#!/usr/bin/env python3
"""Run every download-free toy lab and build one visual dashboard.

The default run is intentionally offline: it uses synthetic/local data and tiny
models only.  The optional coding-agent stage uses an already-installed Podman
image with ``--pull=never`` and therefore also never downloads an image.
"""

# ruff: noqa: E402 -- make a source checkout directly executable before local imports

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fmlab.agent.container_runner import PodmanRunner
from fmlab.agent.demo import demo_responses
from fmlab.agent.loop import CodingAgent
from fmlab.agent.providers import ScriptedProvider
from fmlab.agent.report import render_trace_html
from fmlab.agent.sandbox import ResourceLimits
from fmlab.agent.workspace import AgentWorkspace
from fmlab.artifacts import ExperimentResult, system_snapshot
from fmlab.config import load_config
from fmlab.llm.experiments import TinyExperimentConfig, run_tiny_from_scratch
from fmlab.llm.smoke import run_offline_llm_smoke_suite
from fmlab.llm.tokenizer_lab import train_and_compare_tokenizers
from fmlab.ml.runner import EXPERIMENTS, run_experiment
from fmlab.paths import LabPaths
from fmlab.reporting import build_dashboard
from fmlab.vlm.demo import run_offline_demo


SUCCESS_STATUSES = {"ok", "completed", "passed", "success"}
ExperimentCall = Callable[[], ExperimentResult]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline LLM, VLM, and ML toy labs, then build one HTML dashboard. "
            "No models or datasets are downloaded."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Artifact root (default: $FMLAB_DATA_ROOT/artifacts)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ml",
        help="Directory containing the six ML YAML configs",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--llm-steps", type=int, default=3, help="Steps per objective in the compact LLM suite"
    )
    parser.add_argument(
        "--tiny-steps", type=int, default=20, help="Training steps for the from-scratch decoder"
    )
    parser.add_argument("--vlm-samples", type=int, default=4, help="Synthetic samples per VLM task")
    parser.add_argument(
        "--agent-demo",
        action="store_true",
        help="Also run the deterministic coding agent in a local Podman image",
    )
    parser.add_argument(
        "--agent-image",
        default=PodmanRunner.DEFAULT_IMAGE,
        help="Existing local OCI image; Podman is always invoked with --pull=never",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed experiment instead of completing the report",
    )
    return parser


def _positive(parser: argparse.ArgumentParser, name: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{name} must be positive")


def _resolved_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _failure_result(
    name: str,
    output_dir: Path,
    started_at: float,
    error: BaseException,
) -> ExperimentResult:
    failure = ExperimentResult(
        experiment=name,
        status="failed",
        started_at=started_at,
        finished_at=time.time(),
        metrics={},
        parameters={},
        artifacts=[],
        notes=[traceback.format_exc(limit=8)],
        error=f"{type(error).__name__}: {error}",
    )
    failure.write(output_dir)
    return failure


def _run_stage(
    name: str,
    output_dir: Path,
    call: ExperimentCall,
) -> tuple[ExperimentResult, bool]:
    started = time.time()
    print(f"\n[{name}] starting -> {output_dir}", flush=True)
    try:
        result = call()
    except Exception as error:  # keep later educational tracks runnable
        result = _failure_result(name, output_dir, started, error)
        print(f"[{name}] FAILED: {result.error}", file=sys.stderr, flush=True)
        return result, False
    succeeded = result.status.lower() in SUCCESS_STATUSES
    marker = "done" if succeeded else f"status={result.status}"
    print(f"[{name}] {marker} ({result.duration_seconds:.2f}s)", flush=True)
    return result, succeeded


def _run_podman_agent(
    output_dir: Path,
    sandbox_root: Path,
    *,
    image: str,
) -> ExperimentResult:
    started = time.time()
    workspace = AgentWorkspace(sandbox_root / "workspace")
    runner = PodmanRunner(
        workspace.root,
        image=image,
        limits=ResourceLimits(timeout_seconds=15, cpu_seconds=8, memory_mb=512),
    )
    preflight = runner.preflight()
    if not preflight.available:
        raise RuntimeError(preflight.detail)

    agent = CodingAgent(
        ScriptedProvider(demo_responses()),
        workspace,
        runner,  # PodmanRunner deliberately implements the RestrictedRunner protocol.
        max_steps=10,
    )
    agent_result = agent.run("Implement and test an integer add function.")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_json = agent_result.write(output_dir / "trace.json")
    trace_html = render_trace_html(agent_result, output_dir / "trace.html")
    sandbox_json = output_dir / "sandbox.json"
    sandbox_json.write_text(
        json.dumps(runner.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ExperimentResult(
        experiment="agent_podman_demo",
        status=agent_result.status,
        started_at=started,
        finished_at=time.time(),
        metrics={
            "steps": agent_result.steps,
            "completed": agent_result.status == "completed",
            "network_isolated": True,
        },
        parameters={"image": image, "pull_policy": "never"},
        artifacts=[str(trace_json), str(trace_html), str(sandbox_json)],
        notes=[
            "The scripted model first creates a bug, observes a failing assertion, fixes it, and reruns it.",
            "Generated code executes with no network and only the task workspace mounted from the host.",
        ],
        error=None if agent_result.status == "completed" else agent_result.summary,
    )


def _write_summary(
    destination: Path,
    *,
    started_at: float,
    output_root: Path,
    device: str,
    records: list[dict[str, object]],
    dashboard: Path | None,
) -> Path:
    succeeded = sum(bool(record["succeeded"]) for record in records)
    payload = {
        "suite": "foundation-model-toy-suite",
        "status": "completed" if succeeded == len(records) else "completed_with_failures",
        "started_at": started_at,
        "finished_at": time.time(),
        "duration_seconds": time.time() - started_at,
        "offline": True,
        "resolved_device": device,
        "output_root": str(output_root),
        "dashboard": str(dashboard) if dashboard else None,
        "experiments_total": len(records),
        "experiments_succeeded": succeeded,
        "experiments_failed": len(records) - succeeded,
        "system": system_snapshot(),
        "experiments": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _positive(parser, "--llm-steps", args.llm_steps)
    _positive(parser, "--tiny-steps", args.tiny_steps)
    _positive(parser, "--vlm-samples", args.vlm_samples)

    # Belt-and-suspenders protection: every default stage is already local, and
    # these flags prevent optional Hugging Face helpers from reaching the hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    paths = LabPaths.from_env()
    paths.ensure()
    for key, value in paths.environment().items():
        os.environ.setdefault(key, value)
    output_root = (args.output_root or paths.artifacts).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = _resolved_device(args.device)
    suite_started = time.time()
    records: list[dict[str, object]] = []
    failed = False

    def run(name: str, destination: Path, call: ExperimentCall) -> bool:
        nonlocal failed
        result, succeeded = _run_stage(name, destination, call)
        # Some external workflows do not write automatically; make the launcher
        # contract uniform without changing artifacts already written by a lab.
        if not (destination / "result.json").exists():
            result.write(destination)
        records.append(
            {
                "name": name,
                "status": result.status,
                "succeeded": succeeded,
                "duration_seconds": result.duration_seconds,
                "result": str(destination / "result.json"),
                "error": result.error,
            }
        )
        failed = failed or not succeeded
        return succeeded

    stages: list[tuple[str, Path, ExperimentCall]] = [
        (
            "llm_offline_suite",
            output_root / "llm" / "offline-suite",
            lambda: run_offline_llm_smoke_suite(
                output_root / "llm" / "offline-suite",
                steps=args.llm_steps,
                device=device,
                seed=args.seed,
            ),
        ),
        (
            "llm_tiny_from_scratch",
            output_root / "llm" / "tiny-from-scratch",
            lambda: run_tiny_from_scratch(
                output_root / "llm" / "tiny-from-scratch",
                TinyExperimentConfig(steps=args.tiny_steps, seed=args.seed, device=device),
            ),
        ),
        (
            "llm_tokenizer_bpe",
            output_root / "llm" / "tokenizer-bpe",
            lambda: train_and_compare_tokenizers(output_root / "llm" / "tokenizer-bpe"),
        ),
        (
            "vlm_offline_demo",
            output_root / "vlm" / "offline-demo",
            lambda: run_offline_demo(
                output_root / "vlm" / "offline-demo",
                sample_count=args.vlm_samples,
                seed=args.seed,
            ),
        ),
    ]

    for name, destination, call in stages:
        if not run(name, destination, call) and args.fail_fast:
            break

    if not (failed and args.fail_fast):
        ml_records_start = len(records)
        for name in EXPERIMENTS:
            destination = output_root / "ml" / name
            config_path = args.config_dir / f"{name}.yaml"

            def ml_call(
                experiment_name: str = name,
                experiment_config_path: Path = config_path,
                experiment_output: Path = destination,
            ) -> ExperimentResult:
                config = (
                    load_config(experiment_config_path) if experiment_config_path.exists() else {}
                )
                config = dict(config or {})
                config["device"] = device
                config["seed"] = args.seed
                return run_experiment(experiment_name, config, experiment_output)

            if not run(f"ml_{name}", destination, ml_call) and args.fail_fast:
                break

        ml_records = records[ml_records_start:]
        ml_summary = output_root / "ml" / "summary.json"
        ml_summary.parent.mkdir(parents=True, exist_ok=True)
        ml_summary.write_text(
            json.dumps(ml_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.agent_demo and not (failed and args.fail_fast):
        agent_output = output_root / "agent" / "podman-demo"
        run(
            "agent_podman_demo",
            agent_output,
            lambda: _run_podman_agent(
                agent_output,
                paths.sandboxes / "toy-suite-podman",
                image=args.agent_image,
            ),
        )

    dashboard: Path | None = None
    try:
        dashboard = build_dashboard(output_root, output_root / "dashboard.html")
        print(f"\n[dashboard] {dashboard}", flush=True)
    except Exception as error:
        failed = True
        records.append(
            {
                "name": "dashboard",
                "status": "failed",
                "succeeded": False,
                "duration_seconds": 0.0,
                "result": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        print(f"[dashboard] FAILED: {error}", file=sys.stderr, flush=True)

    summary = _write_summary(
        output_root / "toy_suite_summary.json",
        started_at=suite_started,
        output_root=output_root,
        device=device,
        records=records,
        dashboard=dashboard,
    )
    print(f"[summary] {summary}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
