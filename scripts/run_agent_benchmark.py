#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fmlab.agent.benchmark import AgentBenchmarkConfig, run_agent_benchmark  # noqa: E402


DEFAULT_CONFIG = Path("configs/agent/reliable_agent_benchmark.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local Reliable Agent Eval Environment with deterministic scripted controls."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--isolation", choices=("process", "bwrap"))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument(
        "--no-transient-failures",
        action="store_true",
        help="Disable deterministic retry-path fault injection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("agent benchmark config must be a YAML mapping")
    values = dict(loaded)
    if args.artifact_dir is not None:
        values["artifact_dir"] = str(args.artifact_dir)
        if "scratch_dir" not in loaded:
            values["scratch_dir"] = str(args.artifact_dir / ".scratch")
    if args.isolation is not None:
        values["isolation"] = args.isolation
    if args.shard_index is not None:
        values["shard_index"] = args.shard_index
    if args.shard_count is not None:
        values["shard_count"] = args.shard_count
    if args.task_limit is not None:
        values["task_limit"] = args.task_limit
    if args.no_transient_failures:
        values["inject_transient_failures"] = False
    config = AgentBenchmarkConfig.from_mapping(values)
    result_path = run_agent_benchmark(config)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["metrics"]
    print(result_path)
    print(result_path.with_name("report.html"))
    print(
        json.dumps(
            {
                "claim_level": result["claim_level"],
                "episodes": metrics["episodes"],
                "pass_at_1": metrics["pass_at_1"],
                "outcome_pass_rate": metrics["outcome_pass_rate"],
                "reward_hack_rate": metrics["reward_hack_rate"],
                "isolation": result["claim"]["execution_isolation"],
                "network_isolated": result["claim"]["network_isolated"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
