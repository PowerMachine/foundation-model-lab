#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from fmlab.config import load_config
from fmlab.distributed import DDPExperimentConfig, run_ddp_correctness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the actual two-process CPU/Gloo DDP correctness lab."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distributed/ddp_cpu.yaml"),
        help="YAML file containing output_dir plus strict DDPExperimentConfig fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory without changing the semantic training contract.",
    )
    return parser


def _load_config(path: Path) -> tuple[DDPExperimentConfig, Path]:
    value = load_config(path)
    if not isinstance(value, dict):
        raise ValueError("DDP YAML must contain a mapping")
    raw: dict[str, Any] = dict(value)
    try:
        output_dir = Path(str(raw.pop("output_dir")))
    except KeyError as exc:
        raise ValueError("DDP YAML requires output_dir") from exc
    return DDPExperimentConfig.from_mapping(raw), output_dir


def _write_failed_result(output_dir: Path, error: Exception, started_at: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "cpu_ddp_correctness_exact_resume",
        "status": "failed",
        "claim_level": "none",
        "simulated": False,
        "started_at": started_at,
        "finished_at": time.time(),
        "error": f"{type(error).__name__}: {error}",
        "claim_boundary": {
            "evidenced": "The attempted run failed; no DDP correctness claim is permitted.",
            "not_evidenced": ["any successful distributed execution"],
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.time()
    output_dir = args.output_dir
    try:
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            raise RuntimeError("this PyTorch build does not provide torch.distributed Gloo")
        config, configured_output = _load_config(args.config)
        output_dir = output_dir or configured_output
        reproduction = f"python scripts/run_ddp_correctness.py --config {args.config}"
        if args.output_dir is not None:
            reproduction += f" --output-dir {args.output_dir}"
        result = run_ddp_correctness(
            output_dir,
            config,
            reproduction_command=reproduction,
        )
    except Exception as exc:
        if output_dir is not None:
            _write_failed_result(output_dir, exc, started_at)
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(Path(output_dir).resolve()),
                "gradient_max_abs": result["metrics"]["parity"]["gradient"]["max_absolute_error"],
                "resume_max_abs": result["metrics"]["resume"]["state_error"]["max_absolute_error"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
