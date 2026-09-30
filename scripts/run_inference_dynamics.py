#!/usr/bin/env python3
"""Run the offline Inference Dynamics Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmlab.config import load_config
from fmlab.systems import InferenceDynamicsConfig, run_inference_dynamics


DEFAULT_CONFIG = Path("configs/systems/inference_dynamics.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare serving policies and modeled capacity, then run a real CPU TinyDecoder gate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = load_config(args.config)
    if args.artifact_dir is not None:
        values["artifact_dir"] = str(args.artifact_dir)
    config = InferenceDynamicsConfig.from_mapping(values)
    result_path = run_inference_dynamics(config)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
