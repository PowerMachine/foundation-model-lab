#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fmlab.config import load_config
from fmlab.vlm.lora_e2e import (
    VLMLoRAExperimentConfig,
    run_vlm_lora_experiment,
    write_failed_vlm_lora_result,
)


DEFAULT_CONFIG = Path("configs/vlm/qwen3_vl_8b_lora_e2e.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded Qwen3-VL LoRA SFT using local files only."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=("mock", "local"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--save-adapter",
        action="store_true",
        help="Persist PEFT adapter weights (disabled by default to bound storage).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = load_config(args.config)
    if args.backend:
        values["backend"] = args.backend
    if args.max_steps is not None:
        values["max_steps"] = args.max_steps
    if args.artifact_dir is not None:
        values["artifact_dir"] = str(args.artifact_dir)
    if args.save_adapter:
        values["save_adapter"] = True
    config = VLMLoRAExperimentConfig.from_mapping(values)
    started = time.time()
    try:
        result = run_vlm_lora_experiment(config)
    except BaseException as error:
        failure = write_failed_vlm_lora_result(config, error, started_at=started)
        print(f"failed result: {failure}", file=sys.stderr)
        raise
    print(result)
    print(result.with_name("report.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
