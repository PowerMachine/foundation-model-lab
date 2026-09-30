#!/usr/bin/env python3
"""Run the visual knowledge-work environment and reward-hacking audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmlab.config import load_config
from fmlab.vlm.reward_environment import RewardLabConfig, run_visual_reward_lab


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Override the configured artifact directory"
    )
    args = parser.parse_args()
    raw = load_config(args.config)
    if args.output_dir is not None:
        raw = {**raw, "output_dir": str(args.output_dir)}
    result = run_visual_reward_lab(RewardLabConfig.from_mapping(raw))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
