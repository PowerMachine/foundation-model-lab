#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fmlab.portfolio import ScorecardError, write_portfolio_scorecard


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic JSON/SVG portfolio scorecards from manifest-verified "
            "public evidence. The command fails closed on schema, metric, or hash drift."
        )
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=REPOSITORY_ROOT / "public-evidence",
        help="Canonical public-evidence root (default: repository/public-evidence).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "public-evidence" / "meta" / "portfolio-scorecard",
        help="Directory for scorecard.json and scorecard.svg.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        json_path, svg_path = write_portfolio_scorecard(args.evidence_root, args.output_dir)
    except ScorecardError as exc:
        print(f"portfolio scorecard rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "json": str(json_path),
                "svg": str(svg_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
