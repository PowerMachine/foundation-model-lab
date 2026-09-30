#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fmlab.paths import LabPaths  # noqa: E402
from fmlab.vlm.manifest import prepare_training_manifests  # noqa: E402


def main() -> int:
    paths = LabPaths.from_env()
    parser = argparse.ArgumentParser(description="Prepare canonical VLM training JSONL files")
    parser.add_argument(
        "--source",
        type=Path,
        default=paths.artifacts / "vlm" / "offline-demo" / "datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.datasets / "vlm",
    )
    args = parser.parse_args()
    summary = prepare_training_manifests(args.source, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
