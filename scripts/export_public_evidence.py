#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fmlab.release import (
    MAX_PUBLIC_FILE_BYTES,
    EvidenceExportError,
    export_public_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a redacted, text-only evidence bundle through a fail-closed allowlist."
        )
    )
    parser.add_argument("source", type=Path, help="Source artifact directory")
    parser.add_argument("destination", type=Path, help="Published public-evidence directory")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="BASENAME",
        help="Explicitly include one additional top-level safe text file; repeat as needed.",
    )
    parser.add_argument(
        "--redact-literal",
        action="append",
        default=[],
        metavar="TEXT",
        help="Additional username, hostname, path, or private literal to redact and rescan.",
    )
    parser.add_argument(
        "--claim-evidenced",
        help="Required when source result.json has no claim_boundary.",
    )
    parser.add_argument(
        "--not-evidenced",
        action="append",
        default=[],
        metavar="BOUNDARY",
        help="One explicit non-claim; repeat. Used with --claim-evidenced.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=MAX_PUBLIC_FILE_BYTES,
        help=f"Per-file cap, at most {MAX_PUBLIC_FILE_BYTES} bytes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.not_evidenced and not args.claim_evidenced:
        parser.error("--not-evidenced requires --claim-evidenced")
    explicit_boundary = None
    if args.claim_evidenced:
        if not args.not_evidenced:
            parser.error("--claim-evidenced requires at least one --not-evidenced")
        explicit_boundary = {
            "evidenced": args.claim_evidenced,
            "not_evidenced": args.not_evidenced,
        }
    try:
        exported = export_public_evidence(
            args.source,
            args.destination,
            additional_files=args.include,
            redact_literals=args.redact_literal,
            claim_boundary=explicit_boundary,
            max_file_bytes=args.max_file_bytes,
        )
    except EvidenceExportError as exc:
        print(f"public evidence export rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(exported.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
