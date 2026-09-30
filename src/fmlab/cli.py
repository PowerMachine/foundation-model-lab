from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent.demo import run_demo
from .doctor import run_doctor, write_doctor_report
from .paths import LabPaths
from .reporting import build_dashboard


def _doctor(args: argparse.Namespace) -> int:
    value = run_doctor()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if args.output:
        write_doctor_report(args.output)
    return 0


def _report(args: argparse.Namespace) -> int:
    paths = LabPaths.from_env()
    source = Path(args.results_root) if args.results_root else paths.artifacts
    destination = Path(args.output) if args.output else source / "dashboard.html"
    output = build_dashboard(source, destination)
    print(output)
    return 0


def _agent_demo(args: argparse.Namespace) -> int:
    result, report_dir = run_demo(args.output_root, isolation=args.isolation)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"report: {report_dir / 'trace.html'}")
    return 0 if result.status == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmlab", description="Visual foundation-model and ML toy laboratory"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect hardware, paths, packages, and models")
    doctor.add_argument("--output")
    doctor.set_defaults(handler=_doctor)

    report = subparsers.add_parser("report", help="Build one HTML dashboard from result files")
    report.add_argument("--results-root")
    report.add_argument("--output")
    report.set_defaults(handler=_report)

    agent = subparsers.add_parser("agent-demo", help="Run deterministic code-agent loop")
    agent.add_argument("--output-root")
    agent.add_argument("--isolation", choices=("podman", "process", "bwrap"), default="podman")
    agent.set_defaults(handler=_agent_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
