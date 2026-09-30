from __future__ import annotations

import json
import time
from pathlib import Path

from fmlab.paths import LabPaths

from .container_runner import PodmanRunner
from .loop import CodingAgent, CodingAgentResult
from .providers import ScriptedProvider
from .report import render_trace_html
from .sandbox import ResourceLimits, RestrictedRunner
from .workspace import AgentWorkspace


def demo_responses() -> list[dict[str, object]]:
    return [
        {
            "tool": "write_file",
            "args": {"path": "calculator.py", "content": "def add(a, b):\n    return a - b\n"},
        },
        {
            "tool": "write_file",
            "args": {
                "path": "test_calculator.py",
                "content": (
                    "from calculator import add\n\n"
                    "assert add(2, 3) == 5, 'add should sum both operands'\n"
                    "print('test passed')\n"
                ),
            },
        },
        {"tool": "run", "args": {"argv": ["python3", "test_calculator.py"]}},
        {
            "tool": "replace_text",
            "args": {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
        },
        {"tool": "run", "args": {"argv": ["python3", "test_calculator.py"]}},
        {
            "final": "Implemented add(), observed a failing test, patched the bug, and passed the test."
        },
    ]


def run_demo(
    output_root: str | Path | None = None,
    *,
    isolation: str = "process",
) -> tuple[CodingAgentResult, Path]:
    paths = LabPaths.from_env()
    paths.ensure()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(output_root) if output_root else paths.sandboxes / f"demo-{stamp}"
    workspace = AgentWorkspace(root / "workspace")
    limits = ResourceLimits(timeout_seconds=10, cpu_seconds=5, memory_mb=512)
    if isolation == "podman":
        runner = PodmanRunner(workspace.root, limits=limits)
    else:
        runner = RestrictedRunner(workspace.root, mode=isolation, limits=limits)
    agent = CodingAgent(ScriptedProvider(demo_responses()), workspace, runner, max_steps=10)
    result = agent.run("Implement and test an integer add function.")
    report_dir = root / "report"
    result.write(report_dir / "trace.json")
    render_trace_html(result, report_dir / "trace.html")
    (report_dir / "sandbox.json").write_text(
        json.dumps(runner.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, report_dir
