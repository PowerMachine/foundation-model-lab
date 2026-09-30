from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fmlab.artifacts import ExperimentResult


def build_agent_summary(sandbox_root: str | Path, output_dir: str | Path) -> ExperimentResult:
    root = Path(sandbox_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: dict[str, dict[str, Any]] = {}
    for trace_path in sorted(root.glob("*/report/trace.json")):
        value = json.loads(trace_path.read_text(encoding="utf-8"))
        events = value.get("trace", [])
        tools = [event for event in events if event.get("kind") == "tool"]
        failed_tools = sum(
            not event.get("payload", {}).get("response", {}).get("ok", False) for event in tools
        )
        isolated_runs = [
            event
            for event in tools
            if event.get("payload", {}).get("response", {}).get("ok", False)
            and event.get("payload", {})
            .get("response", {})
            .get("metadata", {})
            .get("network_isolated")
        ]
        runs[trace_path.parents[1].name] = {
            "status": value.get("status"),
            "steps": value.get("steps"),
            "tool_calls": len(tools),
            "failed_tool_calls": failed_tools,
            "network_isolated_runs": len(isolated_runs),
            "trace_html": str(trace_path.with_suffix(".html")),
        }
    (output / "runs.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completed = sum(value["status"] == "completed" for value in runs.values())
    verified_completed = sum(
        value["status"] == "completed" and value["network_isolated_runs"] > 0
        for value in runs.values()
    )
    result = ExperimentResult(
        experiment="coding_agent_runs",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics={
            "runs": len(runs),
            "completed": completed,
            "verified_isolated_completed": verified_completed,
            "details": runs,
        },
        parameters={"sandbox_root": str(root)},
        artifacts=["runs.json"],
        notes=[
            "The pytest step-limit run is retained as evidence of small-model JSON/tool-use failure.",
            "Only Podman runs count as network-isolated execution evidence.",
        ],
    )
    result.write(output)
    return result
