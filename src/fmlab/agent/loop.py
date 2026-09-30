from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import ModelProvider
from .sandbox import RestrictedRunner
from .types import ToolResponse, TraceEvent, extract_json_object
from .workspace import AgentWorkspace


SYSTEM_PROMPT = """You are a coding agent operating inside one isolated task workspace.
Return exactly one JSON object per turn. Available actions:
{"tool":"list_files","args":{"pattern":"*"}}
{"tool":"read_file","args":{"path":"relative.py","start_line":1,"end_line":400}}
{"tool":"search","args":{"query":"regex","glob":"*.py"}}
{"tool":"write_file","args":{"path":"relative.py","content":"..."}}
{"tool":"replace_text","args":{"path":"relative.py","old":"exact unique text","new":"..."}}
{"tool":"run","args":{"argv":["python3","test_file.py"]}}
When the requested task is complete and tests pass, return {"final":"summary"}.
Never request absolute paths, network access, credentials, or shell syntax.
"""


@dataclass
class CodingAgentResult:
    status: str
    summary: str
    steps: int
    trace: list[TraceEvent]
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "steps": self.steps,
            "workspace": self.workspace,
            "trace": [event.to_dict() for event in self.trace],
        }

    def write(self, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class CodingAgent:
    def __init__(
        self,
        provider: ModelProvider,
        workspace: AgentWorkspace,
        runner: RestrictedRunner,
        *,
        max_steps: int = 20,
        require_passing_run_after_change: bool = True,
    ) -> None:
        if runner.workspace != workspace.root:
            raise ValueError("Runner and file tools must share the exact workspace")
        self.provider = provider
        self.workspace = workspace
        self.runner = runner
        self.max_steps = max_steps
        self.require_passing_run_after_change = require_passing_run_after_change

    def _execute_tool(self, name: str, args: dict[str, Any]) -> ToolResponse:
        try:
            if name == "list_files":
                return self.workspace.list_files(pattern=str(args.get("pattern", "*")))
            if name == "read_file":
                return self.workspace.read_file(
                    str(args["path"]),
                    int(args.get("start_line", 1)),
                    int(args.get("end_line", 400)),
                )
            if name == "search":
                return self.workspace.search(str(args["query"]), str(args.get("glob", "*.py")))
            if name == "write_file":
                return self.workspace.write_file(str(args["path"]), str(args["content"]))
            if name == "replace_text":
                return self.workspace.replace_text(
                    str(args["path"]), str(args["old"]), str(args["new"])
                )
            if name == "run":
                argv = args.get("argv")
                if not isinstance(argv, list):
                    return ToolResponse(False, "run.argv must be a list")
                return self.runner.run([str(item) for item in argv]).tool_response()
            return ToolResponse(False, f"Unknown tool: {name}")
        except Exception as exc:
            return ToolResponse(False, f"{type(exc).__name__}: {exc}")

    def run(self, task: str) -> CodingAgentResult:
        started = time.monotonic()
        trace: list[TraceEvent] = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        last_change_step = 0
        last_passing_run_step = 0
        for step in range(1, self.max_steps + 1):
            try:
                raw = self.provider.complete(messages)
            except Exception as exc:
                summary = f"Model provider failed: {type(exc).__name__}: {exc}"
                trace.append(
                    TraceEvent(
                        step, "provider_error", {"error": summary}, time.monotonic() - started
                    )
                )
                return CodingAgentResult(
                    "provider_error", summary, step - 1, trace, str(self.workspace.root)
                )
            trace.append(TraceEvent(step, "model", {"response": raw}, time.monotonic() - started))
            try:
                action = extract_json_object(raw)
            except ValueError as exc:
                response = ToolResponse(False, str(exc))
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": response.prompt_value()},
                    ]
                )
                continue
            if "final" in action:
                if (
                    self.require_passing_run_after_change
                    and last_change_step > 0
                    and last_passing_run_step < last_change_step
                ):
                    response = ToolResponse(
                        False,
                        "Completion rejected: no successful test/run occurred after the latest file change.",
                    )
                    trace.append(
                        TraceEvent(
                            step,
                            "validation",
                            {"accepted": False, "reason": response.output},
                            time.monotonic() - started,
                        )
                    )
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": response.prompt_value()},
                        ]
                    )
                    continue
                summary = str(action["final"])
                trace.append(
                    TraceEvent(step, "final", {"summary": summary}, time.monotonic() - started)
                )
                return CodingAgentResult(
                    "completed", summary, step, trace, str(self.workspace.root)
                )
            name = action.get("tool")
            args = action.get("args", {})
            if not isinstance(name, str) or not isinstance(args, dict):
                response = ToolResponse(False, "Action requires string tool and object args")
            else:
                response = self._execute_tool(name, args)
            if response.ok and name in {"write_file", "replace_text"}:
                last_change_step = step
            if response.ok and name == "run":
                last_passing_run_step = step
            trace.append(
                TraceEvent(
                    step,
                    "tool",
                    {"tool": name, "args": args, "response": json.loads(response.prompt_value())},
                    time.monotonic() - started,
                )
            )
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": response.prompt_value()},
                ]
            )
        return CodingAgentResult(
            "step_limit",
            f"Stopped after {self.max_steps} steps",
            self.max_steps,
            trace,
            str(self.workspace.root),
        )
