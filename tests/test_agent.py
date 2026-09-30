from __future__ import annotations

from pathlib import Path

import pytest

from fmlab.agent.demo import run_demo
from fmlab.agent.sandbox import RestrictedRunner
from fmlab.agent.workspace import AgentWorkspace


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path / "workspace")
    with pytest.raises(ValueError):
        workspace.resolve("../escape.py")


def test_replace_is_exact(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path / "workspace")
    assert workspace.write_file("a.py", "x = 1\n").ok
    assert workspace.replace_text("a.py", "x = 1", "x = 2").ok
    assert "x = 2" in workspace.read_file("a.py").output
    assert not workspace.replace_text("a.py", "missing", "value").ok


def test_process_runner_allowlist_and_no_inline_code(tmp_path: Path) -> None:
    runner = RestrictedRunner(tmp_path, mode="process")
    with pytest.raises(PermissionError):
        runner.run(["bash", "-lc", "echo unsafe"])
    with pytest.raises(PermissionError):
        runner.run(["python3", "-c", "print('unsafe mode')"])


def test_full_demo_produces_passing_trace(tmp_path: Path) -> None:
    result, report = run_demo(tmp_path / "demo", isolation="process")
    assert result.status == "completed"
    assert result.steps == 6
    assert (report / "trace.json").is_file()
    assert (report / "trace.html").is_file()
    assert "return a + b" in (tmp_path / "demo/workspace/calculator.py").read_text()
