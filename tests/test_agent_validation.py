from pathlib import Path

from fmlab.agent.loop import CodingAgent
from fmlab.agent.providers import ScriptedProvider
from fmlab.agent.sandbox import RestrictedRunner
from fmlab.agent.workspace import AgentWorkspace


def test_agent_rejects_false_success_after_failed_run(tmp_path: Path) -> None:
    workspace = AgentWorkspace(tmp_path / "workspace")
    runner = RestrictedRunner(workspace.root, mode="process")
    provider = ScriptedProvider(
        [
            {
                "tool": "write_file",
                "args": {"path": "broken.py", "content": "raise SystemExit(2)\n"},
            },
            {"tool": "run", "args": {"argv": ["python3", "broken.py"]}},
            {"final": "Everything passed"},
        ]
    )
    result = CodingAgent(provider, workspace, runner, max_steps=5).run("Make it pass")
    assert result.status != "completed"
    assert any(event.kind == "validation" for event in result.trace)
