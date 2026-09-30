from __future__ import annotations

from pathlib import Path

import pytest

from fmlab.agent.container_runner import PodmanRunner
from fmlab.agent.sandbox import ResourceLimits


def test_podman_command_builder_needs_no_image(tmp_path: Path) -> None:
    limits = ResourceLimits(
        timeout_seconds=17,
        cpu_seconds=11,
        memory_mb=768,
        output_bytes=10_000,
        file_bytes=2_000_000,
        processes=13,
    )
    runner = PodmanRunner(
        tmp_path / "workspace",
        image="localhost/test-python:3.12",
        limits=limits,
        cpus=0.75,
        tmpfs_mb=48,
    )

    command = runner.build_command(
        ["python3", "tests/test_example.py"], container_name="fixed-dry-run"
    )

    assert command[:2] == ["podman", "run"]
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=all" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--pids-limit=13" in command
    assert "--memory=768m" in command
    assert "--memory-swap=768m" in command
    assert "--cpus=0.75" in command
    assert "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=48m,mode=1777" in command
    assert command.count("--mount") == 1
    mount = command[command.index("--mount") + 1]
    assert mount == f"type=bind,src={runner.workspace},dst=/workspace,rw=true"

    image_index = command.index("localhost/test-python:3.12")
    assert command[image_index + 1 :] == ["python3", "tests/test_example.py"]
    assert "bash" not in command
    assert "sh" not in command


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-lc", "id"],
        ["/usr/bin/python3", "test.py"],
        ["python3", "-c", "print(1)"],
        ["python3", "-cprint(1)"],
        ["python3", "-Bc", "print(1)"],
        ["python3", "-Bm", "pip"],
        ["python3"],
        ["python3", "-m", "pip", "install", "x"],
        ["python3", "../outside.py"],
        ["pytest", "/etc/passwd"],
        ["python3", "bad\narg.py"],
    ],
)
def test_podman_runner_rejects_unsafe_argv(tmp_path: Path, argv: list[str]) -> None:
    runner = PodmanRunner(tmp_path, image="localhost/test-python:3.12")
    with pytest.raises((PermissionError, ValueError)):
        runner.build_command(argv)


def test_podman_runner_allows_pytest_module_and_workspace_path(tmp_path: Path) -> None:
    runner = PodmanRunner(tmp_path, image="localhost/test-python:3.12")
    command = runner.build_command(["python3", "-m", "pytest", "/workspace/tests"])
    assert command[-4:] == ["python3", "-m", "pytest", "/workspace/tests"]


def test_preflight_reports_missing_runtime_without_checking_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = PodmanRunner(tmp_path, image="localhost/not-installed:latest")
    monkeypatch.setattr("fmlab.agent.container_runner.shutil.which", lambda _: None)

    result = runner.preflight()

    assert not result.available
    assert not result.runtime_usable
    assert not result.image_present
    assert "not found" in result.detail
    assert runner.availability()["available"] is False


def test_describe_has_only_one_host_mount(tmp_path: Path) -> None:
    runner = PodmanRunner(tmp_path, image="localhost/test-python:3.12")
    description = runner.describe()
    assert description["network_isolated"] is True
    assert description["rootfs_read_only"] is True
    assert description["host_mounts"] == [
        {"host": str(tmp_path.resolve()), "container": "/workspace"}
    ]
