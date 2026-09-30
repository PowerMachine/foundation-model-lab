from __future__ import annotations

import os
import resource
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .types import ToolResponse


@dataclass(frozen=True)
class ResourceLimits:
    timeout_seconds: int = 30
    cpu_seconds: int = 20
    memory_mb: int = 2048
    output_bytes: int = 200_000
    file_bytes: int = 50_000_000
    processes: int = 32


@dataclass
class RunRecord:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    isolation: str
    network_isolated: bool
    timed_out: bool = False

    def tool_response(self) -> ToolResponse:
        combined = self.stdout
        if self.stderr:
            combined += ("\n" if combined else "") + "[stderr]\n" + self.stderr
        return ToolResponse(
            ok=self.returncode == 0 and not self.timed_out,
            output=combined,
            metadata={
                "argv": self.argv,
                "returncode": self.returncode,
                "duration_seconds": round(self.duration_seconds, 4),
                "isolation": self.isolation,
                "network_isolated": self.network_isolated,
                "timed_out": self.timed_out,
            },
        )


class RestrictedRunner:
    """Run argument-vector commands in a bounded workspace.

    `bwrap` mode denies network and exposes a minimal read-only runtime. `process`
    mode is intentionally marked as not network isolated and is suitable only for
    trusted smoke demonstrations. No mode uses a shell.
    """

    DEFAULT_ALLOWLIST = {"python", "python3", "pytest"}

    def __init__(
        self,
        workspace: str | Path,
        *,
        mode: str = "bwrap",
        allowlist: set[str] | None = None,
        limits: ResourceLimits | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        if mode not in {"bwrap", "process"}:
            raise ValueError("mode must be 'bwrap' or 'process'")
        self.allowlist = allowlist or set(self.DEFAULT_ALLOWLIST)
        self.limits = limits or ResourceLimits()

    def _validate(self, argv: Sequence[str]) -> list[str]:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must be a non-empty list of strings")
        executable_name = Path(argv[0]).name
        if executable_name not in self.allowlist:
            raise PermissionError(f"Executable not allowed: {executable_name}")
        executable = shutil.which(argv[0])
        if executable is None:
            raise FileNotFoundError(argv[0])
        if self.mode == "process" and any(item in {"-c", "--command"} for item in argv[1:]):
            raise PermissionError("Inline code is disabled in process isolation mode")
        for item in argv[1:]:
            if "\x00" in item:
                raise ValueError("NUL byte in argument")
            if item.startswith("/"):
                candidate = Path(item).resolve()
                if not candidate.is_relative_to(self.workspace):
                    raise PermissionError(f"Absolute argument escapes workspace: {item}")
        return [executable, *argv[1:]]

    def _limit_process(self) -> None:
        memory = self.limits.memory_mb * 1024 * 1024
        resource.setrlimit(
            resource.RLIMIT_CPU, (self.limits.cpu_seconds, self.limits.cpu_seconds + 1)
        )
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.limits.file_bytes, self.limits.file_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (self.limits.processes, self.limits.processes))
        os.setsid()

    def _bwrap_argv(self, command: list[str]) -> list[str]:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError("bubblewrap is required for bwrap mode")
        wrapper = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(self.workspace),
            "/workspace",
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
        ]
        for host_path in ("/usr", "/bin", "/lib", "/lib64", "/opt/miniforge3"):
            if Path(host_path).exists():
                wrapper.extend(["--ro-bind", host_path, host_path])
        return wrapper + command

    def run(self, argv: Sequence[str]) -> RunRecord:
        command = self._validate(argv)
        isolation = self.mode
        network_isolated = self.mode == "bwrap"
        executed = self._bwrap_argv(command) if self.mode == "bwrap" else command
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPATH": str(self.workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        started = time.monotonic()
        try:
            process = subprocess.run(
                executed,
                cwd=self.workspace,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.limits.timeout_seconds,
                check=False,
                preexec_fn=self._limit_process,
            )
            stdout = process.stdout[: self.limits.output_bytes]
            stderr = process.stderr[: self.limits.output_bytes]
            return RunRecord(
                argv=list(argv),
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=time.monotonic() - started,
                isolation=isolation,
                network_isolated=network_isolated,
            )
        except subprocess.TimeoutExpired as exc:
            return RunRecord(
                argv=list(argv),
                returncode=124,
                stdout=(exc.stdout or "")[: self.limits.output_bytes],
                stderr=(exc.stderr or "")[: self.limits.output_bytes],
                duration_seconds=time.monotonic() - started,
                isolation=isolation,
                network_isolated=network_isolated,
                timed_out=True,
            )

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "network_isolated": self.mode == "bwrap",
            "allowlist": sorted(self.allowlist),
            "limits": asdict(self.limits),
        }
