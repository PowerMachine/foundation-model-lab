from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence

from .sandbox import ResourceLimits, RunRecord


_IMAGE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,254}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class PodmanPreflight:
    """Result of a local, non-pulling Podman availability check."""

    available: bool
    runtime_path: str | None
    runtime_usable: bool
    image: str
    image_present: bool
    workspace: str
    workspace_writable: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PodmanRunner:
    """Run allowlisted argv commands in a locked-down, rootless Podman container.

    The class intentionally has the same ``run`` and ``describe`` shape as
    :class:`RestrictedRunner`.  Building a command is a pure operation: it does
    not inspect, pull, or start the configured image.  ``preflight`` is the
    explicit check for runtime and local-image availability.

    This boundary permits arbitrary code *inside* the container.  Its host
    boundary is the single read/write workspace bind mount; no model, home, or
    socket directories are mounted.
    """

    DEFAULT_IMAGE = "localhost/fmlab-python:3.12"
    DEFAULT_ALLOWLIST = frozenset({"python", "python3", "pytest"})
    DEFAULT_PYTHON_MODULES = frozenset({"pytest"})

    def __init__(
        self,
        workspace: str | Path,
        *,
        image: str = DEFAULT_IMAGE,
        podman_binary: str = "podman",
        allowlist: set[str] | frozenset[str] | None = None,
        python_modules: set[str] | frozenset[str] | None = None,
        limits: ResourceLimits | None = None,
        cpus: float = 1.0,
        tmpfs_mb: int = 256,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.image = self._validate_image(image)
        self.podman_binary = self._validate_runtime_name(podman_binary)
        self.allowlist = frozenset(allowlist or self.DEFAULT_ALLOWLIST)
        self.python_modules = frozenset(python_modules or self.DEFAULT_PYTHON_MODULES)
        self.limits = limits or ResourceLimits()
        self.cpus = float(cpus)
        self.tmpfs_mb = int(tmpfs_mb)
        self._validate_configuration()

    @staticmethod
    def _validate_image(image: str) -> str:
        if not isinstance(image, str) or not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("image must be one safe OCI image reference")
        if image.startswith("-") or ".." in image:
            raise ValueError("unsafe OCI image reference")
        return image

    @staticmethod
    def _validate_runtime_name(runtime: str) -> str:
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("podman_binary must be a non-empty string")
        if "\x00" in runtime or Path(runtime).name != "podman":
            raise ValueError("podman_binary must name the Podman executable")
        return runtime

    def _validate_configuration(self) -> None:
        if not self.allowlist or any(
            not item or Path(item).name != item or "/" in item for item in self.allowlist
        ):
            raise ValueError("allowlist entries must be bare executable names")
        if any(not item or not item.replace("_", "").isalnum() for item in self.python_modules):
            raise ValueError("python_modules entries must be simple module names")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.tmpfs_mb <= 0:
            raise ValueError("tmpfs_mb must be positive")
        numeric_limits = (
            self.limits.timeout_seconds,
            self.limits.cpu_seconds,
            self.limits.memory_mb,
            self.limits.output_bytes,
            self.limits.file_bytes,
            self.limits.processes,
        )
        if any(value <= 0 for value in numeric_limits):
            raise ValueError("all resource limits must be positive")
        # Podman's comma-delimited --mount syntax cannot represent this path safely.
        if "," in str(self.workspace) or "\n" in str(self.workspace):
            raise ValueError("workspace path contains a character unsafe for Podman --mount")

    def validate_argv(self, argv: Sequence[str]) -> list[str]:
        """Validate a container command without resolving it on the host."""
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a non-empty sequence of strings")
        if len(argv) > 128:
            raise ValueError("argv has too many arguments")
        if any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("argv must contain only non-empty strings")
        if sum(len(item) for item in argv) > 32_768:
            raise ValueError("argv is too large")
        for item in argv:
            if len(item) > 4_096:
                raise ValueError("one argv item is too large")
            if any(ord(char) < 32 or ord(char) == 127 for char in item):
                raise ValueError("control character in argv")

        executable = argv[0]
        if Path(executable).name != executable or executable not in self.allowlist:
            raise PermissionError(f"Executable not allowed: {executable}")

        arguments = list(argv[1:])
        for argument in arguments:
            # Absolute paths may refer only to the mounted workspace. Relative
            # traversal is unnecessary for tests and makes review ambiguous.
            value = (
                argument.split("=", 1)[1]
                if argument.startswith("-") and "=" in argument
                else argument
            )
            if value.startswith("/"):
                path = PurePosixPath(value)
                if (
                    path != PurePosixPath("/workspace")
                    and PurePosixPath("/workspace") not in path.parents
                ):
                    raise PermissionError(f"Absolute path is outside /workspace: {value}")
            if ".." in PurePosixPath(value).parts:
                raise PermissionError(f"Path traversal is disabled: {value}")

        if executable in {"python", "python3"}:
            self._validate_python_arguments(arguments)
        return [executable, *arguments]

    def _validate_python_arguments(self, arguments: list[str]) -> None:
        if not arguments:
            raise PermissionError("Python requires an explicit script or allowlisted module")

        # Interpret only Python's leading options. Once a script is selected, its
        # own arguments are opaque and cannot turn back into interpreter flags.
        safe_flags = {"-B", "-E", "-I", "-P", "-s", "-S", "-u"}
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument == "-" or argument in {"-i", "--interactive", "--"}:
                raise PermissionError("interactive or stdin Python execution is disabled")
            if argument == "-c" or argument.startswith("--command"):
                raise PermissionError("inline Python execution is disabled")
            if argument == "-m":
                if index + 1 >= len(arguments) or arguments[index + 1] not in self.python_modules:
                    raise PermissionError("Python module execution is not allowlisted")
                return
            if argument.startswith("-m") and len(argument) > 2:
                if argument[2:] not in self.python_modules:
                    raise PermissionError("Python module execution is not allowlisted")
                return
            if argument in safe_flags:
                index += 1
                continue
            if argument.startswith("-"):
                # Reject combined forms such as -Bc and -Bm, which Python accepts
                # and which could otherwise bypass exact -c/-m validation.
                raise PermissionError(f"Python interpreter option is not allowlisted: {argument}")
            return

    def build_command(
        self,
        argv: Sequence[str],
        *,
        container_name: str = "fmlab-dry-run",
        runtime_path: str | None = None,
    ) -> list[str]:
        """Build the exact Podman argv; does not require Podman or the image."""
        command = self.validate_argv(argv)
        if not _CONTAINER_NAME.fullmatch(container_name):
            raise ValueError("unsafe container name")
        runtime = runtime_path or self.podman_binary
        if Path(runtime).name != "podman":
            raise ValueError("runtime_path must resolve to Podman")

        mount = f"type=bind,src={self.workspace},dst=/workspace,rw=true"
        return [
            runtime,
            "run",
            "--rm",
            "--pull=never",
            f"--name={container_name}",
            "--network=none",
            "--read-only",
            "--read-only-tmpfs=false",
            "--image-volume=ignore",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--userns=keep-id",
            "--ipc=private",
            "--pid=private",
            "--no-healthcheck",
            "--no-hosts",
            "--http-proxy=false",
            "--systemd=false",
            "--log-driver=none",
            "--stop-timeout=2",
            f"--timeout={self.limits.timeout_seconds}",
            f"--pids-limit={self.limits.processes}",
            f"--memory={self.limits.memory_mb}m",
            f"--memory-swap={self.limits.memory_mb}m",
            f"--cpus={self.cpus:g}",
            f"--ulimit=cpu={self.limits.cpu_seconds}:{self.limits.cpu_seconds + 1}",
            f"--ulimit=fsize={self.limits.file_bytes}:{self.limits.file_bytes}",
            "--ulimit=nofile=256:256",
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={self.tmpfs_mb}m,mode=1777",
            "--mount",
            mount,
            "--workdir=/workspace",
            "--unsetenv-all",
            "--env=PATH=/usr/local/bin:/usr/bin:/bin",
            "--env=HOME=/tmp",
            "--env=LANG=C.UTF-8",
            "--env=LC_ALL=C.UTF-8",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=PYTHONUNBUFFERED=1",
            self.image,
            *command,
        ]

    def preflight(self, *, timeout_seconds: float = 5.0) -> PodmanPreflight:
        """Check Podman and the configured local image without pulling anything."""
        runtime_path = shutil.which(self.podman_binary)
        writable = self.workspace.is_dir() and os.access(
            self.workspace, os.R_OK | os.W_OK | os.X_OK
        )
        if runtime_path is None:
            return PodmanPreflight(
                False,
                None,
                False,
                self.image,
                False,
                str(self.workspace),
                writable,
                "Podman executable was not found",
            )

        environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
        try:
            version = subprocess.run(
                [runtime_path, "version", "--format", "{{.Client.Version}}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return PodmanPreflight(
                False,
                runtime_path,
                False,
                self.image,
                False,
                str(self.workspace),
                writable,
                f"Podman is not usable: {exc}",
            )
        if version.returncode != 0:
            detail = (version.stderr or version.stdout).strip()[:500]
            return PodmanPreflight(
                False,
                runtime_path,
                False,
                self.image,
                False,
                str(self.workspace),
                writable,
                f"Podman version check failed: {detail}",
            )

        try:
            image_check = subprocess.run(
                [runtime_path, "image", "exists", self.image],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=environment,
            )
            image_present = image_check.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            image_present = False

        available = writable and image_present
        if not writable:
            detail = "Workspace is not readable and writable"
        elif not image_present:
            detail = "Configured image is not present locally; no pull was attempted"
        else:
            detail = "Podman, workspace, and local image are ready"
        return PodmanPreflight(
            available,
            runtime_path,
            True,
            self.image,
            image_present,
            str(self.workspace),
            writable,
            detail,
        )

    def availability(self) -> dict[str, object]:
        """Dictionary convenience wrapper for UI/report generation."""
        return self.preflight().to_dict()

    def is_available(self) -> bool:
        return self.preflight().available

    @staticmethod
    def _drain(stream: BinaryIO, limit: int, destination: bytearray) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            remaining = limit - len(destination)
            if remaining > 0:
                destination.extend(chunk[:remaining])

    @staticmethod
    def _decode(data: bytearray, limit: int) -> str:
        text = bytes(data).decode("utf-8", errors="replace")
        if len(data) >= limit:
            text += "\n...[output truncated]"
        return text

    @staticmethod
    def _force_remove(runtime: str, container_name: str) -> None:
        try:
            subprocess.run(
                [runtime, "rm", "--force", "--ignore", container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def run(self, argv: Sequence[str]) -> RunRecord:
        check = self.preflight()
        if not check.available or check.runtime_path is None:
            raise RuntimeError(f"Podman runner is unavailable: {check.detail}")

        container_name = f"fmlab-{uuid.uuid4().hex[:16]}"
        executed = self.build_command(
            argv, container_name=container_name, runtime_path=check.runtime_path
        )
        started = time.monotonic()
        environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
        process = subprocess.Popen(
            executed,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workspace,
            env=environment,
            shell=False,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout = bytearray()
        stderr = bytearray()
        readers = [
            threading.Thread(
                target=self._drain,
                args=(process.stdout, self.limits.output_bytes, stdout),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(process.stderr, self.limits.output_bytes, stderr),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            process.wait(timeout=self.limits.timeout_seconds + 5)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._force_remove(check.runtime_path, container_name)
            process.wait(timeout=5)
        finally:
            process.stdout.close()
            process.stderr.close()
            for reader in readers:
                reader.join(timeout=2)

        return RunRecord(
            argv=list(argv),
            returncode=124 if timed_out else process.returncode,
            stdout=self._decode(stdout, self.limits.output_bytes),
            stderr=self._decode(stderr, self.limits.output_bytes),
            duration_seconds=time.monotonic() - started,
            isolation="podman",
            network_isolated=True,
            timed_out=timed_out,
        )

    def describe(self) -> dict[str, object]:
        return {
            "mode": "podman",
            "image": self.image,
            "network_isolated": True,
            "rootfs_read_only": True,
            "host_mounts": [{"host": str(self.workspace), "container": "/workspace"}],
            "allowlist": sorted(self.allowlist),
            "python_modules": sorted(self.python_modules),
            "cpus": self.cpus,
            "tmpfs_mb": self.tmpfs_mb,
            "limits": asdict(self.limits),
        }
