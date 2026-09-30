from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    experiment: str
    status: str
    started_at: float
    finished_at: float
    metrics: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["duration_seconds"] = self.duration_seconds
        return value

    def write(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "result.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def now_run_dir(artifacts_root: Path, experiment: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = artifacts_root / experiment / stamp
    suffix = 0
    path = base
    while path.exists():
        suffix += 1
        path = Path(f"{base}-{suffix}")
    path.mkdir(parents=True)
    return path


def system_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snapshot["gpu"] = torch.cuda.get_device_name(0)
            snapshot["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            snapshot["cuda"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover - diagnostic only
        snapshot["torch_error"] = str(exc)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if commit.returncode == 0:
            snapshot["git_commit"] = commit.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return snapshot
