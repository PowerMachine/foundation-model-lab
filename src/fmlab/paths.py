from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = (
    Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "foundation-model-lab"
)
DEFAULT_MODEL_ROOT = Path.home() / "models"


@dataclass(frozen=True)
class LabPaths:
    """All heavyweight paths are centralized so experiments never duplicate models."""

    data_root: Path
    model_root: Path

    @classmethod
    def from_env(cls) -> "LabPaths":
        return cls(
            data_root=Path(os.getenv("FMLAB_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser(),
            model_root=Path(os.getenv("FMLAB_MODEL_ROOT", str(DEFAULT_MODEL_ROOT))).expanduser(),
        )

    @property
    def artifacts(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def datasets(self) -> Path:
        return self.data_root / "datasets"

    @property
    def checkpoints(self) -> Path:
        return self.data_root / "checkpoints"

    @property
    def cache(self) -> Path:
        return self.data_root / "cache"

    @property
    def sandboxes(self) -> Path:
        return self.data_root / "agent-sandboxes"

    @property
    def tmp(self) -> Path:
        return self.data_root / "tmp"

    def ensure(self) -> None:
        for path in (
            self.data_root,
            self.artifacts,
            self.datasets,
            self.checkpoints,
            self.cache,
            self.sandboxes,
            self.tmp,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def model(self, relative: str) -> Path:
        candidate = (self.model_root / relative).resolve()
        root = self.model_root.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Model path escapes model root: {relative}")
        return candidate

    def environment(self) -> dict[str, str]:
        """Environment variables recommended for child training processes."""
        return {
            "FMLAB_DATA_ROOT": str(self.data_root),
            "FMLAB_MODEL_ROOT": str(self.model_root),
            "HF_HOME": str(self.cache / "huggingface"),
            "HF_DATASETS_CACHE": str(self.cache / "huggingface" / "datasets"),
            "TORCH_HOME": str(self.cache / "torch"),
            "TMPDIR": str(self.tmp),
        }
