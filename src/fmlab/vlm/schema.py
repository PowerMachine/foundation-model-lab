from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class MediaInput:
    """One image or video attached to a model request."""

    path: Path
    kind: Literal["image", "video"] = "image"

    def as_chat_content(self) -> dict[str, str]:
        return {"type": self.kind, self.kind: str(self.path)}


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    answer: str
    answer_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    text: str
    backend: str
    latency_seconds: float
    simulated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingCase:
    id: str
    image_path: Path
    object_name: str
    present: bool
    bbox_xyxy: tuple[int, int, int, int] | None = None
    prompt: str | None = None

    def question(self) -> str:
        return self.prompt or (
            f"Is there a {self.object_name} in this image? Answer only yes or no."
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_path"] = str(self.image_path)
        return value
