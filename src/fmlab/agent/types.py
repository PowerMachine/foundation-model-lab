from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    step: int
    kind: str
    payload: dict[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResponse:
    ok: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_value(self, limit: int = 12_000) -> str:
        output = self.output
        truncated = len(output) > limit
        if truncated:
            output = output[:limit] + "\n...[truncated]"
        return json.dumps(
            {"ok": self.ok, "output": output, "metadata": self.metadata},
            ensure_ascii=False,
        )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object without accepting executable model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    candidates = [0]
    candidates.extend(index for index, char in enumerate(stripped) if char == "{")
    for start in dict.fromkeys(candidates):
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a JSON object")
