from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .schema import GenerationResult, MediaInput


@dataclass(frozen=True)
class TinyVLMConfig:
    """Configuration marker for a weight-free, deterministic baseline."""

    mode: str = "tiny"


class TinyVLMRunner:
    """A dependency-light baseline with the same ``infer`` shape as Qwen3VLRunner.

    It answers only structured context lookups and object-presence questions.  It
    is useful for tests and report plumbing; every output is marked simulated.
    """

    config = TinyVLMConfig()

    @property
    def loaded(self) -> bool:
        return False

    def infer(
        self,
        prompt: str,
        media: Sequence[MediaInput | Path | str] = (),
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        context = context or {}
        answer, rule = self._answer(prompt, context)
        return GenerationResult(
            text=answer,
            backend="tiny-structured-baseline",
            latency_seconds=time.perf_counter() - started,
            simulated=True,
            metadata={
                "rule": rule,
                "media_count": len(media),
                "warning": "Weight-free baseline; not a VLM benchmark.",
            },
        )

    @staticmethod
    def _answer(prompt: str, context: dict[str, Any]) -> tuple[str, str]:
        if "expected_answer" in context:
            return str(context["expected_answer"]), "expected_answer"
        low = prompt.casefold()
        fields = context.get("fields", {})
        for key, value in fields.items():
            if str(key).replace("_", " ").casefold() in low:
                return str(value), "field_lookup"
        objects = {str(value).casefold() for value in context.get("objects", [])}
        for object_name in objects:
            if object_name in low:
                return "yes", "known_object_presence"
        if objects and re.search(r"\b(?:is there|do you see)\b", low):
            return "no", "known_object_absence"
        return "unknown", "fallback"
