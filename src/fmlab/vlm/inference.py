from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from fmlab.paths import LabPaths

from .schema import GenerationResult, MediaInput


MODEL_RELATIVE_PATHS = {
    "8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "30b-thinking-fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "32b-instruct-fp8": "Qwen/Qwen3-VL-32B-Instruct-FP8",
}


def resolve_qwen_vl_path(
    profile: str = "8b-instruct",
    model_root: Path | None = None,
    *,
    require_exists: bool = True,
) -> Path:
    """Resolve a known local model without downloading or copying it."""

    if profile not in MODEL_RELATIVE_PATHS:
        available = ", ".join(sorted(MODEL_RELATIVE_PATHS))
        raise ValueError(f"Unknown VLM profile {profile!r}; choose one of: {available}")
    root = model_root or LabPaths.from_env().model_root
    path = root / MODEL_RELATIVE_PATHS[profile]
    if require_exists and not path.is_dir():
        raise FileNotFoundError(f"Local VLM not found: {path}")
    return path


@dataclass
class Qwen3VLConfig:
    """Runtime configuration for local Qwen3-VL inference.

    ``mode='offline'`` is a deterministic teaching simulator.  It never imports
    PyTorch/Transformers and must not be interpreted as a model quality result.
    ``mode='local'`` loads only files already present at ``model_path``.
    """

    model_path: Path | None = None
    profile: str = "8b-instruct"
    mode: Literal["offline", "local"] = "offline"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.2
    trust_remote_code: bool = True
    min_pixels: int | None = None
    max_pixels: int | None = None
    model_root: Path = field(default_factory=lambda: LabPaths.from_env().model_root)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def resolved_path(self) -> Path:
        return self.model_path or resolve_qwen_vl_path(self.profile, self.model_root)

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.mode == "local" and not self.resolved_path().is_dir():
            raise FileNotFoundError(self.resolved_path())


class Qwen3VLRunner:
    """Lazy multimodal inference wrapper for images, multiple images, and video."""

    def __init__(self, config: Qwen3VLConfig | None = None) -> None:
        self.config = config or Qwen3VLConfig()
        self.config.validate()
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load a local checkpoint on first use; offline mode remains weight-free."""

        if self.config.mode == "offline" or self.loaded:
            return
        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:  # Compatible fallback for remote-code model releases.
                from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError as exc:  # pragma: no cover - optional heavyweight path
            raise RuntimeError(
                "Local VLM mode requires the 'vlm' dependencies: pip install -e '.[vlm]'"
            ) from exc

        dtype: Any = self.config.torch_dtype
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype, None)
            if dtype is None:
                raise ValueError(f"Unknown torch dtype: {self.config.torch_dtype}")
        kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "torch_dtype": dtype,
            "local_files_only": True,
            "trust_remote_code": self.config.trust_remote_code,
        }
        self._model = ModelClass.from_pretrained(str(self.config.resolved_path()), **kwargs)
        processor_kwargs = {
            "local_files_only": True,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.min_pixels
        if self.config.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.max_pixels
        self._processor = AutoProcessor.from_pretrained(
            str(self.config.resolved_path()), **processor_kwargs
        )

    def infer(
        self,
        prompt: str,
        media: Sequence[MediaInput | Path | str] = (),
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Generate a response.

        ``context`` is used exclusively by the offline simulator.  Supplying an
        ``expected_answer`` makes toy pipelines exactly reproducible.
        """

        started = time.perf_counter()
        if self.config.mode == "offline":
            text, rule = _offline_answer(prompt, context or {})
            return GenerationResult(
                text=text,
                backend="offline-rule-simulator",
                latency_seconds=time.perf_counter() - started,
                simulated=True,
                metadata={"rule": rule, "warning": "Not a real VLM quality measurement."},
            )

        self.load()
        normalized = [_normalize_media(item) for item in media]
        content = [item.as_chat_content() for item in normalized]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        assert self._processor is not None and self._model is not None
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_batch_to_model(inputs, self._model)
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            **self.config.generation_kwargs,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
        generated = self._model.generate(**inputs, **generation_kwargs)
        input_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated, strict=True)]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return GenerationResult(
            text=text.strip(),
            backend=f"qwen3-vl:{self.config.resolved_path().name}",
            latency_seconds=time.perf_counter() - started,
            metadata={"media_count": len(normalized)},
        )

    def infer_batch(
        self,
        requests: Sequence[tuple[str, Sequence[MediaInput | Path | str], dict[str, Any]]],
    ) -> list[GenerationResult]:
        """Simple sequential batch that is safe for heterogeneous visual shapes."""

        return [self.infer(prompt, media, context=context) for prompt, media, context in requests]


def _normalize_media(item: MediaInput | Path | str) -> MediaInput:
    if isinstance(item, MediaInput):
        return item
    path = Path(item)
    video_suffixes = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    return MediaInput(path=path, kind="video" if path.suffix.lower() in video_suffixes else "image")


def _move_batch_to_model(batch: Any, model: Any) -> Any:
    if hasattr(batch, "to"):
        try:
            return batch.to(model.device)
        except (AttributeError, ValueError):
            return batch
    return batch


def _offline_answer(prompt: str, context: dict[str, Any]) -> tuple[str, str]:
    if "expected_answer" in context:
        return str(context["expected_answer"]), "expected_answer"

    low = prompt.casefold()
    objects = {str(item).casefold() for item in context.get("objects", [])}
    presence_match = re.search(
        r"(?:is there|do you see|contains?).*?\b(?:a|an)?\s*([\w -]+?)(?:\?| in)", low
    )
    if objects and presence_match:
        query = presence_match.group(1).strip()
        present = query in objects or any(query in obj or obj in query for obj in objects)
        return ("yes" if present else "no"), "object_presence"

    fields = context.get("fields", {})
    for key, value in fields.items():
        if str(key).replace("_", " ").casefold() in low:
            return str(value), "field_lookup"

    values = context.get("values", {})
    if values:
        if "highest" in low or "maximum" in low or "largest" in low:
            return str(max(values, key=values.get)), "chart_argmax"
        if "lowest" in low or "minimum" in low or "smallest" in low:
            return str(min(values, key=values.get)), "chart_argmin"

    return "unknown", "fallback"
