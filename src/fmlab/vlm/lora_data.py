from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class VLMTrainingExample:
    """One image/question/answer turn expanded from a canonical manifest row."""

    id: str
    source_id: str
    task: str
    image_path: Path
    question: str
    answer: str
    answer_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_path"] = str(self.image_path)
        return value


@dataclass(frozen=True)
class GroupedSplit:
    train: tuple[VLMTrainingExample, ...]
    evaluation: tuple[VLMTrainingExample, ...]
    train_source_ids: tuple[str, ...]
    evaluation_source_ids: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "train_examples": len(self.train),
            "evaluation_examples": len(self.evaluation),
            "train_sources": len(self.train_source_ids),
            "evaluation_sources": len(self.evaluation_source_ids),
            "source_overlap": sorted(set(self.train_source_ids) & set(self.evaluation_source_ids)),
            "train_tasks": dict(sorted(Counter(item.task for item in self.train).items())),
            "evaluation_tasks": dict(
                sorted(Counter(item.task for item in self.evaluation).items())
            ),
        }


def load_canonical_vlm_manifest(
    manifest_path: str | Path,
    *,
    require_images: bool = True,
) -> list[VLMTrainingExample]:
    """Load and expand the lab's canonical JSONL manifest.

    Document and chart rows contain a ``qa`` list and expand to one example per
    question. Grounding rows are already atomic and become yes/no examples.
    Every path is resolved relative to the manifest, so copied manifests remain
    portable. Duplicate example ids are rejected instead of silently leaking into
    both sides of an evaluation.
    """

    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    examples: list[VLMTrainingExample] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number} must contain a JSON object")
        image_path = _resolve_image_path(source, row, line_number)
        if require_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        manifest_source_id = str(row.get("id") or f"row-{line_number}")
        task = str(row.get("task") or "unknown")
        qa_items = row.get("qa")
        if qa_items is not None:
            if not isinstance(qa_items, list) or not qa_items:
                raise ValueError(f"{source}:{line_number} qa must be a non-empty list")
            for qa_index, qa in enumerate(qa_items):
                if not isinstance(qa, dict):
                    raise ValueError(f"{source}:{line_number} qa[{qa_index}] must be an object")
                example = VLMTrainingExample(
                    id=str(qa.get("id") or f"{manifest_source_id}-qa-{qa_index}"),
                    source_id=_image_group_id(image_path),
                    task=task,
                    image_path=image_path,
                    question=_required_text(qa, "question", source, line_number),
                    answer=_required_text(qa, "answer", source, line_number),
                    answer_type=str(qa.get("answer_type") or "text"),
                    metadata={
                        "manifest_source_id": manifest_source_id,
                        "sample": dict(row.get("metadata") or {}),
                        "qa": dict(qa.get("metadata") or {}),
                    },
                )
                _append_unique(examples, seen_ids, example, source, line_number)
            continue

        if "object_name" in row and "present" in row:
            object_name = str(row["object_name"])
            question = str(row.get("prompt") or "").strip() or (
                f"Is there a {object_name} in this image? Answer only yes or no."
            )
            example = VLMTrainingExample(
                id=manifest_source_id,
                source_id=_image_group_id(image_path),
                task=task,
                image_path=image_path,
                question=question,
                answer="yes" if bool(row["present"]) else "no",
                answer_type="boolean",
                metadata={
                    "object_name": object_name,
                    "present": bool(row["present"]),
                    "bbox_xyxy": row.get("bbox_xyxy"),
                },
            )
            _append_unique(examples, seen_ids, example, source, line_number)
            continue

        if "question" in row and "answer" in row:
            example = VLMTrainingExample(
                id=manifest_source_id,
                source_id=_image_group_id(image_path),
                task=task,
                image_path=image_path,
                question=_required_text(row, "question", source, line_number),
                answer=_required_text(row, "answer", source, line_number),
                answer_type=str(row.get("answer_type") or "text"),
                metadata={
                    "manifest_source_id": manifest_source_id,
                    "declared_source_id": row.get("source_id"),
                    **dict(row.get("metadata") or {}),
                },
            )
            _append_unique(examples, seen_ids, example, source, line_number)
            continue

        raise ValueError(
            f"{source}:{line_number} is neither a qa container, grounding row, nor atomic QA"
        )

    if not examples:
        raise ValueError(f"Manifest contains no training examples: {source}")
    return examples


def grouped_holdout_split(
    examples: Sequence[VLMTrainingExample],
    *,
    holdout_fraction: float = 0.25,
    seed: int = 17,
) -> GroupedSplit:
    """Split by source image while covering tasks when the holdout budget permits.

    QA rows never cross images. Deterministic greedy selection gives each task a
    holdout group before filling remaining slots in seeded order.
    """

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if len(examples) < 2:
        raise ValueError("At least two examples are required")

    groups: dict[str, list[VLMTrainingExample]] = {}
    for example in examples:
        groups.setdefault(example.source_id, []).append(example)
    if len(groups) < 2:
        raise ValueError("At least two source groups are required for leakage-safe evaluation")

    group_ids = sorted(groups, key=lambda item: _seeded_digest(item, seed))
    evaluation_count = min(
        len(group_ids) - 1,
        max(1, round(len(group_ids) * holdout_fraction)),
    )
    selected: list[str] = []
    uncovered_tasks = {example.task for example in examples}
    for group_id in group_ids:
        group_tasks = {example.task for example in groups[group_id]}
        if group_tasks & uncovered_tasks and len(selected) < evaluation_count:
            selected.append(group_id)
            uncovered_tasks -= group_tasks
    for group_id in group_ids:
        if len(selected) >= evaluation_count:
            break
        if group_id not in selected:
            selected.append(group_id)
    evaluation_ids = tuple(selected)
    evaluation_set = set(evaluation_ids)
    train_ids = tuple(group_id for group_id in group_ids if group_id not in evaluation_set)
    train: list[VLMTrainingExample] = []
    evaluation: list[VLMTrainingExample] = []
    for example in examples:
        (evaluation if example.source_id in evaluation_set else train).append(example)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return GroupedSplit(tuple(train), tuple(evaluation), train_ids, evaluation_ids)


def limit_examples_by_group(
    examples: Sequence[VLMTrainingExample],
    limit: int | None,
) -> tuple[VLMTrainingExample, ...]:
    """Apply an example budget while keeping deterministic task interleaving."""

    if limit is None:
        return tuple(examples)
    if limit < 1:
        raise ValueError("example limit must be positive")
    buckets: dict[str, list[VLMTrainingExample]] = {}
    for example in examples:
        buckets.setdefault(example.task, []).append(example)
    selected: list[VLMTrainingExample] = []
    while len(selected) < limit and any(buckets.values()):
        for task in sorted(buckets):
            if buckets[task] and len(selected) < limit:
                selected.append(buckets[task].pop(0))
    return tuple(selected)


def build_multimodal_messages(
    example: VLMTrainingExample,
    *,
    include_answer: bool,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(example.image_path)},
                {"type": "text", "text": example.question},
            ],
        }
    )
    if include_answer:
        messages.append({"role": "assistant", "content": example.answer})
    return messages


class Qwen3VLAssistantOnlyCollator:
    """Batch Qwen3-VL conversations with loss only on assistant tokens.

    Qwen3-VL's current local template does not expose Jinja ``generation`` spans.
    We therefore tokenize the prompt-only and complete conversations, verify that
    the former is an exact token prefix of the latter, and mask everything except
    the suffix. This also masks image tokens, system text, user text, and padding.
    """

    def __init__(
        self,
        processor: Any,
        *,
        max_sequence_length: int = 1024,
        system_prompt: str | None = None,
    ) -> None:
        if max_sequence_length < 32:
            raise ValueError("max_sequence_length must be at least 32")
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.system_prompt = system_prompt

    def __call__(self, examples: Sequence[VLMTrainingExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        complete = [
            build_multimodal_messages(item, include_answer=True, system_prompt=self.system_prompt)
            for item in examples
        ]
        prompts = [
            build_multimodal_messages(item, include_answer=False, system_prompt=self.system_prompt)
            for item in examples
        ]
        processor_kwargs = {"padding": True, "return_tensors": "pt"}
        complete_batch = self.processor.apply_chat_template(
            complete,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        prompt_batch = self.processor.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        labels = complete_batch["input_ids"].clone()
        labels.fill_(-100)
        full_attention = complete_batch["attention_mask"]
        prompt_attention = prompt_batch["attention_mask"]

        for row_index, example in enumerate(examples):
            full_positions = full_attention[row_index].bool().nonzero(as_tuple=False).flatten()
            prompt_positions = prompt_attention[row_index].bool().nonzero(as_tuple=False).flatten()
            if len(full_positions) > self.max_sequence_length:
                raise ValueError(
                    f"{example.id} expands to {len(full_positions)} tokens, exceeding "
                    f"max_sequence_length={self.max_sequence_length}; lower max_pixels or shorten text"
                )
            full_ids = complete_batch["input_ids"][row_index, full_positions]
            prompt_ids = prompt_batch["input_ids"][row_index, prompt_positions]
            if len(prompt_ids) >= len(full_ids):
                raise ValueError(f"{example.id} has no assistant answer tokens after processing")
            if not full_ids[: len(prompt_ids)].equal(prompt_ids):
                raise ValueError(
                    f"{example.id} prompt tokens are not a prefix of the complete conversation"
                )
            answer_positions = full_positions[len(prompt_ids) :]
            labels[row_index, answer_positions] = complete_batch["input_ids"][
                row_index, answer_positions
            ]
        complete_batch["labels"] = labels
        return complete_batch


class ToyMultimodalProcessor:
    """Dependency-light processor used to exercise the same collator in CPU tests."""

    pad_token_id = 0
    vocab_size = 263

    def apply_chat_template(
        self,
        conversation: list[Any],
        *,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
        return_tensors: str | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        del return_dict, return_tensors, processor_kwargs
        import torch

        conversations = (
            conversation if conversation and isinstance(conversation[0], list) else [conversation]
        )
        sequences: list[list[int]] = []
        image_features: list[list[float]] = []
        for messages in conversations:
            tokens = [1]
            path = ""
            for message in messages:
                role = message["role"]
                if role == "system":
                    tokens.extend([7, *_byte_tokens(str(message["content"])), 4])
                elif role == "user":
                    tokens.extend([2, 3])
                    for content in message["content"]:
                        if content.get("type") == "image":
                            path = str(content.get("image") or "")
                        elif content.get("type") == "text":
                            tokens.extend(_byte_tokens(str(content.get("text") or "")))
                    tokens.append(4)
                elif role == "assistant":
                    tokens.extend([5, *_byte_tokens(str(message["content"])), 6])
            if add_generation_prompt:
                tokens.append(5)
            sequences.append(tokens)
            image_features.append(_toy_image_features(Path(path)))

        width = max(map(len, sequences))
        input_ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence)
            attention_mask[index, : len(sequence)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.tensor(image_features, dtype=torch.float32),
        }


def _resolve_image_path(source: Path, row: dict[str, Any], line_number: int) -> Path:
    raw = str(row.get("image_path") or "").strip()
    if not raw:
        raise ValueError(f"{source}:{line_number} is missing image_path")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _required_text(row: dict[str, Any], key: str, source: Path, line_number: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{source}:{line_number} is missing {key}")
    return value


def _append_unique(
    examples: list[VLMTrainingExample],
    seen_ids: set[str],
    example: VLMTrainingExample,
    source: Path,
    line_number: int,
) -> None:
    if example.id in seen_ids:
        raise ValueError(f"{source}:{line_number} duplicates example id {example.id!r}")
    seen_ids.add(example.id)
    examples.append(example)


def _image_group_id(path: Path) -> str:
    """Return a content identity when possible, with resolved path fallback.

    The manifest row id is deliberately never used as the split group: separate
    rows can legally ask different questions about the same image. Content
    hashing also catches copied files; missing/unreadable files retain a stable
    resolved-path identity for dependency-light tests.
    """

    resolved = path.expanduser().resolve()
    digest = _cached_image_sha256(str(resolved))
    return f"image-sha256:{digest}" if digest is not None else f"image-path:{resolved}"


@lru_cache(maxsize=4_096)
def _cached_image_sha256(resolved_path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(resolved_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _seeded_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _byte_tokens(value: str) -> Iterable[int]:
    return (byte + 7 for byte in value.encode("utf-8"))


def _toy_image_features(path: Path) -> list[float]:
    payload = path.read_bytes() if path.is_file() else str(path).encode()
    digest = hashlib.sha256(payload).digest()
    return [byte / 255.0 for byte in digest[:8]]
