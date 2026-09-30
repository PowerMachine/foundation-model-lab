from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import random
import re
import socket
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from .inference import Qwen3VLConfig, Qwen3VLRunner
from .schema import GenerationResult


SCHEMA_VERSION = "1.1"
GIB = 1024**3


class TeacherRunner(Protocol):
    def infer(
        self,
        prompt: str,
        media: Sequence[Path | str] = (),
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class DistillationExample:
    id: str
    source_id: str
    task: str
    image_path: Path
    question: str
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_sha256(self) -> str:
        return _sha256_file(self.image_path)

    @property
    def example_key(self) -> str:
        payload = "\n".join((self.task, self.image_sha256, _normalize_text(self.question)))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheFilterConfig:
    min_answer_chars: int = 1
    max_answer_chars: int = 512
    reject_simulated: bool = True
    reject_reference_in_question: bool = True
    reject_placeholders: bool = True
    reject_refusals: bool = True
    deduplicate_inputs: bool = True

    def validate(self) -> None:
        if self.min_answer_chars < 0:
            raise ValueError("min_answer_chars cannot be negative")
        if self.max_answer_chars < self.min_answer_chars:
            raise ValueError("max_answer_chars must be >= min_answer_chars")


@dataclass
class StudentLoRAConfig:
    model_path: Path
    cache_path: Path
    output_dir: Path
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    freeze_vision_encoder: bool = True
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 20
    gradient_accumulation_steps: int = 4
    max_examples: int = 64
    max_sequence_length: int = 2048
    max_minutes: float = 30.0
    seed: int = 7
    bf16: bool = True
    require_cuda: bool = True
    cuda_device: int = 0
    min_free_vram_gib: float = 24.0
    min_pixels: int | None = 65_536
    max_pixels: int | None = 262_144
    audit_path: Path | None = None
    allow_simulated_targets: bool = False
    save_adapter: bool = True
    mock: bool = False

    def validate(self, *, require_files: bool = True) -> None:
        if self.rank < 1 or self.alpha < 1 or not 0 <= self.dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1)")
        if self.max_steps < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("max_steps and gradient_accumulation_steps must be positive")
        if self.max_examples < 1 or self.max_sequence_length < 1 or self.max_minutes <= 0:
            raise ValueError("resource bounds must be positive")
        if self.cuda_device < 0 or self.min_free_vram_gib < 0:
            raise ValueError("cuda_device and min_free_vram_gib cannot be negative")
        if self.min_pixels is not None and self.min_pixels < 1:
            raise ValueError("min_pixels must be positive or None")
        if self.max_pixels is not None and self.max_pixels < 1:
            raise ValueError("max_pixels must be positive or None")
        if (
            self.min_pixels is not None
            and self.max_pixels is not None
            and self.max_pixels < self.min_pixels
        ):
            raise ValueError("max_pixels must be >= min_pixels")

        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if require_files:
            if not self.cache_path.is_file():
                raise FileNotFoundError(self.cache_path)
            if not self.mock and not self.model_path.is_dir():
                raise FileNotFoundError(self.model_path)
            if not self.mock and (self.audit_path is None or not self.audit_path.is_file()):
                raise FileNotFoundError(self.audit_path or "audit_path is required")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("model_path", "cache_path", "output_dir", "audit_path"):
            value[key] = str(value[key]) if value[key] is not None else None
        return value


def load_distillation_examples(
    manifest_path: Path, *, max_examples: int | None = None, require_images: bool = True
) -> list[DistillationExample]:
    """Flatten nested VQA and flat grounding manifests into one stable schema."""

    rows = _read_jsonl(manifest_path)
    examples: list[DistillationExample] = []
    for row in rows:
        source_id = str(row.get("id", "")).strip()
        image_path = Path(str(row.get("image_path", ""))).expanduser()
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        if not source_id or not str(row.get("image_path", "")).strip():
            raise ValueError(f"Manifest row requires id and image_path: {row}")
        if require_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        task = str(row.get("task", "unknown"))
        qa_items = row.get("qa")
        if isinstance(qa_items, list):
            for item in qa_items:
                examples.append(
                    DistillationExample(
                        id=str(item.get("id") or f"{source_id}-{len(examples)}"),
                        source_id=source_id,
                        task=task,
                        image_path=image_path,
                        question=str(item.get("question", "")).strip(),
                        reference_answer=str(item.get("answer", "")).strip(),
                        metadata={
                            "source_metadata": row.get("metadata", {}),
                            "qa_metadata": item.get("metadata", {}),
                            "answer_type": item.get("answer_type", "text"),
                        },
                    )
                )
        elif task == "grounding" or "present" in row:
            object_name = str(row.get("object_name", "object"))
            question = row.get("prompt") or (
                f"Is there a {object_name} in this image? Answer only yes or no."
            )
            examples.append(
                DistillationExample(
                    id=source_id,
                    source_id=source_id,
                    task=task,
                    image_path=image_path,
                    question=str(question),
                    reference_answer="yes" if bool(row.get("present")) else "no",
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {"id", "image_path", "prompt", "task"}
                    },
                )
            )
        else:
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", row.get("reference_answer", ""))).strip()
            if not question:
                raise ValueError(f"Unsupported manifest row without qa/question: {source_id}")
            examples.append(
                DistillationExample(
                    id=source_id,
                    source_id=source_id,
                    task=task,
                    image_path=image_path,
                    question=question,
                    reference_answer=answer,
                    metadata=dict(row.get("metadata", {})),
                )
            )
        if max_examples is not None and len(examples) >= max_examples:
            return examples[:max_examples]
    return examples


def cache_teacher_responses(
    manifest_path: Path,
    output_path: Path,
    teacher: TeacherRunner,
    *,
    max_examples: int | None = None,
    max_minutes: float | None = 60.0,
    resume: bool = True,
    simulation_context: bool = False,
    system_prompt: str = (
        "Answer the visual question accurately and concisely. Do not mention hidden labels."
    ),
    teacher_descriptor: dict[str, Any] | None = None,
    split_seed: int = 17,
    eval_fraction: float = 0.25,
    cuda_device: int = 0,
    min_free_vram_gib: float | None = None,
) -> dict[str, Any]:
    """Generate a contract-bound teacher cache with image-grouped train/eval splits.

    A resume is accepted only when the complete generation contract and every cached
    row still match the manifest and image bytes.  The production teacher should run
    in its own process so its VRAM is released before student training starts.
    """

    started_wall = _utc_now()
    started = time.perf_counter()
    if max_minutes is not None and max_minutes <= 0:
        raise ValueError("max_minutes must be positive or None")
    if type(simulation_context) is not bool:
        raise TypeError("simulation_context must be an exact boolean")
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    if cuda_device < 0:
        raise ValueError("cuda_device cannot be negative")

    examples = load_distillation_examples(manifest_path, max_examples=max_examples)
    keys = [example.example_key for example in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest produces duplicate example keys")
    split_by_image = assign_image_grouped_splits(
        examples, seed=split_seed, eval_fraction=eval_fraction
    )
    split_summary = _split_summary(examples, split_by_image)
    descriptor = teacher_descriptor or {"class": type(teacher).__name__}
    contract = _teacher_cache_contract(
        manifest_path=manifest_path,
        teacher_descriptor=descriptor,
        system_prompt=system_prompt,
        simulation_context=simulation_context,
        max_examples=max_examples,
        split_seed=split_seed,
        eval_fraction=eval_fraction,
    )
    contract_sha256 = _canonical_sha256(contract)
    provenance_path = output_path.with_name(f"{output_path.stem}.provenance.json")
    preflight_path = output_path.with_name(f"{output_path.stem}.preflight.json")

    existing = _read_jsonl(output_path) if resume and output_path.is_file() else []
    if resume and output_path.is_file():
        if not provenance_path.is_file():
            raise ValueError("Cannot resume cache without its provenance sidecar")
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("output_sha256") != _sha256_file(output_path):
            raise ValueError("Teacher cache content SHA does not match provenance")
        if previous.get("contract_sha256") != contract_sha256:
            raise ValueError("Teacher cache resume contract mismatch")
        _validate_resumable_rows(
            existing,
            examples,
            split_by_image=split_by_image,
            contract_sha256=contract_sha256,
            split_seed=split_seed,
            eval_fraction=eval_fraction,
        )

    rows = list(existing)
    completed_keys = {str(row["example_key"]) for row in existing}
    new_examples = [example for example in examples if example.example_key not in completed_keys]
    deadline = started + max_minutes * 60 if max_minutes is not None else None
    generated = 0
    failures: list[dict[str, str]] = []
    stopped_by_time_budget = False
    resource_preflight: dict[str, Any] = {"status": "not_requested"}

    if not new_examples:
        _write_json(
            preflight_path,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": "teacher_current_process_preflight",
                "status": "skipped_no_new_work",
                "created_at": _utc_now(),
                "contract_sha256": contract_sha256,
            },
        )
    elif min_free_vram_gib is not None:
        resource_preflight = cuda_vram_preflight(
            cuda_device=cuda_device, min_free_vram_gib=min_free_vram_gib
        )
        _write_json(
            preflight_path,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": "teacher_resource_preflight",
                "created_at": _utc_now(),
                "contract_sha256": contract_sha256,
                **resource_preflight,
            },
        )
        if resource_preflight["status"] != "ready":
            if not output_path.is_file():
                _write_jsonl_atomic(output_path, rows)
            provenance = _teacher_provenance_payload(
                status="blocked_resource",
                manifest_path=manifest_path,
                output_path=output_path,
                provenance_path=provenance_path,
                preflight_path=preflight_path,
                descriptor=descriptor,
                contract=contract,
                contract_sha256=contract_sha256,
                system_prompt=system_prompt,
                examples=examples,
                rows=rows,
                existing=existing,
                generated=0,
                max_minutes=max_minutes,
                stopped_by_time_budget=False,
                failures=[],
                split_summary=split_summary,
                started_wall=started_wall,
                started=started,
                resource_preflight=resource_preflight,
            )
            _write_json(provenance_path, provenance)
            return {**provenance, "provenance_path": str(provenance_path.resolve())}

    current_process_preflight_passed = not new_examples
    for example in new_examples:
        if deadline is not None and time.perf_counter() >= deadline:
            stopped_by_time_budget = True
            break
        context = {"expected_answer": example.reference_answer} if simulation_context else None
        try:
            result = teacher.infer(
                example.question,
                [example.image_path],
                context=context,
                system_prompt=system_prompt,
            )
            if type(result.simulated) is not bool:
                raise TypeError("teacher result simulated flag must be an exact boolean")
        except Exception as exc:
            failure = {"id": example.id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            if not current_process_preflight_passed:
                _write_json(
                    preflight_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stage": "teacher_one_request_preflight",
                        "status": "failed",
                        "created_at": _utc_now(),
                        "contract_sha256": contract_sha256,
                        "example_id": example.id,
                        "error": failure["error"],
                        "teacher": descriptor,
                        "resource_preflight": resource_preflight,
                        "actionable_checks": [
                            "Confirm the local model directory and config.json are readable.",
                            "For FP8, verify Transformers FineGrainedFP8 support and CUDA capability.",
                            "Check selected-device free VRAM and stop unrelated workloads first.",
                        ],
                    },
                )
                break
            continue

        if not current_process_preflight_passed:
            _write_json(
                preflight_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "teacher_one_request_preflight",
                    "status": "complete",
                    "created_at": _utc_now(),
                    "contract_sha256": contract_sha256,
                    "example_id": example.id,
                    "backend": result.backend,
                    "latency_seconds": result.latency_seconds,
                    "simulated": result.simulated,
                    "resource_preflight": resource_preflight,
                },
            )
            current_process_preflight_passed = True

        image_sha256 = example.image_sha256
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "cache_contract_sha256": contract_sha256,
                "id": example.id,
                "source_id": example.source_id,
                "example_key": example.example_key,
                "task": example.task,
                "split": split_by_image[image_sha256],
                "split_seed": split_seed,
                "eval_fraction": eval_fraction,
                "image_path": str(example.image_path.resolve()),
                "image_sha256": image_sha256,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "teacher_answer": result.text.strip(),
                "teacher_backend": result.backend,
                "teacher_simulated": result.simulated,
                "teacher_latency_seconds": result.latency_seconds,
                "teacher_metadata": result.metadata,
                "source_metadata": example.metadata,
                "generated_at": _utc_now(),
            }
        )
        generated += 1
        completed_keys.add(example.example_key)
        _write_jsonl_atomic(output_path, rows)

    _write_jsonl_atomic(output_path, rows)
    requested_keys = set(keys)
    cached_requested = completed_keys & requested_keys
    if len(cached_requested) == len(requested_keys) and not failures:
        status = "complete"
    elif rows:
        status = "partial"
    else:
        status = "failed"
    provenance = _teacher_provenance_payload(
        status=status,
        manifest_path=manifest_path,
        output_path=output_path,
        provenance_path=provenance_path,
        preflight_path=preflight_path,
        descriptor=descriptor,
        contract=contract,
        contract_sha256=contract_sha256,
        system_prompt=system_prompt,
        examples=examples,
        rows=rows,
        existing=existing,
        generated=generated,
        max_minutes=max_minutes,
        stopped_by_time_budget=stopped_by_time_budget,
        failures=failures,
        split_summary=split_summary,
        started_wall=started_wall,
        started=started,
        resource_preflight=resource_preflight,
    )
    _write_json(provenance_path, provenance)
    return {**provenance, "provenance_path": str(provenance_path.resolve())}


def audit_teacher_cache(
    raw_cache_path: Path,
    accepted_cache_path: Path,
    audit_path: Path,
    *,
    filters: CacheFilterConfig | None = None,
) -> dict[str, Any]:
    """Verify image/schema integrity, then filter weak teacher responses."""

    config = filters or CacheFilterConfig()
    config.validate()
    started = time.perf_counter()
    rows = _read_jsonl(raw_cache_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasons: Counter[str] = Counter()
    agreements: list[float] = []
    token_f1_scores: list[float] = []
    lengths: list[int] = []
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    split_images: dict[str, set[str]] = {"train": set(), "eval": set()}
    teacher_provenance_path = raw_cache_path.with_name(f"{raw_cache_path.stem}.provenance.json")
    if teacher_provenance_path.is_file():
        teacher_provenance = json.loads(teacher_provenance_path.read_text(encoding="utf-8"))
        if teacher_provenance.get("output_sha256") != _sha256_file(raw_cache_path):
            raise ValueError("Raw teacher cache SHA does not match teacher provenance")

    for row in rows:
        raw_answer = row.get("teacher_answer")
        answer = raw_answer.strip() if isinstance(raw_answer, str) else ""
        question = str(row.get("question", "")).strip()
        reference = str(row.get("reference_answer", "")).strip()
        key = str(row.get("example_key") or _fallback_example_key(row))
        row_reasons: list[str] = []
        if not isinstance(raw_answer, str):
            row_reasons.append("invalid_teacher_answer")
        if len(answer) < config.min_answer_chars:
            row_reasons.append("empty_or_too_short")
        if len(answer) > config.max_answer_chars:
            row_reasons.append("too_long")
        simulated = row.get("teacher_simulated")
        if type(simulated) is not bool:
            row_reasons.append("invalid_teacher_simulated")
        elif config.reject_simulated and simulated:
            row_reasons.append("simulated_target")
        split = row.get("split")
        if split not in {"train", "eval"}:
            row_reasons.append("invalid_split")
        normalized_answer = _normalize_text(answer)
        if config.reject_placeholders and normalized_answer in {
            "unknown",
            "n a",
            "na",
            "none",
            "null",
            "i dont know",
            "i don t know",
            "i do not know",
        }:
            row_reasons.append("placeholder")
        if config.reject_refusals and re.search(
            r"\b(?:cannot|can't|unable to|sorry|not able to|as an ai)\b", answer, re.I
        ):
            row_reasons.append("refusal")
        normalized_reference = _normalize_text(reference)
        normalized_question = _normalize_text(question)
        if (
            config.reject_reference_in_question
            and len(normalized_reference) >= 3
            and normalized_reference not in {"yes", "no", "true", "false"}
            and normalized_reference in normalized_question
        ):
            row_reasons.append("reference_leaked_in_question")
        if config.deduplicate_inputs and key in seen:
            row_reasons.append("duplicate_input")
        image_path = Path(str(row.get("image_path", "")))
        image_sha256 = row.get("image_sha256")
        if not image_path.is_file():
            row_reasons.append("missing_image")
        elif not isinstance(image_sha256, str) or len(image_sha256) != 64:
            row_reasons.append("missing_image_sha256")
        elif _sha256_file(image_path) != image_sha256:
            row_reasons.append("image_sha256_mismatch")

        task = str(row.get("task", "unknown"))
        if row_reasons:
            for reason in sorted(set(row_reasons)):
                reasons[reason] += 1
                per_task[task][f"rejected:{reason}"] += 1
            rejected.append({**row, "rejection_reasons": sorted(set(row_reasons))})
            continue
        seen.add(key)
        assert isinstance(split, str) and isinstance(image_sha256, str)
        split_images[split].add(image_sha256)
        exact = float(normalized_answer == normalized_reference) if reference else math.nan
        f1 = _token_f1(answer, reference) if reference else math.nan
        accepted.append({**row, "audit_exact_match": exact, "audit_token_f1": f1})
        lengths.append(len(answer))
        per_task[task]["accepted"] += 1
        if not math.isnan(exact):
            agreements.append(exact)
            token_f1_scores.append(f1)

    overlap = split_images["train"] & split_images["eval"]
    if overlap:
        raise ValueError(f"Image leakage across accepted train/eval rows: {sorted(overlap)}")
    _write_jsonl_atomic(accepted_cache_path, accepted)
    rejected_path = accepted_cache_path.with_name(f"{accepted_cache_path.stem}.rejected.jsonl")
    _write_jsonl_atomic(rejected_path, rejected)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "stage": "teacher_cache_audit",
        "status": "complete" if rows else "failed",
        "claim_level": "wiring",
        "quality_evaluated": False,
        "created_at": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "filters": asdict(config),
        "counts": {
            "raw": len(rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
        },
        "split_integrity": {
            "train_examples": sum(row.get("split") == "train" for row in accepted),
            "eval_examples": sum(row.get("split") == "eval" for row in accepted),
            "train_image_groups": len(split_images["train"]),
            "eval_image_groups": len(split_images["eval"]),
            "image_overlap_count": 0,
            "image_overlap_sha256": [],
        },
        "rejection_reasons": dict(sorted(reasons.items())),
        "quality_audit_not_a_training_filter": {
            "reference_exact_match": _safe_mean(agreements),
            "reference_token_f1": _safe_mean(token_f1_scores),
            "note": "Teacher/reference disagreement is reported, not silently removed.",
        },
        "answer_length_chars": {
            "mean": _safe_mean(lengths),
            "median": statistics.median(lengths) if lengths else 0.0,
            "max": max(lengths, default=0),
        },
        "per_task": {task: dict(counts) for task, counts in sorted(per_task.items())},
        "artifacts": {
            "raw_cache": str(raw_cache_path.resolve()),
            "raw_sha256": _sha256_file(raw_cache_path),
            "accepted_cache": str(accepted_cache_path.resolve()),
            "accepted_sha256": _sha256_file(accepted_cache_path),
            "rejected_cache": str(rejected_path.resolve()),
            "teacher_provenance": (
                str(teacher_provenance_path.resolve())
                if teacher_provenance_path.is_file()
                else None
            ),
        },
    }
    _write_json(audit_path, audit)
    return audit


def train_student_lora(config: StudentLoRAConfig) -> dict[str, Any]:
    """Train the 8B adapter only from audited rows explicitly assigned to train."""

    config.validate()
    all_rows = _read_jsonl(config.cache_path)
    if not all_rows:
        raise ValueError("No accepted teacher targets are available")
    invalid_flags = [
        index
        for index, row in enumerate(all_rows)
        if type(row.get("teacher_simulated")) is not bool
    ]
    if invalid_flags:
        raise ValueError(f"Invalid teacher_simulated boolean at rows {invalid_flags}")
    simulated_count = sum(row["teacher_simulated"] for row in all_rows)
    if simulated_count and not config.allow_simulated_targets:
        raise ValueError(
            f"Refusing to train on {simulated_count} simulated targets; "
            "set allow_simulated_targets only for pipeline tests"
        )

    cache_sha256 = _sha256_file(config.cache_path)
    audit_verification: dict[str, Any]
    if config.audit_path is not None:
        audit_verification = _verify_accepted_cache_audit(
            config.cache_path, config.audit_path, cache_sha256=cache_sha256, row_count=len(all_rows)
        )
    elif config.mock:
        audit_verification = {"status": "not_required_for_mock"}
    else:  # guarded by config.validate; kept explicit for direct invariants
        raise ValueError("Actual student training requires audit_path")

    train_rows = _validated_student_train_rows(all_rows, allow_missing_split=config.mock)
    train_rows = train_rows[: config.max_examples]
    if not train_rows:
        raise ValueError("No rows explicitly assigned to split=train are available")
    if config.mock:
        return _train_mock(
            config,
            train_rows,
            cache_sha256=cache_sha256,
            audit_verification=audit_verification,
        )

    resource_preflight = cuda_vram_preflight(
        cuda_device=config.cuda_device,
        min_free_vram_gib=config.min_free_vram_gib,
    )
    if resource_preflight["status"] != "ready":
        return _blocked_student_training(
            config,
            train_rows,
            cache_sha256=cache_sha256,
            audit_verification=audit_verification,
            resource_preflight=resource_preflight,
        )
    return _train_actual(
        config,
        train_rows,
        cache_sha256=cache_sha256,
        audit_verification=audit_verification,
        resource_preflight=resource_preflight,
    )


def _verify_accepted_cache_audit(
    cache_path: Path,
    audit_path: Path,
    *,
    cache_sha256: str,
    row_count: int,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise ValueError("Teacher cache audit is not complete")
    artifacts = audit.get("artifacts", {})
    counts = audit.get("counts", {})
    if artifacts.get("accepted_sha256") != cache_sha256:
        raise ValueError("Accepted cache SHA does not match audit")
    if counts.get("accepted") != row_count:
        raise ValueError("Accepted cache row count does not match audit")
    audited_path = artifacts.get("accepted_cache")
    if not audited_path or Path(str(audited_path)).resolve() != cache_path.resolve():
        raise ValueError("Accepted cache path does not match audit")
    return {
        "status": "verified",
        "audit_path": str(audit_path.resolve()),
        "accepted_sha256": cache_sha256,
        "accepted_count": row_count,
    }


def _validated_student_train_rows(
    rows: Sequence[dict[str, Any]], *, allow_missing_split: bool
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    split_images: dict[str, set[str]] = {"train": set(), "eval": set()}
    for index, original in enumerate(rows):
        row = dict(original)
        split = row.get("split")
        if split is None and allow_missing_split:
            split = "train"
            row["split"] = split
        if split not in {"train", "eval"}:
            raise ValueError(f"Invalid or missing split at accepted-cache row {index}")
        image_path = Path(str(row.get("image_path", "")))
        image_sha256 = row.get("image_sha256")
        if not image_path.is_file():
            raise ValueError(f"Missing image at accepted-cache row {index}")
        if not isinstance(image_sha256, str) or _sha256_file(image_path) != image_sha256:
            raise ValueError(f"Image SHA mismatch at accepted-cache row {index}")
        split_images[str(split)].add(image_sha256)
        normalized.append(row)
    overlap = split_images["train"] & split_images["eval"]
    if overlap:
        raise ValueError(f"Image leakage across student train/eval rows: {sorted(overlap)}")
    return [row for row in normalized if row["split"] == "train"]


def _blocked_student_training(
    config: StudentLoRAConfig,
    rows: list[dict[str, Any]],
    *,
    cache_sha256: str,
    audit_verification: dict[str, Any],
    resource_preflight: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _training_metrics_base(
        config,
        rows,
        [],
        started,
        cache_sha256=cache_sha256,
        audit_verification=audit_verification,
    )
    metrics.update(
        {
            "status": "blocked_resource",
            "simulated": False,
            "quality_evaluated": False,
            "claim_level": "wiring",
            "resource_preflight": resource_preflight,
            "adapter_saved": None,
            "trainable_parameters": None,
            "total_parameters": None,
            "trainable_percent": None,
            "peak_gpu_allocated_bytes": 0,
            "gpu": None,
        }
    )
    _write_json(config.output_dir / "student.preflight.json", resource_preflight)
    _write_json(config.output_dir / "training_metrics.json", metrics)
    return metrics


def _train_mock(
    config: StudentLoRAConfig,
    rows: list[dict[str, Any]],
    *,
    cache_sha256: str,
    audit_verification: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(config.seed)
    losses = []
    for step in range(1, config.max_steps + 1):
        loss = 2.2 * math.exp(-0.18 * step) + 0.22 + rng.uniform(-0.015, 0.015)
        losses.append({"step": step, "loss": round(loss, 6)})
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mock_adapter = config.output_dir / "mock_adapter.json"
    _write_json(
        mock_adapter,
        {
            "warning": "Deterministic pipeline artifact; contains no learned weights.",
            "rank": config.rank,
            "seed": config.seed,
        },
    )
    metrics = _training_metrics_base(
        config,
        rows,
        losses,
        started,
        cache_sha256=cache_sha256,
        audit_verification=audit_verification,
    )
    metrics.update(
        {
            "status": "complete",
            "simulated": True,
            "claim_level": "wiring",
            "quality_evaluated": False,
            "adapter_saved": str(mock_adapter),
            "trainable_parameters": None,
            "total_parameters": None,
            "trainable_percent": None,
            "parameter_counts_status": "not_measured_mock",
            "peak_gpu_allocated_bytes": 0,
            "gpu": None,
        }
    )
    _write_json(config.output_dir / "training_metrics.json", metrics)
    return metrics


def _train_actual(
    config: StudentLoRAConfig,
    rows: list[dict[str, Any]],
    *,
    cache_sha256: str,
    audit_verification: dict[str, Any],
    resource_preflight: dict[str, Any],
) -> dict[str, Any]:
    if resource_preflight.get("status") != "ready":
        raise AssertionError("Actual model load reached without a passing VRAM preflight")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError as exc:  # pragma: no cover - heavyweight optional path
        raise RuntimeError(
            "Actual student training requires torch, transformers, and peft"
        ) from exc

    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this bounded 8B recipe")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(config.cuda_device)
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(config.cuda_device)
    started = time.perf_counter()
    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_bf16_supported() else torch.float16
    device_map: Any = {"": config.cuda_device} if torch.cuda.is_available() else None
    model = ModelClass.from_pretrained(
        str(config.model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    processor_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if config.min_pixels is not None:
        processor_kwargs["min_pixels"] = config.min_pixels
    if config.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.max_pixels
    processor = AutoProcessor.from_pretrained(str(config.model_path), **processor_kwargs)
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, lora)
    if config.freeze_vision_encoder:
        for name, parameter in model.named_parameters():
            if "visual" in name.casefold() or "vision" in name.casefold():
                parameter.requires_grad = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    micro_step = 0
    deadline = started + config.max_minutes * 60

    for step in range(1, config.max_steps + 1):
        step_losses: list[float] = []
        for _ in range(config.gradient_accumulation_steps):
            row = rows[micro_step % len(rows)]
            micro_step += 1
            try:
                batch = _encode_training_row(processor, row, config.max_sequence_length)
            except ValueError as exc:
                skipped[str(exc)] += 1
                continue
            device = trainable_parameters[0].device
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            output = model(**batch)
            output.loss.backward()
            step_losses.append(float(output.loss.detach().cpu()))
        if not step_losses:
            raise RuntimeError(f"No trainable examples remained; skipped={dict(skipped)}")
        # Divide by the number of successful micro-batches, not the requested count.
        # This preserves average-loss scaling when an over-length row is skipped.
        for parameter in trainable_parameters:
            if parameter.grad is not None:
                parameter.grad.div_(len(step_losses))
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append({"step": step, "loss": _safe_mean(step_losses)})
        if time.perf_counter() >= deadline:
            break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path: str | None = None
    if config.save_adapter:
        adapter_dir = config.output_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(config.output_dir / "processor")
        adapter_path = str(adapter_dir)
    metrics = _training_metrics_base(
        config,
        rows,
        losses,
        started,
        cache_sha256=cache_sha256,
        audit_verification=audit_verification,
    )
    metrics.update(
        {
            "status": "complete",
            "simulated": False,
            "claim_level": "wiring",
            "quality_evaluated": False,
            "resource_preflight": resource_preflight,
            "adapter_saved": adapter_path,
            "steps_completed": len(losses),
            "stopped_by_time_budget": len(losses) < config.max_steps,
            "skipped_examples": dict(skipped),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": 100.0 * trainable / total if total else 0.0,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(config.cuda_device))
                if torch.cuda.is_available()
                else 0
            ),
            "gpu": (
                torch.cuda.get_device_name(config.cuda_device)
                if torch.cuda.is_available()
                else None
            ),
        }
    )
    _write_json(config.output_dir / "training_metrics.json", metrics)
    return metrics


def _encode_training_row(processor: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    image_path = str(row["image_path"])
    user_content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": str(row["question"])},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": str(row["teacher_answer"])},
    ]
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    sequence_length = int(full["input_ids"].shape[-1])
    if sequence_length > max_length:
        raise ValueError("sequence_too_long")
    prompt_length = int(prompt["input_ids"].shape[-1])
    prefix_length = 0
    for prompt_id, full_id in zip(prompt["input_ids"][0], full["input_ids"][0], strict=False):
        if int(prompt_id) != int(full_id):
            break
        prefix_length += 1
    if prefix_length != prompt_length:
        raise ValueError("prompt_not_exact_prefix")
    labels = full["input_ids"].clone()
    labels[:, :prefix_length] = -100
    if "attention_mask" in full:
        labels[full["attention_mask"].eq(0)] = -100
    if not labels.ne(-100).any():
        raise ValueError("no_assistant_tokens")
    full["labels"] = labels
    return dict(full)


def write_distillation_report(
    output_dir: Path, audit: dict[str, Any], training: dict[str, Any]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = _write_loss_svg(output_dir / "loss_curve.svg", training.get("losses", []))
    training_status = training.get("status")
    if training_status in {"complete", "completed"}:
        report_status = "success"
    elif training_status == "blocked_resource":
        report_status = "blocked_resource"
    else:
        report_status = "failed"
    result = {
        "status": report_status,
        "simulated": bool(training.get("simulated")),
        "experiment": "vlm_response_distillation",
        "claim_level": "wiring",
        "quality_evaluated": False,
        "metrics": {
            "raw_teacher_examples": audit.get("counts", {}).get("raw", 0),
            "accepted_teacher_examples": audit.get("counts", {}).get("accepted", 0),
            "teacher_acceptance_rate": audit.get("counts", {}).get("acceptance_rate", 0.0),
            "teacher_reference_exact_match": audit.get(
                "quality_audit_not_a_training_filter", {}
            ).get("reference_exact_match", 0.0),
            "initial_loss": training.get("initial_loss", 0.0),
            "final_loss": training.get("final_loss", 0.0),
            "steps_completed": training.get("steps_completed", len(training.get("losses", []))),
            "trainable_percent": training.get("trainable_percent"),
            "peak_gpu_allocated_bytes": training.get("peak_gpu_allocated_bytes", 0),
            "duration_seconds": training.get("duration_seconds", 0.0),
        },
        "provenance": {
            "teacher_cache_sha256": audit.get("artifacts", {}).get("raw_sha256"),
            "accepted_cache_sha256": training.get("cache_sha256"),
            "audit_verification": training.get("audit_verification"),
            "student_model": training.get("config", {}).get("model_path"),
            "runtime": training.get("runtime", {}),
        },
        "artifacts": {
            "report": str((output_dir / "report.html").resolve()),
            "loss_curve": str(chart_path.resolve()),
            "training_metrics": str(
                (
                    Path(training.get("config", {}).get("output_dir", output_dir))
                    / "training_metrics.json"
                ).resolve()
            ),
            **audit.get("artifacts", {}),
        },
        "interpretation": (
            "Mock metrics validate orchestration only; they are not model quality evidence."
            if training.get("simulated")
            else "Response-level distillation trains on cached sequences, not teacher logits."
        ),
    }
    _write_json(output_dir / "result.json", result)
    _write_report_html(output_dir / "report.html", result, audit, training, chart_path)
    return result


def run_mock_distillation(
    manifest_path: Path, output_dir: Path, *, max_examples: int = 12, max_steps: int = 5
) -> dict[str, Any]:
    """Run the complete graph deterministically without allocating a model or GPU."""

    cache_dir = output_dir / "cache"
    raw_cache = cache_dir / "teacher_raw.jsonl"
    accepted_cache = cache_dir / "teacher_accepted.jsonl"
    audit_path = cache_dir / "audit.json"
    teacher = Qwen3VLRunner(Qwen3VLConfig(mode="offline"))
    cache_teacher_responses(
        manifest_path,
        raw_cache,
        teacher,
        max_examples=max_examples,
        resume=False,
        simulation_context=True,
        teacher_descriptor={"mode": "offline", "purpose": "deterministic pipeline test"},
    )
    audit = audit_teacher_cache(
        raw_cache,
        accepted_cache,
        audit_path,
        filters=CacheFilterConfig(reject_simulated=False),
    )
    training_dir = output_dir / "student"
    training = train_student_lora(
        StudentLoRAConfig(
            model_path=Path("mock-qwen3-vl-8b"),
            cache_path=accepted_cache,
            output_dir=training_dir,
            audit_path=audit_path,
            max_steps=max_steps,
            max_examples=max_examples,
            allow_simulated_targets=True,
            mock=True,
        )
    )
    return write_distillation_report(output_dir, audit, training)


def _training_metrics_base(
    config: StudentLoRAConfig,
    rows: list[dict[str, Any]],
    losses: list[dict[str, Any]],
    started: float,
    *,
    cache_sha256: str,
    audit_verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "student_lora_training",
        "claim_level": "wiring",
        "quality_evaluated": False,
        "created_at": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "config": config.to_dict(),
        "cache_sha256": cache_sha256,
        "audit_verification": audit_verification,
        "examples_loaded": len(rows),
        "train_examples_loaded": len(rows),
        "steps_completed": len(losses),
        "losses": losses,
        "initial_loss": losses[0]["loss"] if losses else 0.0,
        "final_loss": losses[-1]["loss"] if losses else 0.0,
        "runtime": _runtime_snapshot(),
        "base_model_fingerprint": (
            _model_metadata_fingerprint(config.model_path)
            if config.model_path.is_dir()
            else {"path": str(config.model_path), "mock": True}
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_image_grouped_splits(
    examples: Sequence[DistillationExample], *, seed: int, eval_fraction: float
) -> dict[str, str]:
    """Assign complete image groups to deterministic train/eval partitions."""

    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    image_hashes = sorted({example.image_sha256 for example in examples})
    if not image_hashes:
        return {}
    ranked = sorted(
        image_hashes,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    eval_groups = 0
    if eval_fraction > 0.0 and len(ranked) > 1:
        eval_groups = max(1, min(len(ranked) - 1, round(len(ranked) * eval_fraction)))
    eval_hashes = set(ranked[:eval_groups])
    assignment = {
        image_sha256: "eval" if image_sha256 in eval_hashes else "train"
        for image_sha256 in image_hashes
    }
    _split_summary(examples, assignment)
    return assignment


def _split_summary(
    examples: Sequence[DistillationExample], split_by_image: dict[str, str]
) -> dict[str, Any]:
    groups: dict[str, set[str]] = {"train": set(), "eval": set()}
    example_counts: Counter[str] = Counter()
    for example in examples:
        image_sha256 = example.image_sha256
        split = split_by_image.get(image_sha256)
        if split not in groups:
            raise ValueError(f"Invalid or absent split for image {image_sha256}")
        groups[split].add(image_sha256)
        example_counts[split] += 1
    overlap = groups["train"] & groups["eval"]
    if overlap:
        raise AssertionError(f"Image leakage across train/eval: {sorted(overlap)}")
    return {
        "train_examples": example_counts["train"],
        "eval_examples": example_counts["eval"],
        "train_image_groups": len(groups["train"]),
        "eval_image_groups": len(groups["eval"]),
        "image_overlap_count": 0,
        "image_overlap_sha256": [],
    }


def _teacher_cache_contract(
    *,
    manifest_path: Path,
    teacher_descriptor: dict[str, Any],
    system_prompt: str,
    simulation_context: bool,
    max_examples: int | None,
    split_seed: int,
    eval_fraction: float,
) -> dict[str, Any]:
    return {
        "cache_schema_version": SCHEMA_VERSION,
        "manifest_sha256": _sha256_file(manifest_path),
        "teacher_descriptor": teacher_descriptor,
        "teacher_generation": teacher_descriptor.get("generation"),
        "system_prompt": system_prompt,
        "simulation_context": simulation_context,
        "max_examples": max_examples,
        "split": {
            "method": "image_sha256_grouped",
            "seed": split_seed,
            "eval_fraction": eval_fraction,
        },
    }


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resumable_rows(
    rows: Sequence[dict[str, Any]],
    examples: Sequence[DistillationExample],
    *,
    split_by_image: dict[str, str],
    contract_sha256: str,
    split_seed: int,
    eval_fraction: float,
) -> None:
    current = {example.example_key: example for example in examples}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        key = row.get("example_key")
        if not isinstance(key, str) or key not in current:
            raise ValueError(f"Stale teacher cache row at index {index}")
        if key in seen:
            raise ValueError(f"Duplicate teacher cache example_key: {key}")
        seen.add(key)
        example = current[key]
        image_sha256 = example.image_sha256
        expected = {
            "schema_version": SCHEMA_VERSION,
            "cache_contract_sha256": contract_sha256,
            "source_id": example.source_id,
            "task": example.task,
            "image_path": str(example.image_path.resolve()),
            "image_sha256": image_sha256,
            "question": example.question,
            "reference_answer": example.reference_answer,
            "split": split_by_image[image_sha256],
            "split_seed": split_seed,
            "eval_fraction": eval_fraction,
        }
        mismatches = [name for name, value in expected.items() if row.get(name) != value]
        if mismatches:
            raise ValueError(
                f"Teacher cache row {key} disagrees with current contract: {mismatches}"
            )
        if type(row.get("teacher_simulated")) is not bool:
            raise ValueError(f"Teacher cache row {key} has invalid teacher_simulated flag")
        if not isinstance(row.get("teacher_answer"), str):
            raise ValueError(f"Teacher cache row {key} has a non-string answer")
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file() or _sha256_file(image_path) != row["image_sha256"]:
            raise ValueError(f"Teacher cache row {key} image bytes changed")


def _teacher_provenance_payload(
    *,
    status: str,
    manifest_path: Path,
    output_path: Path,
    provenance_path: Path,
    preflight_path: Path,
    descriptor: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    system_prompt: str,
    examples: Sequence[DistillationExample],
    rows: Sequence[dict[str, Any]],
    existing: Sequence[dict[str, Any]],
    generated: int,
    max_minutes: float | None,
    stopped_by_time_budget: bool,
    failures: Sequence[dict[str, str]],
    split_summary: dict[str, Any],
    started_wall: str,
    started: float,
    resource_preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "teacher_response_cache",
        "status": status,
        "claim_level": "wiring",
        "quality_evaluated": False,
        "started_at": started_wall,
        "finished_at": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256_file(output_path),
        "provenance_path": str(provenance_path.resolve()),
        "teacher": descriptor,
        "generation_contract": contract,
        "contract_sha256": contract_sha256,
        "one_request_preflight": str(preflight_path.resolve()),
        "resource_preflight": resource_preflight,
        "system_prompt": system_prompt,
        "requested_examples": len(examples),
        "cached_examples": len(rows),
        "generated_this_run": generated,
        "resumed_examples": len(existing),
        "max_minutes": max_minutes,
        "stopped_by_time_budget": stopped_by_time_budget,
        "failures": list(failures),
        "split": {**contract["split"], **split_summary},
        "mean_latency_seconds": _safe_mean(
            [float(row.get("teacher_latency_seconds", 0.0)) for row in rows]
        ),
        "runtime": _runtime_snapshot(),
        "gpu_resources": _gpu_resource_snapshot(),
    }


def _probe_cuda_memory(cuda_device: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch_not_installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda_unavailable", "device_count": 0}
    device_count = int(torch.cuda.device_count())
    if cuda_device >= device_count:
        return {
            "available": False,
            "reason": "cuda_device_out_of_range",
            "device_count": device_count,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    return {
        "available": True,
        "device_count": device_count,
        "name": torch.cuda.get_device_name(cuda_device),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def cuda_vram_preflight(
    *,
    cuda_device: int,
    min_free_vram_gib: float,
    probe: Any | None = None,
) -> dict[str, Any]:
    """Check the selected device before any teacher/student model load."""

    if cuda_device < 0 or min_free_vram_gib < 0:
        raise ValueError("cuda_device and min_free_vram_gib cannot be negative")
    snapshot = (probe or _probe_cuda_memory)(cuda_device)
    required_bytes = int(min_free_vram_gib * GIB)
    base = {
        "selected_cuda_device": cuda_device,
        "min_free_vram_gib": min_free_vram_gib,
        "required_free_bytes": required_bytes,
        **snapshot,
    }
    if not snapshot.get("available"):
        return {"status": "blocked_resource", **base}
    free_bytes = int(snapshot.get("free_bytes", 0))
    return {
        "status": "ready" if free_bytes >= required_bytes else "blocked_resource",
        **base,
        "free_vram_gib": free_bytes / GIB,
        "total_vram_gib": int(snapshot.get("total_bytes", 0)) / GIB,
        "reason": None if free_bytes >= required_bytes else "insufficient_free_vram",
    }


def inspect_local_teacher_runtime(model_path: Path) -> dict[str, Any]:
    """Static FP8/runtime inspection; a real request preflight is still required."""

    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") or {}
    quantization_method = str(quantization.get("quant_method", "none"))
    fp8_quantizer_available = False
    quantizer_error: str | None = None
    if quantization_method == "fp8":
        try:
            from transformers.quantizers.quantizer_finegrained_fp8 import (
                FineGrainedFP8HfQuantizer,
            )

            _ = FineGrainedFP8HfQuantizer
            fp8_quantizer_available = True
        except (ImportError, RuntimeError) as exc:
            quantizer_error = f"{type(exc).__name__}: {exc}"
    gpu: dict[str, Any] | None = None
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": int(properties.total_memory),
            }
    except ImportError:
        pass
    if quantization_method == "fp8" and not fp8_quantizer_available:
        static_status = "blocked_missing_fp8_quantizer"
    elif gpu is None:
        static_status = "incomplete_no_cuda_device"
    else:
        static_status = "static_checks_passed_request_not_yet_tested"
    return {
        "status": static_status,
        "model_path": str(model_path.resolve()),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "quantization": {
            "method": quantization_method,
            "format": quantization.get("fmt"),
            "activation_scheme": quantization.get("activation_scheme"),
            "ignored_layers_count": len(quantization.get("ignored_layers", [])),
        },
        "transformers_finegrained_fp8_available": fp8_quantizer_available,
        "quantizer_import_error": quantizer_error,
        "gpu": gpu,
        "note": "vLLM/SGLang is model-card preferred; HF support needs the request preflight.",
    }


def _model_metadata_fingerprint(path: Path) -> dict[str, Any]:
    names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    )
    files = {name: _sha256_file(path / name) for name in names if (path / name).is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "path": str(path.resolve()),
        "metadata_sha256": digest,
        "scope": "configuration metadata only; weight files intentionally not re-hashed",
        "files": files,
    }


def _runtime_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            packages[name] = None
    gpu: dict[str, Any] | None = None
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(properties.total_memory),
                "cuda_version": torch.version.cuda,
            }
    except ImportError:
        pass
    return {
        "timestamp": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": packages,
        "gpu": gpu,
    }


def _gpu_resource_snapshot() -> dict[str, Any] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except ImportError:
        return None


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _token_f1(prediction: str, reference: str) -> float:
    pred = _normalize_text(prediction).split()
    ref = _normalize_text(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _fallback_example_key(row: dict[str, Any]) -> str:
    value = "\n".join(
        (
            str(row.get("task", "unknown")),
            str(Path(str(row.get("image_path", ""))).resolve()),
            _normalize_text(str(row.get("question", ""))),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else 0.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_loss_svg(path: Path, losses: Sequence[dict[str, Any]]) -> Path:
    width, height = 760, 330
    values = [float(item["loss"]) for item in losses]
    if values:
        low, high = min(values), max(values)
        span = max(high - low, 1e-6)
        points = []
        for index, value in enumerate(values):
            x = 70 + index * 640 / max(len(values) - 1, 1)
            y = 265 - (value - low) * 205 / span
            points.append(f"{x:.1f},{y:.1f}")
        polyline = (
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="3"/>'
        )
    else:
        low = high = 0.0
        polyline = ""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/><text x="24" y="32" font-size="22">Student LoRA loss</text>
<line x1="70" y1="60" x2="70" y2="265" stroke="#64748b"/><line x1="70" y1="265" x2="710" y2="265" stroke="#64748b"/>
{polyline}<text x="24" y="68" font-size="13">{high:.4f}</text><text x="24" y="265" font-size="13">{low:.4f}</text>
<text x="650" y="300" font-size="13">optimizer step</text></svg>"""
    path.write_text(svg, encoding="utf-8")
    return path


def _write_report_html(
    path: Path,
    result: dict[str, Any],
    audit: dict[str, Any],
    training: dict[str, Any],
    chart_path: Path,
) -> Path:
    metrics = result["metrics"]
    cards = "".join(
        f"<div class='card'><span>{html.escape(str(key))}</span><b>{html.escape(str(value))}</b></div>"
        for key, value in metrics.items()
    )
    rejection_rows = (
        "".join(
            f"<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>"
            for reason, count in audit.get("rejection_reasons", {}).items()
        )
        or "<tr><td>none</td><td>0</td></tr>"
    )
    mode_notice = (
        "MOCK RUN: orchestration evidence only; no model weights were loaded."
        if result["simulated"]
        else "ACTUAL RUN: the report describes a cached-response student LoRA run."
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>VLM response distillation</title>
<style>body{{font-family:system-ui;margin:2rem;background:#f6f8fb;color:#172033}}.cards{{display:flex;flex-wrap:wrap;gap:.7rem}}.card{{background:white;border:1px solid #d8e0ea;padding:.8rem;border-radius:8px;min-width:150px}}.card span{{display:block;color:#64748b;font-size:.8rem}}.notice{{padding:1rem;background:#fff4cc;border-left:4px solid #d99b00}}img{{max-width:760px;width:100%;background:white}}table{{border-collapse:collapse;background:white}}td,th{{border:1px solid #d8e0ea;padding:.5rem}}code{{word-break:break-all}}</style></head>
<body><h1>VLM response distillation: 32B teacher cache → 8B LoRA</h1><p class="notice">{html.escape(mode_notice)}</p>
<p>The teacher and student are deliberately executed in separate processes, so their weights never need to coexist in GPU memory.</p>
<div class="cards">{cards}</div><h2>Loss evidence</h2><img src="{html.escape(chart_path.name)}" alt="loss curve">
<h2>Filter audit</h2><table><tr><th>reason</th><th>count</th></tr>{rejection_rows}</table>
<h2>Reproducibility</h2><p>Teacher cache: <code>{html.escape(str(result["provenance"].get("teacher_cache_sha256")))}</code></p>
<p>Accepted cache: <code>{html.escape(str(result["provenance"].get("accepted_cache_sha256")))}</code></p>
<p>{html.escape(result["interpretation"])}</p><details><summary>Training configuration</summary><pre>{html.escape(json.dumps(training.get("config", {}), ensure_ascii=False, indent=2))}</pre></details>
</body></html>"""
    path.write_text(document, encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resource-bounded VLM response distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    mock = subparsers.add_parser("mock", help="Run the deterministic end-to-end pipeline")
    mock.add_argument("--manifest", type=Path, required=True)
    mock.add_argument("--output-dir", type=Path, required=True)
    mock.add_argument("--max-examples", type=int, default=12)
    mock.add_argument("--max-steps", type=int, default=5)

    cache = subparsers.add_parser("cache-teacher", help="Load only the teacher and cache responses")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--teacher-model", type=Path, required=True)
    cache.add_argument("--max-examples", type=int, default=64)
    cache.add_argument("--max-new-tokens", type=int, default=128)
    cache.add_argument("--max-minutes", type=float, default=60.0)
    cache.add_argument("--split-seed", type=int, default=17)
    cache.add_argument("--eval-fraction", type=float, default=0.25)
    cache.add_argument("--cuda-device", type=int, default=0)
    cache.add_argument("--min-free-vram-gib", type=float, default=48.0)
    cache.add_argument("--min-pixels", type=int, default=65_536)
    cache.add_argument("--max-pixels", type=int, default=262_144)
    cache.add_argument("--execute", action="store_true", help="Required heavy-stage safety gate")

    audit = subparsers.add_parser("audit", help="Filter and audit a raw teacher cache")
    audit.add_argument("--raw-cache", type=Path, required=True)
    audit.add_argument("--accepted-cache", type=Path, required=True)
    audit.add_argument("--audit-json", type=Path, required=True)
    audit.add_argument("--allow-simulated", action="store_true")

    train = subparsers.add_parser("train-student", help="Load only the student and train LoRA")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--student-model", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--audit-json", type=Path, required=True)
    train.add_argument("--max-examples", type=int, default=64)
    train.add_argument("--max-steps", type=int, default=20)
    train.add_argument("--max-minutes", type=float, default=30.0)
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--cuda-device", type=int, default=0)
    train.add_argument("--min-free-vram-gib", type=float, default=24.0)
    train.add_argument("--min-pixels", type=int, default=65_536)
    train.add_argument("--max-pixels", type=int, default=262_144)
    train.add_argument("--execute", action="store_true", help="Required heavy-stage safety gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "mock":
        result = run_mock_distillation(
            args.manifest,
            args.output_dir,
            max_examples=args.max_examples,
            max_steps=args.max_steps,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "cache-teacher":
        if not args.execute:
            raise SystemExit("Refusing to load the 32B teacher without explicit --execute")
        teacher_config = Qwen3VLConfig(
            mode="local",
            model_path=args.teacher_model,
            profile="32b-instruct-fp8",
            max_new_tokens=args.max_new_tokens,
            torch_dtype="auto",
            device_map=f"cuda:{args.cuda_device}",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        result = cache_teacher_responses(
            args.manifest,
            args.output,
            Qwen3VLRunner(teacher_config),
            max_examples=args.max_examples,
            max_minutes=args.max_minutes,
            split_seed=args.split_seed,
            eval_fraction=args.eval_fraction,
            cuda_device=args.cuda_device,
            min_free_vram_gib=args.min_free_vram_gib,
            teacher_descriptor={
                "model": _model_metadata_fingerprint(args.teacher_model),
                "static_runtime_inspection": inspect_local_teacher_runtime(args.teacher_model),
                "generation": asdict(teacher_config),
            },
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 2 if result.get("status") == "blocked_resource" else 0
    if args.command == "audit":
        result = audit_teacher_cache(
            args.raw_cache,
            args.accepted_cache,
            args.audit_json,
            filters=CacheFilterConfig(reject_simulated=not args.allow_simulated),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train-student":
        if not args.execute:
            raise SystemExit("Refusing to load the 8B student without explicit --execute")
        training = train_student_lora(
            StudentLoRAConfig(
                model_path=args.student_model,
                cache_path=args.cache,
                output_dir=args.output_dir,
                audit_path=args.audit_json,
                cuda_device=args.cuda_device,
                min_free_vram_gib=args.min_free_vram_gib,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_examples=args.max_examples,
                max_steps=args.max_steps,
                max_minutes=args.max_minutes,
                rank=args.rank,
            )
        )
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        result = write_distillation_report(args.output_dir, audit, training)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status") == "blocked_resource" else 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
