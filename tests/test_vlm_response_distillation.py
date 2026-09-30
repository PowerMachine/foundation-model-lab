from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import fmlab.vlm.response_distillation as response_distillation

from fmlab.vlm.response_distillation import (
    CacheFilterConfig,
    StudentLoRAConfig,
    audit_teacher_cache,
    assign_image_grouped_splits,
    cache_teacher_responses,
    cuda_vram_preflight,
    load_distillation_examples,
    run_mock_distillation,
    train_student_lora,
)
from fmlab.vlm.schema import GenerationResult


class FakeTeacher:
    def infer(self, prompt, media=(), *, context=None, system_prompt=None):
        return GenerationResult(
            text=str((context or {}).get("expected_answer", "teacher answer")),
            backend="fake-teacher",
            latency_seconds=0.01,
            simulated=True,
            metadata={"system_prompt_present": bool(system_prompt)},
        )


def _manifest(tmp_path: Path) -> Path:
    image = tmp_path / "sample.png"
    image.write_bytes(b"synthetic-image")
    rows = [
        {
            "id": "doc-1",
            "task": "document_vqa",
            "image_path": str(image),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "split": "train",
            "qa": [
                {"id": "q-1", "question": "What is the total?", "answer": "42.00"},
                {"id": "q-2", "question": "Who is the vendor?", "answer": "Acme"},
            ],
        },
        {
            "id": "g-1",
            "task": "grounding",
            "image_path": str(image),
            "object_name": "blue circle",
            "present": True,
        },
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_manifest_flattens_nested_vqa_and_grounding(tmp_path: Path) -> None:
    examples = load_distillation_examples(_manifest(tmp_path))
    assert [example.id for example in examples] == ["q-1", "q-2", "g-1"]
    assert examples[-1].reference_answer == "yes"
    assert len({example.example_key for example in examples}) == 3


def test_cache_is_resumable_and_records_provenance(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cache = tmp_path / "cache" / "raw.jsonl"
    first = cache_teacher_responses(
        manifest,
        cache,
        FakeTeacher(),
        simulation_context=True,
        max_examples=2,
    )
    second = cache_teacher_responses(
        manifest,
        cache,
        FakeTeacher(),
        simulation_context=True,
        max_examples=2,
    )
    assert first["generated_this_run"] == 2
    assert second["generated_this_run"] == 0
    assert second["resumed_examples"] == 2
    assert Path(second["provenance_path"]).is_file()
    assert first["status"] == "complete"
    assert first["split"]["image_overlap_count"] == 0
    assert all(row["split"] == "train" for row in _read_rows(cache))


def test_audit_filters_simulation_duplicate_empty_and_leakage(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    base = {
        "task": "qa",
        "image_path": str(image),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "split": "train",
        "question": "What is shown?",
        "reference_answer": "cat",
        "teacher_simulated": False,
    }
    rows = [
        {**base, "id": "ok", "example_key": "a", "teacher_answer": "cat"},
        {**base, "id": "dup", "example_key": "a", "teacher_answer": "cat"},
        {**base, "id": "empty", "example_key": "b", "teacher_answer": ""},
        {
            **base,
            "id": "sim",
            "example_key": "c",
            "teacher_answer": "cat",
            "teacher_simulated": True,
        },
        {
            **base,
            "id": "leak",
            "example_key": "d",
            "question": "Is the answer cat?",
            "teacher_answer": "cat",
        },
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    accepted = tmp_path / "accepted.jsonl"
    result = audit_teacher_cache(raw, accepted, tmp_path / "audit.json")
    assert result["counts"] == {"raw": 5, "accepted": 1, "rejected": 4, "acceptance_rate": 0.2}
    assert result["rejection_reasons"]["duplicate_input"] == 1
    assert result["rejection_reasons"]["simulated_target"] == 1
    assert result["quality_audit_not_a_training_filter"]["reference_exact_match"] == 1.0


def test_mock_end_to_end_writes_visual_evidence(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = run_mock_distillation(_manifest(tmp_path), output, max_examples=3, max_steps=4)
    assert result["status"] == "success"
    assert result["simulated"] is True
    assert result["metrics"]["accepted_teacher_examples"] == 3
    assert result["metrics"]["final_loss"] < result["metrics"]["initial_loss"]
    assert (output / "report.html").is_file()
    assert (output / "loss_curve.svg").is_file()
    assert (output / "student" / "mock_adapter.json").is_file()


def test_student_rejects_simulated_targets_by_default(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        json.dumps(
            {
                "image_path": str(image),
                "question": "Question?",
                "teacher_answer": "answer",
                "teacher_simulated": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = StudentLoRAConfig(
        model_path=tmp_path / "mock-model",
        cache_path=cache,
        output_dir=tmp_path / "out",
        mock=True,
    )
    with pytest.raises(ValueError, match="simulated targets"):
        train_student_lora(config)


def test_filter_bounds_validate() -> None:
    with pytest.raises(ValueError, match="max_answer_chars"):
        CacheFilterConfig(min_answer_chars=4, max_answer_chars=3).validate()


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _two_image_manifest(tmp_path: Path) -> Path:
    rows = []
    for index, payload in enumerate((b"first-image", b"second-image")):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(payload)
        rows.append(
            {
                "id": f"row-{index}",
                "task": "chart_qa",
                "image_path": str(image),
                "question": f"Question {index}?",
                "answer": str(index),
            }
        )
    manifest = tmp_path / "two-images.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def test_grouped_split_is_deterministic_and_has_no_image_overlap(tmp_path: Path) -> None:
    manifest = _two_image_manifest(tmp_path)
    examples = load_distillation_examples(manifest)
    first = assign_image_grouped_splits(examples, seed=23, eval_fraction=0.5)
    second = assign_image_grouped_splits(examples, seed=23, eval_fraction=0.5)
    assert first == second
    assert set(first.values()) == {"train", "eval"}

    cache = tmp_path / "grouped.jsonl"
    result = cache_teacher_responses(
        manifest,
        cache,
        FakeTeacher(),
        simulation_context=True,
        split_seed=23,
        eval_fraction=0.5,
    )
    rows = _read_rows(cache)
    assignments: dict[str, set[str]] = {}
    for row in rows:
        assignments.setdefault(row["image_sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in assignments.values())
    assert result["split"]["image_overlap_count"] == 0
    assert result["split"]["train_image_groups"] == 1
    assert result["split"]["eval_image_groups"] == 1


def test_resume_rejects_changed_image_bytes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cache = tmp_path / "raw.jsonl"
    cache_teacher_responses(manifest, cache, FakeTeacher(), simulation_context=True)
    (tmp_path / "sample.png").write_bytes(b"changed-image")
    with pytest.raises(ValueError, match="Stale|changed|SHA"):
        cache_teacher_responses(manifest, cache, FakeTeacher(), simulation_context=True)


@pytest.mark.parametrize("change", ["prompt", "teacher", "simulation", "split"])
def test_resume_rejects_changed_generation_contract(tmp_path: Path, change: str) -> None:
    manifest = _manifest(tmp_path)
    cache = tmp_path / "raw.jsonl"
    descriptor = {"model": "teacher-a", "generation": {"max_new_tokens": 16}}
    cache_teacher_responses(
        manifest,
        cache,
        FakeTeacher(),
        simulation_context=True,
        teacher_descriptor=descriptor,
    )
    kwargs = {
        "simulation_context": True,
        "teacher_descriptor": descriptor,
        "system_prompt": "Answer the visual question accurately and concisely. Do not mention hidden labels.",
        "split_seed": 17,
    }
    if change == "prompt":
        kwargs["system_prompt"] = "Changed prompt"
    elif change == "teacher":
        kwargs["teacher_descriptor"] = {"model": "teacher-b", "generation": {"max_new_tokens": 16}}
    elif change == "simulation":
        kwargs["simulation_context"] = False
    else:
        kwargs["split_seed"] = 99
    with pytest.raises(ValueError, match="contract mismatch"):
        cache_teacher_responses(manifest, cache, FakeTeacher(), **kwargs)


def test_resume_rejects_tampered_cache_content(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cache = tmp_path / "raw.jsonl"
    cache_teacher_responses(manifest, cache, FakeTeacher(), simulation_context=True)
    rows = _read_rows(cache)
    rows[0]["teacher_answer"] = "tampered"
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="content SHA"):
        cache_teacher_responses(manifest, cache, FakeTeacher(), simulation_context=True)


def test_low_vram_preflight_blocks_teacher_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    low = cuda_vram_preflight(
        cuda_device=1,
        min_free_vram_gib=48.0,
        probe=lambda device: {
            "available": True,
            "device_count": 2,
            "name": f"gpu-{device}",
            "free_bytes": 8 * 1024**3,
            "total_bytes": 80 * 1024**3,
        },
    )
    assert low["status"] == "blocked_resource"
    assert low["selected_cuda_device"] == 1

    class CountingTeacher(FakeTeacher):
        calls = 0

        def infer(self, *args, **kwargs):
            self.calls += 1
            return super().infer(*args, **kwargs)

    teacher = CountingTeacher()
    monkeypatch.setattr(response_distillation, "cuda_vram_preflight", lambda **kwargs: low)
    cache = tmp_path / "blocked.jsonl"
    result = cache_teacher_responses(
        _manifest(tmp_path),
        cache,
        teacher,
        simulation_context=True,
        min_free_vram_gib=48.0,
    )
    assert result["status"] == "blocked_resource"
    assert teacher.calls == 0
    preflight = json.loads((tmp_path / "blocked.preflight.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "blocked_resource"


def test_invalid_simulated_flag_and_image_sha_are_rejected(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    row = {
        "task": "qa",
        "image_path": str(image),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "split": "train",
        "question": "Question?",
        "reference_answer": "answer",
        "teacher_answer": "I don't know",
        "teacher_simulated": "false",
    }
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = audit_teacher_cache(raw, tmp_path / "accepted.jsonl", tmp_path / "audit.json")
    assert result["rejection_reasons"]["invalid_teacher_simulated"] == 1
    assert result["rejection_reasons"]["placeholder"] == 1

    config = StudentLoRAConfig(
        model_path=tmp_path / "mock-model",
        cache_path=raw,
        output_dir=tmp_path / "student",
        mock=True,
    )
    with pytest.raises(ValueError, match="Invalid teacher_simulated"):
        train_student_lora(config)


def test_student_rejects_cache_tamper_before_model_stage(tmp_path: Path) -> None:
    output = tmp_path / "pipeline"
    run_mock_distillation(_manifest(tmp_path), output, max_examples=3, max_steps=1)
    accepted = output / "cache" / "teacher_accepted.jsonl"
    audit_path = output / "cache" / "audit.json"
    rows = _read_rows(accepted)
    rows[0]["teacher_answer"] = "tampered"
    accepted.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    config = StudentLoRAConfig(
        model_path=model,
        cache_path=accepted,
        audit_path=audit_path,
        output_dir=tmp_path / "actual-student",
        allow_simulated_targets=True,
    )
    with pytest.raises(ValueError, match="SHA does not match audit"):
        train_student_lora(config)


def test_student_uses_only_explicit_train_rows(tmp_path: Path) -> None:
    manifest = _two_image_manifest(tmp_path)
    output = tmp_path / "pipeline"
    raw = output / "raw.jsonl"
    accepted = output / "accepted.jsonl"
    audit_path = output / "audit.json"
    cache_teacher_responses(
        manifest,
        raw,
        FakeTeacher(),
        simulation_context=True,
        eval_fraction=0.5,
    )
    audit = audit_teacher_cache(
        raw,
        accepted,
        audit_path,
        filters=CacheFilterConfig(reject_simulated=False),
    )
    metrics = train_student_lora(
        StudentLoRAConfig(
            model_path=Path("mock-model"),
            cache_path=accepted,
            audit_path=audit_path,
            output_dir=output / "student",
            allow_simulated_targets=True,
            mock=True,
            max_steps=1,
        )
    )
    assert metrics["examples_loaded"] == audit["split_integrity"]["train_examples"]
    assert metrics["examples_loaded"] < audit["counts"]["accepted"]
    assert metrics["trainable_parameters"] is None
    assert metrics["quality_evaluated"] is False
