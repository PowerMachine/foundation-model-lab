from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from fmlab.vlm.lora_data import (
    Qwen3VLAssistantOnlyCollator,
    ToyMultimodalProcessor,
    grouped_holdout_split,
    load_canonical_vlm_manifest,
)
from fmlab.vlm.lora_e2e import (
    InsufficientVRAMError,
    VLMLoRAExperimentConfig,
    enforce_cuda_device_budget,
    run_vlm_lora_experiment,
    write_failed_vlm_lora_result,
)


def _write_manifest(tmp_path: Path) -> Path:
    rows = []
    for index in range(4):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(f"fake-png-{index}".encode())
        rows.append(
            {
                "id": f"source-{index}",
                "task": "document_vqa" if index % 2 == 0 else "chart_qa",
                "image_path": str(image),
                "qa": [
                    {
                        "id": f"qa-{index}-a",
                        "question": f"Question {index} A?",
                        "answer": f"answer-{index}-a",
                    },
                    {
                        "id": f"qa-{index}-b",
                        "question": f"Question {index} B?",
                        "answer": f"answer-{index}-b",
                    },
                ],
            }
        )
    grounding_image = tmp_path / "scene.png"
    grounding_image.write_bytes(b"fake-scene")
    rows.extend(
        [
            {
                "id": "present",
                "task": "grounding",
                "image_path": str(grounding_image),
                "object_name": "blue square",
                "present": True,
            },
            {
                "id": "absent",
                "task": "grounding",
                "image_path": str(grounding_image),
                "object_name": "red circle",
                "present": False,
            },
        ]
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_manifest_expansion_and_grouped_split_prevent_image_leakage(tmp_path: Path) -> None:
    examples = load_canonical_vlm_manifest(_write_manifest(tmp_path))

    assert len(examples) == 10
    assert {item.answer for item in examples if item.task == "grounding"} == {"yes", "no"}

    split = grouped_holdout_split(examples, holdout_fraction=0.4, seed=3)
    train_images = {item.image_path for item in split.train}
    evaluation_images = {item.image_path for item in split.evaluation}
    assert train_images.isdisjoint(evaluation_images)
    assert split.summary()["source_overlap"] == []
    coverage_split = grouped_holdout_split(examples, holdout_fraction=0.6, seed=3)
    assert {item.task for item in coverage_split.evaluation} == {
        "document_vqa",
        "chart_qa",
        "grounding",
    }


def test_collator_masks_system_image_user_and_padding() -> None:
    processor = ToyMultimodalProcessor()
    from fmlab.vlm.lora_data import VLMTrainingExample

    examples = [
        VLMTrainingExample(
            id="short",
            source_id="one",
            task="document_vqa",
            image_path=Path("missing-a.png"),
            question="Q?",
            answer="A",
        ),
        VLMTrainingExample(
            id="long",
            source_id="two",
            task="chart_qa",
            image_path=Path("missing-b.png"),
            question="A somewhat longer question?",
            answer="a longer answer",
        ),
    ]
    batch = Qwen3VLAssistantOnlyCollator(
        processor,
        max_sequence_length=256,
        system_prompt="Follow format.",
    )(examples)

    supervised = batch["labels"].ne(-100)
    assert supervised.sum(dim=1).tolist() == [2, 16]
    assert torch.equal(batch["labels"][supervised], batch["input_ids"][supervised])
    assert torch.all(batch["labels"][batch["attention_mask"].eq(0)] == -100)
    first_supervised = supervised[0].nonzero().min().item()
    assert first_supervised > 4  # system, image, and user prefixes are masked


def test_mock_e2e_writes_trace_provenance_result_and_report(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    phases: list[str] = []

    def evaluation_hook(model, processor, examples, phase, config):
        del model, processor, config
        phases.append(phase)
        score = 0.25 if phase == "before" else 0.5
        return {
            "phase": phase,
            "metric_kind": "teacher_forced_proxy",
            "assistant_token_accuracy": score,
            "loss": 2.0 if phase == "before" else 1.5,
            "records": [
                {
                    "id": item.id,
                    "task": item.task,
                    "reference": item.answer,
                    "prediction": "mock",
                    "score": score,
                }
                for item in examples
            ],
        }

    config = VLMLoRAExperimentConfig(
        model_path=tmp_path / "not-loaded",
        dataset_manifest=manifest,
        artifact_dir=artifact_dir,
        backend="mock",
        experiment="qwen3_vl_8b_lora_custom_test",
        max_steps=2,
        gradient_accumulation_steps=1,
        max_train_examples=4,
        max_eval_examples=3,
        max_sequence_length=256,
    )
    result_path = run_vlm_lora_experiment(config, evaluation_hook=evaluation_hook)
    result = json.loads(result_path.read_text())

    assert phases == ["before", "after"]
    assert result["status"] == "completed"
    assert result["metrics"]["simulated"] is True
    assert result["experiment"] == "qwen3_vl_8b_lora_custom_test"
    assert result["metrics"]["completed_steps"] == 2
    assert result["metrics"]["comparison"]["delta"] == 0.25
    assert result["metrics"]["dataset"]["source_overlap"] == []
    assert (artifact_dir / "training_trace.jsonl").is_file()
    assert (artifact_dir / "provenance.json").is_file()
    assert (artifact_dir / "report.html").is_file()


def test_config_aliases_existing_output_dir_for_adapter(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    config = VLMLoRAExperimentConfig.from_mapping(
        {
            "model_path": str(tmp_path / "model"),
            "dataset_manifest": str(manifest),
            "artifact_dir": str(tmp_path / "artifact"),
            "output_dir": str(tmp_path / "adapter"),
            "backend": "mock",
        }
    )
    assert config.adapter_output_dir == tmp_path / "adapter"


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="Unknown VLM LoRA config keys"):
        VLMLoRAExperimentConfig.from_mapping(
            {
                "model_path": "unused",
                "dataset_manifest": manifest,
                "artifact_dir": "unused",
                "backend": "mock",
                "max_step": 2,
            }
        )


def test_explicit_cuda_device_budget_is_per_device_and_fail_closed() -> None:
    class FakeCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 2

        def mem_get_info(self, index):
            assert index == 1
            return 5 * 1024**3, 8 * 1024**3

        def get_device_name(self, index):
            return f"fake-{index}"

    api = FakeCuda()
    snapshot = enforce_cuda_device_budget(1, 4.0, cuda_api=api)
    assert snapshot["cuda_device"] == 1
    assert snapshot["free_gib"] == 5.0
    with pytest.raises(InsufficientVRAMError, match="cuda:1"):
        enforce_cuda_device_budget(1, 6.0, cuda_api=api)


def test_blocked_resource_result_retains_only_current_audit_artifacts(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    artifact_dir = tmp_path / "blocked"
    artifact_dir.mkdir()
    (artifact_dir / "training_trace.jsonl").write_text("stale\n")
    config = VLMLoRAExperimentConfig(
        model_path=tmp_path / "model",
        dataset_manifest=manifest,
        artifact_dir=artifact_dir,
        experiment="qwen3_vl_8b_lora_qv_ablation",
        backend="mock",
    )
    result_path = write_failed_vlm_lora_result(config, InsufficientVRAMError("cuda:0 busy"))
    payload = json.loads(result_path.read_text())

    assert payload["status"] == "blocked_resource"
    assert payload["experiment"] == "qwen3_vl_8b_lora_qv_ablation"
    assert {Path(item).name for item in payload["artifacts"]} == {
        "provenance.json",
        "report.html",
        "result.json",
    }


def test_separate_rows_sharing_image_identity_never_cross_split(tmp_path: Path) -> None:
    shared = tmp_path / "shared.png"
    shared.write_bytes(b"same-image")
    copy = tmp_path / "copy.png"
    copy.write_bytes(b"same-image")
    other = tmp_path / "other.png"
    other.write_bytes(b"other-image")
    rows = [
        {"id": "a", "task": "doc", "image_path": str(shared), "question": "A?", "answer": "A"},
        {"id": "b", "task": "doc", "image_path": str(copy), "question": "B?", "answer": "B"},
        {"id": "c", "task": "doc", "image_path": str(other), "question": "C?", "answer": "C"},
    ]
    manifest = tmp_path / "separate-rows.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    split = grouped_holdout_split(load_canonical_vlm_manifest(manifest), seed=9)
    train_ids = {item.id for item in split.train}
    evaluation_ids = {item.id for item in split.evaluation}
    assert {"a", "b"} <= train_ids or {"a", "b"} <= evaluation_ids
