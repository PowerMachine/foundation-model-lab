from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmlab.vlm.distillation_evaluation import (
    evaluate_distillation_predictions,
    generate_simulated_demo,
    load_aligned_predictions,
    load_train_source_ids,
)


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _conditions(tmp_path: Path) -> dict[str, Path]:
    common = [
        {
            "id": "one",
            "source_id": "eval-a",
            "task": "chart",
            "answer_type": "number",
            "reference": "42",
        },
        {
            "id": "two",
            "source_id": "eval-b",
            "task": "document",
            "answer_type": "text",
            "reference": "Seoul",
        },
    ]
    answers = {
        "base_8b": ["40", "Busan"],
        "gold_lora": ["42", "Seoul"],
        "distilled_lora": ["42.0", "Seoul"],
        "teacher_32b": ["41", "Seoul"],
    }
    return {
        condition: _write(
            tmp_path / f"{condition}.jsonl",
            [{**row, "prediction": prediction} for row, prediction in zip(common, values)],
        )
        for condition, values in answers.items()
    }


def test_full_simulated_reporting_path_is_explicitly_non_quality_evidence(tmp_path: Path) -> None:
    demo = generate_simulated_demo(tmp_path / "inputs")
    report = evaluate_distillation_predictions(
        demo["condition_paths"],
        tmp_path / "report",
        train_source_ids=load_train_source_ids(demo["train_manifest"]),
        bootstrap_samples=200,
        simulated=True,
        resource_metrics={"distilled_lora": {"peak_vram_gib": 17.5}},
    )

    assert report["simulated"] is True
    assert report["claim_level"] == "wiring_only"
    assert "not evidence of model quality" in report["claim_boundary"]
    assert report["validation"]["source_isolation"]["status"] == "passed"
    assert report["conditions"]["distilled_lora"]["by_task"]["chart"]["count"] == 3
    assert report["teacher_gap_recovery"]["type_accuracy"]["status"] == "defined"
    assert report["failure_gallery"]["teacher_wrong_student_right"]
    assert report["resource_metrics"]["distilled_lora"]["peak_vram_gib"] == 17.5
    assert Path(report["artifacts"]["json"]).is_file()
    document = Path(report["artifacts"]["html"]).read_text(encoding="utf-8")
    assert "SIMULATED DATA" in document
    assert "teacher_wrong_student_right" in document


def test_alignment_rejects_duplicates_missing_ids_and_reference_changes(tmp_path: Path) -> None:
    conditions = _conditions(tmp_path)
    duplicate_rows = [
        {"id": "one", "reference": "42", "prediction": "42"},
        {"id": "one", "reference": "42", "prediction": "42"},
    ]
    conditions["gold_lora"] = _write(tmp_path / "duplicate.jsonl", duplicate_rows)
    with pytest.raises(ValueError, match="duplicate id"):
        load_aligned_predictions(conditions)

    conditions = _conditions(tmp_path)
    conditions["gold_lora"] = _write(
        tmp_path / "missing.jsonl",
        [
            {
                "id": "one",
                "source_id": "eval-a",
                "task": "chart",
                "answer_type": "number",
                "reference": "42",
                "prediction": "42",
            }
        ],
    )
    with pytest.raises(ValueError, match="misaligned ids"):
        load_aligned_predictions(conditions)

    conditions = _conditions(tmp_path)
    rows = [json.loads(line) for line in conditions["gold_lora"].read_text().splitlines()]
    rows[0]["reference"] = "43"
    conditions["gold_lora"] = _write(tmp_path / "reference.jsonl", rows)
    with pytest.raises(ValueError, match="misaligned reference"):
        load_aligned_predictions(conditions)


def test_source_leakage_is_rejected_and_paired_deltas_are_retained(tmp_path: Path) -> None:
    conditions = _conditions(tmp_path)
    with pytest.raises(ValueError, match="source overlap"):
        evaluate_distillation_predictions(
            conditions,
            tmp_path / "bad",
            train_source_ids={"train-x", "eval-a"},
            bootstrap_samples=100,
        )

    report = evaluate_distillation_predictions(
        conditions,
        tmp_path / "good",
        train_source_ids={"train-x"},
        bootstrap_samples=100,
    )
    comparison = report["comparisons"]["vs_base_8b"]["distilled_lora"]
    assert comparison["type_accuracy"]["delta"] == 1.0
    assert comparison["by_task"]["chart"]["type_accuracy"]["delta"] == 1.0
    vs_gold = report["comparisons"]["vs_gold_lora"]["distilled_lora"]
    assert vs_gold["type_accuracy"]["delta"] == 0.0
    assert report["claim_level"] == "heldout_comparison"
    assert "multi-seed" in report["claim_boundary"]


def test_train_source_loader_uses_only_valid_train_split_rows(tmp_path: Path) -> None:
    cache = _write(
        tmp_path / "accepted.jsonl",
        [
            {"image_sha256": "train-hash", "split": "train"},
            {"image_sha256": "eval-hash", "split": "eval"},
        ],
    )
    assert load_train_source_ids(cache) == {"train-hash"}
    bad = _write(tmp_path / "bad.jsonl", [{"source_id": "x", "split": "validation"}])
    with pytest.raises(ValueError, match="invalid split"):
        load_train_source_ids(bad)
