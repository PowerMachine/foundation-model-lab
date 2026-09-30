from __future__ import annotations

import json

import pytest

from fmlab.vlm.research import (
    aggregate_predictions,
    grouped_split,
    normalized_levenshtein_similarity,
    paired_bootstrap_delta,
    score_answer,
    write_provenance,
)


def test_grouped_split_prevents_image_leakage() -> None:
    rows = [
        {"id": f"{image}-{question}", "image_path": image, "question": question}
        for image in ("a.png", "b.png", "c.png", "d.png")
        for question in ("q1", "q2")
    ]
    train, evaluation = grouped_split(rows, eval_fraction=0.25, seed=3)
    assert {row["image_path"] for row in train}.isdisjoint(
        {row["image_path"] for row in evaluation}
    )
    assert len(train) + len(evaluation) == len(rows)


def test_answer_metrics_and_slices() -> None:
    assert normalized_levenshtein_similarity("invoice", "invo1ce") == pytest.approx(6 / 7)
    assert score_answer("Total: 297.0", "297.00", answer_type="number")["type_accuracy"] == 1
    metrics = aggregate_predictions(
        [
            {"prediction": "yes", "reference": "yes", "task": "grounding"},
            {
                "prediction": "41.0",
                "reference": "41",
                "answer_type": "number",
                "task": "chart",
            },
        ]
    )
    assert metrics["count"] == 2
    assert metrics["type_accuracy"] == 1
    assert set(metrics["by_task"]) == {"chart", "grounding"}


def test_paired_bootstrap_and_provenance(tmp_path) -> None:
    bootstrap = paired_bootstrap_delta([0, 0, 1, 0], [1, 1, 1, 1], samples=200)
    assert bootstrap["delta"] == 0.75
    assert bootstrap["probability_positive"] > 0.9

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"one"}\n', encoding="utf-8")
    output = write_provenance(
        tmp_path / "provenance.json",
        config={"seed": 7},
        inputs=[manifest],
        system={"gpu": "test"},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == 7
    assert len(next(iter(payload["inputs"].values()))) == 64
