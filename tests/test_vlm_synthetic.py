from pathlib import Path

import pytest

pytest.importorskip("PIL")

from fmlab.vlm.synthetic import (  # noqa: E402
    generate_chart_dataset,
    generate_document_dataset,
    read_manifest,
)


def test_document_generation_round_trip(tmp_path: Path) -> None:
    samples = generate_document_dataset(tmp_path / "docs", count=2, seed=3)
    restored = read_manifest(tmp_path / "docs" / "manifest.jsonl")

    assert len(samples) == len(restored) == 2
    assert all(sample.image_path.is_file() for sample in samples)
    assert samples[0].metadata["fields"] == restored[0].metadata["fields"]
    assert samples[0].qa[0].answer == restored[0].qa[0].answer


def test_chart_labels_are_derived_from_values(tmp_path: Path) -> None:
    sample = generate_chart_dataset(tmp_path / "charts", count=1, seed=5)[0]
    values = sample.metadata["values"]
    answers = {qa.metadata["operation"]: qa.answer for qa in sample.qa}

    assert answers["argmax"] == max(values, key=values.get)
    assert answers["argmin"] == min(values, key=values.get)
    assert int(answers["difference"]) == max(values.values()) - min(values.values())
    assert int(answers["sum"]) == sum(values.values())
