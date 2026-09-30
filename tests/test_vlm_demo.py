from pathlib import Path

import pytest

pytest.importorskip("PIL")

from fmlab.vlm.demo import run_offline_demo  # noqa: E402


def test_complete_offline_demo_writes_visual_reports(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = run_offline_demo(output, sample_count=1, seed=2)

    assert result.status == "completed"
    assert result.metrics["images"] == 3
    assert (output / "index.html").is_file()
    assert (output / "gallery.html").is_file()
    assert (output / "evaluations" / "document_compare" / "report.html").is_file()
    assert (output / "evaluations" / "chart_qa" / "operation_accuracy.svg").is_file()
    assert (output / "evaluations" / "grounding" / "predictions.jsonl").is_file()
    assert (output / "video" / "offline_sampling_plan.svg").is_file()
