from pathlib import Path

import pytest

from fmlab.vlm.grounding import bbox_iou, parse_bbox, parse_presence
from fmlab.vlm.metrics import exact_match, numeric_match, token_f1
from fmlab.vlm.training import VLMDistillationRecipe, VLMLoRARecipe
from fmlab.vlm.video import plan_frame_indices, write_sampling_plan


def test_text_and_numeric_metrics() -> None:
    assert exact_match("  BLUE square! ", "blue square") == 1.0
    assert numeric_match("$1,234.50", "1234.5") == 1.0
    assert token_f1("blue square", "blue small square") == pytest.approx(0.8)


def test_presence_and_bbox_parsing() -> None:
    assert parse_presence("Yes.") is True
    assert parse_presence("No, it is absent.") is False
    assert parse_presence("yes at first, but no") is None
    assert parse_bbox("[10, 20, 110, 220]") == (10.0, 20.0, 110.0, 220.0)
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_uniform_frame_plan_includes_endpoints(tmp_path: Path) -> None:
    frames = plan_frame_indices(total_frames=301, fps=30.0, max_frames=4)
    assert [frame.index for frame in frames] == [0, 100, 200, 300]
    assert frames[-1].timestamp_seconds == 10.0
    json_path, svg_path = write_sampling_plan(tmp_path / "plan.json", frames)
    assert json_path.is_file()
    assert "10.00s" in svg_path.read_text(encoding="utf-8")


def test_recipe_validation(tmp_path: Path) -> None:
    lora = VLMLoRARecipe(tmp_path / "model", tmp_path / "data.jsonl", tmp_path / "out")
    lora.validate()
    lora.rank = 0
    with pytest.raises(ValueError, match="rank"):
        lora.validate()

    distill = VLMDistillationRecipe(
        tmp_path / "teacher",
        tmp_path / "student",
        tmp_path / "data.jsonl",
        tmp_path / "targets.jsonl",
        tmp_path / "out",
        hard_label_weight=1.1,
    )
    with pytest.raises(ValueError, match="hard_label_weight"):
        distill.validate()
