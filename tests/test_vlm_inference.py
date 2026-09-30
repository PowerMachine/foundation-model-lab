from pathlib import Path

import pytest

from fmlab.vlm.inference import Qwen3VLConfig, Qwen3VLRunner, resolve_qwen_vl_path


def test_offline_runner_is_lazy_and_reproducible() -> None:
    runner = Qwen3VLRunner(Qwen3VLConfig(mode="offline"))
    result = runner.infer("What is the total?", context={"expected_answer": "42.50"})

    assert runner.loaded is False
    assert result.text == "42.50"
    assert result.simulated is True
    assert result.metadata["rule"] == "expected_answer"


def test_resolve_local_profile(tmp_path: Path) -> None:
    model = tmp_path / "Qwen" / "Qwen3-VL-8B-Instruct"
    model.mkdir(parents=True)
    assert resolve_qwen_vl_path("8b-instruct", tmp_path) == model


def test_unknown_profile_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="8b-instruct"):
        resolve_qwen_vl_path("not-a-model", tmp_path)
