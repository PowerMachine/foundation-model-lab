from fmlab.vlm.tiny import TinyVLMRunner


def test_tiny_runner_never_loads_weights() -> None:
    runner = TinyVLMRunner()
    result = runner.infer(
        "Is there a blue square in the image?",
        context={"objects": ["blue square", "red circle"]},
    )

    assert runner.loaded is False
    assert result.text == "yes"
    assert result.simulated is True
    assert result.backend == "tiny-structured-baseline"
