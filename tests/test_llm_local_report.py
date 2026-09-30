import json
from pathlib import Path

from fmlab.llm.local_report import build_local_model_report


def test_local_report_aggregates_workflow_metrics(tmp_path: Path) -> None:
    for relative, value in {
        "continued_pretraining/workflow_metrics.json": {"train_loss": 2.0},
        "full_sft/workflow_metrics.json": {"train_loss": 1.0, "trainable_percent": 100},
        "lora/workflow_metrics.json": {"train_loss": 1.2, "trainable_percent": 0.5},
        "qlora/workflow_metrics.json": {"train_loss": 1.1, "trainable_percent": 0.8},
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    result = build_local_model_report(tmp_path)
    assert result.metrics["lora_trainable_percent"] == 0.5
    assert (tmp_path / "local-model-suite/report.html").is_file()
    assert (tmp_path / "local-model-suite/trainable_percent.png").is_file()
