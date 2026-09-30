from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmlab.ml import (
    run_anomaly,
    run_calibration,
    run_contrastive,
    run_diffusion,
    run_graph,
    run_timeseries,
)


CASES = [
    (
        "diffusion",
        run_diffusion,
        {
            "seed": 1,
            "device": "cpu",
            "image_size": 6,
            "dataset_size": 16,
            "batch_size": 8,
            "timesteps": 4,
            "train_steps": 2,
            "hidden_channels": 8,
            "learning_rate": 0.002,
            "sample_count": 4,
        },
    ),
    (
        "contrastive",
        run_contrastive,
        {
            "seed": 2,
            "device": "cpu",
            "class_count": 4,
            "image_size": 6,
            "embedding_dim": 6,
            "hidden_dim": 8,
            "batch_size": 4,
            "train_steps": 3,
            "learning_rate": 0.003,
            "image_noise": 0.1,
        },
    ),
    (
        "anomaly",
        run_anomaly,
        {
            "seed": 3,
            "device": "cpu",
            "feature_dim": 3,
            "train_normal_count": 32,
            "test_normal_count": 16,
            "test_anomaly_count": 8,
            "hidden_dim": 8,
            "bottleneck_dim": 2,
            "epochs": 3,
            "learning_rate": 0.003,
            "threshold_quantile": 0.9,
        },
    ),
    (
        "timeseries",
        run_timeseries,
        {
            "seed": 4,
            "device": "cpu",
            "series_length": 96,
            "seasonal_period": 8,
            "window": 12,
            "train_fraction": 0.65,
            "noise_std": 0.1,
            "hidden_dim": 8,
            "epochs": 3,
            "learning_rate": 0.006,
            "ridge": 0.01,
        },
    ),
    (
        "graph",
        run_graph,
        {
            "seed": 5,
            "device": "cpu",
            "class_count": 3,
            "nodes_per_class": 8,
            "feature_dim": 4,
            "hidden_dim": 8,
            "within_class_edge_probability": 0.35,
            "between_class_edge_probability": 0.03,
            "feature_signal": 0.6,
            "train_per_class": 2,
            "validation_per_class": 2,
            "epochs": 3,
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
        },
    ),
    (
        "calibration",
        run_calibration,
        {
            "seed": 6,
            "device": "cpu",
            "class_count": 3,
            "train_count": 48,
            "validation_count": 24,
            "test_count": 30,
            "hidden_dim": 8,
            "epochs": 3,
            "learning_rate": 0.008,
            "logit_multiplier": 2.0,
            "calibration_bins": 5,
            "temperature_candidates": 5,
        },
    ),
]


@pytest.mark.parametrize(("name", "function", "config"), CASES)
def test_ml_experiment_writes_metrics_and_visuals(
    name: str, function: object, config: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / name
    result = function(config, output)  # type: ignore[operator]

    assert result.status == "completed"
    assert result.metrics
    assert (output / "result.json").is_file()
    written = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert written["status"] == "completed"
    assert written["metrics"] == result.metrics
    assert len(list(output.glob("*.png"))) >= 3
    for artifact in result.artifacts:
        assert (output / artifact).is_file()
