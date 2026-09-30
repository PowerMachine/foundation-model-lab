from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmlab.artifacts import ExperimentResult

from .common import choose_device, finish, mae, merged, prepare, rmse, save_figure


DEFAULTS: dict[str, Any] = {
    "seed": 23,
    "device": "auto",
    "series_length": 480,
    "seasonal_period": 24,
    "window": 36,
    "train_fraction": 0.72,
    "noise_std": 0.12,
    "hidden_dim": 32,
    "epochs": 180,
    "learning_rate": 0.006,
    "ridge": 0.01,
}


def make_series(length: int, period: int, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float32)
    trend = 0.0025 * time
    season = 0.85 * np.sin(2 * np.pi * time / period)
    harmonic = 0.23 * np.cos(4 * np.pi * time / period + 0.4)
    slow_cycle = 0.2 * np.sin(2 * np.pi * time / (period * 6))
    noise = rng.normal(0.0, noise_std, size=length)
    return (trend + season + harmonic + slow_cycle + noise).astype(np.float32)


def _windows(values: np.ndarray, window: int, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([values[index - window : index] for index in targets])
    return features.astype(np.float32), values[targets].astype(np.float32)


class ForecastMLP(nn.Module):
    def __init__(self, window: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(window, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.network(windows).squeeze(-1)


def _scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(prediction, target), "mae": mae(prediction, target)}


def run_timeseries(
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/ml/timeseries",
) -> ExperimentResult:
    cfg = merged(DEFAULTS, config)
    output, started = prepare(output_dir, int(cfg["seed"]))
    device = choose_device(str(cfg["device"]))
    length = int(cfg["series_length"])
    period = int(cfg["seasonal_period"])
    window = int(cfg["window"])
    if window < period:
        raise ValueError("window must be at least seasonal_period")
    series = make_series(length, period, float(cfg["noise_std"]), int(cfg["seed"]))
    split = int(length * float(cfg["train_fraction"]))
    if not window < split < length:
        raise ValueError("train_fraction leaves no valid train or test windows")

    mean = float(series[:split].mean())
    std = float(series[:split].std())
    normalized = (series - mean) / max(std, 1e-6)
    train_indices = np.arange(window, split)
    test_indices = np.arange(split, length)
    x_train, y_train = _windows(normalized, window, train_indices)
    x_test, _ = _windows(normalized, window, test_indices)
    target = series[test_indices]

    persistence = series[test_indices - 1]
    seasonal = series[test_indices - period]
    design_train = np.column_stack((x_train, np.ones(len(x_train), dtype=np.float32)))
    design_test = np.column_stack((x_test, np.ones(len(x_test), dtype=np.float32)))
    regularizer = np.eye(design_train.shape[1], dtype=np.float32) * float(cfg["ridge"])
    regularizer[-1, -1] = 0.0
    ridge_weights = np.linalg.solve(
        design_train.T @ design_train + regularizer,
        design_train.T @ y_train,
    )
    ridge_prediction = (design_test @ ridge_weights) * std + mean

    model = ForecastMLP(window, int(cfg["hidden_dim"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    train_x_tensor = torch.from_numpy(x_train).to(device)
    train_y_tensor = torch.from_numpy(y_train).to(device)
    losses: list[float] = []
    model.train()
    for _ in range(int(cfg["epochs"])):
        prediction = model(train_x_tensor)
        loss = F.mse_loss(prediction, train_y_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        mlp_prediction = model(torch.from_numpy(x_test).to(device)).cpu().numpy() * std + mean

    predictions = {
        "persistence": persistence,
        "seasonal_naive": seasonal,
        "ridge_autoregression": ridge_prediction,
        "tiny_mlp": mlp_prediction,
    }
    scores = {name: _scores(values, target) for name, values in predictions.items()}
    best_model = min(scores, key=lambda name: scores[name]["rmse"])

    artifacts: list[Path] = []
    fig, axis = plt.subplots(figsize=(11, 4))
    shown = min(len(test_indices), 144)
    positions = test_indices[:shown]
    axis.plot(positions, target[:shown], color="black", linewidth=2, label="actual")
    for name, values in predictions.items():
        axis.plot(positions, values[:shown], linewidth=1.1, alpha=0.8, label=name)
    axis.axvline(split, color="gray", linestyle="--", alpha=0.7)
    axis.set(title="One-step forecasts on the held-out future", xlabel="time", ylabel="value")
    axis.legend(ncol=3, fontsize=8)
    axis.grid(alpha=0.2)
    artifacts.append(save_figure(fig, output / "forecast_comparison.png"))

    names = list(scores)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, metric in zip(axes, ("rmse", "mae"), strict=True):
        values = [scores[name][metric] for name in names]
        bars = axis.bar(names, values, color=["#999999", "#8172b3", "#4c72b0", "#55a868"])
        axis.set(title=metric.upper(), ylabel=metric)
        axis.tick_params(axis="x", rotation=25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    artifacts.append(save_figure(fig, output / "error_bars.png"))

    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(losses, color="#55a868")
    axis.set(title="Tiny MLP training", xlabel="epoch", ylabel="normalized MSE")
    axis.grid(alpha=0.25)
    artifacts.append(save_figure(fig, output / "mlp_training_loss.png"))

    metrics = {
        "models": scores,
        "best_test_rmse_model": best_model,
        "mlp_initial_loss": losses[0],
        "mlp_final_loss": losses[-1],
        "mlp_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_points": len(train_indices),
        "test_points": len(test_indices),
        "device": str(device),
    }
    return finish(
        experiment="ml_timeseries",
        started_at=started,
        output_dir=output,
        metrics=metrics,
        parameters=cfg,
        artifacts=artifacts,
        notes=[
            "The split is chronological: every test target occurs after all training targets.",
            "Simple persistence and seasonal baselines prevent over-crediting a neural network on easy data.",
        ],
    )
