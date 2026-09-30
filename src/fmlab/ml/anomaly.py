from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmlab.artifacts import ExperimentResult

from .common import binary_metrics, choose_device, finish, merged, prepare, save_figure


DEFAULTS: dict[str, Any] = {
    "seed": 17,
    "device": "auto",
    "feature_dim": 4,
    "train_normal_count": 512,
    "test_normal_count": 240,
    "test_anomaly_count": 120,
    "hidden_dim": 16,
    "bottleneck_dim": 2,
    "epochs": 180,
    "learning_rate": 0.003,
    "threshold_quantile": 0.95,
}


def _manifold_features(latent: np.ndarray, dimension: int) -> np.ndarray:
    candidates = [
        latent / 3.0,
        np.sin(latent),
        np.cos(latent),
        (latent / 3.0) ** 2,
        np.sin(2.0 * latent),
        np.cos(2.0 * latent),
    ]
    while len(candidates) < dimension:
        power = len(candidates) - 2
        candidates.append((latent / 3.0) ** power)
    return np.stack(candidates[:dimension], axis=1).astype(np.float32)


def make_anomaly_data(
    train_count: int,
    normal_count: int,
    anomaly_count: int,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dimension < 2:
        raise ValueError("feature_dim must be at least 2 for the visual experiment")
    rng = np.random.default_rng(seed)

    def normal(count: int) -> np.ndarray:
        latent = rng.uniform(-3.0, 3.0, size=count)
        values = _manifold_features(latent, dimension)
        return values + rng.normal(0.0, 0.055, size=values.shape).astype(np.float32)

    train = normal(train_count)
    test_normal = normal(normal_count)
    anomaly_latent = rng.uniform(-3.0, 3.0, size=anomaly_count)
    anomalies = _manifold_features(anomaly_latent, dimension)
    corrupted_dimension = rng.integers(1, dimension, size=anomaly_count)
    direction = rng.choice(np.array([-1.0, 1.0]), size=anomaly_count)
    anomalies[np.arange(anomaly_count), corrupted_dimension] += direction * rng.uniform(
        0.8, 1.5, size=anomaly_count
    )
    anomalies += rng.normal(0.0, 0.055, size=anomalies.shape).astype(np.float32)
    test = np.concatenate((test_normal, anomalies)).astype(np.float32)
    labels = np.concatenate((np.zeros(normal_count), np.ones(anomaly_count))).astype(np.int64)
    permutation = rng.permutation(len(test))
    return train, test[permutation], labels[permutation]


class Autoencoder(nn.Module):
    def __init__(self, dimension: int, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.Tanh(), nn.Linear(hidden, dimension)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def _mahalanobis(
    values: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray
) -> np.ndarray:
    centered = values - mean
    return np.einsum("ni,ij,nj->n", centered, inverse_covariance, centered)


def run_anomaly(
    config: dict[str, Any] | None = None, output_dir: str | Path = "artifacts/ml/anomaly"
) -> ExperimentResult:
    cfg = merged(DEFAULTS, config)
    output, started = prepare(output_dir, int(cfg["seed"]))
    device = choose_device(str(cfg["device"]))
    train, test, labels = make_anomaly_data(
        int(cfg["train_normal_count"]),
        int(cfg["test_normal_count"]),
        int(cfg["test_anomaly_count"]),
        int(cfg["feature_dim"]),
        int(cfg["seed"]),
    )
    feature_mean = train.mean(axis=0)
    feature_std = train.std(axis=0).clip(min=1e-5)
    normalized_train = (train - feature_mean) / feature_std
    normalized_test = (test - feature_mean) / feature_std

    statistical_mean = normalized_train.mean(axis=0)
    covariance = np.cov(normalized_train, rowvar=False) + np.eye(normalized_train.shape[1]) * 1e-3
    inverse_covariance = np.linalg.pinv(covariance)
    statistical_train_scores = _mahalanobis(normalized_train, statistical_mean, inverse_covariance)
    statistical_scores = _mahalanobis(normalized_test, statistical_mean, inverse_covariance)
    quantile = float(cfg["threshold_quantile"])
    statistical_threshold = float(np.quantile(statistical_train_scores, quantile))

    train_tensor = torch.from_numpy(normalized_train).to(device)
    test_tensor = torch.from_numpy(normalized_test).to(device)
    model = Autoencoder(
        int(cfg["feature_dim"]), int(cfg["hidden_dim"]), int(cfg["bottleneck_dim"])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    losses: list[float] = []
    model.train()
    for _ in range(int(cfg["epochs"])):
        reconstruction = model(train_tensor)
        loss = F.mse_loss(reconstruction, train_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        train_reconstruction = model(train_tensor)
        test_reconstruction = model(test_tensor)
        autoencoder_train_scores = (
            ((train_reconstruction - train_tensor) ** 2).mean(dim=1).cpu().numpy()
        )
        autoencoder_scores = ((test_reconstruction - test_tensor) ** 2).mean(dim=1).cpu().numpy()
    autoencoder_threshold = float(np.quantile(autoencoder_train_scores, quantile))
    statistical_metrics = binary_metrics(labels, statistical_scores, statistical_threshold)
    autoencoder_metrics = binary_metrics(labels, autoencoder_scores, autoencoder_threshold)

    artifacts: list[Path] = []
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(losses, color="#c44e52")
    axis.set(title="Autoencoder reconstruction training", xlabel="epoch", ylabel="MSE")
    axis.grid(alpha=0.25)
    artifacts.append(save_figure(fig, output / "autoencoder_loss.png"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(test[labels == 0, 0], test[labels == 0, 1], s=12, alpha=0.55, label="normal")
    axes[0].scatter(test[labels == 1, 0], test[labels == 1, 1], s=16, alpha=0.7, label="anomaly")
    axes[0].set_title("Ground truth")
    axes[0].legend(fontsize=8)
    for axis, scores, title in (
        (axes[1], statistical_scores, "Mahalanobis score"),
        (axes[2], autoencoder_scores, "Autoencoder error"),
    ):
        points = axis.scatter(test[:, 0], test[:, 1], c=scores, s=14, cmap="magma")
        axis.set_title(title)
        fig.colorbar(points, ax=axis)
    for axis in axes:
        axis.set(xlabel="feature 0", ylabel="feature 1")
    artifacts.append(save_figure(fig, output / "anomaly_maps.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, scores, threshold, title in (
        (axes[0], statistical_scores, statistical_threshold, "Mahalanobis"),
        (axes[1], autoencoder_scores, autoencoder_threshold, "Autoencoder"),
    ):
        axis.hist(scores[labels == 0], bins=25, alpha=0.65, label="normal")
        axis.hist(scores[labels == 1], bins=25, alpha=0.65, label="anomaly")
        axis.axvline(threshold, color="black", linestyle="--", label="train 95% threshold")
        axis.set(title=title, xlabel="anomaly score", ylabel="count")
        axis.legend(fontsize=8)
    artifacts.append(save_figure(fig, output / "score_distributions.png"))

    metrics = {
        "mahalanobis": statistical_metrics,
        "autoencoder": autoencoder_metrics,
        "autoencoder_initial_loss": losses[0],
        "autoencoder_final_loss": losses[-1],
        "autoencoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    return finish(
        experiment="ml_anomaly",
        started_at=started,
        output_dir=output,
        metrics=metrics,
        parameters=cfg,
        artifacts=artifacts,
        notes=[
            "Only normal samples train the detector; thresholds come from a normal-score quantile.",
            "Mahalanobis assumes an elliptical distribution, while the autoencoder learns a nonlinear manifold.",
        ],
    )
