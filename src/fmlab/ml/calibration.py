from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmlab.artifacts import ExperimentResult

from .common import choose_device, classification_accuracy, finish, merged, prepare, save_figure


DEFAULTS: dict[str, Any] = {
    "seed": 31,
    "device": "auto",
    "class_count": 3,
    "train_count": 600,
    "validation_count": 240,
    "test_count": 360,
    "hidden_dim": 32,
    "epochs": 180,
    "learning_rate": 0.008,
    "logit_multiplier": 2.8,
    "calibration_bins": 10,
    "temperature_candidates": 100,
}


def make_classification_data(
    count: int, class_count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, class_count, endpoint=False)
    centers = np.column_stack((1.7 * np.cos(angles), 1.7 * np.sin(angles)))
    labels = rng.integers(0, class_count, size=count)
    features = centers[labels] + rng.normal(0.0, 1.05, size=(count, 2))
    return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels).long()


class ConfidenceMLP(nn.Module):
    def __init__(self, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def calibration_statistics(
    probabilities: torch.Tensor, labels: torch.Tensor, bin_count: int
) -> tuple[float, list[dict[str, float]]]:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0, 1, bin_count + 1, device=probabilities.device)
    expected_error = torch.tensor(0.0, device=probabilities.device)
    bins: list[dict[str, float]] = []
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        count = int(selected.sum().item())
        if count:
            average_confidence = confidence[selected].mean()
            average_accuracy = correct[selected].mean()
            weight = selected.float().mean()
            expected_error += weight * (average_confidence - average_accuracy).abs()
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": count,
                    "confidence": float(average_confidence),
                    "accuracy": float(average_accuracy),
                }
            )
        else:
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "confidence": float((lower + upper) / 2),
                    "accuracy": float("nan"),
                }
            )
    return float(expected_error), bins


def _probability_metrics(
    logits: torch.Tensor, labels: torch.Tensor, bin_count: int
) -> tuple[dict[str, float], list[dict[str, float]]]:
    probabilities = logits.softmax(dim=1)
    ece, bins = calibration_statistics(probabilities, labels, bin_count)
    targets = F.one_hot(labels, num_classes=logits.shape[1]).float()
    return (
        {
            "accuracy": classification_accuracy(logits, labels),
            "mean_confidence": float(probabilities.max(dim=1).values.mean()),
            "nll": float(F.cross_entropy(logits, labels)),
            "brier_score": float(((probabilities - targets) ** 2).sum(dim=1).mean()),
            "ece": ece,
        },
        bins,
    )


def select_temperature(logits: torch.Tensor, labels: torch.Tensor, candidates: int) -> float:
    candidates = max(candidates, 3)
    temperatures = torch.logspace(np.log10(0.25), np.log10(6.0), candidates, device=logits.device)
    losses = torch.stack([F.cross_entropy(logits / value, labels) for value in temperatures])
    return float(temperatures[losses.argmin()].item())


def _plot_reliability(axis: plt.Axes, bins: list[dict[str, float]], title: str, color: str) -> None:
    valid = [value for value in bins if value["count"] > 0]
    confidence = [value["confidence"] for value in valid]
    accuracy = [value["accuracy"] for value in valid]
    axis.plot([0, 1], [0, 1], "--", color="black", alpha=0.7, label="perfect calibration")
    axis.plot(confidence, accuracy, marker="o", color=color, label="observed")
    axis.fill_between(confidence, confidence, accuracy, color=color, alpha=0.16)
    axis.set(
        title=title, xlabel="mean confidence", ylabel="fraction correct", xlim=(0, 1), ylim=(0, 1)
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)


def run_calibration(
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/ml/calibration",
) -> ExperimentResult:
    cfg = merged(DEFAULTS, config)
    output, started = prepare(output_dir, int(cfg["seed"]))
    device = choose_device(str(cfg["device"]))
    class_count = int(cfg["class_count"])
    train_x, train_y = make_classification_data(
        int(cfg["train_count"]), class_count, int(cfg["seed"])
    )
    validation_x, validation_y = make_classification_data(
        int(cfg["validation_count"]), class_count, int(cfg["seed"]) + 1
    )
    test_x, test_y = make_classification_data(
        int(cfg["test_count"]), class_count, int(cfg["seed"]) + 2
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    validation_x, validation_y = validation_x.to(device), validation_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    model = ConfidenceMLP(int(cfg["hidden_dim"]), class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    losses: list[float] = []
    for _ in range(int(cfg["epochs"])):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    multiplier = float(cfg["logit_multiplier"])
    with torch.no_grad():
        validation_logits = model(validation_x) * multiplier
        test_logits = model(test_x) * multiplier
    temperature = select_temperature(
        validation_logits, validation_y, int(cfg["temperature_candidates"])
    )
    calibrated_logits = test_logits / temperature
    before, bins_before = _probability_metrics(test_logits, test_y, int(cfg["calibration_bins"]))
    after, bins_after = _probability_metrics(
        calibrated_logits, test_y, int(cfg["calibration_bins"])
    )

    artifacts: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _plot_reliability(axes[0], bins_before, f"Before (ECE={before['ece']:.3f})", "#c44e52")
    _plot_reliability(axes[1], bins_after, f"After (ECE={after['ece']:.3f})", "#55a868")
    fig.suptitle("Reliability diagrams: confidence should match accuracy")
    artifacts.append(save_figure(fig, output / "reliability_diagram.png"))

    confidence_before = test_logits.softmax(dim=1).max(dim=1).values.cpu().numpy()
    confidence_after = calibrated_logits.softmax(dim=1).max(dim=1).values.cpu().numpy()
    fig, axis = plt.subplots(figsize=(6, 3.6))
    axis.hist(confidence_before, bins=15, alpha=0.60, label="before")
    axis.hist(confidence_after, bins=15, alpha=0.60, label="after")
    axis.set(title="Confidence distribution", xlabel="maximum class probability", ylabel="count")
    axis.legend()
    artifacts.append(save_figure(fig, output / "confidence_histogram.png"))

    grid_x, grid_y = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(-4, 4, 120))
    grid = torch.from_numpy(
        np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        grid_logits = model(grid) * multiplier
        confidence_maps = [
            grid_logits.softmax(dim=1).max(dim=1).values.reshape(grid_x.shape).cpu().numpy(),
            (grid_logits / temperature)
            .softmax(dim=1)
            .max(dim=1)
            .values.reshape(grid_x.shape)
            .cpu()
            .numpy(),
        ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, values, title in zip(
        axes,
        confidence_maps,
        ("Before scaling", f"After scaling (T={temperature:.2f})"),
        strict=True,
    ):
        image = axis.contourf(
            grid_x, grid_y, values, levels=np.linspace(0.33, 1.0, 14), cmap="viridis"
        )
        axis.scatter(
            test_x[:, 0].cpu(), test_x[:, 1].cpu(), c=test_y.cpu(), cmap="tab10", s=5, alpha=0.35
        )
        axis.set(title=title, xlabel="feature 0", ylabel="feature 1")
        fig.colorbar(image, ax=axis, label="confidence")
    artifacts.append(save_figure(fig, output / "confidence_surface.png"))

    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(losses, color="#4c72b0")
    axis.set(title="Classifier training", xlabel="epoch", ylabel="cross entropy")
    axis.grid(alpha=0.25)
    artifacts.append(save_figure(fig, output / "training_loss.png"))

    metrics = {
        "before_temperature_scaling": before,
        "after_temperature_scaling": after,
        "selected_temperature": temperature,
        "injected_logit_multiplier": multiplier,
        "ece_reduction": before["ece"] - after["ece"],
        "nll_reduction": before["nll"] - after["nll"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    return finish(
        experiment="ml_calibration",
        started_at=started,
        output_dir=output,
        metrics=metrics,
        parameters=cfg,
        artifacts=artifacts,
        notes=[
            "Temperature scaling changes confidence but never changes the predicted class.",
            "A logit multiplier intentionally demonstrates overconfidence; temperature is selected on validation data only.",
        ],
    )
