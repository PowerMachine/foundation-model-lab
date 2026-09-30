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
    "seed": 29,
    "device": "auto",
    "class_count": 3,
    "nodes_per_class": 30,
    "feature_dim": 8,
    "hidden_dim": 16,
    "within_class_edge_probability": 0.20,
    "between_class_edge_probability": 0.018,
    "feature_signal": 0.55,
    "train_per_class": 5,
    "validation_per_class": 5,
    "epochs": 160,
    "learning_rate": 0.012,
    "weight_decay": 0.0005,
}


def make_community_graph(
    class_count: int,
    nodes_per_class: int,
    feature_dim: int,
    within_probability: float,
    between_probability: float,
    feature_signal: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    node_count = class_count * nodes_per_class
    labels = np.repeat(np.arange(class_count), nodes_per_class)
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    for left in range(node_count):
        for right in range(left + 1, node_count):
            probability = (
                within_probability if labels[left] == labels[right] else between_probability
            )
            if rng.random() < probability:
                adjacency[left, right] = adjacency[right, left] = 1.0
    # Avoid isolated nodes by connecting them inside their community.
    for node in range(node_count):
        if adjacency[node].sum() == 0:
            start = labels[node] * nodes_per_class
            candidates = np.arange(start, start + nodes_per_class)
            candidates = candidates[candidates != node]
            neighbour = int(rng.choice(candidates))
            adjacency[node, neighbour] = adjacency[neighbour, node] = 1.0

    centroids = rng.normal(0.0, 1.0, size=(class_count, feature_dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-6)
    features = rng.normal(0.0, 1.0, size=(node_count, feature_dim)).astype(np.float32)
    features += feature_signal * centroids[labels]

    angles = np.linspace(0, 2 * np.pi, class_count, endpoint=False)
    centers = np.column_stack((2.8 * np.cos(angles), 2.8 * np.sin(angles)))
    positions = centers[labels] + rng.normal(0.0, 0.65, size=(node_count, 2))
    return (
        torch.from_numpy(features),
        torch.from_numpy(adjacency),
        torch.from_numpy(labels).long(),
        positions,
    )


def make_masks(
    labels: torch.Tensor, train_per_class: int, validation_per_class: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train = torch.zeros(len(labels), dtype=torch.bool)
    validation = torch.zeros_like(train)
    for label in labels.unique():
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        if len(indices) <= train_per_class + validation_per_class:
            raise ValueError("nodes_per_class must exceed train_per_class + validation_per_class")
        train[indices[:train_per_class]] = True
        validation[indices[train_per_class : train_per_class + validation_per_class]] = True
    return train, validation, ~(train | validation)


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    with_self = adjacency + torch.eye(len(adjacency), device=adjacency.device)
    inverse_sqrt_degree = with_self.sum(dim=1).clamp(min=1).pow(-0.5)
    return inverse_sqrt_degree[:, None] * with_self * inverse_sqrt_degree[None, :]


class MLPNodeClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, class_count)
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        del adjacency
        return self.layers(features)


class GCNNodeClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, class_count, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(adjacency @ self.input(features))
        return adjacency @ self.output(hidden)


class DenseGraphAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.source_score = nn.Linear(output_dim, 1, bias=False)
        self.target_score = nn.Linear(output_dim, 1, bias=False)
        self.last_attention: torch.Tensor | None = None

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(features)
        scores = self.source_score(hidden) + self.target_score(hidden).T
        scores = F.leaky_relu(scores, negative_slope=0.2)
        mask = adjacency > 0
        attention = torch.softmax(scores.masked_fill(~mask, -1e9), dim=1)
        self.last_attention = attention.detach()
        return attention @ hidden


class GATNodeClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.input = DenseGraphAttention(feature_dim, hidden_dim)
        self.output = DenseGraphAttention(hidden_dim, class_count)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.input(features, adjacency))
        return self.output(hidden, adjacency)


def _train_model(
    model: nn.Module,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, float], list[float], list[float], torch.Tensor]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses: list[float] = []
    validation_accuracies: list[float] = []
    for _ in range(epochs):
        model.train()
        logits = model(features, adjacency)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(features, adjacency)
            validation_accuracies.append(
                classification_accuracy(validation_logits[validation_mask], labels[validation_mask])
            )
    model.eval()
    with torch.no_grad():
        logits = model(features, adjacency)
    metrics = {
        "train_accuracy": classification_accuracy(logits[train_mask], labels[train_mask]),
        "validation_accuracy": classification_accuracy(
            logits[validation_mask], labels[validation_mask]
        ),
        "test_accuracy": classification_accuracy(logits[test_mask], labels[test_mask]),
        "final_loss": losses[-1],
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
    }
    return metrics, losses, validation_accuracies, logits.cpu()


def _draw_graph(
    axis: plt.Axes,
    adjacency: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    title: str,
    train_mask: np.ndarray | None = None,
) -> None:
    rows, columns = np.where(np.triu(adjacency, 1) > 0)
    for left, right in zip(rows, columns, strict=True):
        axis.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="#b0b0b0",
            alpha=0.20,
            linewidth=0.45,
            zorder=0,
        )
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=labels,
        cmap="tab10",
        s=28,
        edgecolors="white",
        linewidths=0.35,
        zorder=1,
    )
    if train_mask is not None:
        axis.scatter(
            positions[train_mask, 0],
            positions[train_mask, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            s=70,
            label="labeled for training",
        )
        axis.legend(fontsize=7, loc="upper right")
    axis.set_title(title)
    axis.axis("off")


def run_graph(
    config: dict[str, Any] | None = None, output_dir: str | Path = "artifacts/ml/graph"
) -> ExperimentResult:
    cfg = merged(DEFAULTS, config)
    output, started = prepare(output_dir, int(cfg["seed"]))
    device = choose_device(str(cfg["device"]))
    features, adjacency_raw, labels, positions = make_community_graph(
        int(cfg["class_count"]),
        int(cfg["nodes_per_class"]),
        int(cfg["feature_dim"]),
        float(cfg["within_class_edge_probability"]),
        float(cfg["between_class_edge_probability"]),
        float(cfg["feature_signal"]),
        int(cfg["seed"]),
    )
    train_mask, validation_mask, test_mask = make_masks(
        labels,
        int(cfg["train_per_class"]),
        int(cfg["validation_per_class"]),
        int(cfg["seed"]),
    )
    attention_adjacency = adjacency_raw + torch.eye(len(adjacency_raw))
    convolution_adjacency = normalized_adjacency(adjacency_raw)
    features = features.to(device)
    labels_device = labels.to(device)
    train_mask_device = train_mask.to(device)
    validation_mask_device = validation_mask.to(device)
    test_mask_device = test_mask.to(device)

    model_specs: dict[str, tuple[nn.Module, torch.Tensor]] = {
        "mlp": (
            MLPNodeClassifier(
                int(cfg["feature_dim"]), int(cfg["hidden_dim"]), int(cfg["class_count"])
            ),
            convolution_adjacency,
        ),
        "gcn": (
            GCNNodeClassifier(
                int(cfg["feature_dim"]), int(cfg["hidden_dim"]), int(cfg["class_count"])
            ),
            convolution_adjacency,
        ),
        "gat": (
            GATNodeClassifier(
                int(cfg["feature_dim"]), int(cfg["hidden_dim"]), int(cfg["class_count"])
            ),
            attention_adjacency,
        ),
    }
    all_metrics: dict[str, dict[str, float]] = {}
    all_losses: dict[str, list[float]] = {}
    all_validation: dict[str, list[float]] = {}
    all_logits: dict[str, torch.Tensor] = {}
    for offset, (name, (model, model_adjacency)) in enumerate(model_specs.items()):
        torch.manual_seed(int(cfg["seed"]) + offset)
        model = model.to(device)
        metrics, losses, validation, logits = _train_model(
            model,
            features,
            model_adjacency.to(device),
            labels_device,
            train_mask_device,
            validation_mask_device,
            test_mask_device,
            int(cfg["epochs"]),
            float(cfg["learning_rate"]),
            float(cfg["weight_decay"]),
        )
        all_metrics[name] = metrics
        all_losses[name] = losses
        all_validation[name] = validation
        all_logits[name] = logits

    artifacts: list[Path] = []
    adjacency_numpy = adjacency_raw.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    _draw_graph(
        axes[0], adjacency_numpy, positions, labels.numpy(), "True communities", train_mask.numpy()
    )
    _draw_graph(
        axes[1],
        adjacency_numpy,
        positions,
        all_logits["gcn"].argmax(dim=1).numpy(),
        "GCN predictions",
    )
    _draw_graph(
        axes[2],
        adjacency_numpy,
        positions,
        all_logits["gat"].argmax(dim=1).numpy(),
        "GAT predictions",
    )
    artifacts.append(save_figure(fig, output / "community_predictions.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in model_specs:
        axes[0].plot(all_losses[name], label=name)
        axes[1].plot(all_validation[name], label=name)
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="cross entropy")
    axes[1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1.05))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    artifacts.append(save_figure(fig, output / "training_curves.png"))

    fig, axis = plt.subplots(figsize=(6, 3.6))
    names = list(model_specs)
    values = [all_metrics[name]["test_accuracy"] for name in names]
    bars = axis.bar(names, values, color=["#999999", "#4c72b0", "#c44e52"])
    axis.set(title="Held-out node classification", ylabel="test accuracy", ylim=(0, 1.05))
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom"
        )
    artifacts.append(save_figure(fig, output / "model_comparison.png"))

    best_model = max(all_metrics, key=lambda name: all_metrics[name]["test_accuracy"])
    metrics: dict[str, Any] = {
        "models": all_metrics,
        "best_test_model": best_model,
        "edge_count": int(adjacency_raw.sum().item() // 2),
        "labeled_node_count": int(train_mask.sum()),
        "test_node_count": int(test_mask.sum()),
        "device": str(device),
    }
    return finish(
        experiment="ml_graph",
        started_at=started,
        output_dir=output,
        metrics=metrics,
        parameters=cfg,
        artifacts=artifacts,
        notes=[
            "MLP sees only node features; GCN averages neighbours; GAT learns neighbour weights.",
            "Black outlines mark the few labeled nodes, so this is semi-supervised node classification.",
        ],
    )
