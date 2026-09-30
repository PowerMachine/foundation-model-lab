from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmlab.artifacts import ExperimentResult

from .common import choose_device, finish, merged, prepare, save_figure


DEFAULTS: dict[str, Any] = {
    "seed": 11,
    "device": "auto",
    "class_count": 8,
    "image_size": 10,
    "embedding_dim": 16,
    "hidden_dim": 48,
    "batch_size": 8,
    "train_steps": 140,
    "learning_rate": 0.003,
    "image_noise": 0.15,
}


def make_symbol_prototypes(class_count: int, size: int) -> torch.Tensor:
    """Create one deterministic visual symbol for every synthetic caption ID."""
    prototypes = torch.full((class_count, 1, size, size), -1.0)
    locations = [
        (2, 2),
        (2, size - 3),
        (size - 3, 2),
        (size - 3, size - 3),
        (size // 2, size // 2),
    ]
    for label in range(class_count):
        row, column = locations[label % len(locations)]
        if (label // len(locations)) % 2 == 0:
            prototypes[label, 0, max(0, row - 1) : min(size, row + 2), column] = 1.0
            prototypes[label, 0, row, max(0, column - 1) : min(size, column + 2)] = 1.0
        else:
            for offset in range(-1, 2):
                if 0 <= row + offset < size and 0 <= column + offset < size:
                    prototypes[label, 0, row + offset, column + offset] = 1.0
                if 0 <= row + offset < size and 0 <= column - offset < size:
                    prototypes[label, 0, row + offset, column - offset] = 1.0
        # A class-specific border tick disambiguates labels when locations repeat.
        prototypes[label, 0, 0, label % size] = 1.0
    return prototypes


class ImageEncoder(nn.Module):
    def __init__(self, image_size: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(images), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.embedding(token_ids)), dim=-1)


class TinyCLIP(nn.Module):
    def __init__(
        self, class_count: int, image_size: int, hidden_dim: int, embedding_dim: int
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_size, hidden_dim, embedding_dim)
        self.text_encoder = TextEncoder(class_count, embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(token_ids)
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * image_embeddings @ text_embeddings.T


def _recall_at_k(similarity: torch.Tensor, k: int) -> float:
    k = min(k, similarity.shape[1])
    expected = torch.arange(similarity.shape[0], device=similarity.device)[:, None]
    retrieved = similarity.topk(k, dim=1).indices
    return float((retrieved == expected).any(dim=1).float().mean().item())


def _pca_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return centered @ right[:2].T


def run_contrastive(
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/ml/contrastive",
) -> ExperimentResult:
    cfg = merged(DEFAULTS, config)
    output, started = prepare(output_dir, int(cfg["seed"]))
    device = choose_device(str(cfg["device"]))
    class_count = int(cfg["class_count"])
    batch_size = min(int(cfg["batch_size"]), class_count)
    prototypes = make_symbol_prototypes(class_count, int(cfg["image_size"])).to(device)
    model = TinyCLIP(
        class_count,
        int(cfg["image_size"]),
        int(cfg["hidden_dim"]),
        int(cfg["embedding_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    losses: list[float] = []

    with torch.no_grad():
        initial_similarity = model(prototypes, torch.arange(class_count, device=device))
        initial_recall = _recall_at_k(initial_similarity, 1)

    model.train()
    for _ in range(int(cfg["train_steps"])):
        token_ids = torch.randperm(class_count, device=device)[:batch_size]
        clean = prototypes[token_ids]
        images = (clean + float(cfg["image_noise"]) * torch.randn_like(clean)).clamp(-1, 1)
        logits = model(images, token_ids)
        targets = torch.arange(batch_size, device=device)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        evaluation_images = (
            prototypes + float(cfg["image_noise"]) * torch.randn_like(prototypes)
        ).clamp(-1, 1)
        image_embeddings = model.image_encoder(evaluation_images)
        text_embeddings = model.text_encoder(torch.arange(class_count, device=device))
        similarity = image_embeddings @ text_embeddings.T

    artifacts: list[Path] = []
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(losses, color="#55a868")
    axis.set(
        title="Symmetric image-text contrastive loss", xlabel="optimization step", ylabel="loss"
    )
    axis.grid(alpha=0.25)
    artifacts.append(save_figure(fig, output / "training_loss.png"))

    matrix = similarity.detach().cpu().numpy()
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set(
        title="Cosine similarity (the diagonal is the correct pairing)",
        xlabel="text symbol ID",
        ylabel="image symbol ID",
        xticks=range(class_count),
        yticks=range(class_count),
    )
    fig.colorbar(image, ax=axis, label="cosine similarity")
    artifacts.append(save_figure(fig, output / "similarity_matrix.png"))

    both = torch.cat((image_embeddings, text_embeddings)).detach().cpu().numpy()
    points = _pca_2d(both)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(points[:class_count, 0], points[:class_count, 1], marker="o", label="image")
    axis.scatter(points[class_count:, 0], points[class_count:, 1], marker="x", label="text")
    for label in range(class_count):
        axis.plot(
            [points[label, 0], points[class_count + label, 0]],
            [points[label, 1], points[class_count + label, 1]],
            color="gray",
            alpha=0.35,
        )
        axis.annotate(str(label), points[label])
    axis.set(title="Shared embedding space (PCA projection)", xlabel="PC1", ylabel="PC2")
    axis.legend()
    artifacts.append(save_figure(fig, output / "embedding_space.png"))

    metrics = {
        "initial_image_to_text_recall_at_1": initial_recall,
        "image_to_text_recall_at_1": _recall_at_k(similarity, 1),
        "image_to_text_recall_at_3": _recall_at_k(similarity, 3),
        "text_to_image_recall_at_1": _recall_at_k(similarity.T, 1),
        "text_to_image_recall_at_3": _recall_at_k(similarity.T, 3),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "learned_logit_scale": float(model.logit_scale.exp().detach().cpu()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    return finish(
        experiment="ml_contrastive",
        started_at=started,
        output_dir=output,
        metrics=metrics,
        parameters=cfg,
        artifacts=artifacts,
        notes=[
            "This is the core CLIP idea without external images: matching pairs attract and mismatches repel.",
            "Each synthetic caption is represented by a learned token ID to isolate contrastive learning.",
        ],
    )
