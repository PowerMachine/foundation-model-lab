"""Small training loops with metrics intended for learning, not scale."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20
    batch_size: int = 4
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def evaluate_language_model(
    model: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    target = torch.device(device)
    model.to(target).eval()
    losses: list[float] = []
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            output = model(**_to_device(batch, target))
            if output.loss is not None and torch.isfinite(output.loss):
                losses.append(output.loss.item())
    if not losses:
        raise RuntimeError("evaluation produced no finite losses")
    mean_loss = float(np.mean(losses))
    return {"loss": mean_loss, "perplexity": float(np.exp(min(mean_loss, 20.0)))}


def train_language_model(
    model: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    config: TrainConfig,
) -> list[dict[str, float | int]]:
    """Train any model returning ``.loss`` and retain a compact learning trace."""

    seed_everything(config.seed)
    device = resolve_device(config.device)
    model.to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    batches = itertools.cycle(loader)
    history: list[dict[str, float | int]] = []
    for step in range(1, config.steps + 1):
        batch = _to_device(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(f"non-finite or missing loss at step {step}")
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": output.loss.detach().item(),
                "gradient_norm": float(gradient_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    return history


def count_parameters(model: nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }
