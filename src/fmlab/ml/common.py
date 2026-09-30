from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

from fmlab.artifacts import ExperimentResult


def merged(defaults: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(defaults)
    if override:
        unknown = sorted(set(override) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        result.update(override)
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def prepare(output_dir: str | Path, seed: int) -> tuple[Path, float]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    return path, time.time()


def finish(
    *,
    experiment: str,
    started_at: float,
    output_dir: Path,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    artifacts: list[Path],
    notes: list[str],
) -> ExperimentResult:
    result = ExperimentResult(
        experiment=experiment,
        status="completed",
        started_at=started_at,
        finished_at=time.time(),
        metrics=to_jsonable(metrics),
        parameters=to_jsonable(parameters),
        artifacts=[str(path.relative_to(output_dir)) for path in artifacts],
        notes=notes,
    )
    result.write(output_dir)
    return result


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def mse(values: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((values - targets) ** 2))


def mae(values: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(values - targets)))


def rmse(values: np.ndarray, targets: np.ndarray) -> float:
    return math.sqrt(mse(values, targets))


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC as the probability that a positive outranks a negative."""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auroc": binary_auroc(labels, scores),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "threshold": float(threshold),
    }
