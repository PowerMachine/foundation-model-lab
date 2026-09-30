from __future__ import annotations

import math
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
    "seed": 7,
    "device": "auto",
    "image_size": 8,
    "dataset_size": 256,
    "batch_size": 64,
    "timesteps": 24,
    "train_steps": 160,
    "hidden_channels": 32,
    "learning_rate": 0.002,
    "sample_count": 16,
}


def make_pattern_images(count: int, size: int, seed: int) -> torch.Tensor:
    """Create a tiny image distribution with bars, crosses, and diagonals."""
    rng = np.random.default_rng(seed)
    images = np.full((count, 1, size, size), -1.0, dtype=np.float32)
    margin = max(1, size // 4)
    for index in range(count):
        kind = index % 4
        position = int(rng.integers(margin, max(margin + 1, size - margin)))
        if kind == 0:
            images[index, 0, :, position] = 1.0
        elif kind == 1:
            images[index, 0, position, :] = 1.0
        elif kind == 2:
            np.fill_diagonal(images[index, 0], 1.0)
            if rng.random() < 0.5:
                images[index, 0] = np.fliplr(images[index, 0])
        else:
            images[index, 0, :, position] = 1.0
            images[index, 0, position, :] = 1.0
        images[index] += rng.normal(0.0, 0.06, size=(1, size, size)).astype(np.float32)
    return torch.from_numpy(np.clip(images, -1.0, 1.0))


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device))
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TinyNoisePredictor(nn.Module):
    def __init__(self, hidden_channels: int, time_dimension: int = 32) -> None:
        super().__init__()
        self.time_dimension = time_dimension
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dimension, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.input = nn.Conv2d(1, hidden_channels, 3, padding=1)
        self.middle = nn.Sequential(
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.output = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden_channels, 1, 3, padding=1))

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        hidden = self.input(noisy)
        time = self.time_mlp(sinusoidal_embedding(timesteps, self.time_dimension))
        hidden = hidden + time[:, :, None, None]
        return self.output(hidden + self.middle(hidden))


def _extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    return values[timesteps].view(-1, *([1] * (ndim - 1)))


def q_sample(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    cumulative_alphas: torch.Tensor,
) -> torch.Tensor:
    signal = _extract(cumulative_alphas.sqrt(), timesteps, clean.ndim)
    noise_scale = _extract((1.0 - cumulative_alphas).sqrt(), timesteps, clean.ndim)
    return signal * clean + noise_scale * noise


@torch.no_grad()
def reverse_sample(
    model: nn.Module,
    sample_count: int,
    image_size: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    samples = torch.randn(sample_count, 1, image_size, image_size, device=device)
    for step in reversed(range(len(betas))):
        t = torch.full((sample_count,), step, dtype=torch.long, device=device)
        predicted_noise = model(samples, t)
        mean = samples - betas[step] / (1.0 - cumulative_alphas[step]).sqrt() * predicted_noise
        mean = mean / alphas[step].sqrt()
        if step > 0:
            posterior_variance = (
                betas[step] * (1.0 - cumulative_alphas[step - 1]) / (1.0 - cumulative_alphas[step])
            )
            samples = mean + posterior_variance.sqrt() * torch.randn_like(samples)
        else:
            samples = mean
    return samples.clamp(-1.0, 1.0).cpu()


def _plot_grid(images: torch.Tensor, title: str, path: Path) -> Path:
    count = len(images)
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.8 * columns, 1.8 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, image in zip(axes.ravel(), images, strict=False):
        axis.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
        axis.axis("off")
    fig.suptitle(title)
    return save_figure(fig, path)


def run_diffusion(
    config: dict[str, Any] | None = None, output_dir: str | Path = "artifacts/ml/diffusion"
) -> ExperimentResult:
    cfg = merged(DEFAULTS, config)
    output, started = prepare(output_dir, int(cfg["seed"]))
    device = choose_device(str(cfg["device"]))
    size = int(cfg["image_size"])
    data = make_pattern_images(int(cfg["dataset_size"]), size, int(cfg["seed"]))
    betas = torch.linspace(1e-4, 0.12, int(cfg["timesteps"]), device=device)
    alphas = 1.0 - betas
    cumulative_alphas = torch.cumprod(alphas, dim=0)
    model = TinyNoisePredictor(int(cfg["hidden_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    losses: list[float] = []

    model.train()
    batch_size = min(int(cfg["batch_size"]), len(data))
    for _ in range(int(cfg["train_steps"])):
        indices = torch.randint(0, len(data), (batch_size,))
        clean = data[indices].to(device)
        timesteps = torch.randint(0, len(betas), (batch_size,), device=device)
        noise = torch.randn_like(clean)
        noisy = q_sample(clean, timesteps, noise, cumulative_alphas)
        loss = F.mse_loss(model(noisy, timesteps), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    generated = reverse_sample(
        model,
        int(cfg["sample_count"]),
        size,
        betas,
        alphas,
        cumulative_alphas,
        device,
    )
    artifacts: list[Path] = []
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(losses, color="#4c72b0")
    axis.set(title="DDPM noise-prediction training", xlabel="optimization step", ylabel="MSE")
    axis.grid(alpha=0.25)
    artifacts.append(save_figure(fig, output / "training_loss.png"))

    clean = data[:1].to(device)
    checkpoints = [0, len(betas) // 3, 2 * len(betas) // 3, len(betas) - 1]
    fixed_noise = torch.randn_like(clean)
    forward_images = [
        q_sample(clean, torch.tensor([step], device=device), fixed_noise, cumulative_alphas).cpu()[
            0
        ]
        for step in checkpoints
    ]
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(7, 2.2))
    for axis, image, step in zip(axes, forward_images, checkpoints, strict=True):
        axis.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
        axis.set_title(f"t={step}")
        axis.axis("off")
    fig.suptitle("Forward diffusion: structure is gradually destroyed")
    artifacts.append(save_figure(fig, output / "forward_process.png"))
    artifacts.append(_plot_grid(generated, "Reverse diffusion samples", output / "samples.png"))

    flat_generated = generated.flatten(1)
    flat_training = data.flatten(1)
    nearest_mse = (
        ((flat_generated[:, None] - flat_training[None, :]) ** 2).mean(dim=-1).min(dim=1).values
    )
    metrics = {
        "initial_noise_mse": losses[0],
        "final_noise_mse": losses[-1],
        "best_noise_mse": min(losses),
        "generated_nearest_training_mse": float(nearest_mse.mean()),
        "generated_pixel_std": float(generated.std()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    return finish(
        experiment="ml_diffusion",
        started_at=started,
        output_dir=output,
        metrics=metrics,
        parameters=cfg,
        artifacts=artifacts,
        notes=[
            "The model predicts added Gaussian noise; reverse sampling repeatedly removes that estimate.",
            "The nearest-training MSE is a toy fidelity indicator, not a perceptual generation metric.",
        ],
    )
