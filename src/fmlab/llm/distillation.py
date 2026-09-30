"""Knowledge-distillation objectives for teacher/student experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 2.0
    soft_target_weight: float = 0.7
    hard_target_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.soft_target_weight < 0 or self.hard_target_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.soft_target_weight + self.hard_target_weight == 0:
            raise ValueError("at least one loss weight must be positive")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have the same shape")
    student = student_logits[:, :-1]
    teacher = teacher_logits[:, :-1].detach()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    temperature = config.temperature
    token_kl = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    soft = (token_kl * valid).sum() / valid.sum().clamp_min(1)
    soft = soft * temperature**2
    hard = F.cross_entropy(
        student.reshape(-1, student.shape[-1]), targets.reshape(-1), ignore_index=-100
    )
    total = config.soft_target_weight * soft + config.hard_target_weight * hard
    return total, {
        "total_loss": total.detach().item(),
        "soft_kl": soft.detach().item(),
        "hard_cross_entropy": hard.detach().item(),
    }
