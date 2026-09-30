"""Direct Preference Optimization (DPO) math in a compact, testable form."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sequence_log_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    average: bool = False,
) -> torch.Tensor:
    """Return masked log-probability for each sequence."""

    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = (
        F.log_softmax(shifted_logits.float(), dim=-1)
        .gather(-1, safe_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    totals = (token_logps * mask).sum(dim=-1)
    if average:
        totals = totals / mask.sum(dim=-1).clamp_min(1)
    return totals


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the reference-relative DPO logistic objective."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    losses = -F.logsigmoid(logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses.mean(), {
        "loss": losses.mean().detach().item(),
        "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    }
