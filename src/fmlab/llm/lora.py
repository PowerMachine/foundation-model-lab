"""Dependency-free LoRA and an inspectable QLoRA simulation for tiny models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import symmetric_quantize


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    quantization_bits: int | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.quantization_bits not in (None, 4, 8):
            raise ValueError("quantization_bits must be None, 4, or 8")


class FrozenQuantizedLinear(nn.Module):
    """Frozen quantize/dequantize linear layer used by the offline QLoRA demo."""

    def __init__(self, source: nn.Linear, *, bits: int) -> None:
        super().__init__()
        quantized = symmetric_quantize(source.weight.detach().cpu(), bits=bits)
        self.register_buffer("quantized_weight", quantized.values)
        self.register_buffer("scales", quantized.scales)
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bits = bits

    def weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale_shape = (self.out_features,) + (1,) * (self.quantized_weight.ndim - 1)
        return (
            self.quantized_weight.to(device=device).float()
            * self.scales.to(device=device).reshape(scale_shape)
        ).to(dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(values.device, values.dtype)
        return F.linear(values, self.weight(device=values.device, dtype=values.dtype), bias)


class LoRALinear(nn.Module):
    def __init__(self, source: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if config.quantization_bits is None:
            self.base: nn.Module = source
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
        else:
            self.base = FrozenQuantizedLinear(source, bits=config.quantization_bits)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = config.rank
        self.scale = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_a = nn.Parameter(torch.empty(config.rank, source.in_features))
        self.lora_b = nn.Parameter(torch.zeros(source.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        adapter = F.linear(F.linear(self.dropout(values), self.lora_a), self.lora_b)
        return self.base(values) + adapter * self.scale

    def merged_weight(self) -> torch.Tensor:
        if not isinstance(self.base, nn.Linear):
            raise RuntimeError("quantized LoRA weights cannot be losslessly merged")
        return self.base.weight.detach() + self.scale * (self.lora_b @ self.lora_a)


def _replace_targets(module: nn.Module, config: LoRAConfig, prefix: str = "") -> list[str]:
    replaced: list[str] = []
    for child_name, child in list(module.named_children()):
        qualified = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and child_name in config.target_modules:
            setattr(module, child_name, LoRALinear(child, config))
            replaced.append(qualified)
        else:
            replaced.extend(_replace_targets(child, config, qualified))
    return replaced


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Freeze a model and replace matching linear modules with LoRA layers."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = _replace_targets(model, config)
    if not replaced:
        raise ValueError(f"no linear modules matched {config.target_modules}")
    return replaced


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }


def parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }
