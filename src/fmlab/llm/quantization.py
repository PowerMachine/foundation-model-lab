"""Transparent weight-only quantization simulations and benchmarks.

The toy implementation stores signed quantized values in int8 tensors even for
4-bit experiments.  ``estimated_packed_bytes`` reports the size a packed backend
would use.  This separation makes quantization error inspectable on CPU while
avoiding a compiled CUDA dependency.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class QuantizedTensor:
    values: torch.Tensor
    scales: torch.Tensor
    bits: int
    original_shape: tuple[int, ...]
    per_channel: bool

    def dequantize(self) -> torch.Tensor:
        if self.per_channel:
            shape = (self.scales.shape[0],) + (1,) * (self.values.ndim - 1)
            return self.values.float() * self.scales.reshape(shape)
        return self.values.float() * self.scales

    @property
    def estimated_packed_bytes(self) -> int:
        value_bytes = math.ceil(self.values.numel() * self.bits / 8)
        return value_bytes + self.scales.numel() * self.scales.element_size()


def symmetric_quantize(
    values: torch.Tensor, *, bits: int = 8, per_channel: bool = True
) -> QuantizedTensor:
    """Quantize a tensor symmetrically, normally per output channel."""

    if bits not in (2, 3, 4, 8):
        raise ValueError("bits must be one of 2, 3, 4, or 8")
    if not values.is_floating_point():
        raise TypeError("only floating-point tensors can be quantized")
    levels = 2 ** (bits - 1) - 1
    source = values.detach().float()
    use_channels = per_channel and source.ndim >= 2
    if use_channels:
        reduce_dims = tuple(range(1, source.ndim))
        maxima = source.abs().amax(dim=reduce_dims)
        scales = (maxima / levels).clamp_min(torch.finfo(torch.float32).eps)
        reshape = (source.shape[0],) + (1,) * (source.ndim - 1)
        quantized = torch.round(source / scales.reshape(reshape)).clamp(-levels, levels)
    else:
        scales = (source.abs().max() / levels).clamp_min(torch.finfo(torch.float32).eps)
        quantized = torch.round(source / scales).clamp(-levels, levels)
    return QuantizedTensor(
        values=quantized.to(torch.int8),
        scales=scales,
        bits=bits,
        original_shape=tuple(values.shape),
        per_channel=use_channels,
    )


def fake_quantize(values: torch.Tensor, *, bits: int = 4) -> torch.Tensor:
    """Quantize then dequantize, preserving the source dtype and device."""

    quantized = symmetric_quantize(values, bits=bits)
    return quantized.dequantize().to(device=values.device, dtype=values.dtype)


@dataclass(frozen=True)
class QuantizationReport:
    bits: int
    parameter_count: int
    original_bytes: int
    estimated_packed_bytes: int
    compression_ratio: float
    mean_absolute_error: float
    root_mean_squared_error: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bits": self.bits,
            "parameter_count": self.parameter_count,
            "original_bytes": self.original_bytes,
            "estimated_packed_bytes": self.estimated_packed_bytes,
            "compression_ratio": self.compression_ratio,
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "cosine_similarity": self.cosine_similarity,
        }


def benchmark_weight_quantization(model: nn.Module, *, bits: int) -> QuantizationReport:
    originals: list[torch.Tensor] = []
    reconstructed: list[torch.Tensor] = []
    original_bytes = 0
    packed_bytes = 0
    for parameter in model.parameters():
        if not parameter.is_floating_point() or parameter.ndim < 2:
            continue
        source = parameter.detach().cpu().float()
        quantized = symmetric_quantize(source, bits=bits)
        originals.append(source.flatten())
        reconstructed.append(quantized.dequantize().flatten())
        original_bytes += parameter.numel() * parameter.element_size()
        packed_bytes += quantized.estimated_packed_bytes
    if not originals:
        raise ValueError("model has no matrix-shaped floating-point parameters")
    source_vector = torch.cat(originals)
    restored_vector = torch.cat(reconstructed)
    difference = source_vector - restored_vector
    cosine = torch.nn.functional.cosine_similarity(
        source_vector[None], restored_vector[None]
    ).item()
    return QuantizationReport(
        bits=bits,
        parameter_count=source_vector.numel(),
        original_bytes=original_bytes,
        estimated_packed_bytes=packed_bytes,
        compression_ratio=original_bytes / packed_bytes,
        mean_absolute_error=difference.abs().mean().item(),
        root_mean_squared_error=difference.square().mean().sqrt().item(),
        cosine_similarity=cosine,
    )


def fake_quantized_copy(model: nn.Module, *, bits: int) -> nn.Module:
    """Return a float model whose matrix weights have quantization error."""

    copied = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in copied.parameters():
            if parameter.is_floating_point() and parameter.ndim >= 2:
                parameter.copy_(fake_quantize(parameter, bits=bits))
    return copied


@torch.inference_mode()
def benchmark_forward_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    warmup: int = 2,
    iterations: int = 10,
) -> dict[str, float]:
    """Measure median-like average latency for a common input batch."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    model.eval()
    device = next(model.parameters()).device
    inputs = input_ids.to(device)
    for _ in range(warmup):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = input_ids.numel() * iterations
    return {
        "latency_ms": elapsed * 1_000 / iterations,
        "tokens_per_second": tokens / elapsed,
    }


def compare_logits(
    reference_model: nn.Module,
    candidate_models: Iterable[tuple[str, nn.Module]],
    input_ids: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Compare output distributions after simulated quantization."""

    reference_model.eval()
    with torch.inference_mode():
        reference = reference_model(input_ids).logits.float()
        output: dict[str, dict[str, float]] = {}
        for name, candidate in candidate_models:
            candidate.eval()
            logits = candidate(input_ids).logits.float()
            difference = reference - logits
            output[name] = {
                "logit_mae": difference.abs().mean().item(),
                "logit_rmse": difference.square().mean().sqrt().item(),
            }
    return output
