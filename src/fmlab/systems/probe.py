"""Real CPU probe for TinyDecoder output fidelity under fake weight quantization."""

from __future__ import annotations

import math
import platform
import statistics
import time
from typing import Any

import torch
from torch.nn import functional as F

from fmlab.llm.model import TinyDecoderConfig, TinyDecoderLM
from fmlab.llm.quantization import benchmark_weight_quantization, fake_quantized_copy

from .config import ProbeConfig


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@torch.inference_mode()
def _latency_samples(
    model: TinyDecoderLM,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | list[float]]:
    model.eval()
    for _ in range(warmup):
        model(input_ids)
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        model(input_ids)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    p50 = _percentile(samples, 0.50)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "p50_ms": p50,
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "input_tokens_per_second_at_p50": input_ids.numel() / p50 * 1_000.0,
    }


def _fidelity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    reference_log_prob = F.log_softmax(reference, dim=-1)
    candidate_log_prob = F.log_softmax(candidate, dim=-1)
    reference_prob = reference_log_prob.exp()
    kl = (reference_prob * (reference_log_prob - candidate_log_prob)).sum(dim=-1).mean()
    cosine = F.cosine_similarity(
        reference.reshape(-1, reference.shape[-1]),
        candidate.reshape(-1, candidate.shape[-1]),
        dim=-1,
    ).mean()
    difference = reference - candidate
    return {
        "top1_agreement": (reference.argmax(dim=-1) == candidate.argmax(dim=-1))
        .float()
        .mean()
        .item(),
        "cosine_similarity": cosine.item(),
        "kl_divergence_reference_to_candidate": kl.item(),
        "logit_mae": difference.abs().mean().item(),
        "logit_rmse": difference.square().mean().sqrt().item(),
        "logit_max_abs": difference.abs().max().item(),
    }


def _gate_variant(
    name: str,
    fidelity: dict[str, float],
    *,
    latency_ratio: float,
    config: ProbeConfig,
) -> tuple[bool, list[str]]:
    prefix = "int8" if name == "fake_int8" else "int4"
    gates = config.gates
    failures: list[str] = []
    minimum_top1 = getattr(gates, f"{prefix}_min_top1_agreement")
    minimum_cosine = getattr(gates, f"{prefix}_min_cosine_similarity")
    maximum_kl = getattr(gates, f"{prefix}_max_kl_divergence")
    if fidelity["top1_agreement"] < minimum_top1:
        failures.append(f"top1_agreement<{minimum_top1}")
    if fidelity["cosine_similarity"] < minimum_cosine:
        failures.append(f"cosine_similarity<{minimum_cosine}")
    if fidelity["kl_divergence_reference_to_candidate"] > maximum_kl:
        failures.append(f"kl_divergence>{maximum_kl}")
    if latency_ratio > gates.max_latency_ratio:
        failures.append(f"latency_ratio>{gates.max_latency_ratio}")
    return not failures, failures


def run_tiny_model_probe(config: ProbeConfig) -> dict[str, Any]:
    """Measure real CPU logits and latency without claiming packed INT execution."""

    if not config.enabled:
        return {
            "evidence_class": "not_run",
            "enabled": False,
            "overall_gate_pass": None,
            "claim_boundary": "CPU probe disabled by strict configuration.",
        }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.num_threads)
    try:
        torch.manual_seed(config.seed)
        model_config = TinyDecoderConfig(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ffn_hidden_size=config.ffn_hidden_size,
            dropout=0.0,
        )
        reference_model = TinyDecoderLM(model_config).cpu().eval()
        input_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=input_generator,
        )
        fake_int8 = fake_quantized_copy(reference_model, bits=8).cpu().eval()
        fake_int4 = fake_quantized_copy(reference_model, bits=4).cpu().eval()
        variants = {
            "fp32": reference_model,
            "fake_int8": fake_int8,
            "fake_int4": fake_int4,
        }
        with torch.inference_mode():
            logits = {name: model(input_ids).logits.cpu() for name, model in variants.items()}
        latencies = {
            name: _latency_samples(
                model,
                input_ids,
                warmup=config.warmup,
                iterations=config.iterations,
            )
            for name, model in variants.items()
        }
        quant_reports = {
            "fake_int8": benchmark_weight_quantization(reference_model, bits=8).to_dict(),
            "fake_int4": benchmark_weight_quantization(reference_model, bits=4).to_dict(),
        }
        reference_latency = float(latencies["fp32"]["p50_ms"])
        results: dict[str, Any] = {
            "fp32": {
                "runtime": "torch_cpu_fp32",
                "latency": latencies["fp32"],
                "fidelity_vs_fp32": {
                    "top1_agreement": 1.0,
                    "cosine_similarity": 1.0,
                    "kl_divergence_reference_to_candidate": 0.0,
                },
                "latency_ratio_vs_fp32": 1.0,
                "gate_pass": True,
                "gate_failures": [],
            }
        }
        for name in ("fake_int8", "fake_int4"):
            fidelity = _fidelity(logits["fp32"], logits[name])
            latency_ratio = float(latencies[name]["p50_ms"]) / reference_latency
            gate_pass, failures = _gate_variant(
                name,
                fidelity,
                latency_ratio=latency_ratio,
                config=config,
            )
            results[name] = {
                "runtime": "torch_cpu_fp32_with_dequantized_fake_quantized_weights",
                "weight_quantization": quant_reports[name],
                "latency": latencies[name],
                "fidelity_vs_fp32": fidelity,
                "latency_ratio_vs_fp32": latency_ratio,
                "gate_pass": gate_pass,
                "gate_failures": failures,
            }
        parameter_count = sum(parameter.numel() for parameter in reference_model.parameters())
        return {
            "evidence_class": "actual_cpu_tiny_model_measurement",
            "enabled": True,
            "claim_boundary": (
                "INT8/INT4 are weight quantize-dequantize simulations executed by FP32 "
                "PyTorch kernels. Fidelity and CPU latency are measured; packed-integer "
                "kernel speedup is not measured or claimed."
            ),
            "hardware": {
                "device": "cpu",
                "processor": platform.processor() or "unknown",
                "torch_version": torch.__version__,
                "num_threads": config.num_threads,
            },
            "model": {
                "class": "TinyDecoderLM",
                "parameter_count": parameter_count,
                "config": {
                    "vocab_size": config.vocab_size,
                    "max_seq_len": config.max_seq_len,
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "n_heads": config.n_heads,
                    "ffn_hidden_size": config.ffn_hidden_size,
                },
            },
            "input": {
                "seed": config.seed + 1,
                "batch_size": config.batch_size,
                "sequence_length": config.sequence_length,
                "input_token_count": input_ids.numel(),
            },
            "gate_thresholds": {
                "int8_min_top1_agreement": config.gates.int8_min_top1_agreement,
                "int8_min_cosine_similarity": config.gates.int8_min_cosine_similarity,
                "int8_max_kl_divergence": config.gates.int8_max_kl_divergence,
                "int4_min_top1_agreement": config.gates.int4_min_top1_agreement,
                "int4_min_cosine_similarity": config.gates.int4_min_cosine_similarity,
                "int4_max_kl_divergence": config.gates.int4_max_kl_divergence,
                "max_latency_ratio": config.gates.max_latency_ratio,
            },
            "variants": results,
            "overall_gate_pass": all(
                bool(results[name]["gate_pass"]) for name in ("fake_int8", "fake_int4")
            ),
        }
    finally:
        torch.set_num_threads(previous_threads)
