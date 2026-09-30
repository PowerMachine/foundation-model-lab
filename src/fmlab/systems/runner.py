"""End-to-end orchestration for the offline Inference Dynamics Lab."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from fmlab.artifacts import ExperimentResult, system_snapshot

from .capacity import CLAIM_BOUNDARY as CAPACITY_CLAIM_BOUNDARY
from .capacity import run_capacity_sweep
from .config import InferenceDynamicsConfig
from .probe import run_tiny_model_probe
from .reporting import (
    write_html_report,
    write_capacity_svg,
    write_json,
    write_jsonl,
    write_kv_svg,
    write_pareto_svg,
    write_slo_svg,
)
from .simulator import SimulationOutput, generate_request_trace, simulate_serving


CLAIM_BOUNDARY = [
    "Scheduler, queue, fault, and paged-KV numbers are deterministic discrete-event simulation outputs; they are not measurements of vLLM, a GPU, or a production service.",
    "TinyDecoder latency and logits are actual CPU measurements on this host and are reported separately from simulator numbers.",
    "The INT8/INT4 variants use dequantized fake-quantized weights with FP32 PyTorch kernels; no packed-integer kernel speedup is claimed.",
    "Theoretical peak and achieved efficiency are values within the configured service-time model, not hardware roofline measurements.",
    CAPACITY_CLAIM_BOUNDARY,
]


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _output_dict(output: SimulationOutput) -> dict[str, Any]:
    return {
        "policy": output.policy,
        "metrics": output.metrics,
        "root_causes": list(output.root_causes),
    }


def _relative_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return (candidate - baseline) / baseline


def compare_policies(outputs: list[SimulationOutput]) -> dict[str, Any]:
    static = next(output for output in outputs if output.policy["kind"] == "static_fcfs")
    continuous = next(output for output in outputs if output.policy["kind"] == "continuous_fcfs")
    static_p99 = static.metrics["latency_ms"]["e2e"]["p99"]
    continuous_p99 = continuous.metrics["latency_ms"]["e2e"]["p99"]
    static_throughput = static.metrics["output_token_throughput_per_second"]
    continuous_throughput = continuous.metrics["output_token_throughput_per_second"]
    static_goodput = static.metrics["slo_goodput_tokens_per_second"]
    continuous_goodput = continuous.metrics["slo_goodput_tokens_per_second"]
    attributions: list[dict[str, Any]] = []
    if static_p99 is not None and continuous_p99 is not None and static_p99 > continuous_p99:
        attributions.append(
            {
                "cause": "static_head_of_line_blocking",
                "evidence": {
                    "static_e2e_p99_ms": static_p99,
                    "continuous_e2e_p99_ms": continuous_p99,
                    "static_wasted_decode_token_slots": static.metrics[
                        "static_wasted_decode_token_slots"
                    ],
                },
            }
        )
    if continuous.metrics["kv"]["peak_utilization"] > static.metrics["kv"]["peak_utilization"]:
        attributions.append(
            {
                "cause": "continuous_concurrency_increases_kv_pressure",
                "evidence": {
                    "static_peak_kv": static.metrics["kv"]["peak_utilization"],
                    "continuous_peak_kv": continuous.metrics["kv"]["peak_utilization"],
                    "continuous_allocation_failures": continuous.metrics["kv"][
                        "allocation_failures"
                    ],
                },
            }
        )
    return {
        "baseline_policy": static.policy["name"],
        "candidate_policy": continuous.policy["name"],
        "continuous_relative_e2e_p99_change": _relative_change(continuous_p99, static_p99),
        "continuous_relative_output_throughput_change": _relative_change(
            continuous_throughput, static_throughput
        ),
        "continuous_relative_slo_goodput_change": _relative_change(
            continuous_goodput, static_goodput
        ),
        "cross_policy_root_cause_attribution": attributions,
    }


def run_inference_dynamics(config: InferenceDynamicsConfig) -> Path:
    """Run simulations and the separate CPU probe, then write portfolio artifacts."""

    started = time.time()
    output_dir = config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = generate_request_trace(config.trace, seed=config.seed)
    outputs = [
        simulate_serving(
            requests,
            engine=config.engine,
            fault=config.fault,
            policy=policy,
        )
        for policy in config.policies
    ]
    comparison = compare_policies(outputs)
    capacity_sweep = run_capacity_sweep(config)
    probe = run_tiny_model_probe(config.probe)

    workload_rows = [request.to_dict() for request in requests]
    simulation_document = {
        "evidence_class": "deterministic_discrete_event_simulation",
        "request_trace_sha256": _fingerprint(workload_rows),
        "config_sha256": _fingerprint(config.to_dict()),
        "policies": [_output_dict(output) for output in outputs],
        "comparison": comparison,
        "claim_boundary": CLAIM_BOUNDARY[0],
    }
    events = [event for output in outputs for event in output.events]
    request_results = [row for output in outputs for row in output.requests]
    kv_samples = [sample for output in outputs for sample in output.kv_samples]

    artifacts = [
        "workload.json",
        "simulation.json",
        "trace.jsonl",
        "requests.jsonl",
        "kv_trace.jsonl",
        "actual_probe.json",
        "capacity_sweep.json",
        "capacity.svg",
        "pareto.svg",
        "slo.svg",
        "kv.svg",
        "report.html",
        "result.json",
    ]
    write_json(output_dir / "workload.json", workload_rows)
    write_json(output_dir / "simulation.json", simulation_document)
    write_jsonl(output_dir / "trace.jsonl", events)
    write_jsonl(output_dir / "requests.jsonl", request_results)
    write_jsonl(output_dir / "kv_trace.jsonl", kv_samples)
    write_json(output_dir / "actual_probe.json", probe)
    write_json(output_dir / "capacity_sweep.json", capacity_sweep)
    write_capacity_svg(output_dir / "capacity.svg", capacity_sweep)
    write_pareto_svg(output_dir / "pareto.svg", outputs)
    write_slo_svg(output_dir / "slo.svg", outputs)
    write_kv_svg(output_dir / "kv.svg", outputs)
    write_html_report(
        output_dir / "report.html",
        outputs=outputs,
        probe=probe,
        comparison=comparison,
        claim_boundary=CLAIM_BOUNDARY,
        capacity_sweep=capacity_sweep,
        artifacts=artifacts,
    )

    probe_pass = probe.get("overall_gate_pass")
    status = "completed" if probe_pass in (True, None) else "completed_with_gate_failure"
    metrics = {
        "evidence_contract": {
            "simulation": "deterministic_discrete_event_simulation",
            "capacity_sweep": capacity_sweep["evidence_class"],
            "tiny_model_probe": probe.get("evidence_class"),
            "fake_quantization_runtime": (
                "dequantized weights executed by FP32 kernels; no INT kernel speedup claim"
            ),
        },
        "simulation": {output.policy["name"]: output.metrics for output in outputs},
        "comparison": comparison,
        "capacity_sweep": capacity_sweep,
        "actual_cpu_probe": {
            "enabled": probe.get("enabled"),
            "overall_gate_pass": probe_pass,
            "variants": {
                name: {
                    "top1_agreement": value["fidelity_vs_fp32"].get("top1_agreement"),
                    "cosine_similarity": value["fidelity_vs_fp32"].get("cosine_similarity"),
                    "kl_divergence": value["fidelity_vs_fp32"].get(
                        "kl_divergence_reference_to_candidate"
                    ),
                    "latency_p50_ms": value["latency"]["p50_ms"],
                    "latency_ratio_vs_fp32": value["latency_ratio_vs_fp32"],
                    "gate_pass": value["gate_pass"],
                }
                for name, value in probe.get("variants", {}).items()
            },
        },
        "request_trace_sha256": simulation_document["request_trace_sha256"],
        "config_sha256": simulation_document["config_sha256"],
        "system": system_snapshot(),
    }
    result = ExperimentResult(
        experiment=config.experiment,
        status=status,
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters=config.to_dict(),
        artifacts=artifacts,
        notes=CLAIM_BOUNDARY,
    )
    result_path = result.write(output_dir)
    result_document = json.loads(result_path.read_text(encoding="utf-8"))
    result_document["claim_level"] = "controlled_systems_study"
    result_document["evidence_classes"] = {
        "scheduler_queue_kv_fault": "deterministic_discrete_event_simulation",
        "capacity_sweep": "deterministic_discrete_event_capacity_study",
        "tiny_model_probe": "actual_cpu_tiny_model_measurement",
        "fake_quantization_runtime": (
            "dequantized_fake_quantized_weights_executed_by_fp32_pytorch_kernels"
        ),
    }
    result_document["claim_boundary"] = {
        "evidenced": [
            "Deterministic request, scheduler, bounded-queue, paged-KV, and fault behavior within the configured discrete-event model.",
            "Paired multi-seed simulated capacity curves and threshold-defined knees under the configured workload and service-time assumptions.",
            "Actual CPU TinyDecoder FP32 versus fake-INT8/fake-INT4 logit fidelity and wall-clock latency for the recorded tiny configuration.",
        ],
        "not_evidenced": [
            "Measured vLLM, GPU, distributed, or production-service latency, throughput, capacity, reliability, or cost.",
            "Packed INT8/INT4 kernel speedup, runtime memory reduction, or hardware-native integer execution.",
            "Foundation-model quality, production traffic representativeness, or causal diagnosis outside the configured simulator.",
        ],
    }
    write_json(result_path, result_document)
    return result_path
