from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmlab.systems.capacity import run_capacity_sweep
from fmlab.systems.config import (
    EngineConfig,
    FaultConfig,
    InferenceDynamicsConfig,
    PolicyConfig,
    ProbeConfig,
    ProbeGates,
    TraceConfig,
)
from fmlab.systems.probe import run_tiny_model_probe
from fmlab.systems.runner import run_inference_dynamics
from fmlab.systems.simulator import (
    PagedKVAllocator,
    RequestSpec,
    generate_request_trace,
    simulate_serving,
)


def _engine(**overrides: object) -> EngineConfig:
    values = {
        "max_batch_size": 2,
        "queue_capacity": 8,
        "static_batch_wait_ms": 0.0,
        "prefill_base_ms": 0.1,
        "prefill_token_ms": 0.0,
        "decode_base_ms": 1.0,
        "decode_per_sequence_ms": 0.5,
        "slo_e2e_ms": 100.0,
        "max_simulation_ms": 1_000.0,
        "kv_block_size_tokens": 4,
        "kv_total_blocks": 32,
        "max_preemptions_per_request": 1,
    }
    values.update(overrides)
    return EngineConfig(**values)


def _no_fault() -> FaultConfig:
    return FaultConfig(enabled=False, start_ms=0.0, duration_ms=0.0, service_multiplier=1.0)


def _mapping(artifact_dir: Path) -> dict[str, object]:
    return {
        "experiment": "inference_dynamics",
        "artifact_dir": str(artifact_dir),
        "seed": 7,
        "trace": {
            "pattern": "mixed",
            "request_count": 12,
            "poisson_rate_rps": 30.0,
            "prompt_tokens": [4, 8],
            "output_tokens": [2, 4],
            "burst_every": 6,
            "burst_size": 3,
            "burst_gap_ms": 0.1,
            "timeout_ms": 200.0,
            "cancellation_fraction": 0.0,
            "cancellation_after_ms": [10.0, 20.0],
        },
        "engine": {
            "max_batch_size": 3,
            "queue_capacity": 5,
            "static_batch_wait_ms": 1.0,
            "prefill_base_ms": 0.1,
            "prefill_token_ms": 0.01,
            "decode_base_ms": 0.3,
            "decode_per_sequence_ms": 0.05,
            "slo_e2e_ms": 100.0,
            "max_simulation_ms": 1000.0,
            "kv_block_size_tokens": 4,
            "kv_total_blocks": 20,
            "max_preemptions_per_request": 1,
        },
        "fault": {
            "enabled": True,
            "start_ms": 20.0,
            "duration_ms": 20.0,
            "service_multiplier": 2.0,
        },
        "policies": [
            {"name": "static", "kind": "static_fcfs", "preemption_policy": "none"},
            {
                "name": "continuous",
                "kind": "continuous_fcfs",
                "preemption_policy": "preempt_largest",
            },
        ],
        "capacity_sweep": {
            "enabled": True,
            "poisson_rates_rps": [8.0, 40.0],
            "request_count": 12,
            "seeds": [101, 202, 303],
            "minimum_completion_ratio": 0.80,
            "minimum_slo_attainment": 0.80,
            "maximum_terminal_failure_ratio": 0.20,
        },
        "probe": {
            "enabled": True,
            "seed": 11,
            "num_threads": 1,
            "batch_size": 1,
            "sequence_length": 8,
            "warmup": 0,
            "iterations": 1,
            "vocab_size": 32,
            "max_seq_len": 8,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_hidden_size": 16,
            "gates": {
                "int8_min_top1_agreement": 0.0,
                "int8_min_cosine_similarity": 0.0,
                "int8_max_kl_divergence": 10.0,
                "int4_min_top1_agreement": 0.0,
                "int4_min_cosine_similarity": 0.0,
                "int4_max_kl_divergence": 10.0,
                "max_latency_ratio": 100.0,
            },
        },
    }


def test_mixed_trace_is_deterministic_and_contains_burst_gaps() -> None:
    config = TraceConfig(
        pattern="mixed",
        request_count=20,
        poisson_rate_rps=10.0,
        burst_every=8,
        burst_size=4,
        burst_gap_ms=0.25,
        cancellation_fraction=0.2,
    )
    first = generate_request_trace(config, seed=42)
    second = generate_request_trace(config, seed=42)
    assert first == second
    gaps = [later.arrival_ms - earlier.arrival_ms for earlier, later in zip(first, first[1:])]
    assert sum(abs(gap - 0.25) < 1e-7 for gap in gaps) >= 6
    assert all(request.prompt_tokens > 0 and request.output_tokens > 0 for request in first)


def test_paged_kv_allocator_tracks_fragmentation_oom_and_release() -> None:
    allocator = PagedKVAllocator(total_blocks=2, block_size_tokens=4)
    assert allocator.reserve("a", 5)
    assert allocator.used_blocks == 2
    assert allocator.internal_fragmentation_tokens == 3
    assert allocator.internal_fragmentation_ratio == pytest.approx(3 / 8)
    assert not allocator.reserve("b", 1)
    assert allocator.allocation_failures == 1
    assert allocator.release("a") == 2
    assert allocator.used_blocks == 0
    assert allocator.snapshot()["external_fragmentation_ratio"] == 0.0


def test_continuous_batching_avoids_static_padding_for_variable_outputs() -> None:
    requests = [
        RequestSpec("short", 0.0, 4, 1, 200.0),
        RequestSpec("long", 0.0, 4, 5, 200.0),
    ]
    static = simulate_serving(
        requests,
        engine=_engine(),
        fault=_no_fault(),
        policy=PolicyConfig("static", "static_fcfs", "none"),
    )
    continuous = simulate_serving(
        requests,
        engine=_engine(),
        fault=_no_fault(),
        policy=PolicyConfig("continuous", "continuous_fcfs", "none"),
    )
    assert static.metrics["static_wasted_decode_token_slots"] == 4
    assert continuous.metrics["static_wasted_decode_token_slots"] == 0
    assert continuous.metrics["makespan_ms"] < static.metrics["makespan_ms"]
    assert static.metrics["latency_ms"]["e2e"]["p99"] is not None


def test_bounded_queue_deadline_and_cancellation_are_accounted() -> None:
    requests = [
        RequestSpec("cancel", 0.0, 4, 20, 100.0, cancel_at_ms=3.0),
        RequestSpec("timeout", 0.1, 4, 20, 4.0),
        RequestSpec("rejected", 0.2, 4, 2, 100.0),
    ]
    output = simulate_serving(
        requests,
        engine=_engine(
            max_batch_size=1,
            queue_capacity=1,
            decode_base_ms=3.0,
            decode_per_sequence_ms=0.0,
        ),
        fault=_no_fault(),
        policy=PolicyConfig("continuous", "continuous_fcfs", "none"),
    )
    counts = output.metrics["status_counts"]
    assert counts["cancelled"] == 1
    assert counts["timed_out"] == 1
    assert counts["rejected"] == 1
    assert output.metrics["rejection_rate"] == pytest.approx(1 / 3)
    assert any(item["cause"] == "queue_overload_backpressure" for item in output.root_causes)


def test_kv_preemption_is_explicit_in_trace() -> None:
    requests = [
        RequestSpec("a", 0.0, 4, 4, 100.0),
        RequestSpec("b", 0.0, 4, 4, 100.0),
    ]
    output = simulate_serving(
        requests,
        engine=_engine(
            max_batch_size=2,
            kv_total_blocks=3,
            queue_capacity=4,
            decode_base_ms=1.0,
            decode_per_sequence_ms=0.0,
        ),
        fault=_no_fault(),
        policy=PolicyConfig("preempt", "continuous_fcfs", "preempt_largest"),
    )
    assert any(event["event"] == "kv_preempted" for event in output.events)
    assert output.metrics["kv"]["allocation_failures"] >= 1
    assert any(item["cause"] == "kv_capacity_pressure" for item in output.root_causes)


def test_fault_slowdown_has_attributable_extra_service_time() -> None:
    output = simulate_serving(
        [RequestSpec("a", 0.0, 4, 4, 100.0)],
        engine=_engine(),
        fault=FaultConfig(enabled=True, start_ms=0.0, duration_ms=50.0, service_multiplier=3.0),
        policy=PolicyConfig("continuous", "continuous_fcfs", "none"),
    )
    assert output.metrics["fault"]["impacted_operations"] > 0
    assert output.metrics["fault"]["extra_service_ms"] > 0
    assert any(item["cause"] == "injected_service_slowdown" for item in output.root_causes)


def test_strict_config_rejects_unknown_keys_and_requires_both_policies(tmp_path: Path) -> None:
    mapping = _mapping(tmp_path)
    mapping["surprise"] = True
    with pytest.raises(ValueError, match="unknown config keys"):
        InferenceDynamicsConfig.from_mapping(mapping)
    mapping = _mapping(tmp_path)
    mapping["engine"]["mystery"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown engine keys"):
        InferenceDynamicsConfig.from_mapping(mapping)
    mapping = _mapping(tmp_path)
    mapping["policies"] = [mapping["policies"][0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="must include"):
        InferenceDynamicsConfig.from_mapping(mapping)


def test_capacity_config_is_strict_sorted_and_requires_three_seeds(tmp_path: Path) -> None:
    mapping = _mapping(tmp_path)
    mapping["capacity_sweep"]["mystery"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown capacity_sweep keys"):
        InferenceDynamicsConfig.from_mapping(mapping)

    mapping = _mapping(tmp_path)
    mapping["capacity_sweep"]["seeds"] = [1, 2]  # type: ignore[index]
    with pytest.raises(ValueError, match="at least three seeds"):
        InferenceDynamicsConfig.from_mapping(mapping)

    mapping = _mapping(tmp_path)
    mapping["capacity_sweep"]["poisson_rates_rps"] = [8.0, 8.0]  # type: ignore[index]
    with pytest.raises(ValueError, match="strictly increasing"):
        InferenceDynamicsConfig.from_mapping(mapping)


def test_capacity_sweep_preserves_seed_rows_pairing_and_uncertainty(tmp_path: Path) -> None:
    mapping = _mapping(tmp_path)
    mapping["trace"]["cancellation_fraction"] = 0.25  # type: ignore[index]
    config = InferenceDynamicsConfig.from_mapping(mapping)
    study = run_capacity_sweep(config)
    assert study["enabled"] is True
    assert study["row_count"] == 2 * 3 * 2
    assert study["paired_trace_count"] == 2 * 3
    assert study["design"]["fault_injection"].startswith("disabled")
    for rate in config.capacity_sweep.poisson_rates_rps:
        for seed in config.capacity_sweep.seeds:
            paired = [
                row
                for row in study["rows"]
                if row["configured_poisson_rate_rps"] == rate and row["seed"] == seed
            ]
            assert len(paired) == len(config.policies)
            assert len({row["trace_sha256"] for row in paired}) == 1
            assert len({row["request_semantics_sha256"] for row in paired}) == 1
    for seed in config.capacity_sweep.seeds:
        assert (
            len({row["request_semantics_sha256"] for row in study["rows"] if row["seed"] == seed})
            == 1
        )
    assert len(study["aggregates"]) == 2 * 2
    for aggregate in study["aggregates"]:
        summary = aggregate["metrics"]["completion_ratio"]
        assert summary["n"] == 3
        assert summary["min"] <= summary["mean"] <= summary["max"]
        assert summary["population_stddev"] >= 0
    assert set(study["knees"]) == {policy.name for policy in config.policies}
    assert all(
        knee["rule_id"] == "prefix_first_mean_threshold_violation_v1"
        for knee in study["knees"].values()
    )
    assert "not measured vLLM" in study["claim_boundary"]


def test_actual_cpu_probe_labels_fake_int_runtime_and_checks_fidelity() -> None:
    config = ProbeConfig(
        enabled=True,
        seed=1,
        num_threads=1,
        batch_size=1,
        sequence_length=8,
        warmup=0,
        iterations=1,
        vocab_size=32,
        max_seq_len=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffn_hidden_size=16,
        gates=ProbeGates(
            int8_min_top1_agreement=0.0,
            int8_min_cosine_similarity=0.0,
            int8_max_kl_divergence=10.0,
            int4_min_top1_agreement=0.0,
            int4_min_cosine_similarity=0.0,
            int4_max_kl_divergence=10.0,
            max_latency_ratio=100.0,
        ),
    )
    result = run_tiny_model_probe(config)
    assert result["evidence_class"] == "actual_cpu_tiny_model_measurement"
    assert result["overall_gate_pass"] is True
    assert "FP32 PyTorch kernels" in result["claim_boundary"]
    assert result["variants"]["fake_int8"]["fidelity_vs_fp32"]["top1_agreement"] <= 1
    assert result["variants"]["fake_int4"]["latency"]["p50_ms"] > 0


def test_end_to_end_run_writes_machine_readable_and_visual_evidence(tmp_path: Path) -> None:
    config = InferenceDynamicsConfig.from_mapping(_mapping(tmp_path))
    result_path = run_inference_dynamics(config)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["claim_level"] == "controlled_systems_study"
    assert (
        result["evidence_classes"]["capacity_sweep"]
        == "deterministic_discrete_event_capacity_study"
    )
    assert result["evidence_classes"]["tiny_model_probe"] == "actual_cpu_tiny_model_measurement"
    assert set(result["claim_boundary"]) == {"evidenced", "not_evidenced"}
    assert "vLLM" in " ".join(result["claim_boundary"]["not_evidenced"])
    assert result["metrics"]["actual_cpu_probe"]["overall_gate_pass"] is True
    assert set(result["metrics"]["simulation"]) == {"static", "continuous"}
    assert result["metrics"]["capacity_sweep"]["row_count"] == 12
    assert result["metrics"]["capacity_sweep"]["design"]["paired_policy_trials"] is True
    for name in (
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
    ):
        assert (tmp_path / name).is_file()
    assert "deterministic_discrete_event_simulation" in (tmp_path / "simulation.json").read_text()
    assert "Claim boundary" in (tmp_path / "report.html").read_text()
    assert "Sustainable-load capacity study" in (tmp_path / "report.html").read_text()
    assert "not measured vLLM" in (tmp_path / "capacity_sweep.json").read_text()


def test_systems_scope_has_no_patch_backups() -> None:
    root = Path(__file__).resolve().parents[1]
    scoped_roots = [root / "src" / "fmlab" / "systems", root / "configs" / "systems"]
    dirty = sorted(
        str(path.relative_to(root))
        for scoped_root in scoped_roots
        for path in scoped_root.rglob("*")
        if path.is_file() and path.suffix in {".orig", ".rej"}
    )
    scoped_files = [
        root / "scripts" / "run_inference_dynamics.py",
        root / "tests" / "test_inference_dynamics.py",
        root / "docs" / "inference_dynamics.md",
    ]
    dirty.extend(
        str(candidate.relative_to(root))
        for path in scoped_files
        for suffix in (".orig", ".rej")
        if (candidate := Path(f"{path}{suffix}")).is_file()
    )
    assert dirty == []
