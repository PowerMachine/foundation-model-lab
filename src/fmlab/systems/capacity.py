"""Paired, seeded capacity sweeps for the serving-system simulator."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, replace
from typing import Any

from .config import FaultConfig, InferenceDynamicsConfig
from .simulator import RequestSpec, generate_request_trace, simulate_serving


EVIDENCE_CLASS = "deterministic_discrete_event_capacity_study"
CLAIM_BOUNDARY = (
    "Capacity rows and knees are simulator results under configured service-time and KV "
    "assumptions; they are not measured vLLM, GPU, or production serving capacity."
)
KNEE_RULE_ID = "prefix_first_mean_threshold_violation_v1"


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _summary(values: list[float | int | None]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values if value is not None]
    if not observed:
        return {
            "n": 0,
            "missing": len(values),
            "mean": None,
            "min": None,
            "max": None,
            "population_stddev": None,
        }
    return {
        "n": len(observed),
        "missing": len(values) - len(observed),
        "mean": statistics.fmean(observed),
        "min": min(observed),
        "max": max(observed),
        "population_stddev": statistics.pstdev(observed),
    }


def _realized_offered_load_rps(requests: list[RequestSpec]) -> tuple[float, float]:
    if len(requests) < 2:
        return 0.0, 0.0
    observation_ms = requests[-1].arrival_ms - requests[0].arrival_ms
    if observation_ms <= 0:
        return 0.0, observation_ms
    return (len(requests) - 1) / observation_ms * 1_000.0, observation_ms


def _request_semantics(requests: list[RequestSpec]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": request.request_id,
            "prompt_tokens": request.prompt_tokens,
            "output_tokens": request.output_tokens,
            "timeout_ms": request.timeout_ms,
            "cancel_after_ms": (
                None
                if request.cancel_at_ms is None
                else round(request.cancel_at_ms - request.arrival_ms, 6)
            ),
        }
        for request in requests
    ]


def _capacity_row(
    *,
    configured_rate_rps: float,
    seed: int,
    requests: list[RequestSpec],
    offered_load_rps: float,
    observation_ms: float,
    output: Any,
) -> dict[str, Any]:
    metrics = output.metrics
    counts = metrics["status_counts"]
    request_count = int(metrics["request_count"])
    terminal_failures = sum(counts[name] for name in ("rejected", "timed_out", "oom", "aborted"))
    makespan_ms = float(metrics["makespan_ms"])
    return {
        "configured_poisson_rate_rps": configured_rate_rps,
        "seed": seed,
        "policy": output.policy["name"],
        "policy_kind": output.policy["kind"],
        "preemption_policy": output.policy["preemption_policy"],
        "request_count": request_count,
        "trace_sha256": _fingerprint([request.to_dict() for request in requests]),
        "request_semantics_sha256": _fingerprint(_request_semantics(requests)),
        "arrival_observation_ms": observation_ms,
        "offered_load_requests_per_second": offered_load_rps,
        "completed_load_requests_per_second": counts["completed"] / makespan_ms * 1_000.0,
        "e2e_p99_ms": metrics["latency_ms"]["e2e"]["p99"],
        "slo_attainment": metrics["slo_attainment"],
        "completion_ratio": metrics["completion_rate"],
        "slo_goodput_tokens_per_second": metrics["slo_goodput_tokens_per_second"],
        "status_counts": dict(counts),
        "rejection_rate": counts["rejected"] / request_count,
        "timeout_rate": counts["timed_out"] / request_count,
        "oom_rate": counts["oom"] / request_count,
        "terminal_failure_ratio": terminal_failures / request_count,
        "kv_peak_utilization": metrics["kv"]["peak_utilization"],
        "kv_allocation_failures": metrics["kv"]["allocation_failures"],
        "kv_allocation_failures_per_request": (
            metrics["kv"]["allocation_failures"] / request_count
        ),
    }


def _aggregate_rows(
    config: InferenceDynamicsConfig, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sweep = config.capacity_sweep
    metric_names = (
        "offered_load_requests_per_second",
        "completed_load_requests_per_second",
        "e2e_p99_ms",
        "slo_attainment",
        "completion_ratio",
        "slo_goodput_tokens_per_second",
        "rejection_rate",
        "timeout_rate",
        "oom_rate",
        "terminal_failure_ratio",
        "kv_peak_utilization",
        "kv_allocation_failures_per_request",
    )
    aggregates: list[dict[str, Any]] = []
    for policy in config.policies:
        for configured_rate in sweep.poisson_rates_rps:
            trials = [
                row
                for row in rows
                if row["policy"] == policy.name
                and row["configured_poisson_rate_rps"] == configured_rate
            ]
            metrics = {name: _summary([row[name] for row in trials]) for name in metric_names}
            totals = {
                status: sum(int(row["status_counts"][status]) for row in trials)
                for status in ("completed", "rejected", "timed_out", "cancelled", "oom", "aborted")
            }
            completion_mean = float(metrics["completion_ratio"]["mean"] or 0.0)
            attainment_mean = float(metrics["slo_attainment"]["mean"] or 0.0)
            failure_mean = float(metrics["terminal_failure_ratio"]["mean"] or 0.0)
            failed_criteria: list[str] = []
            if completion_mean < sweep.minimum_completion_ratio:
                failed_criteria.append("completion_ratio")
            if attainment_mean < sweep.minimum_slo_attainment:
                failed_criteria.append("slo_attainment")
            if failure_mean > sweep.maximum_terminal_failure_ratio:
                failed_criteria.append("terminal_failure_ratio")
            aggregates.append(
                {
                    "configured_poisson_rate_rps": configured_rate,
                    "policy": policy.name,
                    "policy_kind": policy.kind,
                    "preemption_policy": policy.preemption_policy,
                    "trial_count": len(trials),
                    "seeds": [int(row["seed"]) for row in trials],
                    "metrics": metrics,
                    "status_counts_total": totals,
                    "sustainable": not failed_criteria,
                    "failed_criteria": failed_criteria,
                }
            )
    return aggregates


def find_sustainable_knees(
    config: InferenceDynamicsConfig, aggregates: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Find the first prefix-breaking rate for each policy using mean thresholds."""

    knees: dict[str, dict[str, Any]] = {}
    for policy in config.policies:
        policy_rows = sorted(
            (row for row in aggregates if row["policy"] == policy.name),
            key=lambda row: row["configured_poisson_rate_rps"],
        )
        last_sustainable = None
        first_unsustainable = None
        for row in policy_rows:
            if not row["sustainable"]:
                first_unsustainable = row
                break
            last_sustainable = row
        if first_unsustainable is None:
            classification = "right_censored_above_test_range"
        elif last_sustainable is None:
            classification = "below_test_range"
        else:
            classification = "observed_interval"
        knees[policy.name] = {
            "rule_id": KNEE_RULE_ID,
            "classification": classification,
            "last_sustainable_configured_rate_rps": (
                None
                if last_sustainable is None
                else last_sustainable["configured_poisson_rate_rps"]
            ),
            "first_unsustainable_configured_rate_rps": (
                None
                if first_unsustainable is None
                else first_unsustainable["configured_poisson_rate_rps"]
            ),
            "last_sustainable_realized_offered_load_rps_mean": (
                None
                if last_sustainable is None
                else last_sustainable["metrics"]["offered_load_requests_per_second"]["mean"]
            ),
            "first_unsustainable_realized_offered_load_rps_mean": (
                None
                if first_unsustainable is None
                else first_unsustainable["metrics"]["offered_load_requests_per_second"]["mean"]
            ),
            "first_failed_criteria": (
                [] if first_unsustainable is None else first_unsustainable["failed_criteria"]
            ),
        }
    return knees


def run_capacity_sweep(config: InferenceDynamicsConfig) -> dict[str, Any]:
    """Run paired policies over seeded arrival-rate trials and aggregate uncertainty."""

    sweep = config.capacity_sweep
    knee_rule = {
        "id": KNEE_RULE_ID,
        "description": (
            "Rates are sorted ascending. The sustainable prefix ends at the first rate whose "
            "across-seed mean violates any configured threshold; later stochastic recoveries "
            "cannot move the knee upward."
        ),
        "criteria": {
            "completion_ratio_mean_gte": sweep.minimum_completion_ratio,
            "slo_attainment_mean_gte": sweep.minimum_slo_attainment,
            "terminal_failure_ratio_mean_lte": sweep.maximum_terminal_failure_ratio,
        },
    }
    base = {
        "enabled": sweep.enabled,
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "design": {
            "paired_policy_trials": True,
            "request_semantics": "inherited from trace config; only request_count and Poisson rate change",
            "trace_pattern": config.trace.pattern,
            "fault_injection": "disabled to isolate nominal modeled capacity",
            "configured_poisson_rates_rps": list(sweep.poisson_rates_rps),
            "request_count_per_trial": sweep.request_count,
            "seeds": list(sweep.seeds),
            "uncertainty": "mean, min, max, and population standard deviation across fixed seeds",
            "knee_rule": knee_rule,
        },
        "rows": [],
        "aggregates": [],
        "knees": {},
    }
    if not sweep.enabled:
        return base

    rows: list[dict[str, Any]] = []
    nominal_fault = FaultConfig(
        enabled=False,
        start_ms=0.0,
        duration_ms=0.0,
        service_multiplier=1.0,
    )
    for configured_rate in sweep.poisson_rates_rps:
        trace_config = replace(
            config.trace,
            poisson_rate_rps=configured_rate,
            request_count=sweep.request_count,
        )
        for seed in sweep.seeds:
            requests = generate_request_trace(trace_config, seed=seed)
            offered_load, observation_ms = _realized_offered_load_rps(requests)
            required_horizon_ms = observation_ms + trace_config.timeout_ms + 1.0
            engine = replace(
                config.engine,
                max_simulation_ms=max(config.engine.max_simulation_ms, required_horizon_ms),
            )
            for policy in config.policies:
                output = simulate_serving(
                    requests,
                    engine=engine,
                    fault=nominal_fault,
                    policy=policy,
                )
                rows.append(
                    _capacity_row(
                        configured_rate_rps=configured_rate,
                        seed=seed,
                        requests=requests,
                        offered_load_rps=offered_load,
                        observation_ms=observation_ms,
                        output=output,
                    )
                )
    aggregates = _aggregate_rows(config, rows)
    base["rows"] = rows
    base["aggregates"] = aggregates
    base["knees"] = find_sustainable_knees(config, aggregates)
    base["row_count"] = len(rows)
    base["paired_trace_count"] = len(sweep.poisson_rates_rps) * len(sweep.seeds)
    base["config"] = asdict(sweep)
    return base
