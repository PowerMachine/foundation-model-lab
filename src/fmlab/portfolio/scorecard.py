from __future__ import annotations

import hashlib
import html
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCORECARD_SCHEMA = "fmlab.portfolio-scorecard/v1"
MANIFEST_SCHEMA = "1.0"
SHA256_LENGTH = 64

_BUNDLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "qwen3_vl_lora": (
        "vlm/qwen3-vl-8b-lora-real-2step",
        ("result.json",),
    ),
    "visual_reward": (
        "vlm/visual-reward-environment",
        ("result.json",),
    ),
    "agent_eval": (
        "agent/reliable-agent-benchmark",
        ("result.json",),
    ),
    "ddp_correctness": (
        "distributed/ddp-correctness",
        ("result.json",),
    ),
    "inference_dynamics": (
        "systems/inference-dynamics",
        ("result.json", "capacity_sweep.json", "actual_probe.json"),
    ),
}

_DDP_GATES = {
    "gradient",
    "first_update",
    "final_weight",
    "resume_state",
    "resume_loss",
    "sharding",
    "set_epoch",
    "no_sync",
    "checkpoint_contract",
    "checkpoint_integrity",
    "actual_two_rank_gloo",
}
_SYSTEM_POLICIES = {"static_fcfs", "continuous_fcfs", "continuous_fcfs_preempt"}
_PROBE_VARIANTS = {"fp32", "fake_int8", "fake_int4"}


class ScorecardError(RuntimeError):
    """Canonical evidence is absent, invalid, or inconsistent."""


def _fail(path: str, message: str) -> None:
    raise ScorecardError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "expected a finite number")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer")
    return value


def _field(value: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in value:
        _fail(path, f"missing required field {key!r}")
    return value[key]


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    observed = set(value)
    if observed != expected:
        _fail(path, f"expected keys {sorted(expected)}, observed {sorted(observed)}")


def _expect_equal(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        _fail(path, f"expected {expected!r}, observed {value!r}")


def _expect_close(value: float, expected: float, path: str) -> None:
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        _fail(path, f"expected {expected!r}, observed {value!r}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ScorecardError(f"{label}: cannot read {path.name}: {exc}") from exc
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScorecardError(f"{label}: invalid UTF-8 JSON: {exc}") from exc
    return _mapping(parsed, label), payload


def _safe_bundle_path(evidence_root: Path, relative: str) -> Path:
    root = evidence_root.resolve()
    candidate = evidence_root / relative
    if candidate.is_symlink():
        _fail(relative, "bundle symlinks are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ScorecardError(f"{relative}: bundle is missing: {exc}") from exc
    if not resolved.is_relative_to(root):
        _fail(relative, "bundle escapes the evidence root")
    if not resolved.is_dir():
        _fail(relative, "bundle is not a directory")
    return resolved


def _claim_boundary(manifest: Mapping[str, Any], path: str) -> dict[str, Any]:
    boundary = _mapping(_field(manifest, "claim_boundary", path), f"{path}.claim_boundary")
    evidenced_raw = _field(boundary, "evidenced", f"{path}.claim_boundary")
    if isinstance(evidenced_raw, str):
        evidenced = [_text(evidenced_raw, f"{path}.claim_boundary.evidenced")]
    else:
        evidenced = [
            _text(item, f"{path}.claim_boundary.evidenced[{index}]")
            for index, item in enumerate(
                _sequence(evidenced_raw, f"{path}.claim_boundary.evidenced")
            )
        ]
    if not evidenced:
        _fail(f"{path}.claim_boundary.evidenced", "must contain at least one claim")
    not_evidenced_raw = _sequence(
        _field(boundary, "not_evidenced", f"{path}.claim_boundary"),
        f"{path}.claim_boundary.not_evidenced",
    )
    not_evidenced = [
        _text(item, f"{path}.claim_boundary.not_evidenced[{index}]")
        for index, item in enumerate(not_evidenced_raw)
    ]
    if not not_evidenced:
        _fail(f"{path}.claim_boundary.not_evidenced", "must contain at least one non-claim")
    return {"evidenced": evidenced, "not_evidenced": not_evidenced}


def _load_bundle(
    evidence_root: Path,
    key: str,
    relative: str,
    requested_documents: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    bundle = _safe_bundle_path(evidence_root, relative)
    manifest_path = bundle / "evidence_manifest.json"
    if manifest_path.is_symlink():
        _fail(f"{relative}/evidence_manifest.json", "symlinks are forbidden")
    manifest, manifest_payload = _read_json(manifest_path, f"{relative}/evidence_manifest.json")
    _expect_equal(
        _field(manifest, "schema_version", f"{relative}/evidence_manifest.json"),
        MANIFEST_SCHEMA,
        f"{relative}/evidence_manifest.json.schema_version",
    )
    boundary = _claim_boundary(manifest, f"{relative}/evidence_manifest.json")
    records_raw = _sequence(
        _field(manifest, "files", f"{relative}/evidence_manifest.json"),
        f"{relative}/evidence_manifest.json.files",
    )
    if not records_raw:
        _fail(f"{relative}/evidence_manifest.json.files", "manifest cannot be empty")

    records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records_raw):
        record_path = f"{relative}/evidence_manifest.json.files[{index}]"
        record = _mapping(raw_record, record_path)
        name = _text(_field(record, "path", record_path), f"{record_path}.path")
        if Path(name).name != name or name in {".", ".."}:
            _fail(f"{record_path}.path", "must be one safe basename")
        if name in records:
            _fail(f"{record_path}.path", f"duplicate manifest record {name!r}")
        expected_sha = _text(
            _field(record, "public_sha256", record_path), f"{record_path}.public_sha256"
        )
        if len(expected_sha) != SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            _fail(f"{record_path}.public_sha256", "expected lowercase SHA-256")
        public_path = bundle / name
        if public_path.is_symlink() or not public_path.is_file():
            _fail(f"{relative}/{name}", "missing, non-regular, or symlinked public file")
        payload = public_path.read_bytes()
        observed_sha = _sha256(payload)
        if observed_sha != expected_sha:
            _fail(
                f"{relative}/{name}",
                f"manifest SHA-256 mismatch: expected {expected_sha}, observed {observed_sha}",
            )
        records[name] = {
            "path": f"{relative}/{name}",
            "sha256": observed_sha,
            "bytes": len(payload),
        }

    documents: dict[str, Mapping[str, Any]] = {}
    for name in requested_documents:
        if name not in records:
            _fail(f"{relative}/evidence_manifest.json", f"required document {name!r} not listed")
        document, _ = _read_json(bundle / name, f"{relative}/{name}")
        documents[name] = document

    source = {
        "key": key,
        "bundle": relative,
        "manifest_sha256": _sha256(manifest_payload),
        "verified_file_count": len(records),
        "files": [records[name] for name in sorted(records)],
        "claim_boundary": boundary,
    }
    return documents, source


def _validate_visual_reward(result: Mapping[str, Any]) -> dict[str, Any]:
    path = "visual_reward.result"
    _expect_equal(_field(result, "experiment", path), "visual_knowledge_reward_environment", path)
    _expect_equal(_field(result, "status", path), "completed", f"{path}.status")
    metrics = _mapping(_field(result, "metrics", path), f"{path}.metrics")
    dataset = _mapping(_field(metrics, "dataset", f"{path}.metrics"), f"{path}.metrics.dataset")
    preference = _mapping(
        _field(metrics, "preference_accuracy", f"{path}.metrics"),
        f"{path}.metrics.preference_accuracy",
    )
    bootstrap = _mapping(
        _field(preference, "robust_minus_naive", f"{path}.metrics.preference_accuracy"),
        f"{path}.metrics.preference_accuracy.robust_minus_naive",
    )
    hacking = _mapping(
        _field(metrics, "reward_hacking", f"{path}.metrics"),
        f"{path}.metrics.reward_hacking",
    )

    naive = _number(_field(preference, "naive", path), f"{path}.preference.naive")
    robust = _number(_field(preference, "robust", path), f"{path}.preference.robust")
    delta = _number(_field(bootstrap, "delta", path), f"{path}.preference.delta")
    ci_low = _number(_field(bootstrap, "ci95_low", path), f"{path}.preference.ci95_low")
    ci_high = _number(_field(bootstrap, "ci95_high", path), f"{path}.preference.ci95_high")
    _expect_close(delta, robust - naive, f"{path}.preference.delta_consistency")
    if not ci_low <= delta <= ci_high:
        _fail(f"{path}.preference.bootstrap_ci", "95% interval does not contain the estimate")
    source_overlap = _sequence(_field(dataset, "source_overlap", path), f"{path}.source_overlap")

    return {
        "evidence_mode": "scripted",
        "qualifier": "deterministic synthetic visual tasks and scripted candidates",
        "scale": {
            "train_examples": _integer(_field(dataset, "train_examples", path), path),
            "evaluation_examples": _integer(_field(dataset, "evaluation_examples", path), path),
            "train_sources": _integer(_field(dataset, "train_sources", path), path),
            "evaluation_sources": _integer(_field(dataset, "evaluation_sources", path), path),
            "source_overlap_count": len(source_overlap),
            "preference_pairs": _integer(_field(dataset, "preference_pairs", path), path),
            "eval_candidates": _integer(_field(dataset, "eval_candidates", path), path),
            "reward_hack_challenges": _integer(_field(hacking, "challenge_count", path), path),
        },
        "performance": {
            "naive_preference_accuracy": naive,
            "robust_preference_accuracy": robust,
            "absolute_delta": delta,
            "bootstrap_95_ci": [ci_low, ci_high],
            "bootstrap_samples": _integer(_field(bootstrap, "samples", path), path),
            "comparison_pairs": _integer(_field(bootstrap, "pairs", path), path),
            "naive_false_acceptance_rate": _number(
                _field(hacking, "naive_false_acceptance_rate", path), path
            ),
            "robust_false_acceptance_rate": _number(
                _field(hacking, "robust_false_acceptance_rate", path), path
            ),
            "detector_precision": _number(_field(hacking, "detector_precision", path), path),
            "detector_recall": _number(_field(hacking, "detector_recall", path), path),
        },
    }


def _validate_qwen3_vl_lora(result: Mapping[str, Any]) -> dict[str, Any]:
    path = "qwen3_vl_lora.result"
    _expect_equal(_field(result, "experiment", path), "qwen3_vl_8b_lora_e2e", path)
    _expect_equal(_field(result, "status", path), "completed", f"{path}.status")
    metrics = _mapping(_field(result, "metrics", path), f"{path}.metrics")
    _expect_equal(_field(metrics, "backend", path), "local", f"{path}.metrics.backend")
    _expect_equal(_field(metrics, "simulated", path), False, f"{path}.metrics.simulated")
    completed_steps = _integer(_field(metrics, "completed_steps", path), path)
    requested_steps = _integer(_field(metrics, "requested_steps", path), path)
    if completed_steps != requested_steps:
        _fail(f"{path}.metrics.completed_steps", "bounded smoke run did not complete all steps")
    _expect_equal(_field(metrics, "stopped_early", path), False, f"{path}.metrics.stopped_early")

    parameters = _mapping(_field(metrics, "parameters", path), f"{path}.metrics.parameters")
    dataset = _mapping(_field(metrics, "dataset", path), f"{path}.metrics.dataset")
    if _sequence(_field(dataset, "source_overlap", path), f"{path}.dataset.source_overlap"):
        _fail(f"{path}.metrics.dataset.source_overlap", "grouped holdout has source overlap")
    comparison = _mapping(_field(metrics, "comparison", path), f"{path}.metrics.comparison")
    before = _number(_field(comparison, "before", path), f"{path}.comparison.before")
    after = _number(_field(comparison, "after", path), f"{path}.comparison.after")
    delta = _number(_field(comparison, "delta", path), f"{path}.comparison.delta")
    _expect_close(delta, after - before, f"{path}.comparison.delta_consistency")
    bootstrap = _mapping(
        _field(comparison, "paired_bootstrap", path), f"{path}.comparison.paired_bootstrap"
    )
    _expect_close(
        _number(_field(bootstrap, "delta", path), f"{path}.comparison.bootstrap.delta"),
        delta,
        f"{path}.comparison.bootstrap.delta_consistency",
    )
    runtime = _mapping(_field(metrics, "runtime_seconds", path), f"{path}.metrics.runtime_seconds")
    memory = _mapping(
        _field(metrics, "gpu_memory_training_peak", path),
        f"{path}.metrics.gpu_memory_training_peak",
    )
    backend = _mapping(_field(metrics, "backend_details", path), f"{path}.metrics.backend_details")
    _expect_equal(
        _field(backend, "local_files_only", path), True, f"{path}.backend.local_files_only"
    )
    _expect_equal(_field(backend, "vision_frozen", path), True, f"{path}.backend.vision_frozen")

    return {
        "evidence_mode": "actual",
        "evidence_tier": "actual_gpu_smoke",
        "qualifier": (
            "actual local BF16 load, grouped greedy evaluation, and bounded LoRA optimizer "
            "steps; descriptive metrics are not quality-improvement evidence"
        ),
        "scale": {
            "total_parameters": _integer(_field(parameters, "total_parameters", path), path),
            "trainable_parameters": _integer(
                _field(parameters, "trainable_parameters", path), path
            ),
            "trainable_percent": _number(_field(parameters, "trainable_percent", path), path),
            "optimizer_steps": completed_steps,
            "supervised_tokens": _integer(_field(metrics, "total_supervised_tokens", path), path),
            "budgeted_train_examples": _integer(
                _field(dataset, "budgeted_train_examples", path), path
            ),
            "heldout_grouped_examples": _integer(
                _field(dataset, "budgeted_evaluation_examples", path), path
            ),
            "source_overlap_count": 0,
        },
        "performance": {
            "evidence_tier": "actual_gpu_smoke",
            "quality_improvement_claimed": False,
            "heldout_metric": _text(_field(comparison, "metric", path), path),
            "heldout_exact_match_before": before,
            "heldout_exact_match_after": after,
            "heldout_exact_match_delta": delta,
            "heldout_paired_examples": _integer(_field(comparison, "paired_examples", path), path),
            "heldout_delta_bootstrap_95_ci": [
                _number(_field(bootstrap, "ci95_low", path), path),
                _number(_field(bootstrap, "ci95_high", path), path),
            ],
            "training_seconds": _number(_field(runtime, "training_steps", path), path),
            "total_seconds": _number(_field(runtime, "total", path), path),
            "peak_allocated_gib": _number(
                _field(memory, "aggregate_peak_allocated_gib", path), path
            ),
            "supervised_tokens_per_second": _number(
                _field(metrics, "supervised_tokens_per_second", path), path
            ),
        },
    }


def _policy_metrics(policy: Mapping[str, Any], path: str) -> dict[str, Any]:
    ci = _sequence(_field(policy, "pass_at_1_ci_95", path), f"{path}.pass_at_1_ci_95")
    if len(ci) != 2:
        _fail(f"{path}.pass_at_1_ci_95", "expected [low, high]")
    return {
        "episodes": _integer(_field(policy, "episodes", path), path),
        "pass_at_1": _number(_field(policy, "pass_at_1", path), path),
        "pass_at_1_ci_95": [
            _number(ci[0], f"{path}.pass_at_1_ci_95[0]"),
            _number(ci[1], f"{path}.pass_at_1_ci_95[1]"),
        ],
        "outcome_pass_rate": _number(_field(policy, "outcome_pass_rate", path), path),
        "reward_hack_rate": _number(_field(policy, "reward_hack_rate", path), path),
        "tool_calls": _integer(_field(policy, "tool_calls", path), path),
        "tool_error_rate": _number(_field(policy, "tool_error_rate", path), path),
    }


def _validate_agent(result: Mapping[str, Any]) -> dict[str, Any]:
    path = "agent_eval.result"
    _expect_equal(_field(result, "schema", path), "fmlab.reliable-agent-benchmark/v1", path)
    _expect_equal(_field(result, "status", path), "completed", f"{path}.status")
    _expect_equal(
        _field(result, "claim_level", path),
        "real_local_environment_measurement",
        f"{path}.claim_level",
    )
    claim = _mapping(_field(result, "claim", path), f"{path}.claim")
    _expect_equal(
        _field(claim, "policy_type", f"{path}.claim"),
        "deterministic_scripted_controls",
        f"{path}.claim.policy_type",
    )
    _expect_equal(_field(claim, "model_inference", f"{path}.claim"), False, path)
    metrics = _mapping(_field(result, "metrics", path), f"{path}.metrics")
    by_policy = _mapping(_field(metrics, "by_policy", path), f"{path}.metrics.by_policy")
    _exact_keys(
        by_policy,
        {"oracle_patch_scripted", "shortcut_probe_scripted"},
        f"{path}.metrics.by_policy",
    )
    oracle = _policy_metrics(
        _mapping(by_policy["oracle_patch_scripted"], f"{path}.oracle"), f"{path}.oracle"
    )
    shortcut = _policy_metrics(
        _mapping(by_policy["shortcut_probe_scripted"], f"{path}.shortcut"),
        f"{path}.shortcut",
    )
    episodes = _integer(_field(metrics, "episodes", path), f"{path}.metrics.episodes")
    if episodes != oracle["episodes"] + shortcut["episodes"]:
        _fail(f"{path}.metrics.episodes", "does not equal the sum of policy episodes")
    execution = _mapping(_field(metrics, "execution", path), f"{path}.metrics.execution")
    injected = _integer(_field(execution, "transient_failures_injected", path), path)
    retried = _integer(_field(execution, "episodes_requiring_retry", path), path)
    if retried != injected:
        _fail(f"{path}.metrics.execution", "injected failures were not all retried")

    return {
        "environment_mode": "actual",
        "policy_mode": "scripted",
        "qualifier": "actual local process execution with deterministic scripted controls",
        "scale": {
            "tasks": _integer(_field(metrics, "tasks", path), path),
            "episodes": episodes,
            "policies": _integer(_field(metrics, "policies", path), path),
            "tool_calls": _integer(_field(metrics, "tool_calls", path), path),
            "transient_failures_injected": injected,
            "episodes_requiring_retry": retried,
            "resumed_episodes": _integer(_field(execution, "resumed_episodes", path), path),
        },
        "performance": {
            "oracle_scripted_control": oracle,
            "shortcut_scripted_control": shortcut,
            "pass_at_1_gap": oracle["pass_at_1"] - shortcut["pass_at_1"],
        },
    }


def _validate_ddp(result: Mapping[str, Any]) -> dict[str, Any]:
    path = "ddp_correctness.result"
    _expect_equal(_field(result, "experiment", path), "cpu_ddp_correctness_exact_resume", path)
    _expect_equal(_field(result, "status", path), "completed", f"{path}.status")
    _expect_equal(_field(result, "claim_level", path), "controlled_correctness", path)
    _expect_equal(_field(result, "simulated", path), False, f"{path}.simulated")
    parameters = _mapping(_field(result, "parameters", path), f"{path}.parameters")
    metrics = _mapping(_field(result, "metrics", path), f"{path}.metrics")
    gates = _mapping(_field(metrics, "gates", path), f"{path}.metrics.gates")
    _exact_keys(gates, _DDP_GATES, f"{path}.metrics.gates")
    gate_values = {
        name: _boolean(value, f"{path}.metrics.gates.{name}") for name, value in gates.items()
    }
    parity = _mapping(_field(metrics, "parity", path), f"{path}.metrics.parity")
    gradient = _mapping(_field(parity, "gradient", path), f"{path}.metrics.parity.gradient")
    final_weight = _mapping(
        _field(parity, "final_weight", path), f"{path}.metrics.parity.final_weight"
    )
    resume = _mapping(_field(metrics, "resume", path), f"{path}.metrics.resume")
    state_error = _mapping(
        _field(resume, "state_error", path), f"{path}.metrics.resume.state_error"
    )
    training = _mapping(_field(metrics, "training", path), f"{path}.metrics.training")

    return {
        "evidence_mode": "actual",
        "qualifier": "actual two-process CPU/Gloo controlled correctness run",
        "scale": {
            "world_size": _integer(_field(parameters, "world_size", path), path),
            "backend": _text(_field(parameters, "backend", path), path),
            "device": _text(_field(parameters, "device", path), path),
            "dataset_size": _integer(_field(parameters, "dataset_size", path), path),
            "optimizer_steps": _integer(_field(parameters, "optimizer_steps", path), path),
            "effective_global_batch": _integer(
                _field(training, "effective_global_batch", path), path
            ),
            "gate_count": len(gate_values),
        },
        "performance": {
            "all_correctness_gates_pass": all(gate_values.values()),
            "passed_gate_count": sum(gate_values.values()),
            "gradient_max_absolute_error": _number(
                _field(gradient, "max_absolute_error", path), path
            ),
            "gradient_max_relative_error": _number(
                _field(gradient, "max_relative_error", path), path
            ),
            "final_weight_max_absolute_error": _number(
                _field(final_weight, "max_absolute_error", path), path
            ),
            "resume_state_max_absolute_error": _number(
                _field(state_error, "max_absolute_error", path), path
            ),
            "resume_loss_max_absolute_error": _number(
                _field(resume, "max_loss_absolute_error", path), path
            ),
            "resume_exact_state_equal": _boolean(_field(resume, "exact_state_equal", path), path),
            "absolute_tolerance": _number(_field(parity, "absolute_tolerance", path), path),
            "relative_tolerance": _number(_field(parity, "relative_tolerance", path), path),
        },
    }


def _aggregate_stat(
    value: Any,
    path: str,
    *,
    expected_samples: int,
) -> dict[str, Any]:
    stats = _mapping(value, path)
    samples = _integer(_field(stats, "n", path), f"{path}.n")
    missing = _integer(_field(stats, "missing", path), f"{path}.missing")
    if samples != expected_samples or missing != 0:
        _fail(
            path,
            f"expected {expected_samples} complete samples, observed n={samples}, missing={missing}",
        )
    mean = _number(_field(stats, "mean", path), f"{path}.mean")
    minimum = _number(_field(stats, "min", path), f"{path}.min")
    maximum = _number(_field(stats, "max", path), f"{path}.max")
    stddev = _number(_field(stats, "population_stddev", path), f"{path}.population_stddev")
    if not minimum <= mean <= maximum:
        _fail(path, "mean must lie within [min, max]")
    if stddev < 0:
        _fail(path, "population standard deviation cannot be negative")
    return {
        "n": samples,
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "population_stddev": stddev,
    }


def _capacity_summary(
    capacity: Mapping[str, Any], result_summary: Mapping[str, Any]
) -> dict[str, Any]:
    path = "inference_dynamics.capacity_sweep"
    _expect_equal(_field(capacity, "enabled", path), True, f"{path}.enabled")
    _expect_equal(
        _field(capacity, "evidence_class", path),
        "deterministic_discrete_event_capacity_study",
        f"{path}.evidence_class",
    )
    rows = _sequence(_field(capacity, "rows", path), f"{path}.rows")
    row_count = _integer(_field(capacity, "row_count", path), f"{path}.row_count")
    if row_count != len(rows):
        _fail(f"{path}.row_count", "does not equal len(rows)")
    paired_trace_count = _integer(
        _field(capacity, "paired_trace_count", path), f"{path}.paired_trace_count"
    )
    config = _mapping(_field(capacity, "config", path), f"{path}.config")
    rates = [
        _number(value, f"{path}.config.poisson_rates_rps")
        for value in _sequence(_field(config, "poisson_rates_rps", path), path)
    ]
    seeds = [
        _integer(value, f"{path}.config.seeds")
        for value in _sequence(_field(config, "seeds", path), path)
    ]
    if paired_trace_count != len(rates) * len(seeds):
        _fail(f"{path}.paired_trace_count", "does not equal rate_count * seed_count")
    if row_count != paired_trace_count * len(_SYSTEM_POLICIES):
        _fail(f"{path}.row_count", "does not equal paired traces * policy count")
    knees = _mapping(_field(capacity, "knees", path), f"{path}.knees")
    _exact_keys(knees, _SYSTEM_POLICIES, f"{path}.knees")
    summary_rows = _mapping(result_summary, f"{path}.result_summary")
    _expect_equal(_field(summary_rows, "row_count", path), row_count, path)
    _expect_equal(_field(summary_rows, "paired_trace_count", path), paired_trace_count, path)
    _expect_equal(_field(summary_rows, "knees", path), knees, f"{path}.knees_consistency")

    summarized_knees: dict[str, Any] = {}
    for policy in sorted(_SYSTEM_POLICIES):
        knee = _mapping(knees[policy], f"{path}.knees.{policy}")
        summarized_knees[policy] = {
            "last_sustainable_configured_rate_rps": _number(
                _field(knee, "last_sustainable_configured_rate_rps", path), path
            ),
            "first_unsustainable_configured_rate_rps": _number(
                _field(knee, "first_unsustainable_configured_rate_rps", path), path
            ),
            "last_sustainable_realized_offered_load_rps_mean": _number(
                _field(knee, "last_sustainable_realized_offered_load_rps_mean", path), path
            ),
            "first_unsustainable_realized_offered_load_rps_mean": _number(
                _field(knee, "first_unsustainable_realized_offered_load_rps_mean", path), path
            ),
            "first_failed_criteria": [
                _text(item, f"{path}.knees.{policy}.first_failed_criteria")
                for item in _sequence(_field(knee, "first_failed_criteria", path), path)
            ],
        }

    aggregates_raw = _sequence(_field(capacity, "aggregates", path), f"{path}.aggregates")
    if len(aggregates_raw) != len(rates) * len(_SYSTEM_POLICIES):
        _fail(f"{path}.aggregates", "expected one aggregate per configured rate and policy")
    continuous_rate = summarized_knees["continuous_fcfs"]["last_sustainable_configured_rate_rps"]
    matching_aggregates: list[Mapping[str, Any]] = []
    for index, raw_aggregate in enumerate(aggregates_raw):
        aggregate_path = f"{path}.aggregates[{index}]"
        aggregate = _mapping(raw_aggregate, aggregate_path)
        policy = _text(_field(aggregate, "policy", aggregate_path), f"{aggregate_path}.policy")
        rate = _number(
            _field(aggregate, "configured_poisson_rate_rps", aggregate_path),
            f"{aggregate_path}.configured_poisson_rate_rps",
        )
        if policy == "continuous_fcfs" and math.isclose(rate, continuous_rate):
            matching_aggregates.append(aggregate)
    if len(matching_aggregates) != 1:
        _fail(
            f"{path}.aggregates",
            "expected exactly one continuous aggregate at the last sustainable knee",
        )
    continuous_aggregate = matching_aggregates[0]
    _expect_equal(
        _field(continuous_aggregate, "trial_count", path),
        len(seeds),
        f"{path}.continuous_knee.trial_count",
    )
    aggregate_seeds = [
        _integer(value, f"{path}.continuous_knee.seeds")
        for value in _sequence(
            _field(continuous_aggregate, "seeds", path), f"{path}.continuous_knee.seeds"
        )
    ]
    _expect_equal(aggregate_seeds, seeds, f"{path}.continuous_knee.seeds")
    aggregate_metrics = _mapping(
        _field(continuous_aggregate, "metrics", path), f"{path}.continuous_knee.metrics"
    )
    continuous_uncertainty = {
        "policy": "continuous_fcfs",
        "configured_rate_rps": continuous_rate,
        "realized_offered_load_requests_per_second": _aggregate_stat(
            _field(aggregate_metrics, "offered_load_requests_per_second", path),
            f"{path}.continuous_knee.offered_load_requests_per_second",
            expected_samples=len(seeds),
        ),
        "e2e_p99_ms": _aggregate_stat(
            _field(aggregate_metrics, "e2e_p99_ms", path),
            f"{path}.continuous_knee.e2e_p99_ms",
            expected_samples=len(seeds),
        ),
        "slo_goodput_tokens_per_second": _aggregate_stat(
            _field(aggregate_metrics, "slo_goodput_tokens_per_second", path),
            f"{path}.continuous_knee.slo_goodput_tokens_per_second",
            expected_samples=len(seeds),
        ),
    }

    request_count = _integer(_field(config, "request_count", path), f"{path}.request_count")
    return {
        "configured_rates_rps": rates,
        "seeds": seeds,
        "policy_count": len(_SYSTEM_POLICIES),
        "row_count": row_count,
        "paired_trace_count": paired_trace_count,
        "request_count_per_trial": request_count,
        "unique_trace_requests": paired_trace_count * request_count,
        "policy_request_executions": row_count * request_count,
        "knees": summarized_knees,
        "last_sustainable_continuous_uncertainty": continuous_uncertainty,
    }


def _validate_probe(probe: Mapping[str, Any], result_summary: Mapping[str, Any]) -> dict[str, Any]:
    path = "inference_dynamics.actual_probe"
    _expect_equal(
        _field(probe, "evidence_class", path),
        "actual_cpu_tiny_model_measurement",
        f"{path}.evidence_class",
    )
    _expect_equal(_field(probe, "enabled", path), True, f"{path}.enabled")
    variants = _mapping(_field(probe, "variants", path), f"{path}.variants")
    _exact_keys(variants, _PROBE_VARIANTS, f"{path}.variants")
    summary_variants = _mapping(
        _field(result_summary, "variants", f"{path}.result_summary"),
        f"{path}.result_summary.variants",
    )
    _exact_keys(summary_variants, _PROBE_VARIANTS, f"{path}.result_summary.variants")
    summarized: dict[str, Any] = {}
    for name in sorted(_PROBE_VARIANTS):
        variant = _mapping(variants[name], f"{path}.variants.{name}")
        fidelity = _mapping(
            _field(variant, "fidelity_vs_fp32", path), f"{path}.variants.{name}.fidelity"
        )
        latency = _mapping(_field(variant, "latency", path), f"{path}.variants.{name}.latency")
        output = {
            "top1_agreement": _number(_field(fidelity, "top1_agreement", path), path),
            "cosine_similarity": _number(_field(fidelity, "cosine_similarity", path), path),
            "kl_divergence": _number(
                _field(fidelity, "kl_divergence_reference_to_candidate", path), path
            ),
            "latency_p50_ms": _number(_field(latency, "p50_ms", path), path),
            "latency_p95_ms": _number(_field(latency, "p95_ms", path), path),
            "latency_ratio_vs_fp32": _number(_field(variant, "latency_ratio_vs_fp32", path), path),
            "gate_pass": _boolean(_field(variant, "gate_pass", path), path),
        }
        summary = _mapping(summary_variants[name], f"{path}.result_summary.{name}")
        for source_key, summary_key in (
            ("top1_agreement", "top1_agreement"),
            ("cosine_similarity", "cosine_similarity"),
            ("kl_divergence", "kl_divergence"),
            ("latency_p50_ms", "latency_p50_ms"),
            ("latency_ratio_vs_fp32", "latency_ratio_vs_fp32"),
        ):
            _expect_close(
                output[source_key],
                _number(_field(summary, summary_key, path), path),
                f"{path}.{name}.{source_key}_summary_consistency",
            )
        _expect_equal(_field(summary, "gate_pass", path), output["gate_pass"], path)
        summarized[name] = output

    model = _mapping(_field(probe, "model", path), f"{path}.model")
    input_spec = _mapping(_field(probe, "input", path), f"{path}.input")
    overall = _boolean(_field(probe, "overall_gate_pass", path), path)
    _expect_equal(_field(result_summary, "overall_gate_pass", path), overall, path)
    return {
        "evidence_mode": "actual",
        "qualifier": _text(_field(probe, "claim_boundary", path), f"{path}.claim_boundary"),
        "parameter_count": _integer(_field(model, "parameter_count", path), path),
        "input_token_count": _integer(_field(input_spec, "input_token_count", path), path),
        "overall_gate_pass": overall,
        "variants": summarized,
    }


def _validate_inference(
    result: Mapping[str, Any],
    capacity: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    path = "inference_dynamics.result"
    _expect_equal(_field(result, "experiment", path), "inference_dynamics", path)
    _expect_equal(_field(result, "status", path), "completed", f"{path}.status")
    _expect_equal(_field(result, "claim_level", path), "controlled_systems_study", path)
    classes = _mapping(_field(result, "evidence_classes", path), f"{path}.evidence_classes")
    _exact_keys(
        classes,
        {
            "scheduler_queue_kv_fault",
            "capacity_sweep",
            "tiny_model_probe",
            "fake_quantization_runtime",
        },
        f"{path}.evidence_classes",
    )
    _expect_equal(
        classes["scheduler_queue_kv_fault"],
        "deterministic_discrete_event_simulation",
        f"{path}.evidence_classes.scheduler_queue_kv_fault",
    )
    metrics = _mapping(_field(result, "metrics", path), f"{path}.metrics")
    simulation = _mapping(_field(metrics, "simulation", path), f"{path}.metrics.simulation")
    _exact_keys(simulation, _SYSTEM_POLICIES, f"{path}.metrics.simulation")
    request_counts = {
        _integer(_field(_mapping(policy, f"{path}.simulation.{name}"), "request_count", path), path)
        for name, policy in simulation.items()
    }
    if len(request_counts) != 1:
        _fail(f"{path}.metrics.simulation", "policies must share one paired request trace")
    comparison = _mapping(_field(metrics, "comparison", path), f"{path}.metrics.comparison")
    _expect_equal(_field(comparison, "baseline_policy", path), "static_fcfs", path)
    _expect_equal(_field(comparison, "candidate_policy", path), "continuous_fcfs", path)
    capacity_summary = _capacity_summary(
        capacity,
        _mapping(_field(metrics, "capacity_sweep", path), f"{path}.metrics.capacity_sweep"),
    )
    probe_summary = _validate_probe(
        probe,
        _mapping(_field(metrics, "actual_cpu_probe", path), f"{path}.metrics.actual_cpu_probe"),
    )

    static = _mapping(simulation["static_fcfs"], f"{path}.simulation.static_fcfs")
    continuous = _mapping(simulation["continuous_fcfs"], f"{path}.simulation.continuous_fcfs")
    static_p99 = _number(
        _field(
            _mapping(_field(_mapping(_field(static, "latency_ms", path), path), "e2e", path), path),
            "p99",
            path,
        ),
        path,
    )
    continuous_p99 = _number(
        _field(
            _mapping(
                _field(_mapping(_field(continuous, "latency_ms", path), path), "e2e", path), path
            ),
            "p99",
            path,
        ),
        path,
    )
    static_throughput = _number(_field(static, "output_token_throughput_per_second", path), path)
    continuous_throughput = _number(
        _field(continuous, "output_token_throughput_per_second", path), path
    )
    static_goodput = _number(_field(static, "slo_goodput_tokens_per_second", path), path)
    continuous_goodput = _number(_field(continuous, "slo_goodput_tokens_per_second", path), path)

    throughput_delta = _number(
        _field(comparison, "continuous_relative_output_throughput_change", path), path
    )
    p99_delta = _number(_field(comparison, "continuous_relative_e2e_p99_change", path), path)
    goodput_delta = _number(
        _field(comparison, "continuous_relative_slo_goodput_change", path), path
    )
    _expect_close(p99_delta, continuous_p99 / static_p99 - 1.0, f"{path}.p99_delta")
    _expect_close(
        throughput_delta,
        continuous_throughput / static_throughput - 1.0,
        f"{path}.throughput_delta",
    )
    _expect_close(
        goodput_delta,
        continuous_goodput / static_goodput - 1.0,
        f"{path}.slo_goodput_delta",
    )

    return {
        "simulation": {
            "evidence_mode": "simulated",
            "qualifier": "deterministic discrete-event scheduler/queue/KV/fault model",
            "paired_request_count": next(iter(request_counts)),
            "policy_count": len(simulation),
            "policy_request_executions": next(iter(request_counts)) * len(simulation),
            "static_e2e_p99_ms": static_p99,
            "continuous_e2e_p99_ms": continuous_p99,
            "static_output_tokens_per_second": static_throughput,
            "continuous_output_tokens_per_second": continuous_throughput,
            "static_slo_goodput_tokens_per_second": static_goodput,
            "continuous_slo_goodput_tokens_per_second": continuous_goodput,
            "continuous_relative_e2e_p99_change": p99_delta,
            "continuous_relative_output_throughput_change": throughput_delta,
            "continuous_relative_slo_goodput_change": goodput_delta,
        },
        "capacity": {
            "evidence_mode": "simulated",
            "qualifier": "paired multi-seed deterministic discrete-event capacity study",
            **capacity_summary,
        },
        "actual_cpu_probe": probe_summary,
    }


def build_portfolio_scorecard(evidence_root: str | Path) -> dict[str, Any]:
    """Build a fail-closed scorecard from canonical public-evidence bundles."""

    root = Path(evidence_root)
    if root.is_symlink() or not root.is_dir():
        raise ScorecardError(
            f"public evidence root is missing, not a directory, or symlinked: {root}"
        )

    documents: dict[str, dict[str, Mapping[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for key, (relative, requested) in _BUNDLES.items():
        bundle_documents, source = _load_bundle(root, key, relative, requested)
        documents[key] = bundle_documents
        sources.append(source)

    qwen3_vl_lora = _validate_qwen3_vl_lora(documents["qwen3_vl_lora"]["result.json"])
    visual = _validate_visual_reward(documents["visual_reward"]["result.json"])
    agent = _validate_agent(documents["agent_eval"]["result.json"])
    ddp = _validate_ddp(documents["ddp_correctness"]["result.json"])
    inference = _validate_inference(
        documents["inference_dynamics"]["result.json"],
        documents["inference_dynamics"]["capacity_sweep.json"],
        documents["inference_dynamics"]["actual_probe.json"],
    )

    sources.sort(key=lambda item: item["key"])
    source_set_payload = json.dumps(
        sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    components = [
        {
            "component": "vlm.qwen3_vl_8b_lora",
            "mode": "actual",
            "qualifier": "actual GPU smoke only; no quality-improvement claim",
        },
        {
            "component": "agent.local_execution_environment",
            "mode": "actual",
            "qualifier": "local process execution; not a security boundary",
        },
        {
            "component": "ddp.cpu_gloo_correctness",
            "mode": "actual",
            "qualifier": "two-process CPU/Gloo controlled run",
        },
        {
            "component": "systems.tiny_cpu_probe",
            "mode": "actual",
            "qualifier": "FP32 kernels; INT8/INT4 weights are quantize-dequantize simulations",
        },
        {
            "component": "agent.policy_controls",
            "mode": "scripted",
            "qualifier": "oracle and adversarial controls; no model inference",
        },
        {
            "component": "vlm.visual_reward_audit",
            "mode": "scripted",
            "qualifier": "synthetic tasks and scripted candidates; no VLM generation",
        },
        {
            "component": "systems.scheduler_queue_kv_fault",
            "mode": "simulated",
            "qualifier": "deterministic discrete-event model",
        },
        {
            "component": "systems.capacity_sweep",
            "mode": "simulated",
            "qualifier": "paired multi-seed discrete-event study",
        },
    ]
    mode_counts = Counter(component["mode"] for component in components)
    claims = {source["key"]: source["claim_boundary"] for source in sources}
    capacity_scale = inference["capacity"]

    return {
        "schema": SCORECARD_SCHEMA,
        "title": "Foundation Model Lab — Evidence Scorecard",
        "source_set": {
            "sha256": _sha256(source_set_payload),
            "integrity": "all public files verified against bundle manifests",
            "bundle_count": len(sources),
            "verified_file_count": sum(source["verified_file_count"] for source in sources),
            "bundles": sources,
        },
        "evidence_modes": {
            "definitions": {
                "actual": "wall-clock or multi-process execution occurred on the recorded local setup",
                "scripted": "deterministic controls evaluate the harness, not model capability",
                "simulated": "results arise from the declared discrete-event model",
            },
            "component_counts": {
                mode: mode_counts[mode] for mode in ("actual", "scripted", "simulated")
            },
            "components": components,
        },
        "experimental_scale": {
            "qwen3_vl_lora_actual_gpu_smoke": qwen3_vl_lora["scale"],
            "visual_reward": visual["scale"],
            "agent_eval": agent["scale"],
            "ddp_correctness": ddp["scale"],
            "inference": {
                "single_trace_policy_request_executions": inference["simulation"][
                    "policy_request_executions"
                ],
                "capacity_rates": len(capacity_scale["configured_rates_rps"]),
                "capacity_seeds": len(capacity_scale["seeds"]),
                "capacity_paired_traces": capacity_scale["paired_trace_count"],
                "capacity_rows": capacity_scale["row_count"],
                "capacity_unique_trace_requests": capacity_scale["unique_trace_requests"],
                "capacity_policy_request_executions": capacity_scale["policy_request_executions"],
                "tiny_probe_parameters": inference["actual_cpu_probe"]["parameter_count"],
                "tiny_probe_input_tokens": inference["actual_cpu_probe"]["input_token_count"],
            },
        },
        "performance": {
            "qwen3_vl_lora_actual_gpu_smoke": qwen3_vl_lora["performance"],
            "visual_reward_scripted": visual["performance"],
            "agent_scripted_controls_on_actual_environment": agent["performance"],
            "ddp_actual_cpu_gloo": ddp["performance"],
            "inference_single_trace_simulated": inference["simulation"],
            "inference_capacity_simulated": inference["capacity"],
            "inference_tiny_cpu_actual": inference["actual_cpu_probe"],
        },
        "uncertainty": {
            "qwen3_vl_lora": {
                "method": "paired bootstrap over three source-grouped synthetic holdout examples",
                "heldout_delta_95_ci": qwen3_vl_lora["performance"][
                    "heldout_delta_bootstrap_95_ci"
                ],
            },
            "visual_reward": {
                "method": "bootstrap over comparison pairs",
                "samples": visual["performance"]["bootstrap_samples"],
                "absolute_delta_95_ci": visual["performance"]["bootstrap_95_ci"],
            },
            "agent": {
                "method": "reported pass@1 interval per deterministic control",
                "oracle_pass_at_1_95_ci": agent["performance"]["oracle_scripted_control"][
                    "pass_at_1_ci_95"
                ],
                "shortcut_pass_at_1_95_ci": agent["performance"]["shortcut_scripted_control"][
                    "pass_at_1_ci_95"
                ],
            },
            "inference_capacity": {
                "method": "mean/min/max/population standard deviation over fixed seeds",
                "seed_count": len(capacity_scale["seeds"]),
                "knee_rule": "first across-seed mean threshold violation after sustainable prefix",
                "last_sustainable_continuous": capacity_scale[
                    "last_sustainable_continuous_uncertainty"
                ],
            },
            "ddp": {
                "method": "deterministic parity against declared numerical tolerances",
                "absolute_tolerance": ddp["performance"]["absolute_tolerance"],
                "relative_tolerance": ddp["performance"]["relative_tolerance"],
            },
        },
        "claim_boundaries": claims,
    }


def _svg_text(x: int, y: int, value: str, css_class: str, *, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" class="{css_class}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def _percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def render_scorecard_svg(scorecard: Mapping[str, Any]) -> str:
    """Render a dependency-free, deterministic SVG from a validated scorecard."""

    if scorecard.get("schema") != SCORECARD_SCHEMA:
        raise ScorecardError("cannot render an unknown scorecard schema")
    performance = _mapping(scorecard.get("performance"), "scorecard.performance")
    scale = _mapping(scorecard.get("experimental_scale"), "scorecard.experimental_scale")
    source_set = _mapping(scorecard.get("source_set"), "scorecard.source_set")
    qwen3_vl_lora = _mapping(
        performance["qwen3_vl_lora_actual_gpu_smoke"], "scorecard.qwen3_vl_lora"
    )
    visual = _mapping(performance["visual_reward_scripted"], "scorecard.visual")
    agent = _mapping(
        performance["agent_scripted_controls_on_actual_environment"], "scorecard.agent"
    )
    ddp = _mapping(performance["ddp_actual_cpu_gloo"], "scorecard.ddp")
    simulated = _mapping(performance["inference_single_trace_simulated"], "scorecard.inference")
    capacity = _mapping(performance["inference_capacity_simulated"], "scorecard.capacity")
    probe = _mapping(performance["inference_tiny_cpu_actual"], "scorecard.probe")
    qwen3_vl_lora_scale = _mapping(
        scale["qwen3_vl_lora_actual_gpu_smoke"], "scorecard.qwen3_vl_lora_scale"
    )
    visual_scale = _mapping(scale["visual_reward"], "scorecard.visual_scale")
    agent_scale = _mapping(scale["agent_eval"], "scorecard.agent_scale")
    ddp_scale = _mapping(scale["ddp_correctness"], "scorecard.ddp_scale")
    continuous_knee = _mapping(
        _mapping(capacity["knees"], "scorecard.capacity.knees")["continuous_fcfs"],
        "scorecard.capacity.knees.continuous_fcfs",
    )
    oracle = _mapping(agent["oracle_scripted_control"], "scorecard.agent.oracle")
    shortcut = _mapping(agent["shortcut_scripted_control"], "scorecard.agent.shortcut")
    fp32 = _mapping(_mapping(probe["variants"], "scorecard.probe.variants")["fp32"], "fp32")

    source_sha = _text(source_set["sha256"], "scorecard.source_set.sha256")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="960" viewBox="0 0 1200 960" role="img" aria-labelledby="title desc">',
        '<title id="title">Foundation Model Lab evidence scorecard</title>',
        '<desc id="desc">Five experiment cards distinguish actual, scripted, and simulated evidence and show scale, deltas, uncertainty, and claim boundaries.</desc>',
        "<style>",
        ".bg{fill:#08111f}.card{fill:#101d31;stroke:#263b59;stroke-width:1}.title{fill:#f6f8fb;font:700 30px system-ui,sans-serif}.subtitle{fill:#9cb0ca;font:400 14px system-ui,sans-serif}.cardTitle{fill:#f6f8fb;font:700 19px system-ui,sans-serif}.metric{fill:#77e1bd;font:700 29px system-ui,sans-serif}.body{fill:#c8d4e5;font:400 14px system-ui,sans-serif}.muted{fill:#8297b3;font:400 12px system-ui,sans-serif}.label{font:700 11px system-ui,sans-serif;letter-spacing:.8px}.actual{fill:#52d4a7}.scripted{fill:#ffc66d}.simulated{fill:#83b8ff}.divider{stroke:#263b59;stroke-width:1}.warn{fill:#ff9d9d;font:400 12px system-ui,sans-serif}",
        "</style>",
        '<rect class="bg" width="1200" height="960" rx="18"/>',
        _svg_text(48, 60, "Foundation Model Lab — Evidence Scorecard", "title"),
        _svg_text(
            48,
            88,
            "Manifest-verified results · evidence modes are explicit · no aggregate vanity score",
            "subtitle",
        ),
        _svg_text(900, 55, "ACTUAL", "label actual"),
        _svg_text(900, 76, "SCRIPTED", "label scripted"),
        _svg_text(900, 97, "SIMULATED", "label simulated"),
    ]

    cards = [
        (48, 128, 540, 276),
        (612, 128, 540, 276),
        (48, 428, 540, 276),
        (612, 428, 540, 276),
        (48, 728, 1104, 120),
    ]
    for x, y, width, height in cards:
        lines.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" class="card"/>'
        )

    # Visual reward card.
    lines.extend(
        [
            _svg_text(76, 164, "Visual Reward Robustness", "cardTitle"),
            _svg_text(558, 164, "SCRIPTED", "label scripted", anchor="end"),
            _svg_text(76, 211, f"+{float(visual['absolute_delta']) * 100:.1f} pp", "metric"),
            _svg_text(76, 238, "robust - naive preference accuracy", "body"),
            _svg_text(
                76,
                270,
                f"{_percent(float(visual['naive_preference_accuracy']))} to {_percent(float(visual['robust_preference_accuracy']))}",
                "body",
            ),
            _svg_text(
                76,
                294,
                f"bootstrap 95% CI  +{float(visual['bootstrap_95_ci'][0]) * 100:.1f} to +{float(visual['bootstrap_95_ci'][1]) * 100:.1f} pp",
                "body",
            ),
            _svg_text(
                76,
                320,
                f"{visual_scale['eval_candidates']} candidates · {visual_scale['reward_hack_challenges']} attacks · 0 source overlap",
                "body",
            ),
            _svg_text(76, 366, "Boundary: synthetic tasks; not VLM quality or online RL", "warn"),
        ]
    )

    # Agent eval card.
    lines.extend(
        [
            _svg_text(640, 164, "Reliable Agent Evaluation", "cardTitle"),
            _svg_text(1122, 164, "ACTUAL ENV · SCRIPTED POLICY", "label scripted", anchor="end"),
            _svg_text(
                640,
                211,
                f"{int(oracle['episodes'])}/{int(oracle['episodes'])} vs 0/{int(shortcut['episodes'])}",
                "metric",
            ),
            _svg_text(640, 238, "oracle vs shortcut pass@1 controls", "body"),
            _svg_text(
                640,
                270,
                f"{agent_scale['tasks']} tasks · {agent_scale['episodes']} episodes · {agent_scale['tool_calls']} tool calls",
                "body",
            ),
            _svg_text(
                640,
                294,
                f"{agent_scale['transient_failures_injected']} injected failures · all retried",
                "body",
            ),
            _svg_text(640, 320, "hidden outcome + process-integrity grading", "body"),
            _svg_text(
                640, 366, "Boundary: no LLM inference; process mode is not a sandbox", "warn"
            ),
        ]
    )

    # Inference card.
    lines.extend(
        [
            _svg_text(76, 464, "Inference Dynamics", "cardTitle"),
            _svg_text(558, 464, "SIMULATED + ACTUAL CPU", "label simulated", anchor="end"),
            _svg_text(
                76,
                511,
                f"+{_percent(float(simulated['continuous_relative_output_throughput_change']))}",
                "metric",
            ),
            _svg_text(76, 538, "modeled output throughput, continuous vs static", "body"),
            _svg_text(
                76,
                570,
                f"p99 {_percent(float(simulated['continuous_relative_e2e_p99_change']))} · SLO goodput +{_percent(float(simulated['continuous_relative_slo_goodput_change']))}",
                "body",
            ),
            _svg_text(
                76,
                594,
                f"modeled knee {continuous_knee['last_sustainable_realized_offered_load_rps_mean']:.1f}–{continuous_knee['first_unsustainable_realized_offered_load_rps_mean']:.1f} offered rps",
                "body",
            ),
            _svg_text(
                76,
                618,
                f"81 rows · 3 seeds · actual TinyDecoder FP32 p50 {float(fp32['latency_p50_ms']):.3f} ms",
                "body",
            ),
            _svg_text(
                76, 666, "Boundary: capacity is modeled; tiny probe uses CPU FP32 kernels", "warn"
            ),
        ]
    )

    # DDP card.
    lines.extend(
        [
            _svg_text(640, 464, "Distributed Correctness", "cardTitle"),
            _svg_text(1122, 464, "ACTUAL", "label actual", anchor="end"),
            _svg_text(
                640,
                511,
                f"{ddp['passed_gate_count']}/{ddp_scale['gate_count']} gates",
                "metric",
            ),
            _svg_text(640, 538, "actual CPU/Gloo correctness gates passed", "body"),
            _svg_text(
                640,
                570,
                f"gradient max |error| {float(ddp['gradient_max_absolute_error']):.2e}",
                "body",
            ),
            _svg_text(
                640,
                594,
                f"final weight max |error| {float(ddp['final_weight_max_absolute_error']):.2e}",
                "body",
            ),
            _svg_text(
                640,
                618,
                f"resume state/loss error {float(ddp['resume_state_max_absolute_error']):.1f} / {float(ddp['resume_loss_max_absolute_error']):.1f}",
                "body",
            ),
            _svg_text(640, 666, "Boundary: 2-rank CPU/Gloo; no NCCL or scaling claim", "warn"),
        ]
    )

    # Actual GPU VLM LoRA smoke card. Descriptive evaluation is intentionally not scored.
    lines.extend(
        [
            _svg_text(76, 764, "Qwen3-VL-8B LoRA Pipeline", "cardTitle"),
            _svg_text(1122, 764, "ACTUAL GPU SMOKE", "label actual", anchor="end"),
            _svg_text(
                76,
                812,
                f"{qwen3_vl_lora_scale['optimizer_steps']} steps",
                "metric",
            ),
            _svg_text(222, 792, "actual BF16 LoRA optimizer updates", "body"),
            _svg_text(
                222,
                816,
                f"{int(qwen3_vl_lora_scale['trainable_parameters']):,} trainable · {float(qwen3_vl_lora['peak_allocated_gib']):.2f} GiB peak · {float(qwen3_vl_lora['total_seconds']):.2f} s total",
                "body",
            ),
            _svg_text(
                720,
                792,
                f"held-out EM {_percent(float(qwen3_vl_lora['heldout_exact_match_before']))} to {_percent(float(qwen3_vl_lora['heldout_exact_match_after']))}",
                "body",
            ),
            _svg_text(
                720,
                816,
                "3 grouped synthetic examples · Δ 0 · no quality improvement claimed",
                "warn",
            ),
        ]
    )

    lines.extend(
        [
            '<line x1="48" y1="880" x2="1152" y2="880" class="divider"/>',
            _svg_text(
                48,
                914,
                f"SOURCE SET  {source_sha[:16]}…  ·  {source_set['bundle_count']} bundles  ·  {source_set['verified_file_count']} files SHA-256 verified",
                "muted",
            ),
            _svg_text(
                48,
                940,
                "Read every result with its claim boundary. Actual is not production; scripted is not model capability; simulated is not measured serving.",
                "muted",
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_portfolio_scorecard(
    evidence_root: str | Path,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Validate canonical evidence and atomically publish JSON and SVG scorecards."""

    scorecard = build_portfolio_scorecard(evidence_root)
    json_payload = (
        json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    svg_payload = render_scorecard_svg(scorecard).encode("utf-8")
    destination = Path(output_directory)
    json_path = destination / "scorecard.json"
    svg_path = destination / "scorecard.svg"
    manifest_path = destination / "evidence_manifest.json"
    _atomic_write(json_path, json_payload)
    _atomic_write(svg_path, svg_payload)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "generator": "fmlab.portfolio.scorecard",
        "destination_name": destination.name,
        "claim_boundary": {
            "evidenced": (
                "Deterministic aggregation of five manifest-verified public-evidence "
                "source bundles; every displayed metric is recomputed from canonical JSON."
            ),
            "not_evidenced": [
                "A composite model-quality score, leaderboard rank, or production readiness.",
                "Evidence beyond the source bundles and their individual claim boundaries.",
            ],
        },
        "claim_boundary_source": "explicit_export_argument",
        "files": [
            {
                "path": "scorecard.json",
                "source_sha256": _sha256(json_payload),
                "public_sha256": _sha256(json_payload),
                "public_bytes": len(json_payload),
            },
            {
                "path": "scorecard.svg",
                "source_sha256": _sha256(svg_payload),
                "public_sha256": _sha256(svg_payload),
                "public_bytes": len(svg_payload),
            },
        ],
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(manifest_path, manifest_payload)
    return json_path, svg_path
