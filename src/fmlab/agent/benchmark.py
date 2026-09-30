from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fmlab.paths import LabPaths

from .benchmark_ledger import AtomicJsonlStore, AtomicResumeLedger
from .benchmark_policies import BenchmarkPolicy, build_policies
from .benchmark_report import render_benchmark_html, render_metrics_svg
from .benchmark_tasks import build_task_suite, task_suite_fingerprint
from .benchmark_types import TaskSpec, sha256_json
from .eval_environment import ReliableAgentEnvironment
from .sandbox import ResourceLimits


BENCHMARK_SCHEMA = "fmlab.reliable-agent-benchmark/v1"
EPISODE_SCHEMA = "fmlab.agent-episode/v2"


def _expanded_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


@dataclass(frozen=True)
class AgentBenchmarkConfig:
    artifact_dir: Path
    scratch_dir: Path
    policies: tuple[str, ...] = (
        "oracle_patch_scripted",
        "shortcut_probe_scripted",
    )
    isolation: str = "process"
    shard_index: int = 0
    shard_count: int = 1
    task_limit: int | None = None
    seed: int = 20260804
    bootstrap_resamples: int = 1000
    max_retries: int = 1
    inject_transient_failures: bool = True
    transient_failure_modulus: int = 5
    timeout_seconds: int = 5
    cpu_seconds: int = 3
    memory_mb: int = 512

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> AgentBenchmarkConfig:
        known = {
            "artifact_dir",
            "bootstrap_resamples",
            "cpu_seconds",
            "inject_transient_failures",
            "isolation",
            "max_retries",
            "memory_mb",
            "policies",
            "scratch_dir",
            "seed",
            "shard_count",
            "shard_index",
            "task_limit",
            "timeout_seconds",
            "transient_failure_modulus",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown agent benchmark config keys: {unknown}")
        default_artifact = LabPaths.from_env().artifacts / "agent" / "reliable-agent-benchmark"
        artifact = _expanded_path(
            values.get(
                "artifact_dir",
                default_artifact,
            )
        )
        scratch = _expanded_path(values.get("scratch_dir", artifact / ".scratch"))
        policy_values = values.get(
            "policies",
            ["oracle_patch_scripted", "shortcut_probe_scripted"],
        )
        if not isinstance(policy_values, list) or not all(
            isinstance(item, str) for item in policy_values
        ):
            raise ValueError("policies must be a list of strings")
        config = cls(
            artifact_dir=artifact,
            scratch_dir=scratch,
            policies=tuple(policy_values),
            isolation=str(values.get("isolation", "process")),
            shard_index=int(values.get("shard_index", 0)),
            shard_count=int(values.get("shard_count", 1)),
            task_limit=(None if values.get("task_limit") is None else int(values["task_limit"])),
            seed=int(values.get("seed", 20260804)),
            bootstrap_resamples=int(values.get("bootstrap_resamples", 1000)),
            max_retries=int(values.get("max_retries", 1)),
            inject_transient_failures=bool(values.get("inject_transient_failures", True)),
            transient_failure_modulus=int(values.get("transient_failure_modulus", 5)),
            timeout_seconds=int(values.get("timeout_seconds", 5)),
            cpu_seconds=int(values.get("cpu_seconds", 3)),
            memory_mb=int(values.get("memory_mb", 512)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.isolation not in {"process", "bwrap"}:
            raise ValueError("isolation must be process or bwrap")
        if len(self.policies) < 2:
            raise ValueError("at least two policy controls are required")
        build_policies(self.policies)
        if self.shard_count < 1 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        if self.task_limit is not None and self.task_limit < 1:
            raise ValueError("task_limit must be positive when set")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if self.transient_failure_modulus < 2:
            raise ValueError("transient_failure_modulus must be at least 2")
        if min(self.timeout_seconds, self.cpu_seconds, self.memory_mb) < 1:
            raise ValueError("resource limits must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact_dir"] = str(self.artifact_dir)
        value["scratch_dir"] = str(self.scratch_dir)
        value["policies"] = list(self.policies)
        return value


def deterministic_shard(task_uid: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(task_uid.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def episode_identity(
    spec: TaskSpec, policy: BenchmarkPolicy, config: AgentBenchmarkConfig
) -> tuple[str, str]:
    contract = {
        "schema": EPISODE_SCHEMA,
        "task_fingerprint_sha256": spec.fingerprint,
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "environment_contract": "reset-observe-score/v2",
        "isolation": config.isolation,
        "limits": {
            "timeout_seconds": config.timeout_seconds,
            "cpu_seconds": config.cpu_seconds,
            "memory_mb": config.memory_mb,
        },
    }
    contract_sha = sha256_json(contract)
    return "episode-" + contract_sha[:32], contract_sha


def bootstrap_ci(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(set(values)) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.fmean(sample))
    estimates.sort()
    low = estimates[math.floor(0.025 * (len(estimates) - 1))]
    high = estimates[math.ceil(0.975 * (len(estimates) - 1))]
    return [float(low), float(high)]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    passes = [float(bool(row.get("passed"))) for row in rows]
    outcomes = [float(bool(row.get("outcome_passed"))) for row in rows]
    hacks = [float(bool(row.get("reward_hack_signals"))) for row in rows]
    tool_calls = sum(int(row.get("tool_calls", 0)) for row in rows)
    tool_errors = sum(int(row.get("tool_errors", 0)) for row in rows)
    unsafe = sum(int(row.get("unsafe_action_attempts", 0)) for row in rows)
    latencies = [float(row.get("latency_seconds", 0.0)) for row in rows]
    return {
        "episodes": len(rows),
        "pass_at_1": statistics.fmean(passes) if passes else 0.0,
        "pass_at_1_ci_95": bootstrap_ci(passes, resamples=resamples, seed=seed),
        "outcome_pass_rate": statistics.fmean(outcomes) if outcomes else 0.0,
        "reward_hack_rate": statistics.fmean(hacks) if hacks else 0.0,
        "mean_steps": statistics.fmean([float(row.get("steps", 0)) for row in rows])
        if rows
        else 0.0,
        "p50_latency_seconds": _percentile(latencies, 0.5),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "tool_error_rate": tool_errors / tool_calls if tool_calls else 0.0,
        "unsafe_action_rate": unsafe / tool_calls if tool_calls else 0.0,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "unsafe_action_attempts": unsafe,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inject_transient(episode_id: str, config: AgentBenchmarkConfig) -> bool:
    if not config.inject_transient_failures:
        return False
    digest = hashlib.sha256((episode_id + ":transient-v1").encode()).hexdigest()
    return int(digest[:8], 16) % config.transient_failure_modulus == 0


def _run_episode(
    spec: TaskSpec,
    policy: BenchmarkPolicy,
    config: AgentBenchmarkConfig,
    *,
    episode_id: str,
    contract_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = ResourceLimits(
        timeout_seconds=config.timeout_seconds,
        cpu_seconds=config.cpu_seconds,
        memory_mb=config.memory_mb,
        output_bytes=20_000,
        file_bytes=2_000_000,
        processes=16,
    )
    attempt_events: list[dict[str, Any]] = []
    injected = _inject_transient(episode_id, config)
    final_error: Exception | None = None
    for attempt in range(1, config.max_retries + 2):
        if injected and attempt == 1:
            attempt_events.append(
                {
                    "attempt": attempt,
                    "kind": "injected_transient_failure",
                    "retryable": True,
                }
            )
            final_error = RuntimeError("deterministic injected transient failure")
            continue
        environment = ReliableAgentEnvironment(
            spec,
            config.scratch_dir / "workspaces",
            config.scratch_dir / "private-graders",
            isolation=config.isolation,
            limits=limits,
        )
        started = time.monotonic()
        try:
            observation = environment.reset()
            policy.run(environment, observation)
            grade = environment.score()
            elapsed = time.monotonic() - started
            attempt_events.append({"attempt": attempt, "kind": "completed", "retryable": False})
            trace = {
                "episode_id": episode_id,
                "episode_schema": EPISODE_SCHEMA,
                "contract_sha256": contract_sha,
                "task_uid": spec.uid,
                "task_fingerprint_sha256": spec.fingerprint,
                "policy_id": policy.policy_id,
                "policy_version": policy.version,
                "attempt_events": attempt_events,
                "observation": observation.to_dict(),
                "actions": [record.to_dict() for record in environment.trace],
                "grade": grade.to_dict(),
            }
            summary = {
                "episode_id": episode_id,
                "status": "completed",
                "contract_sha256": contract_sha,
                "task_uid": spec.uid,
                "task_id": spec.task_id,
                "perturbation_id": spec.perturbation_id,
                "task_fingerprint_sha256": spec.fingerprint,
                "policy_id": policy.policy_id,
                "policy_version": policy.version,
                "passed": grade.passed,
                "outcome_passed": grade.outcome.passed,
                "failure_category": grade.failure_category,
                "steps": grade.process.steps,
                "tool_calls": grade.process.tool_calls,
                "tool_errors": grade.process.tool_errors,
                "unsafe_action_attempts": grade.process.unsafe_action_attempts,
                "reward_hack_signals": list(grade.process.reward_hack_signals),
                "latency_seconds": elapsed,
                "hidden_cases_total": grade.outcome.cases_total,
                "hidden_cases_passed": grade.outcome.cases_passed,
                "isolation": grade.outcome.isolation,
                "network_isolated": grade.outcome.network_isolated,
                "attempts": attempt,
                "transient_failure_injected": injected,
                "trace_sha256": sha256_json(trace),
            }
            return summary, trace
        except Exception as error:
            final_error = error
            attempt_events.append(
                {
                    "attempt": attempt,
                    "kind": "infrastructure_error",
                    "retryable": attempt <= config.max_retries,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            environment.close()

    assert final_error is not None
    trace = {
        "episode_id": episode_id,
        "episode_schema": EPISODE_SCHEMA,
        "contract_sha256": contract_sha,
        "task_uid": spec.uid,
        "task_fingerprint_sha256": spec.fingerprint,
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "attempt_events": attempt_events,
        "actions": [],
        "infrastructure_error": f"{type(final_error).__name__}: {final_error}",
    }
    summary = {
        "episode_id": episode_id,
        "status": "failed",
        "contract_sha256": contract_sha,
        "task_uid": spec.uid,
        "task_id": spec.task_id,
        "perturbation_id": spec.perturbation_id,
        "task_fingerprint_sha256": spec.fingerprint,
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "passed": False,
        "outcome_passed": False,
        "failure_category": "infrastructure_error",
        "steps": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "unsafe_action_attempts": 0,
        "reward_hack_signals": [],
        "latency_seconds": 0.0,
        "hidden_cases_total": len(spec.hidden_cases),
        "hidden_cases_passed": 0,
        "isolation": config.isolation,
        "network_isolated": False,
        "attempts": config.max_retries + 1,
        "transient_failure_injected": injected,
        "trace_sha256": sha256_json(trace),
    }
    return summary, trace


def run_agent_benchmark(config: AgentBenchmarkConfig) -> Path:
    config.validate()
    started_wall = datetime.now(UTC)
    started_monotonic = time.monotonic()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.scratch_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = build_task_suite()
    selected = tuple(
        task
        for task in all_tasks
        if deterministic_shard(task.uid, config.shard_count) == config.shard_index
    )
    if config.task_limit is not None:
        selected = selected[: config.task_limit]
    if not selected:
        raise ValueError("selected shard contains no tasks")
    policies = build_policies(config.policies)

    ledger = AtomicResumeLedger(config.artifact_dir / "ledger.jsonl")
    trace_store = AtomicJsonlStore(
        config.artifact_dir / "traces.jsonl",
        key_field="episode_id",
        schema="fmlab.agent-traces/v1",
    )
    existing = ledger.read_all()
    planned: list[tuple[TaskSpec, BenchmarkPolicy, str, str]] = []
    for task in selected:
        for policy in policies:
            episode_id, contract_sha = episode_identity(task, policy, config)
            planned.append((task, policy, episode_id, contract_sha))

    resumed = 0
    executed = 0
    for task, policy, episode_id, contract_sha in planned:
        previous = existing.get(episode_id)
        if previous and previous.get("status") == "completed":
            if previous.get("contract_sha256") != contract_sha:
                raise ValueError(f"resume contract mismatch for {episode_id}")
            resumed += 1
            continue
        summary, trace = _run_episode(
            task,
            policy,
            config,
            episode_id=episode_id,
            contract_sha=contract_sha,
        )
        trace_store.upsert(trace)
        ledger.upsert(summary)
        existing[episode_id] = summary
        executed += 1

    final_ledger = ledger.read_all()
    rows = [final_ledger[episode_id] for _, _, episode_id, _ in planned]
    overall = _summarize_rows(
        rows,
        resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    by_policy: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(policies):
        by_policy[policy.policy_id] = _summarize_rows(
            [row for row in rows if row["policy_id"] == policy.policy_id],
            resamples=config.bootstrap_resamples,
            seed=config.seed + index + 1,
        )
        by_policy[policy.policy_id]["description"] = policy.description

    failure_taxonomy = Counter(str(row["failure_category"]) for row in rows)
    hack_categories = Counter(
        signal for row in rows for signal in row.get("reward_hack_signals", [])
    )
    metrics = {
        **overall,
        "tasks": len(selected),
        "policies": len(policies),
        "by_policy": by_policy,
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
        "reward_hack_categories": dict(sorted(hack_categories.items())),
        "execution": {
            "executed_episodes": executed,
            "resumed_episodes": resumed,
            "transient_failures_injected": sum(
                bool(row.get("transient_failure_injected")) for row in rows
            ),
            "episodes_requiring_retry": sum(int(row.get("attempts", 1)) > 1 for row in rows),
            "max_attempts_observed": max(int(row.get("attempts", 1)) for row in rows),
        },
    }
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    network_isolated = bool(completed_rows) and all(
        bool(row.get("network_isolated")) for row in completed_rows
    )
    full_suite_fp = task_suite_fingerprint(all_tasks)
    task_manifest = {
        "schema": "fmlab.agent-task-manifest/v1",
        "suite_fingerprint_sha256": full_suite_fp,
        "selected_task_uids": [task.uid for task in selected],
        "tasks": [task.public_manifest() for task in all_tasks],
    }
    _write_json_atomic(config.artifact_dir / "task_manifest.json", task_manifest)
    _write_json_atomic(
        config.artifact_dir / "run_manifest.json",
        {
            "schema": BENCHMARK_SCHEMA,
            "config": config.to_dict(),
            "config_sha256": sha256_json(config.to_dict()),
            "suite_fingerprint_sha256": full_suite_fp,
            "planned_episode_ids": [episode_id for _, _, episode_id, _ in planned],
        },
    )
    finished_wall = datetime.now(UTC)
    result = {
        "schema": BENCHMARK_SCHEMA,
        "experiment": "reliable_agent_eval_environment",
        "status": (
            "completed" if all(row.get("status") == "completed" for row in rows) else "partial"
        ),
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "duration_seconds": time.monotonic() - started_monotonic,
        "claim_level": "real_local_environment_measurement",
        "claim": {
            "scope": (
                "Local Python microtasks were executed and scored with deterministic scripted "
                "policy controls. This measures the eval environment, retry, integrity, and "
                "reporting machinery; it is not LLM inference and not a model-quality claim."
            ),
            "policy_type": "deterministic_scripted_controls",
            "model_inference": False,
            "execution_isolation": config.isolation,
            "network_isolated": network_isolated,
            "hidden_expected_values_exposed_to_observation_or_workspace": False,
            "in_process_policy_implementation_trusted": True,
            "grader_assets_materialized_after_policy": True,
        },
        "metrics": metrics,
        "parameters": config.to_dict(),
        "provenance": {
            "suite_fingerprint_sha256": full_suite_fp,
            "task_spec_schema": selected[0].schema_version,
            "episode_schema": EPISODE_SCHEMA,
            "ledger_schema": "fmlab.agent-ledger/v1",
            "shard": {
                "index": config.shard_index,
                "count": config.shard_count,
                "assignment": "sha256(task_uid)[0:8] modulo shard_count",
            },
        },
        "artifacts": [
            "ledger.jsonl",
            "metrics.svg",
            "report.html",
            "result.json",
            "run_manifest.json",
            "task_manifest.json",
            "traces.jsonl",
        ],
        "notes": [
            "pass@1 requires both hidden outcome success and process-integrity success.",
            "Injected transient failures are infrastructure retries, not additional policy attempts.",
            "The oracle scripted policy reads a declared reference patch and is only a harness upper bound.",
            "The shortcut scripted policy intentionally probes reward-hacking detectors.",
            (
                "Python policy implementations are trusted in-process harness code; hidden values "
                "are excluded from the model observation and workspace, not isolated from arbitrary "
                "Python introspection inside that trusted process."
            ),
            "Process mode is not a security boundary and does not isolate networking."
            if config.isolation == "process"
            else "Bubblewrap mode reports network isolation only for completed episodes.",
        ],
    }
    _write_json_atomic(config.artifact_dir / "result.json", result)
    render_metrics_svg(by_policy, config.artifact_dir / "metrics.svg")
    render_benchmark_html(
        result,
        rows,
        task_manifest["tasks"],
        config.artifact_dir / "report.html",
    )
    return config.artifact_dir / "result.json"
