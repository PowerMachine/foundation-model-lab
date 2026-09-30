"""Strict configuration contracts for the inference dynamics experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _section(value: Any, *, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    return value


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _integer_choices(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list")
    return tuple(_integer(item, name=f"{name}[]", minimum=1) for item in value)


@dataclass(frozen=True)
class TraceConfig:
    pattern: str = "mixed"
    request_count: int = 72
    poisson_rate_rps: float = 18.0
    prompt_tokens: tuple[int, ...] = (8, 16, 24, 32)
    output_tokens: tuple[int, ...] = (4, 8, 12, 16)
    burst_every: int = 18
    burst_size: int = 6
    burst_gap_ms: float = 0.25
    timeout_ms: float = 450.0
    cancellation_fraction: float = 0.06
    cancellation_after_ms: tuple[float, float] = (40.0, 180.0)

    @classmethod
    def from_mapping(cls, value: Any) -> TraceConfig:
        raw = _section(
            value,
            name="trace",
            allowed={
                "pattern",
                "request_count",
                "poisson_rate_rps",
                "prompt_tokens",
                "output_tokens",
                "burst_every",
                "burst_size",
                "burst_gap_ms",
                "timeout_ms",
                "cancellation_fraction",
                "cancellation_after_ms",
            },
        )
        pattern = raw.get("pattern", cls.pattern)
        if pattern not in {"poisson", "burst", "mixed"}:
            raise ValueError("trace.pattern must be poisson, burst, or mixed")
        cancellation = raw.get("cancellation_after_ms", list(cls.cancellation_after_ms))
        if not isinstance(cancellation, list) or len(cancellation) != 2:
            raise TypeError("trace.cancellation_after_ms must contain [minimum, maximum]")
        cancel_bounds = (
            _number(cancellation[0], name="trace.cancellation_after_ms[0]", minimum=0.0),
            _number(cancellation[1], name="trace.cancellation_after_ms[1]", minimum=0.0),
        )
        if cancel_bounds[1] < cancel_bounds[0]:
            raise ValueError("trace cancellation maximum must be >= minimum")
        result = cls(
            pattern=pattern,
            request_count=_integer(
                raw.get("request_count", cls.request_count),
                name="trace.request_count",
                minimum=1,
            ),
            poisson_rate_rps=_number(
                raw.get("poisson_rate_rps", cls.poisson_rate_rps),
                name="trace.poisson_rate_rps",
                minimum=1e-9,
            ),
            prompt_tokens=_integer_choices(
                raw.get("prompt_tokens", list(cls.prompt_tokens)),
                name="trace.prompt_tokens",
            ),
            output_tokens=_integer_choices(
                raw.get("output_tokens", list(cls.output_tokens)),
                name="trace.output_tokens",
            ),
            burst_every=_integer(
                raw.get("burst_every", cls.burst_every),
                name="trace.burst_every",
                minimum=2,
            ),
            burst_size=_integer(
                raw.get("burst_size", cls.burst_size),
                name="trace.burst_size",
                minimum=2,
            ),
            burst_gap_ms=_number(
                raw.get("burst_gap_ms", cls.burst_gap_ms),
                name="trace.burst_gap_ms",
                minimum=0.0,
            ),
            timeout_ms=_number(
                raw.get("timeout_ms", cls.timeout_ms),
                name="trace.timeout_ms",
                minimum=1e-9,
            ),
            cancellation_fraction=_number(
                raw.get("cancellation_fraction", cls.cancellation_fraction),
                name="trace.cancellation_fraction",
                minimum=0.0,
            ),
            cancellation_after_ms=cancel_bounds,
        )
        if result.cancellation_fraction > 1.0:
            raise ValueError("trace.cancellation_fraction must be <= 1")
        if result.burst_size > result.burst_every and result.pattern == "mixed":
            raise ValueError("mixed trace burst_size must be <= burst_every")
        return result


@dataclass(frozen=True)
class EngineConfig:
    max_batch_size: int = 8
    queue_capacity: int = 20
    static_batch_wait_ms: float = 8.0
    prefill_base_ms: float = 0.8
    prefill_token_ms: float = 0.045
    decode_base_ms: float = 0.75
    decode_per_sequence_ms: float = 0.055
    slo_e2e_ms: float = 220.0
    max_simulation_ms: float = 30_000.0
    kv_block_size_tokens: int = 8
    kv_total_blocks: int = 72
    max_preemptions_per_request: int = 1

    @classmethod
    def from_mapping(cls, value: Any) -> EngineConfig:
        raw = _section(
            value,
            name="engine",
            allowed={
                "max_batch_size",
                "queue_capacity",
                "static_batch_wait_ms",
                "prefill_base_ms",
                "prefill_token_ms",
                "decode_base_ms",
                "decode_per_sequence_ms",
                "slo_e2e_ms",
                "max_simulation_ms",
                "kv_block_size_tokens",
                "kv_total_blocks",
                "max_preemptions_per_request",
            },
        )
        return cls(
            max_batch_size=_integer(
                raw.get("max_batch_size", cls.max_batch_size),
                name="engine.max_batch_size",
                minimum=1,
            ),
            queue_capacity=_integer(
                raw.get("queue_capacity", cls.queue_capacity),
                name="engine.queue_capacity",
                minimum=1,
            ),
            static_batch_wait_ms=_number(
                raw.get("static_batch_wait_ms", cls.static_batch_wait_ms),
                name="engine.static_batch_wait_ms",
                minimum=0.0,
            ),
            prefill_base_ms=_number(
                raw.get("prefill_base_ms", cls.prefill_base_ms),
                name="engine.prefill_base_ms",
                minimum=0.0,
            ),
            prefill_token_ms=_number(
                raw.get("prefill_token_ms", cls.prefill_token_ms),
                name="engine.prefill_token_ms",
                minimum=0.0,
            ),
            decode_base_ms=_number(
                raw.get("decode_base_ms", cls.decode_base_ms),
                name="engine.decode_base_ms",
                minimum=1e-9,
            ),
            decode_per_sequence_ms=_number(
                raw.get("decode_per_sequence_ms", cls.decode_per_sequence_ms),
                name="engine.decode_per_sequence_ms",
                minimum=0.0,
            ),
            slo_e2e_ms=_number(
                raw.get("slo_e2e_ms", cls.slo_e2e_ms),
                name="engine.slo_e2e_ms",
                minimum=1e-9,
            ),
            max_simulation_ms=_number(
                raw.get("max_simulation_ms", cls.max_simulation_ms),
                name="engine.max_simulation_ms",
                minimum=1.0,
            ),
            kv_block_size_tokens=_integer(
                raw.get("kv_block_size_tokens", cls.kv_block_size_tokens),
                name="engine.kv_block_size_tokens",
                minimum=1,
            ),
            kv_total_blocks=_integer(
                raw.get("kv_total_blocks", cls.kv_total_blocks),
                name="engine.kv_total_blocks",
                minimum=1,
            ),
            max_preemptions_per_request=_integer(
                raw.get("max_preemptions_per_request", cls.max_preemptions_per_request),
                name="engine.max_preemptions_per_request",
                minimum=0,
            ),
        )


@dataclass(frozen=True)
class FaultConfig:
    enabled: bool = True
    start_ms: float = 140.0
    duration_ms: float = 120.0
    service_multiplier: float = 3.0

    @classmethod
    def from_mapping(cls, value: Any) -> FaultConfig:
        raw = _section(
            value,
            name="fault",
            allowed={"enabled", "start_ms", "duration_ms", "service_multiplier"},
        )
        result = cls(
            enabled=_boolean(raw.get("enabled", cls.enabled), name="fault.enabled"),
            start_ms=_number(raw.get("start_ms", cls.start_ms), name="fault.start_ms", minimum=0.0),
            duration_ms=_number(
                raw.get("duration_ms", cls.duration_ms),
                name="fault.duration_ms",
                minimum=0.0,
            ),
            service_multiplier=_number(
                raw.get("service_multiplier", cls.service_multiplier),
                name="fault.service_multiplier",
                minimum=1.0,
            ),
        )
        if result.enabled and result.duration_ms <= 0:
            raise ValueError("enabled fault.duration_ms must be positive")
        return result


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    kind: str
    preemption_policy: str = "none"

    @classmethod
    def from_mapping(cls, value: Any, *, index: int) -> PolicyConfig:
        raw = _section(
            value,
            name=f"policies[{index}]",
            allowed={"name", "kind", "preemption_policy"},
        )
        name = raw.get("name")
        kind = raw.get("kind")
        preemption = raw.get("preemption_policy", "none")
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"policies[{index}].name must be a non-empty string")
        if kind not in {"static_fcfs", "continuous_fcfs"}:
            raise ValueError(f"policies[{index}].kind is invalid")
        if preemption not in {"none", "preempt_largest"}:
            raise ValueError(f"policies[{index}].preemption_policy is invalid")
        if kind == "static_fcfs" and preemption != "none":
            raise ValueError("static_fcfs only supports preemption_policy=none")
        return cls(name=name, kind=kind, preemption_policy=preemption)


@dataclass(frozen=True)
class ProbeGates:
    int8_min_top1_agreement: float = 0.90
    int8_min_cosine_similarity: float = 0.999
    int8_max_kl_divergence: float = 0.01
    int4_min_top1_agreement: float = 0.65
    int4_min_cosine_similarity: float = 0.97
    int4_max_kl_divergence: float = 0.08
    max_latency_ratio: float = 3.0

    @classmethod
    def from_mapping(cls, value: Any) -> ProbeGates:
        raw = _section(
            value,
            name="probe.gates",
            allowed={
                "int8_min_top1_agreement",
                "int8_min_cosine_similarity",
                "int8_max_kl_divergence",
                "int4_min_top1_agreement",
                "int4_min_cosine_similarity",
                "int4_max_kl_divergence",
                "max_latency_ratio",
            },
        )
        kwargs = {
            key: _number(raw.get(key, getattr(cls, key)), name=f"probe.gates.{key}", minimum=0.0)
            for key in raw.keys()
            | {
                "int8_min_top1_agreement",
                "int8_min_cosine_similarity",
                "int8_max_kl_divergence",
                "int4_min_top1_agreement",
                "int4_min_cosine_similarity",
                "int4_max_kl_divergence",
                "max_latency_ratio",
            }
        }
        result = cls(**kwargs)
        for name in (
            "int8_min_top1_agreement",
            "int8_min_cosine_similarity",
            "int4_min_top1_agreement",
            "int4_min_cosine_similarity",
        ):
            if getattr(result, name) > 1.0:
                raise ValueError(f"probe.gates.{name} must be <= 1")
        return result


@dataclass(frozen=True)
class ProbeConfig:
    enabled: bool = True
    seed: int = 31415
    num_threads: int = 1
    batch_size: int = 3
    sequence_length: int = 24
    warmup: int = 3
    iterations: int = 15
    vocab_size: int = 260
    max_seq_len: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    ffn_hidden_size: int = 96
    gates: ProbeGates = ProbeGates()

    @classmethod
    def from_mapping(cls, value: Any) -> ProbeConfig:
        raw = _section(
            value,
            name="probe",
            allowed={
                "enabled",
                "seed",
                "num_threads",
                "batch_size",
                "sequence_length",
                "warmup",
                "iterations",
                "vocab_size",
                "max_seq_len",
                "d_model",
                "n_layers",
                "n_heads",
                "ffn_hidden_size",
                "gates",
            },
        )
        result = cls(
            enabled=_boolean(raw.get("enabled", cls.enabled), name="probe.enabled"),
            seed=_integer(raw.get("seed", cls.seed), name="probe.seed", minimum=0),
            num_threads=_integer(
                raw.get("num_threads", cls.num_threads), name="probe.num_threads", minimum=1
            ),
            batch_size=_integer(
                raw.get("batch_size", cls.batch_size), name="probe.batch_size", minimum=1
            ),
            sequence_length=_integer(
                raw.get("sequence_length", cls.sequence_length),
                name="probe.sequence_length",
                minimum=2,
            ),
            warmup=_integer(raw.get("warmup", cls.warmup), name="probe.warmup", minimum=0),
            iterations=_integer(
                raw.get("iterations", cls.iterations), name="probe.iterations", minimum=1
            ),
            vocab_size=_integer(
                raw.get("vocab_size", cls.vocab_size), name="probe.vocab_size", minimum=8
            ),
            max_seq_len=_integer(
                raw.get("max_seq_len", cls.max_seq_len), name="probe.max_seq_len", minimum=2
            ),
            d_model=_integer(raw.get("d_model", cls.d_model), name="probe.d_model", minimum=4),
            n_layers=_integer(raw.get("n_layers", cls.n_layers), name="probe.n_layers", minimum=1),
            n_heads=_integer(raw.get("n_heads", cls.n_heads), name="probe.n_heads", minimum=1),
            ffn_hidden_size=_integer(
                raw.get("ffn_hidden_size", cls.ffn_hidden_size),
                name="probe.ffn_hidden_size",
                minimum=4,
            ),
            gates=ProbeGates.from_mapping(raw.get("gates", {})),
        )
        if result.sequence_length > result.max_seq_len:
            raise ValueError("probe.sequence_length exceeds probe.max_seq_len")
        if result.d_model % result.n_heads:
            raise ValueError("probe.d_model must be divisible by probe.n_heads")
        if (result.d_model // result.n_heads) % 2:
            raise ValueError("probe attention head dimension must be even for RoPE")
        return result


@dataclass(frozen=True)
class CapacitySweepConfig:
    enabled: bool = False
    poisson_rates_rps: tuple[float, ...] = (4.0, 8.0, 12.0, 18.0, 24.0, 32.0)
    request_count: int = 72
    seeds: tuple[int, ...] = (101, 202, 303)
    minimum_completion_ratio: float = 0.90
    minimum_slo_attainment: float = 0.90
    maximum_terminal_failure_ratio: float = 0.10

    @classmethod
    def from_mapping(cls, value: Any) -> CapacitySweepConfig:
        raw = _section(
            value,
            name="capacity_sweep",
            allowed={
                "enabled",
                "poisson_rates_rps",
                "request_count",
                "seeds",
                "minimum_completion_ratio",
                "minimum_slo_attainment",
                "maximum_terminal_failure_ratio",
            },
        )
        rates_raw = raw.get("poisson_rates_rps", list(cls.poisson_rates_rps))
        if not isinstance(rates_raw, list) or not rates_raw:
            raise TypeError("capacity_sweep.poisson_rates_rps must be a non-empty list")
        rates = tuple(
            _number(item, name="capacity_sweep.poisson_rates_rps[]", minimum=1e-9)
            for item in rates_raw
        )
        seeds_raw = raw.get("seeds", list(cls.seeds))
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise TypeError("capacity_sweep.seeds must be a non-empty list")
        seeds = tuple(
            _integer(item, name="capacity_sweep.seeds[]", minimum=0) for item in seeds_raw
        )
        result = cls(
            enabled=_boolean(raw.get("enabled", cls.enabled), name="capacity_sweep.enabled"),
            poisson_rates_rps=rates,
            request_count=_integer(
                raw.get("request_count", cls.request_count),
                name="capacity_sweep.request_count",
                minimum=2,
            ),
            seeds=seeds,
            minimum_completion_ratio=_number(
                raw.get("minimum_completion_ratio", cls.minimum_completion_ratio),
                name="capacity_sweep.minimum_completion_ratio",
                minimum=0.0,
            ),
            minimum_slo_attainment=_number(
                raw.get("minimum_slo_attainment", cls.minimum_slo_attainment),
                name="capacity_sweep.minimum_slo_attainment",
                minimum=0.0,
            ),
            maximum_terminal_failure_ratio=_number(
                raw.get("maximum_terminal_failure_ratio", cls.maximum_terminal_failure_ratio),
                name="capacity_sweep.maximum_terminal_failure_ratio",
                minimum=0.0,
            ),
        )
        if result.enabled and len(result.seeds) < 3:
            raise ValueError("enabled capacity_sweep requires at least three seeds")
        if len(result.seeds) != len(set(result.seeds)):
            raise ValueError("capacity_sweep.seeds must be unique")
        if any(later <= earlier for earlier, later in zip(rates, rates[1:])):
            raise ValueError("capacity_sweep.poisson_rates_rps must be strictly increasing")
        if len(rates) * len(seeds) > 128:
            raise ValueError("capacity_sweep supports at most 128 rate/seed trials")
        for name in (
            "minimum_completion_ratio",
            "minimum_slo_attainment",
            "maximum_terminal_failure_ratio",
        ):
            if getattr(result, name) > 1.0:
                raise ValueError(f"capacity_sweep.{name} must be <= 1")
        return result


@dataclass(frozen=True)
class InferenceDynamicsConfig:
    artifact_dir: Path
    seed: int
    trace: TraceConfig
    engine: EngineConfig
    fault: FaultConfig
    policies: tuple[PolicyConfig, ...]
    probe: ProbeConfig
    capacity_sweep: CapacitySweepConfig = CapacitySweepConfig()
    experiment: str = "inference_dynamics"

    @classmethod
    def from_mapping(cls, value: Any) -> InferenceDynamicsConfig:
        raw = _section(
            value,
            name="config",
            allowed={
                "experiment",
                "artifact_dir",
                "seed",
                "trace",
                "engine",
                "fault",
                "policies",
                "probe",
                "capacity_sweep",
            },
        )
        experiment = raw.get("experiment", "inference_dynamics")
        if experiment != "inference_dynamics":
            raise ValueError("experiment must be inference_dynamics")
        artifact = raw.get("artifact_dir")
        if not isinstance(artifact, str) or not artifact.strip():
            raise TypeError("artifact_dir must be a non-empty path string")
        policies_raw = raw.get("policies")
        if not isinstance(policies_raw, list) or not policies_raw:
            raise TypeError("policies must be a non-empty list")
        policies = tuple(
            PolicyConfig.from_mapping(item, index=index) for index, item in enumerate(policies_raw)
        )
        names = [policy.name for policy in policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        kinds = {policy.kind for policy in policies}
        if kinds != {"static_fcfs", "continuous_fcfs"}:
            raise ValueError("policies must include static_fcfs and continuous_fcfs")
        return cls(
            artifact_dir=Path(artifact).expanduser(),
            seed=_integer(raw.get("seed", 20260804), name="seed", minimum=0),
            trace=TraceConfig.from_mapping(raw.get("trace", {})),
            engine=EngineConfig.from_mapping(raw.get("engine", {})),
            fault=FaultConfig.from_mapping(raw.get("fault", {})),
            policies=policies,
            probe=ProbeConfig.from_mapping(raw.get("probe", {})),
            capacity_sweep=CapacitySweepConfig.from_mapping(raw.get("capacity_sweep", {})),
            experiment=experiment,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact_dir"] = str(self.artifact_dir)
        return value
