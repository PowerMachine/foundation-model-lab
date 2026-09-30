"""Deterministic discrete-event model of an LLM serving scheduler.

The simulator intentionally models service times instead of pretending to run a
real serving engine.  Its output is useful for controlled scheduler experiments,
not for claims about vLLM, GPU kernels, or production capacity.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import EngineConfig, FaultConfig, PolicyConfig, TraceConfig


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    timeout_ms: float
    cancel_at_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_request_trace(config: TraceConfig, *, seed: int) -> list[RequestSpec]:
    """Generate a deterministic Poisson, burst, or mixed request trace."""

    rng = random.Random(seed)
    requests: list[RequestSpec] = []
    arrival_ms = 0.0
    mean_rate_per_ms = config.poisson_rate_rps / 1_000.0
    for index in range(config.request_count):
        if index:
            if config.pattern == "burst":
                in_burst = index % config.burst_size != 0
            elif config.pattern == "mixed":
                position = index % config.burst_every
                in_burst = 0 < position < config.burst_size
            else:
                in_burst = False
            gap = config.burst_gap_ms if in_burst else rng.expovariate(mean_rate_per_ms)
            arrival_ms += gap
        cancel_at = None
        if rng.random() < config.cancellation_fraction:
            cancel_after = rng.uniform(*config.cancellation_after_ms)
            cancel_at = arrival_ms + cancel_after
        requests.append(
            RequestSpec(
                request_id=f"req-{index:04d}",
                arrival_ms=round(arrival_ms, 9),
                prompt_tokens=rng.choice(config.prompt_tokens),
                output_tokens=rng.choice(config.output_tokens),
                timeout_ms=config.timeout_ms,
                cancel_at_ms=None if cancel_at is None else round(cancel_at, 9),
            )
        )
    return requests


class PagedKVAllocator:
    """Non-contiguous fixed-block allocator with observable internal fragmentation."""

    def __init__(self, *, total_blocks: int, block_size_tokens: int) -> None:
        if total_blocks < 1 or block_size_tokens < 1:
            raise ValueError("KV allocator dimensions must be positive")
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self._free_blocks = list(range(total_blocks))
        self._owners: dict[str, list[int]] = {}
        self._tokens: dict[str, int] = {}
        self.allocation_failures = 0
        self.peak_used_blocks = 0
        self.peak_fragmentation_ratio = 0.0

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self._free_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks

    @property
    def internal_fragmentation_tokens(self) -> int:
        return sum(
            len(blocks) * self.block_size_tokens - self._tokens[owner]
            for owner, blocks in self._owners.items()
        )

    @property
    def internal_fragmentation_ratio(self) -> float:
        capacity = self.used_blocks * self.block_size_tokens
        return self.internal_fragmentation_tokens / capacity if capacity else 0.0

    def blocks_for(self, request_id: str) -> int:
        return len(self._owners.get(request_id, ()))

    def reserve(self, request_id: str, token_count: int) -> bool:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        needed = math.ceil(token_count / self.block_size_tokens) if token_count else 0
        owned = self._owners.setdefault(request_id, [])
        additional = needed - len(owned)
        if additional > len(self._free_blocks):
            self.allocation_failures += 1
            if not owned:
                self._owners.pop(request_id, None)
            return False
        if additional > 0:
            allocated = self._free_blocks[:additional]
            del self._free_blocks[:additional]
            owned.extend(allocated)
        elif additional < 0:
            released = owned[additional:]
            del owned[additional:]
            self._free_blocks.extend(released)
            self._free_blocks.sort()
        self._tokens[request_id] = token_count
        if not owned:
            self._owners.pop(request_id, None)
            self._tokens.pop(request_id, None)
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)
        self.peak_fragmentation_ratio = max(
            self.peak_fragmentation_ratio, self.internal_fragmentation_ratio
        )
        return True

    def release(self, request_id: str) -> int:
        released = self._owners.pop(request_id, [])
        self._tokens.pop(request_id, None)
        self._free_blocks.extend(released)
        self._free_blocks.sort()
        return len(released)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "total_blocks": self.total_blocks,
            "utilization": self.utilization,
            "internal_fragmentation_tokens": self.internal_fragmentation_tokens,
            "internal_fragmentation_ratio": self.internal_fragmentation_ratio,
            # Paged allocation is non-contiguous, so external fragmentation is zero
            # by construction. Internal tail waste remains explicitly measured.
            "external_fragmentation_ratio": 0.0,
        }


@dataclass
class _RequestState:
    spec: RequestSpec
    status: str = "pending"
    queue_enter_ms: float | None = None
    queue_wait_ms: float = 0.0
    start_ms: float | None = None
    first_token_ms: float | None = None
    end_ms: float | None = None
    generated_tokens: int = 0
    token_times_ms: list[float] = field(default_factory=list)
    preemptions: int = 0
    needs_recompute: bool = False
    terminal_reason: str | None = None

    @property
    def timeout_at_ms(self) -> float:
        return self.spec.arrival_ms + self.spec.timeout_ms

    def next_client_deadline(self) -> tuple[float, str]:
        if self.spec.cancel_at_ms is not None and self.spec.cancel_at_ms <= self.timeout_at_ms:
            return self.spec.cancel_at_ms, "cancelled"
        return self.timeout_at_ms, "timed_out"


@dataclass(frozen=True)
class SimulationOutput:
    policy: dict[str, str]
    metrics: dict[str, Any]
    requests: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    kv_samples: tuple[dict[str, Any], ...]
    root_causes: tuple[dict[str, Any], ...]


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {"p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)}


class _ServingSimulation:
    def __init__(
        self,
        requests: list[RequestSpec],
        *,
        engine: EngineConfig,
        fault: FaultConfig,
        policy: PolicyConfig,
    ) -> None:
        if not requests:
            raise ValueError("at least one request is required")
        self.specs = sorted(requests, key=lambda item: (item.arrival_ms, item.request_id))
        self.engine = engine
        self.fault = fault
        self.policy = policy
        self.states = {spec.request_id: _RequestState(spec) for spec in self.specs}
        self.pending_index = 0
        self.queue: list[_RequestState] = []
        self.active: list[_RequestState] = []
        self.allocator = PagedKVAllocator(
            total_blocks=engine.kv_total_blocks,
            block_size_tokens=engine.kv_block_size_tokens,
        )
        self.now_ms = self.specs[0].arrival_ms
        self.stop_ms = self.now_ms + engine.max_simulation_ms
        self.events: list[dict[str, Any]] = []
        self.kv_samples: list[dict[str, Any]] = []
        self.queue_peak = 0
        self.static_batch_slots = 0
        self.static_wasted_token_slots = 0
        self.prefill_busy_ms = 0.0
        self.decode_busy_ms = 0.0
        self.idle_ms = 0.0
        self.fault_impacted_operations = 0
        self.fault_extra_ms = 0.0

    def event(self, event: str, at_ms: float, **values: Any) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "policy": self.policy.name,
                "time_ms": round(at_ms, 9),
                "event": event,
                **values,
            }
        )

    def sample_kv(self, at_ms: float, *, cause: str) -> None:
        self.kv_samples.append(
            {
                "policy": self.policy.name,
                "time_ms": round(at_ms, 9),
                "cause": cause,
                **self.allocator.snapshot(),
            }
        )

    def enqueue_arrival(self, spec: RequestSpec) -> None:
        state = self.states[spec.request_id]
        self.event(
            "arrival",
            spec.arrival_ms,
            request_id=spec.request_id,
            prompt_tokens=spec.prompt_tokens,
            output_tokens=spec.output_tokens,
        )
        if len(self.queue) >= self.engine.queue_capacity:
            self.finalize(
                state,
                status="rejected",
                at_ms=spec.arrival_ms,
                reason="queue_capacity",
            )
            return
        state.status = "queued"
        state.queue_enter_ms = spec.arrival_ms
        self.queue.append(state)
        self.queue_peak = max(self.queue_peak, len(self.queue))
        self.event(
            "queued", spec.arrival_ms, request_id=spec.request_id, queue_depth=len(self.queue)
        )

    def finalize(
        self,
        state: _RequestState,
        *,
        status: str,
        at_ms: float,
        reason: str,
    ) -> None:
        if state in self.queue:
            self.queue.remove(state)
        if state in self.active:
            self.active.remove(state)
        released = self.allocator.release(state.spec.request_id)
        state.status = status
        state.end_ms = at_ms
        state.terminal_reason = reason
        self.event(
            status,
            at_ms,
            request_id=state.spec.request_id,
            reason=reason,
            generated_tokens=state.generated_tokens,
            released_kv_blocks=released,
        )
        if released:
            self.sample_kv(at_ms, cause=status)

    def next_deadline(self) -> tuple[float, _RequestState, str] | None:
        candidates = []
        for state in (*self.queue, *self.active):
            deadline, status = state.next_client_deadline()
            candidates.append((deadline, state.spec.request_id, state, status))
        if not candidates:
            return None
        deadline, _, state, status = min(candidates, key=lambda item: (item[0], item[1]))
        return deadline, state, status

    def process_external_until(self, end_ms: float) -> None:
        epsilon = 1e-9
        while True:
            next_arrival = (
                self.specs[self.pending_index].arrival_ms
                if self.pending_index < len(self.specs)
                else math.inf
            )
            deadline_item = self.next_deadline()
            next_deadline = deadline_item[0] if deadline_item else math.inf
            next_time = min(next_arrival, next_deadline)
            if next_time > end_ms + epsilon:
                break
            # Expiry wins ties, making bounded-queue semantics explicit.
            while True:
                item = self.next_deadline()
                if item is None or item[0] > next_time + epsilon:
                    break
                deadline, state, status = item
                self.finalize(state, status=status, at_ms=deadline, reason="client_deadline")
            while (
                self.pending_index < len(self.specs)
                and self.specs[self.pending_index].arrival_ms <= next_time + epsilon
            ):
                spec = self.specs[self.pending_index]
                self.pending_index += 1
                self.enqueue_arrival(spec)

    def duration_with_fault(self, start_ms: float, work_ms: float) -> tuple[float, float]:
        if not self.fault.enabled or work_ms <= 0:
            return work_ms, 0.0
        fault_start = self.fault.start_ms
        fault_end = fault_start + self.fault.duration_ms
        cursor = start_ms
        remaining_work = work_ms
        elapsed = 0.0
        if cursor < fault_start:
            normal = min(remaining_work, fault_start - cursor)
            cursor += normal
            elapsed += normal
            remaining_work -= normal
        if remaining_work > 0 and cursor < fault_end and cursor >= fault_start:
            possible_work = (fault_end - cursor) / self.fault.service_multiplier
            slowed_work = min(remaining_work, possible_work)
            slowed_wall = slowed_work * self.fault.service_multiplier
            cursor += slowed_wall
            elapsed += slowed_wall
            remaining_work -= slowed_work
        elapsed += remaining_work
        return elapsed, elapsed - work_ms

    def advance_busy(self, work_ms: float, *, phase: str, batch_slots: int) -> None:
        start = self.now_ms
        elapsed, extra = self.duration_with_fault(start, work_ms)
        end = start + elapsed
        if extra > 0:
            self.event(
                "fault_slowdown",
                start,
                phase=phase,
                batch_slots=batch_slots,
                base_duration_ms=work_ms,
                extra_duration_ms=extra,
            )
        self.process_external_until(end)
        self.now_ms = end
        if phase == "prefill":
            self.prefill_busy_ms += elapsed
        else:
            self.decode_busy_ms += elapsed
        if extra > 0:
            self.fault_impacted_operations += 1
            self.fault_extra_ms += extra

    def preempt_largest(self, *, requester: _RequestState | None) -> bool:
        if self.policy.preemption_policy != "preempt_largest":
            return False
        if len(self.queue) >= self.engine.queue_capacity:
            return False
        candidates = [
            state
            for state in self.active
            if state is not requester
            and state.preemptions < self.engine.max_preemptions_per_request
        ]
        if not candidates:
            return False
        victim = max(
            candidates,
            key=lambda state: (
                self.allocator.blocks_for(state.spec.request_id),
                state.spec.prompt_tokens + state.generated_tokens,
                -state.spec.arrival_ms,
            ),
        )
        blocks = self.allocator.release(victim.spec.request_id)
        self.active.remove(victim)
        victim.status = "queued"
        victim.queue_enter_ms = self.now_ms
        victim.preemptions += 1
        victim.needs_recompute = True
        self.queue.append(victim)
        self.queue_peak = max(self.queue_peak, len(self.queue))
        self.event(
            "kv_preempted",
            self.now_ms,
            request_id=victim.spec.request_id,
            released_kv_blocks=blocks,
            preemptions=victim.preemptions,
        )
        self.sample_kv(self.now_ms, cause="preemption")
        return True

    def reserve_or_preempt(self, state: _RequestState, token_count: int) -> bool:
        if self.allocator.reserve(state.spec.request_id, token_count):
            self.sample_kv(self.now_ms, cause="reserve")
            return True
        if self.preempt_largest(requester=state) and self.allocator.reserve(
            state.spec.request_id, token_count
        ):
            self.sample_kv(self.now_ms, cause="reserve_after_preemption")
            return True
        return False

    def admit(self, count: int) -> list[_RequestState]:
        admitted: list[_RequestState] = []
        while self.queue and len(admitted) < count:
            state = self.queue[0]
            context_tokens = state.spec.prompt_tokens + state.generated_tokens
            required_blocks = math.ceil(context_tokens / self.engine.kv_block_size_tokens)
            if required_blocks > self.engine.kv_total_blocks:
                self.finalize(
                    state,
                    status="oom",
                    at_ms=self.now_ms,
                    reason="request_exceeds_kv_capacity",
                )
                continue
            if not self.reserve_or_preempt(state, context_tokens):
                break
            self.queue.pop(0)
            if state.queue_enter_ms is not None:
                state.queue_wait_ms += self.now_ms - state.queue_enter_ms
            if state.start_ms is None:
                state.start_ms = self.now_ms
            state.status = "active"
            self.active.append(state)
            admitted.append(state)
            self.event(
                "admitted",
                self.now_ms,
                request_id=state.spec.request_id,
                recompute=state.needs_recompute,
                context_tokens=context_tokens,
            )
            state.needs_recompute = False
        if admitted:
            total_context = sum(
                state.spec.prompt_tokens + state.generated_tokens for state in admitted
            )
            work_ms = self.engine.prefill_base_ms + (
                self.engine.prefill_token_ms * total_context / math.sqrt(len(admitted))
            )
            self.event(
                "prefill_start",
                self.now_ms,
                request_ids=[state.spec.request_id for state in admitted],
                context_tokens=total_context,
                modeled_work_ms=work_ms,
            )
            self.advance_busy(work_ms, phase="prefill", batch_slots=len(admitted))
            self.event(
                "prefill_end",
                self.now_ms,
                surviving_request_ids=[
                    state.spec.request_id for state in admitted if state.status == "active"
                ],
            )
        return admitted

    def decode_step(self) -> None:
        for state in list(self.active):
            next_tokens = state.spec.prompt_tokens + state.generated_tokens + 1
            if not self.reserve_or_preempt(state, next_tokens):
                self.finalize(
                    state,
                    status="oom",
                    at_ms=self.now_ms,
                    reason="kv_growth_allocation_failure",
                )
        participants = list(self.active)
        if not participants:
            return
        batch_slots = (
            self.static_batch_slots if self.policy.kind == "static_fcfs" else len(participants)
        )
        if self.policy.kind == "static_fcfs":
            self.static_wasted_token_slots += max(0, batch_slots - len(participants))
        work_ms = self.engine.decode_base_ms + self.engine.decode_per_sequence_ms * batch_slots
        self.event(
            "decode_step_start",
            self.now_ms,
            active_request_ids=[state.spec.request_id for state in participants],
            charged_batch_slots=batch_slots,
            modeled_work_ms=work_ms,
        )
        self.advance_busy(work_ms, phase="decode", batch_slots=batch_slots)
        for state in participants:
            if state.status != "active" or state not in self.active:
                continue
            state.generated_tokens += 1
            state.token_times_ms.append(self.now_ms)
            if state.first_token_ms is None:
                state.first_token_ms = self.now_ms
            self.event(
                "token",
                self.now_ms,
                request_id=state.spec.request_id,
                token_index=state.generated_tokens,
            )
            if state.generated_tokens >= state.spec.output_tokens:
                self.finalize(
                    state,
                    status="completed",
                    at_ms=self.now_ms,
                    reason="all_output_tokens_generated",
                )

    def next_idle_event_time(self, *, static_wait: bool) -> float:
        candidates: list[float] = []
        if self.pending_index < len(self.specs):
            candidates.append(self.specs[self.pending_index].arrival_ms)
        deadline = self.next_deadline()
        if deadline:
            candidates.append(deadline[0])
        if static_wait and self.queue:
            oldest = min(state.queue_enter_ms or state.spec.arrival_ms for state in self.queue)
            candidates.append(oldest + self.engine.static_batch_wait_ms)
        return min(candidates) if candidates else math.inf

    def run(self) -> SimulationOutput:
        self.sample_kv(self.now_ms, cause="start")
        self.process_external_until(self.now_ms)
        iterations = 0
        while True:
            iterations += 1
            if iterations > 1_000_000:
                raise RuntimeError("simulation failed to converge")
            unfinished = [
                state
                for state in self.states.values()
                if state.status in {"pending", "queued", "active"}
            ]
            if not unfinished:
                break
            if self.now_ms >= self.stop_ms:
                for state in list(unfinished):
                    self.finalize(
                        state,
                        status="aborted",
                        at_ms=self.stop_ms,
                        reason="simulation_time_limit",
                    )
                self.now_ms = self.stop_ms
                break

            if self.policy.kind == "continuous_fcfs":
                capacity = self.engine.max_batch_size - len(self.active)
                if capacity > 0 and self.queue:
                    self.admit(capacity)
            elif not self.active and self.queue:
                oldest = min(state.queue_enter_ms or state.spec.arrival_ms for state in self.queue)
                ready = (
                    len(self.queue) >= self.engine.max_batch_size
                    or self.now_ms - oldest >= self.engine.static_batch_wait_ms - 1e-9
                    or self.pending_index >= len(self.specs)
                )
                if ready:
                    admitted = self.admit(self.engine.max_batch_size)
                    self.static_batch_slots = len(admitted)

            if self.active:
                self.decode_step()
                if self.policy.kind == "static_fcfs" and not self.active:
                    self.static_batch_slots = 0
                continue

            static_wait = self.policy.kind == "static_fcfs" and bool(self.queue)
            target = self.next_idle_event_time(static_wait=static_wait)
            if math.isinf(target):
                break
            target = min(target, self.stop_ms)
            if target <= self.now_ms + 1e-9:
                # Consume simultaneous deadlines/arrivals before asking the scheduler again.
                self.process_external_until(self.now_ms)
                continue
            self.idle_ms += target - self.now_ms
            self.process_external_until(target)
            self.now_ms = target

        requests = tuple(self.request_row(state) for state in self.states.values())
        metrics = self.build_metrics(list(requests))
        causes = tuple(self.attribute_root_causes(metrics))
        return SimulationOutput(
            policy=asdict(self.policy),
            metrics=metrics,
            requests=requests,
            events=tuple(self.events),
            kv_samples=tuple(self.kv_samples),
            root_causes=causes,
        )

    def request_row(self, state: _RequestState) -> dict[str, Any]:
        ttft = (
            state.first_token_ms - state.spec.arrival_ms
            if state.first_token_ms is not None
            else None
        )
        e2e = state.end_ms - state.spec.arrival_ms if state.end_ms is not None else None
        inter_token = [
            later - earlier
            for earlier, later in zip(state.token_times_ms, state.token_times_ms[1:])
        ]
        tpot = sum(inter_token) / len(inter_token) if inter_token else None
        return {
            **state.spec.to_dict(),
            "policy": self.policy.name,
            "status": state.status,
            "terminal_reason": state.terminal_reason,
            "start_ms": state.start_ms,
            "first_token_ms": state.first_token_ms,
            "end_ms": state.end_ms,
            "generated_tokens": state.generated_tokens,
            "queue_time_ms": state.queue_wait_ms,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "itl_ms": inter_token,
            "e2e_ms": e2e,
            "preemptions": state.preemptions,
            "met_slo": state.status == "completed"
            and e2e is not None
            and e2e <= self.engine.slo_e2e_ms,
        }

    def build_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [row for row in rows if row["status"] == "completed"]
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in (
                "completed",
                "rejected",
                "timed_out",
                "cancelled",
                "oom",
                "aborted",
            )
        }
        ttft = [float(row["ttft_ms"]) for row in completed if row["ttft_ms"] is not None]
        tpot = [float(row["tpot_ms"]) for row in completed if row["tpot_ms"] is not None]
        itl = [float(value) for row in completed for value in row["itl_ms"]]
        e2e = [float(row["e2e_ms"]) for row in completed if row["e2e_ms"] is not None]
        queue = [float(row["queue_time_ms"]) for row in completed]
        start = min(spec.arrival_ms for spec in self.specs)
        finish = max((row["end_ms"] or start) for row in rows)
        makespan_ms = max(finish - start, 1e-9)
        completed_tokens = sum(int(row["generated_tokens"]) for row in completed)
        slo_rows = [row for row in completed if row["met_slo"]]
        slo_tokens = sum(int(row["generated_tokens"]) for row in slo_rows)
        throughput = completed_tokens / makespan_ms * 1_000.0
        theoretical_step = self.engine.decode_base_ms + (
            self.engine.decode_per_sequence_ms * self.engine.max_batch_size
        )
        theoretical_peak = self.engine.max_batch_size / theoretical_step * 1_000.0
        return {
            "evidence_class": "deterministic_discrete_event_simulation",
            "policy_kind": self.policy.kind,
            "preemption_policy": self.policy.preemption_policy,
            "request_count": len(rows),
            "status_counts": counts,
            "completion_rate": counts["completed"] / len(rows),
            "rejection_rate": counts["rejected"] / len(rows),
            "timeout_rate": counts["timed_out"] / len(rows),
            "cancellation_rate": counts["cancelled"] / len(rows),
            "latency_ms": {
                "ttft": _percentiles(ttft),
                "tpot": _percentiles(tpot),
                "itl": _percentiles(itl),
                "e2e": _percentiles(e2e),
                "queue": _percentiles(queue),
            },
            "completed_output_tokens": completed_tokens,
            "output_token_throughput_per_second": throughput,
            "slo_e2e_ms": self.engine.slo_e2e_ms,
            "slo_goodput_requests_per_second": len(slo_rows) / makespan_ms * 1_000.0,
            "slo_goodput_tokens_per_second": slo_tokens / makespan_ms * 1_000.0,
            "slo_attainment": len(slo_rows) / len(completed) if completed else 0.0,
            "makespan_ms": makespan_ms,
            "busy_ms": {
                "prefill": self.prefill_busy_ms,
                "decode": self.decode_busy_ms,
                "idle": self.idle_ms,
            },
            "scheduler_busy_fraction": (self.prefill_busy_ms + self.decode_busy_ms) / makespan_ms,
            "theoretical_peak_output_tokens_per_second": theoretical_peak,
            "theoretical_vs_achieved_efficiency": min(
                1.0, throughput / theoretical_peak if theoretical_peak else 0.0
            ),
            "queue_peak_requests": self.queue_peak,
            "queue_capacity": self.engine.queue_capacity,
            "static_wasted_decode_token_slots": self.static_wasted_token_slots,
            "kv": {
                "block_size_tokens": self.engine.kv_block_size_tokens,
                "total_blocks": self.engine.kv_total_blocks,
                "peak_used_blocks": self.allocator.peak_used_blocks,
                "peak_utilization": self.allocator.peak_used_blocks / self.engine.kv_total_blocks,
                "peak_internal_fragmentation_ratio": self.allocator.peak_fragmentation_ratio,
                "external_fragmentation_ratio": 0.0,
                "allocation_failures": self.allocator.allocation_failures,
            },
            "fault": {
                "enabled": self.fault.enabled,
                "impacted_operations": self.fault_impacted_operations,
                "extra_service_ms": self.fault_extra_ms,
                "service_multiplier": self.fault.service_multiplier,
            },
        }

    def attribute_root_causes(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        causes: list[dict[str, Any]] = []
        counts = metrics["status_counts"]
        queue_ratio = (
            self.queue_peak / self.engine.queue_capacity if self.engine.queue_capacity else 1.0
        )
        if counts["rejected"] or queue_ratio >= 0.8:
            causes.append(
                {
                    "cause": "queue_overload_backpressure",
                    "confidence": "high" if counts["rejected"] else "medium",
                    "evidence": {
                        "rejected_requests": counts["rejected"],
                        "queue_peak": self.queue_peak,
                        "queue_capacity": self.engine.queue_capacity,
                    },
                }
            )
        if counts["timed_out"]:
            causes.append(
                {
                    "cause": "deadline_exceeded_under_queue_or_service_delay",
                    "confidence": "high",
                    "evidence": {
                        "timed_out_requests": counts["timed_out"],
                        "queue_p99_ms": metrics["latency_ms"]["queue"]["p99"],
                        "fault_extra_service_ms": self.fault_extra_ms,
                    },
                }
            )
        if counts["oom"] or self.allocator.allocation_failures:
            causes.append(
                {
                    "cause": "kv_capacity_pressure",
                    "confidence": "high" if counts["oom"] else "medium",
                    "evidence": {
                        "oom_requests": counts["oom"],
                        "allocation_failures": self.allocator.allocation_failures,
                        "peak_kv_utilization": metrics["kv"]["peak_utilization"],
                    },
                }
            )
        if self.fault_impacted_operations:
            causes.append(
                {
                    "cause": "injected_service_slowdown",
                    "confidence": "known_injection",
                    "evidence": {
                        "impacted_operations": self.fault_impacted_operations,
                        "extra_service_ms": self.fault_extra_ms,
                        "multiplier": self.fault.service_multiplier,
                    },
                }
            )
        if self.static_wasted_token_slots:
            causes.append(
                {
                    "cause": "static_batch_head_of_line_and_padding",
                    "confidence": "high",
                    "evidence": {
                        "wasted_decode_token_slots": self.static_wasted_token_slots,
                    },
                }
            )
        if counts["cancelled"]:
            causes.append(
                {
                    "cause": "client_cancellation",
                    "confidence": "observed",
                    "evidence": {"cancelled_requests": counts["cancelled"]},
                }
            )
        return causes


def simulate_serving(
    requests: list[RequestSpec],
    *,
    engine: EngineConfig,
    fault: FaultConfig,
    policy: PolicyConfig,
) -> SimulationOutput:
    """Run one deterministic policy against an immutable request trace."""

    return _ServingSimulation(requests, engine=engine, fault=fault, policy=policy).run()
