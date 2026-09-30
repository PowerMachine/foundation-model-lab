# Inference Dynamics Lab

This lab turns LLM serving-system concepts into a deterministic, inspectable experiment. It compares FCFS static batching, FCFS continuous batching, and continuous batching with KV-pressure preemption. It also runs a small real CPU correctness probe with the repository's `TinyDecoderLM`.

The lab is deliberately resource bounded. The serving results are produced by a discrete-event simulator; only the tiny-model PyTorch probe is an actual runtime measurement. The report and machine-readable outputs preserve that distinction.

## Questions answered

1. How do static and continuous batching change queueing, token throughput, tail latency, and SLO goodput under the same arrival trace?
2. When does a bounded request queue reject work, and how do deadlines and cancellations propagate through the lifecycle?
3. How does a fixed-size paged-KV pool expose block utilization, internal fragmentation, allocation failures, OOMs, and preemption trade-offs?
4. Can an injected service slowdown be separated from overload and KV-capacity root causes?
5. How closely do fake INT8 and INT4 weights preserve the FP32 logits of a real tiny decoder on CPU?

## Architecture

The execution path is:

`strict YAML config -> deterministic request trace -> policy simulations -> CPU fidelity probe -> JSON/JSONL + SVG + HTML report`

The main implementation is split into:

- `src/fmlab/systems/config.py`: strict dataclass configuration and validation.
- `src/fmlab/systems/simulator.py`: trace generation, discrete-event schedulers, request lifecycle, paged-KV allocator, and root-cause attribution.
- `src/fmlab/systems/probe.py`: real CPU FP32/fake-INT8/fake-INT4 TinyDecoder measurements and correctness gates.
- `src/fmlab/systems/reporting.py`: dependency-light SVG plots and a self-contained HTML report.
- `src/fmlab/systems/runner.py`: artifact orchestration and cross-policy comparison.

## Reproducible experiment

From the repository root:

```bash
PYTHONPATH=src python scripts/run_inference_dynamics.py \
  --config configs/systems/inference_dynamics.yaml
```

The default run writes to:

```text
${FMLAB_DATA_ROOT}/artifacts/systems/inference-dynamics
```

The experiment is deterministic at the workload and simulation layers: it uses an explicit seed, one generated trace shared by every policy, and no wall-clock time in the simulator. CPU latency samples are real measurements and therefore are expected to vary by host load and PyTorch build.

## Workload and lifecycle

Each request contains an arrival time, prompt-token count, requested output-token count, absolute deadline, and an optional cancellation time. Trace modes are:

- `poisson`: exponential inter-arrival times.
- `burst`: compact request groups separated by Poisson-like idle intervals.
- `mixed`: background Poisson traffic with deterministic burst injection.

The bounded queue implements admission backpressure. Requests can finish as `completed`, `rejected`, `timed_out`, `cancelled`, or `oom`. Per-request JSONL records retain arrival, admission, first-token, finish, queueing, preemption, and terminal-reason evidence.

## Scheduler ablations

`static_fcfs` forms a FCFS batch and runs it to completion. Decode work uses the longest requested output length in the batch, so shorter members expose padded/wasted decode-token slots.

`continuous_fcfs` admits queued requests between decode iterations as slots become available. It avoids static decode padding and makes head-of-line and KV-capacity effects observable without changing the arrival trace.

`continuous_fcfs_preempt` adds a bounded `preempt_largest` policy. On KV allocation pressure, the largest active context can be released and requeued. This is an explicit systems ablation: it may improve admission or make progress under pressure, but it can also increase queueing, recomputation proxies, or terminal failures. The non-preempting continuous policy is retained as the clean scheduler baseline.

## Paged-KV model

The allocator uses fixed-size, non-contiguous token blocks:

```text
blocks(request) = ceil(current_context_tokens / block_size_tokens)
```

It reports allocated blocks, useful tokens, reserved capacity, internal-fragmentation tokens and ratio, allocation failures, and peak utilization. External fragmentation is explicitly zero in this abstraction because any free block can satisfy any request; it is not presented as a measurement of a concrete CUDA allocator. Exhaustion produces an OOM or triggers the configured preemption policy.

## Metrics

The simulator exports p50/p95/p99 for:

- TTFT: arrival to first generated token.
- TPOT: generation span divided by post-first-token output intervals.
- ITL: modeled inter-token latency samples.
- E2E latency: arrival to completion.
- Queue time: arrival to first admission.

It also reports output-token throughput, total-token throughput, SLO goodput, request goodput, rejection/timeout/cancellation/OOM rates, busy-time decomposition, static-padding waste, and modeled efficiency. SLO goodput counts output tokens only from requests completed within the configured E2E SLO.

`pareto.svg` visualizes p99 E2E latency against output-token throughput, `slo.svg` compares SLO goodput and terminal counts, and `kv.svg` shows the time-varying block utilization for each policy.

## Fault injection and attribution

The fault window multiplies modeled prefill/decode service time while leaving arrivals unchanged. The output records impacted operations and added service milliseconds. Root-cause attribution is rule-based and evidence-linked:

- queue saturation and rejection imply overload/backpressure;
- KV allocation failures, OOMs, and preemptions imply KV-capacity pressure;
- service time added inside the injected window implies injected slowdown;
- deadline misses are reported separately so they are not silently folded into generic failure.

This is causal instrumentation of the configured model, not a claim that the heuristic diagnoses arbitrary production incidents.

## Actual CPU correctness probe

The probe instantiates the existing `TinyDecoderLM`, fixes the seed and input tokens, and measures FP32 plus fake-quantized INT8/INT4 weight copies. It reports top-1 agreement, cosine similarity, KL divergence, logit error, latency samples, and pass/fail thresholds.

The quantized copies dequantize weights and execute ordinary PyTorch FP32 CPU kernels. Consequently:

- fidelity results are actual tiny-model measurements;
- latency numbers are actual CPU wall-clock measurements;
- they are **not** evidence of packed INT8/INT4 kernel speedup, reduced runtime memory, GPU serving performance, or production-model quality.

The overall correctness gate fails the experiment if either fake-quantized variant violates its configured fidelity or latency guardrail.

## Capacity and load sweep

The optional `capacity_sweep` section runs a nominal-capacity study without replacing the original single-trace fault experiment. For every configured Poisson background rate and fixed seed, one request trace is generated and replayed unchanged across all scheduler policies. Prompt/output distributions, mixed/burst structure, deadlines, and cancellation semantics are inherited from `trace`; only the request count and Poisson-rate parameter change.

Fault injection is disabled inside this sweep to isolate the configured scheduler/service/KV model. Each seed-level row preserves configured rate, realized offered requests/s, completed requests/s, E2E p99, SLO attainment, completion ratio, SLO token goodput, rejection/timeout/OOM rates, KV peak utilization, and allocation-failure pressure. Aggregates report mean, minimum, maximum, and population standard deviation across at least three fixed seeds.

The sustainable-load knee uses `prefix_first_mean_threshold_violation_v1`:

1. Sort configured rates from low to high.
2. A rate is sustainable when the across-seed mean completion ratio and SLO attainment meet their configured minima and the mean rejection+timeout+OOM+abort ratio stays below its configured maximum.
3. The first violation ends the sustainable prefix. A later stochastic recovery cannot move the knee upward.
4. Report the interval between the last sustainable and first unsustainable rates. If every rate passes, the result is right-censored above the tested range; if the first rate fails, capacity is below the tested range.

`capacity_sweep.json` retains all seed rows, aggregate uncertainty, pairing hashes, the exact decision rule, and per-policy knees. `capacity.svg` plots realized offered versus completed load and SLO attainment with seed ranges. These are **simulated capacity results**, not measurements of vLLM, a GPU, or a production deployment.

## Artifacts

- `result.json`: top-level `claim_level`, mixed `evidence_classes`, evidenced/not-evidenced boundary, status, comparisons, and artifact manifest.
- `workload.json`: generated request specifications.
- `simulation.json`: policy metrics, root causes, and claim boundary.
- `trace.jsonl`: ordered scheduler and fault events.
- `requests.jsonl`: per-request lifecycle records.
- `kv_trace.jsonl`: time-varying KV allocator snapshots.
- `actual_probe.json`: real CPU fidelity, latency samples, and gates.
- `capacity_sweep.json`: paired seed rows, uncertainty summaries, thresholds, and modeled knees.
- `capacity.svg`: modeled offered/completed load and SLO-attainment curves with seed ranges.
- `pareto.svg`, `slo.svg`, `kv.svg`: visual evidence.
- `report.html`: portable, self-contained experiment report.

## Claim boundary and next steps

The experiment demonstrates scheduler mechanics, measurement design, deterministic overload reproduction, resource accounting, and correctness-gated quantization on a small model. It does not benchmark vLLM, TensorRT-LLM, CUDA kernels, real prefix caching, distributed tensor/pipeline parallelism, or hardware energy use.

Natural production-oriented extensions are trace replay from a real gateway, a vLLM/TGI benchmark adapter, prefix-cache policies, chunked prefill, speculative decoding, multi-GPU topology models, OpenTelemetry export, and bootstrap confidence intervals over repeated real-runtime trials. These should be added as separate evidence classes instead of re-labeling simulated outputs as measurements.
