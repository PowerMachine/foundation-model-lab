# Frontier Systems Results — 2026-08-04

This report consolidates four resource-bounded tracks selected from the official-role signal review
in [Frontier Research Engineering Alignment](frontier_role_alignment.md). Primary metrics and claim
boundaries were reconciled against each canonical runtime `result.json`; resume-store cardinality was
audited separately. Repository links point only to source, protocols, or sanitized public
derivatives. No runtime-local absolute path or workstation identity is included.

## Evidence-class ledger

| Track | Execution that actually occurred | Evidence class | Never infer from it |
| --- | --- | --- | --- |
| Visual reward environment | Deterministic synthetic visual tasks were scored against five scripted candidate types. | **Scripted synthetic environment measurement** | VLM generation quality, human preference, online RL, production safety, or rollout scale. |
| Reliable agent evaluation | Real local files, safe tools, Python tests, private grading, ledgers, retries, and reports ran; both policies were deterministic scripts. | **Actual local environment measurement with scripted controls** | LLM/coding-agent capability, OS sandbox security, network isolation, SWE-bench scale, or long-horizon production reliability. |
| Inference dynamics | Scheduler/load/KV outputs came from a deterministic discrete-event model; a separate `TinyDecoderLM` fidelity/latency probe ran on CPU. | **Simulation + actual tiny CPU probe** | vLLM/GPU/fleet capacity, packed INT kernel speed, memory reduction, or production-model quality. |
| DDP correctness | Three actual two-process CPU/Gloo jobs and one single-process reference executed. | **Actual CPU distributed correctness** | NCCL/GPU behavior, multi-node recovery, FSDP/ZeRO, large-model convergence, or throughput scaling. |

This separation is the central result: a scripted control may validate a grader, a simulator may test
a scheduling hypothesis, and an actual CPU run may establish a numerical invariant, but those facts
are not interchangeable.

## 1. Visual knowledge-work reward environment

### Method

The environment covers document VQA, chart QA, and bounding-box grounding. Each task commits to the
image bytes, prompt, hidden reference, metadata, split, schema, and fingerprint. Source-image groups
are disjoint across train and evaluation. The action is strict JSON; extra fields, malformed values,
invalid boxes, and invalid confidence fail closed. The robust reward combines type-aware answer
verification, formatting, calibration, grounding IoU where applicable, and an explicit hack penalty.

The evaluation budget contained 12 held-out task instances. Five scripted candidates were applied to
each instance—`oracle`, `wrong_overconfident`, `abstain`, `reward_hacker`, and
`extra_field_injection`—for 60 scored candidates. These candidates audit the reward implementation;
they are not VLM outputs. Training-only generation produced 24 preference pairs marked as simulated.

Dataset accounting remained source-disjoint:

| Field | Value |
| --- | ---: |
| Full train / evaluation examples | 45 / 15 |
| Train / evaluation source groups | 9 / 3 |
| Source overlap | 0 |
| Budgeted train / evaluation examples | 24 / 12 |
| Evaluation candidates | 60 |
| Preference pairs | 24 |
| Unique task fingerprints | 36 |

### Results

| Metric | Naive substring reward | Robust typed reward | Difference |
| --- | ---: | ---: | ---: |
| Scripted preference accuracy | 45.83% | 100.00% | +54.17 points |
| Pearson correlation with true success | 0.3804 | 0.6875 | +0.3070 |
| Reward-hack false acceptance | 100.00% | 0.00% | -100.00 points |

The paired robust-minus-naive preference delta used 48 pairs and 1,000 bootstrap samples: `0.5417`,
95% CI `[0.3958, 0.6875]`, with bootstrap probability of a positive delta `1.0`. Across 24 attack
challenges, the detector recorded precision `1.0`, recall `1.0`, robust false acceptance `0.0`, and
zero clean false positives. The oracle mean robust reward was `0.99575`.

The vulnerable reward gave mean reward `1.0` to both the reward-hacker and extra-field-injection
candidates even though their true success rate was zero. The robust reward gave both a mean of
`-0.97222`, making the negative control visible rather than hiding it behind aggregate accuracy.

### Failure controls and claim boundary

- Attack controls: reward-string manipulation and unexpected JSON-field injection.
- Behavioral controls: overconfident wrong answers and abstention.
- Leakage control: split by source image; reference and hidden metadata absent from observations.
- Lineage control: preferences only from train; attack audit remains held out; simulated teachers are
  labeled.

This supports a claim about deterministic environment/reward behavior on synthetic visual tasks. It
does not support a claim about VLM generation, learned policy improvement, online RL, human
preference agreement, or production safety.

Evidence: [sanitized result](../public-evidence/vlm/visual-reward-environment/result.json),
[HTML report](../public-evidence/vlm/visual-reward-environment/report.html),
[environment contract](../public-evidence/vlm/visual-reward-environment/environment_spec.json), and
[evidence manifest](../public-evidence/vlm/visual-reward-environment/evidence_manifest.json).
Implementation: [reward environment](../src/fmlab/vlm/reward_environment.py),
[configuration](../configs/vlm/visual_reward_environment.yaml), and
[focused tests](../tests/test_vlm_reward_environment.py).

## 2. Reliable agent evaluation environment

### Method

Six Python bug-fix families were each emitted as a canonical and renamed-module perturbation, giving
12 immutable, SHA-256-addressed tasks. A policy observed only the instruction and public files, used
an allowlisted action interface, and finished before hidden grader assets were materialized in a
separate temporary root. A pass required all hidden cases, current unchanged public tests after the
last source edit, sequential trace integrity, and no reward-hacking signal.

Two deterministic controls ran across every task:

- `oracle_patch_scripted` used the declared reference patch to validate the harness upper bound.
- `shortcut_probe_scripted` intentionally exercised false success, hidden-path access, path escape,
  unsafe tools, unvalidated edits, and public-test deletion/tampering.

The controls are trusted in-process harness code. Hidden values are excluded from the model-facing
observation and workspace; they are not claimed to be isolated from arbitrary Python introspection
inside that trusted process.

### Results

| Scope | Episodes | Integrity pass@1 | Raw hidden outcome | Reward-hack rate |
| --- | ---: | ---: | ---: | ---: |
| Overall | 24 | 50.00% | 58.33% | 50.00% |
| Oracle harness control | 12 | 100.00% | 100.00% | 0.00% |
| Shortcut negative control | 12 | 0.00% | 16.67% | 100.00% |

Overall pass@1 had a seeded 95% bootstrap interval `[29.17%, 70.83%]`. The gap between raw outcome
pass (`58.33%`) and integrity-qualified pass (`50.00%`) is intentional: two shortcut episodes reached
the hidden answer but failed the required process validation. Across 74 tool calls, six returned
errors and six were classified as unsafe attempts.

Recorded reward-hack signals were:

| Signal | Episodes/signals |
| --- | ---: |
| Success forgery | 12 |
| Path escape attempt | 4 |
| Hidden-grader access attempt | 2 |
| Public-test deletion | 2 |
| Public-test tampering | 2 |
| Unsafe-tool attempt | 2 |

Five deterministic transient infrastructure faults were injected. Every affected episode completed
within the one-retry budget, with a maximum of two attempts. An identical rerun executed zero new
episodes and resumed all 24; the ledger and trace store remained at 24 unique rows each.

### Failure controls and claim boundary

- Outcome/process separation prevents a forged public test from becoming a pass.
- Post-hoc hashes detect delete-and-recreate public-test tampering.
- Workspace resolution records and blocks path escape; non-allowlisted tools fail closed.
- Atomic, hash-verified JSONL stores and content-addressed episode contracts make resume auditable.
- Fault injection tests retry machinery without granting a policy another scored attempt.

The run used process-level resource/path controls with `network_isolated=false`; it is not an OS
security boundary. Model inference was false. This is evidence about evaluator, grader, retry,
integrity, and report behavior under scripted controls—not coding-agent quality.

Evidence: [sanitized result](../public-evidence/agent/reliable-agent-benchmark/result.json),
[HTML report](../public-evidence/agent/reliable-agent-benchmark/report.html),
[task commitments](../public-evidence/agent/reliable-agent-benchmark/task_manifest.json), and
[evidence manifest](../public-evidence/agent/reliable-agent-benchmark/evidence_manifest.json).
Implementation: [environment](../src/fmlab/agent/eval_environment.py),
[runner](../src/fmlab/agent/benchmark.py), [configuration](../configs/agent/reliable_agent_benchmark.yaml),
and [focused tests](../tests/test_agent_benchmark.py).

## 3. Inference dynamics

### Evidence split and method

The main system study is deterministic discrete-event simulation. Static FCFS, continuous FCFS, and
continuous FCFS with largest-context preemption replay the same request specifications. The model
tracks queue admission, deadlines, cancellation, decode progress, fixed-block paged KV, internal
fragmentation, allocation failures, preemption, and terminal states. Separately, a real tiny decoder
runs on CPU to test FP32 versus fake-quantized weight fidelity and latency.

The single-trace experiment includes a `3.0x` service slowdown fault. The capacity sweep disables
that fault so nominal load, scheduler, and KV effects are not confounded with the injected slowdown.

### Single-trace scheduler and fault comparison

Relative to static FCFS, continuous FCFS changed modeled output-token throughput by `+65.38%`, p99
end-to-end latency by `-9.96%` (`361.924 ms` to `325.860 ms`), and SLO token goodput by `+218.02%`.
The static scheduler recorded 481 wasted decode-token slots, supporting the configured
head-of-line/padding attribution.

The slowdown affected 8 operations for static FCFS and 10 for each continuous policy, adding about
`106.667 ms` of modeled service time per policy. All policies reached all 36 modeled KV blocks; the
single trace recorded 2, 64, and 95 allocation failures for static, continuous, and
continuous-with-preemption respectively. These counts describe this configured allocator/scheduler
interaction; they are not GPU allocator measurements.

### Capacity sweep and knees

The latest capacity study used:

- 9 configured Poisson background rates: `4, 8, 12, 18, 24, 32, 48, 72, 96` requests/s;
- 3 fixed seeds: `1103, 2207, 3301`;
- 96 requests per unique trace;
- 27 paired traces (`9 rates × 3 seeds`), each replayed across 3 policies;
- 81 policy trials and 27 policy-by-rate aggregates; and
- mean/min/max/population-standard-deviation uncertainty across seeds.

The sustainable-prefix rule stops at the first across-seed mean threshold violation. It requires
completion ratio at least `0.90`, SLO attainment at least `0.90`, and terminal-failure ratio at most
`0.10`; a later stochastic recovery cannot move the knee upward.

| Policy | Last sustainable configured rate | First unsustainable configured rate | Realized offered-load mean interval | First failed criterion |
| --- | ---: | ---: | ---: | --- |
| Static FCFS | 24 | 32 | 29.858–39.785 requests/s | SLO attainment |
| Continuous FCFS | 32 | 48 | 39.785–59.603 requests/s | Completion ratio; terminal-failure ratio |
| Continuous FCFS + preempt largest | 32 | 48 | 39.785–59.603 requests/s | SLO attainment |

These are observed intervals under the configured service and KV model, not measured server
capacity. The realized offered rate differs from the configured Poisson component because the trace
inherits mixed/burst request semantics.

### Actual tiny CPU probe

| Variant | Top-1 agreement | Cosine similarity | KL divergence | CPU latency p50 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1.0000 | 1.000000 | 0 | 0.290256 ms |
| Fake INT8 | 1.0000 | 0.999971 | `3.31e-7` | 0.291368 ms |
| Fake INT4 | 1.0000 | 0.991100 | `1.19e-4` | 0.288552 ms |

All configured fidelity/latency gates passed. The INT8/INT4 copies dequantize weights and execute
ordinary FP32 PyTorch kernels, so the small latency differences do not evidence packed-integer
kernel speed, reduced runtime memory, GPU performance, or production-model accuracy.

### Claim boundary

The simulation establishes behavior of the implemented queue/scheduler/KV/service model and the
actual probe establishes tiny-model CPU fidelity under fake quantization. Neither establishes vLLM,
TensorRT-LLM, CUDA-kernel, real prefix-cache, speculative-decoding, distributed-serving, fleet SLO,
or hardware-efficiency performance.

Evidence: [sanitized result](../public-evidence/systems/inference-dynamics/result.json),
[HTML report](../public-evidence/systems/inference-dynamics/report.html),
[actual CPU probe](../public-evidence/systems/inference-dynamics/actual_probe.json),
[capacity sweep](../public-evidence/systems/inference-dynamics/capacity_sweep.json), and
[evidence manifest](../public-evidence/systems/inference-dynamics/evidence_manifest.json).
Implementation: [protocol](inference_dynamics.md),
[configuration](../configs/systems/inference_dynamics.yaml),
[simulator](../src/fmlab/systems/simulator.py), [CPU probe](../src/fmlab/systems/probe.py),
[capacity logic](../src/fmlab/systems/capacity.py), and
[focused tests](../tests/test_inference_dynamics.py).

## 4. CPU/Gloo DDP correctness and exact resume

### Method

The experiment ran three actual two-process CPU/Gloo stages:

1. an uninterrupted two-optimizer-step DDP job;
2. a one-step DDP job that atomically saved model, optimizer, per-rank RNG, sampler, and cursor state;
3. a fresh two-rank process group that verified and resumed the checkpoint through step two.

A single-process reference differentiated the same effective global batches. The workload used
float64, two ranks, 16 samples, local batch 2, two accumulation microbatches, effective global batch
8, two optimizer steps, SGD with momentum `0.9`, and explicit `DistributedSampler.set_epoch` calls.
The recorded run completed in `4.503 s`.

### Results

| Correctness measurement | Recorded value |
| --- | ---: |
| Gradient max absolute / relative error | `1.1102e-16` / `1.3796e-15` |
| First-update max absolute error | `2.7756e-17` |
| Final-weight max absolute error | `2.7756e-17` |
| Resumed-state max absolute error | `0.0` (exact tensor equality) |
| Resumed-loss max absolute error | `0.0` |
| Coverage / duplicates / missing per audited epoch | `16 / 0 / 0` |
| `no_sync` microbatches | `2` per rank |
| Checkpoint size | `22,725` bytes |
| Estimated communication | `928` bytes/rank over two synchronizations |

Every result gate passed: gradient, first update, final weight, resume state, resume loss, sharding,
`set_epoch`, `no_sync`, checkpoint contract, checkpoint integrity, and observed actual two-rank Gloo.
Epochs 0 and 1 were deterministic, complete, overlap-free, and used different permutations.

### Failure controls

- A 20 ms delay was injected on rank 1 immediately before the synchronized backward of optimizer
  step 0. The maximum observed rank step-duration skew was `0.000115 s`; synchronization propagates a
  straggler's cost, so low skew is not interpreted as absence of the injected delay.
- A changed config contract was rejected before applying state.
- A one-byte checkpoint corruption was rejected by SHA-256 before deserialization/state loading.
- Atomic replace plus a checksum sidecar guarded checkpoint publication.
- The communication number is the documented ring analytical estimate; measured network bytes were
  false.

### Claim boundary

This run evidences near-machine-precision equivalence between an actual tiny two-process CPU/Gloo DDP
workload and a single-process global-batch reference, plus exact uninterrupted-vs-resumed state and
loss equivalence. It does not evidence NCCL/GPU collective behavior, multi-node networking, elastic
rank-failure recovery, FSDP, ZeRO, tensor/pipeline parallelism, large-model convergence, production
throughput scaling, or measured communication traffic.

Evidence: [sanitized result](../public-evidence/distributed/ddp-correctness/result.json),
[HTML report](../public-evidence/distributed/ddp-correctness/report.html),
[resume figure](../public-evidence/distributed/ddp-correctness/resume_equivalence.svg),
[sharding figure](../public-evidence/distributed/ddp-correctness/sample_sharding.svg), and
[evidence manifest](../public-evidence/distributed/ddp-correctness/evidence_manifest.json).
Implementation: [experiment](../src/fmlab/distributed/experiment.py),
[checkpoint contract](../src/fmlab/distributed/checkpoint.py),
[configuration](../configs/distributed/ddp_cpu.yaml), and
[focused tests](../tests/test_distributed_correctness.py).

## Cross-track interpretation

The four tracks form one research-engineering loop:

```text
typed multimodal tasks and reward contracts
                  ↓
hidden outcome/process evaluation and attack controls
                  ↓
capacity, latency, KV, and correctness diagnosis
                  ↓
distributed state integrity and exact recovery
                  ↓
sanitized, hash-linked public evidence with explicit boundaries
```

The strongest actual claims are deliberately narrow: tiny CPU quantization fidelity and two-rank
CPU/Gloo correctness. The environment claims are broader in engineering surface but use scripted
policies/candidates. The serving study explores more operating points but remains simulated. A
production-oriented next step must preserve those labels while adding real VLM optimization, real
model agent policies, vLLM/SGLang trace replay, and NCCL evidence.

Public artifacts were produced through the [sanitized evidence exporter](public_evidence.md). Its
allowlist, redaction, staging, and hash manifest are release controls, not a general DLP or legal
review guarantee.
