# CPU DDP Correctness & Exact-Resume Lab

This lab tests distributed-training **semantics**, not scaling performance. It executes three
actual two-process jobs with `torch.distributed` and the Gloo backend on CPU:

1. an uninterrupted two-step DDP run;
2. a one-step run that atomically saves a checkpoint;
3. a fresh two-rank process group that resumes the checkpoint and completes step two.

The same sample schedule is replayed by a single-process global-batch reference. Float64 keeps
reduction-order error visible and small rather than hiding it behind a loose tolerance.

## Questions and invariants

### Does DDP average the same gradient as the global batch?

For world size `W` and `A` local accumulation microbatches, each local loss is divided by `A`.
The final microbatch leaves `no_sync`, so DDP averages the accumulated rank gradients:

```text
g_ddp = (1/W) sum_rank [(1/A) sum_microbatch g(rank, microbatch)]
```

The reference concatenates the same `W × A × local_batch` examples and differentiates their mean
loss once. The report compares the first averaged gradient, the first optimizer update, and final
weights tensor by tensor with absolute and relative errors.

### Is sharding deterministic and leakage-free?

The production path uses `torch.utils.data.DistributedSampler`. It calls `set_epoch(0)` before
iteration. The audit independently constructs epochs 0 and 1 and requires:

- the same `(seed, epoch, rank)` to repeat exactly;
- every sample to appear exactly once per epoch;
- no missing sample, padding duplicate, or cross-rank overlap;
- epoch 1 to have a different permutation from epoch 0.

The configured dataset size is exactly divisible by world size. This protocol deliberately fails
configuration validation instead of silently introducing `DistributedSampler` padding duplicates.

### Is resume exact?

The rank-0 checkpoint contains:

- model and optimizer state;
- Torch and Python RNG state for both ranks;
- sampler epoch, next local microbatch, and completed optimizer-step cursor;
- separate config, dataset, and combined contract SHA-256 values;
- a schema version.

It is written to a unique temporary file, flushed and `fsync`ed, atomically replaced, then bound
to a SHA-256 sidecar. Loading verifies bytes, schema, and all contracts before applying model or
optimizer state. The lab proves both negative paths:

- a changed config contract is rejected;
- a one-byte-corrupted checkpoint is rejected by SHA-256 before deserialization/state loading.

The resumed model and uninterrupted model are compared tensor by tensor. The loss sequences are
also aligned by optimizer-step id and compared.

## Workload

The committed configuration uses a deterministic float64 MLP regression problem:

| Field | Value |
|---|---:|
| world size / backend | 2 / Gloo |
| device | CPU only |
| dataset samples | 16 |
| local batch | 2 |
| accumulation steps | 2 |
| effective global batch | 8 |
| optimizer steps | 2 |
| optimizer | SGD, momentum 0.9 |

Rank 1 receives a bounded 20 ms delay immediately before the synchronized backward of step 0.
The trace records the injected delay and per-rank step durations. Because collectives propagate a
straggler's cost to peers, small observed rank-duration skew does not mean the delay was absent.
This is a controlled straggler audit, not rank-failure recovery.

### Recorded host demo — 2026-08-04

The requested demo completed in 4.50 seconds with every correctness gate true.

| Measurement | Recorded value |
|---|---:|
| gradient max absolute / relative error | `1.11e-16` / `1.38e-15` |
| first-update max absolute error | `2.78e-17` |
| final-weight max absolute error | `2.78e-17` |
| resumed-state max absolute error | `0.0` (bitwise exact) |
| resumed-loss max absolute error | `0.0` |
| epoch coverage / duplicate / missing | `16 / 0 / 0` |
| exercised no-sync microbatches | `2` per rank |
| checkpoint bytes | `22,725` |
| estimated communication | `928` bytes per rank over two synchronizations |

Both the changed-contract and corrupted-byte probes were rejected before state application. The
artifact is stored at `${FMLAB_DATA_ROOT}/artifacts/distributed/ddp-correctness`.

## Run

No network or GPU is used.

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
python scripts/run_ddp_correctness.py \
  --config configs/distributed/ddp_cpu.yaml
```

Use a separate output without changing the semantic training contract:

```bash
python scripts/run_ddp_correctness.py \
  --config configs/distributed/ddp_cpu.yaml \
  --output-dir /tmp/ddp-correctness
```

## Artifact contract

```text
ddp-correctness/
├── result.json                 # canonical gates, parity, resume, timing and claim boundary
├── provenance.json             # config, hashes, runtime, CPU/Gloo and offline declaration
├── training_trace.jsonl        # stage/rank/step/microbatch samples, losses and timing
├── resume_step_1.pt            # atomic model/optimizer/RNG/cursor checkpoint
├── resume_step_1.pt.sha256     # byte-integrity sidecar
├── sample_sharding.svg         # epoch/rank sample assignment
├── resume_equivalence.svg      # uninterrupted vs resumed loss
└── report.html                 # human-readable evidence summary
```

`result.json` is successful only when every gate passes: gradient/update/final-weight parity,
resume state/loss parity, complete deterministic sharding, exercised `no_sync`, changed-contract
rejection, corruption rejection, and observed actual two-rank CPU/Gloo execution.

Communication bytes use the ring approximation
`2 × (world_size - 1) / world_size × gradient_bytes` per rank per synchronization. Gloo may choose
a different collective algorithm, so the result explicitly marks this value as estimated rather
than measured network traffic.

## Tests

```bash
pytest -q tests/test_distributed_correctness.py
ruff check src/fmlab/distributed scripts/run_ddp_correctness.py \
  tests/test_distributed_correctness.py
```

The integration test starts six worker processes in total across the three sequential two-rank
stages. It does not replace the recorded host demo; it validates the same contract in an isolated
temporary directory.

## Claim boundary

Passing evidence supports this statement:

> A deterministic tiny float64 workload produced near-machine-precision agreement between an
> actual two-process CPU/Gloo DDP run and a single-process global-batch reference, and exact state
> agreement between uninterrupted and checkpoint-resumed two-rank runs.

It does **not** evidence NCCL/GPU collective behavior, multi-node networking, elastic rank-failure
recovery, FSDP/ZeRO/tensor/pipeline parallelism, large-model convergence, or production throughput
scaling. Reported communication bytes are an analytical estimate.
