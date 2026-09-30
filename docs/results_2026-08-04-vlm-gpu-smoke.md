# Actual Qwen3-VL GPU Smoke — 2026-08-04

## Outcome

A local Qwen3-VL-8B-Instruct BF16 model completed two real LoRA optimizer steps on one
RTX PRO 6000 Blackwell GPU. This promotes Track A from wiring-only evidence to a bounded
**actual GPU smoke**. It does not promote the run to an experiment or benchmark.

The separate Qwen3-VL-32B-Instruct-FP8 teacher attempt passed its resource gate and allocated the
model, but failed in the selected FP8 kernel before generating the first response. It produced zero
teacher rows, so no real distillation audit or student run followed.

## Claim boundary

This report evidences:

- actual local BF16 loading of all `8,774,791,408` Qwen3-VL-8B parameters;
- a frozen vision tower and `7,667,712` trainable LoRA parameters (`0.08738%`) on the text
  attention `q/k/v/o` projections;
- assistant-only supervision, two optimizer steps, GPU memory, timing, and grouped greedy
  before/after evaluation;
- a real 32B resource/model-load preflight followed by an FP8-kernel compatibility failure.

It does **not** evidence:

- generalized VLM quality, adaptation improvement, or convergence;
- an inference-speed improvement;
- multi-seed variance, a public benchmark, human preference, or online RL;
- successful 32B teacher generation, an audited real teacher cache, or 32B→8B distillation quality.

## 8B run contract

| Item | Value |
|---|---|
| Evidence class | actual GPU smoke |
| Backend | local, `simulated: false` |
| Model precision | BF16 |
| Vision encoder | frozen |
| LoRA scope | text attention `q_proj/k_proj/v_proj/o_proj` |
| LoRA rank / alpha | 8 / 16 |
| Manifest | 60 synthetic visual QA examples |
| Split | image-SHA grouped, 45 train / 15 evaluation, 9 / 3 image groups, zero overlap |
| Bounded subset | 16 train / 3 evaluation; chart, document, and grounding all represented |
| Optimizer budget | 2 steps |
| Adapter persistence | disabled |

The implementation and bounded configuration are
[`lora_e2e.py`](../src/fmlab/vlm/lora_e2e.py),
[`run_vlm_lora.py`](../scripts/run_vlm_lora.py), and
[`qwen3_vl_8b_lora_e2e.yaml`](../configs/vlm/qwen3_vl_8b_lora_e2e.yaml).

## Measured execution

| Metric | Measured value |
|---|---:|
| Total parameters | 8,774,791,408 |
| Trainable LoRA parameters | 7,667,712 (0.08738%) |
| Completed optimizer steps | 2 |
| Supervised assistant tokens | 44 |
| Supervised throughput | 30.28 tokens/s |
| Model load | 34.26 s |
| Training steps | 1.45 s |
| Total wall time | 36.75 s |
| Peak allocated memory during training | 17.45 GiB |
| Peak reserved memory during training | 18.52 GiB |

The run used adapter persistence off, so these values do not imply an exported or deployed adapter.

## Paired holdout result

The frozen bounded holdout contains one image-group-disjoint synthetic example for each task:
chart QA, document VQA, and grounding.

| Metric | Before | After | Paired delta |
|---|---:|---:|---:|
| Exact match | 1.000 | 1.000 | 0.000 |
| ANLS | 1.000 | 1.000 | 0.000 |
| Type-aware accuracy | 1.000 | 1.000 | 0.000 |

The exact-match paired bootstrap interval was `[0,0]` over three pairs. The base model was already
perfect on this tiny synthetic subset, so the run cannot demonstrate quality improvement. The
before/after mean latencies were collected in a fixed execution order; model warm-up and cache
effects are not separated. Their difference is therefore not an adaptation-speedup result.

The canonical sanitized artifacts are:

- [result JSON](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/result.json)
- [human-readable report](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/report.html)
- [redacted provenance](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/provenance.json)
- [evidence manifest](../public-evidence/vlm/qwen3-vl-8b-lora-real-2step/evidence_manifest.json)

The manifest records per-file source/public SHA-256 values, 27 redactions, an allowlist, and the
explicit unsupported-claim list. Weights, adapters, optimizer state, prediction JSONL, and local
machine paths are not published.

## 32B teacher negative result

A later actual Qwen3-VL-32B-Instruct-FP8 attempt ran with the GPU free:

| Stage | Result |
|---|---|
| 48 GiB free-VRAM gate | passed |
| FP8 metadata/runtime preflight | passed |
| Model allocation | completed, about 32.6 GiB allocated |
| First greedy generation | failed |
| Failure | Transformers-selected `kernels-community/finegrained-fp8` v1 raised `Unknown recipe` |
| Teacher cache | 0 rows |
| Real cache audit / student training | not run |

The failure occurred on the Blackwell runtime after model allocation, not at the earlier
insufficient-resource gate. The latest v4 kernel exposes a changed API and is not a drop-in
replacement for the Transformers integration used by this environment. No compatibility workaround
is represented as a completed result.

This raw failure provenance remains outside the public repository because it contains local runtime
identity and paths. There is intentionally no public bundle for the failed 32B attempt.

## Reproduce the bounded 8B smoke

Prepare the canonical manifest first, verify that the configured GPU has at least 24 GiB free, and
run:

```bash
source .venv/bin/activate
source scripts/env.sh
python scripts/run_vlm_lora.py \
  --config configs/vlm/qwen3_vl_8b_lora_e2e.yaml \
  --backend local \
  --max-steps 2 \
  --artifact-dir "$FMLAB_DATA_ROOT/artifacts/vlm/qwen3-vl-8b-lora-local-smoke"
```

The default does not persist an adapter. Add `--save-adapter` only when adapter licensing, storage,
and release handling have been reviewed.

## Promotion path

A stronger claim requires a releasable non-synthetic visual dataset, the registered image-grouped
split, identical decoding and pixel budgets, zero-shot and gold-LoRA baselines, at least three seeds,
paired intervals and task slices, plus failure cases, VRAM, GPU-minutes, and adapter bytes. Real
32B→8B distillation additionally requires a compatible FP8 runtime, non-empty hash-pinned teacher
cache, audit acceptance evidence, and a separately executed student comparison.
