# VLM evidence report — 2026-08-03

This report freezes the evidence boundary for two low-compute multimodal projects:

1. gold-label Qwen3-VL-8B LoRA with a frozen vision encoder;
2. sequence-level response distillation from Qwen3-VL-32B-Instruct-FP8 to an 8B LoRA student.

It separates implementation evidence, real preprocessing, resource preflight, actual model
execution, and quality evaluation. The presence of code is not presented as a training result.

## Executive result

Both VLM systems are implemented end to end and validated at **Wiring** level. Real local
Qwen3-VL preprocessing also passed at `real_processor_wiring` level. The host had only about
6.4 GiB free VRAM because another service occupied the GPU, so both bounded heavy paths stopped
cleanly before model loading:

- 8B LoRA required 24 GiB free;
- 32B teacher generation required 48 GiB free.

No real 8B optimizer step or real 32B teacher inference is claimed. A previously recorded
single-image Qwen3-VL-8B BF16 inference remains a **Smoke** result, not a dataset-level result.

## Verification snapshot

| Check | Result |
|---|---|
| Test suite | **96 passed** |
| Ruff lint | clean |
| Ruff formatting check | 94 files formatted |
| Network/model download | none; local paths only |
| Real 8B LoRA model load | not reached; resource gate blocked first |
| Real 32B teacher model load | not reached; resource gate blocked first |

The VLM unit and integration tests use tiny/mock backends where appropriate, so CI does not
require a GPU, private model store, or network access. The real processor check is opt-in host
validation.

## Evidence ledger

| Evidence | Claim level | Simulated | Model weights loaded | Interpretation |
|---|---|---:|---:|---|
| Offline document/chart/grounding demo | Wiring | yes | no | data, metrics, and reports are connected |
| Real Qwen3VLProcessor check | `real_processor_wiring` | no | no | local image processing, batching, and labels are correct |
| Tiny LoRA end-to-end run | Wiring | yes | tiny only | optimizer, split, evaluation, provenance, and report paths work |
| Full response-distillation mock | Wiring | yes | no | cache, resume/audit, student, and report orchestration works |
| 8B LoRA resource preflight | Resource evidence | no | no | workload was safely rejected under GPU contention |
| 32B teacher resource preflight | Resource evidence | no | no | FP8 metadata/runtime prerequisites were inspected; workload was rejected |
| Existing single-image 8B inference | Smoke | no | yes | one bounded inference worked on this host |
| Held-out evaluator implementation | Wiring | test fixtures | no | aligned scoring and failure analysis work; no real comparison yet |

## Canonical dataset and leakage control

The source manifests contain nested document/chart QA and atomic grounding rows. The common
loader expands them to 60 atomic multimodal examples.

| Task | Examples |
|---|---:|
| Document VQA | 24 |
| Chart QA | 20 |
| Grounding | 16 |
| **Total** | **60** |

Split identity is the SHA-256 of image bytes, not a manifest row id or path. This keeps all
questions about identical pixels on one side of the split even if paths or row ids differ.

The final task-aware split was:

| Partition | Examples | Image groups | Chart | Document | Grounding |
|---|---:|---:|---:|---:|---:|
| Full train | 45 | 9 | 15 | 18 | 12 |
| Full evaluation | 15 | 3 | 5 | 6 | 4 |
| Bounded LoRA train subset | 16 | grouped subset | 6 | 5 | 5 |
| Bounded LoRA evaluation subset | 3 | grouped subset | 1 | 1 | 1 |

Image overlap was empty and no required task was missing from either bounded subset.

## Real processor evidence

The host loaded the real local `Qwen3VLProcessor`, two canonical images, the multimodal chat
template, and the production assistant-only collator. Model weights were deliberately not loaded.

| Measurement | Value |
|---|---|
| Examples | `document-000-invoice_id`, `document-000-date` |
| `input_ids` shape | `[2, 403]` |
| `pixel_values` shape | `[2992, 1536]` |
| `image_grid_thw` | `[1, 44, 34]` for both rows |
| Merged visual tokens | 374 per row |
| Sequence tokens | 403 / 401 |
| Supervised assistant tokens | 13 / 12 |
| Supervised padding violations | 0 |
| Duration | 0.215 s |

The local chat template did not supply a usable assistant-token mask. The collator therefore
tokenizes prompt-only and complete conversations, verifies that the former is an exact prefix,
and masks that prefix plus padding. Any mismatch or over-budget sequence fails instead of silently
truncating image or answer tokens.

Artifact directory on the validation host:

```text
${FMLAB_DATA_ROOT}/artifacts/vlm/qwen3-vl-real-processor-check
```

## Track A — Qwen3-VL-8B LoRA

### Implemented contract

- local-only Qwen3-VL loading;
- explicit CUDA device, with `device_map=auto` forbidden for training;
- frozen vision encoder;
- rank-8, alpha-16 PEFT LoRA on language `q/k/v/o` projections;
- target-scope and trainable-parameter audit;
- assistant-suffix-only cross entropy;
- gradient accumulation, clipping, a cooperative training-loop time budget, and a hard step cap;
- input pixels limited to 65,536–401,408;
- before/after generation for a real backend and teacher-forced proxy for the tiny backend;
- paired per-example output, bootstrap summary, parameter counts, runtime, VRAM, JSON and HTML;
- adapter persistence disabled by default.

### Final wiring run

The tiny backend completed three optimizer steps on the bounded, task-aware subsets. Its final
fabricated tiny loss was `5.525444746`. This value is **not Qwen3-VL loss**, is not generation
quality, and supports only the statement that the optimizer and reporting paths are connected.

Artifact directory:

```text
${FMLAB_DATA_ROOT}/artifacts/vlm/qwen3-vl-8b-lora-mock-task-aware
```

### Real resource preflight

The local backend requested one explicit GPU and a minimum of 24 GiB free VRAM. Approximately
6.4 GiB was free. The run returned `blocked_resource` before model load and exited cleanly while
retaining `result.json`, provenance, and HTML report evidence. This is expected external resource
contention, not an 8B training failure.

No real 8B forward/backward or optimizer step was reached.

## Track B — 32B to 8B response distillation

### Implemented contract

The stages are deliberately separate operating-system processes:

1. teacher response cache;
2. CPU cache audit;
3. student LoRA training;
4. separate model-generation jobs and CPU held-out evaluation.

The teacher cache is bound to a SHA-256 contract over the manifest, model metadata, generation
settings, pixel budget, and image-grouped split. Resume requires both the same contract and a
matching previous cache hash. Rows preserve the image SHA, split, latency, backend, timestamp, and
an exact-boolean `teacher_simulated` field.

The audit verifies raw-cache provenance and rejects invalid simulated flags, simulated real-run
targets, missing or changed images, duplicate inputs, invalid splits, train/eval overlap,
placeholders, refusals, answer-length violations, and reference leakage. Rejections retain their
reason. Teacher/reference disagreement is reported but is not silently used as a selection
filter.

Before student loading, the accepted-cache SHA and row count must match the audit. Image hashes and
split isolation are verified again, and only `split=train` rows can enter training.

### Full mock result

| Measurement | Value |
|---|---:|
| Raw simulated teacher rows | 60 |
| Accepted rows | 60 |
| Train / evaluation | 45 / 15 |
| Train / evaluation image groups | 9 / 3 |
| Image overlap | 0 |
| Document / chart / grounding | 24 / 20 / 16 |
| Mock student steps | 5 |
| Mock loss | `2.052309 → 1.115530` |
| `quality_evaluated` | `false` |

The loss is fabricated by the mock path and only validates orchestration.

Artifact directory:

```text
${FMLAB_DATA_ROOT}/artifacts/vlm/response-distillation-mock-full
```

### 32B resource preflight

| Field | Recorded value |
|---|---|
| Device | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Compute capability | 12.0 |
| Config quantization | FP8 E4M3 / dynamic |
| Runtime quantizer | available |
| Required free VRAM | 48 GiB |
| Observed free VRAM | approximately 6.38 GiB |
| Status | `blocked_resource` |
| Model load | not reached |

The preflight completed before loading the 32B model. It does not prove Transformers inference
compatibility; the one-request preflight still has to pass when capacity is available. No teacher
responses were generated by the real 32B model in this run.

## Held-out evaluation contract

The dependency-light evaluator consumes aligned JSONL predictions from:

- base 8B;
- gold-label 8B LoRA;
- distilled 8B LoRA;
- optionally the 32B teacher.

It fails closed on missing/duplicate ids, reference/task/type/source mismatches, or overlap with a
provided training manifest. It reports exact match, ANLS, type-aware accuracy, task slices,
paired-bootstrap deltas with 95% intervals, probability of positive delta, teacher-gap recovery,
and a direction-aware failure gallery.

A non-simulated single-seed output is deliberately labeled `heldout_comparison`, not Experiment.
Promotion requires all of the following:

1. a versioned image-SHA split and frozen evaluation examples;
2. identical prompts, decoding, pixel limits, and example budgets across conditions;
3. zero-shot and gold-LoRA baselines;
4. at least seeds 7, 17, and 29;
5. paired intervals and document/chart/grounding slices;
6. adapter bytes, peak VRAM, latency, wall time, and GPU-minutes;
7. retained predictions and failure cases.

## Reproduction commands

### Real processor

```bash
cd /path/to/foundation-model-lab
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/check_vlm_processor.py \
  --model-path "$FMLAB_MODEL_ROOT/Qwen/Qwen3-VL-8B-Instruct" \
  --manifest "$FMLAB_DATA_ROOT/datasets/vlm/mixed/manifest.jsonl" \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/qwen3-vl-real-processor-check"
```

### LoRA wiring

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/run_vlm_lora.py \
  --config configs/vlm/qwen3_vl_8b_lora_e2e.yaml \
  --backend mock \
  --max-steps 3 \
  --artifact-dir "$FMLAB_DATA_ROOT/artifacts/vlm/qwen3-vl-8b-lora-mock-task-aware"
```

### Distillation wiring

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python -m fmlab.vlm.response_distillation mock \
  --manifest "$FMLAB_DATA_ROOT/datasets/vlm/mixed/manifest.jsonl" \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/response-distillation-mock-full" \
  --max-examples 60 \
  --max-steps 5
```

### Held-out evaluator

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/evaluate_vlm_distillation.py \
  --base "$FMLAB_DATA_ROOT/predictions/vlm/base_8b.jsonl" \
  --gold "$FMLAB_DATA_ROOT/predictions/vlm/gold_lora.jsonl" \
  --distilled "$FMLAB_DATA_ROOT/predictions/vlm/distilled_lora.jsonl" \
  --teacher "$FMLAB_DATA_ROOT/predictions/vlm/teacher_32b.jsonl" \
  --train-manifest "$FMLAB_DATA_ROOT/generated/vlm/distillation/teacher-32b-accepted.jsonl" \
  --bootstrap-samples 2000 \
  --seed 17 \
  --output-dir "$FMLAB_DATA_ROOT/artifacts/vlm/distillation-heldout"
```

When the audited cache is supplied, `--train-manifest` uses only `split=train` rows for leakage
checks. The real three-process commands are documented in
[the response-distillation protocol](vlm_response_distillation.md).

## Storage budget

At the measurement point before final documentation:

| Storage | Size | Interpretation |
|---|---:|---|
| Repository source | about 2.5 MB | code, tests, configs, documentation |
| External lab root | about 653 MB | environment, cache, data, and artifacts |
| VLM artifacts | about 1.2 MB | JSON, JSONL, HTML, SVG, small images |
| Existing Qwen3-VL-8B | about 17 GB | shared model, not copied |
| Existing Qwen3-VL-32B FP8 | about 34 GB | shared model, not copied |
| Rank-8 `q/k/v/o` adapter | 7,667,712 parameters; about 14.6 MiB BF16 or 29.3 MiB FP32 | budget 15–31 MiB each |

With adapter-only saves and no optimizer state, the two new tracks add comfortably less than
1 GB. Persisting full checkpoints or optimizer state can exceed that budget and is disabled by
the bounded defaults.

## Portfolio interpretation

The strongest evidence in this revision is systems and research-method evidence:

- content-identity leakage control rather than row-level splitting;
- exact assistant supervision for a real multimodal processor;
- trainable-scope auditing rather than assuming PEFT targeting is correct;
- resumable, fingerprinted teacher data lineage;
- fail-closed cache and simulated-data contracts;
- resource-aware process isolation and preflight;
- generation-independent, paired held-out evaluation;
- explicit claim promotion instead of presenting smoke losses as quality.

The next meaningful evidence is not a longer mock run. It is one bounded real 8B gradient smoke
after GPU capacity becomes available, followed by cached 32B responses and a three-seed held-out
comparison against zero-shot and gold-label LoRA.

Implementation references:

- [Official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Transformers multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora)
