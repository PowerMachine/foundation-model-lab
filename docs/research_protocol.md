# Multimodal research protocol

This document defines what counts as evidence in Foundation Model Lab. It is deliberately
stricter than a demo checklist: every claim must be tied to a held-out slice, a reproducible
configuration, and a resource budget.

## Research questions

1. How much task adaptation can a language-side LoRA obtain when the Qwen3-VL vision encoder
   is frozen?
2. When does response distillation from a 32B teacher improve an 8B student over gold-only
   LoRA, and when does it merely copy teacher errors?
3. Which improvements survive grouped hold-out evaluation across document templates, chart
   operations, and present/absent grounding cases?
4. What accuracy is purchased per trainable parameter, GiB of peak VRAM, and GPU-minute?

## Claim levels

| Level | Required evidence | Allowed wording |
|---|---|---|
| Wiring | deterministic simulator or tiny surrogate | “pipeline validated” |
| Smoke | real local model, 1–5 optimizer steps | “model/API/gradient path validated” |
| Experiment | grouped hold-out, baseline, ≥3 seeds | “improved on this controlled dataset” |
| Benchmark | public test protocol and competitive baselines | “competitive on benchmark X” |

Simulator accuracy and one-sample inference must never be presented as model-quality results.

## Leakage-resistant data protocol

- Split by `image_path`, never by individual question. Questions sharing pixels remain in one
  split.
- For documents, additionally report a template-held-out split.
- For charts, slice results by operation: lookup, sum, difference, argmin, and argmax.
- For grounding, keep positive and negative cases balanced and report hallucination rate.
- Store the SHA-256 of every manifest and the exact split seed in `provenance.json`.
- Keep a frozen evaluation manifest that no training or teacher-generation stage may rewrite.

## Metrics

Primary quality metrics:

- exact match and ANLS for document text;
- numeric accuracy with an explicit tolerance for chart arithmetic;
- precision, recall, F1, hallucination rate, and bbox IoU for grounding;
- paired per-example deltas with a 95% bootstrap interval.

Efficiency metrics:

- trainable/total parameters and adapter bytes;
- peak allocated and reserved VRAM;
- examples/s, tokens/s, wall time, and GPU-minutes;
- teacher-cache time separated from student-training time.

## Low-compute ablation matrix

Run the smallest discriminating experiment first. The recommended matrix uses the same split
and at most 20 optimizer steps per cell.

| Axis | Values | Purpose |
|---|---|---|
| supervision | gold / teacher / filtered teacher | isolate distillation value |
| LoRA rank | 4 / 8 / 16 | capacity-efficiency frontier |
| targets | q,v / q,k,v,o | identify necessary projections |
| vision | frozen / last block trainable | test visual adaptation only if needed |
| pixels | 256² / 512² equivalent budget | quantify OCR-resolution trade-off |
| seed | 7 / 17 / 29 | expose run variance |

Promote a setting to a longer run only if its confidence interval and failure slices justify the
extra compute.

## Distillation protocol

Teacher generation and student training are sequential stages. The 32B teacher is unloaded
before the 8B student is created, so both models never need to coexist in VRAM. Each cached row
records teacher identity, decoding parameters, latency, source manifest hash, reference answer,
and whether filtering accepted it.

Required comparisons:

1. frozen 8B baseline;
2. 8B LoRA on gold labels;
3. 8B LoRA on raw teacher responses;
4. 8B LoRA on filtered teacher responses.

Filtering must be auditable. Empty answers, exact prompt leakage, duplicate responses, invalid
numeric formats, and known teacher/reference contradictions are counted rather than silently
dropped.

## Failure analysis

Every evaluation record should retain the input ID, task slice, reference, prediction, normalized
score, and latency. Reports group failures into OCR/transcription, arithmetic, localization,
hallucination, formatting, truncation, and infrastructure errors. A negative result remains an
artifact and is discussed alongside successful runs.

## Reproducibility checklist

- local-only model path and immutable model identifier;
- config snapshot and package versions;
- random seed and deterministic split;
- input manifest hashes;
- exact command, status, and error field;
- JSON metrics plus a human-readable HTML report;
- adapter-only checkpoint, never a duplicate base model;
- unit tests that use a tiny/mock backend and do not require a GPU or network.
