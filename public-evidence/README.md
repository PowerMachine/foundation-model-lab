# Public Evidence

These small, sanitized bundles make the portfolio claims reviewable from GitHub without publishing
model weights, private traces, checkpoints, or workstation identity.

Start with the aggregate [scorecard figure](meta/portfolio-scorecard/scorecard.svg),
[machine-readable scorecard](meta/portfolio-scorecard/scorecard.json), and
[scorecard manifest](meta/portfolio-scorecard/evidence_manifest.json). The scorecard is a derived
overview, not a sixth source experiment: there are **five source evidence bundles**. The recursive
integrity verifier sees six manifest-bearing directories because the aggregate scorecard has its
own provenance manifest.

## Five-minute reviewer path

| Time | Open | Verify |
|---:|---|---|
| 0:00–0:30 | [Aggregate scorecard](meta/portfolio-scorecard/scorecard.svg) | Evidence classes and unsupported claims remain visible. |
| 0:30–1:20 | [Actual 8B LoRA report](vlm/qwen3-vl-8b-lora-real-2step/report.html) · [result](vlm/qwen3-vl-8b-lora-real-2step/result.json) | Real BF16 load and two optimizer steps; three-example baseline already perfect, so no gain claim. |
| 1:20–2:10 | [Visual reward report](vlm/visual-reward-environment/report.html) | Scripted/synthetic reward ablation and attack controls, not VLM quality. |
| 2:10–3:00 | [Agent report](agent/reliable-agent-benchmark/report.html) | Actual local harness with scripted controls, not LLM capability. |
| 3:00–4:00 | [Inference report](systems/inference-dynamics/report.html) | Simulation is separated from the actual tiny CPU probe. |
| 4:00–5:00 | [DDP report](distributed/ddp-correctness/report.html) | Actual two-rank CPU parity, exact resume, and fail-closed validation. |

## Source evidence bundles

| Track | Evidence class | Review path |
|---|---|---|
| Qwen3-VL-8B LoRA, two-step | actual local BF16 model and actual GPU optimizer smoke; no quality-improvement claim | [manifest](vlm/qwen3-vl-8b-lora-real-2step/evidence_manifest.json) · [result](vlm/qwen3-vl-8b-lora-real-2step/result.json) · [report](vlm/qwen3-vl-8b-lora-real-2step/report.html) |
| Visual reward environment | deterministic synthetic environment/grader audit; scripted policies | [manifest](vlm/visual-reward-environment/evidence_manifest.json) · [result](vlm/visual-reward-environment/result.json) · [figure](vlm/visual-reward-environment/reward_audit.svg) |
| Reliable agent evaluation | real local harness execution with scripted positive/attack controls; not LLM quality | [manifest](agent/reliable-agent-benchmark/evidence_manifest.json) · [result](agent/reliable-agent-benchmark/result.json) · [figure](agent/reliable-agent-benchmark/metrics.svg) |
| Inference dynamics | scheduler/capacity simulation plus an actual tiny CPU fidelity probe | [manifest](systems/inference-dynamics/evidence_manifest.json) · [result](systems/inference-dynamics/result.json) · [capacity figure](systems/inference-dynamics/capacity.svg) |
| CPU DDP correctness | actual two-process CPU/Gloo parity and exact-resume audit | [manifest](distributed/ddp-correctness/evidence_manifest.json) · [result](distributed/ddp-correctness/result.json) · [resume figure](distributed/ddp-correctness/resume_equivalence.svg) |

## Boundary

The source artifact directory remains external to Git. The fail-closed exporter selects only
allowlisted UTF-8 files, redacts local paths and workstation literals, validates the rendered
format, rescans a staging directory, and records source/public SHA-256 values before atomic publish.

JSONL traces, hidden grader payloads, model/adaptor weights, checkpoints, optimizer state, and
binary files are excluded. These controls reduce accidental disclosure; they are not a general DLP,
de-identification, legal, or license review guarantee.

