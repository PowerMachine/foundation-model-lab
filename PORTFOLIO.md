# Reliable Multimodal Agent Systems Research Engineer

I build research systems in which multimodal data and reward contracts, agent environments,
distributed training semantics, inference behavior, and release evidence remain connected and
auditable. The organizing principle is not model size; it is whether a reviewer can recover the
hypothesis, implementation, measurement, failure controls, and unsupported claims from code and
machine-readable artifacts.

This portfolio targets research-engineering work at the intersection of multimodal post-training,
agent evaluation, and ML systems. The role-signal snapshot and official postings used to choose the
work are documented in [Frontier Research Engineering Alignment](docs/frontier_role_alignment.md).

## Five-minute reviewer path

| Time | Review | What to verify |
| ---: | --- | --- |
| 0:00–0:25 | [Aggregate scorecard](public-evidence/meta/portfolio-scorecard/scorecard.svg) and [JSON](public-evidence/meta/portfolio-scorecard/scorecard.json) | Five source bundles are summarized without merging actual, scripted, or simulated evidence classes. |
| 0:25–1:10 | [Actual 8B LoRA report](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/report.html) and [result](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/result.json) | Real BF16 load and two GPU optimizer steps; the three-example baseline was already perfect, so no gain claim. |
| 1:10–1:55 | [Visual reward report](public-evidence/vlm/visual-reward-environment/report.html) and [result](public-evidence/vlm/visual-reward-environment/result.json) | Hidden references, source-group holdout, robust-vs-naive reward ablation, and attack detection. |
| 1:55–2:40 | [Reliable agent report](public-evidence/agent/reliable-agent-benchmark/report.html) and [result](public-evidence/agent/reliable-agent-benchmark/result.json) | Outcome/process grading, attack taxonomy, bounded retry, and exact resume under scripted controls. |
| 2:40–3:50 | [Inference dynamics report](public-evidence/systems/inference-dynamics/report.html) and [result](public-evidence/systems/inference-dynamics/result.json) | Paired 3-seed simulated load sweep, policy knees, fault attribution, and separately labeled actual CPU probe. |
| 3:50–4:40 | [CPU DDP report](public-evidence/distributed/ddp-correctness/report.html) and [result](public-evidence/distributed/ddp-correctness/result.json) | Actual two-rank parity, exact resume, deterministic sharding, and fail-closed checkpoint validation. |
| 4:40–5:00 | [Role alignment and gaps](docs/frontier_role_alignment.md#remaining-high-value-gaps) | What was measured and what still requires non-synthetic data, more seeds, production runtime, or external evidence. |

## Evidence vocabulary

- **Actual GPU smoke** means real model weights, forward/backward, optimizer, memory, and timing ran
  at a deliberately tiny budget; it is not a convergence, quality-improvement, or benchmark claim.
- **Actual CPU execution** means real PyTorch or `torch.distributed` work ran locally and wall-clock
  or numerical results were measured.
- **Actual local environment measurement** means real tools, files, graders, ledgers, and retries ran,
  while the evaluated policy was a deterministic scripted harness control rather than a model.
- **Scripted synthetic environment** means deterministic candidates validate data/reward behavior;
  they do not measure VLM quality or online RL.
- **Discrete-event simulation** means scheduler, queue, KV, and load behavior follows an explicit
  service model rather than vLLM, GPU, or production telemetry.
- **Wiring or smoke** validates a code/data/gradient path at bounded scale and is not promoted to an
  experiment or benchmark claim.

## Hiring signal to evidence matrix

| Public role signal | Code and protocol | Measured or public evidence | Boundary retained |
| --- | --- | --- | --- |
| Agent environments, graders, tool behavior: [OpenAI Codex](https://openai.com/careers/research-engineer-codex-san-francisco/), [OpenAI Frontier Evals & Environments](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/), [Anthropic Model Evaluations](https://job-boards.greenhouse.io/anthropic/jobs/5198255008) | [Environment](src/fmlab/agent/eval_environment.py), [benchmark runner](src/fmlab/agent/benchmark.py), [tests](tests/test_agent_benchmark.py) | [24-episode result](public-evidence/agent/reliable-agent-benchmark/result.json), [task commitments](public-evidence/agent/reliable-agent-benchmark/task_manifest.json), [evidence manifest](public-evidence/agent/reliable-agent-benchmark/evidence_manifest.json) | Two scripted controls over 12 Python microtasks; not an LLM result, SWE-bench-scale result, OS sandbox, or production rollout. |
| Multimodal adaptation and held-out evaluation: [Anthropic Visual Knowledge Work](https://job-boards.greenhouse.io/anthropic/jobs/5074217008), [Apple Multimodal Foundation Models](https://jobs.apple.com/en-gr/details/200672934-3956/ai-research-scientist-multimodal-foundation-models-architecture-pre-training-distillation?team=MLAI) | [LoRA runner](src/fmlab/vlm/lora_e2e.py), [grouped data contract](src/fmlab/vlm/lora_data.py), [GPU smoke report](docs/results_2026-08-04-vlm-gpu-smoke.md) | [Actual 8B result](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/result.json) and [evidence manifest](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/evidence_manifest.json) | Real BF16 load and two assistant-only GPU optimizer steps on synthetic data; the three-example baseline was already 100%, so no improvement, convergence, or speedup claim. |
| Visual reward and preference data: [Anthropic Visual Knowledge Work](https://job-boards.greenhouse.io/anthropic/jobs/5074217008) | [Reward environment](src/fmlab/vlm/reward_environment.py), [research protocol](docs/research_protocol.md), [tests](tests/test_vlm_reward_environment.py) | [Reward result](public-evidence/vlm/visual-reward-environment/result.json), [environment contract](public-evidence/vlm/visual-reward-environment/environment_spec.json), [audit figure](public-evidence/vlm/visual-reward-environment/reward_audit.svg) | Deterministic synthetic tasks and scripted candidates; not VLM generation, human-preference agreement, online RL, or rollout scale. |
| Reproducible post-training and distributed correctness: [Anthropic Production Model Post-Training](https://job-boards.greenhouse.io/anthropic/jobs/4613592008), [OpenAI Research Engineer](https://openai.com/careers/research-engineer-san-francisco/) | [DDP experiment](src/fmlab/distributed/experiment.py), [checkpoint contract](src/fmlab/distributed/checkpoint.py), [tests](tests/test_distributed_correctness.py); [LLM objectives](docs/llm_track.md) | [Actual CPU/Gloo result](public-evidence/distributed/ddp-correctness/result.json), [resume equivalence](public-evidence/distributed/ddp-correctness/resume_equivalence.svg), [sharding audit](public-evidence/distributed/ddp-correctness/sample_sharding.svg) | Actual two-process CPU/Gloo correctness on one tiny deterministic workload; no NCCL, multi-node, FSDP/ZeRO, convergence, or scaling claim. |
| Throughput, tail latency, correctness, and capacity: [OpenAI Model Inference](https://openai.com/careers/software-engineer-model-inference-san-francisco/), [Anthropic Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), [NVIDIA ML Systems Research](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Research-Scientist--ML-Systems---New-College-Grad-2026_JR2010161) | [Simulator](src/fmlab/systems/simulator.py), [actual CPU probe](src/fmlab/systems/probe.py), [capacity logic](src/fmlab/systems/capacity.py), [tests](tests/test_inference_dynamics.py) | [Public result](public-evidence/systems/inference-dynamics/result.json), [81-trial/27-aggregate sweep](public-evidence/systems/inference-dynamics/capacity_sweep.json), and [experiment protocol](docs/inference_dynamics.md) | Scheduler/load results are simulated; fake INT8/INT4 weights run through FP32 kernels; no vLLM, packed-kernel, GPU, or fleet-capacity claim. |
| Reliable evaluation statistics and failure analysis: [Anthropic Model Evaluations](https://job-boards.greenhouse.io/anthropic/jobs/5198255008), [Google DeepMind Research Engineer](https://deepmind.google/careers/) | Grouped source holdouts, paired bootstrap, content-addressed tasks, strict configuration, deterministic shards, failure taxonomies, and negative controls across the four tracks | [Consolidated results](docs/results_2026-08-04-frontier-systems.md) and track-specific machine-readable results above | Small controlled studies expose variance and failure modes; they are not external benchmark or live-checkpoint regressions. |
| Safe, reviewable open-source release | [Release exporter](src/fmlab/release.py), [release policy](docs/open_source_release.md), [exporter tests](tests/test_public_evidence.py) | Five sanitized source [public-evidence bundles](public-evidence/) with per-file hashes, redaction counts, allowlists, and explicit claim boundaries; the aggregate scorecard adds a sixth manifest-bearing directory | Fail-closed text-artifact publication is not general DLP, malware scanning, legal review, or a semantic privacy guarantee. |

## Capability profile

### Research and evaluation

- Converts a broad system question into an explicit invariant, paired comparison, failure injection,
  and machine-readable claim boundary.
- Uses source-group holdouts, fixed seeds, task/config fingerprints, bootstrap intervals, and
  threshold-sensitive capacity decisions rather than a single favorable run.
- Preserves negative evidence: vulnerable rewards, attack policies, overload knees, corrupt
  checkpoints, insufficient-resource stops, and the 32B FP8-kernel failure remain visible.

### Models and post-training

- Implements a transparent decoder-only Transformer with RoPE, RMSNorm, SwiGLU, causal MHA/GQA,
  tied embeddings, and attention visualization; see [LLM track](docs/llm_track.md).
- Implements CPU-scale continued pretraining, assistant-only SFT, LoRA, simulated QLoRA,
  quantization fidelity, knowledge distillation, DPO, and retrieval evaluation.
- Provides local-only Qwen and Qwen3-VL workflows with strict model/data contracts, adapter-only
  outputs, vision-freeze/LoRA-target audits, teacher-cache lineage, and fail-fast resource gates;
  executed a real BF16 8B two-step LoRA smoke with 7.67M trainable parameters and 17.45 GiB peak
  allocated training memory. See [VLM track](docs/vlm_track.md) and
  [actual GPU report](docs/results_2026-08-04-vlm-gpu-smoke.md).
- Keeps tiny/simulated objective evidence separate from real-model wiring, smoke, controlled
  experiments, and public benchmarks.

### Multimodal environments and agents

- Builds typed visual observations/actions across document QA, chart QA, and grounding, with hidden
  references, answer-type verification, IoU checks, calibration, and reward-hacking penalties.
- Builds immutable coding tasks, ephemeral workspaces, private outcome grading, process grading,
  safe tool dispatch, trace evidence, deterministic sharding, atomic ledgers, and bounded retry.
- Treats trusted in-process harness code, model-facing observations, filesystem isolation, and
  network isolation as different security boundaries.

### ML systems

- Studies static/continuous scheduling, bounded queues, deadlines, cancellations, paged KV,
  preemption, overload, tail latency, SLO goodput, and sustainable-load knees with paired traces.
- Separates discrete-event service-model outputs from actual tiny-model CPU fidelity/latency.
- Verifies DDP averaged-gradient semantics, `no_sync` accumulation, sampler behavior, atomic
  checkpoint integrity, RNG/cursor restoration, and uninterrupted-vs-resumed equivalence.

### Reproducibility and release

- Produces canonical JSON, JSONL traces, config/input hashes, SVG/HTML reports, strict schemas,
  seeded experiments, and portable relative-link documentation.
- Publishes through a text-only allowlist, hidden staging, pre/post redaction scans, atomic install,
  and evidence manifests while excluding checkpoints, JSONL traces, weights, and caches.

## Verifiable English resume bullets

- Executed a bounded actual-GPU Qwen3-VL-8B LoRA smoke: loaded 8.77B BF16 parameters, trained
  7.67M text-attention adapter parameters with the vision tower frozen for two assistant-only
  optimizer steps, and measured 17.45 GiB peak allocated memory; retained the zero-delta
  three-example result as smoke evidence rather than claiming improvement.
- Built a reliable coding-agent evaluation environment with 12 immutable task perturbations and 24
  locally executed episodes; combined hidden outcome and process-integrity graders, detected attacks
  in 100% of scripted negative-control episodes, recovered five injected transient faults within one
  retry, and resumed 24/24 episodes with zero duplicate execution.
- Designed a hidden-reference visual reward environment spanning document QA, chart QA, and
  grounding; improved scripted preference accuracy from 45.8% to 100.0% (paired delta 54.2 points,
  95% bootstrap CI 39.6–68.8) and reduced scripted reward-hack false acceptance from 100% to 0% on
  24 challenge cases with no clean false positives.
- Implemented a paired three-seed inference capacity study covering 81 discrete-event policy trials
  and 27 aggregates; identified modeled configured-Poisson-rate knee intervals of 24–32 requests/s
  for static FCFS and 32–48 requests/s for both continuous policies under explicit completion, SLO,
  and terminal-failure thresholds.
- Executed an actual two-rank CPU/Gloo DDP correctness study with `no_sync` accumulation; matched a
  single-process global-batch reference to `1.11e-16` maximum gradient absolute error, reproduced
  model state and losses exactly after checkpoint resume, and rejected both contract drift and byte
  corruption before state application.
- Implemented a from-scratch modern decoder and bounded post-training stack spanning SFT, LoRA,
  simulated QLoRA, quantization, distillation, DPO, and RAG, with attention/loss diagnostics and an
  evidence ledger that prevents tiny or simulated runs from being presented as frontier-model gains.
- Built a fail-closed public-evidence exporter using allowlisted text artifacts, hidden staging,
  path/identity redaction, post-export rescanning, atomic replacement, and per-file SHA-256 manifests;
  published five sanitized source review bundles without weights, checkpoints, caches, or private
  traces, plus a separately manifested aggregate scorecard.

## Public evidence index

| Track | Result | Human report | Integrity metadata |
| --- | --- | --- | --- |
| Qwen3-VL-8B LoRA, actual GPU smoke | [result.json](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/result.json) | [report.html](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/report.html) | [evidence manifest](public-evidence/vlm/qwen3-vl-8b-lora-real-2step/evidence_manifest.json) |
| Visual reward environment | [result.json](public-evidence/vlm/visual-reward-environment/result.json) | [report.html](public-evidence/vlm/visual-reward-environment/report.html) | [evidence manifest](public-evidence/vlm/visual-reward-environment/evidence_manifest.json) |
| Reliable agent evaluation | [result.json](public-evidence/agent/reliable-agent-benchmark/result.json) | [report.html](public-evidence/agent/reliable-agent-benchmark/report.html) | [evidence manifest](public-evidence/agent/reliable-agent-benchmark/evidence_manifest.json) |
| CPU DDP correctness | [result.json](public-evidence/distributed/ddp-correctness/result.json) | [report.html](public-evidence/distributed/ddp-correctness/report.html) | [evidence manifest](public-evidence/distributed/ddp-correctness/evidence_manifest.json) |
| Inference dynamics | [result.json](public-evidence/systems/inference-dynamics/result.json) | [report.html](public-evidence/systems/inference-dynamics/report.html) | [evidence manifest](public-evidence/systems/inference-dynamics/evidence_manifest.json) |

The five source bundles are summarized by the [aggregate scorecard figure](public-evidence/meta/portfolio-scorecard/scorecard.svg)
and [scorecard JSON](public-evidence/meta/portfolio-scorecard/scorecard.json). Its own
[provenance manifest](public-evidence/meta/portfolio-scorecard/evidence_manifest.json) makes six
manifest-bearing directories without turning the derived scorecard into a source
experiment.

The publication mechanism and exclusion policy are documented in
[Sanitized public evidence exporter](docs/public_evidence.md).

## Remaining gaps and next evidence promotions

1. Run a real three-seed grouped-holdout Qwen3-VL LoRA/distillation comparison on releasable,
   non-synthetic visual data; publish task slices, confidence intervals, VRAM, and GPU-minutes.
2. Add online RL/RLVR rollouts against executable visual rewards; evaluate reward overoptimization,
   policy generalization, and grader transfer rather than only scripted preference pairs.
3. Replay the inference workload against vLLM or SGLang on an available GPU and calibrate simulator
   service/KV parameters from traces; publish that real-runtime evidence as a separately versioned
   bundle without relabeling the current simulation-plus-CPU evidence.
4. Re-run distributed invariants on NCCL multi-GPU, then measure topology-aware scaling and failure
   recovery separately from semantic correctness.
5. Add causal activation patching/steering and scalable activation collection to the existing
   transparent-transformer work, aligned with the [Anthropic Interpretability](https://job-boards.greenhouse.io/anthropic/jobs/4980430008) signal.
6. Convert at least one track into external evidence: a public benchmark submission, workshop paper,
   reproducibility report, or merged upstream contribution.

These are gaps, not hidden roadmap completions. The next promotion should add a stronger evidence
class, not another uncalibrated toy result.
