# Frontier Research Engineering Alignment

Snapshot date: 2026-08-04. Job descriptions change; the links below are evidence for the design
priorities at this date, not endorsements or claims of equivalence to work performed inside these
organizations.

## Why the lab is organized this way

Current frontier research-engineering roles repeatedly ask for one connected capability:

```text
data and environments
        ↓
post-training and reward signals
        ↓
reliable evaluation and diagnosis
        ↓
efficient, observable execution
        ↓
failure-driven data and system improvement
```

The portfolio therefore emphasizes falsifiable contracts and failure analysis over a long list of
model demos. A small real measurement is retained as such; a simulator, scripted policy, CPU
distributed run, or one-example smoke test is never promoted into a frontier-scale claim.

## Public-role signal matrix

| Repeated hiring signal | Representative official roles | Evidence in this repository | Current boundary |
|---|---|---|---|
| Agent environments, graders, long-horizon/tool behavior | [OpenAI Codex Research Engineer](https://openai.com/careers/research-engineer-codex-san-francisco/), [OpenAI Frontier Evals & Environments](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/) | typed tool loop, restricted execution, hidden outcome/process graders, attack tasks, deterministic replay and failure taxonomy | scripted-policy benchmark; not SWE-bench or production rollout scale |
| Evaluation reliability, statistics, regressions, dashboards | [Anthropic Model Evaluations](https://job-boards.greenhouse.io/anthropic/jobs/5198255008), [Google DeepMind Research Engineer](https://deepmind.google/careers/) | grouped holdouts, paired bootstrap intervals, source-contamination checks, resumable eval ledger, retry/fault audit, HTML dashboards | local execution; not hundreds of live training checkpoints |
| VLM data strategy, rewards, held-out generalization | [Anthropic Visual Knowledge Work](https://job-boards.greenhouse.io/anthropic/jobs/5074217008), [Apple Multimodal Foundation Models](https://jobs.apple.com/en-gr/details/200672934-3956/ai-research-scientist-multimodal-foundation-models-architecture-pre-training-distillation?team=MLAI) | Qwen3-VL LoRA and response-distillation systems plus hidden-reference visual environment, typed reward, reward-hacking audit and preference-data lineage | synthetic environment and bounded model smoke evidence; no real visual RL run |
| Production post-training and reproducible distributed training | [Anthropic Production Model Post-Training](https://job-boards.greenhouse.io/anthropic/jobs/4613592008), [OpenAI Research Engineer](https://openai.com/careers/research-engineer-san-francisco/) | SFT/LoRA/QLoRA/DPO/distillation objectives, cache contracts, CPU DDP gradient parity, exact-resume and corruption rejection | two-process CPU/Gloo correctness only; no NCCL or multi-node scaling claim |
| Inference throughput, tail latency, correctness and capacity | [OpenAI Model Inference](https://openai.com/careers/software-engineer-model-inference-san-francisco/), [Anthropic Inference Systems](https://job-boards.greenhouse.io/anthropic/jobs/5224564008), [NVIDIA ML Systems Research](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Research-Scientist--ML-Systems---New-College-Grad-2026_JR2010161) | continuous/static scheduling study, paged-KV pressure, overload/backpressure, SLO goodput and real tiny-model quantization correctness gate | discrete-event fleet model plus CPU tiny-model measurement; not vLLM fleet telemetry |
| Causal access to model internals and safety diagnosis | [Anthropic Interpretability](https://job-boards.greenhouse.io/anthropic/jobs/4980430008) | attention visualization and explicit small-transformer internals | activation patching/steering and scaled activation infrastructure remain future work |

## Senior-level evidence contract

For each flagship track, reviewers should be able to answer the following from committed code and
generated JSON rather than prose alone:

1. What hypothesis or system invariant was tested?
2. Which inputs, code/config version, split, seed, and runtime produced the result?
3. What failure was injected, and did the system distinguish model, data, grader, and
   infrastructure failures?
4. Which metric would block a release, and how sensitive is the decision to variance or a chosen
   threshold?
5. What does the evidence *not* support?

The repository's claim ledger uses Wiring, Smoke, controlled comparison, Experiment, and Benchmark
labels to keep those boundaries reviewable.

## Remaining high-value gaps

The following should not be hidden by additional toy breadth:

- a real three-seed held-out LoRA/distillation comparison on non-synthetic VLM data;
- online RL/RLVR using executable rewards rather than only generated preference pairs;
- real vLLM/SGLang traces validating the serving simulator on an available GPU;
- NCCL multi-GPU or multi-node training measurements;
- causal activation patching/steering on a trained model;
- an external benchmark, paper submission, or merged upstream contribution.

These gaps are stated explicitly because strong research engineering includes knowing which
conclusion has not yet been earned.
