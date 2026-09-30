# Visual Knowledge Work Reward Environment

This track turns the canonical document, chart, and grounding data into an auditable single-step
RL environment. The reference answer and task metadata remain hidden from the observation. A
strict JSON action contract, type-aware answer verifier, grounding IoU check, confidence term, and
explicit reward-hacking penalty produce the training reward.

The lab compares that verifier with an intentionally vulnerable substring reward. Scripted
responses include correct actions, overconfident errors, abstentions, grader manipulation, and
unexpected-field injection. Their purpose is to validate the environment and reward, not to
represent a trained VLM.

## Contracts

- Every task is pinned by the image content SHA-256, prompt, hidden reference, metadata, split,
  schema version, and task fingerprint.
- Source-image groups never cross train and evaluation.
- Evaluation observations contain no reference or hidden metadata.
- Preference pairs are emitted only from the train split; held-out attack rows remain evaluation
  only.
- Invalid JSON, extra keys, malformed confidence, and invalid boxes fail closed.
- `teacher_simulated: true` is retained on scripted preference rows so downstream training cannot
  mistake them for human or real-model preferences.

## Run

```bash
source ${FMLAB_DATA_ROOT}/envs/fmlab/bin/activate
source scripts/env.sh
python scripts/run_visual_reward_lab.py \
  --config configs/vlm/visual_reward_environment.yaml
```

Outputs include `environment_spec.json`, `reward_audit.jsonl`, `preference_pairs.jsonl`, an SVG,
an HTML report, provenance, and canonical `result.json`.

## Claim boundary

This is a measured environment-and-grader result on deterministic synthetic tasks. It demonstrates
hidden-state separation, reward-hacking detection, preference-data lineage, and evaluation
plumbing. It is not VLM quality, online RL, human-preference agreement, distributed rollout scale,
or evidence that the reward is complete for real visual knowledge work.
