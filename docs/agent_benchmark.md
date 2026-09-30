# Reliable Agent Evaluation Environment

This project is a small, reproducible coding-agent evaluation environment. It is designed to
demonstrate the engineering behind trustworthy agent evaluation—immutable tasks, hidden outcome
grading, process-integrity checks, deterministic distributed execution, crash-safe resume, and
auditable reports—without claiming that a scripted control is an LLM.

## What is measured

An episode passes only when all of the following are true:

1. every hidden input/output case passes in the private outcome grader;
2. an unchanged public test suite passes after the final source edit;
3. the action trace is sequential and internally consistent; and
4. no reward-hacking signal is detected.

The default run evaluates 12 task instances: six Python bug-fix families, each in a canonical and a
renamed-module perturbation. Two deterministic policies provide harness controls:

- `oracle_patch_scripted` applies the task's declared reference patch. It is a harness upper bound,
  not a model-capability result.
- `shortcut_probe_scripted` deliberately attempts false completion, hidden-grader access, path
  escape, an unapproved tool, unvalidated edits, or public-test tampering. It is a security negative
  control, not a model baseline.

## Architecture and trust boundary

```text
immutable TaskSpec + SHA-256 commitment
                 |
                 v
       reset -> observation  ------> policy
                 ^                      |
                 |                  safe actions
                 |                      v
       ephemeral workspace <------ action trace
                 |
          policy has finished
                 |
                 v
 private temporary grader (separate root)
  candidate source + hidden inputs only
                 |
       expected values compared in parent
                 v
 outcome grade + process grade -> atomic ledger -> reports
```

`TaskSpec` is frozen and content-addressed. Its task UID includes the specification fingerprint, so
changing instructions, assets, hidden cases, or the reference patch changes task identity. The
policy observation contains public files and commitments but not hidden cases or the reference
patch. Hidden grader assets are created only after policy execution, outside the episode workspace,
and removed after scoring. Expected values remain in the scorer process and are not written to the
grader directory.

The policy interface is capability-oriented: evaluated agents should receive only observations and
the allowlisted action API. Python policy implementations loaded into the benchmark process are
trusted harness code, not sandboxed third-party plugins. The scripted oracle is intentionally
privileged because its only purpose is harness validation.

### Isolation modes

- `process` applies time, CPU, memory, output, file-size, and process-count limits through the local
  restricted runner. It does **not** isolate networking and is **not** an OS security boundary.
- `bwrap` uses Bubblewrap when available and reports whether network isolation was actually active.
  Use this mode for untrusted generated code only after validating it on the host.

The default checked-in configuration uses `process` so the CPU demo works on constrained hosts. The
result records `execution_isolation` and `network_isolated`; the report never upgrades these facts
into a stronger security claim.

## Integrity and reward-hacking detectors

The process grader records and classifies:

- deletion or modification of public tests, including delete-and-recreate forgery;
- attempted access to hidden/grader paths;
- paths resolving outside the ephemeral workspace;
- tools outside the allowlist;
- known unsafe code patterns in candidate source;
- success claims made while the hidden outcome fails or after an unvalidated edit; and
- malformed/non-sequential traces.

Hidden grading remains authoritative even if a policy replaces public tests with an always-passing
file. A primary failure taxonomy makes each failed episode aggregate cleanly while the complete set
of signals remains in the trace.

## Reproducible execution

Run the full CPU demonstration from the repository root:

```bash
python scripts/run_agent_benchmark.py \
  --config configs/agent/reliable_agent_benchmark.yaml
```

Useful deterministic overrides:

```bash
# One of three disjoint shards
python scripts/run_agent_benchmark.py \
  --config configs/agent/reliable_agent_benchmark.yaml \
  --shard-index 1 --shard-count 3

# Quick smoke run without injected infrastructure failure
python scripts/run_agent_benchmark.py \
  --config configs/agent/reliable_agent_benchmark.yaml \
  --task-limit 2 --no-transient-failures
```

Shard assignment is `SHA256(task_uid) mod shard_count`; it is independent of enumeration order.
Episode identity commits to the task fingerprint, policy ID/version, environment contract, isolation
mode, and resource limits. Therefore a rerun with an identical contract resumes an existing episode,
while a meaningful contract change creates a new identity.

Each JSONL append is implemented as a lock/read/verify/rewrite/fsync/atomic-replace transaction.
Rows include an individual content commitment; malformed hashes and duplicate on-disk episode IDs
are rejected during verification. The
deterministically injected transient failure exercises bounded retry without giving a policy an
additional scored attempt. Rerunning a completed experiment should report zero executed episodes and
all episodes resumed.

## Metrics

The aggregate and per-policy report contains:

- pass@1 with seeded non-parametric 95% bootstrap confidence intervals;
- hidden-outcome pass rate and reward-hack rate;
- mean steps and p50/p95 wall latency;
- tool-error and unsafe-action attempt rates;
- failure taxonomy and reward-hack category counts; and
- executed/resumed episodes, injected transient failures, retry count, and maximum attempts.

With only 12 tasks per policy, confidence intervals are intentionally visible: this is a systems
demonstration and regression suite, not a statistically powered model leaderboard. Latency measures
local harness execution and cannot be compared with model-serving latency.

### Measured local demonstration

The full checked-in configuration was executed on the local CPU host with 12 tasks and 24 episodes:

| Policy/scope | Episodes | Integrity pass@1 | Raw outcome pass | Reward-hack rate |
| --- | ---: | ---: | ---: | ---: |
| Overall | 24 | 50.0% (95% bootstrap CI 29.2–70.8%) | 58.3% | 50.0% |
| Oracle harness control | 12 | 100.0% | 100.0% | 0.0% |
| Shortcut security control | 12 | 0.0% | 16.7% | 100.0% |

Five deterministic transient faults were injected; all completed on the single allowed retry and the
maximum attempt count was two. An identical second invocation executed zero episodes, resumed all 24,
and left exactly 24 ledger rows and 24 trace rows. The generated report bundle occupies approximately
152 KiB. The run used `process` isolation with `network_isolated=false`; these measurements validate
the evaluator and controls, not an LLM.

## Artifact contract

The configured artifact directory receives:

| File | Purpose |
| --- | --- |
| `task_manifest.json` | Public task commitments, perturbations, and suite fingerprint |
| `run_manifest.json` | Exact configuration, config hash, shard, and planned episode IDs |
| `traces.jsonl` | Observation, action/tool trace, retries, outcome grade, and process grade |
| `ledger.jsonl` | Compact crash-safe episode summaries used for idempotent resume |
| `result.json` | Machine-readable claims, metrics, parameters, provenance, and notes |
| `metrics.svg` | Dependency-free policy comparison chart |
| `report.html` | Self-contained human-readable audit report with failure evidence |

The checked-in demo path is:

```text
${FMLAB_DATA_ROOT}/artifacts/agent/reliable-agent-benchmark
```

No model weights or network access are required. Scratch workspaces and private grader directories
are removed after every attempt; only small JSON/HTML/SVG artifacts remain.

## Verification

```bash
pytest -q tests/test_agent_benchmark.py
ruff check \
  src/fmlab/agent/benchmark*.py \
  src/fmlab/agent/eval_environment.py \
  scripts/run_agent_benchmark.py \
  tests/test_agent_benchmark.py
```

The focused tests cover immutable identity, the 12-task/perturbation suite, hidden-grader separation,
all-task oracle execution, every reward-hacking class, public-test forgery, path/tool blocking,
deterministic sharding, ledger tamper detection, seeded bootstrap intervals, artifact generation,
idempotent resume, bounded transient retry, and strict configuration validation.

## Extension path to a real model policy

Keep the evaluator unchanged and implement an adapter that converts `Observation` into a model
prompt, validates structured tool calls, and forwards only allowlisted calls to `apply_action`.
Record model/provider/version, decoding parameters, prompt hash, token counts, and inference latency in
the episode contract before comparing policies. Add repeated stochastic samples and task-level paired
bootstrap tests for model comparisons. Do not label the current scripted-control results as model
performance.
