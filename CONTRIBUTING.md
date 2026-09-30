# Contributing to Foundation Model Lab

Thank you for improving the lab. Contributions are evaluated on whether they make a research or
systems claim more reproducible, more inspectable, or more honestly bounded—not on demo size.

## Before opening a change

- Search existing issues and use the structured bug, evidence, or research-proposal form.
- For a substantial experiment, agree on a falsifiable question, baseline, resource budget, and
  intended evidence level before implementing it.
- Keep model weights, private data, teacher caches, hidden grader payloads, credentials, and raw
  workstation artifacts outside the repository.
- Report suspected vulnerabilities privately according to [SECURITY.md](SECURITY.md).

Small bug fixes, tests, documentation corrections, and CI repairs can go directly to a pull
request. A contribution does not need a GPU: CPU/offline paths are the required review baseline.

## Local CPU/offline setup

Python 3.11 and 3.12 are supported by CI. From a fresh virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make bootstrap-cpu
```

The install command uses the official PyTorch CPU wheel index. Tests set Hugging Face and
Transformers offline flags, hide CUDA devices, and write temporary artifacts outside the source
tree. They do not require a token, local checkpoint, or model download.

Run the same checks as CI:

```bash
make check
make toy
```

Use `make site-check` after changing the static portfolio and `make evidence` after changing a
sanitized bundle or its manifest.

## Evidence contract

Every result must declare the strongest level actually supported:

| Level | Required interpretation |
|---|---|
| Wiring | A deterministic mock/tiny path connects the components; no capability claim. |
| Smoke | A bounded real local model or runtime path works on the recorded host. |
| Controlled study | A declared synthetic, simulator, or actual-runtime testbed includes controls and invariant/statistical gates. |
| Experiment | A fixed leakage-safe holdout, relevant baselines, and at least three seeds support a dataset-scoped result. |
| Benchmark | A public protocol and competitive baselines support only that protocol's claim. |

Do not blur scripted policy behavior into LLM capability, simulation into hardware measurement,
fake quantization into packed-integer execution, or a tiny controlled study into production
validation. Preserve negative results, failed seeds, uncertainty, and the explicit non-claims.

## Experiment design checklist

A research-facing pull request should normally include:

1. a falsifiable question and predeclared primary metric;
2. a baseline and, where relevant, a negative or attack control;
3. a leakage unit such as source image, entity, patient, graph, or time boundary;
4. deterministic seeds plus repeated trials appropriate to the claim;
5. task/failure slices and uncertainty rather than a single aggregate;
6. CPU/GPU time, peak memory, storage, sample/token/pixel limits, and stop criteria;
7. machine-readable `result.json` and a derived human-readable report; and
8. an `evidenced`/`not_evidenced` boundary that survives public sanitization.

Code presence is not a result. README-level measured numbers need a checked-in sanitized bundle or
must be marked pending.

## Public evidence and data hygiene

Raw artifacts stay outside Git. Use the allowlist-based exporter documented in
`docs/public_evidence.md`; inspect the derivative and run:

```bash
make evidence
```

The verifier checks index coverage, manifest schema, every public byte count and SHA-256, claim
boundaries, top-level allowlisting, UTF-8/text constraints, symlinks, size limits, and common local
path/secret signatures. These controls do not replace human privacy, license, or semantic review.

Before submitting, inspect all text, HTML, SVG, JSON, and image pixels. Confirm data/model licenses
and do not publish personal information, internal identifiers, copied benchmark assets, or
uncertain model-derived artifacts.

## Code and test expectations

- Keep changes narrow and use existing module/config/report patterns.
- Add deterministic tests for success, invalid input, and failure/recovery paths.
- Never make a unit test depend on the network, a shared checkpoint, a GPU, or an installed
  container image.
- Real-model and GPU experiments remain explicit opt-in host validation.
- Use Ruff for linting and formatting; do not mix unrelated mechanical rewrites into a research
  change.
- Keep workflow permissions minimal and never use `pull_request_target` to execute contributor
  code.

## Pull requests

Complete the pull-request template with the objective, evidence source, claim boundary, exact
verification commands, and safe artifact links. Reviewers should be able to reconstruct which
inputs changed, which metric supports the conclusion, and which conclusion remains unsupported.

By contributing, you agree that your contribution is licensed under this repository's MIT license
and that you have the right to submit every included code, data, and visual asset.
