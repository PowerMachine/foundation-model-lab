## Research question or engineering objective

<!-- State the falsifiable question or bounded implementation objective. -->

## Change summary

<!-- Explain the smallest relevant design change and why it is needed. -->

## Evidence and claim boundary

- Declared level: <!-- Wiring / Smoke / Controlled study / Experiment / Benchmark -->
- Evidence source: <!-- scripted/synthetic / simulator / actual CPU / actual GPU/local model -->
- Evidenced:
- Not evidenced:
- Public bundle or result path:

## Verification

<!-- List exact commands and outcomes. Prefer `make check`, then targeted tests. -->

```text
command -> outcome
```

## Review checklist

- [ ] I used grouped splits or another explicit leakage control where data is involved.
- [ ] I included a baseline or negative control appropriate to the claim.
- [ ] I kept scripted, simulated, actual-runtime, and fake-quantization results visibly distinct.
- [ ] I recorded seeds/trials, sample budget, uncertainty, failures, and resource cost where relevant.
- [ ] CPU/offline tests do not require a GPU, secret, local model, or network download.
- [ ] No model weights, private data, hidden grader payloads, credentials, or local absolute paths are included.
- [ ] README/report numbers link to sanitized evidence or are explicitly marked pending.
- [ ] I updated tests and ran lint/format checks for changed Python code.
- [ ] I reviewed third-party licenses and data/model redistribution boundaries.
