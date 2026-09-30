# Security Policy

## Supported versions

This is a research repository rather than a hosted service. Security fixes target the current
`main` branch and the latest published release, if one exists.

| Version | Supported |
|---|---|
| Current `main` | Yes |
| Latest release | Yes |
| Older commits, forks, and external model artifacts | No |

## Report a vulnerability privately

Use **Security → Report a vulnerability** in this repository so maintainers can discuss the issue
in a private GitHub Security Advisory. Include:

- the affected file, revision, and execution mode;
- a minimal reproduction or proof of concept;
- realistic impact and required attacker capabilities;
- whether private data, credentials, generated code, model files, or sandbox boundaries are
  involved; and
- a suggested mitigation, if known.

Do not open a public issue with exploit details, secrets, personal information, hidden grader
content, or unredacted paths. If private vulnerability reporting is unavailable, open a short
public issue asking a maintainer to establish a private contact channel without describing the
vulnerability.

Maintainers aim to acknowledge a complete report within seven days and provide an initial triage
within fourteen days. These are best-effort targets, not a service-level agreement. Please allow a
reasonable remediation and coordinated-disclosure window before public discussion.

## Security-relevant scope

Reports are especially useful for:

- arbitrary command execution, path traversal, or isolation bypass in agent runners;
- unsafe archive, checkpoint, pickle, YAML, HTML, or artifact handling;
- leakage of credentials, local identity, private prompts/data, or hidden evaluator state;
- public-evidence sanitization or manifest-integrity bypass;
- workflow permission escalation, untrusted fork execution, or supply-chain compromise;
- denial of service that escapes declared CPU, memory, process, storage, or timeout bounds; and
- model-loading paths that unexpectedly fetch remote code or files despite an offline/local-only
  contract.

Model hallucination, ordinary accuracy regressions, jailbreaks without a concrete protected
boundary, and differences between a simulator and production system are generally research-quality
issues rather than vulnerabilities. Use the evidence issue form unless confidentiality is needed.

## Trust boundaries

- Checked-in CPU CI executes repository code without GPU, secret, model checkpoint, or Hugging Face
  network access.
- `process` isolation applies resource/path controls but is not an OS or network security boundary.
- Rootless containers reduce exposure but do not prove complete isolation.
- Model weights, adapters, private datasets, teacher caches, and raw runtime artifacts are outside
  the public repository boundary.
- Sanitized evidence checks reduce accidental disclosure; they are not a general DLP,
  de-identification, malware-scanning, or legal-review system.

Never load an untrusted checkpoint or execute generated code outside an appropriately isolated,
disposable environment.

## Disclosure and credit

After a fix is available, maintainers may publish a GitHub Security Advisory describing affected
versions, impact, mitigation, and reporter credit. Tell us if you prefer to remain anonymous. We
will not request access to unrelated systems or private data to validate a report.
