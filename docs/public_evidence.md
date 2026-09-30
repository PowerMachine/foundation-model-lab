# Sanitized public evidence exporter

`fmlab.release` produces small reviewable evidence bundles for a public Git repository without
copying training traces, model state, or workstation identity. It treats the source artifact as
untrusted input and publishes only after a second scan of a hidden staging directory.

## Default allowlist

Only existing top-level files matching these entries are selected automatically:

- `result.json`
- `report.html`
- `provenance.json`
- `*.svg`

`result.json` is required because every public bundle must carry a claim boundary. Additional
files require a repeated `--include BASENAME` argument. An additional name must be one basename,
not a path, and must have one of these text extensions:

```text
.json .html .svg .yaml .yml .txt .md
```

JSONL, checkpoints, model weights, pickle files, and names that imply hidden traces, optimizer
state, adapters, or weights are rejected even when explicitly requested. They cannot be made
public merely by adding `--include`.

## Fail-closed checks

Before publication, the exporter requires all of the following:

1. source, destination components, and every source entry are real paths rather than symlinks;
2. source and destination do not overlap;
3. every selected name stays at the top level and cannot escape staging;
4. each selected file is no larger than 20 MiB, UTF-8 text, and contains no NUL byte;
5. JSON, YAML, SVG, and HTML remain parseable after redaction;
6. local `/home/...` and `/data/...` paths, user/hostname fields, current workstation identity,
   and every configured literal are replaced with `REDACTED_LOCAL`;
7. a second scan, including HTML entity decoding, finds no forbidden path or literal;
8. `result.json` supplies a complete `claim_boundary`, or the CLI supplies one explicitly.

The literal list automatically includes the current home directory, username, login name, and
hostname. Use `--redact-literal` repeatedly for old hosts, shared mount prefixes, project names,
or other private strings that may appear in historical artifacts. Literal values themselves are
never written to the public manifest.

## Staging and atomic update

Files are written below a hidden sibling such as `.ddp-correctness.staging-<id>`. A failed
redaction or policy check leaves this directory clearly marked as staging and never modifies the
existing public destination. This makes partial output inspectable without presenting it as a
release.

After the complete staging tree passes the post-export scan, it atomically replaces the public
directory. An existing destination is first renamed to a private backup; if publication fails it
is restored. The backup is removed only after the new directory has been installed.

## Evidence manifest

`evidence_manifest.json` is written atomically after all selected files. It records:

- UTC generation time and exporter schema;
- source artifact basename, never its absolute path;
- source and public SHA-256 for every copied file;
- source/public byte counts and per-file redaction count;
- total redaction count;
- exact default and explicit allowlist policy;
- the sanitized claim boundary and whether it came from `result.json` or an explicit argument.

The manifest cannot include its own hash without recursion. Every other public file hash is
covered, and the whole staging directory is rescanned after the manifest is written.

## CLI

General form:

```bash
python scripts/export_public_evidence.py SOURCE_ARTIFACT PUBLIC_DESTINATION \
  --redact-literal OLD_HOST \
  --include SAFE_ADDITIONAL_FILE.json
```

### Visual reward environment

The visual reward source predates the repository-wide `claim_boundary` field. Its public boundary
is therefore supplied explicitly. JSONL preference rows and reward-audit traces are intentionally
not included.

```bash
python scripts/export_public_evidence.py \
  "$FMLAB_DATA_ROOT/artifacts/vlm/visual-reward-environment" \
  public-evidence/vlm/visual-reward-environment \
  --include environment_spec.json \
  --claim-evidenced "Measured environment-and-grader behavior on deterministic synthetic visual tasks." \
  --not-evidenced "VLM generation quality or online RL" \
  --not-evidenced "human-preference agreement or production safety" \
  --not-evidenced "distributed rollout scale"
```

### CPU/Gloo DDP correctness

The DDP result already declares its claim boundary. Checkpoints, checksum sidecars, and JSONL
training traces stay outside the public bundle.

```bash
python scripts/export_public_evidence.py \
  "$FMLAB_DATA_ROOT/artifacts/distributed/ddp-correctness" \
  public-evidence/distributed/ddp-correctness
```

## Verification

```bash
pytest -q tests/test_public_evidence.py
ruff check src/fmlab/release.py scripts/export_public_evidence.py \
  tests/test_public_evidence.py
ruff format --check src/fmlab/release.py scripts/export_public_evidence.py \
  tests/test_public_evidence.py
```

The tests cover allowlist-only copying, deterministic hashes, redaction counts, safe replacement
of an existing public directory, traversal rejection, selected binary and oversize files,
directory symlinks, forbidden explicit JSONL/hidden traces, missing claim boundaries, and a
post-scan failure caused by an HTML-entity-encoded local path.

## Claim boundary

The exporter demonstrates deterministic artifact selection, configured literal/path redaction,
integrity metadata, and fail-closed publication for the tested text formats. It is not a general
data-loss-prevention system, malware scanner, legal release review, de-identification guarantee,
or proof that an artifact contains no sensitive semantic information. Human review of the final
bundle and its upstream data/model licenses remains required.
