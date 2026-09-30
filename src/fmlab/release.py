from __future__ import annotations

import getpass
import hashlib
import html
import json
import os
import re
import shutil
import socket
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "1.0"
MAX_PUBLIC_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_EXACT_NAMES = ("result.json", "report.html", "provenance.json")
ALLOWED_EXTENSIONS = frozenset({".json", ".html", ".svg", ".yaml", ".yml", ".txt", ".md"})
FORBIDDEN_EXTENSIONS = frozenset(
    {".jsonl", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".pickle", ".pkl"}
)
FORBIDDEN_NAME_MARKERS = ("hidden", "trace", "checkpoint", "optimizer", "adapter", "weight")
REDACTION = "REDACTED_LOCAL"

_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:home|data)/[^\s\"'<>]+")
_SENSITIVE_KEY_VALUE = re.compile(
    r"(?im)(?P<prefix>(?:\"(?:username|user|hostname|host)\"|"
    r"(?:username|user|hostname|host))\s*:\s*)"
    r"(?P<quote>[\"']?)(?P<value>[^\"'\r\n,}]+)(?P=quote)"
)


class EvidenceExportError(RuntimeError):
    """Base error for a public-evidence export that was not published."""


class EvidencePolicyError(EvidenceExportError):
    """An input violates the public artifact policy."""


class EvidenceRedactionError(EvidenceExportError):
    """Sensitive content remained after the configured redaction pass."""


@dataclass(frozen=True)
class PublicEvidenceExport:
    destination: Path
    manifest: Path
    files: tuple[str, ...]
    redaction_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": str(self.destination),
            "manifest": str(self.manifest),
            "files": list(self.files),
            "redaction_count": self.redaction_count,
        }


class _HTMLValidator(HTMLParser):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_no_symlink_components(path: Path) -> None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise EvidencePolicyError(f"symlink path component is forbidden: {current}")
        if not current.exists():
            break


def _reject_tree_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise EvidencePolicyError(f"symlink directory is forbidden: {root}")
    if not root.exists():
        return
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*subdirectories, *filenames]:
            candidate = base / name
            if candidate.is_symlink():
                raise EvidencePolicyError(f"symlink entry is forbidden: {candidate}")


def _validate_simple_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise EvidencePolicyError(f"additional file must be one basename, not a path: {value!r}")
    if "/" in value or "\\" in value or "\x00" in value:
        raise EvidencePolicyError(f"unsafe additional filename: {value!r}")
    return value


def _validate_public_filename(name: str, *, explicit: bool) -> None:
    suffix = Path(name).suffix.lower()
    lowered = name.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        raise EvidencePolicyError(f"model/checkpoint/JSONL extension is forbidden: {name}")
    if suffix not in ALLOWED_EXTENSIONS:
        raise EvidencePolicyError(f"unexpected public evidence extension: {name}")
    if explicit and any(marker in lowered for marker in FORBIDDEN_NAME_MARKERS):
        raise EvidencePolicyError(f"hidden trace/model-state style filename is forbidden: {name}")


def _selected_files(source: Path, additional_files: Sequence[str]) -> list[Path]:
    _reject_tree_symlinks(source)
    selected: dict[str, Path] = {}
    for name in DEFAULT_EXACT_NAMES:
        candidate = source / name
        if candidate.exists():
            if not candidate.is_file():
                raise EvidencePolicyError(f"allowlisted source is not a regular file: {candidate}")
            _validate_public_filename(name, explicit=False)
            selected[name] = candidate
    for candidate in sorted(source.glob("*.svg")):
        if not candidate.is_file() or candidate.is_symlink():
            raise EvidencePolicyError(f"allowlisted SVG is not a regular file: {candidate}")
        _validate_public_filename(candidate.name, explicit=False)
        selected[candidate.name] = candidate
    for raw_name in additional_files:
        name = _validate_simple_name(str(raw_name))
        _validate_public_filename(name, explicit=True)
        candidate = source / name
        if not candidate.is_file() or candidate.is_symlink():
            raise EvidencePolicyError(f"explicit public evidence file is missing or unsafe: {name}")
        selected[name] = candidate
    if "result.json" not in selected:
        raise EvidencePolicyError("result.json is required for a public evidence claim boundary")
    return [selected[name] for name in sorted(selected)]


def _default_literals() -> tuple[str, ...]:
    values = {
        str(Path.home()),
        getpass.getuser(),
        socket.gethostname(),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    }
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def _redaction_literals(configured: Sequence[str]) -> tuple[str, ...]:
    values = {*_default_literals()}
    for value in configured:
        if not isinstance(value, str) or not value:
            raise EvidencePolicyError("redaction literals must be non-empty strings")
        if value == REDACTION:
            raise EvidencePolicyError("a redaction literal cannot equal the replacement token")
        values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_sensitive_key(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}{match.group('quote')}{REDACTION}{match.group('quote')}"


def _sanitize_text(text: str, literals: Sequence[str]) -> tuple[str, int]:
    sanitized, count = _LOCAL_PATH.subn(REDACTION, text)
    sanitized, key_count = _SENSITIVE_KEY_VALUE.subn(_redact_sensitive_key, sanitized)
    count += key_count
    for literal in literals:
        occurrences = sanitized.count(literal)
        if occurrences:
            sanitized = sanitized.replace(literal, REDACTION)
            count += occurrences
    return sanitized, count


def _forbidden_findings(text: str, literals: Sequence[str]) -> list[str]:
    decoded = html.unescape(text)
    findings: list[str] = []
    path_match = _LOCAL_PATH.search(decoded)
    if path_match:
        findings.append(f"local_path:{path_match.group(0)[:120]}")
    for literal in literals:
        if literal in decoded:
            findings.append(
                f"configured_literal:{hashlib.sha256(literal.encode()).hexdigest()[:12]}"
            )
    for match in _SENSITIVE_KEY_VALUE.finditer(decoded):
        if match.group("value").strip() != REDACTION:
            findings.append(f"sensitive_key:{match.group('prefix').strip()}")
    return findings


def _validate_text_format(name: str, text: str) -> None:
    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".json":
            json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            yaml.safe_load(text)
        elif suffix == ".svg":
            ET.fromstring(text)
        elif suffix == ".html":
            parser = _HTMLValidator(convert_charrefs=True)
            parser.feed(text)
            parser.close()
    except Exception as exc:
        raise EvidencePolicyError(f"sanitized {name} is not valid {suffix} text: {exc}") from exc


def _read_sanitized_file(
    path: Path,
    *,
    max_file_bytes: int,
    literals: Sequence[str],
) -> tuple[bytes, dict[str, Any]]:
    size = path.stat().st_size
    if size > max_file_bytes:
        raise EvidencePolicyError(
            f"public evidence file exceeds {max_file_bytes} bytes: {path.name} ({size})"
        )
    payload = path.read_bytes()
    if b"\x00" in payload:
        raise EvidencePolicyError(f"binary/NUL content is forbidden: {path.name}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidencePolicyError(f"public evidence must be UTF-8 text: {path.name}") from exc
    sanitized, redaction_count = _sanitize_text(text, literals)
    findings = _forbidden_findings(sanitized, literals)
    if findings:
        raise EvidenceRedactionError(f"forbidden content remains in {path.name}: {findings}")
    _validate_text_format(path.name, sanitized)
    public_bytes = sanitized.encode("utf-8")
    return public_bytes, {
        "path": path.name,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "public_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "source_bytes": len(payload),
        "public_bytes": len(public_bytes),
        "redaction_count": redaction_count,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sanitize_claim_boundary(
    value: Mapping[str, Any], literals: Sequence[str]
) -> tuple[dict[str, Any], int]:
    text = json.dumps(dict(value), ensure_ascii=False)
    sanitized, count = _sanitize_text(text, literals)
    findings = _forbidden_findings(sanitized, literals)
    if findings:
        raise EvidenceRedactionError(f"forbidden content remains in claim boundary: {findings}")
    decoded = json.loads(sanitized)
    if not isinstance(decoded, dict) or not decoded.get("evidenced"):
        raise EvidencePolicyError("claim boundary requires a non-empty evidenced field")
    not_evidenced = decoded.get("not_evidenced")
    if not isinstance(not_evidenced, list) or not not_evidenced:
        raise EvidencePolicyError("claim boundary requires a non-empty not_evidenced list")
    return decoded, count


def _extract_claim_boundary(
    sanitized_result: bytes,
    explicit: Mapping[str, Any] | None,
    literals: Sequence[str],
) -> tuple[dict[str, Any], str, int]:
    result = json.loads(sanitized_result.decode("utf-8"))
    source_boundary = result.get("claim_boundary") if isinstance(result, dict) else None
    if explicit is not None:
        boundary, count = _sanitize_claim_boundary(explicit, literals)
        return boundary, "explicit_export_argument", count
    if not isinstance(source_boundary, dict):
        raise EvidencePolicyError(
            "source result.json has no claim_boundary; provide an explicit public claim boundary"
        )
    boundary, count = _sanitize_claim_boundary(source_boundary, literals)
    return boundary, "sanitized_result.json", count


def _scan_staging(stage: Path, literals: Sequence[str]) -> None:
    for path in sorted(stage.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise EvidencePolicyError(f"unexpected staging entry: {path.name}")
        payload = path.read_bytes()
        if b"\x00" in payload:
            raise EvidencePolicyError(f"binary staging output is forbidden: {path.name}")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidencePolicyError(f"staging output is not UTF-8: {path.name}") from exc
        findings = _forbidden_findings(text, literals)
        if findings:
            raise EvidenceRedactionError(
                f"post-export scan found forbidden content in {path.name}: {findings}"
            )


def _publish(stage: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise EvidencePolicyError("existing destination must be one real directory")
        _reject_tree_symlinks(destination)
        os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except Exception:
            os.replace(backup, destination)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(stage, destination)


def export_public_evidence(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    additional_files: Sequence[str] = (),
    redact_literals: Sequence[str] = (),
    claim_boundary: Mapping[str, Any] | None = None,
    max_file_bytes: int = MAX_PUBLIC_FILE_BYTES,
) -> PublicEvidenceExport:
    if max_file_bytes < 1 or max_file_bytes > MAX_PUBLIC_FILE_BYTES:
        raise EvidencePolicyError(f"max_file_bytes must be between 1 and {MAX_PUBLIC_FILE_BYTES}")
    source_input = Path(source_dir).expanduser().absolute()
    destination = Path(destination_dir).expanduser().absolute()
    _ensure_no_symlink_components(source_input)
    _ensure_no_symlink_components(destination)
    if not source_input.is_dir() or source_input.is_symlink():
        raise EvidencePolicyError(f"source artifact directory is missing or unsafe: {source_input}")
    source = source_input.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if (
        destination_resolved == source
        or destination_resolved.is_relative_to(source)
        or source.is_relative_to(destination_resolved)
    ):
        raise EvidencePolicyError("source and destination directories must not overlap")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _ensure_no_symlink_components(destination.parent)
    literals = _redaction_literals(redact_literals)
    selected = _selected_files(source, additional_files)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    records: list[dict[str, Any]] = []
    public_payloads: dict[str, bytes] = {}
    try:
        for source_file in selected:
            public_bytes, record = _read_sanitized_file(
                source_file,
                max_file_bytes=max_file_bytes,
                literals=literals,
            )
            target = stage / source_file.name
            if target.parent != stage or target.name != source_file.name:
                raise EvidencePolicyError(f"destination escape rejected: {source_file.name}")
            _atomic_write(target, public_bytes)
            records.append(record)
            public_payloads[source_file.name] = public_bytes

        boundary, boundary_source, boundary_redactions = _extract_claim_boundary(
            public_payloads["result.json"], claim_boundary, literals
        )
        file_redactions = sum(int(record["redaction_count"]) for record in records)
        total_redactions = file_redactions + boundary_redactions
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "generator": "fmlab.release.export_public_evidence",
            "source_artifact_name": source.name,
            "destination_name": destination.name,
            "claim_boundary": boundary,
            "claim_boundary_source": boundary_source,
            "redaction": {
                "file_redaction_count": file_redactions,
                "manifest_metadata_redaction_count": boundary_redactions,
                "total_redaction_count": total_redactions,
                "configured_literal_count": len(literals),
                "replacement": REDACTION,
            },
            "policy": {
                "default_exact_names": list(DEFAULT_EXACT_NAMES),
                "default_globs": ["*.svg"],
                "explicit_additional_files": sorted(set(additional_files)),
                "max_file_bytes": max_file_bytes,
                "utf8_text_only": True,
                "symlinks_forbidden": True,
                "jsonl_model_weights_hidden_traces_forbidden": True,
            },
            "files": records,
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write(stage / "evidence_manifest.json", manifest_bytes)
        _scan_staging(stage, literals)
        _publish(stage, destination)
    except Exception as exc:
        # A failed export is deliberately left under its hidden staging name. The public
        # destination is untouched and reviewers cannot mistake partial output for a release.
        if isinstance(exc, EvidenceExportError):
            raise
        raise EvidenceExportError(
            f"public evidence export failed in staging {stage}: {exc}"
        ) from exc
    return PublicEvidenceExport(
        destination=destination,
        manifest=destination / "evidence_manifest.json",
        files=tuple([record["path"] for record in records] + ["evidence_manifest.json"]),
        redaction_count=total_redactions,
    )
