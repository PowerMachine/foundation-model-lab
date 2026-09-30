#!/usr/bin/env python3
"""Fail-closed integrity checks for checked-in public evidence bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


MAX_FILE_BYTES = 20 * 1024 * 1024
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_SUFFIXES = {".html", ".json", ".md", ".svg", ".txt", ".yaml", ".yml"}
FORBIDDEN_TEXT = {
    "local Unix home path": re.compile(r"/(?:home|Users)/[^\s<>'\"]+"),
    "local data mount": re.compile(r"/data/[^\s<>'\"]+"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.files_checked = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path, audit: Audit) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        audit.errors.append(f"{path}: invalid UTF-8 JSON: {error}")
        return None
    if not isinstance(value, dict):
        audit.errors.append(f"{path}: top-level JSON must be an object")
        return None
    return value


def _check_text(path: Path, content: bytes, audit: Audit) -> None:
    audit.require(b"\x00" not in content, f"{path}: NUL byte is forbidden")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        audit.errors.append(f"{path}: public evidence must be UTF-8 text: {error}")
        return
    for label, pattern in FORBIDDEN_TEXT.items():
        audit.require(pattern.search(text) is None, f"{path}: detected {label}")


def _valid_boundary(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    evidenced = value.get("evidenced")
    evidenced_ok = isinstance(evidenced, str) and bool(evidenced.strip())
    evidenced_ok = evidenced_ok or (
        isinstance(evidenced, list)
        and bool(evidenced)
        and all(isinstance(item, str) and item.strip() for item in evidenced)
    )
    not_evidenced = value.get("not_evidenced")
    return evidenced_ok and (
        isinstance(not_evidenced, list)
        and bool(not_evidenced)
        and all(isinstance(item, str) and item.strip() for item in not_evidenced)
    )


def _check_bundle(root: Path, manifest: Path, index_text: str, audit: Audit) -> None:
    bundle = manifest.parent
    relative_manifest = manifest.relative_to(root).as_posix()
    audit.require(relative_manifest in index_text, f"{manifest}: missing from public index")

    data = _read_json(manifest, audit)
    if data is None:
        return
    audit.require(data.get("schema_version") == "1.0", f"{manifest}: unsupported schema")
    audit.require(
        data.get("destination_name") == bundle.name,
        f"{manifest}: destination_name must equal bundle directory",
    )
    audit.require(
        _valid_boundary(data.get("claim_boundary")), f"{manifest}: invalid claim boundary"
    )
    audit.require(
        data.get("claim_boundary_source") in {"sanitized_result.json", "explicit_export_argument"},
        f"{manifest}: unknown claim_boundary_source",
    )

    rows = data.get("files")
    if not isinstance(rows, list) or not rows:
        audit.errors.append(f"{manifest}: files must be a non-empty list")
        return

    declared: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            audit.errors.append(f"{manifest}: files[{index}] must be an object")
            continue
        name = row.get("path")
        if not isinstance(name, str):
            audit.errors.append(f"{manifest}: files[{index}].path must be a string")
            continue
        pure = PurePosixPath(name)
        safe_name = (
            not pure.is_absolute()
            and len(pure.parts) == 1
            and pure.name not in {"", ".", "..", manifest.name}
        )
        if not safe_name:
            audit.errors.append(f"{manifest}: unsafe public path {name!r}")
            continue
        if name in declared:
            audit.errors.append(f"{manifest}: duplicate file row {name!r}")
            continue
        declared.add(name)
        path = bundle / name
        if not path.is_file() or path.is_symlink():
            audit.errors.append(f"{manifest}: missing or symlinked file {name!r}")
            continue
        audit.require(path.suffix.lower() in SAFE_SUFFIXES, f"{path}: unsafe file extension")
        content = path.read_bytes()
        audit.files_checked += 1
        audit.require(len(content) <= MAX_FILE_BYTES, f"{path}: exceeds 20 MiB")
        audit.require(row.get("public_bytes") == len(content), f"{path}: byte count mismatch")
        public_sha = row.get("public_sha256")
        source_sha = row.get("source_sha256")
        audit.require(
            isinstance(public_sha, str) and HEX_SHA256.fullmatch(public_sha) is not None,
            f"{path}: invalid public_sha256",
        )
        audit.require(
            isinstance(source_sha, str) and HEX_SHA256.fullmatch(source_sha) is not None,
            f"{path}: invalid source_sha256",
        )
        audit.require(public_sha == _sha256(content), f"{path}: SHA-256 mismatch")
        _check_text(path, content, audit)

    actual = {path.name for path in bundle.iterdir() if path.is_file() and path != manifest}
    unexpected_directories = [path.name for path in bundle.iterdir() if path.is_dir()]
    audit.require(
        actual == declared, f"{manifest}: declared files differ from bundle: {actual ^ declared}"
    )
    audit.require(not unexpected_directories, f"{manifest}: nested directories are forbidden")
    _check_text(manifest, manifest.read_bytes(), audit)


def verify(root: Path) -> Audit:
    audit = Audit()
    audit.require(
        root.is_dir() and not root.is_symlink(), f"{root}: evidence root is missing/unsafe"
    )
    if audit.errors:
        return audit

    index = root / "README.md"
    audit.require(index.is_file() and not index.is_symlink(), f"{index}: public index is required")
    if not index.is_file():
        return audit
    index_text = index.read_text(encoding="utf-8")
    _check_text(index, index_text.encode(), audit)

    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    audit.require(not symlinks, f"{root}: symlinks are forbidden: {symlinks}")

    manifests = sorted(root.rglob("evidence_manifest.json"))
    audit.require(bool(manifests), f"{root}: no evidence manifests found")
    for manifest in manifests:
        relative = manifest.relative_to(root)
        audit.require(
            len(relative.parts) >= 3,
            f"{manifest}: manifest must be nested under a track and run directory",
        )
        _check_bundle(root, manifest, index_text, audit)

    covered = {manifest.parent.resolve() for manifest in manifests}
    for path in root.rglob("*"):
        if not path.is_file() or path == index:
            continue
        audit.require(
            path.parent.resolve() in covered,
            f"{path}: file is outside a manifested evidence bundle",
        )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("public-evidence"))
    args = parser.parse_args(argv)
    audit = verify(args.root.resolve())
    if audit.errors:
        for error in audit.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    manifests = len(list(args.root.rglob("evidence_manifest.json")))
    print(f"public evidence OK: {manifests} bundles, {audit.files_checked} hashed files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
