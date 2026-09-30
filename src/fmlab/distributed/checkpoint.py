from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .core import SCHEMA_VERSION


class CheckpointError(RuntimeError):
    """Base class for fail-closed checkpoint validation errors."""


class CheckpointIntegrityError(CheckpointError):
    """The checkpoint bytes or schema are invalid."""


class CheckpointContractError(CheckpointError):
    """The checkpoint belongs to a different data or training contract."""


@dataclass(frozen=True)
class CheckpointWriteMetrics:
    path: str
    checksum_path: str
    sha256: str
    bytes: int
    save_seconds: float
    atomic_replace: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "checksum_path": self.checksum_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "save_seconds": self.save_seconds,
            "atomic_replace": self.atomic_replace,
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_atomic_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> CheckpointWriteMetrics:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    started = time.perf_counter()
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        checksum = file_sha256(target)
        checksum_path = target.with_suffix(target.suffix + ".sha256")
        _atomic_text(checksum_path, f"{checksum}  {target.name}\n")
    finally:
        if temporary.exists():
            temporary.unlink()
    return CheckpointWriteMetrics(
        path=str(target),
        checksum_path=str(checksum_path),
        sha256=checksum,
        bytes=target.stat().st_size,
        save_seconds=time.perf_counter() - started,
    )


def _expected_checksum(path: Path) -> str:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise CheckpointIntegrityError(f"checkpoint checksum sidecar is missing: {checksum_path}")
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or len(fields[0]) != 64:
        raise CheckpointIntegrityError("checkpoint checksum sidecar is malformed")
    return fields[0]


def load_validated_checkpoint(
    path: str | Path,
    *,
    expected_config_contract_hash: str,
    expected_dataset_contract_hash: str,
    expected_combined_contract_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path)
    started = time.perf_counter()
    if not target.is_file():
        raise CheckpointIntegrityError(f"checkpoint does not exist: {target}")
    expected_sha = _expected_checksum(target)
    observed_sha = file_sha256(target)
    if observed_sha != expected_sha:
        raise CheckpointIntegrityError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointIntegrityError(f"checkpoint deserialization failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError("checkpoint payload must be a mapping")
    required = {
        "schema_version",
        "config_contract_hash",
        "dataset_contract_hash",
        "combined_contract_hash",
        "model_state",
        "optimizer_state",
        "torch_rng_by_rank",
        "python_rng_by_rank",
        "sampler_cursor",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CheckpointIntegrityError(f"checkpoint is missing required fields: {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CheckpointIntegrityError(
            f"unsupported checkpoint schema: {payload['schema_version']!r}"
        )
    contracts = {
        "config_contract_hash": expected_config_contract_hash,
        "dataset_contract_hash": expected_dataset_contract_hash,
        "combined_contract_hash": expected_combined_contract_hash,
    }
    for field, expected in contracts.items():
        observed = payload.get(field)
        if observed != expected:
            raise CheckpointContractError(
                f"{field} mismatch: expected {expected}, observed {observed}"
            )
    cursor = payload["sampler_cursor"]
    if not isinstance(cursor, dict) or set(cursor) != {
        "epoch",
        "next_microbatch",
        "optimizer_step",
    }:
        raise CheckpointIntegrityError("sampler_cursor has an invalid schema")
    if any(not isinstance(cursor[key], int) or cursor[key] < 0 for key in cursor):
        raise CheckpointIntegrityError("sampler_cursor values must be non-negative integers")
    metrics = {
        "path": str(target),
        "sha256": observed_sha,
        "bytes": target.stat().st_size,
        "load_seconds": time.perf_counter() - started,
        "integrity_verified": True,
        "contract_verified": True,
    }
    return payload, metrics
