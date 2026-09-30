from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .benchmark_types import LEDGER_SCHEMA, canonical_json, sha256_json


class AtomicJsonlStore:
    """Hash-verified JSONL store using lock + fsync + same-filesystem replace."""

    def __init__(
        self,
        path: str | Path,
        *,
        key_field: str,
        schema: str,
    ) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.schema = schema
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _normalize(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.key_field not in value or not str(value[self.key_field]):
            raise ValueError(f"record requires {self.key_field}")
        record = dict(value)
        record["_store_schema"] = self.schema
        record.pop("_record_sha256", None)
        record["_record_sha256"] = sha256_json(record)
        return record

    def _read_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if record.get("_store_schema") != self.schema:
                raise ValueError(f"store schema mismatch at line {line_number}")
            claimed = record.get("_record_sha256")
            payload = dict(record)
            payload.pop("_record_sha256", None)
            if claimed != sha256_json(payload):
                raise ValueError(f"record hash mismatch at line {line_number}")
            key = str(record.get(self.key_field, ""))
            if not key or key in records:
                raise ValueError(f"missing or duplicate key at line {line_number}: {key!r}")
            records[key] = record
        return records

    def read_all(self) -> dict[str, dict[str, Any]]:
        with self._lock(exclusive=False):
            return self._read_unlocked()

    def _write_unlocked(self, records: dict[str, dict[str, Any]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for key in sorted(records):
                    stream.write(canonical_json(records[key]) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()

    def upsert(self, value: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(value)
        key = str(normalized[self.key_field])
        with self._lock(exclusive=True):
            records = self._read_unlocked()
            records[key] = normalized
            self._write_unlocked(records)
        return normalized


class AtomicResumeLedger(AtomicJsonlStore):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path, key_field="episode_id", schema=LEDGER_SCHEMA)
