from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal


TASK_SPEC_SCHEMA = "fmlab.agent-task/v1"
LEDGER_SCHEMA = "fmlab.agent-ledger/v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or "\x00" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"asset path must be a safe relative POSIX path: {value!r}")


@dataclass(frozen=True)
class FileAsset:
    path: str
    content: str
    role: Literal["source", "public_test"] = "source"

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if self.role not in {"source", "public_test"}:
            raise ValueError(f"unsupported asset role: {self.role}")

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def fingerprint_payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "content": self.content,
            "role": self.role,
        }


@dataclass(frozen=True)
class HiddenCase:
    """JSON-encoded hidden input and expected value.

    JSON strings keep the frozen task specification deeply immutable. Expected values stay
    in the scorer process and are never written into the agent workspace or grader assets.
    """

    args_json: str
    expected_json: str

    def __post_init__(self) -> None:
        args = json.loads(self.args_json)
        json.loads(self.expected_json)
        if not isinstance(args, list):
            raise ValueError("hidden case args_json must encode a list")
        if canonical_json(args) != self.args_json:
            raise ValueError("hidden case args_json must use canonical JSON")
        if canonical_json(json.loads(self.expected_json)) != self.expected_json:
            raise ValueError("hidden case expected_json must use canonical JSON")

    @classmethod
    def build(cls, args: list[Any], expected: Any) -> HiddenCase:
        return cls(canonical_json(args), canonical_json(expected))

    def args(self) -> list[Any]:
        return json.loads(self.args_json)

    def expected(self) -> Any:
        return json.loads(self.expected_json)


@dataclass(frozen=True)
class TextPatch:
    path: str
    old: str
    new: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if not self.old or self.old == self.new:
            raise ValueError("reference patch must replace non-empty text with a new value")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    perturbation_id: str
    instruction: str
    module: str
    function: str
    public_test_path: str
    assets: tuple[FileAsset, ...]
    hidden_cases: tuple[HiddenCase, ...]
    reference_patch: TextPatch
    tags: tuple[str, ...] = ()
    schema_version: str = TASK_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TASK_SPEC_SCHEMA:
            raise ValueError(f"unsupported task schema: {self.schema_version}")
        if not self.task_id or not self.perturbation_id or not self.instruction:
            raise ValueError("task identity and instruction must be non-empty")
        _validate_relative_path(self.public_test_path)
        paths = [asset.path for asset in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("task assets must have unique paths")
        if self.public_test_path not in paths:
            raise ValueError("public_test_path must name a declared asset")
        if not any(
            asset.path == self.public_test_path and asset.role == "public_test"
            for asset in self.assets
        ):
            raise ValueError("public_test_path must have the public_test role")
        if self.reference_patch.path not in paths:
            raise ValueError("reference patch must target a declared asset")
        if not self.hidden_cases:
            raise ValueError("at least one hidden case is required")

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "perturbation_id": self.perturbation_id,
            "instruction": self.instruction,
            "module": self.module,
            "function": self.function,
            "public_test_path": self.public_test_path,
            "assets": [asset.fingerprint_payload() for asset in self.assets],
            "hidden_cases": [asdict(case) for case in self.hidden_cases],
            "reference_patch": asdict(self.reference_patch),
            "tags": list(self.tags),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.fingerprint_payload())

    @property
    def uid(self) -> str:
        return f"{self.task_id}:{self.perturbation_id}:{self.fingerprint[:16]}"

    def public_manifest(self) -> dict[str, Any]:
        """Publish commitments without disclosing hidden expected values."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "perturbation_id": self.perturbation_id,
            "task_uid": self.uid,
            "fingerprint_sha256": self.fingerprint,
            "instruction": self.instruction,
            "assets": [
                {
                    "path": asset.path,
                    "role": asset.role,
                    "content_sha256": asset.content_sha256,
                }
                for asset in self.assets
            ],
            "hidden_case_count": len(self.hidden_cases),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ObservedFile:
    path: str
    role: str
    content: str
    content_sha256: str


@dataclass(frozen=True)
class Observation:
    schema_version: str
    task_uid: str
    task_fingerprint_sha256: str
    instruction: str
    files: tuple[ObservedFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_uid": self.task_uid,
            "task_fingerprint_sha256": self.task_fingerprint_sha256,
            "instruction": self.instruction,
            "files": [asdict(item) for item in self.files],
        }


@dataclass(frozen=True)
class ActionRecord:
    step: int
    tool: str
    args: dict[str, Any]
    ok: bool
    output: str
    duration_seconds: float
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeGrade:
    passed: bool
    cases_total: int
    cases_passed: int
    failures: tuple[dict[str, Any], ...]
    duration_seconds: float
    isolation: str
    network_isolated: bool
    hidden_assets_separate: bool = True
    hidden_assets_materialized_after_policy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessGrade:
    passed: bool
    steps: int
    tool_calls: int
    tool_errors: int
    unsafe_action_attempts: int
    reward_hack_signals: tuple[str, ...]
    public_validation_after_last_change: bool
    trace_integrity: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeGrade:
    passed: bool
    outcome: OutcomeGrade
    process: ProcessGrade
    failure_category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "outcome": self.outcome.to_dict(),
            "process": self.process.to_dict(),
            "failure_category": self.failure_category,
        }
