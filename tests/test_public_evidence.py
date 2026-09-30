from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import fmlab.release as release

from fmlab.release import (
    EvidencePolicyError,
    EvidenceRedactionError,
    export_public_evidence,
    file_sha256,
)


def _write_source(
    root: Path,
    *,
    claim_boundary: bool = True,
    report_text: str | None = None,
) -> Path:
    root.mkdir()
    result: dict[str, object] = {
        "experiment": "public-test",
        "status": "completed",
        "artifact_path": "/data/alice/private/run",
        "artifacts": ["events.jsonl", "model.pt", "view.svg"],
    }
    if claim_boundary:
        result["claim_boundary"] = {
            "evidenced": "Measured wiring at /home/alice/research.",
            "not_evidenced": ["production quality on build-host"],
        }
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "username": "alice",
                "hostname": "build-host",
                "workspace": "/home/alice/project",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "report.html").write_text(
        report_text or "<html><body>source=/data/alice/private/run host=build-host</body></html>",
        encoding="utf-8",
    )
    (root / "view.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>alice</text></svg>',
        encoding="utf-8",
    )
    (root / "environment_spec.json").write_text(
        json.dumps({"manifest": "/data/alice/dataset.jsonl"}), encoding="utf-8"
    )
    (root / "events.jsonl").write_text('{"hidden": true}\n', encoding="utf-8")
    (root / "model.pt").write_bytes(b"not-a-public-model")
    (root / "hidden_trace.txt").write_text("secret trace", encoding="utf-8")
    return root


def test_export_is_allowlisted_redacted_hashed_and_atomically_replaces_destination(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    destination = tmp_path / "public"
    destination.mkdir()
    (destination / "stale.txt").write_text("old release", encoding="utf-8")
    original_result_sha = file_sha256(source / "result.json")

    exported = export_public_evidence(
        source,
        destination,
        additional_files=["environment_spec.json"],
        redact_literals=["alice", "build-host"],
    )

    expected = {
        "environment_spec.json",
        "evidence_manifest.json",
        "provenance.json",
        "report.html",
        "result.json",
        "view.svg",
    }
    assert {path.name for path in destination.iterdir()} == expected
    assert set(exported.files) == expected
    assert not (destination / "stale.txt").exists()
    assert not (destination / "events.jsonl").exists()
    assert not (destination / "model.pt").exists()
    assert not (destination / "hidden_trace.txt").exists()
    assert exported.redaction_count > 0

    for path in destination.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "/data/" not in text
        assert "alice" not in text
        assert "build-host" not in text

    manifest = json.loads((destination / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["source_artifact_name"] == "source"
    assert manifest["claim_boundary_source"] == "sanitized_result.json"
    assert manifest["redaction"]["total_redaction_count"] == exported.redaction_count
    assert manifest["policy"]["jsonl_model_weights_hidden_traces_forbidden"] is True
    records = {row["path"]: row for row in manifest["files"]}
    assert records["result.json"]["source_sha256"] == original_result_sha
    assert records["result.json"]["public_sha256"] == file_sha256(destination / "result.json")
    assert any(row["source_sha256"] != row["public_sha256"] for row in records.values())
    for name, row in records.items():
        assert row["public_sha256"] == hashlib.sha256((destination / name).read_bytes()).hexdigest()

    # Source evidence is immutable; export only writes the destination/staging sibling.
    assert file_sha256(source / "result.json") == original_result_sha


def test_json_redaction_preserves_colliding_path_key_entries() -> None:
    source = '{"inputs":{"/data/private/a.png":"aaa","/data/private/b.png":"bbb"}}\n'
    sanitized, count = release._sanitize_text(source, ())
    repaired, repairs = release._repair_duplicate_json_keys(sanitized)
    decoded = json.loads(repaired)

    assert count == 2
    assert repairs == 1
    assert decoded["inputs"] == {
        "REDACTED_LOCAL": "aaa",
        "REDACTED_LOCAL_002": "bbb",
    }


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../escape.json", "basename"),
        ("events.jsonl", "extension is forbidden"),
        ("hidden_trace.txt", "style filename is forbidden"),
        ("table.csv", "unexpected public evidence extension"),
    ],
)
def test_explicit_additional_file_policy_rejects_escape_traces_and_extensions(
    tmp_path: Path, name: str, message: str
) -> None:
    source = _write_source(tmp_path / "source")
    (source / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(EvidencePolicyError, match=message):
        export_public_evidence(
            source,
            tmp_path / "public",
            additional_files=[name],
            redact_literals=["alice", "build-host"],
        )


def test_selected_binary_size_and_any_source_symlink_are_fail_closed(tmp_path: Path) -> None:
    binary_source = _write_source(tmp_path / "binary")
    (binary_source / "notes.txt").write_bytes(b"safe-prefix\x00binary")
    with pytest.raises(EvidencePolicyError, match="binary/NUL"):
        export_public_evidence(
            binary_source,
            tmp_path / "binary-public",
            additional_files=["notes.txt"],
            redact_literals=["alice", "build-host"],
        )

    large_source = _write_source(tmp_path / "large")
    (large_source / "notes.txt").write_text("x" * 129, encoding="utf-8")
    with pytest.raises(EvidencePolicyError, match="exceeds 128 bytes"):
        export_public_evidence(
            large_source,
            tmp_path / "large-public",
            additional_files=["notes.txt"],
            redact_literals=["alice", "build-host"],
            max_file_bytes=128,
        )

    symlink_source = _write_source(tmp_path / "symlink")
    (symlink_source / "unsafe-link").symlink_to(symlink_source / "result.json")
    with pytest.raises(EvidencePolicyError, match="symlink entry"):
        export_public_evidence(
            symlink_source,
            tmp_path / "symlink-public",
            redact_literals=["alice", "build-host"],
        )


def test_post_scan_failure_keeps_existing_destination_and_partial_output_in_staging(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path / "source",
        report_text=("<html><body>encoded=&#47;home&#47;alice&#47;private</body></html>"),
    )
    destination = tmp_path / "public"
    destination.mkdir()
    marker = destination / "previous-release.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(EvidenceRedactionError, match="forbidden content remains"):
        export_public_evidence(
            source,
            destination,
            redact_literals=["alice", "build-host"],
        )

    assert marker.read_text(encoding="utf-8") == "keep me"
    staging = list(tmp_path.glob(".public.staging-*"))
    assert len(staging) == 1
    assert staging[0].is_dir()
    assert not (staging[0] / "evidence_manifest.json").exists()


def test_missing_source_claim_requires_explicit_boundary_and_destination_cannot_overlap(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source", claim_boundary=False)
    with pytest.raises(EvidencePolicyError, match="no claim_boundary"):
        export_public_evidence(
            source,
            tmp_path / "rejected-public",
            redact_literals=["alice", "build-host"],
        )

    boundary = {
        "evidenced": "Deterministic environment wiring was measured.",
        "not_evidenced": ["real-model quality", "production safety"],
    }
    destination = tmp_path / "accepted-public"
    exported = export_public_evidence(
        source,
        destination,
        redact_literals=["alice", "build-host"],
        claim_boundary=boundary,
    )
    manifest = json.loads(exported.manifest.read_text(encoding="utf-8"))
    assert manifest["claim_boundary"] == boundary
    assert manifest["claim_boundary_source"] == "explicit_export_argument"

    with pytest.raises(EvidencePolicyError, match="must not overlap"):
        export_public_evidence(
            source,
            source / "child",
            redact_literals=["alice", "build-host"],
            claim_boundary=boundary,
        )
