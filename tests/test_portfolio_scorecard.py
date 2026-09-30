from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fmlab.portfolio import (
    ScorecardError,
    build_portfolio_scorecard,
    render_scorecard_svg,
    write_portfolio_scorecard,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_EVIDENCE = REPOSITORY_ROOT / "public-evidence"
CANONICAL_SCORECARD = CANONICAL_EVIDENCE / "meta" / "portfolio-scorecard"


def _reseal_result(bundle: Path, result: dict[str, object]) -> None:
    result_path = bundle / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    manifest_path = bundle / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in manifest["files"] if row["path"] == "result.json")
    record["public_sha256"] = digest
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_canonical_scorecard_is_deterministic_typed_and_honest(tmp_path: Path) -> None:
    first = build_portfolio_scorecard(CANONICAL_EVIDENCE)
    second = build_portfolio_scorecard(CANONICAL_EVIDENCE)
    assert first == second
    assert first["schema"] == "fmlab.portfolio-scorecard/v1"
    assert first["source_set"]["bundle_count"] == 5
    assert first["source_set"]["verified_file_count"] == 26
    assert first["source_set"]["integrity"].startswith("all public files verified")
    assert first["evidence_modes"]["component_counts"] == {
        "actual": 4,
        "scripted": 2,
        "simulated": 2,
    }

    qwen = first["performance"]["qwen3_vl_lora_actual_gpu_smoke"]
    assert qwen["evidence_tier"] == "actual_gpu_smoke"
    assert qwen["quality_improvement_claimed"] is False
    assert qwen["heldout_paired_examples"] == 3
    assert qwen["heldout_exact_match_delta"] == 0.0
    assert qwen["peak_allocated_gib"] == pytest.approx(17.445631980895996)

    reward = first["performance"]["visual_reward_scripted"]
    assert reward["absolute_delta"] == pytest.approx(0.5416666666666666)
    assert reward["bootstrap_95_ci"] == pytest.approx([0.3958333333333333, 0.6875])
    assert reward["naive_false_acceptance_rate"] == 1.0
    assert reward["robust_false_acceptance_rate"] == 0.0

    agent = first["performance"]["agent_scripted_controls_on_actual_environment"]
    assert agent["oracle_scripted_control"]["pass_at_1"] == 1.0
    assert agent["shortcut_scripted_control"]["pass_at_1"] == 0.0

    ddp = first["performance"]["ddp_actual_cpu_gloo"]
    assert ddp["all_correctness_gates_pass"] is True
    assert ddp["passed_gate_count"] == 11
    assert ddp["resume_state_max_absolute_error"] == 0.0

    capacity = first["performance"]["inference_capacity_simulated"]
    assert capacity["row_count"] == 81
    assert capacity["paired_trace_count"] == 27
    assert capacity["policy_request_executions"] == 7776
    assert capacity["knees"]["continuous_fcfs"]["last_sustainable_configured_rate_rps"] == 32.0
    capacity_uncertainty = capacity["last_sustainable_continuous_uncertainty"]
    assert capacity_uncertainty["e2e_p99_ms"]["n"] == 3
    assert capacity_uncertainty["e2e_p99_ms"]["mean"] == pytest.approx(193.85821535735363)
    assert capacity_uncertainty["e2e_p99_ms"]["population_stddev"] == pytest.approx(
        2.83111713694164
    )

    svg = render_scorecard_svg(first)
    ET.fromstring(svg)
    assert "ACTUAL GPU SMOKE" in svg
    assert "no quality improvement claimed" in svg
    assert "SIMULATED" in svg

    json_path, svg_path = write_portfolio_scorecard(CANONICAL_EVIDENCE, tmp_path / "out")
    first_json = json_path.read_bytes()
    first_svg = svg_path.read_bytes()
    write_portfolio_scorecard(CANONICAL_EVIDENCE, tmp_path / "out")
    assert json_path.read_bytes() == first_json
    assert svg_path.read_bytes() == first_svg
    manifest = json.loads((tmp_path / "out" / "evidence_manifest.json").read_text())
    assert manifest["destination_name"] == "out"
    assert [row["path"] for row in manifest["files"]] == [
        "scorecard.json",
        "scorecard.svg",
    ]
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_checked_in_scorecard_matches_manifest_verified_sources(tmp_path: Path) -> None:
    generated = tmp_path / "portfolio-scorecard"
    write_portfolio_scorecard(CANONICAL_EVIDENCE, generated)

    for name in ("scorecard.json", "scorecard.svg", "evidence_manifest.json"):
        assert (CANONICAL_SCORECARD / name).read_bytes() == (generated / name).read_bytes()

    assert sorted(path.name for path in CANONICAL_SCORECARD.iterdir()) == [
        "evidence_manifest.json",
        "scorecard.json",
        "scorecard.svg",
    ]


def test_manifest_tampering_is_rejected_before_metrics_are_used(tmp_path: Path) -> None:
    evidence = tmp_path / "public-evidence"
    shutil.copytree(CANONICAL_EVIDENCE, evidence)
    result = evidence / "distributed" / "ddp-correctness" / "result.json"
    result.write_bytes(result.read_bytes() + b"\n")

    with pytest.raises(ScorecardError, match="manifest SHA-256 mismatch"):
        build_portfolio_scorecard(evidence)


@pytest.mark.parametrize(
    ("bundle_relative", "mutation", "message"),
    [
        (
            "agent/reliable-agent-benchmark",
            lambda value: value.update({"schema": "fmlab.reliable-agent-benchmark/v2"}),
            "reliable-agent-benchmark/v1",
        ),
        (
            "vlm/visual-reward-environment",
            lambda value: value["metrics"]["preference_accuracy"]["robust_minus_naive"].pop(
                "delta"
            ),
            "missing required field 'delta'",
        ),
    ],
)
def test_resealed_schema_or_metric_drift_fails_closed(
    tmp_path: Path,
    bundle_relative: str,
    mutation: object,
    message: str,
) -> None:
    evidence = tmp_path / "public-evidence"
    shutil.copytree(CANONICAL_EVIDENCE, evidence)
    bundle = evidence / bundle_relative
    result = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    mutation(result)  # type: ignore[operator]
    _reseal_result(bundle, result)

    with pytest.raises(ScorecardError, match=message):
        build_portfolio_scorecard(evidence)
