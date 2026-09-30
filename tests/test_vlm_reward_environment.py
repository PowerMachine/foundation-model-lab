from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmlab.vlm.lora_data import load_canonical_vlm_manifest
from fmlab.vlm.reward_environment import (
    RewardLabConfig,
    VisualKnowledgeEnvironment,
    VisualTaskSpec,
    parse_action,
    run_visual_reward_lab,
    score_response,
    scripted_candidates,
)


def _manifest(tmp_path: Path) -> Path:
    rows = []
    for index in range(6):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(f"pixels-{index}".encode())
        if index == 4:
            rows.append(
                {
                    "id": "ground-present",
                    "task": "grounding",
                    "image_path": str(image),
                    "object_name": "blue square",
                    "present": True,
                    "bbox_xyxy": [10, 20, 80, 90],
                }
            )
        elif index == 5:
            rows.append(
                {
                    "id": "ground-absent",
                    "task": "grounding",
                    "image_path": str(image),
                    "object_name": "red circle",
                    "present": False,
                }
            )
        else:
            task = "document_vqa" if index % 2 == 0 else "chart_qa"
            rows.append(
                {
                    "id": f"row-{index}",
                    "task": task,
                    "image_path": str(image),
                    "qa": [
                        {
                            "id": f"qa-{index}",
                            "question": "What is the value?",
                            "answer": str(40 + index),
                            "answer_type": "number",
                        }
                    ],
                }
            )
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_environment_hides_reference_and_robust_reward_rejects_hack(tmp_path: Path) -> None:
    example = load_canonical_vlm_manifest(_manifest(tmp_path))[0]
    task = VisualTaskSpec.from_example(example, split="eval")
    environment = VisualKnowledgeEnvironment([task])
    observation = environment.reset(task.id)
    assert "reference" not in observation
    assert "metadata" not in observation

    oracle = environment.step(scripted_candidates(task)["oracle"])
    with pytest.raises(RuntimeError, match="terminated"):
        environment.step(scripted_candidates(task)["reward_hacker"])
    environment.reset(task.id)
    attack = environment.step(scripted_candidates(task)["reward_hacker"])
    assert oracle.true_success and oracle.robust_reward > 0.9
    assert attack.naive_reward == 1.0
    assert attack.robust_reward < 0.0
    assert attack.hack_detected


def test_grounding_requires_valid_bbox_for_present_object(tmp_path: Path) -> None:
    examples = load_canonical_vlm_manifest(_manifest(tmp_path))
    example = next(item for item in examples if item.id == "ground-present")
    task = VisualTaskSpec.from_example(example, split="eval")
    good = score_response(task, scripted_candidates(task)["oracle"])
    missing = score_response(task, json.dumps({"answer": "yes", "confidence": 0.9}))
    assert good.true_success and good.grounding_score == 1.0
    assert not missing.true_success and missing.grounding_score == 0.0


def test_action_parser_is_strict() -> None:
    assert not parse_action("not json").valid
    assert not parse_action(json.dumps({"answer": "x", "confidence": 2})).valid
    assert not parse_action(json.dumps({"answer": "x", "confidence": 0.5, "reward": 1})).valid
    assert parse_action(json.dumps({"answer": "x", "confidence": 0.5})).valid


def test_reward_lab_writes_disjoint_preferences_and_audit(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = run_visual_reward_lab(
        RewardLabConfig(
            manifest_path=_manifest(tmp_path),
            output_dir=output,
            max_train_examples=4,
            max_eval_examples=2,
            bootstrap_samples=200,
        )
    )
    assert result.status == "completed"
    assert result.metrics["dataset"]["source_overlap"] == []
    assert (
        result.metrics["preference_accuracy"]["robust"]
        > result.metrics["preference_accuracy"]["naive"]
    )
    assert result.metrics["reward_hacking"]["naive_false_acceptance_rate"] == 1.0
    assert result.metrics["reward_hacking"]["robust_false_acceptance_rate"] == 0.0
    assert (output / "report.html").is_file()
    preference_rows = [
        json.loads(line) for line in (output / "preference_pairs.jsonl").read_text().splitlines()
    ]
    audit_rows = [
        json.loads(line) for line in (output / "reward_audit.jsonl").read_text().splitlines()
    ]
    assert preference_rows and all(row["split"] == "train" for row in preference_rows)
    assert not {row["source_id"] for row in preference_rows} & {
        row["source_id"] for row in audit_rows
    }


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown visual reward"):
        RewardLabConfig.from_mapping(
            {
                "manifest_path": str(_manifest(tmp_path)),
                "output_dir": str(tmp_path / "out"),
                "surprise": True,
            }
        )
