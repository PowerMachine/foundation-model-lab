from __future__ import annotations

import json
from pathlib import Path

from fmlab.vlm.manifest import TASKS, prepare_training_manifests


def test_prepare_training_manifests_copies_images_and_builds_mixed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for task in TASKS:
        images = source / task / "images"
        images.mkdir(parents=True)
        image = images / f"{task}.png"
        image.write_bytes(b"png")
        row = {"id": task, "image_path": f"images/{image.name}"}
        (source / task / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    destination = tmp_path / "prepared"
    summary = prepare_training_manifests(source, destination)

    assert summary["counts"]["mixed"] == 3
    assert (destination / "documents" / "manifest.jsonl").is_file()
    mixed = [
        json.loads(line)
        for line in (destination / "mixed" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["task"] for row in mixed} == {"document", "chart", "grounding"}
    assert all(Path(row["image_path"]).is_file() for row in mixed)
