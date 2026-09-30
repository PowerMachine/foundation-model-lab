from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


TASKS = ("documents", "charts", "grounding")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def prepare_training_manifests(source_dataset_root: Path, output_root: Path) -> dict[str, Any]:
    """Materialize stable VLM training manifests from the offline synthetic dataset.

    The visual demo keeps relative image paths next to each task manifest.  Training
    recipes instead need paths that remain unambiguous when task rows are mixed, so
    this function copies the tiny generated images into the canonical dataset area
    and rewrites every ``image_path`` as an absolute path.
    """

    source = source_dataset_root.expanduser().resolve()
    destination = output_root.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mixed_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    manifests: dict[str, str] = {}

    for task in TASKS:
        source_task = source / task
        source_manifest = source_task / "manifest.jsonl"
        if not source_manifest.is_file():
            raise FileNotFoundError(source_manifest)
        rows = _read_jsonl(source_manifest)
        target_task = destination / task
        target_images = target_task / "images"
        source_images = source_task / "images"
        if source_images.is_dir():
            shutil.copytree(source_images, target_images, dirs_exist_ok=True)

        normalized: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            raw_image = Path(str(value.get("image_path", "")))
            if not raw_image.name:
                raise ValueError(f"{source_manifest} contains a row without image_path")
            source_image = raw_image if raw_image.is_absolute() else source_task / raw_image
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            target_image = target_images / source_image.name
            if not target_image.is_file():
                shutil.copy2(source_image, target_image)
            value["image_path"] = str(target_image.resolve())
            value.setdefault("task", task.rstrip("s"))
            normalized.append(value)

        task_manifest = _write_jsonl(target_task / "manifest.jsonl", normalized)
        manifests[task] = str(task_manifest)
        counts[task] = len(normalized)
        mixed_rows.extend(normalized)

    mixed_manifest = _write_jsonl(destination / "mixed" / "manifest.jsonl", mixed_rows)
    summary = {
        "source_dataset_root": str(source),
        "output_root": str(destination),
        "manifests": {**manifests, "mixed": str(mixed_manifest)},
        "counts": {**counts, "mixed": len(mixed_rows)},
    }
    (destination / "preparation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
