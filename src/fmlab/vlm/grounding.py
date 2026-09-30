from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

from fmlab.artifacts import ExperimentResult

from .inference import Qwen3VLRunner
from .reporting import write_evaluation_html, write_rows, write_score_bars
from .schema import GroundingCase


OBJECT_STYLES = {
    "red circle": ("circle", "#d1495b"),
    "blue square": ("square", "#2878b5"),
    "green triangle": ("triangle", "#3c9d5d"),
    "orange diamond": ("diamond", "#e6842a"),
}


def generate_grounding_dataset(
    output_dir: Path,
    *,
    count: int = 6,
    seed: int = 29,
) -> list[GroundingCase]:
    """Render simple scenes with balanced present/absent object queries."""

    if count < 1:
        raise ValueError("count must be positive")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cases: list[GroundingCase] = []
    object_names = list(OBJECT_STYLES)
    for index in range(count):
        present_names = rng.sample(object_names, k=2)
        image_path = image_dir / f"scene-{index:03d}.png"
        boxes = _draw_scene(image_path, present_names, rng)
        absent_names = [name for name in object_names if name not in present_names]
        for name in present_names:
            cases.append(
                GroundingCase(
                    id=f"scene-{index:03d}-{_slug(name)}-present",
                    image_path=image_path,
                    object_name=name,
                    present=True,
                    bbox_xyxy=tuple(boxes[name]),
                )
            )
        # One absent query per present query keeps hallucination-rate interpretation clear.
        for name in absent_names:
            cases.append(
                GroundingCase(
                    id=f"scene-{index:03d}-{_slug(name)}-absent",
                    image_path=image_path,
                    object_name=name,
                    present=False,
                )
            )
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return cases


def evaluate_grounding(
    cases: list[GroundingCase],
    runner: Qwen3VLRunner,
    output_dir: Path,
) -> ExperimentResult:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    objects_by_image: dict[Path, list[str]] = {}
    for case in cases:
        if case.present:
            objects_by_image.setdefault(case.image_path, []).append(case.object_name)
    rows: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    for case in cases:
        result = runner.infer(
            case.question(),
            [case.image_path],
            context={
                "expected_answer": "yes" if case.present else "no",
                "objects": objects_by_image.get(case.image_path, []),
            },
        )
        predicted = parse_presence(result.text)
        if predicted is True and case.present:
            tp += 1
        elif predicted is False and not case.present:
            tn += 1
        elif predicted is True and not case.present:
            fp += 1
        else:
            fn += 1
        rows.append(
            {
                "case": case.id,
                "image_path": str(case.image_path),
                "object": case.object_name,
                "present": case.present,
                "prediction_text": result.text,
                "predicted_present": predicted,
                "correct": predicted == case.present,
                "backend": result.backend,
                "simulated": result.simulated,
            }
        )
    accuracy = (tp + tn) / len(cases) if cases else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hallucination_rate = fp / (fp + tn) if fp + tn else 0.0
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": hallucination_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    predictions = write_rows(output_dir / "predictions.jsonl", rows)
    chart = write_score_bars(
        output_dir / "grounding_metrics.svg",
        {
            key: metrics[key]
            for key in ("accuracy", "precision", "recall", "f1", "hallucination_rate")
        },
        title="Object presence and hallucination",
    )
    report = write_evaluation_html(
        output_dir / "report.html",
        title="Grounding and hallucination evaluation",
        summary=metrics,
        rows=rows,
        score_chart=chart,
        notice=(
            "Hallucination rate is false-positive / absent-query count. "
            "Offline predictions test only the evaluator."
        ),
    )
    experiment = ExperimentResult(
        experiment="vlm_grounding",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters={"cases": len(cases), "vlm_mode": runner.config.mode},
        artifacts=[str(predictions), str(chart), str(report)],
    )
    result_path = experiment.write(output_dir)
    experiment.artifacts.append(str(result_path))
    return experiment


def parse_presence(text: str) -> bool | None:
    """Parse yes/no while rejecting explanations that contain both answers."""

    tokens = re.findall(r"\b(?:yes|no)\b", text.casefold())
    if not tokens or len(set(tokens)) > 1:
        return None
    return tokens[0] == "yes"


def bbox_iou(
    first: tuple[float, float, float, float] | list[float],
    second: tuple[float, float, float, float] | list[float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def parse_bbox(text: str) -> tuple[float, float, float, float] | None:
    numbers = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if len(numbers) < 4:
        return None
    x1, y1, x2, y2 = numbers[:4]
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def evaluate_box_predictions(
    cases: Iterable[GroundingCase],
    predictions: dict[str, tuple[float, float, float, float] | None],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    scores = []
    hits = 0
    for case in cases:
        if not case.present or case.bbox_xyxy is None:
            continue
        prediction = predictions.get(case.id)
        score = bbox_iou(case.bbox_xyxy, prediction) if prediction is not None else 0.0
        scores.append(score)
        hits += int(score >= iou_threshold)
    return {
        "box_count": float(len(scores)),
        "mean_iou": sum(scores) / len(scores) if scores else 0.0,
        f"accuracy_iou@{iou_threshold:g}": hits / len(scores) if scores else 0.0,
    }


def draw_box_overlay(
    image_path: Path,
    output_path: Path,
    boxes: Iterable[tuple[tuple[int, int, int, int], str, str]],
) -> Path:
    """Draw ``(box, label, color)`` tuples for qualitative inspection."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Box overlays require Pillow") from exc
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, label, color in boxes:
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 3, box[1] + 3), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _draw_scene(path: Path, names: list[str], rng: random.Random) -> dict[str, list[int]]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Grounding data generation requires Pillow") from exc
    image = Image.new("RGB", (720, 480), "#f5f6f7")
    draw = ImageDraw.Draw(image)
    boxes: dict[str, list[int]] = {}
    anchors = [(100, 120), (430, 235)]
    rng.shuffle(anchors)
    for name, (x, y) in zip(names, anchors, strict=True):
        size = rng.randint(90, 130)
        box = [x, y, x + size, y + size]
        shape, color = OBJECT_STYLES[name]
        if shape == "circle":
            draw.ellipse(box, fill=color)
        elif shape == "square":
            draw.rectangle(box, fill=color)
        elif shape == "triangle":
            draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=color)
        else:
            draw.polygon(
                [
                    (x + size // 2, y),
                    (x, y + size // 2),
                    (x + size // 2, y + size),
                    (x + size, y + size // 2),
                ],
                fill=color,
            )
        boxes[name] = box
    image.save(path)
    return boxes


def _slug(value: str) -> str:
    return value.replace(" ", "-")
