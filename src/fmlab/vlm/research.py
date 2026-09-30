from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def normalize_answer(value: Any) -> str:
    """Conservative normalization for short-form VQA answers."""

    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?`'\"")


def normalized_levenshtein_similarity(left: Any, right: Any) -> float:
    """ANLS-style similarity in [0, 1] without an external dependency."""

    a, b = normalize_answer(left), normalize_answer(right)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for row, char_a in enumerate(a, start=1):
        current = [row]
        for column, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char_a != char_b),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def _number(value: Any) -> float | None:
    match = _NUMBER.search(normalize_answer(value).replace(",", ""))
    return float(match.group()) if match else None


def score_answer(
    prediction: Any,
    reference: Any,
    *,
    answer_type: str = "text",
    numeric_atol: float = 1e-3,
    anls_threshold: float = 0.5,
) -> dict[str, float]:
    """Return exact, ANLS, and type-aware accuracy for one VQA answer."""

    exact = float(normalize_answer(prediction) == normalize_answer(reference))
    similarity = normalized_levenshtein_similarity(prediction, reference)
    anls = similarity if similarity >= anls_threshold else 0.0
    kind = answer_type.casefold()
    if kind in {"number", "integer", "float", "numeric"}:
        predicted_number, reference_number = _number(prediction), _number(reference)
        typed = float(
            predicted_number is not None
            and reference_number is not None
            and math.isclose(predicted_number, reference_number, abs_tol=numeric_atol)
        )
    else:
        typed = exact
    return {"exact_match": exact, "anls": anls, "type_accuracy": typed}


def grouped_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "image_path",
    eval_fraction: float = 0.25,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by image/group so questions from one image cannot leak across splits."""

    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be in (0, 1)")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if group_key not in row:
            raise KeyError(f"missing group key: {group_key}")
        grouped[str(row[group_key])].append(dict(row))
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("grouped split requires at least two distinct groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    evaluation_groups = set(
        groups[: max(1, min(len(groups) - 1, round(len(groups) * eval_fraction)))]
    )
    train = [row for group in groups if group not in evaluation_groups for row in grouped[group]]
    evaluation = [row for group in groups if group in evaluation_groups for row in grouped[group]]
    return train, evaluation


def aggregate_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate global and task-sliced metrics from prediction records."""

    records = [dict(row) for row in rows]
    if not records:
        return {"count": 0, "exact_match": 0.0, "anls": 0.0, "type_accuracy": 0.0}
    scored: list[dict[str, Any]] = []
    for row in records:
        scores = score_answer(
            row.get("prediction", ""),
            row.get("reference", ""),
            answer_type=str(row.get("answer_type", "text")),
        )
        scored.append({**row, **scores})

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(items),
            "exact_match": statistics.fmean(float(item["exact_match"]) for item in items),
            "anls": statistics.fmean(float(item["anls"]) for item in items),
            "type_accuracy": statistics.fmean(float(item["type_accuracy"]) for item in items),
        }

    slices: dict[str, Any] = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_task[str(row.get("task", "unknown"))].append(row)
    for task, items in sorted(by_task.items()):
        slices[task] = summarize(items)
    return {**summarize(scored), "by_task": slices, "records": scored}


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 2_000,
    seed: int = 7,
) -> dict[str, float | int]:
    """Paired bootstrap CI for a candidate-minus-baseline metric delta."""

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate must be non-empty and paired")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    rng = random.Random(seed)
    count = len(baseline)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        deltas.append(statistics.fmean(candidate[index] - baseline[index] for index in indices))
    deltas.sort()
    lower = deltas[int(0.025 * (samples - 1))]
    upper = deltas[int(0.975 * (samples - 1))]
    observed = statistics.fmean(c - b for b, c in zip(baseline, candidate, strict=True))
    return {
        "pairs": count,
        "samples": samples,
        "delta": observed,
        "ci95_low": lower,
        "ci95_high": upper,
        "probability_positive": sum(delta > 0 for delta in deltas) / samples,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(
    output_path: str | Path,
    *,
    config: Mapping[str, Any],
    inputs: Iterable[str | Path],
    system: Mapping[str, Any] | None = None,
) -> Path:
    """Write immutable config/input fingerprints alongside an experiment."""

    destination = Path(output_path)
    payload = {
        "config": json.loads(json.dumps(config, default=str)),
        "inputs": {
            str(Path(path).resolve()): file_sha256(path) for path in inputs if Path(path).is_file()
        },
        "system": dict(system or {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
