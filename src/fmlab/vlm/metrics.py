from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


def normalize_answer(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[$€£₩,%]", "", value)
    value = re.sub(r"[^\w.\-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def numeric_match(prediction: str, reference: str, *, atol: float = 1e-6) -> float:
    pred = _first_number(prediction)
    ref = _first_number(reference)
    if pred is None or ref is None:
        return exact_match(prediction, reference)
    return float(math.isclose(pred, ref, abs_tol=atol, rel_tol=1e-6))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate_scores(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"count": 0.0, "exact_match": 0.0, "token_f1": 0.0}
    return {
        "count": float(len(rows)),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "token_f1": sum(row["token_f1"] for row in rows) / len(rows),
    }


def _first_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None
