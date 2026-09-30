"""Held-out, paired evaluation for response-distilled vision-language models.

The module deliberately does not load a model.  It compares predictions produced by
separate base, supervised-LoRA, distilled-LoRA, and (optionally) teacher runs so an
expensive generation job can be audited and re-scored on CPU.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .research import (
    aggregate_predictions,
    file_sha256,
    paired_bootstrap_delta,
)


REQUIRED_CONDITIONS = ("base_8b", "gold_lora", "distilled_lora")
METRICS = ("exact_match", "anls", "type_accuracy")
ALIGNMENT_FIELDS = ("reference", "task", "answer_type")
GALLERY_CATEGORIES = (
    "base_wrong_distilled_right",
    "base_right_distilled_wrong",
    "gold_wrong_distilled_right",
    "gold_right_distilled_wrong",
    "teacher_wrong_student_right",
    "teacher_right_student_wrong",
    "all_wrong",
    "partial_text_match",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: each JSONL row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{source}: prediction file is empty")
    return rows


def _source_id(row: Mapping[str, Any]) -> str | None:
    # Prefer content identity when prediction/export jobs preserve it. Paths and
    # row ids can differ while still pointing at identical pixels.
    for key in ("image_sha256", "source_id", "image_path", "source"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _index_rows(rows: Sequence[Mapping[str, Any]], *, condition: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, original in enumerate(rows, start=1):
        row = dict(original)
        if "id" not in row or not str(row["id"]).strip():
            raise ValueError(f"{condition}: row {position} has no non-empty id")
        if "reference" not in row:
            raise ValueError(f"{condition}: id={row['id']!r} has no reference")
        if "prediction" not in row:
            raise ValueError(f"{condition}: id={row['id']!r} has no prediction")
        example_id = str(row["id"])
        if example_id in indexed:
            raise ValueError(f"{condition}: duplicate id {example_id!r}")
        row["id"] = example_id
        row.setdefault("task", "unknown")
        row.setdefault("answer_type", "text")
        indexed[example_id] = row
    return indexed


def load_aligned_predictions(
    condition_paths: Mapping[str, str | Path],
    *,
    train_source_ids: Iterable[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load prediction files and fail closed on pairing or source leakage errors."""

    missing = [condition for condition in REQUIRED_CONDITIONS if condition not in condition_paths]
    if missing:
        raise ValueError(f"missing required conditions: {', '.join(missing)}")

    indexed = {
        condition: _index_rows(_read_jsonl(path), condition=condition)
        for condition, path in condition_paths.items()
    }
    canonical_name = "base_8b"
    canonical = indexed[canonical_name]
    canonical_ids = set(canonical)
    for condition, rows in indexed.items():
        ids = set(rows)
        if ids != canonical_ids:
            missing_ids = sorted(canonical_ids - ids)
            extra_ids = sorted(ids - canonical_ids)
            raise ValueError(
                f"{condition}: misaligned ids; missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )
        for example_id, base_row in canonical.items():
            row = rows[example_id]
            for field in ALIGNMENT_FIELDS:
                if row.get(field) != base_row.get(field):
                    raise ValueError(
                        f"{condition}: id={example_id!r} has misaligned {field}: "
                        f"{row.get(field)!r} != {base_row.get(field)!r}"
                    )
            if _source_id(row) != _source_id(base_row):
                raise ValueError(f"{condition}: id={example_id!r} has a misaligned source")

    ordered_ids = sorted(canonical)
    aligned = {
        condition: [rows[example_id] for example_id in ordered_ids]
        for condition, rows in indexed.items()
    }
    eval_sources = {_source_id(row) for row in aligned[canonical_name]}
    eval_sources.discard(None)
    if train_source_ids is None:
        source_validation: dict[str, Any] = {
            "status": "not_checked",
            "reason": "No training-source manifest was supplied.",
            "eval_source_count": len(eval_sources),
        }
    else:
        train_sources = {str(source) for source in train_source_ids if str(source).strip()}
        if any(_source_id(row) is None for row in aligned[canonical_name]):
            raise ValueError(
                "cannot verify train/eval isolation: at least one evaluation row has no "
                "source_id, image_path, or source"
            )
        overlap = sorted(train_sources & eval_sources)
        if overlap:
            raise ValueError(f"train/eval source overlap detected: {overlap[:10]}")
        source_validation = {
            "status": "passed",
            "train_source_count": len(train_sources),
            "eval_source_count": len(eval_sources),
            "overlap_count": 0,
        }

    validation = {
        "id_alignment": "passed",
        "reference_alignment": "passed",
        "paired_examples": len(ordered_ids),
        "source_isolation": source_validation,
    }
    return aligned, validation


def load_train_source_ids(path: str | Path) -> set[str]:
    """Read train identities from a manifest or an audited split cache."""

    sources: set[str] = set()
    for position, row in enumerate(_read_jsonl(path), start=1):
        split = row.get("split")
        if split is not None and split not in {"train", "eval"}:
            raise ValueError(f"{path}: row {position} has invalid split {split!r}")
        if split == "eval":
            continue
        source = _source_id(row)
        if source is None:
            raise ValueError(
                f"{path}: training row {position} has no source_id, image_path, or source"
            )
        sources.add(source)
    return sources


def _summary(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate = aggregate_predictions(rows)
    records = list(aggregate.pop("records"))
    return aggregate, records


def _metric_vector(records: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in records]


def _paired_comparison(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    result = {
        metric: paired_bootstrap_delta(
            _metric_vector(baseline, metric),
            _metric_vector(candidate, metric),
            samples=samples,
            seed=seed,
        )
        for metric in METRICS
    }
    tasks = sorted({str(row.get("task", "unknown")) for row in baseline})
    per_task: dict[str, Any] = {}
    for offset, task in enumerate(tasks, start=1):
        baseline_slice = [row for row in baseline if str(row.get("task", "unknown")) == task]
        candidate_slice = [row for row in candidate if str(row.get("task", "unknown")) == task]
        per_task[task] = {
            metric: paired_bootstrap_delta(
                _metric_vector(baseline_slice, metric),
                _metric_vector(candidate_slice, metric),
                samples=samples,
                seed=seed + offset,
            )
            for metric in METRICS
        }
    result["by_task"] = per_task
    return result


def _gap_recovery(condition_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if "teacher_32b" not in condition_metrics:
        return {
            "status": "not_defined",
            "reason": "teacher_32b predictions were not supplied",
        }
    output: dict[str, Any] = {"status": "computed", "definition": "(distilled-base)/(teacher-base)"}
    for metric in METRICS:
        base = float(condition_metrics["base_8b"][metric])
        distilled = float(condition_metrics["distilled_lora"][metric])
        teacher = float(condition_metrics["teacher_32b"][metric])
        denominator = teacher - base
        if abs(denominator) < 1e-12:
            output[metric] = {
                "status": "undefined",
                "reason": "teacher and base have equal scores",
                "base": base,
                "distilled": distilled,
                "teacher": teacher,
            }
        else:
            fraction = (distilled - base) / denominator
            output[metric] = {
                "status": "defined",
                "fraction": fraction,
                "percent": 100.0 * fraction,
                "base": base,
                "distilled": distilled,
                "teacher": teacher,
            }
    return output


def _example_audit(scored: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    by_condition = {
        condition: {str(row["id"]): row for row in rows} for condition, rows in scored.items()
    }
    examples: list[dict[str, Any]] = []
    for example_id in sorted(by_condition["base_8b"]):
        base = by_condition["base_8b"][example_id]
        gold = by_condition["gold_lora"][example_id]
        distilled = by_condition["distilled_lora"][example_id]

        def correct(row: Mapping[str, Any]) -> bool:
            return float(row["type_accuracy"]) == 1.0

        categories: list[str] = []
        if not correct(base) and correct(distilled):
            categories.append("base_wrong_distilled_right")
        if correct(base) and not correct(distilled):
            categories.append("base_right_distilled_wrong")
        if not correct(gold) and correct(distilled):
            categories.append("gold_wrong_distilled_right")
        if correct(gold) and not correct(distilled):
            categories.append("gold_right_distilled_wrong")
        teacher = by_condition.get("teacher_32b", {}).get(example_id)
        if teacher is not None and not correct(teacher) and correct(distilled):
            categories.append("teacher_wrong_student_right")
        if teacher is not None and correct(teacher) and not correct(distilled):
            categories.append("teacher_right_student_wrong")
        rows = [condition_rows[example_id] for condition_rows in by_condition.values()]
        if not any(correct(row) for row in rows):
            categories.append("all_wrong")
        if any(0.0 < float(row["anls"]) < 1.0 for row in rows):
            categories.append("partial_text_match")

        conditions = {
            condition: {
                "prediction": row["prediction"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
            for condition, rows_by_id in by_condition.items()
            for row in (rows_by_id[example_id],)
        }
        examples.append(
            {
                "id": example_id,
                "source_id": _source_id(base),
                "task": base.get("task", "unknown"),
                "answer_type": base.get("answer_type", "text"),
                "reference": base["reference"],
                "categories": categories,
                "conditions": conditions,
            }
        )
    return examples


def _gallery(examples: Sequence[Mapping[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    gallery: dict[str, list[dict[str, Any]]] = {category: [] for category in GALLERY_CATEGORIES}
    for example in examples:
        for category in example["categories"]:
            if category in gallery and len(gallery[category]) < limit:
                gallery[category].append(dict(example))
    return gallery


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_html(report: Mapping[str, Any]) -> str:
    metric_rows = []
    for condition, metrics in report["conditions"].items():
        metric_rows.append(
            [
                condition,
                metrics["count"],
                f"{metrics['exact_match']:.3f}",
                f"{metrics['anls']:.3f}",
                f"{metrics['type_accuracy']:.3f}",
            ]
        )
    comparison_rows = []
    for baseline, candidates in report["comparisons"].items():
        for candidate, comparison in candidates.items():
            for metric in METRICS:
                values = comparison[metric]
                comparison_rows.append(
                    [
                        candidate,
                        baseline.removeprefix("vs_"),
                        metric,
                        f"{values['delta']:+.3f}",
                        f"[{values['ci95_low']:+.3f}, {values['ci95_high']:+.3f}]",
                        f"{values['probability_positive']:.3f}",
                    ]
                )
    task_rows = []
    for condition, metrics in report["conditions"].items():
        for task, values in metrics["by_task"].items():
            task_rows.append(
                [
                    condition,
                    task,
                    values["count"],
                    f"{values['exact_match']:.3f}",
                    f"{values['anls']:.3f}",
                    f"{values['type_accuracy']:.3f}",
                ]
            )
    gallery_sections = []
    for category, examples in report["failure_gallery"].items():
        gallery_rows = [
            [
                example["id"],
                example["task"],
                example["reference"],
                example["conditions"]["base_8b"]["prediction"],
                example["conditions"]["distilled_lora"]["prediction"],
                example["conditions"].get("teacher_32b", {}).get("prediction", "n/a"),
            ]
            for example in examples
        ]
        gallery_sections.append(
            f"<details><summary>{html.escape(category)} ({len(examples)})</summary>"
            + _html_table(["id", "task", "reference", "base", "distilled", "teacher"], gallery_rows)
            + "</details>"
        )
    resource_json = html.escape(
        json.dumps(report["resource_metrics"], ensure_ascii=False, indent=2)
    )
    validation_json = html.escape(json.dumps(report["validation"], ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VLM distillation held-out evaluation</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1180px;margin:auto;padding:2rem;color:#17202a}}
.notice{{border:2px solid #b35c00;background:#fff4df;padding:1rem;border-radius:8px}}
table{{border-collapse:collapse;width:100%;margin:.8rem 0 1.5rem}}th,td{{border:1px solid #ccd1d1;padding:.45rem;text-align:left}}
th{{background:#eef3f5}}code,pre{{background:#f4f6f7;padding:.6rem;overflow:auto}}details{{margin:.6rem 0}}
</style></head><body>
<h1>VLM response distillation: held-out paired evaluation</h1>
<p class="notice"><strong>Claim boundary:</strong> {html.escape(report["claim_boundary"])}</p>
<p>Generated: {html.escape(report["generated_at"])}; paired examples: {report["validation"]["paired_examples"]}.</p>
<h2>Condition metrics</h2>
{_html_table(["condition", "n", "exact", "ANLS", "type-aware"], metric_rows)}
<h2>Task slices</h2>
{_html_table(["condition", "task", "n", "exact", "ANLS", "type-aware"], task_rows)}
<h2>Paired bootstrap deltas</h2>
{_html_table(["candidate", "baseline", "metric", "delta", "95% CI", "P(delta>0)"], comparison_rows)}
<p>Intervals quantify sampling variation on this fixed paired set; they do not establish external validity.</p>
<h2>Teacher-gap recovery</h2><pre>{html.escape(json.dumps(report["teacher_gap_recovery"], indent=2))}</pre>
<h2>Failure gallery</h2>{"".join(gallery_sections)}
<h2>Validation</h2><pre>{validation_json}</pre>
<h2>Optional resource metrics</h2><pre>{resource_json}</pre>
</body></html>"""


def evaluate_distillation_predictions(
    condition_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    train_source_ids: Iterable[str] | None = None,
    resource_metrics: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 2_000,
    seed: int = 7,
    gallery_limit: int = 12,
    simulated: bool = False,
) -> dict[str, Any]:
    """Evaluate aligned held-out predictions and write machine/reader-facing reports."""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if gallery_limit < 1:
        raise ValueError("gallery_limit must be positive")
    aligned, validation = load_aligned_predictions(
        condition_paths, train_source_ids=train_source_ids
    )
    summaries: dict[str, Any] = {}
    scored: dict[str, list[dict[str, Any]]] = {}
    for condition, rows in aligned.items():
        summaries[condition], scored[condition] = _summary(rows)

    comparisons: dict[str, Any] = {"vs_base_8b": {}, "vs_gold_lora": {}}
    for condition in scored:
        if condition != "base_8b":
            comparisons["vs_base_8b"][condition] = _paired_comparison(
                scored["base_8b"],
                scored[condition],
                samples=bootstrap_samples,
                seed=seed,
            )
        if condition != "gold_lora":
            comparisons["vs_gold_lora"][condition] = _paired_comparison(
                scored["gold_lora"],
                scored[condition],
                samples=bootstrap_samples,
                seed=seed + 101,
            )

    examples = _example_audit(scored)
    source_checked = validation["source_isolation"]["status"] == "passed"
    if simulated:
        claim_boundary = (
            "SIMULATED DATA: values validate alignment, metrics, bootstrap, and reporting code only; "
            "they are not evidence of model quality."
        )
    else:
        claim_boundary = (
            "This is a paired held-out comparison for the supplied prediction files. Promotion "
            "to an Experiment claim still requires the registered multi-seed protocol; it does "
            "not establish production performance or broader generalization."
        )
    if not source_checked:
        claim_boundary += (
            " Train/eval source isolation was not verified because no manifest was supplied."
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "vlm_response_distillation_heldout_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulated": simulated,
        "claim_level": "wiring_only" if simulated else "heldout_comparison",
        "claim_boundary": claim_boundary,
        "validation": validation,
        "inputs": {
            condition: {
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(path),
            }
            for condition, path in condition_paths.items()
        },
        "configuration": {
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "gallery_limit": gallery_limit,
            "correctness_for_gallery": "type_accuracy == 1.0",
        },
        "conditions": summaries,
        "comparisons": comparisons,
        "teacher_gap_recovery": _gap_recovery(summaries),
        "resource_metrics": dict(resource_metrics or {}),
        "failure_gallery": _gallery(examples, gallery_limit),
        "examples": examples,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "distillation_evaluation.json"
    html_path = destination / "distillation_evaluation.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")
    report["artifacts"] = {"json": str(json_path), "html": str(html_path)}
    return report


def generate_simulated_demo(output_dir: str | Path) -> dict[str, Any]:
    """Generate deterministic synthetic predictions that exercise every report stage."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cases = [
        ("doc-1", "eval-invoice-a", "document", "number", "297.00"),
        ("doc-2", "eval-invoice-b", "document", "text", "Acme Labs"),
        ("doc-3", "eval-receipt-c", "document", "text", "Seoul"),
        ("chart-1", "eval-chart-a", "chart", "text", "Q4"),
        ("chart-2", "eval-chart-b", "chart", "number", "42"),
        ("chart-3", "eval-chart-c", "chart", "text", "North"),
        ("ground-1", "eval-scene-a", "grounding", "text", "yes"),
        ("ground-2", "eval-scene-b", "grounding", "text", "blue box"),
    ]
    predictions = {
        "base_8b": ["279", "Acme", "Busan", "Q3", "40", "North", "no", "red box"],
        "gold_lora": ["297.00", "Acme Labs", "Seoul", "Q4", "42", "South", "yes", "blue box"],
        "distilled_lora": ["297", "Acme Labs", "Seoul", "Q4", "42", "North", "yes", "blue box"],
        "teacher_32b": ["297.00", "Acme Labs", "Seoul", "Q4", "41", "North", "yes", "blue box"],
    }
    condition_paths: dict[str, Path] = {}
    for condition, answers in predictions.items():
        path = destination / f"{condition}.jsonl"
        rows = [
            {
                "id": example_id,
                "source_id": source_id,
                "task": task,
                "answer_type": answer_type,
                "reference": reference,
                "prediction": answer,
                "simulated": True,
            }
            for (example_id, source_id, task, answer_type, reference), answer in zip(
                cases, answers, strict=True
            )
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        condition_paths[condition] = path
    train_manifest = destination / "train_sources.jsonl"
    train_manifest.write_text(
        "".join(
            json.dumps({"id": f"train-{index}", "source_id": f"train-source-{index}"}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    return {"condition_paths": condition_paths, "train_manifest": train_manifest}


def _resource_metrics(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resource metrics JSON must contain an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="base_8b predictions JSONL")
    parser.add_argument("--gold", type=Path, help="gold_lora predictions JSONL")
    parser.add_argument("--distilled", type=Path, help="distilled_lora predictions JSONL")
    parser.add_argument("--teacher", type=Path, help="optional teacher_32b predictions JSONL")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--resource-metrics", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gallery-limit", type=int, default=12)
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument(
        "--demo", action="store_true", help="generate deterministic synthetic wiring data"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        demo = generate_simulated_demo(args.output_dir / "demo_inputs")
        condition_paths = demo["condition_paths"]
        train_manifest = demo["train_manifest"]
        simulated = True
    else:
        required = {"--base": args.base, "--gold": args.gold, "--distilled": args.distilled}
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"missing required arguments without --demo: {', '.join(missing)}")
        condition_paths = {
            "base_8b": args.base,
            "gold_lora": args.gold,
            "distilled_lora": args.distilled,
        }
        if args.teacher is not None:
            condition_paths["teacher_32b"] = args.teacher
        train_manifest = args.train_manifest
        simulated = args.simulated
    train_sources = load_train_source_ids(train_manifest) if train_manifest is not None else None
    report = evaluate_distillation_predictions(
        condition_paths,
        args.output_dir,
        train_source_ids=train_sources,
        resource_metrics=_resource_metrics(args.resource_metrics),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        gallery_limit=args.gallery_limit,
        simulated=simulated,
    )
    print(report["artifacts"]["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
