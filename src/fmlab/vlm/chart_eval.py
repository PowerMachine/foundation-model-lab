from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fmlab.artifacts import ExperimentResult

from .inference import Qwen3VLRunner
from .metrics import exact_match, numeric_match, token_f1
from .reporting import write_evaluation_html, write_rows, write_score_bars
from .synthetic import SyntheticSample


def evaluate_chart_qa(
    samples: list[SyntheticSample],
    runner: Qwen3VLRunner,
    output_dir: Path,
) -> ExperimentResult:
    """Evaluate lookup and arithmetic chart questions separately."""

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    by_operation: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        values = sample.metadata.get("values", {})
        for qa in sample.qa:
            result = runner.infer(
                qa.question,
                [sample.image_path],
                context={"expected_answer": qa.answer, "values": values},
            )
            score = (
                numeric_match(result.text, qa.answer)
                if qa.answer_type in {"integer", "number"}
                else exact_match(result.text, qa.answer)
            )
            operation = str(qa.metadata.get("operation", "other"))
            by_operation[operation].append(score)
            rows.append(
                {
                    "sample": sample.id,
                    "image_path": str(sample.image_path),
                    "chart_type": sample.metadata.get("chart_type"),
                    "operation": operation,
                    "question": qa.question,
                    "reference": qa.answer,
                    "prediction": result.text,
                    "exact_match": score,
                    "token_f1": token_f1(result.text, qa.answer),
                    "backend": result.backend,
                    "simulated": result.simulated,
                }
            )
    operation_accuracy = {
        operation: sum(scores) / len(scores) for operation, scores in sorted(by_operation.items())
    }
    overall = sum(row["exact_match"] for row in rows) / len(rows) if rows else 0.0
    predictions_path = write_rows(output_dir / "predictions.jsonl", rows)
    score_path = write_score_bars(
        output_dir / "operation_accuracy.svg",
        operation_accuracy,
        title="Chart QA accuracy by operation",
    )
    report_path = write_evaluation_html(
        output_dir / "report.html",
        title="Synthetic chart QA",
        summary={"samples": len(samples), "questions": len(rows), "overall_accuracy": overall},
        rows=rows,
        score_chart=score_path,
        notice=(
            "Offline answers validate label/evaluation/report plumbing only. "
            "Use the local 8B profile to measure visual reasoning."
        ),
    )
    experiment = ExperimentResult(
        experiment="vlm_chart_qa",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics={"overall_accuracy": overall, "operation_accuracy": operation_accuracy},
        parameters={"vlm_mode": runner.config.mode, "samples": len(samples)},
        artifacts=[str(predictions_path), str(score_path), str(report_path)],
        notes=["Synthetic labels are recomputed from chart source values."],
    )
    result_path = experiment.write(output_dir)
    experiment.artifacts.append(str(result_path))
    return experiment
