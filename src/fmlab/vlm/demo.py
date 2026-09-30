from __future__ import annotations

import html
import json
import time
from pathlib import Path

from fmlab.artifacts import ExperimentResult

from .chart_eval import evaluate_chart_qa
from .document_compare import SidecarOCR, run_document_comparison
from .grounding import evaluate_grounding, generate_grounding_dataset
from .inference import Qwen3VLConfig, Qwen3VLRunner
from .reporting import write_gallery
from .synthetic import generate_chart_dataset, generate_document_dataset
from .training import create_teacher_response_dataset
from .video import plan_frame_indices, write_sampling_plan


def run_offline_demo(
    output_dir: Path,
    *,
    sample_count: int = 4,
    seed: int = 7,
) -> ExperimentResult:
    """Run the complete VLM track without loading a model or using the network."""

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = generate_document_dataset(
        output_dir / "datasets" / "documents", count=sample_count, seed=seed
    )
    charts = generate_chart_dataset(
        output_dir / "datasets" / "charts", count=sample_count, seed=seed + 1
    )
    grounding_cases = generate_grounding_dataset(
        output_dir / "datasets" / "grounding", count=sample_count, seed=seed + 2
    )
    all_images = [sample.image_path for sample in documents + charts]
    all_images.extend(sorted({case.image_path for case in grounding_cases}))
    gallery = write_gallery(
        output_dir / "gallery.html", title="VLM synthetic data", images=all_images
    )

    runner = Qwen3VLRunner(Qwen3VLConfig(mode="offline"))
    document_result = run_document_comparison(
        documents,
        runner,
        output_dir / "evaluations" / "document_compare",
        ocr_backend=SidecarOCR(character_error_rate=0.06, seed=seed),
    )
    chart_result = evaluate_chart_qa(charts, runner, output_dir / "evaluations" / "chart_qa")
    grounding_result = evaluate_grounding(
        grounding_cases, runner, output_dir / "evaluations" / "grounding"
    )
    teacher_data = create_teacher_response_dataset(
        documents + charts, runner, output_dir / "distillation" / "teacher_responses.jsonl"
    )
    frames = plan_frame_indices(total_frames=1800, fps=30.0, max_frames=12, strategy="uniform")
    video_json, video_svg = write_sampling_plan(
        output_dir / "video" / "offline_sampling_plan.json", frames
    )
    index = _write_index(
        output_dir / "index.html",
        {
            "Synthetic image gallery": gallery,
            "OCR vs VLM report": Path(document_result.artifacts[2]),
            "Chart QA report": Path(chart_result.artifacts[2]),
            "Grounding report": Path(grounding_result.artifacts[2]),
            "Video sampling timeline": video_svg,
            "Teacher response data": teacher_data,
        },
    )
    metrics = {
        "document_accuracy": document_result.metrics["accuracy"],
        "chart_accuracy": chart_result.metrics["overall_accuracy"],
        "grounding_accuracy": grounding_result.metrics["accuracy"],
        "grounding_hallucination_rate": grounding_result.metrics["hallucination_rate"],
        "images": len(all_images),
    }
    experiment = ExperimentResult(
        experiment="vlm_offline_demo",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters={"sample_count": sample_count, "seed": seed, "mode": "offline"},
        artifacts=[
            str(index),
            str(gallery),
            str(video_json),
            str(video_svg),
            str(teacher_data),
        ],
        notes=[
            "Offline VLM answers are deterministic simulators and are not model benchmarks.",
            "Open index.html to inspect images, predictions, metrics, and video-frame planning.",
        ],
    )
    experiment.write(output_dir)
    return experiment


def _write_index(path: Path, links: dict[str, Path]) -> Path:
    import os

    items = []
    for label, target in links.items():
        href = Path(os.path.relpath(target.resolve(), path.parent.resolve())).as_posix()
        items.append(f'<li><a href="{html.escape(href)}">{html.escape(label)}</a></li>')
    payload = {
        "mode": "offline",
        "interpretation": "pipeline validation only",
        "next_step": "repeat with configs/vlm/qwen3_vl_8b_local.yaml",
    }
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>VLM lab</title>
<style>body{{font-family:system-ui;max-width:850px;margin:3rem auto;padding:0 1rem}}
.warning{{background:#fff4cc;border-left:4px solid #e0a800;padding:1rem}}li{{margin:.8rem 0}}</style>
</head><body><h1>Visual language model lab</h1>
<p class="warning"><strong>Offline simulator:</strong> these scores verify data, evaluation, and reporting. They do not measure Qwen3-VL.</p>
<ul>{"".join(items)}</ul><h2>Run metadata</h2><pre>{html.escape(json.dumps(payload, indent=2))}</pre>
</body></html>"""
    path.write_text(document, encoding="utf-8")
    return path
