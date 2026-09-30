#!/usr/bin/env python3
"""Validate real Qwen3-VL preprocessing without loading model weights."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any

from fmlab.vlm.lora_data import (
    Qwen3VLAssistantOnlyCollator,
    load_canonical_vlm_manifest,
)
from fmlab.vlm.research import file_sha256


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _visual_tokens(batch: dict[str, Any], processor: Any) -> list[int]:
    grid = batch.get("image_grid_thw")
    if grid is None:
        return []
    merge = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1))
    return [int(row.prod().item()) // (merge * merge) for row in grid]


def run_check(
    model_path: Path,
    manifest: Path,
    output_dir: Path,
    *,
    batch_size: int = 2,
    min_pixels: int = 65_536,
    max_pixels: int = 401_408,
    max_sequence_length: int = 1_024,
) -> Path:
    from transformers import AutoProcessor

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    examples = load_canonical_vlm_manifest(manifest)
    if batch_size < 1 or batch_size > len(examples):
        raise ValueError("batch_size must be within the expanded manifest size")
    selected = examples[:batch_size]
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    batch = Qwen3VLAssistantOnlyCollator(
        processor,
        max_sequence_length=max_sequence_length,
    )(selected)
    labels = batch["labels"]
    attention = batch["attention_mask"]
    supervised = labels.ne(-100).sum(dim=1).tolist()
    result = {
        "experiment": "qwen3_vl_real_processor_check",
        "status": "success",
        "claim_level": "real_processor_wiring",
        "simulated": False,
        "model_weights_loaded": False,
        "processor_class": type(processor).__name__,
        "model_path": str(model_path.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "examples": [item.id for item in selected],
        "input_ids_shape": _shape(batch.get("input_ids")),
        "pixel_values_shape": _shape(batch.get("pixel_values")),
        "image_grid_thw": (batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else None),
        "visual_tokens": _visual_tokens(batch, processor),
        "sequence_tokens": attention.sum(dim=1).tolist(),
        "supervised_assistant_tokens": supervised,
        "padding_label_violations": int(labels[attention.eq(0)].ne(-100).sum().item()),
        "prefix_alignment": "exact_or_collator_would_fail",
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "duration_seconds": time.perf_counter() - started,
        "interpretation": (
            "Real local processor and images were used; this validates multimodal batching and "
            "assistant-only labels, not model inference or training quality."
        ),
    }
    if not all(value > 0 for value in supervised):
        raise RuntimeError("at least one row has no supervised assistant token")
    if result["padding_label_violations"]:
        raise RuntimeError("padding tokens were left supervised")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in result.items()
    )
    (output_dir / "report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Qwen3-VL processor check</title>"
        "<style>body{font-family:system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5e0;padding:.55rem;"
        "text-align:left}th{width:32%;background:#f3f6fa}code{word-break:break-all}</style></head>"
        f"<body><h1>Qwen3-VL real processor check</h1><p>{html.escape(result['interpretation'])}"
        f"</p><table>{cards}</table></body></html>",
        encoding="utf-8",
    )
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=65_536)
    parser.add_argument("--max-pixels", type=int, default=401_408)
    parser.add_argument("--max-sequence-length", type=int, default=1_024)
    args = parser.parse_args()
    print(run_check(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
