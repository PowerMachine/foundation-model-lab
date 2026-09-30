from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from fmlab.artifacts import ExperimentResult


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_run"}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"value": value}


def build_local_model_report(llm_root: str | Path) -> ExperimentResult:
    root = Path(llm_root)
    output = root / "local-model-suite"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs = {
        "continued_pretraining": _read(root / "continued_pretraining/workflow_metrics.json"),
        "full_sft": _read(root / "full_sft/workflow_metrics.json"),
        "lora_3b": _read(root / "lora/workflow_metrics.json"),
        "qlora_3b_nf4": _read(root / "qlora/workflow_metrics.json"),
        "distillation_3b_to_05b": _read(root / "distillation/distillation_metrics.json"),
        "dpo_05b": _read(root / "dpo/dpo_metrics.json"),
        "int4_05b": _read(root / "quantization/int4/quantization_benchmark.json"),
    }

    trainable_names = ["full_sft", "lora_3b", "qlora_3b_nf4"]
    trainable_values = [float(runs[name].get("trainable_percent", 0)) for name in trainable_names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(trainable_names, trainable_values, color=["#eb6a5b", "#55a8e1", "#68c28a"])
    ax.set_ylabel("trainable parameters (%)")
    ax.set_title("Actual local-model training footprint")
    ax.bar_label(bars, fmt="%.3f%%")
    ax.set_ylim(0, max(trainable_values + [1]) * 1.15)
    fig.tight_layout()
    chart = output / "trainable_percent.png"
    fig.savefig(chart, dpi=150)
    plt.close(fig)

    rows = []
    for name, value in runs.items():
        if "train_loss" in value:
            key_metric = f"train loss={float(value['train_loss']):.4f}"
        elif "final_loss" in value:
            key_metric = f"final loss={float(value['final_loss']):.4f}"
        elif "tokens_per_second" in value:
            key_metric = f"tokens/s={float(value['tokens_per_second']):.2f}"
        else:
            key_metric = str(value.get("status", "recorded"))
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(key_metric)}</td>"
            f"<td><pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></td></tr>"
        )
    report = output / "report.html"
    report.write_text(
        "<!doctype html><html lang='ko'><meta charset='utf-8'><title>Local LLM runs</title>"
        "<style>body{font:15px system-ui;max-width:1300px;margin:2rem auto}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:.5rem;vertical-align:top}pre{white-space:pre-wrap}</style>"
        "<body><h1>Actual local Qwen smoke runs</h1>"
        "<p>모든 실행은 로컬 가중치, 짧은 내장 데이터, 2 step, save_model=false를 사용했다.</p>"
        "<p>서로 다른 모델/목적의 loss 절대값은 직접 순위 비교 대상이 아니다.</p>"
        "<img style='max-width:100%' src='trainable_percent.png'>"
        f"<table><tr><th>run</th><th>key metric</th><th>full metrics</th></tr>{''.join(rows)}</table></body></html>",
        encoding="utf-8",
    )
    metrics = {
        "continued_pretraining_train_loss": runs["continued_pretraining"].get("train_loss"),
        "full_sft_train_loss": runs["full_sft"].get("train_loss"),
        "lora_train_loss": runs["lora_3b"].get("train_loss"),
        "lora_trainable_percent": runs["lora_3b"].get("trainable_percent"),
        "qlora_train_loss": runs["qlora_3b_nf4"].get("train_loss"),
        "qlora_trainable_percent": runs["qlora_3b_nf4"].get("trainable_percent"),
        "distillation_initial_loss": runs["distillation_3b_to_05b"].get("initial_loss"),
        "distillation_final_loss": runs["distillation_3b_to_05b"].get("final_loss"),
        "dpo_train_loss": runs["dpo_05b"].get("train_loss"),
        "int4_tokens_per_second": runs["int4_05b"].get("tokens_per_second"),
        "int4_peak_vram_bytes": runs["int4_05b"].get("peak_vram_bytes"),
    }
    result = ExperimentResult(
        experiment="llm_actual_local_model_suite",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters={"max_steps": 2, "save_model": False, "local_files_only": True},
        artifacts=["trainable_percent.png", "report.html"],
        notes=[
            "These are real local Qwen/PEFT/bitsandbytes/TRL runs, not simulations.",
            "The data and step counts are smoke-sized; loss values are pipeline evidence, not quality benchmarks.",
        ],
    )
    result.write(output)
    return result
