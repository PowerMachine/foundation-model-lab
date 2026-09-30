"""Optional TRL-backed DPO runner for the locally stored Qwen models."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from .optional import OptionalDependencyError, require_optional
from .workflows import LocalWorkflowConfig, WorkflowKind, _load_transformers_model, _read_jsonl


SMOKE_PREFERENCES = [
    {
        "prompt": "양자화의 장점을 한 문장으로 설명해줘.",
        "chosen": "가중치 표현 비트를 줄여 메모리 사용량을 낮춥니다.",
        "rejected": "양자화는 언제나 정확도를 높이고 단점이 없습니다.",
    },
    {
        "prompt": "근거 문서에 답이 없으면 어떻게 해야 해?",
        "chosen": "근거가 없다고 명확히 말해야 합니다.",
        "rejected": "그럴듯한 내용을 만들어 답하면 됩니다.",
    },
]


def _supported_arguments(callable_object: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    return {name: value for name, value in candidates.items() if name in parameters}


def run_local_dpo(config: LocalWorkflowConfig) -> dict[str, object]:
    """Run a capped, offline DPO experiment using TRL.

    TRL has changed a few constructor names across releases, so signatures are
    inspected at runtime. Unsupported versions fail with an upgrade hint rather
    than an opaque ``unexpected keyword`` traceback.
    """

    if config.kind != WorkflowKind.DPO:
        raise ValueError("DPO workflow config required")
    config.validate()
    trl = require_optional("trl", extra="llm", purpose="DPO")
    datasets = require_optional("datasets", extra="llm", purpose="DPO")
    tokenizer, model = _load_transformers_model(config)
    records = _read_jsonl(config.train_file, limit=config.sample_limit) or SMOKE_PREFERENCES
    required = {"prompt", "chosen", "rejected"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"DPO record {index} is missing fields: {sorted(missing)}")
    train_dataset = datasets.Dataset.from_list(records[: config.sample_limit])
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dpo_config_values = {
        "output_dir": str(destination),
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "max_length": config.max_length,
        "beta": float(config.extra.get("beta", 0.1)),
    }
    try:
        arguments = trl.DPOConfig(**_supported_arguments(trl.DPOConfig, dpo_config_values))
        trainer_values = {
            "model": model,
            "ref_model": None,
            "args": arguments,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
        }
        trainer = trl.DPOTrainer(**_supported_arguments(trl.DPOTrainer, trainer_values))
    except TypeError as exc:
        raise OptionalDependencyError(
            "Installed TRL API is incompatible with this DPO adapter. "
            "Install the project llm extra again to align transformers/trl versions."
        ) from exc
    output = trainer.train()
    if config.save_model:
        trainer.save_model(str(destination / "model"))
    metrics: dict[str, object] = {
        **output.metrics,
        "samples": len(train_dataset),
        "beta": float(config.extra.get("beta", 0.1)),
        "saved_model": config.save_model,
    }
    (destination / "dpo_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
