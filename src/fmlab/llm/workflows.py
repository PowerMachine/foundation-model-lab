"""Local-only Hugging Face workflows plus inspectable workflow planning.

Nothing is downloaded: model loading always sets ``local_files_only=True``.
The default configuration is intentionally a two-step smoke run.  Larger runs
must be requested explicitly by changing the YAML configuration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from fmlab.paths import LabPaths

from .distillation import DistillationConfig, distillation_loss
from .optional import OptionalDependencyError, require_optional


class WorkflowKind(str, Enum):
    CONTINUED_PRETRAINING = "continued_pretraining"
    FULL_SFT = "full_sft"
    LORA = "lora"
    QLORA = "qlora"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    DPO = "dpo"
    RAG = "rag"


@dataclass(frozen=True)
class LocalWorkflowConfig:
    kind: WorkflowKind
    model_path: str
    output_dir: str
    train_file: str | None = None
    teacher_model_path: str | None = None
    smoke: bool = True
    max_steps: int = 2
    sample_limit: int = 16
    max_length: int = 128
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    dtype: str = "bfloat16"
    device_map: str | None = None
    trust_remote_code: bool = False
    save_model: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    quantization_bits: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LocalWorkflowConfig":
        data = dict(values)
        data["kind"] = WorkflowKind(data["kind"])
        if "target_modules" in data:
            data["target_modules"] = tuple(data["target_modules"])
        return cls(**data)

    def validate(self) -> None:
        model = Path(self.model_path).expanduser()
        if self.kind != WorkflowKind.RAG and not model.is_dir():
            expected_root = LabPaths.from_env().model_root / "Qwen"
            raise FileNotFoundError(
                f"Local model directory does not exist: {model}. "
                f"Available models are expected below {expected_root}."
            )
        if self.teacher_model_path and not Path(self.teacher_model_path).expanduser().is_dir():
            raise FileNotFoundError(
                f"Teacher model directory does not exist: {self.teacher_model_path}"
            )
        if self.train_file and not Path(self.train_file).expanduser().is_file():
            raise FileNotFoundError(f"Training file does not exist: {self.train_file}")
        if self.max_steps < 1 or self.max_length < 8 or self.batch_size < 1:
            raise ValueError("max_steps/batch_size must be positive and max_length must be >= 8")
        if self.kind == WorkflowKind.QLORA and self.quantization_bits not in (4, 8):
            raise ValueError("QLoRA requires quantization_bits: 4 or 8")


WORKFLOW_DESCRIPTIONS: dict[WorkflowKind, dict[str, object]] = {
    WorkflowKind.CONTINUED_PRETRAINING: {
        "learns": "domain vocabulary, facts, and style through next-token prediction",
        "data": "JSONL with a text field",
        "optional_packages": ["transformers"],
    },
    WorkflowKind.FULL_SFT: {
        "learns": "prompt-to-response behavior by updating every model parameter",
        "data": "JSONL with prompt/response or messages",
        "optional_packages": ["transformers"],
    },
    WorkflowKind.LORA: {
        "learns": "low-rank updates while base weights stay frozen",
        "data": "same as SFT",
        "optional_packages": ["transformers", "peft"],
    },
    WorkflowKind.QLORA: {
        "learns": "LoRA adapters over a frozen 4/8-bit base model",
        "data": "same as SFT",
        "optional_packages": ["transformers", "peft", "bitsandbytes"],
    },
    WorkflowKind.QUANTIZATION: {
        "learns": "memory/latency/quality trade-off at reduced weight precision",
        "data": "representative prompts",
        "optional_packages": ["transformers", "bitsandbytes"],
    },
    WorkflowKind.DISTILLATION: {
        "learns": "student behavior from teacher probability distributions",
        "data": "JSONL text or prompt/response",
        "optional_packages": ["transformers"],
    },
    WorkflowKind.DPO: {
        "learns": "chosen responses over rejected responses relative to a reference",
        "data": "JSONL with prompt/chosen/rejected",
        "optional_packages": ["transformers", "trl", "datasets"],
    },
    WorkflowKind.RAG: {
        "learns": "retrieval quality separately from answer generation",
        "data": "documents plus relevance-labelled queries",
        "optional_packages": [],
    },
}


def workflow_plan(config: LocalWorkflowConfig) -> dict[str, object]:
    description = WORKFLOW_DESCRIPTIONS[config.kind]
    warnings: list[str] = []
    if config.smoke:
        warnings.append("smoke=true: sample_limit and max_steps intentionally cap the run")
    if config.save_model and config.kind == WorkflowKind.FULL_SFT:
        warnings.append("saving a full checkpoint can consume roughly the original model size")
    if config.kind in (WorkflowKind.QLORA, WorkflowKind.QUANTIZATION):
        warnings.append("bitsandbytes modes require a supported CUDA GPU; toy simulation does not")
    return {
        "kind": config.kind.value,
        **description,
        "model_path": config.model_path,
        "teacher_model_path": config.teacher_model_path,
        "max_steps": config.max_steps,
        "sample_limit": config.sample_limit,
        "offline_only": True,
        "warnings": warnings,
    }


SMOKE_TEXTS = [
    "어텐션은 문맥 속 토큰 사이의 관련성을 계산한다.",
    "LoRA는 작은 저랭크 행렬만 학습하는 효율적인 방법이다.",
    "양자화는 가중치 표현 비트를 줄여 메모리를 절약한다.",
    "검색 평가는 생성 평가와 분리해야 실패 원인을 알 수 있다.",
]

SMOKE_INSTRUCTIONS = [
    {
        "prompt": "LoRA를 한 문장으로 설명해줘.",
        "response": "원본을 고정하고 저랭크 변화량만 학습합니다.",
    },
    {"prompt": "양자화의 장점은?", "response": "모델 메모리와 전송 용량을 줄일 수 있습니다."},
    {
        "prompt": "RAG의 검색 평가는 왜 필요한가?",
        "response": "생성 실패와 검색 실패를 구분할 수 있기 때문입니다.",
    },
]


def _read_jsonl(path: str | None, *, limit: int) -> list[dict[str, Any]]:
    if path is None:
        return []
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(value)
            if len(records) >= limit:
                break
    return records


class _ListDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in choices:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(choices)}")
    return choices[name]


def _load_transformers_model(config: LocalWorkflowConfig) -> tuple[Any, Any]:
    transformers = require_optional("transformers", extra="llm", purpose=config.kind.value)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": _dtype(config.dtype),
    }
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    if config.quantization_bits in (4, 8):
        require_optional(
            "bitsandbytes", extra="llm", purpose=f"{config.quantization_bits}-bit loading"
        )
        if not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes quantization requires CUDA in this lab")
        load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=config.quantization_bits == 4,
            load_in_8bit=config.quantization_bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_path, **load_kwargs)
    return tokenizer, model


def _training_rows(config: LocalWorkflowConfig, tokenizer: Any) -> list[dict[str, torch.Tensor]]:
    records = _read_jsonl(config.train_file, limit=config.sample_limit)
    if config.kind == WorkflowKind.CONTINUED_PRETRAINING:
        texts = [str(record["text"]) for record in records] if records else SMOKE_TEXTS
        formatted = [(text, None) for text in texts]
    else:
        instruction_records = records or SMOKE_INSTRUCTIONS
        formatted = []
        for record in instruction_records:
            if "messages" in record and hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
                formatted.append((text, None))
            else:
                prompt = str(record["prompt"])
                response = str(record["response"])
                prefix = f"User: {prompt}\nAssistant: "
                formatted.append((prefix + response, prefix))
    rows: list[dict[str, torch.Tensor]] = []
    for text, prefix in formatted[: config.sample_limit]:
        encoded = tokenizer(
            text,
            max_length=config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        labels = input_ids.clone().masked_fill(attention_mask.eq(0), -100)
        if prefix is not None:
            prefix_length = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
            labels[: min(prefix_length, labels.numel())] = -100
        rows.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return rows


def run_local_training(config: LocalWorkflowConfig) -> dict[str, object]:
    """Run continued pretraining, full SFT, LoRA, or QLoRA on a local model."""

    if config.kind not in {
        WorkflowKind.CONTINUED_PRETRAINING,
        WorkflowKind.FULL_SFT,
        WorkflowKind.LORA,
        WorkflowKind.QLORA,
    }:
        raise ValueError(f"run_local_training does not handle {config.kind.value}")
    config.validate()
    transformers = require_optional("transformers", extra="llm", purpose=config.kind.value)
    tokenizer, model = _load_transformers_model(config)
    if config.kind in (WorkflowKind.LORA, WorkflowKind.QLORA):
        peft = require_optional("peft", extra="llm", purpose=config.kind.value)
        if config.kind == WorkflowKind.QLORA:
            model = peft.prepare_model_for_kbit_training(model)
        peft_config = peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            task_type="CAUSAL_LM",
        )
        model = peft.get_peft_model(model, peft_config)
    rows = _training_rows(config, tokenizer)
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arguments = transformers.TrainingArguments(
        output_dir=str(destination),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=config.dtype == "bfloat16" and torch.cuda.is_available(),
        fp16=config.dtype == "float16" and torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=_ListDataset(rows),
        data_collator=transformers.default_data_collator,
    )
    started = time.perf_counter()
    train_output = trainer.train()
    elapsed = time.perf_counter() - started
    if config.save_model:
        trainer.save_model(str(destination / "model"))
        tokenizer.save_pretrained(str(destination / "model"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    metrics: dict[str, object] = {
        **train_output.metrics,
        "elapsed_seconds_measured": elapsed,
        "samples": len(rows),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
        "saved_model": config.save_model,
    }
    (destination / "workflow_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def run_local_distillation(config: LocalWorkflowConfig) -> dict[str, object]:
    """Perform a short teacher/student KL + hard-label training run."""

    if config.kind != WorkflowKind.DISTILLATION:
        raise ValueError("distillation config required")
    config.validate()
    if not config.teacher_model_path:
        raise ValueError("teacher_model_path is required for distillation")
    tokenizer, student = _load_transformers_model(config)
    teacher_config = LocalWorkflowConfig(
        **{
            **asdict(config),
            "kind": WorkflowKind.DISTILLATION,
            "model_path": config.teacher_model_path,
            "teacher_model_path": None,
            "quantization_bits": None,
        }
    )
    _, teacher = _load_transformers_model(teacher_config)
    if (
        student.get_output_embeddings().weight.shape[0]
        != teacher.get_output_embeddings().weight.shape[0]
    ):
        raise ValueError("teacher and student must share the same output vocabulary")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device_map is None:
        student.to(device)
        teacher.to(device)
    teacher.eval()
    student.train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    rows = _training_rows(
        LocalWorkflowConfig(**{**asdict(config), "kind": WorkflowKind.CONTINUED_PRETRAINING}),
        tokenizer,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    loss_rows: list[dict[str, float]] = []
    objective = DistillationConfig(
        temperature=float(config.extra.get("temperature", 2.0)),
        soft_target_weight=float(config.extra.get("soft_target_weight", 0.7)),
        hard_target_weight=float(config.extra.get("hard_target_weight", 0.3)),
    )
    for step in range(config.max_steps):
        row = rows[step % len(rows)]
        batch = {name: value.unsqueeze(0).to(device) for name, value in row.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_logits = teacher(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        student_logits = student(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        loss, components = distillation_loss(
            student_logits, teacher_logits, batch["labels"], objective
        )
        loss.backward()
        optimizer.step()
        loss_rows.append(components)
    metrics: dict[str, object] = {
        "steps": config.max_steps,
        "initial_loss": loss_rows[0]["total_loss"],
        "final_loss": loss_rows[-1]["total_loss"],
        "history": loss_rows,
    }
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "distillation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


@torch.inference_mode()
def run_local_generation_benchmark(
    config: LocalWorkflowConfig,
    prompts: Sequence[str] | None = None,
) -> dict[str, object]:
    """Measure load memory, latency, and tokens/s for one precision setting."""

    if config.kind != WorkflowKind.QUANTIZATION:
        raise ValueError("quantization config required")
    config.validate()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started_load = time.perf_counter()
    tokenizer, model = _load_transformers_model(config)
    load_seconds = time.perf_counter() - started_load
    if config.device_map is None:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    examples = list(prompts or ["어텐션의 역할을 짧게 설명해줘."])
    encoded = tokenizer(examples, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(**encoded, max_new_tokens=int(config.extra.get("max_new_tokens", 32)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - started
    generated_tokens = output.shape[1] * output.shape[0] - encoded["input_ids"].numel()
    metrics: dict[str, object] = {
        "quantization_bits": config.quantization_bits,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "new_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / max(generation_seconds, 1e-9),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "outputs": tokenizer.batch_decode(output, skip_special_tokens=True),
    }
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "quantization_benchmark.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def assert_workflow_dependencies(config: LocalWorkflowConfig) -> None:
    """Fail early with one actionable message per missing optional dependency."""

    missing: list[str] = []
    for package in WORKFLOW_DESCRIPTIONS[config.kind]["optional_packages"]:
        try:
            require_optional(str(package), extra="llm", purpose=config.kind.value)
        except OptionalDependencyError:
            missing.append(str(package))
    if missing:
        raise OptionalDependencyError(
            f"{config.kind.value} is missing: {', '.join(missing)}. "
            "Install with: pip install -e '.[llm]'"
        )
