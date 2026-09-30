from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .inference import Qwen3VLRunner
from .synthetic import SyntheticSample


@dataclass
class VLMLoRARecipe:
    model_path: Path
    dataset_manifest: Path
    output_dir: Path
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    freeze_vision_encoder: bool = True
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 2048
    bf16: bool = True

    def validate(self, *, require_files: bool = False) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.per_device_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch sizes must be positive")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")
        if require_files:
            if not self.model_path.is_dir():
                raise FileNotFoundError(self.model_path)
            if not self.dataset_manifest.is_file():
                raise FileNotFoundError(self.dataset_manifest)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("model_path", "dataset_manifest", "output_dir"):
            value[key] = str(value[key])
        return value

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml

            text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        except ImportError:  # pragma: no cover - base project includes PyYAML
            text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        path.write_text(text, encoding="utf-8")
        return path


@dataclass
class VLMDistillationRecipe:
    teacher_model_path: Path
    student_model_path: Path
    dataset_manifest: Path
    targets_path: Path
    output_dir: Path
    temperature: float = 2.0
    hard_label_weight: float = 0.5
    cache_teacher_responses: bool = True
    cache_top_k_logits: int = 0

    def validate(self, *, require_files: bool = False) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.hard_label_weight <= 1:
            raise ValueError("hard_label_weight must be in [0, 1]")
        if self.cache_top_k_logits < 0:
            raise ValueError("cache_top_k_logits cannot be negative")
        if require_files:
            for path in (self.teacher_model_path, self.student_model_path):
                if not path.is_dir():
                    raise FileNotFoundError(path)
            if not self.dataset_manifest.is_file():
                raise FileNotFoundError(self.dataset_manifest)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in value.items():
            if isinstance(item, Path):
                value[key] = str(item)
        return value


def prepare_lora_model(recipe: VLMLoRARecipe) -> tuple[Any, Any]:
    """Load local Qwen3-VL and attach PEFT adapters.

    This is intentionally separate from the offline demo: calling it allocates
    substantial GPU memory.  Training loops can use TRL or plain Transformers.
    """

    recipe.validate(require_files=True)
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError as exc:  # pragma: no cover - heavyweight integration
        raise RuntimeError("VLM LoRA requires transformers, accelerate, peft, and torch") from exc

    model = ModelClass.from_pretrained(
        str(recipe.model_path),
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if recipe.bf16 else "auto",
    )
    processor = AutoProcessor.from_pretrained(
        str(recipe.model_path), local_files_only=True, trust_remote_code=True
    )
    peft_config = LoraConfig(
        r=recipe.rank,
        lora_alpha=recipe.alpha,
        lora_dropout=recipe.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recipe.target_modules,
    )
    model = get_peft_model(model, peft_config)
    if recipe.freeze_vision_encoder:
        for name, parameter in model.named_parameters():
            if "visual" in name.casefold() or "vision" in name.casefold():
                parameter.requires_grad = False
    if recipe.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, processor


def distillation_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    *,
    temperature: float = 2.0,
    hard_label_weight: float = 0.5,
    ignore_index: int = -100,
) -> Any:
    """Combined hard-label CE and teacher-distribution KL loss."""

    if temperature <= 0 or not 0 <= hard_label_weight <= 1:
        raise ValueError("temperature > 0 and hard_label_weight in [0, 1] are required")
    try:
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("distillation_loss requires PyTorch") from exc
    vocabulary = student_logits.shape[-1]
    hard_loss = functional.cross_entropy(
        student_logits.reshape(-1, vocabulary),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
    per_token_kl = functional.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1)
    valid = labels.ne(ignore_index)
    soft_loss = per_token_kl[valid].mean() * (temperature**2)
    return hard_label_weight * hard_loss + (1 - hard_label_weight) * soft_loss


def create_teacher_response_dataset(
    samples: Iterable[SyntheticSample],
    teacher: Qwen3VLRunner,
    output_path: Path,
) -> Path:
    """Cache teacher-generated answers for response-level VLM distillation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for qa in sample.qa:
            result = teacher.infer(
                qa.question,
                [sample.image_path],
                context={"expected_answer": qa.answer, **sample.metadata},
            )
            rows.append(
                {
                    "id": qa.id,
                    "image_path": str(sample.image_path),
                    "question": qa.question,
                    "reference_answer": qa.answer,
                    "teacher_answer": result.text,
                    "teacher_backend": result.backend,
                    "simulated": result.simulated,
                }
            )
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return output_path


def count_trainable_parameters(model: Any) -> dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": float(total),
        "trainable_parameters": float(trainable),
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }
