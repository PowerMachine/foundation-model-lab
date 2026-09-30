from __future__ import annotations

import html
import json
import os
import random
import statistics
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, Sequence

from fmlab.artifacts import ExperimentResult, system_snapshot

from .lora_data import (
    Qwen3VLAssistantOnlyCollator,
    ToyMultimodalProcessor,
    VLMTrainingExample,
    build_multimodal_messages,
    grouped_holdout_split,
    limit_examples_by_group,
    load_canonical_vlm_manifest,
)
from .lora_audit import audit_lora_targeting, enforce_lora_scope
from .research import aggregate_predictions, paired_bootstrap_delta, write_provenance


Backend = Literal["mock", "local"]
EvaluationHook = Callable[
    [Any, Any, Sequence[VLMTrainingExample], str, "VLMLoRAExperimentConfig"],
    dict[str, Any],
]


class ResourceBlockedError(RuntimeError):
    """Expected external resource contention, distinct from implementation failure."""


class InsufficientVRAMError(ResourceBlockedError):
    """The explicitly selected CUDA device cannot safely admit the model."""


@dataclass
class VLMLoRAExperimentConfig:
    """Resource-bounded configuration for real or mock Qwen3-VL LoRA SFT."""

    model_path: Path
    dataset_manifest: Path
    artifact_dir: Path
    experiment: str = "qwen3_vl_8b_lora_e2e"
    adapter_output_dir: Path | None = None
    backend: Backend = "mock"
    seed: int = 17
    holdout_fraction: float = 0.25
    max_train_examples: int = 16
    max_eval_examples: int = 3
    max_steps: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_sequence_length: int = 1024
    max_new_tokens: int = 24
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    freeze_vision_encoder: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    cuda_device: int = 0
    min_pixels: int = 65_536
    max_pixels: int = 401_408
    save_adapter: bool = False
    max_wall_seconds: float = 900.0
    minimum_free_vram_gib: float = 24.0
    system_prompt: str = (
        "Answer the visual question concisely. Follow any requested output format exactly."
    )

    def validate(self, *, require_model: bool | None = None) -> None:
        if self.backend not in {"mock", "local"}:
            raise ValueError("backend must be 'mock' or 'local'")
        if not self.experiment.strip():
            raise ValueError("experiment cannot be empty")
        if not self.dataset_manifest.is_file():
            raise FileNotFoundError(self.dataset_manifest)
        should_require_model = self.backend == "local" if require_model is None else require_model
        if should_require_model and not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in (0, 1)")
        for name in (
            "max_train_examples",
            "max_eval_examples",
            "max_steps",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "max_sequence_length",
            "max_new_tokens",
            "rank",
            "alpha",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_eval_examples < 3:
            raise ValueError("max_eval_examples must be at least 3 for a minimally useful audit")
        if not isinstance(self.cuda_device, int) or self.cuda_device < 0:
            raise ValueError("cuda_device must be a non-negative integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.min_pixels < 1 or self.max_pixels < self.min_pixels:
            raise ValueError("pixel budget must satisfy 0 < min_pixels <= max_pixels")
        if self.max_wall_seconds <= 0 or self.minimum_free_vram_gib < 0:
            raise ValueError("invalid runtime or VRAM budget")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.save_adapter and self.adapter_output_dir is None:
            raise ValueError("adapter_output_dir is required when save_adapter=true")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("model_path", "dataset_manifest", "artifact_dir", "adapter_output_dir"):
            if value[key] is not None:
                value[key] = str(value[key])
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VLMLoRAExperimentConfig":
        known = set(cls.__dataclass_fields__)
        aliases_allowed = {"manifest_path", "output_dir", "device_map"}
        unknown = set(value) - known - aliases_allowed
        if unknown:
            raise ValueError(f"Unknown VLM LoRA config keys: {sorted(unknown)}")
        if "device_map" in value:
            raise ValueError(
                "device_map is forbidden for VLM training; set one explicit cuda_device instead"
            )
        aliases = {
            "dataset_manifest": value.get("dataset_manifest") or value.get("manifest_path"),
            "adapter_output_dir": value.get("adapter_output_dir") or value.get("output_dir"),
        }
        payload = {key: item for key, item in value.items() if key in known}
        payload.update({key: item for key, item in aliases.items() if item is not None})
        for key in ("model_path", "dataset_manifest", "artifact_dir", "adapter_output_dir"):
            if payload.get(key) is not None:
                payload[key] = Path(str(payload[key])).expanduser()
        return cls(**payload)


class TinyAdapterVLM:
    """A frozen tiny multimodal LM with a trainable low-rank residual adapter."""

    def __new__(cls, *, vocab_size: int = 263, hidden_size: int = 32, rank: int = 4) -> Any:
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.vision_projector = nn.Linear(8, hidden_size)
                self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
                self.adapter_down = nn.Linear(hidden_size, rank, bias=False)
                self.adapter_up = nn.Linear(rank, hidden_size, bias=False)
                nn.init.zeros_(self.adapter_up.weight)
                for parameter in self.parameters():
                    parameter.requires_grad = False
                for module in (self.adapter_down, self.adapter_up):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
                self.config = SimpleNamespace(use_cache=False)

            @property
            def device(self) -> Any:
                return next(self.parameters()).device

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any | None = None,
                pixel_values: Any | None = None,
                labels: Any | None = None,
                **_: Any,
            ) -> Any:
                del attention_mask
                hidden = self.embedding(input_ids)
                if pixel_values is not None:
                    hidden = hidden + self.vision_projector(pixel_values).unsqueeze(1)
                hidden, _ = self.recurrent(hidden)
                hidden = hidden + self.adapter_up(torch.tanh(self.adapter_down(hidden)))
                logits = self.lm_head(hidden)
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, vocab_size),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                return SimpleNamespace(loss=loss, logits=logits)

            def save_pretrained(self, output_dir: str | Path) -> None:
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "adapter_down": self.adapter_down.state_dict(),
                        "adapter_up": self.adapter_up.state_dict(),
                    },
                    destination / "toy_adapter.pt",
                )
                (destination / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "backend": "mock",
                            "rank": rank,
                            "hidden_size": hidden_size,
                            "warning": "Teaching backend only; not a Qwen adapter.",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        return Model()


def load_local_qwen3_vl_lora(
    config: VLMLoRAExperimentConfig,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load only local Qwen3-VL files and attach PEFT LoRA adapters."""

    config.validate(require_model=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError as exc:  # pragma: no cover - heavyweight optional path
        raise RuntimeError(
            "Local VLM LoRA requires torch, transformers, accelerate, and peft"
        ) from exc
    device_budget = enforce_cuda_device_budget(config.cuda_device, config.minimum_free_vram_gib)

    dtype: Any = torch.bfloat16 if config.bf16 else "auto"
    model = ModelClass.from_pretrained(
        str(config.model_path),
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": f"cuda:{config.cuda_device}"},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _enforce_single_cuda_residency(model, config.cuda_device)
    processor = AutoProcessor.from_pretrained(
        str(config.model_path),
        local_files_only=True,
        trust_remote_code=True,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    candidate_audit = audit_lora_targeting(model, config.target_modules, include_trainable=False)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, peft_config)
    if config.freeze_vision_encoder:
        for name, parameter in model.named_parameters():
            low_name = name.casefold()
            if "visual" in low_name or "vision" in low_name:
                parameter.requires_grad = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    trainable_audit = audit_lora_targeting(model, config.target_modules)
    enforce_lora_scope(trainable_audit, freeze_vision_encoder=config.freeze_vision_encoder)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return (
        model,
        processor,
        {
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "local_files_only": True,
            "vision_frozen": config.freeze_vision_encoder,
            "cuda_device": config.cuda_device,
            "device_budget_before_load": device_budget,
            "target_audit": {
                "before_peft": candidate_audit,
                "after_peft": trainable_audit,
            },
        },
    )


def run_vlm_lora_experiment(
    config: VLMLoRAExperimentConfig,
    *,
    evaluation_hook: EvaluationHook | None = None,
) -> Path:
    """Run data expansion, leakage-safe eval, LoRA SFT, and visual reporting."""

    import torch

    config.validate()
    started_at = time.time()
    wall_started = time.perf_counter()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    examples = load_canonical_vlm_manifest(config.dataset_manifest)
    split = grouped_holdout_split(
        examples,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed,
    )
    train_examples = limit_examples_by_group(split.train, config.max_train_examples)
    evaluation_examples = limit_examples_by_group(split.evaluation, config.max_eval_examples)
    if not train_examples or not evaluation_examples:
        raise ValueError("Resource limits removed all train or evaluation examples")

    provenance_path = write_provenance(
        config.artifact_dir / "provenance.json",
        config=config.to_dict(),
        inputs=[config.dataset_manifest, *(item.image_path for item in examples)],
        system=system_snapshot(),
    )

    load_started = time.perf_counter()
    if config.backend == "local":
        _reset_cuda_peaks()
        try:
            model, processor, backend_details = load_local_qwen3_vl_lora(config)
        except ResourceBlockedError as error:
            return write_failed_vlm_lora_result(
                config, error, started_at=started_at, provenance_path=provenance_path
            )
    else:
        model = TinyAdapterVLM(rank=min(config.rank, 16))
        processor = ToyMultimodalProcessor()
        backend_details = {
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "local_files_only": True,
            "warning": "Mock backend validates mechanics, not Qwen3-VL quality.",
        }
    model_load_seconds = time.perf_counter() - load_started
    memory_after_load = _cuda_memory_snapshot() if config.backend == "local" else {}

    collator = Qwen3VLAssistantOnlyCollator(
        processor,
        max_sequence_length=config.max_sequence_length,
        system_prompt=config.system_prompt,
    )
    parameter_metrics = _parameter_metrics(model)
    hook = evaluation_hook or (
        _evaluate_generation if config.backend == "local" else _evaluate_teacher_forced
    )

    before_started = time.perf_counter()
    before = hook(model, processor, evaluation_examples, "before", config)
    before_seconds = time.perf_counter() - before_started

    if config.backend == "local":
        _reset_cuda_peaks()
    trace, stop_reason = _train_adapter(
        model,
        collator,
        train_examples,
        config,
        wall_started=wall_started,
    )
    train_seconds = sum(float(row["duration_seconds"]) for row in trace)
    training_memory = _cuda_memory_snapshot() if config.backend == "local" else {}

    after_started = time.perf_counter()
    after = hook(model, processor, evaluation_examples, "after", config)
    after_seconds = time.perf_counter() - after_started
    comparison = _compare_evaluations(before, after, seed=config.seed)

    trace_path = config.artifact_dir / "training_trace.jsonl"
    _write_jsonl(trace_path, trace)
    before_path = config.artifact_dir / "predictions_before.jsonl"
    after_path = config.artifact_dir / "predictions_after.jsonl"
    _write_jsonl(before_path, list(before.get("records") or []))
    _write_jsonl(after_path, list(after.get("records") or []))

    artifacts = [str(trace_path), str(before_path), str(after_path), str(provenance_path)]
    adapter_path: Path | None = None
    if config.save_adapter:
        assert config.adapter_output_dir is not None
        adapter_path = config.adapter_output_dir
        model.save_pretrained(adapter_path)
        artifacts.append(str(adapter_path))

    finished_at = time.time()
    total_supervised_tokens = sum(int(row["supervised_tokens"]) for row in trace)
    metrics: dict[str, Any] = {
        "backend": config.backend,
        "simulated": config.backend == "mock",
        "completed_steps": len(trace),
        "requested_steps": config.max_steps,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "final_loss": trace[-1]["loss"] if trace else None,
        "best_loss": min((row["loss"] for row in trace), default=None),
        "total_supervised_tokens": total_supervised_tokens,
        "supervised_tokens_per_second": (
            total_supervised_tokens / train_seconds if train_seconds else 0.0
        ),
        "parameters": parameter_metrics,
        "dataset": {
            **split.summary(),
            "manifest_examples": len(examples),
            "budgeted_train_examples": len(train_examples),
            "budgeted_evaluation_examples": len(evaluation_examples),
            "budgeted_train_tasks": dict(
                sorted(Counter(item.task for item in train_examples).items())
            ),
            "budgeted_evaluation_tasks": dict(
                sorted(Counter(item.task for item in evaluation_examples).items())
            ),
            "missing_budgeted_evaluation_tasks": sorted(
                {item.task for item in examples} - {item.task for item in evaluation_examples}
            ),
        },
        "evaluation_before": before,
        "evaluation_after": after,
        "comparison": comparison,
        "runtime_seconds": {
            "model_load": model_load_seconds,
            "evaluation_before": before_seconds,
            "training_steps": train_seconds,
            "evaluation_after": after_seconds,
            "total": time.perf_counter() - wall_started,
        },
        "gpu_memory_after_load": memory_after_load,
        "gpu_memory_training_peak": training_memory,
        "backend_details": backend_details,
    }
    report_path = config.artifact_dir / "report.html"
    result_path = config.artifact_dir / "result.json"
    artifacts.extend([str(report_path), str(result_path)])
    result = ExperimentResult(
        experiment=config.experiment,
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        metrics=metrics,
        parameters=config.to_dict(),
        artifacts=artifacts,
        notes=[
            "Evaluation is grouped by source image to prevent QA leakage.",
            "Only assistant answer tokens contribute to cross-entropy loss.",
            "This bounded synthetic run is a systems smoke test, not a model-quality benchmark.",
            "Mock results are explicitly simulated and must not be presented as Qwen3-VL results."
            if config.backend == "mock"
            else "All model and processor files were loaded with local_files_only=True.",
        ],
    )
    result.write(config.artifact_dir)
    render_vlm_lora_report(result_path, report_path)
    return result_path


def write_failed_vlm_lora_result(
    config: VLMLoRAExperimentConfig,
    error: BaseException,
    *,
    started_at: float | None = None,
    provenance_path: Path | None = None,
) -> Path:
    """Persist an auditable failure instead of losing an expensive run's context."""

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    if provenance_path is None:
        provenance_path = write_provenance(
            config.artifact_dir / "provenance.json",
            config=config.to_dict(),
            inputs=[config.dataset_manifest],
            system=system_snapshot(),
        )
    status = "blocked_resource" if isinstance(error, ResourceBlockedError) else "failed"
    result_path = config.artifact_dir / "result.json"
    report_path = config.artifact_dir / "report.html"
    artifacts = [str(provenance_path), str(report_path), str(result_path)]
    result = ExperimentResult(
        experiment=config.experiment,
        status=status,
        started_at=started_at if started_at is not None else time.time(),
        finished_at=time.time(),
        metrics={"backend": config.backend, "simulated": config.backend == "mock"},
        parameters=config.to_dict(),
        error=f"{type(error).__name__}: {error}",
        artifacts=artifacts,
        notes=[
            "Expected external resource contention; no model load was attempted."
            if status == "blocked_resource"
            else "The failure is retained without reclassifying the implementation error."
        ],
    )
    path = result.write(config.artifact_dir)
    render_vlm_lora_report(path, config.artifact_dir / "report.html")
    return path


def render_vlm_lora_report(result_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(result_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(output_path) if output_path else source.with_name("report.html")
    metrics = payload.get("metrics") or {}
    status = str(payload.get("status", "unknown"))
    if status != "completed":
        body = (
            "<h1>Qwen3-VL LoRA run failed</h1>"
            f"<p class='bad'>{html.escape(str(payload.get('error') or 'unknown error'))}</p>"
        )
    else:
        trace_path = source.with_name("training_trace.jsonl")
        trace = _read_jsonl_if_exists(trace_path)
        before = metrics.get("evaluation_before") or {}
        after = metrics.get("evaluation_after") or {}
        params = metrics.get("parameters") or {}
        runtime = metrics.get("runtime_seconds") or {}
        dataset = metrics.get("dataset") or {}
        comparison = metrics.get("comparison") or {}
        body = f"""
<header><p class='eyebrow'>RESOURCE-BOUNDED MULTIMODAL ADAPTATION</p>
<h1>Qwen3-VL-8B · LoRA SFT</h1><p>assistant-only loss · image-group holdout · local-only weights</p></header>
<section class='cards'>
{_card("Backend", metrics.get("backend"), "simulated" if metrics.get("simulated") else "real local")}
{_card("Steps", metrics.get("completed_steps"), f"requested {metrics.get('requested_steps')}")}
{_card("Trainable", _percent(params.get("trainable_percent")), _integer(params.get("trainable_parameters")))}
{_card("Final loss", _decimal(metrics.get("final_loss")), f"best {_decimal(metrics.get('best_loss'))}")}
</section>
<section class='grid'><article><h2>Optimization trace</h2>{_loss_svg(trace)}</article>
<article><h2>Leakage-safe data contract</h2><dl>
<dt>Manifest examples</dt><dd>{dataset.get("manifest_examples")}</dd>
<dt>Budgeted train / eval</dt><dd>{dataset.get("budgeted_train_examples")} / {dataset.get("budgeted_evaluation_examples")}</dd>
<dt>Source overlap</dt><dd>{html.escape(str(dataset.get("source_overlap")))}</dd>
<dt>Assistant tokens</dt><dd>{metrics.get("total_supervised_tokens")}</dd>
</dl></article></section>
<section><h2>Held-out evaluation</h2>{_evaluation_table(before, after, comparison)}</section>
<section class='grid'><article><h2>Runtime (seconds)</h2>{_mapping_table(runtime)}</article>
<article><h2>Memory & backend</h2>{_mapping_table(metrics.get("gpu_memory_training_peak") or metrics.get("backend_details") or {})}</article></section>
<aside><strong>Interpretation boundary.</strong> The synthetic held-out set validates the full training and evaluation path. It is not a claim of general VLM quality. Mock backend values validate mechanics only.</aside>
"""
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>VLM LoRA report</title>
<style>{_REPORT_CSS}</style></head><body><main>{body}</main></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def _train_adapter(
    model: Any,
    collator: Qwen3VLAssistantOnlyCollator,
    examples: Sequence[VLMTrainingExample],
    config: VLMLoRAExperimentConfig,
    *,
    wall_started: float,
) -> tuple[list[dict[str, Any]], str | None]:
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Model has no trainable adapter parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    trace: list[dict[str, Any]] = []
    cursor = 0
    stop_reason: str | None = None
    input_device = _model_input_device(model)
    micro_batch_size = config.per_device_batch_size

    for optimizer_step in range(1, config.max_steps + 1):
        if time.perf_counter() - wall_started >= config.max_wall_seconds:
            stop_reason = f"max_wall_seconds={config.max_wall_seconds} reached"
            break
        step_started = time.perf_counter()
        micro_losses: list[float] = []
        supervised_tokens = 0
        for _ in range(config.gradient_accumulation_steps):
            batch_examples = [
                examples[(cursor + offset) % len(examples)] for offset in range(micro_batch_size)
            ]
            cursor += micro_batch_size
            batch = _move_batch(collator(batch_examples), input_device)
            supervised_tokens += int(batch["labels"].ne(-100).sum().item())
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if config.backend == "local" and config.bf16
                else nullcontext()
            )
            with context:
                output = model(**batch)
                loss = output.loss / config.gradient_accumulation_steps
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {optimizer_step}")
            loss.backward()
            micro_losses.append(float(loss.detach().cpu()) * config.gradient_accumulation_steps)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if config.backend == "local":
            torch.cuda.synchronize()
        trace.append(
            {
                "step": optimizer_step,
                "loss": statistics.fmean(micro_losses),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": config.learning_rate,
                "supervised_tokens": supervised_tokens,
                "duration_seconds": time.perf_counter() - step_started,
            }
        )
    return trace, stop_reason


def _evaluate_teacher_forced(
    model: Any,
    processor: Any,
    examples: Sequence[VLMTrainingExample],
    phase: str,
    config: VLMLoRAExperimentConfig,
) -> dict[str, Any]:
    import torch

    collator = Qwen3VLAssistantOnlyCollator(
        processor,
        max_sequence_length=config.max_sequence_length,
        system_prompt=config.system_prompt,
    )
    was_training = model.training
    model.eval()
    losses: list[float] = []
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for example in examples:
            batch = _move_batch(collator([example]), _model_input_device(model))
            output = model(**batch)
            shift_logits = output.logits[:, :-1]
            shift_labels = batch["labels"][:, 1:]
            valid = shift_labels.ne(-100)
            predicted = shift_logits.argmax(-1)
            correct = int((predicted[valid] == shift_labels[valid]).sum().item())
            count = int(valid.sum().item())
            score = correct / count if count else 0.0
            losses.append(float(output.loss.detach().cpu()))
            records.append(
                {
                    "id": example.id,
                    "task": example.task,
                    "reference": example.answer,
                    "prediction": "<teacher-forced-token-proxy>",
                    "answer_type": example.answer_type,
                    "assistant_token_accuracy": score,
                    "score": score,
                    "assistant_tokens": count,
                }
            )
    if was_training:
        model.train()
    return {
        "phase": phase,
        "metric_kind": "teacher_forced_proxy",
        "count": len(records),
        "loss": statistics.fmean(losses),
        "assistant_token_accuracy": statistics.fmean(row["score"] for row in records),
        "records": records,
        "warning": "Mock teacher-forced metrics are not generation quality.",
    }


def _evaluate_generation(
    model: Any,
    processor: Any,
    examples: Sequence[VLMTrainingExample],
    phase: str,
    config: VLMLoRAExperimentConfig,
) -> dict[str, Any]:
    import torch

    was_training = model.training
    model.eval()
    old_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if old_cache is not None:
        model.config.use_cache = True
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for example in examples:
            request_started = time.perf_counter()
            messages = build_multimodal_messages(
                example,
                include_answer=False,
                system_prompt=config.system_prompt,
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            if prompt_tokens > config.max_sequence_length:
                raise ValueError(
                    f"{example.id} generation prompt has {prompt_tokens} tokens, exceeding "
                    f"max_sequence_length={config.max_sequence_length}; lower max_pixels"
                )
            inputs = _move_batch(inputs, _model_input_device(model))
            generated = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            )
            prompt_length = inputs["input_ids"].shape[1]
            answer_ids = generated[:, prompt_length:]
            prediction = processor.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            records.append(
                {
                    "id": example.id,
                    "source_id": example.source_id,
                    "task": example.task,
                    "question": example.question,
                    "reference": example.answer,
                    "prediction": prediction,
                    "answer_type": example.answer_type,
                    "latency_seconds": time.perf_counter() - request_started,
                }
            )
    if old_cache is not None:
        model.config.use_cache = old_cache
    if was_training:
        model.train()
    aggregate = aggregate_predictions(records)
    aggregate.update(
        {
            "phase": phase,
            "metric_kind": "greedy_generation",
            "mean_latency_seconds": statistics.fmean(row["latency_seconds"] for row in records),
        }
    )
    return aggregate


def _compare_evaluations(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    before_records = {str(row.get("id")): row for row in before.get("records") or []}
    after_records = {str(row.get("id")): row for row in after.get("records") or []}
    common = sorted(before_records.keys() & after_records.keys())
    if not common:
        return {"paired_examples": 0}
    if before.get("metric_kind") == "greedy_generation":
        baseline = [float(before_records[key].get("exact_match", 0.0)) for key in common]
        candidate = [float(after_records[key].get("exact_match", 0.0)) for key in common]
        metric = "exact_match"
    else:
        baseline = [float(before_records[key].get("score", 0.0)) for key in common]
        candidate = [float(after_records[key].get("score", 0.0)) for key in common]
        metric = "assistant_token_accuracy"
    value: dict[str, Any] = {
        "paired_examples": len(common),
        "metric": metric,
        "before": statistics.fmean(baseline),
        "after": statistics.fmean(candidate),
        "delta": statistics.fmean(candidate) - statistics.fmean(baseline),
    }
    if len(common) >= 2:
        value["paired_bootstrap"] = paired_bootstrap_delta(
            baseline,
            candidate,
            samples=1_000,
            seed=seed,
        )
    return value


def _parameter_metrics(model: Any) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100 * trainable / total if total else 0.0,
    }


def _model_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return next(model.parameters()).device


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def enforce_cuda_device_budget(
    cuda_device: int,
    minimum_gib: float,
    *,
    cuda_api: Any | None = None,
) -> dict[str, Any]:
    """Validate one explicit training GPU immediately before model loading."""

    if cuda_api is None:
        import torch

        cuda_api = torch.cuda
    if not cuda_api.is_available():
        raise ResourceBlockedError(
            "Local 8B LoRA requires CUDA; use backend='mock' for CPU mechanics checks"
        )
    device_count = int(cuda_api.device_count())
    if cuda_device < 0 or cuda_device >= device_count:
        raise ValueError(
            f"cuda_device={cuda_device} is invalid for {device_count} visible CUDA device(s)"
        )
    free_bytes, total_bytes = cuda_api.mem_get_info(cuda_device)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    if free_gib < minimum_gib:
        raise InsufficientVRAMError(
            f"cuda:{cuda_device} has only {free_gib:.1f} GiB free; "
            f"minimum_free_vram_gib={minimum_gib:.1f}. Wait for capacity."
        )
    return {
        "cuda_device": cuda_device,
        "name": str(cuda_api.get_device_name(cuda_device)),
        "free_gib": free_gib,
        "total_gib": total_gib,
        "minimum_free_vram_gib": minimum_gib,
    }


def _enforce_single_cuda_residency(model: Any, cuda_device: int) -> None:
    """Reject implicit CPU/disk offload and placement on any other GPU."""

    raw_map = getattr(model, "hf_device_map", None)
    if isinstance(raw_map, Mapping) and raw_map:
        placements = set(raw_map.values())
    else:
        placements = {str(parameter.device) for parameter in model.parameters()}
    normalized = {
        f"cuda:{placement}" if isinstance(placement, int) else str(placement).casefold()
        for placement in placements
    }
    expected = f"cuda:{cuda_device}"
    if normalized != {expected}:
        raise RuntimeError(
            "Local LoRA training forbids CPU/disk/offload or multi-device placement; "
            + f"expected only {expected}, observed {sorted(normalized)}"
        )


def _reset_cuda_peaks() -> None:
    import torch

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def _cuda_memory_snapshot() -> dict[str, Any]:
    import torch

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "allocated_gib": torch.cuda.memory_allocated(index) / 1024**3,
                "reserved_gib": torch.cuda.memory_reserved(index) / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 1024**3,
                "free_gib": free / 1024**3,
                "total_gib": total / 1024**3,
            }
        )
    return {
        "devices": devices,
        "aggregate_peak_allocated_gib": sum(d["peak_allocated_gib"] for d in devices),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _card(label: str, value: Any, detail: Any) -> str:
    return (
        "<article class='card'><span>"
        + html.escape(str(label))
        + "</span><strong>"
        + html.escape(str(value))
        + "</strong><small>"
        + html.escape(str(detail))
        + "</small></article>"
    )


def _decimal(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}%"


def _integer(value: Any) -> str:
    return "—" if value is None else f"{int(value):,} params"


def _loss_svg(trace: Sequence[Mapping[str, Any]]) -> str:
    if not trace:
        return "<p>No optimizer steps completed.</p>"
    values = [float(row["loss"]) for row in trace]
    low, high = min(values), max(values)
    span = max(high - low, 1e-8)
    points = []
    for index, value in enumerate(values):
        x = 24 + index * (552 / max(1, len(values) - 1))
        y = 164 - (value - low) / span * 125
        points.append(f"{x:.1f},{y:.1f}")
    return (
        "<svg viewBox='0 0 600 190' role='img' aria-label='training loss'>"
        "<path d='M24 164H576M24 39V164' class='axis'/>"
        f"<polyline points='{' '.join(points)}' class='loss'/><text x='28' y='32'>{high:.4f}</text>"
        f"<text x='28' y='182'>{low:.4f}</text></svg>"
    )


def _evaluation_table(
    before: Mapping[str, Any], after: Mapping[str, Any], comparison: Mapping[str, Any]
) -> str:
    metric = str(comparison.get("metric") or "metric")
    rows = [
        (metric, comparison.get("before"), comparison.get("after"), comparison.get("delta")),
        (
            "loss",
            before.get("loss"),
            after.get("loss"),
            _difference(before.get("loss"), after.get("loss")),
        ),
        (
            "exact_match",
            before.get("exact_match"),
            after.get("exact_match"),
            _difference(before.get("exact_match"), after.get("exact_match")),
        ),
        (
            "ANLS",
            before.get("anls"),
            after.get("anls"),
            _difference(before.get("anls"), after.get("anls")),
        ),
    ]
    cells = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_decimal(left)}</td><td>{_decimal(right)}</td><td>{_decimal(delta)}</td></tr>"
        for name, left, right, delta in rows
        if left is not None or right is not None
    )
    return f"<table><thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Δ</th></tr></thead><tbody>{cells}</tbody></table>"


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _mapping_table(value: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(_format_value(item))}</td></tr>"
        for key, item in value.items()
        if key != "devices"
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


_REPORT_CSS = """
:root{color-scheme:dark;--bg:#081018;--panel:#101e2b;--line:#284457;--ink:#e9f4f7;--muted:#9db1bc;--a:#47e0b0;--b:#64a7ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#17344a,transparent 35%),var(--bg);color:var(--ink);font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}main{max-width:1180px;margin:auto;padding:48px 24px 72px}header{padding:36px;border:1px solid var(--line);background:linear-gradient(135deg,#102637,#101a26);border-radius:18px}h1{font:700 clamp(2rem,5vw,4rem)/1.02 system-ui;margin:.2rem 0}h2{font:650 1.2rem system-ui}.eyebrow{color:var(--a);letter-spacing:.16em}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card,article,section{border:1px solid var(--line);background:rgba(16,30,43,.86);border-radius:14px;padding:20px}.card span,.card small,dt{color:var(--muted)}.card strong{display:block;font:700 1.55rem system-ui;margin:8px 0}.grid{display:grid;grid-template-columns:1.3fr .7fr;gap:16px;margin:16px 0}section{margin:16px 0}.grid section{margin:0}svg{width:100%;height:auto}.axis{stroke:#496578;fill:none}.loss{stroke:var(--a);stroke-width:4;fill:none;stroke-linejoin:round;stroke-linecap:round}svg text{fill:var(--muted);font-size:12px}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px}th{color:var(--muted)}dl{display:grid;grid-template-columns:1fr 1fr;gap:8px}dd{margin:0;text-align:right}aside{border-left:4px solid var(--a);padding:16px 20px;background:#10251f}.bad{color:#ff8d96}@media(max-width:820px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}@media(max-width:480px){.cards{grid-template-columns:1fr}}
"""
