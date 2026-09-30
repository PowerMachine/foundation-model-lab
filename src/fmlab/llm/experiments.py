"""End-to-end tiny decoder experiment that emits human-readable artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from fmlab.artifacts import ExperimentResult

from .data import ByteTokenizer, CausalTextDataset
from .model import TinyDecoderConfig, TinyDecoderLM
from .training import TrainConfig, count_parameters, evaluate_language_model, train_language_model
from .visualization import save_attention_svg, save_loss_curve, save_metrics_html


DEFAULT_CORPUS = [
    "어텐션은 각 토큰이 이전 토큰에서 필요한 정보를 고르는 연산이다. ",
    "파인튜닝은 이미 배운 모델을 새로운 데이터와 목적에 맞게 조정한다. ",
    "LoRA는 원본 가중치를 고정하고 작은 저랭크 행렬만 학습한다. ",
    "Quantization reduces memory by representing weights with fewer bits. ",
    "작은 모델은 실험이 빨라서 원인과 결과를 비교하기 좋다. ",
] * 8


@dataclass(frozen=True)
class TinyExperimentConfig:
    context_length: int = 48
    d_model: int = 48
    n_layers: int = 2
    n_heads: int = 4
    ffn_hidden_size: int = 128
    steps: int = 20
    batch_size: int = 4
    learning_rate: float = 3e-3
    seed: int = 42
    device: str = "auto"
    max_new_tokens: int = 24


def run_tiny_from_scratch(
    output_dir: str | Path,
    config: TinyExperimentConfig | None = None,
    *,
    texts: list[str] | None = None,
) -> ExperimentResult:
    """Train a tiny model and save loss, generation, and attention artifacts."""

    started = time.time()
    settings = config or TinyExperimentConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    corpus = texts or DEFAULT_CORPUS
    dataset = CausalTextDataset(
        corpus,
        tokenizer,
        context_length=settings.context_length,
        stride=settings.context_length // 2,
    )
    model_config = TinyDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=settings.context_length,
        d_model=settings.d_model,
        n_layers=settings.n_layers,
        n_heads=settings.n_heads,
        ffn_hidden_size=settings.ffn_hidden_size,
    )
    torch.manual_seed(settings.seed)
    model = TinyDecoderLM(model_config)
    device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device
    if device == "auto":
        device = "cpu"
    before = evaluate_language_model(model, dataset, device=device, max_batches=4)
    history = train_language_model(
        model,
        dataset,
        TrainConfig(
            steps=settings.steps,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            seed=settings.seed,
            device=settings.device,
        ),
    )
    after = evaluate_language_model(model, dataset, device=device, max_batches=4)
    model.eval()
    prompt = "어텐션"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generator = torch.Generator(device=prompt_ids.device).manual_seed(settings.seed)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=settings.max_new_tokens,
        temperature=0.8,
        top_k=24,
        generator=generator,
    )[0].cpu()
    generated_text = tokenizer.decode(generated)

    example_ids = dataset[0]["input_ids"][: min(settings.context_length, 24)].unsqueeze(0)
    example_ids = example_ids.to(next(model.parameters()).device)
    with torch.inference_mode():
        attention_output = model(example_ids, return_attentions=True)
    assert attention_output.attentions is not None
    attention = attention_output.attentions[0][0, 0]

    config_path = destination / "config.json"
    config_path.write_text(
        json.dumps(
            {"experiment": asdict(settings), "model": asdict(model_config)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path = destination / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    sample_path = destination / "generated_sample.txt"
    sample_path.write_text(generated_text, encoding="utf-8")
    curve_path = save_loss_curve(history, destination / "loss_curve.png")
    attention_path = save_attention_svg(
        attention,
        tokenizer.token_labels(example_ids[0].cpu()),
        destination / "attention_layer0_head0.svg",
    )
    metrics: dict[str, object] = {
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
        "loss_reduction": before["loss"] - after["loss"],
        **count_parameters(model),
        "dataset_chunks": len(dataset),
        "device": str(next(model.parameters()).device),
    }
    report_path = save_metrics_html(
        "Tiny decoder-only Transformer",
        metrics,
        [curve_path.name, attention_path.name, sample_path.name, history_path.name],
        destination / "report.html",
    )
    artifacts = [
        str(config_path),
        str(history_path),
        str(sample_path),
        str(curve_path),
        str(attention_path),
        str(report_path),
    ]
    result = ExperimentResult(
        experiment="llm_tiny_from_scratch",
        status="ok",
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters=asdict(settings),
        artifacts=artifacts,
        notes=[
            "Byte-level tokenizer is a transparent baseline, not a production tokenizer.",
            "Attention SVG shows layer 0, head 0 for one short training sequence.",
        ],
    )
    result.write(destination)
    return result
