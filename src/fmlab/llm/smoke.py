"""A fast offline suite demonstrating all LLM objectives without model downloads."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from fmlab.artifacts import ExperimentResult

from .data import ByteTokenizer, CausalTextDataset, InstructionDataset, InstructionExample
from .distillation import DistillationConfig, distillation_loss
from .lora import LoRAConfig, inject_lora, parameter_summary
from .model import TinyDecoderConfig, TinyDecoderLM
from .preference import dpo_loss, sequence_log_probabilities
from .quantization import benchmark_weight_quantization, fake_quantized_copy
from .retrieval import BM25Retriever, Document, RetrievalQuery, evaluate_retrieval
from .training import TrainConfig, evaluate_language_model, train_language_model
from .visualization import save_loss_curve, save_metrics_html


def _encode_preference(
    tokenizer: ByteTokenizer,
    prompt: str,
    response: str,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(f"Q:{prompt}\nA:", add_bos=True)
    response_ids = tokenizer.encode(response, add_eos=True)
    response_ids = response_ids[: max(2, max_length // 2)]
    prompt_ids = prompt_ids[: max_length - len(response_ids)]
    ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    padding = max_length - len(ids)
    return (
        torch.tensor([ids + [tokenizer.pad_token_id] * padding]),
        torch.tensor([labels + [-100] * padding]),
        torch.tensor([[1] * len(ids) + [0] * padding]),
    )


def run_offline_llm_smoke_suite(
    output_dir: str | Path,
    *,
    steps: int = 3,
    device: str = "cpu",
    seed: int = 7,
) -> ExperimentResult:
    """Exercise pretraining, SFT, PEFT, quantization, KD, DPO, and RAG.

    It is deliberately small enough for CI/CPU. Results demonstrate mechanics,
    not useful language quality. The local-model YAMLs scale the same concepts to
    Qwen checkpoints.
    """

    started = time.time()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = ByteTokenizer()
    model_config = TinyDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ffn_hidden_size=64,
    )
    general = [
        "attention connects tokens. lora trains small adapters. quantization saves memory. "
    ] * 12
    pretrain_data = CausalTextDataset(general, tokenizer, context_length=32, stride=16)
    base = TinyDecoderLM(model_config)
    before = evaluate_language_model(base, pretrain_data, device=device, max_batches=2)
    pretrain_history = train_language_model(
        base,
        pretrain_data,
        TrainConfig(steps=steps, batch_size=2, learning_rate=0.003, device=device, seed=seed),
    )
    after = evaluate_language_model(base, pretrain_data, device=device, max_batches=2)
    save_loss_curve(pretrain_history, destination / "pretraining_loss.png")

    domain_data = CausalTextDataset(
        ["법률 문서는 조문과 판례 근거를 인용한다. "] * 12,
        tokenizer,
        context_length=32,
        stride=16,
    )
    domain_before = evaluate_language_model(base, domain_data, device=device, max_batches=2)
    cpt_history = train_language_model(
        base,
        domain_data,
        TrainConfig(steps=steps, batch_size=2, learning_rate=0.001, device=device, seed=seed),
    )
    domain_after = evaluate_language_model(base, domain_data, device=device, max_batches=2)
    save_loss_curve(cpt_history, destination / "continued_pretraining_loss.png")

    sft_data = InstructionDataset(
        [
            InstructionExample("2+2?", "4"),
            InstructionExample("LoRA?", "low rank adapter"),
        ],
        tokenizer,
        max_length=64,
        prompt_template="Q:{prompt}\nA:",
    )
    full_sft = copy.deepcopy(base)
    sft_history = train_language_model(
        full_sft,
        sft_data,
        TrainConfig(steps=steps, batch_size=2, learning_rate=0.001, device=device, seed=seed),
    )
    save_loss_curve(sft_history, destination / "full_sft_loss.png")

    lora_model = copy.deepcopy(base)
    lora_modules = inject_lora(
        lora_model,
        LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=("q_proj", "v_proj")),
    )
    lora_history = train_language_model(
        lora_model,
        sft_data,
        TrainConfig(steps=steps, batch_size=2, learning_rate=0.005, device=device, seed=seed),
    )
    save_loss_curve(lora_history, destination / "lora_loss.png")

    qlora_model = copy.deepcopy(base)
    qlora_modules = inject_lora(
        qlora_model,
        LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=("q_proj", "v_proj"),
            quantization_bits=4,
        ),
    )
    qlora_history = train_language_model(
        qlora_model,
        sft_data,
        TrainConfig(steps=steps, batch_size=2, learning_rate=0.005, device=device, seed=seed),
    )
    save_loss_curve(qlora_history, destination / "qlora_loss.png")

    quant8 = benchmark_weight_quantization(base, bits=8)
    quant4 = benchmark_weight_quantization(base, bits=4)
    quantized4 = fake_quantized_copy(base, bits=4)
    quantized_loss = evaluate_language_model(quantized4, domain_data, device=device, max_batches=2)

    teacher = copy.deepcopy(base).to(device).eval()
    student_config = TinyDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_hidden_size=32,
    )
    student = TinyDecoderLM(student_config).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.003)
    distill_rows: list[dict[str, float]] = []
    batch = next(iter(torch.utils.data.DataLoader(pretrain_data, batch_size=2)))
    batch = {name: value.to(device) for name, value in batch.items()}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_logits = teacher(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        student_logits = student(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss, pieces = distillation_loss(
            student_logits,
            teacher_logits,
            batch["labels"],
            DistillationConfig(temperature=2.0),
        )
        loss.backward()
        optimizer.step()
        distill_rows.append(pieces)

    policy = copy.deepcopy(base).to(device).train()
    reference = copy.deepcopy(base).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    chosen_ids, chosen_labels, chosen_mask = _encode_preference(
        tokenizer, "safe?", "cite evidence", max_length=48
    )
    rejected_ids, rejected_labels, rejected_mask = _encode_preference(
        tokenizer, "safe?", "invent evidence", max_length=48
    )
    tensors = [
        tensor.to(device)
        for tensor in (
            chosen_ids,
            chosen_labels,
            chosen_mask,
            rejected_ids,
            rejected_labels,
            rejected_mask,
        )
    ]
    chosen_ids, chosen_labels, chosen_mask, rejected_ids, rejected_labels, rejected_mask = tensors
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    dpo_rows: list[dict[str, float]] = []
    for _ in range(steps):
        policy_optimizer.zero_grad(set_to_none=True)
        chosen_logits = policy(chosen_ids, attention_mask=chosen_mask).logits
        rejected_logits = policy(rejected_ids, attention_mask=rejected_mask).logits
        with torch.inference_mode():
            reference_chosen = reference(chosen_ids, attention_mask=chosen_mask).logits
            reference_rejected = reference(rejected_ids, attention_mask=rejected_mask).logits
        loss, dpo_metrics = dpo_loss(
            sequence_log_probabilities(chosen_logits, chosen_labels),
            sequence_log_probabilities(rejected_logits, rejected_labels),
            sequence_log_probabilities(reference_chosen, chosen_labels),
            sequence_log_probabilities(reference_rejected, rejected_labels),
            beta=0.1,
        )
        loss.backward()
        policy_optimizer.step()
        dpo_rows.append(dpo_metrics)

    documents = [
        Document("attention", "어텐션은 토큰 사이 관련성을 가중합한다."),
        Document("lora", "LoRA는 원본을 고정하고 저랭크 행렬을 학습한다."),
        Document("quant", "양자화는 더 적은 비트로 가중치를 표현한다."),
    ]
    rag_metrics = evaluate_retrieval(
        BM25Retriever(documents),
        [
            RetrievalQuery("q1", "저랭크 행렬 학습", frozenset({"lora"})),
            RetrievalQuery("q2", "가중치 비트 절약", frozenset({"quant"})),
        ],
        top_k=2,
    )

    metrics: dict[str, object] = {
        "from_scratch_loss_before": before["loss"],
        "from_scratch_loss_after": after["loss"],
        "continued_pretraining_domain_loss_before": domain_before["loss"],
        "continued_pretraining_domain_loss_after": domain_after["loss"],
        "full_sft_final_loss": sft_history[-1]["loss"],
        "lora_final_loss": lora_history[-1]["loss"],
        "lora_parameter_summary": parameter_summary(lora_model),
        "lora_modules": lora_modules,
        "qlora_final_loss": qlora_history[-1]["loss"],
        "qlora_parameter_summary": parameter_summary(qlora_model),
        "qlora_modules": qlora_modules,
        "int8": quant8.to_dict(),
        "int4": quant4.to_dict(),
        "int4_domain_loss": quantized_loss["loss"],
        "distillation_initial_loss": distill_rows[0]["total_loss"],
        "distillation_final_loss": distill_rows[-1]["total_loss"],
        "dpo_initial_loss": dpo_rows[0]["loss"],
        "dpo_final_loss": dpo_rows[-1]["loss"],
        "rag_recall_at_2": rag_metrics["recall@2"],
        "rag_mrr": rag_metrics["mrr"],
        "device": device,
    }
    metrics_path = destination / "suite_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    traces_path = destination / "objective_traces.json"
    traces_path.write_text(
        json.dumps(
            {"distillation": distill_rows, "dpo": dpo_rows, "rag": rag_metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = save_metrics_html(
        "Offline LLM objective smoke suite",
        metrics,
        [
            metrics_path.name,
            traces_path.name,
            "pretraining_loss.png",
            "continued_pretraining_loss.png",
            "full_sft_loss.png",
            "lora_loss.png",
            "qlora_loss.png",
        ],
        destination / "report.html",
    )
    result = ExperimentResult(
        experiment="llm_offline_smoke_suite",
        status="ok",
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters={"steps_per_objective": steps, "device": device, "seed": seed},
        artifacts=[str(metrics_path), str(traces_path), str(report_path)],
        notes=[
            "QLoRA and INT4 are inspectable simulations; packed CUDA kernels are benchmarked separately.",
            "Tiny random models demonstrate objective mechanics, not language quality.",
        ],
    )
    result.write(destination)
    return result
