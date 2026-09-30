from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from fmlab.llm.distillation import DistillationConfig, distillation_loss
from fmlab.llm.experiments import TinyExperimentConfig, run_tiny_from_scratch
from fmlab.llm.optional import OptionalDependencyError, require_optional
from fmlab.llm.preference import dpo_loss, sequence_log_probabilities
from fmlab.llm.retrieval import (
    BM25Retriever,
    DenseRetriever,
    Document,
    RetrievalQuery,
    build_grounded_prompt,
    evaluate_retrieval,
)
from fmlab.llm.workflows import LocalWorkflowConfig, WorkflowKind, workflow_plan


def test_distillation_combines_soft_and_hard_targets() -> None:
    student = torch.randn(2, 6, 11, requires_grad=True)
    teacher = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    labels[0, :2] = -100
    loss, pieces = distillation_loss(student, teacher, labels, DistillationConfig(temperature=2.0))
    assert torch.isfinite(loss)
    assert pieces["soft_kl"] >= 0
    loss.backward()
    assert student.grad is not None


def test_sequence_logps_and_dpo_reward_preferred_policy() -> None:
    chosen = torch.tensor([-2.0, -1.0])
    rejected = torch.tensor([-5.0, -3.0])
    reference_chosen = torch.tensor([-4.0, -3.0])
    reference_rejected = torch.tensor([-4.5, -3.5])
    loss, metrics = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.2)
    assert loss.item() < 0.6932
    assert metrics["reward_accuracy"] == 1.0

    logits = torch.randn(1, 5, 7)
    labels = torch.tensor([[-100, -100, 2, 3, 4]])
    logp = sequence_log_probabilities(logits, labels)
    assert logp.shape == (1,)
    assert logp.item() < 0


def test_bm25_dense_retrieval_and_grounded_prompt() -> None:
    documents = [
        Document("lora", "LoRA는 저랭크 어댑터를 학습한다."),
        Document("quant", "양자화는 가중치 비트를 줄인다."),
        Document("rag", "RAG는 관련 문서를 검색한다."),
    ]
    query = RetrievalQuery("q", "가중치 비트 양자화", frozenset({"quant"}))
    bm25 = BM25Retriever(documents)
    report = evaluate_retrieval(bm25, [query], top_k=2)
    assert report["recall@2"] == 1.0
    hits = bm25.search(query.text, top_k=2)
    prompt = build_grounded_prompt(query.text, hits)
    assert "[quant]" in prompt and "근거 없음" in prompt

    vectors = {
        documents[0].text: [1.0, 0.0],
        documents[1].text: [0.0, 1.0],
        documents[2].text: [-1.0, 0.0],
        "memory": [0.0, 1.0],
    }

    def encoder(texts):
        return np.array([vectors[text] for text in texts], dtype=np.float32)

    dense = DenseRetriever(documents, encoder)
    assert dense.search("memory", top_k=1)[0].document.id == "quant"


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "missing_llm_package":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(OptionalDependencyError, match=r"pip install -e.*llm"):
        require_optional("missing_llm_package", extra="llm", purpose="test workflow")


def test_workflow_plan_is_offline_and_validates_local_path(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = LocalWorkflowConfig(
        kind=WorkflowKind.LORA,
        model_path=str(model_dir),
        output_dir=str(tmp_path / "output"),
        max_steps=2,
    )
    config.validate()
    plan = workflow_plan(config)
    assert plan["offline_only"] is True
    assert "peft" in plan["optional_packages"]


def test_tiny_experiment_writes_visual_report(tmp_path) -> None:
    result = run_tiny_from_scratch(
        tmp_path,
        TinyExperimentConfig(
            context_length=16,
            d_model=16,
            n_layers=1,
            n_heads=2,
            ffn_hidden_size=32,
            steps=1,
            batch_size=2,
            learning_rate=0.002,
            device="cpu",
            max_new_tokens=2,
        ),
        texts=["attention lora quantization " * 4],
    )
    assert result.status == "ok"
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "attention_layer0_head0.svg").is_file()
    assert (tmp_path / "report.html").is_file()
    assert result.metrics["total"] > 0
