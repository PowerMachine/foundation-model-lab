import pytest
import torch

from fmlab.llm.data import ByteTokenizer, CausalTextDataset
from fmlab.llm.model import TinyDecoderConfig, TinyDecoderLM
from fmlab.llm.training import TrainConfig, evaluate_language_model, train_language_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression test")
def test_cuda_evaluation_does_not_turn_parameters_into_inference_tensors() -> None:
    tokenizer = ByteTokenizer()
    dataset = CausalTextDataset(["attention test " * 8], tokenizer, context_length=16)
    model = TinyDecoderLM(
        TinyDecoderConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=16,
            d_model=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            ffn_hidden_size=32,
        )
    )
    evaluate_language_model(model, dataset, device="cuda", max_batches=1)
    history = train_language_model(
        model,
        dataset,
        TrainConfig(steps=1, batch_size=1, device="cuda", learning_rate=1e-3),
    )
    assert len(history) == 1
