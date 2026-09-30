from __future__ import annotations

import copy

import torch

from fmlab.llm.data import ByteTokenizer, CausalTextDataset, InstructionDataset, InstructionExample
from fmlab.llm.lora import LoRAConfig, inject_lora, parameter_summary
from fmlab.llm.model import RMSNorm, TinyDecoderConfig, TinyDecoderLM
from fmlab.llm.quantization import benchmark_weight_quantization, fake_quantized_copy


def tiny_model() -> TinyDecoderLM:
    return TinyDecoderLM(
        TinyDecoderConfig(
            vocab_size=260,
            max_seq_len=24,
            d_model=16,
            n_layers=1,
            n_heads=2,
            ffn_hidden_size=32,
            dropout=0.0,
        )
    )


def test_byte_tokenizer_round_trip_for_korean_and_code() -> None:
    tokenizer = ByteTokenizer()
    text = "안녕, attention! x += 1"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == text
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id


def test_causal_and_sft_datasets_expose_expected_masks() -> None:
    tokenizer = ByteTokenizer()
    causal = CausalTextDataset(["abc " * 20], tokenizer, context_length=12, stride=6)
    assert causal[0]["input_ids"].shape == (12,)
    assert torch.equal(causal[0]["input_ids"], causal[0]["labels"])

    sft = InstructionDataset(
        [InstructionExample("sum?", "four")],
        tokenizer,
        max_length=24,
        prompt_template="Q:{prompt} A:",
    )
    row = sft[0]
    assert (row["labels"] == -100).any()
    assert (row["labels"] >= 0).any()
    assert torch.all(row["labels"][row["attention_mask"] == 0] == -100)


def test_tiny_decoder_loss_attention_and_generation() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    ids = torch.randint(0, 260, (2, 12))
    output = model(ids, labels=ids, return_attentions=True)
    assert output.logits.shape == (2, 12, 260)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert output.attentions is not None and len(output.attentions) == 1
    attention = output.attentions[0]
    assert attention.shape == (2, 2, 12, 12)
    assert torch.allclose(torch.triu(attention[0, 0], diagonal=1), torch.zeros(12, 12))
    generated = model.generate(ids[:1, :4], max_new_tokens=3, temperature=0)
    assert generated.shape == (1, 7)


def test_rms_norm_has_unit_root_mean_square_before_gain() -> None:
    norm = RMSNorm(8, eps=1e-8)
    values = torch.randn(3, 4, 8)
    output = norm(values)
    rms = output.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_lora_freezes_base_and_starts_as_no_op() -> None:
    torch.manual_seed(2)
    original = tiny_model().eval()
    adapted = copy.deepcopy(original).eval()
    ids = torch.randint(0, 260, (1, 8))
    reference = original(ids).logits
    replaced = inject_lora(
        adapted,
        LoRAConfig(rank=2, alpha=4, dropout=0, target_modules=("q_proj", "v_proj")),
    )
    assert replaced == ["blocks.0.attention.q_proj", "blocks.0.attention.v_proj"]
    assert torch.allclose(reference, adapted(ids).logits, atol=1e-6)
    summary = parameter_summary(adapted)
    assert 0 < summary["trainable_parameters"] < summary["total_parameters"]
    trainable_names = [name for name, value in adapted.named_parameters() if value.requires_grad]
    assert trainable_names and all("lora_" in name for name in trainable_names)


def test_qlora_simulation_and_quantization_tradeoff() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    qlora = copy.deepcopy(model)
    inject_lora(
        qlora,
        LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0,
            target_modules=("q_proj", "v_proj"),
            quantization_bits=4,
        ),
    )
    ids = torch.randint(0, 260, (1, 8))
    loss = qlora(ids, labels=ids).loss
    assert loss is not None
    loss.backward()
    assert all(
        parameter.grad is not None for parameter in qlora.parameters() if parameter.requires_grad
    )

    int8 = benchmark_weight_quantization(model, bits=8)
    int4 = benchmark_weight_quantization(model, bits=4)
    assert int4.estimated_packed_bytes < int8.estimated_packed_bytes
    assert int4.root_mean_squared_error >= int8.root_mean_squared_error
    quantized = fake_quantized_copy(model, bits=4)
    assert quantized(ids).logits.shape == model(ids).logits.shape
