"""A readable decoder-only Transformer for from-scratch experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TinyDecoderConfig:
    vocab_size: int = 260
    max_seq_len: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None
    ffn_hidden_size: int = 192
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.max_seq_len < 2 or self.n_layers < 1:
            raise ValueError("max_seq_len and n_layers are too small")


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean centering."""

    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance.to(values.dtype) + self.eps)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, length: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:length].to(device), self.sin[:length].to(device)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    cosine = torch.repeat_interleave(cosine, 2, dim=-1)[None, None, :, :]
    sine = torch.repeat_interleave(sine, 2, dim=-1)[None, None, :, :]
    return values * cosine.to(values.dtype) + _rotate_half(values) * sine.to(values.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyDecoderConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        self.dropout = config.dropout

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = (
            self.k_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )
        value = (
            self.v_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )
        cosine, sine = self.rope(length, device=hidden.device)
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)
        if self.n_kv_heads != self.n_heads:
            repetitions = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(repetitions, dim=1)
            value = value.repeat_interleave(repetitions, dim=1)

        # The explicit implementation is deliberate: tiny experiments can expose
        # the complete attention matrix, something fused kernels usually hide.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        allowed = causal[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(context), probabilities if return_attention else None


class SwiGLU(nn.Module):
    def __init__(self, config: TinyDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyDecoderConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, weights = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            return_attention=return_attention,
        )
        hidden = hidden + self.dropout(attention_output)
        hidden = hidden + self.dropout(self.feed_forward(self.ffn_norm(hidden)))
        return hidden, weights


class TinyDecoderLM(nn.Module):
    """Small GPT-like model with RoPE, RMSNorm, SwiGLU, MHA/GQA, and tied logits."""

    def __init__(self, config: TinyDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
    ) -> LMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )
        hidden = self.token_embedding(input_ids)
        collected: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, weights = block(
                hidden,
                attention_mask=attention_mask,
                return_attention=return_attentions,
            )
            if weights is not None:
                collected.append(weights)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        return LMOutput(
            logits=logits,
            loss=loss,
            attentions=tuple(collected) if return_attentions else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        generated = input_ids
        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            next_logits = self(window).logits[:, -1]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and 0 < top_k < next_logits.shape[-1]:
                    threshold = torch.topk(next_logits, top_k).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, -torch.inf)
                next_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), 1, generator=generator
                )
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        parameters = (parameter for parameter in self.parameters() if parameter.requires_grad)
        if not trainable_only:
            parameters = self.parameters()
        return sum(parameter.numel() for parameter in parameters)
