"""Small, deterministic datasets used by the offline LLM experiments.

The byte tokenizer is intentionally simple: every UTF-8 byte has a stable token
id, so Korean, code, and arbitrary Unicode can be used without downloading a
vocabulary.  It is not meant to be a competitive tokenizer; it is a transparent
baseline against which a trained BPE/SentencePiece tokenizer can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """A reversible UTF-8 byte tokenizer with four explicit special tokens."""

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [self.byte_offset + value for value in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        values: list[int] = []
        special_fragments: list[str] = []
        for token_id in ids:
            value = int(token_id)
            if value >= self.byte_offset and value < self.vocab_size:
                values.append(value - self.byte_offset)
            elif not skip_special_tokens:
                special_fragments.append(self.token_label(value))
        decoded = bytes(values).decode("utf-8", errors="replace")
        return "".join(special_fragments) + decoded

    def token_label(self, token_id: int) -> str:
        special = {
            self.pad_token_id: self.pad_token,
            self.bos_token_id: self.bos_token,
            self.eos_token_id: self.eos_token,
            self.unk_token_id: self.unk_token,
        }
        if token_id in special:
            return special[token_id]
        if self.byte_offset <= token_id < self.vocab_size:
            value = token_id - self.byte_offset
            if 32 <= value <= 126:
                return chr(value)
            return f"0x{value:02x}"
        return f"<{token_id}>"

    def token_labels(self, ids: Iterable[int]) -> list[str]:
        return [self.token_label(int(token_id)) for token_id in ids]


class CausalTextDataset(Dataset[dict[str, torch.Tensor]]):
    """Chunk text into fixed-length causal-language-model examples.

    Each item contains the same sequence in ``input_ids`` and ``labels``.  The
    decoder shifts labels internally, which keeps the data representation easy
    to inspect.  Short final chunks are omitted rather than padded so every token
    shown in a visualization came from the corpus.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: ByteTokenizer,
        *,
        context_length: int = 64,
        stride: int | None = None,
    ) -> None:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        self.context_length = context_length
        self.stride = stride or context_length
        if self.stride < 1:
            raise ValueError("stride must be positive")

        all_ids: list[int] = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        if len(all_ids) < context_length:
            repeats = (context_length + len(all_ids) - 1) // max(len(all_ids), 1)
            all_ids = (all_ids * repeats)[:context_length] if all_ids else [1, 2] * context_length
        self._chunks = [
            torch.tensor(all_ids[start : start + context_length], dtype=torch.long)
            for start in range(0, len(all_ids) - context_length + 1, self.stride)
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self._chunks[index]
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


@dataclass(frozen=True)
class InstructionExample:
    prompt: str
    response: str


class InstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Tiny SFT dataset that masks prompt tokens from the training loss."""

    def __init__(
        self,
        examples: Sequence[InstructionExample],
        tokenizer: ByteTokenizer,
        *,
        max_length: int = 128,
        prompt_template: str = "User: {prompt}\nAssistant: ",
    ) -> None:
        if not examples:
            raise ValueError("at least one instruction example is required")
        if max_length < 4:
            raise ValueError("max_length must be at least 4")
        self.items: list[dict[str, torch.Tensor]] = []
        for example in examples:
            prompt_ids = tokenizer.encode(
                prompt_template.format(prompt=example.prompt), add_bos=True
            )
            response_ids = tokenizer.encode(example.response, add_eos=True)
            response_ids = response_ids[: max_length - 1]
            if response_ids:
                response_ids[-1] = tokenizer.eos_token_id
            prompt_ids = prompt_ids[: max_length - len(response_ids)]
            ids = prompt_ids + response_ids
            prompt_length = len(prompt_ids)
            labels = [-100] * prompt_length + ids[prompt_length:]
            padding = max_length - len(ids)
            self.items.append(
                {
                    "input_ids": torch.tensor(
                        ids + [tokenizer.pad_token_id] * padding, dtype=torch.long
                    ),
                    "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(ids) + [0] * padding),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
