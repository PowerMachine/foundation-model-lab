from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol


class ModelProvider(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class ScriptedProvider:
    """Deterministic provider used to test the entire tool loop offline."""

    def __init__(self, responses: Iterable[str | dict[str, object]]) -> None:
        self.responses = iter(responses)

    def complete(self, messages: list[dict[str, str]]) -> str:
        del messages
        value = next(self.responses)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class LocalOpenAIProvider:
    """Connect to an OpenAI-compatible server bound to localhost (for example vLLM)."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://127.0.0.1:8000/v1/chat/completions",
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Coding agent endpoint must be localhost")
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read())
        return str(value["choices"][0]["message"]["content"])


class TransformersProvider:
    """Load a local causal LM without downloading remote code or weights."""

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 1024) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'llm' extras to use TransformersProvider") from exc
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            device_map="auto",
            dtype="auto",
        )
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def complete(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
