from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fmlab.artifacts import ExperimentResult

from .data import ByteTokenizer


DEFAULT_CORPUS = [
    "어텐션은 문맥 속 토큰 사이의 관계를 학습합니다.",
    "파인튜닝은 사전학습 모델을 특정 작업에 적응시킵니다.",
    "LoRA는 원본 가중치를 고정하고 저랭크 행렬만 학습합니다.",
    "양자화는 16비트 가중치를 8비트 또는 4비트로 줄입니다.",
    "검색 증강 생성은 근거 문서를 먼저 검색합니다.",
    "def attention(query, key, value): return softmax(query @ key.T) @ value",
    "invoice_total = subtotal + tax  # 270.00 + 27.00",
    "한국어 조사와 어미는 공백 기반 토크나이저에서 까다롭습니다.",
]


def train_and_compare_tokenizers(
    output_dir: str | Path,
    *,
    corpus: Sequence[str] = DEFAULT_CORPUS,
    vocab_size: int = 320,
) -> ExperimentResult:
    """Train a tiny byte-level BPE and compare it with the transparent byte baseline."""
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Hugging Face tokenizers to run the BPE lab") from exc

    started = time.time()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not corpus:
        raise ValueError("corpus must not be empty")

    bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    bpe.train_from_iterator(list(corpus) * 4, trainer=trainer)
    bpe.save(str(destination / "tokenizer.json"))

    byte = ByteTokenizer()
    rows = []
    byte_lengths = []
    bpe_lengths = []
    for text in corpus:
        byte_ids = byte.encode(text)
        encoded = bpe.encode(text)
        byte_lengths.append(len(byte_ids))
        bpe_lengths.append(len(encoded.ids))
        rows.append(
            {
                "text": text,
                "byte_count": len(byte_ids),
                "bpe_count": len(encoded.ids),
                "bpe_tokens": encoded.tokens,
                "round_trip": bpe.decode(encoded.ids),
            }
        )

    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, byte_lengths, width, label="UTF-8 bytes")
    ax.bar(x + width / 2, bpe_lengths, width, label="trained byte-BPE")
    ax.set_xlabel("corpus sentence")
    ax.set_ylabel("token count (lower is shorter)")
    ax.set_title("Tokenizer sequence-length comparison")
    ax.set_xticks(x, [str(index) for index in range(len(rows))])
    ax.legend()
    fig.tight_layout()
    chart = destination / "token_lengths.png"
    fig.savefig(chart, dpi=150)
    plt.close(fig)

    table_rows = "".join(
        "<tr>"
        f"<td>{index}</td><td>{html.escape(row['text'])}</td>"
        f"<td>{row['byte_count']}</td><td>{row['bpe_count']}</td>"
        f"<td><code>{html.escape(' | '.join(row['bpe_tokens']))}</code></td>"
        "</tr>"
        for index, row in enumerate(rows)
    )
    report = destination / "report.html"
    report.write_text(
        "<!doctype html><html lang='ko'><meta charset='utf-8'><title>Tokenizer lab</title>"
        "<style>body{font:15px system-ui;max-width:1300px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.5rem;vertical-align:top}"
        "code{overflow-wrap:anywhere}</style><body><h1>Byte vs trained BPE</h1>"
        "<p>같은 문장이 vocabulary 학습 뒤 어떤 subword로 합쳐지는지 비교한다.</p>"
        "<img style='max-width:100%' src='token_lengths.png'><table><tr>"
        "<th>#</th><th>text</th><th>bytes</th><th>BPE</th><th>BPE tokens</th></tr>"
        f"{table_rows}</table></body></html>",
        encoding="utf-8",
    )
    (destination / "examples.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": bpe.get_vocab_size(),
        "average_byte_tokens": float(np.mean(byte_lengths)),
        "average_bpe_tokens": float(np.mean(bpe_lengths)),
        "sequence_reduction_percent": float(
            100 * (1 - np.mean(bpe_lengths) / np.mean(byte_lengths))
        ),
        "all_round_trips_equal": all(row["round_trip"] == row["text"] for row in rows),
    }
    result = ExperimentResult(
        experiment="llm_tokenizer_bpe",
        status="completed",
        started_at=started,
        finished_at=time.time(),
        metrics=metrics,
        parameters={"corpus_size": len(corpus), "vocab_size": vocab_size},
        artifacts=["token_lengths.png", "report.html", "examples.json", "tokenizer.json"],
        notes=[
            "This corpus is intentionally tiny; token efficiency here is educational, not general.",
            "Byte-level BPE remains reversible while merging frequently co-occurring byte sequences.",
        ],
    )
    result.write(destination)
    return result
