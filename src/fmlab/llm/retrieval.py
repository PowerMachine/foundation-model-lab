"""Dependency-free BM25 and dense retrieval evaluation for toy RAG projects."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[가-힣]+", re.UNICODE)


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    text: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float
    rank: int


class BM25Retriever:
    """Classic Okapi BM25 suitable for small, inspectable corpora."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("at least one document is required")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("document ids must be unique")
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(lexical_tokens(doc.text)) for doc in documents]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(documents)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = lexical_tokens(query)
        scored: list[tuple[float, Document]] = []
        for document, frequencies, length in zip(
            self.documents, self.term_frequencies, self.lengths, strict=True
        ):
            score = 0.0
            normalization = self.k1 * (
                1.0 - self.b + self.b * length / max(self.average_length, 1e-9)
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1.0) / (frequency + normalization)
                    )
            scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            SearchHit(document=document, score=score, rank=rank)
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


class DenseRetriever:
    """Cosine retriever accepting any local embedding callback."""

    def __init__(
        self,
        documents: Sequence[Document],
        encoder: Callable[[Sequence[str]], np.ndarray],
    ) -> None:
        if not documents:
            raise ValueError("at least one document is required")
        self.documents = list(documents)
        self.encoder = encoder
        matrix = np.asarray(encoder([document.text for document in documents]), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(documents):
            raise ValueError("encoder must return [items, embedding_dimension]")
        self.embeddings = _normalize(matrix)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        query_vector = np.asarray(self.encoder([query]), dtype=np.float32)
        if query_vector.shape != (1, self.embeddings.shape[1]):
            raise ValueError("query embedding dimension differs from document embeddings")
        scores = self.embeddings @ _normalize(query_vector)[0]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchHit(
                document=self.documents[int(index)],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def evaluate_retrieval(
    retriever: BM25Retriever | DenseRetriever,
    queries: Iterable[RetrievalQuery],
    *,
    top_k: int = 5,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    hit_rates: list[float] = []
    for query in queries:
        if not query.relevant_ids:
            raise ValueError(f"query {query.id!r} has no relevance labels")
        hits = retriever.search(query.text, top_k=top_k)
        retrieved = [hit.document.id for hit in hits]
        relevant_ranks = [
            rank
            for rank, document_id in enumerate(retrieved, start=1)
            if document_id in query.relevant_ids
        ]
        found = len(set(retrieved) & set(query.relevant_ids))
        reciprocal_rank = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
        recall = found / len(query.relevant_ids)
        reciprocal_ranks.append(reciprocal_rank)
        recalls.append(recall)
        hit_rates.append(float(bool(relevant_ranks)))
        rows.append(
            {
                "query_id": query.id,
                "query": query.text,
                "relevant_ids": sorted(query.relevant_ids),
                "retrieved_ids": retrieved,
                "first_relevant_rank": min(relevant_ranks) if relevant_ranks else None,
            }
        )
    if not rows:
        raise ValueError("at least one query is required")
    return {
        f"recall@{top_k}": float(np.mean(recalls)),
        f"hit_rate@{top_k}": float(np.mean(hit_rates)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "queries": rows,
    }


def build_grounded_prompt(query: str, hits: Sequence[SearchHit]) -> str:
    context = "\n\n".join(f"[{hit.document.id}] {hit.document.text}" for hit in hits)
    return (
        "아래 근거만 사용해 질문에 답하고, 사용한 문서 ID를 대괄호로 인용하세요. "
        "근거가 없으면 '근거 없음'이라고 답하세요.\n\n"
        f"근거:\n{context}\n\n질문: {query}\n답변:"
    )
