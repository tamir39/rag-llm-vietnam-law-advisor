"""Retrieval metrics for the RAG component.

The QA schema stores a single ``passage_id`` per question — the *primary*
passage that contains the answer — so per-question Recall@k is binary
(1 if the gold id appears in the top-k retrieved ids, else 0). The corpus
metric is the mean.
"""
from __future__ import annotations

from typing import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Per-question recall: 1 if gold id is in the first k retrieved, else 0."""
    return float(gold_ids[0] in list(retrieved_ids)[:k]) if gold_ids else 0.0


def mean_recall_at_k(
    retrieved_per_q: Sequence[Sequence[str]],
    gold_per_q: Sequence[str],
    k: int = 5,
) -> dict:
    """Mean Recall@k over many questions. ``gold_per_q[i]`` is the single gold id."""
    if not retrieved_per_q:
        return {f"recall@{k}": 0.0, "n": 0}
    hits = [1.0 if gold_per_q[i] in list(retrieved_per_q[i])[:k] else 0.0
            for i in range(len(retrieved_per_q))]
    return {f"recall@{k}": sum(hits) / len(hits), "n": len(hits)}


def mrr(
    retrieved_per_q: Sequence[Sequence[str]],
    gold_per_q: Sequence[str],
    k: int = 10,
) -> dict:
    """Mean Reciprocal Rank — useful complement to Recall@k."""
    rr: list[float] = []
    for retrieved, gold in zip(retrieved_per_q, gold_per_q):
        rank = next((i + 1 for i, pid in enumerate(list(retrieved)[:k]) if pid == gold), 0)
        rr.append(1.0 / rank if rank else 0.0)
    return {f"mrr@{k}": sum(rr) / len(rr) if rr else 0.0, "n": len(rr)}
