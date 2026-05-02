"""Retrieval metrics (Recall@k)."""
from __future__ import annotations


def recall_at_k(retrieved_ids, gold_ids, k: int = 5) -> float:
    raise NotImplementedError
