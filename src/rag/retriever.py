"""Top-k retriever interface."""
from __future__ import annotations


def retrieve(query: str, index, embedder, top_k: int = 5):
    raise NotImplementedError
