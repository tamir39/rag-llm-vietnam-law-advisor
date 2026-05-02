"""Embedding model wrapper (multilingual-e5 by default)."""
from __future__ import annotations


def get_embedder(model_name: str):
    raise NotImplementedError
