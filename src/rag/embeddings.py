"""Embedding model wrapper around ``intfloat/multilingual-e5-base``.

The E5 family was trained with explicit prefixes — ``query:`` for the user
question and ``passage:`` for indexed documents — so we always prepend them.
Vectors are L2-normalized so a FAISS ``IndexFlatIP`` performs cosine search.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class E5Embedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([QUERY_PREFIX + t for t in texts])


def get_embedder(model_name: str, device: str | None = None, batch_size: int = 32) -> E5Embedder:
    return E5Embedder(model_name=model_name, device=device, batch_size=batch_size)
