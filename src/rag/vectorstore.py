"""FAISS index build / save / load for the tax-law KB.

Embeddings are L2-normalized upstream, so ``IndexFlatIP`` (inner product) gives
cosine similarity. We store the index next to a ``meta.jsonl`` file that maps
each row index back to its KB record (``passage_id``, ``doc_id``, ``title``,
``passage_text``, ``url``) so retrieval can return human-readable hits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INDEX_FILENAME = "kb.faiss"
META_FILENAME = "meta.jsonl"


def _import_faiss():
    import faiss  # imported lazily so ``import src.rag.vectorstore`` is cheap
    return faiss


def build_index(passages: list[dict], embedder, out_dir: Path) -> Path:
    """Embed every passage and persist a FAISS index + metadata sidecar.

    ``passages`` is a list of dict rows from ``load_knowledge_base``. Each must
    have a ``passage_text`` field; the others are kept verbatim in the meta file.
    """
    faiss = _import_faiss()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [p["passage_text"] for p in passages]
    vectors = embedder.encode_passages(texts)
    if vectors.dtype != np.float32:
        vectors = vectors.astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out_dir / INDEX_FILENAME))

    with open(out_dir / META_FILENAME, "w", encoding="utf-8") as f:
        for row in passages:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_dir


def load_index(index_dir: Path):
    """Return ``(faiss_index, list_of_metadata_dicts)``."""
    faiss = _import_faiss()
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / INDEX_FILENAME))
    meta: list[dict] = []
    with open(index_dir / META_FILENAME, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return index, meta
