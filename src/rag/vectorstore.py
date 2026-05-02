"""FAISS index build/load/search."""
from __future__ import annotations

from pathlib import Path


def build_index(passages, embedder, out_dir: Path):
    raise NotImplementedError


def load_index(index_dir: Path):
    raise NotImplementedError
