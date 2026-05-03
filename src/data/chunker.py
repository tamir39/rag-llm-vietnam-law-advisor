"""Chunking for the Vietnamese tax-law KB.

Each KB row is already a single legal unit (Chương → Điều → Khoản → điểm) with
average length ~280 chars, max 1752. They are short enough to embed as-is, so
the default chunker is a passthrough.
"""
from __future__ import annotations


def chunk_passage(text: str, max_tokens: int = 256, overlap: int = 32) -> list[str]:
    return [text]
