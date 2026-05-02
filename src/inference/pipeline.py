"""Inference pipeline supporting the 4 experimental configs A/B/C/D."""
from __future__ import annotations

from enum import Enum


class Config(str, Enum):
    A_BASE_NO_RAG = "A"
    B_BASE_RAG = "B"
    C_FT_NO_RAG = "C"
    D_FT_RAG = "D"


def answer(question: str, config: Config) -> str:
    raise NotImplementedError
