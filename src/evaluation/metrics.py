"""Generation metrics: BLEU, ROUGE-L, BERTScore."""
from __future__ import annotations


def compute_bleu(predictions, references):
    raise NotImplementedError


def compute_rouge_l(predictions, references):
    raise NotImplementedError


def compute_bertscore(predictions, references, lang: str = "vi"):
    raise NotImplementedError
