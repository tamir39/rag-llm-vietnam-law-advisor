"""Generation metrics: BLEU (sacrebleu), ROUGE-L, BERTScore.

All metric libraries are imported lazily — they pull large dependencies (PyTorch
for BERTScore, transformers tokenizers, etc.) and we don't want them loaded
when the caller only needs, say, BLEU.
"""
from __future__ import annotations

from typing import Sequence


def compute_bleu(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """Corpus BLEU via sacrebleu (Vietnamese is whitespace-tokenized fine)."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(list(predictions), [list(references)])
    return {
        "bleu": bleu.score,
        "bleu_signature": str(bleu.signature) if hasattr(bleu, "signature") else "",
    }


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """Mean ROUGE-L F1 over the corpus (rouge-score package, stemming off)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    fs: list[float] = []
    for pred, ref in zip(predictions, references):
        fs.append(scorer.score(ref, pred)["rougeL"].fmeasure)
    return {"rougeL": sum(fs) / len(fs) if fs else 0.0, "n": len(fs)}


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "vi",
    model_type: str | None = None,
) -> dict:
    """Mean BERTScore F1. Default uses xlm-roberta via lang='vi'."""
    from bert_score import score as bert_score_fn

    p, r, f = bert_score_fn(
        cands=list(predictions),
        refs=list(references),
        lang=lang,
        model_type=model_type,
        rescale_with_baseline=False,
        verbose=False,
    )
    return {
        "bertscore_p": float(p.mean()),
        "bertscore_r": float(r.mean()),
        "bertscore_f1": float(f.mean()),
        "n": len(predictions),
    }


def compute_all(
    predictions: Sequence[str],
    references: Sequence[str],
    bertscore_lang: str = "vi",
    skip_bertscore: bool = False,
) -> dict:
    out = {}
    out.update(compute_bleu(predictions, references))
    out.update(compute_rouge_l(predictions, references))
    if not skip_bertscore:
        out.update(compute_bertscore(predictions, references, lang=bertscore_lang))
    return out
