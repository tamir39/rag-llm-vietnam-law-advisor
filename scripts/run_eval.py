"""Run the 4-config evaluation matrix and dump per-config metrics + predictions.

Usage on Kaggle (after fine-tuning has produced the LoRA adapter):

    python scripts/run_eval.py                          # run all 4 configs
    python scripts/run_eval.py --configs A B            # subset
    python scripts/run_eval.py --skip-bertscore         # skip the slow metric

Outputs land under ``experiments/results/<config_name>/``:
    predictions.jsonl    — {question, gold, prediction, retrieved_ids, scores}
    metrics.json         — BLEU / ROUGE-L / BERTScore / Recall@5 (if RAG)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, ROOT_DIR, TEST_QA  # noqa: E402
from src.data.loader import load_qa  # noqa: E402
from src.evaluation.metrics import compute_all  # noqa: E402
from src.evaluation.retrieval_eval import mean_recall_at_k, mrr  # noqa: E402
from src.inference.pipeline import InferencePipeline  # noqa: E402

CONFIGS_DIR = ROOT_DIR / "experiments" / "configs"
CONFIG_FILES = {
    "A": "A_base_no_rag.yaml",
    "B": "B_base_with_rag.yaml",
    "C": "C_finetuned_no_rag.yaml",
    "D": "D_finetuned_with_rag.yaml",
}


def _free_vram() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_one(config_key: str, qa_rows: list[dict], skip_bertscore: bool) -> dict:
    cfg_path = CONFIGS_DIR / CONFIG_FILES[config_key]
    print(f"\n=== {config_key}: {cfg_path.name} ===")

    pipe = InferencePipeline(cfg_path)
    try:
        t0 = time.time()
        pipe.load()
        print(f"  loaded in {time.time() - t0:.1f}s")

        out_dir = RESULTS_DIR / pipe.name
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_file = out_dir / "predictions.jsonl"

        predictions: list[str] = []
        references: list[str] = []
        retrieved_per_q: list[list[str]] = []
        gold_per_q: list[str] = []

        with open(pred_file, "w", encoding="utf-8") as f:
            for i, row in enumerate(qa_rows, 1):
                t1 = time.time()
                res = pipe.answer(row["question"])
                predictions.append(res.answer)
                references.append(row["answer"])
                retrieved_per_q.append(res.retrieved_passage_ids)
                gold_per_q.append(row["passage_id"])
                f.write(json.dumps({
                    "idx": i,
                    "question": row["question"],
                    "gold": row["answer"],
                    "gold_passage_id": row["passage_id"],
                    "prediction": res.answer,
                    "retrieved_passage_ids": res.retrieved_passage_ids,
                    "retrieved_scores": res.retrieved_scores,
                    "latency_s": round(time.time() - t1, 3),
                }, ensure_ascii=False) + "\n")
                if i % 10 == 0:
                    print(f"  {i}/{len(qa_rows)}")

        print("  computing metrics ...")
        metrics = compute_all(predictions, references, skip_bertscore=skip_bertscore)
        if pipe.cfg["rag"]["enabled"]:
            metrics.update(mean_recall_at_k(retrieved_per_q, gold_per_q, k=5))
            metrics.update(mrr(retrieved_per_q, gold_per_q, k=10))

        metrics_file = out_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  metrics → {metrics_file}")
        print(f"  predictions → {pred_file}")
        return metrics
    finally:
        # Release the LLM + embedder + FAISS index before the next config loads
        # so we never hold two 4-bit Qwen2.5-7B copies in VRAM at the same time.
        pipe.unload()
        del pipe
        _free_vram()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, default=TEST_QA)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIG_FILES), default=list(CONFIG_FILES))
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore (slow, needs xlm-roberta-large download).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of test questions (smoke-test).")
    args = parser.parse_args()

    qa_rows = list(load_qa(args.qa))
    if args.limit:
        qa_rows = qa_rows[: args.limit]
    print(f"Eval QA: {len(qa_rows)} rows from {args.qa}")
    print(f"Configs: {args.configs}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for key in args.configs:
        summary[key] = run_one(key, qa_rows, args.skip_bertscore)

    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary → {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
