"""Pick 3 illustrative test-set examples for report section 6.3.

Heuristics:
  - "D wins big": largest similarity gap D - A.
  - "A hallucinates": A has very low similarity AND mentions a passage_id not retrieved.
  - "RAG miss": Recall@5 = miss for B/D (gold_passage_id NOT in retrieved_passage_ids).
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"


def _tok(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower(), flags=re.UNICODE))


def sim(p: str, g: str) -> float:
    char = SequenceMatcher(None, p, g).ratio()
    pt, gt = _tok(p), _tok(g)
    rec = len(pt & gt) / max(1, len(gt))
    return 0.5 * char + 0.5 * rec


def load(cfg: str) -> list[dict]:
    return [json.loads(line) for line in (RES / cfg / "predictions.jsonl").open(encoding="utf-8")]


def main() -> None:
    A = load("A_base_no_rag")
    B = load("B_base_with_rag")
    C = load("C_finetuned_no_rag")
    D = load("D_finetuned_with_rag")

    rows = []
    for a, b, c, d in zip(A, B, C, D):
        gold = a["gold"]
        rows.append(
            {
                "idx": a["idx"],
                "question": a["question"],
                "gold": gold,
                "gold_pid": a["gold_passage_id"],
                "preds": {"A": a["prediction"], "B": b["prediction"], "C": c["prediction"], "D": d["prediction"]},
                "retrieved": {"B": b["retrieved_passage_ids"], "D": d["retrieved_passage_ids"]},
                "sims": {
                    "A": sim(a["prediction"], gold),
                    "B": sim(b["prediction"], gold),
                    "C": sim(c["prediction"], gold),
                    "D": sim(d["prediction"], gold),
                },
            }
        )

    # 1) D wins big — D high, A low
    win = max(rows, key=lambda r: r["sims"]["D"] - r["sims"]["A"])
    # 2) A hallucinates — lowest A similarity
    halluc = min(rows, key=lambda r: r["sims"]["A"])
    # 3) RAG miss — gold_pid NOT in D retrieved
    rag_misses = [r for r in rows if r["gold_pid"] not in r["retrieved"]["D"]]
    rag_miss = rag_misses[0] if rag_misses else None

    out = {
        "d_wins_big": win,
        "a_hallucinates": halluc,
        "rag_miss": rag_miss,
    }
    out_path = RES / "qualitative_examples.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    for label, ex in out.items():
        if not ex:
            continue
        print(f"\n--- {label} (idx={ex['idx']}) ---")
        print("Q:", ex["question"][:120])
        print("gold_pid:", ex["gold_pid"])
        print("sims:", {k: round(v, 3) for k, v in ex["sims"].items()})


if __name__ == "__main__":
    main()
