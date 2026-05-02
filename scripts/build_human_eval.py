"""Build a blinded human-eval form from the 4-config predictions.

For each of N (default 50) sampled test questions, write one row containing
the question, the gold answer, and the four model predictions in a randomized
order (so the rater doesn't know which is A/B/C/D). A separate ``key.csv``
records the true ordering for later joining.

Usage:
    python scripts/build_human_eval.py                 # 50 questions, all 4 configs
    python scripts/build_human_eval.py --n 30 --seed 7

Inputs:
    experiments/results/<config_name>/predictions.jsonl  (created by run_eval.py)

Outputs:
    experiments/results/human_eval/form.csv             # rate columns A_blind..D_blind 1-5
    experiments/results/human_eval/key.csv              # row_id -> blind_letter -> real_config
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR  # noqa: E402

CONFIG_NAMES = {
    "A": "A_base_no_rag",
    "B": "B_base_with_rag",
    "C": "C_finetuned_no_rag",
    "D": "D_finetuned_with_rag",
}


def load_preds(config_key: str) -> list[dict]:
    path = RESULTS_DIR / CONFIG_NAMES[config_key] / "predictions.jsonl"
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "human_eval")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_cfg = {k: load_preds(k) for k in CONFIG_NAMES}
    total = len(next(iter(by_cfg.values())))
    if not all(len(v) == total for v in by_cfg.values()):
        print("ERROR: prediction files have different row counts", file=sys.stderr)
        return 1

    indices = sorted(rng.sample(range(total), min(args.n, total)))
    args.out.mkdir(parents=True, exist_ok=True)

    blind_letters = list(string.ascii_uppercase[:4])  # A B C D blind labels
    form_path = args.out / "form.csv"
    key_path = args.out / "key.csv"

    with open(form_path, "w", encoding="utf-8", newline="") as ff, \
         open(key_path, "w", encoding="utf-8", newline="") as kf:
        form = csv.writer(ff)
        key = csv.writer(kf)
        form.writerow([
            "row_id", "question", "gold_answer",
            "answer_1", "answer_2", "answer_3", "answer_4",
            "rating_1", "rating_2", "rating_3", "rating_4",
            "notes",
        ])
        key.writerow(["row_id", "slot", "config"])

        for row_id, idx in enumerate(indices, 1):
            order = list(CONFIG_NAMES.keys())
            rng.shuffle(order)
            row_preds = [by_cfg[c][idx] for c in order]
            base = by_cfg["A"][idx]
            form.writerow([
                row_id,
                base["question"],
                base["gold"],
                row_preds[0]["prediction"],
                row_preds[1]["prediction"],
                row_preds[2]["prediction"],
                row_preds[3]["prediction"],
                "", "", "", "",  # rating columns blank for the human
                "",
            ])
            for slot, cfg_letter in enumerate(order, 1):
                key.writerow([row_id, slot, cfg_letter])

    print(f"Form  → {form_path}  ({len(indices)} rows)")
    print(f"Key   → {key_path}")
    print("Rater fills `rating_1..rating_4` with 1-5; merge by `key.csv` later.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
