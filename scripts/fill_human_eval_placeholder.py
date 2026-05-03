"""Fill `form.csv` with metric-derived placeholder ratings (1-5).

This is **not** a human rating — it bins per-prediction similarity (char-level
SequenceMatcher ratio + token recall vs gold) into 1-5 buckets so the report
has *some* numbers in bảng 6.2. Overwrite `form.csv` with real human ratings
when available, then re-run the join snippet in POST_TRAINING.md.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORM = ROOT / "experiments" / "results" / "human_eval" / "form.csv"
KEY = ROOT / "experiments" / "results" / "human_eval" / "key.csv"

csv.field_size_limit(10**7)


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)


def score(pred: str, gold: str) -> int:
    """Map (pred, gold) similarity to 1..5."""
    if not pred.strip():
        return 1
    char_ratio = SequenceMatcher(None, pred, gold).ratio()  # 0..1
    pt, gt = set(_tok(pred)), set(_tok(gold))
    tok_recall = len(pt & gt) / max(1, len(gt))  # 0..1
    blended = 0.5 * char_ratio + 0.5 * tok_recall  # 0..1
    if blended >= 0.55:
        return 5
    if blended >= 0.40:
        return 4
    if blended >= 0.28:
        return 3
    if blended >= 0.18:
        return 2
    return 1


def main() -> int:
    rows = list(csv.DictReader(open(FORM, encoding="utf-8")))
    print(f"Loaded {len(rows)} rows")

    fieldnames = list(rows[0].keys())
    for r in rows:
        gold = r["gold_answer"]
        for slot in (1, 2, 3, 4):
            r[f"rating_{slot}"] = score(r[f"answer_{slot}"], gold)
        r["notes"] = "metric-derived placeholder"

    with open(FORM, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote ratings into {FORM}")

    # Compute per-config stats
    key = list(csv.DictReader(open(KEY, encoding="utf-8")))
    key_map = {(int(k["row_id"]), int(k["slot"])): k["config"] for k in key}

    by_cfg: dict[str, list[int]] = {"A": [], "B": [], "C": [], "D": []}
    for r in rows:
        rid = int(r["row_id"])
        for slot in (1, 2, 3, 4):
            cfg = key_map[(rid, slot)]
            by_cfg[cfg].append(int(r[f"rating_{slot}"]))

    summary: dict[str, dict] = {}
    print("\n=== bang 6.2 (placeholder) ===")
    print(f"{'cfg':4} {'n':>3} {'mean':>6} {'median':>7} {'pct_ge_4':>9}")
    for cfg in "ABCD":
        vals = sorted(by_cfg[cfg])
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        ge4 = sum(1 for v in vals if v >= 4) / n
        summary[cfg] = {"n": n, "mean": round(mean, 3), "median": median, "pct_ge_4": round(ge4, 3)}
        print(f"{cfg:4} {n:>3} {mean:>6.2f} {median:>7.2f} {ge4:>9.2%}")

    out = FORM.parent / "summary_human_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
