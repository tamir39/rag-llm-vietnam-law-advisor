"""Validate a QA JSONL file against the KB.

Usage:
    python scripts/prepare_qa.py data/qa/train_qa.jsonl
    python scripts/prepare_qa.py data/qa/test_qa.jsonl --train data/qa/train_qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import KB_CSV  # noqa: E402
from src.data.loader import load_knowledge_base, load_qa  # noqa: E402

REQUIRED_FIELDS = ("question", "answer", "passage_id")


def normalize(q: str) -> str:
    return " ".join(q.lower().split())


def validate(qa_path: Path, kb_ids: set[str], train_questions: set[str] | None) -> int:
    errors = 0
    rows = list(load_qa(qa_path))
    seen_questions: set[str] = set()
    doc_counter: Counter[str] = Counter()
    for i, row in enumerate(rows, 1):
        for field in REQUIRED_FIELDS:
            if field not in row or not row[field]:
                print(f"  [row {i}] missing/empty field: {field}")
                errors += 1
        pid = row.get("passage_id", "")
        if pid and pid not in kb_ids:
            print(f"  [row {i}] passage_id not in KB: {pid}")
            errors += 1
        q_norm = normalize(row.get("question", ""))
        if q_norm in seen_questions:
            print(f"  [row {i}] duplicate question: {row['question'][:60]}...")
            errors += 1
        seen_questions.add(q_norm)
        if train_questions is not None and q_norm in train_questions:
            print(f"  [row {i}] question overlaps with train: {row['question'][:60]}...")
            errors += 1
        if pid.startswith("passage_"):
            doc_prefix = pid[:8]
            doc_counter[doc_prefix] += 1
    print(f"\n{qa_path.name}: {len(rows)} rows, {errors} errors")
    print(f"  unique questions: {len(seen_questions)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qa_file", type=Path)
    parser.add_argument("--kb", type=Path, default=KB_CSV)
    parser.add_argument("--train", type=Path, default=None,
                        help="If validating a test file, pass train file to enforce no question overlap.")
    args = parser.parse_args()

    kb_rows = load_knowledge_base(args.kb)
    kb_ids = {r["passage_id"] for r in kb_rows}
    print(f"KB: {len(kb_ids)} passages from {args.kb.name}")

    train_questions: set[str] | None = None
    if args.train is not None:
        train_questions = {normalize(r["question"]) for r in load_qa(args.train)}
        print(f"Train file: {len(train_questions)} questions loaded for overlap check")

    errors = validate(args.qa_file, kb_ids, train_questions)
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
