"""Loaders for the Vietnamese tax-law knowledge base CSV and QA JSONL files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def load_knowledge_base(path: Path) -> list[dict]:
    """Read the KB CSV. The file is UTF-8 with BOM, so we use ``utf-8-sig``."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_qa(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
