# Data

```
data/
├── raw/                # original legal docs (PDF source) — gitignored content
├── knowledge_base/
│   └── knowledge_base.csv   # 600 passages from 5 Vietnamese tax laws (UTF-8 with BOM)
├── processed/          # post-chunking artifacts — gitignored content
└── qa/
    ├── train_qa.jsonl  # ≥ 300 pairs for SFT
    └── test_qa.jsonl   # ≥ 50 harder, no question-text overlap with train
```

## Knowledge base

**Domain:** Vietnamese tax law. **5 documents, 600 passages (avg 284 chars):**

| doc_id      | passages | title                                                         | articles |
|-------------|---------:|---------------------------------------------------------------|---------:|
| doc_000001  |      143 | VBHN 12/VBHN-VPQH 2026 — Luật Thuế giá trị gia tăng (VAT)     | 1–18     |
| doc_000002  |      140 | VBHN 103/VBHN-VPQH 2025 — Luật Thuế thu nhập cá nhân (PIT)    | 1–34     |
| doc_000003  |       58 | Luật 48/2010/QH12 — Thuế sử dụng đất phi nông nghiệp          | 1–12     |
| doc_000004  |      127 | Luật Thuế tiêu thụ đặc biệt 66/2025/QH15 (Excise)             | 1–11     |
| doc_000005  |      132 | Luật 67/2025/QH15 — Luật Thuế thu nhập doanh nghiệp (CIT)     | 1–14     |

### CSV schema

| column         | description                                 |
|----------------|---------------------------------------------|
| `passage_id`   | unique id, sequential (`passage_000001` … `passage_000600`) |
| `doc_id`       | source document id (`doc_000001` … `doc_000005`) |
| `title`        | source legal document title in Vietnamese   |
| `passage_text` | one Khoản / điểm of the law in Vietnamese   |
| `url`          | canonical link to the source on chinhphu.vn |

The file is **UTF-8 with BOM** (`EF BB BF`) — load with `encoding="utf-8-sig"` in Python (already wired in `src/data/loader.py`).

## QA JSONL schema

One JSON object per line:

```json
{"question": "Ai là người nộp thuế GTGT?", "answer": "...", "passage_id": "passage_000004"}
```

For reasoning questions that combine 1–2 related passages, `passage_id` references the **primary** passage that most directly contains the answer.

### QA-type distribution

| type | share | description |
|------|------:|-------------|
| fact | 70% | direct extraction from one passage |
| paraphrase | 15% | same answer, reworded question |
| reasoning | 10% | combines info from 1–2 adjacent passages of the same article |
| tricky | 5% | edge cases, negations, "trừ trường hợp", numerical thresholds |

Test set may share passages with train, but no question-text overlap (enforced by `scripts/prepare_qa.py`).
