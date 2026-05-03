---
language:
  - vi
license: cc-by-4.0
task_categories:
  - question-answering
  - text-retrieval
pretty_name: Vietnamese Tax Law QA + Knowledge Base
size_categories:
  - n<1K
tags:
  - vietnamese
  - legal
  - tax
  - rag
  - qa
configs:
  - config_name: knowledge_base
    data_files:
      - split: train
        path: knowledge_base/knowledge_base.csv
  - config_name: qa
    data_files:
      - split: train
        path: qa/train_qa.jsonl
      - split: test
        path: qa/test_qa.jsonl
---

# Vietnamese Tax Law QA + Knowledge Base

A small but clean Vietnamese-language dataset for retrieval-augmented question answering over **5 Vietnamese tax laws**. Built as the eval/training corpus for the [LawMate](https://github.com/tamir39/rag-llm-vietnam-law-advisor) project (final-year NLP report).

## Contents

### `knowledge_base/knowledge_base.csv` — 600 passages

Pre-chunked legal passages from 5 consolidated Vietnamese tax laws (đã chia theo Chương → Điều → Khoản → điểm).

| Column         | Type   | Description                                                         |
|----------------|--------|---------------------------------------------------------------------|
| `passage_id`   | string | `passage_000001` … `passage_000600` (zero-padded, sequential)       |
| `doc_id`       | string | `doc_000001` … `doc_000005`                                         |
| `title`        | string | Document title in Vietnamese                                        |
| `passage_text` | string | Legal text (86–1752 chars, mean 284)                                |
| `url`          | string | Source URL on `vanban.chinhphu.vn` / `chinhphu.vn`                  |

#### Source documents

| `doc_id`     | Document                                                       | # passages |
|--------------|----------------------------------------------------------------|-----------:|
| `doc_000001` | VBHN 12/VBHN-VPQH 2026 — Luật Thuế Giá trị gia tăng             |        143 |
| `doc_000002` | VBHN 103/VBHN-VPQH 2025 — Luật Thuế Thu nhập cá nhân            |        140 |
| `doc_000003` | Luật 48/2010/QH12 — Thuế sử dụng đất phi nông nghiệp            |         58 |
| `doc_000004` | Luật 66/2025/QH15 — Thuế tiêu thụ đặc biệt                      |        127 |
| `doc_000005` | Luật 67/2025/QH15 — Thuế thu nhập doanh nghiệp                  |        132 |
| **Total**    |                                                                | **600**    |

### `qa/train_qa.jsonl` — 305 train pairs

```json
{"question": "...", "answer": "...", "passage_id": "passage_000XYZ"}
```

Distribution (approximate):
- ~70% factoid (single-passage extractive)
- ~15% paraphrase (rewording the gold)
- ~10% reasoning (multi-step over a single passage)
- ~5% tricky / negative (out-of-scope or numerically tight)

### `qa/test_qa.jsonl` — 54 hand-written test pairs

Held out, harder distribution. Designed to **not overlap** with `train_qa` in question wording (validated by `scripts/prepare_qa.py` in the LawMate repo).

## Loading

```python
from datasets import load_dataset

# Knowledge base
kb = load_dataset("Tamir39/vietnam-tax-qa", "knowledge_base", split="train")

# QA splits
train = load_dataset("Tamir39/vietnam-tax-qa", "qa", split="train")
test  = load_dataset("Tamir39/vietnam-tax-qa", "qa", split="test")
```

## Use cases

- Vietnamese RAG benchmarking (small enough to iterate fast).
- Legal-domain retriever evaluation: 600 passages give meaningful Recall@k signal.
- Vietnamese SFT for legal QA style (used to train the QLoRA adapter [`Tamir39/qwen2_5-7b-vietnam-tax-lora`](https://huggingface.co/Tamir39/qwen2_5-7b-vietnam-tax-lora)).

## Construction

- **KB**: passages extracted from official Vietnamese government legal portals (`vanban.chinhphu.vn`, `chinhphu.vn`). Each passage corresponds to one *điểm* / *khoản* / short *Điều*. Length capped so the entire passage fits as one embedding without further chunking.
- **QA**: drafted by Claude with strict instructions to ground every answer in a specific `passage_id` from the KB (no outside knowledge). A validator checks schema, that every `passage_id` exists, no train/test question-text overlap.

## License & attribution

- Released under **CC-BY-4.0**.
- The underlying legal text is Vietnamese law (public-domain by nature in VN); attribution + URLs to the source portal are preserved in the `url` column.
- If you use this dataset, please cite the LawMate project: https://github.com/tamir39/rag-llm-vietnam-law-advisor

## Limitations

- Test set is small (54 questions) → BLEU/ROUGE variance is high.
- Single annotator for QA generation → no inter-annotator agreement signal.
- Five laws only — does not cover the full Vietnamese tax code (e.g., import/export duty, environmental tax).
- Some VAT-related articles (Điều 13–16) were not parsed from the source PDF and are absent from the KB.
