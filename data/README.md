# Data

```
data/
├── raw/                # original legal docs (HTML/PDF) — gitignored content
├── knowledge_base/     # cleaned passages used for RAG indexing
│   └── knowledge_base_dvs_final.csv   # seed: 600 passages from VBHN 12/VBHN-VPQH 2026
├── processed/          # post-chunking artifacts — gitignored content
└── qa/
    ├── train_qa.jsonl  # ≥ 300 pairs for SFT (target)
    └── test_qa.jsonl   # ≥ 50 manually written test pairs (target)
```

## Knowledge base CSV schema

| column        | description                                 |
|---------------|---------------------------------------------|
| `passage_id`  | unique id (e.g. `passage_000001`)           |
| `doc_id`      | source document id (e.g. `doc_001`)         |
| `title`       | source legal document title                 |
| `passage_text`| full passage text in Vietnamese             |
| `url`         | canonical link to the source document       |

## QA JSONL schema

Each line is a JSON object:

```json
{"id": "qa_0001", "question": "...", "answer": "...", "passage_ids": ["passage_000123"]}
```

`passage_ids` is the gold context used for `Recall@k` evaluation.
