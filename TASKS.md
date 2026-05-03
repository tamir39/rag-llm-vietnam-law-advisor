# Tasks

Tick each item when the corresponding work merges into `develop`.

## Phase 0 — Setup

- [x] Initial commit (main)
- [x] Create `develop` branch
- [x] Scaffold project structure (`feature/init-structure`)
- [x] Windows one-shot launcher (`run.bat`)
- [ ] Add CONTRIBUTING / branch protection notes (optional)

## Phase 1 — Data

- [x] Audit seed `knowledge_base.csv` (encoding, duplicates, length)
- [x] Add source documents (5 tax laws, 600 passages) to `data/knowledge_base/`
- [x] Define chunking strategy (`src/data/chunker.py`)
- [x] Generate ≥ 300 train QA pairs → `data/qa/train_qa.jsonl` (305 pairs)
- [x] Hand-write ≥ 50 test QA pairs → `data/qa/test_qa.jsonl` (54 pairs)

## Phase 2 — RAG

- [x] Implement `src/data/loader.py`, `src/data/chunker.py`
- [x] Implement `src/rag/embeddings.py`, `vectorstore.py`, `retriever.py`
- [x] `scripts/build_index.py` produces a FAISS index
- [x] Prompt templates finalized (`src/rag/prompts.py`)

## Phase 3 — Fine-tuning

- [x] Implement `src/finetune/dataset.py`, `lora_config.py`, `trainer.py`
- [x] Kaggle notebook ready (`notebooks/03_finetune_lora_kaggle.ipynb`)
- [x] Push adapter to HF Hub (`Tamir39/qwen2_5-7b-vietnam-tax-lora`)
- [x] Document training metrics in `docs/report/POST_TRAINING.md` (config snapshot + loss curve in §4.4 of report.md)

## Phase 4 — Inference & Evaluation

- [x] Implement `src/inference/llm.py`, `pipeline.py`
- [x] Implement `src/evaluation/metrics.py`, `retrieval_eval.py`
- [x] `scripts/run_eval.py` wired for A/B/C/D matrix
- [x] `scripts/build_human_eval.py` builds blinded form
- [x] Eval Kaggle notebook (`notebooks/07_eval_kaggle.ipynb`)
- [x] **Run `scripts/run_eval.py` end-to-end → produce `experiments/results/{A,B,C,D}/*.json`**
- [x] Human eval form filled (50 ratings)

## Phase 5 — Demo & Delivery

- [x] Streamlit app working for all 4 configs (`app/streamlit_app.py`)
- [x] Kaggle demo notebook with Cloudflare tunnel (`notebooks/06_demo_kaggle.ipynb`)
- [x] Single-slot model loading + OOM auto-recovery in the demo
- [x] Final report — abstract, §6.1, §6.2, §6.3, §4.4, §8 filled; figures embedded
- [x] Publish dataset on HF (`Tamir39/vietnam-tax-qa`, CC-BY-4.0)
- [ ] Tag release; merge `develop` → `main` (only when instructor signs off)
