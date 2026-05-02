# Tasks

Tick each item when the corresponding work merges into `develop`.

## Phase 0 — Setup

- [x] Initial commit (main)
- [x] Create `develop` branch
- [x] Scaffold project structure (`feature/init-structure`)
- [ ] Add CONTRIBUTING / branch protection notes (optional)

## Phase 1 — Data

- [ ] Audit seed `knowledge_base_dvs_final.csv` (encoding, duplicates, length)
- [ ] Add any extra source documents (decrees, circulars) to `data/raw/`
- [ ] Define chunking strategy (notebook 02)
- [ ] Generate ≥ 300 train QA pairs → `data/qa/train_qa.jsonl`
- [ ] Hand-write ≥ 50 test QA pairs → `data/qa/test_qa.jsonl`

## Phase 2 — RAG

- [ ] Implement `src/data/loader.py`, `src/data/chunker.py`
- [ ] Implement `src/rag/embeddings.py`, `vectorstore.py`, `retriever.py`
- [ ] `scripts/build_index.py` produces a FAISS index
- [ ] Prompt templates finalized (`src/rag/prompts.py`)

## Phase 3 — Fine-tuning

- [ ] Implement `src/finetune/dataset.py`, `trainer.py`
- [ ] Run `notebooks/03_finetune_lora_colab.ipynb` end-to-end on Colab Free
- [ ] Push adapter to HF Hub, mirror to Drive
- [ ] Document training metrics in `docs/report/`

## Phase 4 — Inference & Evaluation

- [ ] Implement `src/inference/llm.py`, `pipeline.py`
- [ ] Implement `src/evaluation/metrics.py`, `retrieval_eval.py`
- [ ] `scripts/run_eval.py` runs A/B/C/D and writes `experiments/results/{A,B,C,D}.json`
- [ ] Human eval form + collect 50 ratings

## Phase 5 — Demo & Delivery

- [ ] Streamlit app working for D (LLM fine-tuned + RAG)
- [ ] Record 3–5 min demo video
- [ ] Final report (15–20 pages)
- [ ] Slides
- [ ] Tag release; merge `develop` → `main` (only when instructor signs off)
