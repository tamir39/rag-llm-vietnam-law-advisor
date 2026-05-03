# LawMate — Vietnamese Tax-Law RAG + LoRA

A Vietnamese question-answering system over **Vietnam's tax law**, combining **Retrieval-Augmented Generation** (FAISS + `multilingual-e5-base`) with a **Qwen2.5-7B-Instruct** model **fine-tuned via QLoRA**.

End-of-semester project for *Introduction to Natural Language Processing*.

> **Live demo (Kaggle + Cloudflare tunnel):** `notebooks/06_demo_kaggle.ipynb`. The Streamlit app at `app/streamlit_app.py` runs identically locally — see [Run the demo](#run-the-demo) below.

---

## Table of contents

1. [Scope](#scope)
2. [Goal: the 4-config matrix](#goal-the-4-config-matrix)
3. [Quick start](#quick-start)
4. [Run the demo](#run-the-demo)
5. [Run the evaluation](#run-the-evaluation)
6. [Full pipeline](#full-pipeline)
7. [Folder structure](#folder-structure)
8. [What each file does](#what-each-file-does)
9. [Models & defaults](#models--defaults)
10. [Branching policy](#branching-policy)
11. [Deliverables](#deliverables)

---

## Scope

Five Vietnamese tax laws — **600 legal passages**, ~284 chars per passage:

| `doc_id`     | Document                                                              | # passages |
|--------------|-----------------------------------------------------------------------|-----------:|
| `doc_000001` | VBHN 12/VBHN-VPQH 2026 — VAT Law (GTGT)                               |        143 |
| `doc_000002` | VBHN 103/VBHN-VPQH 2025 — Personal Income Tax (TNCN)                  |        140 |
| `doc_000003` | Law 48/2010/QH12 — Non-agricultural Land-Use Tax                      |         58 |
| `doc_000004` | Law 66/2025/QH15 — Special Consumption Tax (TTĐB)                     |        127 |
| `doc_000005` | Law 67/2025/QH15 — Corporate Income Tax (TNDN)                        |        132 |

Q&A: **305** training pairs + **54** test pairs in `data/qa/`.

## Goal: the 4-config matrix

Compare four configurations on the same test set:

|                  | No RAG | With RAG |
|------------------|--------|----------|
| **Base LLM**      | A      | B        |
| **Fine-tuned LLM** | C      | D        |

Metrics: **BLEU**, **ROUGE-L**, **BERTScore (vi)**, **Recall@5**, plus a **blinded human eval** (1–5) on 50 sampled questions.

---

## Quick start

```powershell
# Windows (PowerShell)
git clone https://github.com/tamir39/rag-llm-vietnam-law-advisor.git
cd rag-llm-vietnam-law-advisor
git checkout develop
copy .env.example .env                   # then edit .env to set HF_TOKEN

# Option 1 — one-shot Windows launcher (uses uv to bootstrap .venv + deps + index, then opens Streamlit)
.\run.bat

# Option 2 — manual
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_index.py            # build the FAISS index from data/knowledge_base/
streamlit run app/streamlit_app.py
```

```bash
# macOS / Linux
git clone https://github.com/tamir39/rag-llm-vietnam-law-advisor.git
cd rag-llm-vietnam-law-advisor
git checkout develop
cp .env.example .env                     # then edit .env to set HF_TOKEN

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
streamlit run app/streamlit_app.py
```

**Hardware note:** loading Qwen2.5-7B in 4-bit needs ~6 GB of VRAM. On a CPU-only machine the demo will fall back to CPU inference, which is unusably slow for a 7B model — use Kaggle (free P100 / T4) for the actual demo. See `notebooks/06_demo_kaggle.ipynb`.

---

## Run the demo

### On Kaggle (recommended for sharing)

Open **`notebooks/06_demo_kaggle.ipynb`** on Kaggle, attach the `HF_TOKEN` secret, set Accelerator to **GPU P100** or **T4 ×2**, **Internet On**, then **Run all**. After the last cell, you'll get a public `*.trycloudflare.com` URL that anyone can hit. The session lasts up to 12 h.

### Locally

`streamlit run app/streamlit_app.py` (or `.\run.bat` on Windows). The app:

- Loads **one** of the 4 configs at a time into VRAM, with explicit Load / Unload buttons (so flipping between configs never holds two 7B models at once).
- Has two tabs:
  - **Hỏi 1 cấu hình** — single config, single question, with retrieved passages shown.
  - **So sánh nhiều cấu hình** — pick 2–4 configs, ask one question, watch the app load / answer / unload each one sequentially, then see results side-by-side.
- Catches CUDA OOM and recovers automatically with a friendly banner instead of a red traceback.

---

## Run the evaluation

The 4-config matrix is evaluated by `scripts/run_eval.py`, which writes per-config `predictions.jsonl` + `metrics.json` and a top-level `summary.json` under `experiments/results/`. **Do not** run this in the demo notebook — eval needs to load the model directly into the kernel, while the demo deliberately runs it in a Streamlit subprocess.

Use **`notebooks/07_eval_kaggle.ipynb`** instead — same Kaggle setup as the demo (GPU + `HF_TOKEN` + Internet on), Run All, ~1–1.5 h wall-clock for the full matrix. The notebook also runs `scripts/build_human_eval.py` to produce a blinded `form.csv` for the 50-question human eval.

After the run, follow `docs/report/POST_TRAINING.md` to fold the numbers into the report.

---

## Full pipeline

```bash
# 1. Validate the QA data
python scripts/prepare_qa.py data/qa/train_qa.jsonl
python scripts/prepare_qa.py data/qa/test_qa.jsonl --train data/qa/train_qa.jsonl

# 2. Build the FAISS index from the KB CSV
python scripts/build_index.py

# 3. Fine-tune on Kaggle (notebooks/03_finetune_lora_kaggle.ipynb)
#    Pushes the LoRA adapter to Tamir39/qwen2_5-7b-vietnam-tax-lora.

# 4. Run the 4-config evaluation matrix (notebooks/07_eval_kaggle.ipynb)
python scripts/run_eval.py                 # all 4 configs
python scripts/run_eval.py --configs B D   # subset
python scripts/run_eval.py --limit 5       # smoke test

# 5. Build the blinded human-eval form
python scripts/build_human_eval.py --n 50

# 6. Demo (notebooks/06_demo_kaggle.ipynb)
streamlit run app/streamlit_app.py
```

---

## Folder structure

```
.
├── app/
│   └── streamlit_app.py             # Streamlit demo UI
├── checkpoints/                     # LoRA adapter (gitignored; pushed to HF Hub)
├── data/
│   ├── knowledge_base/              # knowledge_base.csv (600 passages)
│   ├── qa/                          # train_qa.jsonl (305) + test_qa.jsonl (54)
│   ├── processed/                   # intermediate artifacts (gitignored)
│   └── raw/                         # raw legal texts
├── docs/report/                     # report.md, POST_TRAINING.md, figures/
├── experiments/
│   ├── configs/                     # A/B/C/D YAML configs (one per cell of the matrix)
│   ├── index/                       # FAISS index built from knowledge_base.csv
│   └── results/                     # predictions.jsonl + metrics.json (after run_eval)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 03_finetune_lora_kaggle.ipynb
│   ├── 06_demo_kaggle.ipynb         # Streamlit demo + Cloudflare tunnel
│   └── 07_eval_kaggle.ipynb         # 4-config eval + human-eval form
├── scripts/
│   ├── prepare_qa.py
│   ├── build_index.py
│   ├── run_eval.py
│   └── build_human_eval.py
├── src/
│   ├── config.py                    # all paths, model ids, retrieval defaults
│   ├── data/
│   │   ├── loader.py                # KB CSV (utf-8-sig) + QA JSONL readers
│   │   └── chunker.py               # passage chunking
│   ├── rag/
│   │   ├── embeddings.py            # E5 wrapper (query:/passage: prefixes, L2-normalize)
│   │   ├── vectorstore.py           # FAISS IndexFlatIP build/load
│   │   ├── retriever.py             # top-k retrieve + format_context()
│   │   └── prompts.py               # SYSTEM_VI, build_rag_prompt, build_no_rag_prompt
│   ├── finetune/
│   │   ├── dataset.py               # JSONL → chat-template SFT dataset
│   │   ├── lora_config.py           # PEFT LoraConfig defaults (r=16)
│   │   └── trainer.py               # SFTTrainer wiring (TRL ≥ 0.12 API)
│   ├── inference/
│   │   ├── llm.py                   # 4-bit Qwen loader + generate()
│   │   └── pipeline.py              # InferencePipeline (load/answer/unload)
│   └── evaluation/
│       ├── metrics.py               # BLEU / ROUGE-L / BERTScore
│       └── retrieval_eval.py        # Recall@k, MRR
├── PLANNING.md                      # architecture & decisions
├── PRD.md                           # product requirements
├── TASKS.md                         # phase-by-phase task list
├── run.bat                          # Windows one-shot launcher (uv + Streamlit)
└── requirements.txt
```

## What each file does

### `app/`
- **`streamlit_app.py`** — the demo UI. Single-slot model loader (one config in VRAM at a time), card-based config picker with live VRAM gauge, single-question and side-by-side compare modes, friendly OOM recovery banner.

### `src/config.py`
Central paths and constants. **Edit here** to point at a different base model / adapter / index location.

### `src/data/`
- **`loader.py`** — `load_kb()` reads the KB CSV (`utf-8-sig`, handles BOM); `load_qa()` reads JSONL.
- **`chunker.py`** — passage chunking helpers (used by `scripts/build_index.py`).

### `src/rag/`
- **`embeddings.py`** — wraps `intfloat/multilingual-e5-base`. Adds the required `query: ` / `passage: ` prefixes and L2-normalizes vectors.
- **`vectorstore.py`** — builds a FAISS `IndexFlatIP` (cosine because vectors are normalized); `save_index()` / `load_index()`.
- **`retriever.py`** — `retrieve(question, index, meta, embedder, top_k)` and `format_context(hits)` for the prompt.
- **`prompts.py`** — `SYSTEM_VI` (Vietnamese system prompt), `build_rag_prompt()`, `build_no_rag_prompt()`.

### `src/finetune/`
- **`dataset.py`** — converts QA JSONL into a chat-template SFT dataset for TRL's `SFTTrainer`.
- **`lora_config.py`** — PEFT `LoraConfig` defaults (`r=16`, the LoRA modules to target on Qwen2.5).
- **`trainer.py`** — wires `SFTTrainer` per the TRL ≥ 0.12 / Transformers ≥ 4.46 API.

### `src/inference/`
- **`llm.py`** — `load_llm()` builds a 4-bit nf4 Qwen2.5-7B with `BitsAndBytesConfig`, attaches the LoRA adapter when given a `lora_path`, and picks `bf16` vs `fp16` based on GPU capability (T4 = fp16). `generate()` runs a single chat-template generation.
- **`pipeline.py`** — `InferencePipeline.load() / answer() / unload()`. The single class the Streamlit app and `run_eval.py` both go through. `unload()` releases VRAM cleanly between configs.

### `src/evaluation/`
- **`metrics.py`** — `compute_bleu` (sacrebleu), `compute_rouge_l` (rouge-score), `compute_bertscore` (bert-score `lang="vi"` → xlm-roberta-large). All imports lazy.
- **`retrieval_eval.py`** — `mean_recall_at_k`, `mrr` (only meaningful for configs with RAG, i.e. B and D).

### `scripts/`
- **`prepare_qa.py`** — validates a JSONL QA file (schema + non-empty fields). With `--train`, also checks the test set has no overlap with training.
- **`build_index.py`** — KB CSV → chunks → embeddings → FAISS index in `experiments/index/`.
- **`run_eval.py`** — drives the 4-config matrix end-to-end, with `pipe.unload()` between configs so VRAM stays bounded to one Qwen at a time. Flags: `--configs A B …`, `--limit N` (smoke test), `--skip-bertscore`.
- **`build_human_eval.py`** — samples N questions, randomizes the 4 predictions per row (blinded), writes `form.csv` (rate it 1–5) and `key.csv` (the un-blinding map).

### `notebooks/`
- **`01_data_exploration.ipynb`** — first-pass look at the KB CSV (encoding, length, duplicates).
- **`03_finetune_lora_kaggle.ipynb`** — QLoRA fine-tune of Qwen2.5-7B on Kaggle. Pushes the adapter to `Tamir39/qwen2_5-7b-vietnam-tax-lora`.
- **`06_demo_kaggle.ipynb`** — runs the Streamlit app on Kaggle and exposes it via a free Cloudflare tunnel.
- **`07_eval_kaggle.ipynb`** — runs `scripts/run_eval.py` for all 4 configs and `scripts/build_human_eval.py` for the blinded rating form.

### `experiments/configs/`
One YAML per cell of the matrix. Same shape:

```yaml
name: D_finetuned_with_rag
description: Fine-tuned LLM (LoRA) with RAG
model:
  base_id: Qwen/Qwen2.5-7B-Instruct
  lora_adapter: checkpoints/qwen2_5-7b-vietnam-tax-lora    # local path (optional)
  hf_lora_repo: Tamir39/qwen2_5-7b-vietnam-tax-lora        # fallback if local missing
rag:
  enabled: true
  embedding_model: intfloat/multilingual-e5-base
  vector_store: faiss
  top_k: 5
generation:
  max_new_tokens: 512
  temperature: 0.2
```

### `run.bat`
Windows one-shot launcher. On first run: bootstraps `.venv` via `uv`, installs `requirements.txt`, builds the FAISS index, then opens Streamlit on `localhost:8501`. Subsequent runs skip the bootstrap and go straight to Streamlit.

### `docs/report/`
- **`report.md`** — the 15–20 page final report.
- **`POST_TRAINING.md`** — the closing-checklist (run eval → human eval → charts → write-up → submit).
- **`figures/`** — generated charts.

---

## Models & defaults

- **Base LLM**: `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit nf4 with `bitsandbytes`. Fits a Kaggle P100 (16 GB) or one T4 (15 GB).
- **Adapter**: [`Tamir39/qwen2_5-7b-vietnam-tax-lora`](https://huggingface.co/Tamir39/qwen2_5-7b-vietnam-tax-lora) (~160 MB).
- **Embeddings**: `intfloat/multilingual-e5-base` (768 dim, normalized, with `query:` / `passage:` prefixes).
- **Vector store**: FAISS `IndexFlatIP` (= cosine since vectors are unit-norm), top-k = 5.
- **Generation**: `temperature=0.2`, `max_new_tokens=512` by default.

Tweak in `src/config.py` or per-config in `experiments/configs/*.yaml`.

## Branching policy

`main` (protected) ← `develop` ← `feature/*`.
**Never** merge into `main` until the project is ready to ship. All integration happens on `develop`.

## Deliverables

- Source code (this repo)
- 15–20 page report (`docs/report/report.md`)
- Slides + 3–5 min demo video (author-prepared)
- Dataset & adapter on Hugging Face Hub (`Tamir39`)
