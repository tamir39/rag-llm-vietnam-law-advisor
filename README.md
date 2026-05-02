# rag-llm-vietnam-law-advisor

Hệ thống hỏi đáp tiếng Việt trên miền **pháp luật thuế Việt Nam**, kết hợp **RAG** (FAISS + `multilingual-e5-base`) với **LLM fine-tuned bằng QLoRA** trên `Qwen/Qwen2.5-7B-Instruct`.

Đồ án cuối kỳ môn *Nhập môn Xử lý ngôn ngữ tự nhiên*.

## Phạm vi

5 luật thuế Việt Nam (600 đoạn văn pháp lý, ~284 ký tự/đoạn):

| `doc_id`     | Văn bản                                                                | Số đoạn |
|--------------|------------------------------------------------------------------------|--------:|
| `doc_000001` | VBHN 12/VBHN-VPQH 2026 — Luật Thuế giá trị gia tăng (GTGT)             |     143 |
| `doc_000002` | VBHN 103/VBHN-VPQH 2025 — Luật Thuế thu nhập cá nhân (TNCN)            |     140 |
| `doc_000003` | Luật 48/2010/QH12 — Thuế sử dụng đất phi nông nghiệp                   |      58 |
| `doc_000004` | Luật 66/2025/QH15 — Thuế tiêu thụ đặc biệt (TTĐB)                      |     127 |
| `doc_000005` | Luật 67/2025/QH15 — Thuế thu nhập doanh nghiệp (TNDN)                  |     132 |

QA: **305** câu huấn luyện + **54** câu kiểm tra (`data/qa/`).

## Mục tiêu

So sánh 4 cấu hình trên cùng bộ test:

|              | Không RAG | Có RAG |
|--------------|-----------|--------|
| LLM gốc       | **A**     | **B**  |
| LLM fine-tuned | **C**     | **D**  |

Chỉ số đánh giá: **BLEU**, **ROUGE-L**, **BERTScore (vi)**, **Recall@5**, và **human eval** (1–5) trên 50 câu mẫu blinded.

## Repo map

```
src/
  config.py              # đường dẫn + tên model + hằng số
  data/loader.py         # đọc KB CSV (utf-8-sig) + QA JSONL
  rag/                   # embeddings.py, vectorstore.py, retriever.py, prompts.py
  finetune/              # dataset.py, lora_config.py, trainer.py
  inference/             # llm.py, pipeline.py
  evaluation/            # metrics.py (BLEU/ROUGE/BERTScore), retrieval_eval.py (Recall@k, MRR)
app/streamlit_app.py     # demo UI
scripts/
  prepare_qa.py          # validate train/test JSONL
  build_index.py         # KB → FAISS index
  run_eval.py            # 4-config matrix → metrics JSON
  build_human_eval.py    # blinded 50-Q form
notebooks/               # 01-data → 02-embed → 03-finetune (Kaggle) → 04-rag → 05-eval
data/                    # knowledge_base.csv + train/test_qa.jsonl
experiments/configs/     # A/B/C/D YAML
experiments/results/     # predictions + metrics (sau khi chạy run_eval)
checkpoints/             # LoRA adapter (gitignored, push lên HF)
docs/report/             # báo cáo + slide
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # Linux/macOS: cp; rồi điền HF_TOKEN
```

## Pipeline đầy đủ

```bash
# 1. Validate dữ liệu
python scripts/prepare_qa.py data/qa/train_qa.jsonl
python scripts/prepare_qa.py data/qa/test_qa.jsonl --train data/qa/train_qa.jsonl

# 2. Build FAISS index từ KB
python scripts/build_index.py

# 3. Fine-tune trên Kaggle (mở notebooks/03_finetune_lora_kaggle.ipynb)
#    → adapter được push lên Tamir39/qwen2_5-7b-vietnam-tax-lora

# 4. Chạy ma trận đánh giá 4 cấu hình
python scripts/run_eval.py                       # tất cả A/B/C/D
python scripts/run_eval.py --configs B D         # tập con

# 5. Tạo form human-eval (blinded)
python scripts/build_human_eval.py --n 50

# 6. Demo
streamlit run app/streamlit_app.py
```

## Models & defaults

- **Base LLM**: `Qwen/Qwen2.5-7B-Instruct` — QLoRA 4-bit nf4, fits Kaggle P100 16 GB
- **Adapter**: [`Tamir39/qwen2_5-7b-vietnam-tax-lora`](https://huggingface.co/Tamir39/qwen2_5-7b-vietnam-tax-lora)
- **Embeddings**: `intfloat/multilingual-e5-base` (384 dim, normalized, prefixes `query:`/`passage:`)
- **Vector store**: FAISS `IndexFlatIP` (= cosine), top-k = 5
- **Generation**: `temperature=0.2`, `max_new_tokens=512`

Tinh chỉnh trong `src/config.py` hoặc YAML tại `experiments/configs/`.

## Môi trường huấn luyện

Kaggle (P100 16 GB hoặc T4 ×2 30 GB, 30 GPU-hr/tuần, phiên 12 giờ).
Notebook hướng dẫn từng bước: `notebooks/03_finetune_lora_kaggle.ipynb`.

## Branching

`main` (protected) ← `develop` ← `feature/*`.
Không bao giờ merge thẳng vào `main` cho đến khi đồ án sẵn sàng nộp.

## Deliverables

- Mã nguồn GitHub (repo này)
- Báo cáo 15–20 trang (`docs/report/report.md`)
- Slide + video demo 3–5 phút (do tác giả chuẩn bị)
- Dataset & checkpoint trên HuggingFace Hub (Tamir39)
