# rag-llm-vietnam-law-advisor

Hệ thống hỏi đáp tiếng Việt trên miền **Luật Thuế giá trị gia tăng (VAT)**, kết hợp **RAG** với **LLM fine-tuned bằng LoRA/QLoRA**.

Đồ án cuối kỳ môn *Nhập môn Xử lý ngôn ngữ tự nhiên*.

## Mục tiêu

So sánh 4 cấu hình trên cùng bộ test 50 câu hỏi:

|              | Không RAG | Có RAG |
|--------------|-----------|--------|
| LLM gốc      | **A**     | **B**  |
| LLM fine-tuned | **C**   | **D**  |

Đánh giá: BLEU, ROUGE-L, BERTScore, Recall@5, human eval.

## Repo map

```
src/            # Python package (data, rag, finetune, inference, evaluation)
app/            # Streamlit demo
scripts/        # build_index, prepare_qa, run_eval
notebooks/      # 01-data → 02-embed → 03-finetune → 04-rag → 05-eval
data/           # knowledge_base CSV + train/test QA jsonl
experiments/    # YAML configs for A/B/C/D, plus results/
checkpoints/    # LoRA adapters (gitignored content)
docs/report/    # 15-20 trang báo cáo
tests/          # smoke tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # rồi điền HF_TOKEN
```

## Quick start (sau khi có data + index + adapter)

```bash
python scripts/build_index.py              # KB CSV → FAISS index
python scripts/run_eval.py --config experiments/configs/D_finetuned_with_rag.yaml
streamlit run app/streamlit_app.py
```

## Models & defaults

- **Base LLM**: `Qwen/Qwen2.5-7B-Instruct` (QLoRA 4-bit, fits Colab Free)
- **Embeddings**: `intfloat/multilingual-e5-base`
- **Vector store**: FAISS (CPU)
- **Top-k**: 5

Sửa trong `src/config.py` hoặc YAML trong `experiments/configs/`.

## Branching

`main` (protected) ← `develop` ← `feature/*`. Không bao giờ merge thẳng vào `main` cho đến khi đồ án sẵn sàng nộp.

## Deliverables

- Mã nguồn GitHub (repo này)
- Báo cáo 15–20 trang (`docs/report/`)
- Slide + video demo 3–5 phút
- Dataset & checkpoint (HuggingFace Hub + Drive)
