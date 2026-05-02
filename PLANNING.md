# Planning — Architecture & Decisions

## High-level flow

```
[Luật VAT (PDF/HTML)] → preprocess → [knowledge_base CSV]
                                            │
                                            ▼
                                   chunker + e5-base
                                            │
                                            ▼
                                     [FAISS index]
                                            │
                Question ─────► retriever (top-k=5) ──┐
                   │                                  ▼
                   ▼                         prompt_template
              [Base LLM]  ←──── LoRA adapter ◄────────┘
              Qwen2.5-7B            (QLoRA SFT on
                                   train_qa.jsonl)
                   │
                   ▼
                Answer + cited passages
```

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base LLM | Qwen2.5-7B-Instruct | Mạnh tiếng Việt, license Apache-2.0, vừa Colab Free với 4-bit QLoRA. |
| Fine-tune | QLoRA (4-bit, r=16) | Chỉ ~50–100 MB adapter; train được trên T4. |
| Embeddings | `intfloat/multilingual-e5-base` | Hỗ trợ tiếng Việt tốt, kích thước vừa, có sẵn trên HF. |
| Vector store | FAISS (CPU) | Không cần server, đóng gói file `.faiss` cùng repo/Drive. |
| Top-k | 5 | Khớp với metric `Recall@5`. |
| Prompt | Bilingual VN system + (RAG) context section | Xem `src/rag/prompts.py`. |
| Demo | Streamlit | Một file, deploy nhanh lên Streamlit Cloud / HF Spaces. |

## Experiment matrix

| ID | Base LLM | LoRA | RAG | Config file |
|----|----------|------|-----|-------------|
| A  | Qwen2.5-7B | – | – | `experiments/configs/A_base_no_rag.yaml` |
| B  | Qwen2.5-7B | – | ✓ | `experiments/configs/B_base_with_rag.yaml` |
| C  | Qwen2.5-7B | ✓ | – | `experiments/configs/C_finetuned_no_rag.yaml` |
| D  | Qwen2.5-7B | ✓ | ✓ | `experiments/configs/D_finetuned_with_rag.yaml` |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Colab Free OOM khi train 7B | QLoRA 4-bit + grad-accum=16 + max_seq_len=2048; fallback PhoGPT-4B. |
| QA pairs không đa dạng → overfit | Sinh QA bằng GPT-4 + lọc thủ công; cân bằng theo chương / điều. |
| Retrieval kém vì chunks quá to | Bắt đầu chunk 256 token, overlap 32; tinh chỉnh sau dựa Recall@5. |
| Báo cáo trễ | Viết song song với code; mỗi notebook xuất một section của report. |
