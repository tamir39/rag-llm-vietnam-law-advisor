# Product Requirements — VN VAT QA System

## 1. Vấn đề & người dùng

Người nộp thuế, kế toán, sinh viên luật cần tra cứu nhanh các quy định trong Luật Thuế GTGT bằng ngôn ngữ tự nhiên tiếng Việt, có dẫn chứng pháp lý.

## 2. Phạm vi

- **Domain**: Luật Thuế giá trị gia tăng (VBHN 12/VBHN-VPQH năm 2026 và các văn bản liên quan).
- **Không trong phạm vi**: tư vấn pháp lý cá nhân hoá, các sắc thuế khác, multi-turn dialog phức tạp.

## 3. Yêu cầu chức năng

| ID | Yêu cầu |
|----|---------|
| F1 | Knowledge base ≥ 600 đoạn passages từ luật VAT (đã có seed CSV). |
| F2 | Bộ huấn luyện ≥ 300 cặp QA, bộ test ≥ 50 cặp QA viết tay. |
| F3 | Pipeline RAG: chunking → embedding → FAISS → top-k retriever → prompt template. |
| F4 | LLM fine-tuned bằng LoRA/QLoRA (1B–7B param) trên Colab Free. |
| F5 | Inference pipeline hỗ trợ 4 cấu hình A/B/C/D qua YAML. |
| F6 | Demo UI (Streamlit) cho người dùng cuối hỏi và xem câu trả lời + nguồn. |

## 4. Yêu cầu phi chức năng

- Chạy được trên Colab Free (T4, 16 GB) cho fine-tune.
- Inference cấu hình B/D phản hồi ≤ ~10 giây cho câu hỏi đơn.
- Toàn bộ artefact reproducible từ `requirements.txt` + scripts.

## 5. Đánh giá

- **Generation**: BLEU, ROUGE-L, BERTScore (lang=vi).
- **Retrieval**: Recall@5.
- **Human eval**: 50 câu, thang điểm 1–5 cho độ chính xác và độ liên quan.

## 6. Sản phẩm bàn giao

- Repo GitHub có README đầy đủ.
- Báo cáo 15–20 trang trong `docs/report/`.
- Slide thuyết trình + video demo 3–5 phút.
- Dataset + checkpoint trên HuggingFace Hub + Google Drive.
