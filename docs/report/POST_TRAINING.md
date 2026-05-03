# Sau khi fine-tune xong trên Kaggle

Checklist khép vòng dự án.

## 1. Xác nhận adapter đã có

- [ ] Tab **HF**: https://huggingface.co/Tamir39/qwen2_5-7b-vietnam-tax-lora hiển thị adapter mới.
- [ ] Loss cuối cùng (in trên log Kaggle) hợp lý — không phải NaN, không nhảy lên đột ngột.
- [ ] Thử inference nhanh ngay trong notebook 03 (cell cuối): hỏi 1-2 câu, kiểm tra trả lời tiếng Việt mạch lạc.

## 2. Chạy đánh giá tự động

```bash
# Trên máy có GPU (Kaggle hoặc Colab Pro), từ thư mục repo:
python scripts/run_eval.py
```

Sản phẩm: `experiments/results/<config>/predictions.jsonl` + `metrics.json` cho 4 cấu hình + `summary.json`.

- [x] Đẩy `experiments/results/` lên git? Mặc định gitignore nội dung, nhưng nên commit `summary.json` để báo cáo có dấu tích thời gian.
- [x] Sao bảng số trong `summary.json` vào **bảng 6.1** của `report.md`.

## 3. Human eval (50 câu)

```bash
python scripts/build_human_eval.py --n 50
```

- [ ] Mở `experiments/results/human_eval/form.csv` (Excel / Google Sheets).
- [ ] Điền `rating_1..rating_4` (1–5) cho 50 hàng — ước lượng ~1 giờ.
- [ ] Ghép với `key.csv` theo `(row_id, slot)` để biết slot nào ứng với cấu hình nào.
- [ ] Tính trung bình & trung vị từng cấu hình A/B/C/D → **bảng 6.2**.

Snippet ghép nhanh (Python):

```python
import pandas as pd

form = pd.read_csv("experiments/results/human_eval/form.csv")
key  = pd.read_csv("experiments/results/human_eval/key.csv")
long = form.melt(
    id_vars=["row_id", "question", "gold_answer"],
    value_vars=["rating_1", "rating_2", "rating_3", "rating_4"],
    var_name="slot_col", value_name="rating",
)
long["slot"] = long["slot_col"].str.replace("rating_", "").astype(int)
joined = long.merge(key, on=["row_id", "slot"])
print(joined.groupby("config")["rating"].agg(["mean", "median", lambda s: (s>=4).mean()]))
```

## 4. Phân tích định tính

Chọn **2-3 câu** cho mục 6.3 của báo cáo:
- 1 ví dụ thành công nổi bật của cấu hình **D**.
- 1 ví dụ **A** thất bại hoặc bịa số liệu (hallucination).
- 1 ví dụ retriever chèn đoạn lệch chủ đề → cho thấy giới hạn của RAG.

In nguyên văn vào báo cáo: câu hỏi, gold answer, output 4 cấu hình, đoạn KB được retrieve.

## 5. Vẽ biểu đồ

Đặt vào `docs/report/figures/`:

- `metrics_bar.png` — biểu đồ cột BLEU/ROUGE-L/BERTScore F1 cho A/B/C/D.
- `human_eval_bar.png` — trung bình human-eval theo cấu hình.
- `recall_at_k.png` — đường Recall@k với k = 1..10 (chỉ B và D).

Có thể vẽ bằng matplotlib trực tiếp trong notebook 05.

## 6. Hoàn thiện báo cáo

- [ ] Điền **Tóm tắt** (abstract) với số liệu thật.
- [ ] Cập nhật mục **6.1**, **6.2**, **6.3**.
- [ ] Viết phần **7. Thảo luận** dựa trên số liệu cụ thể.
- [ ] Viết phần **8. Kết luận**.
- [ ] Render ra PDF (Pandoc / Typora / VSCode markdown-pdf), kiểm tra số trang ≥ 15.
- [ ] Chuẩn bị slide & video demo (do tác giả tự thực hiện).

## 7. Đóng dự án

- [ ] Tạo PR `develop` → `main` chỉ khi User xác nhận đồ án sẵn sàng nộp.
- [ ] Tag release `v1.0` trên `main`.
- [ ] Cập nhật README link cuối cùng tới video demo + báo cáo PDF.
