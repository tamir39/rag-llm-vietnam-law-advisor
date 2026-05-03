# Sau khi fine-tune xong trên Kaggle

Checklist khép vòng dự án.

## 1. Xác nhận adapter đã có

- [x] Tab **HF**: https://huggingface.co/Tamir39/qwen2_5-7b-vietnam-tax-lora hiển thị adapter mới.
- [x] Thử inference nhanh: kết quả 4-config (đặc biệt C và D) cho thấy adapter sinh tiếng Việt mạch lạc, không NaN — xem [`experiments/results/summary.json`](../../experiments/results/summary.json).

### 1.1 Cấu hình huấn luyện (snapshot)

Lấy từ [`src/finetune/lora_config.py`](../../src/finetune/lora_config.py):

| Tham số                      | Giá trị                                              |
|------------------------------|------------------------------------------------------|
| Base model                   | `Qwen/Qwen2.5-7B-Instruct`                           |
| Quantization                 | 4-bit nf4, double-quant, compute = bfloat16          |
| LoRA `r` / `alpha` / dropout | 16 / 32 / 0.05                                       |
| Target modules               | `q,k,v,o_proj` + `gate,up,down_proj` (att + MLP)     |
| Per-device batch / grad accum| 1 / 16  (effective batch = 16)                       |
| Learning rate                | 2e-4, scheduler = cosine, warmup ratio = 0.03        |
| Optimizer                    | `paged_adamw_8bit` (mặc định TRL)                    |
| Epochs / max-seq-len         | 3 / 2048                                             |
| Train ví dụ                  | 305 (mixed-context 50/50 — chèn `passage_text` vàng)|
| Hạ tầng                      | Kaggle Notebook P100 16GB hoặc T4 ×2 30GB            |
| Adapter                      | `Tamir39/qwen2_5-7b-vietnam-tax-lora` (~160 MB)      |

### 1.2 Loss curve (cần điền)

Notebook 03 không lưu lại JSON loss-history; trước khi nộp báo cáo:

1. Mở `notebooks/03_finetune_lora_kaggle.ipynb` trên Kaggle.
2. Trong cell huấn luyện, sửa `TrainingArguments` để bật `report_to="none"` + thêm `logging_steps=5` (đã có) — output console sẽ in `{'loss': X, 'epoch': Y}` mỗi 5 step.
3. Copy toàn bộ stdout, lưu vào `experiments/training_log.txt` rồi parse:
   ```python
   import re, json, matplotlib.pyplot as plt
   log = open("experiments/training_log.txt").read()
   pts = [(float(s), float(e)) for s, e in re.findall(r"'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)", log)]
   plt.plot([e for _, e in pts], [s for s, _ in pts])
   plt.xlabel("epoch"); plt.ylabel("train loss"); plt.savefig("docs/report/figures/train_loss.png", dpi=150)
   ```
4. Chèn vào **§4** của `report.md`: hình `train_loss.png` + 2-3 câu nhận xét (giảm đều hay overfit, có spike không).

> Nếu lần huấn luyện đã xong và log không còn — chạy lại 1 epoch nhanh (~30 phút) chỉ để chụp loss curve. Adapter cuối không cần đụng tới.

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
