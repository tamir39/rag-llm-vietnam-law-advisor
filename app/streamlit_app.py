"""Streamlit demo for LawMate — Vietnamese Tax Q&A.

UX goals:
  * Always show which config is currently loaded in VRAM.
  * Single-slot model loading: switching configs unloads the previous one
    BEFORE allocating the new one, so VRAM stays bounded to one Qwen2.5-7B
    (~6 GB in 4-bit) regardless of how many configs the user clicks through.
  * Compare-mode runs configs sequentially (load → answer → unload), never
    holding more than one model in VRAM.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INDEX_DIR, ROOT_DIR
from src.inference.pipeline import InferencePipeline


# ----- Config catalog --------------------------------------------------------

@dataclass(frozen=True)
class ConfigCard:
    code: str
    title: str
    subtitle: str
    base_lora: str
    rag: str
    yaml_path: Path
    recommended: bool = False


CONFIGS: list[ConfigCard] = [
    ConfigCard("A", "A · Cơ sở", "LLM gốc, không RAG",
               "Base", "Không RAG",
               ROOT_DIR / "experiments/configs/A_base_no_rag.yaml"),
    ConfigCard("B", "B · +RAG", "LLM gốc + truy hồi tài liệu",
               "Base", "Có RAG",
               ROOT_DIR / "experiments/configs/B_base_with_rag.yaml"),
    ConfigCard("C", "C · +LoRA", "LLM fine-tuned, không RAG",
               "Fine-tuned", "Không RAG",
               ROOT_DIR / "experiments/configs/C_finetuned_no_rag.yaml"),
    ConfigCard("D", "D · LoRA + RAG", "Cấu hình đầy đủ — khuyến nghị",
               "Fine-tuned", "Có RAG",
               ROOT_DIR / "experiments/configs/D_finetuned_with_rag.yaml",
               recommended=True),
]
CONFIG_BY_CODE = {c.code: c for c in CONFIGS}


DEFAULT_QS = [
    "Thuế suất thuế giá trị gia tăng đối với hàng hóa xuất khẩu là bao nhiêu?",
    "Đối tượng nào được miễn thuế sử dụng đất phi nông nghiệp?",
    "Hàng hóa nào chịu thuế tiêu thụ đặc biệt?",
    "Thu nhập chịu thuế thu nhập cá nhân gồm những khoản nào?",
    "Thuế suất thuế thu nhập doanh nghiệp phổ thông hiện nay là bao nhiêu?",
]


# ----- Page setup ------------------------------------------------------------

st.set_page_config(page_title="LawMate — VN Tax Q&A",
                   page_icon="⚖️", layout="wide")

st.session_state.setdefault("pipe", None)
st.session_state.setdefault("loaded_code", None)
st.session_state.setdefault("last_qa", None)
st.session_state.setdefault("last_cmp", None)


# ----- Helpers ---------------------------------------------------------------

def _gpu_status() -> tuple[str, str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU", "—"
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        used = torch.cuda.memory_allocated(idx) / 1024**3
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return name, f"{used:.1f} / {total:.1f} GB"
    except Exception as exc:
        return "GPU?", f"({exc})"


def unload_active_pipeline() -> None:
    pipe = st.session_state.get("pipe")
    if pipe is not None:
        try:
            pipe.unload()
        except Exception:
            pass
        st.session_state.pipe = None
        st.session_state.loaded_code = None
        gc.collect()


def _is_oom(exc: BaseException) -> bool:
    """OOM check that works across PyTorch versions (sometimes raised as
    torch.cuda.OutOfMemoryError, sometimes as plain RuntimeError)."""
    return "out of memory" in str(exc).lower()


def _free_vram_after_oom() -> None:
    """Aggressively release VRAM after an OOM so the UI can recover without
    a kernel restart."""
    unload_active_pipeline()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def load_pipeline(code: str) -> InferencePipeline:
    """Load `code` into the single slot, unloading any prior pipeline first."""
    if st.session_state.loaded_code == code and st.session_state.pipe is not None:
        return st.session_state.pipe
    unload_active_pipeline()
    card = CONFIG_BY_CODE[code]
    pipe = InferencePipeline(card.yaml_path)
    pipe.load()
    st.session_state.pipe = pipe
    st.session_state.loaded_code = code
    return pipe


def run_answer(pipe: InferencePipeline, question: str,
               temperature: float, max_new_tokens: int, top_k: int):
    pipe.cfg["generation"]["temperature"] = temperature
    pipe.cfg["generation"]["max_new_tokens"] = max_new_tokens
    if pipe.cfg["rag"]["enabled"]:
        pipe.cfg["rag"]["top_k"] = top_k
    return pipe.answer(question.strip())


# ----- Header ----------------------------------------------------------------

st.title("⚖️ LawMate — Hỏi đáp Luật Thuế Việt Nam")
st.caption(
    "RAG (FAISS + multilingual-e5-base) + Qwen2.5-7B-Instruct (QLoRA fine-tuned). "
    "Phạm vi: GTGT, TNCN, TTĐB, TNDN, Thuế sử dụng đất phi nông nghiệp."
)

# ----- Status bar ------------------------------------------------------------

dev, vram = _gpu_status()
loaded_code = st.session_state.loaded_code
loaded_label = CONFIG_BY_CODE[loaded_code].title if loaded_code else "—"
faiss_ok = (INDEX_DIR / "kb.faiss").exists()

c1, c2, c3, c4 = st.columns([1.6, 1.0, 1.0, 0.6])
c1.metric("Đang tải", value=loaded_label)
c2.metric("GPU", value=dev)
c3.metric("VRAM", value=vram)
c4.metric("FAISS", value="✅" if faiss_ok else "❌")

st.divider()


# ----- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("💡 Câu hỏi gợi ý")
    st.caption("Sao chép một câu hỏi bên dưới và dán vào ô câu hỏi bên phải.")
    for q in DEFAULT_QS:
        st.markdown(f"• {q}")
    st.divider()

    st.header("Tham số sinh")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 384, 64)
    top_k = st.slider("Top-k truy hồi (RAG)", 1, 10, 5, 1)
    st.divider()
    st.markdown("**Thư mục FAISS index**")
    st.code(str(INDEX_DIR), language="text")
    st.divider()
    if st.button("🧹 Giải phóng GPU", use_container_width=True,
                 disabled=(loaded_code is None)):
        unload_active_pipeline()
        st.rerun()


# ----- Config picker (cards) -------------------------------------------------

st.subheader("1. Chọn cấu hình")

cols = st.columns(4)
clicked_code: str | None = None
for col, card in zip(cols, CONFIGS):
    with col:
        is_active = (st.session_state.loaded_code == card.code)
        badge = " · ✅ ĐANG TẢI" if is_active else (" · ⭐" if card.recommended else "")
        with st.container(border=True):
            st.markdown(f"**{card.title}**{badge}")
            st.caption(card.subtitle)
            st.markdown(
                f"- LLM: **{card.base_lora}**\n"
                f"- Truy hồi: **{card.rag}**"
            )
            if st.button(
                "✅ Đang dùng" if is_active else "⬇️ Tải cấu hình",
                key=f"load_{card.code}",
                use_container_width=True,
                type=("primary" if (card.recommended and not is_active) else "secondary"),
                disabled=is_active,
            ):
                clicked_code = card.code

if clicked_code is not None:
    try:
        with st.spinner(f"Đang tải cấu hình {clicked_code} (lần đầu mất 1–2 phút)..."):
            load_pipeline(clicked_code)
        st.rerun()
    except Exception as exc:
        if _is_oom(exc):
            _free_vram_after_oom()
            st.error(
                f"🔴 **Hết bộ nhớ GPU** khi tải cấu hình **{clicked_code}**. "
                "Đã tự động giải phóng VRAM. "
                "Hãy đợi vài giây rồi thử tải lại, hoặc chọn cấu hình khác."
            )
        else:
            raise

st.caption(
    "💡 Chỉ một cấu hình được giữ trong VRAM tại một thời điểm — "
    "khi đổi cấu hình, mô hình cũ được giải phóng trước, mô hình mới mới được tải."
)

st.divider()


# ----- Tabs: single QA vs compare -------------------------------------------

tab_qa, tab_cmp = st.tabs(["🗣️ Hỏi 1 cấu hình", "⚖️ So sánh nhiều cấu hình"])

with tab_qa:
    st.subheader("2. Đặt câu hỏi")
    question = st.text_area("Câu hỏi của bạn", height=120, key="q_single",
                            placeholder="Nhập câu hỏi, hoặc sao chép một gợi ý "
                                        "từ thanh bên trái.")

    loaded = st.session_state.loaded_code
    if loaded is None:
        st.info("⬆️ Tải một cấu hình ở phần trên trước khi hỏi.")
    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        btn = st.button(
            f"Trả lời với cấu hình {loaded}" if loaded else "Trả lời",
            type="primary",
            disabled=(loaded is None or not question.strip()),
            key="ask_single",
            use_container_width=True,
        )
    with col_clear:
        if st.button("Xóa kết quả", key="clear_single",
                     use_container_width=True,
                     disabled=(st.session_state.get("last_qa") is None)):
            st.session_state.last_qa = None
            st.rerun()

    if btn:
        try:
            with st.spinner("Đang sinh câu trả lời..."):
                result = run_answer(st.session_state.pipe, question,
                                    temperature, max_new_tokens, top_k)
        except Exception as exc:
            if _is_oom(exc):
                _free_vram_after_oom()
                st.error(
                    "🔴 **Hết bộ nhớ GPU** khi sinh câu trả lời. "
                    "Đã tự động giải phóng VRAM. "
                    "Hãy giảm **Max new tokens** hoặc **Top-k truy hồi** ở "
                    "thanh bên trái rồi tải lại cấu hình."
                )
                st.stop()
            raise

        pipe = st.session_state.pipe
        retrieved_rows = []
        for pid, score in zip(result.retrieved_passage_ids,
                              result.retrieved_scores):
            row = next((m for m in pipe._faiss_meta
                        if m["passage_id"] == pid), None)
            retrieved_rows.append({
                "pid": pid,
                "score": float(score),
                "title": row["title"] if row else "",
                "passage_text": row["passage_text"] if row else None,
                "url": row.get("url") if row else None,
            })
        st.session_state.last_qa = {
            "config": loaded,
            "question": question.strip(),
            "answer": result.answer,
            "retrieved": retrieved_rows,
        }

    qa = st.session_state.get("last_qa")
    if qa is not None:
        st.subheader("Câu trả lời")
        st.caption(f"Cấu hình **{qa['config']}** · Câu hỏi: _{qa['question']}_")
        st.markdown(qa["answer"])

        if qa["retrieved"]:
            st.subheader("📚 Căn cứ pháp lý đã truy hồi")
            for rank, item in enumerate(qa["retrieved"], start=1):
                with st.expander(
                    f"#{rank}  {item['pid']}  (score={item['score']:.3f}) — "
                    f"{item['title']}"
                ):
                    if item["passage_text"]:
                        st.markdown(f"**Đoạn trích:**\n\n{item['passage_text']}")
                        if item["url"]:
                            st.markdown(f"[Nguồn]({item['url']})")
                    else:
                        st.write("(không tìm thấy metadata)")

with tab_cmp:
    st.subheader("2. Chọn cấu hình muốn so sánh")
    chosen_codes: list[str] = []
    cols2 = st.columns(4)
    for col, card in zip(cols2, CONFIGS):
        with col:
            picked = st.checkbox(
                f"{card.title}", key=f"cmp_{card.code}",
                value=(card.code == "D"),
            )
            st.caption(f"{card.base_lora} · {card.rag}")
            if picked:
                chosen_codes.append(card.code)

    cmp_question = st.text_area(
        "Câu hỏi", height=120, key="q_cmp",
        placeholder="Nhập câu hỏi, hoặc sao chép một gợi ý từ thanh bên trái.",
    )

    st.caption(
        "ℹ️ Mỗi cấu hình được tải / giải phóng tuần tự để VRAM luôn chỉ giữ 1 mô hình. "
        "Tổng thời gian ≈ thời-gian-tải × số-cấu-hình."
    )

    col_run, col_clear_cmp = st.columns([3, 1])
    with col_run:
        run_cmp = st.button(
            f"So sánh {len(chosen_codes)} cấu hình (tuần tự)",
            type="primary",
            disabled=(len(chosen_codes) == 0 or not cmp_question.strip()),
            key="ask_cmp",
            use_container_width=True,
        )
    with col_clear_cmp:
        if st.button("Xóa kết quả", key="clear_cmp",
                     use_container_width=True,
                     disabled=(st.session_state.get("last_cmp") is None)):
            st.session_state.last_cmp = None
            st.rerun()

    if run_cmp:
        results = {}
        oom_codes: list[str] = []
        progress = st.progress(0.0, text="Bắt đầu...")
        for i, code in enumerate(chosen_codes, start=1):
            try:
                progress.progress((i - 1) / len(chosen_codes),
                                  text=f"Đang tải cấu hình {code} "
                                       f"({i}/{len(chosen_codes)})...")
                pipe = load_pipeline(code)
                progress.progress((i - 0.5) / len(chosen_codes),
                                  text=f"Đang sinh câu trả lời cho {code} ...")
                res = run_answer(pipe, cmp_question,
                                 temperature, max_new_tokens, top_k)
                results[code] = {
                    "answer": res.answer,
                    "retrieved": [
                        (pid, float(sc))
                        for pid, sc in zip(res.retrieved_passage_ids,
                                           res.retrieved_scores)
                    ],
                }
            except Exception as exc:
                if _is_oom(exc):
                    _free_vram_after_oom()
                    oom_codes.append(code)
                    results[code] = {
                        "answer": ("🔴 **Hết bộ nhớ GPU** khi chạy cấu hình này. "
                                   "VRAM đã được giải phóng tự động."),
                        "retrieved": [],
                    }
                    continue
                raise
        progress.progress(1.0, text="Hoàn tất.")
        st.session_state.last_cmp = {
            "question": cmp_question.strip(),
            "codes": list(chosen_codes),
            "results": results,
        }
        if oom_codes:
            st.warning(
                f"⚠️ Không đủ VRAM cho cấu hình: {', '.join(oom_codes)}. "
                "Hãy giảm **Max new tokens** / **Top-k** hoặc bỏ chọn cấu "
                "hình nặng hơn rồi thử lại."
            )

    cmp_state = st.session_state.get("last_cmp")
    if cmp_state is not None:
        st.subheader("Kết quả so sánh")
        st.caption(f"Câu hỏi: _{cmp_state['question']}_")
        codes = cmp_state["codes"]
        result_cols = st.columns(len(codes))
        for col, code in zip(result_cols, codes):
            card = CONFIG_BY_CODE[code]
            res = cmp_state["results"][code]
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {card.title}")
                    st.caption(f"{card.base_lora} · {card.rag}")
                    st.markdown(res["answer"])
                    if res["retrieved"]:
                        with st.expander(
                            f"📚 {len(res['retrieved'])} đoạn truy hồi"):
                            for pid, sc in res["retrieved"]:
                                st.markdown(f"- `{pid}` ({sc:.3f})")
