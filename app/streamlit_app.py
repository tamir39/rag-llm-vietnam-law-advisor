"""Streamlit demo: ask Vietnamese tax-law questions across the 4 configs.

Run from the repo root:

    streamlit run app/streamlit_app.py

Requirements before launch:
    1. ``python scripts/build_index.py`` has produced ``experiments/index/``.
    2. (For configs C / D) the LoRA adapter is either at the local path
       ``checkpoints/qwen2_5-7b-vietnam-tax-lora/`` or pullable from
       ``Tamir39/qwen2_5-7b-vietnam-tax-lora`` on the Hub.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INDEX_DIR, ROOT_DIR
from src.inference.pipeline import InferencePipeline


CONFIG_OPTIONS = {
    "A — LLM gốc, không RAG": ROOT_DIR / "experiments" / "configs" / "A_base_no_rag.yaml",
    "B — LLM gốc, có RAG": ROOT_DIR / "experiments" / "configs" / "B_base_with_rag.yaml",
    "C — LLM fine-tuned, không RAG": ROOT_DIR / "experiments" / "configs" / "C_finetuned_no_rag.yaml",
    "D — LLM fine-tuned, có RAG": ROOT_DIR / "experiments" / "configs" / "D_finetuned_with_rag.yaml",
}


st.set_page_config(page_title="Hỏi đáp Luật Thuế VN — RAG + LoRA", page_icon="⚖️", layout="wide")

st.title("⚖️ Hệ thống hỏi đáp Luật Thuế Việt Nam")
st.caption(
    "RAG (FAISS + multilingual-e5-base) + Qwen2.5-7B-Instruct (QLoRA fine-tuned). "
    "Phạm vi: Thuế GTGT, Thuế TNCN, Thuế sử dụng đất phi nông nghiệp, Thuế TTĐB, Thuế TNDN."
)

# ----- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Cấu hình")
    config_label = st.selectbox("Chọn cấu hình", list(CONFIG_OPTIONS.keys()), index=3)

    st.divider()
    st.markdown("**Tham số sinh**")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    max_new_tokens = st.slider("max_new_tokens", 64, 1024, 384, 64)
    top_k_override = st.slider("top-k truy hồi", 1, 10, 5, 1)

    st.divider()
    st.markdown("**Trạng thái**")
    st.caption(f"FAISS index: {'✅' if (INDEX_DIR / 'kb.faiss').exists() else '❌ chưa build'}")


# ----- Pipeline cache --------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải mô hình ... (lần đầu mất 1-2 phút)")
def get_pipeline(config_path_str: str) -> InferencePipeline:
    pipe = InferencePipeline(Path(config_path_str))
    pipe.load()
    return pipe


# Eager-load on startup if LAWMATE_PRELOAD_CONFIG is set (e.g. "D"). This warms
# Streamlit's @st.cache_resource before the public URL goes live, so the first
# visitor doesn't trigger a 15GB download through the tunnel.
import os as _os
_preload_key = _os.environ.get("LAWMATE_PRELOAD_CONFIG")
if _preload_key:
    _label = next((k for k in CONFIG_OPTIONS if k.startswith(_preload_key)), None)
    if _label:
        get_pipeline(str(CONFIG_OPTIONS[_label]))


# ----- Main ------------------------------------------------------------------

DEFAULT_QS = [
    "Thuế suất thuế giá trị gia tăng đối với hàng hóa xuất khẩu là bao nhiêu?",
    "Đối tượng nào được miễn thuế sử dụng đất phi nông nghiệp?",
    "Hàng hóa nào chịu thuế tiêu thụ đặc biệt?",
    "Thu nhập chịu thuế thu nhập cá nhân gồm những khoản nào?",
    "Thuế suất thuế thu nhập doanh nghiệp phổ thông hiện nay là bao nhiêu?",
]
example = st.selectbox("Ví dụ câu hỏi (tùy chọn)", [""] + DEFAULT_QS)
question = st.text_area("Câu hỏi của bạn:", value=example, height=110, key="question")

ask = st.button("Trả lời", type="primary", use_container_width=False)

if ask and question.strip():
    cfg_path = CONFIG_OPTIONS[config_label]
    pipe = get_pipeline(str(cfg_path))

    # Apply per-call overrides without mutating the cached pipeline.
    pipe.cfg["generation"]["temperature"] = temperature
    pipe.cfg["generation"]["max_new_tokens"] = max_new_tokens
    if pipe.cfg["rag"]["enabled"]:
        pipe.cfg["rag"]["top_k"] = top_k_override

    with st.spinner("Đang sinh câu trả lời..."):
        result = pipe.answer(question.strip())

    st.subheader("Trả lời")
    st.write(result.answer)

    if result.retrieved_passage_ids:
        st.subheader("Căn cứ pháp lý đã truy hồi")
        for rank, (pid, score) in enumerate(
            zip(result.retrieved_passage_ids, result.retrieved_scores), start=1
        ):
            row = next((m for m in pipe._faiss_meta if m["passage_id"] == pid), None)
            with st.expander(f"#{rank}  {pid}  (score={score:.3f}) — {row['title'] if row else ''}"):
                if row:
                    st.markdown(f"**Đoạn trích:**\n\n{row['passage_text']}")
                    if row.get("url"):
                        st.markdown(f"[Nguồn]({row['url']})")
                else:
                    st.write("(không tìm thấy metadata)")
elif ask:
    st.warning("Hãy nhập câu hỏi.")
