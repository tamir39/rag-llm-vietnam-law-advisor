"""Streamlit demo: pick config A/B/C/D and ask a VAT question."""
from __future__ import annotations

import streamlit as st

from src.inference.pipeline import Config

st.set_page_config(page_title="VAT QA — RAG + LoRA", page_icon=":scales:")

st.title("Hệ thống hỏi đáp Luật Thuế GTGT (VN)")

config_label = st.sidebar.selectbox(
    "Cấu hình",
    options=[
        "A — LLM gốc, không RAG",
        "B — LLM gốc, có RAG",
        "C — LLM fine-tuned, không RAG",
        "D — LLM fine-tuned, có RAG",
    ],
    index=3,
)
config = Config(config_label[0])

question = st.text_area("Câu hỏi của bạn:", height=120)
if st.button("Trả lời") and question.strip():
    st.info("Inference chưa được triển khai — đây là khung dự án ban đầu.")
