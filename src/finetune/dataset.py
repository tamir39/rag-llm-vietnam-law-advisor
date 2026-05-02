"""Build the SFT dataset for QLoRA fine-tuning of Qwen2.5-7B-Instruct.

Each ``train_qa.jsonl`` row becomes a Qwen chat-template conversation:

    <|im_start|>system
    {SYSTEM_VI}<|im_end|>
    <|im_start|>user
    {question}                           # ``context_mode="none"``
    -- or --
    Ngữ cảnh tham khảo:
    [{passage_id}] {title}
    {passage_text}

    Câu hỏi: {question}
    Hãy trả lời dựa vào ngữ cảnh tham khảo. ...   # ``context_mode="gold" / "mixed"``
    <|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>

Default ``context_mode="mixed"`` balances 50/50 with-context / no-context so the
same adapter is competitive in both config C (FT, no RAG) and D (FT + RAG) of
the experiment matrix.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

from src.config import KB_CSV
from src.data.loader import load_knowledge_base, load_qa
from src.rag.prompts import SYSTEM_VI, build_no_rag_prompt, build_rag_prompt


ContextMode = Literal["none", "gold", "mixed"]


def _kb_index(kb_path: Path) -> dict[str, dict]:
    return {row["passage_id"]: row for row in load_knowledge_base(kb_path)}


def _gold_context(row: dict) -> str:
    return f"[{row['passage_id']}] {row.get('title', '')}\n{row['passage_text']}"


def _format_example(qa: dict, kb: dict[str, dict], with_context: bool) -> dict:
    if with_context:
        ctx_row = kb.get(qa["passage_id"])
        ctx = _gold_context(ctx_row) if ctx_row else ""
        user_text = build_rag_prompt(qa["question"], ctx).split("Câu hỏi:", 1)[1]
        # build_rag_prompt embeds the system prompt; for chat-template SFT we
        # inject system separately, so reconstruct user content cleanly:
        user_text = (
            "Ngữ cảnh tham khảo:\n"
            f"{ctx}\n\n"
            f"Câu hỏi: {qa['question']}\n"
            "Hãy trả lời dựa vào ngữ cảnh tham khảo. "
            "Nếu ngữ cảnh không đủ, hãy nói rõ."
        )
    else:
        user_text = qa["question"]
        # Reference build_no_rag_prompt to keep the import live in case the
        # downstream evaluation code wants the same wording:
        _ = build_no_rag_prompt
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_VI},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": qa["answer"]},
        ]
    }


def build_sft_dataset(
    qa_path: Path,
    tokenizer,
    kb_path: Path = KB_CSV,
    context_mode: ContextMode = "mixed",
    seed: int = 42,
):
    """Return a HuggingFace ``Dataset`` with a single ``text`` column.

    ``text`` is the result of ``tokenizer.apply_chat_template(..., tokenize=False)``
    so SFTTrainer can pass it straight through.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    kb = _kb_index(kb_path)
    rows = list(load_qa(qa_path))

    examples: list[dict] = []
    for qa in rows:
        if context_mode == "none":
            with_ctx = False
        elif context_mode == "gold":
            with_ctx = True
        else:  # mixed
            with_ctx = rng.random() < 0.5
        examples.append(_format_example(qa, kb, with_ctx))

    def render(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(examples)
    ds = ds.map(render, remove_columns=["messages"])
    return ds
