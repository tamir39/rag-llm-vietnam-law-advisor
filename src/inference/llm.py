"""Load base or LoRA-adapted LLM for inference."""
from __future__ import annotations


def load_llm(model_id: str, lora_path=None, load_in_4bit: bool = True):
    raise NotImplementedError
