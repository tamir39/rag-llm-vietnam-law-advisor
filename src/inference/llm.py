"""Load Qwen2.5-7B-Instruct (optionally LoRA-adapted) for inference.

The base model loads in 4-bit nf4 to fit a Kaggle P100 (16GB). When a LoRA
adapter is supplied — either a local path or an HF repo id — it is attached
via ``PeftModel.from_pretrained``.
"""
from __future__ import annotations

from pathlib import Path


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for generation
    return tok


def load_llm(
    model_id: str,
    lora_path: str | Path | None = None,
    load_in_4bit: bool = True,
):
    """Return ``(model, tokenizer)`` ready for ``generate``.

    ``lora_path`` may be a local directory or an HF Hub repo id (``user/name``).
    Pass ``None`` for the base model only.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = True

    if lora_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_path))
        model.eval()

    tokenizer = load_tokenizer(model_id)
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Run a single chat-template generation.

    ``prompt`` is the *user* turn text; we wrap it with the system prompt and
    Qwen chat template here. Returns just the assistant reply.
    """
    import torch
    from src.rag.prompts import SYSTEM_VI

    messages = [
        {"role": "system", "content": SYSTEM_VI},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    do_sample = temperature is not None and temperature > 0
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
