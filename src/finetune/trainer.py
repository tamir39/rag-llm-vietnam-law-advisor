"""QLoRA SFTTrainer wiring for Qwen2.5-7B-Instruct.

Designed to run on Kaggle (P100 16GB or T4×2 30GB, 12-hour session). The
heavy imports (``transformers``, ``peft``, ``trl``, ``bitsandbytes``) happen
inside ``train`` so this module is cheap to import on a CPU-only machine.
"""
from __future__ import annotations

from pathlib import Path

from src.finetune.lora_config import (
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LEN,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    TARGET_MODULES,
    WARMUP_RATIO,
)


def load_tokenizer(base_model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(base_model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    )
    model.config.use_cache = False
    return model


def _peft_config():
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(base_model_id: str, dataset, output_dir: Path, eval_dataset=None, return_trainer: bool = False):
    """Fine-tune ``base_model_id`` with QLoRA and save the adapter to ``output_dir``.

    When ``return_trainer=True``, returns ``(output_dir, trainer)`` so the caller
    can read ``trainer.state.log_history`` for the loss curve.
    """
    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(base_model_id)
    model = load_base_model(base_model_id)
    model = prepare_model_for_kbit_training(model)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_SEQ_LEN,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=_peft_config(),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if return_trainer:
        return output_dir, trainer
    return output_dir
