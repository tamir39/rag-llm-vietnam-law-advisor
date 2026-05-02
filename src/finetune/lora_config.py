"""LoRA / QLoRA hyperparameters tuned for Colab Free."""
from __future__ import annotations

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"

PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 2048
