"""LoRA / QLoRA hyperparameters tuned for Kaggle (P100 16GB or T4×2 30GB)."""
from __future__ import annotations

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Attention + MLP projections — gives the adapter enough capacity to learn
# Vietnamese tax-law style without ballooning the trainable-param count.
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"

PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 2048
