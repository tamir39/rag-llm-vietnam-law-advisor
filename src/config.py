"""Central configuration for paths, model ids, and retrieval defaults."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
QA_DIR = DATA_DIR / "qa"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

KB_CSV = KB_DIR / "knowledge_base.csv"
TRAIN_QA = QA_DIR / "train_qa.jsonl"
TEST_QA = QA_DIR / "test_qa.jsonl"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
INDEX_DIR = ROOT_DIR / "experiments" / "index"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
BASE_LLM = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER = CHECKPOINTS_DIR / "qwen2_5-7b-vietnam-tax-lora"
HF_LORA_REPO = "Tamir39/qwen2_5-7b-vietnam-tax-lora"

VECTOR_STORE = "faiss"
TOP_K = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
