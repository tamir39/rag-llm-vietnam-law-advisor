"""Inference pipeline for the 4-config experiment matrix.

Each config is a ``YAML`` file under ``experiments/configs/`` with the same
shape::

    name: <id>
    description: <text>
    model:
      base_id: Qwen/Qwen2.5-7B-Instruct
      lora_adapter: checkpoints/... | null
      hf_lora_repo: Tamir39/...        # optional, used if local path missing
    rag:
      enabled: true | false
      embedding_model: intfloat/multilingual-e5-base
      vector_store: faiss
      top_k: 5
    generation:
      max_new_tokens: 512
      temperature: 0.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import torch
import gc

import yaml

from src.config import EMBEDDING_MODEL, INDEX_DIR


class Config(str, Enum):
    A_BASE_NO_RAG = "A"
    B_BASE_RAG = "B"
    C_FT_NO_RAG = "C"
    D_FT_RAG = "D"


@dataclass
class PipelineResult:
    answer: str
    retrieved_passage_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)


def _resolve_lora(model_cfg: dict) -> str | None:
    """Prefer local checkpoint dir; fall back to HF repo id if present."""
    local = model_cfg.get("lora_adapter")
    if local:
        p = Path(local)
        if p.exists():
            return str(p)
    return model_cfg.get("hf_lora_repo")


class InferencePipeline:
    def __init__(self, config_path: str | Path):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.name = self.cfg["name"]
        self._model = None
        self._tokenizer = None
        self._embedder = None
        self._faiss_index = None
        self._faiss_meta = None

    def load(self) -> None:
        from src.inference.llm import load_llm

        lora = _resolve_lora(self.cfg["model"])
        self._model, self._tokenizer = load_llm(
            self.cfg["model"]["base_id"],
            lora_path=lora,
            load_in_4bit=True,
        )
        if self.cfg["rag"]["enabled"]:
            from src.rag.embeddings import get_embedder
            from src.rag.vectorstore import load_index

            self._embedder = get_embedder(
                self.cfg["rag"].get("embedding_model", EMBEDDING_MODEL)
            )
            self._faiss_index, self._faiss_meta = load_index(INDEX_DIR)

    def answer(self, question: str) -> PipelineResult:
        if self._model is None:
            raise RuntimeError("Pipeline not loaded — call .load() first.")

        from src.inference.llm import generate
        from src.rag.prompts import build_no_rag_prompt, build_rag_prompt
        from src.rag.retriever import format_context, retrieve

        gen_cfg = self.cfg["generation"]
        if self.cfg["rag"]["enabled"]:
            top_k = self.cfg["rag"].get("top_k", 5)
            hits = retrieve(
                question,
                self._faiss_index,
                self._faiss_meta,
                self._embedder,
                top_k=top_k,
            )
            ctx = format_context(hits)
            prompt = build_rag_prompt(question, ctx).split("\n", 2)[2]
            # build_rag_prompt embeds the system prompt as the first line; the
            # generate() helper re-adds it via the chat template, so we pass
            # only the user-facing portion (everything after the system block).
            prompt = (
                "Ngữ cảnh tham khảo:\n"
                f"{ctx}\n\n"
                f"Câu hỏi: {question}\n"
                "Hãy trả lời dựa vào ngữ cảnh tham khảo. "
                "Nếu ngữ cảnh không đủ, hãy nói rõ."
            )
            text = generate(
                self._model, self._tokenizer, prompt,
                max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                temperature=gen_cfg.get("temperature", 0.2),
            )
            return PipelineResult(
                answer=text,
                retrieved_passage_ids=[h.passage_id for h in hits],
                retrieved_scores=[h.score for h in hits],
            )
        else:
            _ = build_no_rag_prompt  # symmetry; chat template is added in generate
            text = generate(
                self._model, self._tokenizer, question,
                max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                temperature=gen_cfg.get("temperature", 0.2),
            )
            return PipelineResult(answer=text)
        
    def unload(self):
        self._model = None
        self._tokenizer = None
        self._embedder = None
        self._faiss_index = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Pipeline resources cleared.")

_current_pipe = None
_current_config = None

def answer(question: str, config: Config, configs_dir: Path | None = None) -> str:
    global _current_pipe, _current_config
    
    if _current_pipe is not None and _current_config == config:
        return _current_pipe.answer(question).answer

    if _current_pipe is not None:
        _current_pipe.unload()
        _current_pipe = None

    """Convenience one-shot for the demo / CLI; reloads the model each call."""
    from src.config import ROOT_DIR

    configs_dir = configs_dir or (ROOT_DIR / "experiments" / "configs")
    name_map = {
        Config.A_BASE_NO_RAG: "A_base_no_rag.yaml",
        Config.B_BASE_RAG: "B_base_with_rag.yaml",
        Config.C_FT_NO_RAG: "C_finetuned_no_rag.yaml",
        Config.D_FT_RAG: "D_finetuned_with_rag.yaml",
    }
    _current_pipe = InferencePipeline(configs_dir / name_map[config])
    _current_pipe.load()
    _current_config = config
    
    return _current_pipe.answer(question).answer
