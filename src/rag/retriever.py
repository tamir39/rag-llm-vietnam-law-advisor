"""Top-k retriever over the FAISS index produced by ``vectorstore.build_index``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RetrievedPassage:
    rank: int
    score: float
    passage_id: str
    doc_id: str
    title: str
    passage_text: str
    url: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": self.score,
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "passage_text": self.passage_text,
            "url": self.url,
        }


def retrieve(
    query: str,
    index,
    meta: Sequence[dict],
    embedder,
    top_k: int = 5,
) -> list[RetrievedPassage]:
    q = embedder.encode_queries([query])
    scores, idxs = index.search(q, top_k)
    hits: list[RetrievedPassage] = []
    for rank, (score, i) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
        if i < 0:
            continue
        row = meta[i]
        hits.append(
            RetrievedPassage(
                rank=rank,
                score=float(score),
                passage_id=row["passage_id"],
                doc_id=row.get("doc_id", ""),
                title=row.get("title", ""),
                passage_text=row.get("passage_text", ""),
                url=row.get("url", ""),
            )
        )
    return hits


def format_context(hits: Sequence[RetrievedPassage], max_chars_per_hit: int | None = None) -> str:
    """Format hits as a prompt-ready ``Ngữ cảnh tham khảo`` block."""
    parts: list[str] = []
    for h in hits:
        text = h.passage_text
        if max_chars_per_hit is not None and len(text) > max_chars_per_hit:
            text = text[:max_chars_per_hit].rstrip() + "..."
        parts.append(f"[{h.passage_id}] {h.title}\n{text}")
    return "\n\n".join(parts)
