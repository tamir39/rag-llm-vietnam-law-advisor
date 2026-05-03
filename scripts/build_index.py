"""Build the FAISS index for the Vietnamese tax-law KB.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --kb data/knowledge_base/knowledge_base.csv \
                                  --out experiments/index \
                                  --model intfloat/multilingual-e5-base \
                                  --batch-size 32 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EMBEDDING_MODEL, INDEX_DIR, KB_CSV  # noqa: E402
from src.data.loader import load_knowledge_base  # noqa: E402
from src.rag.embeddings import get_embedder  # noqa: E402
from src.rag.vectorstore import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, default=KB_CSV)
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu / mps; default lets sentence-transformers pick.")
    args = parser.parse_args()

    print(f"Loading KB from {args.kb} ...")
    passages = load_knowledge_base(args.kb)
    print(f"  {len(passages)} passages loaded")

    print(f"Loading embedder: {args.model} (device={args.device or 'auto'})")
    embedder = get_embedder(args.model, device=args.device, batch_size=args.batch_size)
    print(f"  embedding dim = {embedder.dim}")

    t0 = time.time()
    out = build_index(passages, embedder, args.out)
    dt = time.time() - t0
    print(f"Index written to {out}/ ({dt:.1f}s for {len(passages)} passages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
