"""Generate report figures from `experiments/results/`.

Outputs to `docs/report/figures/`:
  - metrics_bar.png       — BLEU / ROUGE-L / BERTScore-F1 across A/B/C/D
  - retrieval_bar.png     — Recall@5 and MRR@10 for B/D
  - human_eval_bar.png    — placeholder human-eval means (overwrite when real)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "docs" / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CFG_LABELS = {
    "A": "A\nBase\n(no RAG)",
    "B": "B\nBase\n+RAG",
    "C": "C\nFT\n(no RAG)",
    "D": "D\nFT\n+RAG",
}
COLORS = {"A": "#9aa0a6", "B": "#4285f4", "C": "#fbbc04", "D": "#34a853"}


def fig_metrics() -> None:
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    cfgs = ["A", "B", "C", "D"]
    metrics = [
        ("BLEU",         [s[c]["bleu"]          for c in cfgs], 1.0),
        ("ROUGE-L",      [s[c]["rougeL"]   * 100 for c in cfgs], 1.0),
        ("BERTScore F1", [s[c]["bertscore_f1"] * 100 for c in cfgs], 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, (name, vals, _) in zip(axes, metrics):
        bars = ax.bar([CFG_LABELS[c] for c in cfgs], vals, color=[COLORS[c] for c in cfgs], edgecolor="black", linewidth=0.6)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Chỉ số tự động trên test set 54 câu", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG / "metrics_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_retrieval() -> None:
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    cfgs = ["B", "D"]
    r5 = [s[c]["recall@5"] for c in cfgs]
    mrr = [s[c]["mrr@10"] for c in cfgs]

    x = np.arange(len(cfgs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, r5, width, label="Recall@5", color="#4285f4", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width / 2, mrr, width, label="MRR@10", color="#34a853", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["B (Base+RAG)", "D (FT+RAG)"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Chất lượng truy hồi (E5-base + FAISS)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG / "retrieval_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_human_eval() -> None:
    p = RES / "human_eval" / "summary_human_eval.json"
    if not p.exists():
        print(f"skip: {p} missing")
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    cfgs = ["A", "B", "C", "D"]
    means = [s[c]["mean"] for c in cfgs]
    pct4 = [s[c]["pct_ge_4"] * 5 for c in cfgs]  # scale to same axis

    x = np.arange(len(cfgs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - width / 2, means, width, label="Trung bình (1-5)",
                color=[COLORS[c] for c in cfgs], edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, pct4, width, label="Tỉ lệ ≥ 4 (scaled ×5)",
                color="white", edgecolor="black", hatch="//", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CFG_LABELS[c] for c in cfgs])
    ax.set_ylim(0, 5.5)
    ax.set_title("Human eval — 50 câu, blinded, thang 1–5",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = FIG / "human_eval_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_metrics()
    fig_retrieval()
    fig_human_eval()
