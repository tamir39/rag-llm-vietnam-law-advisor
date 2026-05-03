"""Plot training loss from `experiments/loss_history.json`.

Source file format: the raw `trainer.state.log_history` list dumped by the
harvest cell in `notebooks/03_finetune_lora_kaggle.ipynb`. Each entry is a
dict; the entries we care about have keys `loss`, `epoch`, `step`.

Output: `docs/report/figures/train_loss.png`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "loss_history.json"
OUT = ROOT / "docs" / "report" / "figures" / "train_loss.png"


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. Run the harvest cell on Kaggle first.")
        return 1
    log = json.loads(SRC.read_text(encoding="utf-8"))
    points = [(e["step"], e["loss"], e["epoch"]) for e in log if "loss" in e and "step" in e]
    if not points:
        print("ERROR: no `loss` entries in log_history.")
        return 1

    steps = [p[0] for p in points]
    losses = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(steps, losses, color="#4285f4", linewidth=1.6)
    ax.scatter(steps, losses, color="#4285f4", s=18, zorder=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title(f"QLoRA fine-tuning loss curve ({len(points)} log points)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Mark epoch boundaries on a top axis if available
    epoch_pts = [(e["step"], e["epoch"]) for e in log if "epoch" in e and "step" in e]
    if epoch_pts:
        last_epoch = int(max(p[1] for p in epoch_pts))
        for k in range(1, last_epoch + 1):
            for s, ep in epoch_pts:
                if abs(ep - k) < 0.05:
                    ax.axvline(s, color="grey", linestyle=":", alpha=0.5)
                    ax.text(s, max(losses), f"epoch {k}", rotation=90,
                            va="top", ha="right", fontsize=9, color="grey")
                    break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"final loss: {losses[-1]:.4f}  |  min loss: {min(losses):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
