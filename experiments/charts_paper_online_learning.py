"""
Online learning line chart: L2 centroid vs ML baselines (V3A, §10.3).

All values hardcoded from validated experimental record.

Outputs (paper_figures/):
  fig17_online_learning.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
PAPER_FIGS = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Decision checkpoints after training ends
X_POST = [400, 600, 1000, 1500]

# Initial static accuracy at x=200 (training ends)
INIT = {
    "l2":  94.78,
    "rf":  93.20,
    "lr":  92.38,
    "xgb": 92.24,
}

# Online/retrain accuracy at each checkpoint
DATA = {
    "l2":  [94.3, 94.1, 93.7, 93.7],
    "rf":  [91.5, 91.9, 92.5, 92.9],
    "lr":  [90.4, 91.0, 91.3, 91.7],
    "xgb": [89.3, 90.5, 91.0, 91.5],
}

# Full x and y arrays: x=200 (init) + post checkpoints
X_ALL = [200] + X_POST

SERIES = {
    "l2": {
        "label":   "L2 centroid (online update)",
        "color":   "#2563EB",
        "ls":      "-",
        "lw":      2.5,
        "marker":  "o",
        "ms":      7,
        "zorder":  5,
    },
    "rf": {
        "label":   "Random Forest (retrain/100)",
        "color":   "#059669",
        "ls":      "-",
        "lw":      1.5,
        "marker":  "s",
        "ms":      6,
        "zorder":  4,
    },
    "lr": {
        "label":   "Logistic Regression (retrain/100)",
        "color":   "#D97706",
        "ls":      "-",
        "lw":      1.5,
        "marker":  "^",
        "ms":      6,
        "zorder":  4,
    },
    "xgb": {
        "label":   "XGBoost (retrain/100)",
        "color":   "#DC2626",
        "ls":      "-",
        "lw":      1.5,
        "marker":  "D",
        "ms":      5,
        "zorder":  4,
    },
}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_online_figure() -> None:

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.titlesize": 13,
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.84)

    # -----------------------------------------------------------------------
    # Training period shading (x=0 to x=200)
    # -----------------------------------------------------------------------
    ax.axvspan(0, 200, color="#F1F5F9", zorder=1)
    ax.text(100, 88.25, "Training\nperiod",
            ha="center", va="bottom",
            fontsize=8.5, color="#94A3B8",
            style="italic", zorder=2)

    # -----------------------------------------------------------------------
    # Vertical line at x=200
    # -----------------------------------------------------------------------
    ax.axvline(x=200, color="#64748B", linewidth=1.1,
               linestyle="--", zorder=3)
    ax.text(202, 94.85,
            "Training data ends —\nonline learning begins",
            ha="left", va="top",
            fontsize=8, color="#475569",
            style="italic", zorder=4)

    # -----------------------------------------------------------------------
    # Plot each series
    # -----------------------------------------------------------------------
    for key, meta in SERIES.items():
        y_all = [INIT[key]] + DATA[key]
        ax.plot(X_ALL, y_all,
                color=meta["color"],
                linestyle=meta["ls"],
                linewidth=meta["lw"],
                marker=meta["marker"],
                markersize=meta["ms"],
                label=meta["label"],
                zorder=meta["zorder"])

    # -----------------------------------------------------------------------
    # Annotation at x=400: gap between L2 and XGBoost
    # -----------------------------------------------------------------------
    y_l2_400  = DATA["l2"][0]    # 94.3
    y_xgb_400 = DATA["xgb"][0]   # 89.3
    x_ann     = 400

    ax.annotate("",
                xy=(x_ann + 18, y_xgb_400),
                xytext=(x_ann + 18, y_l2_400),
                arrowprops=dict(arrowstyle="<->", color="#0F172A",
                                lw=1.2, connectionstyle="arc3,rad=0.0"),
                zorder=6)
    ax.text(x_ann + 22, (y_l2_400 + y_xgb_400) / 2,
            "5.0pp\nadvantage",
            ha="left", va="center",
            fontsize=8.5, color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.90),
            zorder=7)

    # -----------------------------------------------------------------------
    # Annotation at x=1500: converging gap
    # -----------------------------------------------------------------------
    y_l2_1500  = DATA["l2"][-1]    # 93.7
    y_xgb_1500 = DATA["xgb"][-1]   # 91.5
    x_ann2     = 1500

    ax.annotate("",
                xy=(x_ann2 + 18, y_xgb_1500),
                xytext=(x_ann2 + 18, y_l2_1500),
                arrowprops=dict(arrowstyle="<->", color="#0F172A",
                                lw=1.2, connectionstyle="arc3,rad=0.0"),
                zorder=6)
    ax.text(x_ann2 + 22, (y_l2_1500 + y_xgb_1500) / 2,
            "2.2pp",
            ha="left", va="center",
            fontsize=8.5, color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.90),
            zorder=7,
            clip_on=False)

    # -----------------------------------------------------------------------
    # Axis formatting
    # -----------------------------------------------------------------------
    ax.set_xlim(0, 1580)
    ax.set_ylim(88.0, 95.2)
    ax.set_xticks([200, 400, 600, 800, 1000, 1200, 1500])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Decisions", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)

    # -----------------------------------------------------------------------
    # Title + subtitle
    # -----------------------------------------------------------------------
    ax.set_title(
        "Online Learning: Centroid Update vs Periodic Batch Retraining",
        fontsize=13, pad=14
    )
    fig.text(
        0.10, 0.875,
        "L2 centroid updates per-decision at O(d) cost.  "
        "ML baselines retrain every 100 decisions.",
        ha="left", va="bottom",
        fontsize=9.5, color="#475569", style="italic",
        transform=fig.transFigure,
    )

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    ax.legend(fontsize=9.5, loc="lower right",
              frameon=True, framealpha=0.95, edgecolor="#E2E8F0")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    PAPER_FIGS.mkdir(exist_ok=True)
    for ext_fmt in ("pdf", "png"):
        out = PAPER_FIGS / f"fig17_online_learning.{ext_fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating fig17_online_learning ...")
    make_online_figure()
    print("Done.")
