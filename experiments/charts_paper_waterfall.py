"""
Waterfall bar chart: full 8-step equation progression, 25% → 98.2%.

All values are hardcoded from the validated experimental record.

Outputs (paper_figures/):
  fig1_waterfall_progression.{pdf,png}
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

STEPS = [
    # (label, accuracy, color)
    ("Random baseline",                 25.00, "#636e72"),
    ("Shared W Hebbian\n(EXP-A)",       49.3,  "#d63031"),
    ("Per-category W\n(EXP-A2)",        51.6,  "#e17055"),
    ("Dot-product centroid\n(EXP-C1)",  61.0,  "#e17055"),
    ("Cosine centroid\n(EXP-C1)",       96.4,  "#a8c956"),
    ("L2 centroid,\nzero learning\n(EXP-C1)", 97.89, "#00b894"),
    ("L2 + warm start\nlearning (EXP-B1)",    98.2,  "#00755e"),
]

LABELS   = [s[0] for s in STEPS]
ACCS     = [s[1] for s in STEPS]
COLORS   = [s[2] for s in STEPS]
N        = len(STEPS)
Y_POS    = list(range(N))    # 0 = bottom (random), 6 = top (best)

# Deltas between adjacent steps (shown as labels on bars)
DELTAS = [ACCS[i] - ACCS[i - 1] for i in range(1, N)]   # 6 values

# Phase separators: between indices 2-3 and 3-4 (0-based)
# i.e., between y=2 and y=3, and between y=3 and y=4
PHASE_SEPS = [2.5, 3.5]   # midpoints in Y space

# Phase label spans (y_center, label)
PHASE_LABELS = [
    (1.0,  "Phase A\nCapacity"),      # covers y=1,2
    (3.0,  "Phase B\nRepresentation"), # covers y=3
    (5.0,  "Phase C\nKernel"),         # covers y=4,5,6
]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_waterfall() -> None:

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.titlesize": 13,
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.26, right=0.88, bottom=0.12, top=0.84)

    # -----------------------------------------------------------------------
    # Horizontal bars
    # -----------------------------------------------------------------------
    bars = ax.barh(Y_POS, ACCS,
                   color=COLORS,
                   height=0.55,
                   edgecolor="white",
                   linewidth=0.8,
                   zorder=3)

    # -----------------------------------------------------------------------
    # Accuracy value labels at end of each bar
    # -----------------------------------------------------------------------
    for i, (bar, acc) in enumerate(zip(bars, ACCS)):
        ax.text(acc + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%",
                va="center", ha="left",
                fontsize=10,
                color="#1E293B",
                zorder=5)

    # -----------------------------------------------------------------------
    # Delta labels between adjacent bars (centered on bar, inset)
    # -----------------------------------------------------------------------
    for i, delta in enumerate(DELTAS):
        y_idx = i + 1          # which step this delta belongs to
        bar   = bars[y_idx]
        sign  = "+" if delta >= 0 else ""
        label = f"{sign}{delta:.1f}pp"
        # Place inside bar, near the left edge
        x_pos = max(ACCS[y_idx - 1] / 2, 1.5)   # midpoint of the increment
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha="center",
                fontsize=8.5,
                color="white" if ACCS[y_idx] > 30 else "#1E293B",
                fontweight="bold",
                zorder=5)

    # -----------------------------------------------------------------------
    # Phase separator dashed lines
    # -----------------------------------------------------------------------
    for y_sep in PHASE_SEPS:
        ax.axhline(y=y_sep, color="#94A3B8", linewidth=1.0,
                   linestyle="--", zorder=2)

    # -----------------------------------------------------------------------
    # Phase labels (to the right, past bar ends)
    # -----------------------------------------------------------------------
    X_PHASE = 101.5   # just past 100%
    PHASE_COLORS = {"A": "#d63031", "B": "#e17055", "C": "#00755e"}
    for y_ctr, plabel in PHASE_LABELS:
        letter = plabel.split()[1]
        color  = PHASE_COLORS.get(letter, "#475569")
        ax.text(X_PHASE, y_ctr, plabel,
                va="center", ha="left",
                fontsize=9,
                color=color,
                fontweight="bold",
                clip_on=False,
                zorder=5)

    # -----------------------------------------------------------------------
    # Double-headed arrow: step 3 (dot-product, 61.0%) → step 5 (L2, 97.89%)
    # Indices 3 and 5 in Y_POS
    # -----------------------------------------------------------------------
    arr_x   = 62.5          # just right of the 61.0 bar end
    y_bot   = Y_POS[3]      # index 3 → dot-product centroid
    y_top   = Y_POS[5]      # index 5 → L2 centroid zero learning
    mid_y   = (y_bot + y_top) / 2

    ax.annotate("",
                xy=(arr_x, y_top),
                xytext=(arr_x, y_bot),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="#0F172A",
                    lw=1.4,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=6)

    ax.text(arr_x + 1.0, mid_y,
            "+36.89pp\nkernel change only",
            va="center", ha="left",
            fontsize=9,
            color="#0F172A",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.90),
            zorder=7)

    # -----------------------------------------------------------------------
    # Axis formatting
    # -----------------------------------------------------------------------
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, N - 0.45)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.set_yticks(Y_POS)
    ax.set_yticklabels(LABELS, fontsize=10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------------
    # Title + subtitle
    # -----------------------------------------------------------------------
    ax.set_xlabel("Classification Accuracy", fontsize=12)
    ax.set_title(
        "The Equation Progression: From 25% to 98.2%",
        fontsize=13, pad=14, loc="left",
        x=0.0,
    )
    fig.text(
        0.26, 0.875,
        "Seven steps, three phases.  "
        "The kernel change accounts for 36.89 of the 73.2 total percentage points.",
        ha="left", va="bottom",
        fontsize=9.5,
        color="#475569",
        style="italic",
        transform=fig.transFigure,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    PAPER_FIGS.mkdir(exist_ok=True)
    for ext_fmt in ("pdf", "png"):
        out = PAPER_FIGS / f"fig1_waterfall_progression.{ext_fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating fig1_waterfall_progression ...")
    make_waterfall()
    print("Done.")
