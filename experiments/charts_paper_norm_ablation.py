"""
Normalization ablation bar chart: cross-graph discovery performance.

All values hardcoded from validated Exp 2 results (paper §4.2).

Outputs (paper_figures/):
  fig18_normalization_ablation.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
PAPER_FIGS = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CONDITIONS = [
    "None\n(raw)",
    "Z-score\nonly",
    "L2 unit\nnorm only",
    "Z-score\n+ L2",
]

MULTIPLIERS = [0.0, 0.0, 3.0, 111.0]    # "× above random"
PRECISION   = [0.000, 0.000, 0.001, 0.048]
RECALL      = [0.000, 0.000, 0.028, 0.147]
F1          = [0.000, 0.000, 0.002, 0.071]

COLORS = ["#d63031", "#d63031", "#e17055", "#00b894"]

# For log scale: zeros plotted at ZERO_PROXY
ZERO_PROXY = 0.5
PLOT_VALS  = [ZERO_PROXY if m == 0.0 else m for m in MULTIPLIERS]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_ablation_figure() -> None:

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.titlesize": 13,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.82)

    x = np.arange(len(CONDITIONS))

    bars = ax.bar(x, PLOT_VALS,
                  color=COLORS,
                  width=0.52,
                  edgecolor="white",
                  linewidth=0.8,
                  zorder=3)

    # -----------------------------------------------------------------------
    # Log scale + custom ticks
    # -----------------------------------------------------------------------
    ax.set_yscale("log")
    YTICKS = [0.5, 1, 3, 10, 100, 111]
    YLABELS = ["0\u00d7", "1\u00d7", "3\u00d7", "10\u00d7", "100\u00d7", "111\u00d7"]
    ax.set_yticks(YTICKS)
    ax.set_yticklabels(YLABELS, fontsize=10)
    ax.set_ylim(0.3, 280)
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    # -----------------------------------------------------------------------
    # Value labels on top of each bar
    # -----------------------------------------------------------------------
    TOP_LABELS = ["0\u00d7", "0\u00d7", "3\u00d7", "111\u00d7"]
    for i, (bar, label, mult) in enumerate(zip(bars, TOP_LABELS, MULTIPLIERS)):
        y_top = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_top * 1.12,
                label,
                ha="center", va="bottom",
                fontsize=11,
                fontweight="bold",
                color=COLORS[i],
                zorder=5)

    # -----------------------------------------------------------------------
    # Annotation for zero bars: "found nothing"
    # -----------------------------------------------------------------------
    for i in range(2):
        bar = bars[i]
        ax.text(bar.get_x() + bar.get_width() / 2,
                ZERO_PROXY * 0.72,
                "found\nnothing",
                ha="center", va="top",
                fontsize=7.5,
                color="white",
                style="italic",
                zorder=5)

    # -----------------------------------------------------------------------
    # PRF label below Z-score+L2 bar
    # -----------------------------------------------------------------------
    ax.text(x[3], 0.33,
            "(P=0.048, R=0.147, F1=0.071)",
            ha="center", va="top",
            fontsize=8,
            color="#00755e",
            style="italic",
            clip_on=False,
            zorder=5)

    # -----------------------------------------------------------------------
    # Bracket arrow from bar 2 (L2 only) to bar 3 (Z+L2)
    # Drawn as two vertical stubs + horizontal connector at a fixed y
    # -----------------------------------------------------------------------
    y_bracket = 18.0
    x2c = x[2] + 0.26      # right edge of bar 2
    x3c = x[3] - 0.26      # left edge of bar 3

    ax.annotate("",
                xy=(x3c, y_bracket),
                xytext=(x2c, y_bracket),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#0F172A",
                    lw=1.2,
                ),
                zorder=6)
    # Small downward ticks at each end
    for xp in (x2c, x3c):
        ax.plot([xp, xp], [y_bracket * 0.82, y_bracket],
                color="#0F172A", lw=1.2, zorder=6)

    ax.text((x2c + x3c) / 2, y_bracket * 1.08,
            "Both required —\nneither alone is sufficient",
            ha="center", va="bottom",
            fontsize=8.5,
            color="#0F172A",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.92),
            zorder=7)

    # -----------------------------------------------------------------------
    # Text box in upper area
    # -----------------------------------------------------------------------
    ax.text(0.50, 0.97,
            "Without normalization, cross-attention finds nothing.\n"
            "Z-score + L2 together: 111\u00d7 above random.",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5,
            color="#1E293B",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FDF4",
                      edgecolor="#86EFAC", alpha=0.92),
            zorder=7)

    # -----------------------------------------------------------------------
    # Axis formatting
    # -----------------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=10.5)
    ax.set_xlabel("Normalization Pipeline", fontsize=12)
    ax.set_ylabel("Discovery Performance (\u00d7 above random baseline)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax.set_title(
        "Normalization Ablation: Cross-Graph Discovery Requires Both Stages",
        fontsize=13, pad=10
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    PAPER_FIGS.mkdir(exist_ok=True)
    for ext_fmt in ("pdf", "png"):
        out = PAPER_FIGS / f"fig18_normalization_ablation.{ext_fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating fig18_normalization_ablation ...")
    make_ablation_figure()
    print("Done.")
