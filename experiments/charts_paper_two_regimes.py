"""
Two-panel composite: architecture validation (left) vs deployment reality (right).

Left:  EXP-C1 kernel progression bar chart (hardcoded centroidal results).
Right: FX1-LEARNING realistic trajectory from results.json (10 seeds, combined mode).

Outputs (paper_figures/):
  fig6_two_regimes.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
FX1_DIR    = Path(__file__).resolve().parent / "fx1_learning"
PAPER_FIGS = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Left panel data — hardcoded EXP-C1 kernel progression
# ---------------------------------------------------------------------------

LEFT_LABELS = [
    "Random\nbaseline",
    "Dot-product\ncentroid",
    "Cosine\ncentroid",
    "L2 centroid\n(zero learning)",
    "L2 + warm\nstart learning",
]
LEFT_ACCS  = [25.0, 61.0, 96.4, 97.89, 98.2]
LEFT_COLORS = ["#636e72", "#d63031", "#a8c956", "#00b894", "#00755e"]

# ---------------------------------------------------------------------------
# Right panel data — from results.json
# ---------------------------------------------------------------------------

with open(FX1_DIR / "results.json") as fh:
    _res = json.load(fh)

STATIC_ACC   = _res["mean_static_acc"] * 100          # 71.455%
CHECKPOINTS  = sorted(int(k) for k in _res["checkpoint_stats"])
CP_MEANS     = [_res["checkpoint_stats"][str(c)]["mean_acc"] * 100
                for c in CHECKPOINTS]
CP_STDS      = [_res["checkpoint_stats"][str(c)]["std_acc"] * 100
                for c in CHECKPOINTS]

# Limit display to decisions 0–1000
MAX_DEC = 1000
X_RIGHT  = [0] + [c for c in CHECKPOINTS if c <= MAX_DEC]
Y_RIGHT  = [STATIC_ACC] + [m for c, m in zip(CHECKPOINTS, CP_MEANS)
                             if c <= MAX_DEC]
Y_STD    = [0.0] + [s for c, s in zip(CHECKPOINTS, CP_STDS)
                     if c <= MAX_DEC]

CENTROIDAL_IDEAL = 97.89   # shown as faint reference line


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_two_regimes() -> None:

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.titlesize": 12,
    })

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18,
                        top=0.84, wspace=0.38)

    # ===================================================================
    # LEFT PANEL — bar chart
    # ===================================================================
    x_l = np.arange(len(LEFT_LABELS))

    bars = ax_l.bar(x_l, LEFT_ACCS,
                    color=LEFT_COLORS,
                    width=0.55,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3)

    # Value labels on top of bars
    for bar, acc in zip(bars, LEFT_ACCS):
        ax_l.text(bar.get_x() + bar.get_width() / 2,
                  acc + 0.8,
                  f"{acc:.1f}%",
                  ha="center", va="bottom",
                  fontsize=9, color="#1E293B", zorder=4)

    # Delta annotations between bars
    DELTAS = [(61.0 - 25.0,   "dot", 0),
              (96.4 - 61.0,   "cos", 1),
              (97.89 - 96.4,  "L2",  2),
              (98.2 - 97.89,  "+lr", 3)]
    for delta, _, idx in DELTAS:
        sign = "+" if delta >= 0 else ""
        mid_x = (x_l[idx] + x_l[idx + 1]) / 2
        ax_l.text(mid_x, 12,
                  f"{sign}{delta:.1f}pp",
                  ha="center", va="bottom",
                  fontsize=7.5, color="#475569",
                  style="italic", zorder=4)

    ax_l.set_xticks(x_l)
    ax_l.set_xticklabels(LEFT_LABELS, fontsize=9, rotation=35, ha="right")
    ax_l.set_ylim(0, 105)
    ax_l.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax_l.tick_params(labelsize=9)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.set_ylabel("Accuracy (%)", fontsize=11)
    ax_l.set_title("Architecture Validation", fontsize=12, pad=4)
    ax_l.text(0.5, 1.01, "(centroidal synthetic)",
              transform=ax_l.transAxes,
              ha="center", va="bottom",
              fontsize=9, color="#475569", style="italic")

    # ===================================================================
    # RIGHT PANEL — line chart
    # ===================================================================

    # ±1 SD band
    y_lo = [max(50, y - s) for y, s in zip(Y_RIGHT, Y_STD)]
    y_hi = [min(100, y + s) for y, s in zip(Y_RIGHT, Y_STD)]
    ax_r.fill_between(X_RIGHT, y_lo, y_hi,
                      color="#2563EB", alpha=0.12, zorder=2)

    # Main trajectory line
    ax_r.plot(X_RIGHT, Y_RIGHT,
              color="#2563EB", linewidth=2.2,
              marker="o", markersize=6,
              label="L2 centroid (10 seeds, combined)",
              zorder=4)

    # Horizontal ref: warm-start baseline
    ax_r.axhline(y=STATIC_ACC, color="#D97706", linewidth=1.2,
                 linestyle="--", zorder=3)
    ax_r.text(MAX_DEC * 0.97, STATIC_ACC + 0.4,
              f"Day 1 warm start ({STATIC_ACC:.1f}%)",
              ha="right", va="bottom",
              fontsize=8, color="#D97706", style="italic")

    # Horizontal ref: centroidal ideal (faint)
    ax_r.axhline(y=CENTROIDAL_IDEAL, color="#94A3B8", linewidth=1.0,
                 linestyle=":", zorder=2)
    ax_r.text(MAX_DEC * 0.97, CENTROIDAL_IDEAL + 0.4,
              f"Centroidal ideal ({CENTROIDAL_IDEAL:.1f}%)",
              ha="right", va="bottom",
              fontsize=8, color="#94A3B8", style="italic")

    # Gap arrow at x=1000
    y_top_1000 = CENTROIDAL_IDEAL
    y_bot_1000 = Y_RIGHT[-1]   # acc at 1000
    ax_r.annotate("",
                  xy=(980, y_top_1000),
                  xytext=(980, y_bot_1000),
                  arrowprops=dict(arrowstyle="<->", color="#64748B",
                                  lw=1.1, connectionstyle="arc3,rad=0.0"),
                  zorder=5)
    ax_r.text(960, (y_top_1000 + y_bot_1000) / 2,
              f"{y_top_1000 - y_bot_1000:.1f}pp\ngap",
              ha="right", va="center",
              fontsize=7.5, color="#64748B",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor="#CBD5E1", alpha=0.88),
              zorder=6)

    ax_r.set_xlim(0, MAX_DEC + 20)
    ax_r.set_ylim(50, 100)
    ax_r.set_xticks([0, 200, 400, 600, 800, 1000])
    ax_r.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax_r.tick_params(labelsize=9)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.set_xlabel("Decisions", fontsize=11)
    ax_r.set_ylabel("Accuracy (%)", fontsize=11)
    ax_r.set_title("Deployment Reality", fontsize=12, pad=4)
    ax_r.text(0.5, 1.01, "(50-seed realistic simulation)",
              transform=ax_r.transAxes,
              ha="center", va="bottom",
              fontsize=9, color="#475569", style="italic")

    # ===================================================================
    # Between-panel annotation (in figure space, centered between axes)
    # ===================================================================
    # ax_l right edge ~0.07 + (0.97-0.07)*1.1/2.1 ≈ 0.54; ax_r left ≈ 0.59
    # midpoint ~0.565
    fig.text(
        0.535, 0.50,
        "Gap = realistic\nnoise floor,\nnot architectural\nlimitation.\n\n"
        "Both numbers\nare real.",
        ha="center", va="center",
        fontsize=8.5,
        color="#475569",
        style="italic",
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC",
                  edgecolor="#CBD5E1", alpha=0.92),
    )

    # ===================================================================
    # Suptitle
    # ===================================================================
    fig.suptitle(
        "Two Accuracy Regimes: Architecture Validation vs Deployment Reality",
        fontsize=13, y=0.97,
    )

    # ===================================================================
    # Save
    # ===================================================================
    PAPER_FIGS.mkdir(exist_ok=True)
    for ext_fmt in ("pdf", "png"):
        out = PAPER_FIGS / f"fig6_two_regimes.{ext_fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Right-panel data from results.json:")
    print(f"  Static warm-start: {STATIC_ACC:.2f}%")
    for x, y, s in zip(X_RIGHT[1:], Y_RIGHT[1:], Y_STD[1:]):
        print(f"  dec={x:4d}: {y:.2f}% ±{s:.2f}%")
    print()
    print("Generating fig6_two_regimes ...")
    make_two_regimes()
    print("Done.")
