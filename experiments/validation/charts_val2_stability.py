"""
VAL-2: Consolidated publication figure -- Push Update Stability.

Reads:
  experiments/validation/push_stability_results.csv
  experiments/validation/push_stability_summary.json

Outputs (paper_figures/):
  val_2_push_stability.{pdf,png}
"""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.viz.bridge_common import COLORS, VIZ_DEFAULTS, setup_axes, save_figure

VAL_DIR    = Path(__file__).resolve().parent
PAPER_FIGS = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Colors per spec (from bridge_common palette)
# ---------------------------------------------------------------------------
COLOR_A = COLORS["mi_static"]          # green   "#059669"
COLOR_B = COLORS["hebbian_undamped"]   # orange  "#D97706"
COLOR_C = COLORS["gt_noise_30"]        # red     "#DC2626"
COLOR_D = COLORS["gt_noise_5"]         # blue    "#2563EB"
COLOR_E = COLORS["gt_noise_15"]        # purple  "#7C3AED"

COND_META = {
    "A_normal":     {"color": COLOR_A, "ls": "-",  "lw": 2.0,
                     "legend": "A: Normal (70% correct)"},
    "B_bad_streak": {"color": COLOR_B, "ls": "-",  "lw": 2.0,
                     "legend": "B: Bad streak (first 100 incorrect)"},
    "C_worst_case": {"color": COLOR_C, "ls": "-",  "lw": 2.0,
                     "legend": "C: Worst case (0% correct, unclipped)"},
    "D_clipped":    {"color": COLOR_D, "ls": "-",  "lw": 3.0,   # THICK — the fix
                     "legend": "D: Clipped [0,1]  (0% correct)"},
    "E_margin":     {"color": COLOR_E, "ls": "--", "lw": 2.0,
                     "legend": "E: Margin guard  (0% correct)"},
}

# ---------------------------------------------------------------------------
# Load data: max norm across seeds per (condition, decision)
# ---------------------------------------------------------------------------

def _load_max_trajectories() -> dict[str, dict[int, float]]:
    """Return {condition: {decision: max_mu_norm_across_seeds}}."""
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(VAL_DIR / "push_stability_results.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            acc[row["condition"]][int(row["decision"])].append(float(row["mu_norm"]))
    result: dict[str, dict[int, float]] = {}
    for cond, dec_map in acc.items():
        result[cond] = {dec: float(np.max(vals)) for dec, vals in dec_map.items()}
    return result


def _load_summary() -> dict:
    with open(VAL_DIR / "push_stability_summary.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_stability_figure(traj: dict[str, dict[int, float]],
                          summary: dict) -> None:

    X_MAX = 200   # Clip all conditions to 0-200

    # -----------------------------------------------------------------------
    # Figure: wide with right margin for outside legend
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.09, right=0.70, bottom=0.18, top=0.88)

    # -----------------------------------------------------------------------
    # Draw condition lines (clipped to X_MAX)
    # -----------------------------------------------------------------------
    end_vals: dict[str, float] = {}   # record Y at last plotted decision

    for cond_key, meta in COND_META.items():
        if cond_key not in traj:
            continue
        dec_map  = traj[cond_key]
        decisions = sorted(d for d in dec_map if d <= X_MAX)
        norms     = [dec_map[d] for d in decisions]

        ax.semilogy(decisions, norms,
                    color=meta["color"],
                    linestyle=meta["ls"],
                    linewidth=meta["lw"],
                    label=meta["legend"],
                    zorder=4 if cond_key == "D_clipped" else 3)

        end_vals[cond_key] = norms[-1]

    # -----------------------------------------------------------------------
    # Horizontal reference lines
    # -----------------------------------------------------------------------
    ax.axhline(y=1.0, color="#94A3B8", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(X_MAX + 2, 1.0, "boundary [0,1]^6",
            va="center", ha="left",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
            color="#64748B",
            clip_on=False)

    ax.axhline(y=10.0, color="#CBD5E1", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(X_MAX + 2, 10.0, "10x warning",
            va="center", ha="left",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
            color="#64748B",
            clip_on=False)

    # -----------------------------------------------------------------------
    # Vertical escape-window marker at x=9
    # Set ylim first so the label y-position is within the visible area.
    # -----------------------------------------------------------------------
    ax.set_ylim(0.5, 2e5)
    ax.axvline(x=9, color="#94A3B8", linewidth=0.9, linestyle=":", zorder=2)
    ax.text(9, 0.55,
            "escape window\n(dec 6-12)",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 1.0,
            color="#64748B",
            zorder=5)

    # -----------------------------------------------------------------------
    # Annotation boxes at decision 200 for C, D, A
    # -----------------------------------------------------------------------
    bbox_style = dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.88)

    # -- C: 4,608x without clip --
    y_C = end_vals.get("C_worst_case", 4607.8)
    ax.annotate(
        "4,608x without clip",
        xy=(200, y_C),
        xytext=(155, y_C * 0.35),
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color=COLOR_C,
        fontweight="bold",
        bbox=bbox_style,
        arrowprops=dict(arrowstyle="->", color=COLOR_C, lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        zorder=6,
    )

    # -- D: 2.24x with clip --
    y_D = end_vals.get("D_clipped", 2.236)
    ax.annotate(
        "2.24x with clip",
        xy=(200, y_D),
        xytext=(148, y_D * 5.5),
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color=COLOR_D,
        fontweight="bold",
        bbox=bbox_style,
        arrowprops=dict(arrowstyle="->", color=COLOR_D, lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=6,
    )

    # -- A: Normal safe at 1.6x --
    # xytext placed between the 1x and 10x lines so the box is fully visible.
    y_A = end_vals.get("A_normal", 1.4)
    ax.annotate(
        "Normal: safe at 1.6x",
        xy=(200, y_A),
        xytext=(140, 4.5),
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color=COLOR_A,
        fontweight="bold",
        bbox=bbox_style,
        arrowprops=dict(arrowstyle="->", color=COLOR_A, lw=1.0,
                        connectionstyle="arc3,rad=0.30"),
        zorder=6,
    )

    # -----------------------------------------------------------------------
    # Axis limits and ticks
    # -----------------------------------------------------------------------
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: _fmt_norm(v))
    )

    # -----------------------------------------------------------------------
    # Legend outside right
    # -----------------------------------------------------------------------
    legend = ax.legend(
        fontsize=VIZ_DEFAULTS["tick_fontsize"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#E2E8F0",
    )

    # -----------------------------------------------------------------------
    # Axis labels + title
    # -----------------------------------------------------------------------
    setup_axes(ax,
               title="Push Update Stability: Five Conditions",
               xlabel="Decisions",
               ylabel="Centroid Max Norm (x)")

    # -----------------------------------------------------------------------
    # Caption below figure
    # -----------------------------------------------------------------------
    fig.text(
        0.395, 0.04,
        ("Normal operation (A) is safe at ~1.6x.  "
         "Adversarial (C) escapes [0,1] in 6 decisions -- 4,608x max norm.  "
         "One-line clip (D) prevents all escape."),
        ha="center", va="top",
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color="#475569",
        style="italic",
    )

    save_figure(fig, "val_2_push_stability", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _fmt_norm(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.0f}K x"
    if v >= 1:
        return f"{v:.0f} x"
    return f"{v:.1f} x"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading trajectories ...")
    traj = _load_max_trajectories()
    print("Loading summary ...")
    summary = _load_summary()
    print("Generating val_2_push_stability ...")
    make_stability_figure(traj, summary)
    print("Done.")
