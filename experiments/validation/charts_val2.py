"""
VAL-2: Push update stability figures.

Reads:
  experiments/validation/push_stability_results.csv
  experiments/validation/push_stability_summary.json

Outputs (paper_figures/):
  val2_norm_trajectories.{pdf,png}  -- all-condition norm trajectories (log scale)
  val2_fix_comparison.{pdf,png}     -- max norm per condition, fix annotations
"""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.viz.bridge_common import COLORS, VIZ_DEFAULTS, setup_axes, save_figure

VAL_DIR    = Path(__file__).resolve().parent
PAPER_FIGS = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Load CSV — mean norm trajectory per (condition, decision)
# ---------------------------------------------------------------------------
def _load_trajectories() -> dict[str, dict[int, float]]:
    """Return {condition_key: {decision: mean_norm_across_seeds}}."""
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(VAL_DIR / "push_stability_results.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            acc[row["condition"]][int(row["decision"])].append(float(row["mu_norm"]))
    result: dict[str, dict[int, float]] = {}
    for cond, dec_map in acc.items():
        result[cond] = {dec: float(np.mean(vals)) for dec, vals in dec_map.items()}
    return result


def _load_summary() -> dict:
    with open(VAL_DIR / "push_stability_summary.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Chart 1: All-condition norm trajectories
# ---------------------------------------------------------------------------
COND_STYLES = {
    "A_normal":     {"color": COLORS["gt_noise_0"],   "ls": "-",   "lw": 2.0, "label": "A: normal (70% correct)"},
    "B_bad_streak": {"color": COLORS["gt_noise_15"],  "ls": "--",  "lw": 2.0, "label": "B: bad streak (100 incorrect then 70%)"},
    "C_worst_case": {"color": COLORS["gt_noise_30"],  "ls": "-",   "lw": 2.5, "label": "C: worst case (100% incorrect, unclipped)"},
    "D_clipped":    {"color": COLORS["mi_static"],    "ls": "-.",  "lw": 2.0, "label": "D: clipped to [0,1]"},
    "E_margin":     {"color": COLORS["hebbian_undamped"], "ls": ":", "lw": 2.0, "label": "E: margin guard (||f−μ||<1)"},
}


def chart_norm_trajectories(traj: dict[str, dict[int, float]]) -> None:
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for cond_key, style in COND_STYLES.items():
        if cond_key not in traj:
            continue
        dec_map = traj[cond_key]
        decisions = sorted(dec_map.keys())
        norms     = [dec_map[d] for d in decisions]
        ax.semilogy(decisions, norms,
                    color=style["color"],
                    linestyle=style["ls"],
                    linewidth=style["lw"],
                    label=style["label"])

    # Reference line at norm=1
    ax.axhline(y=1.0, color="#CBD5E1", linewidth=0.8, linestyle=":")

    ax.set_ylim(bottom=0.5)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"] - 0.5, loc="upper left")

    setup_axes(ax,
               title="VAL-2: Centroid Norm Under Push Update (Eq. 4b) — 10 seeds",
               xlabel="Decision",
               ylabel="Mean Centroid Norm (log scale)")

    save_figure(fig, "val2_norm_trajectories", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Chart 2: Max norm per condition with fix annotation
# ---------------------------------------------------------------------------
def chart_fix_comparison(summary: dict) -> None:
    conditions = [
        ("A\nnormal",     "condition_A_normal",     COLORS["gt_noise_0"],        False),
        ("B\nbad streak", "condition_B_bad_streak",  COLORS["gt_noise_15"],       False),
        ("C\nworst case", "condition_C_worst_case",  COLORS["gt_noise_30"],       False),
        ("D\nclipped",    "condition_D_clipped",     COLORS["mi_static"],         True),
        ("E\nmargin",     "condition_E_margin",      COLORS["hebbian_undamped"],  False),
    ]

    labels  = [c[0] for c in conditions]
    max_norms = []
    in_bounds = []
    colors  = []

    for _, key, color, _ in conditions:
        data = summary[key]
        max_norms.append(data["max_norm_seen"])
        ib = data.get("all_dims_in_bounds", None)
        if ib is None:
            ib = data.get("max_dims_outside", 1) == 0
        in_bounds.append(ib)
        colors.append(color)

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    bars = ax.bar(labels, max_norms, color=colors, width=0.55)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)

    # Annotate each bar
    for bar, mn, ib in zip(bars, max_norms, in_bounds):
        x = bar.get_x() + bar.get_width() / 2
        if mn >= 1000:
            label = f"{mn:.0f}"
        elif mn >= 10:
            label = f"{mn:.1f}"
        else:
            label = f"{mn:.2f}"
        ax.text(x, mn * 1.6, label,
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"])
        status_txt = "✓ bounded" if ib else "✗ escapes"
        status_col = "#059669" if ib else "#DC2626"
        ax.text(x, 0.15, status_txt,
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
                color=status_col)

    ax.axhline(y=1.0, color="#CBD5E1", linewidth=0.8, linestyle=":")

    setup_axes(ax,
               title="VAL-2: Max Centroid Norm per Condition — Clip Prevents Escape",
               xlabel="Update Condition",
               ylabel="Max Norm Seen (log scale)")

    save_figure(fig, "val2_fix_comparison", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data ...")
    traj    = _load_trajectories()
    summary = _load_summary()

    print("Generating val2_norm_trajectories ...")
    chart_norm_trajectories(traj)

    print("Generating val2_fix_comparison ...")
    chart_fix_comparison(summary)

    print("Done.")
