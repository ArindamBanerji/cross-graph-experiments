"""
charts.py — Publication-quality figures for FX-1-LEARNING.
experiments/fx1_learning/charts.py

2 charts:
  1. fx1_learning_trajectory      — overall learning curve vs static + references
  2. fx1_learning_per_category    — per-category learning trajectories (5 panels)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.viz.bridge_common import VIZ_DEFAULTS, save_figure
from experiments.fx1_proxy_real.realistic_generator import SOC_CATEGORIES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REF_STATIC_CENTROIDAL = 0.9181
REF_STATIC_REALISTIC  = 0.7145
REF_LEARNING_CENTROID = 0.9820
GATE_ACC              = 0.82

CAT_COLORS = {
    "travel_anomaly":       "#DC2626",   # red    — highest risk
    "credential_access":    "#D97706",   # amber  — high risk
    "threat_intel_match":   "#2563EB",   # blue
    "insider_behavioral":   "#7C3AED",   # purple
    "cloud_infrastructure": "#059669",   # green
}
CAT_LABELS = {
    "travel_anomaly":       "Travel Anomaly",
    "credential_access":    "Credential Access",
    "threat_intel_match":   "Threat Intel Match",
    "insider_behavioral":   "Insider Behavioral",
    "cloud_infrastructure": "Cloud Infrastructure",
}


# ---------------------------------------------------------------------------
# Chart 1 — Learning trajectory
# ---------------------------------------------------------------------------

def chart_learning_trajectory(
    checkpoint_stats: dict,
    mean_static:      float,
    n_decisions:      int,
    n_seeds:          int,
    rolling_win:      int,
    paper_dir:        str,
) -> None:
    """
    Line chart: overall rolling accuracy vs decision number.
    Reference lines: static realistic, centroidal static, EXP-B1 learning, gate.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    checkpoints = sorted(checkpoint_stats.keys())
    mean_accs   = [checkpoint_stats[cp]["mean_acc"] * 100 for cp in checkpoints]
    std_accs    = [checkpoint_stats[cp]["std_acc"]  * 100 for cp in checkpoints]
    mean_arr    = np.array(mean_accs)
    std_arr     = np.array(std_accs)

    # Learning curve
    ax.plot(checkpoints, mean_arr,
            color="#2563EB", linewidth=2.5, marker="o", markersize=6,
            zorder=5, label=f"Learning (warm start + oracle, rolling-{rolling_win})")
    ax.fill_between(checkpoints,
                    mean_arr - std_arr, mean_arr + std_arr,
                    alpha=0.15, color="#2563EB", zorder=4, label="±1 std")

    # Annotate decision 1000
    if 1000 in checkpoint_stats:
        acc_1000 = checkpoint_stats[1000]["mean_acc"] * 100
        ax.annotate(
            f"{acc_1000:.2f}%",
            xy=(1000, acc_1000),
            xytext=(1000 + 60, acc_1000 + 1.5),
            fontsize=9, fontweight="bold", color="#2563EB",
            arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2),
        )

    # Reference lines
    ax.axhline(mean_static * 100,
               color="#DC2626", linewidth=1.5, linestyle="--", alpha=0.85, zorder=3,
               label=f"Static realistic (FX-1-CORRECTED): {mean_static*100:.2f}%")
    ax.axhline(REF_STATIC_CENTROIDAL * 100,
               color="#94A3B8", linewidth=1.2, linestyle="--", alpha=0.7, zorder=3,
               label=f"Centroidal static (FX-1-CORRECTED): {REF_STATIC_CENTROIDAL*100:.2f}%")
    ax.axhline(REF_LEARNING_CENTROID * 100,
               color="#7C3AED", linewidth=1.2, linestyle="--", alpha=0.7, zorder=3,
               label=f"EXP-B1 centroidal learning: {REF_LEARNING_CENTROID*100:.2f}%")
    ax.axhline(GATE_ACC * 100,
               color="#059669", linewidth=1.8, linestyle=":", alpha=0.9, zorder=3,
               label=f"Gate threshold: {GATE_ACC*100:.0f}%")

    # Gate pass/fail shading
    ax.axhspan(GATE_ACC * 100, 105, alpha=0.04, color="#059669", zorder=1)

    ax.set_xlim(0, n_decisions + 50)
    all_vals = list(mean_arr) + [mean_static * 100, REF_STATIC_CENTROIDAL * 100,
                                  REF_LEARNING_CENTROID * 100, GATE_ACC * 100]
    ax.set_ylim(max(0, min(all_vals) - 4), min(105, max(all_vals) + 4))

    ax.set_xlabel("Decision Number",   fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Rolling Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        f"FX-1-LEARNING: Learning Lift Under Realistic Distributions\n"
        f"Combined mode, N={n_decisions}, {n_seeds} seeds, τ=0.1, "
        f"rolling-{rolling_win} accuracy",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])

    fig.tight_layout()
    save_figure(fig, "fx1_learning_trajectory", paper_dir)


# ---------------------------------------------------------------------------
# Chart 2 — Per-category learning trajectories
# ---------------------------------------------------------------------------

def chart_learning_per_category(
    per_cat_cp_mean:    dict,
    per_cat_static_mean: dict,
    n_decisions:        int,
    rolling_win:        int,
    paper_dir:          str,
) -> None:
    """
    5 subplots (one per SOC category): learning trajectory + static reference.
    travel_anomaly and credential_access highlighted as high-risk categories.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Layout: 5 categories in 2×3 grid (last slot empty)
    axes_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    high_risk = {"travel_anomaly", "credential_access"}

    for idx, cat in enumerate(SOC_CATEGORIES):
        row, col = axes_positions[idx]
        ax = fig.add_subplot(gs[row, col])

        checkpoints = sorted(per_cat_cp_mean[cat].keys())
        accs        = [per_cat_cp_mean[cat][cp] * 100
                       if not np.isnan(per_cat_cp_mean[cat][cp]) else np.nan
                       for cp in checkpoints]

        color = CAT_COLORS[cat]
        lw    = 2.5 if cat in high_risk else 1.8

        # Mask NaN for plotting
        valid_mask = [not np.isnan(a) for a in accs]
        valid_cps  = [c for c, m in zip(checkpoints, valid_mask) if m]
        valid_accs = [a for a, m in zip(accs, valid_mask) if m]

        if valid_cps:
            ax.plot(valid_cps, valid_accs,
                    color=color, linewidth=lw, marker="o", markersize=5,
                    zorder=4, label="Learning")

        # Static reference
        static_acc = per_cat_static_mean[cat] * 100
        ax.axhline(static_acc,
                   color=color, linewidth=1.2, linestyle="--", alpha=0.55, zorder=3,
                   label=f"Static: {static_acc:.1f}%")

        # Gate line
        ax.axhline(GATE_ACC * 100,
                   color="#059669", linewidth=1.2, linestyle=":", alpha=0.7, zorder=3)

        ax.set_xlim(0, n_decisions + 50)
        all_vals_ax = valid_accs + [static_acc, GATE_ACC * 100]
        y_min = max(0,   min(all_vals_ax) - 5)
        y_max = min(105, max(all_vals_ax) + 5)
        ax.set_ylim(y_min, y_max)

        risk_tag = " ⚠" if cat in high_risk else ""
        ax.set_title(
            f"{CAT_LABELS[cat]}{risk_tag}",
            fontsize=VIZ_DEFAULTS["title_fontsize"],
            color=color if cat in high_risk else "#111827",
            fontweight="bold" if cat in high_risk else "normal",
        )
        ax.set_xlabel("Decision", fontsize=7.5)
        ax.set_ylabel("Rolling Acc (%)", fontsize=7.5)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(linestyle="--", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    # Hide unused 6th subplot
    fig.add_subplot(gs[1, 2]).set_visible(False)

    fig.suptitle(
        f"FX-1-LEARNING: Per-Category Learning Trajectories (Combined Mode)\n"
        f"Rolling-{rolling_win} accuracy. ⚠ = high-risk category. "
        f"Dashed = static baseline. Green dotted = 82% gate.",
        fontsize=VIZ_DEFAULTS["title_fontsize"] + 0.5,
        y=1.01,
    )
    save_figure(fig, "fx1_learning_per_category", paper_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(
    checkpoint_stats:    dict,
    mean_static:         float,
    per_cat_cp_mean:     dict,
    per_cat_static_mean: dict,
    n_decisions:         int,
    n_seeds:             int,
    rolling_win:         int,
    paper_dir:           str,
) -> None:
    """Generate all FX-1-LEARNING charts and save to paper_dir (PDF + PNG)."""
    print("\nGenerating FX-1-LEARNING charts...")
    chart_learning_trajectory(
        checkpoint_stats=checkpoint_stats,
        mean_static=mean_static,
        n_decisions=n_decisions,
        n_seeds=n_seeds,
        rolling_win=rolling_win,
        paper_dir=paper_dir,
    )
    chart_learning_per_category(
        per_cat_cp_mean=per_cat_cp_mean,
        per_cat_static_mean=per_cat_static_mean,
        n_decisions=n_decisions,
        rolling_win=rolling_win,
        paper_dir=paper_dir,
    )
    print("All 2 charts saved (PDF + PNG).")
