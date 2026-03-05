"""
Publication-quality charts for EXP-D1: Cross-Category Transfer.

Generates
---------
paper_figures/expD1_transfer_matrix.{pdf,png}     Chart 1 — 5×5 transfer similarity heatmap
paper_figures/expD1_convergence.{pdf,png}          Chart 2 — convergence subplots per target
paper_figures/expD1_speedup.{pdf,png}              Chart 3 — decisions to 90% accuracy
paper_figures/expD1_delta_summary.{pdf,png}        Chart 4 — transfer lift vs config and cold
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.viz.bridge_common import VIZ_DEFAULTS, setup_axes, save_figure


# ---------------------------------------------------------------------------
# Display spec
# ---------------------------------------------------------------------------

COND_COLORS = {
    "cold":     "#94A3B8",  # gray — cold start
    "config":   "#1E3A5F",  # dark blue — warm config
    "transfer": "#059669",  # green — cross-category transfer
}
COND_LABELS = {
    "cold":     "Cold (uniform 0.5)",
    "config":   "Config (warm start)",
    "transfer": "Transfer (cross-cat)",
}
COND_STYLES = {
    "cold":     "--",
    "config":   "-",
    "transfer": "-",
}

CAT_SHORT = [
    "Cred\nAccess",
    "Threat\nIntel",
    "Lateral\nMove",
    "Data\nExfil",
    "Insider\nThreat",
]

CONDITIONS = ["cold", "config", "transfer"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _grouped_stats(df: pd.DataFrame, col: str, group_by: str = "checkpoint"):
    grp   = df.groupby(group_by)[col]
    means = grp.mean()
    stds  = grp.std().fillna(0.0)
    xs    = sorted(means.index.tolist())
    return xs, [float(means[x]) for x in xs], [float(stds[x]) for x in xs]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    """
    Read CSV/JSON from *results_dir* and produce all 4 EXP-D1 charts.
    """
    rdir = Path(results_dir)

    df       = pd.read_csv(rdir / "accuracy_trajectories.csv")
    speed_df = pd.read_csv(rdir / "convergence_speed.csv")
    tm_df    = pd.read_csv(rdir / "transfer_matrix.csv", index_col=0)

    with open(rdir / "summary.json") as fh:
        summary = json.load(fh)

    cfg_path   = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    categories = raw["bridge_common"]["categories"]

    figures_dir = str(ROOT / "paper_figures")

    _chart1_transfer_matrix(tm_df, categories, figures_dir)
    _chart2_convergence(df, categories, figures_dir)
    _chart3_speedup(speed_df, categories, figures_dir)
    _chart4_delta_summary(summary, categories, figures_dir)


# ---------------------------------------------------------------------------
# Chart 1: Transfer similarity heatmap (5×5)
# ---------------------------------------------------------------------------

def _chart1_transfer_matrix(tm_df: pd.DataFrame, categories: list, figures_dir: str) -> None:
    data = tm_df.values.astype(float)   # (5, 5), diagonal = 0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gray out diagonal, use Blues for off-diagonal
    display = data.copy()
    im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Cosine Similarity")

    n = len(categories)
    for i in range(n):
        for j in range(n):
            if i == j:
                # Gray diagonal patch
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#E5E7EB", edgecolor="none", zorder=2,
                ))
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                        color="#9CA3AF", zorder=3)
            else:
                text_color = "white" if data[i, j] > 0.7 else "black"
                ax.text(j, i, f"{data[i, j]:.3f}",
                        ha="center", va="center",
                        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                        color=text_color, zorder=3)

    short_names = [c.replace("_", "\n") for c in categories]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_yticklabels(short_names, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_xlabel("Target Category",  fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Source Category",  fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("EXP-D1: Cross-Category Transfer Similarity",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    fig.tight_layout()
    save_figure(fig, "expD1_transfer_matrix", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Convergence subplots (2×3 grid, 5 targets)
# ---------------------------------------------------------------------------

def _chart2_convergence(df: pd.DataFrame, categories: list, figures_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes_flat = axes.flatten()

    for idx, (cat, cat_short) in enumerate(zip(categories, CAT_SHORT)):
        ax = axes_flat[idx]

        for cond in CONDITIONS:
            sub = df[(df["target_category"] == cat) & (df["condition"] == cond)]
            if sub.empty:
                continue
            xs, ys, es = _grouped_stats(sub, "cumulative_gt_accuracy")
            color = COND_COLORS[cond]
            ls    = COND_STYLES[cond]
            ax.plot(xs, ys, color=color, linewidth=2.0,
                    linestyle=ls, label=COND_LABELS[cond], zorder=3)
            ax.fill_between(
                xs,
                [max(0.0, y - e) for y, e in zip(ys, es)],
                [y + e for y, e in zip(ys, es)],
                color=color, alpha=0.12, zorder=2,
            )

        ax.axhline(0.98, color="#6B7280", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="EXP-B1 warm ref (98%)", zorder=1)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(cat_short.replace("\n", " "), fontsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"] - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx >= 3:
            ax.set_xlabel("Decisions", fontsize=VIZ_DEFAULTS["tick_fontsize"])
        if idx % 3 == 0:
            ax.set_ylabel("Cumul. GT Acc.", fontsize=VIZ_DEFAULTS["tick_fontsize"])
        if idx == 0:
            ax.legend(fontsize=max(6, VIZ_DEFAULTS["tick_fontsize"] - 1), loc="lower right")

    # Hide the unused 6th subplot
    axes_flat[5].set_visible(False)

    fig.suptitle("EXP-D1: Convergence by Initialization Strategy",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    fig.tight_layout()
    save_figure(fig, "expD1_convergence", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Decisions to 90% — grouped bars
# ---------------------------------------------------------------------------

def _chart3_speedup(speed_df: pd.DataFrame, categories: list, figures_dir: str) -> None:
    n_cats  = len(categories)
    n_conds = len(CONDITIONS)
    bar_w   = 0.24
    x       = np.arange(n_cats)
    cap     = 200    # sentinel for "not reached"

    fig, ax = plt.subplots(figsize=(12, 5))

    for j, cond in enumerate(CONDITIONS):
        means, errs, hatches_list = [], [], []
        for cat in categories:
            sub = speed_df[
                (speed_df["target_category"] == cat) &
                (speed_df["condition"]        == cond)
            ]["decisions_to_90pct"]
            # Replace NaN with cap
            vals = sub.fillna(cap).values.astype(float)
            means.append(float(vals.mean()))
            errs.append(float(vals.std()))
            hatches_list.append("//" if vals.mean() >= cap - 0.5 else "")

        offsets = x + (j - n_conds / 2 + 0.5) * bar_w
        for xi, (mean, err, hatch) in enumerate(zip(means, errs, hatches_list)):
            ax.bar(
                offsets[xi], mean, bar_w,
                color=COND_COLORS[cond],
                alpha=0.85,
                hatch=hatch,
                edgecolor="none",
                error_kw={"linewidth": 1.0},
            )
        ax.bar(  # invisible bar for legend
            -1, 0, bar_w,
            color=COND_COLORS[cond],
            alpha=0.85,
            label=COND_LABELS[cond],
        )

    ax.axhline(cap, color="#9CA3AF", linestyle=":", linewidth=1.0,
               alpha=0.7, label=f"Max ({cap} = not reached)", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_SHORT, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0, cap * 1.15)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax, "EXP-D1: Decisions to 90% Accuracy",
               "Target Category", "Decisions (hatched = not reached)")
    save_figure(fig, "expD1_speedup", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Delta summary (transfer vs config, transfer vs cold)
# ---------------------------------------------------------------------------

def _chart4_delta_summary(summary: dict, categories: list, figures_dir: str) -> None:
    vs_config = [summary["transfer_vs_config_delta"][cat] for cat in categories]
    vs_cold   = [summary["transfer_vs_cold_delta"][cat]   for cat in categories]

    x       = np.arange(len(categories))
    bar_w   = 0.32

    fig, ax = plt.subplots(figsize=(11, 5))

    bars_config = ax.bar(
        x - bar_w / 2, vs_config, bar_w,
        color="#059669", alpha=0.85, label="Transfer vs Config",
    )
    bars_cold = ax.bar(
        x + bar_w / 2, vs_cold, bar_w,
        color="#D97706", alpha=0.85, label="Transfer vs Cold",
    )

    # Annotate
    for bar, val in zip(bars_config, vs_config):
        ypos = bar.get_height() + (0.005 if val >= 0 else -0.02)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1%}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    for bar, val in zip(bars_cold, vs_cold):
        ypos = bar.get_height() + (0.005 if val >= 0 else -0.02)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.1%}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"])

    ax.axhline(0.0, color="#374151", linewidth=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_SHORT, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax, "EXP-D1: Transfer Lift vs Config and Cold Start",
               "Target Category", "Accuracy Delta at t=200")
    save_figure(fig, "expD1_delta_summary", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expD1_cross_category_transfer/results/")
    print("Charts saved to paper_figures/expD1_*.{png,pdf}")
