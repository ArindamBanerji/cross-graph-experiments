"""
Publication-quality charts for EXP-E1: Kernel Generalization.

Generates
---------
paper_figures/expE1_kernel_x_distribution.{pdf,png}   Chart 1 — grouped bar
paper_figures/expE1_kernel_ranking.{pdf,png}           Chart 2 — 3 subplots, per-distribution ranking
paper_figures/expE1_dot_vs_l2.{pdf,png}                Chart 3 — dot vs L2, does normalization help?
paper_figures/expE1_mixed_scale_impact.{pdf,png}       Chart 4 — scale sensitivity per kernel
paper_figures/expE1_mahalanobis_vs_l2.{pdf,png}        Chart 5 — L2 vs Mahalanobis
paper_figures/expE1_learning_curves.{pdf,png}          Chart 6 — best vs worst kernel learning
paper_figures/expE1_per_category_heatmap.{pdf,png}     Chart 7 — 12-row heatmap
paper_figures/expE1_gae_recommendation.{pdf,png}       Chart 8 — recommendation summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.viz.bridge_common import VIZ_DEFAULTS, setup_axes, save_figure


# ---------------------------------------------------------------------------
# Color / label constants
# ---------------------------------------------------------------------------

KERNEL_COLORS = {
    "l2":          "#1E3A5F",   # dark blue
    "cosine":      "#059669",   # green
    "dot":         "#94A3B8",   # gray
    "mahalanobis": "#7C3AED",   # purple
}
KERNEL_LABELS = {
    "l2":          "L2 (neg-sq)",
    "cosine":      "Cosine",
    "dot":         "Dot product",
    "mahalanobis": "Mahalanobis",
}
KERNELS = ["l2", "cosine", "dot", "mahalanobis"]

DIST_LABELS = {
    "original":    "Original\n[0,1] range",
    "normalized":  "Normalized\n(zero-mean, unit-var)",
    "mixed_scale": "Mixed Scale\n([0,1] + [0,100] + [0,0.01])",
}
DISTRIBUTIONS = ["original", "normalized", "mixed_scale"]

CATEGORY_SHORT = [
    "Credential\nAccess",
    "Threat Intel\nMatch",
    "Lateral\nMovement",
    "Data\nExfil",
    "Insider\nThreat",
]
CAT_KEYS = ["acc_credential", "acc_threat", "acc_lateral", "acc_exfil", "acc_insider"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    rdir = Path(results_dir)

    with open(rdir / "summary.json") as fh:
        summary = json.load(fh)

    df1 = pd.read_csv(rdir / "phase1_oracle.csv")

    learning_path = rdir / "phase2_learning.csv"
    df2 = pd.read_csv(learning_path) if learning_path.exists() and learning_path.stat().st_size > 0 else pd.DataFrame()

    figures_dir = str(ROOT / "paper_figures")

    _chart1_kernel_x_distribution(df1, summary, figures_dir)
    _chart2_kernel_ranking(summary, figures_dir)
    _chart3_dot_vs_l2(summary, figures_dir)
    _chart4_mixed_scale_impact(summary, figures_dir)
    _chart5_mahalanobis_vs_l2(summary, figures_dir)
    _chart6_learning_curves(df2, summary, figures_dir)
    _chart7_per_category_heatmap(df1, figures_dir)
    _chart8_gae_recommendation(summary, figures_dir)


# ---------------------------------------------------------------------------
# Chart 1: Grouped bar — 3 groups (distributions) × 4 bars (kernels)
# ---------------------------------------------------------------------------

def _chart1_kernel_x_distribution(df: pd.DataFrame, summary: dict, figures_dir: str) -> None:
    n_dist   = len(DISTRIBUTIONS)
    n_kernel = len(KERNELS)
    bar_width = 0.18
    x = np.arange(n_dist)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for j, kernel in enumerate(KERNELS):
        means = []
        stds  = []
        for dist in DISTRIBUTIONS:
            sub = df[(df["distribution"] == dist) & (df["kernel"] == kernel)]
            means.append(sub["overall_accuracy"].mean())
            stds.append(sub["overall_accuracy"].std())

        offset = x + (j - n_kernel / 2 + 0.5) * bar_width
        bars = ax.bar(
            offset, means, bar_width,
            color=KERNEL_COLORS[kernel], alpha=0.85,
            yerr=stds, capsize=3,
            error_kw={"linewidth": 1.0},
            label=KERNEL_LABELS[kernel],
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) + 0.01,
                f"{mean:.0%}",
                ha="center", va="bottom",
                fontsize=max(6, VIZ_DEFAULTS["annotation_fontsize"] - 1),
                rotation=90,
            )

    ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.2,
               alpha=0.9, label="Random (25%)", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS], fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=2, loc="upper right")
    setup_axes(
        ax,
        "EXP-E1: Kernel × Distribution — Centroid Oracle Accuracy",
        "Factor Distribution",
        "Mean Accuracy (10 seeds × 10k alerts)",
    )
    save_figure(fig, "expE1_kernel_x_distribution", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Horizontal bar ranking — 3 subplots (one per distribution)
# ---------------------------------------------------------------------------

def _chart2_kernel_ranking(summary: dict, figures_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, dist in zip(axes, DISTRIBUTIONS):
        ranked = summary["phase1"][dist]["kernel_ranking"]
        means  = [summary["phase1"][dist][k]["mean_accuracy"] for k in ranked]
        best_k = ranked[0]

        colors = [KERNEL_COLORS[k] for k in ranked]
        y = np.arange(len(ranked))

        bars = ax.barh(y, means, 0.55, color=colors, alpha=0.85)

        # Highlight best kernel with bold edge
        bars[0].set_edgecolor("#111827")
        bars[0].set_linewidth(2.0)

        for bar, mean, kernel in zip(bars, means, ranked):
            ax.text(
                mean + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1%}",
                va="center", ha="left",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                fontweight="bold" if kernel == best_k else "normal",
            )

        ax.set_yticks(y)
        ax.set_yticklabels([KERNEL_LABELS[k] for k in ranked],
                           fontsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.set_xlim(0.0, 1.12)
        ax.axvline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.set_title(DIST_LABELS[dist].replace("\n", " "),
                     fontsize=VIZ_DEFAULTS["title_fontsize"] - 1)
        ax.set_xlabel("Accuracy", fontsize=VIZ_DEFAULTS["label_fontsize"])

    fig.suptitle(
        "EXP-E1: Kernel Rankings by Distribution",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "expE1_kernel_ranking", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Dot product vs L2 — does normalization close the gap?
# ---------------------------------------------------------------------------

def _chart3_dot_vs_l2(summary: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    dot_means = [summary["phase1"][d]["dot"]["mean_accuracy"]    for d in DISTRIBUTIONS]
    l2_means  = [summary["phase1"][d]["l2"]["mean_accuracy"]     for d in DISTRIBUTIONS]
    cos_means = [summary["phase1"][d]["cosine"]["mean_accuracy"] for d in DISTRIBUTIONS]

    x = np.arange(len(DISTRIBUTIONS))
    bar_width = 0.24

    bars_l2  = ax.bar(x - bar_width, l2_means,  bar_width, color=KERNEL_COLORS["l2"],
                      alpha=0.85, label=KERNEL_LABELS["l2"])
    bars_cos = ax.bar(x,             cos_means, bar_width, color=KERNEL_COLORS["cosine"],
                      alpha=0.85, label=KERNEL_LABELS["cosine"])
    bars_dot = ax.bar(x + bar_width, dot_means, bar_width, color=KERNEL_COLORS["dot"],
                      alpha=0.85, label=KERNEL_LABELS["dot"])

    # Connect L2 and dot bars with lines and annotate gap
    for i, (l2, dot) in enumerate(zip(l2_means, dot_means)):
        gap_pp = (l2 - dot) * 100
        ax.plot(
            [x[i] - bar_width / 2, x[i] + bar_width * 1.5],
            [l2, dot],
            color="#374151", linewidth=1.0, linestyle="--", alpha=0.6,
        )
        mid_y = (l2 + dot) / 2
        ax.text(
            x[i] + bar_width * 0.5, mid_y,
            f"gap\n{gap_pp:+.0f}pp",
            ha="left", va="center",
            fontsize=max(6, VIZ_DEFAULTS["annotation_fontsize"] - 1),
            color="#374151",
        )

    for bar, val in zip(list(bars_l2) + list(bars_dot), l2_means + dot_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.0%}",
            ha="center", va="bottom",
            fontsize=max(6, VIZ_DEFAULTS["annotation_fontsize"] - 1),
        )

    ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0, alpha=0.9, label="Random (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS], fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(
        ax,
        "EXP-E1: Dot Product vs L2 — Does Normalization Help?",
        "Factor Distribution",
        "Mean Accuracy",
    )
    save_figure(fig, "expE1_dot_vs_l2", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Scale sensitivity — which kernels handle mixed scales?
# ---------------------------------------------------------------------------

def _chart4_mixed_scale_impact(summary: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    n_kernel = len(KERNELS)
    n_dist   = len(DISTRIBUTIONS)
    bar_width = 0.22
    x = np.arange(n_kernel)

    dist_alphas = [0.95, 0.65, 0.40]   # original, normalized, mixed_scale

    for di, dist in enumerate(DISTRIBUTIONS):
        means = [summary["phase1"][dist][k]["mean_accuracy"] for k in KERNELS]
        offset = x + (di - n_dist / 2 + 0.5) * bar_width
        bars = ax.bar(
            offset, means, bar_width,
            color=[KERNEL_COLORS[k] for k in KERNELS],
            alpha=dist_alphas[di],
            label=DIST_LABELS[dist].replace("\n", " "),
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{mean:.0%}",
                ha="center", va="bottom",
                fontsize=max(5, VIZ_DEFAULTS["annotation_fontsize"] - 2),
            )

    ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0, alpha=0.9, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_LABELS[k] for k in KERNELS], fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.12)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(
        ax,
        "EXP-E1: Scale Sensitivity — Which Kernels Handle Mixed Scales?",
        "Kernel",
        "Mean Accuracy",
    )
    save_figure(fig, "expE1_mixed_scale_impact", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 5: L2 vs Mahalanobis — does cluster shape matter?
# ---------------------------------------------------------------------------

def _chart5_mahalanobis_vs_l2(summary: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    l2_means   = [summary["phase1"][d]["l2"]["mean_accuracy"]          for d in DISTRIBUTIONS]
    maha_means = [summary["phase1"][d]["mahalanobis"]["mean_accuracy"] for d in DISTRIBUTIONS]

    x = np.arange(len(DISTRIBUTIONS))
    bar_width = 0.3

    bars_l2   = ax.bar(x - bar_width / 2, l2_means,   bar_width,
                       color=KERNEL_COLORS["l2"],          alpha=0.85, label=KERNEL_LABELS["l2"])
    bars_maha = ax.bar(x + bar_width / 2, maha_means, bar_width,
                       color=KERNEL_COLORS["mahalanobis"], alpha=0.85, label=KERNEL_LABELS["mahalanobis"])

    for bars, means in [(bars_l2, l2_means), (bars_maha, maha_means)]:
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{mean:.1%}",
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            )

    # Annotate delta
    for i, (l2, maha) in enumerate(zip(l2_means, maha_means)):
        delta_pp = (maha - l2) * 100
        y_ann = max(l2, maha) + 0.06
        ax.annotate(
            f"Δ={delta_pp:+.1f}pp",
            xy=(x[i], y_ann),
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS], fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(
        ax,
        "EXP-E1: L2 vs Mahalanobis — Does Cluster Shape Matter?",
        "Factor Distribution",
        "Mean Accuracy",
    )
    save_figure(fig, "expE1_mahalanobis_vs_l2", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 6: Learning curves — best vs worst kernel per distribution
# ---------------------------------------------------------------------------

def _chart6_learning_curves(df2: pd.DataFrame, summary: dict, figures_dir: str) -> None:
    if df2.empty:
        # Generate placeholder chart
        fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])
        ax.text(0.5, 0.5, "No Phase 2 data", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        setup_axes(ax, "EXP-E1: Learning Curves — No Data", "Checkpoint", "Cumulative Accuracy")
        save_figure(fig, "expE1_learning_curves", output_dir=str(ROOT / "paper_figures"))
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, dist in zip(axes, DISTRIBUTIONS):
        dist_data = df2[df2["distribution"] == dist]
        if dist_data.empty:
            ax.set_title(DIST_LABELS[dist].replace("\n", " "))
            continue

        best_k  = summary["phase1"][dist]["best_kernel"]
        worst_k = summary["phase1"][dist]["worst_kernel"]

        for kernel, rank in [(best_k, "best"), (worst_k, "worst")]:
            sub = dist_data[dist_data["kernel"] == kernel]
            if sub.empty:
                continue
            checkpoints = sorted(sub["checkpoint"].unique())
            means = [sub[sub["checkpoint"] == cp]["cumulative_gt_accuracy"].mean() for cp in checkpoints]
            stds  = [sub[sub["checkpoint"] == cp]["cumulative_gt_accuracy"].std()  for cp in checkpoints]
            means = np.array(means)
            stds  = np.array(stds)

            color     = KERNEL_COLORS[kernel]
            linestyle = "-" if rank == "best" else "--"
            label     = f"{KERNEL_LABELS[kernel]} ({rank})"

            ax.plot(checkpoints, means, color=color, linestyle=linestyle,
                    linewidth=2.0, marker="o", markersize=4, label=label)
            ax.fill_between(checkpoints, means - stds, means + stds,
                            color=color, alpha=0.15)

        ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Decisions (t)", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.set_title(DIST_LABELS[dist].replace("\n", " "),
                     fontsize=VIZ_DEFAULTS["title_fontsize"] - 1)
        ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=max(7, VIZ_DEFAULTS["tick_fontsize"] - 1))

    axes[0].set_ylabel("Cumulative Accuracy", fontsize=VIZ_DEFAULTS["label_fontsize"])

    fig.suptitle(
        "EXP-E1: Learning Curves — Best vs Worst Kernel per Distribution",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "expE1_learning_curves", output_dir=str(ROOT / "paper_figures"))


# ---------------------------------------------------------------------------
# Chart 7: Per-category heatmap — 12 rows × 5 columns
# ---------------------------------------------------------------------------

def _chart7_per_category_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    # Build matrix: row = (distribution, kernel), col = category
    row_labels = []
    data_rows  = []

    for dist in DISTRIBUTIONS:
        for kernel in KERNELS:
            sub = df[(df["distribution"] == dist) & (df["kernel"] == kernel)]
            if sub.empty:
                continue
            row_labels.append(f"{dist[:4]}  {KERNEL_LABELS[kernel]}")
            row_vals = [sub[k].mean() for k in CAT_KEYS]
            data_rows.append(row_vals)

    mat = np.array(data_rows, dtype=np.float64)   # (12, 5)
    n_rows, n_cols = mat.shape

    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.55 + 1.2)))

    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            text_color = "white" if mat[i, j] > 0.60 else "black"
            ax.text(j, i, f"{mat[i, j]:.0%}",
                    ha="center", va="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                    color=text_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(CATEGORY_SHORT, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=max(6, VIZ_DEFAULTS["tick_fontsize"] - 1))

    # Draw horizontal separators between distributions (every 4 rows)
    for sep in [4, 8]:
        if sep < n_rows:
            ax.axhline(sep - 0.5, color="#374151", linewidth=1.2, alpha=0.6)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Accuracy")
    setup_axes(ax,
               "EXP-E1: Per-Category Accuracy (All Kernel × Distribution Combinations)",
               "Alert Category", "Distribution / Kernel")
    fig.tight_layout()
    save_figure(fig, "expE1_per_category_heatmap", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 8: GAE recommendation summary
# ---------------------------------------------------------------------------

def _chart8_gae_recommendation(summary: dict, figures_dir: str) -> None:
    guidance = summary.get("gae_guidance", "unknown")
    recommendations = summary.get("recommendations", {})

    fig, ax = plt.subplots(figsize=(10, 5))

    # Find the recommended kernel per distribution and its accuracy
    rec_accs     = {}
    other_accs   = {k: [] for k in KERNELS}

    for dist in DISTRIBUTIONS:
        rec_k = recommendations.get(dist, {}).get("recommended_kernel", "l2")
        rec_accs[dist] = summary["phase1"][dist][rec_k]["mean_accuracy"]
        for k in KERNELS:
            other_accs[k].append(summary["phase1"][dist][k]["mean_accuracy"])

    x = np.arange(len(DISTRIBUTIONS))
    bar_width = 0.16

    # Background bars (all kernels, lighter)
    for j, kernel in enumerate(KERNELS):
        means  = other_accs[kernel]
        offset = x + (j - len(KERNELS) / 2 + 0.5) * bar_width
        ax.bar(offset, means, bar_width, color=KERNEL_COLORS[kernel], alpha=0.30)

    # Foreground: recommended kernel per distribution
    rec_means  = [rec_accs[d]     for d in DISTRIBUTIONS]
    rec_colors = [
        KERNEL_COLORS[recommendations.get(d, {}).get("recommended_kernel", "l2")]
        for d in DISTRIBUTIONS
    ]
    for i, (xpos, mean, col, dist) in enumerate(zip(x, rec_means, rec_colors, DISTRIBUTIONS)):
        rec_k = recommendations.get(dist, {}).get("recommended_kernel", "l2")
        ax.bar(xpos, mean, 0.5, color=col, alpha=0.90,
               edgecolor="#111827", linewidth=1.5, zorder=5)
        ax.text(xpos, mean + 0.025, f"{mean:.1%}",
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                fontweight="bold", zorder=6)
        ax.text(xpos, 0.02, f"→ {KERNEL_LABELS[rec_k]}",
                ha="center", va="bottom",
                fontsize=max(7, VIZ_DEFAULTS["annotation_fontsize"] - 1),
                color="#111827", rotation=0, zorder=6)

    # Legend patches
    handles = [
        mpatches.Patch(color=KERNEL_COLORS[k], alpha=0.85, label=KERNEL_LABELS[k])
        for k in KERNELS
    ]
    ax.legend(handles=handles, fontsize=VIZ_DEFAULTS["tick_fontsize"],
              loc="upper right", ncol=2)

    guidance_text = {
        "hardcode_l2":               "RECOMMENDATION: Hardcode L2",
        "l2_default_with_cosine_option": "RECOMMENDATION: L2 default + cosine option",
        "pluggable_kernels":         "RECOMMENDATION: Pluggable kernels required",
    }.get(guidance, f"GUIDANCE: {guidance}")

    ax.text(0.01, 0.97, guidance_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=VIZ_DEFAULTS["label_fontsize"],
            fontweight="bold",
            color="#1E3A5F",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EFF6FF", alpha=0.9,
                      edgecolor="#1E3A5F"))

    ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0, alpha=0.9,
               label="Random (25%)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.15)
    setup_axes(
        ax,
        "EXP-E1: GAE Kernel Recommendation",
        "Factor Distribution",
        "Mean Accuracy",
    )
    save_figure(fig, "expE1_gae_recommendation", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expE1_kernel_generalization/results/")
    print("Charts saved to paper_figures/expE1_*.{png,pdf}")
