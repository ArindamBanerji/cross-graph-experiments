"""
Publication-quality charts for EXP-C1: Centroid Oracle Diagnostic.

Generates
---------
paper_figures/expC1_method_comparison.{pdf,png}       Chart 1 — overall accuracy by method
paper_figures/expC1_category_breakdown.{pdf,png}      Chart 2 — per-category accuracy by method
paper_figures/expC1_confusion_heatmap.{pdf,png}       Chart 3 — 5-subplot confusion matrices
paper_figures/expC1_comparison_waterfall.{pdf,png}    Chart 4 — waterfall vs EXP-A baselines
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
import numpy as np
import pandas as pd
import yaml

from src.viz.bridge_common import VIZ_DEFAULTS, setup_axes, save_figure


# ---------------------------------------------------------------------------
# Display spec
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "dot": "#1E3A5F",
    "l2":  "#D97706",
    "cos": "#059669",
}
METHOD_LABELS = {
    "dot": "Dot product",
    "l2":  "Nearest centroid (L2)",
    "cos": "Cosine similarity",
}

CATEGORY_SHORT = [
    "Credential\nAccess",
    "Threat Intel\nMatch",
    "Lateral\nMovement",
    "Data\nExfil",
    "Insider\nThreat",
]

ACTION_SHORT = ["auto\nclose", "esc\ntier2", "enrich\nwatch", "esc\nincident"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    """
    Read JSON/CSV from *results_dir* and produce all 4 EXP-C1 charts.

    Parameters
    ----------
    results_dir : str
        Directory containing ``summary.json``, ``confusion_matrices.json``,
        and ``classification_results.csv``.
    """
    rdir = Path(results_dir)

    with open(rdir / "summary.json") as fh:
        summary = json.load(fh)
    with open(rdir / "confusion_matrices.json") as fh:
        cm_data = json.load(fh)

    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    categories = raw["bridge_common"]["categories"]

    figures_dir = str(ROOT / "paper_figures")
    best_method = summary["best_method"]

    _chart1_method_comparison(summary, figures_dir)
    _chart2_category_breakdown(summary, categories, figures_dir)
    _chart3_confusion_heatmap(cm_data, best_method, categories, figures_dir)
    _chart4_comparison_waterfall(summary, figures_dir)


# ---------------------------------------------------------------------------
# Chart 1: Overall accuracy by method — grouped bar with EXP-A reference lines
# ---------------------------------------------------------------------------

def _chart1_method_comparison(summary: dict, figures_dir: str) -> None:
    methods = ["dot", "l2", "cos"]
    means   = [summary["methods"][m]["overall_accuracy"] for m in methods]
    stds    = [summary["methods"][m]["std_accuracy"]     for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]
    labels  = [METHOD_LABELS[m] for m in methods]

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])
    x    = np.arange(len(methods))
    bars = ax.bar(
        x, means, yerr=stds,
        color=colors, alpha=0.85, capsize=5,
        error_kw={"linewidth": 1.2}, width=0.5,
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{mean:.1%}",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        )

    ax.axhline(0.4926, color="#94A3B8", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Shared W Hebbian (49.26%)", zorder=0)
    ax.axhline(0.5161, color="#6B7280", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Per-cat W Hebbian (51.61%)", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], loc="upper left")
    setup_axes(
        ax,
        "EXP-C1: Centroid Oracle -- No Learning, Just Profile Matching",
        "Similarity Method",
        "Overall Accuracy (10 seeds, 10k alerts/seed)",
    )
    save_figure(fig, "expC1_method_comparison", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Per-category accuracy — 3 bars (methods) per category
# ---------------------------------------------------------------------------

def _chart2_category_breakdown(summary: dict, categories: list, figures_dir: str) -> None:
    methods   = ["dot", "l2", "cos"]
    n_cats    = len(categories)
    n_methods = len(methods)
    bar_width = 0.24
    x         = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for j, m in enumerate(methods):
        per_cat = summary["methods"][m]["per_category_accuracy"]
        means   = [per_cat[cat] for cat in categories]
        offsets = x + (j - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            offsets, means, bar_width,
            color=METHOD_COLORS[m], alpha=0.85,
            label=METHOD_LABELS[m],
        )

    ax.axhline(0.25, color="#9CA3AF", linestyle=":", linewidth=1.0,
               alpha=0.8, label="Random baseline (25%)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_SHORT, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=2)
    setup_axes(
        ax,
        "EXP-C1: Per-Category Centroid Accuracy",
        "Alert Category",
        "Classification Accuracy",
    )
    save_figure(fig, "expC1_category_breakdown", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Confusion heatmaps — 5 subplots (one per category), best method
# ---------------------------------------------------------------------------

def _chart3_confusion_heatmap(
    cm_data: dict,
    best_method: str,
    categories: list,
    figures_dir: str,
) -> None:
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(18, 4))

    for ax, cat, short in zip(axes, categories, CATEGORY_SHORT):
        mat      = np.array(cm_data[best_method][cat])  # (4, 4) row=pred, col=true
        col_sums = mat.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        mat_norm = mat / col_sums                        # column-normalized (recall)

        im = ax.imshow(mat_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

        for i in range(4):
            for j in range(4):
                text_color = "white" if mat_norm[i, j] > 0.55 else "black"
                ax.text(
                    j, i, f"{mat_norm[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                    color=text_color,
                )

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(ACTION_SHORT, fontsize=max(6, VIZ_DEFAULTS["tick_fontsize"] - 1))
        ax.set_yticklabels(ACTION_SHORT, fontsize=max(6, VIZ_DEFAULTS["tick_fontsize"] - 1))
        ax.set_xlabel("True action", fontsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.set_ylabel("Predicted",   fontsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.set_title(short.replace("\n", " "), fontsize=VIZ_DEFAULTS["tick_fontsize"])

    fig.suptitle(
        f"EXP-C1: Confusion Matrices (Best Method: {best_method})",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "expC1_confusion_heatmap", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Waterfall — random → shared W → per-cat W → centroid → theoretical max
# ---------------------------------------------------------------------------

def _chart4_comparison_waterfall(summary: dict, figures_dir: str) -> None:
    best_acc = summary["best_accuracy"]
    best_m   = summary["best_method"]

    labels  = [
        "Random\nbaseline",
        "Shared W\nHebbian",
        "Per-cat W\nHebbian",
        f"Centroid oracle\n({METHOD_LABELS[best_m]})",
        "Theoretical\nmax",
    ]
    values     = [0.25, 0.4926, 0.5161, best_acc, 1.0]
    colors     = ["#CBD5E1", "#94A3B8", "#6B7280", "#1E3A5F", "none"]
    edgecolors = ["none",    "none",    "none",    "none",    "#1E3A5F"]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))

    for i, (val, col, ec) in enumerate(zip(values, colors, edgecolors)):
        if col == "none":
            ax.bar(x[i], val, 0.55, color="none", edgecolor=ec,
                   linewidth=1.5, linestyle="--", alpha=0.6)
        else:
            ax.bar(x[i], val, 0.55, color=col, alpha=0.85, edgecolor="none")

        ax.text(
            x[i], val + 0.016,
            f"{val:.1%}",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        )

    # Delta annotations between adjacent bars (skip last → theoretical max)
    for i in range(len(values) - 2):
        delta = values[i + 1] - values[i]
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (values[i] + values[i + 1]) / 2
        ax.annotate(
            f"{delta:+.1%}",
            xy=(mid_x, mid_y),
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            ha="center", va="center",
            color="#374151",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white", alpha=0.8, edgecolor="none",
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.15)
    setup_axes(
        ax,
        "EXP-C1: Scoring Approaches Compared -- Learning vs Knowledge",
        "Approach",
        "Accuracy",
    )
    save_figure(fig, "expC1_comparison_waterfall", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expC1_centroid_oracle/results/")
    print("Charts saved to paper_figures/expC1_*.{png,pdf}")
