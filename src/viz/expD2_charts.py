"""
Publication-quality charts for EXP-D2: Factor Interaction Discovery.

Generates
---------
paper_figures/expD2_single_mi.{pdf,png}          Chart 1 — 5×6 MI heatmap (category × factor)
paper_figures/expD2_interaction_gain.{pdf,png}    Chart 2 — 5 subplots, 6×6 gain matrices
paper_figures/expD2_top_interactions.{pdf,png}    Chart 3 — horizontal bar, top 15 by gain
paper_figures/expD2_augmentation.{pdf,png}        Chart 4 — augmentation lift per seed (or placeholder)
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

from src.viz.bridge_common import COLORS, VIZ_DEFAULTS, setup_axes, save_figure


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(
    results_dir: str,
    *,
    augmentation_ran: bool = False,
    top_pairs: list = (),
    factors: list = (),
    mean_base: float = 0.0,
    mean_aug: float = 0.0,
    aug_lift: float = 0.0,
) -> None:
    """
    Read CSV/JSON from *results_dir* and produce all 4 EXP-D2 charts.

    Parameters
    ----------
    results_dir      : str   — path to experiment results directory
    augmentation_ran : bool  — whether Phase 2 augmentation was executed
    top_pairs        : list  — list of (i,j) factor index pairs used in augmentation
    factors          : list  — list of factor name strings (len 6)
    mean_base        : float — mean baseline accuracy across seeds
    mean_aug         : float — mean augmented accuracy across seeds
    aug_lift         : float — mean_aug - mean_base
    """
    rdir = Path(results_dir)

    mi_single_df = pd.read_csv(rdir / "mi_single.csv")
    mi_inter_df  = pd.read_csv(rdir / "mi_interaction.csv")

    with open(rdir / "top_interactions.json") as fh:
        top_interactions = json.load(fh)

    aug_df = None
    if augmentation_ran and (rdir / "augmentation_results.csv").exists():
        aug_df = pd.read_csv(rdir / "augmentation_results.csv")

    # Load factor/category names from config if not supplied
    if not factors:
        cfg_path = ROOT / "configs" / "default.yaml"
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh)
        factors = raw["bridge_common"]["factors"]

    categories = mi_single_df["category"].tolist()
    figures_dir = str(ROOT / "paper_figures")

    _chart1_single_mi(mi_single_df, categories, factors, figures_dir)
    _chart2_interaction_gain(mi_single_df, mi_inter_df, categories, factors, figures_dir)
    _chart3_top_interactions(top_interactions, figures_dir)
    _chart4_augmentation(
        aug_df, augmentation_ran, top_pairs, factors,
        mean_base, mean_aug, aug_lift, figures_dir,
    )


# ---------------------------------------------------------------------------
# Chart 1: Single-factor MI heatmap (5 categories × 6 factors)
# ---------------------------------------------------------------------------

def _chart1_single_mi(
    mi_single_df: pd.DataFrame,
    categories: list,
    factors: list,
    figures_dir: str,
) -> None:
    n_cats    = len(categories)
    n_factors = len(factors)

    # Build (n_cats, n_factors) array
    data = np.zeros((n_cats, n_factors))
    for c_idx, cat in enumerate(categories):
        row = mi_single_df[mi_single_df["category"] == cat].iloc[0]
        for f_idx, fname in enumerate(factors):
            data[c_idx, f_idx] = float(row[fname])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=float(data.max() * 1.05), aspect="auto")
    plt.colorbar(im, ax=ax, label="Mutual Information (bits)")

    vmax = float(data.max())
    for i in range(n_cats):
        for j in range(n_factors):
            v = data[i, j]
            text_color = "white" if v > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                    color=text_color)

    # Short factor labels (strip underscores)
    flabels = [f.replace("_", "\n") for f in factors]
    clabels = [c.replace("_", "\n") for c in categories]

    ax.set_xticks(range(n_factors))
    ax.set_yticks(range(n_cats))
    ax.set_xticklabels(flabels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_yticklabels(clabels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_xlabel("Factor", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Category", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("EXP-D2: Single-Factor Mutual Information (category × factor)",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    fig.tight_layout()
    save_figure(fig, "expD2_single_mi", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: 6×6 interaction gain matrices — 2×3 grid (5 categories + 1 hidden)
# ---------------------------------------------------------------------------

def _chart2_interaction_gain(
    mi_single_df: pd.DataFrame,
    mi_inter_df: pd.DataFrame,
    categories: list,
    factors: list,
    figures_dir: str,
) -> None:
    n_factors = len(factors)
    pair_indices = [(i, j) for i in range(n_factors) for j in range(i + 1, n_factors)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    # Global vmax for consistent color scale across all subplots
    all_gains = []
    for c_idx, cat in enumerate(categories):
        sub = mi_inter_df[mi_inter_df["category"] == cat]
        all_gains.extend(sub["interaction_gain"].tolist())
    vmax_gain = max(float(np.percentile(all_gains, 98)), 2.0)

    for c_idx, cat in enumerate(categories):
        ax = axes_flat[c_idx]
        sub_single = mi_single_df[mi_single_df["category"] == cat].iloc[0]
        sub_inter  = mi_inter_df[mi_inter_df["category"] == cat]

        # Build 6×6 display matrix
        # Diagonal = single-factor MI (normalized), off-diagonal = gain
        mi_vals   = np.array([float(sub_single[f]) for f in factors])
        mi_max    = max(float(mi_vals.max()), 1e-10)
        diag_norm = mi_vals / mi_max  # 0..1 for color mapping

        matrix = np.zeros((n_factors, n_factors))
        for i in range(n_factors):
            matrix[i, i] = diag_norm[i]

        # Fill off-diagonal with gain (symmetric)
        for _, row in sub_inter.iterrows():
            i = factors.index(row["factor_i"])
            j = factors.index(row["factor_j"])
            gain = float(row["interaction_gain"])
            matrix[i, j] = gain
            matrix[j, i] = gain

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=vmax_gain, aspect="auto")

        # Annotate
        for i in range(n_factors):
            for j in range(n_factors):
                v = matrix[i, j]
                text_color = "white" if v > vmax_gain * 0.75 else "black"
                if i == j:
                    label = f"{mi_vals[i]:.3f}"
                else:
                    label = f"{v:.2f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=max(5, VIZ_DEFAULTS["annotation_fontsize"] - 1.5),
                        color=text_color)

        flabels = [f.replace("_", "\n") for f in factors]
        ax.set_xticks(range(n_factors))
        ax.set_yticks(range(n_factors))
        ax.set_xticklabels(flabels, fontsize=max(5, VIZ_DEFAULTS["tick_fontsize"] - 1))
        ax.set_yticklabels(flabels, fontsize=max(5, VIZ_DEFAULTS["tick_fontsize"] - 1))
        ax.set_title(cat.replace("_", " "), fontsize=VIZ_DEFAULTS["tick_fontsize"])

    # Add shared colorbar
    plt.colorbar(im, ax=axes_flat[:5], label="Gain (off-diag) / Norm. MI (diag)",
                 shrink=0.6)

    # Significant threshold line marker in colorbar label
    axes_flat[5].set_visible(False)

    fig.suptitle("EXP-D2: Interaction Gain Matrices (diagonal = single-MI, off-diag = gain)\n"
                 "Green > 1.5 = significant synergy",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    fig.tight_layout()
    save_figure(fig, "expD2_interaction_gain", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Top 15 interactions by gain — horizontal bar
# ---------------------------------------------------------------------------

def _chart3_top_interactions(top_interactions: list, figures_dir: str) -> None:
    top15 = top_interactions[:15]
    if not top15:
        return

    labels = []
    gains  = []
    colors = []

    for inter in reversed(top15):       # reversed so highest at top
        cat   = inter["category"].replace("_", " ")
        fi    = inter["factor_i"].replace("_", " ")
        fj    = inter["factor_j"].replace("_", " ")
        labels.append(f"{cat}\n{fi} × {fj}")
        g = float(inter["gain"])
        gains.append(g)
        colors.append("#059669" if g > 1.5 else "#94A3B8")

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(labels) + 2))
    y = np.arange(len(labels))
    ax.barh(y, gains, color=colors, alpha=0.85, edgecolor="none")

    # Annotate values
    for yi, g in zip(y, gains):
        ax.text(g + 0.02, yi, f"{g:.3f}",
                va="center", fontsize=VIZ_DEFAULTS["annotation_fontsize"])

    ax.axvline(1.5, color="#DC2626", linestyle="--", linewidth=1.2,
               label="Significance threshold (1.5)", alpha=0.8)
    ax.axvline(1.0, color="#94A3B8", linestyle=":", linewidth=1.0,
               label="Baseline (no synergy = 1.0)", alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_xlim(0.0, max(float(max(gains)) * 1.15, 2.0))
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], loc="lower right")
    setup_axes(ax, "EXP-D2: Top 15 Factor Interactions by Gain",
               "Interaction Gain (MI_ij / (MI_i + MI_j))", "Factor Pair")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, "expD2_top_interactions", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Augmentation lift (per-seed bars + mean) or placeholder
# ---------------------------------------------------------------------------

def _chart4_augmentation(
    aug_df,
    augmentation_ran: bool,
    top_pairs: list,
    factors: list,
    mean_base: float,
    mean_aug: float,
    aug_lift: float,
    figures_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    if not augmentation_ran or aug_df is None:
        # Placeholder
        ax.text(0.5, 0.5,
                "No significant interactions found (gain > 1.5).\n"
                "Augmentation not run.\n\n"
                "Factors are independently informative —\n"
                "a well-designed feature space.",
                ha="center", va="center",
                fontsize=VIZ_DEFAULTS["label_fontsize"],
                color="#374151",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F3F4F6", edgecolor="#D1D5DB"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("EXP-D2: Augmentation Results (not run)",
                     fontsize=VIZ_DEFAULTS["title_fontsize"])
        fig.tight_layout()
        save_figure(fig, "expD2_augmentation", output_dir=figures_dir)
        return

    seeds     = aug_df["seed"].tolist()
    baselines = aug_df["baseline_accuracy"].tolist()
    augmented = aug_df["augmented_accuracy"].tolist()

    n = len(seeds)
    x = np.arange(n)
    bar_w = 0.35

    ax.bar(x - bar_w / 2, baselines, bar_w, color="#1E3A5F", alpha=0.85, label="Baseline (6 factors)")
    ax.bar(x + bar_w / 2, augmented, bar_w, color="#059669", alpha=0.85,
           label=f"Augmented (+{len(top_pairs)} pair features)")

    # Mean reference lines
    ax.axhline(mean_base, color="#1E3A5F", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Mean base {mean_base:.1%}")
    ax.axhline(mean_aug,  color="#059669", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Mean aug  {mean_aug:.1%}")

    # Lift annotation
    lift_str = f"Lift: {aug_lift*100:+.2f}pp"
    ax.text(0.98, 0.05, lift_str,
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=VIZ_DEFAULTS["label_fontsize"],
            color="#059669" if aug_lift >= 0 else "#DC2626",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#D1D5DB"))

    pair_labels = [f"{factors[i]}×{factors[j]}" for i, j in top_pairs] if factors else []
    pairs_str   = ", ".join(pair_labels)
    title = f"EXP-D2: Augmentation Accuracy per Seed\nPairs used: {pairs_str}"

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(max(0.0, min(baselines + augmented) - 0.05), 1.02)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=2)
    setup_axes(ax, title, "Seed", "Accuracy")
    fig.tight_layout()
    save_figure(fig, "expD2_augmentation", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expD2_factor_interactions/results/")
    print("Charts saved to paper_figures/expD2_*.{png,pdf}")
