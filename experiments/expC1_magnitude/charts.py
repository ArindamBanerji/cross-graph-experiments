"""
EXP-C1 SUPPLEMENT: Factor Magnitude Confounding — Chart Generation

Two-panel figure:
  Panel A: Mean centroid value per factor (averaged over all C×A pairs)
  Panel B: Discriminative contribution per factor (dot product vs L2)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure, COLORS


def make_charts(
    factor_names: list[str],
    factor_centroid_means: np.ndarray,
    factor_centroid_stds: np.ndarray,
    dot_contributions: np.ndarray,
    l2_contributions: np.ndarray,
    dot_diff: np.ndarray,
    l2_diff: np.ndarray,
) -> None:
    """Generate and save the EXP-C1 magnitude confounding figure."""

    # Sort factors by centroid mean (ascending) for both panels
    sort_idx = np.argsort(factor_centroid_means)
    sorted_names = [factor_names[i] for i in sort_idx]
    sorted_means = factor_centroid_means[sort_idx]
    sorted_stds  = factor_centroid_stds[sort_idx]

    # Color: red if mean > 0.70 (magnitude-confounding), blue otherwise
    bar_colors = ["#d32f2f" if m > 0.70 else "#1976d2" for m in sorted_means]

    # Normalized separating power for Panel B
    dot_norm = dot_diff / (dot_diff.sum() + 1e-9)   # (d,)
    l2_norm  = l2_diff  / (l2_diff.sum()  + 1e-9)   # (d,)

    dot_sorted = dot_norm[sort_idx]
    l2_sorted  = l2_norm[sort_idx]

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.45, bottom=0.22)

    # ==================================================================
    # Panel A: Factor centroid means
    # ==================================================================
    ax1.set_title(
        "Panel A: Mean centroid value per factor\n(averaged over all category × action pairs)",
        fontsize=12, pad=8,
    )

    bars = ax1.barh(
        range(len(sort_idx)), sorted_means,
        color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.85,
    )
    ax1.errorbar(
        sorted_means, range(len(sort_idx)),
        xerr=sorted_stds, fmt="none", color="black",
        capsize=4, capthick=1.2, linewidth=1.2,
    )
    ax1.set_yticks(range(len(sort_idx)))
    ax1.set_yticklabels(sorted_names, fontsize=11)
    ax1.set_xlabel("Mean centroid value (all C×A pairs)", fontsize=11)
    ax1.set_xlim(0, 1.05)
    ax1.axvline(
        0.70, color="#d32f2f", linewidth=1.2, linestyle="--", alpha=0.7,
        label="0.70 — magnitude-confounding threshold",
    )
    ax1.legend(fontsize=9, loc="lower right")

    # Annotate red bars
    for i, (mean, color) in enumerate(zip(sorted_means, bar_colors)):
        if color == "#d32f2f":
            ax1.text(
                mean + 0.02, i,
                "← dot product\ndominated",
                va="center", fontsize=8.5, color="#d32f2f", style="italic",
            )

    ax1_caption = (
        "Red bars: factors with mean > 0.70 contribute near-constant\n"
        "dot product scores regardless of discriminative value."
    )
    ax1.text(
        0.01, -0.14, ax1_caption,
        transform=ax1.transAxes, fontsize=9, style="italic", color="#555",
    )

    # ==================================================================
    # Panel B: Normalized separating power, dot product vs L2
    # ==================================================================
    y = np.arange(len(sort_idx))
    height = 0.35

    ax2.barh(
        y + height / 2, dot_sorted, height=height,
        color="#d32f2f", alpha=0.80, edgecolor="black", linewidth=0.6,
        label="Dot product",
    )
    ax2.barh(
        y - height / 2, l2_sorted, height=height,
        color="#1976d2", alpha=0.80, edgecolor="black", linewidth=0.6,
        label="L2 distance",
    )

    ax2.set_yticks(y)
    ax2.set_yticklabels(sorted_names, fontsize=11)
    ax2.set_xlabel("Fraction of total separating power (normalized)", fontsize=11)
    ax2.set_title(
        "Panel B: Discriminative contribution per factor\n"
        "(separating power: action 0 vs action 1, category 0)",
        fontsize=12, pad=8,
    )
    ax2.legend(fontsize=10, loc="lower right")
    ax2.axvline(
        1.0 / len(sort_idx), color="gray", linewidth=1.0, linestyle=":",
        label="equal-weight reference",
    )

    # Annotation box at bottom — concrete example
    annotation = (
        "Concrete example (device_trust):\n"
        "  Dot product: 0.90 \u00d7 0.88 = 0.79  (constant for all actions)\n"
        "  L2 distance: (0.90 \u2212 0.88)\u00b2 = 0.0004  (near-zero \u2014 not discriminative)\n"
        "  \u2192 dot product weight is driven by magnitude, not by deviation"
    )
    fig.text(
        0.5, 0.01, annotation,
        ha="center", fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#fff8e1",
            edgecolor="#f9a825", linewidth=1.2,
        ),
    )

    fig.suptitle(
        "EXP-C1: Why Dot Product Fails on Bounded [0,1] Features \u2014 Magnitude Confounding",
        fontsize=14, fontweight="bold", y=1.01,
    )

    save_figure(fig, "expC1_factor_magnitude_distribution", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART] expC1_factor_magnitude_distribution.png + .pdf saved")


if __name__ == "__main__":
    # Allow running charts.py standalone for quick iteration (requires run.py data).
    # Re-runs the full data pipeline then generates charts.
    import importlib.util, runpy
    runpy.run_path(str(Path(__file__).parent / "run.py"), run_name="__main__")
