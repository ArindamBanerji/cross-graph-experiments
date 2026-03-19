"""
PROD-4-WARMUP charts: Coverage Growth vs Warmup Length.

Reads: experiments/prod/prod4_warmup_sweep/prod4_warmup_results.json
Writes: paper_figures/prod4w_*.{pdf,png}

Charts:
  1. prod4w_coverage_growth      — coverage at threshold* by warmup (6 category lines)
  2. prod4w_threshold_descent    — threshold* by warmup (6 category lines)
  3. prod4w_overall_accuracy     — overall accuracy vs warmup (single line)
  4. prod4w_mean_coverage_bar    — mean coverage across categories (bar chart)
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.viz.bridge_common import save_figure, COLORS, VIZ_DEFAULTS


def generate_charts(results: dict, categories: list[str], warmup_levels: list[int]) -> None:

    cat_colors_base = COLORS.get("category_colors", [])
    # Pad to 6 if needed
    EXTRA = ["#EA580C", "#7C3AED", "#0891B2", "#059669"]
    cat_colors = (cat_colors_base + EXTRA)[:len(categories)]

    # -----------------------------------------------------------------------
    # Chart 1 — Coverage growth (THE KEY CHART)
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for i, cat in enumerate(categories):
        covs = []
        for w in warmup_levels:
            c = results[str(w)]["confidence_thresholds"][cat]["coverage_at_star"]
            covs.append(c * 100.0 if c is not None else 0.0)
        ax.plot(warmup_levels, covs, "o-", label=cat,
                color=cat_colors[i], linewidth=2.0, markersize=6)

    ax.axhline(y=20.0, color="#6B7280", linestyle="--", linewidth=1.4, alpha=0.8,
               label="20% target")
    ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Coverage at ≥85% accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Auto-Approve Coverage Growth vs Warmup\n"
        "(η_neg=0.05, τ=0.1, 50 seeds, centroidal synthetic)",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="upper left", ncol=2,
              framealpha=0.85)
    ax.set_xticks(warmup_levels)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    save_figure(fig, "prod4w_coverage_growth", output_dir="paper_figures")
    plt.close()
    print("[CHART 1] prod4w_coverage_growth saved")

    # -----------------------------------------------------------------------
    # Chart 2 — Threshold descent
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for i, cat in enumerate(categories):
        thresholds = []
        for w in warmup_levels:
            t = results[str(w)]["confidence_thresholds"][cat]["threshold_star"]
            thresholds.append(t if t is not None else 1.0)
        ax.plot(warmup_levels, thresholds, "o-", label=cat,
                color=cat_colors[i], linewidth=2.0, markersize=6)

    ax.axhline(y=0.75, color="#059669", linestyle="--", linewidth=1.4, alpha=0.8,
               label="threshold ≤ 0.75 (good coverage zone)")
    ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("threshold* (confidence for ≥85% accuracy)",
                  fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Threshold* Descent with Warmup\n"
        "(lower threshold* = more decisions auto-approved at same accuracy)",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="upper right", ncol=2,
              framealpha=0.85)
    ax.set_xticks(warmup_levels)
    ax.set_ylim(0.50, 1.02)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    save_figure(fig, "prod4w_threshold_descent", output_dir="paper_figures")
    plt.close()
    print("[CHART 2] prod4w_threshold_descent saved")

    # -----------------------------------------------------------------------
    # Chart 3 — Overall accuracy vs warmup
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    accs = [results[str(w)]["overall_accuracy"] * 100.0 for w in warmup_levels]
    ax.plot(warmup_levels, accs, "o-", color="#1E3A5F", linewidth=2.5, markersize=8)

    for w, a in zip(warmup_levels, accs):
        ax.annotate(f"{a:.1f}%", xy=(w, a), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"])

    ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Overall accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Overall Accuracy vs Warmup (η_neg=0.05, τ=0.1)",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    ax.set_xticks(warmup_levels)
    y_min = min(accs) - 2.0
    y_max = min(100.0, max(accs) + 4.0)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    save_figure(fig, "prod4w_overall_accuracy", output_dir="paper_figures")
    plt.close()
    print("[CHART 3] prod4w_overall_accuracy saved")

    # -----------------------------------------------------------------------
    # Chart 4 — Mean coverage bar chart
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    mean_covs = []
    n_cats_list = []
    for w in warmup_levels:
        covs = [
            results[str(w)]["confidence_thresholds"][cat]["coverage_at_star"]
            for cat in categories
            if results[str(w)]["confidence_thresholds"][cat]["coverage_at_star"] is not None
        ]
        mean_covs.append(float(np.mean(covs)) * 100.0 if covs else 0.0)
        n_cats_list.append(len(covs))

    x_labels = [str(w) for w in warmup_levels]
    bar_colors = ["#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"]
    bars = ax.bar(x_labels, mean_covs, color=bar_colors[:len(warmup_levels)],
                  edgecolor="white", linewidth=0.8)

    ax.axhline(y=20.0, color="#6B7280", linestyle="--", linewidth=1.4, alpha=0.8,
               label="20% target")

    for bar, val, n in zip(bars, mean_covs, n_cats_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{val:.1f}%\n({n}/6)",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        )

    ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Mean coverage across categories (%)",
                  fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Mean Auto-Approve Coverage vs Warmup\n"
        "(categories with threshold* at ≥85% accuracy)",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "prod4w_mean_coverage_bar", output_dir="paper_figures")
    plt.close()
    print("[CHART 4] prod4w_mean_coverage_bar saved")

    print()
    print("[DONE] All 4 PROD-4-WARMUP charts written to paper_figures/prod4w_*")


# ---------------------------------------------------------------------------
# Standalone entry point (called via subprocess or directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_path = Path(__file__).parent / "prod4_warmup_results.json"
    with open(results_path) as fh:
        results = json.load(fh)

    meta       = results["meta"]
    categories = meta["domain_config"]   # placeholder — load properly below

    config_path = _REPO_ROOT / "configs" / f"{meta['domain_config']}.yaml"
    import yaml
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    categories    = raw["categories"]
    warmup_levels = meta["warmup_levels"]

    generate_charts(results, categories, warmup_levels)
