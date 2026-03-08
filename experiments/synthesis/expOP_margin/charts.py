"""
EXP-OP-MARGIN chart generation.

Three publication-quality charts (300 DPI, PDF + PNG):
  1. expOPm_margin_distribution        — L2 margin histogram + lambda_flip CDF
  2. expOPm_lambda_sweep               — AUAC delta and p-value vs lambda
  3. expOPm_per_category_margins       — margin boxplots by category
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.viz.bridge_common import save_figure, COLORS

RESULTS_JSON = Path("experiments/synthesis/expOP_margin/results.json")

SIGMA_VALUE  = 0.4
N_SEEDS      = 20
COLOR_B      = "#1E3A5F"
COLOR_C      = "#DC2626"
BONFERRONI_ALPHA = 0.05 / 7   # 7 lambda values


def generate_charts(data: dict) -> None:
    """Generate and save all three EXP-OP-MARGIN charts."""

    margin_stats = data["margin_stats"]
    lambda_sweep = data["lambda_sweep"]
    lambda_values = [float(k) for k in lambda_sweep.keys()]

    all_margins_arr    = np.array(margin_stats["all_margins"])
    lambda_flip_values = all_margins_arr / SIGMA_VALUE

    # ------------------------------------------------------------------
    # CHART 1: Margin distribution histogram + lambda_flip CDF
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — histogram of L2 margins
    ax1.hist(all_margins_arr, bins=60, color=COLOR_B, alpha=0.75, edgecolor="none")
    ref_lambdas = [(0.2, "#94A3B8"), (1.0, "#D97706"), (2.0, "#059669"), (5.0, "#DC2626")]
    for lv, col in ref_lambdas:
        eff_bias = lv * SIGMA_VALUE
        ax1.axvline(eff_bias, color=col, linewidth=1.5, linestyle="--",
                    label=f"lambda={lv:.1f} eff.bias={eff_bias:.2f}")
    ax1.set_xlabel("L2 margin (runner-up - winner distance^2)")
    ax1.set_ylabel("Count")
    ax1.set_title("L2 Decision Margin Distribution")
    ax1.legend(fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2 — CDF of lambda_flip values
    sorted_lf = np.sort(lambda_flip_values)
    cdf = np.arange(1, len(sorted_lf) + 1) / len(sorted_lf)
    ax2.plot(sorted_lf, cdf, color=COLOR_B, linewidth=2)

    ff_map = {
        0.2:  margin_stats["fraction_flippable"]["lambda_0.2"],
        1.0:  margin_stats["fraction_flippable"]["lambda_1.0"],
        2.0:  margin_stats["fraction_flippable"]["lambda_2.0"],
        5.0:  margin_stats["fraction_flippable"]["lambda_5.0"],
    }
    for lv, col in ref_lambdas:
        ff = ff_map[lv]
        ax2.axvline(lv, color=col, linewidth=1.2, linestyle="--", alpha=0.7)
        ax2.axhline(ff, color=col, linewidth=1.0, linestyle=":", alpha=0.7)
        ax2.text(lv + 0.05, ff + 0.015, f"{ff:.1%}", fontsize=7.5, color=col)

    ax2.axhline(0.10, color="grey", linestyle=":", linewidth=1, label="10% threshold")
    ax2.axhline(0.50, color="grey", linestyle="--", linewidth=1, label="50% threshold")
    ax2.set_xlabel("lambda value (required to flip decision)")
    ax2.set_ylabel("Fraction of decisions flippable")
    ax2.set_title("Cumulative Fraction of Decisions Flippable by lambda")
    ax2.legend(fontsize=7.5)
    ax2.set_xlim(0, max(lambda_values) + 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("EXP-OP-MARGIN: L2 Decision Geometry", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "expOPm_margin_distribution", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOPm_margin_distribution.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: Lambda sweep — AUAC delta and p-value
    # ------------------------------------------------------------------
    deltas   = [lambda_sweep[str(lv)]["mean_delta"] for lv in lambda_values]
    stds     = [lambda_sweep[str(lv)]["std_delta"]  for lv in lambda_values]
    c_deltas = [lambda_sweep[str(lv)]["c_vs_a"]     for lv in lambda_values]
    p_values = [lambda_sweep[str(lv)]["p_value"]    for lv in lambda_values]
    ci95     = [1.96 * s / np.sqrt(N_SEEDS) for s in stds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — AUAC delta vs lambda
    ax1.errorbar(lambda_values, deltas, yerr=ci95,
                 color=COLOR_B, marker="o", linewidth=2, capsize=5,
                 label="B vs A (correct op)")
    ax1.plot(lambda_values, c_deltas, color=COLOR_C, marker="s",
             linestyle="--", linewidth=2, label="C vs A (harmful op)")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axvspan(0.2, 0.5, alpha=0.08, color=COLOR_B, label="S4 plateau region")

    # Mark first passing lambda if any
    passing = [lv for lv, p, d in zip(lambda_values, p_values, deltas)
               if p < BONFERRONI_ALPHA and d > 0]
    if passing:
        ax1.axvline(min(passing), color="green", linestyle=":", linewidth=1.5,
                    label=f"lambda_threshold={min(passing):.1f}")

    ax1.set_xlabel("lambda (coupling constant)")
    ax1.set_ylabel("AUAC delta vs baseline")
    ax1.set_title("AUAC Delta vs lambda")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2 — p-value (log scale) vs lambda
    # Replace zero p-values with minimum float for log scale
    p_plot = [max(p, 1e-5) for p in p_values]
    ax2.semilogy(lambda_values, p_plot, color=COLOR_B, marker="o", linewidth=2)
    ax2.axhline(0.05, color="orange", linestyle="--", linewidth=1.5, label="alpha=0.05")
    ax2.axhline(BONFERRONI_ALPHA, color="red", linestyle="--", linewidth=1.5,
                label=f"Bonferroni alpha={BONFERRONI_ALPHA:.4f}")
    ax2.set_xlabel("lambda (coupling constant)")
    ax2.set_ylabel("p-value (log scale)")
    ax2.set_title("Statistical Significance vs lambda")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("EXP-OP-MARGIN: lambda Threshold for Detectable sigma Effect", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "expOPm_lambda_sweep", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOPm_lambda_sweep.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: Per-category margin distribution
    # ------------------------------------------------------------------
    per_cat = margin_stats["per_category"]
    cat_names  = list(per_cat.keys())
    cat_data   = [per_cat[c] for c in cat_names]
    cat_colors = COLORS.get("category_colors", ["#1E3A5F","#D97706","#059669","#DC2626","#7C3AED"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        cat_data,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    for patch, color in zip(bp["boxes"], cat_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Reference lines for lambda × sigma thresholds
    ref_lines = [
        (0.2 * SIGMA_VALUE, "#94A3B8", "lambda=0.2 eff.bias=0.08"),
        (1.0 * SIGMA_VALUE, "#D97706", "lambda=1.0 eff.bias=0.40"),
        (2.0 * SIGMA_VALUE, "#059669", "lambda=2.0 eff.bias=0.80"),
    ]
    for yval, col, lbl in ref_lines:
        ax.axhline(yval, color=col, linestyle="--", linewidth=1.2, alpha=0.8, label=lbl)

    # Annotate per-category fraction flippable at lambda=0.2
    for i, (cat, margins) in enumerate(zip(cat_names, cat_data)):
        margins_arr = np.array(margins)
        ff = float(np.mean(margins_arr / SIGMA_VALUE < 0.2))
        short_cat = cat.split("_")[0][:8]
        ax.text(i + 1, max(margins_arr) * 1.02,
                f"{ff:.0%}", ha="center", fontsize=7.5, color="#6B7280")

    ax.set_xticklabels([c.replace("_", "\n") for c in cat_names], fontsize=8)
    ax.set_xlabel("Category")
    ax.set_ylabel("L2 margin")
    ax.set_title("EXP-OP-MARGIN: Margin Distribution by Category\n"
                 "(% above each box = fraction flippable at lambda=0.2)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOPm_per_category_margins", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] expOPm_per_category_margins.png + .pdf saved")

    print("\nAll 3 charts saved to paper_figures/:")
    for name in [
        "expOPm_margin_distribution",
        "expOPm_lambda_sweep",
        "expOPm_per_category_margins",
    ]:
        print(f"  {name}.png + .pdf")


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    generate_charts(data)
