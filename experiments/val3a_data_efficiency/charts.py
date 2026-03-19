"""
VAL-3A-3 Charts — Data Efficiency Ratio

Single figure: ProfileScorer accuracy band (horizontal) vs XGBoost/RF
learning curves as a function of labeled training set size.
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


def make_charts(r: dict) -> None:
    profile_mean    = r["profile_mean"]
    profile_std     = r["profile_std"]
    xgb_mean        = np.array(r["xgb_mean"])
    xgb_std         = np.array(r["xgb_std"])
    rf_mean         = np.array(r["rf_mean"])
    rf_std          = np.array(r["rf_std"])
    sample_sizes    = list(r["sample_sizes"])
    xgb_crossover   = r["xgb_crossover"]
    rf_crossover    = r["rf_crossover"]
    random_baseline = r["random_baseline"]
    n_warmup        = r.get("n_warmup", 200)
    n_seeds         = r.get("n_seeds", 10)
    C               = r.get("C", "?")
    A               = r.get("A", "?")

    x = np.array(sample_sizes)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(bottom=0.20)

    # --- ProfileScorer horizontal band ---
    ax.axhspan(profile_mean - profile_std, profile_mean + profile_std,
               color="#2e7d32", alpha=0.12, zorder=2)
    ax.axhline(profile_mean, color="#2e7d32", lw=2.2, ls="--", zorder=4,
               label=f"ProfileScorer — warm-start, 0 labeled samples  "
                     f"({profile_mean:.1%} ± {profile_std:.1%})")

    # --- XGBoost curve ---
    ax.plot(x, xgb_mean, color="#1565c0", lw=2.0, marker="o",
            markersize=5, zorder=4, label="XGBoost (requires labeled data)")
    ax.fill_between(x, xgb_mean - xgb_std, xgb_mean + xgb_std,
                    color="#1565c0", alpha=0.12, zorder=3)

    # --- Random Forest curve ---
    ax.plot(x, rf_mean, color="#e65100", lw=2.0, marker="s",
            markersize=5, zorder=4, label="Random Forest (requires labeled data)")
    ax.fill_between(x, rf_mean - rf_std, rf_mean + rf_std,
                    color="#e65100", alpha=0.12, zorder=3)

    # --- Random baseline ---
    ax.axhline(random_baseline, color="#9e9e9e", lw=1.2, ls=":",
               label=f"Random baseline  ({random_baseline:.1%})")

    # --- Crossover annotations ---
    if xgb_crossover != ">2000":
        xgb_idx = sample_sizes.index(xgb_crossover)
        xgb_y   = float(xgb_mean[xgb_idx])
        ax.annotate(
            f"XGBoost needs ~{xgb_crossover}\nsamples to reach\n90% of ProfileScorer",
            xy=(xgb_crossover, xgb_y),
            xytext=(max(50, xgb_crossover - 500), xgb_y - 0.08),
            fontsize=9.5, color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#e3f2fd",
                      edgecolor="#1565c0", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.2),
        )
    else:
        ax.text(1500, float(xgb_mean[-1]) - 0.06,
                f"XGBoost: N={xgb_crossover}\nat 90% threshold",
                fontsize=9, color="#1565c0", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd",
                          edgecolor="#1565c0", alpha=0.9))

    if rf_crossover != ">2000":
        rf_idx = sample_sizes.index(rf_crossover)
        rf_y   = float(rf_mean[rf_idx])
        xt = min(rf_crossover + 150, 1800)
        ax.annotate(
            f"RF needs ~{rf_crossover}\nsamples",
            xy=(rf_crossover, rf_y),
            xytext=(xt, rf_y - 0.06),
            fontsize=9.5, color="#e65100",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3e0",
                      edgecolor="#e65100", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.2),
        )
    else:
        ax.text(1500, float(rf_mean[-1]) + 0.02,
                f"RF: N={rf_crossover}\nat 90% threshold",
                fontsize=9, color="#e65100", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                          edgecolor="#e65100", alpha=0.9))

    # --- "Useful from decision 1" annotation ---
    ax.annotate(
        f"ProfileScorer is useful\nfrom decision 1\n(config replaces labeling)",
        xy=(0, profile_mean),
        xytext=(180, profile_mean - 0.12),
        fontsize=9.5, color="#2e7d32",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f5e9",
                  edgecolor="#2e7d32", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
    )

    # --- Asymptote annotation (if XGBoost never reaches ProfileScorer band) ---
    if float(xgb_mean[-1]) < profile_mean - profile_std:
        mid_y = (float(xgb_mean[-1]) + profile_mean) / 2
        ax.text(
            1650, mid_y,
            "XGBoost asymptotes\nbelow ProfileScorer ceiling\n"
            "Config quality is a\nstructural advantage",
            fontsize=9, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffde7",
                      edgecolor="#f9a825", alpha=0.9),
        )

    y_lo = max(0.0, random_baseline - 0.05)
    y_hi = min(1.02, profile_mean + 0.10)
    ax.set_xlim(-80, 2150)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Labeled training samples", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "V3A: Data Efficiency — ProfileScorer vs ML Baselines\n"
        "Compiled ontology (warm-start) achieves high accuracy with zero labeled data",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10, loc="lower right", framealpha=0.92)

    caption = (
        f"{n_seeds} seeds, C={C}, A={A}, d=6, τ=0.1, η=η_neg=0.05, "
        f"ProfileScorer warmed on {n_warmup} unlabeled decisions. "
        "Shaded bands = ±1 std across seeds."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color="#555", style="italic")

    save_figure(fig, "VAL-3A-3_data_efficiency_ratio", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART] VAL-3A-3_data_efficiency_ratio.png + .pdf saved")


if __name__ == "__main__":
    r = np.load(
        str(Path(__file__).parent / "results.npy"),
        allow_pickle=True,
    ).item()
    make_charts(r)
