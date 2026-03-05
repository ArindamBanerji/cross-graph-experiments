"""
Publication-quality charts for EXP-E2: Scale Test.

Generates
---------
paper_figures/expE2_oracle_scaling.{pdf,png}          Chart 1 - oracle accuracy vs scale
paper_figures/expE2_learning_curves.{pdf,png}         Chart 2 - warm/cold learning at each scale
paper_figures/expE2_scaling_trend.{pdf,png}           Chart 3 - accuracy vs parameter count
paper_figures/expE2_separation_vs_accuracy.{pdf,png}  Chart 4 - separation predicts accuracy
paper_figures/expE2_cold_vs_warm_gap.{pdf,png}        Chart 5 - warm vs cold gap at scale
paper_figures/expE2_decisions_per_centroid.{pdf,png}  Chart 6 - cold-start decisions per centroid
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

from src.viz.bridge_common import VIZ_DEFAULTS, setup_axes, save_figure


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS    = ["small", "medium", "large", "xlarge"]
CONFIG_DIM = {
    "small":  "5x4x6",
    "medium": "10x6x10",
    "large":  "15x8x15",
    "xlarge": "20x10x20",
}
CONFIG_PARAMS = {
    "small":  120,
    "medium": 600,
    "large":  1800,
    "xlarge": 4000,
}

# Blue gradient from light to dark
SCALE_COLORS = ["#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A5F"]

CHECKPOINT_PCTS = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    rdir = Path(results_dir)

    with open(rdir / "summary.json") as fh:
        summary = json.load(fh)

    df1 = pd.read_csv(rdir / "phase1_oracle.csv")
    df2 = pd.read_csv(rdir / "phase2_learning.csv")
    df3 = pd.read_csv(rdir / "phase3_separation.csv")

    figures_dir = str(ROOT / "paper_figures")

    _chart1_oracle_scaling(df1, summary, figures_dir)
    _chart2_learning_curves(df2, summary, figures_dir)
    _chart3_scaling_trend(summary, figures_dir)
    _chart4_separation_vs_accuracy(df1, df3, summary, figures_dir)
    _chart5_cold_vs_warm_gap(summary, figures_dir)
    _chart6_decisions_per_centroid(df2, summary, figures_dir)


# ---------------------------------------------------------------------------
# Chart 1: Oracle accuracy vs scale — bar chart
# ---------------------------------------------------------------------------

def _chart1_oracle_scaling(df: pd.DataFrame, summary: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(CONFIGS))
    means = [summary[c]["oracle_accuracy"]["mean"] for c in CONFIGS]
    stds  = [summary[c]["oracle_accuracy"]["std"]  for c in CONFIGS]

    bars = ax.bar(x, means, 0.55,
                  color=SCALE_COLORS, alpha=0.88,
                  yerr=stds, capsize=5,
                  error_kw={"linewidth": 1.2})

    for bar, mean, c in zip(bars, means, CONFIGS):
        dims  = CONFIG_DIM[c]
        params = CONFIG_PARAMS[c]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) + 0.015,
                f"{mean:.1%}\n{dims}\n({params:,} params)",
                ha="center", va="bottom",
                fontsize=max(6, VIZ_DEFAULTS["annotation_fontsize"] - 1),
                linespacing=1.4)

    ax.axhline(0.90, color="#F59E0B", linestyle="--", linewidth=1.2,
               alpha=0.8, label="90% threshold", zorder=0)
    ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0,
               alpha=0.8, label="Random (25%)", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n{CONFIG_DIM[c]}" for c in CONFIGS],
        fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.18)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax,
               "EXP-E2: Centroid Oracle Accuracy vs Problem Scale",
               "Scale Configuration",
               "Oracle Accuracy (10 seeds)")
    save_figure(fig, "expE2_oracle_scaling", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Learning curves — 4 subplots (warm + cold per config)
# ---------------------------------------------------------------------------

def _chart2_learning_curves(df: pd.DataFrame, summary: dict, figures_dir: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for ax, config, color in zip(axes, CONFIGS, SCALE_COLORS):
        n_dec = summary[config]["n_decisions"]

        for condition, linestyle, label_prefix in [
            ("warm", "-",  "Warm start"),
            ("cold", "--", "Cold start"),
        ]:
            sub = df[(df["scale_config"] == config) & (df["condition"] == condition)]
            if sub.empty:
                continue

            pcts   = sorted(sub["checkpoint_pct"].unique())
            means  = [sub[sub["checkpoint_pct"] == p]["cumulative_accuracy"].mean() for p in pcts]
            stds   = [sub[sub["checkpoint_pct"] == p]["cumulative_accuracy"].std()  for p in pcts]
            ts     = [int(p * n_dec) for p in pcts]
            means  = np.array(means)
            stds   = np.array(stds)

            line_color = color if condition == "warm" else "#94A3B8"
            ax.plot(ts, means, color=line_color, linestyle=linestyle,
                    linewidth=2.0, marker="o", markersize=4,
                    label=f"{label_prefix}")
            ax.fill_between(ts, means - stds, means + stds,
                            color=line_color, alpha=0.15)

        ax.set_xlim(0, n_dec * 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0.25, color="#CBD5E1", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{config}\n({CONFIG_DIM[config]}, {CONFIG_PARAMS[config]:,} params)",
                     fontsize=VIZ_DEFAULTS["title_fontsize"] - 1)
        ax.set_xlabel(f"Decisions (n={n_dec:,})", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=max(7, VIZ_DEFAULTS["tick_fontsize"] - 1))

    axes[0].set_ylabel("Cumulative Accuracy", fontsize=VIZ_DEFAULTS["label_fontsize"])
    fig.suptitle("EXP-E2: Learning Convergence at Scale (Warm vs Cold Start)",
                 fontsize=VIZ_DEFAULTS["title_fontsize"], y=1.01)
    fig.tight_layout()
    save_figure(fig, "expE2_learning_curves", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Accuracy vs parameter count (log scale) — KEY scaling chart
# ---------------------------------------------------------------------------

def _chart3_scaling_trend(summary: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    params = [CONFIG_PARAMS[c] for c in CONFIGS]

    oracle_means = [summary[c]["oracle_accuracy"]["mean"] for c in CONFIGS]
    oracle_stds  = [summary[c]["oracle_accuracy"]["std"]  for c in CONFIGS]
    warm_means   = [summary[c]["warm_t100pct"]["mean"]    for c in CONFIGS]
    warm_stds    = [summary[c]["warm_t100pct"]["std"]     for c in CONFIGS]
    cold_means   = [summary[c]["cold_t100pct"]["mean"]    for c in CONFIGS]
    cold_stds    = [summary[c]["cold_t100pct"]["std"]     for c in CONFIGS]

    def _plot_line(ax, xs, ys, yerr, color, label, linestyle="-"):
        ys   = np.array(ys)
        yerr = np.array(yerr)
        ax.semilogx(xs, ys, color=color, linestyle=linestyle,
                    linewidth=2.2, marker="o", markersize=7, label=label)
        ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.15)
        for x, y, cfg in zip(xs, ys, CONFIGS):
            ax.annotate(
                f"{y:.0%}",
                xy=(x, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                color=color,
            )

    _plot_line(ax, params, oracle_means, oracle_stds,
               "#1E3A5F", "Oracle (no learning)")
    _plot_line(ax, params, warm_means,   warm_stds,
               "#059669", "Warm start (t=100%)", "--")
    _plot_line(ax, params, cold_means,   cold_stds,
               "#94A3B8", "Cold start (t=100%)", ":")

    # Label configs on x-axis
    ax.set_xticks(params)
    ax.set_xticklabels(
        [f"{CONFIG_PARAMS[c]:,}\n({CONFIG_DIM[c]})" for c in CONFIGS],
        fontsize=VIZ_DEFAULTS["tick_fontsize"])

    ax.axhline(0.90, color="#F59E0B", linestyle="--", linewidth=1.0,
               alpha=0.7, label="90% threshold")
    ax.set_ylim(0.0, 1.08)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax,
               "EXP-E2: Accuracy vs Parameter Count",
               "Total Parameters (log scale)",
               "Accuracy")
    save_figure(fig, "expE2_scaling_trend", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Separation vs oracle accuracy — per-category scatter
# ---------------------------------------------------------------------------

def _chart4_separation_vs_accuracy(
    df1: pd.DataFrame,
    df3: pd.DataFrame,
    summary: dict,
    figures_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for config, color in zip(CONFIGS, SCALE_COLORS):
        # Oracle accuracy per seed (mean over seeds = single number for the config)
        oracle_mean = summary[config]["oracle_accuracy"]["mean"]

        # Per-category separation
        sep_sub = df3[df3["scale_config"] == config]
        mean_seps = sep_sub["mean_separation"].values
        n_cats    = len(mean_seps)

        # One dot per category: x=mean_separation, y=oracle_acc (same for all within config)
        # Slight y jitter to reveal density
        rng = np.random.default_rng(int(abs(hash(config))) % (2**31))
        y_jitter = rng.uniform(-0.005, 0.005, size=n_cats)

        ax.scatter(
            mean_seps,
            np.full(n_cats, oracle_mean) + y_jitter,
            color=color, alpha=0.65, s=30,
            label=f"{config} ({CONFIG_DIM[config]})",
            zorder=4,
        )

    # Overall regression line across all config/category data
    all_sep  = df3["mean_separation"].values
    all_acc  = np.array([
        summary[row["scale_config"]]["oracle_accuracy"]["mean"]
        for _, row in df3.iterrows()
    ])
    if len(all_sep) > 2:
        coeffs  = np.polyfit(all_sep, all_acc, 1)
        x_fit   = np.linspace(all_sep.min(), all_sep.max(), 100)
        y_fit   = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color="#374151", linewidth=1.5, linestyle="--",
                label=f"Trend (slope={coeffs[0]:+.2f})", zorder=3, alpha=0.7)

    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax,
               "EXP-E2: Profile Separation Predicts Accuracy",
               "Mean Pairwise Separation (per category)",
               "Oracle Accuracy")
    save_figure(fig, "expE2_separation_vs_accuracy", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 5: Warm vs cold gap — grouped bar at t=100%
# ---------------------------------------------------------------------------

def _chart5_cold_vs_warm_gap(summary: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(CONFIGS))
    bar_width = 0.30

    warm_means = np.array([summary[c]["warm_t100pct"]["mean"] for c in CONFIGS])
    cold_means = np.array([summary[c]["cold_t100pct"]["mean"] for c in CONFIGS])
    warm_stds  = np.array([summary[c]["warm_t100pct"]["std"]  for c in CONFIGS])
    cold_stds  = np.array([summary[c]["cold_t100pct"]["std"]  for c in CONFIGS])

    bars_warm = ax.bar(x - bar_width / 2, warm_means, bar_width,
                       color="#1E3A5F", alpha=0.85,
                       yerr=warm_stds, capsize=4,
                       error_kw={"linewidth": 1.0},
                       label="Warm start (t=100%)")
    bars_cold = ax.bar(x + bar_width / 2, cold_means, bar_width,
                       color="#94A3B8", alpha=0.85,
                       yerr=cold_stds, capsize=4,
                       error_kw={"linewidth": 1.0},
                       label="Cold start (t=100%)")

    for i, (wm, cm) in enumerate(zip(warm_means, cold_means)):
        gap_pp = (wm - cm) * 100
        y_ann  = max(wm, cm) + 0.05
        ax.annotate(
            f"gap\n{gap_pp:+.1f}pp",
            xy=(x[i], y_ann),
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.8, edgecolor="none"),
        )

    for bars, means in [(bars_warm, warm_means), (bars_cold, cold_means)]:
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{m:.0%}",
                    ha="center", va="bottom",
                    fontsize=max(6, VIZ_DEFAULTS["annotation_fontsize"] - 1))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({CONFIG_DIM[c]})" for c in CONFIGS],
        fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.22)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax,
               "EXP-E2: Warm vs Cold Start Gap at Scale",
               "Scale Configuration",
               "Cumulative Accuracy at t=100%")
    save_figure(fig, "expE2_cold_vs_warm_gap", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 6: Cold-start decisions per centroid vs accuracy
# ---------------------------------------------------------------------------

def _chart6_decisions_per_centroid(
    df: pd.DataFrame,
    summary: dict,
    figures_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # Collect (decisions/centroid, cold accuracy at pct=1.0) per config
    x_vals   = []
    y_means  = []
    y_stds   = []
    labels   = []
    colors   = []

    for config, color in zip(CONFIGS, SCALE_COLORS):
        n_centroids = summary[config]["total_centroids"]
        n_decisions = summary[config]["n_decisions"]
        dpc = n_decisions / n_centroids if n_centroids > 0 else 0

        sub = df[(df["scale_config"] == config) &
                 (df["condition"]    == "cold")  &
                 (df["checkpoint_pct"] == 1.00)]
        if sub.empty:
            continue

        x_vals.append(dpc)
        y_means.append(sub["cumulative_accuracy"].mean())
        y_stds.append(sub["cumulative_accuracy"].std())
        labels.append(f"{config}\n({CONFIG_DIM[config]})")
        colors.append(color)

    x_arr  = np.array(x_vals)
    y_arr  = np.array(y_means)
    y_err  = np.array(y_stds)

    ax.plot(x_arr, y_arr, color="#1E3A5F", linewidth=2.0,
            marker="o", markersize=8, zorder=4)

    for xi, yi, yerr_i, lbl, col in zip(x_arr, y_arr, y_err, labels, colors):
        ax.errorbar(xi, yi, yerr=yerr_i, fmt="none",
                    ecolor="#94A3B8", capsize=5, linewidth=1.2)
        ax.scatter([xi], [yi], color=col, s=80, zorder=5)
        ax.annotate(
            f"{lbl}\n{yi:.0%}",
            xy=(xi, yi),
            xytext=(0, 12), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        )

    ax.axhline(0.80, color="#F59E0B", linestyle="--", linewidth=1.2,
               alpha=0.8, label="80% threshold")
    ax.set_ylim(0.0, 1.10)
    ax.set_xlim(0, max(x_vals) * 1.25 if x_vals else 100)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax,
               "EXP-E2: Cold Start -- Decisions per Centroid vs Accuracy",
               "Decisions per Centroid (n_decisions / total_centroids)",
               "Cold-Start Cumulative Accuracy at t=100%")
    save_figure(fig, "expE2_decisions_per_centroid", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expE2_scale_test/results/")
    print("Charts saved to paper_figures/expE2_*.{png,pdf}")
