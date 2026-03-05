"""
Publication-quality charts for EXP-B1: Profile-Based Scoring.

Generates
---------
paper_figures/expB1_warm_vs_cold_vs_centroid.{pdf,png}   Chart 1 — convergence lines
paper_figures/expB1_noise_robustness.{pdf,png}            Chart 2 — grouped bar by noise
paper_figures/expB1_lr_heatmap.{pdf,png}                  Chart 3 — eta × eta_neg grid
paper_figures/expB1_profile_drift.{pdf,png}               Chart 4 — drift vs noise rate
paper_figures/expB1_comparison_waterfall.{pdf,png}        Chart 5 — full progression
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
# Display spec
# ---------------------------------------------------------------------------

CONDITION_COLORS = {
    "centroid_only": "#059669",   # green — no learning baseline
    "profile_warm":  "#1E3A5F",   # dark blue — warm start
    "profile_cold":  "#DC2626",   # red — cold start
}
CONDITION_LABELS = {
    "centroid_only": "Centroid only (no learning)",
    "profile_warm":  "Profile warm (initialized)",
    "profile_cold":  "Profile cold (uniform 0.5)",
}

NOISE_COLORS = {
    0.0:  "#1E3A5F",
    0.15: "#7C3AED",
    0.30: "#DC2626",
}

ETA_VALUES    = [0.01, 0.05, 0.10]
ETA_NEG_VALUES = [0.005, 0.01, 0.05]
NOISE_RATES   = [0.0, 0.15, 0.30]


# ---------------------------------------------------------------------------
# Helper: grouped stats from df
# ---------------------------------------------------------------------------

def _grouped_stats(df: pd.DataFrame, col: str, group_by: str = "checkpoint"):
    grp   = df.groupby(group_by)[col]
    means = grp.mean()
    stds  = grp.std().fillna(0.0)
    xs    = sorted(means.index.tolist())
    return xs, [float(means[x]) for x in xs], [float(stds[x]) for x in xs]


def _select_best(
    df: pd.DataFrame,
    summary: dict,
    condition: str,
    noise_rate: float,
) -> pd.DataFrame:
    """Filter df to best (eta, eta_neg) for given condition and noise_rate."""
    if condition == "centroid_only":
        return df[
            (df["condition"] == "centroid_only") &
            (np.abs(df["noise_rate"] - noise_rate) < 1e-6)
        ]
    # Find best eta/eta_neg from summary configs
    best_acc, best_eta, best_eta_neg = -1.0, 0.0, 0.0
    for eta in ETA_VALUES:
        for eta_neg in ETA_NEG_VALUES:
            key = f"{condition}_{eta}_{eta_neg}_{noise_rate}"
            acc = summary["configs"].get(key, {}).get("mean_accuracy_t1000", 0.0)
            if acc > best_acc:
                best_acc, best_eta, best_eta_neg = acc, eta, eta_neg
    return df[
        (df["condition"] == condition) &
        (np.abs(df["noise_rate"] - noise_rate) < 1e-6) &
        (np.abs(df["eta"]        - best_eta)   < 1e-9) &
        (np.abs(df["eta_neg"]    - best_eta_neg) < 1e-9)
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    """
    Read CSV/JSON from *results_dir* and produce all 5 EXP-B1 charts.

    Parameters
    ----------
    results_dir : str
        Directory containing ``accuracy_trajectories.csv`` and ``summary.json``.
    """
    rdir = Path(results_dir)

    df = pd.read_csv(rdir / "accuracy_trajectories.csv")
    with open(rdir / "summary.json") as fh:
        summary = json.load(fh)

    figures_dir = str(ROOT / "paper_figures")

    _chart1_warm_vs_cold_vs_centroid(df, summary, figures_dir)
    _chart2_noise_robustness(df, summary, figures_dir)
    _chart3_lr_heatmap(df, figures_dir)
    _chart4_profile_drift(df, summary, figures_dir)
    _chart5_comparison_waterfall(summary, figures_dir)


# ---------------------------------------------------------------------------
# Chart 1: Convergence lines — warm, cold, centroid_only at noise=0
# ---------------------------------------------------------------------------

def _chart1_warm_vs_cold_vs_centroid(
    df: pd.DataFrame,
    summary: dict,
    figures_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for condition in ("profile_warm", "profile_cold", "centroid_only"):
        sub = _select_best(df, summary, condition, noise_rate=0.0)
        if sub.empty:
            continue
        xs, ys, es = _grouped_stats(sub, "cumulative_gt_acc")
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(
            xs,
            [max(0.0, y - e) for y, e in zip(ys, es)],
            [y + e for y, e in zip(ys, es)],
            color=color, alpha=0.12, zorder=2,
        )

    ax.axhline(0.4926, color="#94A3B8", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Shared W Hebbian (49.26%)", zorder=1)
    ax.axhline(0.9789, color="#059669", linestyle=":", linewidth=1.2,
               alpha=0.6, label="L2 centroid oracle (97.89%)", zorder=1)

    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], loc="lower right")
    setup_axes(
        ax,
        "EXP-B1: Profile-Based Scoring -- Warm vs Cold vs No Learning",
        "Decisions",
        "Cumulative GT Accuracy",
    )
    save_figure(fig, "expB1_warm_vs_cold_vs_centroid", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Grouped bar — noise robustness
# ---------------------------------------------------------------------------

def _chart2_noise_robustness(
    df: pd.DataFrame,
    summary: dict,
    figures_dir: str,
) -> None:
    conditions = ["profile_warm", "profile_cold", "centroid_only"]
    n_conds    = len(conditions)
    n_noise    = len(NOISE_RATES)
    bar_width  = 0.24
    x          = np.arange(n_noise)

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for j, condition in enumerate(conditions):
        means, errs = [], []
        for noise_rate in NOISE_RATES:
            sub = _select_best(df, summary, condition, noise_rate)
            sub_t1000 = sub[sub["checkpoint"] == 1000]
            if sub_t1000.empty:
                means.append(0.0); errs.append(0.0)
            else:
                means.append(float(sub_t1000["cumulative_gt_acc"].mean()))
                errs.append(float(sub_t1000["cumulative_gt_acc"].std()))
        offsets = x + (j - n_conds / 2 + 0.5) * bar_width
        ax.bar(
            offsets, means, bar_width,
            yerr=errs,
            color=CONDITION_COLORS[condition],
            alpha=0.85, capsize=4,
            label=CONDITION_LABELS[condition],
            error_kw={"linewidth": 1.0},
        )

    ax.axhline(0.4926, color="#94A3B8", linestyle="--", linewidth=1.0,
               alpha=0.7, label="Shared W Hebbian (49.26%)", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Noise={n:.0%}" for n in NOISE_RATES],
        fontsize=VIZ_DEFAULTS["tick_fontsize"],
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=2)
    setup_axes(ax, "EXP-B1: Noise Robustness", "Oracle Noise Rate", "Accuracy at t=1000")
    save_figure(fig, "expB1_noise_robustness", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Heatmap — eta × eta_neg for profile_warm, noise=0
# ---------------------------------------------------------------------------

def _chart3_lr_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    n_eta     = len(ETA_VALUES)
    n_eta_neg = len(ETA_NEG_VALUES)
    data      = np.zeros((n_eta, n_eta_neg))

    for i, eta in enumerate(ETA_VALUES):
        for j, eta_neg in enumerate(ETA_NEG_VALUES):
            sub = df[
                (df["condition"]  == "profile_warm") &
                (df["checkpoint"] == 1000) &
                (df["noise_rate"] == 0.0) &
                (np.abs(df["eta"]      - eta)      < 1e-9) &
                (np.abs(df["eta_neg"]  - eta_neg)  < 1e-9)
            ]
            data[i, j] = float(sub["cumulative_gt_acc"].mean()) if not sub.empty else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean GT Accuracy at t=1000")

    for i in range(n_eta):
        for j in range(n_eta_neg):
            text_color = "white" if data[i, j] > 0.6 else "black"
            ax.text(
                j, i, f"{data[i, j]:.3f}",
                ha="center", va="center",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                color=text_color,
            )

    ax.set_xticks(range(n_eta_neg))
    ax.set_yticks(range(n_eta))
    ax.set_xticklabels([str(e) for e in ETA_NEG_VALUES], fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_yticklabels([str(e) for e in ETA_VALUES],    fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(
        ax,
        "EXP-B1: Learning Rate Grid Search (Warm, Noise=0%)",
        "eta_neg (incorrect push rate)",
        "eta (correct pull rate)",
    )
    save_figure(fig, "expB1_lr_heatmap", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Profile drift trajectories — best warm config, 3 noise rates
# ---------------------------------------------------------------------------

def _chart4_profile_drift(
    df: pd.DataFrame,
    summary: dict,
    figures_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for noise_rate in NOISE_RATES:
        sub = _select_best(df, summary, "profile_warm", noise_rate)
        if sub.empty:
            continue
        xs, ys, es = _grouped_stats(sub, "mean_profile_drift")
        color = NOISE_COLORS[noise_rate]
        label = f"profile_warm (noise={noise_rate:.0%})"
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(
            xs,
            [max(0.0, y - e) for y, e in zip(ys, es)],
            [y + e for y, e in zip(ys, es)],
            color=color, alpha=0.12, zorder=2,
        )

    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(
        ax,
        "EXP-B1: Profile Drift from Initial Values",
        "Decisions",
        "Mean L2 Drift (mu vs initial)",
    )
    save_figure(fig, "expB1_profile_drift", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 5: Waterfall — full comparison from random to profile_warm
# ---------------------------------------------------------------------------

def _chart5_comparison_waterfall(summary: dict, figures_dir: str) -> None:
    vb1_2 = summary.get("best_warm_acc_noise0", 0.0)

    labels = [
        "Random\nbaseline",
        "Shared W\nHebbian",
        "Per-cat W\nHebbian",
        "Dot product\ncentroid",
        "Cosine\ncentroid",
        "L2\ncentroid",
        f"Profile warm\n+ learning",
    ]
    values = [0.25, 0.4926, 0.5161, 0.6100, 0.9642, 0.9789, vb1_2]
    colors = ["#CBD5E1", "#94A3B8", "#6B7280", "#D97706", "#059669", "#1E3A5F", "#7C3AED"]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, values, 0.55, color=colors, alpha=0.85, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.1%}",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        )

    # Delta annotations between adjacent bars
    for i in range(len(values) - 1):
        delta = values[i + 1] - values[i]
        mid_x = (x[i] + x[i + 1]) / 2
        mid_y = (values[i] + values[i + 1]) / 2 + 0.01
        ax.annotate(
            f"{delta:+.1%}",
            xy=(mid_x, mid_y),
            fontsize=max(6, VIZ_DEFAULTS["annotation_fontsize"] - 1),
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
        "EXP-B1: Complete Scoring Approach Comparison",
        "Approach",
        "Accuracy",
    )
    save_figure(fig, "expB1_comparison_waterfall", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expB1_profile_scoring/results/")
    print("Charts saved to paper_figures/expB1_*.{png,pdf}")
