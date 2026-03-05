"""
Publication-quality charts for Experiment 5 (Redesigned): Oracle Fix — Ratio Sweep.

Generates
---------
paper_figures/exp5_ratio_sweep_accuracy.{pdf,png}     Chart 1 — heatmap: oracle × ratio
paper_figures/exp5_oracle_accuracy_best_ratio.{pdf,png} Chart 2 — lines: best-ratio convergence
paper_figures/exp5_fm1_boundary.{pdf,png}             Chart 3 — grouped bars: ratio × oracle
paper_figures/exp5_category_heatmap.{pdf,png}         Chart 4 — heatmap: category × oracle
paper_figures/exp5_warmup_comparison.{pdf,png}        Chart 5 — lines: schedules at GT(15%)
paper_figures/exp5_w_entropy_trajectory.{pdf,png}     Chart 6 — lines: entropy over time
"""
from __future__ import annotations

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
# Oracle display spec
# ---------------------------------------------------------------------------

ORACLE_SPECS: list[tuple[str, float, str, str]] = [
    # (oracle_type, noise_rate, display_label, color_key)
    ("bernoulli",   -1.0,  "Bernoulli",      "bernoulli"),
    ("gt_aligned",   0.0,  "GT(0% noise)",   "gt_noise_0"),
    ("gt_aligned",   0.05, "GT(5% noise)",   "gt_noise_5"),
    ("gt_aligned",   0.15, "GT(15% noise)",  "gt_noise_15"),
    ("gt_aligned",   0.30, "GT(30% noise)",  "gt_noise_30"),
]

RATIO_VALUES: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0]

CATEGORY_COLS  = ["gt_acc_credential", "gt_acc_threat", "gt_acc_lateral",
                  "gt_acc_exfil", "gt_acc_insider"]
CATEGORY_LABELS = ["Credential\nAccess", "Threat Intel\nMatch",
                   "Lateral\nMovement", "Data\nExfil", "Insider\nThreat"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select(df: pd.DataFrame, oracle_type: str, noise_rate: float,
            tol: float = 1e-6) -> pd.DataFrame:
    return df[
        (df["oracle_type"] == oracle_type) &
        ((df["noise_rate"] - noise_rate).abs() < tol)
    ]


def _select_ratio(df: pd.DataFrame, oracle_type: str, noise_rate: float,
                  ratio: float, tol: float = 1e-6) -> pd.DataFrame:
    sub = _select(df, oracle_type, noise_rate)
    return sub[(sub["ratio"] - ratio).abs() < tol]


def _best_ratio(df: pd.DataFrame, oracle_type: str, noise_rate: float) -> float:
    """Return the ratio with highest mean cumulative_gt_acc at the final checkpoint."""
    final_cp = df["checkpoint"].max()
    best_r, best_acc = RATIO_VALUES[0], -1.0
    for r in RATIO_VALUES:
        sub = _select_ratio(df, oracle_type, noise_rate, r)
        sub = sub[sub["checkpoint"] == final_cp]
        if sub.empty:
            continue
        acc = float(sub["cumulative_gt_acc"].mean())
        if acc > best_acc:
            best_acc, best_r = acc, r
    return best_r


def _grouped_stats(df: pd.DataFrame, col: str, group_by: str = "checkpoint"):
    grp   = df.groupby(group_by)[col]
    means = grp.mean()
    stds  = grp.std().fillna(0.0)
    xs    = sorted(means.index.tolist())
    return xs, [float(means[x]) for x in xs], [float(stds[x]) for x in xs]


# ---------------------------------------------------------------------------
# Chart 1: Heatmap — oracle × ratio
# ---------------------------------------------------------------------------

def _chart1_ratio_sweep_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    final_cp = df["checkpoint"].max()
    final_df = df[df["checkpoint"] == final_cp]

    n_oracles = len(ORACLE_SPECS)
    n_ratios  = len(RATIO_VALUES)
    data      = np.zeros((n_oracles, n_ratios))

    oracle_labels = []
    for i, (otype, nrate, label, _) in enumerate(ORACLE_SPECS):
        oracle_labels.append(label)
        for j, ratio in enumerate(RATIO_VALUES):
            sub = _select_ratio(final_df, otype, nrate, ratio)
            if not sub.empty:
                data[i, j] = float(sub["cumulative_gt_acc"].mean())

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_heatmap"])
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean GT Accuracy")

    for i in range(n_oracles):
        for j in range(n_ratios):
            text_color = "white" if data[i, j] > 0.6 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"], color=text_color)

    ax.set_xticks(range(n_ratios))
    ax.set_xticklabels([f"{r:.1f}×" for r in RATIO_VALUES],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_yticks(range(n_oracles))
    ax.set_yticklabels(oracle_labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_xlabel("Asymmetry Ratio (α_incorrect / α_correct)",
                  fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Oracle Type", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("EXP-5: GT Accuracy by Oracle Type × Asymmetry Ratio",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    save_figure(fig, "exp5_ratio_sweep_accuracy", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Line plot — best-ratio convergence per oracle
# ---------------------------------------------------------------------------

def _chart2_oracle_accuracy_best_ratio(df: pd.DataFrame, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for otype, nrate, label, color_key in ORACLE_SPECS:
        best_r = _best_ratio(df, otype, nrate)
        sub    = _select_ratio(df, otype, nrate, best_r)
        if sub.empty:
            continue
        xs, ys, es = _grouped_stats(sub, "cumulative_gt_acc")
        color = COLORS[color_key]
        full_label = f"{label} (ratio={best_r:.1f}×)"
        ax.plot(xs, ys, color=color, linewidth=2.0, label=full_label, zorder=3)
        ax.fill_between(xs,
                        [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=color, alpha=0.15, zorder=2)

    ax.axhline(0.75, color="#6B7280", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Gate (0.75)", zorder=1)
    setup_axes(ax, "EXP-5: Oracle Accuracy (Best Ratio Per Oracle)",
               "Decisions", "Cumulative GT Accuracy")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], loc="lower right")
    save_figure(fig, "exp5_oracle_accuracy_best_ratio", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: Grouped bar — FM1 boundary × ratio interaction
# ---------------------------------------------------------------------------

def _chart3_fm1_boundary(df: pd.DataFrame, figures_dir: str) -> None:
    final_cp = df["checkpoint"].max()
    final_df = df[df["checkpoint"] == final_cp]

    n_ratios  = len(RATIO_VALUES)
    n_oracles = len(ORACLE_SPECS)
    bar_width = 0.14
    x         = np.arange(n_ratios)

    fig, ax = plt.subplots(figsize=(12, 5))

    for j, (otype, nrate, label, color_key) in enumerate(ORACLE_SPECS):
        means, errs = [], []
        for ratio in RATIO_VALUES:
            sub = _select_ratio(final_df, otype, nrate, ratio)
            vals = sub["cumulative_gt_acc"].values if not sub.empty else [0.0]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))
        offsets = x + (j - n_oracles / 2 + 0.5) * bar_width
        ax.bar(offsets, means, bar_width, yerr=errs, color=COLORS[color_key],
               alpha=0.85, capsize=3, label=label, error_kw=dict(linewidth=1.0))

    # FM1 boundary annotation (between ratio index 0 and 1, at noise=4.76%)
    ax.text(0.05, 0.92, "FM1 boundary\n(4.76% noise)",
            transform=ax.transAxes, fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color="#DC2626", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}×" for r in RATIO_VALUES],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=3)
    setup_axes(ax, "EXP-5: FM1 Boundary × Ratio Interaction",
               "Asymmetry Ratio", "Final GT Accuracy (t=1000)")
    save_figure(fig, "exp5_fm1_boundary", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Heatmap — per-category × oracle (best ratio)
# ---------------------------------------------------------------------------

def _chart4_category_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    final_cp = df["checkpoint"].max()
    final_df = df[df["checkpoint"] == final_cp]

    n_cats    = len(CATEGORY_COLS)
    n_oracles = len(ORACLE_SPECS)
    data      = np.zeros((n_cats, n_oracles))

    oracle_labels = []
    for j, (otype, nrate, label, _) in enumerate(ORACLE_SPECS):
        oracle_labels.append(label)
        best_r = _best_ratio(df, otype, nrate)
        sub    = _select_ratio(final_df, otype, nrate, best_r)
        if sub.empty:
            continue
        for i, col in enumerate(CATEGORY_COLS):
            data[i, j] = float(sub[col].mean())

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_heatmap"])
    im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean GT Accuracy")

    for i in range(n_cats):
        for j in range(n_oracles):
            text_color = "white" if data[i, j] > 0.55 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"], color=text_color)

    ax.set_xticks(range(n_oracles))
    ax.set_xticklabels(oracle_labels, fontsize=VIZ_DEFAULTS["tick_fontsize"],
                       rotation=20, ha="right")
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(CATEGORY_LABELS, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_xlabel("Oracle Type", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Alert Category", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("EXP-5: Per-Category Accuracy (Best Config)",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    save_figure(fig, "exp5_category_heatmap", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 5: Warmup vs fixed schedules
# ---------------------------------------------------------------------------

def _chart5_warmup_comparison(wdf: pd.DataFrame, figures_dir: str) -> None:
    SCHEDULE_STYLES: list[tuple[str, str, str]] = [
        ("fixed_2.0",      "#94A3B8", "Fixed 2.0×"),
        ("fixed_optimal",  "#2563EB", "Fixed Optimal"),
        ("warmup",         "#1E3A5F", "Warmup (1.0→2.5)"),
    ]

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for sched, color, label in SCHEDULE_STYLES:
        sub = wdf[wdf["schedule"] == sched]
        if sub.empty:
            continue
        xs, ys, es = _grouped_stats(sub, "cumulative_gt_acc")
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(xs,
                        [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=color, alpha=0.15, zorder=2)

    ax.axhline(0.55, color="#6B7280", linestyle="--", linewidth=1.0,
               alpha=0.7, label="V5.3 gate (0.55)", zorder=1)
    setup_axes(ax, "EXP-5: Warmup Schedule vs Fixed Ratio at 15% Noise",
               "Decisions", "Cumulative GT Accuracy")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    save_figure(fig, "exp5_warmup_comparison", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 6: W entropy over time
# ---------------------------------------------------------------------------

def _chart6_w_entropy_trajectory(df: pd.DataFrame, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

    for otype, nrate, label, color_key in ORACLE_SPECS:
        best_r = _best_ratio(df, otype, nrate)
        sub    = _select_ratio(df, otype, nrate, best_r)
        if sub.empty:
            continue
        xs, ys, es = _grouped_stats(sub, "w_entropy")
        color = COLORS[color_key]
        ax.plot(xs, ys, color=color, linewidth=2.0,
                label=f"{label} (ratio={best_r:.1f}×)", zorder=3)
        ax.fill_between(xs,
                        [max(0.0, y - e) for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=color, alpha=0.15, zorder=2)

    setup_axes(ax, "EXP-5: Weight Matrix Entropy Over Time",
               "Decisions", "W Entropy (lower = more specialised)")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    save_figure(fig, "exp5_w_entropy_trajectory", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    """
    Read CSVs from *results_dir* and produce all 6 EXP-5 charts.

    Parameters
    ----------
    results_dir : str
        Directory containing ``ratio_sweep.csv`` and ``warmup_comparison.csv``.
    """
    rdir = Path(results_dir)
    df   = pd.read_csv(rdir / "ratio_sweep.csv")
    wdf  = pd.read_csv(rdir / "warmup_comparison.csv")

    figures_dir = str(ROOT / "paper_figures")

    _chart1_ratio_sweep_heatmap(df, figures_dir)
    _chart2_oracle_accuracy_best_ratio(df, figures_dir)
    _chart3_fm1_boundary(df, figures_dir)
    _chart4_category_heatmap(df, figures_dir)
    _chart5_warmup_comparison(wdf, figures_dir)
    _chart6_w_entropy_trajectory(df, figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/exp5_oracle_fix/results/")
    print("Charts saved to paper_figures/exp5_*.{png,pdf}")
