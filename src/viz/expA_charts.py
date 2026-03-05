"""
Publication-quality charts for EXP-A: Capacity Ceiling.

Generates
---------
paper_figures/expA_convergence.{pdf,png}       Chart 1 — convergence lines, both profile sets
paper_figures/expA_final_accuracy.{pdf,png}    Chart 2 — grouped bar chart, final accuracy
paper_figures/expA_g_heatmap.{pdf,png}         Chart 3 — learned G matrix heatmap (realistic)
paper_figures/expA_category_breakdown.{pdf,png} Chart 4 — per-category accuracy, realistic
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

from src.viz.bridge_common import VIZ_DEFAULTS, setup_axes, save_figure


# ---------------------------------------------------------------------------
# Color / style spec
# ---------------------------------------------------------------------------

# Colors for scoring configs
SC_COLORS = {
    "w_only":          "#94A3B8",   # gray — baseline
    "w_g_static":      "#D97706",   # amber — offline MI gating
    "w_g_learned":     "#1E3A5F",   # dark navy — online Hebbian gating
    "w_augmented":     "#2D8B4E",   # green — category-augmented factors
    "w_per_category":  "#8B2D8B",   # purple — per-category W matrices
}
SC_LABELS = {
    "w_only":          "W-only (baseline)",
    "w_g_static":      "W + G static (MI)",
    "w_g_learned":     "W + G learned (Hebbian)",
    "w_augmented":     "W augmented (category)",
    "w_per_category":  "W per-category (multi-head)",
}

PROFILE_COLORS = {
    "simplified": "#1E3A5F",
    "realistic":  "#DC2626",
}
PROFILE_LABELS = {
    "simplified": "Simplified (orthogonal)",
    "realistic":  "Realistic (category-varying)",
}

CATEGORY_LABELS = [
    "Credential\nAccess",
    "Threat Intel\nMatch",
    "Lateral\nMovement",
    "Data\nExfil",
    "Insider\nThreat",
]
CATEGORY_COLS = [
    "gt_acc_credential",
    "gt_acc_threat",
    "gt_acc_lateral",
    "gt_acc_exfil",
    "gt_acc_insider",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grouped_stats(df: pd.DataFrame, col: str, group_by: str = "checkpoint"):
    grp   = df.groupby(group_by)[col]
    means = grp.mean()
    stds  = grp.std().fillna(0.0)
    xs    = sorted(means.index.tolist())
    return xs, [float(means[x]) for x in xs], [float(stds[x]) for x in xs]


def _select(df: pd.DataFrame, profile_set: str, scoring_config: str) -> pd.DataFrame:
    return df[
        (df["profile_set"] == profile_set) &
        (df["scoring_config"] == scoring_config)
    ]


# ---------------------------------------------------------------------------
# Chart 1: Convergence lines — both profile sets, all scoring configs
# ---------------------------------------------------------------------------

def _chart1_convergence(df: pd.DataFrame, figures_dir: str) -> None:
    """Side-by-side subplots: left=simplified, right=realistic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    gate_lines = {
        "simplified": (0.75, "VA.1 gate (0.75)"),
        "realistic":  (0.60, "VA.3 ceiling (0.60)"),
    }

    for ax, ps in zip(axes, ["simplified", "realistic"]):
        for sc in ["w_only", "w_g_static", "w_g_learned", "w_augmented", "w_per_category"]:
            sub = _select(df, ps, sc)
            if sub.empty:
                continue
            xs, ys, es = _grouped_stats(sub, "cumulative_gt_acc")
            color = SC_COLORS[sc]
            ax.plot(xs, ys, color=color, linewidth=2.0, label=SC_LABELS[sc], zorder=3)
            ax.fill_between(
                xs,
                [max(0.0, y - e) for y, e in zip(ys, es)],
                [y + e for y, e in zip(ys, es)],
                color=color, alpha=0.15, zorder=2,
            )

        gate_val, gate_label = gate_lines[ps]
        ax.axhline(gate_val, color="#6B7280", linestyle="--", linewidth=1.2,
                   alpha=0.8, label=gate_label, zorder=1)

        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(
            f"EXP-A: {PROFILE_LABELS[ps]}",
            fontsize=VIZ_DEFAULTS["title_fontsize"],
        )
        ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], loc="lower right")

    axes[0].set_ylabel("Cumulative GT Accuracy", fontsize=VIZ_DEFAULTS["label_fontsize"])
    fig.tight_layout()
    save_figure(fig, "expA_convergence", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 2: Grouped bar chart — final accuracy by (profile_set × scoring_config)
# ---------------------------------------------------------------------------

def _chart2_final_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    final_cp = df["checkpoint"].max()
    final_df = df[df["checkpoint"] == final_cp]

    scoring_configs = ["w_only", "w_g_static", "w_g_learned", "w_augmented", "w_per_category"]
    profile_sets    = ["simplified", "realistic"]
    n_sc            = len(scoring_configs)
    n_ps            = len(profile_sets)
    bar_width       = 0.32
    x               = np.arange(n_sc)

    hatches = {"simplified": "", "realistic": "///"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, ps in enumerate(profile_sets):
        means, errs = [], []
        for sc in scoring_configs:
            sub  = _select(final_df, ps, sc)
            vals = sub["cumulative_gt_acc"].values if not sub.empty else [0.0]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))

        offsets = x + (j - n_ps / 2 + 0.5) * bar_width
        ax.bar(
            offsets, means, bar_width,
            yerr=errs,
            color=PROFILE_COLORS[ps],
            hatch=hatches[ps],
            alpha=0.80,
            capsize=4,
            label=PROFILE_LABELS[ps],
            error_kw={"linewidth": 1.0},
        )

    ax.axhline(0.75, color="#1E3A5F", linestyle="--", linewidth=1.0,
               alpha=0.6, label="VA.1 gate (0.75)", zorder=0)
    ax.axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.0,
               alpha=0.6, label="VA.3 ceiling (0.60)", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SC_LABELS[sc] for sc in scoring_configs],
        fontsize=VIZ_DEFAULTS["tick_fontsize"],
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=2, loc="upper left")
    setup_axes(
        ax,
        f"EXP-A: Final GT Accuracy (t={final_cp})",
        "Scoring Configuration",
        "Cumulative GT Accuracy",
    )
    save_figure(fig, "expA_final_accuracy", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 3: G matrix heatmap — realistic profiles, w_g_learned (mean over seeds)
# ---------------------------------------------------------------------------

def _chart3_g_heatmap(g_df: pd.DataFrame, bc: dict, figures_dir: str) -> None:
    """Heatmap of mean G matrix for realistic/w_g_learned: categories × factors."""
    sub = g_df[
        (g_df["profile_set"] == "realistic") &
        (g_df["scoring_config"] == "w_g_learned")
    ]

    categories   = bc["categories"]
    factor_names = bc["factors"]
    n_cats       = len(categories)
    n_facs       = len(factor_names)

    data = np.zeros((n_cats, n_facs))
    for i, cat in enumerate(categories):
        for j, fac in enumerate(factor_names):
            col = f"g_{cat}_{fac}"
            if col in sub.columns and not sub.empty:
                data[i, j] = float(sub[col].mean())
            else:
                data[i, j] = 1.0    # default (UniformGating)

    # Also show w_g_static for comparison (second subplot)
    sub_static = g_df[
        (g_df["profile_set"] == "realistic") &
        (g_df["scoring_config"] == "w_g_static")
    ]
    data_static = np.zeros((n_cats, n_facs))
    for i, cat in enumerate(categories):
        for j, fac in enumerate(factor_names):
            col = f"g_{cat}_{fac}"
            if col in sub_static.columns and not sub_static.empty:
                data_static[i, j] = float(sub_static[col].mean())
            else:
                data_static[i, j] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, data_mat, title_suffix in zip(
        axes,
        [data_static, data],
        ["W + G static (MI)", "W + G learned (Hebbian)"],
    ):
        im = ax.imshow(data_mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, label="Mean Gate Value")

        for i in range(n_cats):
            for j in range(n_facs):
                text_color = "white" if data_mat[i, j] > 0.55 else "black"
                ax.text(
                    j, i, f"{data_mat[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                    color=text_color,
                )

        factor_short = [f.replace("_", "\n") for f in factor_names]
        cat_short    = [c.replace("_", "\n") for c in categories]

        ax.set_xticks(range(n_facs))
        ax.set_xticklabels(factor_short, fontsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(cat_short, fontsize=VIZ_DEFAULTS["tick_fontsize"])
        ax.set_xlabel("Factor", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.set_ylabel("Category", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.set_title(
            f"EXP-A: G Matrix (Realistic) — {title_suffix}",
            fontsize=VIZ_DEFAULTS["title_fontsize"],
        )

    fig.tight_layout()
    save_figure(fig, "expA_g_heatmap", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Chart 4: Per-category accuracy — realistic profiles, all scoring configs
# ---------------------------------------------------------------------------

def _chart4_category_breakdown(df: pd.DataFrame, figures_dir: str) -> None:
    """Grouped bar chart: categories × scoring configs on realistic profiles."""
    final_cp = df["checkpoint"].max()
    final_df = df[(df["checkpoint"] == final_cp) & (df["profile_set"] == "realistic")]

    scoring_configs = ["w_only", "w_g_static", "w_g_learned", "w_augmented", "w_per_category"]
    n_sc        = len(scoring_configs)
    n_cats      = len(CATEGORY_COLS)
    bar_width   = 0.24
    x           = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(13, 5))

    for j, sc in enumerate(scoring_configs):
        sub = final_df[final_df["scoring_config"] == sc]
        means, errs = [], []
        for col in CATEGORY_COLS:
            vals = sub[col].values if not sub.empty else [0.0]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))

        offsets = x + (j - n_sc / 2 + 0.5) * bar_width
        ax.bar(
            offsets, means, bar_width,
            yerr=errs,
            color=SC_COLORS[sc],
            alpha=0.85,
            capsize=3,
            label=SC_LABELS[sc],
            error_kw={"linewidth": 1.0},
        )

    ax.axhline(0.25, color="#9CA3AF", linestyle=":", linewidth=1.0,
               alpha=0.8, label="Random (0.25)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], ncol=3, loc="upper right")
    setup_axes(
        ax,
        f"EXP-A: Per-Category Accuracy — Realistic Profiles (t={final_cp})",
        "Alert Category",
        "Cumulative GT Accuracy",
    )
    save_figure(fig, "expA_category_breakdown", output_dir=figures_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_charts(results_dir: str) -> None:
    """
    Read CSVs from *results_dir* and produce all 4 EXP-A charts.

    Parameters
    ----------
    results_dir : str
        Directory containing ``accuracy_trajectories.csv`` and ``g_matrices.csv``.
    """
    rdir = Path(results_dir)

    traj_path = rdir / "accuracy_trajectories.csv"
    g_path    = rdir / "g_matrices.csv"

    df  = pd.read_csv(traj_path)
    g_df = pd.read_csv(g_path) if g_path.exists() else pd.DataFrame()

    # Load bridge_common for factor/category name lists
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        bc = yaml.safe_load(fh)["bridge_common"]

    figures_dir = str(ROOT / "paper_figures")

    _chart1_convergence(df, figures_dir)
    _chart2_final_accuracy(df, figures_dir)
    _chart3_g_heatmap(g_df, bc, figures_dir)
    _chart4_category_breakdown(df, figures_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_charts("experiments/expA_capacity_ceiling/results/")
    print("Charts saved to paper_figures/expA_*.{png,pdf}")
