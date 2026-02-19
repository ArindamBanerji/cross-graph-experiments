"""
Publication-quality charts for Experiment 1: Scoring Matrix Convergence.

Generates
---------
paper_figures/exp1_convergence.{pdf,png}       Chart 1 -- main convergence curves
paper_figures/exp1_window_accuracy.{pdf,png}   Chart 2 -- window (per-stage) accuracy
paper_figures/exp1_per_action.{pdf,png}        Chart 3 -- per-action breakdown
paper_figures/exp1_weight_evolution.{pdf,png}  Chart 4 -- weight matrix heatmaps
paper_figures/exp1_table.tex                   LaTeX summary table
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # file-only backend; must precede pyplot import

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "exp1_scoring_convergence" / "results"
FIGURES_DIR = ROOT / "paper_figures"
CSV_PATH    = RESULTS_DIR / "convergence_data.csv"
NPZ_PATH    = RESULTS_DIR / "weight_evolution.npz"

# ---------------------------------------------------------------------------
# Visual constants  (mirrors configs/default.yaml visualization block)
# ---------------------------------------------------------------------------

FONT = dict(title=13, label=11, tick=9, anno=8.5)
DPI  = 300

FACTOR_NAMES = [
    "severity", "asset_crit.", "user_risk",
    "time_anom.", "pattern", "context",
]
ACTION_NAMES = ["auto_close", "enrich+watch", "escalate_t2", "escalate_inc"]

METHOD_STYLES: dict[str, dict] = {
    "compounding": dict(
        color="#1E3A5F", ls="-",  lw=2.5, zorder=5, alpha_fill=0.15,
        label="Compounding (20:1 asymmetry + decay)",
    ),
    "fixed_weight": dict(
        color="#94A3B8", ls="--", lw=1.5, zorder=2, alpha_fill=0.10,
        label="Fixed weight",
    ),
    "random_policy": dict(
        color="#CBD5E1", ls=":",  lw=1.5, zorder=1, alpha_fill=0.08,
        label="Random policy",
    ),
    "periodic_retrain": dict(
        color="#EA580C", ls="--", lw=1.5, zorder=4, alpha_fill=0.12,
        label="Periodic retrain",
    ),
    "symmetric": dict(
        color="#DC2626", ls="--", lw=1.5, zorder=3, alpha_fill=0.12,
        label="Symmetric (1:1)",
    ),
}

# ascending zorder so compounding is drawn last (on top)
PLOT_ORDER = [
    "fixed_weight", "random_policy", "symmetric", "periodic_retrain", "compounding",
]

LATEX_ORDER = [
    "compounding", "symmetric", "periodic_retrain", "fixed_weight", "random_policy",
]
LATEX_NAMES = {
    "compounding":      "Compounding (ours)",
    "symmetric":        "Symmetric (1:1)",
    "periodic_retrain": "Periodic retrain",
    "fixed_weight":     "Fixed weight",
    "random_policy":    "Random policy",
}


# ---------------------------------------------------------------------------
# Style / IO helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    plt.rcParams.update({
        "font.size":        FONT["label"],
        "axes.titlesize":   FONT["title"],
        "axes.labelsize":   FONT["label"],
        "xtick.labelsize":  FONT["tick"],
        "ytick.labelsize":  FONT["tick"],
        "legend.fontsize":  FONT["tick"],
        "figure.dpi":       DPI,
        "savefig.dpi":      DPI,
        "pdf.fonttype":     42,   # embed TrueType fonts (required for journals)
        "ps.fonttype":      42,
    })


def _clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.7)


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # stored as fractions [0, 1] -> convert to percentages
    pct_cols = [
        "cumulative_accuracy", "window_accuracy",
        "action_0_accuracy", "action_1_accuracy",
        "action_2_accuracy", "action_3_accuracy",
    ]
    df[pct_cols] = df[pct_cols] * 100.0
    return df


def _agg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-(method, checkpoint) mean / std / 95 % CI."""
    g = df.groupby(["method", "checkpoint"])[col].agg(["mean", "std", "count"])
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    return g.reset_index()


def _val(agg_df: pd.DataFrame, method: str, checkpoint: int,
         stat: str = "mean") -> float:
    mask = (agg_df["method"] == method) & (agg_df["checkpoint"] == checkpoint)
    v = agg_df.loc[mask, stat].values
    return float(v[0]) if len(v) else 0.0


# ---------------------------------------------------------------------------
# Shared line-plot core (used by Charts 1 and 2)
# ---------------------------------------------------------------------------

def _draw_accuracy_lines(
    ax: plt.Axes,
    agg: pd.DataFrame,
    cps: list[int],
    marker: str | None = None,
) -> None:
    for method in PLOT_ORDER:
        s   = METHOD_STYLES[method]
        sub = agg[agg["method"] == method].sort_values("checkpoint")
        xs  = sub["checkpoint"].values
        ys  = sub["mean"].values
        ci  = sub["ci95"].values

        kwargs: dict = dict(
            color=s["color"], linestyle=s["ls"],
            linewidth=s["lw"], label=s["label"], zorder=s["zorder"],
        )
        if marker:
            kwargs.update(marker=marker, markersize=3.5)

        ax.plot(xs, ys, **kwargs)
        ax.fill_between(
            xs, ys - ci, ys + ci,
            color=s["color"], alpha=s["alpha_fill"],
            zorder=s["zorder"] - 1,
        )

    ax.set_xticks(cps)
    ax.set_xticklabels(cps, rotation=30, ha="right", fontsize=FONT["tick"])
    _clean_ax(ax)


# ---------------------------------------------------------------------------
# Chart 1 -- Convergence curves (cumulative accuracy)
# ---------------------------------------------------------------------------

def chart1_convergence(df: pd.DataFrame) -> tuple[Path, Path]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    agg = _agg(df, "cumulative_accuracy")
    cps = sorted(df["checkpoint"].unique())
    _draw_accuracy_lines(ax, agg, cps)

    # -- "Day 1" annotation at checkpoint 50 --
    y50 = _val(agg, "compounding", 50)
    ax.annotate(
        "Day 1",
        xy=(50, y50),
        xytext=(200, y50 + 12),
        fontsize=FONT["anno"],
        color="#1E3A5F",
        arrowprops=dict(arrowstyle="-", color="#1E3A5F", lw=0.8),
    )

    # -- Random baseline reference line --
    ax.axhline(25, color="#A0AEC0", linestyle="--", linewidth=0.9, zorder=0)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.97, 26.5, "Random baseline", transform=trans,
            ha="right", fontsize=FONT["anno"], color="#A0AEC0")

    # -- Summary text box (upper left) --
    comp_mean = _val(agg, "compounding",      5000)
    comp_std  = _val(agg, "compounding",      5000, "std")
    per_mean  = _val(agg, "periodic_retrain", 5000)
    sym_mean  = _val(agg, "symmetric",        5000)
    box = (
        f"Compounding: {comp_mean:.1f}% +/- {comp_std:.1f}%\n"
        f"Periodic retrain: {per_mean:.1f}%\n"
        f"Symmetric: {sym_mean:.1f}%"
    )
    ax.text(
        0.03, 0.97, box, transform=ax.transAxes,
        fontsize=FONT["anno"], verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#E2E8F0", alpha=0.92),
    )

    ax.set_ylim(0, 88)
    ax.set_xlabel("Number of Decisions", fontsize=FONT["label"])
    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=FONT["label"])
    ax.set_title(
        "Scoring Matrix Convergence: Accuracy vs. Decision Count",
        fontsize=FONT["title"],
    )
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#E2E8F0")
    fig.tight_layout()
    return _save(fig, "exp1_convergence")


# ---------------------------------------------------------------------------
# Chart 2 -- Window accuracy
# ---------------------------------------------------------------------------

def chart2_window(df: pd.DataFrame) -> tuple[Path, Path]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    agg = _agg(df, "window_accuracy")
    cps = sorted(df["checkpoint"].unique())
    _draw_accuracy_lines(ax, agg, cps, marker="o")

    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of Decisions", fontsize=FONT["label"])
    ax.set_ylabel("Window Accuracy (%)", fontsize=FONT["label"])
    ax.set_title(
        "Scoring Matrix: Window Accuracy by Decision Stage",
        fontsize=FONT["title"],
    )
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#E2E8F0")
    fig.tight_layout()
    return _save(fig, "exp1_window_accuracy")


# ---------------------------------------------------------------------------
# Chart 3 -- Per-action accuracy (compounding only, 2x2)
# ---------------------------------------------------------------------------

def chart3_per_action(df: pd.DataFrame) -> tuple[Path, Path]:
    _apply_style()
    comp = df[df["method"] == "compounding"].copy()
    n    = comp["seed"].nunique()
    cps  = sorted(comp["checkpoint"].unique())

    action_cols  = [
        "action_0_accuracy", "action_1_accuracy",
        "action_2_accuracy", "action_3_accuracy",
    ]
    action_labels = [
        "auto_close", "enrich_and_watch",
        "escalate_tier2", "escalate_incident",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    for i, (col, name) in enumerate(zip(action_cols, action_labels)):
        ax  = axes[i]
        agg = comp.groupby("checkpoint")[col].agg(["mean", "std"]).reset_index()
        ci  = 1.96 * agg["std"] / np.sqrt(n)

        ax.plot(cps, agg["mean"].values,
                color="#1E3A5F", linewidth=2.0, marker="o", markersize=4,
                zorder=3)
        ax.fill_between(cps, agg["mean"] - ci, agg["mean"] + ci,
                        color="#1E3A5F", alpha=0.15, zorder=2)

        # Final value annotation
        final_val = agg.loc[agg["checkpoint"] == 5000, "mean"].values
        if len(final_val):
            ax.text(
                5000, float(final_val[0]) + 3,
                f"{float(final_val[0]):.1f}%",
                ha="center", fontsize=FONT["anno"], color="#1E3A5F",
            )

        ax.set_title(name, fontsize=FONT["label"], fontweight="bold")
        ax.set_xticks(cps)
        ax.set_xticklabels(cps, rotation=35, ha="right",
                           fontsize=FONT["tick"] - 1)
        ax.set_ylabel("Accuracy (%)", fontsize=FONT["tick"])
        ax.set_ylim(0, 108)
        _clean_ax(ax)

    fig.suptitle(
        "Per-Action Accuracy: Compounding Method  (mean +/- 95% CI, 10 seeds)",
        fontsize=FONT["title"],
    )
    fig.tight_layout()
    return _save(fig, "exp1_per_action")


# ---------------------------------------------------------------------------
# Chart 4 -- Weight evolution heatmaps
# ---------------------------------------------------------------------------

def chart4_weight_heatmap() -> tuple[Path, Path]:
    _apply_style()

    data        = np.load(NPZ_PATH)
    weights     = data["weights"]      # (n_seeds, n_checkpoints, n_actions, n_factors)
    checkpoints = data["checkpoints"]  # [50, 100, 200, 500, 1000, 2000, 5000]

    target_cps = [50, 1000, 5000]
    cp_indices  = [int(np.where(checkpoints == cp)[0][0]) for cp in target_cps]

    # Mean across seeds for each panel
    W_means = [weights[:, ci, :, :].mean(axis=0) for ci in cp_indices]

    # Symmetric colormap limits (shared across all panels)
    vmax = max(max(np.abs(W).max(), 1e-6) for W in W_means)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (W, cp) in enumerate(zip(W_means, target_cps)):
        ax = axes[i]
        im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(FACTOR_NAMES)))
        ax.set_xticklabels(FACTOR_NAMES, rotation=40, ha="right",
                           fontsize=FONT["tick"] - 1)
        ax.set_yticks(range(len(ACTION_NAMES)))
        ax.set_yticklabels(ACTION_NAMES, fontsize=FONT["tick"])
        ax.set_title(f"W at t={cp}", fontsize=FONT["label"])

        cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
        cbar.ax.tick_params(labelsize=FONT["tick"] - 1)

    fig.suptitle(
        "Weight Matrix Evolution: Compounding Method  (mean across 10 seeds)",
        fontsize=FONT["title"],
    )
    fig.tight_layout()
    return _save(fig, "exp1_weight_evolution")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(df: pd.DataFrame) -> Path:
    checkpoints = sorted(df["checkpoint"].unique())
    agg = _agg(df, "cumulative_accuracy")

    col_spec = "l" + "c" * len(checkpoints)
    header   = "Method & " + " & ".join(str(c) for c in checkpoints) + r" \\"

    rows: list[str] = []
    for method in LATEX_ORDER:
        sub   = agg[agg["method"] == method].sort_values("checkpoint")
        cells = [
            f"{row['mean']:.1f} $\\pm$ {row['std']:.1f}"
            for _, row in sub.iterrows()
        ]
        rows.append(LATEX_NAMES[method] + " & " + " & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Scoring matrix accuracy (\%) at decision checkpoints. "
         r"Mean $\pm$ std over 10 seeds. Compounding uses 20:1 asymmetric "
         r"learning with $1/t$ decay.}"),
        r"\label{tab:exp1}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = FIGURES_DIR / "exp1_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {CSV_PATH}\n"
            "Run experiments/exp1_scoring_convergence/run.py first."
        )

    print("Loading results ...")
    df = _load_data()
    print(f"  {len(df)} rows, {df['seed'].nunique()} seeds, "
          f"{df['method'].nunique()} methods")

    generated: list[Path] = []

    print("Chart 1: Convergence curves ...")
    pdf, png = chart1_convergence(df)
    generated += [pdf, png]

    print("Chart 2: Window accuracy ...")
    pdf, png = chart2_window(df)
    generated += [pdf, png]

    print("Chart 3: Per-action accuracy ...")
    pdf, png = chart3_per_action(df)
    generated += [pdf, png]

    print("Chart 4: Weight evolution heatmap ...")
    pdf, png = chart4_weight_heatmap()
    generated += [pdf, png]

    print("LaTeX table ...")
    tex = generate_latex_table(df)
    generated.append(tex)

    print("\nGenerated files:")
    for p in generated:
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            rel = p
        print(f"  {rel}")


if __name__ == "__main__":
    main()
