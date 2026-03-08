"""
EXP-S1b charts — 3 publication-quality charts from results.json
experiments/synthesis/expS1b_degraded_profiles/charts.py

Saves each chart as .png (dpi=150) and .pdf to both exp_dir and paper_figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.viz.synthesis_common import CATEGORIES, ACTIONS, load_results

LAMBDA_VALUES      = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
LAMBDA_NON_ZERO    = [l for l in LAMBDA_VALUES if l > 0]
COND_ORDER         = ["cold_start", "noise_50pct", "noise_25pct", "warm_start"]
COND_LABELS        = {
    "cold_start":  "Cold Start\n(random profiles)",
    "noise_50pct": "Noise 50%\n(GT + σ=0.5)",
    "noise_25pct": "Noise 25%\n(GT + σ=0.25)",
    "warm_start":  "Warm Start\n(GT profiles)",
}
COND_COLORS        = {
    "cold_start":  "#F44336",
    "noise_50pct": "#FF9800",
    "noise_25pct": "#FFC107",
    "warm_start":  "#4CAF50",
}


def _save(fig: plt.Figure, stem: str, dirs: List[str]) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(Path(d) / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: 2×2 panel — accuracy vs lambda, one panel per condition
# ---------------------------------------------------------------------------

def _chart1_accuracy_panels(agg: Dict, stat: Dict, dirs: List[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("EXP-S1b: Synthesis Bias Under Profile Degradation\n"
                 "Accuracy vs λ per degradation condition (10 seeds ±1 std)",
                 fontsize=13, fontweight="bold")

    lambdas = np.array(LAMBDA_VALUES)

    for idx, cond in enumerate(COND_ORDER):
        ax    = axes[idx // 2][idx % 2]
        color = COND_COLORS[cond]

        means = np.array([agg[cond][f"lambda_{l:.3f}"]["mean_accuracy"] for l in LAMBDA_VALUES])
        stds  = np.array([agg[cond][f"lambda_{l:.3f}"]["std_accuracy"]  for l in LAMBDA_VALUES])
        baseline = means[0]

        ax.errorbar(lambdas, means, yerr=stds, fmt="o-", color=color,
                    linewidth=2, markersize=6, capsize=4,
                    label="Mean ±1 std (10 seeds)")
        ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.3,
                   label=f"Baseline: {baseline:.1f}%")

        # Gate threshold line (+3pp)
        ax.axhline(baseline + 3.0, color="#333333", linestyle=":", linewidth=1.0,
                   alpha=0.7, label="+3pp gate")

        # Mark any significant lambda values
        if cond != "warm_start" and cond in stat:
            for lam in LAMBDA_NON_ZERO:
                lam_key = f"lambda_{lam:.3f}"
                if lam_key in stat[cond]:
                    s = stat[cond][lam_key]
                    if s["p_value"] < 0.05:
                        lam_idx = LAMBDA_VALUES.index(lam)
                        ax.plot(lam, means[lam_idx], "*", color="#FFD700",
                                markersize=14, zorder=5, label=f"p<0.05")

        ax.set_title(COND_LABELS[cond], fontsize=11, fontweight="bold")
        ax.set_xlabel("λ", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

        # Auto y-range with 2pp padding
        y_lo = max(0, float(means.min()) - float(stds.max()) - 2)
        y_hi = min(100, float(means.max()) + float(stds.max()) + 3)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.02, 0.53)

        # Legend only on first panel
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    _save(fig, "expS1b_accuracy_vs_lambda_by_condition", dirs)


# ---------------------------------------------------------------------------
# Chart 2: Improvement heatmap — condition × lambda (non-zero only)
# ---------------------------------------------------------------------------

def _chart2_improvement_heatmap(
    agg: Dict,
    stat: Dict,
    dirs: List[str],
) -> None:
    n_conds  = len(COND_ORDER)
    n_lams   = len(LAMBDA_NON_ZERO)

    imp_mat  = np.zeros((n_conds, n_lams))
    p_mat    = np.ones((n_conds, n_lams))

    for i, cond in enumerate(COND_ORDER):
        for j, lam in enumerate(LAMBDA_NON_ZERO):
            lam_key = f"lambda_{lam:.3f}"
            if cond in stat and lam_key in stat[cond]:
                imp_mat[i, j] = stat[cond][lam_key]["improvement_pp"]
                p_mat[i, j]   = stat[cond][lam_key]["p_value"]

    abs_max = max(abs(imp_mat).max(), 0.1)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(imp_mat, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Improvement (pp)")

    col_labels = [f"λ={l:.2f}" for l in LAMBDA_NON_ZERO]
    row_labels = [COND_LABELS[c].replace("\n", " ") for c in COND_ORDER]

    ax.set_xticks(range(n_lams))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_conds))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(n_conds):
        for j in range(n_lams):
            v   = imp_mat[i, j]
            p   = p_mat[i, j]
            sig = p < 0.05
            txt = f"{v:+.2f}pp\np={p:.3f}"
            fw  = "bold" if sig else "normal"
            col = "black" if abs(v) < abs_max * 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, fontweight=fw, color=col)

    ax.set_title(
        "EXP-S1b: Synthesis Improvement (pp) by Degradation × λ\n"
        "Bold = p<0.05.  Gate: >=3pp in any degraded condition.",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "expS1b_improvement_heatmap", dirs)


# ---------------------------------------------------------------------------
# Chart 3: Baseline accuracy by degradation condition
# ---------------------------------------------------------------------------

def _chart3_baseline_bars(agg: Dict, dirs: List[str]) -> None:
    baselines = [
        agg[cond]["lambda_0.000"]["mean_accuracy"]
        for cond in COND_ORDER
    ]
    stds = [
        agg[cond]["lambda_0.000"]["std_accuracy"]
        for cond in COND_ORDER
    ]
    labels = [COND_LABELS[c].replace("\n", " ") for c in COND_ORDER]
    colors = [COND_COLORS[c] for c in COND_ORDER]

    # Sort by accuracy descending
    order   = sorted(range(len(baselines)), key=lambda i: -baselines[i])
    baselines_s = [baselines[i] for i in order]
    stds_s      = [stds[i]      for i in order]
    labels_s    = [labels[i]    for i in order]
    colors_s    = [colors[i]    for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(COND_ORDER))
    bars = ax.bar(x, baselines_s, yerr=stds_s, capsize=5,
                  color=colors_s, alpha=0.85, edgecolor="white")

    for bar, val, std in zip(bars, baselines_s, stds_s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_s, fontsize=10)
    ax.set_ylabel("Baseline Accuracy % (λ=0)", fontsize=11)
    ax.set_ylim(0, min(100, max(baselines_s) + max(stds_s) + 8))
    ax.set_title(
        "EXP-S1b: Profile Degradation Impact on Baseline Accuracy\n"
        "(Shows the gap σ has available to bridge per condition)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate gap from warm_start
    warm_acc = agg["warm_start"]["lambda_0.000"]["mean_accuracy"]
    for i, (acc, label) in enumerate(zip(baselines_s, labels_s)):
        if "Warm" not in label:
            gap = warm_acc - acc
            ax.annotate(
                f"gap\n-{gap:.1f}pp",
                xy=(x[i], acc),
                xytext=(x[i], acc - gap / 2 - 2),
                ha="center", fontsize=8, color="#555555",
            )

    plt.tight_layout()
    _save(fig, "expS1b_degradation_vs_baseline", dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    agg  = results["aggregated"]
    stat = results.get("statistical", {})
    dirs = [exp_dir, paper_dir]

    _chart1_accuracy_panels(agg, stat, dirs)
    _chart2_improvement_heatmap(agg, stat, dirs)
    _chart3_baseline_bars(agg, dirs)

    print(f"EXP-S1b: 3 charts (PNG + PDF) saved to {exp_dir} and {paper_dir}")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    results   = load_results(str(results_path))
    exp_dir   = str(Path(__file__).parent)
    paper_dir = str(REPO_ROOT / "paper_figures")
    make_all_charts(results, exp_dir, paper_dir)
