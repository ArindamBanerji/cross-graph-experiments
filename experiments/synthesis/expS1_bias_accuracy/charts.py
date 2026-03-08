"""
EXP-S1 charts — 4 publication-quality charts from results.json
experiments/synthesis/expS1_bias_accuracy/charts.py

Saves each chart as both .png (dpi=150) and .pdf to:
  - experiments/synthesis/expS1_bias_accuracy/
  - paper_figures/
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

from src.viz.synthesis_common import (
    CATEGORIES, ACTIONS, FIGURE_DEFAULTS, load_results
)

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.2]


# ---------------------------------------------------------------------------
# Save helper: PNG + PDF to two directories
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, stem: str, dirs: List[str]) -> None:
    """Save fig as <stem>.png and <stem>.pdf to each directory in dirs."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(Path(d) / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart1_accuracy_by_lambda(
    agg: Dict,
    stat: Dict,
    best_lam: float,
    dirs: List[str],
) -> None:
    """Line chart: accuracy vs lambda, ±1 std error bars, gate threshold."""
    lambdas  = LAMBDA_VALUES
    means    = [agg[f"lambda_{l:.3f}"]["mean_accuracy"] for l in lambdas]
    stds     = [agg[f"lambda_{l:.3f}"]["std_accuracy"]  for l in lambdas]
    baseline = agg["lambda_0.000"]["mean_accuracy"]
    gate_thr = baseline + 3.0

    fig, ax = plt.subplots(figsize=(10, 6))
    lambdas_arr = np.array(lambdas)
    means_arr   = np.array(means)
    stds_arr    = np.array(stds)

    ax.errorbar(lambdas_arr, means_arr, yerr=stds_arr,
                fmt="o-", color="#4CAF50", linewidth=2, markersize=7,
                capsize=4, label="Mean accuracy ±1 std (10 seeds)")

    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1.5,
               label=f"Baseline (λ=0): {baseline:.2f}%")
    ax.axhline(gate_thr, color="#F44336", linestyle=":", linewidth=1.5,
               alpha=0.8, label=f"Gate threshold: {gate_thr:.2f}% (+3pp)")

    # Annotate best lambda
    best_idx = lambdas.index(best_lam)
    imp      = stat["improvement_pp"]
    p_val    = stat["p_value"]
    ax.annotate(
        f"λ={best_lam}\n{imp:+.2f}pp\np={p_val:.3f}",
        xy=(best_lam, means[best_idx]),
        xytext=(best_lam + 0.01, means[best_idx] + stds[best_idx] + 0.3),
        fontsize=8, color="#FF9800",
        arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.0),
    )

    # Gate PASS/FAIL box
    gate_ok = stat.get("improvement_pp", 0) >= 3.0 and stat.get("p_value", 1) < 0.05
    box_color = "#4CAF50" if gate_ok else "#F44336"
    box_label = "GATE: PASS" if gate_ok else "GATE: FAIL"
    ax.text(0.97, 0.04, box_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="white",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.85))

    ax.set_xlabel("Coupling constant λ", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("EXP-S1: Synthesis Bias vs Baseline Accuracy\n"
                 f"n={500} alerts, 10 seeds.  Gate: ≥3pp, p<0.05",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.22)

    # Y range: auto with 2pp padding
    y_lo = max(0, min(means_arr) - max(stds_arr) - 2)
    y_hi = min(100, max(means_arr) + max(stds_arr) + 2)
    ax.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    _save(fig, "expS1_accuracy_by_lambda", dirs)


def _chart2_category_heatmap(
    agg: Dict,
    dirs: List[str],
) -> None:
    """Heatmap: category × lambda, cell = mean accuracy %, diverging from baseline."""
    baseline_row = [agg["lambda_0.000"]["per_category_mean_accuracy"].get(c, 0.0)
                    for c in CATEGORIES]
    baseline_mean = float(np.mean(baseline_row))

    mat = np.zeros((len(CATEGORIES), len(LAMBDA_VALUES)))
    for j, lam in enumerate(LAMBDA_VALUES):
        lam_key = f"lambda_{lam:.3f}"
        cat_acc = agg[lam_key].get("per_category_mean_accuracy", {})
        for i, cat in enumerate(CATEGORIES):
            mat[i, j] = cat_acc.get(cat, 0.0)

    col_labels = [f"λ={l:.2f}" for l in LAMBDA_VALUES]

    fig, ax = plt.subplots(figsize=(10, 7))
    vmin = baseline_mean - 5
    vmax = baseline_mean + 5
    im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES, fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(mat[i, j] - baseline_mean) < 3 else "white")

    ax.set_title("EXP-S1: Per-Category Accuracy by λ", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "expS1_category_heatmap", dirs)


def _chart3_ece_by_lambda(
    agg: Dict,
    dirs: List[str],
) -> None:
    """ECE vs lambda with gate fail zone."""
    lambdas      = LAMBDA_VALUES
    mean_ece     = [agg[f"lambda_{l:.3f}"]["mean_ece"] for l in lambdas]
    std_ece      = [agg[f"lambda_{l:.3f}"]["std_ece"]  for l in lambdas]
    baseline_ece = agg["lambda_0.000"]["mean_ece"]
    gate_ceil    = baseline_ece + 0.02

    fig, ax = plt.subplots(figsize=(10, 6))
    lambdas_arr = np.array(lambdas)
    ece_arr     = np.array(mean_ece)
    std_arr     = np.array(std_ece)

    ax.errorbar(lambdas_arr, ece_arr, yerr=std_arr,
                fmt="s-", color="#2196F3", linewidth=2, markersize=7,
                capsize=4, label="Mean ECE ±1 std (10 seeds)")

    ax.axhline(baseline_ece, color="#666666", linestyle="--", linewidth=1.5,
               label=f"Baseline ECE (λ=0): {baseline_ece:.4f}")

    # Red shading for FAIL zone
    y_max = max(ece_arr.max() + std_arr.max(), gate_ceil + 0.005)
    ax.axhspan(gate_ceil, y_max + 0.01, alpha=0.15, color="#F44336",
               label=f"FAIL zone: ECE > baseline + 0.02")
    ax.axhline(gate_ceil, color="#F44336", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(max(lambdas) - 0.005, gate_ceil + 0.001,
            "Gate threshold", fontsize=8, color="#F44336", ha="right")

    ax.set_xlabel("Coupling constant λ", fontsize=11)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax.set_title("EXP-S1: Calibration (ECE) by λ\n(lower is better; gate: ≤baseline+0.02)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    _save(fig, "expS1_ece_by_lambda", dirs)


def _chart4_action_shift(
    agg: Dict,
    best_lam: float,
    dirs: List[str],
) -> None:
    """Grouped bar: action distribution at λ=0 vs best_lambda, per action."""
    baseline_dist = agg["lambda_0.000"].get("mean_action_distribution", {})
    best_key      = f"lambda_{best_lam:.3f}"
    best_dist     = agg[best_key].get("mean_action_distribution", {})

    if not baseline_dist:
        return  # No action distribution data

    acts = list(ACTIONS)
    x    = np.arange(len(acts))
    width = 0.35

    baseline_vals = [baseline_dist.get(a, 0.0) * 100 for a in acts]
    best_vals     = [best_dist.get(a, 0.0) * 100 for a in acts]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width,
           label="λ=0 (baseline)", color="#666666", alpha=0.85)
    ax.bar(x + width / 2, best_vals, width,
           label=f"λ={best_lam:.2f} (best)", color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(acts, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean fraction of alerts (%)", fontsize=11)
    ax.set_title(
        f"EXP-S1: Action Distribution Shift Under Synthesis Bias\n"
        f"λ=0 vs λ={best_lam:.2f}  (warm-start, {500} alerts × 10 seeds)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(baseline_vals), max(best_vals)) * 1.2)

    plt.tight_layout()
    _save(fig, "expS1_action_shift", dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    agg  = results["aggregated"]
    stat = results["statistical_test"]
    dirs = [exp_dir, paper_dir]

    best_lam = stat.get("best_lambda")
    if best_lam is None:
        best_lam = LAMBDA_VALUES[1]

    _chart1_accuracy_by_lambda(agg, stat, best_lam, dirs)
    _chart2_category_heatmap(agg, dirs)
    _chart3_ece_by_lambda(agg, dirs)
    _chart4_action_shift(agg, best_lam, dirs)

    print("EXP-S1: 4 charts (PNG + PDF) saved to", exp_dir, "and", paper_dir)


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    results  = load_results(str(results_path))
    exp_dir  = str(Path(__file__).parent)
    paper_dir = str(REPO_ROOT / "paper_figures")
    make_all_charts(results, exp_dir, paper_dir)
