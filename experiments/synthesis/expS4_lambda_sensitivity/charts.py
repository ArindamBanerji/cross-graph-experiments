"""
EXP-S4 charts — 2 charts from results.json
experiments/synthesis/expS4_lambda_sensitivity/charts.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.viz.synthesis_common import FIGURE_DEFAULTS, load_results, CATEGORIES


def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    for d in [exp_dir, paper_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for d in [exp_dir, paper_dir]:
            fig.savefig(Path(d) / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    lambdas = results["lambda_sweep"]
    means   = results["per_lambda_mean_accuracy"]
    stds    = results["per_lambda_std_accuracy"]
    baseline = results["baseline_mean_accuracy"]
    plateau  = results["plateau"]

    # --- Chart 1: Accuracy vs Lambda (main) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    means_arr = np.array(means)
    stds_arr  = np.array(stds)
    lambdas_arr = np.array(lambdas)

    ax.plot(lambdas_arr, means_arr, "o-", color="#4CAF50", linewidth=2.5,
            markersize=7, label="Mean accuracy (10 seeds)", zorder=3)
    ax.fill_between(lambdas_arr, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.18, color="#4CAF50", label="±1 std")

    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1.5,
               label=f"Baseline (λ=0): {baseline:.2f}%")
    gate_line = baseline + 2.0
    ax.axhline(gate_line, color="#FF9800", linestyle=":", linewidth=1.5,
               label=f"Plateau threshold (+2pp): {gate_line:.2f}%", alpha=0.85)

    # Shade plateau
    pl = plateau.get("plateau_lambdas", [])
    if pl:
        ax.axvspan(min(pl), max(pl), alpha=0.15, color="#4CAF50",
                   label=f"Plateau: [{min(pl):.3f}, {max(pl):.3f}] "
                         f"(width={plateau['plateau_width']:.3f})")

    # Mark peak
    lp = plateau.get("lambda_peak")
    if lp is not None:
        ax.axvline(lp, color="#FF9800", linestyle="--", linewidth=1.5,
                   label=f"λ_peak = {lp:.3f}", alpha=0.9)

    gate_pass = plateau["plateau_width"] >= 0.05
    verdict = "✅ PASS" if gate_pass else "❌ FAIL"
    ax.text(0.98, 0.05,
            f"Plateau width: {plateau['plateau_width']:.3f}  {verdict}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            color="#4CAF50" if gate_pass else "#F44336",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_xlabel("Coupling constant λ", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.set_ylabel("Accuracy (%)", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.set_title("EXP-S4: Accuracy vs Coupling Constant λ\n"
                 "(Is there a stable operating region?)", fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, max(lambdas) + 0.01)
    plt.tight_layout()
    save(fig, "expS4_accuracy_vs_lambda.png")

    # --- Chart 2: Per-lambda improvement over baseline (bar) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    improvements = [m - baseline for m in means]
    colors = ["#4CAF50" if imp >= 2.0 else "#FF9800" if imp >= 0 else "#F44336"
              for imp in improvements]
    ax.bar(range(len(lambdas)), improvements, color=colors, alpha=0.85)
    ax.axhline(2.0, color="#FF9800", linestyle="--", linewidth=1.5,
               label=f"Plateau threshold (+2pp)")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.3f}" for l in lambdas], rotation=45, fontsize=8)
    ax.set_xlabel("λ value", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.set_ylabel("Improvement over baseline (pp)", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.set_title("EXP-S4: Per-λ Improvement vs Baseline\n"
                 "(Green = in plateau, orange = positive but below threshold)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, "expS4_per_category_optimal.png")

    print("EXP-S4: 2 charts saved to", exp_dir, "and", paper_dir)


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    results = load_results(str(results_path))
    make_all_charts(results, str(Path(__file__).parent), str(REPO_ROOT / "paper_figures"))
