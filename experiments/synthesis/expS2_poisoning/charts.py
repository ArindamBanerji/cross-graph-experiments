"""
EXP-S2 charts — 3 charts from results.json
experiments/synthesis/expS2_poisoning/charts.py
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

from src.viz.synthesis_common import (
    POISON_COLORS, FIGURE_DEFAULTS, load_results, ACTIONS, CATEGORIES
)

CONDITION_LABELS = {
    "clean_0pct":   "Clean (0% poison)",
    "poison_20pct": "20% poisoned",
    "poison_40pct": "40% poisoned",
}


def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    for d in [exp_dir, paper_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for d in [exp_dir, paper_dir]:
            fig.savefig(Path(d) / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    conditions = results["conditions"]
    summary = results["summary"]
    labels_ordered = ["clean_0pct", "poison_20pct", "poison_40pct"]

    # --- Chart 1: Accuracy comparison (violin/box + bar) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EXP-S2: Poisoning Resilience — Accuracy Under Different Poison Rates",
                 fontweight="bold", fontsize=12)

    # Left: Bar chart mean ± std
    ax = axes[0]
    means = [conditions[l]["mean_accuracy"] for l in labels_ordered]
    stds  = [conditions[l]["std_accuracy"]  for l in labels_ordered]
    colors = [POISON_COLORS.get(l, "#999") for l in labels_ordered]
    x = np.arange(len(labels_ordered))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[l] for l in labels_ordered], fontsize=9)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=10)
    ax.axhline(means[0], color="#333", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(max(0, min(means) - 10), 102)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Mean Accuracy ± Std", fontsize=10)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: Degradation bars
    ax2 = axes[1]
    degradations = [0.0,
                    summary["degradation_20pct"],
                    summary["degradation_40pct"]]
    bars2 = ax2.bar(x, degradations, color=colors, alpha=0.85)
    ax2.axhline(2.0, color="#F44336", linestyle="--", linewidth=1.5,
                label="Gate threshold (2pp)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([CONDITION_LABELS[l] for l in labels_ordered], fontsize=9)
    ax2.set_ylabel("Accuracy Degradation (pp)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title("Degradation vs Clean Baseline", fontsize=10)
    for bar, deg in zip(bars2, degradations):
        ax2.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.03,
                 f"{deg:.2f}pp", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save(fig, "expS2_poisoning_accuracy.png")

    # --- Chart 2: Per-seed accuracy distributions ---
    fig, ax = plt.subplots(figsize=(10, 6))
    all_seed_accs = []
    for l in labels_ordered:
        accs = [r["accuracy"] for r in conditions[l]["per_seed"]]
        all_seed_accs.append(accs)

    bp = ax.boxplot(all_seed_accs, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, l in zip(bp["boxes"], labels_ordered):
        patch.set_facecolor(POISON_COLORS.get(l, "#999"))
        patch.set_alpha(0.7)

    ax.set_xticklabels([CONDITION_LABELS[l] for l in labels_ordered], fontsize=10)
    ax.set_ylabel("Accuracy (%) across 10 seeds", fontsize=10)
    ax.set_title("EXP-S2: Accuracy Distribution by Poison Rate (10 seeds)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, "expS2_seed_distribution.png")

    # --- Chart 3: Safety effectiveness summary ---
    fig, ax = plt.subplots(figsize=(8, 5))
    safety = summary["safety_effectiveness"]
    ax.barh(["Safety\nEffectiveness"], [safety], color="#4CAF50" if safety >= 0.5 else "#F44336",
            alpha=0.85)
    ax.axvline(0.5, color="#F44336", linestyle="--", linewidth=1.5, label="Gate threshold (0.50)")
    ax.set_xlim(0, 1.0)
    ax.text(safety + 0.01, 0, f"{safety:.3f}", va="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Safety Effectiveness Score", fontsize=10)
    ax.set_title("EXP-S2: Safety Effectiveness\n(resilience to 20% poisoning)", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "expS2_safety_effectiveness.png")

    print("EXP-S2: 3 charts saved to", exp_dir, "and", paper_dir)


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    results = load_results(str(results_path))
    make_all_charts(results, str(Path(__file__).parent), str(REPO_ROOT / "paper_figures"))
