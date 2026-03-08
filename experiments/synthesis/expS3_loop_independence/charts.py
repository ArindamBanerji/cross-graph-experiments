"""
EXP-S3 charts — 3 charts from results.json
experiments/synthesis/expS3_loop_independence/charts.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.viz.synthesis_common import (
    CATEGORIES, FIGURE_DEFAULTS, load_results
)

SNAPSHOT_STEPS = [50, 100, 150, 200, 250, 300]


def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    for d in [exp_dir, paper_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for d in [exp_dir, paper_dir]:
            fig.savefig(Path(d) / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    agg = results["aggregated"]

    # --- Chart 1: Frobenius Divergence over time ---
    fig, ax = plt.subplots(figsize=FIGURE_DEFAULTS["figsize_standard"])
    snap = agg["snapshot_mean_relative_diffs"]
    steps = sorted(int(k) for k in snap.keys())
    vals  = [snap[str(s)] if str(s) in snap else snap.get(s, 0.0) for s in steps]

    ax.plot(steps, vals, "o-", color="#2196F3", linewidth=2, markersize=7,
            label="Mean relative Frobenius diff (10 seeds)")
    ax.axhline(0.05, color="#F44336", linestyle="--", linewidth=2,
               label="Gate threshold (5%)")
    ax.fill_between(steps, vals, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Training decisions", fontsize=11)
    ax.set_ylabel("Relative Frobenius: ||μ_A − μ_B||_F / ||μ_B||_F", fontsize=10)
    ax.set_title("EXP-S3: Centroid Divergence Over Time\n"
                 "(With Synthesis vs Without Synthesis)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    gate_pass = agg["mean_relative_frobenius"] <= 0.05
    verdict = "✅ PASS" if gate_pass else "❌ FAIL"
    ax.text(0.98, 0.95, f"Final: {agg['mean_relative_frobenius']:.4f} {verdict}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color="#4CAF50" if gate_pass else "#F44336",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    save(fig, "expS3_frobenius_divergence.png")

    # --- Chart 2: Centroids-alone accuracy (paired bar) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    acc_with    = agg["mean_acc_with_synthesis_centroids"]
    acc_without = agg["mean_acc_without_synthesis_centroids"]
    bars = ax.bar(
        ["Centroids from\nwith-synthesis run", "Centroids from\nwithout-synthesis run"],
        [acc_with, acc_without],
        color=["#FF9800", "#4CAF50"], alpha=0.85, width=0.5
    )
    for bar, val in zip(bars, [acc_with, acc_without]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    diff = abs(acc_with - acc_without)
    ax.set_ylabel("Accuracy (%) with λ=0 scoring", fontsize=10)
    ax.set_title("EXP-S3: Centroid Quality — Independent of Synthesis\n"
                 f"(Both sets scored at λ=0. Diff={diff:.2f}pp, threshold=1pp)",
                 fontweight="bold")
    gate_acc = diff <= 1.0
    color_verdict = "#4CAF50" if gate_acc else "#F44336"
    ax.text(0.98, 0.95, f"Diff: {diff:.2f}pp {'✅' if gate_acc else '❌'}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color=color_verdict,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_ylim(max(0, min(acc_with, acc_without) - 5), 102)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, "expS3_centroids_alone_accuracy.png")

    # --- Chart 3: Per-seed Frobenius distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    per_seed = agg["per_seed_relative_diffs"]
    seeds = list(range(1, len(per_seed) + 1))
    ax.bar(seeds, per_seed, color=["#4CAF50" if v <= 0.05 else "#F44336" for v in per_seed],
           alpha=0.8)
    ax.axhline(0.05, color="#F44336", linestyle="--", linewidth=2, label="Gate threshold")
    ax.axhline(np.mean(per_seed), color="#2196F3", linestyle="-", linewidth=2,
               label=f"Mean = {np.mean(per_seed):.4f}")
    ax.set_xlabel("Seed index", fontsize=10); ax.set_ylabel("Relative Frobenius diff", fontsize=10)
    ax.set_title("EXP-S3: Per-Seed Relative Frobenius Divergence", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, "expS3_centroid_trajectory.png")

    print("EXP-S3: 3 charts saved to", exp_dir, "and", paper_dir)


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    results = load_results(str(results_path))
    make_all_charts(results, str(Path(__file__).parent), str(REPO_ROOT / "paper_figures"))
