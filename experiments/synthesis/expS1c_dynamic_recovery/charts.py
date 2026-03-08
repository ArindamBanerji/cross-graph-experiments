"""
EXP-S1c charts — 3 publication-quality charts from results.json
experiments/synthesis/expS1c_dynamic_recovery/charts.py

Chart 1: Accuracy curves over time per lambda (mean ± 1 std, 10 seeds)
Chart 2: Box plot of recovery time distribution per lambda
Chart 3: Final accuracy bar chart per lambda
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

LAMBDA_VALUES   = [0.0, 0.1, 0.2, 0.5]
WINDOW_SIZE     = 50
MAX_DECISIONS   = 2000
ACCURACY_TARGET = 90.0

LAMBDA_COLORS = {
    0.0: "#607D8B",   # blue-grey  (baseline)
    0.1: "#2196F3",   # blue
    0.2: "#4CAF50",   # green
    0.5: "#FF5722",   # deep orange
}
LAMBDA_LABELS = {lam: f"λ={lam:.1f}" + (" (base)" if lam == 0.0 else "")
                 for lam in LAMBDA_VALUES}


def _save(fig: plt.Figure, stem: str, dirs: List[str]) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(Path(d) / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: Accuracy curves — mean ± 1 std per lambda
# ---------------------------------------------------------------------------

def _chart1_accuracy_curves(agg: Dict, per_lambda: Dict, dirs: List[str]) -> None:
    n_checkpoints = MAX_DECISIONS // WINDOW_SIZE
    x = np.arange(1, n_checkpoints + 1) * WINDOW_SIZE   # 50, 100, ..., 2000

    fig, ax = plt.subplots(figsize=(13, 7))

    for lam in LAMBDA_VALUES:
        lam_key = f"lambda_{lam:.3f}"
        color   = LAMBDA_COLORS[lam]
        label   = LAMBDA_LABELS[lam]

        # Collect curves per seed (pad if any seed is short)
        curves = [r["curve"] for r in per_lambda[lam_key]]
        max_len = max(len(c) for c in curves)
        padded  = np.array([c + [c[-1]] * (max_len - len(c)) if c else [0] * max_len
                            for c in curves])  # (n_seeds, n_checkpoints)

        means = padded.mean(axis=0)
        stds  = padded.std(axis=0)

        ax.plot(x[:len(means)], means, color=color, linewidth=2.2,
                label=label, zorder=3)
        ax.fill_between(x[:len(means)],
                        means - stds, means + stds,
                        alpha=0.15, color=color, zorder=2)

        # Mark mean recovery crossing
        rd_mean = agg[lam_key]["mean_recovery"]
        if rd_mean < MAX_DECISIONS:
            ax.axvline(rd_mean, color=color, linestyle=":", linewidth=1.2,
                       alpha=0.7, zorder=1)
            ax.text(rd_mean + 15, ACCURACY_TARGET - 5,
                    f"{rd_mean:.0f}d", color=color, fontsize=8, va="top")

    # 90% target line
    ax.axhline(ACCURACY_TARGET, color="#333333", linestyle="--", linewidth=1.5,
               label=f"90% target", zorder=4)

    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("Rolling accuracy % (window=50)", fontsize=12)
    ax.set_title("EXP-S1c: Cold-Start Recovery Speed by λ\n"
                 "Loop 2 active (oracle rate=85%). Rolling window=50. Mean ±1 std across 10 seeds.",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, MAX_DECISIONS + 50)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, "expS1c_accuracy_curves", dirs)


# ---------------------------------------------------------------------------
# Chart 2: Recovery time distribution — box plots per lambda
# ---------------------------------------------------------------------------

def _chart2_recovery_distribution(agg: Dict, dirs: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    positions = np.arange(len(LAMBDA_VALUES))
    box_data  = []
    colors_list = []

    for lam in LAMBDA_VALUES:
        lam_key = f"lambda_{lam:.3f}"
        rds     = agg[lam_key]["recovery_decisions_per_seed"]
        box_data.append(rds)
        colors_list.append(LAMBDA_COLORS[lam])

    bps = ax.boxplot(box_data, positions=positions, widths=0.55,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bps["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Annotate never-recovered count and mean above each box
    for i, lam in enumerate(LAMBDA_VALUES):
        lam_key  = f"lambda_{lam:.3f}"
        n_never  = agg[lam_key]["never_recovered"]
        mean_rd  = agg[lam_key]["mean_recovery"]
        max_val  = max(agg[lam_key]["recovery_decisions_per_seed"])
        y_top    = max_val + 30
        ax.text(i, y_top, f"never={n_never}/10\nmean={mean_rd:.0f}d",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.axhline(MAX_DECISIONS, color="#999999", linestyle="--", linewidth=1.0,
               alpha=0.6, label=f"MAX={MAX_DECISIONS} (never recovered)")
    ax.set_xticks(positions)
    ax.set_xticklabels([LAMBDA_LABELS[lam] for lam in LAMBDA_VALUES], fontsize=11)
    ax.set_ylabel("Decisions to reach 90% rolling accuracy", fontsize=11)
    ax.set_title("EXP-S1c: Distribution of Recovery Time by λ\n"
                 "(MAX_DECISIONS for never-recovered. Lower is better.)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, "expS1c_recovery_distribution", dirs)


# ---------------------------------------------------------------------------
# Chart 3: Final accuracy bar chart
# ---------------------------------------------------------------------------

def _chart3_final_accuracy(agg: Dict, dirs: List[str]) -> None:
    means  = [agg[f"lambda_{lam:.3f}"]["mean_final_acc"]  for lam in LAMBDA_VALUES]
    stds   = [agg[f"lambda_{lam:.3f}"]["std_final_acc"]   for lam in LAMBDA_VALUES]
    colors = [LAMBDA_COLORS[lam]                           for lam in LAMBDA_VALUES]
    labels = [LAMBDA_LABELS[lam]                           for lam in LAMBDA_VALUES]

    fig, ax = plt.subplots(figsize=(9, 6))
    x    = np.arange(len(LAMBDA_VALUES))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, edgecolor="white")

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.3,
                f"{m:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final accuracy % (last 100 decisions)", fontsize=11)
    ax.set_ylim(0, min(100, max(means) + max(stds) + 8))
    ax.set_title("EXP-S1c: Final Accuracy After 2000 Decisions\n"
                 "Mean ±1 std (10 seeds). Cold start + Loop 2 learning.",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, "expS1c_final_accuracy", dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    agg        = results["aggregated"]
    per_lambda = results["per_lambda"]
    dirs       = [exp_dir, paper_dir]

    _chart1_accuracy_curves(agg, per_lambda, dirs)
    _chart2_recovery_distribution(agg, dirs)
    _chart3_final_accuracy(agg, dirs)

    print(f"EXP-S1c: 3 charts (PNG + PDF) saved to {exp_dir} and {paper_dir}")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    import json
    with open(results_path) as f:
        results = json.load(f)
    exp_dir   = str(Path(__file__).parent)
    paper_dir = str(REPO_ROOT / "paper_figures")
    make_all_charts(results, exp_dir, paper_dir)
