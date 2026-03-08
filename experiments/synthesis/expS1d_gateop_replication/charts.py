"""
EXP-S1d charts — 3 publication-quality charts from results.json
experiments/synthesis/expS1d_gateop_replication/charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.category_alert_generator import CATEGORIES

LAMBDA_VALUES     = [0.0, 0.1, 0.2, 0.3, 0.5]
LAMBDA_NONZERO    = [0.1, 0.2, 0.3, 0.5]
GATEOP_DELTA_PP   = 0.41    # GATE-OP reference delta (pp)
GATE_THRESHOLD_PP = 0.3     # S1d gate threshold (pp)

LAMBDA_COLORS = {
    0.1: "#2196F3",
    0.2: "#4CAF50",
    0.3: "#FF9800",
    0.5: "#F44336",
}


def _save(fig: plt.Figure, stem: str, dirs: List[str]) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(Path(d) / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: Delta by lambda — with GATE-OP reference lines
# ---------------------------------------------------------------------------

def _chart1_delta_by_lambda(
    agg: Dict, stat: Dict, gate: Dict, dirs: List[str]
) -> None:
    baseline_accs = np.array(agg["lambda_0.000"]["per_seed_accs"])

    deltas = []
    stds   = []
    p_vals = []
    for lam in LAMBDA_NONZERO:
        lam_key  = f"lambda_{lam:.3f}"
        tr_accs  = np.array(agg[lam_key]["per_seed_accs"])
        deltas.append(float(np.mean(tr_accs - baseline_accs)) * 100.0)
        stds.append(float(np.std(tr_accs - baseline_accs)) * 100.0)
        p_vals.append(stat[lam_key]["p_value"])

    x      = np.arange(len(LAMBDA_NONZERO))
    colors = [LAMBDA_COLORS[l] for l in LAMBDA_NONZERO]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.bar(x, deltas, yerr=stds, capsize=5, color=colors, alpha=0.82,
                  edgecolor="white", width=0.6)

    # Mark significant bars
    for i, (delta, p) in enumerate(zip(deltas, p_vals)):
        if p < 0.05:
            ax.text(i, delta + stds[i] + 0.008,
                    "* p<0.05", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")
        ax.text(i, delta - stds[i] - 0.025,
                f"{delta:+.3f}pp\np={p:.3f}",
                ha="center", va="top", fontsize=8, color="#555555")

    # Reference lines
    ax.axhline(0,                    color="#666666", linewidth=1.5, linestyle="-")
    ax.axhline(GATE_THRESHOLD_PP,    color="#4CAF50", linewidth=1.5, linestyle=":",
               label=f"+{GATE_THRESHOLD_PP:.1f}pp gate threshold")
    ax.axhline(GATEOP_DELTA_PP,      color="#FF5722", linewidth=1.5, linestyle="--",
               label=f"+{GATEOP_DELTA_PP:.2f}pp GATE-OP reference")

    # Gate verdict box
    gate_pass = gate.get("S1d_passed", False)
    verdict   = "PASS" if gate_pass else "FAIL"
    box_color = "#C8E6C9" if gate_pass else "#FFCDD2"
    ax.text(0.98, 0.97, f"GATE-S1d: {verdict}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([f"lambda={l:.1f}" for l in LAMBDA_NONZERO], fontsize=11)
    ax.set_ylabel("delta accuracy (pp) vs lambda=0 baseline", fontsize=11)
    ax.set_title(
        "EXP-S1d: sigma Effect on Warm-Profile Accuracy (GATE-OP Replication)\n"
        f"500 warmup decisions + 500 eval decisions. Loop 2 active. n=10 seeds.",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper left")
    ax.tick_params(labelsize=10)

    # Zero line label
    ax.text(-0.45, 0.012, "no effect", fontsize=8, color="#666666", va="bottom")

    plt.tight_layout()
    _save(fig, "expS1d_delta_by_lambda", dirs)


# ---------------------------------------------------------------------------
# Chart 2: Per-category sigma effect heatmap
# ---------------------------------------------------------------------------

def _chart2_category_heatmap(agg: Dict, stat: Dict, dirs: List[str]) -> None:
    n_cats = len(CATEGORIES)
    n_lams = len(LAMBDA_NONZERO)

    delta_mat = np.zeros((n_cats, n_lams))
    p_mat     = np.ones((n_cats, n_lams))

    baseline_per_cat = np.array([
        agg["lambda_0.000"]["per_cat_means"].get(cat, 0.0)
        for cat in CATEGORIES
    ])

    for j, lam in enumerate(LAMBDA_NONZERO):
        lam_key = f"lambda_{lam:.3f}"
        for i, cat in enumerate(CATEGORIES):
            tr_mean  = agg[lam_key]["per_cat_means"].get(cat, 0.0)
            delta_mat[i, j] = (tr_mean - baseline_per_cat[i]) * 100.0

    abs_max = max(abs(delta_mat).max(), 0.1)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(delta_mat, cmap="RdYlGn",
                   vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="delta accuracy (pp)")

    ax.set_xticks(range(n_lams))
    ax.set_xticklabels([f"lambda={l:.1f}" for l in LAMBDA_NONZERO], fontsize=10)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(CATEGORIES, fontsize=10)

    for i in range(n_cats):
        for j in range(n_lams):
            v   = delta_mat[i, j]
            col = "white" if abs(v) > abs_max * 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}pp",
                    ha="center", va="center",
                    fontsize=8.5, color=col)

    ax.set_title(
        "EXP-S1d: Per-Category sigma Effect by lambda\n"
        "(delta accuracy pp vs lambda=0 baseline)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "expS1d_category_heatmap", dirs)


# ---------------------------------------------------------------------------
# Chart 3: GATE-OP vs S1d comparison at lambda=0.5
# ---------------------------------------------------------------------------

def _chart3_gateop_comparison(agg: Dict, stat: Dict, dirs: List[str]) -> None:
    # Extract S1d results at lambda=0.5
    lam_key   = "lambda_0.500"
    baseline_accs  = np.array(agg["lambda_0.000"]["per_seed_accs"])
    treatment_accs = np.array(agg[lam_key]["per_seed_accs"])
    per_seed_deltas = (treatment_accs - baseline_accs) * 100.0

    s1d_mean = float(np.mean(per_seed_deltas))
    s1d_std  = float(np.std(per_seed_deltas))

    fig, ax = plt.subplots(figsize=(8, 7))

    # GATE-OP bar (single value, no error bar — direct sigma assignment)
    bar_gateop = ax.bar([0], [GATEOP_DELTA_PP], color="#FF5722", alpha=0.8,
                        width=0.5, edgecolor="white", label="GATE-OP (direct sigma)")
    ax.text(0, GATEOP_DELTA_PP + 0.008,
            f"+{GATEOP_DELTA_PP:.2f}pp\np=0.0008",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # S1d bar (mean ± std over 10 seeds)
    s1d_color = "#4CAF50" if s1d_mean >= GATE_THRESHOLD_PP else "#2196F3"
    bar_s1d = ax.bar([1], [s1d_mean], yerr=[s1d_std], capsize=6,
                     color=s1d_color, alpha=0.8, width=0.5, edgecolor="white",
                     label="EXP-S1d (claim chain, lambda=0.5)")
    p_s1d   = stat[lam_key]["p_value"]
    ax.text(1, s1d_mean + s1d_std + 0.008,
            f"{s1d_mean:+.2f}pp\np={p_s1d:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reference lines
    ax.axhline(0,                  color="#666666", linewidth=1.2, linestyle="-")
    ax.axhline(GATE_THRESHOLD_PP,  color="#888888", linewidth=1.2, linestyle=":",
               label=f"+{GATE_THRESHOLD_PP:.1f}pp gate threshold")
    ax.axhline(GATEOP_DELTA_PP,    color="#FF5722", linewidth=1.0, linestyle="--",
               alpha=0.5, label="GATE-OP level")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GATE-OP\n(direct sigma,\nlambda=0.5)",
                         "EXP-S1d\n(claim chain,\nlambda=0.5)"],
                        fontsize=11)
    ax.set_ylabel("delta accuracy (pp) vs baseline", fontsize=11)
    ax.set_title(
        "EXP-S1d: GATE-OP Replication Comparison (lambda=0.5)\n"
        "GATE-OP used direct sigma assignment. S1d uses full claim chain.",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=10)

    # Percentage difference annotation
    ratio = s1d_mean / GATEOP_DELTA_PP if GATEOP_DELTA_PP != 0 else 0
    ax.text(0.5, 0.05,
            f"S1d/GATE-OP ratio: {ratio:.0%}\n"
            f"(claim chain vs direct sigma)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    _save(fig, "expS1d_warmup_vs_gateop", dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    agg  = results["aggregated"]
    stat = results["statistical"]
    gate = results["gate"]
    dirs = [exp_dir, paper_dir]

    _chart1_delta_by_lambda(agg, stat, gate, dirs)
    _chart2_category_heatmap(agg, stat, dirs)
    _chart3_gateop_comparison(agg, stat, dirs)

    print(f"EXP-S1d: 3 charts (PNG + PDF) saved to {exp_dir} and {paper_dir}")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    with open(results_path) as f:
        results = json.load(f)
    exp_dir   = str(Path(__file__).parent)
    paper_dir = str(REPO_ROOT / "paper_figures")
    make_all_charts(results, exp_dir, paper_dir)
