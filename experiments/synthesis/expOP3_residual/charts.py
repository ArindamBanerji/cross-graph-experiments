"""
EXP-OP3-RESIDUAL chart generation.

Four publication-quality charts (300 DPI, PDF + PNG):
  1. expOP3_decay_trajectories      -- R_norm vs decision for all conditions
  2. expOP3_early_warning_roc       -- ROC at W=1 for detecting harmful operator
  3. expOP3_per_category_norms      -- per-category R_norm at W=1 and W=4
  4. expOP3_diagnostic_scatter      -- R_norm(W=1) vs AUAC delta from OP2
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.viz.bridge_common import save_figure, COLORS

RESULTS_JSON = Path("experiments/synthesis/expOP3_residual/results.json")
WINDOW_SIZE  = 50

COLOR_B   = "#1E3A5F"
COLOR_C   = "#DC2626"
COLOR_P50 = "#D97706"
COLOR_Bsh = "#2563EB"
COLOR_D   = "#94A3B8"
COLOR_BL  = "#64748B"

WINDOW_X = [w * WINDOW_SIZE for w in range(8)]   # decisions 0,50,100,...,350


def generate_charts(data: dict) -> None:
    """Generate and save all four EXP-OP3 charts."""
    results  = data["per_seed_results"]
    roc      = data["roc_analysis"]
    n_seeds  = 20
    conds    = ["B", "C", "P-50", "B-short", "D-stale"]
    colors   = {
        "B":       COLOR_B,
        "C":       COLOR_C,
        "P-50":    COLOR_P50,
        "B-short": COLOR_Bsh,
        "D-stale": COLOR_D,
    }
    labels   = {
        "B":       "B — Correct (TTL=400)",
        "C":       "C — Harmful (TTL=400)",
        "P-50":    "P-50 — Partial 50% (TTL=400)",
        "B-short": "B-short — Correct (TTL=100)",
        "D-stale": "D-stale — Expires immediately",
    }

    def get_trajs(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["r_norm_trajectory"] for s in range(n_seeds)])

    # ------------------------------------------------------------------
    # CHART 1: R_norm decay trajectories
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    for cond in conds:
        trajs = get_trajs(cond)
        m = trajs.mean(axis=0)
        s = trajs.std(axis=0)
        ci = s / np.sqrt(n_seeds) * 1.96
        ax.plot(WINDOW_X, m, color=colors[cond], linewidth=2, label=labels[cond])
        ax.fill_between(WINDOW_X, m - ci, m + ci, alpha=0.10, color=colors[cond])

    ax.axhline(1.0, color="black", linestyle="-",  linewidth=1, alpha=0.4,
               label="R_norm=1.0 (no change)")
    ax.axhline(0.5, color="black", linestyle=":",  linewidth=1, alpha=0.4,
               label="R_norm=0.5 (half-decay)")
    ax.axvspan(0, 150, alpha=0.05, color=COLOR_B)
    ax.text(5, 1.38, "<- Acute phase ->", fontsize=8, color=COLOR_B, alpha=0.7)
    ax.axvline(WINDOW_SIZE, color=COLOR_BL, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(WINDOW_SIZE + 3, 1.42, "W=1\ndetection", fontsize=7, color=COLOR_BL)
    ax.set_xlabel("Decision (post-shift)")
    ax.set_ylabel("R_norm = ||R(t)||_F / ||R(0)||_F")
    ax.set_title("EXP-OP3: Residual Tracker Decay Trajectories at lambda=0.5")
    ax.set_ylim([0, 1.55])
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP3_decay_trajectories", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOP3_decay_trajectories.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: ROC curve at W=1
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    thresholds = roc["thresholds"]
    tprs       = roc["tpr"]
    fprs       = roc["fpr"]
    precs      = roc["precision"]
    rec_idx    = roc.get("recommended_threshold_idx")

    ax.plot(fprs, tprs, "o-", color=COLOR_C, linewidth=2, markersize=8, zorder=3)
    for i, thr in enumerate(thresholds):
        ax.annotate(
            f"tau={thr:.1f}\n(TPR={tprs[i]:.0%}, FPR={fprs[i]:.0%})",
            (fprs[i], tprs[i]),
            textcoords="offset points",
            xytext=(10, -12),
            fontsize=7,
            color="#374151",
        )

    if rec_idx is not None:
        ax.plot(fprs[rec_idx], tprs[rec_idx], "k*", markersize=16, zorder=4,
                label=f"Recommended: tau={thresholds[rec_idx]:.1f}")

    ax.axvline(0.20, color="grey", linestyle="--", linewidth=1, alpha=0.6,
               label="FPR=0.20 limit")
    # Random classifier diagonal
    ax.plot([0, 1], [0, 1], color="lightgrey", linestyle="--", linewidth=1)

    ax.set_xlabel("False Positive Rate (B incorrectly flagged)")
    ax.set_ylabel("True Positive Rate (C correctly detected)")
    ax.set_title("EXP-OP3: ROC — Detecting Harmful Operator at W=1\n"
                 f"(threshold on R_norm after 50 decisions)")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP3_early_warning_roc", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOP3_early_warning_roc.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: Per-category R_norm at W=1 and W=4
    # ------------------------------------------------------------------
    categories = data.get("categories", [f"cat_{i}" for i in range(5)])
    fig, axes  = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (w_label, w_key) in enumerate(
        [("W=1 (first 50 dec)", "w1"), ("W=4 (200 dec)", "w4")]
    ):
        ax    = axes[ax_idx]
        x     = np.arange(len(categories))
        width = 0.35

        for cond, color, offset in [("B", COLOR_B, -width / 2),
                                     ("C", COLOR_C, +width / 2)]:
            per_ca = np.array([
                results[cond][s][f"per_ca_norms_{w_key}"]
                for s in range(n_seeds)
            ])  # (20, C, A)
            cat_means = per_ca.mean(axis=(0, 2))               # (C,)
            cat_ci    = per_ca.std(axis=(0, 2)) / np.sqrt(n_seeds) * 1.96

            ax.bar(x + offset, cat_means, width=width,
                   color=color, alpha=0.75, label=cond)
            ax.errorbar(x + offset, cat_means, yerr=cat_ci,
                        fmt="none", color="black", capsize=3, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([c[:14] for c in categories], rotation=25, fontsize=8)
        ax.set_title(f"Per-category R_norm at {w_label}")
        if ax_idx == 0:
            ax.set_ylabel("Mean ||R[c,:,:]||_F (avg over actions & seeds)")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.4)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("EXP-OP3: Does Harm Concentrate in Specific Categories?", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "expOP3_per_category_norms", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] expOP3_per_category_norms.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 4: Diagnostic scatter — R_norm(W=1) vs AUAC delta from OP2
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    op2_path = Path("experiments/synthesis/expOP2_harmful/results.json")
    op2_auac_deltas: dict = {}
    if op2_path.exists():
        op2_data = json.load(open(op2_path))
        for cond in ["B", "C", "P-50"]:
            op2_auac_deltas[cond] = [
                op2_data["per_seed_results"][cond][s]["auac"] -
                op2_data["per_seed_results"]["A"][s]["auac"]
                for s in range(n_seeds)
            ]

    scatter_conds = [("B", COLOR_B), ("C", COLOR_C), ("P-50", COLOR_P50)]
    for cond, color in scatter_conds:
        r_w1 = [results[cond][s]["r_norm_w1"] for s in range(n_seeds)]
        if cond in op2_auac_deltas:
            y = op2_auac_deltas[cond]
            ax.scatter(r_w1, y, color=color, alpha=0.65, s=40, label=labels[cond])

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="R_norm=1.0 (detection threshold)")
    ax.axhline(0.0, color="black", linestyle="-",  linewidth=1, alpha=0.3)
    ax.set_xlabel("R_norm at W=1 (tracker signal after 50 decisions)")
    ax.set_ylabel("AUAC delta vs baseline (from EXP-OP2)")
    ax.set_title("EXP-OP3: Does R_norm(W=1) Predict Final Outcome?")
    ax.legend(fontsize=8)
    ax.text(0.04, 0.97,
            "Low R_norm + positive AUAC\n= correct op (ideal)",
            transform=ax.transAxes, fontsize=7, va="top", color=COLOR_B)
    ax.text(0.62, 0.05,
            "High R_norm + negative AUAC\n= harmful op (detect here)",
            transform=ax.transAxes, fontsize=7, va="bottom", color=COLOR_C)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP3_diagnostic_scatter", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 4] expOP3_diagnostic_scatter.png + .pdf saved")

    print("\nAll 4 charts saved to paper_figures/:")
    for name in [
        "expOP3_decay_trajectories",
        "expOP3_early_warning_roc",
        "expOP3_per_category_norms",
        "expOP3_diagnostic_scatter",
    ]:
        print(f"  {name}.png + .pdf")


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    generate_charts(data)
