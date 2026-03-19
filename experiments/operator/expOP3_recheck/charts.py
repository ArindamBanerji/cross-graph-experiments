"""
EXP-OP3-RECHECK Charts

Chart 1 — expOP3_recheck_residual_trajectory:
  Mean R(t) per condition (A=baseline, B=correct, C=harmful) over 8 windows.
  Shaded ±1 std band. Detection threshold line. Gate annotations.

Chart 2 — expOP3_recheck_roc:
  ROC curve for B vs C discrimination using R(W=1).
  AUC annotation. Gate threshold operating point marked.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure

_THIS_DIR    = Path(__file__).parent
RESULTS_FILE = _THIS_DIR / "results.json"

COND_COLORS = {
    "A": "#9e9e9e",    # gray — baseline
    "B": "#2e7d32",    # green — correct
    "C": "#c62828",    # red — harmful
}
COND_LABELS = {
    "A": "A — No operator (baseline, 10% noise)",
    "B": "B — Correct operator (100% truthful)",
    "C": "C — Harmful operator (0% correct, always wrong)",
}


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — Residual trajectory
# ============================================================================

def chart1_trajectory(r: dict) -> None:
    trajs = r["trajectories"]
    gate  = r["gate"]
    n_w   = r["n_decisions"] // r["window_size"]

    x = np.arange(n_w + 1) * r["window_size"]   # decision count at each window

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(bottom=0.16, top=0.88)

    for cond in ["A", "B", "C"]:
        mean = np.array(trajs[cond]["mean"])
        std  = np.array(trajs[cond]["std"])
        color = COND_COLORS[cond]
        lw    = 2.4 if cond == "C" else 2.0

        ax.plot(x, mean, color=color, lw=lw, label=COND_LABELS[cond], zorder=4)
        ax.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.12, zorder=3)

    # Threshold line
    thr = gate["threshold"]
    ax.axhline(thr, color="#e65100", lw=1.5, ls="--", zorder=5,
               label=f"Detection threshold R={thr:.4f}")

    # Vertical line at W=1 (first 50 decisions)
    w1_x = r["window_size"]
    ax.axvline(w1_x, color="black", lw=1.0, ls=":", alpha=0.5, zorder=2,
               label=f"W=1 ({w1_x} decisions)")

    # Annotations
    ax.text(
        w1_x + 3, thr + 0.0003,
        f"θ = {thr:.4f}\n"
        f"C det: {gate['detect_rate']:.0%}\n"
        f"B FA:  {gate['fa_rate']:.0%}",
        fontsize=8.5, color="#e65100", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#e65100", alpha=0.88),
    )

    # Gate verdict
    gate_str = "GATE PASS" if gate["pass"] else "GATE FAIL"
    gate_color = "#2e7d32" if gate["pass"] else "#c62828"
    ax.text(
        0.99, 0.97,
        f"{gate_str}\nAUC={gate['auc']:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold", color=gate_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=gate_color, linewidth=1.5, alpha=0.92),
    )

    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("R(t) = ‖μ(t) − μ(t−W)‖_F   (centroid drift per window)", fontsize=11)
    ax.set_xlim(-5, x[-1] + 15)
    ax.set_ylim(bottom=-0.0005)
    ax.set_xticks(x)

    ax.set_title(
        "EXP-OP3-RECHECK: Residual Tracker Early-Warning (η_neg=0.05, A=4)\n"
        "R(t) = centroid drift in last W=50 decisions. Threshold separates B (correct) from C (harmful).",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92, ncol=1)

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). "
        f"{r['n_seeds']} seeds, warmup={r['n_warmup']}, "
        f"decisions={r['n_decisions']}, "
        f"τ={r['tau']}, η_neg={r['eta_neg']}. "
        "Shaded band = ±1 std."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "expOP3_recheck_residual_trajectory", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOP3_recheck_residual_trajectory.png + .pdf saved")


# ============================================================================
# Chart 2 — ROC curve
# ============================================================================

def chart2_roc(r: dict) -> None:
    roc   = r["roc"]
    gate  = r["gate"]
    fprs  = np.array(roc["fprs"])
    tprs  = np.array(roc["tprs"])
    auc   = roc["auc"]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(bottom=0.14, top=0.88)

    # ROC curve
    ax.plot(fprs, tprs, color="#1565c0", lw=2.5, label=f"R(W=1)  AUC = {auc:.3f}",
            zorder=4)
    ax.fill_between(fprs, tprs, alpha=0.08, color="#1565c0", zorder=3)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="#999", lw=1.2, ls="--", zorder=2,
            label="Random (AUC=0.50)")

    # Operating point (gate threshold)
    op_fpr = gate["fa_rate"]
    op_tpr = gate["detect_rate"]
    ax.scatter([op_fpr], [op_tpr], s=160, color="#e65100",
               zorder=6, edgecolors="black", linewidth=0.8,
               label=f"Operating point\nDet={op_tpr:.0%}, FA={op_fpr:.0%}")
    ax.annotate(
        f"θ={gate['threshold']:.4f}\nDet={op_tpr:.0%}\nFA={op_fpr:.0%}",
        xy=(op_fpr, op_tpr),
        xytext=(op_fpr + 0.08, op_tpr - 0.10),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                  edgecolor="#e65100", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.2),
    )

    # Gate lines
    ax.axhline(gate["detect_rate_gate"], color="#2e7d32", lw=1.0, ls=":",
               alpha=0.7, label=f"Detection gate = {gate['detect_rate_gate']:.0%}")
    ax.axvline(gate["fa_rate_gate"],     color="#c62828", lw=1.0, ls=":",
               alpha=0.7, label=f"FA gate = {gate['fa_rate_gate']:.0%}")

    # AUC gate annotation
    gate_str  = "GATE PASS" if gate["pass"] else "GATE FAIL"
    gate_color = "#2e7d32" if gate["pass"] else "#c62828"
    ax.text(
        0.98, 0.05,
        f"{gate_str}\nAUC={auc:.3f} (gate ≥ {gate['auc_gate']:.2f})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10.5, fontweight="bold", color=gate_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=gate_color, linewidth=1.5, alpha=0.92),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate  (B false-alarmed)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (C detected)", fontsize=12)
    ax.set_title(
        "EXP-OP3-RECHECK: ROC — R(W=1) as B vs C Discriminator\n"
        "Discriminating harmful operator (C) from correct operator (B) at first window.",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92)

    caption = (
        f"soc_product_v50 A=4, {r['n_seeds']} seeds. "
        "B = correct operator, C = harmful operator (0% correct). "
        f"R(W=1) = centroid drift after first {r['window_size']} decisions."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "expOP3_recheck_roc", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOP3_recheck_roc.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_trajectory(r)
    chart2_roc(r)
