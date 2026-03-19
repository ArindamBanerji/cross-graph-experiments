"""
FX-T5-A4 Charts

Chart 1 — fx_t5_a4_per_action_accuracy:
  Grouped bar chart: per-action accuracy A=5 vs A=4.
  Highlights monitor improvement (or lack thereof).
  99% target line for CAUTION actions.

Chart 2 — fx_t5_a4_band_distribution:
  Stacked bar chart: action distribution within the auto-approve band, A=5 vs A=4.
  Shows how the band composition changed with A=4 geometry.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure

_THIS_DIR    = Path(__file__).parent
RESULTS_FILE = _THIS_DIR / "results.json"

ACTIONS = ["escalate", "investigate", "suppress", "monitor"]
ACT_COLORS_A4 = ["#1565c0", "#2e7d32", "#f57f17", "#c62828"]
ACT_COLORS_A5 = ["#90caf9", "#a5d6a7", "#ffe082", "#ef9a9a"]   # lighter for A=5

THREAT_ACTIONS  = {0, 1}
CAUTION_ACTIONS = {2, 3}


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — Per-action accuracy: A=5 vs A=4
# ============================================================================

def chart1_accuracy(r: dict) -> None:
    a5_ref = r["a5_reference"]
    q2_a4  = r["q2_per_action_accuracy"]

    acc5 = [a5_ref[act]["accuracy"]  for act in ACTIONS]
    acc4 = [q2_a4[act]["accuracy"]   for act in ACTIONS]
    ci_lo = [q2_a4[act]["ci_lo"]     for act in ACTIONS]
    ci_hi = [q2_a4[act]["ci_hi"]     for act in ACTIONS]
    yerr  = [[acc4[i] - ci_lo[i] for i in range(4)],
             [ci_hi[i] - acc4[i] for i in range(4)]]

    x       = np.arange(len(ACTIONS))
    bar_w   = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(bottom=0.16, top=0.88)

    bars5 = ax.bar(x - bar_w / 2, acc5, bar_w, color=ACT_COLORS_A5,
                   edgecolor="#555", linewidth=0.8, label="A=5 (FX-T5, realistic gen)",
                   alpha=0.82, zorder=4)
    bars4 = ax.bar(x + bar_w / 2, acc4, bar_w, color=ACT_COLORS_A4,
                   edgecolor="black", linewidth=0.8, label="A=4 (this run, v50 gen)",
                   alpha=0.88, zorder=4)
    ax.errorbar(x + bar_w / 2, acc4, yerr=yerr,
                fmt="none", color="black", capsize=4, capthick=1.2, lw=1.2, zorder=5)

    # Value labels
    for xi, (v5, v4) in enumerate(zip(acc5, acc4)):
        ax.text(xi - bar_w / 2, v5 + 0.002, f"{v5:.1%}",
                ha="center", va="bottom", fontsize=8.5, color="#555")
        ax.text(xi + bar_w / 2, v4 + 0.002, f"{v4:.1%}",
                ha="center", va="bottom", fontsize=8.5,
                color=ACT_COLORS_A4[xi], fontweight="bold")

    # Delta annotations
    for xi, (v5, v4) in enumerate(zip(acc5, acc4)):
        if not np.isnan(v4):
            delta = v4 - v5
            color = "#2e7d32" if delta > 0.02 else ("#c62828" if delta < -0.01 else "#666")
            ax.text(xi, min(v4, v5) - 0.012, f"Δ{delta:+.1%}",
                    ha="center", fontsize=8, color=color, fontstyle="italic")

    # Target lines
    ax.axhline(0.99, color="#c62828", lw=1.2, ls="--", alpha=0.7, zorder=3,
               label="99% target (suppress/monitor)")
    ax.axhline(0.90, color="#1565c0", lw=1.0, ls=":", alpha=0.6, zorder=3,
               label="90% target (escalate/investigate)")

    # CAUTION region shading
    ax.axvspan(1.5, 3.5, color="#ffebee", alpha=0.25, zorder=1)
    ax.text(2.5, 0.84, "CAUTION actions\n(high-cost errors)", ha="center",
            fontsize=8.5, color="#c62828", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS, fontsize=12)
    ax.set_ylabel("Per-action accuracy within band (conf ≥ 0.90)", fontsize=11)
    ax.set_ylim(0.82, 1.035)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "FX-T5-A4: Per-Action Accuracy in Auto-Approve Band — A=5 vs A=4\n"
        "Key question: does removing refer_to_analyst fix monitor accuracy?",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)

    caption = (
        f"soc_product_v50 A=4, C=6, d=6. {r['n_seeds']} seeds, "
        f"warmup={r['n_warmup']}, eval={r['n_decisions']}, "
        f"τ={r['tau']}, η_neg={r['eta_neg']}, noise={r['noise_rate']:.0%}. "
        "A=5 reference: FX-T5-BREAKDOWN (realistic generator, static scorer)."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8,
             color="#555", style="italic")

    save_figure(fig, "fx_t5_a4_per_action_accuracy", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] fx_t5_a4_per_action_accuracy.png + .pdf saved")


# ============================================================================
# Chart 2 — Band composition: A=5 vs A=4
# ============================================================================

def chart2_band_distribution(r: dict) -> None:
    a5_ref = r["a5_reference"]
    q1_a4  = r["q1_action_distribution"]
    q2_a4  = r["q2_per_action_accuracy"]
    a5_ref_q2 = {act: a5_ref[act] for act in ACTIONS}

    pct5 = [a5_ref[act]["pct_band"] for act in ACTIONS]
    pct4 = [q1_a4[act]["pct_band"]  for act in ACTIONS]
    acc5 = [a5_ref[act]["accuracy"]  for act in ACTIONS]
    acc4 = [q2_a4[act]["accuracy"]   for act in ACTIONS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, wspace=0.38)

    # --- Left: Band composition stacked bar ---
    ax = axes[0]
    x = [0, 1]
    bottom5 = 0
    bottom4 = 0
    for act_idx, act_name in enumerate(ACTIONS):
        h5 = pct5[act_idx]
        h4 = pct4[act_idx]
        col_a5 = ACT_COLORS_A5[act_idx]
        col_a4 = ACT_COLORS_A4[act_idx]
        ax.bar([0], [h5], bottom=[bottom5], color=col_a5, edgecolor="#555",
               linewidth=0.6, zorder=4)
        ax.bar([1], [h4], bottom=[bottom4], color=col_a4, edgecolor="black",
               linewidth=0.7, zorder=4, label=act_name)
        # Labels for substantial slices
        if h5 > 0.04:
            ax.text(0, bottom5 + h5 / 2, f"{act_name}\n{h5:.1%}",
                    ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold")
        if h4 > 0.04:
            ax.text(1, bottom4 + h4 / 2, f"{act_name}\n{h4:.1%}",
                    ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold")
        bottom5 += h5
        bottom4 += h4

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A=5 (FX-T5)", "A=4 (this run)"], fontsize=11)
    ax.set_ylabel("Fraction of auto-approve band", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.set_title("Band Composition\n(action distribution within conf ≥ 0.90)",
                 fontsize=11, fontweight="bold")
    total5 = a5_ref["total_band"]
    total4 = r["total_band_decisions"]
    ax.text(0, 1.02, f"n={total5:,}", ha="center", fontsize=9, color="#555")
    ax.text(1, 1.02, f"n={total4:,}", ha="center", fontsize=9, color="#555")

    # --- Right: Accuracy within band comparison ---
    ax2 = axes[1]
    x = np.arange(len(ACTIONS))
    bar_w = 0.35

    ax2.bar(x - bar_w / 2, acc5, bar_w, color=ACT_COLORS_A5,
            edgecolor="#555", linewidth=0.8, alpha=0.82, label="A=5", zorder=4)
    ax2.bar(x + bar_w / 2, acc4, bar_w, color=ACT_COLORS_A4,
            edgecolor="black", linewidth=0.8, alpha=0.88, label="A=4", zorder=4)

    for xi, (v5, v4) in enumerate(zip(acc5, acc4)):
        if not np.isnan(v4):
            delta = v4 - v5
            col = "#2e7d32" if delta > 0.02 else ("#c62828" if delta < -0.01 else "#666")
            ax2.text(xi + bar_w / 2, v4 + 0.003, f"{v4:.1%}",
                     ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

    ax2.axhline(0.99, color="#c62828", lw=1.2, ls="--", alpha=0.7)
    ax2.axhline(0.90, color="#1565c0", lw=1.0, ls=":",  alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ACTIONS, fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim(0.80, 1.04)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_title("Per-Action Accuracy in Band\n(A=5 vs A=4, conf ≥ 0.90)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")

    # Dangerous error annotation
    der4 = r["q3_error_direction"]["dangerous_error_rate"]
    der5 = a5_ref["dangerous_rate"]
    fig.text(0.5, 0.04,
             f"Dangerous error rate:  A=5 = {der5:.2%}  →  A=4 = {der4:.2%}   "
             f"({'REDUCED' if der4 < der5 else 'UNCHANGED'}). "
             f"Cost ratio = {r['cost_ratio']:.0f}:1.  Band threshold = {r['band_threshold']:.2f}.",
             ha="center", fontsize=9, color="#333")

    fig.suptitle(
        "FX-T5-A4: Auto-Approve Band Analysis — A=5 vs A=4 Comparison\n"
        "soc_product_v50 (Phase 0a: refer_to_analyst removed)",
        fontsize=12, fontweight="bold",
    )

    save_figure(fig, "fx_t5_a4_band_distribution", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] fx_t5_a4_band_distribution.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_accuracy(r)
    chart2_band_distribution(r)
