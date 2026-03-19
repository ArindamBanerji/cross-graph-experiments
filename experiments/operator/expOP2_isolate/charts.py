"""
EXP-OP2-ISOLATE Charts

Chart 1 — expOP2_isolate_factorial_nr: 4-panel bar chart (one per arm)
Chart 2 — expOP2_isolate_effect_decomposition: stacked bar effect decomposition
Chart 3 — expOP2_isolate_post_accuracy: line chart, post-expiry accuracy by arm
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure

ARMS_ORDER = ["A_both_fixed", "B_eta_fixed_only", "C_update_fixed_only", "D_both_legacy"]
ARM_LABELS = {
    "A_both_fixed":        "A: both fixed\n(η=0.05, GT✓)",
    "B_eta_fixed_only":    "B: η fixed only\n(η=0.05, no GT)",
    "C_update_fixed_only": "C: update fixed only\n(η=1.0, GT✓)",
    "D_both_legacy":       "D: both bugs\n(η=1.0, no GT)",
}
ARM_COLORS = {
    "A_both_fixed":        "#2e7d32",
    "B_eta_fixed_only":    "#66bb6a",
    "C_update_fixed_only": "#1565c0",
    "D_both_legacy":       "#c62828",
}
OP_ACCURACIES = [0.0, 0.50, 0.75, 1.00]
OP_LABELS     = ["0% (harmful)", "50%", "75%", "100% (correct)"]


def make_charts(
    results: dict,
    decomp: dict,
    raw: dict,
    n_seeds: int,
) -> None:

    # ======================================================================
    # Chart 1: Factorial NR rate — 4-panel (one per arm)
    # ======================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    fig1.subplots_adjust(hspace=0.45, wspace=0.12, top=0.88)

    for idx, arm_name in enumerate(ARMS_ORDER):
        ax = axes[idx // 2][idx % 2]
        color = ARM_COLORS[arm_name]

        nr_vals = [results[f"{arm_name}_op{int(op*100)}"]["nr_rate"] for op in OP_ACCURACIES]
        ci_vals = [results[f"{arm_name}_op{int(op*100)}"]["nr_ci"]   for op in OP_ACCURACIES]

        x = np.arange(len(OP_ACCURACIES))
        bars = ax.bar(x, nr_vals, color=color, alpha=0.82,
                      edgecolor="black", linewidth=0.7)
        ax.errorbar(x, nr_vals, yerr=ci_vals,
                    fmt="none", color="black", capsize=5, capthick=1.3, linewidth=1.3)

        ax.set_xticks(x)
        ax.set_xticklabels(OP_LABELS, fontsize=9, rotation=12)
        ax.set_title(ARM_LABELS[arm_name], fontsize=11, fontweight="bold",
                     color=color, pad=6)
        ax.set_ylabel("Cell drift NR rate", fontsize=9.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, 1.05)

        for bar, val in zip(bars, nr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=color)

    fig1.suptitle(
        "2\u00d72 Factorial: Never-Recover Rate by Arm and Operator Accuracy\n"
        "NR = fraction of (c,a) centroid cells with L2 drift > 0.05 after N_POST decisions",
        fontsize=13, fontweight="bold",
    )
    caption = (f"N={n_seeds} seeds, soc_product_v50 (C=6,A=5,d=6), \u03c4=0.1, \u03b7=0.05, "
               "\u03bb=0.5, TTL=150, N_POST=400. Error bars = 95% CI (z=1.96).")
    fig1.text(0.5, 0.01, caption, ha="center", fontsize=8.5, color="#555", style="italic")

    save_figure(fig1, "expOP2_isolate_factorial_nr", output_dir="paper_figures")
    plt.close(fig1)
    print("[CHART] expOP2_isolate_factorial_nr.png + .pdf saved")

    # ======================================================================
    # Chart 2: Effect decomposition — stacked bar
    # ======================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.subplots_adjust(bottom=0.20)

    op_keys   = [f"op{int(op*100)}" for op in OP_ACCURACIES]
    eta_effs  = [decomp[k]["eta_effect"]    for k in op_keys]
    upd_effs  = [decomp[k]["update_effect"] for k in op_keys]
    inter     = [decomp[k]["interaction"]   for k in op_keys]

    x   = np.arange(len(OP_ACCURACIES))
    w   = 0.5

    # Stack positive and negative contributions separately
    def _stack_bars(ax, x, w, values_list, colors, labels):
        bottoms_pos = np.zeros(len(x))
        bottoms_neg = np.zeros(len(x))
        for vals, color, label in zip(values_list, colors, labels):
            vals_arr = np.array(vals)
            pos_vals = np.where(vals_arr >= 0, vals_arr, 0)
            neg_vals = np.where(vals_arr < 0,  vals_arr, 0)
            ax.bar(x, pos_vals, w, bottom=bottoms_pos, color=color,
                   alpha=0.85, edgecolor="black", linewidth=0.6, label=label)
            ax.bar(x, neg_vals, w, bottom=bottoms_neg, color=color,
                   alpha=0.85, edgecolor="black", linewidth=0.6)
            bottoms_pos += pos_vals
            bottoms_neg += neg_vals

    _stack_bars(
        ax2, x, w,
        [eta_effs, upd_effs, inter],
        ["#1565c0", "#c62828", "#f57f17"],
        ["\u03b7_neg effect (C\u2212A)", "Update bug effect (B\u2212A)", "Interaction"],
    )

    # Overlay total
    totals = [decomp[k]["total"] for k in op_keys]
    ax2.plot(x, totals, "ko--", lw=1.5, ms=7, label="Total (D\u2212A)", zorder=5)

    ax2.axhline(0, color="black", lw=0.9, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(OP_LABELS, fontsize=10)
    ax2.set_ylabel("Change in NR rate (vs Arm A)", fontsize=11)
    ax2.set_title(
        "Effect Decomposition: \u03b7_neg vs Update Bug vs Interaction\n"
        "(All effects relative to Arm A: both fixed)",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(fontsize=10, loc="upper right")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))

    dominant = "\u03b7_neg" if abs(eta_effs[0]) > abs(upd_effs[0]) else "Update bug"
    ax2.text(
        0.02, 0.05,
        f"Dominant effect at 0% op accuracy: {dominant}\n"
        f"\u03b7 effect: {eta_effs[0]:+.1%}   Update bug: {upd_effs[0]:+.1%}   "
        f"Interaction: {inter[0]:+.1%}",
        transform=ax2.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9",
                  edgecolor="#2e7d32", linewidth=1.2),
    )

    fig2.text(0.5, 0.01, caption, ha="center", fontsize=8.5, color="#555", style="italic")
    save_figure(fig2, "expOP2_isolate_effect_decomposition", output_dir="paper_figures")
    plt.close(fig2)
    print("[CHART] expOP2_isolate_effect_decomposition.png + .pdf saved")

    # ======================================================================
    # Chart 3: Post-expiry accuracy by arm
    # ======================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.subplots_adjust(bottom=0.18)

    for arm_name in ARMS_ORDER:
        color = ARM_COLORS[arm_name]
        accs = [results[f"{arm_name}_op{int(op*100)}"]["post_accuracy"] for op in OP_ACCURACIES]
        cis  = [results[f"{arm_name}_op{int(op*100)}"]["post_acc_ci"]   for op in OP_ACCURACIES]
        x_op = np.arange(len(OP_ACCURACIES))
        ax3.plot(x_op, accs, "o-", color=color, lw=2.0, ms=7,
                 label=ARM_LABELS[arm_name].replace("\n", "  "))
        ax3.fill_between(
            x_op,
            [a - c for a, c in zip(accs, cis)],
            [a + c for a, c in zip(accs, cis)],
            color=color, alpha=0.12,
        )

    ax3.set_xticks(np.arange(len(OP_ACCURACIES)))
    ax3.set_xticklabels(OP_LABELS, fontsize=10)
    ax3.set_ylabel("Post-expiry accuracy (dec TTL..N_POST)", fontsize=11)
    ax3.set_title(
        "Post-Expiry Accuracy by Arm\n"
        "(Mean accuracy during dec 150\u2013400 after operator expires)",
        fontsize=13, fontweight="bold",
    )
    ax3.legend(fontsize=9, loc="lower right")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig3.text(0.5, 0.01, caption, ha="center", fontsize=8.5, color="#555", style="italic")
    save_figure(fig3, "expOP2_isolate_post_accuracy", output_dir="paper_figures")
    plt.close(fig3)
    print("[CHART] expOP2_isolate_post_accuracy.png + .pdf saved")


if __name__ == "__main__":
    import json
    _out = Path(__file__).parent
    with open(_out / "isolate_results.json") as f:
        _results = json.load(f)
    with open(_out / "decomp.json") as f:
        _decomp = json.load(f)
    make_charts(_results, _decomp, {}, 50)
