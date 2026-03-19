"""
EXP-OP2-RECHECK Charts

Chart 1 — AUAC by operator accuracy (bar)
Chart 2 — T_recovery survival curves (Kaplan-Meier style)
Chart 3 — Never-recover rate by condition with 95% CI
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

CONDITIONS_ALL   = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]
SENTINEL         = 401
N_POST           = 400
WINDOW_SIZE      = 50
N_WINDOWS        = N_POST // WINDOW_SIZE

_COLORS = {
    "A":     "#616161",
    "B":     "#2e7d32",
    "B-exp": "#66bb6a",
    "C":     "#c62828",
    "C-exp": "#ef9a9a",
    "P-75":  "#1565c0",
    "P-50":  "#42a5f5",
    "P-25":  "#f57f17",
    "P-0":   "#b71c1c",
}

_LABELS = {
    "A":     "A  (no operator)",
    "B":     "B  (correct, TTL=400)",
    "B-exp": "B-exp (correct, TTL=150)",
    "C":     "C  (harmful, TTL=400)",
    "C-exp": "C-exp (harmful, TTL=150)",
    "P-75":  "P-75 (75% correct)",
    "P-50":  "P-50 (50% correct)",
    "P-25":  "P-25 (25% correct)",
    "P-0":   "P-0  (0% correct)",
}


def make_charts(
    results: dict,
    per_cond_acc_curves: dict,
    per_cond_t_rec: dict,
    n_seeds: int,
    tag: str = "expOP2_recheck",
) -> None:
    """Generate all three charts and save to paper_figures/."""

    # ------------------------------------------------------------------
    # Chart 1: AUAC delta by operator accuracy
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 5.5))
    fig1.subplots_adjust(bottom=0.22)

    cond_order  = ["A", "P-0", "P-25", "P-50", "P-75", "B", "B-exp", "C", "C-exp"]
    x_labels    = [_LABELS[c] for c in cond_order]
    auac_deltas = [results[c]["auac_delta"] for c in cond_order]
    bar_colors  = [_COLORS[c] for c in cond_order]

    bars = ax1.bar(
        range(len(cond_order)), auac_deltas,
        color=bar_colors, edgecolor="black", linewidth=0.7, alpha=0.85,
    )
    ax1.axhline(0, color="black", lw=0.9, ls="--")
    ax1.set_xticks(range(len(cond_order)))
    ax1.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
    ax1.set_ylabel("Mean AUAC delta vs pre-shift baseline", fontsize=11)
    ax1.set_title(
        f"EXP-OP2 RECHECK: AUAC by Operator Accuracy\n"
        f"(\u03b7_neg=0.05 fixed, gt_action_index always passed, N={n_seeds})",
        fontsize=13, fontweight="bold",
    )
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}"))

    for bar, delta in zip(bars, auac_deltas):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            delta + (0.0005 if delta >= 0 else -0.001),
            f"{delta:+.4f}",
            ha="center", va="bottom" if delta >= 0 else "top",
            fontsize=8, fontweight="bold",
        )

    caption = (f"N={n_seeds} seeds, soc_product_v50 (C=6,A=5,d=6), "
               "\u03c4=0.1, \u03b7=\u03b7_neg=0.05, \u03bb=0.5, oracle noise=0%.")
    fig1.text(0.5, 0.02, caption, ha="center", fontsize=8.5, color="#555", style="italic")

    save_figure(fig1, f"{tag}_auac_by_accuracy", output_dir="paper_figures")
    plt.close(fig1)
    print(f"[CHART] {tag}_auac_by_accuracy.png + .pdf saved")

    # ------------------------------------------------------------------
    # Chart 2: T_recovery survival curves (Kaplan-Meier style)
    # ------------------------------------------------------------------
    km_conds = ["A", "B", "B-exp", "C", "C-exp", "P-75"]

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    fig2.subplots_adjust(bottom=0.15)

    decision_points = list(range(0, N_POST + WINDOW_SIZE, WINDOW_SIZE)) + [SENTINEL]

    for cond in km_conds:
        t_recs  = np.array(per_cond_t_rec[cond])
        # KM survival: S(t) = fraction with T_recovery > t
        survival = [float((t_recs > t).mean()) for t in decision_points]
        ax2.step(
            decision_points, survival,
            color=_COLORS[cond], lw=2.0, label=_LABELS[cond], where="post",
        )

    ax2.axvline(150, color="black", lw=1.2, ls=":", alpha=0.6)
    ax2.text(155, 0.92, "TTL=150", fontsize=9, color="black")
    ax2.set_xlim(-10, SENTINEL + 20)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("Decisions post-shift", fontsize=11)
    ax2.set_ylabel("Fraction not yet recovered", fontsize=11)
    ax2.set_title(
        f"Recovery Time After Operator Expiry (\u03b7_neg=0.05, fixed update, N={n_seeds})",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(fontsize=9.5, loc="upper right")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # NR annotation
    for cond in ["C", "P-75", "A"]:
        nr = results[cond]["nr_rate"]
        ax2.text(
            SENTINEL + 5, nr + 0.01,
            f"{cond}: {nr:.0%}",
            fontsize=8.5, color=_COLORS[cond], va="bottom",
        )

    save_figure(fig2, f"{tag}_t_recovery_survival", output_dir="paper_figures")
    plt.close(fig2)
    print(f"[CHART] {tag}_t_recovery_survival.png + .pdf saved")

    # ------------------------------------------------------------------
    # Chart 3: Never-recover rate with 95% CI
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 5.5))
    fig3.subplots_adjust(bottom=0.22)

    nr_rates = [results[c]["nr_rate_pct"]  for c in cond_order]
    nr_lo    = [results[c]["nr_ci_lo"]     for c in cond_order]
    nr_hi    = [results[c]["nr_ci_hi"]     for c in cond_order]
    yerr_lo  = [r - lo for r, lo in zip(nr_rates, nr_lo)]
    yerr_hi  = [hi - r for r, hi in zip(nr_rates, nr_hi)]

    ax3.bar(
        range(len(cond_order)), nr_rates,
        color=bar_colors, edgecolor="black", linewidth=0.7, alpha=0.85,
    )
    ax3.errorbar(
        range(len(cond_order)), nr_rates,
        yerr=[yerr_lo, yerr_hi],
        fmt="none", color="black", capsize=5, capthick=1.5, linewidth=1.5,
    )

    # Paradox annotation
    a_nr  = results["A"]["nr_rate_pct"]
    p75_nr = results["P-75"]["nr_rate_pct"]
    paradox = p75_nr > a_nr
    ax3.axhline(a_nr, color=_COLORS["A"], lw=1.2, ls="--", alpha=0.7,
                label=f"Baseline (A): {a_nr:.1f}%")
    ax3.text(
        0.5, 0.93,
        f"P-75 paradox (P-75 NR > baseline): {'YES' if paradox else 'NO'}",
        transform=ax3.transAxes, fontsize=10.5, ha="center",
        color="#c62828" if paradox else "#2e7d32",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1",
                  edgecolor="#f9a825", linewidth=1.2),
    )

    ax3.set_xticks(range(len(cond_order)))
    ax3.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
    ax3.set_ylabel("Never-recover rate (%)", fontsize=11)
    ax3.set_title(
        f"Never-Recover Rate by Operator Accuracy (RECHECK, N={n_seeds})\n"
        f"95% Wilson CI shown. Sentinel = {SENTINEL} decisions.",
        fontsize=13, fontweight="bold",
    )
    ax3.legend(fontsize=9.5)
    ax3.set_ylim(0, max(nr_hi) * 1.25 + 2)

    fig3.text(0.5, 0.02, caption, ha="center", fontsize=8.5, color="#555", style="italic")

    save_figure(fig3, f"{tag}_nr_rate", output_dir="paper_figures")
    plt.close(fig3)
    print(f"[CHART] {tag}_nr_rate.png + .pdf saved")


if __name__ == "__main__":
    import json
    _out = Path(__file__).parent
    with open(_out / "results.json") as f:
        _results = json.load(f)
    _curves = np.load(str(_out / "acc_curves.npy"), allow_pickle=True).item()
    _t_recs = np.load(str(_out / "t_recovery.npy"), allow_pickle=True).item()
    _n      = len(list(_t_recs.values())[0])
    make_charts(_results, _curves, _t_recs, _n, tag="expOP2_recheck")
