"""
EXP-OP2-ACCURACY Charts

Chart 1 — expOP2_accuracy_trajectory: 4-panel accuracy trajectory through phases
           (pre-injection, injection, post-expiry with windows marked)
Chart 2 — expOP2_accuracy_nr_comparison: grouped bar comparing centroid NR vs accuracy NR
Chart 3 — expOP2_accuracy_windows: line chart of W1-W4 accuracy per operator accuracy
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

OP_ACCURACIES = [0.0, 0.50, 0.75, 1.00]
OP_LABELS     = ["0% (harmful)", "50%", "75%", "100% (correct)"]
OP_KEYS       = [f"op{int(op*100)}" for op in OP_ACCURACIES]

OP_COLORS = {
    "op0":   "#c62828",
    "op50":  "#f57f17",
    "op75":  "#1565c0",
    "op100": "#2e7d32",
}

WINDOW_LABELS = ["W1\n(1-50)", "W2\n(51-100)", "W3\n(101-200)", "W4\n(201-400)"]


def make_charts(
    results: dict,
    raw_windows: dict,
    raw_baselines: dict,
    n_seeds: int,
) -> None:

    caption = (
        f"N={n_seeds} seeds, soc_product_v50 (C=6,A=5,d=6), "
        "Arm A (η_neg=0.05, gt always passed), τ=0.1, λ=0.5, TTL=150. "
        "Error bars = 95% CI (z=1.96). Recovery = W4 acc ≥ baseline − 2pp."
    )

    # ===========================================================================
    # Chart 1: Accuracy trajectory — 4-panel (one per op_accuracy)
    # ===========================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig1.subplots_adjust(hspace=0.50, wspace=0.30, top=0.88)

    for idx, op_key in enumerate(OP_KEYS):
        ax = axes[idx // 2][idx % 2]
        color = OP_COLORS[op_key]
        r = results[op_key]

        baseline = r["mean_baseline"]
        windows  = [r["mean_w1"], r["mean_w2"], r["mean_w3"], r["mean_w4"]]
        cis      = [r["ci_w1"],   r["ci_w2"],   r["ci_w3"],   r["ci_w4"]]

        x = np.arange(4)
        ax.plot(x, windows, "o-", color=color, lw=2.0, ms=7)
        ax.fill_between(
            x,
            [w - c for w, c in zip(windows, cis)],
            [w + c for w, c in zip(windows, cis)],
            color=color, alpha=0.15,
        )

        # Baseline reference line
        ax.axhline(baseline, color="black", lw=1.2, ls="--", alpha=0.65,
                   label=f"baseline {baseline:.1%}")
        # Recovery threshold
        thresh = baseline - 0.02
        ax.axhline(thresh, color="#888", lw=0.9, ls=":", alpha=0.7,
                   label=f"threshold {thresh:.1%}")

        ax.set_xticks(x)
        ax.set_xticklabels(WINDOW_LABELS, fontsize=9)
        ax.set_ylabel("Mean accuracy", fontsize=9.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        acc_nr   = r["accuracy_nr_rate"]
        cent_nr  = r["centroid_nr_rate"]
        ax.set_title(
            f"Op accuracy = {OP_LABELS[idx]}\n"
            f"Acc NR={acc_nr:.0%}  Cent NR={cent_nr:.0%}  "
            f"(gap={cent_nr - acc_nr:+.0%})",
            fontsize=10, fontweight="bold", color=color, pad=6,
        )
        ax.legend(fontsize=8, loc="lower right")

        # Shade post-expiry region
        ax.axvspan(-0.5, 3.5, alpha=0.03, color=color)

    fig1.suptitle(
        "Post-Expiry Accuracy by Window — Arm A (Both Bugs Fixed)\n"
        "W4 accuracy vs baseline determines recovery; centroid NR shown for comparison",
        fontsize=13, fontweight="bold",
    )
    fig1.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#555", style="italic")

    save_figure(fig1, "expOP2_accuracy_trajectory", output_dir="paper_figures")
    plt.close(fig1)
    print("[CHART] expOP2_accuracy_trajectory.png + .pdf saved")

    # ===========================================================================
    # Chart 2: NR comparison — grouped bar (centroid NR vs accuracy NR)
    # ===========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.subplots_adjust(bottom=0.18)

    x       = np.arange(len(OP_KEYS))
    w       = 0.35
    acc_nrs  = [results[k]["accuracy_nr_rate"]  for k in OP_KEYS]
    cent_nrs = [results[k]["centroid_nr_rate"]   for k in OP_KEYS]
    acc_cis  = [results[k]["accuracy_nr_ci"]     for k in OP_KEYS]
    gaps     = [results[k]["sensitivity_gap"]    for k in OP_KEYS]

    bars_cent = ax2.bar(
        x - w/2, cent_nrs, w,
        color="#c62828", alpha=0.80, edgecolor="black", linewidth=0.7,
        label="Centroid NR (from isolate)",
    )
    bars_acc = ax2.bar(
        x + w/2, acc_nrs, w,
        color="#1565c0", alpha=0.80, edgecolor="black", linewidth=0.7,
        label="Accuracy NR (this experiment)",
    )
    ax2.errorbar(
        x + w/2, acc_nrs, yerr=acc_cis,
        fmt="none", color="black", capsize=5, capthick=1.3, linewidth=1.3,
    )

    # Gap annotations
    for i, (xv, cn, an, g) in enumerate(zip(x, cent_nrs, acc_nrs, gaps)):
        top = max(cn, an) + 0.03
        ax2.annotate(
            f"gap\n{g:+.0%}",
            xy=(xv, top), ha="center", fontsize=8.5, color="#555",
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(OP_LABELS, fontsize=10)
    ax2.set_ylabel("Never-recover rate", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_ylim(0, 1.15)
    ax2.set_title(
        "Centroid NR vs Accuracy NR — Is Centroid Displacement Over-Sensitive?\n"
        "Gap = centroid_nr − accuracy_nr  (positive → centroid over-counts failures)",
        fontsize=12, fontweight="bold",
    )
    ax2.legend(fontsize=10, loc="upper right")

    # Verdict box
    avg_gap = float(np.mean(gaps))
    if avg_gap > 0.10:
        verdict = f"OVER-SENSITIVE: avg gap = {avg_gap:+.1%}\nSystem scores correctly despite centroid drift"
        box_color = "#e8f5e9"; edge_color = "#2e7d32"
    elif avg_gap > 0.0:
        verdict = f"SLIGHTLY PESSIMISTIC: avg gap = {avg_gap:+.1%}\nMinor over-counting of failure cases"
        box_color = "#fff8e1"; edge_color = "#f9a825"
    else:
        verdict = f"METRICS AGREE: avg gap = {avg_gap:+.1%}\nCentroid drift reliably predicts scoring failure"
        box_color = "#ffebee"; edge_color = "#c62828"

    ax2.text(
        0.02, 0.96, verdict,
        transform=ax2.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color,
                  edgecolor=edge_color, linewidth=1.5),
    )

    fig2.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#555", style="italic")
    save_figure(fig2, "expOP2_accuracy_nr_comparison", output_dir="paper_figures")
    plt.close(fig2)
    print("[CHART] expOP2_accuracy_nr_comparison.png + .pdf saved")

    # ===========================================================================
    # Chart 3: Window accuracy vs operator accuracy — line chart
    # ===========================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.subplots_adjust(bottom=0.15)

    window_colors = ["#c62828", "#f57f17", "#1565c0", "#2e7d32"]
    window_names  = ["W1 (dec 1-50)", "W2 (dec 51-100)", "W3 (dec 101-200)", "W4 (dec 201-400)"]
    window_keys   = ["mean_w1", "mean_w2", "mean_w3", "mean_w4"]
    ci_keys       = ["ci_w1",   "ci_w2",   "ci_w3",   "ci_w4"]

    x_op = np.arange(len(OP_KEYS))

    for wk, ck, wname, wcolor in zip(window_keys, ci_keys, window_names, window_colors):
        vals = [results[k][wk] for k in OP_KEYS]
        cis  = [results[k][ck] for k in OP_KEYS]
        ax3.plot(x_op, vals, "o-", color=wcolor, lw=2.0, ms=7, label=wname)
        ax3.fill_between(
            x_op,
            [v - c for v, c in zip(vals, cis)],
            [v + c for v, c in zip(vals, cis)],
            color=wcolor, alpha=0.10,
        )

    # Baseline reference
    mean_baselines = [results[k]["mean_baseline"] for k in OP_KEYS]
    ax3.plot(x_op, mean_baselines, "s--", color="black", lw=1.5, ms=6,
             label="Pre-injection baseline", alpha=0.65)

    ax3.set_xticks(x_op)
    ax3.set_xticklabels(OP_LABELS, fontsize=10)
    ax3.set_ylabel("Mean accuracy in window", fontsize=11)
    ax3.set_title(
        "Post-Expiry Accuracy by Window and Operator Accuracy\n"
        "W4 (dec 201-400) is the recovery criterion window",
        fontsize=13, fontweight="bold",
    )
    ax3.legend(fontsize=9, loc="lower right")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    fig3.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#555", style="italic")
    save_figure(fig3, "expOP2_accuracy_windows", output_dir="paper_figures")
    plt.close(fig3)
    print("[CHART] expOP2_accuracy_windows.png + .pdf saved")


if __name__ == "__main__":
    import json
    _out = Path(__file__).parent
    with open(_out / "results.json") as f:
        _results = json.load(f)
    _raw_windows   = np.load(str(_out / "raw_windows.npy"),   allow_pickle=True).item()
    _raw_baselines = np.load(str(_out / "raw_baselines.npy"), allow_pickle=True).item()
    # Recover N_SEEDS from data
    _n = len(list(_raw_baselines.values())[0])
    make_charts(_results, _raw_windows, _raw_baselines, _n)
