"""
EXP-OP2 SUPPLEMENT: Post-TTL Expiry Comparison — Chart Generation

Two-panel figure:
  Panel A: Window-by-window AUAC delta trajectory for all 5 conditions
  Panel B: Pre- vs post-expiry mean delta summary bars
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

from src.viz.bridge_common import save_figure, COLORS

DATA_PATH = Path(__file__).parent / "results.npy"

CONDITIONS = ["A", "B", "B-exp", "C", "C-exp"]

STYLE = {
    "A":     dict(color="#616161", lw=1.6, ls="-",  label="A  (no operator)"),
    "B":     dict(color="#2e7d32", lw=2.0, ls="-",  label="B  (correct, active)"),
    "B-exp": dict(color="#66bb6a", lw=2.0, ls="--", label="B-exp  (correct, expires w=3)"),
    "C":     dict(color="#c62828", lw=2.0, ls="-",  label="C  (harmful, active)"),
    "C-exp": dict(color="#ef9a9a", lw=2.0, ls="--", label="C-exp  (harmful, expires w=3)"),
}

WINDOW_SIZE      = 50
POST_EXPIRY_IDX  = list(range(3, 8))
PRE_EXPIRY_IDX   = [0, 1, 2]


def make_charts() -> None:
    """Load results.npy and generate the expOP2 post-expiry comparison figure."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"results.npy not found — run run.py first.\n  Expected: {DATA_PATH}"
        )

    data = np.load(str(DATA_PATH), allow_pickle=True).item()
    delta_mean = {c: data[c].mean(axis=0) for c in CONDITIONS}
    delta_std  = {c: data[c].std(axis=0)  for c in CONDITIONS}
    N_seeds    = data["A"].shape[0]
    n_windows  = data["A"].shape[1]   # 8

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.42)

    x = np.arange(n_windows)

    # ==================================================================
    # Panel A: Window-by-window trajectory
    # ==================================================================
    ax1.set_title(
        "Panel A: AUAC delta by window \u2014 all 5 conditions\n"
        "(window 0 = decisions 0\u201350, window 7 = decisions 350\u2013400)",
        fontsize=12, pad=8,
    )

    for cond in CONDITIONS:
        s = STYLE[cond]
        ax1.plot(x, delta_mean[cond], color=s["color"], lw=s["lw"],
                 ls=s["ls"], label=s["label"], zorder=4)
        ax1.fill_between(
            x,
            delta_mean[cond] - delta_std[cond],
            delta_mean[cond] + delta_std[cond],
            color=s["color"], alpha=0.10, zorder=3,
        )

    # TTL expiry vertical line (between w=2 and w=3)
    ax1.axvline(2.5, color="black", lw=1.4, ls=":", zorder=5)

    # Place TTL label safely after axes are populated
    y_lim_bottom = ax1.get_ylim()[0] if ax1.get_ylim()[0] > -0.04 else -0.032
    ax1.text(
        2.58, y_lim_bottom + 0.003,
        "TTL = 150\n(operator\nexpires)",
        fontsize=9, va="bottom", color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="black", alpha=0.8),
    )

    ax1.axhline(0, color="black", lw=0.9, ls="--", zorder=2)

    # Annotate C-exp lasting damage at window 5
    c_exp_post_val = float(delta_mean["C-exp"][POST_EXPIRY_IDX].mean())
    w5_val = float(delta_mean["C-exp"][5])
    ax1.annotate(
        f"C-exp: {c_exp_post_val:+.4f}\nLasting damage \u2014 did NOT recover",
        xy=(5, w5_val),
        xytext=(4.4, w5_val - 0.010),
        fontsize=9, color="#c62828", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee",
                  edgecolor="#c62828", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
    )

    # Annotate B-exp partial recovery at window 5
    b_exp_post_val = float(delta_mean["B-exp"][POST_EXPIRY_IDX].mean())
    b5_val = float(delta_mean["B-exp"][5])
    ax1.annotate(
        f"B-exp: {b_exp_post_val:+.4f}\n(partial recovery)",
        xy=(5, b5_val),
        xytext=(4.4, b5_val + 0.010),
        fontsize=9, color="#2e7d32",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9",
                  edgecolor="#2e7d32", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"w={i}\n{i*WINDOW_SIZE}\u2013{i*WINDOW_SIZE+WINDOW_SIZE}" for i in x],
        fontsize=9,
    )
    ax1.set_ylabel("AUAC delta vs pre-shift baseline", fontsize=11)
    ax1.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}"))

    # ==================================================================
    # Panel B: Pre vs post expiry summary bars
    # ==================================================================
    ax2.set_title(
        "Panel B: Mean AUAC delta \u2014 pre-expiry (w=0\u20132) vs post-expiry (w=3\u20137)",
        fontsize=12, pad=8,
    )

    n_conds = len(CONDITIONS)
    x_pos   = np.arange(n_conds)
    width   = 0.35

    pre_means  = [float(delta_mean[c][PRE_EXPIRY_IDX].mean())  for c in CONDITIONS]
    post_means = [float(delta_mean[c][POST_EXPIRY_IDX].mean()) for c in CONDITIONS]
    pre_stds   = [float(data[c][:, PRE_EXPIRY_IDX].mean(axis=1).std())  for c in CONDITIONS]
    post_stds  = [float(data[c][:, POST_EXPIRY_IDX].mean(axis=1).std()) for c in CONDITIONS]

    bar_colors = [STYLE[c]["color"] for c in CONDITIONS]

    ax2.bar(
        x_pos - width / 2, pre_means,  width, yerr=pre_stds,
        color=bar_colors, alpha=0.50, edgecolor="black",
        linewidth=0.8, capsize=4, label="Pre-expiry (w=0\u20132)",
    )
    ax2.bar(
        x_pos + width / 2, post_means, width, yerr=post_stds,
        color=bar_colors, alpha=0.90, edgecolor="black",
        linewidth=0.8, capsize=4, label="Post-expiry (w=3\u20137)",
    )

    # Annotations on key post-expiry bars
    c_idx = CONDITIONS.index("C-exp")
    b_idx = CONDITIONS.index("B-exp")

    ax2.text(
        c_idx + width / 2,
        post_means[c_idx] - post_stds[c_idx] - 0.003,
        "LASTING\nDAMAGE",
        ha="center", fontsize=8.5, color="#c62828", fontweight="bold",
    )
    ax2.text(
        b_idx + width / 2,
        post_means[b_idx] + post_stds[b_idx] + 0.002,
        "partial\nrecovery",
        ha="center", fontsize=8.5, color="#2e7d32",
    )

    ax2.axhline(0, color="black", lw=0.9, ls="--")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(CONDITIONS, fontsize=11)
    ax2.set_ylabel("Mean AUAC delta over period", fontsize=11)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}"))

    # Safety implication box
    ax2.text(
        0.01, 0.06,
        "TTL expiry alone is NOT a sufficient safety mechanism.\n"
        "C-exp: harmful centroid damage persists beyond TTL.\n"
        "Checkpoint + rollback (TD-033) is required.",
        transform=ax2.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee",
                  edgecolor="#c62828", linewidth=1.3),
    )

    fig.suptitle(
        "EXP-OP2: TTL Expiry Does Not Reverse Harmful Centroid Damage (C-exp)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    caption = (
        f"N={N_seeds} seeds, \u03bb=0.5, TTL=150, C=5, A={data['A'].shape[1]}, d=6, "
        "\u03c4=0.1, \u03b7=\u03b7_neg=0.05, oracle noise=10%, RNG separation seed+10000."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#555", style="italic")

    save_figure(fig, "expOP2_post_expiry_comparison", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART] expOP2_post_expiry_comparison.png + .pdf saved")


if __name__ == "__main__":
    make_charts()
