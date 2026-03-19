"""
EXP-B1 RECHECK Chart

Chart 1 — expB1_recheck_accuracy_by_noise:
  Grouped bar chart. 3 groups (noise rates). 3 bars per group
  (static, learning, centroid-only). Error bars = 95% CI.
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

NOISE_RATES = [0.0, 0.10, 0.30]
CONDITIONS  = ["static", "learning", "centroid_only"]
COND_LABELS = ["Static\n(no update)", "Learning\n(with update)", "Centroid-only\n(final μ rescore)"]
COND_COLORS = ["#9e9e9e", "#1565c0", "#2e7d32"]

# Published pre-fix reference values for noise=0 and noise=0.30
ORIG_REFS = {
    ("static",   0.0):  0.9789,
    ("learning", 0.0):  0.982,
    ("learning", 0.30): 0.981,
}


def make_charts(summary: dict, n_seeds: int) -> None:
    n_groups   = len(NOISE_RATES)
    n_bars     = len(CONDITIONS)
    group_w    = 0.75
    bar_w      = group_w / n_bars

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(bottom=0.22, top=0.88)

    x_centers = np.arange(n_groups)

    for bi, (cond, label, color) in enumerate(
            zip(CONDITIONS, COND_LABELS, COND_COLORS)):
        x_pos = x_centers + (bi - n_bars / 2 + 0.5) * bar_w
        means = [summary[cond][nr]["mean"] for nr in NOISE_RATES]
        ci95s = [summary[cond][nr]["ci95"]  for nr in NOISE_RATES]

        bars = ax.bar(x_pos, means, bar_w * 0.90,
                      color=color, alpha=0.82, edgecolor="black",
                      linewidth=0.7, label=label, zorder=4)
        ax.errorbar(x_pos, means, yerr=ci95s,
                    fmt="none", color="black",
                    capsize=4, capthick=1.3, linewidth=1.3, zorder=5)

        for xp, mean in zip(x_pos, means):
            ax.text(xp, mean + 0.0025, f"{mean:.1%}",
                    ha="center", va="bottom", fontsize=8.0,
                    fontweight="bold", color=color)

    # Reference lines for original pre-fix values
    orig_plotted: set = set()
    for (cond, nr), orig_val in ORIG_REFS.items():
        nr_idx = NOISE_RATES.index(nr)
        bi     = CONDITIONS.index(cond)
        xc     = nr_idx + (bi - n_bars / 2 + 0.5) * bar_w
        key    = f"{cond}_{nr}"
        if key not in orig_plotted:
            ax.plot([xc - bar_w * 0.45, xc + bar_w * 0.45], [orig_val, orig_val],
                    color="red", lw=1.5, ls="--", zorder=6,
                    label="Pre-fix reference" if not orig_plotted else "_nolegend_")
            orig_plotted.add(key)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"Noise = {nr:.0%}" for nr in NOISE_RATES], fontsize=11)
    ax.set_ylabel("Accuracy (mean ± 95% CI)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Y-axis: zoom in to relevant range
    all_means = [summary[c][nr]["mean"] for c in CONDITIONS for nr in NOISE_RATES]
    all_cis   = [summary[c][nr]["ci95"]  for c in CONDITIONS for nr in NOISE_RATES]
    y_lo = max(0.0, min(all_means) - max(all_cis) - 0.02)
    y_hi = min(1.0, max(all_means) + max(all_cis) + 0.03)
    ax.set_ylim(y_lo, y_hi)

    ax.set_title(
        "EXP-B1 Recheck: Warm-Start Accuracy (\u03b7_neg=0.05, fixed update)\n"
        "soc_product_v50 (C=6, A=5, d=6). Red dashes = pre-fix published values.",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)

    caption = (
        f"N={n_seeds} seeds, N_decisions=1000, \u03c4=0.1, \u03b7=\u03b7_neg=0.05, "
        "warm-start from soc_product_v50 profiles. "
        "Centroid-only = final learned \u03bc rescored on all training alerts."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color="#555", style="italic")

    save_figure(fig, "expB1_recheck_accuracy_by_noise", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART] expB1_recheck_accuracy_by_noise.png + .pdf saved")


if __name__ == "__main__":
    import json
    _out = Path(__file__).parent
    with open(_out / "summary.json") as f:
        raw = json.load(f)
    # Re-key noise rates from strings to floats
    summary = {
        cond: {float(k): v for k, v in d.items()}
        for cond, d in raw["summary"].items()
    }
    make_charts(summary, n_seeds=50)
