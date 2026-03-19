"""
EXP-S2-REPRO charts.

Four publication-quality figures from results.json:

  expS2r_arm0_replication      — Arm 0 accuracy vs poison rate (bar chart)
  expS2r_t_recovery            — Arm A T_recovery boxplot by poison rate
  expS2r_auac_vs_poison        — Arm A mean AUAC ± SD line chart
  expS2r_realistic_auac_arm_b  — Side-by-side Arm A vs Arm B AUAC

All outputs to paper_figures/ via save_figure().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.viz.bridge_common import save_figure


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_RC = {
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
}

_BLUE   = "#2563EB"
_GREEN  = "#059669"
_RED    = "#DC2626"
_ORANGE = "#D97706"
_GRAY   = "#64748B"
_LGRAY  = "#E2E8F0"


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Chart 1: Arm 0 — Frozen synthesis accuracy vs poison rate
# ---------------------------------------------------------------------------

def chart_arm0_replication(data: dict) -> None:
    """
    Bar chart: mean accuracy (%) per poison rate for frozen synthesis (Arm 0).
    Error bars = ±1 SD across seeds.
    """
    plt.rcParams.update(_RC)

    arm0 = data["arm0_frozen"]
    poison_rates = sorted(float(k) for k in arm0.keys())
    labels  = [f"{int(pr*100)}%" for pr in poison_rates]
    means   = [float(np.mean([r["accuracy"] for r in arm0[str(pr)]])) * 100
               for pr in poison_rates]
    stds    = [float(np.std([r["accuracy"]  for r in arm0[str(pr)]])) * 100
               for pr in poison_rates]

    x = np.arange(len(poison_rates))
    colors = [_GREEN, _ORANGE, _RED]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.86)

    bars = ax.bar(x, means, width=0.5,
                  color=colors[:len(poison_rates)],
                  edgecolor="white", linewidth=0.8,
                  yerr=stds, capsize=4,
                  error_kw=dict(ecolor=_GRAY, lw=1.1),
                  zorder=3)

    # Value labels above bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.4,
                f"{m:.1f}%",
                ha="center", va="bottom",
                fontsize=9.5, color="#1E293B", zorder=5)

    # Degradation annotation (0% → 20%)
    if len(means) >= 2:
        deg = means[0] - means[1]
        sign = "-" if deg >= 0 else "+"
        mid_x = (x[0] + x[1]) / 2
        ax.annotate("",
                    xy=(x[1], means[1]),
                    xytext=(x[0], means[0]),
                    arrowprops=dict(arrowstyle="->", color=_GRAY, lw=1.1),
                    zorder=4)
        ax.text(mid_x, (means[0] + means[1]) / 2 + 0.5,
                f"{sign}{abs(deg):.1f}pp",
                ha="center", va="bottom",
                fontsize=9, color=_GRAY, style="italic")

    # Gate line at baseline - 2pp
    baseline = means[0]
    gate_y = baseline - 2.0
    ax.axhline(y=gate_y, color=_ORANGE, linewidth=1.1, linestyle="--", zorder=2)
    ax.text(len(poison_rates) - 1 + 0.38, gate_y + 0.2,
            f"Gate: −2pp ({gate_y:.1f}%)",
            ha="right", va="bottom",
            fontsize=8, color=_ORANGE, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Poison Rate", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(labelsize=9)
    _clean_ax(ax)

    gate_str = data["summary"]["arm0"]["gate"]
    ax.set_title(f"Arm 0: Frozen-Synthesis Accuracy vs Poison Rate  [{gate_str}]",
                 fontsize=12, pad=10)
    fig.text(0.12, 0.875,
             "400 decisions, no centroid update, SynthesisBias applied at scoring only.",
             ha="left", va="bottom",
             fontsize=9, color="#475569", style="italic",
             transform=fig.transFigure)

    save_figure(fig, "expS2r_arm0_replication", output_dir="paper_figures")


# ---------------------------------------------------------------------------
# Chart 2: Arm A — T_recovery boxplot by poison rate
# ---------------------------------------------------------------------------

def chart_t_recovery(data: dict) -> None:
    """
    Boxplot: T_recovery (decisions) per poison rate for Arm A.
    Gate line at 100 decisions.
    """
    plt.rcParams.update(_RC)

    arm_a = data["arm_a_production"]
    poison_rates = sorted(float(k) for k in arm_a.keys())
    labels = [f"{int(pr*100)}%" for pr in poison_rates]

    box_data = [
        [r["t_recovery"] for r in arm_a[str(pr)]]
        for pr in poison_rates
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.86)

    bp = ax.boxplot(box_data,
                    positions=np.arange(len(poison_rates)),
                    widths=0.45,
                    patch_artist=True,
                    medianprops=dict(color=_BLUE, linewidth=2),
                    whiskerprops=dict(color=_GRAY, linewidth=1.1),
                    capprops=dict(color=_GRAY, linewidth=1.1),
                    flierprops=dict(marker="o", markerfacecolor=_GRAY,
                                   markersize=4, alpha=0.6),
                    zorder=3)

    colors = [_GREEN, "#a8c956", _ORANGE, _RED]
    for patch, col in zip(bp["boxes"], colors[:len(poison_rates)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)

    # Gate line at 100 decisions
    ax.axhline(y=100, color=_RED, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(poison_rates) - 1 + 0.38, 102,
            "Gate: 100 decisions",
            ha="right", va="bottom",
            fontsize=8, color=_RED, style="italic")

    # Annotate p90 at 20% poison (index 2 if rates=[0,10,20,30])
    idx_20 = next((i for i, pr in enumerate(poison_rates) if abs(pr - 0.20) < 1e-6), None)
    if idx_20 is not None:
        p90_val = float(np.percentile(box_data[idx_20], 90))
        ax.scatter([idx_20], [p90_val],
                   color="#DC2626", s=60, zorder=5, marker="^",
                   label=f"p90 @ 20% = {p90_val:.0f}")
        ax.legend(fontsize=9, loc="upper left", frameon=True,
                  framealpha=0.95, edgecolor=_LGRAY)

    ax.set_xticks(np.arange(len(poison_rates)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Poison Rate", fontsize=11)
    ax.set_ylabel("T_recovery (decisions)", fontsize=11)
    ax.tick_params(labelsize=9)
    _clean_ax(ax)

    gate_str = data["summary"]["arm_a"]["gate"]
    ax.set_title(f"Arm A: T_recovery by Poison Rate  [{gate_str}]",
                 fontsize=12, pad=10)
    fig.text(0.12, 0.875,
             f"20 seeds, {400} post-shift decisions, sentinel={400} if never recovered.",
             ha="left", va="bottom",
             fontsize=9, color="#475569", style="italic",
             transform=fig.transFigure)

    save_figure(fig, "expS2r_t_recovery", output_dir="paper_figures")


# ---------------------------------------------------------------------------
# Chart 3: Arm A — Mean AUAC ± SD line chart
# ---------------------------------------------------------------------------

def chart_auac_vs_poison(data: dict) -> None:
    """
    Line + shaded-SD: mean AUAC vs poison rate for Arm A.
    """
    plt.rcParams.update(_RC)

    arm_a = data["arm_a_production"]
    poison_rates = sorted(float(k) for k in arm_a.keys())
    pct_labels = [int(pr * 100) for pr in poison_rates]
    means = [float(np.mean([r["auac"] for r in arm_a[str(pr)]])) for pr in poison_rates]
    stds  = [float(np.std ([r["auac"] for r in arm_a[str(pr)]])) for pr in poison_rates]

    x = np.array(pct_labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.86)

    ax.fill_between(x,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=_BLUE, alpha=0.15, zorder=2)
    ax.plot(x, means,
            color=_BLUE, linewidth=2.2,
            marker="o", markersize=7,
            zorder=4, label="Arm A (centroidal)")

    # Annotate the baseline (0% poison)
    ax.text(x[0] + 0.5, means[0] + stds[0] + 0.001,
            f"Baseline: {means[0]:.4f}",
            ha="left", va="bottom",
            fontsize=8.5, color=_BLUE, style="italic")

    ax.set_xticks(pct_labels)
    ax.set_xticklabels([f"{p}%" for p in pct_labels], fontsize=10)
    ax.set_xlabel("Poison Rate", fontsize=11)
    ax.set_ylabel("AUAC", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.tick_params(labelsize=9)
    _clean_ax(ax)

    ax.set_title("Arm A: Mean AUAC vs Poison Rate  (20 seeds)", fontsize=12, pad=10)
    fig.text(0.12, 0.875,
             "AUAC = area under rolling-accuracy curve, N_post=400, window=50.",
             ha="left", va="bottom",
             fontsize=9, color="#475569", style="italic",
             transform=fig.transFigure)

    save_figure(fig, "expS2r_auac_vs_poison", output_dir="paper_figures")


# ---------------------------------------------------------------------------
# Chart 4: Side-by-side Arm A vs Arm B AUAC
# ---------------------------------------------------------------------------

def chart_arm_ab_auac(data: dict) -> None:
    """
    Two-panel figure: left = Arm A AUAC by poison rate,
                      right = Arm B AUAC by poison rate.
    """
    plt.rcParams.update(_RC)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.82,
                        wspace=0.35)

    def _panel(ax, arm_data: dict, title: str, color: str, seeds_n: int) -> None:
        poison_rates = sorted(float(k) for k in arm_data.keys())
        pct  = [int(pr * 100) for pr in poison_rates]
        means = [float(np.mean([r["auac"] for r in arm_data[str(pr)]])) for pr in poison_rates]
        stds  = [float(np.std ([r["auac"] for r in arm_data[str(pr)]])) for pr in poison_rates]

        x = np.arange(len(poison_rates))
        bars = ax.bar(x, means,
                      width=0.5,
                      color=color, alpha=0.75,
                      edgecolor="white", linewidth=0.8,
                      yerr=stds, capsize=4,
                      error_kw=dict(ecolor=_GRAY, lw=1.1),
                      zorder=3)

        # Value labels
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + 0.001,
                    f"{m:.3f}",
                    ha="center", va="bottom",
                    fontsize=8.5, color="#1E293B", zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}%" for p in pct], fontsize=10)
        ax.set_xlabel("Poison Rate", fontsize=11)
        ax.set_ylabel("Mean AUAC", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
        ax.tick_params(labelsize=9)
        _clean_ax(ax)
        ax.set_title(title, fontsize=12, pad=6)
        ax.text(0.5, 1.01, f"({seeds_n} seeds)",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=9, color="#475569", style="italic")

    _panel(ax_l, data["arm_a_production"],
           "Arm A: Centroidal (bridge-common)", _BLUE, seeds_n=20)
    _panel(ax_r, data["arm_b_realistic"],
           "Arm B: Realistic SOC (combined mode)", _GREEN, seeds_n=10)

    fig.suptitle("Poisoning Resilience: AUAC Comparison Arm A vs Arm B",
                 fontsize=13, y=0.97)

    # Gate note for Arm B
    fig.text(0.97, 0.02,
             "Arm B gate: DOMAIN EXPERT REVIEW (no numeric threshold).",
             ha="right", va="bottom",
             fontsize=8.5, color="#475569", style="italic",
             transform=fig.transFigure)

    save_figure(fig, "expS2r_realistic_auac_arm_b", output_dir="paper_figures")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(data: dict) -> None:
    print("\nGenerating EXP-S2-REPRO charts ...")
    chart_arm0_replication(data)
    print("  [1/4] expS2r_arm0_replication")
    chart_t_recovery(data)
    print("  [2/4] expS2r_t_recovery")
    chart_auac_vs_poison(data)
    print("  [3/4] expS2r_auac_vs_poison")
    chart_arm_ab_auac(data)
    print("  [4/4] expS2r_realistic_auac_arm_b")
    print("Done. 8 files in paper_figures/")


if __name__ == "__main__":
    _results_path = Path("experiments/synthesis/expS2_repro/results.json")
    if not _results_path.exists():
        print(f"ERROR: {_results_path} not found. Run run.py first.")
        sys.exit(1)
    with open(_results_path) as fh:
        _data = json.load(fh)
    generate_charts(_data)
