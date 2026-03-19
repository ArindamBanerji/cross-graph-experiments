"""
EXP-OP2-N100 charts.

Three publication-quality figures:

  expOP2n_never_recover_ci         — 9-bar never-recover rate with Wilson CI + N=20 diamonds
  expOP2n_t_recovery_violin        — 9 violin plots of T_recovery, bimodal conditions flagged
  expOP2n_indirect_path_consistency — B-exp bimodality check (histogram + std comparison)
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

from src.viz.bridge_common import save_figure, COLORS

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_RC = {
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
}

_BLUE        = "#2563EB"
_RED         = "#DC2626"
_SALMON      = "#F87171"
_GREEN       = "#059669"
_ORANGE      = "#D97706"
_GRAY        = "#64748B"
_LGRAY       = "#E2E8F0"
_STEEL_BLUE  = "#4A90C4"

N20_NEVER_RECOVER = {
    "A":    20.0,
    "B":     5.0,
    "B-exp": None,
    "C":    35.0,
    "C-exp":35.0,
    "P-75": 20.0,
    "P-50": None,
    "P-25": None,
    "P-0":  None,
}


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Chart 1: Never-recover rate with 95% Wilson CI + N=20 red diamonds
# ---------------------------------------------------------------------------

def chart_never_recover_ci(data: dict) -> None:
    plt.rcParams.update(_RC)

    nr = data["summary"]["never_recover"]
    conditions_all = list(nr.keys())

    # Sort ascending by N=100 never_recover_rate
    conds_sorted = sorted(conditions_all, key=lambda c: nr[c]["pct"])

    rates  = [nr[c]["pct"]    for c in conds_sorted]
    ci_lo  = [nr[c]["pct"] - nr[c]["ci_lo_pct"] for c in conds_sorted]   # lower errbar
    ci_hi  = [nr[c]["ci_hi_pct"] - nr[c]["pct"] for c in conds_sorted]   # upper errbar

    x = np.arange(len(conds_sorted))

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.86)

    # Bars
    bars = ax.bar(x, rates,
                  width=0.55,
                  color=_BLUE, alpha=0.70,
                  edgecolor="white", linewidth=0.8,
                  zorder=3)

    # Asymmetric error bars (Wilson CI)
    ax.errorbar(x, rates,
                yerr=[ci_lo, ci_hi],
                fmt="none",
                ecolor=_GRAY, elinewidth=1.3, capsize=5, capthick=1.3,
                zorder=4)

    # N=20 red diamond markers
    for i, cond in enumerate(conds_sorted):
        ref = N20_NEVER_RECOVER.get(cond)
        if ref is not None:
            ax.scatter([i], [ref],
                       color=_RED, marker="D", s=70, zorder=5,
                       label="N=20 estimate" if i == 0 else "_")

    # Gate lines
    ax.axhline(y=35.0, color=_ORANGE, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(conds_sorted) - 0.4, 35.5,
            "OP2 C estimate (35%)",
            ha="right", va="bottom",
            fontsize=8, color=_ORANGE, style="italic")

    ax.axhline(y=5.0, color=_GREEN, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(conds_sorted) - 0.4, 5.5,
            "Safe threshold (5%)",
            ha="right", va="bottom",
            fontsize=8, color=_GREEN, style="italic")

    ax.axhline(y=10.0, color=_GRAY, linewidth=1.0, linestyle=":", zorder=2)
    ax.text(len(conds_sorted) - 0.4, 10.5,
            "Monitoring threshold (10%)",
            ha="right", va="bottom",
            fontsize=8, color=_GRAY, style="italic")

    # Value labels on bars
    for bar, rate, cond in zip(bars, rates, conds_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2,
                rate + ci_hi[conds_sorted.index(cond)] + 0.5,
                f"{rate:.1f}%",
                ha="center", va="bottom",
                fontsize=8.5, color="#1E293B", zorder=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BLUE, alpha=0.70, label="N=100 rate"),
        Line2D([0], [0], color=_GRAY, lw=1.3, label="95% Wilson CI"),
        Line2D([0], [0], color=_RED, marker="D", lw=0, markersize=8,
               label="N=20 estimate"),
    ]
    ax.legend(handles=legend_handles, fontsize=9.5, loc="upper left",
              frameon=True, framealpha=0.95, edgecolor=_LGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(conds_sorted, fontsize=10)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_ylabel("Never-Recover Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, max(rates) * 1.35 + 5)
    ax.tick_params(labelsize=9)
    _clean_ax(ax)

    ax.set_title(
        "EXP-OP2-N100: Never-Recover Rate by Condition (N=100 vs N=20)",
        fontsize=12, pad=10
    )
    fig.text(
        0.09, 0.875,
        "Blue bars = N=100 with 95% Wilson CI.  Red diamonds = N=20 point estimates.  "
        "Sorted ascending by N=100 rate.",
        ha="left", va="bottom",
        fontsize=9, color="#475569", style="italic",
        transform=fig.transFigure,
    )

    save_figure(fig, "expOP2n_never_recover_ci", output_dir="paper_figures")
    print("[CHART 1] expOP2n_never_recover_ci.png + .pdf saved")


# ---------------------------------------------------------------------------
# Chart 2: T_recovery violin plots, bimodal conditions flagged salmon/red
# ---------------------------------------------------------------------------

def chart_t_recovery_violin(data: dict) -> None:
    plt.rcParams.update(_RC)

    t_arrays   = data["t_recovery_arrays"]
    nr         = data["summary"]["never_recover"]
    t_stats    = data["summary"]["t_rec_stats"]
    conditions_all = list(t_arrays.keys())

    # Sort ascending by median T_recovery
    conds_sorted = sorted(conditions_all,
                          key=lambda c: t_stats[c]["p50"])

    # Bimodal flag: std > 0.5 * mean (same criterion as bimodality_verdict)
    def _is_bimodal(cond: str) -> bool:
        arr = np.array(t_arrays[cond], dtype=float)
        m   = float(arr.mean())
        s   = float(arr.std())
        return s > 0.5 * m if m > 0 else False

    violin_data = [t_arrays[c] for c in conds_sorted]
    colors_v    = [_SALMON if _is_bimodal(c) else _STEEL_BLUE for c in conds_sorted]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.86)

    parts = ax.violinplot(
        violin_data,
        positions=np.arange(len(conds_sorted)),
        widths=0.65,
        showmedians=False,
        showextrema=False,
    )

    # Color each violin body + add median white dot
    for i, (pc, color) in enumerate(zip(parts["bodies"], colors_v)):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor("#334155")
        pc.set_linewidth(0.8)

    # Median white dot
    for i, cond in enumerate(conds_sorted):
        med = np.median(t_arrays[cond])
        ax.scatter([i], [med],
                   color="white", s=30, zorder=5,
                   edgecolors="#334155", linewidths=0.8)

    # IQR bar
    for i, cond in enumerate(conds_sorted):
        arr  = np.array(t_arrays[cond])
        p25  = float(np.percentile(arr, 25))
        p75  = float(np.percentile(arr, 75))
        ax.vlines(i, p25, p75, color="#334155", linewidth=2.5, zorder=4)

    # Reference lines
    sentinel_val = data["config"]["sentinel"]
    ax.axhline(y=sentinel_val, color=_RED, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(conds_sorted) - 0.4, sentinel_val + 2,
            f"Never-recover sentinel ({sentinel_val})",
            ha="right", va="bottom", fontsize=8, color=_RED, style="italic")

    op2_a_t_rec = 178
    ax.axhline(y=op2_a_t_rec, color=_ORANGE, linewidth=1.0, linestyle=":", zorder=2)
    ax.text(len(conds_sorted) - 0.4, op2_a_t_rec + 2,
            f"OP2 cond-A T_rec ({op2_a_t_rec})",
            ha="right", va="bottom", fontsize=8, color=_ORANGE, style="italic")

    # Legend for bimodal flag
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_STEEL_BLUE, alpha=0.65, edgecolor="#334155",
              label="Unimodal (std ≤ 0.5×mean)"),
        Patch(facecolor=_SALMON,     alpha=0.65, edgecolor="#334155",
              label="Bimodal flag (std > 0.5×mean)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9.5, loc="upper left",
              frameon=True, framealpha=0.95, edgecolor=_LGRAY)

    ax.set_xticks(np.arange(len(conds_sorted)))
    ax.set_xticklabels(conds_sorted, fontsize=10)
    ax.set_xlabel("Condition (sorted by median T_recovery)", fontsize=11)
    ax.set_ylabel("T_recovery (decisions)", fontsize=11)
    ax.tick_params(labelsize=9)
    _clean_ax(ax)

    ax.set_title(
        "EXP-OP2-N100: Recovery Speed by Condition (N=100 seeds)",
        fontsize=12, pad=10
    )
    fig.text(
        0.08, 0.875,
        "White dot = median.  Bar = IQR.  Salmon/red = bimodal flag (std > 0.5×mean).  "
        "Sentinel = 401 (never recovered).",
        ha="left", va="bottom",
        fontsize=9, color="#475569", style="italic",
        transform=fig.transFigure,
    )

    save_figure(fig, "expOP2n_t_recovery_violin", output_dir="paper_figures")
    print("[CHART 2] expOP2n_t_recovery_violin.png + .pdf saved")


# ---------------------------------------------------------------------------
# Chart 3: B-exp indirect path bimodality check
# ---------------------------------------------------------------------------

def chart_indirect_path(data: dict) -> None:
    plt.rcParams.update(_RC)

    bexp_vals_n100 = np.array(data["t_recovery_arrays"]["B-exp"], dtype=float)
    bimod          = data["summary"]["bimodality"]
    verdict        = bimod["verdict"]
    evidence       = bimod["evidence"]

    # Try to get B-exp N=20 raw data from OP2 results.json
    op2_path = Path("experiments/synthesis/expOP2_harmful/results.json")
    bexp_vals_n20 = None
    if op2_path.exists():
        try:
            with open(op2_path) as fh:
                op2_data = json.load(fh)
            raw_n20 = op2_data.get("per_seed_results", {}).get("B-exp", [])
            if raw_n20:
                bexp_vals_n20 = np.array([r["t_recovery"] for r in raw_n20], dtype=float)
        except Exception:
            pass

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.82, wspace=0.35)

    # --- Left panel: overlaid histograms ---
    n_bins = 30
    if bexp_vals_n20 is not None:
        ax_l.hist(bexp_vals_n20, bins=n_bins,
                  color=_RED, alpha=0.55, label=f"N=20 (std={float(bexp_vals_n20.std()):.0f})",
                  density=True)
        ax_l.hist(bexp_vals_n100, bins=n_bins,
                  color=_BLUE, alpha=0.55, label=f"N=100 (std={float(bexp_vals_n100.std()):.0f})",
                  density=True)
        ax_l.legend(fontsize=9.5, frameon=True, framealpha=0.95, edgecolor=_LGRAY)
        panel_title = "B-exp T_recovery: N=20 vs N=100"
    else:
        ax_l.hist(bexp_vals_n100, bins=n_bins,
                  color=_BLUE, alpha=0.70,
                  label=f"N=100 (std={float(bexp_vals_n100.std()):.0f})")
        ax_l.legend(fontsize=9.5, frameon=True, framealpha=0.95, edgecolor=_LGRAY)
        panel_title = "B-exp T_recovery: N=100 distribution"

    ax_l.set_xlabel("T_recovery (decisions)", fontsize=11)
    ax_l.set_ylabel("Density", fontsize=11)
    ax_l.set_title(panel_title, fontsize=12, pad=6)
    _clean_ax(ax_l)

    # --- Right panel: std comparison bar + verdict ---
    std_n100 = float(bexp_vals_n100.std())
    std_n20  = float(bexp_vals_n20.std()) if bexp_vals_n20 is not None else None

    bar_labels  = ["N=100"]
    bar_heights = [std_n100]
    bar_colors  = [_BLUE]
    if std_n20 is not None:
        bar_labels  = ["N=20", "N=100"]
        bar_heights = [std_n20, std_n100]
        bar_colors  = [_RED, _BLUE]

    x2 = np.arange(len(bar_labels))
    ax_r.bar(x2, bar_heights,
             width=0.45,
             color=bar_colors, alpha=0.72,
             edgecolor="white", linewidth=0.8,
             zorder=3)

    # Mean line for each bar
    for xi, (lbl, height) in enumerate(zip(bar_labels, bar_heights)):
        if lbl == "N=100":
            arr = bexp_vals_n100
        else:
            arr = bexp_vals_n20
        mean_val = float(arr.mean()) if arr is not None else 0
        half_thresh = 0.5 * mean_val
        ax_r.axhline(y=half_thresh, color=_ORANGE, linewidth=1.0, linestyle="--",
                     alpha=0.7, zorder=2)
        ax_r.text(xi, height + 3,
                  f"std={height:.0f}\nmean={mean_val:.0f}",
                  ha="center", va="bottom",
                  fontsize=8.5, color="#1E293B", zorder=5)

    # Large verdict annotation
    verdict_color = _RED if verdict == "BIMODAL" else _GREEN
    ax_r.text(0.5, 0.92, verdict,
              transform=ax_r.transAxes,
              ha="center", va="top",
              fontsize=22, fontweight="bold",
              color=verdict_color,
              bbox=dict(boxstyle="round,pad=0.4",
                        facecolor="#FEF2F2" if verdict == "BIMODAL" else "#F0FDF4",
                        edgecolor=verdict_color, alpha=0.90),
              zorder=6)

    ratio = evidence.get("std_over_mean_ratio", 0)
    ax_r.text(0.5, 0.70,
              f"std/mean = {ratio:.3f}\n(threshold = 0.50)",
              transform=ax_r.transAxes,
              ha="center", va="top",
              fontsize=10, color="#475569", style="italic")

    ax_r.set_xticks(x2)
    ax_r.set_xticklabels(bar_labels, fontsize=10)
    ax_r.set_ylabel("std(T_recovery)", fontsize=11)
    ax_r.set_title("std Comparison + Verdict", fontsize=12, pad=6)
    _clean_ax(ax_r)

    fig.suptitle(
        "EXP-OP2-N100: B-exp Indirect Path — Bimodality Check",
        fontsize=13, y=0.97,
    )

    save_figure(fig, "expOP2n_indirect_path_consistency", output_dir="paper_figures")
    print("[CHART 3] expOP2n_indirect_path_consistency.png + .pdf saved")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(data: dict) -> None:
    print("\nGenerating EXP-OP2-N100 charts ...")
    chart_never_recover_ci(data)
    chart_t_recovery_violin(data)
    chart_indirect_path(data)
    print("Done. 6 files in paper_figures/")


if __name__ == "__main__":
    _results_path = Path("experiments/synthesis/expOP2_n100/results.json")
    if not _results_path.exists():
        print(f"ERROR: {_results_path} not found. Run run.py first.")
        sys.exit(1)
    with open(_results_path) as fh:
        _data = json.load(fh)
    generate_charts(_data)
