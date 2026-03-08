"""
charts.py — Publication-quality figures for FX-T5-BREAKDOWN.
experiments/fx_t5_breakdown/charts.py

4 charts:
  1. fx_t5_action_distribution  — stacked bar: band vs overall distribution
  2. fx_t5_per_action_accuracy  — horizontal bars with 95% CI and thresholds
  3. fx_t5_error_direction      — stacked bar of error types
  4. fx_t5_cost_weighted        — cost-benefit waterfall under 20:1 asymmetry
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.viz.bridge_common import VIZ_DEFAULTS, save_figure
from experiments.fx1_proxy_real.realistic_generator import SOC_ACTIONS, SOC_CATEGORIES

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

ACTION_COLORS = {
    "escalate":    "#1D4ED8",   # dark blue   — threat
    "investigate": "#60A5FA",   # light blue  — threat
    "suppress":    "#DC2626",   # red         — caution (high-cost)
    "monitor":     "#F97316",   # orange      — caution (high-cost)
}
ERROR_COLORS = {
    "dangerous":      "#DC2626",
    "safe":           "#D97706",
    "over_escalation": "#2563EB",
}
THREAT_ACTIONS  = {0, 1}
CAUTION_ACTIONS = {2, 3}

COST_RATIO          = 20.0
TARGET_ACC_CAUTION  = 0.99
TARGET_ACC_THREAT   = 0.90


# ---------------------------------------------------------------------------
# Chart 1 — Action distribution: band vs overall
# ---------------------------------------------------------------------------

def chart_action_distribution(
    action_count: dict[int, int],
    action_pct:   dict[int, float],
    all_decisions: list,
    total_band:   int,
    paper_dir:    str,
) -> None:
    """
    Side-by-side stacked bar: auto-approve band distribution vs overall GT distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Overall GT distribution (from all band decisions' GT labels)
    gt_count = {i: sum(1 for d in all_decisions if d.gt == i) for i in range(4)}
    gt_total = sum(gt_count.values())

    bars_data = {
        "Auto-approve\nband (predicted)": [action_pct[i] * 100 for i in range(4)],
        "Overall GT\ndistribution":       [gt_count[i] / gt_total * 100 for i in range(4)],
    }

    x       = np.arange(len(bars_data))
    bottoms = [0.0, 0.0]
    colors  = [ACTION_COLORS[a] for a in SOC_ACTIONS]

    for act_idx, act_name in enumerate(SOC_ACTIONS):
        vals = [bars_data[label][act_idx] for label in bars_data]
        bars = ax.bar(x, vals, bottom=bottoms, width=0.5,
                      color=colors[act_idx], label=act_name, zorder=3)
        # Annotate segments
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(xi, b + v / 2, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(list(bars_data.keys()), fontsize=VIZ_DEFAULTS["tick_fontsize"] + 1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.legend(fontsize=8, loc="upper right",
              title="Action (red/orange = caution, blue = threat)")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "FX-T5: Action Distribution — Auto-Approve Band vs Overall GT\n"
        f"Band = predicted action at conf ≥0.90. "
        f"Red/orange = suppress/monitor (high-cost).",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    fig.tight_layout()
    save_figure(fig, "fx_t5_action_distribution", paper_dir)


# ---------------------------------------------------------------------------
# Chart 2 — Per-action accuracy with CIs and thresholds
# ---------------------------------------------------------------------------

def chart_per_action_accuracy(
    per_action_stats:       dict[int, dict],
    recommended_thresholds: dict[int, Any],
    paper_dir:              str,
) -> None:
    """
    Horizontal bars: per-action accuracy ± 95% CI.
    Reference lines at 90% and 99%. Threshold annotation on each bar.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    y_pos   = np.arange(4)
    accs    = [per_action_stats[i]["accuracy"] * 100 for i in range(4)]
    ci_los  = [per_action_stats[i]["ci_lo"]    * 100 for i in range(4)]
    ci_his  = [per_action_stats[i]["ci_hi"]    * 100 for i in range(4)]
    colors  = [ACTION_COLORS[SOC_ACTIONS[i]] for i in range(4)]

    for i in range(4):
        acc    = accs[i]
        ci_lo  = ci_los[i]
        ci_hi  = ci_his[i]
        xerr_lo = acc - ci_lo
        xerr_hi = ci_hi - acc
        ax.barh(y_pos[i], acc, height=0.5, color=colors[i], alpha=0.85, zorder=3)
        ax.errorbar(acc, y_pos[i],
                    xerr=[[xerr_lo], [xerr_hi]],
                    fmt="none", color="#374151", capsize=5,
                    elinewidth=1.5, zorder=4)

        # Annotate threshold
        thr = recommended_thresholds[i]
        thr_str = (f"thr: {thr:.2f}" if isinstance(thr, float)
                   else f"thr: {thr}")
        ax.text(ci_hi + 0.3, y_pos[i],
                f"{acc:.1f}%  ({thr_str})",
                va="center", ha="left", fontsize=8, color=colors[i])

    # Reference lines
    ax.axvline(90.0, color="#2563EB", linewidth=1.5, linestyle="--", alpha=0.7,
               zorder=2, label="90% — threat action target")
    ax.axvline(99.0, color="#DC2626", linewidth=1.5, linestyle="--", alpha=0.7,
               zorder=2, label="99% — suppress/monitor target")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(SOC_ACTIONS, fontsize=VIZ_DEFAULTS["tick_fontsize"] + 1)
    ax.set_xlim(
        max(0, min(ci_los) - 3),
        min(102, max(ci_his) + 14),
    )
    ax.set_xlabel("Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])

    # Shade suppress/monitor rows
    for i in CAUTION_ACTIONS:
        ax.axhspan(y_pos[i] - 0.32, y_pos[i] + 0.32, alpha=0.06,
                   color="#DC2626", zorder=1)

    ax.set_title(
        "FX-T5: Per-Action Accuracy in Auto-Approve Band (conf ≥0.90)\n"
        "Red/orange = high-cost actions. 99% bar = target for suppress/monitor. "
        "Error bars = 95% CI across 50 seeds.",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    fig.tight_layout()
    save_figure(fig, "fx_t5_per_action_accuracy", paper_dir)


# ---------------------------------------------------------------------------
# Chart 3 — Error direction stacked bar
# ---------------------------------------------------------------------------

def chart_error_direction(
    n_dangerous:          int,
    n_safe:               int,
    n_over_esc:           int,
    total_errors:         int,
    total_band:           int,
    dangerous_error_rate: float,
    paper_dir:            str,
) -> None:
    """
    Stacked bar of error types + annotation of dangerous rate as % of all band decisions.
    """
    if total_errors == 0:
        print("  [chart_error_direction] No errors — skipping chart.")
        return

    fig, (ax_pct, ax_abs) = plt.subplots(1, 2, figsize=(10, 4.5))

    categories = ["Dangerous\n(suppress/monitor\npredicted, GT=threat)",
                  "Safe\n(within-tier\nconfusion)",
                  "Over-escalation\n(threat predicted,\nGT=suppress/monitor)"]
    counts     = [n_dangerous, n_safe, n_over_esc]
    pcts       = [c / total_errors * 100 for c in counts]
    colors_err = [ERROR_COLORS["dangerous"], ERROR_COLORS["safe"],
                  ERROR_COLORS["over_escalation"]]
    x          = np.arange(len(categories))

    # Left: % of errors
    bars = ax_pct.bar(x, pcts, color=colors_err, width=0.55, zorder=3, alpha=0.87)
    for bar, pct in zip(bars, pcts):
        ax_pct.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"], fontweight="bold")

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(categories, fontsize=7)
    ax_pct.set_ylabel("% of all errors", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax_pct.set_ylim(0, max(pcts) * 1.25)
    ax_pct.set_title("Error Type Distribution\n(% of errors)",
                     fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax_pct.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_pct.spines["top"].set_visible(False)
    ax_pct.spines["right"].set_visible(False)

    # Right: % of ALL band decisions
    band_pcts  = [c / total_band * 100 for c in counts]
    bars2 = ax_abs.bar(x, band_pcts, color=colors_err, width=0.55, zorder=3, alpha=0.87)
    for bar, pct in zip(bars2, band_pcts):
        ax_abs.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{pct:.3f}%", ha="center", va="bottom",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"], fontweight="bold")

    # Annotate dangerous rate explicitly
    ax_abs.annotate(
        f"Dangerous rate:\n{dangerous_error_rate*100:.3f}% of band",
        xy=(0, band_pcts[0]),
        xytext=(0.3, band_pcts[0] + max(band_pcts) * 0.25),
        fontsize=8.5, color=ERROR_COLORS["dangerous"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ERROR_COLORS["dangerous"], lw=1.2),
    )

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(categories, fontsize=7)
    ax_abs.set_ylabel("% of ALL auto-approve decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax_abs.set_ylim(0, max(band_pcts) * 1.45)
    ax_abs.set_title("Error Rate per Type\n(% of all band decisions)",
                     fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax_abs.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_abs.spines["top"].set_visible(False)
    ax_abs.spines["right"].set_visible(False)

    fig.suptitle(
        "FX-T5: Error Direction in Auto-Approve Band\n"
        "Dangerous = suppress/monitor predicted when GT was escalate/investigate (missed threat).",
        fontsize=VIZ_DEFAULTS["title_fontsize"] + 0.5,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fx_t5_error_direction", paper_dir)


# ---------------------------------------------------------------------------
# Chart 4 — Cost-weighted waterfall
# ---------------------------------------------------------------------------

def chart_cost_weighted(
    correct_decisions:  int,
    n_dangerous:        int,
    other_errors:       int,
    cost_weighted_score: float,
    cost_weighted_acc:  float,
    total_band:         int,
    paper_dir:          str,
) -> None:
    """
    Waterfall chart: value from correct decisions vs cost of dangerous errors.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = [
        f"Correct\ndecisions\n(×+1)",
        f"Dangerous\nerrors\n(×−{COST_RATIO:.0f})",
        f"Other\nerrors\n(×−1)",
        "Net score",
    ]
    values = [
        correct_decisions * 1.0,
        -n_dangerous * COST_RATIO,
        -other_errors * 1.0,
        cost_weighted_score,
    ]
    colors_wf = ["#059669", "#DC2626", "#D97706",
                 "#2563EB" if cost_weighted_score > 0 else "#DC2626"]

    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors_wf, width=0.55, zorder=3, alpha=0.87)

    for i, (bar, val) in enumerate(zip(bars, values)):
        offset = max(abs(val) * 0.015, 80)
        va     = "bottom" if val >= 0 else "top"
        y_ann  = val + (offset if val >= 0 else -offset)
        ax.text(x[i], y_ann, f"{val:+,.0f}",
                ha="center", va=va, fontsize=9, fontweight="bold",
                color=colors_wf[i])

    ax.axhline(0, color="#374151", linewidth=1.0, zorder=2)

    # Annotate net score interpretation
    net_str = (f"Net positive: {cost_weighted_acc:+.3f}\nAuto-approve viable"
               if cost_weighted_score > 0
               else f"Net negative: {cost_weighted_acc:+.3f}\nAuto-approve harmful")
    text_color = "#059669" if cost_weighted_score > 0 else "#DC2626"
    ax.text(0.98, 0.95, net_str, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=text_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # Nominal accuracy annotation
    ax.text(0.02, 0.95,
            f"Nominal acc: {correct_decisions/total_band*100:.2f}%\n"
            f"Cost-weighted: {cost_weighted_acc:+.4f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=VIZ_DEFAULTS["tick_fontsize"] + 0.5)
    ax.set_ylabel("Score (decisions × weight)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_title(
        f"FX-T5: Cost-Weighted Auto-Approve Analysis ({COST_RATIO:.0f}:1 Asymmetry)\n"
        f"Dangerous error (missed threat) = {COST_RATIO:.0f}× cost of escalation error. "
        f"N={total_band:,} band decisions.",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    fig.tight_layout()
    save_figure(fig, "fx_t5_cost_weighted", paper_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(
    action_count:           dict,
    action_pct:             dict,
    per_action_stats:       dict,
    total_band:             int,
    n_dangerous:            int,
    n_safe:                 int,
    n_over_esc:             int,
    total_errors:           int,
    correct_decisions:      int,
    dangerous_error_rate:   float,
    cost_weighted_acc:      float,
    cost_weighted_score:    float,
    recommended_thresholds: dict,
    threshold_coverage:     dict,
    all_decisions:          list,
    paper_dir:              str,
) -> None:
    """Generate all 4 FX-T5 charts."""
    print("\nGenerating FX-T5 charts...")
    chart_action_distribution(
        action_count=action_count,
        action_pct=action_pct,
        all_decisions=all_decisions,
        total_band=total_band,
        paper_dir=paper_dir,
    )
    chart_per_action_accuracy(
        per_action_stats=per_action_stats,
        recommended_thresholds=recommended_thresholds,
        paper_dir=paper_dir,
    )
    chart_error_direction(
        n_dangerous=n_dangerous,
        n_safe=n_safe,
        n_over_esc=n_over_esc,
        total_errors=total_errors,
        total_band=total_band,
        dangerous_error_rate=dangerous_error_rate,
        paper_dir=paper_dir,
    )
    chart_cost_weighted(
        correct_decisions=correct_decisions,
        n_dangerous=n_dangerous,
        other_errors=n_safe + n_over_esc,
        cost_weighted_score=cost_weighted_score,
        cost_weighted_acc=cost_weighted_acc,
        total_band=total_band,
        paper_dir=paper_dir,
    )
    print("All 4 charts saved (PDF + PNG).")
