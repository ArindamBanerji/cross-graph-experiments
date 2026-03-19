"""
PROD-4-A4 Charts

Chart 1 — prod4_a4_coverage_by_category:
  Per-category coverage vs threshold (A=4) with A=5 curves overlaid.
  Shows whether removing refer_to_analyst changed the confidence distribution.

Chart 2 — prod4_a4_threshold_table:
  Table visualization: recommended thresholds, accuracy, coverage for A=4.
  Side-by-side comparison with A=5 values.
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
import matplotlib.patches as mpatches
import numpy as np

from src.viz.bridge_common import save_figure

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

_THIS_DIR    = Path(__file__).parent
A4_RESULTS   = _THIS_DIR / "prod4_a4_results.json"
A5_RESULTS   = _REPO_ROOT / "experiments" / "prod" / "prod4_final" / "prod4_final_results.json"

with open(A4_RESULTS) as f:
    a4 = json.load(f)

a5_sweep: dict | None = None
a5_conf_thresholds: dict | None = None
try:
    with open(A5_RESULTS) as f:
        a5 = json.load(f)
    a5_sweep = a5.get("sweep_data", {})
    a5_conf_thresholds = a5.get("confidence_thresholds", {})
except FileNotFoundError:
    print("[WARN] prod4_final_results.json not found — A=5 overlay skipped")

CATEGORIES = list(a4["confidence_thresholds"].keys())

# Category display labels
CAT_LABELS = {
    "credential_access":    "Credential\nAccess",
    "threat_intel_match":   "Threat Intel\nMatch",
    "lateral_movement":     "Lateral\nMovement",
    "data_exfiltration":    "Data\nExfiltration",
    "insider_threat":       "Insider\nThreat",
    "cloud_infrastructure": "Cloud\nInfra",
}

# Colors per category
CAT_COLORS = [
    "#1565c0",  # credential_access
    "#2e7d32",  # threat_intel_match
    "#6a1b9a",  # lateral_movement
    "#e65100",  # data_exfiltration
    "#b71c1c",  # insider_threat
    "#00838f",  # cloud_infrastructure
]


# ============================================================================
# Chart 1 — Coverage vs threshold curves (A=4 solid, A=5 dashed)
# ============================================================================

def chart1_coverage_curves() -> None:
    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.42, wspace=0.30, bottom=0.10, top=0.88)

    a4_sweep = a4["sweep_data"]
    thresholds_a4 = sorted(float(k) for k in a4_sweep[CATEGORIES[0]].keys())

    for i, (cat, color) in enumerate(zip(CATEGORIES, CAT_COLORS)):
        ax = axes[i // 3][i % 3]

        # A=4 coverage curve
        cov4 = [a4_sweep[cat][f"{t:.2f}"]["coverage_mean"] for t in thresholds_a4]
        ax.plot(thresholds_a4, cov4, color=color, lw=2.2, ls="-",
                label="A=4 (this run)", zorder=4)

        # A=5 overlay
        if a5_sweep and cat in a5_sweep:
            # A=5 has different threshold keys (may extend to 0.99)
            thresholds_a5 = sorted(float(k) for k in a5_sweep[cat].keys()
                                   if float(k) <= 0.99)
            cov5 = [a5_sweep[cat][f"{t:.2f}"]["coverage_mean"] for t in thresholds_a5]
            ax.plot(thresholds_a5, cov5, color=color, lw=1.6, ls="--",
                    alpha=0.65, label="A=5 (prod4_final)", zorder=3)

        # Mark theta* (A=4)
        conf_thr = a4["confidence_thresholds"].get(cat, {})
        t_star   = conf_thr.get("threshold_star")
        cov_star = conf_thr.get("coverage_at_star")
        if t_star is not None and cov_star is not None:
            ax.axvline(t_star, color=color, lw=1.2, ls=":", alpha=0.8, zorder=3)
            ax.scatter([t_star], [cov_star], s=70, color=color,
                       zorder=6, edgecolors="black", linewidth=0.7)
            ax.text(t_star + 0.005, cov_star + 0.005,
                    f"θ*={t_star:.2f}\n{cov_star:.1%}",
                    fontsize=7.5, color=color, va="bottom")

        # Mark theta* (A=5) if available
        if a5_conf_thresholds and cat in a5_conf_thresholds:
            t5    = a5_conf_thresholds[cat].get("threshold_star")
            cov5s = a5_conf_thresholds[cat].get("coverage_at_star")
            if t5 is not None and cov5s is not None:
                ax.scatter([t5], [cov5s], s=50, color=color, marker="x",
                           zorder=5, linewidth=1.5, alpha=0.7)

        ax.axhline(0.20, color="#777", lw=0.9, ls=":", zorder=1, alpha=0.6)
        ax.axhline(0.40, color="#999", lw=0.7, ls=":", zorder=1, alpha=0.4)
        ax.set_xlim(0.48, 1.00)
        ax.set_ylim(-0.01, min(1.0, max(cov4) + 0.08))
        ax.set_title(CAT_LABELS.get(cat, cat), fontsize=10.5, fontweight="bold",
                     color=color, pad=4)
        ax.set_xlabel("Confidence threshold", fontsize=9)
        ax.set_ylabel("Coverage", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.88)

    fig.suptitle(
        "PROD-4-A4: Coverage vs Confidence Threshold — A=4 vs A=5 Overlay\n"
        "Solid = A=4 (refer_to_analyst removed), Dashed = A=5 (prod4_final). "
        "Dot = θ*, ✕ = A=5 θ*",
        fontsize=13, fontweight="bold",
    )

    caption = (
        f"soc_product_v50 A=4, C=6, d=6. "
        f"{a4['n_seeds']} seeds, warmup={a4['n_warmup']}, "
        f"decisions={a4['n_decisions']}, τ={a4['tau']}, "
        f"η_neg={a4['eta_neg']}, noise={a4['noise_rate']:.0%}. "
        "Accuracy gate = 85%."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "prod4_a4_coverage_by_category", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] prod4_a4_coverage_by_category.png + .pdf saved")


# ============================================================================
# Chart 2 — Threshold recommendation table (A=4 vs A=5)
# ============================================================================

def chart2_threshold_table() -> None:
    # A=5 reference (fallback to hardcoded if JSON unavailable)
    a5_ref_fallback = {
        "credential_access":    {"threshold_star": 0.72, "coverage_at_star": 0.067},
        "data_exfiltration":    {"threshold_star": 0.73, "coverage_at_star": 0.072},
        "lateral_movement":     {"threshold_star": 0.79, "coverage_at_star": 0.043},
        "threat_intel_match":   {"threshold_star": 0.81, "coverage_at_star": 0.033},
        "cloud_infrastructure": {"threshold_star": 0.82, "coverage_at_star": 0.029},
        "insider_threat":       {"threshold_star": 0.87, "coverage_at_star": 0.020},
    }

    a5_ref = a5_conf_thresholds if a5_conf_thresholds else a5_ref_fallback

    n_cats = len(CATEGORIES)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis("off")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.03, right=0.97)

    col_labels = [
        "Category",
        "θ* (A=5)", "Acc (A=5)", "Cov (A=5)",
        "θ* (A=4)", "Acc (A=4)", "Cov (A=4)",
        "Δ Coverage",
    ]
    col_widths = [0.22, 0.09, 0.09, 0.09, 0.09, 0.10, 0.09, 0.11]

    rows = []
    for cat in CATEGORIES:
        c4   = a4["confidence_thresholds"][cat]
        ref5 = a5_ref.get(cat, {})

        t4   = c4.get("threshold_star")
        acc4 = c4.get("accuracy_at_star")
        cov4 = c4.get("coverage_at_star")

        t5   = ref5.get("threshold_star")
        cov5 = ref5.get("coverage_at_star")
        acc5_val = ref5.get("accuracy_at_star") if a5_conf_thresholds else None

        t4_s   = f"{t4:.2f}"   if t4   is not None else "BELOW GATE"
        acc4_s = f"{acc4:.1%}" if acc4 is not None else "---"
        cov4_s = f"{cov4:.1%}" if cov4 is not None else "---"
        t5_s   = f"{t5:.2f}"   if t5   is not None else "---"
        acc5_s = f"{acc5_val:.1%}" if acc5_val is not None else "---"
        cov5_s = f"{cov5:.1%}" if cov5 is not None else "---"

        if cov4 is not None and cov5 is not None:
            delta  = cov4 - cov5
            delta_s = f"{delta:+.1%}"
        else:
            delta_s = "---"
            delta   = 0.0

        rows.append((cat, t5_s, acc5_s, cov5_s, t4_s, acc4_s, cov4_s, delta_s, delta))

    # Draw table
    header_y = 0.88
    row_h    = 0.11
    x_starts = [0.01]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    # Header row
    for j, (label, xs) in enumerate(zip(col_labels, x_starts)):
        ax.text(xs, header_y, label, transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", va="center",
                color="#1a1a1a")

    ax.plot([0.01, 0.99], [header_y - 0.05, header_y - 0.05],
            color="#333", lw=1.2, transform=ax.transAxes)

    for ri, row_data in enumerate(rows):
        y        = header_y - 0.10 - ri * row_h
        cat_name = row_data[0]
        color    = CAT_COLORS[CATEGORIES.index(cat_name)]
        delta    = row_data[8]

        # Alternate row background
        bg_color = "#f5f5f5" if ri % 2 == 0 else "white"
        bg = mpatches.FancyBboxPatch(
            (0.005, y - row_h * 0.5), 0.99, row_h * 0.9,
            boxstyle="square,pad=0",
            facecolor=bg_color, edgecolor="none",
            transform=ax.transAxes, zorder=0,
        )
        ax.add_patch(bg)

        # Category name (colored)
        ax.text(x_starts[0], y, cat_name, transform=ax.transAxes,
                fontsize=9.5, va="center", color=color, fontweight="bold")

        # A=5 values (gray)
        for j, val in enumerate(row_data[1:4], start=1):
            ax.text(x_starts[j], y, val, transform=ax.transAxes,
                    fontsize=9.5, va="center", color="#666666")

        # A=4 values (colored)
        for j, val in enumerate(row_data[4:7], start=4):
            ax.text(x_starts[j], y, val, transform=ax.transAxes,
                    fontsize=9.5, va="center", color=color)

        # Delta (color-coded)
        delta_color = "#2e7d32" if delta >= 0.10 else (
                      "#1565c0" if delta > 0.0 else
                      "#c62828" if delta < 0.0 else "#555555")
        ax.text(x_starts[7], y, row_data[7], transform=ax.transAxes,
                fontsize=9.5, va="center", color=delta_color,
                fontweight="bold" if abs(delta) >= 0.10 else "normal")

    # Horizontal line after last row
    bottom_y = header_y - 0.10 - (n_cats - 1) * row_h - row_h * 0.55
    ax.plot([0.01, 0.99], [bottom_y, bottom_y],
            color="#333", lw=0.8, transform=ax.transAxes)

    ax.set_title(
        "PROD-4-A4: Threshold Recommendation Table — A=4 vs A=5 Comparison\n"
        "Green Δ = coverage gain ≥ 10pp.  Accuracy gate = 85%.  "
        "soc_product_v50 (Phase 0a)",
        fontsize=12, fontweight="bold", pad=10,
    )

    caption = (
        f"{a4['n_seeds']} seeds, warmup={a4['n_warmup']}, "
        f"decisions={a4['n_decisions']}, τ={a4['tau']}, "
        f"η_neg={a4['eta_neg']}, noise={a4['noise_rate']:.0%}."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9,
             color="#555", style="italic")

    save_figure(fig, "prod4_a4_threshold_table", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] prod4_a4_threshold_table.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    chart1_coverage_curves()
    chart2_threshold_table()
