"""
PROD-4 charts: Per-Category Auto-Approve Threshold Calibration.

Regime: centroidal synthetic
Reads: experiments/prod/prod4_threshold_calibration/prod4_threshold_table.json
Writes: paper_figures/prod4_*.{pdf,png}

Charts:
  1. prod4_accuracy_vs_threshold_by_category  — 6 curves, gate line, threshold* dots
  2. prod4_coverage_vs_threshold_by_category  — 6 curves, reference lines, threshold* dots
  3. prod4_threshold_recommendation_table     — matplotlib table with colour coding
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.viz.bridge_common import save_figure, COLORS, VIZ_DEFAULTS

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod4_threshold_calibration" / "prod4_threshold_table.json"

with open(RESULTS_PATH) as fh:
    results = json.load(fh)

categories  = list(results["categories"].keys())
sweep_data  = results["sweep_data"]
cat_data    = results["categories"]
floors      = results["refer_to_analyst_floors"]
gate        = results["accuracy_gate"]
ontology    = results["ontology"]
C, A        = ontology["C"], ontology["A"]

# Sorted threshold values
thresholds = sorted([float(k) for k in sweep_data[categories[0]].keys()])

# 6-color palette: 5 existing + orange for cloud_infrastructure
CAT_COLORS = COLORS["category_colors"] + ["#EA580C"]

# ---------------------------------------------------------------------------
# Chart 1 — Accuracy vs threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for c_idx, cat in enumerate(categories):
    acc_means = []
    for t in thresholds:
        entry = sweep_data[cat][f"{t:.2f}"]
        v = entry["accuracy_mean"]
        acc_means.append(v if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan)

    ax.plot(thresholds, acc_means,
            color=CAT_COLORS[c_idx], linewidth=1.6, label=cat, alpha=0.9)

    # Mark threshold*
    t_star = cat_data[cat]["threshold_star"]
    acc_star = cat_data[cat]["accuracy_at_star"]
    if t_star is not None and acc_star is not None:
        ax.scatter([t_star], [acc_star],
                   color=CAT_COLORS[c_idx], s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

# Gate line
ax.axhline(gate, color="#374151", linewidth=1.4, linestyle="--",
           label=f"Gate {gate:.0%}")

ax.set_xlabel("Confidence threshold", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Accuracy (mean over 50 seeds)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve accuracy vs threshold (centroidal synthetic, 50 seeds, A={A})",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xlim(0.50, 0.99)
ax.set_ylim(0.5, 1.02)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="lower right",
          ncol=2, framealpha=0.85)

fig.tight_layout()
save_figure(fig, "prod4_accuracy_vs_threshold_by_category", output_dir="paper_figures")
plt.close()
print("[CHART 1] saved")

# ---------------------------------------------------------------------------
# Chart 2 — Coverage vs threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for c_idx, cat in enumerate(categories):
    cov_means = [sweep_data[cat][f"{t:.2f}"]["coverage_mean"] for t in thresholds]
    ax.plot(thresholds, cov_means,
            color=CAT_COLORS[c_idx], linewidth=1.6, label=cat, alpha=0.9)

    # Mark threshold*
    t_star = cat_data[cat]["threshold_star"]
    cov_star = cat_data[cat]["coverage_at_star"]
    if t_star is not None and cov_star is not None:
        ax.scatter([t_star], [cov_star],
                   color=CAT_COLORS[c_idx], s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

# Reference lines
ax.axvline(0.90, color="#374151", linewidth=1.4, linestyle="--",
           label="Current global threshold (0.90)")
ax.axhline(0.151, color="#7C3AED", linewidth=1.2, linestyle=":",
           label="PROD-3 global coverage (15.1%)")

ax.set_xlabel("Confidence threshold", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Coverage (fraction of decisions)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve coverage vs threshold (centroidal synthetic, 50 seeds, A={A})",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xlim(0.50, 0.99)
ax.set_ylim(0, 1.0)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="upper right",
          ncol=2, framealpha=0.85)

fig.tight_layout()
save_figure(fig, "prod4_coverage_vs_threshold_by_category", output_dir="paper_figures")
plt.close()
print("[CHART 2] saved")

# ---------------------------------------------------------------------------
# Chart 3 — Threshold recommendation table (matplotlib table)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")

col_labels = ["Category", "threshold*", "accuracy", "coverage", "refer floor"]
row_data   = []
row_colors = []

GREEN  = "#D1FAE5"
ORANGE = "#FEF3C7"
RED    = "#FEE2E2"
HEADER = "#E2E8F0"

for cat in categories:
    t_star   = cat_data[cat]["threshold_star"]
    acc      = cat_data[cat]["accuracy_at_star"]
    cov      = cat_data[cat]["coverage_at_star"]
    ci       = cat_data[cat]["accuracy_ci_at_star"]
    floor_v  = floors[cat]

    if t_star is None:
        row_data.append([cat, "BELOW GATE", "---", "---", f"{floor_v:.2f}"])
        row_colors.append([RED] * 5)
    elif t_star > 0.75:
        ci_str = f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}]" if ci else f"{acc:.1%}"
        row_data.append([cat, f"{t_star:.2f}", ci_str, f"{cov:.1%}", f"{floor_v:.2f}"])
        row_colors.append([ORANGE] * 5)
    else:
        ci_str = f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}]" if ci else f"{acc:.1%}"
        row_data.append([cat, f"{t_star:.2f}", ci_str, f"{cov:.1%}", f"{floor_v:.2f}"])
        row_colors.append([GREEN] * 5)

table = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=row_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.6)

# Style header row
for col_idx in range(len(col_labels)):
    table[0, col_idx].set_facecolor(HEADER)
    table[0, col_idx].set_text_props(fontweight="bold")

ax.set_title(
    f"Recommended per-category thresholds — v5.5 (PROD-4)\n"
    f"Gate: accuracy >= {gate:.0%}  |  centroidal synthetic, 50 seeds",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
    pad=12,
)

# Legend patches
legend_patches = [
    mpatches.Patch(color=GREEN,  label="threshold* ≤ 0.75 and accuracy ≥ 85%"),
    mpatches.Patch(color=ORANGE, label="threshold* > 0.75"),
    mpatches.Patch(color=RED,    label="below gate"),
]
ax.legend(handles=legend_patches, fontsize=VIZ_DEFAULTS["annotation_fontsize"],
          loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, framealpha=0.9)

fig.tight_layout()
save_figure(fig, "prod4_threshold_recommendation_table", output_dir="paper_figures")
plt.close()
print("[CHART 3] saved")
