"""
PROD-4 RERUN charts: Per-Category Threshold Calibration at η_neg=0.05.

Regime: centroidal synthetic
Reads: experiments/prod/prod4_rerun_eta005/prod4_rerun_results.json
Writes: paper_figures/prod4r_*.{pdf,png}

Charts:
  1. prod4r_accuracy_vs_threshold_by_category  — confidence curves, 6 cats, gate line
  2. prod4r_coverage_vs_threshold_by_category  — confidence coverage curves
  3. prod4r_threshold_recommendation_table     — matplotlib table, colour-coded
  4. prod4r_margin_vs_accuracy_by_category     — margin curves, 6 cats, gate line
  5. prod4r_confidence_vs_margin_comparison    — 6-panel: conf (blue) vs margin (red)
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

RESULTS_PATH = (
    _REPO_ROOT / "experiments" / "prod" / "prod4_rerun_eta005" / "prod4_rerun_results.json"
)
with open(RESULTS_PATH) as fh:
    results = json.load(fh)

categories     = list(results["categories"].keys())
conf_sweep     = results["sweep_data"]
margin_sweep   = results["margin_sweep_data"]
cat_data       = results["categories"]
floors         = results["refer_to_analyst_floors"]
gate           = results["accuracy_gate"]
ontology       = results["ontology"]
C, A           = ontology["C"], ontology["A"]

conf_thresholds   = sorted([float(k) for k in conf_sweep[categories[0]].keys()])
margin_thresholds = sorted([float(k) for k in margin_sweep[categories[0]].keys()])

# 6-color palette
CAT_COLORS = COLORS["category_colors"] + ["#EA580C"]

GREEN  = "#D1FAE5"
ORANGE = "#FEF3C7"
RED    = "#FEE2E2"
HEADER = "#E2E8F0"

# ---------------------------------------------------------------------------
# Chart 1 — Accuracy vs confidence threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for c_idx, cat in enumerate(categories):
    acc_means = [
        conf_sweep[cat][f"{t:.2f}"]["accuracy_mean"]
        for t in conf_thresholds
    ]
    # Replace None / nan for plotting
    acc_plot = [v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
                else np.nan for v in acc_means]
    ax.plot(conf_thresholds, acc_plot,
            color=CAT_COLORS[c_idx], linewidth=1.6, label=cat, alpha=0.9)

    t_star   = cat_data[cat]["threshold_star"]
    acc_star = cat_data[cat]["accuracy_at_star"]
    if t_star is not None and acc_star is not None:
        ax.scatter([t_star], [acc_star], color=CAT_COLORS[c_idx],
                   s=60, zorder=5, edgecolors="white", linewidths=0.8)

ax.axhline(gate, color="#374151", linewidth=1.4, linestyle="--",
           label=f"Gate {gate:.0%}")
ax.set_xlabel("Confidence threshold", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Accuracy (mean, 50 seeds)",  fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve accuracy vs confidence threshold\n"
    f"(η_neg=0.05, centroidal synthetic, 50 seeds, A={A})",
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
save_figure(fig, "prod4r_accuracy_vs_threshold_by_category", output_dir="paper_figures")
plt.close()
print("[CHART 1] saved")

# ---------------------------------------------------------------------------
# Chart 2 — Coverage vs confidence threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for c_idx, cat in enumerate(categories):
    cov_means = [conf_sweep[cat][f"{t:.2f}"]["coverage_mean"] for t in conf_thresholds]
    ax.plot(conf_thresholds, cov_means,
            color=CAT_COLORS[c_idx], linewidth=1.6, label=cat, alpha=0.9)

    t_star   = cat_data[cat]["threshold_star"]
    cov_star = cat_data[cat]["coverage_at_star"]
    if t_star is not None and cov_star is not None:
        ax.scatter([t_star], [cov_star], color=CAT_COLORS[c_idx],
                   s=60, zorder=5, edgecolors="white", linewidths=0.8)

ax.axvline(0.90, color="#374151", linewidth=1.4, linestyle="--",
           label="Current global threshold (0.90)")
ax.axhline(0.151, color="#7C3AED", linewidth=1.2, linestyle=":",
           label="PROD-3 global coverage (15.1%)")
ax.set_xlabel("Confidence threshold",                  fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Coverage (fraction of all decisions)",  fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve coverage vs confidence threshold\n"
    f"(η_neg=0.05, centroidal synthetic, 50 seeds, A={A})",
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
save_figure(fig, "prod4r_coverage_vs_threshold_by_category", output_dir="paper_figures")
plt.close()
print("[CHART 2] saved")

# ---------------------------------------------------------------------------
# Chart 3 — Threshold recommendation table (colour-coded)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.axis("off")

col_labels = ["Category", "conf threshold*", "accuracy (CI)", "coverage",
              "margin*", "margin acc", "margin cov", "refer floor"]
row_data, row_colors = [], []

for cat in categories:
    d       = cat_data[cat]
    t_star  = d["threshold_star"]
    acc     = d["accuracy_at_star"]
    cov     = d["coverage_at_star"]
    ci      = d["accuracy_ci_at_star"]
    m_star  = d["margin_star"]
    acc_m   = d["accuracy_at_mstar"]
    cov_m   = d["coverage_at_mstar"]
    floor_v = floors[cat]

    if t_star is None:
        t_str   = "BELOW GATE"
        acc_str = "---"
        cov_str = "---"
        cell_c  = RED
    elif t_star > 0.75:
        t_str   = f"{t_star:.2f}"
        acc_str = f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}]"
        cov_str = f"{cov:.1%}"
        cell_c  = ORANGE
    else:
        t_str   = f"{t_star:.2f}"
        acc_str = f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}]"
        cov_str = f"{cov:.1%}"
        cell_c  = GREEN

    m_str   = f"{m_star:.2f}"  if m_star  is not None else "---"
    am_str  = f"{acc_m:.1%}"   if acc_m   is not None else "---"
    cm_str  = f"{cov_m:.1%}"   if cov_m   is not None else "---"

    row_data.append([cat, t_str, acc_str, cov_str, m_str, am_str, cm_str, f"{floor_v:.2f}"])
    row_colors.append([cell_c] * 8)

table = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=row_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1.0, 1.65)

for col_idx in range(len(col_labels)):
    table[0, col_idx].set_facecolor(HEADER)
    table[0, col_idx].set_text_props(fontweight="bold")

ax.set_title(
    f"Recommended per-category thresholds — v5.5 (PROD-4 RERUN, η_neg=0.05)\n"
    f"Gate: accuracy ≥ {gate:.0%}  |  centroidal synthetic, 50 seeds",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
    pad=12,
)
legend_patches = [
    mpatches.Patch(color=GREEN,  label="threshold* ≤ 0.75 and accuracy ≥ 85%"),
    mpatches.Patch(color=ORANGE, label="threshold* > 0.75"),
    mpatches.Patch(color=RED,    label="below gate"),
]
ax.legend(handles=legend_patches, fontsize=VIZ_DEFAULTS["annotation_fontsize"],
          loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, framealpha=0.9)
fig.tight_layout()
save_figure(fig, "prod4r_threshold_recommendation_table", output_dir="paper_figures")
plt.close()
print("[CHART 3] saved")

# ---------------------------------------------------------------------------
# Chart 4 — Accuracy vs margin threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

for c_idx, cat in enumerate(categories):
    acc_means = [
        margin_sweep[cat][f"{m:.2f}"]["accuracy_mean"]
        for m in margin_thresholds
    ]
    acc_plot = [v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
                else np.nan for v in acc_means]
    ax.plot(margin_thresholds, acc_plot,
            color=CAT_COLORS[c_idx], linewidth=1.8, marker="o", markersize=5,
            label=cat, alpha=0.9)

    m_star   = cat_data[cat]["margin_star"]
    acc_mstar = cat_data[cat]["accuracy_at_mstar"]
    if m_star is not None and acc_mstar is not None:
        ax.scatter([m_star], [acc_mstar], color=CAT_COLORS[c_idx],
                   s=80, zorder=5, edgecolors="white", linewidths=1.0)

ax.axhline(gate, color="#374151", linewidth=1.4, linestyle="--",
           label=f"Gate {gate:.0%}")
ax.set_xlabel("Margin threshold (top1 − top2 probability)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Accuracy (mean, 50 seeds)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve accuracy vs margin (η_neg=0.05, 50 seeds, A={A})",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xlim(0.25, 0.85)
ax.set_ylim(0.5, 1.02)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="lower right",
          ncol=2, framealpha=0.85)
fig.tight_layout()
save_figure(fig, "prod4r_margin_vs_accuracy_by_category", output_dir="paper_figures")
plt.close()
print("[CHART 4] saved")

# ---------------------------------------------------------------------------
# Chart 5 — Confidence vs Margin comparison (6-panel, one per category)
# ---------------------------------------------------------------------------

ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7))
axes_flat  = axes.flatten()

BLUE = COLORS["gt_noise_0"]
RED  = COLORS["gt_noise_30"]

for c_idx, cat in enumerate(categories):
    ax = axes_flat[c_idx]

    # Confidence curve (blue)
    acc_conf = [
        conf_sweep[cat][f"{t:.2f}"]["accuracy_mean"]
        for t in conf_thresholds
    ]
    acc_conf_plot = [v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
                     else np.nan for v in acc_conf]
    ax.plot(conf_thresholds, acc_conf_plot,
            color=BLUE, linewidth=1.6, label="Confidence", alpha=0.9)

    t_star   = cat_data[cat]["threshold_star"]
    acc_star = cat_data[cat]["accuracy_at_star"]
    if t_star is not None and acc_star is not None:
        ax.scatter([t_star], [acc_star], color=BLUE, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"c*={t_star:.2f}", xy=(t_star, acc_star),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5, color=BLUE)

    # Margin curve (red) — plotted on same x-axis scale
    acc_marg = [
        margin_sweep[cat][f"{m:.2f}"]["accuracy_mean"]
        for m in margin_thresholds
    ]
    acc_marg_plot = [v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
                     else np.nan for v in acc_marg]
    ax.plot(margin_thresholds, acc_marg_plot,
            color=RED, linewidth=1.6, linestyle="--", marker="^", markersize=5,
            label="Margin", alpha=0.9)

    m_star    = cat_data[cat]["margin_star"]
    acc_mstar = cat_data[cat]["accuracy_at_mstar"]
    if m_star is not None and acc_mstar is not None:
        ax.scatter([m_star], [acc_mstar], color=RED, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"m*={m_star:.2f}", xy=(m_star, acc_mstar),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5, color=RED)

    ax.axhline(gate, color="#374151", linewidth=1.0, linestyle=":",
               label=f"Gate {gate:.0%}")

    ax.set_title(cat, fontsize=VIZ_DEFAULTS["tick_fontsize"] + 1)
    ax.set_xlabel("Threshold value", fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax.set_ylabel("Accuracy",        fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.5, 1.02)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if c_idx == 0:
        ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
                  loc="lower right")

fig.suptitle(
    f"Confidence vs Margin as Auto-Approve Gate (η_neg=0.05)\n"
    "Blue=confidence threshold  |  Red=margin threshold  |  dots=threshold*",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "prod4r_confidence_vs_margin_comparison", output_dir="paper_figures")
plt.close()
print("[CHART 5] saved")
