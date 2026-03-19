"""
PROD-4 FINAL charts: Per-Category Threshold Calibration at η_neg=0.05.

Regime: centroidal synthetic
Reads: experiments/prod/prod4_final/prod4_final_results.json
Writes: paper_figures/prod4f_*.{pdf,png}

Charts:
  1. prod4f_accuracy_vs_confidence    — confidence curves, 6 cats, gate line, threshold* dots
  2. prod4f_coverage_vs_confidence    — confidence coverage curves, threshold* dots
  3. prod4f_accuracy_vs_margin        — margin curves, 6 cats, gate line, margin* dots
  4. prod4f_confidence_vs_margin      — 6-panel: conf (blue) vs margin (red) per category
  5. prod4f_recommendation_table      — colour-coded table: green ≤0.75, orange >0.75, red=none
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
    _REPO_ROOT / "experiments" / "prod" / "prod4_final" / "prod4_final_results.json"
)
with open(RESULTS_PATH) as fh:
    results = json.load(fh)

categories    = list(results["confidence_thresholds"].keys())
conf_data     = results["confidence_thresholds"]   # per-category threshold* + ci
margin_data   = results["margin_thresholds"]       # per-category margin*
conf_sweep    = results["sweep_data"]              # {cat: {threshold: {accuracy_mean, ...}}}
margin_sweep  = results["margin_sweep_data"]       # {cat: {margin: {accuracy_mean, ...}}}
floors        = results["refer_to_analyst_floors"]
gate          = results["accuracy_gate"]
ontology      = results["ontology"]
C, A          = ontology["C"], ontology["A"]
eta_neg       = results["eta_neg"]
recommendation = results["recommendation"]

conf_thresholds   = sorted([float(k) for k in conf_sweep[categories[0]].keys()])
margin_thresholds = sorted([float(k) for k in margin_sweep[categories[0]].keys()])

# 6-colour palette
CAT_COLORS = COLORS["category_colors"] + ["#EA580C"]

GREEN  = "#D1FAE5"
ORANGE = "#FEF3C7"
RED    = "#FEE2E2"
HEADER = "#E2E8F0"

BLUE_LINE = COLORS["gt_noise_0"]
RED_LINE  = COLORS["gt_noise_30"]

# ---------------------------------------------------------------------------
# Helper: nan-safe list for plotting
# ---------------------------------------------------------------------------

def _safe(vals):
    return [v if (v is not None and not (isinstance(v, float) and np.isnan(v)))
            else np.nan for v in vals]

# ---------------------------------------------------------------------------
# Chart 1 — Accuracy vs confidence threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for c_idx, cat in enumerate(categories):
    acc_means = [conf_sweep[cat][f"{t:.2f}"]["accuracy_mean"] for t in conf_thresholds]
    ax.plot(conf_thresholds, _safe(acc_means),
            color=CAT_COLORS[c_idx], linewidth=1.6, label=cat, alpha=0.9)

    t_star   = conf_data[cat]["threshold_star"]
    acc_star = conf_data[cat]["accuracy_at_star"]
    if t_star is not None and acc_star is not None:
        ax.scatter([t_star], [acc_star], color=CAT_COLORS[c_idx],
                   s=60, zorder=5, edgecolors="white", linewidths=0.8)

ax.axhline(gate, color="#374151", linewidth=1.4, linestyle="--",
           label=f"Gate {gate:.0%}")
ax.set_xlabel("Confidence threshold", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Accuracy (mean, 50 seeds)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve accuracy vs confidence (η_neg={eta_neg}, centroidal synthetic, 50 seeds, A={A})",
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
save_figure(fig, "prod4f_accuracy_vs_confidence", output_dir="paper_figures")
plt.close()
print("[CHART 1] prod4f_accuracy_vs_confidence saved")

# ---------------------------------------------------------------------------
# Chart 2 — Coverage vs confidence threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for c_idx, cat in enumerate(categories):
    cov_means = [conf_sweep[cat][f"{t:.2f}"]["coverage_mean"] for t in conf_thresholds]
    ax.plot(conf_thresholds, cov_means,
            color=CAT_COLORS[c_idx], linewidth=1.6, label=cat, alpha=0.9)

    t_star   = conf_data[cat]["threshold_star"]
    cov_star = conf_data[cat]["coverage_at_star"]
    if t_star is not None and cov_star is not None:
        ax.scatter([t_star], [cov_star], color=CAT_COLORS[c_idx],
                   s=60, zorder=5, edgecolors="white", linewidths=0.8)

ax.axvline(0.90, color="#374151", linewidth=1.4, linestyle="--",
           label="Current global threshold (0.90)")
ax.set_xlabel("Confidence threshold", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Coverage (fraction of all decisions)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Coverage vs confidence threshold (η_neg={eta_neg}, centroidal synthetic, 50 seeds)",
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
save_figure(fig, "prod4f_coverage_vs_confidence", output_dir="paper_figures")
plt.close()
print("[CHART 2] prod4f_coverage_vs_confidence saved")

# ---------------------------------------------------------------------------
# Chart 3 — Accuracy vs margin threshold (6 curves)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

for c_idx, cat in enumerate(categories):
    acc_means = [margin_sweep[cat][f"{m:.2f}"]["accuracy_mean"] for m in margin_thresholds]
    ax.plot(margin_thresholds, _safe(acc_means),
            color=CAT_COLORS[c_idx], linewidth=1.8, marker="o", markersize=5,
            label=cat, alpha=0.9)

    m_star    = margin_data[cat]["margin_star"]
    acc_mstar = margin_data[cat]["accuracy_at_star"]
    if m_star is not None and acc_mstar is not None:
        ax.scatter([m_star], [acc_mstar], color=CAT_COLORS[c_idx],
                   s=80, zorder=5, edgecolors="white", linewidths=1.0)

ax.axhline(gate, color="#374151", linewidth=1.4, linestyle="--",
           label=f"Gate {gate:.0%}")
ax.set_xlabel("Margin threshold (top1 − top2 probability)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Accuracy (mean, 50 seeds)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Auto-approve accuracy vs margin (η_neg={eta_neg}, 50 seeds, A={A})",
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
save_figure(fig, "prod4f_accuracy_vs_margin", output_dir="paper_figures")
plt.close()
print("[CHART 3] prod4f_accuracy_vs_margin saved")

# ---------------------------------------------------------------------------
# Chart 4 — Confidence vs Margin comparison (6-panel, one per category)
# ---------------------------------------------------------------------------

ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7))
axes_flat  = axes.flatten()

for c_idx, cat in enumerate(categories):
    ax = axes_flat[c_idx]

    # Confidence curve (blue)
    acc_conf = [conf_sweep[cat][f"{t:.2f}"]["accuracy_mean"] for t in conf_thresholds]
    ax.plot(conf_thresholds, _safe(acc_conf),
            color=BLUE_LINE, linewidth=1.6, label="Confidence", alpha=0.9)

    t_star   = conf_data[cat]["threshold_star"]
    acc_star = conf_data[cat]["accuracy_at_star"]
    if t_star is not None and acc_star is not None:
        ax.scatter([t_star], [acc_star], color=BLUE_LINE, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"c*={t_star:.2f}", xy=(t_star, acc_star),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5, color=BLUE_LINE)

    # Margin curve (red) — plotted on same x-axis scale [0.25, 1.0]
    acc_marg = [margin_sweep[cat][f"{m:.2f}"]["accuracy_mean"] for m in margin_thresholds]
    ax.plot(margin_thresholds, _safe(acc_marg),
            color=RED_LINE, linewidth=1.6, linestyle="--", marker="^", markersize=5,
            label="Margin", alpha=0.9)

    m_star    = margin_data[cat]["margin_star"]
    acc_mstar = margin_data[cat]["accuracy_at_star"]
    if m_star is not None and acc_mstar is not None:
        ax.scatter([m_star], [acc_mstar], color=RED_LINE, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"m*={m_star:.2f}", xy=(m_star, acc_mstar),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5, color=RED_LINE)

    ax.axhline(gate, color="#374151", linewidth=1.0, linestyle=":",
               label=f"Gate {gate:.0%}")

    ax.set_title(cat, fontsize=VIZ_DEFAULTS["tick_fontsize"] + 1)
    ax.set_xlabel("Threshold value",  fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax.set_ylabel("Accuracy",         fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.5, 1.02)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if c_idx == 0:
        ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
                  loc="lower right")

fig.suptitle(
    f"Confidence vs Margin as Auto-Approve Gate (η_neg={eta_neg}) — PROD-4 FINAL\n"
    "Blue=confidence  |  Red dashed=margin  |  dots=threshold*",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "prod4f_confidence_vs_margin", output_dir="paper_figures")
plt.close()
print("[CHART 4] prod4f_confidence_vs_margin saved")

# ---------------------------------------------------------------------------
# Chart 5 — Recommendation table (colour-coded)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4.0))
ax.axis("off")

col_labels = ["Category", "conf threshold*", "accuracy (CI)", "conf cov",
              "margin*", "margin acc", "margin cov", "refer floor"]
row_data, row_colors = [], []

for cat in categories:
    cd      = conf_data[cat]
    md      = margin_data[cat]
    t_star  = cd["threshold_star"]
    acc     = cd["accuracy_at_star"]
    cov     = cd["coverage_at_star"]
    ci      = cd["accuracy_ci_at_star"]
    m_star  = md["margin_star"]
    acc_m   = md["accuracy_at_star"]
    cov_m   = md["coverage_at_star"]
    floor_v = floors[cat]

    if t_star is None:
        t_str   = "BELOW GATE"
        acc_str = "---"
        cov_str = "---"
        cell_c  = RED
    elif t_star > 0.75:
        t_str   = f"{t_star:.2f}"
        acc_str = (f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}]"
                   if ci else f"{acc:.1%}")
        cov_str = f"{cov:.1%}"
        cell_c  = ORANGE
    else:
        t_str   = f"{t_star:.2f}"
        acc_str = (f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}]"
                   if ci else f"{acc:.1%}")
        cov_str = f"{cov:.1%}"
        cell_c  = GREEN

    m_str  = f"{m_star:.2f}"  if m_star is not None else "---"
    am_str = f"{acc_m:.1%}"   if acc_m  is not None else "---"
    cm_str = f"{cov_m:.1%}"   if cov_m  is not None else "---"

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
    f"PROD-4 Final — Per-Category Auto-Approve Thresholds (η_neg={eta_neg})\n"
    f"Gate: accuracy ≥ {gate:.0%}  |  centroidal synthetic, 50 seeds  |  "
    f"Recommendation: {recommendation}",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
    pad=12,
)
legend_patches = [
    mpatches.Patch(color=GREEN,  label="threshold* ≤ 0.75 (auto-approve viable)"),
    mpatches.Patch(color=ORANGE, label="threshold* > 0.75 (high-bar auto-approve)"),
    mpatches.Patch(color=RED,    label="below gate (no auto-approve)"),
]
ax.legend(handles=legend_patches, fontsize=VIZ_DEFAULTS["annotation_fontsize"],
          loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.9)
fig.tight_layout()
save_figure(fig, "prod4f_recommendation_table", output_dir="paper_figures")
plt.close()
print("[CHART 5] prod4f_recommendation_table saved")

print()
print(f"[DONE] All 5 PROD-4 FINAL charts written to paper_figures/prod4f_*")
