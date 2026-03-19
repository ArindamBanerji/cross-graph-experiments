"""
PROD-3 charts: Shadow Mode Agreement Rate Baseline.

Regime: centroidal synthetic
Reads: experiments/prod/prod3_shadow_baseline/prod3_calibration_table.json
Writes: paper_figures/prod3_*.{pdf,png}

Charts:
  1. prod3_agreement_rate_distribution  — histogram of 50 per-seed overall rates
  2. prod3_per_category_agreement       — horizontal bar chart, 6 categories
  3. prod3_high_confidence_agreement    — dual-axis: agreement + coverage vs threshold
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.viz.bridge_common import save_figure, COLORS, VIZ_DEFAULTS

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod3_shadow_baseline" / "prod3_calibration_table.json"

with open(RESULTS_PATH) as fh:
    results = json.load(fh)

overall     = results["overall"]
per_cat     = results["per_category"]
high_conf   = results["high_conf"]
theta       = results["theta_recommendations"]
per_seed    = results["per_seed_overall"]
ontology    = results["ontology"]
C, A        = ontology["C"], ontology["A"]
categories  = list(per_cat.keys())

# 6-color palette: 5 existing + amber-orange for cloud_infrastructure
CAT_COLORS = COLORS["category_colors"] + ["#EA580C"]

# ---------------------------------------------------------------------------
# Chart 1 — Agreement rate distribution (histogram of 50 seeds)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

ax.hist(per_seed, bins=10, color=COLORS["gt_noise_0"], edgecolor="white", alpha=0.85)
ax.axvline(overall["mean"], color="#DC2626", linewidth=1.8, linestyle="--", label="_nolegend_")
ax.annotate(
    f"mean={overall['mean']:.1%}, 95% CI [{overall['ci_low']:.1%}, {overall['ci_high']:.1%}]",
    xy=(overall["mean"], ax.get_ylim()[1] * 0.5),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
    color="#DC2626",
)
ax.set_xlabel(f"Agreement rate (centroidal synthetic, C={C} A={A})",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Seeds (out of 50)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title("PROD-3: Shadow mode agreement rate distribution",
             fontsize=VIZ_DEFAULTS["title_fontsize"])
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

save_figure(fig, "prod3_agreement_rate_distribution", output_dir="paper_figures")
plt.close()
print("[CHART 1] saved")

# ---------------------------------------------------------------------------
# Chart 2 — Per-category agreement (horizontal bar, sorted high→low)
# ---------------------------------------------------------------------------

sorted_cats = sorted(categories, key=lambda c: per_cat[c]["mean"], reverse=True)
means  = [per_cat[c]["mean"]   for c in sorted_cats]
ci_low = [per_cat[c]["mean"] - per_cat[c]["ci_low"]  for c in sorted_cats]
ci_hi  = [per_cat[c]["ci_high"] - per_cat[c]["mean"] for c in sorted_cats]
colors = [CAT_COLORS[categories.index(c)] for c in sorted_cats]

fig, ax = plt.subplots(figsize=(9, 5))

y_pos = np.arange(len(sorted_cats))
bars  = ax.barh(y_pos, means, xerr=[ci_low, ci_hi],
                color=colors, edgecolor="white", alpha=0.85,
                error_kw={"ecolor": "#374151", "capsize": 4, "linewidth": 1.4})
ax.axvline(overall["mean"], color="#374151", linewidth=1.4, linestyle="--",
           label=f"Overall mean {overall['mean']:.1%}")
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_cats, fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_xlabel("Agreement rate", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Per-category shadow agreement (centroidal synthetic, 50 seeds, A={A})",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.tick_params(axis="x", labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"])
ax.set_xlim(0, 1.0)
fig.tight_layout()

save_figure(fig, "prod3_per_category_agreement", output_dir="paper_figures")
plt.close()
print("[CHART 2] saved")

# ---------------------------------------------------------------------------
# Chart 3 — High-confidence agreement + coverage (dual-axis)
# ---------------------------------------------------------------------------

thresholds  = sorted([float(k) for k in high_conf.keys()])
agr_means   = [high_conf[str(t)]["agreement_mean"]  for t in thresholds]
agr_stds    = [high_conf[str(t)]["agreement_std"]   for t in thresholds]
cov_means   = [high_conf[str(t)]["coverage_mean"]   for t in thresholds]
cov_stds    = [high_conf[str(t)]["coverage_std"]    for t in thresholds]

agr_lo = [m - s for m, s in zip(agr_means, agr_stds)]
agr_hi = [m + s for m, s in zip(agr_means, agr_stds)]
cov_lo = [max(0.0, m - s) for m, s in zip(cov_means, cov_stds)]
cov_hi = [min(1.0, m + s) for m, s in zip(cov_means, cov_stds)]

fig, ax1 = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])
ax2 = ax1.twinx()

blue   = COLORS["gt_noise_0"]
green  = COLORS["mi_static"]

ax1.plot(thresholds, agr_means, color=blue, linewidth=2.0, marker="o", markersize=5, label="Agreement")
ax1.fill_between(thresholds, agr_lo, agr_hi, color=blue, alpha=0.15)

ax2.plot(thresholds, cov_means, color=green, linewidth=2.0, linestyle="--",
         marker="s", markersize=5, label="Coverage")
ax2.fill_between(thresholds, cov_lo, cov_hi, color=green, alpha=0.12)

# Annotate at P>=0.9
t90_idx  = thresholds.index(0.9)
ax1.annotate(
    f"agr={agr_means[t90_idx]:.1%}",
    xy=(0.9, agr_means[t90_idx]),
    xytext=(-38, 8),
    textcoords="offset points",
    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
    color=blue,
)
ax2.annotate(
    f"cov={cov_means[t90_idx]:.1%}",
    xy=(0.9, cov_means[t90_idx]),
    xytext=(-38, -14),
    textcoords="offset points",
    fontsize=VIZ_DEFAULTS["annotation_fontsize"],
    color=green,
)

ax1.set_xlabel("Confidence threshold", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax1.set_ylabel("Agreement rate", fontsize=VIZ_DEFAULTS["label_fontsize"], color=blue)
ax2.set_ylabel("Coverage (fraction of decisions)", fontsize=VIZ_DEFAULTS["label_fontsize"], color=green)
ax1.set_title(
    f"High-confidence agreement vs coverage\n(centroidal synthetic, C={C} A={A}, 50 seeds)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax1.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax2.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax1.set_ylim(0, 1.05)
ax2.set_ylim(0, 1.05)
ax1.spines["top"].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="lower left")

fig.tight_layout()
save_figure(fig, "prod3_high_confidence_agreement", output_dir="paper_figures")
plt.close()
print("[CHART 3] saved")
