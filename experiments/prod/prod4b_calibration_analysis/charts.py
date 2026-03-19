"""
PROD-4b charts: Per-Category Reliability Diagrams and ECE Analysis.

Regime: centroidal synthetic
Reads: experiments/prod/prod4b_calibration_analysis/prod4b_eta_neg_*.json
Writes: paper_figures/prod4b_*.{pdf,png}

Charts:
  1. prod4b_reliability_global_comparison   — side-by-side global diagrams
  2. prod4b_reliability_per_category        — 6-panel, both eta_neg overlaid
  3. prod4b_confidence_distribution         — histogram, both eta_neg overlaid
  4. prod4b_margin_vs_accuracy              — margin as accuracy predictor
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
# Load both result files
# ---------------------------------------------------------------------------

BASE = _REPO_ROOT / "experiments" / "prod" / "prod4b_calibration_analysis"

with open(BASE / "prod4b_eta_neg_0.05.json") as f:
    r05 = json.load(f)
with open(BASE / "prod4b_eta_neg_1.00.json") as f:
    r10 = json.load(f)

CATEGORIES = list(r05["per_category"].keys())
N_BINS     = 10
BIN_EDGES  = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_MIDS   = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# Colors
BLUE  = COLORS["gt_noise_0"]    # eta_neg=0.05 (product)
RED   = COLORS["gt_noise_30"]   # eta_neg=1.0  (experiments)
CAT_COLORS = COLORS["category_colors"] + ["#EA580C"]  # 6th: cloud_infrastructure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bins_to_arrays(bins: list[dict]) -> tuple[list, list, list]:
    """Return (avg_conf, avg_acc, counts) lists, NaN where count==0."""
    confs, accs, counts = [], [], []
    for b in bins:
        confs.append(b["avg_confidence"] if b["count"] > 0 else float("nan"))
        accs.append(b["avg_accuracy"]   if b["count"] > 0 else float("nan"))
        counts.append(b["count"])
    return confs, accs, counts

def _reliability_axes(ax, bins_a, bins_b, label_a, label_b,
                       ece_a, ece_b, title, annotation_fontsize):
    """Draw two reliability curves on ax with diagonal reference."""
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Perfect")

    confs_a, accs_a, cnts_a = _bins_to_arrays(bins_a)
    confs_b, accs_b, cnts_b = _bins_to_arrays(bins_b)

    # Marker size proportional to sqrt(count)
    s_a = [max(10, 4 * (c ** 0.5)) for c in cnts_a]
    s_b = [max(10, 4 * (c ** 0.5)) for c in cnts_b]

    ax.scatter(confs_a, accs_a, s=s_a, color=BLUE, alpha=0.85, zorder=3,
               label=f"{label_a} ECE={ece_a:.4f}")
    ax.scatter(confs_b, accs_b, s=s_b, color=RED, alpha=0.85, zorder=3,
               marker="^", label=f"{label_b} ECE={ece_b:.4f}")

    # Shading between diagonal and each curve (gap = miscalibration)
    valid_a = [(c, a) for c, a in zip(confs_a, accs_a)
               if not (np.isnan(c) or np.isnan(a))]
    valid_b = [(c, a) for c, a in zip(confs_b, accs_b)
               if not (np.isnan(c) or np.isnan(a))]
    if valid_a:
        xa, ya = zip(*valid_a)
        ax.fill_between(xa, xa, ya, alpha=0.08, color=BLUE)
    if valid_b:
        xb, yb = zip(*valid_b)
        ax.fill_between(xb, xb, yb, alpha=0.08, color=RED)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence", fontsize=annotation_fontsize)
    ax.set_ylabel("Fraction correct", fontsize=annotation_fontsize)
    ax.set_title(title, fontsize=VIZ_DEFAULTS["tick_fontsize"] + 0.5)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=annotation_fontsize - 0.5, loc="upper left")


# ---------------------------------------------------------------------------
# Chart 1 — Global reliability diagram (side-by-side)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, r, label, color in [
    (axes[0], r05, "η_neg=0.05 (product)", BLUE),
    (axes[1], r10, "η_neg=1.00 (experiments)", RED),
]:
    bins   = r["global_bins"]
    ece    = r["global_ece"]
    confs, accs, cnts = _bins_to_arrays(bins)
    sizes  = [max(10, 4 * (c ** 0.5)) for c in cnts]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Perfect")
    ax.scatter(confs, accs, s=sizes, color=color, alpha=0.85, zorder=3,
               label=f"ECE={ece:.4f}")
    valid = [(c, a) for c, a in zip(confs, accs)
             if not (np.isnan(c) or np.isnan(a))]
    if valid:
        xv, yv = zip(*valid)
        ax.fill_between(xv, xv, yv, alpha=0.10, color=color)

    ax.annotate(
        f"P≥0.90: {r['pct_above_90']:.1%} of decisions",
        xy=(0.02, 0.92), xycoords="axes fraction",
        fontsize=VIZ_DEFAULTS["annotation_fontsize"], color="#374151",
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Fraction correct",          fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(label, fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="upper left")

fig.suptitle(
    f"Global Reliability Diagram (tau={r05['tau']}, {r05['n_seeds']} seeds, centroidal synthetic)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "prod4b_reliability_global_comparison", output_dir="paper_figures")
plt.close()
print("[CHART 1] saved")

# ---------------------------------------------------------------------------
# Chart 2 — Per-category reliability (6-panel, both eta_neg overlaid)
# ---------------------------------------------------------------------------

n_cats = len(CATEGORIES)
ncols  = 3
nrows  = (n_cats + ncols - 1) // ncols   # 2 rows × 3 cols

fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7))
axes_flat  = axes.flatten()

ann_fs = VIZ_DEFAULTS["annotation_fontsize"]

for c_idx, cat in enumerate(CATEGORIES):
    ax      = axes_flat[c_idx]
    bins_05 = r05["per_category"][cat]["bins"]
    bins_10 = r10["per_category"][cat]["bins"]
    ece_05  = r05["per_category"][cat]["ece"]
    ece_10  = r10["per_category"][cat]["ece"]
    acc_05  = r05["per_category"][cat]["accuracy"]
    acc_10  = r10["per_category"][cat]["accuracy"]

    _reliability_axes(
        ax, bins_05, bins_10,
        label_a=f"η={0.05}", label_b=f"η={1.0}",
        ece_a=ece_05, ece_b=ece_10,
        title=f"{cat}\nacc: {acc_05:.1%} | {acc_10:.1%}",
        annotation_fontsize=ann_fs,
    )

# Hide any spare panel
for spare in axes_flat[n_cats:]:
    spare.set_visible(False)

fig.suptitle(
    f"Per-Category Reliability (tau={r05['tau']}, {r05['n_seeds']} seeds)\n"
    "Blue=η_neg=0.05 (product)  |  Red=η_neg=1.0 (experiments)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "prod4b_reliability_per_category", output_dir="paper_figures")
plt.close()
print("[CHART 2] saved")

# ---------------------------------------------------------------------------
# Chart 3 — Confidence distribution
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

hist_05 = np.array(r05["confidence_histogram"], dtype=float)
hist_10 = np.array(r10["confidence_histogram"], dtype=float)
total_05 = hist_05.sum()
total_10 = hist_10.sum()

x = BIN_MIDS
width = 0.04

ax.bar(x - width / 2, hist_05 / total_05, width=width,
       color=BLUE, alpha=0.75, label=f"η_neg=0.05  P≥0.90:{r05['pct_above_90']:.1%}")
ax.bar(x + width / 2, hist_10 / total_10, width=width,
       color=RED,  alpha=0.75, label=f"η_neg=1.00  P≥0.90:{r10['pct_above_90']:.1%}")

ax.axvline(0.90, color="#374151", linewidth=1.4, linestyle="--", label="P=0.90")

ax.set_xlabel("Confidence (max softmax probability)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Fraction of decisions",                fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Confidence Distribution (tau={r05['tau']}, centroidal synthetic, {r05['n_seeds']} seeds)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"])
fig.tight_layout()
save_figure(fig, "prod4b_confidence_distribution", output_dir="paper_figures")
plt.close()
print("[CHART 3] saved")

# ---------------------------------------------------------------------------
# Chart 4 — Margin vs accuracy (judges' hypothesis test)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

for r, color, label in [
    (r05, BLUE, "η_neg=0.05"),
    (r10, RED,  "η_neg=1.00"),
]:
    mb_data = r["margin_accuracy"]
    if not mb_data:
        continue
    xs    = [b["margin_lo"] + 0.05 for b in mb_data]   # bin midpoint approx
    accs  = [b["accuracy"]         for b in mb_data]
    cnts  = [b["count"]            for b in mb_data]
    sizes = [max(20, 5 * (c ** 0.5)) for c in cnts]

    ax.plot(xs, accs, color=color, linewidth=1.8, marker="o", markersize=5,
            label=label, alpha=0.9)
    ax.scatter(xs, accs, s=sizes, color=color, alpha=0.4, zorder=3)

ax.axhline(0.85, color="#374151", linewidth=1.2, linestyle="--",
           label="85% gate")
ax.set_xlabel("Decision margin (top1 − top2 probability)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Accuracy",  fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Margin (top1−top2) as Accuracy Predictor\n"
    "(tests judges' hypothesis: margin > confidence as gating signal)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(0.4, 1.02)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"])
fig.tight_layout()
save_figure(fig, "prod4b_margin_vs_accuracy", output_dir="paper_figures")
plt.close()
print("[CHART 4] saved")
