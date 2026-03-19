"""
DISC-1 charts: Composite Discriminant + IKS Reverse Engineering.

Reads: experiments/prod/disc1_composite_discriminant/disc1_results.json
Writes: paper_figures/disc1_*.{pdf,png}

Charts:
  1. disc1_model_comparison         — coverage at 85% prec: 5 models (A-E)
  2. disc1_feature_importance       — Model E coefficients, horizontal bars
  3. disc1_coverage_trajectory      — coverage@85% vs cumulative decisions
  4. disc1_iks_v2_trajectory        — IKS v2 score + components vs decisions
  5. disc1_feature_correlation_heatmap — 13×13 pairwise correlation matrix
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

# When exec'd from run.py, __file__ == run.py, so _SCRIPT_DIR points to the
# experiment directory — same location as disc1_results.json.
_CHARTS_DIR  = Path(__file__).resolve().parent
RESULTS_PATH = _CHARTS_DIR / "disc1_results.json"

with open(RESULTS_PATH) as fh:
    results = json.load(fh)

model_results   = results["model_results"]
feat_names      = results["feature_names"]
window_coverage = results["window_coverage"]
feat_corr_y     = results["feature_correlations_with_correct"]
corr_matrix     = np.array(results["full_correlation_matrix"])
per_cat         = results["per_category_model_e"]

model_names   = list(model_results.keys())
windows       = sorted([int(k) for k in window_coverage.keys()])
baseline_cov  = model_results["A_confidence_only"]["coverage_at_85_precision"]

GREEN  = "#D1FAE5"
GRAY   = "#E5E7EB"
BLUE   = "#3B82F6"
RED    = "#EF4444"
HEADER = "#E2E8F0"

# ---------------------------------------------------------------------------
# Chart 1 — Model comparison: coverage at 85% precision (A–E)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

covs_85 = [model_results[m]["coverage_at_85_precision"] * 100.0 for m in model_names]
aucs    = [model_results[m]["auc"] for m in model_names]
colors  = [GREEN if c > baseline_cov * 100.0 else GRAY for c in covs_85]

# Short display names
short_names = [m.split("_", 1)[1].replace("_", " ").title() for m in model_names]
x = np.arange(len(model_names))
bars = ax.bar(x, covs_85, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

ax.axhline(y=baseline_cov * 100.0, color="#374151", linestyle="--", linewidth=1.4,
           alpha=0.8, label=f"Baseline (conf only): {baseline_cov:.1%}")

for bar, cov, auc in zip(bars, covs_85, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{cov:.1f}%\nAUC={auc:.3f}",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5)

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=15, ha="right",
                   fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_ylabel("Coverage at 85% Precision (%)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Composite Discriminant: Coverage at 85% Precision\n"
    "(Frozen scorer, centroidal synthetic, 50 seeds, 5-fold CV)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"])
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_figure(fig, "disc1_model_comparison", output_dir="paper_figures")
plt.close()
print("[CHART 1] disc1_model_comparison saved")

# ---------------------------------------------------------------------------
# Chart 2 — Feature importance: Model E coefficients
# ---------------------------------------------------------------------------

coefs_e = model_results["E_all_features"]["coefficients"]
feats   = list(coefs_e.keys())
vals    = list(coefs_e.values())

# Sort by absolute value descending
order = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
feats_sorted = [feats[i] for i in order]
vals_sorted  = [vals[i]  for i in order]
bar_colors   = [BLUE if v >= 0 else RED for v in vals_sorted]

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(feats_sorted))
ax.barh(y_pos, vals_sorted, color=bar_colors, edgecolor="white", linewidth=0.6)
ax.axvline(x=0, color="#374151", linewidth=1.0, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(feats_sorted, fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_xlabel("Logistic Regression Coefficient (Model E, L2 regularized)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Feature Importance in Composite Discriminant (Model E)\n"
    "Blue = increases P(correct) | Red = decreases P(correct)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
legend_patches = [
    mpatches.Patch(color=BLUE, label="positive coefficient"),
    mpatches.Patch(color=RED,  label="negative coefficient"),
]
ax.legend(handles=legend_patches, fontsize=VIZ_DEFAULTS["annotation_fontsize"],
          loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_figure(fig, "disc1_feature_importance", output_dir="paper_figures")
plt.close()
print("[CHART 2] disc1_feature_importance saved")

# ---------------------------------------------------------------------------
# Chart 3 — Coverage trajectory: coverage@85% vs cumulative decisions
# ---------------------------------------------------------------------------

covs_traj = [window_coverage[str(w)]["mean_coverage_85"] * 100.0 for w in windows]
accs_traj = [window_coverage[str(w)]["mean_accuracy"]    * 100.0 for w in windows]

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

ax.plot(windows, covs_traj, "o-", color=BLUE, linewidth=2.2, markersize=7,
        label="Coverage at 85% precision (Model E)")
ax.plot(windows, accs_traj, "s--", color="#6B7280", linewidth=1.6, markersize=5,
        alpha=0.7, label="Overall accuracy (baseline)")

for w, c in zip(windows, covs_traj):
    ax.annotate(f"{c:.1f}%", xy=(w, c), xytext=(0, 8),
                textcoords="offset points", ha="center",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5, color=BLUE)

ax.axhline(y=85.0, color="#374151", linestyle=":", linewidth=1.2, alpha=0.6,
           label="85% precision gate")
ax.set_xscale("log")
ax.set_xlabel("Cumulative decisions (log scale)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Coverage / Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Coverage Growth with Decision Accumulation (Frozen Scorer)\n"
    "IKS story: trust coverage compounds as graph enriches",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xticks(windows)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2, linestyle=":")
fig.tight_layout()
save_figure(fig, "disc1_coverage_trajectory", output_dir="paper_figures")
plt.close()
print("[CHART 3] disc1_coverage_trajectory saved")

# ---------------------------------------------------------------------------
# Chart 4 — IKS v2 trajectory with component breakdown
# ---------------------------------------------------------------------------

comp_keys  = ["graph_richness", "decision_maturity", "trust_coverage", "factor_quality"]
comp_label = ["Graph Richness", "Decision Maturity", "Trust Coverage", "Factor Quality"]
comp_colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"]

iks_vals       = [window_coverage[str(w)]["iks_v2"] for w in windows]
comp_vals_list = [[window_coverage[str(w)]["iks_components"][k]
                   for w in windows] for k in comp_keys]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: IKS v2 total
ax = axes[0]
ax.plot(windows, iks_vals, "o-", color="#1E3A5F", linewidth=2.5, markersize=8,
        label="IKS v2 (total)")
for w, v in zip(windows, iks_vals):
    ax.annotate(f"{v:.1f}", xy=(w, v), xytext=(0, 8),
                textcoords="offset points", ha="center",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"])
ax.set_xscale("log")
ax.set_xlabel("Cumulative decisions (log scale)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("IKS v2 score (0–100)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title("IKS v2 Trajectory", fontsize=VIZ_DEFAULTS["title_fontsize"])
ax.set_xticks(windows)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_ylim(0, 105)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2, linestyle=":")

# Right: stacked area — component contributions
ax = axes[1]
bottom = np.zeros(len(windows))
for vals, label, color in zip(comp_vals_list, comp_label, comp_colors):
    contribution = np.array(vals) / 4.0   # each component contributes 1/4 to IKS
    ax.fill_between(windows, bottom, bottom + contribution,
                    alpha=0.75, color=color, label=label)
    bottom += contribution

ax.set_xscale("log")
ax.set_xlabel("Cumulative decisions (log scale)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Component contribution to IKS v2 (points)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title("IKS v2 Component Breakdown", fontsize=VIZ_DEFAULTS["title_fontsize"])
ax.set_xticks(windows)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_ylim(0, 105)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], loc="upper left",
          framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2, linestyle=":")

fig.suptitle(
    "IKS v2 — Institutional Knowledge Accumulation\n"
    "(Frozen scorer, centroidal synthetic, 50 seeds)",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "disc1_iks_v2_trajectory", output_dir="paper_figures")
plt.close()
print("[CHART 4] disc1_iks_v2_trajectory saved")

# ---------------------------------------------------------------------------
# Chart 5 — Feature correlation heatmap (13×13)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 8))

# Nicer short labels
short_labels = [
    "conf", "margin", "entropy", "top3", "p_std",
    "d_ratio", "d_gap",
    "f_ext", "f_norm", "f_cen",
    "cat_cnt", "roll_acc", "pos",
]

im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

ax.set_xticks(range(len(feat_names)))
ax.set_yticks(range(len(feat_names)))
ax.set_xticklabels(short_labels, rotation=45, ha="right",
                   fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5)
ax.set_yticklabels(short_labels, fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5)

# Annotate each cell
for i in range(len(feat_names)):
    for j in range(len(feat_names)):
        val = corr_matrix[i, j]
        text_color = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=5.5, color=text_color)

ax.set_title(
    "Feature Independence Matrix (13 discriminant features)\n"
    "Red = correlated | Blue = anti-correlated | near-white = orthogonal",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "disc1_feature_correlation_heatmap", output_dir="paper_figures")
plt.close()
print("[CHART 5] disc1_feature_correlation_heatmap saved")

print()
print("[DONE] All 5 DISC-1 charts written to paper_figures/disc1_*")
