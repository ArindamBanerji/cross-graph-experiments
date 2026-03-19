"""
DISC-2 charts: Frozen vs Learned Composite Discriminant.

Reads: experiments/prod/disc2_frozen_vs_learned/disc2_results.json
Writes: paper_figures/disc2_*.{pdf,png}

Charts:
  1. disc2_frozen_vs_learned      — grouped bars: frozen vs learned composite coverage
  2. disc2_learning_lift          — heatmap of disc coverage lift (learned - frozen)
  3. disc2_conf_vs_composite      — scatter: confidence-only vs composite coverage
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

_CHARTS_DIR  = Path(__file__).resolve().parent
RESULTS_PATH = _CHARTS_DIR / "disc2_results.json"

with open(RESULTS_PATH) as fh:
    results = json.load(fh)

# Reconstruct sweep parameters
NOISE_RATES  = sorted({v["noise_rate"] for v in results.values()})
DELTA_LEVELS = sorted({v["delta"]      for v in results.values()})
CONDITIONS   = ["frozen", "learned"]

FROZEN_COLOR  = "#3B82F6"    # blue
LEARNED_COLOR = "#EF4444"    # red

def get(noise: float, delta: float, condition: str) -> dict:
    return results[f"noise{noise:.2f}_delta{delta:.2f}_{condition}"]

# ---------------------------------------------------------------------------
# Chart 1 — Grouped bars: frozen vs learned composite coverage
# ---------------------------------------------------------------------------

# One group per (noise, delta), two bars per group (frozen, learned)
n_groups  = len(NOISE_RATES) * len(DELTA_LEVELS)
group_labels = [f"n={n:.2f}\nδ={d:.2f}"
                for n in NOISE_RATES for d in DELTA_LEVELS]
x     = np.arange(n_groups)
BAR_W = 0.32

frozen_disc  = [get(n, d, "frozen")["composite_coverage_85"] * 100.0
                for n in NOISE_RATES for d in DELTA_LEVELS]
learned_disc = [get(n, d, "learned")["composite_coverage_85"] * 100.0
                for n in NOISE_RATES for d in DELTA_LEVELS]
frozen_conf  = [get(n, d, "frozen")["confidence_coverage_85"] * 100.0
                for n in NOISE_RATES for d in DELTA_LEVELS]

fig, ax = plt.subplots(figsize=(12, 5))

bars_f = ax.bar(x - BAR_W / 2, frozen_disc,  BAR_W,
                color=FROZEN_COLOR,  alpha=0.85, label="Frozen scorer (composite)",
                edgecolor="white", linewidth=0.7)
bars_l = ax.bar(x + BAR_W / 2, learned_disc, BAR_W,
                color=LEARNED_COLOR, alpha=0.85, label="Learned scorer (composite)",
                edgecolor="white", linewidth=0.7)

# Annotate bars with absolute values
for bar, val in zip(bars_f, frozen_disc):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 1.0,
            color=FROZEN_COLOR)
for bar, val in zip(bars_l, learned_disc):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 1.0,
            color=LEARNED_COLOR)

# Confidence baseline as step line for reference
ax.step(np.append(x - 0.5, x[-1] + 0.5),
        frozen_conf + [frozen_conf[-1]],
        where="post",
        color="#6B7280", linestyle=":", linewidth=1.3, alpha=0.7,
        label="Confidence-only (frozen, reference)")

ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_ylabel("Coverage at 85% Precision (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Composite Discriminant: Frozen vs Learned Scorer\n"
    "Blue = frozen composite | Red = learned composite | Dotted = confidence-only baseline",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
fig.tight_layout()
save_figure(fig, "disc2_frozen_vs_learned", output_dir="paper_figures")
plt.close()
print("[CHART 1] disc2_frozen_vs_learned saved")

# ---------------------------------------------------------------------------
# Chart 2 — Heatmap: disc coverage lift (learned - frozen) per condition
# ---------------------------------------------------------------------------

# Rows = (noise, delta), single column = disc lift
row_labels = [f"n={n:.2f}, δ={d:.2f}"
              for n in NOISE_RATES for d in DELTA_LEVELS]
n_rows = len(row_labels)

lift_matrix = np.zeros((n_rows, 1))
for r_idx, (noise, delta) in enumerate(
        [(n, d) for n in NOISE_RATES for d in DELTA_LEVELS]):
    f_disc = get(noise, delta, "frozen")["composite_coverage_85"]
    l_disc = get(noise, delta, "learned")["composite_coverage_85"]
    lift_matrix[r_idx, 0] = (l_disc - f_disc) * 100.0

fig, ax = plt.subplots(figsize=(4, 6))
vmax = max(abs(lift_matrix.max()), abs(lift_matrix.min()), 0.5)
im   = ax.imshow(lift_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.6, label="Disc coverage lift (learned − frozen, pp)")

ax.set_xticks([0])
ax.set_xticklabels(["composite\ncoverage lift"], fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])

for r_idx in range(n_rows):
    val       = lift_matrix[r_idx, 0]
    text_col  = "black" if abs(val) < vmax * 0.6 else "white"
    ax.text(0, r_idx, f"{val:+.1f}pp",
            ha="center", va="center",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color=text_col)

ax.set_title(
    "Learning Lift on Composite Coverage\n"
    "Green = improvement | Red = degradation",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "disc2_learning_lift", output_dir="paper_figures")
plt.close()
print("[CHART 2] disc2_learning_lift saved")

# ---------------------------------------------------------------------------
# Chart 3 — Scatter: confidence-only vs composite coverage
# ---------------------------------------------------------------------------

# Each condition is one point; colored by frozen (blue) vs learned (red)
fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for condition, color, marker, label in [
    ("frozen",  FROZEN_COLOR,  "o", "Frozen scorer"),
    ("learned", LEARNED_COLOR, "s", "Learned scorer"),
]:
    x_vals = []
    y_vals = []
    for noise in NOISE_RATES:
        for delta in DELTA_LEVELS:
            r = get(noise, delta, condition)
            x_vals.append(r["confidence_coverage_85"] * 100.0)
            y_vals.append(r["composite_coverage_85"]  * 100.0)

    ax.scatter(x_vals, y_vals, color=color, marker=marker,
               s=70, alpha=0.85, label=label, zorder=3)

    # Label each point with (noise, delta)
    for xi, yi, noise, delta in zip(x_vals, y_vals, NOISE_RATES * len(DELTA_LEVELS),
                                    [d for _ in NOISE_RATES for d in DELTA_LEVELS]):
        ax.annotate(f"n={noise:.2f}\nδ={delta:.2f}",
                    xy=(xi, yi), xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=4.5, color=color, alpha=0.8)

# Diagonal: composite = confidence (no benefit)
all_conf_vals = [get(n, d, c)["confidence_coverage_85"] * 100.0
                 for n in NOISE_RATES for d in DELTA_LEVELS for c in CONDITIONS]
diag_min = max(0.0, min(all_conf_vals) - 5.0)
diag_max = min(100.0, max(all_conf_vals) + 5.0)
ax.plot([diag_min, diag_max], [diag_min, diag_max],
        color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.6,
        label="Diagonal (no composite benefit)", zorder=1)

ax.set_xlabel("Confidence-only Coverage at 85% Precision (%)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Composite Discriminant Coverage at 85% Precision (%)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Confidence vs Composite Gating\n"
    "Points above diagonal = composite adds value | "
    "Blue=frozen, Red=learned",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2, linestyle=":")
fig.tight_layout()
save_figure(fig, "disc2_conf_vs_composite", output_dir="paper_figures")
plt.close()
print("[CHART 3] disc2_conf_vs_composite saved")

print()
print("[DONE] All 3 DISC-2 charts written to paper_figures/disc2_*")
