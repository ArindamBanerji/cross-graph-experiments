"""
SHIFT-1 charts: Learning Capacity Under Prior Mismatch.

Reads: experiments/prod/shift1_prior_mismatch/shift1_results.json
Writes: paper_figures/shift1_*.{pdf,png}

Charts:
  1. shift1_learning_lift       — learning lift vs δ (3 warmup lines)
  2. shift1_coverage_by_delta   — auto-approve coverage vs δ (3 warmup lines)
  3. shift1_centroid_drift      — centroid drift vs δ (3 warmup lines)
  4. shift1_frozen_vs_learned   — grouped bars frozen/learned at warmup=1000
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

RESULTS_PATH = Path(__file__).parent / "shift1_results.json"
with open(RESULTS_PATH) as fh:
    results = json.load(fh)

# Reconstruct sweep parameters from the stored keys
DELTA_MAGNITUDES = sorted({v["delta"] for v in results.values()})
WARMUP_LEVELS    = sorted({v["warmup"] for v in results.values()})

def get(delta: float, warmup: int) -> dict:
    return results[f"d{delta:.2f}_w{warmup}"]

# Colour palette: one colour per warmup level
WARMUP_COLORS = ["#3B82F6", "#F59E0B", "#10B981"]   # blue, amber, green
WARMUP_LABELS = [f"warmup={w}" for w in WARMUP_LEVELS]

BLUE_BARS  = "#3B82F6"
RED_BARS   = "#EF4444"

# ---------------------------------------------------------------------------
# Chart 1 — Learning lift vs δ (3 warmup lines)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for i, w in enumerate(WARMUP_LEVELS):
    lifts = [get(d, w)["learning_lift"] * 100.0 for d in DELTA_MAGNITUDES]
    ax.plot(DELTA_MAGNITUDES, lifts, "o-",
            color=WARMUP_COLORS[i], linewidth=2.0, markersize=6,
            label=WARMUP_LABELS[i])

ax.axhline(y=0.0, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.7,
           label="no lift (zero line)")
ax.set_xlabel("Prior mismatch δ", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Learning lift (learned − frozen accuracy, pp)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Learning Lift vs Prior Mismatch (η_neg=0.05, τ=0.1, 30 seeds)\n"
    "If lift > 0 at δ > 0 but ≈ 0 at δ=0 → circularity confirmed",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.set_xticks(DELTA_MAGNITUDES)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":")
fig.tight_layout()
save_figure(fig, "shift1_learning_lift", output_dir="paper_figures")
plt.close()
print("[CHART 1] shift1_learning_lift saved")

# ---------------------------------------------------------------------------
# Chart 2 — Auto-approve coverage vs δ (learned condition, 3 warmup lines)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for i, w in enumerate(WARMUP_LEVELS):
    covs = []
    for d in DELTA_MAGNITUDES:
        c = get(d, w)["learning"]["coverage_at_star"]
        covs.append(c * 100.0 if c is not None else 0.0)
    ax.plot(DELTA_MAGNITUDES, covs, "o-",
            color=WARMUP_COLORS[i], linewidth=2.0, markersize=6,
            label=WARMUP_LABELS[i])

ax.axhline(y=20.0, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.7,
           label="20% coverage target")
ax.set_xlabel("Prior mismatch δ", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Coverage at ≥85% accuracy (%, learned)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Auto-Approve Coverage vs Prior Mismatch (learned condition)\n"
    "If coverage rises with δ → warmup plateau was a test artifact",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.set_xticks(DELTA_MAGNITUDES)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":")
fig.tight_layout()
save_figure(fig, "shift1_coverage_by_delta", output_dir="paper_figures")
plt.close()
print("[CHART 2] shift1_coverage_by_delta saved")

# ---------------------------------------------------------------------------
# Chart 3 — Centroid drift vs δ (3 warmup lines)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for i, w in enumerate(WARMUP_LEVELS):
    drifts = [get(d, w)["mean_centroid_drift"] for d in DELTA_MAGNITUDES]
    ax.plot(DELTA_MAGNITUDES, drifts, "o-",
            color=WARMUP_COLORS[i], linewidth=2.0, markersize=6,
            label=WARMUP_LABELS[i])

# Overlay the expected shift (mean L2 distance = δ for unconstrained cells)
expected = [get(d, WARMUP_LEVELS[0])["mean_shift_from_mu0"] for d in DELTA_MAGNITUDES]
ax.plot(DELTA_MAGNITUDES, expected, "s--",
        color="#6B7280", linewidth=1.4, markersize=5, alpha=0.7,
        label="target shift (‖μ_true − μ₀‖ mean)")

ax.set_xlabel("Prior mismatch δ", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Mean centroid drift ‖μ_learned − μ₀‖ (per cell, L2)",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    "Centroid Drift vs Prior Mismatch (learning condition)\n"
    "Drift ≈ target shift → centroids tracking μ_true",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.set_xticks(DELTA_MAGNITUDES)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":")
fig.tight_layout()
save_figure(fig, "shift1_centroid_drift", output_dir="paper_figures")
plt.close()
print("[CHART 3] shift1_centroid_drift saved")

# ---------------------------------------------------------------------------
# Chart 4 — Frozen vs Learned accuracy, grouped bars at warmup=1000
# ---------------------------------------------------------------------------

TARGET_WARMUP = 1000
if TARGET_WARMUP not in WARMUP_LEVELS:
    TARGET_WARMUP = max(WARMUP_LEVELS)

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

n_deltas  = len(DELTA_MAGNITUDES)
bar_width = 0.35
x         = np.arange(n_deltas)

frozen_accs  = [get(d, TARGET_WARMUP)["frozen"]["overall_accuracy"] * 100.0
                for d in DELTA_MAGNITUDES]
learned_accs = [get(d, TARGET_WARMUP)["learning"]["overall_accuracy"] * 100.0
                for d in DELTA_MAGNITUDES]

bars_f = ax.bar(x - bar_width / 2, frozen_accs,  bar_width,
                label=f"Frozen (μ₀, no updates)",
                color=RED_BARS, alpha=0.85, edgecolor="white", linewidth=0.8)
bars_l = ax.bar(x + bar_width / 2, learned_accs, bar_width,
                label=f"Learned (warmup={TARGET_WARMUP})",
                color=BLUE_BARS, alpha=0.85, edgecolor="white", linewidth=0.8)

# Annotate lift on top of learned bars
for bar_f, bar_l, f_acc, l_acc in zip(bars_f, bars_l, frozen_accs, learned_accs):
    lift = l_acc - f_acc
    if abs(lift) >= 0.05:
        ax.text(
            bar_l.get_x() + bar_l.get_width() / 2,
            bar_l.get_height() + 0.3,
            f"{lift:+.1f}pp",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
            color="#1E40AF",
        )

ax.axhline(y=ACCURACY_GATE * 100.0, color="#374151", linestyle="--",
           linewidth=1.2, alpha=0.7, label=f"Gate {ACCURACY_GATE:.0%}")
ax.set_xlabel("Prior mismatch δ", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel("Overall accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Frozen vs Learned Accuracy at Warmup={TARGET_WARMUP} (η_neg=0.05)\n"
    "Bars labelled with learning lift in percentage points",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xticks(x)
ax.set_xticklabels([f"δ={d:.2f}" for d in DELTA_MAGNITUDES])
y_min = min(min(frozen_accs), min(learned_accs)) - 2.0
y_max = min(100.0, max(max(frozen_accs), max(learned_accs)) + 5.0)
ax.set_ylim(y_min, y_max)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

ACCURACY_GATE = 0.85   # needed for annotation above (re-declared for standalone use)
save_figure(fig, "shift1_frozen_vs_learned", output_dir="paper_figures")
plt.close()
print("[CHART 4] shift1_frozen_vs_learned saved")

print()
print("[DONE] All 4 SHIFT-1 charts written to paper_figures/shift1_*")
