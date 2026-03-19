"""
SHIFT-2 charts: Rigorous Learning Impact Verification.

Reads: experiments/prod/shift2_learning_verification/shift2_results.json
Writes: paper_figures/shift2_*.{pdf,png}

Charts:
  1. shift2_accuracy_lift_heatmap    — accuracy lift vs frozen (rows: noise×delta, cols: warmup)
  2. shift2_coverage_lift_heatmap    — coverage@85% lift vs frozen (same layout)
  3. shift2_noise_sensitivity        — accuracy lift vs noise_rate at warmup=1000 (2 delta lines)
  4. shift2_zero_noise_diagnostic    — grouped bars at noise=0.00 (THE diagnostic chart)
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
RESULTS_PATH = _CHARTS_DIR / "shift2_results.json"

with open(RESULTS_PATH) as fh:
    results = json.load(fh)

# Reconstruct sweep parameters from stored keys
NOISE_RATES     = sorted({v["noise_rate"] for v in results.values()})
DELTA_LEVELS    = sorted({v["delta"]      for v in results.values()})
N_WARMUP_LEVELS = sorted({v["n_warmup"]   for v in results.values()})
ACCURACY_GATE   = 0.85

def get(noise: float, delta: float, warmup: int) -> dict:
    return results[f"noise{noise:.2f}_delta{delta:.2f}_warmup{warmup}"]

# Warmup levels excluding the frozen baseline (warmup=0)
warmup_non_zero = [w for w in N_WARMUP_LEVELS if w > 0]

# Row labels for heatmaps: (noise, delta) combinations
row_labels = [f"n={n:.2f} δ={d:.2f}"
              for n in NOISE_RATES for d in DELTA_LEVELS]
n_rows = len(row_labels)
col_labels = [str(w) for w in N_WARMUP_LEVELS]

# ---------------------------------------------------------------------------
# Chart 1 — Accuracy lift heatmap
# ---------------------------------------------------------------------------

# Build lift matrix: rows = (noise, delta), cols = warmup
lift_acc = np.zeros((n_rows, len(N_WARMUP_LEVELS)))
abs_acc  = np.zeros((n_rows, len(N_WARMUP_LEVELS)))

for r_idx, (noise, delta) in enumerate(
        [(n, d) for n in NOISE_RATES for d in DELTA_LEVELS]):
    frozen_acc = get(noise, delta, 0)["overall_accuracy"]
    for c_idx, w in enumerate(N_WARMUP_LEVELS):
        acc = get(noise, delta, w)["overall_accuracy"]
        lift_acc[r_idx, c_idx] = (acc - frozen_acc) * 100.0
        abs_acc[r_idx, c_idx]  = acc * 100.0

fig, ax = plt.subplots(figsize=(8, 5))
vmax = max(abs(lift_acc.max()), abs(lift_acc.min()), 0.1)
im   = ax.imshow(lift_acc, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy lift (pp vs frozen)")

ax.set_xticks(range(len(N_WARMUP_LEVELS)))
ax.set_yticks(range(n_rows))
ax.set_xticklabels([f"w={w}" for w in N_WARMUP_LEVELS],
                   fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_yticklabels(row_labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])

# Annotate cells: absolute accuracy (top) + lift sign (bottom)
for r_idx in range(n_rows):
    for c_idx in range(len(N_WARMUP_LEVELS)):
        lift_val = lift_acc[r_idx, c_idx]
        abs_val  = abs_acc[r_idx, c_idx]
        text_col = "black" if abs(lift_val) < vmax * 0.5 else "white"
        label    = f"{abs_val:.1f}%"
        if c_idx > 0:                   # show lift for non-zero warmup cols
            label += f"\n({lift_val:+.1f})"
        ax.text(c_idx, r_idx, label, ha="center", va="center",
                fontsize=6.5, color=text_col)

ax.set_title(
    "Learning Lift: Accuracy (warmup vs frozen)\n"
    "Green = improvement | Red = degradation | col w=0 shows frozen baseline",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "shift2_accuracy_lift_heatmap", output_dir="paper_figures")
plt.close()
print("[CHART 1] shift2_accuracy_lift_heatmap saved")

# ---------------------------------------------------------------------------
# Chart 2 — Coverage@85% lift heatmap
# ---------------------------------------------------------------------------

lift_cov = np.zeros((n_rows, len(N_WARMUP_LEVELS)))
abs_cov  = np.zeros((n_rows, len(N_WARMUP_LEVELS)))

for r_idx, (noise, delta) in enumerate(
        [(n, d) for n in NOISE_RATES for d in DELTA_LEVELS]):
    frozen_cov = get(noise, delta, 0)["coverage_85"]
    for c_idx, w in enumerate(N_WARMUP_LEVELS):
        cov = get(noise, delta, w)["coverage_85"]
        lift_cov[r_idx, c_idx] = (cov - frozen_cov) * 100.0
        abs_cov[r_idx, c_idx]  = cov * 100.0

fig, ax = plt.subplots(figsize=(8, 5))
vmax_c = max(abs(lift_cov.max()), abs(lift_cov.min()), 0.1)
im = ax.imshow(lift_cov, cmap="RdYlGn", vmin=-vmax_c, vmax=vmax_c, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Coverage@85% lift (pp vs frozen)")

ax.set_xticks(range(len(N_WARMUP_LEVELS)))
ax.set_yticks(range(n_rows))
ax.set_xticklabels([f"w={w}" for w in N_WARMUP_LEVELS],
                   fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_yticklabels(row_labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])

for r_idx in range(n_rows):
    for c_idx in range(len(N_WARMUP_LEVELS)):
        l_val = lift_cov[r_idx, c_idx]
        a_val = abs_cov[r_idx, c_idx]
        text_col = "black" if abs(l_val) < vmax_c * 0.5 else "white"
        label    = f"{a_val:.1f}%"
        if c_idx > 0:
            label += f"\n({l_val:+.1f})"
        ax.text(c_idx, r_idx, label, ha="center", va="center",
                fontsize=6.5, color=text_col)

ax.set_title(
    "Learning Lift: Coverage at 85% Precision (warmup vs frozen)\n"
    "Green = improvement | Red = degradation | col w=0 shows frozen baseline",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "shift2_coverage_lift_heatmap", output_dir="paper_figures")
plt.close()
print("[CHART 2] shift2_coverage_lift_heatmap saved")

# ---------------------------------------------------------------------------
# Chart 3 — Noise sensitivity: accuracy lift vs noise_rate at warmup=1000
# ---------------------------------------------------------------------------

TARGET_WARMUP = 1000
if TARGET_WARMUP not in N_WARMUP_LEVELS:
    TARGET_WARMUP = max(N_WARMUP_LEVELS)

DELTA_COLORS = ["#3B82F6", "#EF4444"]    # blue for δ=0, red for δ>0
DELTA_STYLES = ["-o", "--s"]

fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_wide"])

for i, delta in enumerate(DELTA_LEVELS):
    lifts = []
    for noise in NOISE_RATES:
        frozen_acc  = get(noise, delta, 0)["overall_accuracy"]
        learned_acc = get(noise, delta, TARGET_WARMUP)["overall_accuracy"]
        lifts.append((learned_acc - frozen_acc) * 100.0)
    ax.plot(NOISE_RATES, lifts, DELTA_STYLES[i],
            color=DELTA_COLORS[i], linewidth=2.0, markersize=7,
            label=f"δ={delta:.2f}")
    for x, y in zip(NOISE_RATES, lifts):
        ax.annotate(f"{y:+.2f}pp", xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
                    color=DELTA_COLORS[i])

ax.axhline(y=0.0, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.7,
           label="no lift (break-even)")
ax.set_xlabel("Noise rate", fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_ylabel(f"Accuracy lift over frozen (pp, warmup={TARGET_WARMUP})",
              fontsize=VIZ_DEFAULTS["label_fontsize"])
ax.set_title(
    f"Learning Lift vs Noise Rate (warmup={TARGET_WARMUP}, frozen eval)\n"
    "Key: does lift go negative as noise increases?",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
ax.set_xticks(NOISE_RATES)
ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25, linestyle=":")
fig.tight_layout()
save_figure(fig, "shift2_noise_sensitivity", output_dir="paper_figures")
plt.close()
print("[CHART 3] shift2_noise_sensitivity saved")

# ---------------------------------------------------------------------------
# Chart 4 — Zero-noise diagnostic: grouped bars at noise=0.00
# ---------------------------------------------------------------------------

ZERO_NOISE = 0.0
BAR_W = 0.18
x = np.arange(len(N_WARMUP_LEVELS))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for col, (metric_key, metric_label, ylim_pad) in enumerate([
        ("overall_accuracy",  "Accuracy (%)",            2.0),
        ("coverage_85",       "Coverage at 85% prec (%)", 0.5),
]):
    ax = axes[col]

    for i, delta in enumerate(DELTA_LEVELS):
        vals    = [get(ZERO_NOISE, delta, w)[metric_key] * 100.0
                   for w in N_WARMUP_LEVELS]
        offsets = (np.arange(len(DELTA_LEVELS)) - (len(DELTA_LEVELS) - 1) / 2.0) * BAR_W
        bars    = ax.bar(x + offsets[i], vals, BAR_W,
                         label=f"δ={delta:.2f}",
                         color=DELTA_COLORS[i], alpha=0.85,
                         edgecolor="white", linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + ylim_pad * 0.15,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"] - 0.5,
                    color=DELTA_COLORS[i])

    all_vals = [get(ZERO_NOISE, d, w)[metric_key] * 100.0
                for d in DELTA_LEVELS for w in N_WARMUP_LEVELS]
    y_min = max(0.0, min(all_vals) - ylim_pad * 3)
    y_max = min(100.0, max(all_vals) + ylim_pad * 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([f"w={w}" for w in N_WARMUP_LEVELS],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_xlabel("Warmup decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel(metric_label, fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.legend(fontsize=VIZ_DEFAULTS["annotation_fontsize"], framealpha=0.85)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle=":", axis="y")

fig.suptitle(
    "Zero-Noise Diagnostic: Does Learning Help With Perfect Labels? (noise=0.00)\n"
    "If δ=0.10 bars improve with warmup → learning works, noise was the bottleneck",
    fontsize=VIZ_DEFAULTS["title_fontsize"],
)
fig.tight_layout()
save_figure(fig, "shift2_zero_noise_diagnostic", output_dir="paper_figures")
plt.close()
print("[CHART 4] shift2_zero_noise_diagnostic saved")

print()
print("[DONE] All 4 SHIFT-2 charts written to paper_figures/shift2_*")
