"""
EXP-OP1 chart generation.

Four publication-quality charts (300 DPI, PDF + PNG):
  1. expOP1_auac_curves       — accuracy trajectories for all four conditions
  2. expOP1_auac_delta        — AUAC delta vs baseline boxplots
  3. expOP1_t70_comparison    — time-to-70% bar chart for A, B, D
  4. expOP1_harmful_recovery  — Loop 2 recovery from harmful operator
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.viz.bridge_common import save_figure, COLORS

RESULTS_JSON = Path("experiments/synthesis/expOP1_scalar_loop2/results.json")
FIGURES_DIR  = Path("paper_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colors — fallback to explicit hex if keys absent from COLORS dict
COLOR_A = COLORS.get("baseline_fixed", "#94A3B8")   # slate grey
COLOR_B = COLORS.get("main",           "#1E3A5F")   # deep blue
COLOR_C = "#DC2626"                                  # red
COLOR_D = "#D97706"                                  # amber

WINDOW_SIZE = 50   # must match run.py WINDOW_SIZE


def generate_charts(all_results: dict, summary_stats: dict) -> None:
    """Generate and save all four EXP-OP1 charts."""

    # ------------------------------------------------------------------
    # Extract accuracy curves — shape (10, n_windows)
    # ------------------------------------------------------------------
    curves_A = np.array([r["accuracy_curve"] for r in all_results["condition_A"]])
    curves_B = np.array([r["accuracy_curve"] for r in all_results["condition_B"]])
    curves_C = np.array([r["accuracy_curve"] for r in all_results["condition_C"]])
    curves_D = np.array([r["accuracy_curve"] for r in all_results["condition_D"]])

    n_windows = curves_A.shape[1]
    x = np.arange(n_windows)

    # ------------------------------------------------------------------
    # CHART 1: Accuracy trajectories — all four conditions
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for curves, color, label, ls in [
        (curves_A, COLOR_A, "A — Baseline",          "-"),
        (curves_B, COLOR_B, "B — Correct operator",  "-"),
        (curves_C, COLOR_C, "C — Harmful operator",  "--"),
        (curves_D, COLOR_D, "D — Expiring operator", ":"),
    ]:
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    # Vertical line at approximate D-expires position
    expire_x = max(0, 150 - WINDOW_SIZE // 2)
    ax.axvline(expire_x, color=COLOR_D, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(expire_x + 1, 0.15, "D expires", fontsize=8, color=COLOR_D)

    # Reference lines
    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)

    ax.set_xlabel("Decision (post-shift window)")
    ax.set_ylabel("Rolling accuracy (window=50)")
    ax.set_title("EXP-OP1: Accuracy Trajectories — Four Conditions")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1_auac_curves")
    print("[CHART 1] expOP1_auac_curves.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: AUAC delta vs baseline — boxplots
    # ------------------------------------------------------------------
    auacs_A = np.array([r["auac"] for r in all_results["condition_A"]])
    auacs_B = np.array([r["auac"] for r in all_results["condition_B"]])
    auacs_C = np.array([r["auac"] for r in all_results["condition_C"]])
    auacs_D = np.array([r["auac"] for r in all_results["condition_D"]])

    delta_B = auacs_B - auacs_A
    delta_C = auacs_C - auacs_A
    delta_D = auacs_D - auacs_A

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [delta_B, delta_C, delta_D],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], [COLOR_B, COLOR_C, COLOR_D]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="black", linewidth=1)

    # Annotate p-value for B vs A above the first box
    p_val = summary_stats["B_vs_A"]["p_value"]
    ax.text(1, float(delta_B.max()) + 0.005, f"p={p_val:.3f}", ha="center", fontsize=9)

    ax.set_xticklabels(["B vs A\n(correct)", "C vs A\n(harmful)", "D vs A\n(expiring)"])
    ax.set_ylabel("AUAC delta (condition - baseline)")
    ax.set_title("EXP-OP1: AUAC Delta vs Baseline (10 seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1_auac_delta")
    print("[CHART 2] expOP1_auac_delta.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: T70 comparison — A, B, D
    # ------------------------------------------------------------------
    def mean_t70_std(cond_key: str) -> tuple[float, float]:
        vals = [r["t70"] for r in all_results[cond_key] if r["t70"] is not None]
        if not vals:
            return float("nan"), 0.0
        return float(np.mean(vals)), float(np.std(vals))

    t70_A, std_A = mean_t70_std("condition_A")
    t70_B, std_B = mean_t70_std("condition_B")
    t70_D, std_D = mean_t70_std("condition_D")

    labels_t   = ["A — Baseline", "B — Correct", "D — Expiring"]
    values_t   = [t70_A, t70_B, t70_D]
    stds_t     = [std_A, std_B, std_D]
    colors_t   = [COLOR_A, COLOR_B, COLOR_D]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        labels_t, values_t,
        color=colors_t, alpha=0.85,
        yerr=stds_t, capsize=5,
    )
    for bar, val in zip(bars, values_t):
        if not np.isnan(val) and val > 10:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 5,
                f"{val:.0f}",
                ha="center", va="top", fontsize=9,
                color="white", fontweight="bold",
            )

    if any(np.isnan(v) for v in values_t):
        ax.text(
            0.5, 0.95, "NaN = accuracy never reached 70%",
            transform=ax.transAxes, ha="center", fontsize=8, color="grey",
        )

    ax.set_ylabel("T70 decision index (lower = faster)")
    ax.set_title("EXP-OP1: Time to 70% Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1_t70_comparison")
    print("[CHART 3] expOP1_t70_comparison.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 4: Harmful operator — Loop 2 recovery
    # ------------------------------------------------------------------
    mean_A = curves_A.mean(axis=0)
    mean_C = curves_C.mean(axis=0)
    std_C  = curves_C.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean_A, color=COLOR_A, linestyle="-",  linewidth=2, label="A — Baseline")
    ax.plot(x, mean_C, color=COLOR_C, linestyle="--", linewidth=2, label="C — Harmful operator")
    ax.fill_between(x, mean_C - std_C, mean_C + std_C, color=COLOR_C, alpha=0.15)

    # Shade region where harmful operator is below baseline
    ax.fill_between(
        x, mean_A, mean_C,
        where=(mean_C < mean_A),
        color=COLOR_C, alpha=0.10,
        label="Loop 2 not recovered",
    )

    # Find first crossover after index 5
    crossover = None
    for i in range(6, n_windows):
        if mean_C[i] >= mean_A[i] and mean_C[i - 1] < mean_A[i - 1]:
            crossover = i
            break

    if crossover is not None:
        ax.axvline(
            crossover, color="green", linestyle="--", linewidth=1.5,
            label=f"Recovery ~decision {crossover}",
        )
        ax.set_title("EXP-OP1: Harmful Operator — Loop 2 Recovers")
    else:
        ax.set_title("EXP-OP1: Harmful Operator — Loop 2 Does Not Recover in Window")

    ax.set_xlabel("Decision (post-shift window)")
    ax.set_ylabel("Rolling accuracy (window=50)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1_harmful_recovery")
    print("[CHART 4] expOP1_harmful_recovery.png + .pdf saved")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nAll charts saved to paper_figures/:")
    print("  expOP1_auac_curves.png      + .pdf")
    print("  expOP1_auac_delta.png       + .pdf")
    print("  expOP1_t70_comparison.png   + .pdf")
    print("  expOP1_harmful_recovery.png + .pdf")


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    generate_charts(data["all_results"], data["summary_stats"])
