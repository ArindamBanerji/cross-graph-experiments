"""
EXP-OP1-REVISED chart generation.

Four publication-quality charts (300 DPI, PDF + PNG):
  1. expOP1r_auac_curves          — trajectories for all six conditions
  2. expOP1r_auac_delta           — AUAC delta boxplots (5 comparisons)
  3. expOP1r_harmful_vs_correct   — OP1 near-ceiling vs OP1R with-headroom
  4. expOP1r_cold_start_benefit   — operator benefit: cold vs warm start
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

RESULTS_JSON = Path("experiments/synthesis/expOP1_revised/results.json")
FIGURES_DIR  = Path("paper_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_A = COLORS.get("baseline_fixed", "#94A3B8")   # slate grey — partial warmup baseline
COLOR_B = COLORS.get("main",           "#1E3A5F")   # deep blue — correct operator
COLOR_C = "#DC2626"                                  # red — harmful operator
COLOR_D = "#D97706"                                  # amber — expiring operator
COLOR_E = "#059669"                                  # green — cold start + correct op
COLOR_F = "#6B7280"                                  # dark grey — cold start baseline

WINDOW_SIZE = 50   # must match run.py


def generate_charts(all_results: dict, summary_stats: dict) -> None:
    """Generate and save all four EXP-OP1-REVISED charts."""

    # ------------------------------------------------------------------
    # Extract accuracy curves — shape (10, n_windows)
    # ------------------------------------------------------------------
    curves_A = np.array([r["accuracy_curve"] for r in all_results["condition_A"]])
    curves_B = np.array([r["accuracy_curve"] for r in all_results["condition_B"]])
    curves_C = np.array([r["accuracy_curve"] for r in all_results["condition_C"]])
    curves_D = np.array([r["accuracy_curve"] for r in all_results["condition_D"]])
    curves_E = np.array([r["accuracy_curve"] for r in all_results["condition_E"]])
    curves_F = np.array([r["accuracy_curve"] for r in all_results["condition_F"]])

    n_windows = curves_A.shape[1]
    x = np.arange(n_windows)

    # ------------------------------------------------------------------
    # CHART 1: Accuracy trajectories — all six conditions
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    for curves, color, label, ls in [
        (curves_A, COLOR_A, "A — Baseline (partial warmup)", "-"),
        (curves_B, COLOR_B, "B — Correct operator",          "-"),
        (curves_C, COLOR_C, "C — Harmful operator",          "--"),
        (curves_D, COLOR_D, "D — Expiring operator",         ":"),
        (curves_E, COLOR_E, "E — Cold start + correct op",   "-."),
        (curves_F, COLOR_F, "F — Cold start baseline",       "--"),
    ]:
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    # Vertical line at approximate D-expires position (decision 200)
    expire_x = max(0, 200 - WINDOW_SIZE // 2)
    ax.axvline(expire_x, color=COLOR_D, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(expire_x + 1, 0.25, "D expires", color=COLOR_D, fontsize=8)

    # Reference lines
    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)

    ax.set_xlabel("Decision (post-shift window)")
    ax.set_ylabel("Rolling accuracy (window=50)")
    ax.set_title("EXP-OP1-REVISED: Accuracy Trajectories — Six Conditions")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1r_auac_curves", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOP1r_auac_curves.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: AUAC delta boxplots — five comparisons
    # ------------------------------------------------------------------
    auacs_A = np.array([r["auac"] for r in all_results["condition_A"]])
    auacs_B = np.array([r["auac"] for r in all_results["condition_B"]])
    auacs_C = np.array([r["auac"] for r in all_results["condition_C"]])
    auacs_D = np.array([r["auac"] for r in all_results["condition_D"]])
    auacs_E = np.array([r["auac"] for r in all_results["condition_E"]])
    auacs_F = np.array([r["auac"] for r in all_results["condition_F"]])

    delta_BA = auacs_B - auacs_A
    delta_CA = auacs_C - auacs_A
    delta_DA = auacs_D - auacs_A
    delta_EF = auacs_E - auacs_F
    delta_BF = auacs_B - auacs_F

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        [delta_BA, delta_CA, delta_DA, delta_EF, delta_BF],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    box_colors = [COLOR_B, COLOR_C, COLOR_D, COLOR_E, COLOR_B]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="black", linewidth=1)

    # Annotate p-values for B-A and E-F
    p_BA = summary_stats["B_vs_A"]["p_value"]
    p_EF = summary_stats["E_vs_F"]["p_value"]
    ax.text(1, float(delta_BA.max()) + 0.005, f"p={p_BA:.3f}", ha="center", fontsize=8)
    ax.text(4, float(delta_EF.max()) + 0.005, f"p={p_EF:.3f}", ha="center", fontsize=8)

    ax.set_xticklabels([
        "B-A\n(warm+op\nvs warm)",
        "C-A\n(harmful\nvs warm)",
        "D-A\n(expiring\nvs warm)",
        "E-F\n(cold+op\nvs cold)",
        "B-F\n(warm+op\nvs cold)",
    ], fontsize=8)
    ax.set_ylabel("AUAC delta")
    ax.set_title("EXP-OP1-REVISED: AUAC Deltas (10 seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1r_auac_delta", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOP1r_auac_delta.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: Near-ceiling (OP1) vs with-headroom (OP1-REVISED)
    # ------------------------------------------------------------------
    op1_path = Path("experiments/synthesis/expOP1_scalar_loop2/results.json")
    op1_exists = op1_path.exists()

    if op1_exists:
        with open(op1_path) as fh:
            op1_data = json.load(fh)["all_results"]
        op1_curves_A = np.array([r["accuracy_curve"] for r in op1_data["condition_A"]])
        op1_curves_C = np.array([r["accuracy_curve"] for r in op1_data["condition_C"]])
        op1_auac_A   = float(np.mean([r["auac"] for r in op1_data["condition_A"]]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(7, 5))
        ax1 = None

    # Panel 1 — OP1 original (near-ceiling)
    if ax1 is not None:
        x1 = np.arange(op1_curves_A.shape[1])
        ax1.plot(x1, op1_curves_A.mean(axis=0), color=COLOR_A, linewidth=2,
                 label=f"A — Baseline (AUAC={op1_auac_A:.3f})")
        ax1.plot(x1, op1_curves_C.mean(axis=0), color=COLOR_C, linestyle="--",
                 linewidth=2, label="C — Harmful operator")
        ax1.fill_between(
            x1,
            op1_curves_A.mean(axis=0),
            op1_curves_C.mean(axis=0),
            where=(op1_curves_C.mean(axis=0) < op1_curves_A.mean(axis=0)),
            color=COLOR_C, alpha=0.10,
        )
        ax1.axhline(0.70, color="grey", linestyle=":", linewidth=1)
        ax1.axhline(0.90, color="grey", linestyle=":", linewidth=1)
        ax1.set_xlabel("Decision (post-shift window)")
        ax1.set_ylabel("Rolling accuracy (window=50)")
        ax1.set_title("OP1 original: near-ceiling (no headroom)")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

    # Panel 2 — OP1-REVISED (with headroom)
    mean_A2 = curves_A.mean(axis=0)
    mean_B2 = curves_B.mean(axis=0)
    mean_C2 = curves_C.mean(axis=0)
    std_C2  = curves_C.std(axis=0)

    ax2.plot(x, mean_A2, color=COLOR_A, linewidth=2, label="A — Baseline (partial warmup)")
    ax2.plot(x, mean_B2, color=COLOR_B, linewidth=2, label="B — Correct operator")
    ax2.plot(x, mean_C2, color=COLOR_C, linestyle="--", linewidth=2, label="C — Harmful operator")
    ax2.fill_between(x, mean_C2 - std_C2, mean_C2 + std_C2, color=COLOR_C, alpha=0.12)
    ax2.fill_between(
        x, mean_A2, mean_C2,
        where=(mean_C2 < mean_A2),
        color=COLOR_C, alpha=0.10, label="Harmful below baseline",
    )

    # Find crossover for harmful recovery
    crossover2 = None
    for i in range(6, n_windows):
        if mean_C2[i] >= mean_A2[i] and mean_C2[i - 1] < mean_A2[i - 1]:
            crossover2 = i
            break

    if crossover2 is not None:
        ax2.axvline(crossover2, color="green", linestyle="--", linewidth=1.5,
                    label=f"Recovery ~decision {crossover2}")
        ax2.set_title("OP1-REVISED: Harmful op hurts (headroom created)")
    else:
        ax2.set_title("OP1-REVISED: Harmful operator — Loop 2 catches up")

    ax2.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax2.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax2.set_xlabel("Decision (post-shift window)")
    ax2.set_ylabel("Rolling accuracy (window=50)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("EXP-OP1: Near-Ceiling vs With-Headroom Comparison", fontsize=13)
    fig.tight_layout()

    save_figure(fig, "expOP1r_harmful_vs_correct", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] expOP1r_harmful_vs_correct.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 4: Cold start vs warm start — operator benefit
    # ------------------------------------------------------------------
    mean_E = curves_E.mean(axis=0)
    mean_F = curves_F.mean(axis=0)
    std_E  = curves_E.std(axis=0)
    std_F  = curves_F.std(axis=0)
    mean_A4 = curves_A.mean(axis=0)
    mean_B4 = curves_B.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Cold conditions: dashed lines with fill
    ax.plot(x, mean_F, color=COLOR_F, linestyle="--", linewidth=2,
            label="F — Cold start baseline")
    ax.fill_between(x, mean_F - std_F, mean_F + std_F, color=COLOR_F, alpha=0.12)

    ax.plot(x, mean_E, color=COLOR_E, linestyle="--", linewidth=2,
            label="E — Cold start + correct op")
    ax.fill_between(x, mean_E - std_E, mean_E + std_E, color=COLOR_E, alpha=0.12)

    # Warm conditions: solid lines
    ax.plot(x, mean_A4, color=COLOR_A, linestyle="-", linewidth=2,
            label="A — Warm baseline")
    ax.plot(x, mean_B4, color=COLOR_B, linestyle="-", linewidth=2,
            label="B — Warm + correct op")

    # Annotate gap between E and F at approximately decision 100
    # decision 100 corresponds to window index ~(100 - window_size + 1) if rolling
    ann_idx = min(max(0, 100 - WINDOW_SIZE + 1), n_windows - 1)
    gap = float(mean_E[ann_idx] - mean_F[ann_idx])
    ann_y = float((mean_E[ann_idx] + mean_F[ann_idx]) / 2)
    ax.annotate(
        f"gap={gap:+.3f}",
        xy=(ann_idx, ann_y),
        xytext=(ann_idx + 8, ann_y + 0.06),
        fontsize=8,
        color=COLOR_E,
        arrowprops={"arrowstyle": "->", "color": COLOR_E, "lw": 1},
    )

    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel("Rolling accuracy (window=50)")
    ax.set_xlabel("Decision (post-shift window)")
    ax.set_title("EXP-OP1-REVISED: Operator Benefit — Cold Start vs Warm Start")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1r_cold_start_benefit", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 4] expOP1r_cold_start_benefit.png + .pdf saved")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nAll charts saved to paper_figures/:")
    for name in [
        "expOP1r_auac_curves",
        "expOP1r_auac_delta",
        "expOP1r_harmful_vs_correct",
        "expOP1r_cold_start_benefit",
    ]:
        print(f"  {name}.png + .pdf")


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    generate_charts(data["all_results"], data["summary_stats"])
