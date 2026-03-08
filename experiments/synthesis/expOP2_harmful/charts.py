"""
EXP-OP2-HARMFUL chart generation.

Four publication-quality charts (300 DPI, PDF + PNG):
  1. expOP2_acute_phase                 -- per-window delta vs baseline (B/C/C-exp/P-50)
  2. expOP2_recovery_trajectories       -- full accuracy trajectories (A/B/B-exp/C/C-exp)
  3. expOP2_partial_accuracy_threshold  -- AUAC delta by operator accuracy %
  4. expOP2_t_recovery                  -- T_recovery boxplots sorted ascending
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

RESULTS_JSON = Path("experiments/synthesis/expOP2_harmful/results.json")
WINDOW_SIZE  = 50
N_POST_SHIFT = 400

COLOR_B    = "#1E3A5F"
COLOR_Bexp = "#2563EB"
COLOR_C    = "#DC2626"
COLOR_Cexp = "#F87171"
COLOR_BL   = "#94A3B8"
COLOR_P75  = "#059669"
COLOR_P50  = "#D97706"
COLOR_P25  = "#7C3AED"

def generate_charts(data: dict) -> None:
    """Generate and save all four EXP-OP2 charts."""
    results   = data["per_seed_results"]
    summary   = data["summary"]
    cond_list = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]
    n_seeds   = len(results["A"])

    # x-axis: decisions 50..400 (rolling window of size 50, one point per decision)
    n_curve  = len(results["A"][0]["accuracy_curve"])   # = N_POST_SHIFT - WINDOW_SIZE + 1 = 351
    WINDOW_X = np.arange(WINDOW_SIZE, WINDOW_SIZE + n_curve)

    def get_curves(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["accuracy_curve"] for s in range(n_seeds)])

    def delta_curves(cond: str, ref: str = "A") -> np.ndarray:
        return get_curves(cond) - get_curves(ref)

    # ------------------------------------------------------------------
    # CHART 1: Acute-phase per-window delta
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    plot_conds = [
        ("B",     COLOR_B,    "-",  "Correct op (B, TTL=400)"),
        ("C",     COLOR_C,    "--", "Harmful op (C, TTL=400)"),
        ("C-exp", COLOR_Cexp, ":",  "Harmful op expired (C-exp, TTL=150)"),
        ("P-50",  COLOR_P50,  "-.", "50% correct (P-50)"),
    ]

    for cond, color, ls, label in plot_conds:
        dc   = delta_curves(cond)
        mean_d = dc.mean(axis=0)
        std_d  = dc.std(axis=0)
        ci95   = std_d / np.sqrt(n_seeds) * 1.96
        ax.plot(WINDOW_X, mean_d, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=5, label=label)
        ax.fill_between(WINDOW_X, mean_d - ci95, mean_d + ci95,
                        alpha=0.10, color=color)

    ax.axvspan(0, 150, alpha=0.06, color=COLOR_B, label="Acute phase (0-150 decisions)")
    ax.axvline(150, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(153, 0, "TTL=150\nexpires", fontsize=7, color="#64748B", va="bottom")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Decision (post-shift, midpoint of window)")
    ax.set_ylabel("Accuracy delta vs baseline (A)")
    ax.set_title("EXP-OP2: Per-Window sigma Effect — Acute Phase vs Recovery")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP2_acute_phase", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOP2_acute_phase.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: Recovery trajectories
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    traj_conds = [
        ("A",     COLOR_BL,   "-",  "A — Baseline"),
        ("B",     COLOR_B,    "-",  "B — Correct op (TTL=400)"),
        ("C",     COLOR_C,    "--", "C — Harmful op (TTL=400)"),
        ("C-exp", COLOR_Cexp, ":",  "C-exp — Harmful op (TTL=150)"),
        ("B-exp", COLOR_Bexp, "-.", "B-exp — Correct op (TTL=150)"),
    ]

    for cond, color, ls, label in traj_conds:
        c = get_curves(cond)
        m = c.mean(axis=0)
        ci = c.std(axis=0) / np.sqrt(n_seeds) * 1.96
        ax.plot(WINDOW_X, m, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(WINDOW_X, m - ci, m + ci, alpha=0.10, color=color)

    # Shade harm region (C < A)
    mA = get_curves("A").mean(axis=0)
    mC = get_curves("C").mean(axis=0)
    ax.fill_between(WINDOW_X, mA, mC, where=mC < mA,
                    alpha=0.15, color=COLOR_C, label="Harm region (C < A)")

    ax.axvline(150, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.7)

    # Pre-shift baseline reference — mean of baseline_pre_shift across seeds
    baseline_ref = float(np.mean([results["A"][s]["baseline_pre_shift"]
                                   for s in range(n_seeds)]))
    ax.axhline(baseline_ref, color="#64748B", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(8, baseline_ref + 0.0005, "Pre-shift baseline", fontsize=7, color="#64748B")

    ax.set_xlabel("Decision (post-shift)")
    ax.set_ylabel(f"Rolling accuracy (window={WINDOW_SIZE})")
    ax.set_title("EXP-OP2: Recovery Trajectories at lambda=0.5")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.93, 1.00)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP2_recovery_trajectories", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOP2_recovery_trajectories.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: Partial accuracy threshold
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    acc_levels = [0, 25, 50, 75, 100]
    cond_map   = {0: "P-0", 25: "P-25", 50: "P-50", 75: "P-75", 100: "B"}

    means, cis, ps = [], [], []
    for pct in acc_levels:
        cond = cond_map[pct]
        d    = delta_curves(cond).mean(axis=1)   # per-seed mean-window-delta
        means.append(float(d.mean()))
        cis.append(float(d.std() / np.sqrt(n_seeds) * 1.96))
        ps.append(summary["auac_p_values"][cond])

    bar_colors = [
        COLOR_C if m < 0 else (COLOR_P50 if m < 0.002 else COLOR_B)
        for m in means
    ]
    ax.bar(acc_levels, means, width=12, color=bar_colors, alpha=0.75)
    ax.errorbar(acc_levels, means, yerr=cis, fmt="none",
                color="black", capsize=4, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=1)

    # Mark threshold
    threshold = next(
        (pct for pct, m, p in zip(acc_levels, means, ps) if m > 0 and p < 0.05),
        None,
    )
    if threshold is not None:
        ax.axvline(threshold - 6, color=COLOR_P75, linestyle="--", linewidth=1.5,
                   label=f"Threshold: >={threshold}% correct -> net positive")
        ax.legend(fontsize=9)

    # Annotate p-values
    for xi, (pct, m, p) in enumerate(zip(acc_levels, means, ps)):
        sig = "*" if p < 0.05 else ""
        ax.text(pct, m + cis[xi] + 0.0001,
                f"p={p:.3f}{sig}", ha="center", fontsize=7.5,
                color="black")

    ax.set_xlabel("Operator accuracy (% cells with correct direction)")
    ax.set_ylabel("AUAC delta vs baseline (mean window delta)")
    ax.set_title("EXP-OP2: Partial Accuracy Threshold\n"
                 "(positive = operator net helps; * = p<0.05)")
    ax.set_xticks(acc_levels)
    ax.set_xticklabels([f"{p}%" for p in acc_levels])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP2_partial_accuracy_threshold", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] expOP2_partial_accuracy_threshold.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 4: T_recovery boxplots sorted ascending
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    t_rec_means = {
        cond: float(np.mean([results[cond][s]["t_recovery"] for s in range(n_seeds)]))
        for cond in cond_list
    }
    sorted_conds = sorted(cond_list, key=lambda c: t_rec_means[c])

    cond_colors = {
        "A":     COLOR_BL,
        "B":     COLOR_B,
        "B-exp": COLOR_Bexp,
        "C":     COLOR_C,
        "C-exp": COLOR_Cexp,
        "P-75":  COLOR_P75,
        "P-50":  COLOR_P50,
        "P-25":  COLOR_P25,
        "P-0":   COLOR_C,
    }

    for i, cond in enumerate(sorted_conds):
        t_vals = np.array([results[cond][s]["t_recovery"] for s in range(n_seeds)])
        ax.boxplot(
            t_vals,
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=cond_colors.get(cond, COLOR_BL), alpha=0.65),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="#64748B"),
            capprops=dict(color="#64748B"),
            flierprops=dict(marker=".", color="#64748B", markersize=4),
        )
        sentinel_pct = float(np.mean(t_vals >= N_POST_SHIFT)) * 100
        if sentinel_pct > 10:
            ax.text(i, N_POST_SHIFT + 8, f"{sentinel_pct:.0f}%\nnever",
                    fontsize=6, ha="center", color=COLOR_C)

    ax.axhline(N_POST_SHIFT, color=COLOR_C, linestyle=":", linewidth=1,
               alpha=0.6, label=f"Sentinel = {N_POST_SHIFT} (never recovered)")
    ax.set_xticks(range(len(sorted_conds)))
    ax.set_xticklabels(sorted_conds, rotation=30, fontsize=9)
    ax.set_ylabel("T_recovery (decisions to return within 1pp of pre-shift)")
    ax.set_title("EXP-OP2: Recovery Speed by Condition (sorted ascending)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP2_t_recovery", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 4] expOP2_t_recovery.png + .pdf saved")

    print("\nAll 4 charts saved to paper_figures/:")
    for name in [
        "expOP2_acute_phase",
        "expOP2_recovery_trajectories",
        "expOP2_partial_accuracy_threshold",
        "expOP2_t_recovery",
    ]:
        print(f"  {name}.png + .pdf")


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    generate_charts(data)
