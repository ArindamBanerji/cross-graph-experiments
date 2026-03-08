"""
EXP-OP1-FINAL chart generation.

Five publication-quality charts (300 DPI, PDF + PNG):
  1. expOP1f_window_sweep         -- AUAC delta by lambda (sweep bar chart)
  2. expOP1f_trajectories_lambda05 -- all 6 conditions at lambda=0.5
  3. expOP1f_directionality        -- B vs A and C vs A across lambda values
  4. expOP1f_stable_operation      -- G vs H (sigma neutral without shift)
  5. expOP1f_loop2_overshoot       -- D expiry recovery + C overshoot
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

FIGURES_DIR = Path("paper_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_A  = "#94A3B8"   # slate grey — baseline
COLOR_B  = "#1E3A5F"   # deep blue — correct operator
COLOR_C  = "#DC2626"   # red — harmful operator
COLOR_D  = "#D97706"   # amber — expiring operator
COLOR_G  = "#059669"   # green — stable + correct op
COLOR_H  = "#9CA3AF"   # light grey — stable baseline

SIGMA_VALUE   = 0.4
WINDOW_SIZE   = 50
N_POST_SHIFT  = 400
TTL_EXPIRE    = 150


def generate_charts(results_path: Path) -> None:
    """Load results from results_path and generate all 5 charts."""
    with open(results_path) as fh:
        data = json.load(fh)

    sweep_results = data["sweep_results"]
    all_results   = data["all_results"]
    summary       = data["summary_stats"]
    lambda_sweep  = data["config"]["lambda_sweep"]
    bonf_alpha    = data["config"]["bonferroni_alpha"]
    n_seeds       = len(data["config"]["seeds"])

    # ------------------------------------------------------------------
    # CHART 1: Window sweep — AUAC delta by lambda (bar chart)
    # ------------------------------------------------------------------
    means_B = [sweep_results[str(lv)]["mean_delta_B"] for lv in lambda_sweep]
    stds_B  = [sweep_results[str(lv)]["std_delta_B"]  for lv in lambda_sweep]
    means_C = [sweep_results[str(lv)]["mean_delta_C"] for lv in lambda_sweep]
    p_vals  = [sweep_results[str(lv)]["p_value"]       for lv in lambda_sweep]
    ci95    = [1.96 * s / np.sqrt(n_seeds) for s in stds_B]

    pass_colors = [
        COLOR_B if sweep_results[str(lv)]["gate_pass"] else "#93C5FD"
        for lv in lambda_sweep
    ]

    x = np.arange(len(lambda_sweep))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_B = ax.bar(x - bar_w / 2, means_B, bar_w,
                    color=pass_colors, alpha=0.85, label="B vs A (correct op)",
                    yerr=ci95, capsize=4, ecolor="#64748B")
    bars_C = ax.bar(x + bar_w / 2, means_C, bar_w,
                    color=COLOR_C, alpha=0.70, label="C vs A (harmful op)")

    ax.axhline(0, color="black", linewidth=1)
    for xi, (pv, lv) in enumerate(zip(p_vals, lambda_sweep)):
        if pv < bonf_alpha and means_B[lambda_sweep.index(lv)] > 0:
            ax.text(xi - bar_w / 2, means_B[lambda_sweep.index(lv)] + ci95[lambda_sweep.index(lv)] + 0.0005,
                    "*", ha="center", fontsize=12, color=COLOR_B, fontweight="bold")

    ax.axhspan(-0.001, 0.001, alpha=0.05, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv:.1f}" for lv in lambda_sweep])
    ax.set_xlabel("lambda (coupling constant)")
    ax.set_ylabel("AUAC delta vs baseline A")
    ax.set_title("EXP-OP1-FINAL: Narrow Lambda Sweep\n"
                 f"(* = Bonferroni PASS, alpha={bonf_alpha:.4f}, dark blue = gate pass)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1f_window_sweep", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOP1f_window_sweep.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: Trajectories at lambda=0.5 — all 6 conditions
    # ------------------------------------------------------------------
    def get_curves(cond_key: str) -> np.ndarray:
        return np.array([r["accuracy_curve"] for r in all_results[cond_key]])

    curves = {
        c: get_curves(c)
        for c in ["condition_A", "condition_B", "condition_C",
                  "condition_D", "condition_G", "condition_H"]
    }
    n_windows = curves["condition_A"].shape[1]
    xw = np.arange(n_windows)

    fig, ax = plt.subplots(figsize=(11, 5))
    for cond, color, label, ls in [
        ("condition_A", COLOR_A, "A — Baseline",              "-"),
        ("condition_B", COLOR_B, "B — Correct operator",      "-"),
        ("condition_C", COLOR_C, "C — Harmful operator",      "--"),
        ("condition_D", COLOR_D, "D — Expiring operator",     ":"),
        ("condition_G", COLOR_G, "G — Stable + correct op",   "-."),
        ("condition_H", COLOR_H, "H — Stable baseline",       "--"),
    ]:
        mean = curves[cond].mean(axis=0)
        std  = curves[cond].std(axis=0)
        ax.plot(xw, mean, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(xw, mean - std, mean + std, color=color, alpha=0.10)

    expire_x = max(0, (TTL_EXPIRE // WINDOW_SIZE) - 1)
    ax.axvline(expire_x, color=COLOR_D, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(expire_x + 0.3, 0.20, "D expires", color=COLOR_D, fontsize=8)

    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Decision (window index, post-shift)")
    ax.set_ylabel(f"Rolling accuracy (window={WINDOW_SIZE})")
    ax.set_title("EXP-OP1-FINAL: lambda=0.5 -- All Six Conditions")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1f_trajectories_lambda05", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOP1f_trajectories_lambda05.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: Directionality — B vs A and C vs A across lambda
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — AUAC delta lines
    ax1.plot(lambda_sweep, means_B, color=COLOR_B, marker="o", linewidth=2,
             label="B vs A — Correct (expect >0)")
    ax1.plot(lambda_sweep, means_C, color=COLOR_C, marker="s", linestyle="--",
             linewidth=2, label="C vs A — Harmful (expect <0)")
    ax1.fill_between(lambda_sweep, means_B, means_C, alpha=0.07, color="grey")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("AUAC delta vs baseline A")
    ax1.set_title("Correct vs Harmful Directionality")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2 — p-value (log scale)
    p_plot = [max(p, 1e-5) for p in p_vals]
    ax2.semilogy(lambda_sweep, p_plot, color=COLOR_B, marker="o", linewidth=2,
                 label="p-value (B vs A)")
    ax2.axhline(0.05, color="orange", linestyle="--", linewidth=1.5, label="alpha=0.05")
    ax2.axhline(bonf_alpha, color="red", linestyle="--", linewidth=1.5,
                label=f"Bonferroni={bonf_alpha:.4f}")
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("p-value (log scale)")
    ax2.set_title("Statistical Significance vs lambda")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("EXP-OP1-FINAL: Directionality and Significance", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "expOP1f_directionality", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] expOP1f_directionality.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 4: Stable operation — G vs H
    # ------------------------------------------------------------------
    mean_G = curves["condition_G"].mean(axis=0)
    mean_H = curves["condition_H"].mean(axis=0)
    std_G  = curves["condition_G"].std(axis=0)
    std_H  = curves["condition_H"].std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xw, mean_G, color=COLOR_G, linestyle="-", linewidth=2,
            label="G — Stable + correct operator")
    ax.plot(xw, mean_H, color=COLOR_H, linestyle="--", linewidth=2,
            label="H — Stable baseline")
    ax.fill_between(xw, mean_G - std_G, mean_G + std_G, color=COLOR_G, alpha=0.10)
    ax.fill_between(xw, mean_H - std_H, mean_H + std_H, color=COLOR_H, alpha=0.10)

    # Shade differences
    ax.fill_between(xw, mean_H, mean_G, where=(mean_G < mean_H),
                    color=COLOR_C, alpha=0.15, label="sigma hurts (G < H)")
    ax.fill_between(xw, mean_H, mean_G, where=(mean_G >= mean_H),
                    color=COLOR_G, alpha=0.10, label="sigma helps (G >= H)")

    final_delta = float(mean_G[-1] - mean_H[-1])
    ax.annotate(
        f"final delta={final_delta:+.4f}",
        xy=(n_windows - 1, float((mean_G[-1] + mean_H[-1]) / 2)),
        xytext=(n_windows - 25, float((mean_G[-1] + mean_H[-1]) / 2) + 0.05),
        fontsize=8, color=COLOR_G,
        arrowprops={"arrowstyle": "->", "color": COLOR_G, "lw": 1},
    )

    gvh_mean = summary["G_vs_H"]["mean"]
    stable_ok = summary["G_vs_H"]["stable_ok"]
    ax.text(0.05, 0.07, f"G-H mean delta = {gvh_mean:+.4f}  "
            f"{'[PASS: sigma neutral]' if stable_ok else '[FAIL: sigma disrupts stable]'}",
            transform=ax.transAxes, fontsize=8,
            color=COLOR_G if stable_ok else COLOR_C)

    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel(f"Rolling accuracy (window={WINDOW_SIZE})")
    ax.set_xlabel("Decision (stable-period window index)")
    ax.set_title("EXP-OP1-FINAL: Stable-Operation Control -- Does sigma Hurt?")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1f_stable_operation", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 4] expOP1f_stable_operation.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 5: Loop-2 overshoot — D expiry recovery + C harm + B benefit
    # ------------------------------------------------------------------
    mean_A = curves["condition_A"].mean(axis=0)
    mean_B = curves["condition_B"].mean(axis=0)
    mean_C = curves["condition_C"].mean(axis=0)
    mean_D = curves["condition_D"].mean(axis=0)
    std_A  = curves["condition_A"].std(axis=0)
    std_C  = curves["condition_C"].std(axis=0)
    std_D  = curves["condition_D"].std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — D expiry recovery
    ax1.plot(xw, mean_A, color=COLOR_A, linestyle="-",  linewidth=2, label="A — Baseline")
    ax1.plot(xw, mean_B, color=COLOR_B, linestyle="-",  linewidth=2, label="B — Correct (full TTL)")
    ax1.plot(xw, mean_D, color=COLOR_D, linestyle=":",  linewidth=2, label="D — Expiring (TTL=150)")
    ax1.fill_between(xw, mean_D - std_D, mean_D + std_D, color=COLOR_D, alpha=0.12)

    ax1.axvline(expire_x, color=COLOR_D, linestyle="--", linewidth=1.5,
                alpha=0.6, label=f"D expires (decision ~{TTL_EXPIRE})")
    ax1.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax1.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax1.set_xlabel("Decision window (post-shift)")
    ax1.set_ylabel(f"Rolling accuracy (window={WINDOW_SIZE})")
    ax1.set_title("D Expiry Recovery vs Full-TTL Operator")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2 — C harmful overshoot and Loop-2 recovery
    ax2.plot(xw, mean_A, color=COLOR_A, linestyle="-", linewidth=2, label="A — Baseline")
    ax2.plot(xw, mean_C, color=COLOR_C, linestyle="--", linewidth=2, label="C — Harmful operator")
    ax2.fill_between(xw, mean_C - std_C, mean_C + std_C, color=COLOR_C, alpha=0.12)
    ax2.fill_between(
        xw, mean_A, mean_C,
        where=(mean_C < mean_A),
        color=COLOR_C, alpha=0.12, label="Harmful below baseline",
    )
    ax2.fill_between(
        xw, mean_A, mean_C,
        where=(mean_C >= mean_A),
        color=COLOR_G, alpha=0.10, label="Loop-2 recovery above baseline",
    )

    # Annotate minimum point of C (worst harm)
    worst_idx = int(np.argmin(mean_C))
    ax2.annotate(
        f"worst harm @ win {worst_idx}\nacc={mean_C[worst_idx]:.3f}",
        xy=(worst_idx, float(mean_C[worst_idx])),
        xytext=(worst_idx + 5, float(mean_C[worst_idx]) - 0.08),
        fontsize=8, color=COLOR_C,
        arrowprops={"arrowstyle": "->", "color": COLOR_C, "lw": 1},
    )

    ax2.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax2.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax2.set_xlabel("Decision window (post-shift)")
    ax2.set_ylabel(f"Rolling accuracy (window={WINDOW_SIZE})")
    ax2.set_title("C Harmful Op — Overshoot & Loop-2 Recovery")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("EXP-OP1-FINAL: Loop-2 Overshoot and Recovery Dynamics", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "expOP1f_loop2_overshoot", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 5] expOP1f_loop2_overshoot.png + .pdf saved")

    # Summary
    print("\nAll 5 charts saved to paper_figures/:")
    for name in [
        "expOP1f_window_sweep",
        "expOP1f_trajectories_lambda05",
        "expOP1f_directionality",
        "expOP1f_stable_operation",
        "expOP1f_loop2_overshoot",
    ]:
        print(f"  {name}.png + .pdf")


if __name__ == "__main__":
    generate_charts(Path("experiments/synthesis/expOP1_final/results.json"))
