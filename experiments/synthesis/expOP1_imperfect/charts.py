"""
EXP-OP1-IMPERFECT chart generation.

Six publication-quality charts (300 DPI, PDF + PNG):
  1. expOP1i_baseline_by_epsilon       — headroom created by noise level
  2. expOP1i_delta_by_epsilon          — sigma benefit scales with imperfection
  3. expOP1i_trajectories_epsilon010   — all 6 conditions at epsilon=0.10
  4. expOP1i_directionality_check      — correct vs harmful across epsilon
  5. expOP1i_stable_operation          — sigma neutral without campaign shift
  6. expOP1i_t70_by_epsilon            — T70 speedup by epsilon
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

RESULTS_JSON = Path("experiments/synthesis/expOP1_imperfect/results.json")
FIGURES_DIR  = Path("paper_figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_B  = "#1E3A5F"   # deep blue — correct operator
COLOR_C  = "#DC2626"   # red — harmful operator
COLOR_D  = "#D97706"   # amber — expiring operator
COLOR_G  = "#059669"   # green — stable + operator
COLOR_BL = "#94A3B8"   # slate — baseline (conditions A and H)

EPSILON_COLORS = {
    "0.00": "#CBD5E1",
    "0.05": "#94A3B8",
    "0.10": "#1E3A5F",
    "0.15": "#7C3AED",
    "0.20": "#DC2626",
}

WINDOW_SIZE  = 50
N_POST_SHIFT = 400


def generate_charts(
    all_results: dict,
    summary_stats: dict,
    epsilon_levels: list,
) -> None:
    """Generate and save all six EXP-OP1-IMPERFECT charts."""

    eps_keys = [f"{e:.2f}" for e in epsilon_levels]

    # ------------------------------------------------------------------
    # CHART 1: Baseline AUAC by epsilon — headroom created
    # ------------------------------------------------------------------
    baseline_means = [summary_stats[k]["baseline_auac"]     for k in eps_keys]
    baseline_stds  = [summary_stats[k]["baseline_auac_std"] for k in eps_keys]
    bar_colors     = [EPSILON_COLORS[k] for k in eps_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [str(e) for e in epsilon_levels],
        baseline_means,
        yerr=baseline_stds,
        color=bar_colors, alpha=0.85, capsize=5,
    )
    ax.axhline(0.97, color="#94A3B8", linestyle="--", linewidth=1.5,
               label="OP1 baseline (~0.97)")
    for bar, val in zip(bars, baseline_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylabel("Baseline AUAC (condition A, no operator)")
    ax.set_xlabel("Profile noise epsilon")
    ax.set_title("EXP-OP1-IMPERFECT: Headroom Created by Profile Noise")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1i_baseline_by_epsilon", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] expOP1i_baseline_by_epsilon.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 2: AUAC delta (B vs A) by epsilon
    # ------------------------------------------------------------------
    deltas_mean = [summary_stats[k]["B_vs_A"]["mean"] for k in eps_keys]
    deltas_std  = [summary_stats[k]["B_vs_A"]["std"]  for k in eps_keys]
    p_vals      = [summary_stats[k]["B_vs_A"]["p"]    for k in eps_keys]
    n_seeds     = len(all_results[eps_keys[0]]["condition_A"])
    ci95        = [1.96 * s / np.sqrt(n_seeds) for s in deltas_std]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (ek, dm, ci, pv) in enumerate(zip(eps_keys, deltas_mean, ci95, p_vals)):
        color = EPSILON_COLORS[ek]
        # Marker size capped to avoid inf
        msize = min(300, max(30, 10 / max(pv, 1e-4)))
        ax.scatter([epsilon_levels[i]], [dm], color=color, s=msize, zorder=5)
        ax.errorbar([epsilon_levels[i]], [dm], yerr=[[ci], [ci]],
                    color=color, linewidth=2, capsize=5)
        # Star/cross annotation at epsilon=0.10
        if abs(epsilon_levels[i] - 0.10) < 1e-6:
            marker = "*PASS*" if summary_stats[ek]["B_vs_A"]["gate_pass"] else "FAIL"
            ax.text(epsilon_levels[i], dm + ci + 0.002, marker,
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1)
    ax.text(0.01, 0.003, "Bonferroni alpha=0.0125 threshold", fontsize=7.5, color="grey")
    ax.set_ylabel("AUAC delta (B - A)")
    ax.set_xlabel("Profile noise epsilon")
    ax.set_title("EXP-OP1-IMPERFECT: sigma Benefit Scales with Profile Imperfection")
    ax.set_xticks(epsilon_levels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1i_delta_by_epsilon", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] expOP1i_delta_by_epsilon.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 3: Trajectories at epsilon=0.10
    # ------------------------------------------------------------------
    eps10 = "0.10"
    cond_data_10 = all_results[eps10]

    curves = {
        c: np.array([r["accuracy_curve"] for r in cond_data_10[c]])
        for c in ["condition_A", "condition_B", "condition_C",
                  "condition_D", "condition_G", "condition_H"]
    }
    n_windows = curves["condition_A"].shape[1]
    x = np.arange(n_windows)

    fig, ax = plt.subplots(figsize=(11, 5))
    for cond, color, label, ls in [
        ("condition_A", COLOR_BL,         "A — Baseline",             "-"),
        ("condition_B", COLOR_B,           "B — Correct operator",     "-"),
        ("condition_C", COLOR_C,           "C — Harmful operator",     "--"),
        ("condition_D", COLOR_D,           "D — Expiring operator",    ":"),
        ("condition_G", COLOR_G,           "G — Stable + correct op",  "-."),
        ("condition_H", "#9CA3AF",         "H — Stable baseline",      "--"),
    ]:
        mean = curves[cond].mean(axis=0)
        std  = curves[cond].std(axis=0)
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)

    # D expires at decision 200 — window index approximation
    expire_x = max(0, (200 // WINDOW_SIZE) - 1)
    ax.axvline(expire_x, color=COLOR_D, linestyle="--", linewidth=1, alpha=0.5)
    ax.text(expire_x + 0.5, 0.25, "D expires", color=COLOR_D, fontsize=8)

    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Decision (post-shift window)")
    ax.set_ylabel("Rolling accuracy (window=50)")
    ax.set_title("EXP-OP1-IMPERFECT: epsilon=0.10 -- All Six Conditions")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1i_trajectories_epsilon010", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] expOP1i_trajectories_epsilon010.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 4: Directionality check — correct vs harmful across epsilon
    # ------------------------------------------------------------------
    ba_means = [summary_stats[k]["B_vs_A"]["mean"] for k in eps_keys]
    ca_means = [summary_stats[k]["C_vs_A"]["mean"] for k in eps_keys]
    eps_x    = epsilon_levels

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eps_x, ba_means, color=COLOR_B, marker="o", linestyle="-",
            linewidth=2, label="B vs A — Correct operator (expect positive)")
    ax.plot(eps_x, ca_means, color=COLOR_C, marker="s", linestyle="--",
            linewidth=2, label="C vs A — Harmful operator (expect negative)")
    ax.fill_between(eps_x, ba_means, ca_means, alpha=0.08, color="grey")
    ax.axhline(0, color="black", linewidth=1)
    ax.text(max(eps_x) * 0.6, max(max(ba_means), abs(min(ca_means))) * 0.7,
            "Lines should diverge\nas epsilon increases",
            fontsize=8, color="grey", ha="left")
    ax.set_ylabel("AUAC delta vs baseline (A)")
    ax.set_xlabel("Profile noise epsilon")
    ax.set_title("EXP-OP1-IMPERFECT: Directionality Check -- Correct vs Harmful")
    ax.legend(fontsize=8)
    ax.set_xticks(eps_x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1i_directionality_check", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 4] expOP1i_directionality_check.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 5: Stable-operation control at epsilon=0.10
    # ------------------------------------------------------------------
    curves_G = curves["condition_G"]
    curves_H = curves["condition_H"]
    mean_G = curves_G.mean(axis=0)
    mean_H = curves_H.mean(axis=0)
    std_G  = curves_G.std(axis=0)
    std_H  = curves_H.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mean_G, color=COLOR_G,  linestyle="-",  linewidth=2,
            label="G — Stable + correct operator")
    ax.plot(x, mean_H, color=COLOR_BL, linestyle="--", linewidth=2,
            label="H — Stable baseline")
    ax.fill_between(x, mean_G - std_G, mean_G + std_G, color=COLOR_G,  alpha=0.10)
    ax.fill_between(x, mean_H - std_H, mean_H + std_H, color=COLOR_BL, alpha=0.10)

    # Shade where operator hurts (red) and where it helps (green)
    ax.fill_between(x, mean_H, mean_G, where=(mean_G < mean_H),
                    color=COLOR_C, alpha=0.15, label="sigma hurts")
    ax.fill_between(x, mean_H, mean_G, where=(mean_G >= mean_H),
                    color=COLOR_G, alpha=0.10, label="sigma helps")

    # Annotate delta at final window
    final_delta = float(mean_G[-1] - mean_H[-1])
    ax.annotate(
        f"final delta={final_delta:+.4f}",
        xy=(n_windows - 1, float((mean_G[-1] + mean_H[-1]) / 2)),
        xytext=(n_windows - 30, float((mean_G[-1] + mean_H[-1]) / 2) + 0.05),
        fontsize=8, color=COLOR_G,
        arrowprops={"arrowstyle": "->", "color": COLOR_G, "lw": 1},
    )
    ax.axhline(0.70, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel("Rolling accuracy (window=50)")
    ax.set_xlabel("Decision (stable period)")
    ax.set_title("EXP-OP1-IMPERFECT: Stable-Operation Control -- Does sigma Hurt?")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1i_stable_operation", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 5] expOP1i_stable_operation.png + .pdf saved")

    # ------------------------------------------------------------------
    # CHART 6: T70 by epsilon — A vs B
    # ------------------------------------------------------------------
    t70_A_means, t70_A_stds = [], []
    t70_B_means, t70_B_stds = [], []
    has_nan = False

    for ek in eps_keys:
        t70_A_vals = [r["t70"] if r["t70"] is not None else N_POST_SHIFT
                      for r in all_results[ek]["condition_A"]]
        t70_B_vals = [r["t70"] if r["t70"] is not None else N_POST_SHIFT
                      for r in all_results[ek]["condition_B"]]
        if any(r["t70"] is None for r in all_results[ek]["condition_A"]):
            has_nan = True
        if any(r["t70"] is None for r in all_results[ek]["condition_B"]):
            has_nan = True
        t70_A_means.append(float(np.mean(t70_A_vals)))
        t70_A_stds.append(float(np.std(t70_A_vals)))
        t70_B_means.append(float(np.mean(t70_B_vals)))
        t70_B_stds.append(float(np.std(t70_B_vals)))

    n_eps    = len(eps_keys)
    x_eps    = np.arange(n_eps)
    bar_w    = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_A = ax.bar(x_eps - bar_w / 2, t70_A_means, bar_w,
                    label="A — Baseline", color=COLOR_BL, alpha=0.85,
                    yerr=t70_A_stds, capsize=4)
    bars_B = ax.bar(x_eps + bar_w / 2, t70_B_means, bar_w,
                    label="B — Correct operator", color=COLOR_B, alpha=0.85,
                    yerr=t70_B_stds, capsize=4)

    if has_nan:
        ax.text(0.5, 0.95, "* = never reached 70% (sentinel = N_POST_SHIFT)",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="grey")

    ax.set_xticks(x_eps)
    ax.set_xticklabels([str(e) for e in epsilon_levels])
    ax.set_xlabel("Profile noise epsilon")
    ax.set_ylabel("Decisions to reach 70% accuracy")
    ax.set_title("EXP-OP1-IMPERFECT: T70 Speedup by Epsilon Level")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "expOP1i_t70_by_epsilon", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 6] expOP1i_t70_by_epsilon.png + .pdf saved")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nAll 6 charts saved to paper_figures/:")
    for name in [
        "expOP1i_baseline_by_epsilon",
        "expOP1i_delta_by_epsilon",
        "expOP1i_trajectories_epsilon010",
        "expOP1i_directionality_check",
        "expOP1i_stable_operation",
        "expOP1i_t70_by_epsilon",
    ]:
        print(f"  {name}.png + .pdf")


if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    generate_charts(data["all_results"], data["summary_stats"], data["epsilon_levels"])
