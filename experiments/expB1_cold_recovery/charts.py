"""
EXP-B1 SUPPLEMENT: Cold Start Recovery Trajectory — Chart Generation

Single-panel figure showing ProfileScorer accuracy trajectory from random
initialisation over 1,000 decisions, with warm-start reference lines and
gap annotation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure, COLORS

# Published reference values from EXP-B1
WARM_LEARNING  = 0.982   # 98.2% warm start + learning
CENTROID_ONLY  = 0.980   # 98.0% centroid only (no learning)


def make_charts(
    all_cold_trajectories: np.ndarray,
    N_decisions: int = 1000,
    window: int = 50,
) -> None:
    """Generate and save the EXP-B1 cold start recovery figure."""

    mean_traj = all_cold_trajectories.mean(axis=0)   # (951,)
    std_traj  = all_cold_trajectories.std(axis=0)

    measured_cold_initial = float(mean_traj[0])
    measured_cold_final   = float(mean_traj[-1])
    gap = WARM_LEARNING - measured_cold_final

    # X-axis: rolling window endpoints. Window=50 → first window covers dec 1..50,
    # centred at ~25; last window covers dec 951..1000, centred at ~975.
    x = np.arange(window, N_decisions + 1)   # 50 .. 1000  (len 951)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(bottom=0.18)

    # --- Main trajectory: mean ± std ---
    ax.plot(
        x, mean_traj,
        color="#1565c0", linewidth=2.0, zorder=4,
        label=f"Cold start (random init)  \u2192  {measured_cold_final:.1%} at dec 1000",
    )
    ax.fill_between(
        x, mean_traj - std_traj, mean_traj + std_traj,
        color="#1565c0", alpha=0.15, zorder=3, label="\u00b11 std (10 seeds)",
    )

    # --- Reference lines ---
    ax.axhline(
        WARM_LEARNING, color="#c62828", linewidth=1.8, linestyle="--", zorder=2,
        label=f"Warm start + learning  {WARM_LEARNING:.1%}  (EXP-B1)",
    )
    ax.axhline(
        CENTROID_ONLY, color="#616161", linewidth=1.4, linestyle=":", zorder=2,
        label=f"Centroid only (no learning)  {CENTROID_ONLY:.1%}",
    )

    # --- Gap annotation: double-headed arrow from cold final to warm ceiling ---
    ax.annotate(
        "",
        xy=(950, WARM_LEARNING - 0.001),
        xytext=(950, measured_cold_final + 0.001),
        arrowprops=dict(
            arrowstyle="<->", color="#c62828",
            lw=1.5, mutation_scale=12,
        ),
    )
    ax.text(
        960, (WARM_LEARNING + measured_cold_final) / 2,
        f"{gap:.1%}pp\ngap",
        ha="left", va="center", fontsize=10, color="#c62828", fontweight="bold",
    )

    # --- Checkpoint annotation: decision 200 ---
    idx_200 = int(np.searchsorted(x, 200))
    acc_at_200 = float(mean_traj[idx_200]) if idx_200 < len(mean_traj) else float(mean_traj[-1])
    ax.axvline(200, color="#e65100", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.text(
        205, 0.62,
        f"dec 200\n~{acc_at_200:.0%}",
        ha="left", va="bottom", fontsize=9.5, color="#e65100",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                  edgecolor="#e65100", alpha=0.85),
    )

    # --- Cold start initial annotation ---
    ax.annotate(
        f"Cold start: ~{measured_cold_initial:.0%}",
        xy=(x[0], measured_cold_initial),
        xytext=(80, measured_cold_initial - 0.06),
        ha="left", fontsize=10, color="#1565c0",
        arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.2),
    )

    # --- Key insight box ---
    insight = (
        f"Cold-warm gap at 1,000 decisions: {gap:.1%}pp\n"
        "Never closes without expert profile initialisation.\n"
        "Warm start (DomainConfig) is a production requirement."
    )
    ax.text(
        0.36, 0.14, insight,
        transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#e8f5e9",
                  edgecolor="#388e3c", linewidth=1.2),
    )

    ax.set_xlim(0, 1050)
    ax.set_ylim(0.48, 1.03)
    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("Accuracy (rolling 50-decision mean)", fontsize=12)
    ax.set_title(
        "EXP-B1: Cold Start Recovery \u2014 ProfileScorer from Random Initialisation",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    caption = (
        "10 seeds, A=4, C=5, d=6, \u03c4=0.1, \u03b7=\u03b7_neg=0.05, noise=0%, "
        "centroids initialised uniform random [0,1]. "
        "Reference lines from EXP-B1 published results."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color="#555", style="italic")

    save_figure(fig, "expB1_cold_start_recovery", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART] expB1_cold_start_recovery.png + .pdf saved")


if __name__ == "__main__":
    # Standalone: load previously saved trajectories and regenerate chart.
    npy_path = Path(__file__).parent / "cold_trajectories.npy"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"cold_trajectories.npy not found — run run.py first.\n  Expected: {npy_path}"
        )
    all_cold = np.load(str(npy_path))
    print(f"Loaded trajectories: shape={all_cold.shape}")
    make_charts(all_cold)
