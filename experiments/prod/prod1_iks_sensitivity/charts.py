"""
PROD-1 Charts

Chart 1 — prod1_iks_trajectory_by_kappa:  IKS mean trajectory per κ, ±1 std band
Chart 2 — prod1_iks_at_decision_milestones: grouped bars at t=50, 100, 200
Chart 3 — prod1_kappa_recommendation:  scatter of IKS(200) vs κ with κ* annotation
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure, COLORS

RESULTS_PATH = Path(__file__).parent / "results" / "iks_results.npy"


def make_charts() -> None:
    r = np.load(str(RESULTS_PATH), allow_pickle=True).item()

    results      = r["results"]
    summary      = r["summary"]
    kappa_star   = r["kappa_star"]
    kappa_values = r["kappa_values"]
    N_decisions  = int(r["N_decisions"])
    N_seeds      = int(r["N_seeds"])
    C_dim        = r.get("C", "?")
    A_dim        = r.get("A", "?")
    d_dim        = r.get("d", "?")

    # Color palette: one per kappa, highlight kappa_star
    base_colors = ["#9e9e9e", "#64b5f6", "#1565c0", "#2e7d32", "#f57f17", "#c62828"]
    colors = {k: c for k, c in zip(kappa_values, base_colors)}

    caption_base = (
        f"{N_seeds} seeds, C={C_dim}, A={A_dim}, d={d_dim}, "
        "\u03c4=0.1, \u03b7=\u03b7_neg=0.05, oracle noise=10%, warm-start centroids."
    )

    # =========================================================================
    # Chart 1: IKS trajectory by kappa
    # =========================================================================
    fig1, ax = plt.subplots(figsize=(12, 6.5))
    fig1.subplots_adjust(bottom=0.18)

    x = np.arange(1, N_decisions + 1)
    for kappa in kappa_values:
        arr       = np.array(results[kappa])   # (N_seeds, N_decisions)
        mean_traj = arr.mean(axis=0)
        std_traj  = arr.std(axis=0)
        lw        = 2.8 if kappa == kappa_star else 1.6
        ls        = "-"  if kappa == kappa_star else "--"
        label     = (
            f"\u03ba={kappa:.2f}  IKS(200)={summary[kappa]['mean']:.1f}"
            + (" \u2190 \u03ba*" if kappa == kappa_star else "")
        )
        ax.plot(x, mean_traj, color=colors[kappa], lw=lw, ls=ls,
                label=label, zorder=4)
        ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj,
                        color=colors[kappa], alpha=0.08, zorder=3)

    ax.axhline(15,  color="#2e7d32", lw=1.4, ls=":", zorder=2,
               label="IKS=15  (interpretable floor)")
    ax.axhline(40,  color="#e65100", lw=1.4, ls=":", zorder=2,
               label="IKS=40  (saturation risk)")
    ax.axhline(100, color="#9e9e9e", lw=0.8, ls=":", zorder=1)
    ax.axvline(200, color="black",   lw=1.2, ls="--", alpha=0.5,
               label="t=200  (demo window boundary)")
    ax.axhspan(15, 40, color="#a5d6a7", alpha=0.12, zorder=1,
               label="Interpretable zone [15, 40]")

    if kappa_star is not None:
        s = summary[kappa_star]
        # Place annotation to right of t=200 line; adjust y so it stays in plot
        ann_y = max(18, min(s["mean"] - 12, 65))
        ax.text(
            202, ann_y,
            f"\u03ba* = {kappa_star}\nIKS(200) = {s['mean']:.1f}\n"
            f"in [15,40]: {s['pct_in_range']:.0%} of seeds",
            fontsize=9.5, color=colors[kappa_star],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=colors[kappa_star], linewidth=1.3),
        )

    ax.set_xlim(0, 220)
    ax.set_ylim(-2, 108)
    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("IKS value", fontsize=12)
    ax.set_title(
        "PROD-1: IKS trajectory by normalization constant \u03ba\n"
        "Target: IKS(200) \u2208 [15, 40] in \u226590% of seeds",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92, ncol=2)

    fig1.text(0.5, 0.02, caption_base, ha="center", fontsize=9,
              color="#555", style="italic")

    save_figure(fig1, "prod1_iks_trajectory_by_kappa", output_dir="paper_figures")
    plt.close(fig1)
    print("[CHART 1] prod1_iks_trajectory_by_kappa.png + .pdf saved")

    # =========================================================================
    # Chart 2: IKS at decision milestones
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(12, 5.5))
    fig2.subplots_adjust(bottom=0.18)

    milestones       = [49, 99, 199]    # 0-based decision indices → t=50, 100, 200
    milestone_labels = ["t=50", "t=100", "t=200"]
    n_kappa          = len(kappa_values)
    n_milestones     = len(milestones)
    group_width      = 0.8
    bar_width        = group_width / n_kappa

    for ki, kappa in enumerate(kappa_values):
        arr = np.array(results[kappa])
        for mi, (dec_idx, _mlabel) in enumerate(zip(milestones, milestone_labels)):
            vals     = arr[:, dec_idx]
            mean_v   = float(vals.mean())
            ci95     = 1.96 * float(vals.std()) / np.sqrt(N_seeds)
            x_pos    = mi + (ki - n_kappa / 2 + 0.5) * bar_width
            in_range = 15 <= mean_v <= 40
            color    = colors[kappa] if in_range else "#ef9a9a"
            edge     = "black" if kappa == kappa_star else "none"
            ax.bar(x_pos, mean_v, bar_width * 0.9,
                   color=color, edgecolor=edge, linewidth=1.5,
                   alpha=0.85, zorder=4)
            ax.errorbar(x_pos, mean_v, yerr=ci95,
                        color="black", capsize=3, capthick=1.2, lw=1.2, zorder=5)

    ax.axhspan(15, 40, color="#a5d6a7", alpha=0.15, zorder=1)
    ax.axhline(15,  color="#2e7d32", lw=1.2, ls=":", zorder=2)
    ax.axhline(40,  color="#e65100", lw=1.2, ls=":", zorder=2)
    ax.axhline(100, color="#9e9e9e", lw=0.8, ls=":", zorder=1)

    ax.set_xticks(range(n_milestones))
    ax.set_xticklabels(milestone_labels, fontsize=12)
    ax.set_ylim(-2, 108)
    ax.set_ylabel("IKS value (mean \u00b1 95% CI)", fontsize=11)
    ax.set_title(
        "PROD-1: IKS value at demo window milestones\n"
        "Green band = interpretable zone [15,40]. Red bars = outside zone.",
        fontsize=13, fontweight="bold",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k],
                       label=f"\u03ba={k:.2f}" + (" \u2190 \u03ba*" if k == kappa_star else ""))
        for k in kappa_values
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.92, ncol=3)

    fig2.text(0.5, 0.02, caption_base, ha="center", fontsize=9,
              color="#555", style="italic")

    save_figure(fig2, "prod1_iks_at_decision_milestones", output_dir="paper_figures")
    plt.close(fig2)
    print("[CHART 2] prod1_iks_at_decision_milestones.png + .pdf saved")

    # =========================================================================
    # Chart 3: kappa recommendation scatter
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(9, 5.5))
    fig3.subplots_adjust(bottom=0.22)

    means_200  = [summary[k]["mean"] for k in kappa_values]
    stds_200   = [summary[k]["std"]  for k in kappa_values]
    colors_pts = [colors[k] for k in kappa_values]

    ax.scatter(kappa_values, means_200, s=120, c=colors_pts,
               zorder=5, edgecolors="black", linewidth=0.8)
    ax.errorbar(kappa_values, means_200, yerr=stds_200,
                fmt="none", color="black", capsize=5, capthick=1.2,
                lw=1.2, zorder=4)

    ax.axhspan(15, 40, color="#a5d6a7", alpha=0.18, zorder=1,
               label="Interpretable zone [15, 40]")
    ax.axhline(15, color="#2e7d32", lw=1.2, ls=":", zorder=2)
    ax.axhline(40, color="#e65100", lw=1.2, ls=":", zorder=2)

    if kappa_star is not None:
        s = summary[kappa_star]
        ax.scatter([kappa_star], [s["mean"]], s=220, color=colors[kappa_star],
                   marker="*", zorder=6, edgecolors="black", linewidth=0.8,
                   label=f"\u03ba* = {kappa_star}  (recommended)")
        # Choose annotation direction based on position along x axis
        x_offset = 0.03 if kappa_star <= 0.20 else -0.12
        y_offset = 8.0  if s["mean"] < 60 else -18.0
        ax.annotate(
            f"\u03ba* = {kappa_star}\n"
            f"IKS(200) = {s['mean']:.1f} \u00b1 {s['std']:.1f}\n"
            f"In [15,40]: {s['pct_in_range']:.0%} of seeds\n"
            f"Demo outcome: IKS \u2265 15 in {s['pct_above_15']:.0%} of seeds",
            xy=(kappa_star, s["mean"]),
            xytext=(kappa_star + x_offset, s["mean"] + y_offset),
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9",
                      edgecolor="#2e7d32", linewidth=1.2),
            arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.2),
        )
    else:
        ax.text(
            0.5, 0.5,
            "FLAG: No \u03ba* found in [0.05, 0.30]\nSee run.py output for diagnosis",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#c62828",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee",
                      edgecolor="#c62828", linewidth=1.5),
        )

    ax.set_xlabel("\u03ba (normalization constant)", fontsize=12)
    ax.set_ylabel("IKS(200) mean \u00b1 std  (50 seeds)", fontsize=11)
    ax.set_title(
        "PROD-1: \u03ba recommendation \u2014 IKS(200) target zone [15, 40]",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xticks(kappa_values)

    fig3.text(0.5, 0.02, caption_base, ha="center", fontsize=9,
              color="#555", style="italic")

    save_figure(fig3, "prod1_kappa_recommendation", output_dir="paper_figures")
    plt.close(fig3)
    print("[CHART 3] prod1_kappa_recommendation.png + .pdf saved")


if __name__ == "__main__":
    make_charts()
