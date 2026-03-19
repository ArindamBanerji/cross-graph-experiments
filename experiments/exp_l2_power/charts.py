"""
P4-POWER Charts

Chart 1 — exp_l2_power_heatmap:
  Heatmap: N (y-axis) vs inflation (x-axis), color = promotion rate.
  Contour line at 80% power. Contour line at 5% false positive.
  Title: "Gate 1 Detection Power: N vs Inflation"

Chart 2 — exp_l2_power_curves:
  Power curves: one line per inflation level.
  X = N, Y = promotion rate. Horizontal line at 80%.
  Title: "Power Curves by Inflation Level"
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.viz.bridge_common import save_figure

RESULTS_FILE = _REPO_ROOT / "results" / "exp_l2_power.json"

POWER_TARGET = 0.80
FP_TARGET    = 0.05

# Inflation colors: light→dark as inflation grows
INFL_COLORS = ["#9e9e9e", "#1565c0", "#2e7d32", "#e65100", "#c62828", "#6a1b9a"]


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# =============================================================================
# Chart 1 — Heatmap
# =============================================================================

def chart1_heatmap(data: dict) -> None:
    N_values    = data["n_values"]
    infl_values = data["inflation_values"]
    rates_raw   = data["rates"]

    # Build matrix: rows = N (y), cols = inflation (x)
    matrix = np.array([
        [rates_raw[f"N{N}_infl{int(infl*100)}pp"] for infl in infl_values]
        for N in N_values
    ])  # shape (len_N, len_infl)

    x_labels = [f"{int(v*100)}pp" for v in infl_values]
    y_labels = [str(n) for n in N_values]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.10, right=0.92)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "power", ["#ffffff", "#fffde7", "#fff176", "#ffb300", "#e65100", "#b71c1c"], N=256
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Promotion rate (power)", fontsize=11)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # Contour lines at 80% power and 5% FP
    # Need fine grid for contours — interpolate on a denser grid
    xi = np.linspace(0, len(infl_values) - 1, 200)
    yi = np.linspace(0, len(N_values) - 1, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    from scipy.interpolate import RegularGridInterpolator
    x_grid = np.arange(len(infl_values), dtype=float)
    y_grid = np.arange(len(N_values),    dtype=float)
    interp = RegularGridInterpolator((y_grid, x_grid), matrix, method="linear")
    Zi = interp((Yi, Xi))

    # 80% power contour
    cs80 = ax.contour(Xi, Yi, Zi, levels=[POWER_TARGET],
                      colors=["#1565c0"], linewidths=2.5, linestyles="-")
    ax.clabel(cs80, fmt={POWER_TARGET: "80% power"}, fontsize=9, inline=True,
              inline_spacing=4)

    # 5% FP contour (horizontal at inflation=0 doesn't make a curve;
    # instead annotate FP values in the 0pp column)

    # Cell value annotations
    for ri, N in enumerate(N_values):
        for ci, infl in enumerate(infl_values):
            val = matrix[ri, ci]
            txt = f"{val*100:.0f}%"
            col = "white" if val > 0.55 else "black"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

    # Tick labels
    ax.set_xticks(range(len(infl_values)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("True inflation (Δq = attacker's quality boost)", fontsize=12)
    ax.set_ylabel("N (decisions per variant)", fontsize=12)

    ax.set_title(
        "Gate 1 Detection Power: N vs Inflation\n"
        f"P(promote | data) > {int(POWER_TARGET*100)}%  via Beta posterior normal approximation. "
        f"Δ_min={data['config']['delta_min']}, P_thresh={data['config']['promo_threshold']}.",
        fontsize=11.5, fontweight="bold",
    )

    cfg = data["config"]
    caption = (
        f"N_seeds={cfg['n_seeds']} per cell. Baseline quality={cfg['baseline_quality']}. "
        f"N split equally between variants. Blue contour = 80% power boundary. "
        "Col 0pp = false positive rate (should be <5%)."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5, color="#555", style="italic")

    save_figure(fig, "exp_l2_power_heatmap", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] exp_l2_power_heatmap.png + .pdf saved")


# =============================================================================
# Chart 2 — Power curves
# =============================================================================

def chart2_curves(data: dict) -> None:
    N_values    = data["n_values"]
    infl_values = data["inflation_values"]
    rates_raw   = data["rates"]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, right=0.97)

    for ci, infl in enumerate(infl_values):
        y = [rates_raw[f"N{N}_infl{int(infl*100)}pp"] * 100 for N in N_values]
        lbl = f"Δ={int(infl*100)}pp" if infl > 0 else "Δ=0pp (false positive)"
        ls  = "--" if infl == 0 else "-"
        lw  = 1.5 if infl == 0 else 2.2
        ax.plot(N_values, y, color=INFL_COLORS[ci], lw=lw, ls=ls,
                marker="o", markersize=6, label=lbl, zorder=4)

        # Annotate final N value
        ax.text(N_values[-1] + 10, y[-1], f"{y[-1]:.0f}%",
                color=INFL_COLORS[ci], fontsize=8.5, va="center")

    # 80% power line
    ax.axhline(POWER_TARGET * 100, color="#333", lw=1.8, ls="--", zorder=3,
               label=f"{int(POWER_TARGET*100)}% power target")

    # 5% FP line
    ax.axhline(FP_TARGET * 100, color="#e65100", lw=1.4, ls=":", zorder=3,
               label=f"{int(FP_TARGET*100)}% FP ceiling", alpha=0.8)

    # Shade the target zone
    ax.fill_between([N_values[0], N_values[-1]], POWER_TARGET * 100, 100,
                    color="#e8f5e9", alpha=0.40, zorder=1, label="≥80% zone")

    # Annotate min-N crossings for each inflation
    for ci, infl in enumerate(infl_values[1:], start=1):
        for i, N in enumerate(N_values):
            val = rates_raw[f"N{N}_infl{int(infl*100)}pp"]
            if val >= POWER_TARGET:
                ax.axvline(N, color=INFL_COLORS[ci], lw=0.9, ls=":",
                           alpha=0.55, zorder=2)
                ax.text(N, POWER_TARGET * 100 - 4,
                        f"N={N}", ha="center", va="top",
                        fontsize=7.5, color=INFL_COLORS[ci], alpha=0.85)
                break

    ax.set_xscale("log")
    ax.set_xticks(N_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("N (total decisions, split equally between variants)", fontsize=12)
    ax.set_ylabel("Promotion rate = estimated power (%)", fontsize=12)
    ax.set_ylim(-2, 107)
    ax.set_xlim(N_values[0] * 0.85, N_values[-1] * 1.15)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92, ncol=2)

    ax.set_title(
        "Power Curves by Inflation Level\n"
        f"Gate 1: P(Beta_target > Beta_baseline + Δ_min | data) > {data['config']['promo_threshold']}. "
        f"Vertical dotted = first N reaching 80% power.",
        fontsize=11.5, fontweight="bold",
    )

    cfg = data["config"]
    caption = (
        f"N_seeds={cfg['n_seeds']} per cell. Δ_min={cfg['delta_min']}, "
        f"baseline={cfg['baseline_quality']}. Log x-axis. "
        "Dashed gray = 80% target; orange dotted = 5% FP ceiling."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5, color="#555", style="italic")

    save_figure(fig, "exp_l2_power_curves", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] exp_l2_power_curves.png + .pdf saved")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data = load()
    chart1_heatmap(data)
    chart2_curves(data)
