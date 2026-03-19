"""
Bridge B Phase B Diagnostic Charts

Chart 1 — bridge_b_diagnostic_heatmap:
  Heatmap: ε (y-axis) vs e₀ (x-axis), color = G₄/G₁ reduction %.
  Annotate P2 operating point (ε=0.05, e₀=0.15).
  30% threshold contour if achievable.

Chart 2 — bridge_b_diagnostic_per_factor:
  Bar chart: per-factor G₄/G₁ reduction at (ε=0.05, e₀=0.15).
  Highlight threat_intel_enrichment (expected largest G4 benefit).
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

RESULTS_FILE = _REPO_ROOT / "results" / "bridge_b_phase_b_diagnostic.json"

FACTOR_LABELS = {
    "travel_match":            "travel_match\n(extrapolated)",
    "asset_criticality":       "asset_criticality\n(measured)",
    "threat_intel_enrichment": "threat_intel\n(measured, G4×0.70)",
    "pattern_history":         "pattern_history\n(measured)",
    "time_anomaly":            "time_anomaly\n(extrapolated)",
    "device_trust":            "device_trust\n(extrapolated)",
}

FACTOR_ORDER = [
    "travel_match",
    "asset_criticality",
    "threat_intel_enrichment",
    "pattern_history",
    "time_anomaly",
    "device_trust",
]


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — Heatmap: ε × e₀ → G₄/G₁ reduction %
# ============================================================================

def chart1_heatmap(r: dict) -> None:
    eps_values = r["eps_values"]          # 5 values (y axis)
    e0_values  = r["e0_values"]           # 4 values (x axis)
    grid       = np.array(r["reduction_grid"])   # shape (5, 4)

    p2_eps = r["p2_operating_point"]["eps"]
    p2_e0  = r["p2_operating_point"]["e0"]
    p2_ei  = eps_values.index(p2_eps)
    p2_e0i = e0_values.index(p2_e0)

    floor_confirmed = r["floor_proximity_confirmed"]
    spear_eps       = r["spearman_eps_vs_reduction"]
    spear_e0        = r["spearman_e0_vs_reduction"]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.12, right=0.90)

    # Color: diverging around 30% gate
    vmin = max(0.0, grid.min() - 2.0)
    vmax = grid.max() + 2.0
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="upper")

    # Annotate each cell
    for ei in range(len(eps_values)):
        for e0i in range(len(e0_values)):
            v = grid[ei, e0i]
            text_color = "black" if 15 < v < 50 else "white"
            ax.text(e0i, ei, f"{v:.1f}%",
                    ha="center", va="center", fontsize=11.5,
                    fontweight="bold", color=text_color)

    # P2 operating point box
    rect = plt.Rectangle(
        (p2_e0i - 0.45, p2_ei - 0.45), 0.90, 0.90,
        fill=False, edgecolor="#1565c0", linewidth=3.0, zorder=6,
    )
    ax.add_patch(rect)
    ax.text(p2_e0i, p2_ei + 0.55,
            f"P2 ({p2_eps:.2f}, {p2_e0:.2f})\n{grid[p2_ei, p2_e0i]:.1f}%",
            ha="center", va="bottom", fontsize=8.5,
            color="#1565c0", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#1565c0", alpha=0.9))

    # 30% gate contour (approximate via contour on the grid)
    try:
        cs = ax.contour(
            np.arange(len(e0_values)),
            np.arange(len(eps_values)),
            grid,
            levels=[30.0],
            colors=["#1a237e"], linewidths=2.0, linestyles="--",
        )
        ax.clabel(cs, fmt="30%% gate", fontsize=9, inline=True,
                  inline_spacing=4)
    except Exception:
        pass   # contour may fail if no crossing exists

    # Axes
    ax.set_xticks(range(len(e0_values)))
    ax.set_xticklabels([f"e₀={v:.2f}" for v in e0_values], fontsize=11)
    ax.set_yticks(range(len(eps_values)))
    ax.set_yticklabels([f"ε={v:.2f}" for v in eps_values], fontsize=11)
    ax.set_xlabel("Initial error e₀ per component", fontsize=12)
    ax.set_ylabel("Convergence threshold ε", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("G₄/G₁ reduction in N_converge (%)", fontsize=10)
    cbar.ax.axhline(30.0, color="#1a237e", lw=2.0, ls="--")

    # Gate/hypothesis box
    verdict_color = "#2e7d32" if floor_confirmed else "#c62828"
    verdict_str   = "FLOOR CONFIRMED" if floor_confirmed else "NOT CONFIRMED"
    ax.text(
        0.99, 0.97,
        f"{verdict_str}\nρ(ε, red) = {spear_eps:.3f}\nρ(e₀, red) = {spear_e0:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=verdict_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=verdict_color, linewidth=1.5, alpha=0.92),
    )

    ax.set_title(
        "Bridge B Diagnostic: G₄/G₁ Convergence Acceleration by Threshold and Initial Error\n"
        "Floor-proximity hypothesis: reduction increases as ε decreases "
        "(threat_intel e_inf ≈ 0.0499 ≈ ε=0.05)",
        fontsize=11.5, fontweight="bold",
    )

    caption = (
        "soc_product_v50 (C=6, A=4, d=6). η=0.05, ρ=0.8, N_EFF=1.111. "
        "G₄ applies threat_intel×0.70 + base/N_EFF. "
        "30 seeds × 6C × 4A = 720 cells. Blue box = P2 operating point."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_diagnostic_heatmap", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] bridge_b_diagnostic_heatmap.png + .pdf saved")


# ============================================================================
# Chart 2 — Per-factor G₄/G₁ reduction bar chart
# ============================================================================

def chart2_per_factor(r: dict) -> None:
    pf = r["per_factor"]
    p2_eps = r["p2_operating_point"]["eps"]
    p2_e0  = r["p2_operating_point"]["e0"]

    names      = FACTOR_ORDER
    labels     = [FACTOR_LABELS[n] for n in names]
    reductions = [pf[n]["reduction"] for n in names]
    e_inf_g1   = [pf[n]["e_inf_g1"] for n in names]
    e_inf_g4   = [pf[n]["e_inf_g4"] for n in names]
    g1_nc      = [pf[n]["g1_mean"]   for n in names]
    g4_nc      = [pf[n]["g4_mean"]   for n in names]

    THREAT_IDX = names.index("threat_intel_enrichment")

    def bar_color(i: int, v: float) -> str:
        if i == THREAT_IDX:
            return "#6a1b9a"   # purple — the G4-targeted factor
        if v >= 30.0:
            return "#2e7d32"
        if v >= 10.0:
            return "#f57f17"
        return "#9e9e9e"

    colors = [bar_color(i, v) for i, v in enumerate(reductions)]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.40)

    # ---- Left panel: Reduction bar chart ----
    ax = axes[0]
    bars = ax.bar(x, reductions, width=0.6, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.88, zorder=4)

    for xi, (v, bar) in enumerate(zip(reductions, bars)):
        ax.text(xi, v + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color=colors[xi])

    # e_inf annotation below each bar
    for xi in range(len(names)):
        ei1 = e_inf_g1[xi]
        ei4 = e_inf_g4[xi]
        marker = " ≈ε" if abs(ei1 - p2_eps) < 0.003 else ""
        ax.text(xi, -1.5, f"e∞_G1={ei1:.4f}{marker}\ne∞_G4={ei4:.4f}",
                ha="center", va="top", fontsize=7.5, color="#555")

    # 30% gate line
    ax.axhline(30.0, color="#2e7d32", lw=2.0, ls="--", zorder=5,
               label="30% gate")
    ax.axhline(0.0, color="black", lw=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("G₄/G₁ reduction in 1D N_converge (%)", fontsize=11)
    ax.set_ylim(-3.5, max(max(reductions) * 1.30, 35))
    ax.set_title(
        f"Per-Factor Enrichment Effect\n(ε={p2_eps}, e₀={p2_e0} — P2 operating point)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")

    # Highlight purple = G4 target
    ax.text(THREAT_IDX, max(reductions) * 1.18,
            "G4 target\n(×0.70)", ha="center", fontsize=8.5,
            color="#6a1b9a", fontweight="bold")

    # ---- Right panel: N_converge comparison G1 vs G4 per factor ----
    ax2 = axes[1]
    bar_w = 0.35
    bars_g1 = ax2.bar(x - bar_w / 2, g1_nc, bar_w,
                      color="#9e9e9e", edgecolor="black", linewidth=0.8,
                      label="G₁ (base noise)", alpha=0.88, zorder=4)
    bars_g4 = ax2.bar(x + bar_w / 2, g4_nc, bar_w,
                      color=colors, edgecolor="black", linewidth=0.8,
                      label="G₄ (enriched)", alpha=0.88, zorder=4)

    for xi, (v1, v4) in enumerate(zip(g1_nc, g4_nc)):
        ax2.text(xi + bar_w / 2, v4 + 5, f"{v4:.0f}",
                 ha="center", va="bottom", fontsize=8, color=colors[xi],
                 fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, ha="center")
    ax2.set_ylabel("Mean N_converge (1D, 1000 cells)", fontsize=11)
    ax2.set_title(
        "G₁ vs G₄ N_converge per Factor\n(1D simulation, single-factor isolation)",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(fontsize=10)

    caption = (
        f"1D per-factor simulation: {r.get('n_decisions',2000)} max decisions, "
        f"N=1000 cells, mu_true=0.5, η={r['eta']}, ε={p2_eps}, e₀={p2_e0}. "
        "G4 applies threat_intel_enrichment variance × 0.70. "
        "Purple bar = G4-targeted factor."
    )
    fig.text(0.5, 0.03, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    fig.suptitle(
        "Bridge B Diagnostic: Per-Factor Enrichment Effect "
        f"at P2 Operating Point (ε={p2_eps}, e₀={p2_e0})",
        fontsize=12, fontweight="bold",
    )

    save_figure(fig, "bridge_b_diagnostic_per_factor", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] bridge_b_diagnostic_per_factor.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_heatmap(r)
    chart2_per_factor(r)
