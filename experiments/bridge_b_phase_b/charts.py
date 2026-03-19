"""
Bridge B Phase B Charts

Chart 1 — bridge_b_phase_b_convergence_curves:
  Mean error trajectory per G level. Horizontal ε=0.05 line.
  Vertical markers at mean N_converge per level.

Chart 2 — bridge_b_phase_b_reduction_bar:
  % reduction in mean N_converge vs G₁ for G₂, G₃, G₄.
  30% success threshold line.
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
import numpy as np

from src.viz.bridge_common import save_figure

_THIS_DIR    = Path(__file__).parent
RESULTS_FILE = _REPO_ROOT / "results" / "bridge_b_phase_b.json"

G_LEVELS  = ["G1", "G2", "G3", "G4"]
G_COLORS  = {
    "G1": "#9e9e9e",    # gray — baseline
    "G2": "#1565c0",    # blue
    "G3": "#2e7d32",    # green
    "G4": "#6a1b9a",    # purple — full enrichment
}
G_LABELS_SHORT = {
    "G1": "G₁ Single SIEM",
    "G2": "G₂ Two SIEMs (ρ=0.8)",
    "G3": "G₃ + Entity resolution",
    "G4": "G₄ + ThreatIndicators",
}


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — Error convergence trajectories
# ============================================================================

def chart1_convergence(r: dict) -> None:
    eps         = r["eps"]
    n_dec       = r["n_decisions"]
    levels      = r["levels"]
    reductions  = r["reductions"]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88)

    for g_level in G_LEVELS:
        s      = levels[g_level]
        # Reconstruct x axis from downsampled trajectory (every 10th point)
        traj   = np.array(s["mean_error_traj"])
        x      = np.arange(len(traj)) * 10        # decisions (downsampled by 10)
        color  = G_COLORS[g_level]
        mean_n = s["mean_n_converge"]
        lw     = 2.6 if g_level == "G4" else 1.8

        red_str = ("baseline" if g_level == "G1"
                   else f"−{reductions[g_level]:.1f}%")
        label = f"{G_LABELS_SHORT[g_level]}  (N_conv={mean_n:.0f}, {red_str})"

        ax.plot(x, traj, color=color, lw=lw, label=label, zorder=4)

        # Vertical marker at mean N_converge (if within window)
        if mean_n < n_dec * 0.98:
            ax.axvline(mean_n, color=color, lw=1.0, ls=":", alpha=0.7, zorder=3)
            ax.text(mean_n + 8, traj[0] * (0.75 - G_LEVELS.index(g_level) * 0.08),
                    f"N_conv={mean_n:.0f}",
                    fontsize=7.5, color=color, rotation=90, va="top", alpha=0.85)

    # ε threshold
    ax.axhline(eps, color="#e65100", lw=1.5, ls="--", zorder=5,
               label=f"ε = {eps} (convergence threshold)")

    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("Mean max-component error  max‖μ_t − μ*‖", fontsize=11)
    ax.set_xlim(-20, n_dec + 50)
    ax.set_ylim(0, max(
        levels["G1"]["mean_error_traj"][0] if levels["G1"]["mean_error_traj"] else 0.2,
        0.20,
    ) * 1.08)

    gate_str   = "PASS ✓" if r["gates"]["overall_pass"] else "FAIL ✗"
    gate_color = "#2e7d32" if r["gates"]["overall_pass"] else "#c62828"
    g4_red     = r["gates"]["g4_reduction_pct"]
    ax.text(
        0.99, 0.97,
        f"{gate_str}\nG₄/G₁ −{g4_red:.1f}% (gate ≥30%)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10.5, fontweight="bold", color=gate_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=gate_color, linewidth=1.5, alpha=0.92),
    )

    ax.set_title(
        "Bridge B Phase B: Centroid Convergence by Graph Maturity Level\n"
        "Single-cell simulation (each (c,a) centroid independent). "
        "Mean over 1200 cells (50 seeds × 6 categories × 4 actions).",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). "
        f"η={r['eta']}, ε={r['eps']}, e₀_base={r['e0_base']}, ρ={r['rho']}. "
        f"Factor variances from P1 (FX-1-PROXY-REAL: 3 measured + 3 extrapolated)."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_b_convergence_curves", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] bridge_b_phase_b_convergence_curves.png + .pdf saved")


# ============================================================================
# Chart 2 — Reduction bar chart
# ============================================================================

def chart2_reduction_bar(r: dict) -> None:
    reductions  = r["reductions"]
    levels_data = r["levels"]
    gates       = r["gates"]

    compare_levels = ["G2", "G3", "G4"]
    red_vals = [reductions[g] for g in compare_levels]

    def bar_color(v: float) -> str:
        if v >= 30.0: return "#2e7d32"
        if v >= 15.0: return "#f57f17"
        return "#c62828"

    colors = [bar_color(v) for v in red_vals]
    labels = [G_LABELS_SHORT[g] for g in compare_levels]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(bottom=0.18, top=0.88)

    x = np.arange(len(compare_levels))
    bars = ax.bar(x, red_vals, width=0.55, color=colors, edgecolor="black",
                  linewidth=0.8, alpha=0.88, zorder=4)

    for xi, (v, bar) in enumerate(zip(red_vals, bars)):
        ax.text(xi, v + 0.4, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=colors[xi])
        # Also print mean N_converge below label
        mean_n = levels_data[compare_levels[xi]]["mean_n_converge"]
        mean_g1 = levels_data["G1"]["mean_n_converge"]
        ax.text(xi, -1.8, f"N_conv={mean_n:.0f}\n(G₁={mean_g1:.0f})",
                ha="center", va="top", fontsize=8.5, color="#555")

    # 30% success gate line
    ax.axhline(30.0, color="#2e7d32", lw=2.0, ls="--", zorder=5,
               label="30% success gate")
    # 15% reference line
    ax.axhline(15.0, color="#f57f17", lw=1.2, ls=":", alpha=0.8, zorder=3,
               label="15% amber zone")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% reduction in mean N_converge vs G₁", fontsize=11)
    ax.set_ylim(-3, max(max(red_vals) * 1.25, 35))

    gate_str   = "PASS ✓" if gates["overall_pass"] else "FAIL ✗"
    gate_color = "#2e7d32" if gates["overall_pass"] else "#c62828"
    ax.text(
        0.99, 0.97,
        f"{gate_str}\nG₄/G₁: −{gates['g4_reduction_pct']:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold", color=gate_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=gate_color, linewidth=1.5, alpha=0.92),
    )

    ax.set_title(
        "Bridge B Phase B: Convergence Acceleration by Graph Enrichment Level\n"
        "% reduction in mean N_converge vs G₁ baseline. Green bars ≥ 30% success gate.",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left", framealpha=0.92)

    caption = (
        f"Baseline G₁ mean N_converge = {levels_data['G1']['mean_n_converge']:.0f} decisions. "
        f"G₂ uses ρ={r['rho']} correlated sources. G₃ adds entity resolution (e₀×0.85). "
        f"G₄ adds ThreatIndicator accumulation (threat_intel variance ×0.70)."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_b_reduction_bar", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] bridge_b_phase_b_reduction_bar.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_convergence(r)
    chart2_reduction_bar(r)
