"""
Bridge B Phase C Charts

Chart 1 — bridge_b_phase_c_eta_eff_trajectory:
  η_eff per window (mean ± std across cells).
  Vertical dashed lines at G-level transitions.
  Annotations: G₁, G₂, G₃, G₄ regions.

Chart 2 — bridge_b_phase_c_error_trajectory:
  Mean error ‖μ_t − μ*‖ over all 5000 decisions (cross-cell mean).
  Vertical lines at transitions. Log scale y-axis.
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

RESULTS_FILE = _REPO_ROOT / "results" / "bridge_b_phase_c.json"

G_COLORS = {
    "G1": "#9e9e9e",   # gray
    "G2": "#1565c0",   # blue
    "G3": "#2e7d32",   # green
    "G4": "#6a1b9a",   # purple
}

G_LABELS_LONG = {
    "G1": "G₁ Single SIEM",
    "G2": "G₂ Two SIEMs (ρ=0.8)",
    "G3": "G₃ + Entity Resolution",
    "G4": "G₄ + ThreatIndicators",
}

# Window → G level (matching run.py)
WINDOW_G = ["G1", "G2", "G2", "G3", "G3", "G3", "G4", "G4", "G4", "G4"]
WINDOW_SIZE = 500

# G-level regions for shading: (g_level, first_window, last_window)
G_REGIONS = [
    ("G1", 0, 0),
    ("G2", 1, 2),
    ("G3", 3, 5),
    ("G4", 6, 9),
]

# Transition x-positions (decision numbers)
TRANSITIONS = [500, 1500, 3000]   # G1→G2, G2→G3, G3→G4
TRANSITION_LABELS = ["G₁→G₂", "G₂→G₃", "G₃→G₄"]


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — η_eff per window
# ============================================================================

def chart1_eta_eff_trajectory(r: dict) -> None:
    n_windows = r["n_windows"]
    window_g  = r["window_g_levels"]
    windows   = r["windows"]

    eta_means = np.array([windows[f"W{wi}"]["eta_eff_mean"] for wi in range(n_windows)])
    eta_stds  = np.array([windows[f"W{wi}"]["eta_eff_std"]  for wi in range(n_windows)])
    n_active  = np.array([windows[f"W{wi}"]["n_active_cells"] for wi in range(n_windows)])

    # x-axis: window centres (in decision units)
    x_centres = np.array([(wi + 0.5) * WINDOW_SIZE for wi in range(n_windows)])
    x_labels  = [f"W{wi}\n{wi*WINDOW_SIZE}–{(wi+1)*WINDOW_SIZE}" for wi in range(n_windows)]

    spear      = r["spearman"]
    g_summary  = r["g_level_summary"]
    transitions = r["transitions"]
    gates       = r["gates"]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(bottom=0.20, top=0.88)

    # Shade G-level regions
    region_alpha = 0.10
    for g_level, w_start, w_end in G_REGIONS:
        x0 = w_start * WINDOW_SIZE
        x1 = (w_end + 1) * WINDOW_SIZE
        col = G_COLORS[g_level]
        ax.axvspan(x0, x1, color=col, alpha=region_alpha, zorder=1)
        mid_x = (x0 + x1) / 2
        ax.text(mid_x, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.06,
                G_LABELS_LONG[g_level],
                ha="center", va="top", fontsize=9, color=col,
                fontweight="bold", alpha=0.9,
                transform=ax.get_xaxis_transform())

    # Transition lines
    for trans_x, trans_label in zip(TRANSITIONS, TRANSITION_LABELS):
        ax.axvline(trans_x, color="#555", lw=1.5, ls="--", alpha=0.7, zorder=3)
        ax.text(trans_x + 30, 0.98, trans_label,
                transform=ax.get_xaxis_transform(),
                fontsize=8, color="#555", rotation=90, va="top", alpha=0.8)

    # η_eff bars / points per window
    bar_w  = WINDOW_SIZE * 0.60
    colors = [G_COLORS[WINDOW_G[wi]] for wi in range(n_windows)]

    bars = ax.bar(x_centres, eta_means, width=bar_w,
                  color=colors, edgecolor="black", linewidth=0.7,
                  alpha=0.82, zorder=4)

    # Error bars (std)
    active_mask = n_active > 0
    ax.errorbar(
        x_centres[active_mask], eta_means[active_mask],
        yerr=eta_stds[active_mask],
        fmt="none", color="black", capsize=4, capthick=1.2, lw=1.2, zorder=5,
    )

    # Annotations: n_active per window
    for wi, (xc, m, n) in enumerate(zip(x_centres, eta_means, n_active)):
        if n > 0:
            ax.text(xc, m + eta_stds[wi] + 0.0002, f"n={n}",
                    ha="center", va="bottom", fontsize=7, color="#555")

    # G-level mean η_eff horizontal marks
    for g_level, region in [("G1",[0,0]), ("G2",[1,2]), ("G3",[3,5]), ("G4",[6,9])]:
        x0 = region[0] * WINDOW_SIZE
        x1 = (region[1] + 1) * WINDOW_SIZE
        g_eta = g_summary[g_level]["mean_eta_eff"]
        if not (g_eta != g_eta):   # not NaN
            ax.plot([x0, x1], [g_eta, g_eta],
                    color=G_COLORS[g_level], lw=2.0, ls="-", alpha=0.6, zorder=6)

    ax.set_xlabel("Decision number (window centre)", fontsize=12)
    ax.set_ylabel("Effective learning rate η_eff", fontsize=12)
    ax.set_xticks(x_centres)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlim(-WINDOW_SIZE * 0.3, 10.3 * WINDOW_SIZE)
    ymax = max(float((eta_means + eta_stds).max()) * 1.35, 0.01)
    ax.set_ylim(0, ymax)

    # Spearman annotation
    spear_pass  = gates["spearman_pass"]
    mono_pass   = gates["monotonic_pass"]
    overall     = gates["overall_evidence"]
    verdict_col = "#2e7d32" if overall else ("#f57f17" if (spear_pass or mono_pass) else "#c62828")
    rho_str     = f"{spear['rho']:.4f}" if spear["rho"] is not None else "N/A"
    p_str       = f"{spear['p_value']:.4f}" if spear["p_value"] is not None else "N/A"
    mono_str    = f"{r['transitions']['monotonic_count']}/3 transitions ↑"

    ax.text(
        0.99, 0.97,
        f"Spearman ρ={rho_str}  p={p_str}\n{mono_str}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=verdict_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=verdict_col, linewidth=1.5, alpha=0.92),
    )

    ax.set_title(
        "Bridge B Phase C: Effective Learning Rate Over Time (Graph Enrichment)\n"
        "η_eff = 1 − exp(log-linear slope of error within 500-decision window). "
        "Pre-convergence cells only.",
        fontsize=12, fontweight="bold",
    )

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). "
        f"{r['n_seeds']} seeds × {r['ontology']['C']}C × {r['ontology']['A']}A = "
        f"{r['n_seeds']*r['ontology']['C']*r['ontology']['A']} cells. "
        f"η={r['eta']}, ε={r['eps']}, e₀={r['e0_base']}, ρ={r['rho']}. "
        f"Window = {r['window_size']} decisions."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_eta_eff_trajectory", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] bridge_b_phase_c_eta_eff_trajectory.png + .pdf saved")


# ============================================================================
# Chart 2 — Full error trajectory over 5000 decisions
# ============================================================================

def chart2_error_trajectory(r: dict) -> None:
    # Reconstruct full error traj from per-window data
    windows = r["windows"]
    n_windows = r["n_windows"]
    window_size = r["window_size"]

    full_traj = []
    for wi in range(n_windows):
        full_traj.extend(windows[f"W{wi}"]["mean_error_traj"])
    full_traj = np.array(full_traj)   # shape (5000,)

    n_decisions = len(full_traj)
    x = np.arange(n_decisions)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88)

    # Shade G-level regions
    for g_level, w_start, w_end in G_REGIONS:
        x0 = w_start * WINDOW_SIZE
        x1 = (w_end + 1) * WINDOW_SIZE
        col = G_COLORS[g_level]
        ax.axvspan(x0, x1, color=col, alpha=0.08, zorder=1)

    # G-level labels (top)
    for g_level, w_start, w_end in G_REGIONS:
        x0 = w_start * WINDOW_SIZE
        x1 = (w_end + 1) * WINDOW_SIZE
        mid_x = (x0 + x1) / 2
        col = G_COLORS[g_level]
        ax.text(mid_x, 1.01, G_LABELS_LONG[g_level],
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5,
                color=col, fontweight="bold")

    # Transition lines
    for trans_x, trans_label in zip(TRANSITIONS, TRANSITION_LABELS):
        ax.axvline(trans_x, color="#555", lw=1.5, ls="--", alpha=0.7, zorder=4)
        ymin, ymax_cur = ax.get_ylim()
        ax.text(trans_x + 20, 0.04, trans_label,
                fontsize=8.5, color="#555", alpha=0.85)

    # Error trajectory — colored by G level
    prev_end = 0
    for g_level, w_start, w_end in G_REGIONS:
        t0 = w_start * WINDOW_SIZE
        t1 = (w_end + 1) * WINDOW_SIZE
        seg_x = x[t0:t1]
        seg_y = full_traj[t0:t1]
        col   = G_COLORS[g_level]
        ax.plot(seg_x, seg_y, color=col, lw=1.8, zorder=5,
                label=G_LABELS_LONG[g_level])

    # ε threshold
    ax.axhline(r["eps"], color="#e65100", lw=1.5, ls=":", zorder=6,
               label=f"ε = {r['eps']} (convergence threshold)")

    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("Mean max-component error  max‖μ_t − μ*‖", fontsize=11)
    ax.set_xlim(-50, n_decisions + 50)

    # Use log scale if range spans >2 orders of magnitude
    y_range = float(full_traj[:50].mean()) / max(float(full_traj[-50:].mean()), 1e-8)
    if y_range > 20:
        ax.set_yscale("log")
        ax.set_ylabel("Mean max-component error (log scale)", fontsize=11)

    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    gates      = r["gates"]
    verdict_col = "#2e7d32" if gates["overall_evidence"] else "#f57f17"
    ax.text(
        0.99, 0.97,
        r["verdict_short"],
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=verdict_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=verdict_col, linewidth=1.5, alpha=0.92),
    )

    ax.set_title(
        "Bridge B Phase C: Centroid Error Trajectory with Graph Enrichment\n"
        "Mean max-component error across all cells over 5000 decisions. "
        "Each color segment = active G level.",
        fontsize=12, fontweight="bold",
    )

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). "
        f"Timeline: G₁=0–500, G₂=500–1500, G₃=1500–3000, G₄=3000–5000. "
        f"η={r['eta']}, ε={r['eps']}, e₀={r['e0_base']}, ρ={r['rho']}. "
        f"{r['n_seeds']} seeds × {r['ontology']['C']*r['ontology']['A']} cells."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_error_trajectory", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] bridge_b_phase_c_error_trajectory.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_eta_eff_trajectory(r)
    chart2_error_trajectory(r)
