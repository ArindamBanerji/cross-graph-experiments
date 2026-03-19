"""
Bridge B Phase C v3 Charts

Chart 1 — bridge_b_phase_c_v3_error_trajectory:
  Full 4500-decision error trajectory. Vertical red lines at shift points.
  Shows: convergence → flat → SHIFT → re-convergence → SHIFT → re-convergence.

Chart 2 — bridge_b_phase_c_v3_reconvergence_comparison:
  Three overlaid re-convergence curves from their respective shift point (t=0 each).
  Curve 1: initial convergence (sparse graph).
  Curve 2: re-convergence after shift 1 (medium graph).
  Curve 3: re-convergence after shift 2 (richest graph).
  If curve 3 drops fastest: visual γ>1 evidence.

Chart 3 — bridge_b_phase_c_v3_graph_noise:
  Dual axis: G(entities) and G(IOCs) (left) + noise level (right).
  Vertical lines at shift points.
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

RESULTS_FILE = _REPO_ROOT / "results" / "bridge_b_phase_c_v3.json"

# Colors for the three phases
PHASE_COLORS = ["#9e9e9e", "#1565c0", "#2e7d32"]
PHASE_LABELS = [
    "Phase 1: Initial conv. (sparse graph)",
    "Phase 2: Re-conv. after Shift 1 (medium graph)",
    "Phase 3: Re-conv. after Shift 2 (rich graph)",
]
SHIFT_COLORS = ["#e65100", "#c62828"]
DS = 5   # downsampling factor


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — Full error trajectory
# ============================================================================

def chart1_error_trajectory(r: dict) -> None:
    error_ds = np.array(r["error_traj_ds"])
    n_pts    = len(error_ds)
    x        = np.arange(n_pts) * DS

    phase_len = r["phase_len"]
    delta     = r["delta_shift"]
    eps       = r["eps_reconverge"]
    stats     = r["statistics"]
    nc        = r["n_converge"]
    acc       = r["accuracy_per_phase"]

    shift_ts  = [phase_len, 2 * phase_len]
    shift_colors = SHIFT_COLORS

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88)

    # Shade phases
    x_ends = [0, phase_len, 2 * phase_len, 3 * phase_len]
    for ph in range(3):
        ax.axvspan(x_ends[ph], x_ends[ph + 1], color=PHASE_COLORS[ph],
                   alpha=0.07, zorder=1)
        mid = (x_ends[ph] + x_ends[ph + 1]) / 2
        ax.text(mid, 1.01, f"Phase {ph+1}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5,
                color=PHASE_COLORS[ph], fontweight="bold")

    # Error curve, colored per phase
    for ph in range(3):
        t0 = x_ends[ph] // DS
        t1 = x_ends[ph + 1] // DS
        ax.plot(x[t0:t1 + 1], error_ds[t0:t1 + 1],
                color=PHASE_COLORS[ph], lw=2.0, zorder=4,
                label=PHASE_LABELS[ph])

    # Shift lines
    for i, (ts, sc) in enumerate(zip(shift_ts, shift_colors)):
        ax.axvline(ts, color=sc, lw=2.0, ls="--", zorder=5, alpha=0.85)
        ax.text(ts + 30, 0.95, f"Shift {i+1}\nδ={delta}",
                transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=sc, va="top", fontweight="bold")

    # EPS_RECONVERGE line
    ax.axhline(eps, color="#555", lw=1.2, ls=":", zorder=3, alpha=0.7,
               label=f"ε_reconverge = {eps}")

    # N_converge annotations per phase
    y_nc = eps * 1.02
    for ph, (nc_m, nc_frac, ph_col) in enumerate([
        (nc["n_init_mean"], nc["n_init_frac"], PHASE_COLORS[0]),
        (nc["n_rc1_mean"],  nc["n_rc1_frac"],  PHASE_COLORS[1]),
        (nc["n_rc2_mean"],  nc["n_rc2_frac"],  PHASE_COLORS[2]),
    ]):
        t_abs = x_ends[ph] + nc_m
        if nc_frac > 0.3 and t_abs < x_ends[ph + 1]:
            ax.axvline(t_abs, color=ph_col, lw=1.0, ls=":", alpha=0.6, zorder=3)
            ax.text(t_abs + 20, y_nc, f"N={nc_m:.0f}\n({nc_frac:.0%})",
                    fontsize=7.5, color=ph_col, va="bottom")

    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("Mean L2 centroid displacement ‖μ_t − μ*‖", fontsize=11)
    ax.set_xlim(-50, r["n_total"] + 50)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)

    # Verdict annotation
    ev   = stats["evidence_count"]
    col  = "#2e7d32" if ev >= 3 else ("#f57f17" if ev >= 2 else "#c62828")
    rc_r = r["acceleration_ratios"]["rc1_rc2"]
    ax.text(
        0.01, 0.97,
        f"N_rc1/N_rc2={rc_r:.3f}  {'↑ faster ✓' if rc_r > 1 else '↓ not faster'}\n"
        f"t-test p={stats['paired_ttest_p']:.4f}  "
        f"Spearman ρ={stats['spearman_rho']:.4f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, fontweight="bold", color=col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=col, linewidth=1.5, alpha=0.92),
    )

    ax.set_title(
        "Bridge B Phase C v3: Re-Convergence After Distribution Shifts (Endogenous Enrichment)\n"
        f"Three phases × {r['phase_len']} decisions. Shifts of δ={delta} per unit vector at "
        f"t={phase_len} and t={2*phase_len}. Vertical dotted = N_converge.",
        fontsize=11.5, fontweight="bold",
    )

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). "
        f"{r['n_seeds']} seeds, η={r['eta']}, e₀={r['e0']}. "
        f"Enrichment: entity_denom={r['graph_growth_params']['entity_denom']:.0f}, "
        f"ioc_denom={r['graph_growth_params']['ioc_denom']:.0f} (no cap). "
        "Rates are environment-specific estimates."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_v3_error_trajectory", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] bridge_b_phase_c_v3_error_trajectory.png + .pdf saved")


# ============================================================================
# Chart 2 — Overlaid re-convergence curves (all starting from t=0)
# ============================================================================

def chart2_reconvergence_comparison(r: dict) -> None:
    phase_error = {
        ph: np.array(r["phase_error_ds"][f"phase{ph+1}"])
        for ph in range(3)
    }
    n_pts_ph = [len(phase_error[ph]) for ph in range(3)]
    x_ph     = [np.arange(n_pts_ph[ph]) * DS for ph in range(3)]

    phase_len = r["phase_len"]
    eps       = r["eps_reconverge"]
    stats     = r["statistics"]
    nc        = r["n_converge"]
    t_eval    = r["t_eval"]
    err_T     = r["error_at_T_eval"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.38)

    # ---- Left: raw (non-normalized) trajectories ----
    ax = axes[0]

    for ph in range(3):
        label = (f"Phase {ph+1}: {PHASE_LABELS[ph].split(':')[1].strip()}"
                 if ':' in PHASE_LABELS[ph] else PHASE_LABELS[ph])
        ax.plot(x_ph[ph], phase_error[ph],
                color=PHASE_COLORS[ph], lw=2.2 if ph == 2 else 1.7,
                zorder=5 - ph, label=PHASE_LABELS[ph], alpha=0.9)

    # N_converge markers
    for ph, nc_m, nc_frac in [
        (0, nc["n_init_mean"], nc["n_init_frac"]),
        (1, nc["n_rc1_mean"],  nc["n_rc1_frac"]),
        (2, nc["n_rc2_mean"],  nc["n_rc2_frac"]),
    ]:
        if nc_frac > 0.3 and nc_m < phase_len:
            ax.axvline(nc_m, color=PHASE_COLORS[ph], lw=1.2, ls=":", alpha=0.7, zorder=3)
            ax.text(nc_m + 20, eps * 1.05, f"N={nc_m:.0f}",
                    fontsize=7.5, color=PHASE_COLORS[ph])

    # T_eval line
    ax.axvline(t_eval, color="#555", lw=1.2, ls="--", alpha=0.6, zorder=2)
    ax.text(t_eval + 15, 0.97, f"T={t_eval}", transform=ax.get_xaxis_transform(),
            fontsize=8, color="#555", va="top")

    ax.axhline(eps, color="#555", lw=1.0, ls=":", alpha=0.6, zorder=2,
               label=f"ε = {eps}")

    ax.set_xlabel("Decisions since phase start", fontsize=11)
    ax.set_ylabel("Mean L2 error ‖μ_t − μ*‖", fontsize=11)
    ax.set_xlim(-20, phase_len + 20)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.set_title("Re-Convergence Curves\n(from phase start, raw L2 error)",
                 fontsize=11, fontweight="bold")

    # ---- Right: error at T_eval bar chart ----
    ax2 = axes[1]
    ph_labels = [f"Phase {i+1}" for i in range(3)]
    err_T_vals = [err_T["phase1"], err_T["phase2"], err_T["phase3"]]

    bars = ax2.bar(range(3), err_T_vals, width=0.55,
                   color=PHASE_COLORS, edgecolor="black", linewidth=0.8,
                   alpha=0.88, zorder=4)

    for i, (ph_l, v) in enumerate(zip(ph_labels, err_T_vals)):
        ax2.text(i, v + 0.001, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=PHASE_COLORS[i])

    # Delta annotations between consecutive phases
    for i in range(2):
        delta_v = err_T_vals[i + 1] - err_T_vals[i]
        col = "#2e7d32" if delta_v < 0 else "#c62828"
        ax2.text(i + 0.5, max(err_T_vals[i], err_T_vals[i + 1]) + 0.005,
                 f"Δ={delta_v:+.4f}",
                 ha="center", fontsize=8.5, color=col, style="italic")

    ax2.axhline(eps, color="#555", lw=1.0, ls=":", alpha=0.6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(ph_labels, fontsize=11)
    ax2.set_ylabel(f"Mean L2 error at T={t_eval} decisions", fontsize=11)
    ax2.set_ylim(0, max(err_T_vals) * 1.22)
    ax2.set_title(f"Error at T={t_eval} Decisions After Phase Start\n"
                  f"Lower = faster re-convergence = γ>1 evidence",
                  fontsize=11, fontweight="bold")

    # Overall verdict box
    ev   = stats["evidence_count"]
    col  = "#2e7d32" if ev >= 3 else ("#f57f17" if ev >= 2 else "#c62828")
    rc_r = r["acceleration_ratios"]["rc1_rc2"]
    fig.text(
        0.5, 0.06,
        f"N_rc1/N_rc2 = {rc_r:.3f}  |  "
        f"Paired t-test p = {stats['paired_ttest_p']:.4f}  |  "
        f"Phase 3 faster in {stats['rc_order_frac']:.0%} of seeds  |  "
        f"Evidence level: {ev}/3",
        ha="center", fontsize=10, fontweight="bold", color=col,
    )

    fig.suptitle(
        "Bridge B Phase C v3: Re-Convergence Speed vs Graph Maturity\n"
        "Same shift magnitude δ=0.10 each time — richer graph should produce faster recovery.",
        fontsize=12, fontweight="bold",
    )

    save_figure(fig, "bridge_b_phase_c_v3_reconvergence_comparison",
                output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] bridge_b_phase_c_v3_reconvergence_comparison.png + .pdf saved")


# ============================================================================
# Chart 3 — Graph state + noise co-evolution
# ============================================================================

def chart3_graph_noise(r: dict) -> None:
    g_ent_ds   = np.array(r["g_entities_ds"])
    g_ioc_ds   = np.array(r["g_iocs_ds"])
    noise_ds   = np.array(r["noise_traj_ds"])
    n_pts      = len(g_ent_ds)
    x          = np.arange(n_pts) * DS

    phase_len  = r["phase_len"]
    delta      = r["delta_shift"]
    ns         = r["noise_at_shifts"]
    stats      = r["statistics"]

    fig, ax1   = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, right=0.87)
    ax2 = ax1.twinx()

    # Phase shading
    x_ends = [0, phase_len, 2 * phase_len, 3 * phase_len]
    for ph in range(3):
        ax1.axvspan(x_ends[ph], x_ends[ph + 1], color=PHASE_COLORS[ph],
                    alpha=0.06, zorder=1)

    # G(entities) and G(IOCs) on left axis
    ax1.plot(x, g_ent_ds, color="#1565c0", lw=2.0, zorder=4, label="G(entities)")
    ax1.plot(x, g_ioc_ds, color="#c62828", lw=1.8, ls="--", zorder=4, label="G(IOCs)")

    # Enrichment threshold lines (50% reduction points)
    ax1.axhline(r["graph_growth_params"]["entity_denom"], color="#1565c0",
                lw=1.0, ls=":", alpha=0.5, label=f"entity_denom={r['graph_growth_params']['entity_denom']:.0f}")
    ax1.axhline(r["graph_growth_params"]["ioc_denom"], color="#c62828",
                lw=1.0, ls=":", alpha=0.5, label=f"ioc_denom={r['graph_growth_params']['ioc_denom']:.0f}")

    # Noise on right axis
    ax2.plot(x, noise_ds, color="#6a1b9a", lw=2.0, zorder=5, label="Noise σ² (right)")

    # Shift lines
    for i, ts in enumerate([phase_len, 2 * phase_len]):
        ax1.axvline(ts, color=SHIFT_COLORS[i], lw=2.0, ls="--", alpha=0.8, zorder=5)
        ax1.text(ts + 30, 0.97, f"Shift {i+1}\nδ={delta}",
                 transform=ax1.get_xaxis_transform(),
                 fontsize=8.5, color=SHIFT_COLORS[i], va="top", fontweight="bold")

    # Noise annotations at shift points
    ax2.scatter([phase_len, 2 * phase_len],
                [ns["shift1"], ns["shift2"]],
                color="#6a1b9a", s=80, zorder=6, marker="D")
    ax2.text(phase_len + 30, ns["shift1"] + 0.0005,
             f"σ²={ns['shift1']:.5f}", fontsize=8, color="#6a1b9a")
    ax2.text(2 * phase_len + 30, ns["shift2"] + 0.0005,
             f"σ²={ns['shift2']:.5f}\n({ns['reduction_pct']:.1f}% below Shift 1)",
             fontsize=8, color="#6a1b9a")

    ax1.set_xlabel("Decision number", fontsize=12)
    ax1.set_ylabel("G(t): entity and IOC counts", fontsize=11, color="#1565c0")
    ax2.set_ylabel("Mean factor noise σ²", fontsize=11, color="#6a1b9a")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#6a1b9a")

    # Legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               fontsize=9, loc="upper left", framealpha=0.92)

    ev  = stats["evidence_count"]
    col = "#2e7d32" if ev >= 3 else ("#f57f17" if ev >= 2 else "#c62828")
    ax1.text(
        0.99, 0.97,
        f"Noise reduction shift1→shift2: {ns['reduction_pct']:.1f}%",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=col, linewidth=1.5, alpha=0.92),
    )

    ax1.set_title(
        "Bridge B Phase C v3: Graph Enrichment and Factor Noise Co-Evolution\n"
        f"Entity and IOC accumulation reduce factor noise endogenously. "
        f"N={r['n_seeds']} seeds mean. Shift points marked.",
        fontsize=11.5, fontweight="bold",
    )

    caption = (
        f"P(entity)={r['graph_growth_params']['p_entity']}/decision, "
        f"P(IOC)={r['graph_growth_params']['p_ioc']}/decision (environment-specific estimates). "
        f"entity_denom={r['graph_growth_params']['entity_denom']:.0f}, "
        f"ioc_denom={r['graph_growth_params']['ioc_denom']:.0f}, no cap → "
        "progressive noise reduction across all three phases."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_v3_graph_noise", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] bridge_b_phase_c_v3_graph_noise.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_error_trajectory(r)
    chart2_reconvergence_comparison(r)
    chart3_graph_noise(r)
