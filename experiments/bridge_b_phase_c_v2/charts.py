"""
Bridge B Phase C v2 Charts

Chart 1 — bridge_b_phase_c_v2_dual_trajectory:
  Dual y-axes: mean error (left, decreasing) + mean noise (right, decreasing).
  X = decision number 0-5000.

Chart 2 — bridge_b_phase_c_v2_eta_eff:
  η_eff per window (bars) with noise level overlay (line).

Chart 3 — bridge_b_phase_c_v2_graph_growth:
  G(t) components over time: entities, IOCs, cross-links, total decisions.
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

RESULTS_FILE = _REPO_ROOT / "results" / "bridge_b_phase_c_v2.json"

G_COLORS = {
    "entities":    "#1565c0",   # blue
    "iocs":        "#c62828",   # red
    "crosslinks":  "#6a1b9a",   # purple
    "decisions":   "#2e7d32",   # green
}

WINDOW_SIZE = 500
N_WINDOWS   = 10


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================================
# Chart 1 — Dual trajectory: error + noise
# ============================================================================

def chart1_dual_trajectory(r: dict) -> None:
    ds        = 10   # downsampling factor
    error_ds  = np.array(r["error_traj_ds"])   # every 10th step
    noise_ds  = np.array(r["noise_traj_ds"])
    n_points  = len(error_ds)
    x         = np.arange(n_points) * ds        # actual decision numbers

    windows    = r["windows"]
    n_windows  = r["n_windows"]
    spear      = r["spearman"]
    gates      = r["gates"]
    noise_info = r["noise"]

    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, right=0.88)

    ax2 = ax1.twinx()

    # Shade 500-decision windows alternately
    for wi in range(n_windows):
        t0 = wi * WINDOW_SIZE
        t1 = (wi + 1) * WINDOW_SIZE
        ax1.axvspan(t0, t1, color=("#e3f2fd" if wi % 2 == 0 else "#fafafa"),
                    alpha=0.5, zorder=1)

    # Error trajectory (left axis) — colored by segment to show change
    ax1.plot(x, error_ds, color="#1565c0", lw=2.0, zorder=4, label="Mean error ‖μ_t − μ*‖")

    # Noise trajectory (right axis)
    ax2.plot(x, noise_ds, color="#e65100", lw=1.8, ls="--", zorder=5,
             label="Mean noise σ² (right axis)", alpha=0.85)

    # Window boundary lines
    for wi in range(1, n_windows):
        ax1.axvline(wi * WINDOW_SIZE, color="#aaa", lw=0.8, ls=":", zorder=3, alpha=0.7)

    # Mark window η_eff centers
    for wi in range(n_windows):
        w = windows[f"W{wi}"]
        mid_x  = (w["t_start"] + w["t_end"]) / 2
        eta    = w["eta_eff"]
        ax1.text(mid_x, 0.01, f"η={eta:.4f}",
                 ha="center", va="bottom", fontsize=7.5,
                 color="#1565c0", alpha=0.8, rotation=0)

    ax1.set_xlabel("Decision number", fontsize=12)
    ax1.set_ylabel("Mean L2 centroid displacement ‖μ_t − μ*‖", fontsize=11,
                   color="#1565c0")
    ax2.set_ylabel("Mean factor noise σ² (current_sigma.mean())", fontsize=11,
                   color="#e65100")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#e65100")
    ax1.set_xlim(-50, len(x) * ds + 50)

    # Spearman + verdict annotation
    overall = gates["overall_evidence"]
    ann_col = "#2e7d32" if overall else "#c62828"
    red_pct = noise_info["reduction_pct"]
    ax1.text(
        0.99, 0.97,
        f"ρ(window, η_eff)={spear['rho_window_eta']:.4f}  p={spear['p_window_eta']:.4f}\n"
        f"Noise reduction W0→W9: {red_pct:.1f}%",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=ann_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=ann_col, linewidth=1.5, alpha=0.92),
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9.5, loc="upper right", framealpha=0.92)

    ax1.set_title(
        "Bridge B Phase C v2: Endogenous Enrichment — Error and Noise Co-Evolution\n"
        "Full scoring (C=6, A=4). Graph state G(t) grows with each decision, "
        "reducing factor noise endogenously.",
        fontsize=11.5, fontweight="bold",
    )

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). "
        f"{r['n_seeds']} seeds, η={r['eta']}, e₀={r['e0']}. "
        f"P(entity)={r['graph_growth_params']['p_entity']}, "
        f"P(IOC)={r['graph_growth_params']['p_ioc']}. "
        "Enrichment rates are environment-specific estimates."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_v2_dual_trajectory", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] bridge_b_phase_c_v2_dual_trajectory.png + .pdf saved")


# ============================================================================
# Chart 2 — η_eff per window with noise overlay
# ============================================================================

def chart2_eta_eff(r: dict) -> None:
    windows   = r["windows"]
    n_windows = r["n_windows"]
    spear     = r["spearman"]
    gates     = r["gates"]

    w_labels  = [f"W{wi}\n{wi*WINDOW_SIZE}–{(wi+1)*WINDOW_SIZE}" for wi in range(n_windows)]
    eta_vals  = np.array([windows[f"W{wi}"]["eta_eff"]    for wi in range(n_windows)])
    r2_vals   = np.array([windows[f"W{wi}"]["eta_r2"]     for wi in range(n_windows)])
    noise_vals= np.array([windows[f"W{wi}"]["noise_mean"] for wi in range(n_windows)])
    ent_vals  = np.array([windows[f"W{wi}"]["g_entities"] for wi in range(n_windows)])
    x         = np.arange(n_windows)

    fig, ax1  = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(bottom=0.22, top=0.88, right=0.88)
    ax2 = ax1.twinx()

    # η_eff bars
    bar_colors = plt.cm.Blues(np.linspace(0.35, 0.85, n_windows))
    bars = ax1.bar(x, eta_vals, width=0.55, color=bar_colors,
                   edgecolor="black", linewidth=0.7, alpha=0.88, zorder=4,
                   label="η_eff (left)")

    # R² annotations
    for xi, (e, r2) in enumerate(zip(eta_vals, r2_vals)):
        ax1.text(xi, e + abs(e) * 0.05 + 1e-5, f"R²={r2:.2f}",
                 ha="center", va="bottom", fontsize=7.5, color="#555")

    # Noise line (right axis)
    ax2.plot(x, noise_vals, color="#e65100", lw=2.0, marker="o",
             markersize=6, zorder=5, label="Noise σ² (right)")

    # Entities as secondary context (normalized)
    ent_norm = ent_vals / max(ent_vals.max(), 1.0) * noise_vals.max()
    ax2.plot(x, ent_norm, color="#1565c0", lw=1.4, ls="--", marker="s",
             markersize=5, zorder=4, alpha=0.7, label="G(entities) normalized")

    # Trend line for η_eff
    if len(eta_vals) > 2:
        z = np.polyfit(x, eta_vals, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), color="#9e9e9e", lw=1.5, ls="-.", zorder=3,
                 alpha=0.7, label="η_eff linear trend")

    ax1.set_xticks(x)
    ax1.set_xticklabels(w_labels, fontsize=8.5)
    ax1.set_xlabel("Window", fontsize=12)
    ax1.set_ylabel("Effective learning rate η_eff", fontsize=11, color="#1565c0")
    ax2.set_ylabel("Mean factor noise σ²", fontsize=11, color="#e65100")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#e65100")

    # Spearman annotation
    overall   = gates["overall_evidence"]
    ann_col   = "#2e7d32" if overall else ("#f57f17" if gates.get("evidence_level", 0) >= 1 else "#c62828")
    spear_str = (f"ρ(window, η_eff)={spear['rho_window_eta']:.4f}  "
                 f"p={spear['p_window_eta']:.4f}  "
                 f"{'PASS ✓' if spear['spearman_pass'] else 'FAIL ✗'}")
    ax1.text(
        0.01, 0.97, spear_str,
        transform=ax1.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold", color=ann_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=ann_col, linewidth=1.5, alpha=0.92),
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="upper right", framealpha=0.92)

    ax1.set_title(
        "Bridge B Phase C v2: Effective Learning Rate vs Endogenous Graph Enrichment\n"
        "η_eff = 1 − exp(log-linear slope) per 500-decision window. "
        "R² shown above each bar.",
        fontsize=11.5, fontweight="bold",
    )

    caption = (
        f"soc_product_v50 (C={r['ontology']['C']}, A={r['ontology']['A']}, "
        f"d={r['ontology']['d']}). {r['n_seeds']} seeds. "
        f"If η_eff increases as noise decreases: simulation evidence consistent with γ>1. "
        "Enrichment rates are environment-specific estimates."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_v2_eta_eff", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] bridge_b_phase_c_v2_eta_eff.png + .pdf saved")


# ============================================================================
# Chart 3 — Graph state G(t) accumulation
# ============================================================================

def chart3_graph_growth(r: dict) -> None:
    ds            = 10
    g_entities    = np.array(r["g_entities_ds"])
    g_iocs        = np.array(r["g_iocs_ds"])
    g_crosslinks  = np.array(r["g_crosslinks_ds"])
    g_decisions   = np.array(r["g_decisions_ds"])
    n_points      = len(g_entities)
    x             = np.arange(n_points) * ds

    windows       = r["windows"]
    n_windows     = r["n_windows"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.35, bottom=0.10, top=0.92)

    plot_specs = [
        (axes[0, 0], g_entities,   "#1565c0", "G(unique_entities)",
         "Unique entities seen", "TravelRecord + Asset + Device nodes"),
        (axes[0, 1], g_iocs,       "#c62828", "G(threat_indicators)",
         "ThreatIndicators accumulated", "IOC accumulation → threat_intel noise ↓"),
        (axes[1, 0], g_crosslinks, "#6a1b9a", "G(cross_category_links)",
         "Cross-category links", "Entities shared across categories"),
        (axes[1, 1], g_decisions,  "#2e7d32", "G(total_decisions)",
         "Total decisions accumulated", "Drives all enrichment dimensions"),
    ]

    for ax, y, col, label, ylabel, subtitle in plot_specs:
        ax.plot(x, y, color=col, lw=2.0, zorder=4)
        ax.fill_between(x, y, alpha=0.12, color=col, zorder=2)

        # Window boundary lines
        for wi in range(1, n_windows):
            ax.axvline(wi * WINDOW_SIZE, color="#aaa", lw=0.8, ls=":", alpha=0.6, zorder=3)

        # Annotate final value
        ax.text(x[-1], float(y[-1]), f" {float(y[-1]):.0f}",
                va="center", ha="left", fontsize=9, color=col, fontweight="bold")

        # Slope annotation: linear fit
        if len(y) > 2 and float(y[-1]) > float(y[0]):
            rate = (float(y[-1]) - float(y[0])) / (x[-1] - x[0])
            ax.text(0.05, 0.92, f"Rate: {rate:.2f}/decision",
                    transform=ax.transAxes, fontsize=8.5, color=col)

        ax.set_xlabel("Decision number", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-50, x[-1] + 150)
        ax.set_title(f"{label}\n{subtitle}", fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=0)

    # Enrichment thresholds overlay on entities plot
    ax_ent = axes[0, 0]
    ax_ent.axhline(100.0, color="#1565c0", lw=1.2, ls="--", alpha=0.6,
                   label="100 entities → 50% reduction (j=0,1,5)")
    ax_ent.legend(fontsize=8, loc="upper left")

    ax_ioc = axes[0, 1]
    ax_ioc.axhline(50.0, color="#c62828", lw=1.2, ls="--", alpha=0.6,
                   label="50 IOCs → 50% reduction (j=2)")
    ax_ioc.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Bridge B Phase C v2: Graph State Accumulation Over 5000 Decisions\n"
        f"Endogenous flywheel — each decision adds to G(t), reducing factor noise. "
        f"N={r['n_seeds']} seeds, mean shown.",
        fontsize=12, fontweight="bold",
    )

    caption = (
        f"P(new entity)={r['graph_growth_params']['p_entity']}/decision, "
        f"P(new IOC)={r['graph_growth_params']['p_ioc']}/decision, "
        f"P(cross-link)={r['graph_growth_params']['p_crosslink']}/decision "
        f"[after {r['graph_growth_params']['crosslink_min']} total]. "
        "These rates are environment-specific estimates."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "bridge_b_phase_c_v2_graph_growth", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] bridge_b_phase_c_v2_graph_growth.png + .pdf saved")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    r = load()
    chart1_dual_trajectory(r)
    chart2_eta_eff(r)
    chart3_graph_growth(r)
