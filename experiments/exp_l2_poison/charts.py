"""
EXP-L2-POISON Charts

Chart 1 — exp_l2_poison_promotion_rates:
  Grouped bar: v4 (red) and v2 (green) promotion rate per condition.

Chart 2 — exp_l2_poison_gate_effectiveness:
  Stacked bar per condition: which gate blocked v4 promotion.

Chart 3 — exp_l2_poison_conservation_trajectory:
  Mean α·q·V over decisions for conditions A, C, D.
  Horizontal line at θ_min = 0.467.
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
import matplotlib.patches as mpatches
import numpy as np

from src.viz.bridge_common import save_figure

RESULTS_FILE = _REPO_ROOT / "results" / "exp_l2_poison.json"
THETA_MIN    = 0.467

COND_ORDER  = ["A", "B", "C", "D", "E"]
COND_LABELS = {
    "A": "A\n(0% poison)",
    "B": "B\n(10%)",
    "C": "C\n(20%)",
    "D": "D\n(40%)",
    "E": "E\n(20% tgt)",
}

GATE_COLORS = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a"]
GATE_LABELS = ["G1: Superiority", "G2: Correctness",
               "G3: Conservation", "G4: Variance"]


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# =============================================================================
# Chart 1 — Promotion rates
# =============================================================================

def chart1_promotion_rates(data: dict) -> None:
    conds  = data["conditions"]
    x      = np.arange(len(COND_ORDER))
    width  = 0.32

    v4_rates = [conds[c]["v4_promoted_rate"] * 100 for c in COND_ORDER]
    v2_rates = [conds[c]["v2_promoted_rate"] * 100 for c in COND_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(bottom=0.18, top=0.88)

    bars_v4 = ax.bar(x - width / 2, v4_rates, width, color="#c62828",
                     edgecolor="black", linewidth=0.7, alpha=0.88,
                     label="v4 (TARGET, quality=0.55)")
    bars_v2 = ax.bar(x + width / 2, v2_rates, width, color="#2e7d32",
                     edgecolor="black", linewidth=0.7, alpha=0.88,
                     label="v2 (BEST, quality=0.85)")

    # Value labels
    for bar in bars_v4:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9, color="#c62828")
    for bar in bars_v2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9, color="#2e7d32")

    # Success criteria lines
    ax.axhline(5,  color="#c62828", lw=1.2, ls="--", alpha=0.5,
               label="v4 gate: 5% (B/C)", zorder=1)
    ax.axhline(80, color="#2e7d32", lw=1.2, ls="--", alpha=0.5,
               label="v2 gate: 80% (A)", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=10)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Promotion rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    ax.set_title(
        "L2 Poisoning: Target Variant Promotion Rate\n"
        "v4 (attacker target, quality=0.55) vs v2 (best, quality=0.85). "
        "Four-condition gate with θ_min=0.467.",
        fontsize=11.5, fontweight="bold",
    )

    cfg = data["config"]
    caption = (
        f"N={cfg['n_seeds']} seeds × {cfg['n_decisions']} decisions. "
        f"Thompson Sampling (Beta posterior). "
        f"θ_min={THETA_MIN} (Gate 3 conservation floor). "
        "Dashed lines = success criteria."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "exp_l2_poison_promotion_rates", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] exp_l2_poison_promotion_rates.png + .pdf saved")


# =============================================================================
# Chart 2 — Gate effectiveness (stacked bar)
# =============================================================================

def chart2_gate_effectiveness(data: dict) -> None:
    conds  = data["conditions"]
    # Only poisoned conditions
    cond_show = ["B", "C", "D", "E"]
    x      = np.arange(len(cond_show))

    # gate_effectiveness is fraction of blocking events per gate
    gate_eff = np.array([
        conds[c]["gate_effectiveness"] for c in cond_show
    ])  # shape (4_conds, 4_gates)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(bottom=0.18, top=0.88)

    bottoms = np.zeros(len(cond_show))
    bars_list = []
    for gi in range(4):
        vals = gate_eff[:, gi]
        bars = ax.bar(x, vals * 100, bottom=bottoms * 100,
                      color=GATE_COLORS[gi], edgecolor="white",
                      linewidth=0.5, alpha=0.88, label=GATE_LABELS[gi],
                      width=0.55)
        bars_list.append(bars)
        # Annotate segments > 8%
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.08:
                ax.text(xi, (b + v / 2) * 100,
                        f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({conds[c]['label']})" for c in cond_show], fontsize=10)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Fraction of blocking events (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    ax.set_title(
        "Gate Effectiveness Against Adversarial Poisoning\n"
        "Fraction of gate-check events where each gate blocked v4 promotion.",
        fontsize=11.5, fontweight="bold",
    )

    cfg = data["config"]
    caption = (
        f"N={cfg['n_seeds']} seeds × {cfg['n_decisions']} decisions. "
        "Each bar = % of blocking events attributable to each gate. "
        "Bars sum to ≤100% (some checks blocked by multiple gates)."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "exp_l2_poison_gate_effectiveness", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] exp_l2_poison_gate_effectiveness.png + .pdf saved")


# =============================================================================
# Chart 3 — Conservation trajectory
# =============================================================================

def chart3_conservation_trajectory(data: dict) -> None:
    conds    = data["conditions"]
    n_dec    = data["config"]["n_decisions"]
    ds       = max(n_dec // 100, 1)

    show_conds = ["A", "C", "D"]
    colors     = {"A": "#2e7d32", "C": "#e65100", "D": "#c62828"}
    labels     = {"A": "A (0% poison)", "C": "C (20% poison)", "D": "D (40% poison)"}

    # Downsampled x-axis
    x = np.arange(len(conds["A"]["aqv_mean_traj"])) * ds

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.15, top=0.88, right=0.97)

    for cond_name in show_conds:
        traj = np.array(conds[cond_name]["aqv_mean_traj"])
        ax.plot(x, traj, color=colors[cond_name], lw=2.2,
                label=labels[cond_name], zorder=4)

    # θ_min floor
    ax.axhline(THETA_MIN, color="#555", lw=1.8, ls="--", zorder=5,
               label=f"θ_min = {THETA_MIN} (Gate 3 floor)")

    # Shade below θ_min
    ax.fill_between([x[0], x[-1]], 0, THETA_MIN,
                    color="#ffcccc", alpha=0.25, zorder=1,
                    label="Danger zone (α·q·V < θ_min)")

    # Mark first breach for C and D
    for cond_name in ["C", "D"]:
        tc = conds[cond_name]["mean_t_conservation"]
        if tc is not None:
            ax.axvline(tc, color=colors[cond_name], lw=1.2, ls=":",
                       alpha=0.8, zorder=3)
            ax.text(tc + 5, THETA_MIN * 1.03,
                    f"G3 triggers\n(Cond {cond_name}, t≈{tc:.0f})",
                    fontsize=8, color=colors[cond_name], va="bottom")

    # Early-warning annotations
    for cond_name in ["C", "D"]:
        ta = conds[cond_name]["mean_t_accuracy_drop"]
        if ta is not None:
            ax.axvline(ta, color=colors[cond_name], lw=1.0, ls="-.",
                       alpha=0.6, zorder=3)
            ax.text(ta + 5, THETA_MIN * 0.75,
                    f"Acc drop\n(Cond {cond_name}, t≈{ta:.0f})",
                    fontsize=8, color=colors[cond_name], va="top")

    ax.set_xlabel("Decision number", fontsize=12)
    ax.set_ylabel("α_t · q_t · V_t  (conservation signal)", fontsize=11)
    ax.set_xlim(0, n_dec)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    ax.set_title(
        "Conservation Law Signal Under Adversarial Poisoning\n"
        "α·q·V = override_rate × override_quality × verified_per_day. "
        f"Gate 3 blocks promotion when signal < θ_min={THETA_MIN}.",
        fontsize=11.5, fontweight="bold",
    )

    # Early-warning summary annotation
    cl_rates = [conds[c]["cl_early_warning_rate"] for c in ["B", "C", "D"]
                if conds[c]["cl_early_warning_rate"] is not None]
    mean_cl  = float(np.mean(cl_rates)) if cl_rates else 0.0

    ax.text(
        0.02, 0.96,
        f"CL early warning rate (B/C/D mean): {mean_cl*100:.0f}%\n"
        f"Conservation law fires before accuracy degrades\nin majority of poisoned seeds.",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        color="#c62828" if mean_cl > 0.60 else "#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#aaa", linewidth=1.2, alpha=0.93),
    )

    cfg = data["config"]
    caption = (
        f"Mean α·q·V trajectory across {cfg['n_seeds']} seeds. "
        "Malicious analysts inflate α but zero q (override_quality=0) → α·q product drops. "
        "Dotted vertical = mean G3 trigger; dash-dot = mean accuracy drop."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5,
             color="#555", style="italic")

    save_figure(fig, "exp_l2_poison_conservation_trajectory",
                output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 3] exp_l2_poison_conservation_trajectory.png + .pdf saved")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data = load()
    chart1_promotion_rates(data)
    chart2_gate_effectiveness(data)
    chart3_conservation_trajectory(data)
