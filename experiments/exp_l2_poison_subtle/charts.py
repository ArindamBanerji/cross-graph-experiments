"""
P4-F Charts

Chart 1 — exp_l2_poison_subtle_comparison:
  Bar chart: v4 promotion rate for conditions A, F, G.
  Power analysis reference lines (crude equivalents).
  Title: "Subtle Poisoning: Gate Detection vs Inflation Level"
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

RESULTS_FILE = _REPO_ROOT / "results" / "exp_l2_poison_subtle.json"

COND_COLORS = {"A": "#2e7d32", "F": "#e65100", "G": "#1565c0"}
COND_LABELS = {
    "A": "A\n(0%, control)",
    "F": "F\n(subtle +40pp)",
    "G": "G\n(subtle +10pp)",
}


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# =============================================================================
# Chart 1 — Comparison bar chart
# =============================================================================

def chart1_comparison(data: dict) -> None:
    conds   = data["conditions"]
    cond_keys = ["A", "F", "G"]

    v4_rates = [conds[c]["v4_promoted_rate"] * 100 for c in cond_keys]
    v2_rates = [conds[c]["v2_promoted_rate"] * 100 for c in cond_keys]

    x     = np.arange(len(cond_keys))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.35)

    # ---- Left panel: v4 promotion rate ----
    ax = axes[0]
    bars_v4 = ax.bar(x, v4_rates, width + 0.15,
                     color=[COND_COLORS[c] for c in cond_keys],
                     edgecolor="black", linewidth=0.8, alpha=0.88)

    for bar, val in zip(bars_v4, v4_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # Power analysis reference lines for F and G (crude attacker equivalents)
    # P4-POWER at N~100: 30pp→73%, 10pp→13%
    power_refs = {"F": 73.0, "G": 13.0}
    for ci, c in enumerate(cond_keys):
        if c in power_refs:
            ref = power_refs[c]
            ax.plot([x[ci] - 0.35, x[ci] + 0.35], [ref, ref],
                    color="black", lw=1.8, ls="--", alpha=0.6, zorder=5)
            ax.text(x[ci] + 0.37, ref, f"Crude ref\n{ref:.0f}%",
                    ha="left", va="center", fontsize=7.5, color="#555", alpha=0.85)

    # 5% threshold
    ax.axhline(5, color="#c62828", lw=1.5, ls=":", alpha=0.7,
               label="5% false positive ceiling")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in cond_keys], fontsize=10)
    ax.set_ylabel("v4 (attacker target) promotion rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("v4 Promotion Rate\n(Attacker target, quality=0.55)",
                 fontsize=11, fontweight="bold")

    # ---- Right panel: breakdown by gate ----
    ax2 = axes[1]
    gate_names = ["G1\n(superiority)", "G2\n(correctness)",
                  "G3\n(conservation)", "G4\n(variance)"]
    gate_colors = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a"]
    x2 = np.arange(len(gate_names))
    w2 = 0.22
    offsets = np.linspace(-(len(cond_keys)-1)*w2/2, (len(cond_keys)-1)*w2/2, len(cond_keys))

    for ci, c in enumerate(cond_keys):
        eff = [conds[c]["gate_effectiveness"][gi] * 100 for gi in range(4)]
        total_blocks = sum(conds[c]["gate_block_total"])
        if total_blocks == 0:
            eff = [0.0] * 4
        bars = ax2.bar(x2 + offsets[ci], eff, w2,
                       color=[COND_COLORS[c]] * 4,
                       edgecolor="black", linewidth=0.6, alpha=0.80,
                       label=f"Cond {c}")
        for bar, val in zip(bars, eff):
            if val > 5:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                         f"{val:.0f}%", ha="center", va="bottom", fontsize=7,
                         color=COND_COLORS[c])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(gate_names, fontsize=10)
    ax2.set_ylabel("Fraction of v4 blocking events (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("Gate Blocking Breakdown\n(% of all v4 block events per gate)",
                  fontsize=11, fontweight="bold")

    # ---- Figure-level elements ----
    fig.suptitle(
        "Subtle Poisoning: Gate Detection vs Inflation Level\n"
        "Subtle attacker inflates ONLY acceptance signal for v4 (weight=0.4). "
        "Other metrics (res_time, correct, overrides) are genuine.",
        fontsize=12, fontweight="bold",
    )

    cfg = data["config"]
    # Conservation note
    aqv = data.get("expected_aqv_f_g", 0.0)
    caption = (
        f"N={cfg['n_seeds']} seeds × {cfg['n_decisions']} decisions. "
        f"θ_min={cfg['theta_min']}, Δ_min={cfg['delta_min']}. "
        f"Dashed = P4-POWER crude-attacker reference (total quality inflation). "
        f"Expected α·q·V (F/G) = {aqv:.3f} (θ_min={cfg['theta_min']})."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8, color="#555", style="italic")

    # Annotation: why subtle fails
    axes[0].text(
        0.03, 0.97,
        "Subtle attacker cannot make\nv4 appear superior to v0:\n"
        "  E[R_v4_mixed] < E[R_v0]\n"
        "→ Gate 1 trivially blocks",
        transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1",
                  edgecolor="#f9a825", linewidth=1.2, alpha=0.94),
    )

    save_figure(fig, "exp_l2_poison_subtle_comparison", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] exp_l2_poison_subtle_comparison.png + .pdf saved")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data = load()
    chart1_comparison(data)
