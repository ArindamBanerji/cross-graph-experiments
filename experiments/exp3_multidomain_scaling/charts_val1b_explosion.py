"""
VAL-1B: Consolidated publication figure — Norm Explosion.

Reads:
  experiments/exp3_multidomain_scaling/norm_tracking_summary.json
  experiments/exp3_multidomain_scaling/norm_tracking.csv

Outputs (paper_figures/):
  val_1b_norm_explosion.{pdf,png}
"""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.viz.bridge_common import COLORS, VIZ_DEFAULTS, setup_axes, save_figure

EXP_DIR    = Path(__file__).resolve().parent
PAPER_FIGS = ROOT / "paper_figures"

# Colors chosen from bridge_common palette
COLOR_NO_LN  = COLORS["gt_noise_30"]   # "#DC2626" red   — without LayerNorm
COLOR_WITH_LN = COLORS["mi_static"]    # "#059669" green — with LayerNorm
COLOR_THRESH  = "#475569"              # slate-600 — 2× threshold dashed line


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_summary() -> dict:
    with open(EXP_DIR / "norm_tracking_summary.json") as fh:
        return json.load(fh)


def _load_seed_sweep_means() -> dict[int, list[float]]:
    """
    Return {sweep: [mean_norm_per_seed]} where each element is the
    domain-averaged mean_norm for one seed.  Gives 10 values per sweep
    for computing ±1σ across seeds.
    """
    # Accumulate: {(seed, sweep): [mean_norm per domain]}
    acc: dict[tuple, list[float]] = defaultdict(list)
    with open(EXP_DIR / "norm_tracking.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (int(row["seed"]), int(row["sweep"]))
            acc[key].append(float(row["mean_norm"]))
    # Average across domains → one value per (seed, sweep)
    per_sweep: dict[int, list[float]] = defaultdict(list)
    for (seed, sweep), vals in acc.items():
        per_sweep[sweep].append(float(np.mean(vals)))
    return dict(per_sweep)


# ---------------------------------------------------------------------------
# Build main figure
# ---------------------------------------------------------------------------

def make_explosion_figure(summary: dict,
                          seed_sweep_means: dict[int, list[float]]) -> None:

    sweeps     = summary["sweeps"]                  # [0,1,2,3,4,5]
    mean_norms = summary["overall_mean_norm"]        # normalized; sweep0 = 1.0
    dom_data   = summary["per_domain_final_sweep"]

    # ±1σ across seeds at each sweep
    sigma = [float(np.std(seed_sweep_means[s])) for s in sweeps]
    lo    = [max(1e-3, mean_norms[s] - sigma[s]) for s in sweeps]
    hi    = [mean_norms[s] + sigma[s]            for s in sweeps]

    # ------------------------------------------------------------------
    # Figure layout — wide enough for inset comfortably
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.18, top=0.88, left=0.10, right=0.97)

    # ------------------------------------------------------------------
    # "Without LayerNorm" line + ±1σ band
    # ------------------------------------------------------------------
    ax.semilogy(sweeps, mean_norms,
                color=COLOR_NO_LN, linewidth=2.5,
                marker="o", markersize=6, zorder=4,
                label="Without LayerNorm")
    ax.fill_between(sweeps, lo, hi,
                    color=COLOR_NO_LN, alpha=0.15, zorder=3)

    # ------------------------------------------------------------------
    # "With LayerNorm" flat reference line at 1.0
    # ------------------------------------------------------------------
    layernorm_vals = [1.0] * len(sweeps)
    ax.semilogy(sweeps, layernorm_vals,
                color=COLOR_WITH_LN, linewidth=2.5,
                marker="^", markersize=6, zorder=4,
                label="With LayerNorm")

    # ------------------------------------------------------------------
    # 2× threshold dashed line
    # ------------------------------------------------------------------
    ax.axhline(y=2.0, color=COLOR_THRESH, linewidth=1.2,
               linestyle="--", zorder=2)
    ax.text(5.05, 2.0, "2× threshold",
            va="center", ha="left",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color=COLOR_THRESH)

    # ------------------------------------------------------------------
    # Annotation: sweep 1
    # ------------------------------------------------------------------
    s1_y = mean_norms[1]   # 1.18 — line value at sweep 1
    ax.annotate(
        "~2× after 1 sweep",
        xy=(1, s1_y),
        xytext=(1.35, 4.5),
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color=COLOR_NO_LN,
        arrowprops=dict(arrowstyle="->", color=COLOR_NO_LN,
                        lw=1.0, connectionstyle="arc3,rad=0.2"),
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Annotation: sweep 5
    # ------------------------------------------------------------------
    s5_y = mean_norms[5]   # 1,243,653
    ax.annotate(
        "~2.9 million×",
        xy=(5, s5_y),
        xytext=(3.55, s5_y * 0.12),
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color=COLOR_NO_LN,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_NO_LN,
                        lw=1.0, connectionstyle="arc3,rad=-0.15"),
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------
    ax.set_xticks(sweeps)
    ax.set_xlim(-0.3, 5.5)       # leave room for threshold label
    ax.set_ylim(0.5, 5e7)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: _fmt_growth(v))
    )
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"], loc="upper right",
              bbox_to_anchor=(0.99, 0.99))

    setup_axes(ax,
               title="Eq. 13 Norm Explosion: Why LayerNorm Is Non-Negotiable",
               xlabel="Enrichment Sweeps",
               ylabel="Norm Growth (×)")

    # ------------------------------------------------------------------
    # Inset: per-domain growth_max horizontal bar chart
    # ------------------------------------------------------------------
    # Coordinates in axes fraction: [left, bottom, width, height]
    ax_in = ax.inset_axes([0.04, 0.40, 0.29, 0.48])
    _draw_inset(ax_in, dom_data)

    # ------------------------------------------------------------------
    # Caption below figure
    # ------------------------------------------------------------------
    caption = (
        "Without normalization: geometric ~40× per sweep. "
        "With LayerNorm: bounded at ~1×. Production requirement."
    )
    fig.text(0.5, 0.04, caption,
             ha="center", va="top",
             fontsize=VIZ_DEFAULTS["annotation_fontsize"],
             color="#475569",
             style="italic")

    save_figure(fig, "val_1b_norm_explosion", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Inset helper
# ---------------------------------------------------------------------------

_INSET_DOMAIN_LABELS = {
    "security":        "Security",
    "decision_history":"Dec. Hist.",
    "threat_intel":    "Threat Intel",
    "network_flow":    "Net. Flow",
    "asset_inventory": "Asset Inv.",
    "user_behavior":   "User Beh.",
}


def _draw_inset(ax_in: plt.Axes, dom_data: dict) -> None:
    domains = list(dom_data.keys())
    growths = [dom_data[d]["growth_max"] for d in domains]

    # Sort ascending so smallest is at top of horizontal bars
    order   = sorted(range(len(domains)), key=lambda i: growths[i])
    domains = [domains[i] for i in order]
    growths = [growths[i] for i in order]
    labels  = [_INSET_DOMAIN_LABELS.get(d, d) for d in domains]

    # Use numeric y-positions to avoid categorical/numeric axis conflict
    y_pos  = list(range(len(labels)))
    colors = COLORS["category_colors"]

    bars = ax_in.barh(y_pos, growths,
                      color=[colors[i % len(colors)] for i in range(len(y_pos))],
                      height=0.6)

    ax_in.set_yticks(y_pos)
    ax_in.set_yticklabels(labels, fontsize=5.5)
    ax_in.set_xscale("log")
    ax_in.set_xlim(left=1e4)
    ax_in.set_title("Per-domain growth\n(sweep 5)",
                    fontsize=6.5, pad=3)
    ax_in.tick_params(labelsize=5.5)
    ax_in.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: _fmt_growth(v))
    )
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.tick_params(axis="x", labelrotation=30)

    # Light value labels at bar ends
    for bar, g in zip(bars, growths):
        ax_in.text(g * 1.05, bar.get_y() + bar.get_height() / 2,
                   _fmt_growth(g),
                   va="center", ha="left", fontsize=5.0)


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _fmt_growth(v: float) -> str:
    if v >= 1e6:
        return f"{v/1e6:.1f}M×"
    if v >= 1e3:
        return f"{v/1e3:.0f}K×"
    if v >= 2:
        return f"{v:.0f}×"
    return f"{v:.1f}×"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading summary ...")
    summary = _load_summary()

    print("Loading CSV for per-seed std band ...")
    seed_sweep_means = _load_seed_sweep_means()

    print("Generating val_1b_norm_explosion ...")
    make_explosion_figure(summary, seed_sweep_means)
    print("Done.")
