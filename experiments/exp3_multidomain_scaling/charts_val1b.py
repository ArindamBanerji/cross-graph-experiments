"""
VAL-1B: Norm tracking figures.

Reads:
  experiments/exp3_multidomain_scaling/norm_tracking_summary.json
  experiments/exp3_multidomain_scaling/norm_tracking.csv

Outputs (paper_figures/):
  val1b_norm_growth.{pdf,png}   -- log-scale norm trajectory over sweeps
  val1b_per_domain.{pdf,png}    -- per-domain final growth_max bar chart
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.viz.bridge_common import COLORS, VIZ_DEFAULTS, setup_axes, save_figure

EXP_DIR     = Path(__file__).resolve().parent
PAPER_FIGS  = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _load_summary() -> dict:
    with open(EXP_DIR / "norm_tracking_summary.json") as fh:
        return json.load(fh)


def _load_csv_mean_per_sweep() -> dict[int, dict[str, float]]:
    """Return {sweep: {domain: mean_norm}} averaged across seeds."""
    from collections import defaultdict
    rows_by_key: dict[tuple, list[float]] = defaultdict(list)
    with open(EXP_DIR / "norm_tracking.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (int(row["sweep"]), row["domain"])
            rows_by_key[key].append(float(row["mean_norm"]))
    result: dict[int, dict[str, float]] = {}
    for (sweep, domain), vals in rows_by_key.items():
        result.setdefault(sweep, {})[domain] = float(np.mean(vals))
    return result


# ---------------------------------------------------------------------------
# Chart 1: Norm growth over sweeps (log scale)
# ---------------------------------------------------------------------------
def chart_norm_growth(summary: dict) -> None:
    sweeps       = summary["sweeps"]
    mean_norms   = summary["overall_mean_norm"]
    max_norms    = summary["overall_max_norm"]
    final_growth = summary["final_growth_factor_max"]

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    ax.semilogy(sweeps, mean_norms,
                color=COLORS["gt_noise_5"], linewidth=2.0,
                marker="o", markersize=5, label="Mean norm")
    ax.semilogy(sweeps, max_norms,
                color=COLORS["gt_noise_30"], linewidth=2.0,
                marker="s", markersize=5, linestyle="--", label="Max norm")

    # Annotate final growth factor
    ax.annotate(
        f"{final_growth / 1e6:.1f}M× growth",
        xy=(sweeps[-1], max_norms[-1]),
        xytext=(-80, -30),
        textcoords="offset points",
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color=COLORS["gt_noise_30"],
        arrowprops=dict(arrowstyle="->", color=COLORS["gt_noise_30"], lw=1.0),
    )

    ax.set_xticks(sweeps)
    ax.set_xlim(-0.2, max(sweeps) + 0.2)
    ax.axhline(y=1.0, color="#94A3B8", linewidth=0.8, linestyle=":")
    ax.text(0.05, 1.15, "initial norm = 1.0", fontsize=7, color="#64748B",
            transform=ax.get_yaxis_transform())

    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])
    setup_axes(ax,
               title="VAL-1B: Embedding Norm Growth Without LayerNorm (Eq. 13)",
               xlabel="Enrichment Sweep",
               ylabel="Embedding Norm (log scale)")

    save_figure(fig, "val1b_norm_growth", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Chart 2: Per-domain final growth factor
# ---------------------------------------------------------------------------
def chart_per_domain(summary: dict) -> None:
    domain_data = summary["per_domain_final_sweep"]
    domains     = list(domain_data.keys())
    growths     = [domain_data[d]["growth_max"] for d in domains]

    # Sort ascending
    order   = sorted(range(len(domains)), key=lambda i: growths[i])
    domains = [domains[i] for i in order]
    growths = [growths[i] for i in order]

    # Friendly labels
    labels = [d.replace("_", " ").title() for d in domains]
    colors = COLORS["category_colors"][: len(domains)]

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    bars = ax.barh(labels, growths, color=colors[::-1], height=0.55)

    # Annotate bar values
    for bar, g in zip(bars, growths):
        w = bar.get_width()
        ax.text(w * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{g / 1e6:.2f}M×" if g >= 1e6 else f"{g / 1e3:.0f}K×",
                va="center", ha="left",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"])

    ax.set_xscale("log")
    ax.set_xlim(left=1.0)
    setup_axes(ax,
               title="VAL-1B: Per-Domain Max-Norm Growth at Sweep 5",
               xlabel="Growth Factor (log scale)",
               ylabel="")

    fig.tight_layout()
    save_figure(fig, "val1b_per_domain", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    summary = _load_summary()
    print("Generating val1b_norm_growth ...")
    chart_norm_growth(summary)
    print("Generating val1b_per_domain ...")
    chart_per_domain(summary)
    print("Done.")
