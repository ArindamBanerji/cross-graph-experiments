"""
Log-log scaling chart: Discovery scaling D(n) ~ n^b, n=2-15 domains.

Reads:
  experiments/exp3_multidomain_scaling/results/extended_scaling_data.csv
  experiments/exp3_multidomain_scaling/results/extended_scaling_fit.json

Outputs (paper_figures/):
  fig9_scaling_11point.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
PAPER_FIGS = ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_csv() -> dict[int, list[float]]:
    """Return {n_domains: [total_discoveries per seed]}."""
    acc: dict[int, list[float]] = defaultdict(list)
    with open(RESULTS / "extended_scaling_data.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            acc[int(row["n_domains"])].append(float(row["total_discoveries"]))
    return dict(acc)


def _load_fit() -> dict:
    with open(RESULTS / "extended_scaling_fit.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def make_scaling_figure(data: dict[int, list[float]], fit: dict) -> None:

    # -----------------------------------------------------------------------
    # Aggregate: mean and std of total_discoveries per domain count
    # -----------------------------------------------------------------------
    domain_counts = sorted(data.keys())
    means = np.array([np.mean(data[n]) for n in domain_counts])
    stds  = np.array([np.std(data[n],  ddof=1) for n in domain_counts])
    x     = np.array(domain_counts, dtype=float)

    # -----------------------------------------------------------------------
    # Fit parameters from JSON
    # -----------------------------------------------------------------------
    ext     = fit["extended_range"]
    a_ext   = ext["a"]          # 206.8988
    b_ext   = ext["fitted_exponent"]   # 2.1127
    b_lo    = ext["ci_95_lower"]       # 2.0894
    b_hi    = ext["ci_95_upper"]       # 2.1361
    r2_ext  = ext["r_squared"]         # 0.999883

    a_quad  = fit["alternative_models_full_range"]["pure_quadratic"]["a"]  # 274.7851

    # -----------------------------------------------------------------------
    # Figure + axes
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({
        "font.family":   "serif",
        "font.size":     11,
        "axes.titlesize": 13,
    })
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.88)

    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)

    # -----------------------------------------------------------------------
    # CI shaded band around b=2.11 fit
    # -----------------------------------------------------------------------
    y_hi = a_ext * x_fit ** b_hi
    y_lo = a_ext * x_fit ** b_lo
    ax.fill_between(x_fit, y_lo, y_hi,
                    color="#2563EB", alpha=0.12, zorder=2,
                    label="95% CI [2.09, 2.14]")

    # -----------------------------------------------------------------------
    # Fit lines
    # -----------------------------------------------------------------------
    # 1. Best-fit b=2.11 — solid blue
    ax.plot(x_fit, a_ext * x_fit ** b_ext,
            color="#2563EB", linewidth=2.2, linestyle="-", zorder=4,
            label=f"b = {b_ext:.2f} (11-point fit)")

    # 2. Pure quadratic b=2.0 — dashed gray
    ax.plot(x_fit, a_quad * x_fit ** 2.0,
            color="#94A3B8", linewidth=1.6, linestyle="--", zorder=3,
            label="b = 2.0 (quadratic)")

    # 3. Original 5-point b=2.30 — dotted orange
    a_orig = fit["original_range"]["a"]   # 145.6168
    ax.plot(x_fit, a_orig * x_fit ** 2.30,
            color="#D97706", linewidth=1.6, linestyle=":", zorder=3,
            label="b = 2.30 (original 5-point)")

    # -----------------------------------------------------------------------
    # Scatter: mean discoveries with ±1 std error bars
    # -----------------------------------------------------------------------
    ax.errorbar(x, means, yerr=stds,
                fmt="o", color="#1E293B",
                markersize=6, capsize=4, capthick=1.2,
                linewidth=1.0, zorder=5,
                label="Data (mean ±1 SD, 10 seeds)")

    # -----------------------------------------------------------------------
    # Log-log axes
    # -----------------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1.8, 18)
    ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    # -----------------------------------------------------------------------
    # Labels + title
    # -----------------------------------------------------------------------
    ax.set_xlabel("Number of Domains (n)", fontsize=12)
    ax.set_ylabel("Total Discoveries", fontsize=12)
    ax.set_title(
        r"Discovery Scaling: $D(n) \propto n^b$, $n = 2\text{--}15$ Domains",
        fontsize=13, pad=10
    )

    # -----------------------------------------------------------------------
    # Annotation box
    # -----------------------------------------------------------------------
    ax.text(
        0.97, 0.05,
        f"b = {b_ext:.2f},  95% CI [{b_lo:.2f}, {b_hi:.2f}]\n"
        f"R\u00b2 = {r2_ext:.4f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.92),
    )

    # -----------------------------------------------------------------------
    # Legend: lower right
    # -----------------------------------------------------------------------
    ax.legend(
        fontsize=9.5,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#E2E8F0",
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    PAPER_FIGS.mkdir(exist_ok=True)
    for ext_fmt in ("pdf", "png"):
        out = PAPER_FIGS / f"fig9_scaling_11point.{ext_fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading CSV ...")
    data = _load_csv()
    print(f"  Domain counts: {sorted(data.keys())}")
    print(f"  Seeds per count: {len(next(iter(data.values())))}")

    print("Loading fit JSON ...")
    fit = _load_fit()
    print(f"  b_ext={fit['extended_range']['fitted_exponent']}, "
          f"R2={fit['extended_range']['r_squared']:.6f}")

    print("Generating fig9_scaling_11point ...")
    make_scaling_figure(data, fit)
    print("Done.")
