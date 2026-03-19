"""
FX1-PROXY-REAL charts.

Three publication-quality figures:
  fx1r_factor_distributions        — 3-panel histograms: real vs Gaussian fits
  fx1r_kl_divergence_from_synthetic — 3 bars of KL divergence
  fx1r_distribution_statistics      — statistics table as colored grid
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.viz.bridge_common import save_figure, COLORS


_RC = {
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
}

_BLUE   = "#2563EB"
_RED    = "#DC2626"
_GREEN  = "#059669"
_ORANGE = "#D97706"
_GRAY   = "#64748B"
_LGRAY  = "#E2E8F0"

FACTOR_LABELS = {
    "threat_intel":      "Threat Intel Score",
    "asset_criticality": "Asset Criticality Proxy",
    "pattern_history":   "Pattern History Proxy",
}

SYNTHETIC_REFS = {
    "threat_intel":      {"mean": 0.5, "std": 0.20},
    "asset_criticality": {"mean": 0.5, "std": 0.25},
    "pattern_history":   {"mean": 0.3, "std": 0.20},
}


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Chart 1: Factor distributions — 3 panels
# ---------------------------------------------------------------------------

def chart_factor_distributions(factors: dict, stats: dict) -> None:
    plt.rcParams.update(_RC)

    keys = list(FACTOR_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.82, wspace=0.35)

    bins = np.linspace(0.0, 1.0, 31)
    x_fit = np.linspace(0.0, 1.0, 200)

    for ax, key in zip(axes, keys):
        vals = factors[key]
        ref  = SYNTHETIC_REFS[key]
        st   = stats[key]

        if len(vals) > 0:
            # Histogram
            ax.hist(vals, bins=bins,
                    color=_BLUE, alpha=0.65,
                    density=True, zorder=3,
                    label=f"Real (N={len(vals)})")

        # Fitted Gaussian (real data)
        r_mean, r_std = st["mean"], st["std"]
        if len(vals) > 1 and r_std > 0:
            pdf_real = (1.0 / (r_std * np.sqrt(2 * np.pi)) *
                        np.exp(-0.5 * ((x_fit - r_mean) / r_std)**2))
            ax.plot(x_fit, pdf_real,
                    color=_RED, linewidth=1.8, linestyle="--",
                    zorder=4, label=f"Fit μ={r_mean:.2f} σ={r_std:.2f}")

        # Synthetic centroidal reference Gaussian
        s_mean, s_std = ref["mean"], ref["std"]
        pdf_synth = (1.0 / (s_std * np.sqrt(2 * np.pi)) *
                     np.exp(-0.5 * ((x_fit - s_mean) / s_std)**2))
        ax.plot(x_fit, pdf_synth,
                color=_GREEN, linewidth=1.5, linestyle="-.",
                zorder=3, label=f"Synthetic μ={s_mean:.2f} σ={s_std:.2f}")

        # Annotation
        ax.text(0.97, 0.97,
                f"skew={st['skewness']:.2f}\nN={st['n']}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9, color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=_LGRAY, alpha=0.88))

        ax.set_xlim(0, 1)
        ax.set_xlabel("Factor Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(FACTOR_LABELS[key], fontsize=11, pad=4)
        ax.legend(fontsize=7.5, loc="upper left",
                  frameon=True, framealpha=0.90, edgecolor=_LGRAY)
        _clean_ax(ax)

    fig.suptitle("Real IOC vs Synthetic Centroidal Factor Distributions",
                 fontsize=13, y=0.97)

    save_figure(fig, "fx1r_factor_distributions", output_dir="paper_figures")
    print("[CHART 1] fx1r_factor_distributions.png + .pdf saved")


# ---------------------------------------------------------------------------
# Chart 2: KL divergence bars
# ---------------------------------------------------------------------------

def chart_kl_divergence(kl_values: dict) -> None:
    plt.rcParams.update(_RC)

    keys   = list(FACTOR_LABELS.keys())
    labels = [FACTOR_LABELS[k] for k in keys]
    kls    = [kl_values[k] for k in keys]

    def _bar_color(kl: float) -> str:
        if kl < 0.3:   return _GREEN
        if kl < 0.5:   return _ORANGE
        return _RED

    colors = [_bar_color(kl) for kl in kls]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.86)

    bars = ax.bar(x, kls,
                  width=0.52,
                  color=colors,
                  edgecolor="white", linewidth=0.8,
                  zorder=3)

    # Value labels
    for bar, kl in zip(bars, kls):
        if not (kl != kl):   # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2,
                    kl + 0.01,
                    f"{kl:.3f}",
                    ha="center", va="bottom",
                    fontsize=10, color="#1E293B", fontweight="bold", zorder=5)

    # Reference lines
    ax.axhline(y=0.1, color=_GREEN, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(keys) - 0.5, 0.115,
            "Low divergence (0.1)",
            ha="right", va="bottom", fontsize=8, color=_GREEN, style="italic")

    ax.axhline(y=0.5, color=_RED, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(keys) - 0.5, 0.515,
            "Recalibration needed (0.5)",
            ha="right", va="bottom", fontsize=8, color=_RED, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("KL Divergence (real ‖ synthetic)", fontsize=11)
    y_max = max(max(kls, default=0) * 1.4, 0.6)
    ax.set_ylim(0, y_max)
    ax.tick_params(labelsize=9)
    _clean_ax(ax)

    ax.set_title("KL Divergence: Real IOC Data vs Synthetic Centroidal",
                 fontsize=12, pad=10)
    fig.text(0.12, 0.875,
             "Green = low divergence (<0.3).  Orange = moderate.  Red = high (>0.5).",
             ha="left", va="bottom",
             fontsize=9, color="#475569", style="italic",
             transform=fig.transFigure)

    save_figure(fig, "fx1r_kl_divergence_from_synthetic", output_dir="paper_figures")
    print("[CHART 2] fx1r_kl_divergence_from_synthetic.png + .pdf saved")


# ---------------------------------------------------------------------------
# Chart 3: Statistics table as colored grid
# ---------------------------------------------------------------------------

def chart_distribution_statistics(stats: dict) -> None:
    plt.rcParams.update(_RC)

    keys   = list(FACTOR_LABELS.keys())
    labels = [FACTOR_LABELS[k] for k in keys]

    row_names = ["mean", "std", "skewness", "kurtosis"]
    row_labels = ["Mean", "Std Dev", "Skewness", "Excess Kurtosis"]

    # Build table data: rows = statistics, cols = real factors + synthetic
    col_names  = labels + ["Synthetic\n(uniform ref)"]
    n_rows = len(row_names)
    n_cols = len(col_names)

    # Collect values
    cell_vals = np.full((n_rows, n_cols), float("nan"))
    for j, key in enumerate(keys):
        st = stats[key]
        for i, rn in enumerate(row_names):
            cell_vals[i, j] = st.get(rn, float("nan"))

    # Synthetic reference column (last)
    synth_stats = {
        "mean":     0.5,
        "std":      0.22,    # avg of the three synthetic stds
        "skewness": 0.0,
        "kurtosis": -1.2,    # uniform-ish → negative excess kurtosis
    }
    for i, rn in enumerate(row_names):
        cell_vals[i, -1] = synth_stats[rn]

    # Color cells: compare real to synthetic reference per statistic
    synth_vals = cell_vals[:, -1]
    cell_colors = []
    for i in range(n_rows):
        row_colors = []
        for j in range(n_cols - 1):
            v   = cell_vals[i, j]
            ref = synth_vals[i]
            if np.isnan(v) or np.isnan(ref):
                row_colors.append("#F1F5F9")
            else:
                delta_rel = abs(v - ref) / (abs(ref) + 0.1)
                if delta_rel < 0.5:
                    row_colors.append("#D1FAE5")   # light green
                elif delta_rel < 1.5:
                    row_colors.append("#FEF3C7")   # light orange
                else:
                    row_colors.append("#FEE2E2")   # light red
        row_colors.append("#F0FDF4")  # synthetic ref col always light green
        cell_colors.append(row_colors)

    # Build text table
    cell_text = []
    for i in range(n_rows):
        row_text = []
        for j in range(n_cols):
            v = cell_vals[i, j]
            row_text.append(f"{v:.3f}" if not np.isnan(v) else "n/a")
        cell_text.append(row_text)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.82)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_names,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")

    # Style row label col
    for i in range(1, n_rows + 1):
        cell = tbl[i, -1]
        cell.set_facecolor("#334155")
        cell.set_text_props(color="white", fontweight="bold")

    fig.suptitle("Distribution Statistics: Real IOC vs Synthetic Centroidal",
                 fontsize=13, y=0.97)
    fig.text(0.5, 0.01,
             "Cell color: green = within 50% of synthetic, orange = 50–150%, red = >150% deviation.",
             ha="center", va="bottom",
             fontsize=8.5, color="#475569", style="italic")

    save_figure(fig, "fx1r_distribution_statistics", output_dir="paper_figures")
    print("[CHART 3] fx1r_distribution_statistics.png + .pdf saved")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(factors: dict, stats: dict, kl_values: dict) -> None:
    print("\nGenerating FX1-PROXY-REAL charts ...")
    chart_factor_distributions(factors, stats)
    chart_kl_divergence(kl_values)
    chart_distribution_statistics(stats)
    print("Done. 6 files in paper_figures/")


if __name__ == "__main__":
    import json
    _results_path = Path(__file__).resolve().parent / "results.json"
    if not _results_path.exists():
        print(f"ERROR: {_results_path} not found. Run run.py first.")
        sys.exit(1)
    with open(_results_path) as fh:
        d = json.load(fh)
    _factors = {k: np.array(v) for k, v in d["factors"].items()}
    generate_charts(_factors, d["stats"], d["kl_divergence"])
