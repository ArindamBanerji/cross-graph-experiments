"""
charts.py — Publication-quality figures for FX-1-PROXY-REAL.
experiments/fx1_proxy_real/charts.py

6 charts:
  1. fx1_accuracy_by_mode       — accuracy per distribution mode (L2, bar + error)
  2. fx1_ece_by_mode            — ECE per distribution mode (L2, bar + error)
  3. fx1_per_category_combined  — per-category accuracy, combined mode (horiz bar)
  4. fx1_accuracy_vs_ece_scatter— accuracy vs ECE scatter with gate quadrants (L2)
  5. fx1_mahalanobis_vs_l2      — grouped bar: Mahalanobis vs L2 by mode
  6. fx1_confidence_bands       — confidence band fractions + accuracy (combined mode)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.viz.bridge_common import VIZ_DEFAULTS, setup_axes, save_figure

# ---------------------------------------------------------------------------
# Mode display names and colors
# ---------------------------------------------------------------------------

MODE_LABELS = {
    "centroidal":  "Centroidal\n(baseline)",
    "heavy_tail":  "Heavy\nTail",
    "correlated":  "Correlated\nFactors",
    "overlapping": "Overlapping\nClasses",
    "combined":    "Combined\n(realistic)",
}

MODE_COLORS = {
    "centroidal":  "#94A3B8",   # grey — baseline
    "heavy_tail":  "#60A5FA",   # light blue
    "correlated":  "#2563EB",   # blue
    "overlapping": "#7C3AED",   # purple
    "combined":    "#DC2626",   # red — most realistic / worst case
}

MODES = ["centroidal", "heavy_tail", "correlated", "overlapping", "combined"]

# Reference lines
ECE_V3B_REF     = 0.036   # V3B reference: ECE=0.036 at tau=0.1 centroidal
GATE_ACC_THRESH = 80.0    # gate: combined accuracy >= 80%
GATE_ECE_THRESH = 0.10    # gate: combined ECE <= 0.10

# Confidence band colors
BAND_COLORS = {
    "auto_approve": "#059669",   # green
    "agent_zone":   "#D97706",   # amber
    "human_review": "#DC2626",   # red
}
BAND_LABELS = {
    "auto_approve": "Auto-approve\n(conf ≥ 0.90)",
    "agent_zone":   "Agent zone\n(0.60–0.90)",
    "human_review": "Human review\n(conf < 0.60)",
}


# ---------------------------------------------------------------------------
# Chart 1 — Accuracy by mode (L2)
# ---------------------------------------------------------------------------

def chart_accuracy_by_mode(results: dict, paper_dir: str) -> None:
    """Bar chart: L2 accuracy per distribution mode, error bars ±1 std."""
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    x     = np.arange(len(MODES))
    width = 0.55
    centroidal_acc = results["centroidal"]["l2"]["mean_acc"] * 100

    for i, mode in enumerate(MODES):
        r   = results[mode]["l2"]
        acc = r["mean_acc"] * 100
        std = r["std_acc"]  * 100
        ax.bar(
            i, acc,
            width=width,
            color=MODE_COLORS[mode],
            yerr=std,
            capsize=4,
            error_kw={"linewidth": 1.2, "ecolor": "#374151"},
            zorder=3,
            label=MODE_LABELS[mode],
        )
        if mode != "centroidal":
            delta = acc - centroidal_acc
            sign  = "+" if delta >= 0 else ""
            ax.text(
                i, acc + std + 0.5,
                f"{sign}{delta:.1f}pp",
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                color=MODE_COLORS[mode], fontweight="bold",
            )
        else:
            ax.text(
                i, acc + std + 0.5,
                f"{acc:.2f}%",
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                color="#374151",
            )

    ax.axhline(centroidal_acc, color="#94A3B8", linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=2, label=f"Centroidal ref: {centroidal_acc:.2f}%")
    ax.axhline(GATE_ACC_THRESH, color="#DC2626", linewidth=1.2, linestyle=":",
               alpha=0.8, zorder=2, label=f"Gate threshold: {GATE_ACC_THRESH:.0f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])
    all_accs = [results[m]["l2"]["mean_acc"] * 100 for m in MODES]
    all_stds = [results[m]["l2"]["std_acc"]  * 100 for m in MODES]
    ax.set_ylim(
        max(0, min(a - s for a, s in zip(all_accs, all_stds)) - 5),
        min(105, centroidal_acc + 5),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)

    n     = results["centroidal"]["l2"].get("n_alerts", 2000)
    nseed = results["centroidal"]["l2"].get("n_seeds",  10)
    setup_axes(ax,
               title="FX-1-PROXY-REAL: Accuracy Under Realistic Factor Distributions",
               xlabel="Distribution Mode",
               ylabel="Accuracy (%)")
    ax.set_title(
        f"FX-1-PROXY-REAL: Accuracy Under Realistic Factor Distributions\n"
        f"N={n}, n_seeds={nseed}, τ=0.1, L2 kernel — Gate: ≥{GATE_ACC_THRESH:.0f}%",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    save_figure(fig, "fx1_accuracy_by_mode", paper_dir)


# ---------------------------------------------------------------------------
# Chart 2 — ECE by mode (L2)
# ---------------------------------------------------------------------------

def chart_ece_by_mode(results: dict, paper_dir: str) -> None:
    """Bar chart: L2 ECE per distribution mode, error bars ±1 std."""
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    x     = np.arange(len(MODES))
    width = 0.55

    for i, mode in enumerate(MODES):
        r   = results[mode]["l2"]
        ece = r["mean_ece"]
        std = r["std_ece"]
        ax.bar(
            i, ece,
            width=width,
            color=MODE_COLORS[mode],
            yerr=std,
            capsize=4,
            error_kw={"linewidth": 1.2, "ecolor": "#374151"},
            zorder=3,
        )
        ax.text(
            i, ece + std + 0.001,
            f"{ece:.4f}",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color=MODE_COLORS[mode], fontweight="bold",
        )

    ax.axhline(ECE_V3B_REF, color="#94A3B8", linewidth=1.0, linestyle="--",
               alpha=0.8, zorder=2,
               label=f"V3B ref: ECE={ECE_V3B_REF} at τ=0.1 (centroidal synthetic)")
    ax.axhline(GATE_ECE_THRESH, color="#DC2626", linewidth=1.2, linestyle=":",
               alpha=0.8, zorder=2,
               label=f"Gate threshold: ECE={GATE_ECE_THRESH}")

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=7.5, loc="upper left")

    setup_axes(ax,
               title="FX-1-PROXY-REAL: Calibration (ECE) Under Realistic Distributions",
               xlabel="Distribution Mode",
               ylabel="Expected Calibration Error (ECE)")
    ax.set_title(
        "FX-1-PROXY-REAL: Calibration (ECE) Under Realistic Distributions\n"
        f"V3B reference: ECE={ECE_V3B_REF} at τ=0.1 (centroidal synthetic).",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    save_figure(fig, "fx1_ece_by_mode", paper_dir)


# ---------------------------------------------------------------------------
# Chart 3 — Per-category accuracy, combined mode
# ---------------------------------------------------------------------------

def chart_per_category_combined(
    combined_per_cat:     dict,
    overall_combined_acc: float,
    centroidal_per_cat:   dict,
    paper_dir:            str,
) -> None:
    """Horizontal bar: per-category accuracy in combined mode (L2)."""
    sorted_cats = sorted(combined_per_cat.items(), key=lambda x: x[1])
    cat_names   = [c.replace("_", "\n") for c, _ in sorted_cats]
    acc_vals    = [v * 100 for _, v in sorted_cats]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = [
        MODE_COLORS["combined"] if v < overall_combined_acc * 100 else MODE_COLORS["centroidal"]
        for v in acc_vals
    ]
    bars = ax.barh(cat_names, acc_vals, color=colors, height=0.6, zorder=3)

    for bar, val in zip(bars, acc_vals):
        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"], fontweight="bold",
        )

    ax.axvline(overall_combined_acc * 100, color="#DC2626", linewidth=1.2,
               linestyle="--", alpha=0.8,
               label=f"Combined overall: {overall_combined_acc*100:.2f}%")
    if centroidal_per_cat:
        centroidal_mean = np.mean(list(centroidal_per_cat.values())) * 100
        ax.axvline(centroidal_mean, color="#94A3B8", linewidth=1.0,
                   linestyle=":", alpha=0.8,
                   label=f"Centroidal mean: {centroidal_mean:.2f}%")

    ax.set_xlim(left=max(0, min(acc_vals) - 5), right=105)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Alert Category",  fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        "FX-1-PROXY-REAL: Per-Category Accuracy (Combined Mode, L2)\n"
        "Which categories are most sensitive to realistic distributions?",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    fig.tight_layout()
    save_figure(fig, "fx1_per_category_combined", paper_dir)


# ---------------------------------------------------------------------------
# Chart 4 — Accuracy vs ECE scatter (L2)
# ---------------------------------------------------------------------------

def chart_accuracy_vs_ece_scatter(results: dict, paper_dir: str) -> None:
    """Scatter: accuracy (x) vs ECE (y), one point per mode (L2 kernel)."""
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    for mode in MODES:
        r   = results[mode]["l2"]
        acc = r["mean_acc"] * 100
        ece = r["mean_ece"]
        ax.scatter(acc, ece, color=MODE_COLORS[mode], s=120, zorder=5,
                   edgecolors="white", linewidths=1.0)
        offset_y = 0.0005 if mode != "centroidal" else -0.0015
        ax.annotate(
            MODE_LABELS[mode].replace("\n", " "),
            xy=(acc, ece),
            xytext=(acc + 0.15, ece + offset_y),
            fontsize=7.5, ha="left", va="center",
            color=MODE_COLORS[mode],
        )

    ax.axvline(GATE_ACC_THRESH, color="#DC2626", linewidth=1.2, linestyle=":",
               alpha=0.7, label=f"Accuracy gate: {GATE_ACC_THRESH:.0f}%")
    ax.axhline(GATE_ECE_THRESH, color="#7C3AED", linewidth=1.2, linestyle=":",
               alpha=0.7, label=f"ECE gate: {GATE_ECE_THRESH}")

    all_accs = [results[m]["l2"]["mean_acc"] * 100 for m in MODES]
    all_eces = [results[m]["l2"]["mean_ece"]       for m in MODES]
    x_min = max(0, min(all_accs) - 3)
    x_max = min(105, max(all_accs) + 3)
    y_min = 0
    y_max = max(all_eces) * 1.3

    ax.text(GATE_ACC_THRESH + 0.3, y_max * 0.92, "⚠ High acc,\nhigh ECE",
            fontsize=7, color="#7C3AED", alpha=0.6)
    ax.text(x_min + 0.2, y_max * 0.92, "✗ Low acc,\nhigh ECE",
            fontsize=7, color="#DC2626", alpha=0.6)
    ax.text(GATE_ACC_THRESH + 0.3, y_min + y_max * 0.03, "✓ Ideal:\nhigh acc, low ECE",
            fontsize=7, color="#059669", alpha=0.7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.legend(fontsize=8, loc="upper left")

    setup_axes(ax,
               title="FX-1-PROXY-REAL: Accuracy vs Calibration Trade-off (L2)",
               xlabel="Accuracy (%)",
               ylabel="Expected Calibration Error (ECE)")
    save_figure(fig, "fx1_accuracy_vs_ece_scatter", paper_dir)


# ---------------------------------------------------------------------------
# Chart 5 — Mahalanobis vs L2 grouped bar
# ---------------------------------------------------------------------------

def chart_mahalanobis_vs_l2(results: dict, paper_dir: str) -> None:
    """
    Grouped bar chart: accuracy for L2 vs Mahalanobis, across all 5 modes.
    Annotates delta (Maha - L2) above each pair.
    """
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    x     = np.arange(len(MODES))
    width = 0.35

    colors_l2   = [MODE_COLORS[m] for m in MODES]
    colors_maha = ["#1E3A5F", "#1D4ED8", "#1E40AF", "#4C1D95", "#991B1B"]
    # Darker variants for Mahalanobis bars

    for i, mode in enumerate(MODES):
        acc_l2   = results[mode]["l2"]["mean_acc"] * 100
        std_l2   = results[mode]["l2"]["std_acc"]  * 100
        acc_maha = results[mode]["mahalanobis"]["mean_acc"] * 100
        std_maha = results[mode]["mahalanobis"]["std_acc"]  * 100

        ax.bar(i - width/2, acc_l2, width=width,
               color=colors_l2[i], alpha=0.90,
               yerr=std_l2, capsize=3,
               error_kw={"linewidth": 1.0, "ecolor": "#374151"},
               zorder=3, label="L2" if i == 0 else "_nolegend_")
        ax.bar(i + width/2, acc_maha, width=width,
               color=colors_maha[i], alpha=0.90,
               yerr=std_maha, capsize=3,
               error_kw={"linewidth": 1.0, "ecolor": "#374151"},
               zorder=3, label="Mahalanobis" if i == 0 else "_nolegend_",
               hatch="//", edgecolor="white", linewidth=0.5)

        # Delta annotation above the pair
        delta = acc_maha - acc_l2
        sign  = "+" if delta >= 0 else ""
        top   = max(acc_l2 + std_l2, acc_maha + std_maha) + 0.5
        ax.text(i, top, f"{sign}{delta:.1f}pp",
                ha="center", va="bottom",
                fontsize=VIZ_DEFAULTS["annotation_fontsize"],
                color="#374151", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])

    all_accs = [results[m][k]["mean_acc"] * 100
                for m in MODES for k in ("l2", "mahalanobis")]
    all_stds = [results[m][k]["std_acc"]  * 100
                for m in MODES for k in ("l2", "mahalanobis")]
    ax.set_ylim(
        max(0, min(a - s for a, s in zip(all_accs, all_stds)) - 5),
        min(105, max(all_accs) + 7),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=8, loc="lower left")

    setup_axes(ax,
               title="FX-1-PROXY-REAL: Mahalanobis vs L2 Kernel Accuracy",
               xlabel="Distribution Mode",
               ylabel="Accuracy (%)")
    ax.set_title(
        "FX-1-PROXY-REAL: Mahalanobis vs L2 Kernel Accuracy\n"
        "Δ = Mahalanobis − L2 (positive = Mahalanobis advantage)",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    save_figure(fig, "fx1_mahalanobis_vs_l2", paper_dir)


# ---------------------------------------------------------------------------
# Chart 6 — Confidence band breakdown (combined mode)
# ---------------------------------------------------------------------------

def chart_confidence_bands(
    bands_l2: dict,
    bands_mh: dict,
    paper_dir: str,
) -> None:
    """
    Side-by-side grouped bar: confidence band fractions + accuracy annotations.
    Left group = L2, right group = Mahalanobis.
    Bands: auto_approve, agent_zone, human_review.
    """
    band_names = ["auto_approve", "agent_zone", "human_review"]
    fig, (ax_frac, ax_acc) = plt.subplots(1, 2, figsize=(10, 4.5))

    x     = np.arange(len(band_names))
    width = 0.35

    # --- Subplot 1: Band fractions ---
    for i, bname in enumerate(band_names):
        frac_l2 = bands_l2[bname]["fraction"] * 100
        frac_mh = bands_mh[bname]["fraction"] * 100
        ax_frac.bar(i - width/2, frac_l2, width=width,
                    color=BAND_COLORS[bname], alpha=0.85, zorder=3,
                    label="L2" if i == 0 else "_nolegend_")
        ax_frac.bar(i + width/2, frac_mh, width=width,
                    color=BAND_COLORS[bname], alpha=0.55, zorder=3,
                    hatch="//", edgecolor="white", linewidth=0.5,
                    label="Mahalanobis" if i == 0 else "_nolegend_")
        ax_frac.text(i - width/2, frac_l2 + 0.5, f"{frac_l2:.1f}%",
                     ha="center", va="bottom",
                     fontsize=VIZ_DEFAULTS["annotation_fontsize"])
        ax_frac.text(i + width/2, frac_mh + 0.5, f"{frac_mh:.1f}%",
                     ha="center", va="bottom",
                     fontsize=VIZ_DEFAULTS["annotation_fontsize"])

    ax_frac.set_xticks(x)
    ax_frac.set_xticklabels([BAND_LABELS[b] for b in band_names],
                             fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax_frac.set_ylim(0, 105)
    ax_frac.set_ylabel("% of Decisions", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax_frac.set_title("Decision Volume by Confidence Band",
                      fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax_frac.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_frac.legend(fontsize=8, loc="upper right")
    ax_frac.spines["top"].set_visible(False)
    ax_frac.spines["right"].set_visible(False)

    # --- Subplot 2: Band accuracy ---
    for i, bname in enumerate(band_names):
        acc_l2 = bands_l2[bname].get("accuracy")
        acc_mh = bands_mh[bname].get("accuracy")
        val_l2 = (acc_l2 * 100) if acc_l2 is not None else 0.0
        val_mh = (acc_mh * 100) if acc_mh is not None else 0.0

        ax_acc.bar(i - width/2, val_l2, width=width,
                   color=BAND_COLORS[bname], alpha=0.85, zorder=3,
                   label="L2" if i == 0 else "_nolegend_")
        ax_acc.bar(i + width/2, val_mh, width=width,
                   color=BAND_COLORS[bname], alpha=0.55, zorder=3,
                   hatch="//", edgecolor="white", linewidth=0.5,
                   label="Mahalanobis" if i == 0 else "_nolegend_")

        if acc_l2 is not None:
            ax_acc.text(i - width/2, val_l2 + 0.5, f"{val_l2:.1f}%",
                        ha="center", va="bottom",
                        fontsize=VIZ_DEFAULTS["annotation_fontsize"])
        if acc_mh is not None:
            ax_acc.text(i + width/2, val_mh + 0.5, f"{val_mh:.1f}%",
                        ha="center", va="bottom",
                        fontsize=VIZ_DEFAULTS["annotation_fontsize"])

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([BAND_LABELS[b] for b in band_names],
                            fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax_acc.set_ylim(0, 110)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax_acc.set_title("Accuracy Within Each Confidence Band",
                     fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax_acc.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    fig.suptitle(
        "FX-1-PROXY-REAL: Confidence Band Breakdown (Combined Mode, seed=42)\n"
        "Solid = L2  |  Hatched = Mahalanobis",
        fontsize=VIZ_DEFAULTS["title_fontsize"] + 0.5,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fx1_confidence_bands", paper_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(
    results:            dict,
    combined_per_cat:   dict,
    centroidal_per_cat: dict,
    bands_l2:           dict,
    bands_mh:           dict,
    paper_dir:          str,
) -> None:
    """Generate all 6 FX-1 charts and save to paper_dir (PDF + PNG)."""
    print("\nGenerating FX-1 charts...")
    chart_accuracy_by_mode(results, paper_dir)
    chart_ece_by_mode(results, paper_dir)
    chart_per_category_combined(
        combined_per_cat,
        overall_combined_acc=results["combined"]["l2"]["mean_acc"],
        centroidal_per_cat=centroidal_per_cat,
        paper_dir=paper_dir,
    )
    chart_accuracy_vs_ece_scatter(results, paper_dir)
    chart_mahalanobis_vs_l2(results, paper_dir)
    chart_confidence_bands(bands_l2, bands_mh, paper_dir)
    print("All 6 charts saved (PDF + PNG).")
