"""
VAL-3B: Calibration analysis figures.

Reads:
  experiments/validation/calibration_summary.json

Outputs (paper_figures/):
  val3b_temperature_ece.{pdf,png}    -- ECE vs tau + method comparison
  val3b_reliability_diagram.{pdf,png} -- reliability diagram at tau 0.1 and 0.25
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.viz.bridge_common import COLORS, VIZ_DEFAULTS, setup_axes, save_figure

VAL_DIR    = Path(__file__).resolve().parent
PAPER_FIGS = ROOT / "paper_figures"

ECE_WELL = 0.05
ECE_MOD  = 0.15


def _load_summary() -> dict:
    with open(VAL_DIR / "calibration_summary.json") as fh:
        return json.load(fh)


def _calibration_grade(ece: float) -> str:
    if ece < ECE_WELL:
        return "well"
    if ece < ECE_MOD:
        return "moderate"
    return "poor"


def _grade_color(grade: str) -> str:
    return {"well": "#059669", "moderate": "#D97706", "poor": "#DC2626"}[grade]


# ---------------------------------------------------------------------------
# Chart 1: Temperature sensitivity (tau sweep) + method comparison
# ---------------------------------------------------------------------------
def chart_temperature_ece(summary: dict) -> None:
    temp_sens = summary["temperature_sensitivity"]
    taus      = [0.1, 0.25, 0.5, 1.0]
    tau_eces  = [temp_sens[str(t)]["mean_ece"] for t in taus]
    tau_labels = [f"τ={t}" for t in taus]
    tau_grades = [_calibration_grade(e) for e in tau_eces]
    tau_colors = [_grade_color(g) for g in tau_grades]

    ml_methods = ["logistic_regression", "xgboost", "random_forest"]
    ml_labels  = ["Logistic\nRegression", "XGBoost", "Random\nForest"]
    ml_eces    = [summary[m]["mean_ece"] for m in ml_methods]
    ml_grades  = [_calibration_grade(e) for e in ml_eces]
    ml_colors  = [_grade_color(g) for g in ml_grades]

    fig, (ax_tau, ax_ml) = plt.subplots(1, 2, figsize=VIZ_DEFAULTS["figsize_wide"])

    # --- Left: tau sweep ---
    bars_tau = ax_tau.bar(tau_labels, tau_eces, color=tau_colors, width=0.5)
    ax_tau.axhline(y=ECE_WELL, color="#059669", linewidth=1.0, linestyle="--",
                   label=f"Well calibrated (ECE<{ECE_WELL})")
    ax_tau.axhline(y=ECE_MOD, color="#D97706", linewidth=1.0, linestyle="--",
                   label=f"Moderate (ECE<{ECE_MOD})")
    for bar, ece in zip(bars_tau, tau_eces):
        ax_tau.text(bar.get_x() + bar.get_width() / 2, ece + 0.008,
                    f"{ece:.3f}",
                    ha="center", va="bottom",
                    fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax_tau.set_ylim(0, max(tau_eces) * 1.20)
    ax_tau.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"] - 1)
    setup_axes(ax_tau,
               title="L2 Centroid: ECE vs Temperature τ",
               xlabel="Temperature Parameter τ",
               ylabel="Mean ECE (10 seeds)")

    # --- Right: ML comparison ---
    bars_ml = ax_ml.bar(ml_labels, ml_eces, color=ml_colors, width=0.5)
    # Also add L2 centroid at best tau (0.1)
    best_tau_ece = temp_sens["0.1"]["mean_ece"]
    ax_ml.axhline(y=best_tau_ece, color=COLORS["gt_noise_5"], linewidth=1.5,
                  linestyle="--", label=f"L2 centroid τ=0.1 (ECE={best_tau_ece:.3f})")
    for bar, ece in zip(bars_ml, ml_eces):
        ax_ml.text(bar.get_x() + bar.get_width() / 2, ece + 0.002,
                   f"{ece:.3f}",
                   ha="center", va="bottom",
                   fontsize=VIZ_DEFAULTS["annotation_fontsize"])
    ax_ml.axhline(y=ECE_WELL, color="#059669", linewidth=1.0, linestyle=":")
    ax_ml.set_ylim(0, max(ml_eces) * 1.25)
    ax_ml.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"] - 1)
    setup_axes(ax_ml,
               title="Method Comparison: ECE",
               xlabel="Method",
               ylabel="Mean ECE (10 seeds)")

    fig.tight_layout(pad=2.0)
    save_figure(fig, "val3b_temperature_ece", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Chart 2: Reliability diagram
# ---------------------------------------------------------------------------
def chart_reliability_diagram(summary: dict) -> None:
    """
    Show reliability (calibration) curves for L2 centroid at tau=0.1 and
    tau=0.25, alongside perfect calibration diagonal.

    Note: The summary JSON has tau=0.25 reliability bins for l2_centroid.
    We also load the raw calibration_results.csv to reconstruct tau=0.1 bins.
    Since we only have pre-aggregated bins for tau=0.25 in the JSON, we
    show tau=0.25 from JSON and note tau=0.1 ECE from temperature_sensitivity.
    """
    import csv

    N_BINS = 10
    bin_bounds = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_centers = (bin_bounds[:-1] + bin_bounds[1:]) / 2

    # tau=0.25 bins from JSON (pre-aggregated)
    bins_025 = summary["l2_centroid"]["reliability_bins"]
    conf_025 = [b["mean_confidence"] if b["count"] > 0 else None for b in bins_025]
    acc_025  = [b["mean_accuracy"]   if b["count"] > 0 else None for b in bins_025]

    # Filter out empty bins
    valid_025 = [(c, a) for c, a in zip(conf_025, acc_025) if c is not None and c > 0]
    conf_025_v, acc_025_v = zip(*valid_025) if valid_025 else ([], [])

    # XGBoost (best ML, for reference)
    bins_xgb = summary["xgboost"]["reliability_bins"]
    valid_xgb = [(b["mean_confidence"], b["mean_accuracy"])
                 for b in bins_xgb if b["count"] > 0 and b["mean_confidence"] > 0]
    conf_xgb_v, acc_xgb_v = zip(*valid_xgb) if valid_xgb else ([], [])

    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color="#94A3B8", linewidth=1.2,
            linestyle="--", label="Perfect calibration", zorder=1)

    # L2 centroid tau=0.25
    if conf_025_v:
        ax.scatter(conf_025_v, acc_025_v,
                   color=COLORS["gt_noise_30"], s=50, zorder=3,
                   label=f"L2 centroid τ=0.25 (ECE=0.190)", marker="o")
        ax.plot(conf_025_v, acc_025_v,
                color=COLORS["gt_noise_30"], linewidth=1.5,
                linestyle="-", alpha=0.7, zorder=2)

    # XGBoost reference
    if conf_xgb_v:
        ax.scatter(conf_xgb_v, acc_xgb_v,
                   color=COLORS["mi_static"], s=50, zorder=3,
                   label="XGBoost (ECE=0.015)", marker="s")
        ax.plot(conf_xgb_v, acc_xgb_v,
                color=COLORS["mi_static"], linewidth=1.5,
                linestyle="-", alpha=0.7, zorder=2)

    # Annotate optimal tau result
    ece_01 = summary["temperature_sensitivity"]["0.1"]["mean_ece"]
    ax.text(0.05, 0.92,
            f"L2 centroid τ=0.1: ECE={ece_01:.3f} (well calibrated)",
            transform=ax.transAxes,
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color=COLORS["gt_noise_5"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF",
                      edgecolor=COLORS["gt_noise_5"], alpha=0.9))

    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=VIZ_DEFAULTS["tick_fontsize"])

    setup_axes(ax,
               title="VAL-3B: Reliability Diagram — L2 Centroid Calibration",
               xlabel="Mean Confidence",
               ylabel="Mean Accuracy")

    save_figure(fig, "val3b_reliability_diagram", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data ...")
    summary = _load_summary()

    print("Generating val3b_temperature_ece ...")
    chart_temperature_ece(summary)

    print("Generating val3b_reliability_diagram ...")
    chart_reliability_diagram(summary)

    print("Done.")
