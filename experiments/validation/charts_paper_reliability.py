"""
Publication reliability diagram: L2 tau=0.25 vs tau=0.1 vs XGBoost.

Data sources:
  - tau=0.25 bins: calibration_summary.json  (l2_centroid.reliability_bins)
  - XGBoost bins:  calibration_summary.json  (xgboost.reliability_bins)
  - tau=0.1 bins:  recomputed via helpers from charts_val3b_reliability.py
                   (same fixed seeds, deterministic — not a new experiment)

Outputs (paper_figures/):
  fig5_reliability_diagram.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent.parent
VAL_DIR = Path(__file__).resolve().parent
PAPER_FIGS = ROOT / "paper_figures"

sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# tau=0.1 bins — import the existing computation from charts_val3b_reliability
# ---------------------------------------------------------------------------
# This imports the already-written deterministic helper; same seeds, same result.
from experiments.validation.charts_val3b_reliability import (
    _compute_tau01_bins,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_bins(bins: list[dict]) -> tuple[list[float], list[float]]:
    """Return (conf_vals, acc_vals) for non-empty, non-zero-confidence bins."""
    valid = [(b["mean_confidence"], b["mean_accuracy"])
             for b in bins if b["count"] > 0 and b["mean_confidence"] > 0]
    if not valid:
        return [], []
    conf, acc = zip(*valid)
    return list(conf), list(acc)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_reliability_figure(summary: dict, bins_01: list[dict]) -> None:

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.titlesize": 13,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.11, top=0.90)

    # -----------------------------------------------------------------------
    # Extract bins
    # -----------------------------------------------------------------------
    conf_025, acc_025 = _valid_bins(summary["l2_centroid"]["reliability_bins"])
    conf_01,  acc_01  = _valid_bins(bins_01)
    conf_xgb, acc_xgb = _valid_bins(summary["xgboost"]["reliability_bins"])

    conf_025 = np.array(conf_025); acc_025 = np.array(acc_025)
    conf_01  = np.array(conf_01);  acc_01  = np.array(acc_01)
    conf_xgb = np.array(conf_xgb); acc_xgb = np.array(acc_xgb)

    # -----------------------------------------------------------------------
    # Perfect calibration diagonal
    # -----------------------------------------------------------------------
    ax.plot([0, 1], [0, 1],
            color="#94A3B8", linewidth=1.4, linestyle="--",
            zorder=1, label="Perfect calibration")

    # -----------------------------------------------------------------------
    # Shading: tau=0.25 gap above diagonal (underconfident — acc > conf)
    # -----------------------------------------------------------------------
    if len(conf_025) > 1:
        ax.fill_between(conf_025, conf_025, acc_025,
                        color="#DC2626", alpha=0.13, zorder=2,
                        label="_nolegend_")

    # -----------------------------------------------------------------------
    # XGBoost — green squares
    # -----------------------------------------------------------------------
    ece_xgb = summary["xgboost"]["mean_ece"]
    if len(conf_xgb) > 0:
        ax.plot(conf_xgb, acc_xgb,
                color="#059669", linewidth=1.5, linestyle="-",
                alpha=0.75, zorder=3)
        ax.scatter(conf_xgb, acc_xgb,
                   color="#059669", s=55, marker="s", zorder=4,
                   label=f"XGBoost (ECE={ece_xgb:.3f})")

    # -----------------------------------------------------------------------
    # L2 tau=0.25 — red circles
    # -----------------------------------------------------------------------
    ece_025 = summary["l2_centroid"]["mean_ece"]
    if len(conf_025) > 0:
        ax.plot(conf_025, acc_025,
                color="#DC2626", linewidth=1.8, linestyle="-",
                zorder=5)
        ax.scatter(conf_025, acc_025,
                   color="#DC2626", s=60, marker="o", zorder=6,
                   label=f"L2 \u03c4=0.25 (ECE={ece_025:.3f})")

    # -----------------------------------------------------------------------
    # L2 tau=0.1 — blue circles
    # -----------------------------------------------------------------------
    ece_01 = summary["temperature_sensitivity"]["0.1"]["mean_ece"]
    if len(conf_01) > 0:
        ax.plot(conf_01, acc_01,
                color="#2563EB", linewidth=1.8, linestyle="-",
                zorder=7)
        ax.scatter(conf_01, acc_01,
                   color="#2563EB", s=60, marker="o", zorder=8,
                   label=f"L2 \u03c4=0.1 (ECE={ece_01:.3f})")

    # -----------------------------------------------------------------------
    # Axis limits, ticks, spines
    # -----------------------------------------------------------------------
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -----------------------------------------------------------------------
    # Labels + title
    # -----------------------------------------------------------------------
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Mean Actual Accuracy", fontsize=12)
    ax.set_title(
        "Reliability Diagram: Calibration at \u03c4=0.25 vs \u03c4=0.1",
        fontsize=13, pad=10
    )

    # -----------------------------------------------------------------------
    # Annotation: underconfidence label for tau=0.25
    # -----------------------------------------------------------------------
    ax.text(
        0.38, 0.82,
        "Underconfident\n(acc > confidence)",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=9.5,
        color="#DC2626",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2",
                  edgecolor="#FCA5A5", alpha=0.88),
    )

    # -----------------------------------------------------------------------
    # Legend: lower right
    # -----------------------------------------------------------------------
    ax.legend(
        fontsize=10,
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
        out = PAPER_FIGS / f"fig5_reliability_diagram.{ext_fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading calibration_summary.json ...")
    with open(VAL_DIR / "calibration_summary.json") as fh:
        summary = json.load(fh)

    print("Computing tau=0.1 bins (deterministic, 10 fixed seeds) ...")
    from experiments.validation.run_baseline_comparison import _load_config
    bc, rp = _load_config()
    bins_01 = _compute_tau01_bins(bc, rp)
    print(f"  tau=0.1 bins computed: {sum(1 for b in bins_01 if b['count'] > 0)} non-empty bins")

    print("Generating fig5_reliability_diagram ...")
    make_reliability_figure(summary, bins_01)
    print("Done.")
