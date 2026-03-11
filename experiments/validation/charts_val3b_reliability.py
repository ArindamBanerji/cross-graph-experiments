"""
VAL-3B: Consolidated publication reliability diagram.

Two-panel side-by-side:
  Left:  L2 centroid tau=0.25 (poorly calibrated, ECE=0.190)
  Right: L2 centroid tau=0.10 (well calibrated,   ECE=0.036)

tau=0.25 bins are available in calibration_summary.json.
tau=0.10 bins are recomputed here (not stored in JSON) by running the
L2 centroid model on the same seeds/data used in run_calibration_analysis.py.

Outputs (paper_figures/):
  val_3b_reliability_diagram.{pdf,png}
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

# Import data-generation helpers from the original experiment modules.
# These are the exact same helpers used in run_calibration_analysis.py so
# that the tau=0.1 bins match what the run would have produced.
from experiments.validation.run_baseline_comparison import (
    _load_config, _generate, _build_mu_from_data,
    SEEDS, N_TRAIN_STATIC, N_TEST_STATIC, N_ACTIONS,
)
from experiments.validation.run_calibration_analysis import (
    _l2_probs, compute_ece,
)

VAL_DIR    = Path(__file__).resolve().parent
PAPER_FIGS = ROOT / "paper_figures"

N_BINS      = 10
BIN_BOUNDS  = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTERS = (BIN_BOUNDS[:-1] + BIN_BOUNDS[1:]) / 2

COLOR_LINE = COLORS["gt_noise_5"]     # blue  "#2563EB" — reliability points/line
COLOR_DIAG = "#94A3B8"                # slate — diagonal reference


# ---------------------------------------------------------------------------
# Compute tau=0.1 bins (not stored in JSON)
# ---------------------------------------------------------------------------

def _compute_tau01_bins(bc: dict, rp: dict) -> list[dict]:
    """
    Run L2 centroid at tau=0.1 for all seeds; return aggregated bin data
    (weighted average across seeds, same method as run_calibration_analysis).
    """
    all_seed_bins: list[list[dict]] = []

    for seed in SEEDS:
        alerts = _generate(seed, N_TRAIN_STATIC + N_TEST_STATIC, bc, rp)
        train  = alerts[:N_TRAIN_STATIC]
        test   = alerts[N_TEST_STATIC:]          # mirrors run_calibration_analysis
        y_true = np.array([a.gt_action_index for a in test])

        mu    = _build_mu_from_data(train)
        probs = np.vstack([_l2_probs(mu, a, 0.1) for a in test])
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)

        _, bdata = compute_ece(confs, preds, y_true, n_bins=N_BINS)
        all_seed_bins.append(bdata)

    return _agg_bins(all_seed_bins)


def _agg_bins(bin_lists: list[list[dict]]) -> list[dict]:
    """Weighted-average bin accuracy and confidence across seeds."""
    agg = []
    for b in range(N_BINS):
        total = sum(bl[b]["count"] for bl in bin_lists)
        if total == 0:
            agg.append({"bin": b, "count": 0,
                        "mean_accuracy": 0.0, "mean_confidence": 0.0})
            continue
        w_acc  = sum(bl[b]["accuracy"]   * bl[b]["count"] for bl in bin_lists)
        w_conf = sum(bl[b]["confidence"] * bl[b]["count"] for bl in bin_lists)
        agg.append({
            "bin":             b,
            "count":           total,
            "mean_accuracy":   w_acc  / total,
            "mean_confidence": w_conf / total,
        })
    return agg


# ---------------------------------------------------------------------------
# Draw a single reliability panel
# ---------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    bins: list[dict],
    title: str,
    fill_color: str,
    annotation: str,
) -> None:
    """
    Draw reliability diagram on ax.
    - Gray dashed diagonal (perfect calibration)
    - Blue filled circles connected by line for non-empty bins
    - fill_between reliability line and diagonal (fill_color)
    - Count bars along bottom via twinx (gray, semi-transparent)
    """

    # -- Separate non-empty bins --
    valid = [(b["mean_confidence"], b["mean_accuracy"], b["count"])
             for b in bins if b["count"] > 0 and b["mean_confidence"] > 0]

    conf_v  = np.array([v[0] for v in valid])
    acc_v   = np.array([v[1] for v in valid])

    # -- Count bars: twinx, behind the reliability data --
    ax2 = ax.twinx()
    counts_all = [b["count"] for b in bins]
    max_count  = max(counts_all) if max(counts_all) > 0 else 1
    ax2.bar(BIN_CENTERS, counts_all,
            width=0.085, color="#94A3B8", alpha=0.25, zorder=1)
    ax2.set_ylim(0, max_count * 5)   # bars occupy bottom ~20% of panel
    ax2.set_yticks([])
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # Bring main axis forward so it renders on top
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_alpha(0.0)

    # -- Perfect calibration diagonal --
    ax.plot([0, 1], [0, 1], color=COLOR_DIAG, linewidth=1.3,
            linestyle="--", zorder=2, label="Perfect calibration")

    # -- Shading between diagonal and reliability curve --
    if len(conf_v) > 1:
        ax.fill_between(conf_v, conf_v, acc_v,
                        color=fill_color, alpha=0.18, zorder=3)

    # -- Reliability line + scatter --
    if len(conf_v) > 0:
        ax.plot(conf_v, acc_v,
                color=COLOR_LINE, linewidth=1.8, zorder=4)
        ax.scatter(conf_v, acc_v,
                   color=COLOR_LINE, s=55, zorder=5, marker="o")

    # -- Annotation box in panel center --
    ax.text(0.50, 0.12, annotation,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.90))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=VIZ_DEFAULTS["title_fontsize"] - 1, pad=6)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_reliability_figure(bins_025: list[dict],
                            bins_01:  list[dict]) -> None:

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20,
                        top=0.84, wspace=0.30)

    # -- Left panel: tau=0.25 (poorly calibrated) --
    _draw_panel(
        ax_l, bins_025,
        title="L2 Centroid tau=0.25  --  Underconfident (ECE=0.190)",
        fill_color="#DC2626",    # red
        annotation="All bins: accuracy > confidence",
    )

    # -- Right panel: tau=0.1 (well calibrated) --
    _draw_panel(
        ax_r, bins_01,
        title="L2 Centroid tau=0.1  --  Well Calibrated (ECE=0.036)",
        fill_color="#059669",    # green
        annotation="ECE=0.036 -- confidence matches accuracy",
    )

    # -- Shared axis labels --
    fig.text(0.525, 0.09, "Mean Predicted Confidence",
             ha="center", va="top",
             fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax_l.set_ylabel("Actual Accuracy",
                    fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax_r.set_ylabel("")   # shared label on left only

    # -- Super-title --
    fig.suptitle("Reliability Diagrams: Temperature Tuning Fixes Calibration",
                 fontsize=VIZ_DEFAULTS["title_fontsize"] + 1,
                 y=0.97)

    # -- Caption --
    fig.text(
        0.525, 0.03,
        ("Both panels: same model, same data, same accuracy.  "
         "Only tau changes.  "
         "ECE drops from 0.190 to 0.036 -- from poorly calibrated to well calibrated.  "
         "Confidence scores become trustworthy for automation gating."),
        ha="center", va="top",
        fontsize=VIZ_DEFAULTS["annotation_fontsize"],
        color="#475569",
        style="italic",
    )

    save_figure(fig, "val_3b_reliability_diagram", output_dir=str(PAPER_FIGS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading calibration summary (tau=0.25 bins) ...")
    with open(VAL_DIR / "calibration_summary.json") as fh:
        summary = json.load(fh)
    bins_025 = summary["l2_centroid"]["reliability_bins"]

    print("Loading config ...")
    bc, rp = _load_config()

    print("Computing tau=0.1 bins across 10 seeds ...")
    bins_01 = _compute_tau01_bins(bc, rp)

    print("Generating val_3b_reliability_diagram ...")
    make_reliability_figure(bins_025, bins_01)
    print("Done.")
