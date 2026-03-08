"""
charts.py — Publication-quality figures for FX-2.
experiments/fx2_noise_distributions/charts.py

2 charts:
  1. fx2_accuracy_trajectories — 3-panel rolling accuracy per bias pattern
  2. fx2_centroid_drift        — bar chart of centroid drift per pattern
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

from src.viz.bridge_common import VIZ_DEFAULTS, save_figure
from experiments.fx2_noise_distributions.bias_generator import (
    BiasPattern,
    BiasedFeedbackSimulator,
)

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

PATTERN_META = {
    BiasPattern.POST_INCIDENT_ESCALATION.value: {
        "label":  "Post-Incident\nEscalation",
        "color":  "#DC2626",
        "short":  "Post-Incident",
    },
    BiasPattern.ALERT_FATIGUE.value: {
        "label":  "Alert\nFatigue",
        "color":  "#D97706",
        "short":  "Alert Fatigue",
    },
    BiasPattern.EXPERTISE_GRADIENT.value: {
        "label":  "Expertise\nGradient",
        "color":  "#7C3AED",
        "short":  "Expertise Gradient",
    },
}

REF_STATIC_REALISTIC = 0.7145
DRIFT_WARNING        = 0.15    # design guard threshold
DRIFT_LOOP2_REF      = 0.0028  # EXP-S3 Loop 2 firewall reference


# ---------------------------------------------------------------------------
# Chart 1 — Accuracy trajectories (3 panels)
# ---------------------------------------------------------------------------

def chart_accuracy_trajectories(
    all_results: dict,
    patterns:    list[BiasPattern],
    n_decisions: int,
    window_size: int,
    incident_at: int,
    paper_dir:   str,
) -> None:
    """
    3-panel figure: rolling-window accuracy trajectory per bias pattern.
    Each panel shows: mean trajectory ±1std, reference baseline, bias event marker.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    decisions = np.arange(1, n_decisions + 1)

    for ax, pattern in zip(axes, patterns):
        pname = pattern.value
        meta  = PATTERN_META[pname]
        r     = all_results[pname]

        mean_roll = np.array(r["mean_roll"])
        std_roll  = np.array(r["std_roll"])

        # Main trajectory
        ax.plot(decisions, mean_roll * 100,
                color=meta["color"], linewidth=2.2, zorder=4,
                label=f"Rolling-{window_size} acc (mean)")
        ax.fill_between(decisions,
                        (mean_roll - std_roll) * 100,
                        (mean_roll + std_roll) * 100,
                        alpha=0.15, color=meta["color"], zorder=3, label="±1 std")

        # Reference: static realistic baseline
        ax.axhline(REF_STATIC_REALISTIC * 100,
                   color="#94A3B8", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2,
                   label=f"Static baseline: {REF_STATIC_REALISTIC*100:.2f}%")

        # Recovery threshold line
        ax.axhline(75.0,
                   color="#059669", linewidth=1.2, linestyle=":", alpha=0.8, zorder=2,
                   label="Recovery gate: 75%")

        # Event markers
        if pattern == BiasPattern.POST_INCIDENT_ESCALATION:
            ax.axvline(incident_at, color="#DC2626", linewidth=1.5,
                       linestyle="--", alpha=0.6, zorder=3, label=f"Incident @ dec {incident_at}")
            ax.axvspan(incident_at,
                       incident_at + BiasedFeedbackSimulator.POST_INCIDENT_WINDOW,
                       alpha=0.07, color="#DC2626", zorder=1, label="Bias window (50 dec)")
        elif pattern == BiasPattern.ALERT_FATIGUE:
            onset = BiasedFeedbackSimulator.FATIGUE_ONSET
            ax.axvline(onset, color="#D97706", linewidth=1.5,
                       linestyle="--", alpha=0.6, zorder=3, label=f"Fatigue onset @ dec {onset}")
            ax.axvspan(onset, n_decisions, alpha=0.05, color="#D97706", zorder=1)

        # Mark recovery if exists
        rec = r["recovery_dec_75"]
        if rec is not None and rec <= n_decisions:
            ax.axvline(rec, color="#059669", linewidth=1.2,
                       linestyle="-.", alpha=0.7, zorder=3,
                       label=f"Recovery @ dec {rec}")

        # Annotations: bias rate, final acc
        final_acc   = r["final_acc"] * 100
        bias_rate   = r["mean_bias_rate"] * 100
        min_acc_val = r["min_acc"] * 100
        ax.text(0.97, 0.04,
                f"Bias rate: {bias_rate:.1f}%\nMin acc: {min_acc_val:.1f}%\nFinal acc: {final_acc:.1f}%",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        all_vals = list(mean_roll * 100) + [REF_STATIC_REALISTIC * 100, 75.0]
        y_min    = max(0,   min(all_vals) - 5)
        y_max    = min(105, max(all_vals) + 6)
        ax.set_xlim(0, n_decisions + 10)
        ax.set_ylim(y_min, y_max)

        ax.set_title(meta["short"], fontsize=VIZ_DEFAULTS["title_fontsize"],
                     color=meta["color"], fontweight="bold")
        ax.set_xlabel("Decision Number", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.set_ylabel("Rolling Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
        ax.legend(fontsize=6.5, loc="lower right", ncol=1)
        ax.grid(linestyle="--", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])

    fig.suptitle(
        f"FX-2: Accuracy Under Analyst Bias Patterns (Combined Realistic Data)\n"
        f"Rolling-{window_size} accuracy. Grey dashed = static baseline ({REF_STATIC_REALISTIC*100:.2f}%). "
        f"Green dotted = 75% recovery gate.",
        fontsize=VIZ_DEFAULTS["title_fontsize"] + 0.5,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fx2_accuracy_trajectories", paper_dir)


# ---------------------------------------------------------------------------
# Chart 2 — Centroid drift bar chart
# ---------------------------------------------------------------------------

def chart_centroid_drift(
    all_results: dict,
    patterns:    list[BiasPattern],
    paper_dir:   str,
) -> None:
    """
    Bar chart: Frobenius centroid drift per bias pattern.
    Reference lines: 0.15 (design guard threshold), 0.0028 (EXP-S3 firewall).
    Annotate recovery decision or "No recovery" on each bar.
    """
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])

    x     = np.arange(len(patterns))
    width = 0.55
    labels = [PATTERN_META[p.value]["label"] for p in patterns]
    colors = [PATTERN_META[p.value]["color"] for p in patterns]

    for i, pattern in enumerate(patterns):
        pname = pattern.value
        r     = all_results[pname]
        drift = r["mean_drift"]
        std   = r["std_drift"]

        ax.bar(
            i, drift,
            width=width,
            color=colors[i],
            alpha=0.85,
            yerr=std,
            capsize=5,
            error_kw={"linewidth": 1.2, "ecolor": "#374151"},
            zorder=3,
        )

        # Annotate recovery on bar
        rec_75  = r["recovery_dec_75"]
        rec_str = f"Recovery: dec {rec_75}" if rec_75 is not None else "No recovery"
        ax.text(
            i, drift + std + 0.003,
            f"{drift:.4f}\n{rec_str}",
            ha="center", va="bottom",
            fontsize=VIZ_DEFAULTS["annotation_fontsize"],
            color=colors[i],
        )

    # Reference lines
    ax.axhline(DRIFT_WARNING, color="#DC2626", linewidth=1.5, linestyle=":",
               alpha=0.8, zorder=2, label=f"Design guard threshold: {DRIFT_WARNING}")
    ax.axhline(DRIFT_LOOP2_REF, color="#94A3B8", linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=2,
               label=f"EXP-S3 Loop 2 firewall: {DRIFT_LOOP2_REF}")

    # Shade above warning threshold
    all_drift = [all_results[p.value]["mean_drift"] for p in patterns]
    y_max = max(all_drift) * 1.35 + 0.03
    ax.axhspan(DRIFT_WARNING, y_max, alpha=0.04, color="#DC2626", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_ylim(bottom=0, top=y_max)
    ax.set_ylabel("Centroid Drift (Frobenius norm)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.set_title(
        "FX-2: Centroid Drift Under Analyst Bias\n"
        "Frobenius norm of μ_final − μ_initial. "
        f"Red dotted = design guard ({DRIFT_WARNING}). "
        f"Grey dashed = EXP-S3 ref ({DRIFT_LOOP2_REF}).",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    fig.tight_layout()
    save_figure(fig, "fx2_centroid_drift", paper_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_charts(
    all_results: dict,
    patterns:    list[BiasPattern],
    n_decisions: int,
    window_size: int,
    incident_at: int,
    paper_dir:   str,
) -> None:
    """Generate all FX-2 charts and save to paper_dir (PDF + PNG)."""
    print("\nGenerating FX-2 charts...")
    chart_accuracy_trajectories(
        all_results=all_results,
        patterns=patterns,
        n_decisions=n_decisions,
        window_size=window_size,
        incident_at=incident_at,
        paper_dir=paper_dir,
    )
    chart_centroid_drift(
        all_results=all_results,
        patterns=patterns,
        paper_dir=paper_dir,
    )
    print("All 2 charts saved (PDF + PNG).")
