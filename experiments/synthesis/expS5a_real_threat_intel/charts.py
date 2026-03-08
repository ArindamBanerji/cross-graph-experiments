"""
EXP-S5a charts — 4 publication-quality charts from results.json
experiments/synthesis/expS5a_real_threat_intel/charts.py

Chart 1: sigma heatmap (real threat intel → sigma tensor)
Chart 2: category distribution (KEV + NVD claim coverage)
Chart 3: urgency distribution (histogram by source)
Chart 4: source volume comparison (KEV vs NVD)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.viz.synthesis_common import (
    FIGURE_DEFAULTS, load_results, plot_sigma_heatmap,
)

# Canonical SOC taxonomy for this experiment
CATEGORIES = [
    "travel_anomaly", "credential_access", "threat_intel_match",
    "insider_behavioral", "cloud_infrastructure",
]
ACTIONS = ["escalate", "investigate", "suppress", "monitor"]


def _save(fig: plt.Figure, stem: str, dirs: List[str]) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{stem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(Path(d) / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: sigma heatmap
# ---------------------------------------------------------------------------

def _chart1_sigma_heatmap(results: dict, dirs: List[str]) -> None:
    sigma = np.array(results["sigma_tensor"])
    sources = results["data_sources"]
    n_active = sources["active_after_threshold"]

    plot_sigma_heatmap(
        sigma=sigma,
        title=(
            f"EXP-S5a: Synthesis Bias sigma from Real Threat Intel\n"
            f"(CISA KEV + NVD — {n_active} active claims, lambda=1.0 for inspection)"
        ),
        save_path=str(Path(dirs[0]) / "expS5a_sigma_heatmap.png"),
        categories=CATEGORIES,
        actions=ACTIONS,
    )
    # Also save to paper_figures if second dir given
    if len(dirs) > 1:
        plot_sigma_heatmap(
            sigma=sigma,
            title=(
                f"EXP-S5a: Synthesis Bias sigma from Real Threat Intel\n"
                f"(CISA KEV + NVD — {n_active} active claims, lambda=1.0 for inspection)"
            ),
            save_path=str(Path(dirs[1]) / "expS5a_sigma_heatmap.png"),
            categories=CATEGORIES,
            actions=ACTIONS,
        )
    # Save PDF copies
    for d in dirs:
        plot_sigma_heatmap(
            sigma=sigma,
            title=(
                f"EXP-S5a: Synthesis Bias sigma from Real Threat Intel\n"
                f"(CISA KEV + NVD — {n_active} active claims)"
            ),
            save_path=str(Path(d) / "expS5a_sigma_heatmap.pdf"),
            categories=CATEGORIES,
            actions=ACTIONS,
        )


# ---------------------------------------------------------------------------
# Chart 2: category distribution
# ---------------------------------------------------------------------------

def _chart2_category_distribution(results: dict, dirs: List[str]) -> None:
    cat_dist = results["category_distribution"]
    sources  = results["data_sources"]

    cat_names  = list(cat_dist.keys())
    cat_counts = [cat_dist[c] for c in cat_names]
    colors     = ["#2196F3" if v > 0 else "#CCCCCC" for v in cat_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cat_names, cat_counts, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, cat_counts):
        if val > 0:
            ax.text(
                bar.get_width() + max(cat_counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, fontweight="bold",
            )
    ax.set_xlabel("Number of claims affecting category", fontsize=10)
    ax.set_title(
        "EXP-S5a: Real Threat Intel — Category Coverage\n"
        f"Total: {sources['total_claims']} claims, "
        f"{sources['active_after_threshold']} active after threshold",
        fontweight="bold", fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, max(cat_counts, default=1) * 1.15)
    plt.tight_layout()
    _save(fig, "expS5a_category_distribution", dirs)


# ---------------------------------------------------------------------------
# Chart 3: urgency distribution
# ---------------------------------------------------------------------------

def _chart3_urgency_distribution(results: dict, dirs: List[str]) -> None:
    urgency_vals = results.get("urgency_stats", {}).get("values", [])
    if not urgency_vals:
        return

    sources = results["data_sources"]
    n_kev = sources["kev_claims"]
    n_nvd = sources["nvd_claims"]
    total = n_kev + n_nvd

    # Split by source order: KEV first, then NVD
    kev_urgencies = urgency_vals[:n_kev]
    nvd_urgencies = urgency_vals[n_kev: n_kev + n_nvd]

    bins = np.linspace(0.0, 1.0, 11)

    fig, ax = plt.subplots(figsize=(9, 5))
    if kev_urgencies:
        ax.hist(kev_urgencies, bins=bins, alpha=0.75, color="#F44336",
                label=f"CISA KEV (n={len(kev_urgencies)}, actively exploited)",
                edgecolor="white")
    if nvd_urgencies:
        ax.hist(nvd_urgencies, bins=bins, alpha=0.75, color="#FF9800",
                label=f"NVD (n={len(nvd_urgencies)}, CVSS-derived)",
                edgecolor="white")

    mean_all = float(np.mean(urgency_vals)) if urgency_vals else 0.0
    ax.axvline(mean_all, color="#333333", linewidth=1.5, linestyle="--",
               label=f"Overall mean={mean_all:.3f}")

    ax.set_xlabel("Urgency score (0=low, 1=critical)", fontsize=10)
    ax.set_ylabel("Number of claims", fontsize=10)
    ax.set_title(
        f"EXP-S5a: Claim Urgency Distribution\n"
        f"({total} total claims: KEV={n_kev} + NVD={n_nvd})",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    _save(fig, "expS5a_urgency_distribution", dirs)


# ---------------------------------------------------------------------------
# Chart 4: source volume comparison
# ---------------------------------------------------------------------------

def _chart4_source_comparison(results: dict, dirs: List[str]) -> None:
    sources = results["data_sources"]

    src_names  = ["CISA KEV\n(actively exploited)", "NVD\n(recent CVEs)"]
    src_counts = [sources["kev_claims"], sources["nvd_claims"]]
    colors     = ["#F44336", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(src_names, src_counts, color=colors, alpha=0.85,
                  width=0.5, edgecolor="white")
    for bar, val in zip(bars, src_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(src_counts, default=1) * 0.02,
            str(val), ha="center", va="bottom",
            fontsize=13, fontweight="bold",
        )

    # Active threshold line
    active = sources["active_after_threshold"]
    total  = sources["total_claims"]
    ax.axhline(active, color="#4CAF50", linewidth=1.5, linestyle=":",
               label=f"Active after threshold: {active} / {total}")

    ax.set_ylabel("Number of claims", fontsize=10)
    ax.set_title(
        "EXP-S5a: Data Source Volume (before threshold filtering)",
        fontweight="bold", fontsize=11,
    )
    ax.set_ylim(0, max(src_counts, default=1) * 1.2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, "expS5a_source_comparison", dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_all_charts(results: dict, exp_dir: str, paper_dir: str) -> None:
    dirs = [exp_dir, paper_dir]

    _chart1_sigma_heatmap(results, dirs)
    _chart2_category_distribution(results, dirs)
    _chart3_urgency_distribution(results, dirs)
    _chart4_source_comparison(results, dirs)

    print(f"EXP-S5a: 4 charts (PNG + PDF) saved to {exp_dir} and {paper_dir}")


if __name__ == "__main__":
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        sys.exit(1)
    results = load_results(str(results_path))
    make_all_charts(
        results,
        str(Path(__file__).parent),
        str(REPO_ROOT / "paper_figures"),
    )
