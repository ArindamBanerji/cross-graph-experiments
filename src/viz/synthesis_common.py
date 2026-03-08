"""
synthesis_common.py — Shared visualization utilities for synthesis experiments

SYNTH-EXP-0 deliverable. Lives in src/viz/ of cross-graph-experiments.
Used by EXP-S1 through S5b chart modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on Windows + headless)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

LAMBDA_COLORS: Dict[float, str] = {
    0.000: "#666666",   # Baseline: grey
    0.025: "#3182bd",
    0.050: "#2196F3",   # Primary blue
    0.075: "#21bdd3",
    0.100: "#4CAF50",   # Green (recommended default)
    0.125: "#8BC34A",
    0.150: "#CDDC39",
    0.175: "#FFC107",
    0.200: "#FF9800",   # Orange (caution)
    0.250: "#F44336",   # Red (high)
    0.300: "#B71C1C",
    0.350: "#880E4F",
    0.400: "#4A148C",
    0.450: "#1A237E",
    0.500: "#0D47A1",
}

POISON_COLORS: Dict[str, str] = {
    "clean_0pct":   "#4CAF50",   # Green
    "poison_20pct": "#FF9800",   # Orange
    "poison_40pct": "#F44336",   # Red
}

CATEGORY_COLORS: Dict[str, str] = {
    "travel_anomaly":       "#2196F3",
    "credential_access":    "#F44336",
    "threat_intel_match":   "#FF9800",
    "insider_behavioral":   "#9C27B0",
    "cloud_infrastructure": "#4CAF50",
    "healthcare":           "#00BCD4",
}

FIGURE_DEFAULTS = {
    "figsize_standard":  (10, 6),
    "figsize_wide":      (14, 6),
    "figsize_heatmap":   (10, 7),
    "figsize_2panel":    (14, 6),
    "dpi":               150,
    "font_size_title":   13,
    "font_size_labels":  11,
    "font_size_ticks":   9,
}

ACTIONS = ["auto_close", "escalate_tier2", "enrich_and_watch", "escalate_incident"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration", "insider_threat",
]


# ---------------------------------------------------------------------------
# Common plot setup
# ---------------------------------------------------------------------------

def setup_figure(
    title: str,
    figsize: Tuple[float, float] = None,
    tight: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling."""
    figsize = figsize or FIGURE_DEFAULTS["figsize_standard"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=FIGURE_DEFAULTS["font_size_title"], fontweight="bold")
    ax.tick_params(labelsize=FIGURE_DEFAULTS["font_size_ticks"])
    return fig, ax


def setup_subplots(
    nrows: int,
    ncols: int,
    title: str,
    figsize: Tuple[float, float] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    figsize = figsize or (FIGURE_DEFAULTS["figsize_wide"][0] * ncols / 2,
                          FIGURE_DEFAULTS["figsize_standard"][1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=FIGURE_DEFAULTS["font_size_title"], fontweight="bold")
    return fig, axes


def save_figure(
    fig: plt.Figure,
    name: str,
    experiment_dir: str,
    paper_figures_dir: str = "paper_figures",
) -> None:
    """Save figure to both the experiment dir and paper_figures/."""
    for directory in [experiment_dir, paper_figures_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = os.path.join(directory, name)
        fig.savefig(path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reusable plot functions
# ---------------------------------------------------------------------------

def plot_accuracy_by_lambda(
    lambdas: List[float],
    mean_accs: List[float],
    std_accs: List[float],
    baseline_acc: float,
    title: str,
    save_path: str,
    plateau_range: Optional[Tuple[float, float]] = None,
    lambda_peak: Optional[float] = None,
    gate_threshold: Optional[float] = None,
) -> None:
    """
    Standard lambda sweep accuracy chart.
    Used by EXP-S1 and EXP-S4.
    """
    fig, ax = setup_figure(title, figsize=FIGURE_DEFAULTS["figsize_standard"])
    
    # Mean ± std band
    lambdas_arr = np.array(lambdas)
    mean_arr = np.array(mean_accs)
    std_arr = np.array(std_accs)

    ax.plot(lambdas_arr, mean_arr, "o-", color="#4CAF50", linewidth=2, markersize=6,
            label="Mean accuracy (10 seeds)")
    ax.fill_between(lambdas_arr, mean_arr - std_arr, mean_arr + std_arr,
                    alpha=0.2, color="#4CAF50", label="±1 std")

    # Baseline
    ax.axhline(baseline_acc, color="#666666", linestyle="--", linewidth=1.5,
               label=f"Baseline (λ=0): {baseline_acc:.2f}%")

    # Gate threshold
    if gate_threshold is not None:
        ax.axhline(gate_threshold, color="#F44336", linestyle=":", linewidth=1.5,
                   label=f"Gate threshold: {gate_threshold:.2f}%", alpha=0.7)

    # Plateau shading
    if plateau_range is not None:
        ax.axvspan(plateau_range[0], plateau_range[1], alpha=0.12, color="#4CAF50",
                   label=f"Plateau: [{plateau_range[0]:.3f}, {plateau_range[1]:.3f}]")

    # Lambda peak marker
    if lambda_peak is not None:
        ax.axvline(lambda_peak, color="#FF9800", linestyle="--", linewidth=1.5,
                   label=f"λ_peak = {lambda_peak:.3f}", alpha=0.8)

    ax.set_xlabel("Coupling constant λ", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.set_ylabel("Accuracy (%)", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(lambdas) - 0.01, max(lambdas) + 0.01)

    plt.tight_layout()
    # Save directly (caller passes full path)
    fig.savefig(save_path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    save_path: str,
    center: float = None,
    vmin: float = None,
    vmax: float = None,
    fmt: str = ".2f",
    cmap: str = "RdYlGn",
) -> None:
    """
    Category × column heatmap (used by EXP-S1, S3, S4).
    """
    fig, ax = setup_figure(title, figsize=FIGURE_DEFAULTS["figsize_heatmap"])

    im = ax.imshow(
        data, cmap=cmap, aspect="auto",
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=FIGURE_DEFAULTS["font_size_ticks"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FIGURE_DEFAULTS["font_size_ticks"])

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if abs(data[i, j]) > (vmax or data.max()) * 0.6 else "black"
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_sigma_heatmap(
    sigma: np.ndarray,
    title: str,
    save_path: str,
    categories: List[str] = None,
    actions: List[str] = None,
) -> None:
    """
    Standard 6×4 σ tensor heatmap with diverging colormap.
    Used across multiple experiments.
    """
    cats = categories or CATEGORIES
    acts = actions or ACTIONS
    plot_category_heatmap(
        data=sigma,
        row_labels=cats,
        col_labels=acts,
        title=title,
        save_path=save_path,
        cmap="RdBu_r",  # Diverging: red=positive (suppress), blue=negative (escalate)
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        fmt=".3f",
    )


def plot_ece_by_lambda(
    lambdas: List[float],
    mean_ece: List[float],
    std_ece: List[float],
    baseline_ece: float,
    title: str,
    save_path: str,
    gate_ceiling: float = 0.02,
) -> None:
    """
    ECE vs lambda chart with gate threshold.
    Used by EXP-S1.
    """
    fig, ax = setup_figure(title, figsize=FIGURE_DEFAULTS["figsize_standard"])

    ax.plot(lambdas, mean_ece, "s-", color="#2196F3", linewidth=2, markersize=6,
            label="Mean ECE (10 seeds)")
    arr = np.array(mean_ece)
    std_arr = np.array(std_ece)
    ax.fill_between(lambdas, arr - std_arr, arr + std_arr, alpha=0.2, color="#2196F3")

    ax.axhline(baseline_ece, color="#666666", linestyle="--", linewidth=1.5,
               label=f"Baseline ECE (λ=0): {baseline_ece:.4f}")
    ax.axhspan(baseline_ece + gate_ceiling, ax.get_ylim()[1] if ax.get_ylim()[1] > baseline_ece + gate_ceiling else baseline_ece + gate_ceiling + 0.01,
               alpha=0.12, color="#F44336", label=f"FAIL zone: ECE > baseline + {gate_ceiling}")

    ax.set_xlabel("Coupling constant λ", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_action_shift(
    categories: List[str],
    prob_baseline: np.ndarray,
    prob_synthesis: np.ndarray,
    action_name: str,
    action_index: int,
    title: str,
    save_path: str,
) -> None:
    """
    Grouped bar chart: P(action) baseline vs synthesis, per category.
    Used by EXP-S1 and EXP-S5a/b.
    """
    fig, ax = setup_figure(title, figsize=FIGURE_DEFAULTS["figsize_wide"])

    n_cats = len(categories)
    x = np.arange(n_cats)
    width = 0.35

    ax.bar(x - width/2, prob_baseline[:, action_index] * 100, width,
           label=f"λ=0 (baseline)", color="#666666", alpha=0.8)
    ax.bar(x + width/2, prob_synthesis[:, action_index] * 100, width,
           label=f"With synthesis", color="#4CAF50", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right",
                       fontsize=FIGURE_DEFAULTS["font_size_ticks"])
    ax.set_ylabel(f"P({action_name}) %", fontsize=FIGURE_DEFAULTS["font_size_labels"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results I/O utilities
# ---------------------------------------------------------------------------

def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def save_results(results: dict, path: str) -> None:
    """Save experiment results to JSON (with Path creation)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)


def print_gate_result(experiment_id: str, passed: bool, details: str) -> None:
    """Standardized gate result printing."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nGATE-{experiment_id}: {status}")
    print(f"  {details}")
