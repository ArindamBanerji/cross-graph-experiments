"""
Shared visualization utilities for Bridge Layer Experiments (EXP 5-9).

Provides a consistent color palette, layout defaults, and helper functions
used across all bridge experiment charts, ensuring visual coherence and
publication-quality output (300 DPI, PDF + PNG).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # file-only backend; must precede pyplot import

import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS: dict = {
    # Oracle types
    "bernoulli":        "#94A3B8",   # gray — legacy biased oracle
    "gt_noise_0":       "#1E3A5F",   # dark blue — perfect GT oracle
    "gt_noise_5":       "#2563EB",   # blue — 5% noise
    "gt_noise_15":      "#7C3AED",   # purple — 15% noise
    "gt_noise_30":      "#DC2626",   # red — 30% noise
    # Gating mechanisms
    "uniform_gate":     "#94A3B8",   # gray — no gating baseline
    "hebbian_damped":   "#1E3A5F",   # dark blue — damped Hebbian
    "hebbian_undamped": "#D97706",   # amber — undamped Hebbian
    "mi_static":        "#059669",   # green — offline MI gating
    # Per-category palette (5 categories)
    "category_colors":  ["#1E3A5F", "#D97706", "#059669", "#DC2626", "#7C3AED"],
}


# ---------------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------------

VIZ_DEFAULTS: dict = {
    "dpi":                300,
    "title_fontsize":     13,
    "label_fontsize":     11,
    "tick_fontsize":      9,
    "annotation_fontsize": 8.5,
    "figsize_single":     (8, 5),
    "figsize_wide":       (12, 5),
    "figsize_heatmap":    (10, 6),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def setup_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Apply standard title, axis labels, font sizes, and spine cleanup.

    Removes the top and right spines for a clean publication look.
    All font sizes are taken from ``VIZ_DEFAULTS``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    title : str
    xlabel : str
    ylabel : str
    """
    ax.set_title(title,  fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax.set_xlabel(xlabel, fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.tick_params(labelsize=VIZ_DEFAULTS["tick_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: str = "paper_figures",
) -> None:
    """
    Save *fig* as both PDF and PNG at 300 DPI, print the paths, then close.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Base filename without extension (e.g. ``"exp5_convergence"``).
    output_dir : str
        Directory for output files.  Created if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=VIZ_DEFAULTS["dpi"], bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def load_bridge_config(config_path: str = "configs/default.yaml") -> dict:
    """
    Load and return the ``bridge_common`` section from the YAML config.

    Parameters
    ----------
    config_path : str
        Path to the project YAML config (relative to cwd or absolute).

    Returns
    -------
    dict
        Contents of the ``bridge_common`` key.
    """
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"]


def get_category_color(category_index: int) -> str:
    """
    Return the hex color string for the given category index (0–4).

    Parameters
    ----------
    category_index : int
        Index into the five SOC copilot categories.

    Returns
    -------
    str
        Hex color string, e.g. ``"#1E3A5F"``.
    """
    return COLORS["category_colors"][category_index]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # a+b. Create a figure with setup_axes and save it
    fig, ax = plt.subplots(figsize=VIZ_DEFAULTS["figsize_single"])
    ax.plot([0, 1, 2], [0, 1, 0], color=COLORS["gt_noise_0"], linewidth=2)
    setup_axes(ax, "Bridge Test Chart", "Step", "Value")
    save_figure(fig, "bridge_test")

    # c. Assert both files were created, then clean up
    for ext in ("pdf", "png"):
        p = Path("paper_figures") / f"bridge_test.{ext}"
        assert p.exists(), f"FAIL: {p} was not created by save_figure()"
        p.unlink()
        print(f"  Deleted test file: {p}")

    # d. load_bridge_config: verify expected keys are present
    cfg = load_bridge_config()
    required_keys = ("n_categories", "categories", "actions", "factors",
                     "seeds", "scoring", "category_factor_means",
                     "category_gt_distributions")
    for key in required_keys:
        assert key in cfg, f"FAIL: bridge_common missing key '{key}'"
    print(f"\n  bridge_common keys present: {list(cfg.keys())}")
    assert cfg["n_categories"] == 5, (
        f"FAIL: n_categories={cfg['n_categories']} expected 5"
    )

    # e. get_category_color: all five return valid hex strings
    colors = [get_category_color(i) for i in range(5)]
    for i, c in enumerate(colors):
        assert isinstance(c, str) and c.startswith("#"), (
            f"FAIL: get_category_color({i}) returned {c!r}"
        )
    print(f"  Category colors: {colors}")

    print("\nAll checks passed")
