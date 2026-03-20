"""
BLOCK-5B-PROXY: Cross-persona comparison charts.
Reads results/all_harness_results.json produced by run_all_harnesses.py.

Chart 1 — block5b_convergence_heatmap   (9 × 6 heatmap)
Chart 2 — block5b_tau_comparison        (bar chart by persona)
Chart 3 — block5b_accuracy_trajectories (9 lines over 60 days)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Colour/style mappings
INDUSTRY_COLORS = {
    "Financial Services": "#2166ac",
    "Healthcare":         "#d73027",
    "Technology":         "#4dac26",
}
JUDGE_STYLES = {
    "grok":    "solid",
    "gpt4o":   "dashed",
    "gemini":  "dotted",
}
JUDGE_LABELS = {
    "grok":  "Grok",
    "gpt4o": "GPT-4o",
    "gemini": "Gemini",
}

CATEGORIES_ABBREV = {
    "credential_access":    "cred_access",
    "threat_intel_match":   "threat_intel",
    "lateral_movement":     "lateral_mvmt",
    "data_exfiltration":    "data_exfil",
    "insider_threat":       "insider_threat",
    "cloud_infrastructure": "cloud_infra",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = PAPER_FIGS / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ── Chart 1: Convergence heatmap ─────────────────────────────────────────────
def chart_convergence_heatmap(data):
    results    = data["results"]
    persona_ids = list(results.keys())
    categories  = list(next(iter(results.values()))["prod5"]["categories"].keys())
    n_personas  = len(persona_ids)
    n_cats      = len(categories)

    # Build matrix: rows=personas, cols=categories
    # Value = mean_weeks; NaN = not converged (<80%)
    matrix = np.full((n_personas, n_cats), np.nan)
    for ri, pid in enumerate(persona_ids):
        for ci, cat in enumerate(categories):
            cr = results[pid]["prod5"]["categories"][cat]
            if cr["mean_weeks"] is not None:
                matrix[ri, ci] = cr["mean_weeks"]

    # Cap at 8.6 weeks (= 60 days) for colour scale
    MAX_WKS = 8.6
    plot_matrix = np.where(np.isnan(matrix), MAX_WKS + 1, np.minimum(matrix, MAX_WKS))

    # Colormap: green (fast) → yellow → red (slow) → white (no convergence)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.cm as cm
    base_cmap = cm.get_cmap("RdYlGn_r", 256)
    colors    = base_cmap(np.linspace(0, 1, 256))
    colors    = np.vstack([colors, [1, 1, 1, 1]])   # append white for NaN/not-converged
    cmap      = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(9, 5))
    vmin, vmax = 0, MAX_WKS + 1.5
    im = ax.imshow(plot_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Axis labels
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels([CATEGORIES_ABBREV.get(c, c) for c in categories],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_personas))

    # Persona row labels: ID + industry initial
    industry_initial = {"Financial Services": "F", "Healthcare": "H", "Technology": "T"}
    ylabels = []
    for pid in persona_ids:
        ind = results[pid]["industry"]
        ylabels.append(f"{pid} [{industry_initial.get(ind, '?')}]")
    ax.set_yticklabels(ylabels, fontsize=9)

    # Annotate cells
    for ri in range(n_personas):
        for ci in range(n_cats):
            val = matrix[ri, ci]
            if np.isnan(val):
                txt = "—"
                col = "gray"
            else:
                txt = f"{val:.1f}w"
                col = "white" if val > MAX_WKS * 0.6 else "black"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=7.5, color=col, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Weeks to convergence", fontsize=9)
    cbar.set_ticks([0, 2, 4, 6, MAX_WKS])
    cbar.set_ticklabels(["0", "2", "4", "6", "—"])
    cbar.ax.text(0.5, 1.05, "(white = no conv.)", transform=cbar.ax.transAxes,
                 fontsize=7, ha="center")

    ax.set_title("Per-Category Convergence by Customer Persona  (ε=0.10, 60-day sim)")
    fig.tight_layout()
    save(fig, "block5b_convergence_heatmap")


# ── Chart 2: Optimal τ by persona ────────────────────────────────────────────
def chart_tau_comparison(data):
    results     = data["results"]
    persona_ids = list(results.keys())
    n           = len(persona_ids)

    opt_taus  = [results[pid]["td034"]["optimal_tau"] for pid in persona_ids]
    industries = [results[pid]["industry"]            for pid in persona_ids]
    bar_colors = [INDUSTRY_COLORS.get(ind, "gray")    for ind in industries]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(n)
    bars = ax.bar(x, opt_taus, color=bar_colors, alpha=0.8, width=0.6, zorder=3)

    # Annotate bars
    for xi, (bar, tau) in enumerate(zip(bars, opt_taus)):
        ax.text(xi, tau + 0.003, f"{tau:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # τ=0.10 reference line
    ax.axhline(0.10, color="black", linewidth=1.4, linestyle="--",
               label="τ = 0.10 (default)", zorder=2)
    ax.text(n - 0.5, 0.102, "τ=0.10 default", fontsize=8, color="black", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(persona_ids, rotation=0, fontsize=9)
    ax.set_ylabel("Optimal Temperature τ*")
    ax.set_title("Optimal Temperature by Customer Profile (TD-034 τ Recalibration)")
    ax.set_ylim(0, max(opt_taus) * 1.20)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, zorder=0)

    # Legend (industry colors)
    patches = [mpatches.Patch(color=col, label=ind, alpha=0.8)
               for ind, col in INDUSTRY_COLORS.items()]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="black",
                                             linestyle="--", label="τ=0.10 default")],
              loc="upper right", fontsize=8, framealpha=0.85)

    save(fig, "block5b_tau_comparison")


# ── Chart 3: Accuracy trajectories ───────────────────────────────────────────
def chart_accuracy_trajectories(data):
    results     = data["results"]
    persona_ids = list(results.keys())
    days        = np.arange(1, 61)

    fig, ax = plt.subplots(figsize=(9, 5))

    legend_handles = []
    for pid in persona_ids:
        res     = results[pid]
        acc     = np.array(res["prod5"]["daily_acc_mean"], dtype=float)
        ind     = res["industry"]
        judge   = res.get("judge", "grok")
        color   = INDUSTRY_COLORS.get(ind, "gray")
        ls      = JUDGE_STYLES.get(judge, "solid")

        # 7-day rolling mean for readability
        smooth = np.convolve(acc, np.ones(7) / 7, mode="same")
        smooth[:3]  = np.nanmean(acc[:7])
        smooth[-3:] = np.nanmean(acc[-7:])

        line, = ax.plot(days, smooth, color=color, linestyle=ls,
                        linewidth=1.6, alpha=0.85)
        ax.text(days[-1] + 0.3, smooth[-1], pid, fontsize=7.5,
                color=color, va="center")

    # Industry legend
    ind_patches = [mpatches.Patch(color=col, label=ind, alpha=0.85)
                   for ind, col in INDUSTRY_COLORS.items()]
    # Judge style legend
    judge_lines = [plt.Line2D([0], [0], color="gray", linestyle=ls,
                               linewidth=1.6, label=JUDGE_LABELS[judge])
                   for judge, ls in JUDGE_STYLES.items()]
    ax.legend(handles=ind_patches + judge_lines,
              loc="lower right", ncol=2, fontsize=8, framealpha=0.85)

    ax.set_xlabel("Day")
    ax.set_ylabel("Accuracy (7-day rolling mean)")
    ax.set_title("Accuracy Trajectories Across Customer Personas (PROD-5 60-day sim)")
    ax.set_xlim(1, 65)
    ax.grid(True, linestyle=":", alpha=0.35)
    save(fig, "block5b_accuracy_trajectories")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = RESULTS_DIR / "all_harness_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Run run_all_harnesses.py first. Missing: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    print("Generating BLOCK-5B charts...")
    chart_convergence_heatmap(data)
    chart_tau_comparison(data)
    chart_accuracy_trajectories(data)
    print("Done. 3 charts × 2 formats = 6 files in paper_figures/.")
