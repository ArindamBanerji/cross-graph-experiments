"""
PROD-5: Convergence Rate — Publication-quality figures.
Reads results/prod5_convergence.json produced by run.py.

Chart 1 — prod5_convergence_by_category:
  6 category convergence curves (daily mean L2 error), ε=0.05 line,
  vertical convergence-day markers.

Chart 2 — prod5_predicted_vs_simulated:
  Predicted (L-08) vs simulated weeks scatter, diagonal + 1.5× band.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

EPS_CONV = 0.05
DAYS     = 60

# 6-category colour palette (qualitative, print-safe)
CAT_COLORS = [
    "#1f78b4",  # credential_access  — blue
    "#33a02c",  # threat_intel_match — green
    "#e31a1c",  # lateral_movement   — red
    "#ff7f00",  # data_exfiltration  — orange
    "#6a3d9a",  # insider_threat     — purple
    "#b15928",  # cloud_infrastructure — brown
]

CAT_LABELS = {
    "credential_access":    "cred_access",
    "threat_intel_match":   "threat_intel",
    "lateral_movement":     "lateral_mvmt",
    "data_exfiltration":    "data_exfil",
    "insider_threat":       "insider_threat",
    "cloud_infrastructure": "cloud_infra",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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


# ── Chart 1: Per-category convergence curves ──────────────────────────────────
def chart_convergence_curves(data):
    categories = data["categories"]
    days       = np.arange(1, DAYS + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for ci, cat in enumerate(categories):
        err_curve = np.array(data["simulated"][cat]["daily_mean_error_mean"])
        conv_day  = data["simulated"][cat]["converge_day_mean"]
        color     = CAT_COLORS[ci]
        label     = CAT_LABELS.get(cat, cat)

        ax.plot(days, err_curve, linewidth=1.8, color=color, label=label)

        # Vertical convergence marker
        if conv_day is not None:
            ax.axvline(conv_day, color=color, linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(conv_day + 0.4, err_curve[0] * (0.80 - ci * 0.07),
                    f"d={conv_day:.0f}", fontsize=7, color=color, va="top")

    # ε threshold line
    ax.axhline(EPS_CONV, color="black", linestyle="--", linewidth=1.2,
               label=f"ε = {EPS_CONV} (convergence)")

    ax.set_xlabel("Day")
    ax.set_ylabel("Mean per-action L2 error")
    ax.set_title("Per-Category Convergence (60-Day Production Simulation)")
    ax.set_xlim(1, DAYS)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", framealpha=0.85, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.35)

    # Summary note
    gate_str = "PASS" if data["gate_pass"] else "FAIL"
    n_not = sum(
        data["simulated"][c]["not_converged_seeds"]
        for c in categories
    )
    note = f"Gate: {gate_str}  |  alerts/day={data['config']['alerts_per_day']}"
    if n_not:
        note += f"  |  {n_not} seed-category pairs did not converge"
    ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=8,
            va="bottom", bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="lightyellow", alpha=0.8))

    save(fig, "prod5_convergence_by_category")


# ── Chart 2: Predicted vs simulated scatter ───────────────────────────────────
def chart_predicted_vs_simulated(data):
    categories = data["categories"]
    preds, sims, labels, colors = [], [], [], []

    for ci, cat in enumerate(categories):
        c = data["comparison"][cat]
        if c["predicted_weeks"] is None or c["simulated_weeks"] is None:
            continue
        preds.append(c["predicted_weeks"])
        sims.append(c["simulated_weeks"])
        labels.append(CAT_LABELS.get(cat, cat))
        colors.append(CAT_COLORS[ci])

    preds = np.array(preds)
    sims  = np.array(sims)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Axis range
    max_val = max(preds.max(), sims.max()) * 1.15
    ax_range = np.linspace(0, max_val, 200)

    # 1.5× on-track band: y ≤ 1.5x (shaded below the 1.5x line)
    ax.fill_between(ax_range, 0, 1.5 * ax_range, alpha=0.10, color="#4dac26",
                    label="On-track zone (simulated ≤ 1.5× predicted)")

    # Perfect prediction diagonal
    ax.plot(ax_range, ax_range, "k--", linewidth=1.2, label="Perfect prediction (y = x)")

    # 1.5× boundary line
    ax.plot(ax_range, 1.5 * ax_range, color="#d73027", linewidth=1.0, linestyle="-.",
            label="1.5× limit (on-track boundary)")

    # Scatter points
    for i, (p, s, lbl, col) in enumerate(zip(preds, sims, labels, colors)):
        ax.scatter(p, s, color=col, s=90, zorder=5)
        # Label offset: alternate above/below
        va = "bottom" if i % 2 == 0 else "top"
        dy = 0.08 if va == "bottom" else -0.08
        ax.annotate(lbl, xy=(p, s), xytext=(p + 0.05, s + dy),
                    fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5))

    ax.set_xlabel("L-08 Predicted Convergence (weeks)")
    ax.set_ylabel("Simulated Convergence (weeks)")
    ax.set_title("L-08 Calendar Prediction vs Simulated Convergence")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.85, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)

    # Gate summary
    gate_str = "PASS" if data["gate_pass"] else "FAIL"
    off      = data["off_track_categories"]
    note     = f"Gate: {gate_str}"
    if off:
        note += f"\nOff-track: {', '.join(CAT_LABELS.get(c, c) for c in off)}"
    ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    save(fig, "prod5_predicted_vs_simulated")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = RESULTS_DIR / "prod5_convergence.json"
    if not p.exists():
        raise FileNotFoundError(f"Run run.py first. Missing: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    print("Generating PROD-5 paper figures...")
    chart_convergence_curves(data)
    chart_predicted_vs_simulated(data)
    print("Done. 2 charts × 2 formats = 4 files in paper_figures/.")
