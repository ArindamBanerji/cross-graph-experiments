"""
TD-034: τ Recalibration — Publication-quality figures.
Reads results/td034_tau_synthetic.json produced by run.py.

Chart 1 — td034_ece_vs_tau:  ECE mean ± std vs τ, gate line, τ=0.10 marker.
Chart 2 — td034_reliability_diagram:  Reliability diagram at τ=0.10.
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

# Publication style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ECE_GATE = 0.05


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = PAPER_FIGS / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ── Chart 1: ECE vs τ ─────────────────────────────────────────────────────────
def chart_ece_vs_tau(data: dict) -> None:
    tau_vals  = []
    ece_means = []
    ece_stds  = []

    for tau_str, r in data["results"].items():
        tau_vals.append(r["tau"])
        ece_means.append(r["ece_mean"])
        ece_stds.append(r["ece_std"])

    tau_vals  = np.array(tau_vals)
    ece_means = np.array(ece_means)
    ece_stds  = np.array(ece_stds)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Std band
    ax.fill_between(tau_vals, ece_means - ece_stds, ece_means + ece_stds,
                    alpha=0.18, color="#2166ac", label="±1 std (50 seeds)")

    # Mean line
    ax.plot(tau_vals, ece_means, "o-", color="#2166ac", linewidth=2,
            markersize=6, label="Mean ECE")

    # Gate threshold
    ax.axhline(ECE_GATE, color="#d73027", linewidth=1.5, linestyle="--",
               label=f"Gate (ECE ≤ {ECE_GATE})")

    # Mark τ=0.10
    idx_010 = list(tau_vals).index(0.10) if 0.10 in list(tau_vals) else None
    if idx_010 is not None:
        ax.plot(0.10, ece_means[idx_010], "D", color="#f46d43", markersize=9,
                zorder=5, label="τ = 0.10 (current default)")
        ax.annotate(
            f"τ=0.10\nECE={ece_means[idx_010]:.4f}",
            xy=(0.10, ece_means[idx_010]),
            xytext=(0.10 + 0.015, ece_means[idx_010] + 0.003),
            fontsize=9, color="#f46d43",
            arrowprops=dict(arrowstyle="-", color="#f46d43", lw=0.8),
        )

    # Optimal τ marker
    opt_tau = data["optimal_tau"]
    opt_ece = data["optimal_ece"]
    if abs(opt_tau - 0.10) > 1e-9:
        ax.plot(opt_tau, opt_ece, "s", color="#1a9850", markersize=9,
                zorder=5, label=f"Optimal τ = {opt_tau:.2f}")

    ax.set_xlabel("Temperature τ")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("ECE vs Temperature (A=4 Synthetic)")
    ax.set_xticks(tau_vals)
    ax.set_xticklabels([f"{t:.2f}" for t in tau_vals])
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.4)

    gate_str = "PASS" if data["gate_pass"] else "FAIL"
    ax.text(0.02, 0.97,
            f"Optimal τ={data['optimal_tau']:.2f}  ECE={data['optimal_ece']:.4f}  Gate: {gate_str}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    save(fig, "td034_ece_vs_tau")


# ── Chart 2: Reliability diagram at τ=0.10 ───────────────────────────────────
def chart_reliability_diagram(data: dict) -> None:
    rd       = data["reliability_diagram"]
    bin_conf = np.array(rd["bin_confidence"])
    bin_acc  = np.array(rd["bin_accuracy"], dtype=float)  # may contain NaN
    bin_cnt  = np.array(rd["bin_counts"])
    n_bins   = len(bin_conf)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration", zorder=1)

    # Bar-style calibration plot (skip empty bins)
    width = 1.0 / n_bins
    for i in range(n_bins):
        if bin_cnt[i] == 0 or np.isnan(bin_acc[i]):
            continue
        gap = bin_acc[i] - bin_conf[i]
        color = "#d73027" if gap < 0 else "#4dac26"
        # Base bar (confidence)
        ax.bar(bin_conf[i], bin_conf[i], width=width * 0.7, alpha=0.25,
               color="#2166ac", align="center", zorder=2)
        # Accuracy bar
        ax.bar(bin_conf[i], bin_acc[i], width=width * 0.7, alpha=0.7,
               color=color, align="center", zorder=3)

    # Dots on top of accuracy bars
    valid = ~np.isnan(bin_acc) & (bin_cnt > 0)
    ax.scatter(bin_conf[valid], bin_acc[valid], color="#1a1a1a", s=30, zorder=5,
               label="Bin accuracy")

    # ECE annotation
    tau_10  = data["results"]["0.10"]
    ece_val = tau_10["ece_mean"]
    ax.text(0.05, 0.93,
            f"ECE = {ece_val:.4f}\n(τ = 0.10, N={rd['n_samples']:,})",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    over_patch  = mpatches.Patch(color="#4dac26", alpha=0.7, label="Over-confident")
    under_patch = mpatches.Patch(color="#d73027", alpha=0.7, label="Under-confident")
    ax.legend(handles=[over_patch, under_patch], loc="lower right", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction Correct (Accuracy)")
    ax.set_title("Calibration Diagram at τ = 0.10")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)

    save(fig, "td034_reliability_diagram")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_path = RESULTS_DIR / "td034_tau_synthetic.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Run run.py first. Missing: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    print("Generating TD-034 paper figures...")
    chart_ece_vs_tau(data)
    chart_reliability_diagram(data)
    print("Done. 2 charts × 2 formats = 4 files saved to paper_figures/.")
