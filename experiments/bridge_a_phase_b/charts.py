"""
B-A Phase B: Production A/B — Publication-quality figures.
Reads results/bridge_a_phase_b.json produced by run.py.

Chart 1 — bridge_a_phase_b_acceptance_trajectory
Chart 2 — bridge_a_phase_b_reward_distribution
Chart 3 — bridge_a_phase_b_conservation
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

THETA_MIN = 0.434
DAYS      = 90

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

COLOR_A  = "#2166ac"   # blue — Group A
COLOR_B  = "#d73027"   # red  — Group B
COLOR_V1 = "#4dac26"   # green — variant 1 promoted


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = PAPER_FIGS / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


# ── Chart 1: Acceptance rate trajectory over 90 days ─────────────────────────
def chart_acceptance_trajectory(data):
    traj_A = np.array(data["trajectories"]["daily_acceptance_A"], dtype=float)
    traj_B = np.array(data["trajectories"]["daily_acceptance_B"], dtype=float)
    days   = np.arange(1, DAYS + 1)

    # Smooth with 7-day rolling mean for readability
    def rolling(arr, w=7):
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            lo = max(0, i - w // 2)
            hi = min(len(arr), i + w // 2 + 1)
            vals = arr[lo:hi]
            out[i] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
        return out

    smooth_A = rolling(traj_A)
    smooth_B = rolling(traj_B)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(days, traj_A, alpha=0.25, color=COLOR_A, linewidth=0.8)
    ax.plot(days, traj_B, alpha=0.25, color=COLOR_B, linewidth=0.8)
    ax.plot(days, smooth_A, color=COLOR_A, linewidth=2.0, label="Group A — fixed v0")
    ax.plot(days, smooth_B, color=COLOR_B, linewidth=2.0, label="Group B — Thompson adaptive")

    # Mark mean promotion day if variant 1 was promoted
    promo = data["promotion"]
    if promo["promotion_rate"] > 0 and promo["mean_decisions_to_promotion"] is not None:
        # Convert from decision count to approximate day
        n_B_analysts = len(data["group_b_analysts"])
        dec_per_day  = n_B_analysts * 0.85
        promo_day    = promo["mean_decisions_to_promotion"] / dec_per_day
        ax.axvline(promo_day, color=COLOR_V1, linestyle="--", linewidth=1.5,
                   label=f"Variant 1 promoted (≈day {promo_day:.0f})")
        ax.text(promo_day + 0.8, 0.62, f"v1 promoted\n{promo['promotion_rate']*100:.0f}% of seeds",
                fontsize=8, color=COLOR_V1, va="bottom")

    # Quality reference lines
    ax.axhline(0.70, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(0.80, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(DAYS + 0.5, 0.70, "q=0.70", fontsize=8, color="gray", va="center")
    ax.text(DAYS + 0.5, 0.80, "q=0.80", fontsize=8, color="gray", va="center")

    ax.set_xlabel("Day")
    ax.set_ylabel("Acceptance Rate (7-day rolling mean)")
    ax.set_title("A/B Study: Acceptance Rate Over 90 Days")
    ax.set_xlim(1, DAYS + 3)
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.35)

    # Annotation box
    d = data["statistical_tests"]
    note = (f"Δ = {d['delta_acceptance']*100:+.1f}pp  |  "
            f"Cohen's d = {d['cohens_d']:.2f}  |  p = {d['mann_whitney_p']:.4f}")
    ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    save(fig, "bridge_a_phase_b_acceptance_trajectory")


# ── Chart 2: Composite reward distribution ────────────────────────────────────
def chart_reward_distribution(data):
    R_A = np.array(data["distributions"]["reward_A_sample"])
    R_B = np.array(data["distributions"]["reward_B_sample"])

    fig, ax = plt.subplots(figsize=(6, 5))

    positions = [1, 2]
    parts = ax.violinplot([R_A, R_B], positions=positions,
                          showmedians=True, showextrema=True, widths=0.55)

    colors = [COLOR_A, COLOR_B]
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor(col)

    for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part_name].set_color("black")
        parts[part_name].set_linewidth(1.0)

    # Overlay box-plot style IQR markers
    for i, (arr, col) in enumerate(zip([R_A, R_B], colors)):
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        ax.scatter(positions[i], q50, color=col, s=30, zorder=5)
        ax.vlines(positions[i], q25, q75, lw=3, color=col, alpha=0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Group A\n(fixed v0)", "Group B\n(Thompson)"])
    ax.set_ylabel("Composite Reward R")
    ax.set_title("Composite Reward Distribution: Fixed vs Adaptive")

    # Mean annotations
    for pos, arr, col in zip(positions, [R_A, R_B], colors):
        ax.text(pos, arr.max() + 0.01,
                f"μ={arr.mean():.3f}", fontsize=9, ha="center", color=col)

    d = data["statistical_tests"]
    note = (f"Δμ = {data['metrics']['group_b']['reward_mean'] - data['metrics']['group_a']['reward_mean']:+.4f}  |  "
            f"Cohen's d = {d['cohens_d']:.2f}")
    ax.text(0.5, 0.04, note, transform=ax.transAxes, fontsize=9, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    save(fig, "bridge_a_phase_b_reward_distribution")


# ── Chart 3: Conservation law signal for Group B ─────────────────────────────
def chart_conservation(data):
    signal = np.array(data["trajectories"]["conservation_signal_mean"])
    days   = np.arange(1, DAYS + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Fill under signal
    ax.fill_between(days, 0, signal, alpha=0.20, color=COLOR_B)
    ax.plot(days, signal, color=COLOR_B, linewidth=1.8, label="α·q·V (Group B daily)")

    # θ_min line
    ax.axhline(THETA_MIN, color="#d73027", linewidth=1.5, linestyle="--",
               label=f"θ_min = {THETA_MIN}")
    ax.text(DAYS + 0.3, THETA_MIN, f"θ_min={THETA_MIN}", fontsize=8,
            color="#d73027", va="center")

    # Variant 1 promotion day (if applicable)
    promo = data["promotion"]
    if promo["promotion_rate"] > 0 and promo["mean_decisions_to_promotion"] is not None:
        n_B_analysts = len(data["group_b_analysts"])
        dec_per_day  = n_B_analysts * 0.85
        promo_day    = promo["mean_decisions_to_promotion"] / dec_per_day
        ax.axvline(promo_day, color=COLOR_V1, linestyle="--", linewidth=1.2,
                   label=f"Variant 1 promoted (≈day {promo_day:.0f})")

    ax.set_xlabel("Day")
    ax.set_ylabel("Conservation Signal α·q·V")
    ax.set_title("Conservation Law Signal During A/B Study")
    ax.set_xlim(1, DAYS + 2)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.35)

    breach_rate = data["conservation"]["breach_rate"]
    if breach_rate == 0:
        status_txt = "Never breached"
        bbox_col   = "#d9f7be"
    else:
        status_txt = f"Breached in {breach_rate*100:.1f}% of seeds"
        bbox_col   = "#ffebe6"
    ax.text(0.02, 0.95, status_txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_col, alpha=0.85))

    save(fig, "bridge_a_phase_b_conservation")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = RESULTS_DIR / "bridge_a_phase_b.json"
    if not p.exists():
        raise FileNotFoundError(f"Run run.py first. Missing: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    print("Generating B-A Phase B paper figures...")
    chart_acceptance_trajectory(data)
    chart_reward_distribution(data)
    chart_conservation(data)
    print("Done. 3 charts × 2 formats = 6 files in paper_figures/.")
