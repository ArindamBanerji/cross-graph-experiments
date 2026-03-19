"""
META-3 Charts

Chart 1 — meta3_w_vs_signal:
  X = signal level (as multiple of θ_min).
  Y = W (log scale). Two curves: Hoeffding and Chebyshev.
  Horizontal line at W=14. Vertical line at 1.1× θ_min.
  Title: "Breach Window vs Signal Level"

Chart 2 — meta3_bootstrap_validation:
  X = W. Y = false breach rate (%).
  Horizontal line at 5%. Mark W=14.
  Title: "Bootstrap Validation: False Breach Rate by Window Size"
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.viz.bridge_common import save_figure

RESULTS_FILE = _REPO_ROOT / "results" / "meta3_breach_window.json"

W_ENGINEERING = 14
DELTA         = 0.05


def load() -> dict:
    with open(RESULTS_FILE) as f:
        return json.load(f)


# =============================================================================
# Chart 1 — W vs signal level
# =============================================================================

def chart1_w_vs_signal(data: dict) -> None:
    theta_min = data["config"]["theta_min"]
    ss        = data["signal_stats"]
    s_range   = ss["range"]
    s_var     = ss["var"]

    # Fine x-axis: signal as multiple of theta_min, from 1.01 to 3.0
    mult_vals = np.linspace(1.01, 3.0, 500)
    s_vals    = mult_vals * theta_min

    def hoeffding(s: float) -> float:
        eps = s - theta_min
        if eps <= 0:
            return float("nan")
        return -math.log(DELTA) * s_range ** 2 / (2.0 * eps ** 2)

    def chebyshev(s: float) -> float:
        eps = s - theta_min
        if eps <= 0:
            return float("nan")
        return s_var / (DELTA * eps ** 2)

    w_hoef = np.array([hoeffding(s) for s in s_vals])
    w_cheb = np.array([chebyshev(s) for s in s_vals])

    # Clip for log-scale display
    w_hoef = np.clip(w_hoef, 0.1, 1e5)
    w_cheb = np.clip(w_cheb, 0.1, 1e5)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, right=0.97)

    ax.semilogy(mult_vals, w_hoef, color="#1565c0", lw=2.5, label="Hoeffding bound (tight)")
    ax.semilogy(mult_vals, w_cheb, color="#c62828", lw=2.0, ls="--",
                label="Chebyshev bound (loose)", alpha=0.85)

    # Horizontal line at W=14
    ax.axhline(W_ENGINEERING, color="#2e7d32", lw=2.0, ls="-.",
               label=f"W={W_ENGINEERING} (engineering choice)")

    # Vertical line at 1.1× θ_min (marginal regime)
    ax.axvline(1.1, color="#e65100", lw=1.8, ls=":",
               label="1.1× θ_min (marginal regime)")

    # Vertical line at empirical mean
    mu_mult = ss["mult_theta"]
    ax.axvline(mu_mult, color="#6a1b9a", lw=1.5, ls=":",
               label=f"Empirical mean ({mu_mult:.2f}× θ_min)", alpha=0.8)

    # Regime shading
    ax.axvspan(1.0,  1.1,  alpha=0.08, color="#c62828", label="Dangerous (1.0–1.1×)")
    ax.axvspan(1.1,  1.2,  alpha=0.08, color="#ff9800", label="Marginal (1.1–1.2×)")
    ax.axvspan(1.2,  1.5,  alpha=0.06, color="#ffee58", label="Moderate (1.2–1.5×)")
    ax.axvspan(1.5,  3.0,  alpha=0.04, color="#a5d6a7", label="Easy (>1.5×)")

    # Annotate the W=14 crossing for each curve
    for w_arr, col, name in [(w_hoef, "#1565c0", "Hoeffding"), (w_cheb, "#c62828", "Chebyshev")]:
        # Find where W crosses W_ENGINEERING from above
        crossings = np.where(np.diff((w_arr > W_ENGINEERING).astype(int)) == -1)[0]
        if len(crossings) > 0:
            ci = crossings[0]
            mx = mult_vals[ci]
            ax.axvline(mx, color=col, lw=0.9, ls=":", alpha=0.55)
            ax.text(mx + 0.02, W_ENGINEERING * 1.4,
                    f"{name}\ncross\n{mx:.2f}×",
                    fontsize=7.5, color=col, ha="left", va="bottom", alpha=0.85)

    ax.set_xlabel("Signal level (multiple of θ_min)", fontsize=12)
    ax.set_ylabel("Required window W (days, log scale)", fontsize=12)
    ax.set_xlim(1.0, 3.0)
    ax.set_ylim(0.5, 2000)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92, ncol=2)

    ax.set_title(
        "Breach Window vs Signal Level\n"
        f"W = −ln(δ)·R²/(2ε²)  [Hoeffding],  W = Var/(δ·ε²)  [Chebyshev]. "
        f"δ={DELTA}, θ_min={theta_min}, R={s_range:.3f}.",
        fontsize=11.5, fontweight="bold",
    )

    cfg = data["config"]
    caption = (
        f"Signal = α·q·V_day. α~Beta({cfg['alpha_params'][0]},{cfg['alpha_params'][1]}), "
        f"q~Beta({cfg['q_params'][0]},{cfg['q_params'][1]}), "
        f"V~Poisson({cfg['v_nominal']:.0f}/day). "
        f"Range R={s_range:.3f} from {cfg['n_seeds']}×{cfg['n_days']} simulation. "
        "Log y-axis. Engineering choice W=14 shown as dash-dot green line."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8, color="#555", style="italic")

    save_figure(fig, "meta3_w_vs_signal", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 1] meta3_w_vs_signal.png + .pdf saved")


# =============================================================================
# Chart 2 — Bootstrap validation
# =============================================================================

def chart2_bootstrap(data: dict) -> None:
    bt     = data["bootstrap_table"]
    theta_min = data["config"]["theta_min"]
    w_vals = [r["W"]               for r in bt]
    fbr    = [r["false_breach_rate"] * 100 for r in bt]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, right=0.97)

    colors = ["#c62828" if r["false_breach_rate"] >= DELTA else "#2e7d32" for r in bt]
    bars = ax.bar(w_vals, fbr, width=2.8, color=colors, edgecolor="black",
                  linewidth=0.8, alpha=0.88, zorder=4)

    for bar, val, r in zip(bars, fbr, bt):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.1,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#c62828" if r["false_breach_rate"] >= DELTA else "#2e7d32")

    # 5% line
    ax.axhline(DELTA * 100, color="#555", lw=2.0, ls="--",
               label=f"{DELTA*100:.0f}% false-alarm ceiling", zorder=5)

    # W=14 marker
    ax.axvline(W_ENGINEERING, color="#1565c0", lw=2.2, ls="-.",
               label=f"W={W_ENGINEERING} (engineering choice)", zorder=5)

    # Hoeffding reference
    w_h = data.get("w_hoeffding_marginal", None)
    if w_h and w_h < 50:
        ax.axvline(w_h, color="#6a1b9a", lw=1.5, ls=":",
                   label=f"Hoeffding W = {w_h:.1f}", zorder=4, alpha=0.8)

    # Shade adequate region
    ax.fill_between([min(w_vals) - 2, max(w_vals) + 2], 0, DELTA * 100,
                    color="#e8f5e9", alpha=0.40, zorder=1, label="<5% zone (adequate)")

    ax.set_xticks(w_vals)
    ax.set_xticklabels([f"W={w}" for w in w_vals], fontsize=10)
    ax.set_xlabel("Window size W (days)", fontsize=12)
    ax.set_ylabel("False breach rate (%)", fontsize=12)
    ax.set_ylim(0, max(fbr) * 1.3 + 1)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.92)

    verdict = data["verdict"]
    v_text  = dict(validated="VALIDATED", overly_conservative="CONSERVATIVE",
                   insufficient="INSUFFICIENT").get(verdict["result"], verdict["result"])
    v_col   = dict(validated="#2e7d32", overly_conservative="#f57f17",
                   insufficient="#c62828").get(verdict["result"], "#555")

    ax.text(
        0.02, 0.96,
        f"W={W_ENGINEERING}: {verdict['fbr_at_W14']*100:.2f}% false breach rate\n"
        f"Verdict: {v_text}\n"
        f"Recommended W: {verdict['recommended_W']}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold",
        color=v_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=v_col, linewidth=1.5, alpha=0.93),
    )

    ax.set_title(
        f"Bootstrap Validation: False Breach Rate by Window Size\n"
        f"Marginal regime: signal = 1.1× θ_min = {1.1*theta_min:.4f}. "
        f"Target: false breach rate < {DELTA*100:.0f}%.",
        fontsize=11.5, fontweight="bold",
    )

    cfg = data["config"]
    caption = (
        f"N={cfg['bootstrap_n']:,} bootstrap samples per W. "
        "Signals drawn from empirical 50×90-day pool, scaled to marginal regime. "
        "Green = adequate (<5%), red = insufficient (≥5%)."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.5, color="#555", style="italic")

    save_figure(fig, "meta3_bootstrap_validation", output_dir="paper_figures")
    plt.close(fig)
    print("[CHART 2] meta3_bootstrap_validation.png + .pdf saved")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    data = load()
    chart1_w_vs_signal(data)
    chart2_bootstrap(data)
