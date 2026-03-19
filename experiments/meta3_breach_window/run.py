"""
META-3: Breach Window Derivation

Validates or corrects the W=14 day engineering choice for ConservationMonitor.

The key quantity is the breach detection window W — the minimum number of
daily α·q·V observations needed to confirm a genuine drop below θ_min
with false-alarm probability < δ = 0.05.

Two analytic bounds are computed and compared:
  1. Hoeffding bound (tight, requires range R):
       W ≥ −ln(δ) · R² / (2 · ε²)   where ε = signal_mean − θ_min
       Implements gae.calibration.compute_breach_window() formula.
  2. Chebyshev bound (looser, requires only variance):
       W ≥ Var / (δ · ε²)

Both are validated empirically via 10,000-sample bootstrap at the marginal
signal level (1.1× θ_min).
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

RESULTS_FILE = _REPO_ROOT / "results" / "meta3_breach_window.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS      = 50
N_DAYS       = 90
DELTA        = 0.05          # false-alarm probability target
THETA_MIN    = 0.434         # from derive_theta_min

ALERTS_PER_DAY       = 200
VERIFICATION_RATE    = 0.30
V_NOMINAL            = ALERTS_PER_DAY * VERIFICATION_RATE   # = 60 verified/day

# Beta parameters for daily alpha and q
ALPHA_A, ALPHA_B = 6.0, 14.0    # mean = 6/20 = 0.30
Q_A,     Q_B     = 16.0, 4.0    # mean = 16/20 = 0.80

W_ENGINEERING = 14   # the engineering choice we are validating

BOOTSTRAP_N = 10_000
W_VALUES    = [7, 10, 14, 21, 28]


# ---------------------------------------------------------------------------
# Hoeffding breach window  (implements gae.calibration.compute_breach_window)
# ---------------------------------------------------------------------------

def compute_breach_window(var: float, signal_mean: float, theta_min: float,
                          delta: float = 0.05,
                          signal_range: float | None = None) -> float:
    """
    Hoeffding bound: minimum W (days) such that
        P(W-day sample mean < theta_min | true mean = signal_mean) < delta

    Derivation:
        P(sample_mean < theta_min) <= exp(-2W*epsilon^2 / R^2)  < delta
        => W >= -ln(delta) * R^2 / (2 * epsilon^2)

    Parameters
    ----------
    var          : variance of the daily signal (used for Chebyshev; not Hoeffding)
    signal_mean  : true (long-run) signal mean
    theta_min    : breach threshold
    delta        : false-alarm probability ceiling (default 0.05)
    signal_range : R = max_signal - min_signal (required for Hoeffding)

    Returns
    -------
    W : float  (inf if signal_mean <= theta_min)
    """
    epsilon = signal_mean - theta_min
    if epsilon <= 0.0:
        return float("inf")
    if signal_range is None:
        raise ValueError("signal_range (R = max - min) required for Hoeffding bound")
    R = signal_range
    return -math.log(delta) * R * R / (2.0 * epsilon * epsilon)


def chebyshev_breach_window(var: float, signal_mean: float, theta_min: float,
                             delta: float = 0.05) -> float:
    """
    Chebyshev (Markov on sample variance) bound:
        W >= Var / (delta * epsilon^2)

    Looser than Hoeffding but requires only variance, not range.
    """
    epsilon = signal_mean - theta_min
    if epsilon <= 0.0:
        return float("inf")
    return var / (delta * epsilon * epsilon)


# ---------------------------------------------------------------------------
# Step 1: Simulate daily α·q·V signals
# ---------------------------------------------------------------------------

def simulate_daily_signals(rng: np.random.Generator) -> np.ndarray:
    """
    Simulate N_SEEDS × N_DAYS daily signal values.
    Returns flat array of length N_SEEDS * N_DAYS.
    """
    signals: list[float] = []
    for _ in range(N_SEEDS):
        for _ in range(N_DAYS):
            n_decisions = int(rng.poisson(V_NOMINAL))
            alpha       = float(rng.beta(ALPHA_A, ALPHA_B))
            q           = float(rng.beta(Q_A,     Q_B))
            V_day       = float(n_decisions)          # decisions observed that day
            signals.append(alpha * q * V_day)
    return np.array(signals)


# ---------------------------------------------------------------------------
# Step 4: Bootstrap validation
# ---------------------------------------------------------------------------

def bootstrap_false_breach_rate(signals: np.ndarray, signal_target: float,
                                 W: int, rng: np.random.Generator) -> float:
    """
    Draw BOOTSTRAP_N samples of length W (with replacement) from the subset
    of signals whose seed-mean ≈ signal_target (marginal regime).
    Return fraction of samples whose mean falls below THETA_MIN.

    For the marginal case (1.1× θ_min) we use ALL signal data, since
    the empirical distribution is symmetric around the overall mean;
    to target a specific regime we scale the signals.
    """
    # Scale factor to shift mean to signal_target without changing variance shape
    scale = signal_target / signals.mean() if signals.mean() > 0 else 1.0
    scaled = signals * scale

    count = 0
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(scaled, size=W, replace=True)
        if sample.mean() < THETA_MIN:
            count += 1
    return count / BOOTSTRAP_N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("=== META-3: BREACH WINDOW DERIVATION ===")
    print("=" * 62)
    print(f"N_SEEDS={N_SEEDS}  N_DAYS={N_DAYS}  θ_min={THETA_MIN}")
    print(f"δ={DELTA}  V_nominal={V_NOMINAL:.0f}/day")
    print(f"Alpha ~ Beta({ALPHA_A},{ALPHA_B}) mean={ALPHA_A/(ALPHA_A+ALPHA_B):.2f}")
    print(f"q     ~ Beta({Q_A},{Q_B}) mean={Q_A/(Q_A+Q_B):.2f}")
    print()

    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Step 1: Simulate
    # -----------------------------------------------------------------------
    print("Step 1: Simulating daily signals ...", flush=True)
    signals = simulate_daily_signals(rng)
    print(f"  Generated {len(signals):,} daily signal values")

    # -----------------------------------------------------------------------
    # Step 2: Signal statistics
    # -----------------------------------------------------------------------
    mu     = float(signals.mean())
    sigma  = float(signals.std())
    s_var  = float(signals.var())
    s_min  = float(signals.min())
    s_max  = float(signals.max())
    s_range = s_max - s_min

    print()
    print("Signal statistics (50 seeds × 90 days):")
    print(f"  Mean : {mu:.4f}")
    print(f"  Std  : {sigma:.4f}")
    print(f"  Var  : {s_var:.4f}")
    print(f"  Min  : {s_min:.4f}")
    print(f"  Max  : {s_max:.4f}")
    print(f"  Range: {s_range:.4f}")
    print(f"  θ_min = {THETA_MIN}  (signal/θ_min ratio = {mu/THETA_MIN:.2f}×)")

    # -----------------------------------------------------------------------
    # Step 3: W at multiple signal levels
    # -----------------------------------------------------------------------
    signal_levels = [
        ("Healthy (μ)",  mu),
        ("2× θ_min",     2.0   * THETA_MIN),
        ("1.5× θ_min",   1.5   * THETA_MIN),
        ("1.2× θ_min",   1.2   * THETA_MIN),
        ("1.1× θ_min",   1.1   * THETA_MIN),
        ("1.05× θ_min",  1.05  * THETA_MIN),
        ("At θ_min",     THETA_MIN),
    ]

    print()
    print("W by signal level (δ=5%):")
    header = (f"  {'Signal level':<16} {'Value':>7} {'ε=s-θ':>8} "
              f"{'W_Hoef':>9} {'W_Cheb':>9} {'Regime'}")
    sep = "  " + "-" * 70
    print(header)
    print(sep)

    w_table: list[dict] = []
    marginal_signal = 1.1 * THETA_MIN

    def regime(mult: float) -> str:
        if mult >= 2.0:   return "trivial"
        if mult >= 1.5:   return "easy"
        if mult >= 1.2:   return "moderate"
        if mult >= 1.1:   return "marginal"
        if mult > 1.0:    return "dangerous"
        return "breached"

    for label, s_val in signal_levels:
        mult   = s_val / THETA_MIN
        eps    = s_val - THETA_MIN
        if eps > 0:
            w_h = compute_breach_window(s_var, s_val, THETA_MIN, DELTA, s_range)
            w_c = chebyshev_breach_window(s_var, s_val, THETA_MIN, DELTA)
            w_h_str = f"{w_h:7.1f}"
            w_c_str = f"{w_c:7.1f}"
        else:
            w_h, w_c = float("inf"), float("inf")
            w_h_str = w_c_str = "     inf"
        reg = regime(mult)
        print(f"  {label:<16} {s_val:>7.3f} {eps:>8.3f} {w_h_str:>9} {w_c_str:>9}  {reg}")
        w_table.append(dict(label=label, signal=s_val, epsilon=eps,
                            mult_theta=mult, w_hoeffding=w_h, w_chebyshev=w_c,
                            regime=reg))

    # -----------------------------------------------------------------------
    # Step 4: Bootstrap validation at marginal (1.1× θ_min)
    # -----------------------------------------------------------------------
    print()
    print(f"Bootstrap validation (marginal regime, signal = 1.1× θ_min = {marginal_signal:.4f}):")
    print(f"  (N_BOOTSTRAP={BOOTSTRAP_N:,} samples per W)")
    print()
    print(f"  {'W':>4}  {'False breach rate':>18}  {'< 5%?':>8}")
    print("  " + "-" * 36)

    bootstrap_table: list[dict] = []
    for W in W_VALUES:
        fbr = bootstrap_false_breach_rate(signals, marginal_signal, W, rng)
        adequate = "YES" if fbr < DELTA else "NO"
        print(f"  {W:>4}  {fbr*100:>16.2f}%  {adequate:>8}")
        bootstrap_table.append(dict(W=W, false_breach_rate=fbr, adequate=fbr < DELTA))

    # Hoeffding at marginal for reference
    w_h_marginal = compute_breach_window(s_var, marginal_signal, THETA_MIN, DELTA, s_range)
    w_c_marginal = chebyshev_breach_window(s_var, marginal_signal, THETA_MIN, DELTA)
    print(f"\n  Hoeffding W at marginal = {w_h_marginal:.1f}")
    print(f"  Chebyshev W at marginal = {w_c_marginal:.1f}")

    # -----------------------------------------------------------------------
    # Step 5: Verdict
    # -----------------------------------------------------------------------
    fbr_14 = next(r["false_breach_rate"] for r in bootstrap_table if r["W"] == W_ENGINEERING)
    first_adequate = next((r["W"] for r in bootstrap_table if r["adequate"]), None)

    print()
    print("=" * 62)
    print("VERDICT:")

    if fbr_14 < 0.01:
        print(f"  W={W_ENGINEERING} gives {fbr_14*100:.2f}% false breach rate — OVERLY CONSERVATIVE.")
        smaller = next((r["W"] for r in bootstrap_table if r["adequate"]), W_ENGINEERING)
        print(f"  W={smaller} would suffice at 5% false-alarm target.")
        verdict = "overly_conservative"
        recommended_W = smaller
    elif fbr_14 < DELTA:
        print(f"  W={W_ENGINEERING} gives {fbr_14*100:.2f}% false breach rate — VALIDATED (<5%).")
        print(f"  Engineering choice W={W_ENGINEERING} is principled.")
        verdict = "validated"
        recommended_W = W_ENGINEERING
    else:
        print(f"  W={W_ENGINEERING} gives {fbr_14*100:.2f}% false breach rate — INSUFFICIENT (>5%).")
        if first_adequate:
            print(f"  Recommend W={first_adequate} (first W with <5% false-alarm rate).")
        verdict = "insufficient"
        recommended_W = first_adequate

    print()
    print(f"  Signal statistics recap:")
    print(f"    Empirical mean = {mu:.4f} = {mu/THETA_MIN:.2f}× θ_min  → regime: {regime(mu/THETA_MIN)}")
    print(f"    At healthy operating point, W is irrelevant (breach obvious).")
    print(f"    W={W_ENGINEERING} is calibrated for MARGINAL regime (1.1× θ_min).")
    print()
    print(f"  RECOMMENDED W: {recommended_W}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    result = dict(
        config = dict(
            n_seeds=N_SEEDS, n_days=N_DAYS, delta=DELTA, theta_min=THETA_MIN,
            v_nominal=V_NOMINAL, alpha_params=[ALPHA_A, ALPHA_B],
            q_params=[Q_A, Q_B], bootstrap_n=BOOTSTRAP_N,
            w_engineering=W_ENGINEERING,
        ),
        signal_stats = dict(
            mean=mu, std=sigma, var=s_var,
            min=s_min, max=s_max, range=s_range,
            mult_theta=mu / THETA_MIN,
        ),
        w_table          = w_table,
        bootstrap_table  = bootstrap_table,
        w_hoeffding_marginal = w_h_marginal,
        w_chebyshev_marginal = w_c_marginal,
        verdict          = dict(
            result       = verdict,
            fbr_at_W14   = fbr_14,
            recommended_W= recommended_W,
        ),
    )

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"Results saved -> {RESULTS_FILE}")

    # Charts
    from experiments.meta3_breach_window.charts import chart1_w_vs_signal, chart2_bootstrap
    chart1_w_vs_signal(result)
    chart2_bootstrap(result)


if __name__ == "__main__":
    main()
