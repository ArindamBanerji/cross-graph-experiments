"""
P4-POWER: Gate 1 Detection Power Analysis

For each (N, inflation) combination, simulate 200 seeds of a simplified
Gate 1 superiority test to determine the minimum sample size needed to
detect attacker-inflated quality at 80% power.

Setup per seed:
  - Baseline quality = 0.70
  - Target quality   = 0.70 + inflation
  - N decisions split equally between the two variants
  - Outcomes: Bernoulli(true_quality)
  - Beta posterior (uniform prior), normal approximation
  - Promoted if P(mu_target > mu_baseline + DELTA_MIN | data) > 0.95

Sweep:
  N_values        = [50, 100, 150, 200, 300, 500, 750, 1000]
  inflation_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
  N_SEEDS         = 200
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS          = 200
BASELINE_QUALITY = 0.70
DELTA_MIN        = 0.05
PROMO_THRESHOLD  = 0.95          # P > 0.95 → promoted

N_VALUES         = [50, 100, 150, 200, 300, 500, 750, 1000]
INFLATION_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

POWER_TARGET     = 0.80          # 80% power threshold
FP_TARGET        = 0.05          # ≤5% false positive
DAYS_PER_DECISION = 5.0          # 5 decisions/day (for feasibility calc)
PILOT_DAYS       = 90            # 90-day pilot window

RESULTS_FILE = _REPO_ROOT / "results" / "exp_l2_power.json"


# ---------------------------------------------------------------------------
# Core: Beta posterior normal approximation
# ---------------------------------------------------------------------------

def _beta_posterior_params(k: int, n: int) -> tuple[float, float]:
    """
    Beta(1+k, 1+n-k) posterior (uniform prior).
    Returns (mean, variance).
    """
    a = 1.0 + k
    b = 1.0 + (n - k)
    s = a + b
    mean = a / s
    var  = (a * b) / (s * s * (s + 1.0))
    return mean, var


def promotion_prob(k_t: int, n_t: int, k_b: int, n_b: int) -> float:
    """
    P(Beta_target > Beta_baseline + DELTA_MIN | data)
    via normal approximation to the difference of two Beta posteriors.
    """
    mean_t, var_t = _beta_posterior_params(k_t, n_t)
    mean_b, var_b = _beta_posterior_params(k_b, n_b)
    diff_mean = mean_t - mean_b
    diff_std  = math.sqrt(max(var_t + var_b, 1e-12))
    z    = (diff_mean - DELTA_MIN) / diff_std
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return prob


# ---------------------------------------------------------------------------
# Simulation: one (N, inflation) cell
# ---------------------------------------------------------------------------

def run_cell(N: int, inflation: float, rng: np.random.Generator) -> float:
    """
    Simulate N_SEEDS seeds for a given (N, inflation).
    Returns promotion_rate (fraction of seeds promoted).
    """
    q_target   = min(BASELINE_QUALITY + inflation, 1.0)
    q_baseline = BASELINE_QUALITY

    n_t = N // 2          # decisions allocated to target
    n_b = N - n_t         # decisions allocated to baseline

    promotions = 0
    for _ in range(N_SEEDS):
        k_t = int(rng.binomial(n_t, q_target))
        k_b = int(rng.binomial(n_b, q_baseline))
        p   = promotion_prob(k_t, n_t, k_b, n_b)
        if p > PROMO_THRESHOLD:
            promotions += 1

    return promotions / N_SEEDS


# ---------------------------------------------------------------------------
# Min N for target power
# ---------------------------------------------------------------------------

def min_n_for_power(rates: dict, inflation: float,
                    target: float = POWER_TARGET) -> int | None:
    """
    Scan N_VALUES in order; return first N where promotion_rate >= target.
    Returns None if never reached.
    """
    for N in N_VALUES:
        if rates[(N, inflation)] >= target:
            return N
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("=== P4-POWER: GATE 1 DETECTION POWER ANALYSIS ===")
    print("=" * 62)
    print(f"N_SEEDS={N_SEEDS}  DELTA_MIN={DELTA_MIN}  PROMO_THRESHOLD={PROMO_THRESHOLD}")
    print(f"Baseline quality={BASELINE_QUALITY}")
    print(f"Sweep: N={N_VALUES}, inflation={INFLATION_VALUES}")
    print()

    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Run all cells
    # -----------------------------------------------------------------------
    rates: dict[tuple[int, float], float] = {}
    total = len(N_VALUES) * len(INFLATION_VALUES)
    done  = 0
    for N in N_VALUES:
        for infl in INFLATION_VALUES:
            rates[(N, infl)] = run_cell(N, infl, rng)
            done += 1
        print(f"  N={N:4d} complete ({done}/{total} cells)", flush=True)

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    infl_labels = [f"{int(v*100):2d}pp" for v in INFLATION_VALUES]
    col_w = 7

    print()
    print("Promotion rate (%) by N and inflation:")
    print()
    header = f"  {'N':>5} |" + "|".join(f" {lbl:^{col_w}} " for lbl in infl_labels)
    sep    = "  " + "-" * 7 + "+" + ("+".join(["-" * (col_w + 2)] * len(INFLATION_VALUES)))
    print(header)
    print(sep)
    for N in N_VALUES:
        row = f"  {N:>5} |"
        for infl in INFLATION_VALUES:
            pct = rates[(N, infl)] * 100
            row += f" {pct:5.1f}%  |"
        print(row)

    # -----------------------------------------------------------------------
    # Min N for 80% power table
    # -----------------------------------------------------------------------
    print()
    print("Minimum N for 80% power at each inflation:")
    print(f"  {'Inflation':<12} {'Min N':>8} {'Days@5/day':>12} {'Feasible 90d?':>15}")
    print("  " + "-" * 50)

    power_table: list[dict] = []
    for infl in INFLATION_VALUES[1:]:   # skip 0pp (false positive row)
        mn = min_n_for_power(rates, infl)
        label = f"{int(infl*100)}pp"
        if mn is None:
            days_str   = "N/A"
            feasible   = "NO (needs >1000)"
            mn_display = ">1000"
        else:
            days = mn / DAYS_PER_DECISION
            days_str   = f"{days:.0f}"
            feasible   = f"{'YES' if days <= PILOT_DAYS else 'NO (needs >' + str(int(days)) + 'd)'}"
            mn_display = str(mn)
        print(f"  {label:<12} {mn_display:>8} {days_str:>12} {feasible:>15}")
        power_table.append(dict(
            inflation_pp    = int(infl * 100),
            min_n_80pct     = mn,
            days_at_5perday = None if mn is None else mn / DAYS_PER_DECISION,
            feasible_90d    = None if mn is None else (mn / DAYS_PER_DECISION <= PILOT_DAYS),
        ))

    # -----------------------------------------------------------------------
    # False positive rates
    # -----------------------------------------------------------------------
    fp_ns = [50, 100, 200, 500]
    print()
    print("False positive rate at inflation=0:")
    row = "  " + " | ".join(
        f"N={n}: {rates[(n, 0.00)]*100:.1f}%" for n in fp_ns
    )
    print(row)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    mn_10pp = min_n_for_power(rates, 0.10)
    mn_5pp  = min_n_for_power(rates, 0.05)
    fp_200  = rates[(200, 0.00)]

    print()
    print("INTERPRETATION:")
    if mn_10pp is not None and mn_10pp <= 200:
        print(f"  [10pp] Detectable at N={mn_10pp} (<= 200). Gate is adequate → run P4-F (subtle).")
    elif mn_10pp is not None and mn_10pp <= 500:
        print(f"  [10pp] Needs N={mn_10pp} (200 < N <= 500). Gate marginally adequate; longer window advised.")
    else:
        print(f"  [10pp] Needs N={mn_10pp if mn_10pp else '>1000'}. Gate needs redesign: SPRT or K=2 or longer window.")

    if mn_5pp is None:
        print("  [5pp ] Never detectable at N<=1000. Subtle poisoning is a residual risk.")
        print("         Mitigate with outcome verification sampling.")
    else:
        print(f"  [5pp ] Detectable at N={mn_5pp}.")

    if fp_200 > FP_TARGET:
        print(f"  [FP  ] False positive rate at N=200: {fp_200*100:.1f}% > {FP_TARGET*100:.0f}%. Gate too permissive.")
    else:
        print(f"  [FP  ] False positive rate at N=200: {fp_200*100:.1f}% <= {FP_TARGET*100:.0f}%. FP control OK.")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    rates_serializable = {
        f"N{N}_infl{int(infl*100)}pp": rates[(N, infl)]
        for N in N_VALUES for infl in INFLATION_VALUES
    }

    result = dict(
        config = dict(
            n_seeds          = N_SEEDS,
            baseline_quality = BASELINE_QUALITY,
            delta_min        = DELTA_MIN,
            promo_threshold  = PROMO_THRESHOLD,
            power_target     = POWER_TARGET,
            fp_target        = FP_TARGET,
            days_per_decision= DAYS_PER_DECISION,
            pilot_days       = PILOT_DAYS,
        ),
        n_values         = N_VALUES,
        inflation_values = INFLATION_VALUES,
        rates            = rates_serializable,
        power_table      = power_table,
        false_positive   = {str(n): rates[(n, 0.00)] for n in N_VALUES},
        interpretation   = dict(
            min_n_10pp    = mn_10pp,
            min_n_5pp     = mn_5pp,
            fp_at_n200    = fp_200,
        ),
    )

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"Results saved -> {RESULTS_FILE}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    from experiments.exp_l2_power.charts import chart1_heatmap, chart2_curves
    chart1_heatmap(result)
    chart2_curves(result)


if __name__ == "__main__":
    main()
