"""
EXP-OP-MARGIN: L2 Margin Distribution + Lambda Threshold Diagnostic.

QUESTION: What is the distribution of L2 decision margins? What lambda is
required for sigma to flip a median-margin decision? At what lambda does
AUAC delta become statistically detectable?

This is a DIAGNOSTIC, not a gate experiment.

PART 1: Measure margin distribution across 20 seeds × 500 alerts (no Loop 2).
PART 2: Lambda sweep [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] — same OP1 paired
         design. Uses lambda_override in HarnessConfig to bypass OperatorSpec
         lambda_s <= 0.5 validation constraint.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.category_alert_generator import (
    CategoryAlertGenerator, CATEGORIES, ACTIONS,
)
from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle
from src.models.operator_spec import OperatorSpec
from src.models.operator_registry import OperatorRegistry
from src.eval.op_harness import OPHarness, HarnessConfig, HarnessResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
]

N_ALERTS      = 500
N_PRE_SHIFT   = 200
N_POST_SHIFT  = 400
WINDOW_SIZE   = 50
SIGMA_VALUE   = 0.4          # fixed sigma magnitude; sweep lambda only
TAU           = 0.1
ETA           = 0.05
ETA_NEG       = 1.0
TTL_FULL      = 400
N_FACTORS     = 6

LAMBDA_VALUES = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}

BONFERRONI_ALPHA = 0.05 / len(LAMBDA_VALUES)   # ~0.0071

RESULTS_PATH = Path("experiments/synthesis/expOP_margin/results.json")


# ---------------------------------------------------------------------------
# Confirmed shims
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    """Build (C, A, d) tensor from CategoryAlertGenerator nested profiles dict."""
    C, A, d = len(CATEGORIES), len(ACTIONS), N_FACTORS
    mu = np.zeros((C, A, d), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def _evaluate(result_action_index: int, gt_action_index: int) -> bool:
    """Inline oracle — confirmed compatible replacement for oracle.evaluate()."""
    return bool(result_action_index == gt_action_index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_correct_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = +SIGMA_VALUE
    sigma[:, esc_idx] = -SIGMA_VALUE
    return sigma


def build_harmful_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = -SIGMA_VALUE
    sigma[:, esc_idx] = +SIGMA_VALUE
    return sigma


def run_pre_shift(seed: int, profiles: np.ndarray) -> np.ndarray:
    """Run N_PRE_SHIFT pre-shift decisions. Returns post-training mu."""
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_PRE_SHIFT)
    for alert in alerts:
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = _evaluate(result.action_index, alert.gt_action_index)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    return scorer.mu.copy()


def run_post_shift(
    starting_mu: np.ndarray,
    alerts: list,
    registry: OperatorRegistry,
    lambda_override: float | None = None,
) -> HarnessResult:
    """Run post-shift via OPHarness. lambda_override bypasses spec constraint."""
    scorer = ProfileScorer(starting_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=N_POST_SHIFT,
        snapshot_interval=WINDOW_SIZE,
        use_synthesis=True,
        lambda_override=lambda_override,
        window_size=WINDOW_SIZE,
    )
    harness = OPHarness(scorer, oracle, registry, config)
    return harness.run(alerts)


def permutation_test(
    deltas: np.ndarray,
    n_permutations: int = 10000,
    rng_seed: int = 0,
) -> float:
    """One-sided permutation test: P(mean(random_sign * deltas) >= observed)."""
    observed = float(deltas.mean())
    rng = np.random.default_rng(rng_seed)
    count = sum(
        1 for _ in range(n_permutations)
        if float((rng.choice([-1, 1], len(deltas)) * deltas).mean()) >= observed
    )
    return count / n_permutations


# ---------------------------------------------------------------------------
# PART 1: Margin Measurement
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PART 1: MARGIN DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Measuring margins: {N_ALERTS} alerts x {len(SEEDS)} seeds ...", flush=True)

    all_margins:      list[float] = []
    all_lambda_flip:  list[float] = []
    per_category_margins: dict[str, list[float]] = defaultdict(list)

    C_dim = len(CATEGORIES)
    A_dim = len(ACTIONS)
    ac_idx  = ACTIONS.index("auto_close")
    esc_idx = ACTIONS.index("escalate_incident")

    for seed in SEEDS:
        gen = CategoryAlertGenerator(seed=seed)
        gt_profiles = _build_profiles_tensor(gen)
        alerts = gen.generate(N_ALERTS)

        for alert in alerts:
            c = alert.category_index
            f = alert.factors
            # L2 squared distances to each action centroid
            dists = np.array([
                float(np.sum((f - gt_profiles[c, a, :]) ** 2))
                for a in range(A_dim)
            ])
            dists_sorted = np.sort(dists)
            margin     = float(dists_sorted[1] - dists_sorted[0])
            lambda_flip = margin / SIGMA_VALUE

            all_margins.append(margin)
            all_lambda_flip.append(lambda_flip)
            per_category_margins[alert.category].append(margin)

    all_margins_arr    = np.array(all_margins)
    all_lambda_flip_arr = np.array(all_lambda_flip)

    pcts = [10, 25, 50, 75, 90]
    margin_pcts = np.percentile(all_margins_arr, pcts)
    lf_pcts     = np.percentile(all_lambda_flip_arr, pcts)
    (margin_p10, margin_p25, margin_p50, margin_p75, margin_p90) = margin_pcts
    (lf_p10, lf_p25, lf_p50, lf_p75, lf_p90) = lf_pcts

    fraction_flippable   = float(np.mean(all_lambda_flip_arr < 0.2))
    fraction_flippable_1 = float(np.mean(all_lambda_flip_arr < 1.0))
    fraction_flippable_2 = float(np.mean(all_lambda_flip_arr < 2.0))
    fraction_flippable_5 = float(np.mean(all_lambda_flip_arr < 5.0))

    print(f"N alerts analyzed: {len(all_margins)} ({N_ALERTS} x {len(SEEDS)} seeds)")
    print(f"\nL2 Margin (winner vs runner-up):")
    print(f"  p10={margin_p10:.4f}  p25={margin_p25:.4f}  p50={margin_p50:.4f}")
    print(f"  p75={margin_p75:.4f}  p90={margin_p90:.4f}")
    print(f"\nlambda required to flip each decision (lambda_flip = margin / sigma):")
    print(f"  p10={lf_p10:.4f}  p25={lf_p25:.4f}  p50={lf_p50:.4f}")
    print(f"  p75={lf_p75:.4f}  p90={lf_p90:.4f}")
    print(f"\nFraction of decisions flippable at each lambda:")
    print(f"  lambda=0.2  (current): {fraction_flippable:.1%}")
    print(f"  lambda=1.0:            {fraction_flippable_1:.1%}")
    print(f"  lambda=2.0:            {fraction_flippable_2:.1%}")
    print(f"  lambda=5.0:            {fraction_flippable_5:.1%}")
    print(f"\nConclusion: to flip >=10% of decisions, need lambda >= {lf_p10/SIGMA_VALUE:.2f}")
    print(f"           to flip >=50% of decisions, need lambda >= {lf_p50/SIGMA_VALUE:.2f}")

    # -----------------------------------------------------------------------
    # PART 2: Lambda Sweep
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("PART 2: LAMBDA SWEEP")
    print("=" * 60)

    lambda_results: dict = {}

    for lv in LAMBDA_VALUES:
        print(f"\n=== lambda={lv:.1f} ===", flush=True)
        deltas_B: list[float] = []
        deltas_C: list[float] = []
        baseline_auacs: list[float] = []

        # Register with lambda_s=0.2 (passes OperatorSpec validation <= 0.5).
        # Actual coupling controlled by lambda_override in HarnessConfig.
        # lambda_override=None at lv=0.0 means neutral synthesis (zero sigma effect).
        lo = lv if lv > 0.0 else None   # None → neutral synthesis at lambda=0

        for seed in SEEDS:
            print(f"  seed {seed}...", flush=True)

            gen_pre  = CategoryAlertGenerator(seed=seed)
            gt_profiles = _build_profiles_tensor(gen_pre)
            pre_shift_mu = run_pre_shift(seed, gt_profiles)

            post_gen = CategoryAlertGenerator(seed=seed + 10000)
            post_shift_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

            sigma_correct = build_correct_sigma0(C_dim, A_dim, ac_idx, esc_idx)
            sigma_harmful = build_harmful_sigma0(C_dim, A_dim, ac_idx, esc_idx)

            # Condition A — baseline, empty registry
            reg_A = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            result_A = run_post_shift(pre_shift_mu, post_shift_alerts, reg_A, lambda_override=lo)

            # Condition B — correct operator
            reg_B = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            reg_B.register(OperatorSpec(
                operator_id=f"correct_{seed}_lv{lv}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_correct,
                lambda_s=min(lv, 0.5) if lv > 0 else 0.2,   # valid spec value
                ttl_decisions=TTL_FULL,
            ), pre_shift_mu)
            result_B = run_post_shift(pre_shift_mu, post_shift_alerts, reg_B, lambda_override=lo)

            # Condition C — harmful operator
            reg_C = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            reg_C.register(OperatorSpec(
                operator_id=f"harmful_{seed}_lv{lv}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_harmful,
                lambda_s=min(lv, 0.5) if lv > 0 else 0.2,
                ttl_decisions=TTL_FULL,
            ), pre_shift_mu)
            result_C = run_post_shift(pre_shift_mu, post_shift_alerts, reg_C, lambda_override=lo)

            auac_A = float(result_A.auac_result.auac)
            auac_B = float(result_B.auac_result.auac)
            auac_C = float(result_C.auac_result.auac)
            deltas_B.append(auac_B - auac_A)
            deltas_C.append(auac_C - auac_A)
            baseline_auacs.append(auac_A)

        deltas_B_arr = np.array(deltas_B)
        deltas_C_arr = np.array(deltas_C)
        p_val = permutation_test(deltas_B_arr)

        lambda_results[lv] = {
            "B_vs_A_delta":   deltas_B_arr,
            "C_vs_A_delta":   deltas_C_arr,
            "baseline_auac":  float(np.mean(baseline_auacs)),
            "p_value":        p_val,
        }

    # -----------------------------------------------------------------------
    # Gate check printout
    # -----------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("PART 2: LAMBDA SWEEP -- GATE CHECK")
    print("=" * 75)
    print(f"{'lv':>6}  {'Eff. bias':>10}  {'B-A delta':>12}  {'p (raw)':>9}  "
          f"{'Bonf<0.0071':>12}  {'C-A delta':>10}")
    print("-" * 75)
    for lv in LAMBDA_VALUES:
        eff_bias   = lv * SIGMA_VALUE
        r          = lambda_results[lv]
        delta_mean = float(r["B_vs_A_delta"].mean())
        delta_p    = float(r["p_value"])
        bonf       = "[PASS]" if (delta_p < BONFERRONI_ALPHA and delta_mean > 0) else "[----]"
        c_delta    = float(r["C_vs_A_delta"].mean())
        print(f"{lv:>6.1f}  {eff_bias:>10.3f}  {delta_mean:>+12.4f}  "
              f"{delta_p:>9.4f}  {bonf:>12}  {c_delta:>+10.4f}")

    # Find threshold lambda
    passing = [
        lv for lv in LAMBDA_VALUES
        if float(lambda_results[lv]["p_value"]) < BONFERRONI_ALPHA
        and float(lambda_results[lv]["B_vs_A_delta"].mean()) > 0
    ]

    print()
    if passing:
        lv_thresh = min(passing)
        eff_thresh = lv_thresh * SIGMA_VALUE
        ratio = lf_p50 / eff_thresh if eff_thresh > 0 else float("inf")
        print(f"  lambda_threshold (first passing): {lv_thresh:.1f}")
        print(f"  Effective bias at threshold: {eff_thresh:.3f}")
        print(f"  Compares to margin p50:      {lf_p50:.4f}")
        print(f"  Ratio (margin p50 / eff.bias): {ratio:.2f}x")
    else:
        max_delta = max(float(lambda_results[lv]["B_vs_A_delta"].mean()) for lv in LAMBDA_VALUES)
        print(f"  No lambda value produced a passing result.")
        print(f"  Maximum AUAC delta seen: {max_delta:+.4f}")
        print(f"  At lambda=10.0 (eff.bias={10.0*SIGMA_VALUE:.1f}): "
              f"delta={float(lambda_results[10.0]['B_vs_A_delta'].mean()):+.4f}, "
              f"p={float(lambda_results[10.0]['p_value']):.4f}")

    print("\n  ARCHITECTURAL VERDICT:")
    if passing and min(passing) <= 1.0:
        print(f"  sigma works -- requires lambda up to {min(passing):.1f}. "
              f"S4 plateau was lambda=0.2-0.5 (frozen profiles).")
        print(f"  With Loop 2 running, higher lambda is needed. Architecturally viable.")
    elif passing and min(passing) <= 5.0:
        print(f"  sigma works but requires lambda={min(passing):.1f} -- "
              f"significantly above S4 plateau.")
        print(f"  May need re-calibrating S4 with Loop 2 active. Document the gap.")
    else:
        print(f"  sigma does NOT produce measurable AUAC delta at any tested lambda.")
        print(f"  The L2 mechanism may be fundamentally insensitive to scalar sigma bias.")
        print(f"  Consider: rank-1 operator or alternative scoring modification.")

    print("=" * 75)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    results_data = {
        "margin_stats": {
            "all_margins": [float(x) for x in all_margins],
            "percentiles": {
                "p10": float(margin_p10), "p25": float(margin_p25),
                "p50": float(margin_p50), "p75": float(margin_p75),
                "p90": float(margin_p90),
            },
            "lambda_flip_percentiles": {
                "p10": float(lf_p10), "p25": float(lf_p25),
                "p50": float(lf_p50), "p75": float(lf_p75),
                "p90": float(lf_p90),
            },
            "fraction_flippable": {
                "lambda_0.2": float(fraction_flippable),
                "lambda_1.0": float(fraction_flippable_1),
                "lambda_2.0": float(fraction_flippable_2),
                "lambda_5.0": float(fraction_flippable_5),
            },
            "per_category": {
                cat: [float(x) for x in vals]
                for cat, vals in per_category_margins.items()
            },
        },
        "lambda_sweep": {
            str(lv): {
                "mean_delta":    float(lambda_results[lv]["B_vs_A_delta"].mean()),
                "std_delta":     float(lambda_results[lv]["B_vs_A_delta"].std()),
                "p_value":       float(lambda_results[lv]["p_value"]),
                "c_vs_a":        float(lambda_results[lv]["C_vs_A_delta"].mean()),
                "baseline_auac": float(lambda_results[lv]["baseline_auac"]),
            }
            for lv in LAMBDA_VALUES
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fout:
        json.dump(results_data, fout, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------

    from experiments.synthesis.expOP_margin.charts import generate_charts
    generate_charts(results_data)
