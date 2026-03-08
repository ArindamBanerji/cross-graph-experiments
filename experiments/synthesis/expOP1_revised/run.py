"""
EXP-OP1-REVISED: Scalar sigma with Loop 2 Running — Headroom Design.

Changes from OP1:
  N_PRE_SHIFT:  300 -> 100  (Loop 2 partially trained, not converged)
  CAMPAIGN:     escalate_incident 0.75 -> 0.90  (harder shift)
  N_POST_SHIFT: 300 -> 400  (longer recovery window)
  Condition E:  cold start + correct operator (maximum headroom)
  Condition F:  cold start baseline (no pre-shift, no operator)

GATE: mean AUAC delta (B vs A) > 0, p < 0.05
FALSIFICATION: STOP only if AUAC delta <= 0 AND T70 speedup <= 0 AND final acc delta <= 0
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

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

SEEDS          = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
N_PRE_SHIFT    = 100
N_POST_SHIFT   = 400
WINDOW_SIZE    = 50
LAMBDA_S       = 0.2
TAU            = 0.1
ETA            = 0.05
ETA_NEG        = 1.0
TTL_FULL       = 400
TTL_HALF       = 200
N_FACTORS      = 6

RESULTS_PATH   = Path("experiments/synthesis/expOP1_revised/results.json")

# Harder campaign shift: escalate_incident = 90% GT probability for all categories
CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}


# ---------------------------------------------------------------------------
# Confirmed shims (do not re-inspect source files)
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    """Build (C, A, d) tensor from CategoryAlertGenerator nested profiles dict."""
    C = len(CATEGORIES)
    A = len(ACTIONS)
    d = N_FACTORS
    mu = np.zeros((C, A, d), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def _evaluate(result_action_index: int, gt_action_index: int) -> bool:
    """Inline oracle — confirmed compatible replacement for oracle.evaluate()."""
    return bool(result_action_index == gt_action_index)


# ---------------------------------------------------------------------------
# Sigma helpers
# ---------------------------------------------------------------------------

def build_correct_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    """Correct operator: discourage auto_close (+0.4), encourage escalate_incident (-0.4)."""
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = +0.4
    sigma[:, esc_idx] = -0.4
    return sigma


def build_harmful_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    """Harmful operator: inverted signs — wrong direction."""
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = -0.4
    sigma[:, esc_idx] = +0.4
    return sigma


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pre_shift(seed: int, profiles_tensor: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    """
    Run pre-shift phase (N_PRE_SHIFT=100 decisions): Loop 2, no operator.
    Uses normal GT distribution (no campaign dict).
    Returns (post-training centroid mu, correct_flags).
    """
    scorer = ProfileScorer(profiles_tensor.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_PRE_SHIFT)

    correct_flags: list[bool] = []
    for alert in alerts:
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = _evaluate(result.action_index, alert.gt_action_index)
        correct_flags.append(is_correct)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    return scorer.mu.copy(), correct_flags


def run_post_shift(
    starting_mu: np.ndarray,
    post_shift_alerts: list,
    registry: OperatorRegistry,
) -> HarnessResult:
    """
    Run post-shift phase via OPHarness (Loop 2 running + optional operator).
    starting_mu may be pre_shift_mu (warm) or config_profiles (cold).
    """
    scorer = ProfileScorer(starting_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=N_POST_SHIFT,
        snapshot_interval=50,
        use_synthesis=True,
        window_size=WINDOW_SIZE,
    )
    harness = OPHarness(scorer, oracle, registry, config)
    return harness.run(post_shift_alerts)


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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    all_results: dict[str, list] = {
        "condition_A": [],
        "condition_B": [],
        "condition_C": [],
        "condition_D": [],
        "condition_E": [],
        "condition_F": [],
    }

    for seed in SEEDS:
        print(f"Seed {seed}...", flush=True)

        gen = CategoryAlertGenerator(seed=seed)
        config_profiles = _build_profiles_tensor(gen)   # raw, untrained

        C_dim   = len(CATEGORIES)
        A_dim   = len(ACTIONS)
        ac_idx  = ACTIONS.index("auto_close")
        esc_idx = ACTIONS.index("escalate_incident")

        # Pre-shift — identical starting point for conditions A/B/C/D
        pre_shift_mu, _ = run_pre_shift(seed, config_profiles)

        # Post-shift alerts — shared across all six conditions
        post_gen = CategoryAlertGenerator(seed=seed)
        post_shift_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

        sigma_correct = build_correct_sigma0(C_dim, A_dim, ac_idx, esc_idx)
        sigma_harmful = build_harmful_sigma0(C_dim, A_dim, ac_idx, esc_idx)

        # Condition A — baseline: warm start, empty registry
        reg_A = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        result_A = run_post_shift(pre_shift_mu, post_shift_alerts, reg_A)

        # Condition B — correct operator, full TTL, warm start
        reg_B = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_B.register(OperatorSpec(
            operator_id=f"correct_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
        ), pre_shift_mu)
        result_B = run_post_shift(pre_shift_mu, post_shift_alerts, reg_B)

        # Condition C — harmful operator, full TTL, warm start
        reg_C = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_C.register(OperatorSpec(
            operator_id=f"harmful_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_harmful, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
        ), pre_shift_mu)
        result_C = run_post_shift(pre_shift_mu, post_shift_alerts, reg_C)

        # Condition D — correct operator, half TTL (expires at decision 200)
        reg_D = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_D.register(OperatorSpec(
            operator_id=f"expiring_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_HALF,
        ), pre_shift_mu)
        result_D = run_post_shift(pre_shift_mu, post_shift_alerts, reg_D)

        # Condition E — cold start + correct operator (maximum headroom)
        reg_E = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_E.register(OperatorSpec(
            operator_id=f"cold_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
        ), config_profiles)   # validate against config_profiles (no pre-shift training)
        result_E = run_post_shift(config_profiles, post_shift_alerts, reg_E)

        # Condition F — cold start baseline (no pre-shift, no operator)
        reg_F = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        result_F = run_post_shift(config_profiles, post_shift_alerts, reg_F)

        # Record all six conditions
        for cond_key, result, label in [
            ("condition_A", result_A, "baseline_partial_warmup"),
            ("condition_B", result_B, "correct_operator"),
            ("condition_C", result_C, "harmful_operator"),
            ("condition_D", result_D, "expiring_operator"),
            ("condition_E", result_E, "cold_start_with_operator"),
            ("condition_F", result_F, "cold_start_baseline"),
        ]:
            all_results[cond_key].append({
                "seed":           seed,
                "label":          label,
                "auac":           float(result.auac_result.auac),
                "t70":            int(result.auac_result.t70)
                                      if result.auac_result.t70 is not None else None,
                "t90":            int(result.auac_result.t90)
                                      if result.auac_result.t90 is not None else None,
                "final_accuracy": float(result.auac_result.final_accuracy),
                "regret":         float(result.auac_result.cumulative_regret),
                "accuracy_curve": [float(v) for v in result.auac_result.accuracy_curve],
                "n_expired":      int(result.n_operators_expired),
            })

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    auacs_A = np.array([r["auac"] for r in all_results["condition_A"]])
    auacs_B = np.array([r["auac"] for r in all_results["condition_B"]])
    auacs_C = np.array([r["auac"] for r in all_results["condition_C"]])
    auacs_D = np.array([r["auac"] for r in all_results["condition_D"]])
    auacs_E = np.array([r["auac"] for r in all_results["condition_E"]])
    auacs_F = np.array([r["auac"] for r in all_results["condition_F"]])

    delta_BA = auacs_B - auacs_A
    delta_CA = auacs_C - auacs_A
    delta_DA = auacs_D - auacs_A
    delta_EF = auacs_E - auacs_F
    delta_BF = auacs_B - auacs_F

    p_BA = permutation_test(delta_BA)
    p_CA = permutation_test(delta_CA)
    p_DA = permutation_test(delta_DA)
    p_EF = permutation_test(delta_EF)
    p_BF = permutation_test(delta_BF)

    def t70_speedup(cond_key_x: str, cond_key_baseline: str) -> float:
        t70s_base = [r["t70"] for r in all_results[cond_key_baseline] if r["t70"] is not None]
        t70s_x    = [r["t70"] for r in all_results[cond_key_x]        if r["t70"] is not None]
        if not t70s_base or not t70s_x:
            return float("nan")
        return float(np.mean(t70s_base) - np.mean(t70s_x))  # positive = faster

    summary_stats = {
        "B_vs_A": {
            "mean_auac_delta":  float(delta_BA.mean()),
            "std_auac_delta":   float(delta_BA.std()),
            "mean_t70_speedup": t70_speedup("condition_B", "condition_A"),
            "p_value":          p_BA,
            "gate_pass":        bool(delta_BA.mean() > 0 and p_BA < 0.05),
        },
        "C_vs_A": {
            "mean_auac_delta":  float(delta_CA.mean()),
            "std_auac_delta":   float(delta_CA.std()),
            "mean_t70_speedup": t70_speedup("condition_C", "condition_A"),
            "p_value":          p_CA,
            "gate_pass":        bool(delta_CA.mean() > 0 and p_CA < 0.05),
        },
        "D_vs_A": {
            "mean_auac_delta":  float(delta_DA.mean()),
            "std_auac_delta":   float(delta_DA.std()),
            "mean_t70_speedup": t70_speedup("condition_D", "condition_A"),
            "p_value":          p_DA,
            "gate_pass":        bool(delta_DA.mean() > 0 and p_DA < 0.05),
        },
        "E_vs_F": {
            "mean_auac_delta":  float(delta_EF.mean()),
            "std_auac_delta":   float(delta_EF.std()),
            "mean_t70_speedup": t70_speedup("condition_E", "condition_F"),
            "p_value":          p_EF,
            "gate_pass":        bool(delta_EF.mean() > 0 and p_EF < 0.05),
        },
        "B_vs_F": {
            "mean_auac_delta":  float(delta_BF.mean()),
            "std_auac_delta":   float(delta_BF.std()),
            "mean_t70_speedup": t70_speedup("condition_B", "condition_F"),
            "p_value":          p_BF,
            "gate_pass":        bool(delta_BF.mean() > 0 and p_BF < 0.05),
        },
    }

    # -----------------------------------------------------------------------
    # Gate check output
    # -----------------------------------------------------------------------

    print("=" * 60)
    print("EXP-OP1-REVISED GATE CHECK")
    print("=" * 60)
    print(f"Baseline AUAC (A, partial warmup): {auacs_A.mean():.4f}")
    print(f"Baseline AUAC (F, cold start):     {auacs_F.mean():.4f}")
    print()
    print(f"B vs A (correct op, partial warmup):")
    print(f"  AUAC delta  = {delta_BA.mean():+.4f} +/- {delta_BA.std():.4f}")
    print(f"  p-value     = {p_BA:.4f}")
    print(f"  T70 speedup = {t70_speedup('condition_B', 'condition_A'):.1f} decisions")
    if summary_stats["B_vs_A"]["gate_pass"]:
        print("  GATE-OP1-REVISED: PASS")
    else:
        print("  GATE-OP1-REVISED: FAIL")
        fa_delta = (
            np.array([r["final_accuracy"] for r in all_results["condition_B"]])
            - np.array([r["final_accuracy"] for r in all_results["condition_A"]])
        )
        print(f"  Falsification: AUAC>0={delta_BA.mean()>0}, "
              f"T70>0={t70_speedup('condition_B','condition_A')>0}, "
              f"FinalAcc>0={fa_delta.mean()>0}")
        if delta_BA.mean() <= 0 and t70_speedup("condition_B", "condition_A") <= 0 and fa_delta.mean() <= 0:
            print("  ALL THREE FALSIFICATION CONDITIONS MET -- STOP")
    print()
    print(f"C vs A (harmful op):   delta={delta_CA.mean():+.4f}, p={p_CA:.4f}  (expect negative)")
    print(f"D vs A (expiring op):  delta={delta_DA.mean():+.4f}, p={p_DA:.4f}")
    print()
    print(f"E vs F (cold+op vs cold baseline):")
    print(f"  AUAC delta  = {delta_EF.mean():+.4f} +/- {delta_EF.std():.4f}")
    print(f"  p-value     = {p_EF:.4f}")
    print(f"  T70 speedup = {t70_speedup('condition_E', 'condition_F'):.1f} decisions")
    if delta_EF.mean() > 0 and p_EF < 0.05:
        print("  COLD-START GATE: PASS -- sigma helps when Loop 2 is naive")
    else:
        print("  COLD-START GATE: FAIL")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"all_results": all_results, "summary_stats": summary_stats}, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------

    from experiments.synthesis.expOP1_revised.charts import generate_charts
    generate_charts(all_results, summary_stats)
