"""
EXP-OP1: Scalar sigma with Loop 2 Running.

QUESTION: Does scalar synthesis bias sigma improve action selection accuracy
during a distributional shift when Loop 2 is running simultaneously?

GATE: mean AUAC delta (B vs A) > 0, paired permutation test p < 0.05

FALSIFICATION: if AUAC delta <= 0 AND T70 speedup <= 0 AND final acc delta <= 0
across all lambda conditions -> STOP.

Four conditions (identical post-shift alert sequence per seed):
  A: Baseline    — Loop 2 only, no operator
  B: Correct op  — Loop 2 + correctly-aligned operator (TTL_FULL)
  C: Harmful op  — Loop 2 + inverted-sign operator (TTL_FULL)
  D: Expiring op — Loop 2 + correct operator but TTL_HALF
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

SEEDS         = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
N_PRE_SHIFT   = 300
N_POST_SHIFT  = 300
WINDOW_SIZE   = 50
LAMBDA_S      = 0.2
TAU           = 0.1
ETA           = 0.05
ETA_NEG       = 1.0
TTL_FULL      = 300
TTL_HALF      = 150
N_FACTORS     = 6

RESULTS_PATH  = Path("experiments/synthesis/expOP1_scalar_loop2/results.json")

# Post-shift campaign: escalate_incident boosted to 75% for all categories
CAMPAIGN = {cat: {"escalate_incident": 0.75} for cat in CATEGORIES}

# Pre-shift suppression: escalate_incident suppressed to 5% (rarely correct)
SUPPRESSED = {cat: "escalate_incident" for cat in CATEGORIES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    """Build (C, A, d) centroid tensor from generator's nested profile dict."""
    C = len(gen.categories)
    A = len(gen.actions)
    d = len(gen.factors)
    mu = np.zeros((C, A, d), dtype=np.float64)
    for c, cat in enumerate(gen.categories):
        for a, act in enumerate(gen.actions):
            mu[c, a, :] = gen.profiles[cat][act]
    return mu


def build_correct_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    """Correct operator: discourage auto_close (+0.4), encourage escalate_incident (-0.4)."""
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = +0.4
    sigma[:, esc_idx] = -0.4
    return sigma


def build_harmful_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    """Harmful operator: inverted signs relative to correct operator."""
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = -0.4
    sigma[:, esc_idx] = +0.4
    return sigma


def run_pre_shift(seed: int, profiles: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    """
    Run pre-shift phase: Loop 2 running, no operator.
    Returns (post-training centroid mu, correct_flags).
    Uses generate_precampaign to suppress escalate_incident in GT distribution.
    """
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate_precampaign(N_PRE_SHIFT, SUPPRESSED)

    correct_flags: list[bool] = []
    for alert in alerts:
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = bool(result.action_index == alert.gt_action_index)
        correct_flags.append(is_correct)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    return scorer.mu.copy(), correct_flags


def run_post_shift(
    pre_shift_mu: np.ndarray,
    post_shift_alerts: list,
    registry: OperatorRegistry,
) -> HarnessResult:
    """
    Run post-shift phase via OPHarness (Loop 2 running + optional operator).
    Fresh ProfileScorer from pre_shift_mu on every call.
    """
    scorer = ProfileScorer(pre_shift_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
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
    """
    One-sided permutation test: P(mean(random_sign * deltas) >= observed).
    H0: the sign of each delta is random (i.e., no systematic effect).
    """
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
    }

    for seed in SEEDS:
        print(f"Seed {seed}...", flush=True)

        # Profiles and dimensions
        gen     = CategoryAlertGenerator(seed=seed)
        profiles = _build_profiles_tensor(gen)
        C_dim   = len(CATEGORIES)
        A_dim   = len(ACTIONS)
        ac_idx  = ACTIONS.index("auto_close")
        esc_idx = ACTIONS.index("escalate_incident")

        # Pre-shift phase — identical starting point for all four conditions
        pre_shift_mu, _ = run_pre_shift(seed, profiles)

        # Post-shift alerts — generated once, shared across all four conditions
        post_gen = CategoryAlertGenerator(seed=seed)
        post_shift_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

        # Sigma tensors
        sigma_correct = build_correct_sigma0(C_dim, A_dim, ac_idx, esc_idx)
        sigma_harmful = build_harmful_sigma0(C_dim, A_dim, ac_idx, esc_idx)

        # Condition A — baseline: empty registry (neutral synthesis applied)
        reg_A = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        result_A = run_post_shift(pre_shift_mu, post_shift_alerts, reg_A)

        # Condition B — correct operator, full TTL
        reg_B = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_B.register(OperatorSpec(
            operator_id=f"correct_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
        ), pre_shift_mu)
        result_B = run_post_shift(pre_shift_mu, post_shift_alerts, reg_B)

        # Condition C — harmful operator (inverted sigma), full TTL
        reg_C = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_C.register(OperatorSpec(
            operator_id=f"harmful_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_harmful, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
        ), pre_shift_mu)
        result_C = run_post_shift(pre_shift_mu, post_shift_alerts, reg_C)

        # Condition D — correct operator, half TTL (expires at decision 150)
        reg_D = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
        reg_D.register(OperatorSpec(
            operator_id=f"expiring_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_HALF,
        ), pre_shift_mu)
        result_D = run_post_shift(pre_shift_mu, post_shift_alerts, reg_D)

        # Record all four conditions
        for cond_key, result, label in [
            ("condition_A", result_A, "baseline"),
            ("condition_B", result_B, "correct_operator"),
            ("condition_C", result_C, "harmful_operator"),
            ("condition_D", result_D, "expiring_operator"),
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

    delta_B = auacs_B - auacs_A
    delta_C = auacs_C - auacs_A
    delta_D = auacs_D - auacs_A

    p_B = permutation_test(delta_B)
    p_C = permutation_test(delta_C)
    p_D = permutation_test(delta_D)

    def t70_speedup(cond_key: str) -> float:
        t70s_A = [r["t70"] for r in all_results["condition_A"] if r["t70"] is not None]
        t70s_X = [r["t70"] for r in all_results[cond_key]      if r["t70"] is not None]
        if not t70s_A or not t70s_X:
            return float("nan")
        return float(np.mean(t70s_A) - np.mean(t70s_X))  # positive = faster

    summary_stats = {
        "B_vs_A": {
            "mean_auac_delta":  float(delta_B.mean()),
            "std_auac_delta":   float(delta_B.std()),
            "mean_t70_speedup": t70_speedup("condition_B"),
            "p_value":          p_B,
            "gate_pass":        bool(delta_B.mean() > 0 and p_B < 0.05),
        },
        "C_vs_A": {
            "mean_auac_delta":  float(delta_C.mean()),
            "std_auac_delta":   float(delta_C.std()),
            "mean_t70_speedup": t70_speedup("condition_C"),
            "p_value":          p_C,
            "gate_pass":        bool(delta_C.mean() > 0 and p_C < 0.05),
        },
        "D_vs_A": {
            "mean_auac_delta":  float(delta_D.mean()),
            "std_auac_delta":   float(delta_D.std()),
            "mean_t70_speedup": t70_speedup("condition_D"),
            "p_value":          p_D,
            "gate_pass":        bool(delta_D.mean() > 0 and p_D < 0.05),
        },
    }

    # -----------------------------------------------------------------------
    # Gate check output
    # -----------------------------------------------------------------------

    print("=" * 60)
    print("EXP-OP1 GATE CHECK")
    print("=" * 60)
    print(f"B vs A (correct operator):")
    print(f"  mean AUAC = A:{auacs_A.mean():.4f}  B:{auacs_B.mean():.4f}")
    print(f"  AUAC delta  = {delta_B.mean():+.4f} +/- {delta_B.std():.4f}")
    print(f"  p-value     = {p_B:.4f}")
    print(f"  T70 speedup = {t70_speedup('condition_B'):.1f} decisions")
    if summary_stats["B_vs_A"]["gate_pass"]:
        print("  GATE-OP1: PASS")
    else:
        print("  GATE-OP1: FAIL")
        print("  Falsification check:")
        print(f"    AUAC delta <= 0: {delta_B.mean() <= 0}")
        print(f"    T70 speedup <= 0: {t70_speedup('condition_B') <= 0}")
        fad = (
            np.array([r["final_accuracy"] for r in all_results["condition_B"]])
            - np.array([r["final_accuracy"] for r in all_results["condition_A"]])
        )
        print(f"    Final acc delta <= 0: {fad.mean() <= 0}")
        if delta_B.mean() <= 0 and t70_speedup("condition_B") <= 0 and fad.mean() <= 0:
            print("  ALL THREE FALSIFICATION CONDITIONS MET -- STOP")

    print(f"\nC vs A (harmful operator):")
    print(f"  mean AUAC = A:{auacs_A.mean():.4f}  C:{auacs_C.mean():.4f}")
    print(f"  AUAC delta  = {delta_C.mean():+.4f} (expect negative)")
    print(f"  p-value     = {p_C:.4f}")

    print(f"\nD vs A (expiring operator):")
    print(f"  mean AUAC = A:{auacs_A.mean():.4f}  D:{auacs_D.mean():.4f}")
    print(f"  AUAC delta  = {delta_D.mean():+.4f}")
    print(f"  p-value     = {p_D:.4f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"all_results": all_results, "summary_stats": summary_stats}, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------

    from experiments.synthesis.expOP1_scalar_loop2.charts import generate_charts
    generate_charts(all_results, summary_stats)
