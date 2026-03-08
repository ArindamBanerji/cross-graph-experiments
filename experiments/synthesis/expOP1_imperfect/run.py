"""
EXP-OP1-IMPERFECT: Scalar sigma with production-realistic (noisy) profiles.

QUESTION: Does scalar synthesis bias improve action selection when profiles are
imperfect — the condition where sigma is actually needed?

Approach: add Gaussian noise epsilon to config profiles to simulate domain-expert
estimates that are directionally correct but quantitatively imperfect. Sweep
epsilon in [0.00, 0.05, 0.10, 0.15, 0.20].

GATE (primary at epsilon=0.10): mean AUAC delta (B vs A) > 0, Bonferroni p < 0.0125
FALSIFICATION: STOP only if AUAC delta <= 0 AND T70 speedup <= 0 AND final acc delta <= 0
               at epsilon=0.10, 0.15, and 0.20 simultaneously.
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

SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
]  # 20 seeds for statistical power

EPSILON_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20]
NOISE_RNG_SEED = 42    # fixed — same profile imperfection across all seeds

N_PRE_SHIFT    = 200
N_POST_SHIFT   = 400
WINDOW_SIZE    = 50
LAMBDA_S       = 0.2
TAU            = 0.1
ETA            = 0.05
ETA_NEG        = 1.0
TTL_FULL       = 400
TTL_HALF       = 200
N_FACTORS      = 6

CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}

RESULTS_PATH = Path("experiments/synthesis/expOP1_imperfect/results.json")

BONFERRONI_ALPHA = 0.05 / 4   # k=4 comparisons per epsilon level


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


def _add_noise(gt_profiles: np.ndarray, epsilon: float) -> np.ndarray:
    """Add Gaussian noise to profiles, clip to [0, 1]. Fixed noise RNG seed."""
    if epsilon == 0.0:
        return gt_profiles.copy()
    rng = np.random.default_rng(seed=NOISE_RNG_SEED)
    noise = rng.normal(0.0, epsilon, gt_profiles.shape)
    return np.clip(gt_profiles + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Sigma helpers
# ---------------------------------------------------------------------------

def build_correct_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = +0.4
    sigma[:, esc_idx] = -0.4
    return sigma


def build_harmful_sigma0(C_dim: int, A_dim: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    sigma = np.zeros((C_dim, A_dim), dtype=np.float64)
    sigma[:, ac_idx]  = -0.4
    sigma[:, esc_idx] = +0.4
    return sigma


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pre_shift(seed: int, starting_profiles: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Run pre-shift phase: N_PRE_SHIFT decisions, no operator, no campaign.
    Uses CategoryAlertGenerator(seed=seed) — pre-shift RNG state.
    Returns (post-training centroid mu, n_correct).
    """
    scorer = ProfileScorer(starting_profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    pre_gen = CategoryAlertGenerator(seed=seed)
    alerts = pre_gen.generate(N_PRE_SHIFT)

    n_correct = 0
    for alert in alerts:
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = _evaluate(result.action_index, alert.gt_action_index)
        if is_correct:
            n_correct += 1
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    return scorer.mu.copy(), n_correct


def run_post_shift(
    starting_mu: np.ndarray,
    alerts: list,
    registry: OperatorRegistry,
) -> HarnessResult:
    """
    Run post-shift phase via OPHarness (Loop 2 running + optional operator).
    """
    scorer = ProfileScorer(starting_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=N_POST_SHIFT,
        snapshot_interval=WINDOW_SIZE,
        use_synthesis=True,
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    all_results: dict = {}

    for epsilon in EPSILON_LEVELS:
        eps_key = f"{epsilon:.2f}"
        all_results[eps_key] = {
            "condition_A": [], "condition_B": [], "condition_C": [],
            "condition_D": [], "condition_G": [], "condition_H": [],
        }
        print(f"\n=== epsilon={epsilon:.2f} ===", flush=True)

        for seed in SEEDS:
            print(f"  seed {seed}...", flush=True)

            gen_gt = CategoryAlertGenerator(seed=seed)
            gt_profiles    = _build_profiles_tensor(gen_gt)
            noisy_profiles = _add_noise(gt_profiles, epsilon)

            C_dim   = len(CATEGORIES)
            A_dim   = len(ACTIONS)
            ac_idx  = ACTIONS.index("auto_close")
            esc_idx = ACTIONS.index("escalate_incident")

            # Pre-shift — same for all campaign conditions A/B/C/D
            pre_shift_mu, _ = run_pre_shift(seed, noisy_profiles)

            # Post-shift alerts — separate RNG (seed + 10000)
            post_gen = CategoryAlertGenerator(seed=seed + 10000)
            post_shift_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

            # Stable-period alerts — separate RNG (seed + 20000)
            stable_gen = CategoryAlertGenerator(seed=seed + 20000)
            stable_alerts = stable_gen.generate(N_POST_SHIFT)

            sigma_correct = build_correct_sigma0(C_dim, A_dim, ac_idx, esc_idx)
            sigma_harmful = build_harmful_sigma0(C_dim, A_dim, ac_idx, esc_idx)

            # Condition A — campaign baseline, no operator
            reg_A = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            result_A = run_post_shift(pre_shift_mu, post_shift_alerts, reg_A)

            # Condition B — campaign + correct operator, full TTL
            reg_B = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            reg_B.register(OperatorSpec(
                operator_id=f"correct_{seed}_{eps_key}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
            ), pre_shift_mu)
            result_B = run_post_shift(pre_shift_mu, post_shift_alerts, reg_B)

            # Condition C — campaign + harmful operator
            reg_C = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            reg_C.register(OperatorSpec(
                operator_id=f"harmful_{seed}_{eps_key}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_harmful, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
            ), pre_shift_mu)
            result_C = run_post_shift(pre_shift_mu, post_shift_alerts, reg_C)

            # Condition D — campaign + correct operator, half TTL
            reg_D = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            reg_D.register(OperatorSpec(
                operator_id=f"expiring_{seed}_{eps_key}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_HALF,
            ), pre_shift_mu)
            result_D = run_post_shift(pre_shift_mu, post_shift_alerts, reg_D)

            # Condition G — stable period + correct operator
            reg_G = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            reg_G.register(OperatorSpec(
                operator_id=f"stable_op_{seed}_{eps_key}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_correct, lambda_s=LAMBDA_S, ttl_decisions=TTL_FULL,
            ), pre_shift_mu)
            result_G = run_post_shift(pre_shift_mu, stable_alerts, reg_G)

            # Condition H — stable period baseline, no operator
            reg_H = OperatorRegistry(n_categories=C_dim, n_actions=A_dim, n_factors=N_FACTORS)
            result_H = run_post_shift(pre_shift_mu, stable_alerts, reg_H)

            # Record all six conditions
            for cond_key, result in [
                ("condition_A", result_A), ("condition_B", result_B),
                ("condition_C", result_C), ("condition_D", result_D),
                ("condition_G", result_G), ("condition_H", result_H),
            ]:
                all_results[eps_key][cond_key].append({
                    "seed":           seed,
                    "auac":           float(result.auac_result.auac),
                    "t70":            int(result.auac_result.t70)
                                          if result.auac_result.t70 is not None else None,
                    "final_accuracy": float(result.auac_result.final_accuracy),
                    "accuracy_curve": [float(v) for v in result.auac_result.accuracy_curve],
                    "n_expired":      int(result.n_operators_expired),
                })

    # -----------------------------------------------------------------------
    # Statistics — per epsilon level
    # -----------------------------------------------------------------------

    summary_stats: dict = {}

    for epsilon in EPSILON_LEVELS:
        eps_key = f"{epsilon:.2f}"
        cond_data = all_results[eps_key]

        auacs = {
            c: np.array([r["auac"] for r in cond_data[c]])
            for c in ["condition_A", "condition_B", "condition_C",
                      "condition_D", "condition_G", "condition_H"]
        }

        deltas = {
            "B_vs_A": auacs["condition_B"] - auacs["condition_A"],
            "C_vs_A": auacs["condition_C"] - auacs["condition_A"],
            "D_vs_A": auacs["condition_D"] - auacs["condition_A"],
            "G_vs_H": auacs["condition_G"] - auacs["condition_H"],
        }

        p_values = {k: permutation_test(v) for k, v in deltas.items()}

        t70_A = np.array([r["t70"] if r["t70"] is not None else N_POST_SHIFT
                          for r in cond_data["condition_A"]])
        t70_B = np.array([r["t70"] if r["t70"] is not None else N_POST_SHIFT
                          for r in cond_data["condition_B"]])
        t70_speedup = float((t70_A - t70_B).mean())

        fa_A = np.array([r["final_accuracy"] for r in cond_data["condition_A"]])
        fa_B = np.array([r["final_accuracy"] for r in cond_data["condition_B"]])

        gate_pass = bool(
            deltas["B_vs_A"].mean() > 0 and
            p_values["B_vs_A"] < BONFERRONI_ALPHA
        )
        falsified = bool(
            deltas["B_vs_A"].mean() <= 0 and
            t70_speedup <= 0 and
            (fa_B - fa_A).mean() <= 0
        )

        summary_stats[eps_key] = {
            "baseline_auac":    float(auacs["condition_A"].mean()),
            "baseline_auac_std": float(auacs["condition_A"].std()),
            "B_vs_A": {
                "mean":      float(deltas["B_vs_A"].mean()),
                "std":       float(deltas["B_vs_A"].std()),
                "p":         p_values["B_vs_A"],
                "gate_pass": gate_pass,
            },
            "C_vs_A": {
                "mean": float(deltas["C_vs_A"].mean()),
                "std":  float(deltas["C_vs_A"].std()),
                "p":    p_values["C_vs_A"],
            },
            "D_vs_A": {
                "mean": float(deltas["D_vs_A"].mean()),
                "std":  float(deltas["D_vs_A"].std()),
                "p":    p_values["D_vs_A"],
            },
            "G_vs_H": {
                "mean": float(deltas["G_vs_H"].mean()),
                "std":  float(deltas["G_vs_H"].std()),
                "p":    p_values["G_vs_H"],
            },
            "t70_speedup":      t70_speedup,
            "falsified":        falsified,
            "bonferroni_alpha": BONFERRONI_ALPHA,
        }

    # -----------------------------------------------------------------------
    # Gate check output
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXP-OP1-IMPERFECT GATE CHECK")
    print("=" * 70)
    print(f"{'eps':>6}  {'Baseline':>10}  {'B-A delta':>12}  "
          f"{'p (raw)':>9}  {'Bonf<0.0125':>12}  {'C-A delta':>10}")
    print("-" * 70)
    for epsilon in EPSILON_LEVELS:
        eps_key = f"{epsilon:.2f}"
        s   = summary_stats[eps_key]
        ba  = s["B_vs_A"]
        ca  = s["C_vs_A"]
        bonf_str = "[PASS]" if ba["gate_pass"] else "[FAIL]"
        print(f"{epsilon:>6.2f}  {s['baseline_auac']:>10.4f}  "
              f"{ba['mean']:>+12.4f}  {ba['p']:>9.4f}  "
              f"{bonf_str:>12}  {ca['mean']:>+10.4f}")
    print()

    # Primary gate at epsilon=0.10
    s10 = summary_stats["0.10"]
    print(f"PRIMARY GATE (epsilon=0.10):")
    print(f"  AUAC delta B vs A = {s10['B_vs_A']['mean']:+.4f} +/- {s10['B_vs_A']['std']:.4f}")
    print(f"  p (raw)           = {s10['B_vs_A']['p']:.4f}")
    print(f"  Bonferroni a=0.0125: {'PASS' if s10['B_vs_A']['gate_pass'] else 'FAIL'}")
    print(f"  T70 speedup       = {s10['t70_speedup']:.1f} decisions")
    print(f"  C vs A (direction)= {s10['C_vs_A']['mean']:+.4f}  (expect negative)")
    print(f"  G vs H (stable op)= {s10['G_vs_H']['mean']:+.4f}  (expect near-zero)")
    print()

    # Falsification across high-epsilon levels
    eps_high = ["0.10", "0.15", "0.20"]
    if all(summary_stats[e]["falsified"] for e in eps_high):
        print("  [STOP] ALL THREE FALSIFICATION CONDITIONS MET AT eps=0.10, 0.15, 0.20")
        print("  sigma does not improve accuracy at any realistic imperfection level.")
    else:
        print("  Falsification: NOT all three conditions met -- result is not a clean stop.")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"all_results": all_results, "summary_stats": summary_stats,
             "epsilon_levels": EPSILON_LEVELS, "n_seeds": len(SEEDS)},
            f, indent=2,
        )
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------

    from experiments.synthesis.expOP1_imperfect.charts import generate_charts
    generate_charts(all_results, summary_stats, EPSILON_LEVELS)
