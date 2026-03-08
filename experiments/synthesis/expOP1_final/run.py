"""
EXP-OP1-FINAL: Narrow Lambda Window + Full Gate Check at lambda=0.5.

QUESTION: What is the operative lambda window for scalar sigma bias with
Loop 2 running? EXP-OP-MARGIN showed lambda=0.5 is the first passing value;
this experiment narrows the range and runs the full gate battery at lambda=0.5.

PART 1: Narrow lambda sweep [0.3, 0.4, 0.5, 0.6, 0.7, 0.8].
         Conditions A (baseline), B (correct op), C (harmful op). 20 seeds.
PART 2: Full 6-condition design at lambda_gate=0.5.
         Conditions A/B/C/D (post-shift) + G/H (stable-operation controls).
         Gate: B-A delta > 0 AND p < Bonferroni_alpha (0.05/6=0.0083).
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
from src.eval.auac import compute_auac

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
]

LAMBDA_SWEEP  = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LAMBDA_GATE   = 0.5
SIGMA_VALUE   = 0.4
TAU           = 0.1
ETA           = 0.05
ETA_NEG       = 1.0
N_PRE_SHIFT   = 200
N_POST_SHIFT  = 400
WINDOW_SIZE   = 50
N_FACTORS     = 6
TTL_FULL      = 400   # operator lives for all N_POST_SHIFT decisions
TTL_EXPIRE    = 150   # condition D expires mid-run

CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}

BONFERRONI_ALPHA = 0.05 / len(LAMBDA_SWEEP)   # 0.05 / 6 = 0.00833

RESULTS_PATH = Path("experiments/synthesis/expOP1_final/results.json")

C_DIM = len(CATEGORIES)
A_DIM = len(ACTIONS)
AC_IDX  = ACTIONS.index("auto_close")
ESC_IDX = ACTIONS.index("escalate_incident")


# ---------------------------------------------------------------------------
# Shims
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    """Build (C, A, d) tensor from CategoryAlertGenerator nested profiles dict."""
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def _evaluate(result_action_index: int, gt_action_index: int) -> bool:
    return bool(result_action_index == gt_action_index)


# ---------------------------------------------------------------------------
# Sigma builders
# ---------------------------------------------------------------------------

def build_correct_sigma0() -> np.ndarray:
    """Positive bias toward escalate_incident (campaign action), penalise auto_close."""
    sigma = np.zeros((C_DIM, A_DIM), dtype=np.float64)
    sigma[:, AC_IDX]  = +SIGMA_VALUE
    sigma[:, ESC_IDX] = -SIGMA_VALUE
    return sigma


def build_harmful_sigma0() -> np.ndarray:
    """Reversed: push toward auto_close, penalise escalate_incident."""
    sigma = np.zeros((C_DIM, A_DIM), dtype=np.float64)
    sigma[:, AC_IDX]  = -SIGMA_VALUE
    sigma[:, ESC_IDX] = +SIGMA_VALUE
    return sigma


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

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
    return OPHarness(scorer, oracle, registry, config).run(alerts)


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


def extract_accuracy_curve(result: HarnessResult) -> list[float]:
    """Rolling accuracy curve from HarnessResult."""
    return [float(v) for v in result.auac_result.accuracy_curve]


# ---------------------------------------------------------------------------
# PART 1: Narrow Lambda Sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("EXP-OP1-FINAL  PART 1: NARROW LAMBDA SWEEP")
    print("=" * 65)
    print(f"Lambdas: {LAMBDA_SWEEP}")
    print(f"Seeds:   {len(SEEDS)}, N_pre={N_PRE_SHIFT}, N_post={N_POST_SHIFT}")
    print(f"Bonferroni alpha: {BONFERRONI_ALPHA:.5f} (0.05/{len(LAMBDA_SWEEP)})")
    print()

    sigma_correct = build_correct_sigma0()
    sigma_harmful = build_harmful_sigma0()

    sweep_results: dict = {}

    for lv in LAMBDA_SWEEP:
        print(f"--- lambda={lv:.1f} ---", flush=True)
        deltas_B:   list[float] = []
        deltas_C:   list[float] = []
        baseline_auacs: list[float] = []

        for seed in SEEDS:
            gen_pre     = CategoryAlertGenerator(seed=seed)
            gt_profiles = _build_profiles_tensor(gen_pre)
            pre_mu      = run_pre_shift(seed, gt_profiles)

            post_gen    = CategoryAlertGenerator(seed=seed + 10000)
            post_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

            # Condition A — baseline, empty registry
            reg_A = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
            res_A = run_post_shift(pre_mu, post_alerts, reg_A, lambda_override=lv)

            # Condition B — correct operator
            reg_B = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
            reg_B.register(OperatorSpec(
                operator_id=f"correct_{seed}_lv{lv:.1f}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_correct, lambda_s=min(lv, 0.5), ttl_decisions=TTL_FULL,
            ), pre_mu)
            res_B = run_post_shift(pre_mu, post_alerts, reg_B, lambda_override=lv)

            # Condition C — harmful operator
            reg_C = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
            reg_C.register(OperatorSpec(
                operator_id=f"harmful_{seed}_lv{lv:.1f}", claim_type="active_campaign",
                rank=0, sigma_0=sigma_harmful, lambda_s=min(lv, 0.5), ttl_decisions=TTL_FULL,
            ), pre_mu)
            res_C = run_post_shift(pre_mu, post_alerts, reg_C, lambda_override=lv)

            auac_A = float(res_A.auac_result.auac)
            auac_B = float(res_B.auac_result.auac)
            auac_C = float(res_C.auac_result.auac)
            deltas_B.append(auac_B - auac_A)
            deltas_C.append(auac_C - auac_A)
            baseline_auacs.append(auac_A)

        arr_B = np.array(deltas_B)
        arr_C = np.array(deltas_C)
        p_val = permutation_test(arr_B, rng_seed=int(lv * 1000))

        sweep_results[lv] = {
            "mean_delta_B": float(arr_B.mean()),
            "std_delta_B":  float(arr_B.std()),
            "mean_delta_C": float(arr_C.mean()),
            "p_value":      float(p_val),
            "gate_pass":    bool(arr_B.mean() > 0 and p_val < BONFERRONI_ALPHA),
            "baseline_auac": float(np.mean(baseline_auacs)),
            "deltas_B":     [float(x) for x in deltas_B],
            "deltas_C":     [float(x) for x in deltas_C],
        }
        bonf = "[PASS]" if sweep_results[lv]["gate_pass"] else "[----]"
        print(f"  B-A={arr_B.mean():+.4f}  C-A={arr_C.mean():+.4f}  "
              f"p={p_val:.4f}  {bonf}")

    # Print sweep table
    print("\n" + "=" * 75)
    print("PART 1 SWEEP TABLE")
    print("=" * 75)
    print(f"{'lambda':>8}  {'eff.bias':>9}  {'B-A mean':>10}  {'B-A std':>8}  "
          f"{'p_raw':>8}  {'Bonf<{:.4f}'.format(BONFERRONI_ALPHA):>12}  {'C-A mean':>10}")
    print("-" * 75)
    for lv in LAMBDA_SWEEP:
        r   = sweep_results[lv]
        eff = lv * SIGMA_VALUE
        bonf = "[PASS]" if r["gate_pass"] else "[----]"
        print(f"{lv:>8.1f}  {eff:>9.3f}  {r['mean_delta_B']:>+10.4f}  "
              f"{r['std_delta_B']:>8.4f}  {r['p_value']:>8.4f}  {bonf:>12}  "
              f"{r['mean_delta_C']:>+10.4f}")

    passing_lv = [lv for lv in LAMBDA_SWEEP if sweep_results[lv]["gate_pass"]]
    passing_neg = [lv for lv in LAMBDA_SWEEP if sweep_results[lv]["mean_delta_B"] < 0]
    lower_edge = min(passing_lv) if passing_lv else None
    upper_edge = min(passing_neg) if passing_neg else None

    print()
    if lower_edge is not None:
        print(f"  Operative window lower edge: lambda={lower_edge:.1f}")
    else:
        print("  No lambda passed gate in sweep.")
    if upper_edge is not None:
        print(f"  Operative window upper edge (overshoot onset): lambda={upper_edge:.1f}")
    else:
        print("  No overshoot detected in sweep range.")

    # ---------------------------------------------------------------------------
    # PART 2: Full 6-Condition Gate at lambda_gate=0.5
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXP-OP1-FINAL  PART 2: FULL GATE AT lambda=0.5")
    print("=" * 65)
    print("Conditions: A/B/C/D (post-shift) + G/H (stable operation)")
    print()

    part2_results: dict = {
        c: [] for c in ["condition_A", "condition_B", "condition_C",
                        "condition_D", "condition_G", "condition_H"]
    }

    for seed in SEEDS:
        print(f"  seed {seed}...", flush=True)

        gen_pre     = CategoryAlertGenerator(seed=seed)
        gt_profiles = _build_profiles_tensor(gen_pre)
        pre_mu      = run_pre_shift(seed, gt_profiles)

        # Post-shift alerts (campaign)
        post_gen    = CategoryAlertGenerator(seed=seed + 10000)
        post_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

        # Stable alerts (normal distribution — no campaign shift)
        stable_gen    = CategoryAlertGenerator(seed=seed + 20000)
        stable_alerts = stable_gen.generate(N_POST_SHIFT)

        lv = LAMBDA_GATE

        # --- Condition A: baseline (empty registry, post-shift alerts) ---
        reg_A = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        res_A = run_post_shift(pre_mu, post_alerts, reg_A, lambda_override=lv)

        # --- Condition B: correct operator (post-shift alerts) ---
        reg_B = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        reg_B.register(OperatorSpec(
            operator_id=f"correct_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=lv, ttl_decisions=TTL_FULL,
        ), pre_mu)
        res_B = run_post_shift(pre_mu, post_alerts, reg_B, lambda_override=lv)

        # --- Condition C: harmful operator (post-shift alerts) ---
        reg_C = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        reg_C.register(OperatorSpec(
            operator_id=f"harmful_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_harmful, lambda_s=lv, ttl_decisions=TTL_FULL,
        ), pre_mu)
        res_C = run_post_shift(pre_mu, post_alerts, reg_C, lambda_override=lv)

        # --- Condition D: expiring operator (TTL=150, expires mid-run) ---
        reg_D = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        reg_D.register(OperatorSpec(
            operator_id=f"expiring_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=lv, ttl_decisions=TTL_EXPIRE,
        ), pre_mu)
        res_D = run_post_shift(pre_mu, post_alerts, reg_D, lambda_override=lv)

        # --- Condition G: stable + correct operator (stable alerts, correct op) ---
        reg_G = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        reg_G.register(OperatorSpec(
            operator_id=f"stable_correct_{seed}", claim_type="active_campaign",
            rank=0, sigma_0=sigma_correct, lambda_s=lv, ttl_decisions=TTL_FULL,
        ), pre_mu)
        res_G = run_post_shift(pre_mu, stable_alerts, reg_G, lambda_override=lv)

        # --- Condition H: stable baseline (stable alerts, empty registry) ---
        reg_H = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        res_H = run_post_shift(pre_mu, stable_alerts, reg_H, lambda_override=lv)

        def _seed_record(res: HarnessResult, seed_val: int) -> dict:
            auac_val = float(res.auac_result.auac)
            t70_val  = res.auac_result.t70
            return {
                "seed":           seed_val,
                "auac":           auac_val,
                "t70":            int(t70_val) if t70_val is not None else None,
                "accuracy_curve": extract_accuracy_curve(res),
            }

        part2_results["condition_A"].append(_seed_record(res_A, seed))
        part2_results["condition_B"].append(_seed_record(res_B, seed))
        part2_results["condition_C"].append(_seed_record(res_C, seed))
        part2_results["condition_D"].append(_seed_record(res_D, seed))
        part2_results["condition_G"].append(_seed_record(res_G, seed))
        part2_results["condition_H"].append(_seed_record(res_H, seed))

    # ---------------------------------------------------------------------------
    # Gate check at lambda=0.5
    # ---------------------------------------------------------------------------

    auacs = {
        cond: np.array([r["auac"] for r in part2_results[cond]])
        for cond in ["condition_A", "condition_B", "condition_C",
                     "condition_D", "condition_G", "condition_H"]
    }

    deltas_BA = auacs["condition_B"] - auacs["condition_A"]
    deltas_CA = auacs["condition_C"] - auacs["condition_A"]
    deltas_DA = auacs["condition_D"] - auacs["condition_A"]
    deltas_GH = auacs["condition_G"] - auacs["condition_H"]

    p_BA = permutation_test(deltas_BA, rng_seed=1)
    p_CA = permutation_test(deltas_CA, rng_seed=2)
    p_GH = permutation_test(deltas_GH, rng_seed=3)

    t70_A = [r["t70"] for r in part2_results["condition_A"]]
    t70_B = [r["t70"] for r in part2_results["condition_B"]]
    t70_A_safe = [v if v is not None else N_POST_SHIFT for v in t70_A]
    t70_B_safe = [v if v is not None else N_POST_SHIFT for v in t70_B]
    t70_speedup = float(np.mean(t70_A_safe) - np.mean(t70_B_safe))  # positive = B faster

    gate_pass   = bool(deltas_BA.mean() > 0 and p_BA < BONFERRONI_ALPHA)
    falsified   = bool(deltas_BA.mean() <= 0 and t70_speedup <= 0)
    dir_ok      = bool(deltas_CA.mean() < 0)   # harmful goes negative
    stable_ok   = bool(abs(deltas_GH.mean()) < 0.005)  # sigma neutral without shift

    summary_stats = {
        "B_vs_A": {
            "mean":      float(deltas_BA.mean()),
            "std":       float(deltas_BA.std()),
            "p_value":   float(p_BA),
            "gate_pass": gate_pass,
        },
        "C_vs_A": {
            "mean":      float(deltas_CA.mean()),
            "std":       float(deltas_CA.std()),
            "p_value":   float(p_CA),
        },
        "D_vs_A": {
            "mean":      float(deltas_DA.mean()),
            "std":       float(deltas_DA.std()),
        },
        "G_vs_H": {
            "mean":      float(deltas_GH.mean()),
            "std":       float(deltas_GH.std()),
            "p_value":   float(p_GH),
            "stable_ok": stable_ok,
        },
        "t70_speedup":     t70_speedup,
        "directionality":  dir_ok,
        "falsified":       falsified,
        "bonferroni_alpha": float(BONFERRONI_ALPHA),
    }

    print("\n" + "=" * 65)
    print("PART 2: GATE-OP CHECK AT lambda=0.5")
    print("=" * 65)
    print(f"  B vs A: delta={deltas_BA.mean():+.4f} +/- {deltas_BA.std():.4f}  "
          f"p={p_BA:.4f}  Bonf={BONFERRONI_ALPHA:.4f}  "
          f"{'[PASS]' if gate_pass else '[FAIL]'}")
    print(f"  C vs A: delta={deltas_CA.mean():+.4f} +/- {deltas_CA.std():.4f}  "
          f"p={p_CA:.4f}  "
          f"{'[PASS]' if dir_ok else '[FAIL]'} (expect negative)")
    print(f"  D vs A: delta={deltas_DA.mean():+.4f} +/- {deltas_DA.std():.4f}  "
          f"(expiring op)")
    print(f"  G vs H: delta={deltas_GH.mean():+.4f} +/- {deltas_GH.std():.4f}  "
          f"p={p_GH:.4f}  "
          f"{'[PASS]' if stable_ok else '[FAIL]'} (expect ~0, stable operation)")
    print(f"  T70 speedup (A - B): {t70_speedup:+.1f} decisions  "
          f"(positive = B faster to 70%)")
    print(f"  Directionality: {'[PASS]' if dir_ok else '[FAIL]'}")
    print(f"  Stable operation: {'[PASS]' if stable_ok else '[FAIL]'}")
    print()
    if gate_pass:
        print("  GATE-OP: [PASS]")
    elif falsified:
        print("  GATE-OP: [FALSIFIED]  -- delta<=0 and no T70 speedup")
    else:
        print("  GATE-OP: [FAIL]  -- positive delta but not significant")
    print("=" * 65)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------

    results_data = {
        "config": {
            "seeds":          SEEDS,
            "lambda_sweep":   LAMBDA_SWEEP,
            "lambda_gate":    LAMBDA_GATE,
            "sigma_value":    SIGMA_VALUE,
            "n_pre_shift":    N_PRE_SHIFT,
            "n_post_shift":   N_POST_SHIFT,
            "window_size":    WINDOW_SIZE,
            "bonferroni_alpha": float(BONFERRONI_ALPHA),
        },
        "sweep_results": {
            str(lv): sweep_results[lv]
            for lv in LAMBDA_SWEEP
        },
        "all_results": part2_results,
        "summary_stats": summary_stats,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fout:
        json.dump(results_data, fout, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ---------------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------------

    from experiments.synthesis.expOP1_final.charts import generate_charts
    generate_charts(RESULTS_PATH)
