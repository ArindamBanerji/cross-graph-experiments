"""
EXP-S2-REPRO: Poisoning Resilience — Three-Arm Replication.

Arms:
  Arm 0 — Frozen-synthesis replication (CategoryAlertGenerator, no centroid update,
           SynthesisBias directly, 3 poison rates, 10 seeds).
  Arm A — Production condition (CategoryAlertGenerator, OPHarness, N_pre=200,
           N_post=400, lambda=0.5, 4 poison rates, 20 seeds).
  Arm B — Realistic AUAC (RealisticAlertGenerator mode='combined', OPHarness,
           SOC taxonomy, 3 poison rates, 10 seeds; DOMAIN EXPERT REVIEW gate).

Gate logic:
  Arm 0: accuracy degradation at 20% poison <= 2pp  → PASS / FAIL
  Arm A: p90 T_recovery < 100 decisions AND never_recover_rate <= 5% at 20% poison
  Arm B: DOMAIN EXPERT REVIEW (descriptive AUAC only, no numeric gate)

Outputs:
  experiments/synthesis/expS2_repro/results.json
  paper_figures/expS2r_*.{pdf,png}  (via charts.py, 4 charts x 2 formats = 8 files)
"""
from __future__ import annotations

import json
import sys
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
from src.models.synthesis import SynthesisBias
from src.models.operator_spec import OperatorSpec
from src.models.operator_registry import OperatorRegistry
from src.eval.op_harness import OPHarness, HarnessConfig, HarnessResult

from experiments.fx1_proxy_real.realistic_generator import (
    RealisticAlertGenerator, SOC_CATEGORIES, SOC_ACTIONS, SOCDomainConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS_10 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
SEEDS_20 = SEEDS_10 + [7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]

LAMBDA_S         = 0.5
SIGMA_VALUE      = 0.4
N_PRE            = 200
N_POST           = 400
WINDOW_SIZE      = 50
TAU              = 0.1
ETA              = 0.05
ETA_NEG          = 1.0
TTL_FULL         = 400

RECOVERY_THRESHOLD_PP = 1.0
RECOVERY_HOLD_WINDOWS = 2

# Bridge-common taxonomy (Arm 0 + Arm A)
C_DIM    = len(CATEGORIES)   # 5
A_DIM    = len(ACTIONS)      # 4
N_FACTORS = 6
AC_IDX   = ACTIONS.index("auto_close")          # 0 — penalize (+0.4)
ESC_IDX  = ACTIONS.index("escalate_incident")   # 3 — boost (-0.4)

# SOC taxonomy (Arm B)
C_SOC       = len(SOC_CATEGORIES)   # 5
A_SOC       = len(SOC_ACTIONS)      # 4
SUPP_SOC_IDX = SOC_ACTIONS.index("suppress")   # 2 — penalize (+0.4)
ESC_SOC_IDX  = SOC_ACTIONS.index("escalate")   # 0 — boost (-0.4)

POISON_RATES_A0 = [0.0, 0.20, 0.40]
POISON_RATES_A  = [0.0, 0.10, 0.20, 0.30]
POISON_RATES_B  = [0.0, 0.20, 0.40]

RESULTS_PATH = Path("experiments/synthesis/expS2_repro/results.json")


# ---------------------------------------------------------------------------
# Sigma constructors
# ---------------------------------------------------------------------------

def build_poisoned_sigma(
    C: int,
    A: int,
    penalize_idx: int,
    boost_idx: int,
    poison_rate: float,
    rng_seed: int,
) -> np.ndarray:
    """
    Build sigma with poison_rate fraction of non-zero cells flipped.

    Correct base: s[:, penalize_idx] = +SIGMA_VALUE (penalise action)
                  s[:, boost_idx]    = -SIGMA_VALUE (boost action)
    Poisoning: flip sign on ceil(2*C * poison_rate) randomly chosen cells.
    RNG is fixed per arm so the flip pattern is the same across seeds.
    """
    s = np.zeros((C, A), dtype=np.float64)
    s[:, penalize_idx] = +SIGMA_VALUE
    s[:, boost_idx]    = -SIGMA_VALUE

    nonzero = (
        [(c, penalize_idx) for c in range(C)] +
        [(c, boost_idx)    for c in range(C)]
    )
    n_flip = int(len(nonzero) * poison_rate)
    if n_flip > 0:
        rng = np.random.default_rng(seed=rng_seed)
        flip_idx = rng.choice(len(nonzero), size=n_flip, replace=False)
        for i in flip_idx:
            r, col = nonzero[i]
            s[r, col] = -s[r, col]
    return s


# Pre-build all sigma tensors once at module level (same flip pattern per arm)
SIGMAS_A0 = {pr: build_poisoned_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, pr, rng_seed=2024)
             for pr in POISON_RATES_A0}
SIGMAS_A  = {pr: build_poisoned_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, pr, rng_seed=2025)
             for pr in POISON_RATES_A}
SIGMAS_B  = {pr: build_poisoned_sigma(C_SOC, A_SOC, SUPP_SOC_IDX, ESC_SOC_IDX, pr, rng_seed=2026)
             for pr in POISON_RATES_B}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def compute_t_recovery(
    accuracy_curve: list,
    baseline_pre_shift: float,
    threshold_pp: float = RECOVERY_THRESHOLD_PP,
    hold_windows: int = RECOVERY_HOLD_WINDOWS,
    sentinel: int = None,
) -> int:
    if sentinel is None:
        sentinel = N_POST
    threshold = baseline_pre_shift - threshold_pp / 100.0
    n = len(accuracy_curve)
    for i in range(n - hold_windows + 1):
        if all(accuracy_curve[i + j] >= threshold for j in range(hold_windows)):
            return i * WINDOW_SIZE
    return sentinel


def make_registry(
    n_categories: int,
    n_actions: int,
    n_factors: int,
    spec: OperatorSpec | None,
    mu: np.ndarray,
) -> OperatorRegistry:
    reg = OperatorRegistry(
        n_categories=n_categories,
        n_actions=n_actions,
        n_factors=n_factors,
    )
    if spec is not None:
        reg.register(spec, mu)
    return reg


# ---------------------------------------------------------------------------
# Arm 0 — Frozen synthesis (no centroid update, SynthesisBias directly)
# ---------------------------------------------------------------------------

def run_arm0_seed(seed: int, sigma: np.ndarray, lambda_val: float) -> dict:
    """
    Frozen synthesis: score with SynthesisBias but never call update().
    Uses GT profiles from CategoryAlertGenerator(seed=seed).
    Returns {accuracy, correct_flags, n_decisions}.
    """
    gen = CategoryAlertGenerator(seed=seed)
    profiles = _build_profiles_tensor(gen)
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    bias = SynthesisBias(
        sigma=sigma.copy(),
        active_claims=1 if lambda_val > 0 else 0,
        lambda_coupling=lambda_val,
    )
    alerts = gen.generate(N_POST)
    correct_flags: list[bool] = []
    for alert in alerts:
        result = scorer.score(alert.factors, alert.category_index, synthesis=bias)
        correct_flags.append(bool(result.action_index == alert.gt_action_index))
        # NO update() — centroids frozen
    return {
        "accuracy":      float(np.mean(correct_flags)),
        "correct_flags": [int(b) for b in correct_flags],
        "n_decisions":   len(correct_flags),
    }


# ---------------------------------------------------------------------------
# Arm A — Production condition via OPHarness (bridge-common taxonomy)
# ---------------------------------------------------------------------------

def run_pre_shift_centroidal(seed: int, profiles: np.ndarray) -> tuple:
    """
    Run N_PRE decisions with no synthesis to warm up centroids.
    Returns (post_mu, baseline_pre_shift_acc).
    """
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_PRE)
    last100: list[bool] = []
    for i, alert in enumerate(alerts):
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = bool(result.action_index == alert.gt_action_index)
        if i >= N_PRE - 100:
            last100.append(is_correct)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    return scorer.mu.copy(), float(np.mean(last100))


def run_post_shift_harness(
    starting_mu: np.ndarray,
    alerts: list,
    registry: OperatorRegistry,
    n_post: int = N_POST,
) -> HarnessResult:
    scorer = ProfileScorer(starting_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=n_post,
        snapshot_interval=WINDOW_SIZE,
        use_synthesis=True,
        lambda_override=LAMBDA_S,
        window_size=WINDOW_SIZE,
    )
    return OPHarness(scorer, oracle, registry, config).run(alerts)


def run_arm_a_seed(seed: int, sigma: np.ndarray) -> dict:
    """
    Full Arm A run for one seed + one poison rate:
    pre-shift training → post-shift OPHarness with poisoned OperatorSpec.
    """
    gen_pre   = CategoryAlertGenerator(seed=seed)
    profiles  = _build_profiles_tensor(gen_pre)
    pre_mu, baseline_acc = run_pre_shift_centroidal(seed, profiles)

    post_gen    = CategoryAlertGenerator(seed=seed + 20000)
    post_alerts = post_gen.generate(N_POST)

    spec = OperatorSpec(
        operator_id=f"S2A_{seed}",
        claim_type="active_campaign",
        rank=0,
        sigma_0=sigma.copy(),
        lambda_s=LAMBDA_S,
        ttl_decisions=TTL_FULL,
    )
    reg = make_registry(C_DIM, A_DIM, N_FACTORS, spec, pre_mu)
    res = run_post_shift_harness(pre_mu, post_alerts, reg, n_post=N_POST)

    acc_curve = [float(v) for v in res.auac_result.accuracy_curve]
    t_rec = compute_t_recovery(acc_curve, baseline_acc)

    return {
        "auac":               float(res.auac_result.auac),
        "accuracy_curve":     acc_curve,
        "t_recovery":         int(t_rec),
        "baseline_pre_shift": float(baseline_acc),
        "final_accuracy":     float(res.auac_result.final_accuracy),
    }


# ---------------------------------------------------------------------------
# Arm B — Realistic AUAC via OPHarness (SOC taxonomy)
# ---------------------------------------------------------------------------

def run_pre_shift_realistic(seed: int) -> tuple:
    """
    Warm up centroids on SOC combined-mode alerts for N_PRE decisions.
    Returns (post_mu_soc, baseline_acc).
    """
    soc_profiles = SOCDomainConfig.get_profile_centroids()   # (5, 4, 6)
    scorer = ProfileScorer(soc_profiles.copy(), SOC_ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen = RealisticAlertGenerator(mode="combined", seed=seed)
    alerts = gen.generate(N_PRE)
    last100: list[bool] = []
    for i, alert in enumerate(alerts):
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = bool(result.action_index == alert.gt_action_index)
        if i >= N_PRE - 100:
            last100.append(is_correct)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    return scorer.mu.copy(), float(np.mean(last100))


def run_post_shift_harness_soc(
    starting_mu: np.ndarray,
    alerts: list,
    registry: OperatorRegistry,
    n_post: int = N_POST,
) -> HarnessResult:
    scorer = ProfileScorer(starting_mu.copy(), SOC_ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=n_post,
        snapshot_interval=WINDOW_SIZE,
        use_synthesis=True,
        lambda_override=LAMBDA_S,
        window_size=WINDOW_SIZE,
    )
    return OPHarness(scorer, oracle, registry, config).run(alerts)


def run_arm_b_seed(seed: int, sigma: np.ndarray) -> dict:
    """
    Full Arm B run for one seed + one poison rate:
    SOC pre-shift training → SOC post-shift OPHarness.
    """
    pre_mu_soc, baseline_acc = run_pre_shift_realistic(seed)

    post_gen = RealisticAlertGenerator(mode="combined", seed=seed + 20000)
    post_alerts = post_gen.generate(N_POST)

    spec = OperatorSpec(
        operator_id=f"S2B_{seed}",
        claim_type="active_campaign",
        rank=0,
        sigma_0=sigma.copy(),
        lambda_s=LAMBDA_S,
        ttl_decisions=TTL_FULL,
    )
    reg = make_registry(C_SOC, A_SOC, N_FACTORS, spec, pre_mu_soc)
    res = run_post_shift_harness_soc(pre_mu_soc, post_alerts, reg, n_post=N_POST)

    acc_curve = [float(v) for v in res.auac_result.accuracy_curve]
    t_rec = compute_t_recovery(acc_curve, baseline_acc)

    return {
        "auac":               float(res.auac_result.auac),
        "accuracy_curve":     acc_curve,
        "t_recovery":         int(t_rec),
        "baseline_pre_shift": float(baseline_acc),
        "final_accuracy":     float(res.auac_result.final_accuracy),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("EXP-S2-REPRO: POISONING RESILIENCE — THREE ARMS")
    print("=" * 65)
    print(f"Arm 0: {len(SEEDS_10)} seeds x {len(POISON_RATES_A0)} poison rates = "
          f"{len(SEEDS_10)*len(POISON_RATES_A0)} runs")
    print(f"Arm A: {len(SEEDS_20)} seeds x {len(POISON_RATES_A)} poison rates = "
          f"{len(SEEDS_20)*len(POISON_RATES_A)} runs")
    print(f"Arm B: {len(SEEDS_10)} seeds x {len(POISON_RATES_B)} poison rates = "
          f"{len(SEEDS_10)*len(POISON_RATES_B)} runs")
    total = (len(SEEDS_10)*len(POISON_RATES_A0) +
             len(SEEDS_20)*len(POISON_RATES_A) +
             len(SEEDS_10)*len(POISON_RATES_B))
    print(f"Total: {total} runs")
    print()

    # -----------------------------------------------------------------------
    # Arm 0 — Frozen synthesis
    # -----------------------------------------------------------------------
    print("--- ARM 0: Frozen synthesis replication ---")
    arm0: dict = {pr: [] for pr in POISON_RATES_A0}

    for pr in POISON_RATES_A0:
        sigma = SIGMAS_A0[pr]
        for seed in SEEDS_10:
            r = run_arm0_seed(seed, sigma, lambda_val=LAMBDA_S)
            arm0[pr].append({"seed": seed, **r})
        mean_acc = float(np.mean([r["accuracy"] for r in arm0[pr]]))
        print(f"  poison={pr*100:.0f}%: mean_acc={mean_acc*100:.2f}%")

    acc_0pct  = float(np.mean([r["accuracy"] for r in arm0[0.00]]))
    acc_20pct = float(np.mean([r["accuracy"] for r in arm0[0.20]]))
    arm0_degradation_20 = (acc_0pct - acc_20pct) * 100
    arm0_gate = "PASS" if arm0_degradation_20 <= 2.0 else "FAIL"
    print(f"  Degradation @ 20% poison: {arm0_degradation_20:.2f}pp  [{arm0_gate}]")

    # -----------------------------------------------------------------------
    # Arm A — Production condition
    # -----------------------------------------------------------------------
    print("\n--- ARM A: Production condition (OPHarness, centroidal) ---")
    arm_a: dict = {pr: [] for pr in POISON_RATES_A}

    for pr in POISON_RATES_A:
        sigma = SIGMAS_A[pr]
        for s_idx, seed in enumerate(SEEDS_20):
            r = run_arm_a_seed(seed, sigma)
            arm_a[pr].append({"seed": seed, **r})
            if (s_idx + 1) % 5 == 0:
                print(f"  poison={pr*100:.0f}%: seed {seed} ({s_idx+1}/{len(SEEDS_20)})", flush=True)

    # Gate evaluation at 20% poison
    t_recs_20 = np.array([r["t_recovery"] for r in arm_a[0.20]])
    p90_t_rec = float(np.percentile(t_recs_20, 90))
    never_recover_rate = float(np.mean(t_recs_20 >= N_POST))
    arma_gate = (
        "PASS" if (p90_t_rec < 100 and never_recover_rate <= 0.05) else "FAIL"
    )
    print(f"\n  Gate @ 20% poison:")
    print(f"    p90 T_recovery = {p90_t_rec:.0f} decisions  (gate: < 100)")
    print(f"    never_recover  = {never_recover_rate*100:.1f}%  (gate: <= 5%)")
    print(f"    --> ARM A gate: {arma_gate}")

    # -----------------------------------------------------------------------
    # Arm B — Realistic AUAC (SOC taxonomy)
    # -----------------------------------------------------------------------
    print("\n--- ARM B: Realistic AUAC (SOC, combined mode) ---")
    arm_b: dict = {pr: [] for pr in POISON_RATES_B}

    for pr in POISON_RATES_B:
        sigma = SIGMAS_B[pr]
        for seed in SEEDS_10:
            r = run_arm_b_seed(seed, sigma)
            arm_b[pr].append({"seed": seed, **r})
        mean_auac = float(np.mean([r["auac"] for r in arm_b[pr]]))
        print(f"  poison={pr*100:.0f}%: mean_AUAC={mean_auac:.4f}")

    print("  --> ARM B gate: DOMAIN EXPERT REVIEW (descriptive only)")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print("\nArm 0 — Frozen synthesis (accuracy at N=400):")
    for pr in POISON_RATES_A0:
        accs = [r["accuracy"] for r in arm0[pr]]
        print(f"  {pr*100:.0f}%: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    print(f"  Gate (<=2pp @ 20%): {arm0_gate}  ({arm0_degradation_20:.2f}pp)")

    print("\nArm A — Production (T_recovery + AUAC):")
    for pr in POISON_RATES_A:
        t_arr = np.array([r["t_recovery"] for r in arm_a[pr]])
        a_arr = np.array([r["auac"]       for r in arm_a[pr]])
        nr = float(np.mean(t_arr >= N_POST)) * 100
        print(f"  {pr*100:.0f}%: T_rec={np.mean(t_arr):.0f}+/-{np.std(t_arr):.0f}  "
              f"AUAC={np.mean(a_arr):.4f}  never_rec={nr:.1f}%")
    print(f"  Gate (p90 T_rec<100, never_rec<=5% @ 20%): {arma_gate}")

    print("\nArm B — Realistic AUAC:")
    for pr in POISON_RATES_B:
        a_arr = np.array([r["auac"] for r in arm_b[pr]])
        t_arr = np.array([r["t_recovery"] for r in arm_b[pr]])
        print(f"  {pr*100:.0f}%: AUAC={np.mean(a_arr):.4f}+/-{np.std(a_arr):.4f}  "
              f"T_rec={np.mean(t_arr):.0f}")
    print("  Gate: DOMAIN EXPERT REVIEW")

    print("=" * 65)

    # -----------------------------------------------------------------------
    # Save results.json
    # -----------------------------------------------------------------------
    def _arm_dict(arm_data: dict) -> dict:
        return {
            str(pr): [
                {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                 for k, v in entry.items()}
                for entry in arm_data[pr]
            ]
            for pr in arm_data
        }

    results = {
        "arm0_frozen": _arm_dict(arm0),
        "arm_a_production": _arm_dict(arm_a),
        "arm_b_realistic": _arm_dict(arm_b),
        "summary": {
            "arm0": {
                "degradation_20pct_pp": float(arm0_degradation_20),
                "gate": arm0_gate,
                "mean_acc_by_poison": {
                    str(pr): float(np.mean([r["accuracy"] for r in arm0[pr]]))
                    for pr in POISON_RATES_A0
                },
            },
            "arm_a": {
                "gate": arma_gate,
                "p90_t_recovery_at_20pct": float(p90_t_rec),
                "never_recover_rate_at_20pct": float(never_recover_rate),
                "mean_auac_by_poison": {
                    str(pr): float(np.mean([r["auac"] for r in arm_a[pr]]))
                    for pr in POISON_RATES_A
                },
                "mean_t_recovery_by_poison": {
                    str(pr): float(np.mean([r["t_recovery"] for r in arm_a[pr]]))
                    for pr in POISON_RATES_A
                },
            },
            "arm_b": {
                "gate": "DOMAIN EXPERT REVIEW",
                "mean_auac_by_poison": {
                    str(pr): float(np.mean([r["auac"] for r in arm_b[pr]]))
                    for pr in POISON_RATES_B
                },
                "mean_t_recovery_by_poison": {
                    str(pr): float(np.mean([r["t_recovery"] for r in arm_b[pr]]))
                    for pr in POISON_RATES_B
                },
            },
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fout:
        json.dump(results, fout, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    from experiments.synthesis.expS2_repro.charts import generate_charts
    generate_charts(json.load(open(RESULTS_PATH)))
