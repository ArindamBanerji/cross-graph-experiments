"""
EXP-OP2: Harmful Claim Resilience + Partial-Accuracy Spectrum.

QUESTIONS:
  (1) Where is the acute-phase sigma benefit concentrated (windows 0-2)?
  (2) Does Loop 2 recover from a harmful operator after it expires?
  (3) Does a correct-but-expired operator leave lasting benefit via Loop 2?
  (4) How wrong can an operator be and still help?

PRIMARY METRIC: T_recovery (decisions until rolling accuracy returns within
1pp of pre-shift baseline and holds for 2 consecutive windows).

Nine conditions at lambda=0.5: A (baseline), B (correct full), B-exp
(correct expiring), C (harmful full), C-exp (harmful expiring), P-75/50/25/0
(partial accuracy spectrum).
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
]

LAMBDA_S          = 0.5
SIGMA_VALUE       = 0.4
N_PRE_SHIFT       = 200
N_POST_SHIFT      = 400
WINDOW_SIZE       = 50
TAU               = 0.1
ETA               = 0.05
ETA_NEG           = 1.0
TTL_FULL          = 400
TTL_HALF          = 150
PARTIAL_RNG_SEED  = 99

CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}

RECOVERY_THRESHOLD_PP = 1.0
RECOVERY_HOLD_WINDOWS = 2

RESULTS_PATH = Path("experiments/synthesis/expOP2_harmful/results.json")

N_FACTORS = 6
C_DIM = len(CATEGORIES)
A_DIM = len(ACTIONS)
AC_IDX  = ACTIONS.index("auto_close")
ESC_IDX = ACTIONS.index("escalate_incident")

conditions = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]


# ---------------------------------------------------------------------------
# Shims
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def _evaluate(result_action_index: int, gt_action_index: int) -> bool:
    return bool(result_action_index == gt_action_index)


def compute_t_recovery(
    accuracy_curve: list,
    baseline_pre_shift: float,
    threshold_pp: float = RECOVERY_THRESHOLD_PP,
    hold_windows: int = RECOVERY_HOLD_WINDOWS,
    sentinel: int = None,
) -> int:
    """
    First decision where accuracy >= (baseline_pre_shift - threshold_pp/100)
    and holds for hold_windows consecutive windows.
    Returns decision number (window_index * WINDOW_SIZE). Sentinel if never.
    """
    if sentinel is None:
        sentinel = N_POST_SHIFT
    threshold = baseline_pre_shift - threshold_pp / 100.0
    n = len(accuracy_curve)
    for i in range(n - hold_windows + 1):
        if all(accuracy_curve[i + j] >= threshold for j in range(hold_windows)):
            return i * WINDOW_SIZE
    return sentinel


# ---------------------------------------------------------------------------
# Sigma constructors
# ---------------------------------------------------------------------------

def build_correct_sigma(C: int, A: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    s = np.zeros((C, A), dtype=np.float64)
    s[:, ac_idx]  = +SIGMA_VALUE
    s[:, esc_idx] = -SIGMA_VALUE
    return s


def build_harmful_sigma(C: int, A: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    return -build_correct_sigma(C, A, ac_idx, esc_idx)


def build_partial_sigma(
    C: int, A: int, ac_idx: int, esc_idx: int, fraction_correct: float
) -> np.ndarray:
    """
    fraction_correct in {0.0, 0.25, 0.50, 0.75, 1.0}.
    Start from correct sigma, flip (1-fraction_correct) of non-zero cells.
    Uses module-level PARTIAL_RNG for deterministic flip pattern across seeds.
    """
    s = build_correct_sigma(C, A, ac_idx, esc_idx)
    nonzero_indices = (
        [(c, ac_idx)  for c in range(C)] +
        [(c, esc_idx) for c in range(C)]
    )
    n_flip = int(len(nonzero_indices) * (1.0 - fraction_correct))
    if n_flip > 0:
        flip_choices = PARTIAL_RNG.choice(len(nonzero_indices), size=n_flip, replace=False)
        for idx in flip_choices:
            r, col = nonzero_indices[idx]
            s[r, col] = -s[r, col]
    return s


# ---------------------------------------------------------------------------
# Pre-build partial sigmas at module level (same flip pattern across seeds)
# ---------------------------------------------------------------------------

PARTIAL_RNG = np.random.default_rng(seed=PARTIAL_RNG_SEED)

sigma_P100 = build_correct_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX)
sigma_P75  = build_partial_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, 0.75)
sigma_P50  = build_partial_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, 0.50)
sigma_P25  = build_partial_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, 0.25)
sigma_P0   = build_harmful_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_pre_shift(seed: int, profiles: np.ndarray) -> tuple:
    """
    Run N_PRE_SHIFT pre-shift decisions.
    Returns (post-training mu, baseline_pre_shift_acc).
    baseline_pre_shift_acc = fraction correct in last 100 pre-shift decisions.
    """
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_PRE_SHIFT)
    last100_correct = []
    for i, alert in enumerate(alerts):
        result = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = _evaluate(result.action_index, alert.gt_action_index)
        if i >= N_PRE_SHIFT - 100:
            last100_correct.append(is_correct)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    baseline_acc = float(np.mean(last100_correct))
    return scorer.mu.copy(), baseline_acc


def run_post_shift(
    starting_mu: np.ndarray,
    alerts: list,
    registry: OperatorRegistry,
    lambda_override: float = LAMBDA_S,
) -> HarnessResult:
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


def make_registry(spec: OperatorSpec | None, mu: np.ndarray) -> OperatorRegistry:
    reg = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
    if spec is not None:
        reg.register(spec, mu)
    return reg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("EXP-OP2: HARMFUL CLAIM RESILIENCE + PARTIAL ACCURACY SPECTRUM")
    print("=" * 65)
    print(f"Seeds: {len(SEEDS)}, lambda={LAMBDA_S}, N_pre={N_PRE_SHIFT}, N_post={N_POST_SHIFT}")
    print(f"Conditions: {conditions}")
    print()

    # results[cond] = list of dicts, one per seed
    results: dict = {cond: [] for cond in conditions}

    cond_specs: dict = {
        "A":     None,
        "B":     lambda mu, seed: OperatorSpec(
                     f"B_{seed}", "active_campaign", 0,
                     sigma_P100, LAMBDA_S, TTL_FULL),
        "B-exp": lambda mu, seed: OperatorSpec(
                     f"Bexp_{seed}", "active_campaign", 0,
                     sigma_P100, LAMBDA_S, TTL_HALF),
        "C":     lambda mu, seed: OperatorSpec(
                     f"C_{seed}", "active_campaign", 0,
                     sigma_P0, LAMBDA_S, TTL_FULL),
        "C-exp": lambda mu, seed: OperatorSpec(
                     f"Cexp_{seed}", "active_campaign", 0,
                     sigma_P0, LAMBDA_S, TTL_HALF),
        "P-75":  lambda mu, seed: OperatorSpec(
                     f"P75_{seed}", "active_campaign", 0,
                     sigma_P75, LAMBDA_S, TTL_FULL),
        "P-50":  lambda mu, seed: OperatorSpec(
                     f"P50_{seed}", "active_campaign", 0,
                     sigma_P50, LAMBDA_S, TTL_FULL),
        "P-25":  lambda mu, seed: OperatorSpec(
                     f"P25_{seed}", "active_campaign", 0,
                     sigma_P25, LAMBDA_S, TTL_FULL),
        "P-0":   lambda mu, seed: OperatorSpec(
                     f"P0_{seed}", "active_campaign", 0,
                     sigma_P0, LAMBDA_S, TTL_FULL),
    }

    for seed_idx, seed in enumerate(SEEDS):
        print(f"  seed {seed} ({seed_idx+1}/{len(SEEDS)})...", flush=True)

        gen_pre     = CategoryAlertGenerator(seed=seed)
        gt_profiles = _build_profiles_tensor(gen_pre)
        pre_mu, baseline_acc = run_pre_shift(seed, gt_profiles)

        post_gen    = CategoryAlertGenerator(seed=seed + 10000)
        post_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

        for cond in conditions:
            spec_fn = cond_specs[cond]
            spec    = spec_fn(pre_mu, seed) if spec_fn is not None else None
            reg     = make_registry(spec, pre_mu)
            res     = run_post_shift(pre_mu, post_alerts, reg)

            acc_curve = [float(v) for v in res.auac_result.accuracy_curve]
            t_rec = compute_t_recovery(acc_curve, baseline_acc)

            results[cond].append({
                "auac":               float(res.auac_result.auac),
                "accuracy_curve":     acc_curve,
                "t_recovery":         int(t_rec),
                "baseline_pre_shift": float(baseline_acc),
                "n_expired":          int(res.n_operators_expired),
            })

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------

    n_seeds = len(SEEDS)

    def get_auacs(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["auac"] for s in range(n_seeds)])

    def get_curves(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["accuracy_curve"] for s in range(n_seeds)])

    auacs_A = get_auacs("A")
    curves_A = get_curves("A")

    # Acute-phase deltas (windows 0-2)
    acute_deltas = {}
    for cond in ["B", "C", "C-exp", "P-50"]:
        acute_deltas[cond] = {}
        c_curves = get_curves(cond)
        for w in [0, 1, 2]:
            d = c_curves[:, w] - curves_A[:, w]
            acute_deltas[cond][w] = (float(d.mean()), float(d.std()))

    # Post-expiry deltas (windows 3-7, after TTL=150 expires)
    post_expiry_deltas = {}
    for cond in ["B-exp", "C-exp"]:
        c_curves = get_curves(cond)
        per_seed = np.mean(c_curves[:, 3:8] - curves_A[:, 3:8], axis=1)
        post_expiry_deltas[cond] = (float(per_seed.mean()), float(per_seed.std()))

    # AUAC deltas + p-values
    auac_deltas = {}
    auac_pvals  = {}
    for cond in conditions[1:]:
        d = get_auacs(cond) - auacs_A
        auac_deltas[cond] = float(d.mean())
        auac_pvals[cond]  = float(permutation_test(d, rng_seed=conditions.index(cond)))

    # T_recovery
    t_rec_stats = {}
    for cond in conditions:
        t_vals = np.array([results[cond][s]["t_recovery"] for s in range(n_seeds)])
        t_rec_stats[cond] = (float(t_vals.mean()), float(t_vals.std()))

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXP-OP2: HARMFUL CLAIM RESILIENCE — FULL REPORT")
    print("=" * 70)

    print("\n--- ACUTE-PHASE DELTAS (windows 0-2, decisions 0-150) ---")
    print(f"{'Condition':>8}  {'Win 0 (0-50)':>14}  {'Win 1 (50-100)':>16}  {'Win 2 (100-150)':>18}")
    print("-" * 62)
    for cond in ["B", "C", "C-exp", "P-50"]:
        row = f"{cond:>8}"
        for w in [0, 1, 2]:
            m, s = acute_deltas[cond][w]
            row += f"  {m:+.4f} +/-{s:.4f}"
        print(row)

    print("\n--- POST-EXPIRY (windows 3-7, after TTL=150 expires) ---")
    for cond in ["B-exp", "C-exp"]:
        m, s = post_expiry_deltas[cond]
        if cond == "B-exp":
            label = "INDIRECT PATH CONFIRMED" if m > 0.0005 else "NOT DETECTED"
        else:
            label = "LOOP2 RECOVERED" if m > 0 else "DID NOT RECOVER"
        print(f"  {cond} post-expiry mean delta = {m:+.4f} +/- {s:.4f}  [{label}]")

    print("\n--- T_RECOVERY (decisions to return within 1pp of pre-shift) ---")
    sorted_conds = sorted(conditions, key=lambda c: t_rec_stats[c][0])
    for cond in sorted_conds:
        m, s = t_rec_stats[cond]
        sent_pct = np.mean([results[cond][ss]["t_recovery"] >= N_POST_SHIFT
                            for ss in range(n_seeds)]) * 100
        sent_str = f"  ({sent_pct:.0f}% never recovered)" if sent_pct > 0 else ""
        print(f"  {cond:>6}: {m:6.1f} +/- {s:5.1f} decisions{sent_str}")

    print("\n--- AUAC DELTAS + p-VALUES ---")
    for cond in conditions[1:]:
        d   = auac_deltas[cond]
        p   = auac_pvals[cond]
        sig = "[p<0.05]" if p < 0.05 else "[n.s.]"
        print(f"  {cond:>6}: AUAC delta = {d:+.4f},  p = {p:.4f}  {sig}")

    print("\n--- PARTIAL ACCURACY THRESHOLD ---")
    partial_pairs = [("P-0", 0), ("P-25", 25), ("P-50", 50), ("P-75", 75), ("B", 100)]
    threshold_pct = None
    for label, pct in partial_pairs:
        d = auac_deltas[label]
        p = auac_pvals[label]
        if d > 0 and p < 0.05:
            sign = "NET POSITIVE"
            if threshold_pct is None:
                threshold_pct = pct
        elif d > 0:
            sign = "MARGINAL"
        else:
            sign = "NET NEGATIVE"
        print(f"  {pct:3d}% correct: delta={d:+.4f},  p={p:.4f}  [{sign}]")
    if threshold_pct is not None:
        print(f"\n  Threshold: operators with >= {threshold_pct}% correct cells are net positive")
    else:
        print("\n  Threshold: no condition reached p<0.05 net positive in this sweep")

    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    results_data = {
        "per_seed_results": {
            cond: [
                {
                    "seed":               SEEDS[s],
                    "auac":               float(results[cond][s]["auac"]),
                    "accuracy_curve":     [float(v) for v in results[cond][s]["accuracy_curve"]],
                    "t_recovery":         int(results[cond][s]["t_recovery"]),
                    "baseline_pre_shift": float(results[cond][s]["baseline_pre_shift"]),
                }
                for s in range(n_seeds)
            ]
            for cond in conditions
        },
        "summary": {
            "auac_deltas":  {c: float(auac_deltas[c]) for c in conditions[1:]},
            "auac_p_values": {c: float(auac_pvals[c])  for c in conditions[1:]},
            "t_recovery_means": {
                c: float(t_rec_stats[c][0]) for c in conditions
            },
            "acute_phase_deltas": {
                cond: {
                    f"window_{w}": float(acute_deltas[cond][w][0])
                    for w in [0, 1, 2]
                }
                for cond in ["B", "C", "C-exp", "P-50"]
            },
            "post_expiry_deltas": {
                c: float(post_expiry_deltas[c][0]) for c in ["B-exp", "C-exp"]
            },
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fout:
        json.dump(results_data, fout, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ---------------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------------

    from experiments.synthesis.expOP2_harmful.charts import generate_charts
    generate_charts(json.load(open(RESULTS_PATH)))
