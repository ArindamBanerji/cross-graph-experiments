"""
EXP-OP3: Residual Tracker as Early-Warning Diagnostic.

QUESTION: Does ||R(t)||_F / ||R(0)||_F distinguish a correct from a harmful
operator within the first 50-100 decisions (window 0-1)?

R[c,a,:](t) = mu_tilde[c,a,:] - mu(t)
  Correct operator: Loop 2 moves mu(t) TOWARD mu_tilde -> R decays
  Harmful operator: Loop 2 moves mu(t) AWAY from mu_tilde -> R plateaus/grows

Six conditions at lambda=0.5: A (baseline), B (correct full), C (harmful full),
P-50 (partial), B-short (correct TTL=100), D-stale (correct TTL=1).
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
from src.models.residual_tracker import ResidualTracker
from src.eval.op_harness import OPHarness, HarnessConfig, HarnessResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
]

LAMBDA_S    = 0.5
SIGMA_VALUE = 0.4
N_PRE       = 200
N_POST      = 400
N_REF       = 200          # reference run to build converged_mu
WINDOW_SIZE = 50
TAU         = 0.1
ETA         = 0.05
ETA_NEG     = 1.0
TTL_FULL    = 400
TTL_SHORT   = 100
TTL_STALE   = 1            # expires after 1 decision (minimum valid TTL)

CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}

ROC_THRESHOLDS = [0.8, 0.9, 1.0, 1.1, 1.2]

# Window keys in centroid_snapshots (decisions 0,50,100,...,350)
WINDOW_KEYS = [i * WINDOW_SIZE for i in range(8)]

RESULTS_PATH = Path("experiments/synthesis/expOP3_residual/results.json")

N_FACTORS = 6
C_DIM = len(CATEGORIES)
A_DIM = len(ACTIONS)
AC_IDX  = ACTIONS.index("auto_close")
ESC_IDX = ACTIONS.index("escalate_incident")

conditions = ["A", "B", "C", "P-50", "B-short", "D-stale"]


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


# ---------------------------------------------------------------------------
# Sigma builders
# ---------------------------------------------------------------------------

def build_correct_sigma() -> np.ndarray:
    s = np.zeros((C_DIM, A_DIM), dtype=np.float64)
    s[:, AC_IDX]  = +SIGMA_VALUE
    s[:, ESC_IDX] = -SIGMA_VALUE
    return s


def build_harmful_sigma() -> np.ndarray:
    return -build_correct_sigma()


def build_partial_sigma_50() -> np.ndarray:
    """50% correct: half of the non-zero cells are flipped."""
    rng = np.random.default_rng(seed=99)
    s = build_correct_sigma()
    nonzero = [(c, AC_IDX) for c in range(C_DIM)] + [(c, ESC_IDX) for c in range(C_DIM)]
    n_flip = int(len(nonzero) * 0.50)
    for idx in rng.choice(len(nonzero), size=n_flip, replace=False):
        r, col = nonzero[idx]
        s[r, col] = -s[r, col]
    return s


sigma_correct  = build_correct_sigma()
sigma_harmful  = build_harmful_sigma()
sigma_partial50 = build_partial_sigma_50()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frob(M: np.ndarray) -> float:
    return float(np.sqrt(np.sum(M ** 2)))


def run_pre_shift(seed: int, profiles: np.ndarray) -> np.ndarray:
    """Run N_PRE pre-shift decisions. Returns post-training mu."""
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen    = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_PRE)
    for alert in alerts:
        result     = scorer.score(alert.factors, alert.category_index, synthesis=None)
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
    lambda_override: float = LAMBDA_S,
) -> HarnessResult:
    scorer = ProfileScorer(starting_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=len(alerts),
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


def compute_r_norm_trajectory(
    mu_tilde: np.ndarray,
    pre_shift_mu: np.ndarray,
    centroid_snapshots: dict,
    window_keys: list,
    converged_mu: np.ndarray,
) -> list:
    """
    Compute normalized R_norm = ||mu_tilde - mu(t)||_F / ||mu_tilde - pre_shift_mu||_F
    at each window key.

    For P-50 where mu_tilde = pre_shift_mu (initial_norm ≈ 0):
    normalizes by ||converged_mu - pre_shift_mu||_F instead.
    """
    initial_norm = frob(mu_tilde - pre_shift_mu)
    if initial_norm < 1e-10:
        # P-50 case: declared no change; normalize by expected shift scale
        scale = frob(converged_mu - pre_shift_mu)
        if scale < 1e-10:
            return [0.0] * len(window_keys)
        return [
            frob(mu_tilde - centroid_snapshots.get(k, pre_shift_mu)) / scale
            for k in window_keys
        ]
    return [
        frob(mu_tilde - centroid_snapshots.get(k, pre_shift_mu)) / initial_norm
        for k in window_keys
    ]


def compute_per_ca_norms(
    mu_tilde: np.ndarray, mu_snapshot: np.ndarray
) -> np.ndarray:
    """Per-(category, action) L2 norm of residual. Shape (C, A)."""
    R = mu_tilde - mu_snapshot
    return np.sqrt(np.sum(R ** 2, axis=-1))   # (C, A)


def acc_curve_8(result: HarnessResult) -> list:
    """Subsample rolling accuracy to 8 values at window boundaries."""
    full = list(result.auac_result.accuracy_curve)
    n    = len(full)
    return [full[min(i * WINDOW_SIZE, n - 1)] for i in range(8)]


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
    print("EXP-OP3: RESIDUAL TRACKER EARLY-WARNING DIAGNOSTIC")
    print("=" * 65)
    print(f"Seeds: {len(SEEDS)}, lambda={LAMBDA_S}, "
          f"N_pre={N_PRE}, N_post={N_POST}, N_ref={N_REF}")
    print(f"Conditions: {conditions}")
    print(f"ROC thresholds: {ROC_THRESHOLDS}")
    print()

    results: dict = {cond: [] for cond in conditions}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"  seed {seed} ({seed_idx + 1}/{len(SEEDS)})...", flush=True)

        gen_pre     = CategoryAlertGenerator(seed=seed)
        gt_profiles = _build_profiles_tensor(gen_pre)
        pre_mu      = run_pre_shift(seed, gt_profiles)

        post_gen    = CategoryAlertGenerator(seed=seed + 10000)
        post_alerts = post_gen.generate_campaign(N_POST, CAMPAIGN)

        # ── Build converged_mu: oracle-only reference run on first N_REF alerts ──
        reg_ref     = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
        res_ref     = run_post_shift(pre_mu.copy(), post_alerts[:N_REF], reg_ref)
        converged_mu = res_ref.final_mu.copy()

        # ── mu_tilde construction ──────────────────────────────────────────────
        mu_tilde_B   = converged_mu.copy()
        mu_tilde_C   = 2.0 * pre_mu - converged_mu        # opposite direction
        mu_tilde_P50 = pre_mu.copy()                       # declares no shift

        # ── Condition specs ────────────────────────────────────────────────────
        cond_cfg = {
            "A": (None, None),
            "B": (OperatorSpec(f"B_{seed}", "active_campaign", 0,
                               sigma_correct, LAMBDA_S, TTL_FULL),
                  mu_tilde_B),
            "C": (OperatorSpec(f"C_{seed}", "active_campaign", 0,
                               sigma_harmful, LAMBDA_S, TTL_FULL),
                  mu_tilde_C),
            "P-50": (OperatorSpec(f"P50_{seed}", "active_campaign", 0,
                                  sigma_partial50, LAMBDA_S, TTL_FULL),
                     mu_tilde_P50),
            "B-short": (OperatorSpec(f"Bsh_{seed}", "active_campaign", 0,
                                     sigma_correct, LAMBDA_S, TTL_SHORT),
                        mu_tilde_B),
            "D-stale": (OperatorSpec(f"Dst_{seed}", "active_campaign", 0,
                                     sigma_correct, LAMBDA_S, TTL_STALE),
                        mu_tilde_B),
        }

        for cond in conditions:
            spec, mu_tilde = cond_cfg[cond]
            reg = make_registry(spec, pre_mu)
            res = run_post_shift(pre_mu, post_alerts, reg)

            snaps = res.centroid_snapshots   # keys: 0, 50, 100, ..., 350, 399

            if cond == "A":
                # No declared intent — tracker undefined, set sentinel
                r_norm_traj    = [0.0] * 8
                per_ca_w1      = np.zeros((C_DIM, A_DIM))
                per_ca_w4      = np.zeros((C_DIM, A_DIM))
            else:
                r_norm_traj = compute_r_norm_trajectory(
                    mu_tilde, pre_mu, snaps, WINDOW_KEYS, converged_mu
                )
                mu_w1 = snaps.get(WINDOW_KEYS[1], pre_mu)   # W=1: key=50
                mu_w4 = snaps.get(WINDOW_KEYS[4], pre_mu)   # W=4: key=200
                per_ca_w1 = compute_per_ca_norms(mu_tilde, mu_w1)
                per_ca_w4 = compute_per_ca_norms(mu_tilde, mu_w4)

            # r_norm_w1 = W=1 value
            r_norm_w1 = float(r_norm_traj[1]) if len(r_norm_traj) > 1 else 0.0

            results[cond].append({
                "seed":               seed,
                "r_norm_trajectory":  [float(v) for v in r_norm_traj],
                "r_norm_w1":          float(r_norm_w1),
                "per_ca_norms_w1":    per_ca_w1.tolist(),
                "per_ca_norms_w4":    per_ca_w4.tolist(),
                "auac":               float(res.auac_result.auac),
                "accuracy_curve":     acc_curve_8(res),
            })

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------

    n_seeds = len(SEEDS)

    def get_field(cond: str, field: str) -> np.ndarray:
        return np.array([results[cond][s][field] for s in range(n_seeds)])

    # R_norm trajectories (mean per window)
    traj_means = {
        cond: get_field(cond, "r_norm_trajectory").mean(axis=0)
        for cond in conditions
    }

    # Distinguishability at W=1
    r_w1_B = get_field("B", "r_norm_w1")
    r_w1_C = get_field("C", "r_norm_w1")
    p_dist  = permutation_test(r_w1_C - r_w1_B, rng_seed=42)

    # Decay speed: fraction of B seeds with R_norm < 0.5 by W=4
    b_decay_frac = float(np.mean(get_field("B", "r_norm_trajectory")[:, 4] < 0.5))

    # ROC analysis
    roc_tprs, roc_fprs, roc_precs = [], [], []
    for thr in ROC_THRESHOLDS:
        tpr = float(np.mean(r_w1_C > thr))
        fpr = float(np.mean(r_w1_B > thr))
        prec = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0.0
        roc_tprs.append(tpr)
        roc_fprs.append(fpr)
        roc_precs.append(float(prec))

    # Recommended threshold: highest TPR with FPR <= 0.20
    rec_idx = None
    best_tpr = -1.0
    for i, (tpr, fpr) in enumerate(zip(roc_tprs, roc_fprs)):
        if fpr <= 0.20 and tpr > best_tpr:
            best_tpr = tpr
            rec_idx = i

    # Per-category R_norm at W=1 for B and C
    per_ca_w1_B = np.array([results["B"][s]["per_ca_norms_w1"] for s in range(n_seeds)])
    per_ca_w1_C = np.array([results["C"][s]["per_ca_norms_w1"] for s in range(n_seeds)])
    per_cat_B_w1 = per_ca_w1_B.mean(axis=(0, 2))   # (C,): mean over seeds and actions
    per_cat_C_w1 = per_ca_w1_C.mean(axis=(0, 2))

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXP-OP3: RESIDUAL TRACKER EARLY-WARNING DIAGNOSTIC")
    print("=" * 70)

    print("\n--- R_NORM TRAJECTORIES (mean across 20 seeds) ---")
    header = f"{'Cond':>8}  " + "  ".join(f"W{w}" for w in range(8))
    print(header)
    print("-" * 65)
    for cond in conditions:
        row = f"{cond:>8}  " + "  ".join(f"{v:.3f}" for v in traj_means[cond])
        note = ""
        if cond == "B"   and traj_means["B"][1] < 1.0:  note = "  [converging at W=1]"
        if cond == "C"   and traj_means["C"][1] >= 1.0: note = "  [flat/growing at W=1]"
        if cond == "P-50":                               note = "  [normalized by shift scale]"
        print(row + note)

    print("\n--- DISTINGUISHABILITY AT W=1 ---")
    print(f"  B (correct) R_norm(W=1): mean={r_w1_B.mean():.4f} +/- {r_w1_B.std():.4f}")
    print(f"  C (harmful) R_norm(W=1): mean={r_w1_C.mean():.4f} +/- {r_w1_C.std():.4f}")
    dist_label = "[PASS]" if p_dist < 0.05 else "[FAIL]"
    print(f"  Distinguishability (C > B at W=1): p={p_dist:.4f}  {dist_label}")

    print("\n--- DECAY SPEED (B condition) ---")
    gate_label = "[PASS]" if b_decay_frac >= 0.80 else "[FAIL]"
    print(f"  B seeds with R_norm < 0.5 by W=4: {b_decay_frac:.1%}  "
          f"(gate: >=80%)  {gate_label}")

    print("\n--- EARLY-WARNING ROC AT W=1 ---")
    print(f"  {'Threshold':>12}  {'TPR (C flagged)':>17}  "
          f"{'FPR (B false-flagged)':>22}  {'Precision':>12}")
    print("  " + "-" * 68)
    for i, (thr, tpr, fpr, prec) in enumerate(
        zip(ROC_THRESHOLDS, roc_tprs, roc_fprs, roc_precs)
    ):
        marker = " <-- RECOMMENDED" if i == rec_idx else ""
        print(f"  {thr:>12.1f}  {tpr:>17.1%}  {fpr:>22.1%}  "
              f"{prec:>12.1%}{marker}")

    print("\n--- RECOMMENDED DETECTION THRESHOLD ---")
    if rec_idx is not None:
        print(f"  tau={ROC_THRESHOLDS[rec_idx]:.1f}: "
              f"TPR={roc_tprs[rec_idx]:.1%}, FPR={roc_fprs[rec_idx]:.1%}, "
              f"Precision={roc_precs[rec_idx]:.1%}")
    else:
        print("  No viable threshold (none achieves >=70% TPR with FPR <=20%)")

    print("\n--- PER-CATEGORY R_NORM AT W=1 (B vs C) ---")
    print(f"  {'Category':>22}  {'B (correct)':>14}  {'C (harmful)':>14}")
    print("  " + "-" * 55)
    for c_idx, cat in enumerate(CATEGORIES):
        print(f"  {cat:>22}  {per_cat_B_w1[c_idx]:>14.4f}  "
              f"{per_cat_C_w1[c_idx]:>14.4f}")

    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    results_data = {
        "per_seed_results": {
            cond: results[cond]
            for cond in conditions
        },
        "roc_analysis": {
            "thresholds":                ROC_THRESHOLDS,
            "tpr":                       roc_tprs,
            "fpr":                       roc_fprs,
            "precision":                 roc_precs,
            "recommended_threshold_idx": rec_idx,
        },
        "distinguishability": {
            "B_mean_w1":  float(r_w1_B.mean()),
            "C_mean_w1":  float(r_w1_C.mean()),
            "p_value":    float(p_dist),
        },
        "decay_speed": {
            "B_frac_below_05_by_w4": float(b_decay_frac),
        },
        "categories": list(CATEGORIES),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fout:
        json.dump(results_data, fout, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ---------------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------------

    from experiments.synthesis.expOP3_residual.charts import generate_charts
    generate_charts(results_data)
