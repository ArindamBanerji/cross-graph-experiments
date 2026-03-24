"""
V-CGA-FROZEN v3 — Powered validation at N=257 (90% power for d=0.241 at p<0.01)
==================================================================================
Power analysis (v1+v2) confirmed d=0.241 is a real, consistent effect but N=50
had only 17.6% power at p<0.01.  Required N for 90% power: 257.

This is the final, pre-registered run.  N=257 committed before running.
All gates, sigma schedules, and simulation logic are identical to v2.

Change vs v2: N_SEEDS 50 → 257.

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_v3.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile, check_conservation
from gae import CovarianceEstimator

# ── Canonical parameters ──────────────────────────────────────────────────────
# N_SEEDS = 257: pre-committed for 90% power at d=0.241, p<0.01 (from power analysis)
N_SEEDS         = 257
N_BOOTSTRAP     = 1200
N_FROZEN_DAYS   = 90
APD             = 10          # alerts per day during frozen window
N_POST_UNFREEZE = 500
THETA_MIN       = 0.467       # canonical — always this value
TAU             = 0.1
ETA_CONFIRM     = 0.05
ETA_OVERRIDE    = 0.01
Q_BAR           = 0.75        # realistic analyst quality
ALPHA           = 0.80        # override rate — realistic SOC

# SOC domain
N_CATEGORIES = 6
N_ACTIONS    = 4
N_FACTORS    = 6

FACTOR_NAMES = [
    "travel_match",
    "asset_criticality",
    "threat_intel_enrichment",
    "time_anomaly",
    "pattern_history",
    "device_trust",
]
# Factor indices for enrichment-sensitive factors
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ENRICHMENT_FACTORS = ["threat_intel_enrichment", "pattern_history", "device_trust"]

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration", "insider_threat", "cloud_infrastructure",
]

# ── Noise model (identical sigma schedule to v1) ───────────────────────────────
# Fixed factors (enrichment-independent)
_SIGMA_FIXED = {
    "travel_match":      0.18,
    "asset_criticality": 0.06,
    "time_anomaly":      0.07,
}
# Base noise for enrichment-sensitive factors (used in condition B and day 0)
_SIGMA_BASE = {
    "threat_intel_enrichment": 0.18,
    "pattern_history":         0.20,
    "device_trust":            0.16,
}


def _enrichment_ramp(day: int) -> float:
    """Smooth ramp 0.0 (day 0) → 1.0 (day 90). CISA KEV growth pattern."""
    return float(day) / 90.0


def _entity_resolution_rate(day: int) -> float:
    """Entity resolution rate: 40% → 75% linearly over 90 days."""
    return 0.40 + (0.75 - 0.40) * (float(day) / 90.0)


def _history_days(day: int) -> float:
    """SIEM history window: 30 days at day 0, grows to 120 days by day 90."""
    return 30.0 + float(day)


def get_sigma_vector(day: int, condition: str) -> np.ndarray:
    """
    Return per-factor sigma vector for a given day and condition.

    Condition A (enrichment): sigma decreases for threat_intel, device_trust,
                               pattern_history over 90 days.
    Condition B (control):    sigma stays constant at Day 0 values throughout.

    sigma_threat_intel:    0.180 → 0.133 (condition A Day 90)
    sigma_pattern_history: 0.200 → 0.100 (condition A Day 90)
    sigma_device_trust:    0.160 → 0.112 (condition A Day 90)
    All others: fixed (same both conditions)
    """
    sigma = np.zeros(N_FACTORS)
    for fname, idx in IDX.items():
        if fname in _SIGMA_FIXED:
            sigma[idx] = _SIGMA_FIXED[fname]
        elif condition == "A":
            if fname == "threat_intel_enrichment":
                sigma[idx] = 0.18 * np.exp(-_enrichment_ramp(day) * 0.3)
            elif fname == "device_trust":
                sigma[idx] = 0.16 * (1.0 - 0.4 * _entity_resolution_rate(day))
            elif fname == "pattern_history":
                sigma[idx] = 0.20 * (_history_days(0) / _history_days(day)) ** 0.5
        else:
            sigma[idx] = _SIGMA_BASE[fname]
    return sigma


def sigma_vector_deterministic(day: int, condition: str) -> np.ndarray:
    """Alias for clarity — sigma is deterministic given (day, condition)."""
    return get_sigma_vector(day, condition)


# ── Ground truth utilities ─────────────────────────────────────────────────────

def build_ground_truth(rng: np.random.RandomState):
    """
    Build ground truth centroids and GT action distributions.
    mu_true[c, a]: true centroid for category c, action a — shape (C, A, d).
    Each category has a slightly different dominant action (not uniform).
    """
    mu_true = rng.uniform(0.15, 0.85, size=(N_CATEGORIES, N_ACTIONS, N_FACTORS))
    # GT distribution: one action dominant per category
    gt_dist = np.ones((N_CATEGORIES, N_ACTIONS)) * 0.1
    for c in range(N_CATEGORIES):
        dominant = c % N_ACTIONS
        gt_dist[c, dominant] = 0.7
    gt_dist = gt_dist / gt_dist.sum(axis=1, keepdims=True)
    cat_weights = np.ones(N_CATEGORIES) / N_CATEGORIES
    return mu_true, gt_dist, cat_weights


def sample_alert(
    rng: np.random.RandomState,
    mu_true: np.ndarray,
    gt_dist: np.ndarray,
    cat_weights: np.ndarray,
    sigma_vec: np.ndarray,
):
    """Sample one alert. Returns (category_idx, gt_action_idx, factor_vector)."""
    c   = int(rng.choice(N_CATEGORIES, p=cat_weights))
    a   = int(rng.choice(N_ACTIONS, p=gt_dist[c]))
    f   = np.clip(mu_true[c, a] + rng.randn(N_FACTORS) * sigma_vec, 0.0, 1.0)
    return c, a, f


def analyst_feedback(
    rng: np.random.RandomState,
    pred_a: int,
    gt_a: int,
    q_bar: float = Q_BAR,
    alpha: float = ALPHA,
):
    """
    Simulate analyst decision.
    Returns (final_action, is_override).
    With prob (1-alpha): confirm — final = pred_a.
    With prob alpha:     override — correct with prob q_bar, else random.
    """
    if rng.rand() < alpha:
        if rng.rand() < q_bar:
            return gt_a, True
        else:
            return int(rng.choice(N_ACTIONS)), True
    else:
        return pred_a, False


# ── IKS (Information Knowledge Score) ─────────────────────────────────────────

def compute_iks(mu_current: np.ndarray, mu_true: np.ndarray, mu_unfreeze: np.ndarray) -> float:
    """
    IKS = 1 - ||mu_current - mu_true||_F / ||mu_unfreeze - mu_true||_F

    Measures fraction of centroid error recovered since unfreeze.
      0.0 = no recovery (mu still at unfreeze state)
      1.0 = full convergence to mu_true
    Negative values indicate divergence (penalized correctly).
    """
    err_now   = np.linalg.norm(mu_current - mu_true)
    err_start = np.linalg.norm(mu_unfreeze - mu_true)
    if err_start < 1e-9:
        return 1.0
    return float(1.0 - err_now / err_start)


# ── Per-seed simulation ────────────────────────────────────────────────────────

def run_one_seed(seed: int) -> dict:
    """
    Run both conditions A and B for a single seed.

    Seed isolation:
      gt_rng (seed)              — ground truth, shared mu_true, mu_init
      boot_rng (seed + 10000)    — bootstrap decisions (shared between A and B)
      rng_a   (seed + 20000)     — condition A frozen + post-unfreeze
      rng_b   (seed + 30000)     — condition B frozen + post-unfreeze

    v2 change (condition A only):
      A CovarianceEstimator is maintained for both conditions during the frozen
      window.  After each 10-day batch in condition A, kernel_weight_refresh()
      is called to update DiagonalKernel weights from accumulated covariance data.
      Condition B gets no kernel_weight_refresh() calls (clean control).
    """
    gt_rng   = np.random.RandomState(seed)
    boot_rng = np.random.RandomState(seed + 10000)
    rng_a    = np.random.RandomState(seed + 20000)
    rng_b    = np.random.RandomState(seed + 30000)

    # Ground truth — fixed for both conditions
    mu_true, gt_dist, cat_weights = build_ground_truth(gt_rng)

    # Initial centroids — shared starting point for bootstrap
    mu_init = gt_rng.uniform(0.15, 0.85, size=(N_CATEGORIES, N_ACTIONS, N_FACTORS))

    # CalibrationProfile: temperature=tau, learning_rate=eta_confirm
    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)

    # ── Bootstrap (shared: same for A and B) ──────────────────────────────────
    boot_scorer = ProfileScorer(
        mu_init.copy(), actions=ACTIONS, categories=CATEGORIES,
        profile=profile, eta_override=ETA_OVERRIDE,
    )
    sigma_boot = get_sigma_vector(0, "B")  # Day 0 = static noise for both
    for _ in range(N_BOOTSTRAP):
        c, gt_a, f = sample_alert(boot_rng, mu_true, gt_dist, cat_weights, sigma_boot)
        res = boot_scorer.score(f, c)
        final_a, _ = analyst_feedback(boot_rng, res.action_index, gt_a)
        correct = (final_a == gt_a)
        boot_scorer.update(f, c, final_a, correct, gt_action_index=gt_a)

    mu_bootstrapped = boot_scorer.mu.copy()

    # ── Branch into conditions A and B ────────────────────────────────────────
    out = {}
    rngs = {"A": rng_a, "B": rng_b}

    for condition in ("A", "B"):
        rng = rngs[condition]

        scorer = ProfileScorer(
            mu_bootstrapped.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        # ── Frozen phase ──────────────────────────────────────────────────────
        scorer.freeze()

        # CovarianceEstimator — tracks factor observations during frozen window.
        # Updated in BOTH conditions so covariance accumulation is symmetric.
        # kernel_weight_refresh() called in condition A only (after each 10-day batch).
        cov_est = CovarianceEstimator(N_FACTORS)

        # Factor observations for sigma estimation at days 1-5 and 86-90
        obs_early = []  # days 1-5
        obs_late  = []  # days 86-90

        for day in range(1, N_FROZEN_DAYS + 1):
            sigma_t = get_sigma_vector(day, condition)
            for _ in range(APD):
                c, gt_a, f = sample_alert(rng, mu_true, gt_dist, cat_weights, sigma_t)
                scorer.score(f, c)
                # Update CovarianceEstimator in BOTH conditions
                cov_est.update(f)
                # Factor observations — measure against nearest true centroid for that (c,gt_a)
                # This gives the NOISE component: deviation from true centroid
                deviation = f - mu_true[c, gt_a]
                if day <= 5:
                    obs_early.append(deviation)
                if day >= 86:
                    obs_late.append(deviation)

            # After each 10-day enrichment step: refresh kernel weights in condition A only.
            # Safe during freeze (learning_enabled=False, mu is never touched).
            if condition == "A" and day % 10 == 0:
                scorer.kernel_weight_refresh(cov_est)

        obs_early = np.array(obs_early)   # (N_early, N_FACTORS)
        obs_late  = np.array(obs_late)    # (N_late,  N_FACTORS)

        # Observed sigma per factor = std of (factor_obs - true_centroid)
        # i.e., the noise amplitude in each factor dimension
        sigma_obs_early = obs_early.std(axis=0)   # shape (N_FACTORS,)
        sigma_obs_late  = obs_late.std(axis=0)

        # ── Post-unfreeze phase ────────────────────────────────────────────────
        scorer.unfreeze()
        mu_at_unfreeze = scorer.mu.copy()

        sigma_post = get_sigma_vector(N_FROZEN_DAYS, condition)
        post_accs  = []
        iks_checkpoints = {}

        for dec in range(1, N_POST_UNFREEZE + 1):
            c, gt_a, f = sample_alert(rng, mu_true, gt_dist, cat_weights, sigma_post)
            res = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(rng, pred_a, gt_a)
            correct = (final_a == gt_a)
            scorer.update(f, c, final_a, correct, gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

            if dec in (100, 200, 300, 400, 500):
                iks_checkpoints[dec] = compute_iks(scorer.mu, mu_true, mu_at_unfreeze)

        out[condition] = {
            "sigma_obs_early": sigma_obs_early.tolist(),
            "sigma_obs_late":  sigma_obs_late.tolist(),
            "post_accs":       post_accs,
            "iks_checkpoints": {str(k): v for k, v in iks_checkpoints.items()},
        }

    return out


# ── N_half computation ─────────────────────────────────────────────────────────

def compute_n_half(post_accs: list, window: int = 50, gap_pp: float = 2.0) -> int:
    """
    Return first decision index where rolling accuracy (window=50) reaches
    within gap_pp percentage points of the final accuracy.

    Final accuracy = mean of last 100 decisions.
    Returns N_POST_UNFREEZE if threshold never reached (conservative).
    """
    arr = np.array(post_accs)
    final_acc = arr[-100:].mean() * 100.0
    threshold = (final_acc - gap_pp) / 100.0

    # Rolling mean
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    # roll[i] covers decisions [i, i+window), so decision index = i + window
    above = np.where(roll >= threshold)[0]
    if len(above) == 0:
        return N_POST_UNFREEZE
    return int(above[0]) + window


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_metrics(all_seed_results: list) -> dict:
    """
    Aggregate M1, M2, M3 across 50 seeds.
    Returns full metric dict including pass/fail verdicts.
    """
    n_seeds = len(all_seed_results)

    # M1: per-factor sigma reduction
    # For each seed: sigma_obs_late[factor] condition A vs B
    sigma_late_A = np.array([r["A"]["sigma_obs_late"] for r in all_seed_results])  # (S, F)
    sigma_late_B = np.array([r["B"]["sigma_obs_late"] for r in all_seed_results])

    sigma_early_A = np.array([r["A"]["sigma_obs_early"] for r in all_seed_results])
    sigma_early_B = np.array([r["B"]["sigma_obs_early"] for r in all_seed_results])

    # sigma reduction per factor: (late_B - late_A) / late_B
    # Positive = A has lower sigma than B
    sigma_reduction_by_factor = {}
    enrichment_passes = 0
    sigma_t_stats = {}
    for fname in ENRICHMENT_FACTORS:
        idx = IDX[fname]
        late_a = sigma_late_A[:, idx]
        late_b = sigma_late_B[:, idx]
        reduction_pct = float(((late_b - late_a) / np.maximum(late_b, 1e-9)).mean() * 100)
        t_stat, p_val = scipy_stats.ttest_rel(late_b, late_a)  # B > A if enrichment works
        effect_d = float((late_b - late_a).mean() / (late_b - late_a).std() + 1e-9)
        passes = (reduction_pct > 10.0 and p_val < 0.01)
        if passes:
            enrichment_passes += 1
        sigma_reduction_by_factor[fname] = {
            "mean_sigma_A":    float(late_a.mean()),
            "mean_sigma_B":    float(late_b.mean()),
            "reduction_pct":   round(reduction_pct, 2),
            "p_value":         round(float(p_val), 6),
            "effect_size":     round(effect_d, 4),
            "individual_pass": passes,
        }
        sigma_t_stats[fname] = (t_stat, p_val)

    # Fixed factors — should show NO difference (sanity check)
    for fname in ["travel_match", "asset_criticality", "time_anomaly"]:
        idx = IDX[fname]
        late_a = sigma_late_A[:, idx]
        late_b = sigma_late_B[:, idx]
        reduction_pct = float(((late_b - late_a) / np.maximum(late_b, 1e-9)).mean() * 100)
        sigma_reduction_by_factor[fname] = {
            "mean_sigma_A":    float(late_a.mean()),
            "mean_sigma_B":    float(late_b.mean()),
            "reduction_pct":   round(reduction_pct, 2),
            "note":            "fixed factor — reduction expected near 0",
        }

    # M1 overall: pooled t-test over all 3 enrichment-sensitive factors
    all_late_a_enriched = np.concatenate([sigma_late_A[:, IDX[f]] for f in ENRICHMENT_FACTORS])
    all_late_b_enriched = np.concatenate([sigma_late_B[:, IDX[f]] for f in ENRICHMENT_FACTORS])
    t_m1, p_m1 = scipy_stats.ttest_rel(all_late_b_enriched, all_late_a_enriched)
    m1_pass = (enrichment_passes >= 2 and p_m1 < 0.01)

    # M2: N_half convergence speed
    n_half_A = np.array([compute_n_half(r["A"]["post_accs"]) for r in all_seed_results])
    n_half_B = np.array([compute_n_half(r["B"]["post_accs"]) for r in all_seed_results])

    diff_BA = n_half_B - n_half_A  # positive = B takes longer = A converges faster
    t_m2, p_m2 = scipy_stats.ttest_rel(n_half_B, n_half_A)
    d_m2 = float(diff_BA.mean() / (diff_BA.std() + 1e-9))
    m2_pass = (float(n_half_A.mean()) < float(n_half_B.mean()) and p_m2 < 0.01 and abs(d_m2) > 0.3)

    # CI for N_half
    ci_a = scipy_stats.t.interval(0.95, n_seeds - 1,
                                   loc=n_half_A.mean(),
                                   scale=scipy_stats.sem(n_half_A))
    ci_b = scipy_stats.t.interval(0.95, n_seeds - 1,
                                   loc=n_half_B.mean(),
                                   scale=scipy_stats.sem(n_half_B))

    # M3: IKS at decision 300
    iks_300_A = np.array([r["A"]["iks_checkpoints"]["300"] for r in all_seed_results])
    iks_300_B = np.array([r["B"]["iks_checkpoints"]["300"] for r in all_seed_results])

    t_m3, p_m3 = scipy_stats.ttest_rel(iks_300_A, iks_300_B)
    m3_pass = (float(iks_300_A.mean()) > float(iks_300_B.mean()) and p_m3 < 0.01)

    # Full IKS trajectories
    iks_traj = {}
    for ck in (100, 200, 300, 400, 500):
        arr_A = np.array([r["A"]["iks_checkpoints"][str(ck)] for r in all_seed_results])
        arr_B = np.array([r["B"]["iks_checkpoints"][str(ck)] for r in all_seed_results])
        iks_traj[ck] = {
            "A_mean": round(float(arr_A.mean()), 4),
            "B_mean": round(float(arr_B.mean()), 4),
            "delta":  round(float(arr_A.mean() - arr_B.mean()), 4),
        }

    # Final accuracy comparison
    final_acc_A = np.array([np.array(r["A"]["post_accs"])[-100:].mean() for r in all_seed_results])
    final_acc_B = np.array([np.array(r["B"]["post_accs"])[-100:].mean() for r in all_seed_results])
    t_fa, p_fa = scipy_stats.ttest_rel(final_acc_A, final_acc_B)

    overall_pass = m1_pass and m2_pass and m3_pass

    return {
        "overall_pass": overall_pass,
        "sigma_reduction": {
            "by_factor":          sigma_reduction_by_factor,
            "enrichment_factors_passing": enrichment_passes,
            "pooled_p_value":     round(float(p_m1), 6),
            "pooled_t_stat":      round(float(t_m1), 4),
            "pass":               m1_pass,
        },
        "convergence_speed": {
            "n_half_A_mean":   round(float(n_half_A.mean()), 1),
            "n_half_A_ci95":   [round(ci_a[0], 1), round(ci_a[1], 1)],
            "n_half_B_mean":   round(float(n_half_B.mean()), 1),
            "n_half_B_ci95":   [round(ci_b[0], 1), round(ci_b[1], 1)],
            "reduction_pct":   round(float((n_half_B.mean() - n_half_A.mean()) /
                                           (n_half_B.mean() + 1e-9) * 100), 1),
            "p_value":         round(float(p_m2), 6),
            "t_stat":          round(float(t_m2), 4),
            "effect_size_d":   round(d_m2, 4),
            "pass":            m2_pass,
        },
        "iks_trajectory": {
            "checkpoints":    {str(k): v for k, v in iks_traj.items()},
            "at_300_A":       round(float(iks_300_A.mean()), 4),
            "at_300_B":       round(float(iks_300_B.mean()), 4),
            "at_300_delta":   round(float(iks_300_A.mean() - iks_300_B.mean()), 4),
            "p_value":        round(float(p_m3), 6),
            "t_stat":         round(float(t_m3), 4),
            "pass":           m3_pass,
        },
        "final_accuracy": {
            "A_mean": round(float(final_acc_A.mean()), 4),
            "B_mean": round(float(final_acc_B.mean()), 4),
            "p_value": round(float(p_fa), 6),
        },
    }


# ── Sigma schedule sanity print ────────────────────────────────────────────────

def print_sigma_schedule():
    print("\nSigma schedule (deterministic model):")
    print(f"  {'Factor':<28} {'Day 0 (A=B)':>12} {'Day 45 (A)':>12} {'Day 90 (A)':>12} {'Day 90 (B)':>12}")
    for fname in FACTOR_NAMES:
        idx = IDX[fname]
        s0 = get_sigma_vector(0, "A")[idx]
        s45 = get_sigma_vector(45, "A")[idx]
        s90a = get_sigma_vector(90, "A")[idx]
        s90b = get_sigma_vector(90, "B")[idx]
        tag = " *" if fname in ENRICHMENT_FACTORS else ""
        print(f"  {fname:<28}{tag} {s0:>12.4f} {s45:>12.4f} {s90a:>12.4f} {s90b:>12.4f}")
    print("  * = enrichment-sensitive factor")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("V-CGA-FROZEN v3 (N=257, 90% power, KernelWeightRefresh — GAE 0.7.1)")
    print("=" * 65)
    print_sigma_schedule()

    print(f"\nRunning {N_SEEDS} seeds × (bootstrap={N_BOOTSTRAP}, frozen={N_FROZEN_DAYS}d×{APD}apd, post={N_POST_UNFREEZE})")
    print(f"THETA_MIN={THETA_MIN} (canonical), TAU={TAU}, Q_BAR={Q_BAR}, ALPHA={ALPHA}")
    print(f"Power: 90% target at d=0.241, p<0.01 → N=257 (pre-committed)")
    print(f"Fix: kernel_weight_refresh() called every 10 days in condition A (9 calls/seed)")

    t0 = time.time()
    all_seed_results = []

    for seed in range(N_SEEDS):
        result = run_one_seed(seed)
        all_seed_results.append(result)
        if (seed + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (seed + 1) / elapsed
            eta = (N_SEEDS - seed - 1) / rate
            print(f"  Seed {seed+1:2d}/{N_SEEDS} done  [{elapsed:.1f}s elapsed, ETA {eta:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(all_seed_results)

    # ── Build results dict ─────────────────────────────────────────────────────
    verdict = "PASS" if metrics["overall_pass"] else "FAIL"

    design_implications = (
        "All three metrics passed. Graph enrichment accelerates post-unfreeze "
        "convergence. Claim validated: graph compounds while centroids wait. "
        "KernelWeightRefresh fix confirmed as root cause of v1 M2 failure."
        if verdict == "PASS" else
        "DESIGN GAP: GraphAttentionBridge required. "
        f"Failed metrics: "
        f"{'M1 ' if not metrics['sigma_reduction']['pass'] else ''}"
        f"{'M2 ' if not metrics['convergence_speed']['pass'] else ''}"
        f"{'M3 ' if not metrics['iks_trajectory']['pass'] else ''}. "
        "GTM materials referencing 'graph compounds while centroids wait' must "
        "be updated to forward-looking only. GraphAttentionBridge goes to Phase 3 queue."
    )

    m2 = metrics["convergence_speed"]
    results = {
        "experiment":   "V-CGA-FROZEN",
        "version":      "v3_powered_n257",
        "fix_applied":  "GAE 0.7.1 kernel_weight_refresh() called after each 10-day enrichment step in condition A",
        "date":         "2026-03-23",
        "n_seeds":      N_SEEDS,
        "verdict":      verdict,
        "runtime_s":    round(elapsed_total, 1),
        "parameters": {
            "n_bootstrap":      N_BOOTSTRAP,
            "n_frozen_days":    N_FROZEN_DAYS,
            "apd":              APD,
            "n_post_unfreeze":  N_POST_UNFREEZE,
            "theta_min":        THETA_MIN,
            "tau":              TAU,
            "eta_confirm":      ETA_CONFIRM,
            "eta_override":     ETA_OVERRIDE,
            "q_bar":            Q_BAR,
            "alpha":            ALPHA,
        },
        "sigma_schedule_day90": {
            fname: {
                "A": round(float(get_sigma_vector(90, "A")[IDX[fname]]), 4),
                "B": round(float(get_sigma_vector(90, "B")[IDX[fname]]), 4),
            }
            for fname in FACTOR_NAMES
        },
        "metrics":              metrics,
        "design_implications":  design_implications,
        "power_analysis": {
            "effect_size_d": 0.241,
            "alpha":         0.01,
            "power_target":  0.90,
            "n_required":    257,
            "n_actual":      257,
        },
        "comparison_to_v1v2": {
            "m2_p_value_v1v2":    0.097324,
            "m2_p_value_v3":      m2["p_value"],
            "m2_effect_size_v1v2": 0.2415,
            "m2_effect_size_v3":  m2["effect_size_d"],
            "n_half_a_v1v2":      75.1,
            "n_half_a_v3":        m2["n_half_A_mean"],
            "n_half_b_v1v2":      93.3,
            "n_half_b_v3":        m2["n_half_B_mean"],
        },
    }

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v3.json"
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    print(f"\nResults saved to {results_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    m1 = metrics["sigma_reduction"]
    m2 = metrics["convergence_speed"]
    m3 = metrics["iks_trajectory"]
    fa = metrics["final_accuracy"]

    print("\n" + "=" * 65)
    print(f"V-CGA-FROZEN v3 (N=257, 90% power): {verdict}")
    print("=" * 65)
    print(f"M1 (sigma reduction):      {'PASS' if m1['pass'] else 'FAIL'}")
    print(f"  Enrichment factors passing: {m1['enrichment_factors_passing']}/3")
    for fname in ENRICHMENT_FACTORS:
        d = m1["by_factor"][fname]
        print(f"    {fname:<30} reduction={d['reduction_pct']:+.1f}%  p={d['p_value']:.4f}  "
              f"{'PASS' if d['individual_pass'] else 'fail'}")
    print(f"  Pooled p={m1['pooled_p_value']:.6f}")
    print()
    print(f"M2 (N_half):               {'PASS' if m2['pass'] else 'FAIL'}")
    print(f"  Condition A: {m2['n_half_A_mean']:.1f} decisions  "
          f"95% CI [{m2['n_half_A_ci95'][0]:.1f}, {m2['n_half_A_ci95'][1]:.1f}]")
    print(f"  Condition B: {m2['n_half_B_mean']:.1f} decisions  "
          f"95% CI [{m2['n_half_B_ci95'][0]:.1f}, {m2['n_half_B_ci95'][1]:.1f}]")
    print(f"  Reduction:  {m2['reduction_pct']:.1f}%  p={m2['p_value']:.6f}  d={m2['effect_size_d']:.3f}")
    print(f"  v1/v2 → v3: N_half_A {results['comparison_to_v1v2']['n_half_a_v1v2']} → {m2['n_half_A_mean']:.1f}  "
          f"N_half_B {results['comparison_to_v1v2']['n_half_b_v1v2']} → {m2['n_half_B_mean']:.1f}")
    print(f"  v1/v2 → v3: p {results['comparison_to_v1v2']['m2_p_value_v1v2']:.6f} → {m2['p_value']:.6f}  "
          f"d {results['comparison_to_v1v2']['m2_effect_size_v1v2']:.4f} → {m2['effect_size_d']:.4f}")
    print()
    print(f"M3 (IKS trajectory):       {'PASS' if m3['pass'] else 'FAIL'}")
    print(f"  IKS at 300 decisions: A={m3['at_300_A']:.4f}  B={m3['at_300_B']:.4f}  "
          f"delta={m3['at_300_delta']:+.4f}  p={m3['p_value']:.6f}")
    print(f"  Full trajectory:")
    for ck in (100, 200, 300, 400, 500):
        ck_data = m3["checkpoints"][str(ck)]
        print(f"    dec {ck:3d}: A={ck_data['A_mean']:.4f}  B={ck_data['B_mean']:.4f}  "
              f"delta={ck_data['delta']:+.4f}")
    print()
    print(f"Final accuracy (post-500):   A={fa['A_mean']:.4f}  B={fa['B_mean']:.4f}  "
          f"p={fa['p_value']:.4f}")
    print()

    fix_str = "closed" if verdict == "PASS" else (
        "partial" if m2["pass"] else "no effect"
    )
    print(f"Fix assessment: {fix_str}")
    print(f"Claim status: ", end="")
    if verdict == "PASS":
        print("VALIDATED — 'graph compounds while centroids wait'")
    else:
        print("STILL GAP — describe precisely:")
        if not m1["pass"]:
            print(f"  M1 FAIL: only {m1['enrichment_factors_passing']}/3 enrichment factors pass")
        if not m2["pass"]:
            print(f"  M2 FAIL: N_half_A={m2['n_half_A_mean']:.1f} vs B={m2['n_half_B_mean']:.1f}, "
                  f"p={m2['p_value']:.4f}, d={m2['effect_size_d']:.3f} (need p<0.01, d>0.3)")
        if not m3["pass"]:
            print(f"  M3 FAIL: IKS_A={m3['at_300_A']:.4f} vs IKS_B={m3['at_300_B']:.4f}, "
                  f"p={m3['p_value']:.4f} (need A>B and p<0.01)")
    print("=" * 65)

    return results


if __name__ == "__main__":
    main()
