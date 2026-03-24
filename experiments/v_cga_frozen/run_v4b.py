"""
V-CGA-FROZEN v4 — Empirical Bayes bootstrap vs cold-start
==========================================================
Tests a DIFFERENT hypothesis from v1-v3 (which tested sigma-reduction ->
convergence rate; result: null, d=-0.010).

v4 hypothesis: compute_enriched_bootstrap_prior() places mu_0 22% closer to
the operational optimum mu*. Because starting distance ||e_0|| = ||mu_0 - mu*||
is smaller, fewer analyst decisions are needed to reach the calibration
neighborhood.  This is about STARTING DISTANCE, not convergence RATE.

TWO CONDITIONS:
  C0 — Cold start: historical decisions generated with un-enriched sigma
       (sigma_profile x 1.5 on enrichment factors), standard (unweighted)
       bootstrap prior.  Learning runs with enriched sigma.
  T2 — Empirical Bayes: historical decisions generated with enriched sigma,
       compute_enriched_bootstrap_prior() prior (inverse-variance weighted).
       Learning runs with same enriched sigma as C0.

Both conditions have IDENTICAL post-bootstrap learning environment.
The ONLY difference is the starting mu_0.

Re-run with corrected GAE 0.7.4 implementation.
Fix: mu += eta * W_normalized * (f - mu)  (kernel-weighted gradient)
Was: f_enriched = f * W_normalized  (scaled vector — wrong)

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_v4b.py
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
from gae.calibration import CalibrationProfile, compute_enriched_bootstrap_prior

# ── Committed parameters (do not change) ──────────────────────────────────────
N_SEEDS          = 100
N_BOOTSTRAP_HIST = 200    # historical decisions used to build mu_0
N_POST_BOOTSTRAP = 500    # learning decisions measured post-bootstrap
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80

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
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ENRICHMENT_FACTORS = ["threat_intel_enrichment", "pattern_history", "device_trust"]

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration", "insider_threat", "cloud_infrastructure",
]

# ── Sigma profiles ─────────────────────────────────────────────────────────────
# Enriched sigma (T2 bootstrap + both conditions' learning environment)
SIGMA_ENRICHED = {
    "threat_intel_enrichment": 0.13,
    "pattern_history":         0.10,
    "device_trust":            0.11,
    "travel_match":            0.18,
    "asset_criticality":       0.06,
    "time_anomaly":            0.07,
}
# Un-enriched sigma for C0 historical data: enrichment factors x1.5
_UNENRICH_MULT = 1.5
SIGMA_UNENRICHED = {
    name: (SIGMA_ENRICHED[name] * _UNENRICH_MULT if name in ENRICHMENT_FACTORS
           else SIGMA_ENRICHED[name])
    for name in FACTOR_NAMES
}


def _sigma_vec(sigma_dict: dict) -> np.ndarray:
    return np.array([sigma_dict[f] for f in FACTOR_NAMES])


SIGMA_VEC_ENRICHED   = _sigma_vec(SIGMA_ENRICHED)
SIGMA_VEC_UNENRICHED = _sigma_vec(SIGMA_UNENRICHED)


# ── Minimal domain_config — only .factor_names required by GAE ────────────────
class _DomainConfig:
    factor_names = FACTOR_NAMES


DOMAIN_CONFIG = _DomainConfig()


# ── Ground truth + operational optimum ────────────────────────────────────────

def build_ground_truth(rng: np.random.RandomState):
    """
    Build mu_true (ground truth centroids) and GT action distribution.
    mu_true is used as mu* (operational optimum): after enough perfectly-labeled
    decisions, ProfileScorer.mu converges to mu_true.
    """
    mu_true = rng.uniform(0.15, 0.85, size=(N_CATEGORIES, N_ACTIONS, N_FACTORS))
    gt_dist = np.ones((N_CATEGORIES, N_ACTIONS)) * 0.1
    for c in range(N_CATEGORIES):
        gt_dist[c, c % N_ACTIONS] = 0.7
    gt_dist = gt_dist / gt_dist.sum(axis=1, keepdims=True)
    cat_weights = np.ones(N_CATEGORIES) / N_CATEGORIES
    return mu_true, gt_dist, cat_weights


# ── Alert generation ───────────────────────────────────────────────────────────

def sample_alert(rng, mu_true, gt_dist, cat_weights, sigma_vec):
    c = int(rng.choice(N_CATEGORIES, p=cat_weights))
    a = int(rng.choice(N_ACTIONS, p=gt_dist[c]))
    f = np.clip(mu_true[c, a] + rng.randn(N_FACTORS) * sigma_vec, 0.0, 1.0)
    return c, a, f


def analyst_feedback(rng, pred_a, gt_a):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(N_ACTIONS))), True
    return pred_a, False


# ── Standard (unweighted) bootstrap for C0 ────────────────────────────────────

def standard_bootstrap(historical_decisions) -> np.ndarray:
    """
    Simple ETA-update bootstrap with uniform (unweighted) factor vectors.
    Same iteration structure as compute_enriched_bootstrap_prior but with
    W_normalized = ones — no inverse-variance weighting.
    This is the C0 baseline: same number of decisions, same algorithm, no enrichment.
    """
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in historical_decisions:
        f = np.asarray(f, dtype=float)
        mu[c, a, :] += ETA_CONFIRM * (f - mu[c, a, :])
        mu[c, a, :] = np.clip(mu[c, a, :], 0.0, 1.0)
    return mu


# ── N_half ────────────────────────────────────────────────────────────────────

def compute_n_half(post_accs: list, window: int = 50, gap_pp: float = 2.0) -> int:
    """
    First decision where rolling accuracy (window=50) reaches within gap_pp pp
    of final accuracy (mean of last 100 decisions).
    Returns N_POST_BOOTSTRAP if threshold never reached (conservative).
    """
    arr = np.array(post_accs)
    final_acc = arr[-100:].mean() * 100.0
    threshold = (final_acc - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    above = np.where(roll >= threshold)[0]
    if len(above) == 0:
        return N_POST_BOOTSTRAP
    return int(above[0]) + window


# ── Per-seed simulation ────────────────────────────────────────────────────────

def run_one_seed(seed: int) -> dict:
    """
    Run both conditions for a single seed.

    Seed isolation:
      gt_rng  (seed)          — ground truth mu_true / mu* / cat_weights
      hist_rng_c0 (seed+10000) — C0 historical decisions (un-enriched sigma)
      hist_rng_t2 (seed+20000) — T2 historical decisions (enriched sigma)
      learn_rng   (seed+30000) — SHARED learning alerts (identical for C0 and T2)
    """
    gt_rng       = np.random.RandomState(seed)
    hist_rng_c0  = np.random.RandomState(seed + 10000)
    hist_rng_t2  = np.random.RandomState(seed + 20000)
    learn_rng    = np.random.RandomState(seed + 30000)

    # Ground truth — shared by both conditions
    mu_true, gt_dist, cat_weights = build_ground_truth(gt_rng)
    mu_star = mu_true   # operational optimum: where mu converges under perfect labeling

    # ── Historical decisions ───────────────────────────────────────────────────
    hist_c0 = [
        sample_alert(hist_rng_c0, mu_true, gt_dist, cat_weights, SIGMA_VEC_UNENRICHED)
        for _ in range(N_BOOTSTRAP_HIST)
    ]
    hist_t2 = [
        sample_alert(hist_rng_t2, mu_true, gt_dist, cat_weights, SIGMA_VEC_ENRICHED)
        for _ in range(N_BOOTSTRAP_HIST)
    ]

    # ── mu_0 computation ──────────────────────────────────────────────────────
    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_t2 = compute_enriched_bootstrap_prior(
        hist_t2, SIGMA_ENRICHED, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
    )

    # ── Starting distance M4 ──────────────────────────────────────────────────
    starting_error_c0 = float(np.linalg.norm(mu0_c0 - mu_star))
    starting_error_t2 = float(np.linalg.norm(mu0_t2 - mu_star))

    # ── Pre-generate shared learning alerts (identical for C0 and T2) ─────────
    # Clone rng state so both conditions replay the same sequence
    learn_rng_c0 = np.random.RandomState(seed + 30000)
    learn_rng_t2 = np.random.RandomState(seed + 30000)

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)

    out = {}
    for cond, mu0, lr in [("C0", mu0_c0, learn_rng_c0), ("T2", mu0_t2, learn_rng_t2)]:
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        # Day-1 accuracy: first 50 alerts before any learning
        day1_correct = 0
        day1_rng = np.random.RandomState(seed + 40000)
        for _ in range(50):
            c, gt_a, f = sample_alert(day1_rng, mu_true, gt_dist, cat_weights, SIGMA_VEC_ENRICHED)
            res = scorer.score(f, c)
            if res.action_index == gt_a:
                day1_correct += 1
        day1_acc = day1_correct / 50.0

        # Post-bootstrap learning — enriched sigma, same alert sequence per seed
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, mu_true, gt_dist, cat_weights, SIGMA_VEC_ENRICHED)
            res = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            correct = (final_a == gt_a)
            scorer.update(f, c, final_a, correct, gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "mu0":           mu0,
            "starting_error": (starting_error_c0 if cond == "C0" else starting_error_t2),
            "day1_acc":      day1_acc,
            "post_accs":     post_accs,
            "n_half":        compute_n_half(post_accs),
        }

    return out


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_metrics(all_seed_results: list) -> dict:
    n_seeds = len(all_seed_results)

    n_half_c0 = np.array([r["C0"]["n_half"] for r in all_seed_results])
    n_half_t2 = np.array([r["T2"]["n_half"] for r in all_seed_results])

    diff = n_half_c0 - n_half_t2   # positive = T2 faster
    t_stat, p_val = scipy_stats.ttest_rel(n_half_c0, n_half_t2)
    d_m2 = float(diff.mean() / (diff.std() + 1e-9))
    reduction_pct = float((n_half_c0.mean() - n_half_t2.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2_pass = (float(n_half_t2.mean()) < float(n_half_c0.mean()) and
               float(p_val) < 0.01 and abs(d_m2) > 0.3)

    ci_c0 = scipy_stats.t.interval(0.95, n_seeds - 1,
                                    loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    ci_t2 = scipy_stats.t.interval(0.95, n_seeds - 1,
                                    loc=n_half_t2.mean(), scale=scipy_stats.sem(n_half_t2))

    # M4: starting distance
    err_c0 = np.array([r["C0"]["starting_error"] for r in all_seed_results])
    err_t2 = np.array([r["T2"]["starting_error"] for r in all_seed_results])
    err_reduction = float((err_c0.mean() - err_t2.mean()) / (err_c0.mean() + 1e-9) * 100)
    m4_pass = bool(err_t2.mean() < err_c0.mean())

    # Day-1 accuracy
    d1_c0 = np.array([r["C0"]["day1_acc"] for r in all_seed_results])
    d1_t2 = np.array([r["T2"]["day1_acc"] for r in all_seed_results])

    # Final accuracy
    final_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in all_seed_results])
    final_t2 = np.array([np.array(r["T2"]["post_accs"])[-100:].mean() for r in all_seed_results])
    _, p_final = scipy_stats.ttest_rel(final_t2, final_c0)

    return {
        "m2": {
            "n_half_c0_mean": round(float(n_half_c0.mean()), 1),
            "n_half_c0_ci95": [round(ci_c0[0], 1), round(ci_c0[1], 1)],
            "n_half_t2_mean": round(float(n_half_t2.mean()), 1),
            "n_half_t2_ci95": [round(ci_t2[0], 1), round(ci_t2[1], 1)],
            "reduction_pct":  round(reduction_pct, 2),
            "p_value":        round(float(p_val), 6),
            "t_stat":         round(float(t_stat), 4),
            "cohens_d":       round(d_m2, 4),
            "pass":           m2_pass,
        },
        "m4": {
            "starting_error_c0":   round(float(err_c0.mean()), 4),
            "starting_error_t2":   round(float(err_t2.mean()), 4),
            "reduction_pct":       round(err_reduction, 2),
            "pass":                m4_pass,
        },
        "day1_accuracy": {
            "c0":      round(float(d1_c0.mean()), 4),
            "t2":      round(float(d1_t2.mean()), 4),
            "delta_pp": round(float((d1_t2.mean() - d1_c0.mean()) * 100), 2),
        },
        "final_accuracy": {
            "c0_mean":  round(float(final_c0.mean()), 4),
            "t2_mean":  round(float(final_t2.mean()), 4),
            "delta_pp": round(float((final_t2.mean() - final_c0.mean()) * 100), 2),
            "p_value":  round(float(p_final), 6),
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("V-CGA-FROZEN v4b (corrected gradient, GAE 0.7.4)")
    print("=" * 65)
    print(f"Hypothesis: enriched mu_0 (T2) closer to mu* => fewer decisions")
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP_HIST={N_BOOTSTRAP_HIST}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"C0 sigma (enrichment factors x{_UNENRICH_MULT}): "
          + ", ".join(f"{f}={SIGMA_UNENRICHED[f]:.2f}" for f in ENRICHMENT_FACTORS))
    print(f"T2 sigma (enriched):              "
          + ", ".join(f"{f}={SIGMA_ENRICHED[f]:.2f}" for f in ENRICHMENT_FACTORS))

    t0 = time.time()
    all_seed_results = []

    for seed in range(N_SEEDS):
        result = run_one_seed(seed)
        all_seed_results.append(result)
        if (seed + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (seed + 1) / elapsed
            eta = (N_SEEDS - seed - 1) / rate
            print(f"  Seed {seed+1:3d}/{N_SEEDS} done  [{elapsed:.1f}s elapsed, ETA {eta:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    metrics = compute_metrics(all_seed_results)
    m2 = metrics["m2"]
    m4 = metrics["m4"]
    d1 = metrics["day1_accuracy"]
    fa = metrics["final_accuracy"]

    overall_pass = m2["pass"]
    verdict = "PASS" if overall_pass else "FAIL"
    claim_status = ("CLAIM-62/63 UNCONDITIONAL" if overall_pass
                    else "CLAIM-62/63 REMAINS CONDITIONAL")

    results = {
        "experiment":  "V-CGA-FROZEN-v4b",
        "version":     "v4b_corrected_gradient_gae_0.7.4",
        "fix_applied": "W_normalized*(f-mu) gradient — not f*W_normalized",
        "comparison_to_v4a": {
            "v4a_verdict": "FAIL — implementation error",
            "v4b_verdict": verdict,
        },
        "date":        "2026-03-23",
        "n_seeds":     N_SEEDS,
        "verdict":     verdict,
        "runtime_s":   round(elapsed_total, 1),
        "parameters": {
            "n_bootstrap_hist": N_BOOTSTRAP_HIST,
            "n_post_bootstrap": N_POST_BOOTSTRAP,
            "theta_min":        THETA_MIN,
            "tau":              TAU,
            "eta_confirm":      ETA_CONFIRM,
            "eta_override":     ETA_OVERRIDE,
            "q_bar":            Q_BAR,
            "alpha":            ALPHA,
        },
        "sigma_profiles": {
            "enriched":   SIGMA_ENRICHED,
            "unenriched": SIGMA_UNENRICHED,
            "unenrich_multiplier": _UNENRICH_MULT,
        },
        "m2":             m2,
        "m4":             m4,
        "day1_accuracy":  d1,
        "final_accuracy": fa,
        "claim_status":   claim_status,
    }

    # Save — do NOT overwrite v1/v2/v3
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v4b.json"

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    print(f"Results saved to {results_path}")

    # ── Verdict printout ───────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"V-CGA-FROZEN v4b (corrected gradient, GAE 0.7.4): {verdict}")
    print("=" * 65)
    print(f"M2 (N_half): C0={m2['n_half_c0_mean']:.1f}, T2={m2['n_half_t2_mean']:.1f}, "
          f"reduction={m2['reduction_pct']:.1f}%, p={m2['p_value']:.4f}, "
          f"d={m2['cohens_d']:.3f} — {'PASS' if m2['pass'] else 'FAIL'}")
    print(f"  C0 95% CI [{m2['n_half_c0_ci95'][0]:.1f}, {m2['n_half_c0_ci95'][1]:.1f}]  "
          f"T2 95% CI [{m2['n_half_t2_ci95'][0]:.1f}, {m2['n_half_t2_ci95'][1]:.1f}]")
    print(f"M4 (starting distance): C0={m4['starting_error_c0']:.3f}, "
          f"T2={m4['starting_error_t2']:.3f}, "
          f"reduction={m4['reduction_pct']:.1f}% — {'PASS' if m4['pass'] else 'FAIL'}")
    print(f"Day-1 accuracy: C0={d1['c0']:.1%}, T2={d1['t2']:.1%}, "
          f"delta={d1['delta_pp']:+.1f}pp")
    print(f"Final accuracy: C0={fa['c0_mean']:.1%}, T2={fa['t2_mean']:.1%}, "
          f"delta={fa['delta_pp']:+.1f}pp, p={fa['p_value']:.4f}")
    print()

    if overall_pass:
        print("CLAIM-62/63 -> UNCONDITIONAL. Empirical Bayes bootstrap validated.")
    else:
        print("CLAIM-62/63 remains CONDITIONAL.")
        if not m2["pass"]:
            d = m2["cohens_d"]
            p = m2["p_value"]
            if abs(d) <= 0.3 and p < 0.01:
                reason = f"effect size below gate (d={d:.3f} <= 0.3, p passes)"
            elif abs(d) > 0.3 and p >= 0.01:
                reason = f"p-value above gate (p={p:.4f} >= 0.01, d passes: {d:.3f})"
            elif m2["n_half_t2_mean"] >= m2["n_half_c0_mean"]:
                reason = f"wrong direction (T2 NOT faster than C0)"
            else:
                reason = f"both gates fail (d={d:.3f}, p={p:.4f})"
            print(f"  M2 fail reason: {reason}")
            print(f"  Actual effect size: d={d:.3f}  "
                  f"(need d>0.3 and p<0.01 and N_half_T2 < N_half_C0)")
        if not m4["pass"]:
            print(f"  M4 fail: T2 starting error ({m4['starting_error_t2']:.3f}) "
                  f">= C0 ({m4['starting_error_c0']:.3f})")
        else:
            print(f"  M4 (starting distance) PASSES: "
                  f"T2 is {m4['reduction_pct']:.1f}% closer to mu*")

    print("=" * 65)
    return results


if __name__ == "__main__":
    main()
