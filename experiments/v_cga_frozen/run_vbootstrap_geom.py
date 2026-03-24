"""
V-BOOTSTRAP-GEOM — geometry-aware bootstrap (GAE 0.7.6)
=========================================================
Tests whether W_geom closes the B1 ceiling (d≈0.21) and achieves d>0.30.

Root cause of B1 ceiling: enriched factors (threat_intel, pattern_history) are
also the primary discriminators → bootstrap gradient fights dominant centroid
structure → d≈0.21–0.23 regardless of Δσ weighting.

Geometry-aware fix:
    W_boot_j = (σ_before_j/σ_after_j)² × (1 − dominant_axis_j)

dominant_axis computed from μ₀_c0 (standard bootstrap) per seed.
High-discriminating factors get near-zero weight; enriched non-discriminating
factors (device_trust) dominate.

C0:      standard bootstrap (σ_before=None path, un-enriched × 1.5)
T_geom:  compute_enriched_bootstrap_prior_geom(mu_current=μ₀_c0)

Both use A1×B1 geometry (realistic SOC, structured μ*).
M4_enriched uses dim 5 (device_trust) ONLY.

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_vbootstrap_geom.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import t as t_dist, nct

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import (
    CalibrationProfile,
    compute_enriched_bootstrap_prior,
    compute_enriched_bootstrap_prior_geom,
    compute_dominant_axis,
)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SEEDS          = 200
N_BOOTSTRAP      = 1200
N_POST_BOOTSTRAP = 500
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80

N_CATEGORIES = 6
N_ACTIONS    = 4
N_FACTORS    = 6

FACTOR_NAMES = [
    "travel_match",            # dim 0
    "asset_criticality",       # dim 1
    "threat_intel_enrichment", # dim 2 — enriched, high-discriminating → attenuated
    "time_anomaly",            # dim 3
    "pattern_history",         # dim 4 — enriched, partially attenuated
    "device_trust",            # dim 5 — enriched, low-discriminating → dominates
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
M4_DIM = 5    # device_trust only — the non-discriminating enriched factor

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Sigma profiles ──────────────────────────────────────────────────────────────
SIGMA_AFTER = {
    "travel_match":            0.11,
    "asset_criticality":       0.06,
    "threat_intel_enrichment": 0.13,   # enriched (was 0.18 in vceiling)
    "time_anomaly":            0.07,
    "pattern_history":         0.10,   # enriched (was 0.20 in vceiling)
    "device_trust":            0.09,   # enriched non-discriminating
}
SIGMA_BEFORE = {
    "travel_match":            0.27,
    "asset_criticality":       0.09,
    "threat_intel_enrichment": 0.27,   # actual pre-enrichment
    "time_anomaly":            0.105,
    "pattern_history":         0.30,   # actual pre-enrichment
    "device_trust":            0.24,   # large Δσ benefit
}
SV_AFTER  = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])
SV_BEFORE = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])

# ── Structured A1×B1 μ* (same as v5/v6/v7/vceiling) ───────────────────────────
_MU_STAR_RAW = {
    ("lateral_movement",    "escalate"):    [0.30, 0.85, 0.80, 0.70, 0.75, 0.40],
    ("lateral_movement",    "investigate"): [0.30, 0.70, 0.60, 0.55, 0.55, 0.40],
    ("lateral_movement",    "suppress"):    [0.30, 0.25, 0.20, 0.20, 0.20, 0.40],
    ("lateral_movement",    "monitor"):     [0.30, 0.45, 0.35, 0.35, 0.35, 0.40],
    ("insider_threat",      "escalate"):    [0.20, 0.80, 0.70, 0.65, 0.80, 0.25],
    ("insider_threat",      "investigate"): [0.20, 0.60, 0.55, 0.50, 0.60, 0.25],
    ("insider_threat",      "suppress"):    [0.20, 0.25, 0.20, 0.20, 0.20, 0.25],
    ("insider_threat",      "monitor"):     [0.20, 0.40, 0.35, 0.30, 0.40, 0.25],
    ("credential_access",   "escalate"):    [0.75, 0.75, 0.75, 0.70, 0.65, 0.35],
    ("credential_access",   "investigate"): [0.60, 0.60, 0.55, 0.55, 0.50, 0.35],
    ("credential_access",   "suppress"):    [0.20, 0.20, 0.20, 0.20, 0.20, 0.35],
    ("credential_access",   "monitor"):     [0.40, 0.35, 0.35, 0.30, 0.30, 0.35],
    ("data_exfiltration",   "escalate"):    [0.35, 0.90, 0.85, 0.75, 0.70, 0.30],
    ("data_exfiltration",   "investigate"): [0.35, 0.70, 0.65, 0.60, 0.55, 0.30],
    ("data_exfiltration",   "suppress"):    [0.35, 0.20, 0.20, 0.20, 0.20, 0.30],
    ("data_exfiltration",   "monitor"):     [0.35, 0.40, 0.35, 0.30, 0.30, 0.30],
    ("cloud_infrastructure","escalate"):    [0.50, 0.65, 0.70, 0.60, 0.55, 0.45],
    ("cloud_infrastructure","investigate"): [0.50, 0.50, 0.55, 0.45, 0.40, 0.45],
    ("cloud_infrastructure","suppress"):    [0.50, 0.20, 0.20, 0.20, 0.20, 0.45],
    ("cloud_infrastructure","monitor"):     [0.50, 0.35, 0.30, 0.25, 0.25, 0.45],
    ("threat_intel_match",  "escalate"):    [0.40, 0.70, 0.90, 0.65, 0.60, 0.35],
    ("threat_intel_match",  "investigate"): [0.40, 0.55, 0.70, 0.50, 0.45, 0.35],
    ("threat_intel_match",  "suppress"):    [0.40, 0.20, 0.20, 0.20, 0.20, 0.35],
    ("threat_intel_match",  "monitor"):     [0.40, 0.35, 0.45, 0.30, 0.25, 0.35],
}

def _build_mu_star():
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = _build_mu_star()

def _gt_dist(mu_star):
    gt = np.ones((N_CATEGORIES, N_ACTIONS)) * 0.1
    for c in range(N_CATEGORIES):
        gt[c, int(np.argmax(np.linalg.norm(mu_star[c], axis=-1)))] = 0.7
    gt /= gt.sum(axis=1, keepdims=True)
    return gt

GT_DIST = _gt_dist(MU_STAR)

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── Utilities ───────────────────────────────────────────────────────────────────
def sample_alert(rng, sigma_vec):
    c = int(rng.choice(N_CATEGORIES))
    a = int(rng.choice(N_ACTIONS, p=GT_DIST[c]))
    f = np.clip(MU_STAR[c, a] + rng.randn(N_FACTORS) * sigma_vec, 0.0, 1.0)
    return c, a, f

def analyst_feedback(rng, pred_a, gt_a):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(N_ACTIONS))), True
    return pred_a, False

def standard_bootstrap(historical_decisions):
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in historical_decisions:
        mu[c, a] += ETA_CONFIRM * (f - mu[c, a])
        mu[c, a]  = np.clip(mu[c, a], 0.0, 1.0)
    return mu

def compute_n_half(post_accs, window=50, gap_pp=2.0):
    arr = np.array(post_accs)
    threshold = (arr[-100:].mean() * 100.0 - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    above = np.where(roll >= threshold)[0]
    return int(above[0]) + window if len(above) else N_POST_BOOTSTRAP

def power_at_n(n, d, alpha=0.01):
    df = n - 1
    nc = d * np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc))

# ── Per-seed simulation ─────────────────────────────────────────────────────────
def run_one_seed(seed: int) -> dict:
    """
    C0:     standard bootstrap with un-enriched σ (sigma_before=None).
    T_geom: geometry-aware bootstrap using μ₀_c0 as mu_current.
    """
    hist_rng_c0   = np.random.RandomState(seed + 10000)
    hist_rng_tg   = np.random.RandomState(seed + 20000)
    learn_rng_c0  = np.random.RandomState(seed + 30000)
    learn_rng_tg  = np.random.RandomState(seed + 30000)  # identical sequence

    # Historical decisions
    hist_c0 = [sample_alert(hist_rng_c0, SV_BEFORE) for _ in range(N_BOOTSTRAP)]
    hist_tg = [sample_alert(hist_rng_tg, SV_AFTER)  for _ in range(N_BOOTSTRAP)]

    # μ₀: C0 standard bootstrap first (needed as mu_current for T_geom)
    mu0_c0 = standard_bootstrap(hist_c0)

    # μ₀: T_geom geometry-aware bootstrap (mu_current = μ₀_c0)
    mu0_tg = compute_enriched_bootstrap_prior_geom(
        hist_tg, SIGMA_AFTER, SIGMA_BEFORE,
        mu_current=mu0_c0,
        domain_config=DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
    )

    # Per-seed dominant_axis (from μ₀_c0 as the geometry reference)
    dom_axis = compute_dominant_axis(mu0_c0)

    # Starting distances (M4: dim 5 only)
    err_total_c0 = float(np.linalg.norm(mu0_c0 - MU_STAR))
    err_total_tg = float(np.linalg.norm(mu0_tg - MU_STAR))
    err_d5_c0    = float(np.linalg.norm(mu0_c0[:, :, M4_DIM] - MU_STAR[:, :, M4_DIM]))
    err_d5_tg    = float(np.linalg.norm(mu0_tg[:, :, M4_DIM] - MU_STAR[:, :, M4_DIM]))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, lr in [("C0", mu0_c0, learn_rng_c0), ("T_geom", mu0_tg, learn_rng_tg)]:
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )
        # Day-1 accuracy
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, SV_AFTER)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        # Post-bootstrap learning
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, SV_AFTER)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "err_total": err_total_c0 if cond == "C0" else err_total_tg,
            "err_d5":    err_d5_c0    if cond == "C0" else err_d5_tg,
            "day1_acc":  day1_correct / 50.0,
            "post_accs": post_accs,
            "n_half":    compute_n_half(post_accs),
        }

    return {**out, "dom_axis": dom_axis.tolist()}

# ── Analysis ────────────────────────────────────────────────────────────────────
def analyse(seed_results: list) -> dict:
    n = len(seed_results)

    n_half_c0 = np.array([r["C0"]["n_half"]     for r in seed_results])
    n_half_tg = np.array([r["T_geom"]["n_half"] for r in seed_results])
    diff = n_half_c0 - n_half_tg
    t_stat, p = scipy_stats.ttest_rel(n_half_c0, n_half_tg)
    d   = float(diff.mean() / (diff.std() + 1e-9))
    red = float((n_half_c0.mean() - n_half_tg.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2  = bool(n_half_tg.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3)

    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(),      scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    citg = scipy_stats.t.interval(0.95, n-1, loc=n_half_tg.mean(), scale=scipy_stats.sem(n_half_tg))

    # M4 (dim 5 — device_trust only)
    err_d5_c0 = float(np.mean([r["C0"]["err_d5"]     for r in seed_results]))
    err_d5_tg = float(np.mean([r["T_geom"]["err_d5"] for r in seed_results]))
    m4_pass   = bool(err_d5_tg < err_d5_c0)
    m4_red    = float((err_d5_c0 - err_d5_tg) / (err_d5_c0 + 1e-9) * 100)

    err_tot_c0 = float(np.mean([r["C0"]["err_total"]     for r in seed_results]))
    err_tot_tg = float(np.mean([r["T_geom"]["err_total"] for r in seed_results]))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"]     for r in seed_results]))
    d1_tg = float(np.mean([r["T_geom"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean()     for r in seed_results])
    fa_tg = np.array([np.array(r["T_geom"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_tg, fa_c0)

    # Average dominant_axis across seeds
    dom_axes = np.array([r["dom_axis"] for r in seed_results])  # (N_SEEDS, N_FACTORS)
    mean_dom_axis = {f: round(float(dom_axes[:, i].mean()), 4) for i, f in enumerate(FACTOR_NAMES)}

    obs_d = abs(d)
    return {
        "m2": {
            "n_half_c0":        round(float(n_half_c0.mean()), 1),
            "n_half_c0_ci95":   [round(ci0[0], 1),  round(ci0[1], 1)],
            "n_half_t_geom":    round(float(n_half_tg.mean()), 1),
            "n_half_tg_ci95":   [round(citg[0], 1), round(citg[1], 1)],
            "diff_mean":        round(float(diff.mean()), 2),
            "diff_ci95":        [round(ci[0], 2), round(ci[1], 2)],
            "reduction_pct":    round(red, 2),
            "p_value":          round(float(p), 6),
            "t_stat":           round(float(t_stat), 4),
            "cohens_d":         round(d, 4),
            "pass":             m2,
        },
        "m4_enriched_dim5": {
            "description": "device_trust dimension only (dim 5)",
            "c0_dist":          round(err_d5_c0, 4),
            "t_geom_dist":      round(err_d5_tg, 4),
            "reduction_pct":    round(m4_red, 2),
            "pass":             m4_pass,
        },
        "m4_total": {
            "c0_dist":          round(err_tot_c0, 4),
            "t_geom_dist":      round(err_tot_tg, 4),
            "pass":             bool(err_tot_tg < err_tot_c0),
        },
        "day1_accuracy": {
            "c0":               round(d1_c0, 4),
            "t_geom":           round(d1_tg, 4),
            "delta_pp":         round((d1_tg - d1_c0) * 100, 2),
        },
        "final_accuracy": {
            "c0_mean":          round(float(fa_c0.mean()), 4),
            "tg_mean":          round(float(fa_tg.mean()), 4),
            "delta_pp":         round(float((fa_tg.mean() - fa_c0.mean()) * 100), 2),
            "p_value":          round(float(p_fa), 6),
        },
        "mean_dominant_axis": mean_dom_axis,
        "power_analysis": {
            "observed_d":       round(obs_d, 4),
            "power_at_n100":    round(power_at_n(100, obs_d), 4),
            "power_at_n150":    round(power_at_n(150, obs_d), 4),
            "power_at_n200":    round(power_at_n(200, obs_d), 4),
        },
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-BOOTSTRAP-GEOM (geometry-aware bootstrap, GAE 0.7.6)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"Geometry: A1×B1 (realistic SOC, structured μ*)")
    print()

    # Print expected W_geom from MU_STAR dominant_axis for reference
    da_star = compute_dominant_axis(MU_STAR)
    sa = SV_AFTER; sb = SV_BEFORE
    er = (sb / sa) ** 2
    wg = er * (1.0 - da_star); wg = np.clip(wg, 1e-6, None)
    wn = wg / wg.mean()
    print("W_geom (from MU_STAR dominant_axis, reference only — per-seed uses μ₀_c0):")
    for i, f in enumerate(FACTOR_NAMES):
        print(f"  {f:<30} dom={da_star[i]:.3f}  enr_ratio={er[i]:.3f}  W_geom={wn[i]:.3f}")
    print()

    t0 = time.time()
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(seed))
        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (seed + 1) / elapsed
            print(f"  Seed {seed+1:3d}/{N_SEEDS}  "
                  f"[{elapsed:.1f}s, ETA {(N_SEEDS-seed-1)/rate:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    stats = analyse(all_results)
    m2    = stats["m2"]
    m4    = stats["m4_enriched_dim5"]
    m4t   = stats["m4_total"]
    d1    = stats["day1_accuracy"]
    fa    = stats["final_accuracy"]
    powa  = stats["power_analysis"]
    dom   = stats["mean_dominant_axis"]

    verdict = "PASS" if m2["pass"] else "FAIL"

    # Comparison to V-CEILING Arm A (Δσ scheme, same geometry)
    arm_a_d, arm_a_p = 0.230, 0.0014
    geom_d = abs(m2["cohens_d"])
    improved = bool(geom_d > arm_a_d)
    delta_str = (f"+{geom_d - arm_a_d:.3f} (improvement)"
                 if improved else f"{geom_d - arm_a_d:.3f} (no improvement)")

    claim_status = (
        "VALIDATED — B1 ceiling CLOSED. CLAIM-62/63 deployment-general."
        if m2["pass"] else
        f"PARTIAL IMPROVEMENT — Geometry awareness improves d ({arm_a_d:.3f}→{geom_d:.3f}) "
        f"but B1 ceiling remains. Two-regime structure partially addressed."
        if (improved and not m2["pass"] and float(m2["p_value"]) < 0.01) else
        "NO IMPROVEMENT — Geometry awareness has no additional benefit over Δσ. "
        "B1 ceiling is fundamental. Two-regime structure confirmed permanent."
        if (not improved and float(m2["p_value"]) < 0.01) else
        "INCONCLUSIVE — Effect present but p or d gate not met."
    )

    # ── Save ────────────────────────────────────────────────────────────────────
    results = {
        "experiment":  "V-BOOTSTRAP-GEOM",
        "version":     "geometry_aware_gae_0.7.6",
        "date":        "2026-03-23",
        "n_seeds":     N_SEEDS,
        "geometry":    "A1×B1 (realistic SOC, threat_intel high-discriminating)",
        "verdict":     verdict,
        "runtime_s":   round(elapsed_total, 1),
        "sigma_after":  SIGMA_AFTER,
        "sigma_before": SIGMA_BEFORE,
        "w_geom_profile": {
            f: round(float(wn[i]), 4) for i, f in enumerate(FACTOR_NAMES)
        },
        "w_geom_notes": {
            "threat_intel_enrichment": "~0 (attenuated — dominant_axis≈1.0)",
            "device_trust":            "dominates (low dominant_axis, large Δσ)",
            "pattern_history":         "partially attenuated (dominant_axis≈0.7)",
            "mu_current_source":       "C0 standard bootstrap μ₀ (per seed)",
        },
        "parameters": {
            "n_bootstrap":       N_BOOTSTRAP,
            "n_post_bootstrap":  N_POST_BOOTSTRAP,
            "theta_min":         THETA_MIN,
            "tau":               TAU,
            "eta_confirm":       ETA_CONFIRM,
            "eta_override":      ETA_OVERRIDE,
            "q_bar":             Q_BAR,
            "alpha":             ALPHA,
        },
        "m2":                m2,
        "m4_enriched_dim5":  m4,
        "m4_total":          m4t,
        "day1_accuracy":     d1,
        "final_accuracy":    fa,
        "mean_dominant_axis": dom,
        "power_analysis":    powa,
        "comparison_to_ceiling_arm_a": {
            "arm_a_d":    arm_a_d,
            "arm_a_p":    arm_a_p,
            "geom_d":     round(geom_d, 4),
            "geom_p":     m2["p_value"],
            "improvement": delta_str,
        },
        "claim_status": claim_status,
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_vbootstrap_geom.json"

    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEnc)
    print(f"Results saved to {out_path}")

    # ── Print verdict ────────────────────────────────────────────────────────────
    def _pf(b): return "PASS" if b else "FAIL"
    print()
    print("=" * 65)
    print(f"V-BOOTSTRAP-GEOM (geometry-aware, N=200, B1 geometry): {verdict}")
    print("=" * 65)
    print(f"M2: C0={m2['n_half_c0']:.1f}, T_geom={m2['n_half_t_geom']:.1f}, "
          f"reduction={m2['reduction_pct']:.1f}%, "
          f"p={m2['p_value']:.4f}, d={abs(m2['cohens_d']):.3f} [{_pf(m2['pass'])}]")
    print(f"  diff CI95: [{m2['diff_ci95'][0]:.2f}, {m2['diff_ci95'][1]:.2f}] decisions")
    print(f"M4 (device_trust dim 5): C0={m4['c0_dist']:.3f}, "
          f"T_geom={m4['t_geom_dist']:.3f}, reduction={m4['reduction_pct']:.1f}% "
          f"[{_pf(m4['pass'])}]")
    print(f"Day-1: C0={d1['c0']:.1%}, T_geom={d1['t_geom']:.1%}, "
          f"delta={d1['delta_pp']:+.1f}pp")
    print(f"Final accuracy: C0={fa['c0_mean']:.1%}, T_geom={fa['tg_mean']:.1%}, "
          f"delta={fa['delta_pp']:+.1f}pp, p={fa['p_value']:.4f}")
    print()
    print("Mean dominant_axis per factor (from μ₀_c0 across seeds):")
    for f, v in dom.items():
        print(f"  {f:<30} {v:.4f}")
    print()
    print(f"Power at observed d={powa['observed_d']:.3f}: "
          f"N=100: {powa['power_at_n100']:.1%}  "
          f"N=150: {powa['power_at_n150']:.1%}  "
          f"N=200: {powa['power_at_n200']:.1%}")
    print()
    print(f"vs V-CEILING Arm A (Δσ): d={arm_a_d:.3f} → geom: d={geom_d:.3f} ({delta_str})")
    print()
    if m2["pass"]:
        print("B1 ceiling CLOSED. CLAIM-62/63 deployment-general.")
    elif improved and not m2["pass"] and float(m2["p_value"]) < 0.01:
        print(f"Partial improvement. B1 ceiling partially addressed. "
              f"Two-regime structure remains.")
    else:
        print("Geometry awareness has no additional benefit. "
              "B1 ceiling is fundamental. Two-regime structure confirmed permanent.")
    print("=" * 65)

if __name__ == "__main__":
    main()
