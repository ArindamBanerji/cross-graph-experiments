"""
V-BOOTSTRAP-MOD — Three bootstrap modifications vs baseline (GAE 0.7.8)
=========================================================================
WS-2 geometry (asset_criticality σ=0.090). N=200 per arm.

M0: Standard Δσ bootstrap (current compute_enriched_bootstrap_prior)
M1: Dominance-aware scaling — W_boot_j × sqrt(W_max/W_j_post)
M2: Covariance propagation — after Δσ update, propagate enriched δμ
    to correlated non-enriched factors

Gate: M1 beats M0 (d_M1 > d_M0+0.03), M2 beats M0 (d_M2 > d_M0+0.03)

Run:
    PYTHONUTF8=1 python experiments/v_bootstrap_mod/run.py
"""

import sys, json, time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile, compute_enriched_bootstrap_prior

N_SEEDS, N_BOOTSTRAP, N_POST_BOOTSTRAP = 200, 1200, 500
THETA_MIN, TAU, ETA_CONFIRM, ETA_OVERRIDE = 0.467, 0.1, 0.05, 0.01
Q_BAR, ALPHA = 0.75, 0.80
N_CATEGORIES, N_ACTIONS, N_FACTORS = 6, 4, 6

FACTOR_NAMES = ["travel_match","asset_criticality","threat_intel_enrichment",
                "time_anomaly","pattern_history","device_trust"]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ACTIONS    = ["monitor","investigate","suppress","escalate"]
CATEGORIES = ["credential_access","threat_intel_match","lateral_movement",
              "data_exfiltration","insider_threat","cloud_infrastructure"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── WS-2 sigma profile ──────────────────────────────────────────────────────────
SIGMA_AFTER = {
    "travel_match":            0.165,
    "asset_criticality":       0.090,
    "threat_intel_enrichment": 0.090,   # enriched
    "time_anomaly":            0.070,
    "pattern_history":         0.095,
    "device_trust":            0.200,
}
SIGMA_BEFORE = {
    "travel_match":            0.165,
    "asset_criticality":       0.090,
    "threat_intel_enrichment": 0.210,   # pre-enrichment only
    "time_anomaly":            0.070,
    "pattern_history":         0.095,
    "device_trust":            0.200,
}
SV_C0 = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])
SV_T2 = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])

# W values post-enrichment
W_POST = np.array([1.0/SIGMA_AFTER[f]**2 for f in FACTOR_NAMES])
W_MAX  = float(W_POST.max())
W_MAX_FACTOR = FACTOR_NAMES[int(np.argmax(W_POST))]

# Δσ weights (standard M0)
W_BOOT_M0 = np.array([SIGMA_BEFORE[f]**2 / SIGMA_AFTER[f]**4 for f in FACTOR_NAMES])

# M1: dominance-aware scaling
# scale_j = sqrt(W_max / W_j_post)  — amplifies secondary factor bootstrap contribution
SCALE_M1 = np.sqrt(W_MAX / W_POST)
W_BOOT_M1 = W_BOOT_M0 * SCALE_M1

# M2: covariance propagation
# After Δσ update step, propagate enriched-factor δμ to correlated non-enriched factors.
# Enriched factor: threat_intel (dim 2)
# Non-enriched with |corr| > 0.30: pattern_history (dim 4) corr=0.45,
#                                   device_trust (dim 5) corr=0.35
ENRICH_IDX = IDX["threat_intel_enrichment"]   # dim 2
CORR_M2 = {
    IDX["pattern_history"]: 0.45,
    IDX["device_trust"]:    0.35,
}
# Propagation weight: corr(j,k) × (W_j/W_k)
W_TI = W_POST[ENRICH_IDX]
PROP_COEFF_M2 = {k: corr * (W_TI / W_POST[k]) for k, corr in CORR_M2.items()}

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

_MU_STAR_RAW = {
    ("lateral_movement","escalate"):    [0.30,0.50,0.75,0.35,0.80,0.65],
    ("lateral_movement","investigate"): [0.30,0.43,0.55,0.35,0.60,0.55],
    ("lateral_movement","suppress"):    [0.30,0.40,0.20,0.35,0.20,0.35],
    ("lateral_movement","monitor"):     [0.30,0.43,0.40,0.35,0.35,0.45],
    ("insider_threat","escalate"):      [0.25,0.55,0.70,0.30,0.75,0.65],
    ("insider_threat","investigate"):   [0.25,0.46,0.50,0.30,0.55,0.55],
    ("insider_threat","suppress"):      [0.25,0.40,0.20,0.30,0.20,0.35],
    ("insider_threat","monitor"):       [0.25,0.42,0.38,0.30,0.32,0.45],
    ("credential_access","escalate"):   [0.35,0.50,0.80,0.40,0.75,0.65],
    ("credential_access","investigate"):[0.35,0.43,0.60,0.40,0.58,0.55],
    ("credential_access","suppress"):   [0.35,0.40,0.20,0.40,0.22,0.35],
    ("credential_access","monitor"):    [0.35,0.42,0.42,0.40,0.33,0.45],
    ("data_exfiltration","escalate"):   [0.30,0.52,0.78,0.35,0.82,0.65],
    ("data_exfiltration","investigate"):[0.30,0.44,0.58,0.35,0.62,0.55],
    ("data_exfiltration","suppress"):   [0.30,0.40,0.20,0.35,0.20,0.35],
    ("data_exfiltration","monitor"):    [0.30,0.42,0.40,0.35,0.32,0.45],
    ("cloud_infrastructure","escalate"):[0.28,0.45,0.72,0.38,0.70,0.65],
    ("cloud_infrastructure","investigate"):[0.28,0.41,0.52,0.38,0.52,0.55],
    ("cloud_infrastructure","suppress"):[0.28,0.40,0.20,0.38,0.20,0.35],
    ("cloud_infrastructure","monitor"): [0.28,0.41,0.38,0.38,0.30,0.45],
    ("threat_intel_match","escalate"):  [0.32,0.52,0.82,0.36,0.78,0.65],
    ("threat_intel_match","investigate"):[0.32,0.44,0.62,0.36,0.58,0.55],
    ("threat_intel_match","suppress"):  [0.32,0.40,0.20,0.36,0.20,0.35],
    ("threat_intel_match","monitor"):   [0.32,0.42,0.44,0.36,0.33,0.45],
}

def _build_mu_star():
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = _build_mu_star()

def _gt_dist():
    gt = np.ones((N_CATEGORIES, N_ACTIONS)) * 0.1
    for c in range(N_CATEGORIES):
        gt[c, int(np.argmax(np.linalg.norm(MU_STAR[c], axis=-1)))] = 0.7
    gt /= gt.sum(axis=1, keepdims=True)
    return gt

GT_DIST = _gt_dist()

def sample_alert(rng, sv):
    c = int(rng.choice(N_CATEGORIES))
    a = int(rng.choice(N_ACTIONS, p=GT_DIST[c]))
    return c, a, np.clip(MU_STAR[c,a] + rng.randn(N_FACTORS)*sv, 0.0, 1.0)

def analyst_feedback(rng, pred_a, gt_a):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(N_ACTIONS))), True
    return pred_a, False

def std_bootstrap(hist):
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in hist:
        mu[c,a] += ETA_CONFIRM*(f - mu[c,a])
        mu[c,a]  = np.clip(mu[c,a], 0.0, 1.0)
    return mu

def m1_bootstrap(hist):
    """Dominance-aware: same Δσ weights scaled by sqrt(W_max/W_j_post)."""
    W_norm = W_BOOT_M1 / W_BOOT_M1.mean()
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in hist:
        grad = W_norm * (f - mu[c,a])
        mu[c,a] += ETA_CONFIRM * grad
        mu[c,a]  = np.clip(mu[c,a], 0.0, 1.0)
    return mu

def m2_bootstrap(hist):
    """Covariance propagation: Δσ bootstrap then propagate enriched δμ."""
    W_norm = W_BOOT_M0 / W_BOOT_M0.mean()
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in hist:
        mu_prev_ti = mu[c, a, ENRICH_IDX]
        grad = W_norm * (f - mu[c,a])
        mu[c,a] += ETA_CONFIRM * grad
        mu[c,a]  = np.clip(mu[c,a], 0.0, 1.0)
        # Propagation: delta on threat_intel propagates to correlated factors
        delta_ti = mu[c, a, ENRICH_IDX] - mu_prev_ti
        for k_idx, coeff in PROP_COEFF_M2.items():
            mu[c, a, k_idx] = np.clip(mu[c, a, k_idx] + delta_ti * coeff, 0.0, 1.0)
    return mu

def compute_n_half(post_accs, window=50, gap_pp=2.0):
    arr = np.array(post_accs)
    thr  = (arr[-100:].mean()*100.0 - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window)/window, mode="valid")
    above = np.where(roll >= thr)[0]
    return int(above[0])+window if len(above) else N_POST_BOOTSTRAP

def run_one_seed(seed, mu0):
    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    lr = np.random.RandomState(seed+30000)
    sc = ProfileScorer(mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
                       profile=profile, eta_override=ETA_OVERRIDE)
    d1_rng = np.random.RandomState(seed+40000)
    d1_ok = sum(sc.score(f, c).action_index == a
                for _ in range(50) for c,a,f in [sample_alert(d1_rng, SV_T2)])
    post_accs = []
    for _ in range(N_POST_BOOTSTRAP):
        c,gt_a,f = sample_alert(lr, SV_T2)
        pred_a = sc.score(f,c).action_index
        final_a,_ = analyst_feedback(lr, pred_a, gt_a)
        sc.update(f, c, final_a, (final_a==gt_a), gt_action_index=gt_a)
        post_accs.append(float(pred_a==gt_a))
    return {"day1_acc": d1_ok/50.0, "post_accs": post_accs,
            "n_half": compute_n_half(post_accs)}

def run_arm(arm_name, boot_fn):
    """Run N_SEEDS for one bootstrap modification arm."""
    results = []
    for seed in range(N_SEEDS):
        rng_t2 = np.random.RandomState(seed+20000)
        hist_t2 = [sample_alert(rng_t2, SV_T2) for _ in range(N_BOOTSTRAP)]
        mu0 = boot_fn(hist_t2)
        results.append(run_one_seed(seed, mu0))
    return results

def analyse(seed_results):
    nh = np.array([r["n_half"] for r in seed_results])
    d1 = float(np.mean([r["day1_acc"] for r in seed_results]))*100
    fa = np.mean([np.array(r["post_accs"])[-100:].mean() for r in seed_results])*100
    return dict(n_half_mean=round(float(nh.mean()),1),
                day1_pp=round(d1,2),
                final_pp=round(fa,2))

def compare_arms(res_a, res_b):
    """Cohen's d and p for n_half(A) vs n_half(B)."""
    nh_a = np.array([r["n_half"] for r in res_a])
    nh_b = np.array([r["n_half"] for r in res_b])
    diff = nh_a - nh_b
    _, p = scipy_stats.ttest_rel(nh_a, nh_b)
    d = float(diff.mean()/(diff.std()+1e-9))
    ci = scipy_stats.t.interval(0.95, len(diff)-1, loc=diff.mean(),
                                 scale=scipy_stats.sem(diff))
    return dict(cohens_d=round(abs(d),4), p_value=round(float(p),6),
                ci_95=[round(ci[0],2),round(ci[1],2)])

def main():
    print("="*68)
    print("V-BOOTSTRAP-MOD (WS-2 geometry, GAE 0.7.8)")
    print("="*68)
    print(f"W_max in T2 arm: {W_MAX:.1f} ({W_MAX_FACTOR})")
    print(f"M1 scale_j values (sqrt(W_max/W_j_post)):")
    for i, f in enumerate(FACTOR_NAMES):
        print(f"  {f:<30} scale={SCALE_M1[i]:.4f}  "
              f"W_boot_M0={W_BOOT_M0[i]:.2f}  W_boot_M1={W_BOOT_M1[i]:.2f}")
    print(f"  W_boot_threat_intel: M0={W_BOOT_M0[IDX['threat_intel_enrichment']]:.2f}  "
          f"M1={W_BOOT_M1[IDX['threat_intel_enrichment']]:.2f}")
    print(f"M2 propagation coefficients (corr × W_j/W_k):")
    for k_idx, coeff in PROP_COEFF_M2.items():
        print(f"  threat_intel → {FACTOR_NAMES[k_idx]}: {coeff:.4f}")
    print()

    t0 = time.time()
    print("Running M0 (standard Δσ)...")
    res_m0 = run_arm("M0", lambda h: compute_enriched_bootstrap_prior(
        h, SIGMA_AFTER, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=SIGMA_BEFORE))
    print(f"  done [{time.time()-t0:.1f}s]")

    t1 = time.time()
    print("Running M1 (dominance-aware)...")
    res_m1 = run_arm("M1", m1_bootstrap)
    print(f"  done [{time.time()-t1:.1f}s]")

    t2 = time.time()
    print("Running M2 (covariance propagation)...")
    res_m2 = run_arm("M2", m2_bootstrap)
    print(f"  done [{time.time()-t2:.1f}s]")

    elapsed = time.time()-t0
    print(f"All arms complete in {elapsed:.1f}s")

    st_m0 = analyse(res_m0)
    st_m1 = analyse(res_m1)
    st_m2 = analyse(res_m2)
    cmp_m0_m1 = compare_arms(res_m0, res_m1)
    cmp_m0_m2 = compare_arms(res_m0, res_m2)
    # Separate d for each arm vs C0 reference (using same C0 as M0's nh for reference)
    m1_vs_m0_d = cmp_m0_m1["cohens_d"]
    m2_vs_m0_d = cmp_m0_m2["cohens_d"]

    # Get individual arm d (vs C0 using per-arm standard bootstrap reference)
    # For reporting, compute individual d vs a shared C0 (standard bootstrap)
    res_c0 = []
    for seed in range(N_SEEDS):
        rng_c0 = np.random.RandomState(seed+10000)
        hist_c0 = [sample_alert(rng_c0, SV_C0) for _ in range(N_BOOTSTRAP)]
        mu0 = std_bootstrap(hist_c0)
        rng = np.random.RandomState(seed+20000)
        hist_t2_dummy = [sample_alert(rng, SV_T2) for _ in range(N_BOOTSTRAP)]
        # C0 uses standard bootstrap from SV_C0 history
        res_c0.append(run_one_seed(seed, mu0))

    def arm_d_vs_c0(res_t):
        nh_c0 = np.array([r["n_half"] for r in res_c0])
        nh_t  = np.array([r["n_half"] for r in res_t])
        diff  = nh_c0 - nh_t
        _, p  = scipy_stats.ttest_rel(nh_c0, nh_t)
        d     = float(diff.mean()/(diff.std()+1e-9))
        ci    = scipy_stats.t.interval(0.95, len(diff)-1, loc=diff.mean(),
                                        scale=scipy_stats.sem(diff))
        return dict(cohens_d=round(abs(d),4), p_value=round(float(p),6),
                    ci_95=[round(ci[0],2),round(ci[1],2)])

    dm0 = arm_d_vs_c0(res_m0)
    dm1 = arm_d_vs_c0(res_m1)
    dm2 = arm_d_vs_c0(res_m2)

    beats_m1 = bool(dm1["cohens_d"] > dm0["cohens_d"] + 0.03)
    beats_m2 = bool(dm2["cohens_d"] > dm0["cohens_d"] + 0.03)

    out = {
        "experiment": "V-BOOTSTRAP-MOD",
        "gae_version": "0.7.8",
        "geometry": "WS-2",
        "w_max_t2": round(W_MAX,1),
        "w_max_factor": W_MAX_FACTOR,
        "m1_scale_values": {f: round(float(SCALE_M1[i]),4) for i,f in enumerate(FACTOR_NAMES)},
        "m2_prop_coefficients": {FACTOR_NAMES[k]: round(v,4) for k,v in PROP_COEFF_M2.items()},
        "arms": {
            "M0": {"cohens_d": dm0["cohens_d"], "p": dm0["p_value"],
                   "ci_95": dm0["ci_95"], "day1_pp": st_m0["day1_pp"],
                   "n_half": st_m0["n_half_mean"], "final_pp": st_m0["final_pp"]},
            "M1": {"cohens_d": dm1["cohens_d"], "p": dm1["p_value"],
                   "ci_95": dm1["ci_95"], "day1_pp": st_m1["day1_pp"],
                   "n_half": st_m1["n_half_mean"], "final_pp": st_m1["final_pp"],
                   "beats_m0": beats_m1},
            "M2": {"cohens_d": dm2["cohens_d"], "p": dm2["p_value"],
                   "ci_95": dm2["ci_95"], "day1_pp": st_m2["day1_pp"],
                   "n_half": st_m2["n_half_mean"], "final_pp": st_m2["final_pp"],
                   "beats_m0": beats_m2},
        },
        "runtime_s": round(elapsed,1),
    }

    class _E(json.JSONEncoder):
        def default(self, o):
            if isinstance(o,(np.bool_,)): return bool(o)
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            return super().default(o)

    p = Path(__file__).parent/"results"/"results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, cls=_E))
    print(f"\nResults saved to {p}")

    print()
    print("="*68)
    print("V-BOOTSTRAP-MOD results (WS-2 geometry):")
    for arm in ["M0","M1","M2"]:
        a = out["arms"][arm]
        bm = f"  beats_M0={'YES' if a.get('beats_m0') else 'NO'}" if arm != "M0" else ""
        print(f"  {arm}: d={a['cohens_d']:.4f}  p={a['p']:.4f}  "
              f"Day-1={a['day1_pp']:+.1f}pp  n_half={a['n_half']:.1f}{bm}")
    print("="*68)

if __name__ == "__main__":
    main()
