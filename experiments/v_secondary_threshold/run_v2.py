"""
V-SECONDARY-THRESHOLD-2 — W-share transition in 13-35% band (GAE 0.7.8)
=========================================================================
Sweeps 20%, 25%, 30% W-share by varying asset_criticality sigma.
Monotonicity tolerance: 0.5pp.

Run:
    PYTHONUTF8=1 python experiments/v_secondary_threshold/run_v2.py
"""

import sys, json, time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile, compute_enriched_bootstrap_prior

N_SEEDS, N_BOOTSTRAP, N_POST_BOOTSTRAP = 100, 1200, 500
THETA_MIN, TAU, ETA_CONFIRM, ETA_OVERRIDE = 0.467, 0.1, 0.05, 0.01
Q_BAR, ALPHA = 0.75, 0.80
N_CATEGORIES, N_ACTIONS, N_FACTORS = 6, 4, 6
MONOTONIC_TOL_PP = 0.5

FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Fixed factors (unchanged between C0 and T2, and across all arms)
_FIXED_SIGMA = {
    "travel_match":            0.210,
    "threat_intel_enrichment": 0.200,
    "time_anomaly":            0.160,
    "pattern_history":         0.190,
}
_DT_SIGMA_BEFORE = 0.220
_DT_SIGMA_AFTER  = 0.100

# Three arms: asset_criticality sigma → target W-share
ARMS = [
    ("ST2-1", 0.061),  # W_ac≈268.6 → W-share≈20%
    ("ST2-2", 0.076),  # W_ac≈173.1 → W-share≈25%
    ("ST2-3", 0.098),  # W_ac≈104.3 → W-share≈30%
]

# Validated healthcare SOC mu* geometry
_MU_STAR_RAW = {
    ("lateral_movement",    "escalate"):    [0.30, 0.50, 0.75, 0.35, 0.80, 0.65],
    ("lateral_movement",    "investigate"): [0.30, 0.43, 0.55, 0.35, 0.60, 0.55],
    ("lateral_movement",    "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("lateral_movement",    "monitor"):     [0.30, 0.43, 0.40, 0.35, 0.35, 0.45],
    ("insider_threat",      "escalate"):    [0.25, 0.55, 0.70, 0.30, 0.75, 0.65],
    ("insider_threat",      "investigate"): [0.25, 0.46, 0.50, 0.30, 0.55, 0.55],
    ("insider_threat",      "suppress"):    [0.25, 0.40, 0.20, 0.30, 0.20, 0.35],
    ("insider_threat",      "monitor"):     [0.25, 0.42, 0.38, 0.30, 0.32, 0.45],
    ("credential_access",   "escalate"):    [0.35, 0.50, 0.80, 0.40, 0.75, 0.65],
    ("credential_access",   "investigate"): [0.35, 0.43, 0.60, 0.40, 0.58, 0.55],
    ("credential_access",   "suppress"):    [0.35, 0.40, 0.20, 0.40, 0.22, 0.35],
    ("credential_access",   "monitor"):     [0.35, 0.42, 0.42, 0.40, 0.33, 0.45],
    ("data_exfiltration",   "escalate"):    [0.30, 0.52, 0.78, 0.35, 0.82, 0.65],
    ("data_exfiltration",   "investigate"): [0.30, 0.44, 0.58, 0.35, 0.62, 0.55],
    ("data_exfiltration",   "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("data_exfiltration",   "monitor"):     [0.30, 0.42, 0.40, 0.35, 0.32, 0.45],
    ("cloud_infrastructure","escalate"):    [0.28, 0.45, 0.72, 0.38, 0.70, 0.65],
    ("cloud_infrastructure","investigate"): [0.28, 0.41, 0.52, 0.38, 0.52, 0.55],
    ("cloud_infrastructure","suppress"):    [0.28, 0.40, 0.20, 0.38, 0.20, 0.35],
    ("cloud_infrastructure","monitor"):     [0.28, 0.41, 0.38, 0.38, 0.30, 0.45],
    ("threat_intel_match",  "escalate"):    [0.32, 0.52, 0.82, 0.36, 0.78, 0.65],
    ("threat_intel_match",  "investigate"): [0.32, 0.44, 0.62, 0.36, 0.58, 0.55],
    ("threat_intel_match",  "suppress"):    [0.32, 0.40, 0.20, 0.36, 0.20, 0.35],
    ("threat_intel_match",  "monitor"):     [0.32, 0.42, 0.44, 0.36, 0.33, 0.45],
}

def _build_mu_star():
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = _build_mu_star()
_dt_spread = float(MU_STAR[:, :, IDX["device_trust"]].max() - MU_STAR[:, :, IDX["device_trust"]].min())

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
    return c, a, np.clip(MU_STAR[c, a] + rng.randn(N_FACTORS) * sv, 0.0, 1.0)

def analyst_feedback(rng, pred_a, gt_a):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(N_ACTIONS))), True
    return pred_a, False

def std_bootstrap(hist):
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in hist:
        mu[c, a] += ETA_CONFIRM * (f - mu[c, a])
        mu[c, a]  = np.clip(mu[c, a], 0.0, 1.0)
    return mu

def compute_n_half(post_accs, window=50, gap_pp=2.0):
    arr = np.array(post_accs)
    thr  = (arr[-100:].mean() * 100.0 - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    above = np.where(roll >= thr)[0]
    return int(above[0]) + window if len(above) else N_POST_BOOTSTRAP

def run_one_seed(seed, sigma_after, sigma_before):
    sv_c0 = np.array([sigma_before[f] for f in FACTOR_NAMES])
    sv_t2 = np.array([sigma_after[f]  for f in FACTOR_NAMES])

    class _DC:
        factor_names = FACTOR_NAMES
    domain_config = _DC()

    rng_c0 = np.random.RandomState(seed + 10000)
    rng_t2 = np.random.RandomState(seed + 20000)
    hist_c0 = [sample_alert(rng_c0, sv_c0) for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(rng_t2, sv_t2) for _ in range(N_BOOTSTRAP)]
    mu0_c0  = std_bootstrap(hist_c0)
    mu0_t2  = compute_enriched_bootstrap_prior(
        hist_t2, sigma_after, domain_config,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=sigma_before)
    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}
    for cond, mu0 in [("C0", mu0_c0), ("T2", mu0_t2)]:
        lr    = np.random.RandomState(seed + 30000)
        sc    = ProfileScorer(mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
                              profile=profile, eta_override=ETA_OVERRIDE)
        d1rng = np.random.RandomState(seed + 40000)
        d1_ok = sum(sc.score(f, c).action_index == a
                    for _ in range(50) for c, a, f in [sample_alert(d1rng, sv_t2)])
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, sv_t2)
            pred_a     = sc.score(f, c).action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            sc.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))
        out[cond] = {"day1_acc": d1_ok / 50.0, "post_accs": post_accs,
                     "n_half": compute_n_half(post_accs)}
    return out

def run_arm(arm_name, ac_sigma):
    sigma_after  = dict(_FIXED_SIGMA)
    sigma_after["asset_criticality"] = ac_sigma
    sigma_after["device_trust"]      = _DT_SIGMA_AFTER

    sigma_before = dict(_FIXED_SIGMA)
    sigma_before["asset_criticality"] = ac_sigma   # unchanged between C0/T2
    sigma_before["device_trust"]      = _DT_SIGMA_BEFORE

    w_dt_t2    = 1.0 / _DT_SIGMA_AFTER**2
    w_ac_t2    = 1.0 / ac_sigma**2
    w_fixed    = sum(1.0 / sigma_after[f]**2 for f in _FIXED_SIGMA)
    w_total_t2 = w_dt_t2 + w_ac_t2 + w_fixed
    w_share    = w_dt_t2 / w_total_t2 * 100.0

    print(f"\n--- {arm_name}: asset_criticality σ={ac_sigma:.3f} ---")
    print(f"  W_device_trust_T2 = {w_dt_t2:.1f}")
    print(f"  W_asset_crit_T2   = {w_ac_t2:.1f}")
    print(f"  W_fixed           = {w_fixed:.1f}")
    print(f"  W_total_T2        = {w_total_t2:.2f}")
    print(f"  W_share           = {w_share:.1f}%")

    t0 = time.time()
    all_results = [run_one_seed(s, sigma_after, sigma_before) for s in range(N_SEEDS)]
    elapsed = time.time() - t0
    print(f"  {N_SEEDS} seeds in {elapsed:.1f}s")

    nh_c0 = np.array([r["C0"]["n_half"] for r in all_results])
    nh_t2 = np.array([r["T2"]["n_half"] for r in all_results])
    diff  = nh_c0 - nh_t2
    _, p  = scipy_stats.ttest_rel(nh_c0, nh_t2)
    d     = float(diff.mean() / (diff.std() + 1e-9))
    ci    = scipy_stats.t.interval(0.95, N_SEEDS - 1,
                                   loc=diff.mean(), scale=scipy_stats.sem(diff))

    d1_c0    = float(np.mean([r["C0"]["day1_acc"] for r in all_results]))
    d1_t2    = float(np.mean([r["T2"]["day1_acc"] for r in all_results]))
    d1_delta = round((d1_t2 - d1_c0) * 100, 2)
    obs_d    = round(abs(d), 4)

    return {
        "ac_sigma":          ac_sigma,
        "actual_w_share_pct": round(w_share, 1),
        "day1_delta_pp":     d1_delta,
        "cohens_d":          obs_d,
        "p_value":           round(float(p), 6),
        "ci_95":             [round(ci[0], 2), round(ci[1], 2)],
        "n_half_c0":         round(float(nh_c0.mean()), 1),
        "n_half_t2":         round(float(nh_t2.mean()), 1),
        "runtime_s":         round(elapsed, 1),
    }

def main():
    print("=" * 65)
    print("V-SECONDARY-THRESHOLD-2 (GAE 0.7.8, N=100 per arm)")
    print("=" * 65)
    print(f"Band: 13–35% W-share | device_trust spread={_dt_spread:.2f} "
          f"σ {_DT_SIGMA_BEFORE}→{_DT_SIGMA_AFTER}")
    print(f"Monotonicity tolerance: {MONOTONIC_TOL_PP}pp")

    arm_results = {}
    prev_d1   = 0.0     # V-SPREAD-CONFIRM reference
    prev_arm  = "V-SPREAD-CONFIRM"
    monotonic = True
    anomaly_arm = None

    for arm_name, ac_sigma in ARMS:
        res = run_arm(arm_name, ac_sigma)
        arm_results[arm_name] = res

        # Monotonicity check with 0.5pp tolerance
        drop = prev_d1 - res["day1_delta_pp"]
        if drop > MONOTONIC_TOL_PP:
            print(f"\nSTOP — monotonicity violated (>{MONOTONIC_TOL_PP}pp): "
                  f"{arm_name} Day-1={res['day1_delta_pp']:+.2f}pp < "
                  f"{prev_arm} Day-1={prev_d1:+.2f}pp  (drop={drop:.2f}pp)")
            monotonic = False
            anomaly_arm = arm_name
            break

        prev_d1  = res["day1_delta_pp"]
        prev_arm = arm_name

    # Transition point: first arm where Day-1 > +2pp
    transition_pct = None
    for name, res in arm_results.items():
        if res["day1_delta_pp"] > 2.0:
            transition_pct = res["actual_w_share_pct"]
            break

    output = {
        "experiment":          "V-SECONDARY-THRESHOLD-2",
        "gae_version":         "0.7.8",
        "target_band":         "13-35% W-share",
        "reference_points": {
            "spread_confirm": {"w_share_pct": 13.0, "day1_pp": 0.0},
            "st1":            {"w_share_pct": 35.2, "day1_pp": 4.64},
            "gs_b":           {"w_share_pct": 40.8, "day1_pp": 4.50},
        },
        "arms":                    arm_results,
        "transition_point_w_share_pct": transition_pct,
        "monotonic":               monotonic,
    }
    if anomaly_arm:
        output["anomaly_arm"] = anomaly_arm

    out_path = Path(__file__).parent / "results" / "results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    # Final report
    print()
    print("=" * 65)
    print("V-SECONDARY-THRESHOLD-2 (13-35% W-share band):")
    print()
    print(f"  {'W-share':<9} {'Day-1 Δ':<10} {'d':<8} {'p':<9} CI95")
    print(f"  {'-'*63}")
    print(f"  ~13%      +0.0pp     0.12     —         —          (ref)")

    for name, res in arm_results.items():
        ws  = res["actual_w_share_pct"]
        d1  = res["day1_delta_pp"]
        d_  = res["cohens_d"]
        p_  = res["p_value"]
        lo  = res["ci_95"][0]
        hi  = res["ci_95"][1]
        print(f"  ~{ws:.0f}%     {d1:+.2f}pp    {d_:.4f}   {p_:.4f}    [{lo:.2f},{hi:.2f}]  ({name})")

    print(f"  ~35%      +4.64pp    0.1397   0.1675    —          (ref)")
    print()

    mon_str = "YES" if monotonic else f"NO — tolerance {MONOTONIC_TOL_PP}pp, anomaly at {anomaly_arm}"
    print(f"  Monotonic: {mon_str}")

    if transition_pct is not None:
        print(f"  Transition point (Day-1 first exceeds +2pp): ~{transition_pct}% W-share")
    else:
        print(f"  Transition point (Day-1 first exceeds +2pp): not reached in ST2-1 through ST2-3")

    print("  Raw numbers for roadmap session review.")
    print("=" * 65)

if __name__ == "__main__":
    main()
