"""
V-GREENFIELD-SPREAD GS-B — greenfield + non-primary factor enriched (GAE 0.7.8)
=================================================================================
Greenfield geometry (all sigma > 0.15). Enrich device_trust ONLY (secondary factor,
spread=0.30). W-share post = 40.8%.

GS-A reference: d=0.3342, Day-1=+6.79pp (threat_intel enriched, spread=0.62)

Run:
    PYTHONUTF8=1 python experiments/v_greenfield_spread/run_gsb.py
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

FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# GS-B: greenfield baseline + device_trust enriched
SIGMA_AFTER = {
    "travel_match":            0.210,
    "asset_criticality":       0.180,
    "threat_intel_enrichment": 0.200,
    "time_anomaly":            0.160,
    "pattern_history":         0.190,
    "device_trust":            0.100,   # enriched
}
SIGMA_BEFORE = {
    "travel_match":            0.210,
    "asset_criticality":       0.180,
    "threat_intel_enrichment": 0.200,
    "time_anomaly":            0.160,
    "pattern_history":         0.190,
    "device_trust":            0.220,   # pre-enrichment greenfield
}

SV_C0 = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])
SV_T2 = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])

W_POST = {f: 1.0 / SIGMA_AFTER[f]**2 for f in FACTOR_NAMES}
W_TOTAL_T2  = sum(W_POST.values())
W_SHARE_PCT = W_POST["device_trust"] / W_TOTAL_T2 * 100

W_BOOT_DT = SIGMA_BEFORE["device_trust"]**2 / SIGMA_AFTER["device_trust"]**4

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# Validated healthcare SOC mu* geometry
_MU_STAR_RAW = {
    ("lateral_movement",   "escalate"):    [0.30, 0.50, 0.75, 0.35, 0.80, 0.65],
    ("lateral_movement",   "investigate"): [0.30, 0.43, 0.55, 0.35, 0.60, 0.55],
    ("lateral_movement",   "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("lateral_movement",   "monitor"):     [0.30, 0.43, 0.40, 0.35, 0.35, 0.45],
    ("insider_threat",     "escalate"):    [0.25, 0.55, 0.70, 0.30, 0.75, 0.65],
    ("insider_threat",     "investigate"): [0.25, 0.46, 0.50, 0.30, 0.55, 0.55],
    ("insider_threat",     "suppress"):    [0.25, 0.40, 0.20, 0.30, 0.20, 0.35],
    ("insider_threat",     "monitor"):     [0.25, 0.42, 0.38, 0.30, 0.32, 0.45],
    ("credential_access",  "escalate"):    [0.35, 0.50, 0.80, 0.40, 0.75, 0.65],
    ("credential_access",  "investigate"): [0.35, 0.43, 0.60, 0.40, 0.58, 0.55],
    ("credential_access",  "suppress"):    [0.35, 0.40, 0.20, 0.40, 0.22, 0.35],
    ("credential_access",  "monitor"):     [0.35, 0.42, 0.42, 0.40, 0.33, 0.45],
    ("data_exfiltration",  "escalate"):    [0.30, 0.52, 0.78, 0.35, 0.82, 0.65],
    ("data_exfiltration",  "investigate"): [0.30, 0.44, 0.58, 0.35, 0.62, 0.55],
    ("data_exfiltration",  "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("data_exfiltration",  "monitor"):     [0.30, 0.42, 0.40, 0.35, 0.32, 0.45],
    ("cloud_infrastructure","escalate"):   [0.28, 0.45, 0.72, 0.38, 0.70, 0.65],
    ("cloud_infrastructure","investigate"):[0.28, 0.41, 0.52, 0.38, 0.52, 0.55],
    ("cloud_infrastructure","suppress"):   [0.28, 0.40, 0.20, 0.38, 0.20, 0.35],
    ("cloud_infrastructure","monitor"):    [0.28, 0.41, 0.38, 0.38, 0.30, 0.45],
    ("threat_intel_match", "escalate"):    [0.32, 0.52, 0.82, 0.36, 0.78, 0.65],
    ("threat_intel_match", "investigate"): [0.32, 0.44, 0.62, 0.36, 0.58, 0.55],
    ("threat_intel_match", "suppress"):    [0.32, 0.40, 0.20, 0.36, 0.20, 0.35],
    ("threat_intel_match", "monitor"):     [0.32, 0.42, 0.44, 0.36, 0.33, 0.45],
}

def _build_mu_star():
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = _build_mu_star()

_dt_spread = float(MU_STAR[:, :, IDX["device_trust"]].max() - MU_STAR[:, :, IDX["device_trust"]].min())
_ti_spread = float(MU_STAR[:, :, IDX["threat_intel_enrichment"]].max() - MU_STAR[:, :, IDX["threat_intel_enrichment"]].min())

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

def run_one_seed(seed):
    rng_c0 = np.random.RandomState(seed + 10000)
    rng_t2 = np.random.RandomState(seed + 20000)
    hist_c0 = [sample_alert(rng_c0, SV_C0) for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(rng_t2, SV_T2) for _ in range(N_BOOTSTRAP)]
    mu0_c0  = std_bootstrap(hist_c0)
    mu0_t2  = compute_enriched_bootstrap_prior(
        hist_t2, SIGMA_AFTER, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=SIGMA_BEFORE)
    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}
    for cond, mu0 in [("C0", mu0_c0), ("T2", mu0_t2)]:
        lr    = np.random.RandomState(seed + 30000)
        sc    = ProfileScorer(mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
                              profile=profile, eta_override=ETA_OVERRIDE)
        d1rng = np.random.RandomState(seed + 40000)
        d1_ok = sum(sc.score(f, c).action_index == a
                    for _ in range(50) for c, a, f in [sample_alert(d1rng, SV_T2)])
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, SV_T2)
            pred_a     = sc.score(f, c).action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            sc.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))
        out[cond] = {"day1_acc": d1_ok / 50.0, "post_accs": post_accs,
                     "n_half": compute_n_half(post_accs)}
    return out

def main():
    print("=" * 65)
    print("V-GREENFIELD-SPREAD GS-B (GAE 0.7.8, N=200)")
    print("=" * 65)
    print(f"Greenfield geometry: all sigma > 0.15")
    print(f"Enriched: device_trust  spread={_dt_spread:.2f}  W_boot={W_BOOT_DT:.1f}  W_share={W_SHARE_PCT:.1f}%")
    print(f"Primary:  threat_intel  spread={_ti_spread:.2f}  (not enriched)")
    print()

    t0 = time.time()
    all_results = [run_one_seed(s) for s in range(N_SEEDS)]
    elapsed = time.time() - t0
    print(f"All {N_SEEDS} seeds complete in {elapsed:.1f}s")

    n = N_SEEDS
    nh_c0 = np.array([r["C0"]["n_half"] for r in all_results])
    nh_t2 = np.array([r["T2"]["n_half"] for r in all_results])
    diff  = nh_c0 - nh_t2
    _, p  = scipy_stats.ttest_rel(nh_c0, nh_t2)
    d     = float(diff.mean() / (diff.std() + 1e-9))
    ci    = scipy_stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=scipy_stats.sem(diff))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in all_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in all_results]))
    d1_delta = round((d1_t2 - d1_c0) * 100, 2)

    fa_c0 = np.mean([np.array(r["C0"]["post_accs"])[-100:].mean() for r in all_results])
    fa_t2 = np.mean([np.array(r["T2"]["post_accs"])[-100:].mean() for r in all_results])

    obs_d = round(abs(d), 4)

    # GS-A reference
    gsa_d     = 0.3342
    gsa_day1  = 6.79

    # Interaction finding
    gsb_meaningful = bool(obs_d > 0.15 and d1_delta > 1.0)
    if obs_d >= 0.25 and d1_delta >= 2.0:
        interaction = "both_variables"
    elif obs_d < 0.15 and d1_delta < 1.0:
        interaction = "spread_only"
    else:
        interaction = "unclear"

    gsb_result = {
        "n_half_c0":       round(float(nh_c0.mean()), 1),
        "n_half_t2":       round(float(nh_t2.mean()), 1),
        "cohens_d":        obs_d,
        "p_value":         round(float(p), 6),
        "ci_95":           [round(ci[0], 2), round(ci[1], 2)],
        "day1_delta_pp":   d1_delta,
        "final_delta_pp":  round(float((fa_t2 - fa_c0) * 100), 2),
        "enriched_factor": "device_trust",
        "spread":          round(_dt_spread, 3),
        "w_share_pct":     round(W_SHARE_PCT, 2),
        "w_boot":          round(W_BOOT_DT, 1),
        "runtime_s":       round(elapsed, 1),
    }

    # Load and update results.json
    results_path = Path(__file__).parent / "results" / "results.json"
    existing = json.loads(results_path.read_text())
    existing["gsb"] = gsb_result
    existing["interaction_finding"] = interaction
    if "stop_reason" in existing:
        existing["stop_reason"] = "GS-A gate was single-factor; gate rescinded"
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"Results saved to {results_path}")

    print()
    print("=" * 65)
    print("V-GREENFIELD-SPREAD:")
    print(f"  GS-B (greenfield + non-primary, spread={_dt_spread:.2f}): "
          f"W-share: {W_SHARE_PCT:.1f}%  "
          f"Day-1: {d1_delta:+.2f}pp  "
          f"d={obs_d:.4f}  p={float(p):.4f}  "
          f"CI=[{ci[0]:.2f},{ci[1]:.2f}]")
    print(f"  GS-A (primary, spread=0.62): Day-1=+{gsa_day1:.2f}pp  d={gsa_d:.4f}")
    print(f"  GS-B (secondary, spread={_dt_spread:.2f}): "
          f"Day-1={d1_delta:+.2f}pp  d={obs_d:.4f}")
    print(f"  Interaction finding: {interaction}")
    print("=" * 65)

if __name__ == "__main__":
    main()
