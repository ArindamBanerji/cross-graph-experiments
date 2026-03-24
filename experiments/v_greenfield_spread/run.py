"""
V-GREENFIELD-SPREAD — Does greenfield W-share rescue non-primary enrichment? (GAE 0.7.8)
==========================================================================================
GS-A: greenfield geometry + PRIMARY enrichment (threat_intel, spread=0.62)
      Expected to replicate Series 2 (d≈0.429, Day-1≈+6.7pp). Gate: d>=0.35.
GS-B: greenfield geometry + NON-PRIMARY enrichment (device_trust, spread=0.30)
      Run only if GS-A passes. W-share GS-B ≈ 40.8%.

Run:
    PYTHONUTF8=1 python experiments/v_greenfield_spread/run.py
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

# ── Greenfield pre-enrichment baseline sigma ────────────────────────────────────
GF_BASE = {
    "travel_match":            0.210,
    "asset_criticality":       0.180,
    "threat_intel_enrichment": 0.200,
    "time_anomaly":            0.160,
    "pattern_history":         0.190,
    "device_trust":            0.220,
}

# ── GS-A: threat_intel enriched to 0.090 ───────────────────────────────────────
GSA_AFTER  = {**GF_BASE, "threat_intel_enrichment": 0.090}
GSA_BEFORE = {**GF_BASE}   # threat_intel before=0.200

# ── GS-B: device_trust enriched to 0.100, threat_intel stays at 0.200 ──────────
GSB_AFTER  = {**GF_BASE, "device_trust": 0.100}
GSB_BEFORE = {**GF_BASE}   # device_trust before=0.220

def _sv(d): return np.array([d[f] for f in FACTOR_NAMES])
def _wshare(sa, factor):
    W = {f: 1/sa[f]**2 for f in FACTOR_NAMES}
    return W[factor] / sum(W.values()) * 100

GSA_WSHARE = _wshare(GSA_AFTER, "threat_intel_enrichment")
GSB_WSHARE = _wshare(GSB_AFTER, "device_trust")

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── μ* (validated healthcare SOC geometry) ──────────────────────────────────────
_MU_RAW = {
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

def _build_mu():
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = _build_mu()
_SPREADS = {f: float(MU_STAR[:,:,i].max()-MU_STAR[:,:,i].min())
            for i,f in enumerate(FACTOR_NAMES)}

def _gt_dist():
    gt = np.ones((N_CATEGORIES, N_ACTIONS)) * 0.1
    for c in range(N_CATEGORIES):
        gt[c, int(np.argmax(np.linalg.norm(MU_STAR[c], axis=-1)))] = 0.7
    gt /= gt.sum(axis=1, keepdims=True)
    return gt

GT_DIST = _gt_dist()

# ── Shared utilities ────────────────────────────────────────────────────────────
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

def compute_n_half(post_accs, window=50, gap_pp=2.0):
    arr = np.array(post_accs)
    thr  = (arr[-100:].mean()*100.0 - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window)/window, mode="valid")
    above = np.where(roll >= thr)[0]
    return int(above[0])+window if len(above) else N_POST_BOOTSTRAP

def run_one_seed(seed, sv_c0, sv_t2, sigma_after, sigma_before):
    rng_c0 = np.random.RandomState(seed+10000)
    rng_t2 = np.random.RandomState(seed+20000)
    hist_c0 = [sample_alert(rng_c0, sv_c0) for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(rng_t2, sv_t2) for _ in range(N_BOOTSTRAP)]
    mu0_c0  = std_bootstrap(hist_c0)
    mu0_t2  = compute_enriched_bootstrap_prior(
        hist_t2, sigma_after, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=sigma_before)
    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}
    for cond, mu0 in [("C0", mu0_c0), ("T2", mu0_t2)]:
        lr    = np.random.RandomState(seed+30000)
        sc    = ProfileScorer(mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
                              profile=profile, eta_override=ETA_OVERRIDE)
        d1rng = np.random.RandomState(seed+40000)
        d1_ok = sum(sc.score(f,c).action_index == a
                    for _ in range(50) for c,a,f in [sample_alert(d1rng, sv_t2)])
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c,gt_a,f = sample_alert(lr, sv_t2)
            pred_a   = sc.score(f,c).action_index
            final_a,_ = analyst_feedback(lr, pred_a, gt_a)
            sc.update(f, c, final_a, (final_a==gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a==gt_a))
        out[cond] = {"day1_acc": d1_ok/50.0, "post_accs": post_accs,
                     "n_half": compute_n_half(post_accs)}
    return out

def analyse(seed_results):
    n = len(seed_results)
    nh_c0 = np.array([r["C0"]["n_half"] for r in seed_results])
    nh_t2 = np.array([r["T2"]["n_half"] for r in seed_results])
    diff  = nh_c0 - nh_t2
    _, p  = scipy_stats.ttest_rel(nh_c0, nh_t2)
    d     = float(diff.mean()/(diff.std()+1e-9))
    ci    = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(), scale=scipy_stats.sem(diff))
    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in seed_results]))
    fa_c0 = np.mean([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_t2 = np.mean([np.array(r["T2"]["post_accs"])[-100:].mean() for r in seed_results])
    return dict(
        n_half_c0     = round(float(nh_c0.mean()),1),
        n_half_t2     = round(float(nh_t2.mean()),1),
        cohens_d      = round(abs(d),4),
        p_value       = round(float(p),6),
        ci_95         = [round(ci[0],2), round(ci[1],2)],
        day1_delta_pp = round((d1_t2-d1_c0)*100, 2),
        final_delta_pp= round(float((fa_t2-fa_c0)*100),2),
    )

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("V-GREENFIELD-SPREAD (GAE 0.7.8)")
    print("="*70)
    print(f"μ* spreads: threat_intel={_SPREADS['threat_intel_enrichment']:.3f}  "
          f"device_trust={_SPREADS['device_trust']:.3f}")
    print()

    # ── GS-A ───────────────────────────────────────────────────────────────────
    sv_a_c0 = _sv(GSA_BEFORE)
    sv_a_t2 = _sv(GSA_AFTER)
    print(f"GS-A: threat_intel enriched (spread=0.62)  "
          f"W-share={GSA_WSHARE:.1f}%")
    t0 = time.time()
    res_a = [run_one_seed(s, sv_a_c0, sv_a_t2, GSA_AFTER, GSA_BEFORE)
             for s in range(N_SEEDS)]
    elapsed_a = time.time()-t0
    st_a = analyse(res_a)
    print(f"  done [{elapsed_a:.1f}s]  "
          f"d={st_a['cohens_d']:.4f}  Day-1={st_a['day1_delta_pp']:+.2f}pp")

    # GS-A sanity gate
    replication_ok = bool(st_a['cohens_d'] >= 0.35)
    if not replication_ok:
        print(f"\nSTOP — GS-A d={st_a['cohens_d']:.4f} < 0.35 geometry mismatch.")
        print("GS-B not run.")
        results = {
            "gsa": {**st_a, "enriched_factor":"threat_intel","spread":0.62,
                    "w_share_pct":round(GSA_WSHARE,2)},
            "gsb": None,
            "interaction_finding": "unclear",
            "stop_reason": f"GS-A d={st_a['cohens_d']:.4f} < 0.35 gate"
        }
        _save(results)
        return

    # ── GS-B ───────────────────────────────────────────────────────────────────
    sv_b_c0 = _sv(GSB_BEFORE)
    sv_b_t2 = _sv(GSB_AFTER)
    print()
    print(f"GS-A passed (d={st_a['cohens_d']:.4f} >= 0.35). Running GS-B...")
    print(f"GS-B: device_trust enriched (spread=0.30)  "
          f"W-share={GSB_WSHARE:.1f}%")
    print(f"  W_device_trust_T2={1/0.100**2:.1f}  W_total_T2={sum(1/GSB_AFTER[f]**2 for f in FACTOR_NAMES):.2f}")
    t1 = time.time()
    res_b = [run_one_seed(s, sv_b_c0, sv_b_t2, GSB_AFTER, GSB_BEFORE)
             for s in range(N_SEEDS)]
    elapsed_b = time.time()-t1
    st_b = analyse(res_b)
    print(f"  done [{elapsed_b:.1f}s]  "
          f"d={st_b['cohens_d']:.4f}  Day-1={st_b['day1_delta_pp']:+.2f}pp")

    # ── Interaction finding ────────────────────────────────────────────────────
    gsa_d = st_a['cohens_d']
    gsb_d = st_b['cohens_d']
    gsa_d1 = st_a['day1_delta_pp']
    gsb_d1 = st_b['day1_delta_pp']

    if gsb_d1 > 2.0 and gsb_d > 0.20:
        interaction = "both_variables"   # W-share does rescue non-primary
    elif gsb_d1 <= 0.5 and gsb_d <= 0.15:
        interaction = "spread_only"      # spread is binding regardless of W-share
    else:
        interaction = "unclear"

    results = {
        "experiment": "V-GREENFIELD-SPREAD",
        "gae_version": "0.7.8",
        "gsa": {
            "enriched_factor": "threat_intel_enrichment",
            "spread": round(_SPREADS['threat_intel_enrichment'],3),
            "w_share_pct": round(GSA_WSHARE,2),
            "day1_delta_pp": st_a['day1_delta_pp'],
            "cohens_d": st_a['cohens_d'],
            "p_value": st_a['p_value'],
            "ci_95": st_a['ci_95'],
            "n_half_c0": st_a['n_half_c0'],
            "n_half_t2": st_a['n_half_t2'],
            "final_delta_pp": st_a['final_delta_pp'],
            "series2_replication": replication_ok,
        },
        "gsb": {
            "enriched_factor": "device_trust",
            "spread": round(_SPREADS['device_trust'],3),
            "w_share_pct": round(GSB_WSHARE,2),
            "day1_delta_pp": st_b['day1_delta_pp'],
            "cohens_d": st_b['cohens_d'],
            "p_value": st_b['p_value'],
            "ci_95": st_b['ci_95'],
            "n_half_c0": st_b['n_half_c0'],
            "n_half_t2": st_b['n_half_t2'],
            "final_delta_pp": st_b['final_delta_pp'],
        },
        "interaction_finding": interaction,
        "runtime_s": round(elapsed_a+elapsed_b, 1),
    }

    _save(results)

    # ── Print verdict ──────────────────────────────────────────────────────────
    rep_label = "CONFIRMED" if replication_ok and abs(gsa_d - 0.429) < 0.10 else \
                "REPLICATION OK" if replication_ok else "MISMATCH"
    print()
    print("="*70)
    print("V-GREENFIELD-SPREAD results:")
    print()
    print(f"GS-A (greenfield + primary, spread=0.62):")
    print(f"  W-share: {GSA_WSHARE:.1f}%  Day-1: {gsa_d1:+.2f}pp  "
          f"d={gsa_d:.4f}  p={st_a['p_value']:.6f}  "
          f"CI=[{st_a['ci_95'][0]:.2f},{st_a['ci_95'][1]:.2f}]")
    print(f"  Replication of Series 2: {rep_label}")
    print()
    print(f"GS-B (greenfield + non-primary, spread=0.30):")
    print(f"  W-share: {GSB_WSHARE:.1f}%  Day-1: {gsb_d1:+.2f}pp  "
          f"d={gsb_d:.4f}  p={st_b['p_value']:.6f}  "
          f"CI=[{st_b['ci_95'][0]:.2f},{st_b['ci_95'][1]:.2f}]")
    print()
    print(f"Interaction finding: {interaction}")
    print("Raw numbers for roadmap session.")
    print("="*70)

def _save(results):
    class _E(json.JSONEncoder):
        def default(self, o):
            if isinstance(o,(np.bool_,)): return bool(o)
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            return super().default(o)
    p = Path(__file__).parent/"results"/"results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, cls=_E))
    print(f"Results saved to {p}")

if __name__ == "__main__":
    main()
