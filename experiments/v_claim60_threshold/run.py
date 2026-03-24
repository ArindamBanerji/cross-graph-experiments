"""
V-CLAIM60-THRESHOLD — W-share calibration sweep, Series 1 (GAE 0.7.8)
=======================================================================
Five personas WS-1 through WS-5. N=100 seeds each.
Only variable: asset_criticality σ (controls W-share dilution of enriched factor).
Enrichment: threat_intel 0.210→0.090 (57% reduction) — identical across all personas.
μ* geometry: validated healthcare SOC (Persona A).

Purpose: calibrate empirically where Day-1 accuracy lift first becomes visible
as W_enriched_share_T2 increases from ~16% (WS-1) to ~24% (WS-5).

Run:
    PYTHONUTF8=1 python experiments/v_claim60_threshold/run.py
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

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SEEDS          = 100     # per persona
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
    "asset_criticality",       # dim 1  ← varies across personas
    "threat_intel_enrichment", # dim 2  ← enriched (C0: 0.210, T2: 0.090)
    "time_anomaly",            # dim 3
    "pattern_history",         # dim 4
    "device_trust",            # dim 5
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Fixed sigma values (identical across all personas) ──────────────────────────
TI_SIGMA_C0 = 0.210   # threat_intel pre-enrichment
TI_SIGMA_T2 = 0.090   # threat_intel post-enrichment

FIXED_SIGMA = {
    "travel_match":    0.165,
    "time_anomaly":    0.070,
    "pattern_history": 0.095,
    "device_trust":    0.200,
}

# ── Asset-criticality sweep: WS-1 to WS-5 ──────────────────────────────────────
PERSONAS = [
    {"name": "WS-1", "ac_sigma": 0.060},
    {"name": "WS-2", "ac_sigma": 0.090},
    {"name": "WS-3", "ac_sigma": 0.120},
    {"name": "WS-4", "ac_sigma": 0.150},
    {"name": "WS-5", "ac_sigma": 0.200},
]

def build_sigma_vecs(ac_sigma):
    """Returns (SV_C0, SV_T2, SIGMA_BEFORE_DICT, SIGMA_AFTER_DICT)."""
    sigma_after = {
        "travel_match":            FIXED_SIGMA["travel_match"],
        "asset_criticality":       ac_sigma,
        "threat_intel_enrichment": TI_SIGMA_T2,
        "time_anomaly":            FIXED_SIGMA["time_anomaly"],
        "pattern_history":         FIXED_SIGMA["pattern_history"],
        "device_trust":            FIXED_SIGMA["device_trust"],
    }
    sigma_before = {f: sigma_after[f] for f in FACTOR_NAMES}
    sigma_before["threat_intel_enrichment"] = TI_SIGMA_C0  # only threat_intel changes

    sigma_c0 = {f: sigma_before[f] for f in FACTOR_NAMES}  # C0 uses before values

    sv_c0 = np.array([sigma_c0[f]    for f in FACTOR_NAMES])
    sv_t2 = np.array([sigma_after[f] for f in FACTOR_NAMES])
    return sv_c0, sv_t2, sigma_before, sigma_after

def w_share(sigma_after_dict, sigma_before_dict):
    """Compute W_enriched_share_T2 for threat_intel."""
    def w(s): return 1.0 / s**2
    w_ti_t2 = w(sigma_after_dict["threat_intel_enrichment"])
    w_total  = sum(w(sigma_after_dict[f]) for f in FACTOR_NAMES)
    return w_ti_t2 / w_total * 100

def w_total(sigma_dict):
    return sum(1.0 / sigma_dict[f]**2 for f in FACTOR_NAMES)

# ── μ* geometry (validated healthcare SOC, Persona A) ──────────────────────────
_MU_STAR_RAW = {
    ("lateral_movement",     "escalate"):    [0.30, 0.50, 0.75, 0.35, 0.80, 0.65],
    ("lateral_movement",     "investigate"): [0.30, 0.43, 0.55, 0.35, 0.60, 0.55],
    ("lateral_movement",     "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("lateral_movement",     "monitor"):     [0.30, 0.43, 0.40, 0.35, 0.35, 0.45],
    ("insider_threat",       "escalate"):    [0.25, 0.55, 0.70, 0.30, 0.75, 0.65],
    ("insider_threat",       "investigate"): [0.25, 0.46, 0.50, 0.30, 0.55, 0.55],
    ("insider_threat",       "suppress"):    [0.25, 0.40, 0.20, 0.30, 0.20, 0.35],
    ("insider_threat",       "monitor"):     [0.25, 0.42, 0.38, 0.30, 0.32, 0.45],
    ("credential_access",    "escalate"):    [0.35, 0.50, 0.80, 0.40, 0.75, 0.65],
    ("credential_access",    "investigate"): [0.35, 0.43, 0.60, 0.40, 0.58, 0.55],
    ("credential_access",    "suppress"):    [0.35, 0.40, 0.20, 0.40, 0.22, 0.35],
    ("credential_access",    "monitor"):     [0.35, 0.42, 0.42, 0.40, 0.33, 0.45],
    ("data_exfiltration",    "escalate"):    [0.30, 0.52, 0.78, 0.35, 0.82, 0.65],
    ("data_exfiltration",    "investigate"): [0.30, 0.44, 0.58, 0.35, 0.62, 0.55],
    ("data_exfiltration",    "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("data_exfiltration",    "monitor"):     [0.30, 0.42, 0.40, 0.35, 0.32, 0.45],
    ("cloud_infrastructure", "escalate"):    [0.28, 0.45, 0.72, 0.38, 0.70, 0.65],
    ("cloud_infrastructure", "investigate"): [0.28, 0.41, 0.52, 0.38, 0.52, 0.55],
    ("cloud_infrastructure", "suppress"):    [0.28, 0.40, 0.20, 0.38, 0.20, 0.35],
    ("cloud_infrastructure", "monitor"):     [0.28, 0.41, 0.38, 0.38, 0.30, 0.45],
    ("threat_intel_match",   "escalate"):    [0.32, 0.52, 0.82, 0.36, 0.78, 0.65],
    ("threat_intel_match",   "investigate"): [0.32, 0.44, 0.62, 0.36, 0.58, 0.55],
    ("threat_intel_match",   "suppress"):    [0.32, 0.40, 0.20, 0.36, 0.20, 0.35],
    ("threat_intel_match",   "monitor"):     [0.32, 0.42, 0.44, 0.36, 0.33, 0.45],
}

def _build_mu_star() -> np.ndarray:
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

def standard_bootstrap(hist):
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in hist:
        mu[c, a] += ETA_CONFIRM * (f - mu[c, a])
        mu[c, a]  = np.clip(mu[c, a], 0.0, 1.0)
    return mu

def compute_n_half(post_accs, window=50, gap_pp=2.0):
    arr = np.array(post_accs)
    threshold = (arr[-100:].mean() * 100.0 - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    above = np.where(roll >= threshold)[0]
    return int(above[0]) + window if len(above) else N_POST_BOOTSTRAP

# ── Per-seed simulation ─────────────────────────────────────────────────────────
def run_one_seed(seed: int, sv_c0, sv_t2, sigma_before, sigma_after,
                 domain_config) -> dict:
    hist_rng_c0 = np.random.RandomState(seed + 10000)
    hist_rng_t2 = np.random.RandomState(seed + 20000)

    hist_c0 = [sample_alert(hist_rng_c0, sv_c0) for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(hist_rng_t2, sv_t2) for _ in range(N_BOOTSTRAP)]

    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_t2 = compute_enriched_bootstrap_prior(
        hist_t2, sigma_after, domain_config,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=sigma_before,
    )

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0 in [("C0", mu0_c0), ("T2", mu0_t2)]:
        lr = np.random.RandomState(seed + 30000)   # shared — identical alert sequence

        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, sv_t2)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, sv_t2)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "day1_acc":  day1_correct / 50.0,
            "post_accs": post_accs,
            "n_half":    compute_n_half(post_accs),
        }
    return out

# ── Analysis ────────────────────────────────────────────────────────────────────
def analyse(seed_results: list) -> dict:
    n = len(seed_results)
    n_half_c0 = np.array([r["C0"]["n_half"] for r in seed_results])
    n_half_t2 = np.array([r["T2"]["n_half"] for r in seed_results])
    diff = n_half_c0 - n_half_t2

    t_stat, p = scipy_stats.ttest_rel(n_half_c0, n_half_t2)
    d = float(diff.mean() / (diff.std() + 1e-9))

    ci = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(), scale=scipy_stats.sem(diff))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_t2 = np.array([np.array(r["T2"]["post_accs"])[-100:].mean() for r in seed_results])

    return {
        "day1_c0":       round(d1_c0, 4),
        "day1_t2":       round(d1_t2, 4),
        "day1_delta_pp": round((d1_t2 - d1_c0) * 100, 2),
        "n_half_c0":     round(float(n_half_c0.mean()), 1),
        "n_half_t2":     round(float(n_half_t2.mean()), 1),
        "diff_mean":     round(float(diff.mean()), 2),
        "ci_95":         [round(ci[0], 2), round(ci[1], 2)],
        "p_value":       round(float(p), 6),
        "cohens_d":      round(abs(d), 4),
        "final_c0":      round(float(fa_c0.mean()), 4),
        "final_t2":      round(float(fa_t2.mean()), 4),
        "final_delta_pp": round(float((fa_t2.mean() - fa_c0.mean()) * 100), 2),
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("V-CLAIM60-THRESHOLD W-share calibration sweep (GAE 0.7.8)")
    print("=" * 72)
    print(f"N_SEEDS={N_SEEDS} per persona, 5 personas (WS-1 to WS-5)")
    print(f"Enrichment: threat_intel {TI_SIGMA_C0}→{TI_SIGMA_T2} "
          f"(57% reduction), ΔW=+{1/TI_SIGMA_T2**2 - 1/TI_SIGMA_C0**2:.1f}")
    print()

    # ── V-CGA-FROZEN v3 reference ──────────────────────────────────────────────
    V3_AC_SIGMA  = 0.06
    V3_TA_SIGMA  = 0.07
    print("V-CGA-FROZEN v3 sigma profile (C0/B arm, from results_v3.json):")
    print(f"  asset_criticality: {V3_AC_SIGMA}")
    print(f"  time_anomaly:      {V3_TA_SIGMA}")
    print()

    # ── W-share pre-print ──────────────────────────────────────────────────────
    print("Pre-run W-share values:")
    persona_wshares = {}
    for p in PERSONAS:
        sv_c0, sv_t2, sb, sa = build_sigma_vecs(p["ac_sigma"])
        sc0 = {f: float(sv_c0[i]) for i, f in enumerate(FACTOR_NAMES)}
        sa_d = {f: float(sv_t2[i]) for i, f in enumerate(FACTOR_NAMES)}
        wt_c0 = w_total(sc0)
        wt_t2 = w_total(sa_d)
        ws    = w_share(sa_d, sb)
        persona_wshares[p["name"]] = round(ws, 2)
        print(f"  {p['name']}: asset_crit σ={p['ac_sigma']:.3f}, "
              f"W(ac)={1/p['ac_sigma']**2:.1f}")
        print(f"    W_total_C0={wt_c0:.1f}  W_total_T2={wt_t2:.1f}")
        print(f"    W_enriched_share_T2={ws:.1f}%")
    print()

    class _DomainConfig:
        factor_names = FACTOR_NAMES
    domain_config = _DomainConfig()

    # ── Run sweep ──────────────────────────────────────────────────────────────
    all_persona_results = []
    sweep_stats = []

    for p_idx, p in enumerate(PERSONAS):
        pname    = p["name"]
        ac_sigma = p["ac_sigma"]
        sv_c0, sv_t2, sigma_before, sigma_after = build_sigma_vecs(ac_sigma)

        t0 = time.time()
        seed_results = []
        for seed in range(N_SEEDS):
            seed_results.append(
                run_one_seed(seed, sv_c0, sv_t2, sigma_before, sigma_after,
                             domain_config)
            )

        elapsed = time.time() - t0
        stats = analyse(seed_results)
        stats["persona"]         = pname
        stats["ac_sigma"]        = ac_sigma
        stats["w_enriched_share_t2_pct"] = persona_wshares[pname]
        stats["runtime_s"]       = round(elapsed, 1)
        sweep_stats.append(stats)

        print(f"  {pname} done [{elapsed:.1f}s]: "
              f"Day-1={stats['day1_delta_pp']:+.1f}pp  "
              f"d={stats['cohens_d']:.3f}  p={stats['p_value']:.4f}  "
              f"n_half={stats['n_half_c0']:.0f}→{stats['n_half_t2']:.0f}")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    ws1 = sweep_stats[0]
    ws5 = sweep_stats[4]
    sanity_ok = True

    print()
    print("Sanity checks:")
    if ws1["day1_delta_pp"] > 1.5:
        print(f"  FAIL: WS-1 Day-1={ws1['day1_delta_pp']:+.1f}pp > +1.5pp "
              f"(geometry/sigma mismatch)")
        sanity_ok = False
    else:
        print(f"  WS-1 Day-1={ws1['day1_delta_pp']:+.1f}pp ≤ +1.5pp [OK]")

    if ws5["day1_delta_pp"] < 1.0:
        print(f"  FAIL: WS-5 Day-1={ws5['day1_delta_pp']:+.1f}pp < +1.0pp "
              f"(μ* geometry issue)")
        sanity_ok = False
    else:
        print(f"  WS-5 Day-1={ws5['day1_delta_pp']:+.1f}pp ≥ +1.0pp [OK]")

    # Monotonicity check on d
    d_vals = [s["cohens_d"] for s in sweep_stats]
    for i in range(1, len(d_vals)):
        drop = d_vals[i-1] - d_vals[i]
        if drop > 0.10:
            print(f"  FLAG: d drops {drop:.3f} from WS-{i} to WS-{i+1} "
                  f"({d_vals[i-1]:.3f}→{d_vals[i]:.3f})")
    print()

    # ── Find threshold (first persona where Day-1 > +2pp) ─────────────────────
    threshold_persona = None
    threshold_wshare  = None
    for s in sweep_stats:
        if s["day1_delta_pp"] > 2.0:
            threshold_persona = s["persona"]
            threshold_wshare  = s["w_enriched_share_t2_pct"]
            break

    # ── Save results ───────────────────────────────────────────────────────────
    save_list = []
    for s in sweep_stats:
        save_list.append({
            "persona":               s["persona"],
            "asset_criticality_sigma": s["ac_sigma"],
            "w_enriched_share_t2_pct": s["w_enriched_share_t2_pct"],
            "day1_delta_pp":         s["day1_delta_pp"],
            "cohens_d":              s["cohens_d"],
            "p_value":               s["p_value"],
            "ci_95":                 s["ci_95"],
            "n_half_c0":             s["n_half_c0"],
            "n_half_t2":             s["n_half_t2"],
            "final_delta_pp":        s["final_delta_pp"],
            "runtime_s":             s["runtime_s"],
        })

    out_dir  = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(save_list, fh, indent=2, cls=_NpEnc)
    print(f"Results saved to {out_path}")

    # ── Print verdict table ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("V-CLAIM60-THRESHOLD W-share calibration sweep (GAE 0.7.8):")
    print()
    hdr = (f"{'Persona':<8} | {'ac_sigma':>8} | {'W_enrich%':>9} | "
           f"{'Day-1 Δ':>8} | {'d':>6} | {'p':>8} | CI95")
    print(hdr)
    print("-" * len(hdr))
    for s in sweep_stats:
        print(f"  {s['persona']:<6} | "
              f"{s['ac_sigma']:>8.3f} | "
              f"{s['w_enriched_share_t2_pct']:>8.1f}% | "
              f"{s['day1_delta_pp']:>+7.1f}pp | "
              f"{s['cohens_d']:>6.3f} | "
              f"{s['p_value']:>8.4f} | "
              f"[{s['ci_95'][0]:.2f}, {s['ci_95'][1]:.2f}]")
    print()
    print("V-CGA-FROZEN v3 sigma profile (retrieved):")
    print(f"  asset_criticality: {V3_AC_SIGMA}")
    print(f"  time_anomaly:      {V3_TA_SIGMA}")
    print()
    if threshold_persona:
        print(f"Observed threshold (Day-1 first exceeds +2pp): "
              f"~{threshold_wshare:.1f}% W-share ({threshold_persona})")
    else:
        print("Observed threshold: Day-1 does not exceed +2pp in WS-1..WS-5")
    print()
    print("Raw numbers for roadmap session review.")
    print("=" * 72)

if __name__ == "__main__":
    main()
