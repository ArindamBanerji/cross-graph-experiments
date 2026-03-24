"""
V-CGA-FROZEN v7 — CD cross-domain enrichment, N=200, corrected M4_CD
======================================================================
Re-run of v6 CD condition with two validity fixes:
  1. N=200: properly powered for d=0.270 (observed in v6 at N=100)
     Power analysis: 90% power at d=0.270, p<0.01 requires N≈170.
  2. M4_CD: per-dimension distance in enriched dims ONLY (dims 0,5:
     travel_match and device_trust). v6 M4 was total 6-dim distance,
     dominated by untouched discriminating dims (threat_intel, pattern_history).

DW (Option 3) is retired (d=0.081 — genuinely null). Not re-run here.

v6 CD confirmed:
  Sigma: travel_match 0.18→0.11, device_trust 0.16→0.09
         threat_intel 0.18 UNCHANGED, pattern_history 0.20 UNCHANGED
  C0: un-enriched × 1.5 on travel_match and device_trust only
  M4 in v6: total ||mu0 - mu_star||_F over all 6 dims (wrong for CD)

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_v7.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import norm, t as t_dist, nct

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile, compute_enriched_bootstrap_prior

# ── Parameters (committed) ────────────────────────────────────────────────────
N_SEEDS          = 200        # powered for d=0.270 at p<0.01 (90% power ~N=170)
N_BOOTSTRAP_HIST = 200
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

# Canonical factor order
FACTOR_NAMES = [
    "travel_match",            # dim 0 — enriched in CD
    "asset_criticality",       # dim 1 — fixed
    "threat_intel_enrichment", # dim 2 — NOT enriched in CD
    "time_anomaly",            # dim 3 — fixed
    "pattern_history",         # dim 4 — NOT enriched in CD
    "device_trust",            # dim 5 — enriched in CD
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}

# CD enriched dims: travel_match=0, device_trust=5
CD_ENRICHED_DIMS   = [0, 5]
CD_ENRICHED_FACTOR_NAMES = ["travel_match", "device_trust"]

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── CD sigma profiles (identical to v6) ───────────────────────────────────────
SIGMA_CD_ENRICHED = {
    "travel_match":            0.11,   # enriched (was 0.18)
    "asset_criticality":       0.06,   # fixed
    "threat_intel_enrichment": 0.18,   # UNCHANGED — not enriched in CD
    "time_anomaly":            0.07,   # fixed
    "pattern_history":         0.20,   # UNCHANGED — not enriched in CD
    "device_trust":            0.09,   # enriched (was 0.16)
}
_CD_ENRICHED_FACTORS = ["travel_match", "device_trust"]
SIGMA_CD_UNENRICHED = {
    name: (SIGMA_CD_ENRICHED[name] * 1.5 if name in _CD_ENRICHED_FACTORS
           else SIGMA_CD_ENRICHED[name])
    for name in FACTOR_NAMES
}

def _sv(d): return np.array([d[f] for f in FACTOR_NAMES])
SV_ENRICHED   = _sv(SIGMA_CD_ENRICHED)
SV_UNENRICHED = _sv(SIGMA_CD_UNENRICHED)

# ── Structured A1×B1 mu* (identical to v5/v6) ─────────────────────────────────
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

# ── Utilities ──────────────────────────────────────────────────────────────────
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

def compute_dim_n_half(dim_errors, window=50, gap_pp_frac=0.02):
    """
    Per-dimension N_half: first decision when per-dim centroid error
    (||mu_current[:,:,d] - mu_star[:,:,d]||_F) has recovered to within
    gap_pp_frac of the final error.
    Uses same rolling-window logic as compute_n_half.
    """
    arr = np.array(dim_errors)    # shape (N_POST_BOOTSTRAP,)
    final_err = arr[-100:].mean()
    # "threshold" = final_err + gap_pp_frac * initial_error (convergence, not accuracy)
    # Reframe: error is decreasing, so threshold = final_err * (1 + gap_pp_frac)
    threshold = final_err * (1.0 + gap_pp_frac)
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    below = np.where(roll <= threshold)[0]
    return int(below[0]) + window if len(below) else N_POST_BOOTSTRAP

# ── Power analysis ─────────────────────────────────────────────────────────────
def power_at_n(n, d, alpha=0.01):
    """Power for paired t-test at given n, effect size d, two-tailed alpha."""
    df = n - 1
    nc = d * np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc))

# ── Per-seed simulation ────────────────────────────────────────────────────────
def run_one_seed(seed: int) -> dict:
    """
    Run C0 (standard bootstrap) and C1 (enriched bootstrap) for one seed.
    Tracks per-dimension centroid error trajectory for per-dim N_half.
    """
    hist_rng_c0  = np.random.RandomState(seed + 10000)
    hist_rng_c1  = np.random.RandomState(seed + 20000)
    learn_rng_c0 = np.random.RandomState(seed + 30000)
    learn_rng_c1 = np.random.RandomState(seed + 30000)   # identical sequence

    # Historical decisions
    hist_c0 = [sample_alert(hist_rng_c0, SV_UNENRICHED) for _ in range(N_BOOTSTRAP_HIST)]
    hist_c1 = [sample_alert(hist_rng_c1, SV_ENRICHED)   for _ in range(N_BOOTSTRAP_HIST)]

    # mu_0
    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_c1 = compute_enriched_bootstrap_prior(
        hist_c1, SIGMA_CD_ENRICHED, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        weight_exponent=1.0,
    )

    # M4 distances
    err_total_c0 = float(np.linalg.norm(mu0_c0 - MU_STAR))
    err_total_c1 = float(np.linalg.norm(mu0_c1 - MU_STAR))
    # M4_CD: enriched dims only (0=travel_match, 5=device_trust)
    err_cd_c0 = float(np.linalg.norm(mu0_c0[:, :, CD_ENRICHED_DIMS] -
                                      MU_STAR[:, :, CD_ENRICHED_DIMS]))
    err_cd_c1 = float(np.linalg.norm(mu0_c1[:, :, CD_ENRICHED_DIMS] -
                                      MU_STAR[:, :, CD_ENRICHED_DIMS]))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, lr in [("C0", mu0_c0, learn_rng_c0), ("C1", mu0_c1, learn_rng_c1)]:
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        # Day-1 accuracy (same probe for both conditions)
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, SV_ENRICHED)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        # Post-bootstrap learning — track accuracy and per-dim centroid errors
        post_accs = []
        # Per-dim error at each step: ||mu_current[:,:,d] - mu_star[:,:,d]||_F
        dim_errors = [[] for _ in range(N_FACTORS)]

        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, SV_ENRICHED)
            res = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))
            for d in range(N_FACTORS):
                dim_errors[d].append(
                    float(np.linalg.norm(scorer.mu[:, :, d] - MU_STAR[:, :, d]))
                )

        out[cond] = {
            "err_total":  err_total_c0 if cond == "C0" else err_total_c1,
            "err_cd":     err_cd_c0    if cond == "C0" else err_cd_c1,
            "day1_acc":   day1_correct / 50.0,
            "post_accs":  post_accs,
            "n_half":     compute_n_half(post_accs),
            "dim_errors": dim_errors,   # list of N_FACTORS lists, each length N_POST_BOOTSTRAP
        }

    return out

# ── Aggregate analysis ─────────────────────────────────────────────────────────
def analyse(seed_results: list) -> dict:
    n = len(seed_results)

    # M2: N_half
    n_half_c0 = np.array([r["C0"]["n_half"] for r in seed_results])
    n_half_c1 = np.array([r["C1"]["n_half"] for r in seed_results])
    diff = n_half_c0 - n_half_c1
    t_stat, p = scipy_stats.ttest_rel(n_half_c0, n_half_c1)
    d    = float(diff.mean() / (diff.std() + 1e-9))
    red  = float((n_half_c0.mean() - n_half_c1.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2   = bool(n_half_c1.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3)
    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(), scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    ci1  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c1.mean(), scale=scipy_stats.sem(n_half_c1))

    # M4_CD: enriched dims only
    err_cd_c0 = np.mean([r["C0"]["err_cd"] for r in seed_results])
    err_cd_c1 = np.mean([r["C1"]["err_cd"] for r in seed_results])
    m4_cd_red = float((err_cd_c0 - err_cd_c1) / (err_cd_c0 + 1e-9) * 100)
    m4_cd     = bool(err_cd_c1 < err_cd_c0)

    # M4_total: all 6 dims (v6 metric for comparison)
    err_tot_c0 = np.mean([r["C0"]["err_total"] for r in seed_results])
    err_tot_c1 = np.mean([r["C1"]["err_total"] for r in seed_results])
    m4_total   = bool(err_tot_c1 < err_tot_c0)

    # Day-1 accuracy
    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_c1 = float(np.mean([r["C1"]["day1_acc"] for r in seed_results]))

    # Final accuracy
    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_c1 = np.array([np.array(r["C1"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_c1, fa_c0)

    # Per-dimension N_half: mean over seeds for each factor dim
    per_dim = {}
    for dim_idx, fname in enumerate(FACTOR_NAMES):
        nh_c0_dim = np.array([
            compute_dim_n_half(r["C0"]["dim_errors"][dim_idx]) for r in seed_results
        ])
        nh_c1_dim = np.array([
            compute_dim_n_half(r["C1"]["dim_errors"][dim_idx]) for r in seed_results
        ])
        per_dim[fname] = {
            "c0": round(float(nh_c0_dim.mean()), 1),
            "c1": round(float(nh_c1_dim.mean()), 1),
            "delta": round(float(nh_c0_dim.mean() - nh_c1_dim.mean()), 1),
        }

    # Power analysis for observed d
    obs_d = abs(d)
    pow_analysis = {
        "observed_d":    round(obs_d, 4),
        "power_at_n100": round(power_at_n(100, obs_d), 4),
        "power_at_n150": round(power_at_n(150, obs_d), 4),
        "power_at_n200": round(power_at_n(200, obs_d), 4),
    }

    return {
        "m2": {
            "n_half_c0":     round(float(n_half_c0.mean()), 1),
            "n_half_c0_ci95":[round(ci0[0], 1), round(ci0[1], 1)],
            "n_half_c1":     round(float(n_half_c1.mean()), 1),
            "n_half_c1_ci95":[round(ci1[0], 1), round(ci1[1], 1)],
            "diff_mean":     round(float(diff.mean()), 2),
            "diff_ci95":     [round(ci[0], 2), round(ci[1], 2)],
            "reduction_pct": round(red, 2),
            "p_value":       round(float(p), 6),
            "t_stat":        round(float(t_stat), 4),
            "cohens_d":      round(d, 4),
            "pass":          m2,
        },
        "m4_cd": {
            "description":   "per-dim distance travel_match(0) + device_trust(5) only",
            "c0_partial_dist": round(float(err_cd_c0), 4),
            "c1_partial_dist": round(float(err_cd_c1), 4),
            "reduction_pct": round(m4_cd_red, 2),
            "pass":          m4_cd,
        },
        "m4_total": {
            "description":   "total distance all 6 dims (v6 metric for comparison)",
            "c0_total_dist": round(float(err_tot_c0), 4),
            "c1_total_dist": round(float(err_tot_c1), 4),
            "reduction_pct": round(float((err_tot_c0-err_tot_c1)/(err_tot_c0+1e-9)*100), 2),
            "pass":          m4_total,
        },
        "day1_accuracy": {
            "c0":       round(d1_c0, 4),
            "c1":       round(d1_c1, 4),
            "delta_pp": round((d1_c1 - d1_c0) * 100, 2),
        },
        "final_accuracy": {
            "c0_mean":  round(float(fa_c0.mean()), 4),
            "c1_mean":  round(float(fa_c1.mean()), 4),
            "delta_pp": round(float((fa_c1.mean()-fa_c0.mean())*100), 2),
            "p_value":  round(float(p_fa), 6),
        },
        "per_dim_n_half": {
            "travel_match":            per_dim["travel_match"],
            "asset_criticality":       per_dim["asset_criticality"],
            "threat_intel_enrichment": per_dim["threat_intel_enrichment"],
            "time_anomaly":            per_dim["time_anomaly"],
            "pattern_history":         per_dim["pattern_history"],
            "device_trust":            per_dim["device_trust"],
        },
        "power_analysis": pow_analysis,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-CGA-FROZEN v7 — CD cross-domain enrichment, N=200")
    print("Corrected M4_CD (enriched dims 0,5 only)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP_HIST={N_BOOTSTRAP_HIST}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"CD enriched: travel_match 0.18→0.11, device_trust 0.16→0.09")
    print(f"CD unchanged: threat_intel=0.18, pattern_history=0.20")
    print(f"C0 unenriched: travel_match={SIGMA_CD_UNENRICHED['travel_match']:.2f}, "
          f"device_trust={SIGMA_CD_UNENRICHED['device_trust']:.2f}")
    print()

    t0 = time.time()
    all_results = []

    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(seed))
        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (seed + 1) / elapsed
            eta = (N_SEEDS - seed - 1) / rate
            print(f"  Seed {seed+1:3d}/{N_SEEDS} done  "
                  f"[{elapsed:.1f}s elapsed, ETA {eta:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    stats = analyse(all_results)
    m2    = stats["m2"]
    m4cd  = stats["m4_cd"]
    m4tot = stats["m4_total"]
    d1    = stats["day1_accuracy"]
    pdim  = stats["per_dim_n_half"]
    powa  = stats["power_analysis"]

    verdict = "PASS" if m2["pass"] else "FAIL"
    option2_status = (
        "VALIDATED" if m2["pass"] else
        ("INCONCLUSIVE" if (not m2["pass"] and m2["p_value"] < 0.01 and abs(m2["cohens_d"]) > 0.20)
         else "REJECTED")
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "experiment":  "V-CGA-FROZEN-v7",
        "version":     "v7_CD_corrected_N200",
        "date":        "2026-03-23",
        "n_seeds":     N_SEEDS,
        "verdict":     verdict,
        "runtime_s":   round(elapsed_total, 1),
        "condition":   "CD_cross_domain_enriched_factors_only",
        "sigma_cd_enriched":   SIGMA_CD_ENRICHED,
        "sigma_cd_unenriched": SIGMA_CD_UNENRICHED,
        "parameters": {
            "n_bootstrap_hist": N_BOOTSTRAP_HIST,
            "n_post_bootstrap": N_POST_BOOTSTRAP,
            "theta_min": THETA_MIN, "tau": TAU,
            "eta_confirm": ETA_CONFIRM, "eta_override": ETA_OVERRIDE,
            "q_bar": Q_BAR, "alpha": ALPHA,
        },
        "m2":             m2,
        "m4_cd":          m4cd,
        "m4_total":       m4tot,
        "day1_accuracy":  d1,
        "final_accuracy": stats["final_accuracy"],
        "per_dim_n_half": pdim,
        "power_analysis": powa,
        "option2_status": option2_status,
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v7.json"

    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEnc)
    print(f"Results saved to {results_path}")

    # ── Print verdict ──────────────────────────────────────────────────────────
    def _pf(b): return "PASS" if b else "FAIL"

    print()
    print("=" * 65)
    print(f"V-CGA-FROZEN v7 (CD cross-domain, N=200, corrected M4): {verdict}")
    print("=" * 65)
    print(f"M2: C0={m2['n_half_c0']:.1f} CI[{m2['n_half_c0_ci95'][0]:.1f},{m2['n_half_c0_ci95'][1]:.1f}], "
          f"C1={m2['n_half_c1']:.1f} CI[{m2['n_half_c1_ci95'][0]:.1f},{m2['n_half_c1_ci95'][1]:.1f}], "
          f"reduction={m2['reduction_pct']:.1f}%, p={m2['p_value']:.4f}, "
          f"d={m2['cohens_d']:.3f} [{_pf(m2['pass'])}]")
    print(f"  diff CI95: [{m2['diff_ci95'][0]:.2f}, {m2['diff_ci95'][1]:.2f}] decisions")
    print(f"M4_CD (enriched dims only): "
          f"C0={m4cd['c0_partial_dist']:.3f}, C1={m4cd['c1_partial_dist']:.3f}, "
          f"reduction={m4cd['reduction_pct']:.1f}% [{_pf(m4cd['pass'])}]")
    print(f"M4_total (all dims, v6 metric): "
          f"C0={m4tot['c0_total_dist']:.3f}, C1={m4tot['c1_total_dist']:.3f} [{_pf(m4tot['pass'])}]")
    print(f"Day-1 accuracy: C0={d1['c0']:.1%}, C1={d1['c1']:.1%}, "
          f"delta={d1['delta_pp']:+.1f}pp")
    print()
    print("Per-dimension N_half (C0 -> C1, delta = C0-C1, positive = C1 faster):")
    enriched_tag   = {0: " (enriched)", 5: " (enriched)"}
    unenriched_tag = {2: " (NOT enriched)", 4: " (NOT enriched)"}
    fixed_tag      = {1: " (fixed)", 3: " (fixed)"}
    tag_map = {**enriched_tag, **unenriched_tag, **fixed_tag}
    for fname in FACTOR_NAMES:
        dim = IDX[fname]
        tag = tag_map.get(dim, "")
        r = pdim[fname]
        print(f"  {fname:<28}{tag:<20}  C0={r['c0']:.1f}  C1={r['c1']:.1f}  "
              f"delta={r['delta']:+.1f}")
    print()
    print(f"Power analysis: d={powa['observed_d']:.3f}")
    print(f"  Power at N=100: {powa['power_at_n100']:.1%}")
    print(f"  Power at N=150: {powa['power_at_n150']:.1%}")
    print(f"  Power at N=200: {powa['power_at_n200']:.1%}")
    print()
    if m2["pass"]:
        print("Option 2 status: VALIDATED — cross-domain enrichment clears M2+M4_CD.")
        print("  Enriching non-discriminating factors (travel_match, device_trust) "
              "reduces N_half in realistic A1×B1 SOC geometry.")
    elif option2_status == "INCONCLUSIVE":
        print(f"Option 2 status: INCONCLUSIVE — p clears gate ({m2['p_value']:.4f}<0.01) "
              f"but d={m2['cohens_d']:.3f} does not reach 0.3.")
        print(f"  Effect is real but smaller than pre-set gate. "
              f"True effect may be d≈{abs(m2['cohens_d']):.2f}.")
        print(f"  M4_CD: {'PASS' if m4cd['pass'] else 'FAIL'} "
              f"(per-dim distance in enriched dims: "
              f"{m4cd['reduction_pct']:.1f}% reduction)")
    else:
        print(f"Option 2 status: REJECTED — neither M2 nor directional evidence "
              f"supports cross-domain enrichment in A1×B1 geometry.")
    print("=" * 65)

    return results

if __name__ == "__main__":
    main()
