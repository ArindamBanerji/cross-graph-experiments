"""
V-BOOTSTRAP-W — Δσ-weighted bootstrap scheme (GAE 0.7.5)
=========================================================
Tests the Δσ weighting scheme: W_j = sigma_before_j² / sigma_after_j⁴

Root issue with v7 W_normalized (d=0.212, misses d>0.3):
  W_normalized uses 1/sigma_after² regardless of enrichment benefit.
  asset_criticality dominates (W_norm=2.24) despite never being enriched.

Δσ scheme fix:
  Factors that improved most get highest weight.
  Fixed factors (sigma_before==sigma_after) reduce to 1/sigma² — same as before.
  device_trust now W_norm=2.037 (highest), asset_criticality=1.450.

W_delta verified:
  device_trust:            2.037  ← enriched CD factor, highest
  asset_criticality:       1.450  ← fixed
  travel_match:            1.155  ← enriched CD factor
  time_anomaly:            1.066  ← fixed
  threat_intel_enrichment: 0.161  ← not enriched in CD
  pattern_history:         0.131  ← not enriched in CD

C0:      standard bootstrap (sigma_before=None, un-enriched sigma × 1.5)
T_delta: Δσ-weighted bootstrap (sigma_before provided, sigma_after=CD enriched)

μ*: Structured A1×B1 SOC geometry (same as v5/v6/v7).
Gates: M2 p<0.01 AND d>0.3, M4_enriched (dims 0,5 only).

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_vbootstrap_w.py
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
from gae.calibration import CalibrationProfile, compute_enriched_bootstrap_prior

# ── Parameters (committed) ────────────────────────────────────────────────────
N_SEEDS          = 200
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

FACTOR_NAMES = [
    "travel_match",            # dim 0 — enriched in CD
    "asset_criticality",       # dim 1 — fixed
    "threat_intel_enrichment", # dim 2 — NOT enriched in CD
    "time_anomaly",            # dim 3 — fixed
    "pattern_history",         # dim 4 — NOT enriched in CD
    "device_trust",            # dim 5 — enriched in CD
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
CD_ENRICHED_DIMS = [0, 5]   # travel_match, device_trust

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Sigma profiles ─────────────────────────────────────────────────────────────
SIGMA_AFTER = {               # enriched CD profile (sigma_after)
    "travel_match":            0.11,
    "asset_criticality":       0.06,
    "threat_intel_enrichment": 0.18,
    "time_anomaly":            0.07,
    "pattern_history":         0.20,
    "device_trust":            0.09,
}
SIGMA_BEFORE = {              # un-enriched (sigma_after × 1.5 for all factors)
    "travel_match":            0.27,   # 0.18 × 1.5 (enriched origin)
    "asset_criticality":       0.09,   # 0.06 × 1.5
    "threat_intel_enrichment": 0.27,   # 0.18 × 1.5
    "time_anomaly":            0.105,  # 0.07 × 1.5
    "pattern_history":         0.30,   # 0.20 × 1.5
    "device_trust":            0.24,   # 0.16 × 1.5 (enriched origin)
}
# C0 standard bootstrap uses un-enriched sigma (same as sigma_before)
SIGMA_C0 = SIGMA_BEFORE

# W_delta pre-computed for reporting (verified against spec)
_sb = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])
_sa = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])
_W_raw  = _sb**2 / _sa**4
_W_norm = _W_raw / _W_raw.mean()
W_DELTA_VALUES = {f: round(float(_W_norm[i]), 4) for i, f in enumerate(FACTOR_NAMES)}

def _sv(d): return np.array([d[f] for f in FACTOR_NAMES])
SV_AFTER  = _sv(SIGMA_AFTER)
SV_BEFORE = _sv(SIGMA_BEFORE)

# ── Structured A1×B1 mu* (identical to v5/v6/v7) ──────────────────────────────
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

def power_at_n(n, d, alpha=0.01):
    df = n - 1
    nc = d * np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc))

# ── Per-seed simulation ────────────────────────────────────────────────────────
def run_one_seed(seed: int) -> dict:
    """
    C0: standard bootstrap with un-enriched sigma (sigma_before=None path).
    T_delta: Δσ-weighted bootstrap with sigma_before provided.
    Both: identical post-bootstrap learning (SV_AFTER, same seed).
    """
    hist_rng_c0    = np.random.RandomState(seed + 10000)
    hist_rng_td    = np.random.RandomState(seed + 20000)
    learn_rng_c0   = np.random.RandomState(seed + 30000)
    learn_rng_td   = np.random.RandomState(seed + 30000)   # identical sequence

    # Historical decisions
    hist_c0 = [sample_alert(hist_rng_c0, SV_BEFORE) for _ in range(N_BOOTSTRAP_HIST)]
    hist_td = [sample_alert(hist_rng_td, SV_AFTER)  for _ in range(N_BOOTSTRAP_HIST)]

    # mu_0
    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_td = compute_enriched_bootstrap_prior(
        hist_td, SIGMA_AFTER, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=SIGMA_BEFORE,
    )

    # Starting distances
    err_total_c0 = float(np.linalg.norm(mu0_c0 - MU_STAR))
    err_total_td = float(np.linalg.norm(mu0_td - MU_STAR))
    err_cd_c0 = float(np.linalg.norm(
        mu0_c0[:, :, CD_ENRICHED_DIMS] - MU_STAR[:, :, CD_ENRICHED_DIMS]))
    err_cd_td = float(np.linalg.norm(
        mu0_td[:, :, CD_ENRICHED_DIMS] - MU_STAR[:, :, CD_ENRICHED_DIMS]))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, lr in [("C0", mu0_c0, learn_rng_c0), ("T_delta", mu0_td, learn_rng_td)]:
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        # Day-1 accuracy (identical probe for both conditions)
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
            res = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "err_total":  err_total_c0 if cond == "C0" else err_total_td,
            "err_cd":     err_cd_c0    if cond == "C0" else err_cd_td,
            "day1_acc":   day1_correct / 50.0,
            "post_accs":  post_accs,
            "n_half":     compute_n_half(post_accs),
        }

    return out

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyse(seed_results: list) -> dict:
    n = len(seed_results)

    n_half_c0 = np.array([r["C0"]["n_half"]      for r in seed_results])
    n_half_td = np.array([r["T_delta"]["n_half"]  for r in seed_results])
    diff = n_half_c0 - n_half_td
    t_stat, p = scipy_stats.ttest_rel(n_half_c0, n_half_td)
    d    = float(diff.mean() / (diff.std() + 1e-9))
    red  = float((n_half_c0.mean() - n_half_td.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2   = bool(n_half_td.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3)

    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(), scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    citd = scipy_stats.t.interval(0.95, n-1, loc=n_half_td.mean(), scale=scipy_stats.sem(n_half_td))

    err_cd_c0 = float(np.mean([r["C0"]["err_cd"]      for r in seed_results]))
    err_cd_td = float(np.mean([r["T_delta"]["err_cd"] for r in seed_results]))
    m4_cd     = bool(err_cd_td < err_cd_c0)
    m4_cd_red = float((err_cd_c0 - err_cd_td) / (err_cd_c0 + 1e-9) * 100)

    err_tot_c0 = float(np.mean([r["C0"]["err_total"]      for r in seed_results]))
    err_tot_td = float(np.mean([r["T_delta"]["err_total"] for r in seed_results]))
    m4_total   = bool(err_tot_td < err_tot_c0)

    d1_c0 = float(np.mean([r["C0"]["day1_acc"]      for r in seed_results]))
    d1_td = float(np.mean([r["T_delta"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean()      for r in seed_results])
    fa_td = np.array([np.array(r["T_delta"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_td, fa_c0)

    obs_d = abs(d)
    return {
        "m2": {
            "n_half_c0":       round(float(n_half_c0.mean()), 1),
            "n_half_c0_ci95":  [round(ci0[0], 1),  round(ci0[1], 1)],
            "n_half_t_delta":  round(float(n_half_td.mean()), 1),
            "n_half_td_ci95":  [round(citd[0], 1), round(citd[1], 1)],
            "diff_mean":       round(float(diff.mean()), 2),
            "diff_ci95":       [round(ci[0], 2), round(ci[1], 2)],
            "reduction_pct":   round(red, 2),
            "p_value":         round(float(p), 6),
            "t_stat":          round(float(t_stat), 4),
            "cohens_d":        round(d, 4),
            "pass":            m2,
        },
        "m4_enriched": {
            "c0_partial_dist":      round(err_cd_c0, 4),
            "t_delta_partial_dist": round(err_cd_td, 4),
            "reduction_pct":        round(m4_cd_red, 2),
            "pass":                 m4_cd,
        },
        "m4_total": {
            "c0_total_dist":      round(err_tot_c0, 4),
            "t_delta_total_dist": round(err_tot_td, 4),
            "reduction_pct":      round(float((err_tot_c0-err_tot_td)/(err_tot_c0+1e-9)*100), 2),
            "pass":               m4_total,
        },
        "day1_accuracy": {
            "c0":       round(d1_c0, 4),
            "t_delta":  round(d1_td, 4),
            "delta_pp": round((d1_td - d1_c0) * 100, 2),
        },
        "final_accuracy": {
            "c0_mean":  round(float(fa_c0.mean()), 4),
            "td_mean":  round(float(fa_td.mean()), 4),
            "delta_pp": round(float((fa_td.mean()-fa_c0.mean())*100), 2),
            "p_value":  round(float(p_fa), 6),
        },
        "power_analysis": {
            "observed_d":    round(obs_d, 4),
            "power_at_n100": round(power_at_n(100, obs_d), 4),
            "power_at_n150": round(power_at_n(150, obs_d), 4),
            "power_at_n200": round(power_at_n(200, obs_d), 4),
        },
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-BOOTSTRAP-W (Δσ-weighted bootstrap, GAE 0.7.5)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP_HIST={N_BOOTSTRAP_HIST}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print("W_delta values:")
    for f in FACTOR_NAMES:
        tag = " (enriched)" if f in ("travel_match", "device_trust") else ""
        print(f"  {f:<30} W_norm={W_DELTA_VALUES[f]:.3f}{tag}")
    print()

    t0 = time.time()
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(seed))
        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (seed + 1) / elapsed
            print(f"  Seed {seed+1:3d}/{N_SEEDS}  [{elapsed:.1f}s, ETA {(N_SEEDS-seed-1)/rate:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    stats  = analyse(all_results)
    m2     = stats["m2"]
    m4e    = stats["m4_enriched"]
    m4t    = stats["m4_total"]
    d1     = stats["day1_accuracy"]
    fa     = stats["final_accuracy"]
    powa   = stats["power_analysis"]

    verdict = "PASS" if m2["pass"] else "FAIL"

    # Improvement over v7 W_normalized
    v7_d, v7_p = 0.212, 0.003
    improved = bool(abs(m2["cohens_d"]) > v7_d)
    improvement_str = (f"+{abs(m2['cohens_d'])-v7_d:.3f} (better)" if improved
                       else f"{abs(m2['cohens_d'])-v7_d:.3f} (worse or same)")

    claim_status = (
        "VALIDATED — Δσ scheme clears M2+M4_enriched in A1×B1 geometry."
        if m2["pass"] and m4e["pass"] else
        "INCONCLUSIVE — p<0.01 but d<0.3; effect real, mechanism partially confirmed."
        if (not m2["pass"] and float(m2["p_value"]) < 0.01) else
        "REJECTED — Δσ scheme does not improve over W_normalized in A1×B1 geometry."
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "experiment":  "V-BOOTSTRAP-W",
        "version":     "delta_sigma_weighted_gae_0.7.5",
        "date":        "2026-03-23",
        "n_seeds":     N_SEEDS,
        "verdict":     verdict,
        "runtime_s":   round(elapsed_total, 1),
        "w_delta_values": W_DELTA_VALUES,
        "sigma_after":   SIGMA_AFTER,
        "sigma_before":  SIGMA_BEFORE,
        "parameters": {
            "n_bootstrap_hist": N_BOOTSTRAP_HIST,
            "n_post_bootstrap": N_POST_BOOTSTRAP,
            "theta_min": THETA_MIN, "tau": TAU,
            "eta_confirm": ETA_CONFIRM, "eta_override": ETA_OVERRIDE,
            "q_bar": Q_BAR, "alpha": ALPHA,
        },
        "m2":              m2,
        "m4_enriched":     m4e,
        "m4_total":        m4t,
        "day1_accuracy":   d1,
        "final_accuracy":  fa,
        "power_analysis":  powa,
        "comparison_to_v7": {
            "v7_d":           v7_d,
            "v7_p":           v7_p,
            "vbootstrap_w_d": round(abs(m2["cohens_d"]), 4),
            "vbootstrap_w_p": m2["p_value"],
            "improvement":    improvement_str,
        },
        "claim_status": claim_status,
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_vbootstrap_w.json"

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
    print(f"V-BOOTSTRAP-W (Δσ-weighted, N=200): {verdict}")
    print("=" * 65)
    print(f"M2: C0={m2['n_half_c0']:.1f} CI[{m2['n_half_c0_ci95'][0]:.1f},{m2['n_half_c0_ci95'][1]:.1f}], "
          f"T_delta={m2['n_half_t_delta']:.1f} CI[{m2['n_half_td_ci95'][0]:.1f},{m2['n_half_td_ci95'][1]:.1f}], "
          f"reduction={m2['reduction_pct']:.1f}%, p={m2['p_value']:.4f}, "
          f"d={m2['cohens_d']:.3f} [{_pf(m2['pass'])}]")
    print(f"  diff CI95: [{m2['diff_ci95'][0]:.2f}, {m2['diff_ci95'][1]:.2f}] decisions")
    print(f"M4_enriched (dims 0,5): "
          f"C0={m4e['c0_partial_dist']:.3f}, T_delta={m4e['t_delta_partial_dist']:.3f}, "
          f"reduction={m4e['reduction_pct']:.1f}% [{_pf(m4e['pass'])}]")
    print(f"M4_total (all dims):    "
          f"C0={m4t['c0_total_dist']:.3f}, T_delta={m4t['t_delta_total_dist']:.3f} [{_pf(m4t['pass'])}]")
    print(f"Day-1: C0={d1['c0']:.1%}, T_delta={d1['t_delta']:.1%}, "
          f"delta={d1['delta_pp']:+.1f}pp")
    print(f"Final accuracy: C0={fa['c0_mean']:.1%}, T_delta={fa['td_mean']:.1%}, "
          f"delta={fa['delta_pp']:+.1f}pp, p={fa['p_value']:.4f}")
    print()
    print(f"vs v7 W_normalized: d={v7_d:.3f} p={v7_p:.3f} "
          f"→ Δσ: d={abs(m2['cohens_d']):.3f} p={m2['p_value']:.4f}  "
          f"({improvement_str})")
    print()
    print(f"Power at observed d={powa['observed_d']:.3f}:  "
          f"N=100: {powa['power_at_n100']:.1%}  "
          f"N=150: {powa['power_at_n150']:.1%}  "
          f"N=200: {powa['power_at_n200']:.1%}")
    print()
    print(f"Claim status: {claim_status}")
    print("=" * 65)

    return results

if __name__ == "__main__":
    main()
