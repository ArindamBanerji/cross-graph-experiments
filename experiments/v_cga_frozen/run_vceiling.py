"""
V-CEILING — three-arm ceiling experiment (GAE 0.7.5)
=====================================================
Determines whether d≈0.21 is the structural ceiling for enriched bootstrap
or whether d>0.30 is achievable in specific deployment conditions.

ARM A — CD replication at N=200 (control arm, replicates V-BOOTSTRAP-W/v7)
ARM B — v5 A1×B0 scaled to N=200 (low-discrim μ* that achieved d=0.311 at N=100)
ARM C — S2P high-Δσ profile (C=5, A=5, d=8) — CONDITIONAL on W_delta check

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_vceiling.py
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

# ── Shared parameters ──────────────────────────────────────────────────────────
N_SEEDS          = 200
N_BOOTSTRAP      = 1200
N_POST_BOOTSTRAP = 500
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80

# ── SOC domain (Arms A & B): 6 categories × 4 actions × 6 factors ─────────────
SOC_N_CAT     = 6
SOC_N_ACT     = 4
SOC_N_FACTORS = 6

SOC_FACTOR_NAMES = [
    "travel_match",            # dim 0 — enriched in CD
    "asset_criticality",       # dim 1 — fixed
    "threat_intel_enrichment", # dim 2 — NOT enriched in CD
    "time_anomaly",            # dim 3 — fixed
    "pattern_history",         # dim 4 — NOT enriched in CD
    "device_trust",            # dim 5 — enriched in CD
]
SOC_IDX = {f: i for i, f in enumerate(SOC_FACTOR_NAMES)}
CD_ENRICHED_DIMS = [0, 5]   # travel_match, device_trust

SOC_ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
SOC_CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
SOC_CAT_IDX = {c: i for i, c in enumerate(SOC_CATEGORIES)}
SOC_ACT_IDX = {a: i for i, a in enumerate(SOC_ACTIONS)}

# SOC sigma profiles
SOC_SIGMA_AFTER = {
    "travel_match":            0.11,
    "asset_criticality":       0.06,
    "threat_intel_enrichment": 0.18,
    "time_anomaly":            0.07,
    "pattern_history":         0.20,
    "device_trust":            0.09,
}
SOC_SIGMA_BEFORE = {
    "travel_match":            0.27,
    "asset_criticality":       0.09,
    "threat_intel_enrichment": 0.27,
    "time_anomaly":            0.105,
    "pattern_history":         0.30,
    "device_trust":            0.24,
}
_soc_sa = np.array([SOC_SIGMA_AFTER[f]  for f in SOC_FACTOR_NAMES])
_soc_sb = np.array([SOC_SIGMA_BEFORE[f] for f in SOC_FACTOR_NAMES])

# ── Structured A1 SOC μ* (shared by Arms A and B) ─────────────────────────────
_SOC_MU_RAW = {
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

def _build_soc_mu_star(flatten_enrichment: bool) -> np.ndarray:
    """
    flatten_enrichment=False → B1: structured, high-discriminating (Arm A)
    flatten_enrichment=True  → B0: threat_intel(dim2) + pattern_history(dim4)
                                   flattened to 0.50 (Arm B)
    """
    mu = np.full((SOC_N_CAT, SOC_N_ACT, SOC_N_FACTORS), 0.5)
    for (cat, act), vec in _SOC_MU_RAW.items():
        mu[SOC_CAT_IDX[cat], SOC_ACT_IDX[act], :] = vec
    if flatten_enrichment:
        mu[:, :, SOC_IDX["threat_intel_enrichment"]] = 0.50
        mu[:, :, SOC_IDX["pattern_history"]]         = 0.50
    return mu

SOC_MU_STAR_B1 = _build_soc_mu_star(flatten_enrichment=False)  # Arm A
SOC_MU_STAR_B0 = _build_soc_mu_star(flatten_enrichment=True)   # Arm B

def _gt_dist(mu_star, n_cat, n_act):
    gt = np.ones((n_cat, n_act)) * 0.1
    for c in range(n_cat):
        gt[c, int(np.argmax(np.linalg.norm(mu_star[c], axis=-1)))] = 0.7
    gt /= gt.sum(axis=1, keepdims=True)
    return gt

SOC_GT_B1 = _gt_dist(SOC_MU_STAR_B1, SOC_N_CAT, SOC_N_ACT)
SOC_GT_B0 = _gt_dist(SOC_MU_STAR_B0, SOC_N_CAT, SOC_N_ACT)

# ── S2P domain (Arm C): 5 categories × 5 actions × 8 factors ──────────────────
S2P_N_CAT     = 5
S2P_N_ACT     = 5
S2P_N_FACTORS = 8

S2P_FACTOR_NAMES = [
    "supplier_risk",      # dim 0 — fixed
    "logistics_risk",     # dim 1 — enriched
    "demand_risk",        # dim 2 — fixed
    "inventory_risk",     # dim 3 — fixed
    "regulatory_risk",    # dim 4 — enriched
    "geopolitical_risk",  # dim 5 — enriched
    "financial_risk",     # dim 6 — fixed
    "environmental_risk", # dim 7 — enriched
]
S2P_IDX = {f: i for i, f in enumerate(S2P_FACTOR_NAMES)}
S2P_ENRICHED_DIMS = [1, 4, 5, 7]   # logistics, regulatory, geopolitical, environmental
S2P_ENRICHED_SET  = {"logistics_risk","regulatory_risk","geopolitical_risk","environmental_risk"}

S2P_ACTIONS    = ["approve", "escalate", "hold", "reject", "expedite"]
S2P_CATEGORIES = [
    "supplier_risk_cat", "logistics_risk_cat", "demand_risk_cat",
    "financial_risk_cat", "geopolitical_risk_cat",
]
S2P_CAT_IDX = {c: i for i, c in enumerate(S2P_CATEGORIES)}
S2P_ACT_IDX = {a: i for i, a in enumerate(S2P_ACTIONS)}

# S2P sigma profiles
S2P_SIGMA_AFTER = {
    "supplier_risk":      0.08,
    "logistics_risk":     0.14,   # enriched from 0.22
    "demand_risk":        0.12,
    "inventory_risk":     0.10,
    "regulatory_risk":    0.07,   # enriched from 0.10
    "geopolitical_risk":  0.13,   # enriched from 0.20
    "financial_risk":     0.08,
    "environmental_risk": 0.16,   # enriched from 0.25
}
S2P_SIGMA_BEFORE = {
    "supplier_risk":      0.12,   # 0.08 × 1.5
    "logistics_risk":     0.22,   # actual pre-enrichment
    "demand_risk":        0.18,   # 0.12 × 1.5
    "inventory_risk":     0.15,   # 0.10 × 1.5
    "regulatory_risk":    0.10,   # actual pre-enrichment
    "geopolitical_risk":  0.20,   # actual pre-enrichment
    "financial_risk":     0.12,   # 0.08 × 1.5
    "environmental_risk": 0.25,   # actual pre-enrichment
}

def _compute_w_delta(factor_names, sigma_after_dict, sigma_before_dict):
    sa = np.array([sigma_after_dict[f]  for f in factor_names])
    sb = np.array([sigma_before_dict[f] for f in factor_names])
    W_raw  = sb**2 / sa**4
    W_norm = W_raw / W_raw.mean()
    return {f: float(W_norm[i]) for i, f in enumerate(factor_names)}

S2P_W_DELTA = _compute_w_delta(S2P_FACTOR_NAMES, S2P_SIGMA_AFTER, S2P_SIGMA_BEFORE)
_s2p_sa = np.array([S2P_SIGMA_AFTER[f]  for f in S2P_FACTOR_NAMES])
_s2p_sb = np.array([S2P_SIGMA_BEFORE[f] for f in S2P_FACTOR_NAMES])

# ── S2P μ* geometry ────────────────────────────────────────────────────────────
# approve vectors (from spec)
_S2P_APPROVE = {
    "supplier_risk_cat":     [0.20, 0.75, 0.30, 0.35, 0.25, 0.70, 0.30, 0.25],
    "logistics_risk_cat":    [0.20, 0.30, 0.70, 0.35, 0.25, 0.30, 0.30, 0.25],
    "demand_risk_cat":       [0.20, 0.30, 0.30, 0.70, 0.25, 0.30, 0.30, 0.25],
    "financial_risk_cat":    [0.20, 0.30, 0.30, 0.35, 0.25, 0.30, 0.75, 0.25],
    "geopolitical_risk_cat": [0.20, 0.30, 0.30, 0.35, 0.25, 0.75, 0.30, 0.25],
}

def _build_s2p_mu_star() -> np.ndarray:
    """
    Structured S2P μ* tensor shape (S2P_N_CAT, S2P_N_ACT, S2P_N_FACTORS).
    Actions: approve=0, escalate=1, hold=2, reject=3, expedite=4
    """
    mu = np.full((S2P_N_CAT, S2P_N_ACT, S2P_N_FACTORS), 0.5)
    for cat, app_vec in _S2P_APPROVE.items():
        ci = S2P_CAT_IDX[cat]
        av = np.array(app_vec)
        # approve: as specified
        mu[ci, S2P_ACT_IDX["approve"], :] = av
        # escalate: inverted toward 0.80 (1 - v, clipped)
        mu[ci, S2P_ACT_IDX["escalate"], :] = np.clip(1.0 - av, 0.15, 0.80)
        # hold: mid-range 0.45-0.55
        mu[ci, S2P_ACT_IDX["hold"], :] = 0.50
        # reject: low values 0.15-0.25
        mu[ci, S2P_ACT_IDX["reject"], :] = 0.20
        # expedite: high urgency 0.70-0.85 — use approve but shifted up
        exp_vec = np.clip(av + 0.50, 0.70, 0.85)
        exp_vec[av >= 0.50] = 0.80   # dominant factors → 0.80
        exp_vec[av < 0.25]  = 0.75   # subdominant → 0.75
        mu[ci, S2P_ACT_IDX["expedite"], :] = exp_vec
    return mu

S2P_MU_STAR = _build_s2p_mu_star()
S2P_GT_DIST = _gt_dist(S2P_MU_STAR, S2P_N_CAT, S2P_N_ACT)

# ── Domain config shims ────────────────────────────────────────────────────────
class _SOCConfig:
    factor_names = SOC_FACTOR_NAMES
class _S2PConfig:
    factor_names = S2P_FACTOR_NAMES

SOC_DOMAIN_CFG = _SOCConfig()
S2P_DOMAIN_CFG = _S2PConfig()

# ── Utilities ──────────────────────────────────────────────────────────────────
def sample_alert(rng, mu_star, gt_dist, sigma_vec, n_cat, n_act):
    c = int(rng.choice(n_cat))
    a = int(rng.choice(n_act, p=gt_dist[c]))
    f = np.clip(mu_star[c, a] + rng.randn(len(sigma_vec)) * sigma_vec, 0.0, 1.0)
    return c, a, f

def analyst_feedback(rng, pred_a, gt_a, n_act):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(n_act))), True
    return pred_a, False

def standard_bootstrap(historical_decisions, n_cat, n_act, n_factors):
    mu = np.full((n_cat, n_act, n_factors), 0.5, dtype=float)
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

# ── Per-seed simulation (generic) ──────────────────────────────────────────────
def run_one_seed(
    seed, mu_star, gt_dist, sigma_after_dict, sigma_before_dict,
    factor_names, enriched_dims, actions, categories, domain_cfg,
    n_cat, n_act, n_factors,
):
    sa_vec = np.array([sigma_after_dict[f]  for f in factor_names])
    sb_vec = np.array([sigma_before_dict[f] for f in factor_names])

    hist_rng_c0  = np.random.RandomState(seed + 10000)
    hist_rng_t2  = np.random.RandomState(seed + 20000)
    learn_rng_c0 = np.random.RandomState(seed + 30000)
    learn_rng_t2 = np.random.RandomState(seed + 30000)   # identical sequence

    # Historical bootstrap data
    hist_c0 = [sample_alert(hist_rng_c0, mu_star, gt_dist, sb_vec, n_cat, n_act)
               for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(hist_rng_t2, mu_star, gt_dist, sa_vec, n_cat, n_act)
               for _ in range(N_BOOTSTRAP)]

    # μ₀ for each condition
    mu0_c0 = standard_bootstrap(hist_c0, n_cat, n_act, n_factors)
    mu0_t2 = compute_enriched_bootstrap_prior(
        hist_t2, sigma_after_dict, domain_cfg,
        n_cat=n_cat, n_act=n_act, n_factors=n_factors,
        sigma_before=sigma_before_dict,
    )

    # Starting distances
    err_total_c0 = float(np.linalg.norm(mu0_c0 - mu_star))
    err_total_t2 = float(np.linalg.norm(mu0_t2 - mu_star))
    err_enr_c0   = float(np.linalg.norm(
        mu0_c0[:, :, enriched_dims] - mu_star[:, :, enriched_dims]))
    err_enr_t2   = float(np.linalg.norm(
        mu0_t2[:, :, enriched_dims] - mu_star[:, :, enriched_dims]))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, lr in [("C0", mu0_c0, learn_rng_c0), ("T2", mu0_t2, learn_rng_t2)]:
        scorer = ProfileScorer(
            mu0.copy(), actions=actions, categories=categories,
            profile=profile, eta_override=ETA_OVERRIDE,
        )
        # Day-1 accuracy
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, mu_star, gt_dist, sa_vec, n_cat, n_act)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        # Post-bootstrap learning
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, mu_star, gt_dist, sa_vec, n_cat, n_act)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a, n_act)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "err_total": err_total_c0 if cond == "C0" else err_total_t2,
            "err_enr":   err_enr_c0   if cond == "C0" else err_enr_t2,
            "day1_acc":  day1_correct / 50.0,
            "post_accs": post_accs,
            "n_half":    compute_n_half(post_accs),
        }
    return out

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyse(seed_results):
    n = len(seed_results)
    n_half_c0 = np.array([r["C0"]["n_half"] for r in seed_results])
    n_half_t2 = np.array([r["T2"]["n_half"] for r in seed_results])
    diff = n_half_c0 - n_half_t2
    t_stat, p = scipy_stats.ttest_rel(n_half_c0, n_half_t2)
    d   = float(diff.mean() / (diff.std() + 1e-9))
    red = float((n_half_c0.mean() - n_half_t2.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2  = bool(n_half_t2.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3)

    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(),         scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(),    scale=scipy_stats.sem(n_half_c0))
    cit2 = scipy_stats.t.interval(0.95, n-1, loc=n_half_t2.mean(),    scale=scipy_stats.sem(n_half_t2))

    err_enr_c0 = float(np.mean([r["C0"]["err_enr"] for r in seed_results]))
    err_enr_t2 = float(np.mean([r["T2"]["err_enr"] for r in seed_results]))
    m4_pass    = bool(err_enr_t2 < err_enr_c0)
    m4_red     = float((err_enr_c0 - err_enr_t2) / (err_enr_c0 + 1e-9) * 100)

    err_tot_c0 = float(np.mean([r["C0"]["err_total"] for r in seed_results]))
    err_tot_t2 = float(np.mean([r["T2"]["err_total"] for r in seed_results]))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_t2 = np.array([np.array(r["T2"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_t2, fa_c0)

    return {
        "n_half_c0":       round(float(n_half_c0.mean()), 1),
        "n_half_c0_ci95":  [round(ci0[0], 1),  round(ci0[1], 1)],
        "n_half_t2":       round(float(n_half_t2.mean()), 1),
        "n_half_t2_ci95":  [round(cit2[0], 1), round(cit2[1], 1)],
        "diff_mean":       round(float(diff.mean()), 2),
        "diff_ci95":       [round(ci[0], 2), round(ci[1], 2)],
        "reduction_pct":   round(red, 2),
        "p_value":         round(float(p), 6),
        "cohens_d":        round(d, 4),
        "m2_pass":         m2,
        "m4_enriched": {
            "c0_dist":    round(err_enr_c0, 4),
            "t2_dist":    round(err_enr_t2, 4),
            "reduction":  round(m4_red, 2),
            "pass":       m4_pass,
        },
        "m4_total": {
            "c0_dist":    round(err_tot_c0, 4),
            "t2_dist":    round(err_tot_t2, 4),
            "pass":       bool(err_tot_t2 < err_tot_c0),
        },
        "day1_accuracy": {
            "c0":       round(d1_c0, 4),
            "t2":       round(d1_t2, 4),
            "delta_pp": round((d1_t2 - d1_c0) * 100, 2),
        },
        "final_accuracy": {
            "c0":       round(float(fa_c0.mean()), 4),
            "t2":       round(float(fa_t2.mean()), 4),
            "delta_pp": round(float((fa_t2.mean() - fa_c0.mean()) * 100), 2),
            "p_value":  round(float(p_fa), 6),
        },
        "power_at_n200": round(power_at_n(200, abs(d)), 4),
    }

# ── Run one arm ────────────────────────────────────────────────────────────────
def run_arm(label, mu_star, gt_dist, sigma_after_d, sigma_before_d,
            factor_names, enriched_dims, actions, categories, domain_cfg,
            n_cat, n_act, n_factors):
    t0 = time.time()
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(
            seed, mu_star, gt_dist, sigma_after_d, sigma_before_d,
            factor_names, enriched_dims, actions, categories, domain_cfg,
            n_cat, n_act, n_factors,
        ))
        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (seed + 1) / elapsed
            print(f"    Seed {seed+1:3d}/{N_SEEDS}  "
                  f"[{elapsed:.1f}s, ETA {(N_SEEDS-seed-1)/rate:.0f}s]")
    elapsed = time.time() - t0
    print(f"  {label} complete in {elapsed:.1f}s")
    return analyse(all_results), elapsed

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-CEILING (GAE 0.7.5) — three-arm ceiling experiment")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print()

    arms_results = {}

    # ── ARM A: CD replication (A1×B1) ─────────────────────────────────────────
    print("ARM A — CD replication (A1×B1 structured μ*, Δσ-weighted, N=200)")
    stats_a, rt_a = run_arm(
        "Arm A",
        SOC_MU_STAR_B1, SOC_GT_B1,
        SOC_SIGMA_AFTER, SOC_SIGMA_BEFORE,
        SOC_FACTOR_NAMES, CD_ENRICHED_DIMS,
        SOC_ACTIONS, SOC_CATEGORIES, SOC_DOMAIN_CFG,
        SOC_N_CAT, SOC_N_ACT, SOC_N_FACTORS,
    )
    arms_results["A_cd_replication"] = stats_a

    # ── ARM B: A1×B0 scaled to N=200 ──────────────────────────────────────────
    print()
    print("ARM B — A1×B0 at N=200 (flattened dims: threat_intel=0.50, pattern_history=0.50)")
    # Verify flattening
    flat_ok = bool(
        np.all(SOC_MU_STAR_B0[:, :, SOC_IDX["threat_intel_enrichment"]] == 0.50) and
        np.all(SOC_MU_STAR_B0[:, :, SOC_IDX["pattern_history"]] == 0.50)
    )
    print(f"  Flattening verified (dims 2,4 == 0.50): {flat_ok}")
    if not flat_ok:
        print("  ERROR: Flattening not applied. STOP.")
        return

    stats_b, rt_b = run_arm(
        "Arm B",
        SOC_MU_STAR_B0, SOC_GT_B0,
        SOC_SIGMA_AFTER, SOC_SIGMA_BEFORE,
        SOC_FACTOR_NAMES, CD_ENRICHED_DIMS,
        SOC_ACTIONS, SOC_CATEGORIES, SOC_DOMAIN_CFG,
        SOC_N_CAT, SOC_N_ACT, SOC_N_FACTORS,
    )
    arms_results["B_a1b0_scaled"] = stats_b

    # ── ARM C: S2P high-Δσ — W_delta verification first ───────────────────────
    print()
    print("ARM C — S2P high-Δσ: W_delta verification")
    print(f"  {'Factor':<25} {'W_norm':>8}  {'Type'}")
    print(f"  {'-'*50}")
    for f in S2P_FACTOR_NAMES:
        tag = "ENRICHED" if f in S2P_ENRICHED_SET else "fixed"
        print(f"  {f:<25} {S2P_W_DELTA[f]:>8.3f}  {tag}")

    enr_W_vals = [S2P_W_DELTA[f] for f in S2P_FACTOR_NAMES if f in S2P_ENRICHED_SET]
    fix_W_vals = [S2P_W_DELTA[f] for f in S2P_FACTOR_NAMES if f not in S2P_ENRICHED_SET]
    mean_enr = float(np.mean(enr_W_vals))
    mean_fix = float(np.mean(fix_W_vals))
    print(f"\n  Mean W enriched={mean_enr:.3f}, mean W fixed={mean_fix:.3f}")
    print(f"  Min W enriched={min(enr_W_vals):.3f}, max W fixed={max(fix_W_vals):.3f}")

    w_check_pass = bool(min(enr_W_vals) > max(fix_W_vals))   # strict: all enriched > all fixed
    w_check_mean = bool(mean_enr > mean_fix)
    print(f"  All enriched > all fixed: {w_check_pass}")
    print(f"  Mean enriched > mean fixed: {w_check_mean}")

    arm_c_ran   = False
    arm_c_note  = ""
    stats_c     = None

    if not w_check_pass and not w_check_mean:
        arm_c_note = (
            "W_delta check FAILED: enriched factors do NOT dominate fixed factors. "
            "Root cause: small-sigma fixed factors (supplier_risk=0.08, financial_risk=0.08) "
            "produce high W_raw = (1.5·sa)²/sa⁴ = 2.25/sa² >> enriched-factor W at larger sa. "
            "The Δσ scheme upweights low-uncertainty factors regardless of enrichment status "
            "when fixed factors have smaller sa than enriched factors. "
            "Arm C not run per spec (report and STOP)."
        )
        print(f"\n  RESULT: {arm_c_note}")
        arms_results["C_s2p_high_delta"] = {
            "ran": False,
            "w_delta_values": S2P_W_DELTA,
            "w_check_all_enriched_gt_fixed": w_check_pass,
            "w_check_mean_enriched_gt_fixed": w_check_mean,
            "mean_enriched_w": round(mean_enr, 4),
            "mean_fixed_w":    round(mean_fix, 4),
            "note": arm_c_note,
        }
    else:
        print("\n  W_delta check passed. Running Arm C...")
        stats_c, rt_c = run_arm(
            "Arm C",
            S2P_MU_STAR, S2P_GT_DIST,
            S2P_SIGMA_AFTER, S2P_SIGMA_BEFORE,
            S2P_FACTOR_NAMES, S2P_ENRICHED_DIMS,
            S2P_ACTIONS, S2P_CATEGORIES, S2P_DOMAIN_CFG,
            S2P_N_CAT, S2P_N_ACT, S2P_N_FACTORS,
        )
        arm_c_ran = True
        arms_results["C_s2p_high_delta"] = {
            "ran": True,
            "w_delta_values": S2P_W_DELTA,
            **stats_c,
        }

    # ── Gate decision matrix ───────────────────────────────────────────────────
    d_a = abs(stats_a["cohens_d"])
    d_b = abs(stats_b["cohens_d"])

    if arm_c_ran and stats_c is not None:
        d_c = abs(stats_c["cohens_d"])
        if d_a < 0.25 and d_b < 0.25 and d_c < 0.21:
            pattern = "A d≈0.21, B d≈0.21, C d<0.21 — ceiling structural"
            recommendation = "revise gate to d>0.20"
        elif d_a < 0.25 and d_b > 0.30 and d_c < 0.21:
            pattern = "A d≈0.21, B d>0.30, C d<0.21 — d>0.30 in low-discrim SOC only"
            recommendation = "hold gate at d>0.30"
        elif d_a < 0.25 and d_b > 0.30 and d_c > 0.30:
            pattern = "A d≈0.21, B d>0.30, C d>0.30 — d>0.30 in two conditions"
            recommendation = "hold gate at d>0.30"
        else:
            pattern = f"Other pattern: d_A={d_a:.3f}, d_B={d_b:.3f}, d_C={d_c:.3f}"
            recommendation = "other — characterize"
        gate_decision = {
            "arm_a_d": round(d_a, 4),
            "arm_b_d": round(d_b, 4),
            "arm_c_d": round(d_c, 4),
            "arm_c_ran": True,
            "pattern": pattern,
            "recommendation": recommendation,
        }
    else:
        # Arm C did not run — "Any other pattern"
        if d_a < 0.25 and d_b > 0.30:
            pattern = "A d≈0.21, B d>0.30, C not run (W check failed) — low-discrim condition confirmed"
            recommendation = "hold gate at d>0.30; Arm C requires revised σ profile"
        elif d_a < 0.25 and d_b < 0.25:
            pattern = "A d≈0.21, B d≈0.21, C not run (W check failed) — ceiling may be structural"
            recommendation = "revise gate to d>0.20 if B confirms no B0 advantage; Arm C requires revised σ profile"
        else:
            pattern = f"Other: d_A={d_a:.3f}, d_B={d_b:.3f}, C not run (W check failed)"
            recommendation = "other — characterize"
        gate_decision = {
            "arm_a_d": round(d_a, 4),
            "arm_b_d": round(d_b, 4),
            "arm_c_d": None,
            "arm_c_ran": False,
            "arm_c_w_check_failure": arm_c_note,
            "pattern": pattern,
            "recommendation": recommendation,
        }

    # Claim status
    claim_passes = {
        "A": bool(stats_a["m2_pass"]),
        "B": bool(stats_b["m2_pass"]),
    }
    if arm_c_ran and stats_c is not None:
        claim_passes["C"] = bool(stats_c["m2_pass"])

    if all(claim_passes.values()) and arm_c_ran:
        claim_status = "VALIDATED — all three arms pass M2; d>0.30 achievable broadly."
    elif claim_passes.get("B") and not claim_passes["A"]:
        claim_status = "CONDITIONAL — d>0.30 requires low-discriminating B0 geometry; A1×B1 ceiling at d≈0.21."
    elif not any(claim_passes.values()):
        claim_status = "INCONCLUSIVE/REJECTED — no arm clears M2 gate."
    else:
        n_pass = sum(claim_passes.values())
        claim_status = f"PARTIAL — {n_pass}/{len(claim_passes)} arms pass M2 gate."

    # ── Build results dict ─────────────────────────────────────────────────────
    def _arm_summary(stats, description, sigma_after_d, sigma_before_d):
        return {
            "description": description,
            "n_half_c0":       stats["n_half_c0"],
            "n_half_t2":       stats["n_half_t2"],
            "d":               round(abs(stats["cohens_d"]), 4),
            "p":               stats["p_value"],
            "ci_95":           stats["diff_ci95"],
            "m4_enriched_pass": stats["m4_enriched"]["pass"],
            "day1_delta_pp":   stats["day1_accuracy"]["delta_pp"],
            "sigma_after":     sigma_after_d,
            "sigma_before":    sigma_before_d,
        }

    results = {
        "experiment":       "V-CEILING",
        "gae_version":      "0.7.5",
        "date":             "2026-03-23",
        "n_seeds_per_arm":  N_SEEDS,
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
        "arms": {
            "A_cd_replication": _arm_summary(
                stats_a,
                "CD condition replication — confirms v7/V-BOOTSTRAP-W stability",
                SOC_SIGMA_AFTER, SOC_SIGMA_BEFORE,
            ),
            "B_a1b0_scaled": _arm_summary(
                stats_b,
                "v5 A1×B0 at N=200 — low-discrim structured mu* (threat_intel+pattern_history flattened)",
                SOC_SIGMA_AFTER, SOC_SIGMA_BEFORE,
            ),
            "C_s2p_high_delta": arms_results["C_s2p_high_delta"],
        },
        "full_stats": {
            "A": stats_a,
            "B": stats_b,
        },
        "gate_decision": gate_decision,
        "claim_status":  claim_status,
    }
    if arm_c_ran and stats_c is not None:
        results["full_stats"]["C"] = stats_c

    # ── Save ───────────────────────────────────────────────────────────────────
    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    out_dir    = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / "results_vceiling.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEnc)
    print(f"\nResults saved to {out_path}")

    # ── Print verdict ──────────────────────────────────────────────────────────
    def _pf(b): return "PASS" if b else "FAIL"
    def _fmt_arm(label, stats, enriched_dims_label):
        d   = abs(stats["cohens_d"])
        p   = stats["p_value"]
        ci  = stats["diff_ci95"]
        m4  = stats["m4_enriched"]["pass"]
        d1  = stats["day1_accuracy"]["delta_pp"]
        m2f = _pf(stats["m2_pass"])
        return (f"{label}: d={d:.3f} p={p:.4f} CI=[{ci[0]:.2f},{ci[1]:.2f}] "
                f"M4_enr({'PASS' if m4 else 'FAIL'}) Day1={d1:+.1f}pp  [{m2f}]")

    print()
    print("=" * 65)
    print("V-CEILING Results:")
    print("=" * 65)
    print(_fmt_arm("Arm A (CD replication)  ", stats_a, "dims 0,5"))
    print(_fmt_arm("Arm B (A1×B0 at N=200)  ", stats_b, "dims 0,5"))
    if arm_c_ran and stats_c is not None:
        print(_fmt_arm("Arm C (S2P high-Δσ)     ", stats_c, "dims 1,4,5,7"))
    else:
        print("Arm C (S2P high-Δσ):     NOT RUN — W_delta check failed")
        print(f"  Root cause: mean W enriched={mean_enr:.3f} < mean W fixed={mean_fix:.3f}")
        print(f"  Fixed factors supplier/financial (sa=0.08) dominate via 1/sa⁴")

    print()
    print(f"Gate decision matrix match: {gate_decision['pattern']}")
    print(f"Recommendation: {gate_decision['recommendation']}")
    print()
    print(f"CLAIM-62/63 status: {claim_status}")
    print("=" * 65)

if __name__ == "__main__":
    main()
