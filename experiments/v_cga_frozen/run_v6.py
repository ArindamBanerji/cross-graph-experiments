"""
V-CGA-FROZEN v6 — Option 2 (cross-domain enrichment) and Option 3 (dampened weights)
=======================================================================================
Both conditions use A1×B1 structured mu* geometry (high-discriminating enriched
factors) — the realistic SOC geometry that FAILED in v5 (d=0.120, p=0.234).

CD — Cross-Domain enrichment (Option 2):
  Enrich ONLY non-discriminating factors: device_trust 0.16→0.09, travel_match 0.18→0.11
  threat_intel and pattern_history sigma UNCHANGED (not enriched).
  Hypothesis: enriching non-discriminating factors avoids overcorrection.

DW — Dampened Weighting (Option 3):
  Same A1×B1 enrichment (threat_intel, pattern_history) as v5.
  BUT weight_exponent=0.5: W=(1/σ²)^0.5=sqrt(1/σ²) instead of 1/σ².
  Hypothesis: dampening reduces overcorrection while preserving directional benefit.

GAE change: weight_exponent param added to compute_enriched_bootstrap_prior()
  in gae/calibration.py (default=1.0 — backward compatible, 490 tests pass).

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_v6.py
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

# ── Parameters (identical to v5 — committed) ──────────────────────────────────
N_SEEDS          = 100
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
    "travel_match",            # idx 0
    "asset_criticality",       # idx 1
    "threat_intel_enrichment", # idx 2
    "time_anomaly",            # idx 3
    "pattern_history",         # idx 4
    "device_trust",            # idx 5
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ENRICHMENT_FACTORS = ["threat_intel_enrichment", "pattern_history", "device_trust"]

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Structured A1×B1 mu* (identical to v5) ────────────────────────────────────
_MU_STAR_A1B1_RAW = {
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

def _build_mu_star() -> np.ndarray:
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_A1B1_RAW.items():
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

# ── Condition sigma profiles ───────────────────────────────────────────────────

# CD — Cross-Domain: enrich ONLY device_trust and travel_match
SIGMA_CD_ENRICHED = {
    "travel_match":            0.11,   # enriched (was 0.18)
    "asset_criticality":       0.06,   # fixed
    "threat_intel_enrichment": 0.18,   # UNCHANGED — not enriched in CD
    "time_anomaly":            0.07,   # fixed
    "pattern_history":         0.20,   # UNCHANGED — not enriched in CD
    "device_trust":            0.09,   # enriched (was 0.16)
}
# C0 for CD: un-enriched x1.5 on the two enriched factors only
_CD_ENRICHED_FACTORS = ["travel_match", "device_trust"]
SIGMA_CD_UNENRICHED = {
    name: (SIGMA_CD_ENRICHED[name] * 1.5 if name in _CD_ENRICHED_FACTORS
           else SIGMA_CD_ENRICHED[name])
    for name in FACTOR_NAMES
}

# DW — Dampened Weighting: same enrichment as v5 A1×B1, weight_exponent=0.5
SIGMA_DW_ENRICHED = {
    "threat_intel_enrichment": 0.13,
    "pattern_history":         0.10,
    "device_trust":            0.11,
    "travel_match":            0.18,
    "asset_criticality":       0.06,
    "time_anomaly":            0.07,
}
_DW_ENRICHED_FACTORS = ["threat_intel_enrichment", "pattern_history", "device_trust"]
SIGMA_DW_UNENRICHED = {
    name: (SIGMA_DW_ENRICHED[name] * 1.5 if name in _DW_ENRICHED_FACTORS
           else SIGMA_DW_ENRICHED[name])
    for name in FACTOR_NAMES
}
DW_WEIGHT_EXPONENT = 0.5

def _sv(d): return np.array([d[f] for f in FACTOR_NAMES])

# ── Domain config ──────────────────────────────────────────────────────────────
class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── Shared utilities ───────────────────────────────────────────────────────────
def sample_alert(rng, sigma_vec):
    c = int(rng.choice(N_CATEGORIES))
    a = int(rng.choice(N_ACTIONS, p=GT_DIST[c]))
    f = np.clip(MU_STAR[c, a] + rng.randn(N_FACTORS) * sigma_vec, 0.0, 1.0)
    return c, a, f

def analyst_feedback(rng, pred_a, gt_a):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(N_ACTIONS))), True
    return pred_a, False

def standard_bootstrap(historical_decisions) -> np.ndarray:
    """Unweighted ETA-update bootstrap (C0)."""
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

# ── Per-seed simulation ────────────────────────────────────────────────────────
def run_one_seed(seed: int,
                 sigma_enriched: dict, sigma_unenriched: dict,
                 weight_exponent: float) -> dict:
    """
    Run C0 (standard bootstrap) and C1 (enriched bootstrap) for one seed.
    mu* and GT distribution are fixed (A1×B1 structured geometry).
    Seed isolation identical to v5.
    """
    hist_rng_c0  = np.random.RandomState(seed + 10000)
    hist_rng_c1  = np.random.RandomState(seed + 20000)
    learn_rng_c0 = np.random.RandomState(seed + 30000)
    learn_rng_c1 = np.random.RandomState(seed + 30000)   # identical sequence

    sv_enriched   = _sv(sigma_enriched)
    sv_unenriched = _sv(sigma_unenriched)

    # Historical decisions
    hist_c0 = [sample_alert(hist_rng_c0, sv_unenriched) for _ in range(N_BOOTSTRAP_HIST)]
    hist_c1 = [sample_alert(hist_rng_c1, sv_enriched)   for _ in range(N_BOOTSTRAP_HIST)]

    # mu_0
    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_c1 = compute_enriched_bootstrap_prior(
        hist_c1, sigma_enriched, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        weight_exponent=weight_exponent,
    )

    err_c0 = float(np.linalg.norm(mu0_c0 - MU_STAR))
    err_c1 = float(np.linalg.norm(mu0_c1 - MU_STAR))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, lr in [("C0", mu0_c0, learn_rng_c0), ("C1", mu0_c1, learn_rng_c1)]:
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        # Day-1 accuracy — same 50-alert probe per seed for both conditions
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, sv_enriched)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        # Post-bootstrap learning
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, sv_enriched)
            res = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "starting_error": err_c0 if cond == "C0" else err_c1,
            "day1_acc":       day1_correct / 50.0,
            "post_accs":      post_accs,
            "n_half":         compute_n_half(post_accs),
        }

    return out

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyse(seed_results: list) -> dict:
    n = len(seed_results)
    n_half_c0 = np.array([r["C0"]["n_half"] for r in seed_results])
    n_half_c1 = np.array([r["C1"]["n_half"] for r in seed_results])
    diff = n_half_c0 - n_half_c1
    t, p = scipy_stats.ttest_rel(n_half_c0, n_half_c1)
    d    = float(diff.mean() / (diff.std() + 1e-9))
    red  = float((n_half_c0.mean() - n_half_c1.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2   = n_half_c1.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3

    ci0 = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    ci1 = scipy_stats.t.interval(0.95, n-1, loc=n_half_c1.mean(), scale=scipy_stats.sem(n_half_c1))

    err_c0 = np.mean([r["C0"]["starting_error"] for r in seed_results])
    err_c1 = np.mean([r["C1"]["starting_error"] for r in seed_results])
    m4     = bool(err_c1 < err_c0)

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_c1 = float(np.mean([r["C1"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_c1 = np.array([np.array(r["C1"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_c1, fa_c0)

    return {
        "n_half_c0":     round(float(n_half_c0.mean()), 1),
        "n_half_c0_ci95":[round(ci0[0], 1), round(ci0[1], 1)],
        "n_half_c1":     round(float(n_half_c1.mean()), 1),
        "n_half_c1_ci95":[round(ci1[0], 1), round(ci1[1], 1)],
        "reduction_pct": round(red, 2),
        "p_value":       round(float(p), 6),
        "cohens_d":      round(d, 4),
        "m2_pass":       bool(m2),
        "m4": {
            "starting_error_c0": round(float(err_c0), 4),
            "starting_error_c1": round(float(err_c1), 4),
            "reduction_pct":     round(float((err_c0-err_c1)/(err_c0+1e-9)*100), 2),
            "pass":              m4,
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
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-CGA-FROZEN v6 — Option 2 (CD) and Option 3 (DW)")
    print("Both use A1×B1 structured mu* (v5 failing geometry)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP_HIST={N_BOOTSTRAP_HIST}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print()

    t0 = time.time()

    # ── CD: Cross-Domain enrichment ────────────────────────────────────────────
    print("Running CD (cross-domain enrichment — Option 2) ...")
    print(f"  Enriched: travel_match=0.18→0.11, device_trust=0.16→0.09")
    print(f"  Unchanged: threat_intel=0.18, pattern_history=0.20")
    cd_results = []
    for seed in range(N_SEEDS):
        cd_results.append(run_one_seed(
            seed, SIGMA_CD_ENRICHED, SIGMA_CD_UNENRICHED, weight_exponent=1.0))
    cd_stats = analyse(cd_results)
    elapsed = time.time() - t0
    print(f"  done [{elapsed:.1f}s]  N_half C0={cd_stats['n_half_c0']:.1f} "
          f"C1={cd_stats['n_half_c1']:.1f}  d={cd_stats['cohens_d']:.3f}  "
          f"p={cd_stats['p_value']:.4f}  "
          f"M2={'PASS' if cd_stats['m2_pass'] else 'fail'}  "
          f"M4={'PASS' if cd_stats['m4']['pass'] else 'fail'}")
    print()

    # ── DW: Dampened Weighting ─────────────────────────────────────────────────
    print("Running DW (dampened weighting, exp=0.5 — Option 3) ...")
    print(f"  Same sigma as v5 A1×B1; weight_exponent={DW_WEIGHT_EXPONENT}")
    dw_results = []
    for seed in range(N_SEEDS):
        dw_results.append(run_one_seed(
            seed, SIGMA_DW_ENRICHED, SIGMA_DW_UNENRICHED,
            weight_exponent=DW_WEIGHT_EXPONENT))
    dw_stats = analyse(dw_results)
    elapsed = time.time() - t0
    print(f"  done [{elapsed:.1f}s]  N_half C0={dw_stats['n_half_c0']:.1f} "
          f"C1={dw_stats['n_half_c1']:.1f}  d={dw_stats['cohens_d']:.3f}  "
          f"p={dw_stats['p_value']:.4f}  "
          f"M2={'PASS' if dw_stats['m2_pass'] else 'fail'}  "
          f"M4={'PASS' if dw_stats['m4']['pass'] else 'fail'}")

    elapsed_total = time.time() - t0
    print(f"\nAll conditions complete in {elapsed_total:.1f}s")

    # ── Verdicts ───────────────────────────────────────────────────────────────
    opt2 = "VALIDATED" if cd_stats["m2_pass"] else "REJECTED"
    opt3 = "VALIDATED" if dw_stats["m2_pass"] else "REJECTED"

    if cd_stats["m2_pass"] or dw_stats["m2_pass"]:
        claim_rec = ("UNCONDITIONAL — validated in A1×B1 geometry via "
                     + ("cross-domain enrichment (Option 2)" if cd_stats["m2_pass"] else "")
                     + (" and " if cd_stats["m2_pass"] and dw_stats["m2_pass"] else "")
                     + ("dampened weighting (Option 3)" if dw_stats["m2_pass"] else "")
                     + ".")
    else:
        claim_rec = ("RETIRE — enriched bootstrap prior provides no measurable N_half "
                     "benefit in realistic A1×B1 SOC geometry under any tested weighting "
                     "scheme or enrichment strategy. Effect is confined to the narrow "
                     "A1×B0 regime (structured mu*, non-discriminating enriched factors).")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "experiment": "V-CGA-FROZEN-v6",
        "version":    "v6_option2_option3",
        "date":       "2026-03-23",
        "n_seeds":    N_SEEDS,
        "runtime_s":  round(elapsed_total, 1),
        "gae_change": {
            "param_added":  "weight_exponent: float = 1.0",
            "function":     "compute_enriched_bootstrap_prior() in gae/calibration.py",
            "tests_passing": 490,
            "backward_compatible": True,
        },
        "mu_star_geometry": "A1×B1 structured SOC (v5 A1_B1 — high-discriminating)",
        "parameters": {
            "n_bootstrap_hist": N_BOOTSTRAP_HIST,
            "n_post_bootstrap": N_POST_BOOTSTRAP,
            "theta_min": THETA_MIN, "tau": TAU,
            "eta_confirm": ETA_CONFIRM, "eta_override": ETA_OVERRIDE,
            "q_bar": Q_BAR, "alpha": ALPHA,
        },
        "conditions": {
            "CD_cross_domain": {
                "description":   "Enrich non-discriminating factors only (travel_match, device_trust)",
                "weight_exponent": 1.0,
                "sigma_enriched": SIGMA_CD_ENRICHED,
                "sigma_unenriched": SIGMA_CD_UNENRICHED,
                "n_half_c0":     cd_stats["n_half_c0"],
                "n_half_c0_ci95": cd_stats["n_half_c0_ci95"],
                "n_half_c1":     cd_stats["n_half_c1"],
                "n_half_c1_ci95": cd_stats["n_half_c1_ci95"],
                "reduction_pct": cd_stats["reduction_pct"],
                "p":             cd_stats["p_value"],
                "d":             cd_stats["cohens_d"],
                "m2_pass":       cd_stats["m2_pass"],
                "m4_pass":       cd_stats["m4"]["pass"],
                "m4_err_c0":     cd_stats["m4"]["starting_error_c0"],
                "m4_err_c1":     cd_stats["m4"]["starting_error_c1"],
                "day1_delta_pp": cd_stats["day1_accuracy"]["delta_pp"],
                "final_acc_c0":  cd_stats["final_accuracy"]["c0_mean"],
                "final_acc_c1":  cd_stats["final_accuracy"]["c1_mean"],
            },
            "DW_dampened_weights": {
                "description":   "sqrt(1/sigma^2) weighting (exponent=0.5) — same enriched dims as v5",
                "weight_exponent": DW_WEIGHT_EXPONENT,
                "sigma_enriched": SIGMA_DW_ENRICHED,
                "sigma_unenriched": SIGMA_DW_UNENRICHED,
                "n_half_c0":     dw_stats["n_half_c0"],
                "n_half_c0_ci95": dw_stats["n_half_c0_ci95"],
                "n_half_c1":     dw_stats["n_half_c1"],
                "n_half_c1_ci95": dw_stats["n_half_c1_ci95"],
                "reduction_pct": dw_stats["reduction_pct"],
                "p":             dw_stats["p_value"],
                "d":             dw_stats["cohens_d"],
                "m2_pass":       dw_stats["m2_pass"],
                "m4_pass":       dw_stats["m4"]["pass"],
                "m4_err_c0":     dw_stats["m4"]["starting_error_c0"],
                "m4_err_c1":     dw_stats["m4"]["starting_error_c1"],
                "day1_delta_pp": dw_stats["day1_accuracy"]["delta_pp"],
                "final_acc_c0":  dw_stats["final_accuracy"]["c0_mean"],
                "final_acc_c1":  dw_stats["final_accuracy"]["c1_mean"],
            },
        },
        "option2_verdict":      opt2,
        "option3_verdict":      opt3,
        "claim_recommendation": claim_rec,
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v6.json"

    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEnc)
    print(f"\nResults saved to {results_path}")

    # ── Print verdict ──────────────────────────────────────────────────────────
    def _pass(b): return "PASS" if b else "FAIL"

    print()
    print("=" * 65)
    cd = cd_stats; cd_m4 = cd["m4"]; cd_d1 = cd["day1_accuracy"]
    dw = dw_stats; dw_m4 = dw["m4"]; dw_d1 = dw["day1_accuracy"]

    print(f"V-CGA-FROZEN v6 — Option 2 (cross-domain enrichment): {_pass(cd['m2_pass'])}")
    print(f"  CD: N_half C0={cd['n_half_c0']:.1f}, C1={cd['n_half_c1']:.1f}, "
          f"reduction={cd['reduction_pct']:.1f}%, p={cd['p_value']:.4f}, "
          f"d={cd['cohens_d']:.3f}")
    print(f"  M4: {_pass(cd_m4['pass'])} "
          f"(C0_err={cd_m4['starting_error_c0']:.3f}, C1_err={cd_m4['starting_error_c1']:.3f}, "
          f"red={cd_m4['reduction_pct']:.1f}%), "
          f"Day-1: {cd_d1['delta_pp']:+.1f}pp")
    print()
    print(f"V-CGA-FROZEN v6 — Option 3 (dampened weighting): {_pass(dw['m2_pass'])}")
    print(f"  DW: N_half C0={dw['n_half_c0']:.1f}, C1={dw['n_half_c1']:.1f}, "
          f"reduction={dw['reduction_pct']:.1f}%, p={dw['p_value']:.4f}, "
          f"d={dw['cohens_d']:.3f}")
    print(f"  M4: {_pass(dw_m4['pass'])} "
          f"(C0_err={dw_m4['starting_error_c0']:.3f}, C1_err={dw_m4['starting_error_c1']:.3f}, "
          f"red={dw_m4['reduction_pct']:.1f}%), "
          f"Day-1: {dw_d1['delta_pp']:+.1f}pp")
    print()
    print(f"Option 2 verdict: {opt2}" + (
        " — cross-domain enrichment works in B1 geometry" if opt2 == "VALIDATED" else
        " — cross-domain enrichment does not clear M2 gate"))
    print(f"Option 3 verdict: {opt3}" + (
        " — dampened weighting fixes overcorrection" if opt3 == "VALIDATED" else
        " — dampened weighting does not clear M2 gate"))
    print()
    print(f"Claim recommendation: {claim_rec}")
    print("=" * 65)

    return results

if __name__ == "__main__":
    main()
