"""
V-CGA-FROZEN v7 spot check — GAE 0.7.8 gradient fix
=====================================================
Re-runs V-CGA-FROZEN v7 (A1×B1 B1-ceiling check) under GAE 0.7.8 with
the fixed DiagonalKernel gradient to confirm the B1 ceiling still holds.

Previous result (GAE 0.7.6, bugged): d=0.212, p=0.003
Expected: d should remain in the 0.21-0.23 band (ceiling structural, not bug artifact).

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_v7_gae078.py
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
    "travel_match",            # dim 0 — enriched in CD
    "asset_criticality",       # dim 1 — fixed
    "threat_intel_enrichment", # dim 2 — enriched in B1
    "time_anomaly",            # dim 3 — fixed
    "pattern_history",         # dim 4 — enriched in B1
    "device_trust",            # dim 5 — enriched in CD
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
CD_ENRICHED_DIMS = [0, 5]   # travel_match, device_trust — M4 dims (same as v7)

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Sigma profiles (identical to V-CGA-FROZEN v7) ───────────────────────────────
SIGMA_AFTER = {
    "travel_match":            0.11,
    "asset_criticality":       0.06,
    "threat_intel_enrichment": 0.18,
    "time_anomaly":            0.07,
    "pattern_history":         0.20,
    "device_trust":            0.09,
}
SIGMA_BEFORE = {
    "travel_match":            0.27,
    "asset_criticality":       0.09,
    "threat_intel_enrichment": 0.27,
    "time_anomaly":            0.105,
    "pattern_history":         0.30,
    "device_trust":            0.24,
}

SV_AFTER  = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])
SV_BEFORE = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])

# W_delta = sigma_before² / sigma_after⁴
_W_raw  = SV_BEFORE**2 / SV_AFTER**4
_W_norm = _W_raw / _W_raw.mean()
W_DELTA_VALUES = {f: round(float(_W_norm[i]), 4) for i, f in enumerate(FACTOR_NAMES)}

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── Structured A1×B1 μ* (NO flattening) ────────────────────────────────────────
# threat_intel_enrichment (dim 2) and pattern_history (dim 4) are kept at their
# discriminating values — this IS the B1 high-discriminating geometry.
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

def _build_mu_star_b1() -> np.ndarray:
    """A1×B1: full structured geometry, no flattening."""
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    # B1: dims 2 and 4 KEPT at discriminating values (no flattening)
    return mu

MU_STAR = _build_mu_star_b1()

# Verify B1 — threat_intel and pattern_history must NOT be uniformly 0.50
assert not np.all(MU_STAR[:, :, IDX["threat_intel_enrichment"]] == 0.50), \
    "B1 check: threat_intel should have structure"
assert not np.all(MU_STAR[:, :, IDX["pattern_history"]] == 0.50), \
    "B1 check: pattern_history should have structure"

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

def power_at_n(n, d, alpha=0.01):
    df = n - 1
    nc = d * np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc))

# ── Per-seed simulation ─────────────────────────────────────────────────────────
def run_one_seed(seed: int) -> dict:
    hist_rng_c0 = np.random.RandomState(seed + 10000)
    hist_rng_t2 = np.random.RandomState(seed + 20000)
    learn_rng   = np.random.RandomState(seed + 30000)

    hist_c0 = [sample_alert(hist_rng_c0, SV_BEFORE) for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(hist_rng_t2, SV_AFTER)  for _ in range(N_BOOTSTRAP)]

    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_t2 = compute_enriched_bootstrap_prior(
        hist_t2, SIGMA_AFTER, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=SIGMA_BEFORE,
    )

    # M4: enriched dims only (travel_match=0, device_trust=5)
    err_enr_c0 = float(np.linalg.norm(
        mu0_c0[:, :, CD_ENRICHED_DIMS] - MU_STAR[:, :, CD_ENRICHED_DIMS]))
    err_enr_t2 = float(np.linalg.norm(
        mu0_t2[:, :, CD_ENRICHED_DIMS] - MU_STAR[:, :, CD_ENRICHED_DIMS]))
    err_tot_c0 = float(np.linalg.norm(mu0_c0 - MU_STAR))
    err_tot_t2 = float(np.linalg.norm(mu0_t2 - MU_STAR))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0 in [("C0", mu0_c0), ("T2", mu0_t2)]:
        lr = np.random.RandomState(seed + 30000)   # same sequence for both
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, SV_AFTER)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, SV_AFTER)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "err_enr":   err_enr_c0 if cond == "C0" else err_enr_t2,
            "err_total": err_tot_c0 if cond == "C0" else err_tot_t2,
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
    d   = float(diff.mean() / (diff.std() + 1e-9))
    red = float((n_half_c0.mean() - n_half_t2.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2  = bool(n_half_t2.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3)

    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(),      scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    cit2 = scipy_stats.t.interval(0.95, n-1, loc=n_half_t2.mean(), scale=scipy_stats.sem(n_half_t2))

    err_enr_c0 = float(np.mean([r["C0"]["err_enr"] for r in seed_results]))
    err_enr_t2 = float(np.mean([r["T2"]["err_enr"] for r in seed_results]))
    m4_pass    = bool(err_enr_t2 < err_enr_c0)

    err_tot_c0 = float(np.mean([r["C0"]["err_total"] for r in seed_results]))
    err_tot_t2 = float(np.mean([r["T2"]["err_total"] for r in seed_results]))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_t2 = np.array([np.array(r["T2"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_t2, fa_c0)

    obs_d = abs(d)
    return {
        "m2": {
            "n_half_c0":      round(float(n_half_c0.mean()), 1),
            "n_half_c0_ci95": [round(ci0[0], 1),  round(ci0[1], 1)],
            "n_half_t2":      round(float(n_half_t2.mean()), 1),
            "n_half_t2_ci95": [round(cit2[0], 1), round(cit2[1], 1)],
            "diff_mean":      round(float(diff.mean()), 2),
            "diff_ci95":      [round(ci[0], 2), round(ci[1], 2)],
            "reduction_pct":  round(red, 2),
            "p_value":        round(float(p), 6),
            "t_stat":         round(float(t_stat), 4),
            "cohens_d":       round(d, 4),
            "pass":           m2,
        },
        "m4_enriched": {
            "c0_dist":        round(err_enr_c0, 4),
            "t2_dist":        round(err_enr_t2, 4),
            "reduction_pct":  round(float((err_enr_c0 - err_enr_t2) / (err_enr_c0 + 1e-9) * 100), 2),
            "pass":           m4_pass,
        },
        "m4_total": {
            "c0_dist":        round(err_tot_c0, 4),
            "t2_dist":        round(err_tot_t2, 4),
            "pass":           bool(err_tot_t2 < err_tot_c0),
        },
        "day1_accuracy": {
            "c0":             round(d1_c0, 4),
            "t2":             round(d1_t2, 4),
            "delta_pp":       round((d1_t2 - d1_c0) * 100, 2),
        },
        "final_accuracy": {
            "c0_mean":        round(float(fa_c0.mean()), 4),
            "t2_mean":        round(float(fa_t2.mean()), 4),
            "delta_pp":       round(float((fa_t2.mean() - fa_c0.mean()) * 100), 2),
            "p_value":        round(float(p_fa), 6),
        },
        "power_analysis": {
            "observed_d":     round(obs_d, 4),
            "power_at_n200":  round(power_at_n(200, obs_d), 4),
        },
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-CGA-FROZEN v7 spot check (GAE 0.7.8, gradient fix)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"Condition: A1×B1 CD cross-domain B1 ceiling check")
    print(f"Previous (GAE 0.7.6 bugged): d=0.212, p=0.003")
    print()
    print("W_delta values (sigma_before²/sigma_after⁴, normalized):")
    for f, w in W_DELTA_VALUES.items():
        print(f"  {f:<30} {w:.4f}")
    print()

    t0 = time.time()
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(seed))
        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (seed + 1) / elapsed
            print(f"  Seed {seed+1:3d}/{N_SEEDS}  "
                  f"[{elapsed:.1f}s, ETA {(N_SEEDS-seed-1)/rate:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    stats = analyse(all_results)
    m2    = stats["m2"]
    m4    = stats["m4_enriched"]
    d1    = stats["day1_accuracy"]
    fa    = stats["final_accuracy"]
    powa  = stats["power_analysis"]

    v7_d   = 0.212
    v7_p   = 0.003
    new_d  = abs(m2["cohens_d"])
    new_p  = m2["p_value"]

    # B1 ceiling status
    if new_d <= 0.25:
        ceiling_status   = "CONFIRMED"
        two_regime_line  = "UNCHANGED"
    else:
        ceiling_status   = f"SHIFTED to d={new_d:.3f}"
        two_regime_line  = "needs re-evaluation"

    # ── Save ─────────────────────────────────────────────────────────────────────
    results = {
        "experiment":  "V-CGA-FROZEN-v7-GAE-0.7.8",
        "version":     "spot_check_gradient_fix",
        "gae_version": "0.7.8",
        "date":        "2026-03-24",
        "n_seeds":     N_SEEDS,
        "condition":   "A1xB1 CD cross-domain B1 ceiling check",
        "sigma_after":  SIGMA_AFTER,
        "sigma_before": SIGMA_BEFORE,
        "w_delta_values": W_DELTA_VALUES,
        "previous_result": {
            "d": v7_d,
            "p": v7_p,
            "cause_of_rerun": "DiagonalKernel gradient bug",
        },
        "m2": {
            "n_half_c0":    m2["n_half_c0"],
            "n_half_t2":    m2["n_half_t2"],
            "reduction_pct": m2["reduction_pct"],
            "p":            new_p,
            "cohens_d":     round(m2["cohens_d"], 4),
            "ci_95":        m2["diff_ci95"],
            "pass":         m2["pass"],
        },
        "m4_enriched": {
            "c0": m4["c0_dist"],
            "t2": m4["t2_dist"],
            "pass": m4["pass"],
        },
        "day1_accuracy": d1,
        "final_accuracy": fa,
        "power_analysis": powa,
        "comparison_to_v7": {
            "v7_d":               v7_d,
            "v7_p":               v7_p,
            "new_d":              round(new_d, 4),
            "new_p":              round(new_p, 6),
            "b1_ceiling_status":  ceiling_status,
        },
        "runtime_s": round(elapsed_total, 1),
    }

    out_dir  = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_v7_gae078.json"

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

    # ── Print verdict ─────────────────────────────────────────────────────────────
    def _pf(b): return "PASS" if b else "FAIL"

    print()
    print("=" * 65)
    print("V-CGA-FROZEN v7 spot check (GAE 0.7.8): B1 ceiling status")
    print("=" * 65)
    print(f"M2: C0={m2['n_half_c0']:.1f}, T2={m2['n_half_t2']:.1f}, "
          f"reduction={m2['reduction_pct']:.1f}%, "
          f"p={new_p:.4f}, d={new_d:.3f} [{_pf(m2['pass'])}]")
    print(f"M4_enriched (dims 0,5): C0={m4['c0_dist']:.3f}, "
          f"T2={m4['t2_dist']:.3f} [{_pf(m4['pass'])}]")
    print(f"Day-1: C0={d1['c0']:.1%}, T2={d1['t2']:.1%}, "
          f"delta={d1['delta_pp']:+.1f}pp")
    print(f"Final: C0={fa['c0_mean']:.1%}, T2={fa['t2_mean']:.1%}, "
          f"delta={fa['delta_pp']:+.1f}pp")
    print()
    print(f"vs v7 (bugged): d={v7_d:.3f} p={v7_p:.3f} "
          f"\u2192 fixed: d={new_d:.3f} p={new_p:.4f}")
    print()
    print(f"B1 ceiling: [{ceiling_status}]")
    print(f"Two-regime structure: [{two_regime_line}]")
    print("Raw numbers for roadmap session review.")
    print("=" * 65)

if __name__ == "__main__":
    main()
