"""
V-CLAIM62-B0-P2 — B0 regime, MDM enrichment, device_trust non-primary (GAE 0.7.8)
====================================================================================
Tests whether Δσ enrichment of a NON-primary factor (device_trust) still produces
d>0.28 convergence acceleration in the A1×B0 topology where threat_intel and
pattern_history remain the high-discriminating dimensions.

B0 condition: device_trust FLATTENED to 0.50 in μ* (low-discriminating).
              threat_intel and pattern_history KEPT as primary (high-spread in μ*).
Enrichment:   device_trust only. sigma_before=0.240, sigma_after=0.120.
All other factors: sigma_before = sigma_after (unchanged).

C0: standard bootstrap (SV_BEFORE), no enrichment
T2: Δσ-weighted bootstrap (SV_AFTER data, sigma_before provided)

Gates:
  M2: d > 0.28, p < 0.01 (on n_half convergence speed)
  M4: W_boot_device_trust highest among all factors (verified pre-run)
  Day-1: T2 > C0 (directional)

Run:
    PYTHONUTF8=1 python experiments/v_claim62_b0/b0_p2_mdm_device_trust/run.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import t as t_dist, nct

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
    "travel_match",            # dim 0
    "asset_criticality",       # dim 1
    "threat_intel_enrichment", # dim 2 — primary (NOT flattened in B0-P2)
    "time_anomaly",            # dim 3
    "pattern_history",         # dim 4 — primary (NOT flattened in B0-P2)
    "device_trust",            # dim 5 — non-primary, enriched (B0 condition)
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Sigma profiles ──────────────────────────────────────────────────────────────
# Only device_trust changes; all other factors: sigma_before = sigma_after
SIGMA_AFTER = {
    "travel_match":            0.165,
    "asset_criticality":       0.060,
    "threat_intel_enrichment": 0.090,
    "time_anomaly":            0.070,
    "pattern_history":         0.100,
    "device_trust":            0.120,   # enriched (from 0.240)
}
SIGMA_BEFORE = {
    "travel_match":            0.165,   # unchanged
    "asset_criticality":       0.060,   # unchanged
    "threat_intel_enrichment": 0.090,   # unchanged
    "time_anomaly":            0.070,   # unchanged
    "pattern_history":         0.100,   # unchanged
    "device_trust":            0.240,   # pre-enrichment
}

SV_AFTER  = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])
SV_BEFORE = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])

# W_boot (Δσ scheme) = sigma_before² / sigma_after⁴
# Fixed factors (sb=sa): W = sb²/sa⁴ = 1/sa² (same as default)
# device_trust: W = 0.240²/0.120⁴ = 277.78 (4× elevated vs default 1/0.120²=69.44)
W_BOOT_RAW  = SV_BEFORE**2 / SV_AFTER**4
W_BOOT_DICT = {f: round(float(W_BOOT_RAW[i]), 4) for i, f in enumerate(FACTOR_NAMES)}

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── M4 pre-check ────────────────────────────────────────────────────────────────
_dt_w = W_BOOT_DICT["device_trust"]
_max_w = max(W_BOOT_DICT.values())
_tied = [f for f, w in W_BOOT_DICT.items() if abs(w - _max_w) < 0.001]
M4_PASS = (_dt_w >= _max_w - 0.001)   # device_trust tied-for or strictly highest
M4_NOTE = "tie" if len(_tied) > 1 else "strict"

# ── Structured A1×B0-P2 μ* ─────────────────────────────────────────────────────
# Same base geometry as V-CEILING. Device_trust (dim 5) FLATTENED to 0.50.
# Threat_intel (dim 2) and pattern_history (dim 4) kept as primary.
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

def _build_mu_star_b0p2() -> np.ndarray:
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    # B0-P2 flattening: device_trust (dim 5) → 0.50 (non-primary)
    # threat_intel (dim 2) and pattern_history (dim 4) KEPT as primary
    mu[:, :, IDX["device_trust"]] = 0.50
    return mu

MU_STAR = _build_mu_star_b0p2()

# Verify B0-P2 condition
assert np.all(MU_STAR[:, :, IDX["device_trust"]] == 0.50), "B0-P2 flatten failed dim5"
# Verify primary factors are NOT flat
assert MU_STAR[:, :, IDX["threat_intel_enrichment"]].std() > 0.10, "threat_intel unexpectedly flat"
assert MU_STAR[:, :, IDX["pattern_history"]].std() > 0.10,         "pattern_history unexpectedly flat"

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

    hist_c0 = [sample_alert(hist_rng_c0, SV_BEFORE) for _ in range(N_BOOTSTRAP)]
    hist_t2 = [sample_alert(hist_rng_t2, SV_AFTER)  for _ in range(N_BOOTSTRAP)]

    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_t2 = compute_enriched_bootstrap_prior(
        hist_t2, SIGMA_AFTER, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
        sigma_before=SIGMA_BEFORE,
    )

    # M4 proxy: closeness to μ* on device_trust dim only
    err_dt_c0 = float(np.linalg.norm(
        mu0_c0[:, :, IDX["device_trust"]] - MU_STAR[:, :, IDX["device_trust"]]))
    err_dt_t2 = float(np.linalg.norm(
        mu0_t2[:, :, IDX["device_trust"]] - MU_STAR[:, :, IDX["device_trust"]]))
    err_tot_c0 = float(np.linalg.norm(mu0_c0 - MU_STAR))
    err_tot_t2 = float(np.linalg.norm(mu0_t2 - MU_STAR))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0 in [("C0", mu0_c0), ("T2", mu0_t2)]:
        lr = np.random.RandomState(seed + 30000)   # shared RNG — identical alert sequence

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
            "err_dt":    err_dt_c0  if cond == "C0" else err_dt_t2,
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
    m2  = bool(float(n_half_t2.mean()) < float(n_half_c0.mean())
               and float(p) < 0.01 and abs(d) > 0.28)

    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(),      scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    cit2 = scipy_stats.t.interval(0.95, n-1, loc=n_half_t2.mean(), scale=scipy_stats.sem(n_half_t2))

    err_dt_c0 = float(np.mean([r["C0"]["err_dt"] for r in seed_results]))
    err_dt_t2 = float(np.mean([r["T2"]["err_dt"] for r in seed_results]))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in seed_results]))

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_t2 = np.array([np.array(r["T2"]["post_accs"])[-100:].mean() for r in seed_results])
    _, p_fa = scipy_stats.ttest_rel(fa_t2, fa_c0)

    obs_d = abs(d)
    return {
        "n_half": {
            "c0_mean":    round(float(n_half_c0.mean()), 1),
            "c0_ci95":    [round(ci0[0], 1),  round(ci0[1], 1)],
            "t2_mean":    round(float(n_half_t2.mean()), 1),
            "t2_ci95":    [round(cit2[0], 1), round(cit2[1], 1)],
            "diff_mean":  round(float(diff.mean()), 2),
            "diff_ci95":  [round(ci[0], 2), round(ci[1], 2)],
            "p_value":    round(float(p), 6),
            "t_stat":     round(float(t_stat), 4),
            "cohens_d":   round(d, 4),
            "m2_pass":    m2,
        },
        "m4_dt_error": {
            "c0_dist":       round(err_dt_c0, 4),
            "t2_dist":       round(err_dt_t2, 4),
            "reduction_pct": round(float((err_dt_c0 - err_dt_t2) / (err_dt_c0 + 1e-9) * 100), 2),
            "t2_closer":     bool(err_dt_t2 < err_dt_c0),
        },
        "day1_accuracy": {
            "c0":       round(d1_c0, 4),
            "t2":       round(d1_t2, 4),
            "delta_pp": round((d1_t2 - d1_c0) * 100, 2),
            "t2_gt_c0": bool(d1_t2 > d1_c0),
        },
        "final_accuracy": {
            "c0_mean":  round(float(fa_c0.mean()), 4),
            "t2_mean":  round(float(fa_t2.mean()), 4),
            "delta_pp": round(float((fa_t2.mean() - fa_c0.mean()) * 100), 2),
            "p_value":  round(float(p_fa), 6),
        },
        "power_analysis": {
            "observed_d":    round(obs_d, 4),
            "power_at_n200": round(power_at_n(200, obs_d), 4),
        },
    }

# ── Sanity checks ────────────────────────────────────────────────────────────────
def sanity_checks(stats) -> dict:
    d    = abs(stats["n_half"]["cohens_d"])
    d1dp = stats["day1_accuracy"]["delta_pp"]
    d1ok = stats["day1_accuracy"]["t2_gt_c0"]
    suspicious = []

    if d < 0.15:
        suspicious.append(f"d={d:.3f} < 0.15 (too small)")
    if d > 0.50:
        suspicious.append(f"d={d:.3f} > 0.50 (too large for B0)")
    if not d1ok:
        suspicious.append(f"Day-1 delta={d1dp:.1f}pp (T2 not better than C0)")

    return {
        "pass":       len(suspicious) == 0,
        "flags":      suspicious,
        "d_in_range": bool(0.15 <= d <= 0.50),
        "day1_ok":    bool(d1ok),
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("V-CLAIM62-B0-P2 (MDM enrichment, device_trust non-primary, GAE 0.7.8)")
    print("=" * 70)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"B0 condition: device_trust FLAT (0.50 in μ*); "
          f"threat_intel+pattern_history PRIMARY")
    print(f"Enriched factor: device_trust only (0.240→0.120, 2.0× ratio)")
    print()

    # ── W_boot pre-print ────────────────────────────────────────────────────────
    print("W_boot values (sigma_before² / sigma_after⁴):")
    for f in FACTOR_NAMES:
        w = W_BOOT_DICT[f]
        tag = " ← enriched" if f == "device_trust" else ""
        print(f"  {f:<30} {w:.4f}{tag}")
    print(f"  Highest W_boot: {max(W_BOOT_DICT, key=lambda k: W_BOOT_DICT[k])} = {_max_w:.4f}")
    if len(_tied) > 1:
        print(f"  NOTE: TIE — factors at max W_boot: {_tied}")
        print(f"  asset_criticality tie is coincidental: W=1/sa²=1/0.060²=277.78")
        print(f"  device_trust is uniquely ELEVATED: default would be 1/0.120²=69.44")
    print(f"  M4 (device_trust highest): {'PASS' if M4_PASS else 'FAIL'} ({M4_NOTE})")
    print()

    # ── Verify C0/T2 symmetry on primary factors ────────────────────────────────
    ti_spread  = MU_STAR[:, :, IDX["threat_intel_enrichment"]].std()
    ph_spread  = MU_STAR[:, :, IDX["pattern_history"]].std()
    dt_spread  = MU_STAR[:, :, IDX["device_trust"]].std()
    print(f"μ* spread check (σ of μ values):")
    print(f"  threat_intel:    {ti_spread:.4f} (primary — HIGH)")
    print(f"  pattern_history: {ph_spread:.4f} (primary — HIGH)")
    print(f"  device_trust:    {dt_spread:.4f} (B0 condition — should be 0.00)")
    print()
    print(f"C0 and T2 identical sigma on threat_intel ({SIGMA_AFTER['threat_intel_enrichment']}) "
          f"and pattern_history ({SIGMA_AFTER['pattern_history']}): "
          f"no asymmetric gradient amplification possible on primary factors")
    print()

    # ── Run seeds ──────────────────────────────────────────────────────────────
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

    stats  = analyse(all_results)
    sane   = sanity_checks(stats)
    nh     = stats["n_half"]
    m4e    = stats["m4_dt_error"]
    d1     = stats["day1_accuracy"]
    fa     = stats["final_accuracy"]
    powa   = stats["power_analysis"]

    obs_d  = abs(nh["cohens_d"])

    # ── Save ─────────────────────────────────────────────────────────────────────
    results = {
        "experiment":      "V-CLAIM62-B0-P2",
        "gae_version":     "0.7.8",
        "date":            "2026-03-24",
        "persona":         "B0-P2: MDM enrichment, device_trust non-primary",
        "enriched_factor": "device_trust",
        "sigma_before":    0.240,
        "sigma_after":     0.120,
        "w_boot_values":   W_BOOT_DICT,
        "w_boot_tie_note": (f"device_trust ties with asset_criticality at {_max_w:.4f}. "
                            f"Tie is coincidental (sa=0.060 → 1/sa²=277.78). "
                            f"device_trust elevation vs default: 277.78 vs 69.44 (4×)."),
        "n_seeds":         N_SEEDS,
        "cohens_d":        round(obs_d, 4),
        "p_value":         nh["p_value"],
        "ci_95":           nh["diff_ci95"],
        "m2_gate_pass":    nh["m2_pass"],
        "m4_enriched_pass": M4_PASS,
        "m4_note":         M4_NOTE,
        "day1_delta_pp":   d1["delta_pp"],
        "final_delta_pp":  fa["delta_pp"],
        "sanity_checks_passed": sane["pass"],
        "sanity_flags":    sane["flags"],
        "n_half_c0":       nh["c0_mean"],
        "n_half_t2":       nh["t2_mean"],
        "n_half_diff_ci95": nh["diff_ci95"],
        "final_accuracy":  fa,
        "power_at_n200":   powa["power_at_n200"],
        "mu_star_spreads": {
            "threat_intel":    round(ti_spread, 4),
            "pattern_history": round(ph_spread, 4),
            "device_trust":    round(dt_spread, 4),
        },
        "runtime_s":       round(elapsed_total, 1),
        "note":            "Raw numbers for roadmap session — no GTM conclusions drawn",
    }

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
        json.dump(results, fh, indent=2, cls=_NpEnc)
    print(f"Results saved to {out_path}")

    # ── Print verdict ─────────────────────────────────────────────────────────────
    def _pf(b): return "PASS" if b else "FAIL"

    print()
    print("=" * 70)
    print("V-CLAIM62-B0-P2 (MDM → device_trust):")
    print("=" * 70)
    print(f"  GAE: 0.7.8")
    print(f"  W_boot values:")
    for f in FACTOR_NAMES:
        tag = " ← enriched" if f == "device_trust" else ""
        print(f"    {f:<30} {W_BOOT_DICT[f]:.4f}{tag}")
    print(f"  W_boot tie note: device_trust={_dt_w:.4f}, "
          f"asset_criticality={W_BOOT_DICT['asset_criticality']:.4f} (coincidental)")
    print(f"  M4 (device_trust highest W_boot): {_pf(M4_PASS)} ({M4_NOTE})")
    print()
    print(f"  Cohen's d: {obs_d:.4f}")
    print(f"  p-value:   {nh['p_value']:.6f}")
    print(f"  95% CI:    [{nh['diff_ci95'][0]:.2f}, {nh['diff_ci95'][1]:.2f}] decisions")
    print(f"  M2 gate (d>0.28, p<0.01): {_pf(nh['m2_pass'])}")
    print()
    print(f"  n_half C0: {nh['c0_mean']:.1f} [{nh['c0_ci95'][0]:.1f}, {nh['c0_ci95'][1]:.1f}]")
    print(f"  n_half T2: {nh['t2_mean']:.1f} [{nh['t2_ci95'][0]:.1f}, {nh['t2_ci95'][1]:.1f}]")
    print()
    print(f"  Day-1 delta:   {d1['delta_pp']:+.2f}pp "
          f"(C0={d1['c0']:.1%}, T2={d1['t2']:.1%}) "
          f"[{'T2>C0' if d1['t2_gt_c0'] else 'T2<=C0'}]")
    print(f"  Final delta:   {fa['delta_pp']:+.2f}pp "
          f"(C0={fa['c0_mean']:.1%}, T2={fa['t2_mean']:.1%})")
    print(f"  Power at d={powa['observed_d']:.3f}: {powa['power_at_n200']:.1%} (N=200)")
    print()
    print(f"  Sanity checks: {_pf(sane['pass'])}"
          + (f" — {sane['flags']}" if sane["flags"] else ""))
    print()
    print("Raw numbers for roadmap session review.")
    print("=" * 70)

if __name__ == "__main__":
    main()
