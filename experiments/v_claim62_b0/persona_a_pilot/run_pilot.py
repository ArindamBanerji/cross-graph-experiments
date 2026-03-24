"""
V-CLAIM62-B0-PA pilot — Persona A geometry, 20-seed pilot (GAE 0.7.8)
=======================================================================
Tests whether Persona A μ* geometry (device_trust PARTIAL discriminator,
spread=0.30, ~48% of primary) produces viable pilot signal before N=200 run.

Key difference from B0-P2: device_trust is NOT flat in μ* (spread=0.30).
threat_intel and pattern_history remain PRIMARY (spread=0.62 each).
Enriched factor: device_trust only. sigma_before=0.200, sigma_after=0.100.

C0: standard bootstrap (SV_BEFORE, device_trust=0.200), no enrichment
T2: Δσ-weighted bootstrap (SV_AFTER data, sigma_before provided)

Pilot gate checks:
  1. Day-1 delta in [+2pp, +6pp]
  2. d_estimate in [0.15, 0.40]
  3. C0 N_half < 120
  4. T2 N_half < C0 N_half

Run:
    PYTHONUTF8=1 python experiments/v_claim62_b0/persona_a_pilot/run_pilot.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile, compute_enriched_bootstrap_prior

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SEEDS          = 20      # pilot only
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
    "threat_intel_enrichment", # dim 2 — primary
    "time_anomaly",            # dim 3
    "pattern_history",         # dim 4 — primary
    "device_trust",            # dim 5 — partial discriminator, enriched
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
    "device_trust":            0.100,   # enriched (from 0.200)
}
SIGMA_BEFORE = {
    "travel_match":            0.165,   # unchanged
    "asset_criticality":       0.060,   # unchanged
    "threat_intel_enrichment": 0.090,   # unchanged
    "time_anomaly":            0.070,   # unchanged
    "pattern_history":         0.100,   # unchanged
    "device_trust":            0.200,   # pre-enrichment
}

SV_AFTER  = np.array([SIGMA_AFTER[f]  for f in FACTOR_NAMES])
SV_BEFORE = np.array([SIGMA_BEFORE[f] for f in FACTOR_NAMES])

# W_boot = sigma_before² / sigma_after⁴
W_BOOT_RAW  = np.array([SIGMA_BEFORE[f]**2 / SIGMA_AFTER[f]**4 for f in FACTOR_NAMES])
W_BOOT_DICT = {f: round(float(W_BOOT_RAW[i]), 4) for i, f in enumerate(FACTOR_NAMES)}

class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── Persona A μ* ───────────────────────────────────────────────────────────────
# device_trust has spread=0.30 (partial discriminator, NOT flat).
# threat_intel (dim 2) and pattern_history (dim 4) are primary (spread=0.62).
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

# ── Spread verification ─────────────────────────────────────────────────────────
SPREADS = {}
for i, f in enumerate(FACTOR_NAMES):
    vals = MU_STAR[:, :, i].flatten()
    SPREADS[f] = round(float(vals.max() - vals.min()), 4)

_dt_spread = SPREADS["device_trust"]
_ti_spread = SPREADS["threat_intel_enrichment"]
_ph_spread = SPREADS["pattern_history"]

assert 0.25 <= _dt_spread <= 0.35, (
    f"device_trust spread={_dt_spread:.3f} out of [0.25, 0.35] — STOP")
assert 0.55 <= _ti_spread <= 0.70, (
    f"threat_intel spread={_ti_spread:.3f} out of [0.55, 0.70] — STOP")
assert 0.55 <= _ph_spread <= 0.70, (
    f"pattern_history spread={_ph_spread:.3f} out of [0.55, 0.70] — STOP")

# W_boot check
_wdt = W_BOOT_DICT["device_trust"]
_wti = W_BOOT_DICT["threat_intel_enrichment"]
assert _wdt > _wti, f"W_boot_device_trust ({_wdt}) not > W_boot_threat_intel ({_wti}) — STOP"

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

    ci   = scipy_stats.t.interval(0.95, n-1, loc=diff.mean(),      scale=scipy_stats.sem(diff))
    ci0  = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    cit2 = scipy_stats.t.interval(0.95, n-1, loc=n_half_t2.mean(), scale=scipy_stats.sem(n_half_t2))

    d1_c0 = float(np.mean([r["C0"]["day1_acc"] for r in seed_results]))
    d1_t2 = float(np.mean([r["T2"]["day1_acc"] for r in seed_results]))
    d1_delta = round((d1_t2 - d1_c0) * 100, 2)

    fa_c0 = np.array([np.array(r["C0"]["post_accs"])[-100:].mean() for r in seed_results])
    fa_t2 = np.array([np.array(r["T2"]["post_accs"])[-100:].mean() for r in seed_results])

    obs_d = abs(d)
    c0_nh = float(n_half_c0.mean())
    t2_nh = float(n_half_t2.mean())

    # Four pilot gates
    g1 = bool(2.0 <= d1_delta <= 6.0)
    g2 = bool(0.15 <= obs_d <= 0.40)
    g3 = bool(c0_nh < 120.0)
    g4 = bool(t2_nh < c0_nh)

    return {
        "n_half_c0":    round(c0_nh, 1),
        "n_half_c0_ci": [round(ci0[0], 1),  round(ci0[1], 1)],
        "n_half_t2":    round(t2_nh, 1),
        "n_half_t2_ci": [round(cit2[0], 1), round(cit2[1], 1)],
        "diff_mean":    round(float(diff.mean()), 2),
        "diff_ci95":    [round(ci[0], 2), round(ci[1], 2)],
        "p_value":      round(float(p), 6),
        "d_estimate":   round(obs_d, 4),
        "day1_c0":      round(d1_c0, 4),
        "day1_t2":      round(d1_t2, 4),
        "day1_delta_pp": d1_delta,
        "final_c0":     round(float(fa_c0.mean()), 4),
        "final_t2":     round(float(fa_t2.mean()), 4),
        "final_delta_pp": round(float((fa_t2.mean() - fa_c0.mean()) * 100), 2),
        "pilot_gates": {
            "1_day1_in_range": g1,
            "2_d_in_range":    g2,
            "3_c0_n_half_lt_120": g3,
            "4_t2_lt_c0":     g4,
        },
        "all_gates_pass": bool(g1 and g2 and g3 and g4),
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("V-CLAIM62-B0-PA pilot (N=20, GAE 0.7.8)")
    print("=" * 68)
    print(f"N_SEEDS={N_SEEDS} (pilot), N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print()

    # ── Spread verification ────────────────────────────────────────────────────
    print("Spread verification (max-min across 24 cat×act pairs):")
    for f in FACTOR_NAMES:
        tag = ""
        if f in ("threat_intel_enrichment", "pattern_history"): tag = " ← primary"
        elif f == "device_trust":   tag = " ← enriched (partial)"
        elif f == "asset_criticality": tag = " ← moderate"
        print(f"  {f:<30} {SPREADS[f]:.4f}{tag}")
    print()

    # ── W_boot print ──────────────────────────────────────────────────────────
    print("W_boot values (sigma_before² / sigma_after⁴):")
    for f in FACTOR_NAMES:
        tag = " ← enriched" if f == "device_trust" else ""
        print(f"  {f:<30} {W_BOOT_DICT[f]:.4f}{tag}")
    print(f"  W_boot_device_trust ({_wdt:.1f}) >> "
          f"W_boot_threat_intel ({_wti:.1f}) [OK]")
    print()

    # ── Run seeds ──────────────────────────────────────────────────────────────
    t0 = time.time()
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(seed))
    elapsed_total = time.time() - t0
    print(f"All {N_SEEDS} seeds complete in {elapsed_total:.1f}s")

    stats = analyse(all_results)
    gates = stats["pilot_gates"]

    # ── Save ─────────────────────────────────────────────────────────────────────
    results = {
        "experiment":   "V-CLAIM62-B0-PA-pilot",
        "gae_version":  "0.7.8",
        "date":         "2026-03-24",
        "n_seeds":      N_SEEDS,
        "spread_verification": {
            f: SPREADS[f] for f in FACTOR_NAMES
        },
        "w_boot_device_trust": _wdt,
        "w_boot_threat_intel": _wti,
        "day1_delta_pp": stats["day1_delta_pp"],
        "n_half_c0":    stats["n_half_c0"],
        "n_half_t2":    stats["n_half_t2"],
        "d_estimate":   stats["d_estimate"],
        "pilot_gate_checks": {
            "1_day1_in_range":    gates["1_day1_in_range"],
            "2_d_in_range":       gates["2_d_in_range"],
            "3_c0_n_half_lt_120": gates["3_c0_n_half_lt_120"],
            "4_t2_lt_c0":         gates["4_t2_lt_c0"],
        },
        "all_gates_pass": stats["all_gates_pass"],
        "proceed_to_n200": stats["all_gates_pass"],
        "detail": {
            "n_half_c0_ci95":  stats["n_half_c0_ci"],
            "n_half_t2_ci95":  stats["n_half_t2_ci"],
            "diff_mean":       stats["diff_mean"],
            "diff_ci95":       stats["diff_ci95"],
            "p_value":         stats["p_value"],
            "day1_c0":         stats["day1_c0"],
            "day1_t2":         stats["day1_t2"],
            "final_c0":        stats["final_c0"],
            "final_t2":        stats["final_t2"],
            "final_delta_pp":  stats["final_delta_pp"],
        },
        "runtime_s": round(elapsed_total, 1),
        "note": "Raw numbers for roadmap session — no GTM conclusions drawn",
    }

    out_dir  = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_pilot.json"

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
    print("=" * 68)
    print("V-CLAIM62-B0-PA pilot (N=20, GAE 0.7.8):")
    print()
    print("Spread verification:")
    print(f"  threat_intel:      {SPREADS['threat_intel_enrichment']:.3f} (primary, expected ~0.62)")
    print(f"  pattern_history:   {SPREADS['pattern_history']:.3f} (primary, expected ~0.62)")
    print(f"  device_trust:      {SPREADS['device_trust']:.3f} (enriched, expected ~0.30)")
    print(f"  asset_criticality: {SPREADS['asset_criticality']:.3f} (moderate, expected ~0.15)")
    print(f"  travel_match:      {SPREADS['travel_match']:.3f} (low)")
    print(f"  time_anomaly:      {SPREADS['time_anomaly']:.3f} (low)")
    print()
    print(f"W_boot_device_trust: {_wdt:.1f} >> W_boot_threat_intel: {_wti:.1f} [OK]")
    print()
    print("Pilot results:")
    print(f"  Day-1 delta: {stats['day1_delta_pp']:+.1f}pp  "
          f"(C0={stats['day1_c0']:.1%}, T2={stats['day1_t2']:.1%})  "
          f"[gate: +2 to +6pp] -> {_pf(gates['1_day1_in_range'])}")
    print(f"  d_estimate:  {stats['d_estimate']:.3f}          "
          f"[gate: 0.15 to 0.40] -> {_pf(gates['2_d_in_range'])}")
    print(f"  C0 N_half:   {stats['n_half_c0']:.1f}           "
          f"[gate: < 120] -> {_pf(gates['3_c0_n_half_lt_120'])}")
    print(f"  T2 N_half:   {stats['n_half_t2']:.1f}           "
          f"[gate: T2 < C0] -> {_pf(gates['4_t2_lt_c0'])}")
    print()
    print(f"  Final delta: {stats['final_delta_pp']:+.2f}pp  "
          f"(C0={stats['final_c0']:.1%}, T2={stats['final_t2']:.1%})")
    print(f"  diff CI95:   [{stats['diff_ci95'][0]:.2f}, {stats['diff_ci95'][1]:.2f}] decisions  "
          f"p={stats['p_value']:.4f}")
    print()
    print(f"All gates pass: {'YES' if stats['all_gates_pass'] else 'NO'}")
    print(f"Proceed to N=200: {'YES' if stats['all_gates_pass'] else 'NO'}")
    print("Raw numbers for roadmap session review.")
    print("=" * 68)

if __name__ == "__main__":
    main()
