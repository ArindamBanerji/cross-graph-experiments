"""
V-CLAIM60-S2P — CLAIM-60 precision substrate validation in S2P domain (GAE 0.7.6)
==================================================================================
RESULTS FOR ROADMAP SESSION ONLY. No GTM conclusions drawn here.

Tests whether graph enrichment produces accuracy lift in S2P manufacturing
procurement via DiagonalKernel (1/σ² weighting). Isolates kernel weight
effect from bootstrap initialization (standard bootstrap for both conditions).

C0:      DiagonalKernel(1/σ_baseline²), live alerts at baseline noise
T_enrich: DiagonalKernel(1/σ_enriched²), live alerts at enriched noise
Both: standard bootstrap from baseline σ (same μ₀)

Domain: S2P manufacturing procurement (C=5, A=5, d=8)
Gates: M1_S2P (≥3 factors with ≥25% σ reduction), M3_S2P (accuracy delta, p<0.10)

Run:
    PYTHONUTF8=1 python experiments/v_claim60_s2p/run.py
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
from gae.calibration import CalibrationProfile
from gae.kernels import DiagonalKernel

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SEEDS          = 100
N_BOOTSTRAP      = 1200
N_POST_BOOTSTRAP = 500
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80

N_CATEGORIES = 5
N_ACTIONS    = 5
N_FACTORS    = 8

FACTOR_NAMES = [
    "supplier_risk",       # dim 0
    "logistics_risk",      # dim 1
    "demand_risk",         # dim 2
    "inventory_risk",      # dim 3
    "regulatory_risk",     # dim 4
    "geopolitical_risk",   # dim 5
    "financial_risk",      # dim 6
    "environmental_risk",  # dim 7
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}

ACTIONS    = ["approve", "escalate", "hold", "reject", "expedite"]
CATEGORIES = [
    "supplier_focus", "logistics_focus", "demand_focus",
    "financial_focus", "geopolitical_focus",
]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Sigma profiles ──────────────────────────────────────────────────────────────
SIGMA_BASELINE = {
    "supplier_risk":      0.12,
    "logistics_risk":     0.22,
    "demand_risk":        0.18,
    "inventory_risk":     0.16,
    "regulatory_risk":    0.15,
    "geopolitical_risk":  0.20,
    "financial_risk":     0.12,
    "environmental_risk": 0.25,
}
SIGMA_ENRICHED = {
    "supplier_risk":      0.09,   # −25%
    "logistics_risk":     0.14,   # −36%
    "demand_risk":        0.13,   # −28%
    "inventory_risk":     0.11,   # −31%
    "regulatory_risk":    0.07,   # −53%
    "geopolitical_risk":  0.13,   # −35%
    "financial_risk":     0.09,   # −25%
    "environmental_risk": 0.16,   # −36%
}

def _sv(d):
    return np.array([d[f] for f in FACTOR_NAMES])

SV_BASELINE = _sv(SIGMA_BASELINE)
SV_ENRICHED = _sv(SIGMA_ENRICHED)

# Raw W = 1/σ² (not normalized — used for reporting and DiagonalKernel)
W_BASELINE_RAW = 1.0 / SV_BASELINE**2
W_ENRICHED_RAW = 1.0 / SV_ENRICHED**2

# ── M1_S2P: factors with ≥25% σ reduction ──────────────────────────────────────
def _compute_m1():
    qualifying = []
    for f in FACTOR_NAMES:
        reduction = (SIGMA_BASELINE[f] - SIGMA_ENRICHED[f]) / SIGMA_BASELINE[f]
        if reduction >= 0.25:
            qualifying.append(f)
    return qualifying

M1_FACTORS = _compute_m1()
M1_PASS    = len(M1_FACTORS) >= 3

# ── Structured S2P μ* ──────────────────────────────────────────────────────────
# Factor order: [supplier_risk, logistics_risk, demand_risk, inventory_risk,
#                regulatory_risk, geopolitical_risk, financial_risk, environmental_risk]
_APPROVE = {
    "supplier_focus":     [0.20, 0.75, 0.30, 0.35, 0.25, 0.70, 0.30, 0.25],
    "logistics_focus":    [0.20, 0.30, 0.70, 0.35, 0.25, 0.30, 0.30, 0.25],
    "demand_focus":       [0.20, 0.30, 0.30, 0.70, 0.25, 0.30, 0.30, 0.25],
    "financial_focus":    [0.20, 0.30, 0.30, 0.35, 0.25, 0.30, 0.75, 0.25],
    "geopolitical_focus": [0.20, 0.30, 0.30, 0.35, 0.25, 0.75, 0.30, 0.25],
}

def _build_mu_star() -> np.ndarray:
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for cat, app_vec in _APPROVE.items():
        ci = CAT_IDX[cat]
        av = np.array(app_vec)
        # approve: as specified
        mu[ci, ACT_IDX["approve"], :]  = av
        # escalate: inverted toward 0.80 — push toward opposite
        mu[ci, ACT_IDX["escalate"], :] = np.clip(1.0 - av, 0.15, 0.80)
        # hold: mid-range 0.50
        mu[ci, ACT_IDX["hold"], :]     = 0.50
        # reject: low values 0.20
        mu[ci, ACT_IDX["reject"], :]   = 0.20
        # expedite: high urgency 0.70-0.85
        exp_vec = np.where(av >= 0.50, 0.80, 0.75)
        mu[ci, ACT_IDX["expedite"], :] = exp_vec
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

def standard_bootstrap(historical_decisions):
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in historical_decisions:
        mu[c, a] += ETA_CONFIRM * (f - mu[c, a])
        mu[c, a]  = np.clip(mu[c, a], 0.0, 1.0)
    return mu

# ── Per-seed simulation ─────────────────────────────────────────────────────────
def run_one_seed(seed: int) -> dict:
    """
    Shared bootstrap (from sigma_baseline) → same μ₀ for both conditions.
    C0:       DiagonalKernel(W_baseline), live alerts at sigma_baseline
    T_enrich: DiagonalKernel(W_enriched), live alerts at sigma_enriched
    The ONLY methodological difference: kernel weights and live data quality.
    No enriched bootstrap prior in either condition.
    """
    # Shared bootstrap history — identical for both conditions
    boot_rng = np.random.RandomState(seed + 10000)
    hist     = [sample_alert(boot_rng, SV_BASELINE) for _ in range(N_BOOTSTRAP)]
    mu0      = standard_bootstrap(hist)   # identical μ₀ for both

    dk_c0     = DiagonalKernel(weights=W_BASELINE_RAW.copy())
    dk_enrich = DiagonalKernel(weights=W_ENRICHED_RAW.copy())

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, dk, sv, seed_offset in [
        ("C0",       dk_c0,     SV_BASELINE, 30000),
        ("T_enrich", dk_enrich, SV_ENRICHED, 30000),   # same RNG seed → same decision sequence
    ]:
        learn_rng = np.random.RandomState(seed + seed_offset)
        scorer = ProfileScorer(
            mu0.copy(),
            actions=ACTIONS,
            categories=CATEGORIES,
            profile=profile,
            eta_override=ETA_OVERRIDE,
            scoring_kernel=dk,
        )

        # Day-1 accuracy
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, sv)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1

        # Post-bootstrap learning
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(learn_rng, sv)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(learn_rng, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "day1_acc":  day1_correct / 50.0,
            "post_accs": post_accs,
        }
    return out

# ── Analysis ────────────────────────────────────────────────────────────────────
def analyse(seed_results: list) -> dict:
    n = len(seed_results)

    d1_c0  = np.array([r["C0"]["day1_acc"]       for r in seed_results])
    d1_en  = np.array([r["T_enrich"]["day1_acc"]  for r in seed_results])

    fa_c0  = np.array([np.mean(r["C0"]["post_accs"][-100:])       for r in seed_results])
    fa_en  = np.array([np.mean(r["T_enrich"]["post_accs"][-100:]) for r in seed_results])

    delta_pp = float((fa_en.mean() - fa_c0.mean()) * 100)
    _, p_two  = scipy_stats.ttest_rel(fa_en, fa_c0)
    p_one     = float(p_two) / 2.0 if delta_pp > 0 else 1.0 - float(p_two) / 2.0
    m3_pass   = bool(delta_pp > 0 and p_one < 0.10)

    ci = scipy_stats.t.interval(0.95, n - 1,
                                loc=(fa_en - fa_c0).mean() * 100,
                                scale=scipy_stats.sem((fa_en - fa_c0) * 100))

    return {
        "day1": {
            "c0":       round(float(d1_c0.mean()), 4),
            "t_enrich": round(float(d1_en.mean()),  4),
            "delta_pp": round(float((d1_en.mean() - d1_c0.mean()) * 100), 2),
        },
        "final": {
            "c0":        round(float(fa_c0.mean()), 4),
            "t_enrich":  round(float(fa_en.mean()),  4),
            "delta_pp":  round(delta_pp, 2),
            "ci_95":     [round(ci[0], 2), round(ci[1], 2)],
            "p_one":     round(p_one, 4),
            "p_two":     round(float(p_two), 4),
        },
        "m3_pass": m3_pass,
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-CLAIM60-S2P (DiagonalKernel enrichment, GAE 0.7.6)")
    print("=" * 65)
    print(f"Domain: S2P manufacturing procurement (C={N_CATEGORIES}, "
          f"A={N_ACTIONS}, d={N_FACTORS})")
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print()

    # ── M1_S2P pre-check ───────────────────────────────────────────────────────
    print(f"M1_S2P: {len(M1_FACTORS)}/8 factors with ≥25% σ reduction "
          f"[{'PASS' if M1_PASS else 'FAIL'}]")
    for f in FACTOR_NAMES:
        red  = (SIGMA_BASELINE[f] - SIGMA_ENRICHED[f]) / SIGMA_BASELINE[f] * 100
        flag = " ✓" if f in M1_FACTORS else ""
        print(f"  {f:<22} {SIGMA_BASELINE[f]:.2f}→{SIGMA_ENRICHED[f]:.2f}"
              f"  (−{red:.0f}%){flag}")
    print()

    # ── Per-factor kernel weights ───────────────────────────────────────────────
    print(f"{'Factor':<22} {'W_before':>10} {'W_after':>10} {'Change':>10}")
    print(f"  {'-'*54}")
    for i, f in enumerate(FACTOR_NAMES):
        wb  = W_BASELINE_RAW[i]
        we  = W_ENRICHED_RAW[i]
        chg = (we / wb - 1) * 100
        print(f"  {f:<22} {wb:>10.1f} {we:>10.1f} {f'+{chg:.0f}%':>10}")
    largest_gain_f = FACTOR_NAMES[int(np.argmax(W_ENRICHED_RAW / W_BASELINE_RAW))]
    print(f"  Largest weight gain: {largest_gain_f}")
    print()

    # ── Run seeds ──────────────────────────────────────────────────────────────
    t0 = time.time()
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_one_seed(seed))
        if (seed + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate    = (seed + 1) / elapsed
            print(f"  Seed {seed+1:3d}/{N_SEEDS}  "
                  f"[{elapsed:.1f}s, ETA {(N_SEEDS-seed-1)/rate:.0f}s]")

    elapsed_total = time.time() - t0
    print(f"\nAll seeds complete in {elapsed_total:.1f}s")

    stats = analyse(all_results)

    # ── Linear scaling check ───────────────────────────────────────────────────
    soc_delta   = 5.0
    soc_nr      = 2.6
    s2p_nr_base = 2.08
    s2p_nr_enr  = 2.29
    pred_base   = round(soc_delta * (s2p_nr_base / soc_nr), 2)   # 4.0pp
    pred_enr    = round(soc_delta * (s2p_nr_enr  / soc_nr), 2)   # 4.4pp
    actual      = stats["final"]["delta_pp"]
    scaling_holds = bool(pred_base - 1.0 <= actual <= pred_enr + 1.0)  # ±1pp tolerance
    if actual >= pred_base:
        scaling_label = "holds"
    elif actual > 0:
        scaling_label = "below prediction"
    else:
        scaling_label = "no lift"

    # ── CLAIM-60 S2P status ────────────────────────────────────────────────────
    m3   = stats["m3_pass"]
    d1ok = bool(stats["day1"]["delta_pp"] > 0)

    if M1_PASS and m3 and actual >= 4.0:
        claim_status = "VALIDATED"
    elif M1_PASS and m3:
        claim_status = "PARTIAL"
    else:
        claim_status = "REJECTED"

    # ── Save ────────────────────────────────────────────────────────────────────
    per_factor_w = {
        "before": {f: round(float(W_BASELINE_RAW[i]), 2) for i, f in enumerate(FACTOR_NAMES)},
        "after":  {f: round(float(W_ENRICHED_RAW[i]), 2) for i, f in enumerate(FACTOR_NAMES)},
        "largest_gain": largest_gain_f,
        "per_factor_change_pct": {
            f: round(float((W_ENRICHED_RAW[i] / W_BASELINE_RAW[i] - 1) * 100), 1)
            for i, f in enumerate(FACTOR_NAMES)
        },
    }

    results = {
        "experiment": "V-CLAIM60-S2P",
        "gae_version": "0.7.6",
        "date": "2026-03-24",
        "domain": "S2P manufacturing procurement",
        "n_seeds": N_SEEDS,
        "noise_ratio": {
            "baseline":      2.08,
            "enriched":      2.29,
            "soc_reference": 2.6,
        },
        "m1_s2p": {
            "factors_with_25pct_reduction": len(M1_FACTORS),
            "factor_list": M1_FACTORS,
            "pass": bool(M1_PASS),
        },
        "m3_s2p": {
            "accuracy_c0":       stats["final"]["c0"],
            "accuracy_t_enrich": stats["final"]["t_enrich"],
            "delta_pp":          stats["final"]["delta_pp"],
            "ci_95":             stats["final"]["ci_95"],
            "p_value":           stats["final"]["p_one"],
            "pass":              bool(m3),
        },
        "day1_accuracy": {
            "c0":       stats["day1"]["c0"],
            "t_enrich": stats["day1"]["t_enrich"],
            "delta_pp": stats["day1"]["delta_pp"],
        },
        "linear_scaling_check": {
            "soc_delta_pp":                soc_delta,
            "soc_noise_ratio":             soc_nr,
            "s2p_noise_ratio_baseline":    s2p_nr_base,
            "s2p_noise_ratio_enriched":    s2p_nr_enr,
            "predicted_s2p_delta_baseline": pred_base,
            "predicted_s2p_delta_enriched": pred_enr,
            "actual_s2p_delta":            actual,
            "scaling_holds":               scaling_holds,
            "scaling_label":               scaling_label,
        },
        "per_factor_weights": per_factor_w,
        "claim60_s2p_status": claim_status,
        "runtime_s": round(elapsed_total, 1),
        "note": "Raw numbers for roadmap session — no GTM conclusions drawn",
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

    # ── Print verdict ────────────────────────────────────────────────────────────
    def _pf(b): return "PASS" if b else "FAIL"

    print()
    print("=" * 65)
    print("V-CLAIM60-S2P Results:")
    print("=" * 65)
    print(f"Noise ratio: baseline=2.08× enriched=2.29× (SOC reference: 2.6×)")
    print()
    print(f"M1_S2P: {len(M1_FACTORS)} factors with ≥25% σ reduction [{_pf(M1_PASS)}]")
    print(f"  Qualifying factors: {', '.join(M1_FACTORS)}")
    print()
    print(f"{'Factor':<22} {'W_before':>10} {'W_after':>10} {'Change':>10}")
    for i, f in enumerate(FACTOR_NAMES):
        wb  = W_BASELINE_RAW[i]
        we  = W_ENRICHED_RAW[i]
        chg = (we / wb - 1) * 100
        print(f"  {f:<22} {wb:>10.1f} {we:>10.1f} {f'+{chg:.0f}%':>10}")
    print()
    print(f"M3_S2P: C0={stats['final']['c0']*100:.1f}%, "
          f"T_enrich={stats['final']['t_enrich']*100:.1f}%, "
          f"delta={actual:+.1f}pp, p={stats['final']['p_one']:.4f} [{_pf(m3)}]")
    print(f"  95% CI on delta: [{stats['final']['ci_95'][0]:.2f}, "
          f"{stats['final']['ci_95'][1]:.2f}]pp")
    print(f"Day-1: C0={stats['day1']['c0']:.1%}, "
          f"T_enrich={stats['day1']['t_enrich']:.1%}, "
          f"delta={stats['day1']['delta_pp']:+.1f}pp")
    print()
    print(f"Linear scaling check:")
    print(f"  SOC: +5.0pp at 2.6× noise ratio")
    print(f"  S2P predicted: +{pred_base:.1f} to +{pred_enr:.1f}pp (linear)")
    print(f"  S2P actual:    {actual:+.1f}pp")
    print(f"  Scaling: [{scaling_label}]")
    print()
    print(f"CLAIM-60 S2P status: {claim_status}")
    print("Raw numbers for roadmap session review.")
    print("=" * 65)

if __name__ == "__main__":
    main()
