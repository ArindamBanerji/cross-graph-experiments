"""
V-ENRICHMENT-NEGATIVE v2 — GAE 0.7.7 gradient fix, N=50 per persona
=====================================================================
SAFETY EXPERIMENT. Pre-ship blocker. Results for roadmap session only.

Changes from v1: GAE 0.7.7 (DiagonalKernel gradient bounded), N_SEEDS=50.
Design, parameters, personas, and safety gate identical to v1.

Run:
    PYTHONUTF8=1 python experiments/v_enrichment_negative/run_v2.py
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
N_SEEDS          = 50        # increased from 20
N_BOOTSTRAP      = 1200
N_POST_BOOTSTRAP = 500
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80
AMBER_SIGMA_THRESHOLD = 0.157

N_CATEGORIES = 6
N_ACTIONS    = 4
N_FACTORS    = 6

FACTOR_NAMES = [
    "travel_match", "asset_criticality", "threat_intel_enrichment",
    "time_anomaly", "pattern_history", "device_trust",
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
SIGMA_BASELINE = {
    "travel_match": 0.18, "asset_criticality": 0.06,
    "threat_intel_enrichment": 0.13, "time_anomaly": 0.07,
    "pattern_history": 0.10, "device_trust": 0.09,
}
SIGMA_P1_BAD = {**SIGMA_BASELINE, "threat_intel_enrichment": 0.22}
SIGMA_P2_BAD = {**SIGMA_BASELINE, "threat_intel_enrichment": 0.25, "pattern_history": 0.20}

def _sv(d):
    return np.array([d[f] for f in FACTOR_NAMES])

def _kernel_weights(sd):
    return 1.0 / _sv(sd)**2

# ── Structured A1×B1 μ* ────────────────────────────────────────────────────────
_MU_RAW = {
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
    for (cat, act), vec in _MU_RAW.items():
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
def sample_alert(rng, sv):
    c = int(rng.choice(N_CATEGORIES))
    a = int(rng.choice(N_ACTIONS, p=GT_DIST[c]))
    f = np.clip(MU_STAR[c, a] + rng.randn(N_FACTORS) * sv, 0.0, 1.0)
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

def run_one_seed(seed, sigma_c0, sigma_bad):
    sv_c0  = _sv(sigma_c0)
    sv_bad = _sv(sigma_bad)
    dk_c0  = DiagonalKernel(weights=_kernel_weights(sigma_c0))
    dk_bad = DiagonalKernel(weights=_kernel_weights(sigma_bad))

    hist_c0  = [sample_alert(np.random.RandomState(seed + 10000), sv_c0)
                for _ in range(N_BOOTSTRAP)]
    hist_bad = [sample_alert(np.random.RandomState(seed + 20000), sv_bad)
                for _ in range(N_BOOTSTRAP)]

    mu0_c0  = standard_bootstrap(hist_c0)
    mu0_bad = standard_bootstrap(hist_bad)

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, dk, sv in [
        ("C0",    mu0_c0,  dk_c0,  sv_c0),
        ("T_bad", mu0_bad, dk_bad, sv_bad),
    ]:
        learn_rng = np.random.RandomState(seed + 30000)
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE, scoring_kernel=dk,
        )
        d1_rng = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng, sv)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(learn_rng, sv)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(learn_rng, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))
        out[cond] = {"day1_acc": day1_correct / 50.0, "post_accs": post_accs}
    return out

def analyse(seed_results, gate_pp=3.0):
    n = len(seed_results)
    d1_c0  = np.array([r["C0"]["day1_acc"]    for r in seed_results])
    d1_bad = np.array([r["T_bad"]["day1_acc"] for r in seed_results])
    fa_c0  = np.array([np.mean(r["C0"]["post_accs"][-100:])    for r in seed_results])
    fa_bad = np.array([np.mean(r["T_bad"]["post_accs"][-100:]) for r in seed_results])
    deg_fa  = (fa_c0 - fa_bad) * 100.0
    deg_d1  = (d1_c0 - d1_bad) * 100.0
    _, p_fa = scipy_stats.ttest_rel(fa_c0, fa_bad)
    ci_fa   = scipy_stats.t.interval(0.95, n-1, loc=deg_fa.mean(),
                                     scale=scipy_stats.sem(deg_fa))
    ci_d1   = scipy_stats.t.interval(0.95, n-1, loc=deg_d1.mean(),
                                     scale=scipy_stats.sem(deg_d1))
    return {
        "day1": {
            "c0_mean":   round(float(d1_c0.mean()), 4),
            "tbad_mean": round(float(d1_bad.mean()), 4),
            "delta_pp":  round(float(deg_d1.mean()), 2),
            "ci_95":     [round(ci_d1[0], 2), round(ci_d1[1], 2)],
        },
        "final": {
            "c0_mean":   round(float(fa_c0.mean()), 4),
            "tbad_mean": round(float(fa_bad.mean()), 4),
            "delta_pp":  round(float(deg_fa.mean()), 2),
            "ci_95":     [round(ci_fa[0], 2), round(ci_fa[1], 2)],
            "p_value":   round(float(p_fa), 4),
        },
        "max_degradation_ci_upper": round(float(ci_fa[1]), 2),
        "gate_pass": bool(float(ci_fa[1]) <= gate_pp),
    }

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-ENRICHMENT-NEGATIVE v2 (GAE 0.7.7, N=50)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS}, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"Safety gate: max degradation CI_upper <= 3pp")
    print()

    personas = {
        "P1_contradictory_tier1_tier2": {
            "description": "CISA KEV vs vendor feed contradiction — same IOC",
            "sigma_c0":  SIGMA_BASELINE,
            "sigma_bad": SIGMA_P1_BAD,
            "sigma_change": {"threat_intel_enrichment": "0.13->0.22"},
            "degraded_factors": ["threat_intel_enrichment"],
        },
        "P2_osint_bulk_tier4": {
            "description": "High-volume OSINT degrades threat_intel + pattern_history",
            "sigma_c0":  SIGMA_BASELINE,
            "sigma_bad": SIGMA_P2_BAD,
            "sigma_change": {
                "threat_intel_enrichment": "0.13->0.25",
                "pattern_history":         "0.10->0.20",
            },
            "degraded_factors": ["threat_intel_enrichment", "pattern_history"],
        },
    }

    all_persona_results = {}
    t0 = time.time()

    for pk, pc in personas.items():
        label = "P1" if "P1" in pk else "P2"
        print(f"Running {label}: {pc['description']}")
        t_p = time.time()
        sr = [run_one_seed(seed, pc["sigma_c0"], pc["sigma_bad"])
              for seed in range(N_SEEDS)]
        st = analyse(sr)
        print(f"  {label} complete in {time.time()-t_p:.1f}s — "
              f"final_delta={st['final']['delta_pp']:.2f}pp  "
              f"CI_upper={st['max_degradation_ci_upper']:.2f}pp  "
              f"gate={'PASS' if st['gate_pass'] else 'FAIL'}")

        sv_bad = _sv(pc["sigma_bad"])
        sv_c0  = _sv(pc["sigma_c0"])
        sm_after = float(sv_bad.mean())
        amber    = bool(sm_after > AMBER_SIGMA_THRESHOLD)
        kw_b = {f: round(1.0 / SIGMA_BASELINE[f]**2, 1) for f in FACTOR_NAMES}
        kw_a = {f: round(1.0 / pc["sigma_bad"][f]**2,  1) for f in FACTOR_NAMES}

        entry = {
            "description":     pc["description"],
            "sigma_change":    pc["sigma_change"],
            "noise_ratio_before": 1.0,
            "noise_ratio_after":  round(sm_after / float(sv_c0.mean()), 3),
            "sigma_mean_after":   round(sm_after, 4),
            "amber_fires":        amber,
            "active_safety_mechanism": "amber_autopause" if amber else "kernel_weight_reduction",
            "day1_accuracy":  {"c0": st["day1"]["c0_mean"],
                               "t_bad": st["day1"]["tbad_mean"],
                               "delta_pp": st["day1"]["delta_pp"]},
            "final_accuracy": {"c0": st["final"]["c0_mean"],
                               "t_bad": st["final"]["tbad_mean"],
                               "delta_pp": st["final"]["delta_pp"],
                               "p_value": st["final"]["p_value"]},
            "ci_95_degradation":        st["final"]["ci_95"],
            "max_degradation_ci_upper": st["max_degradation_ci_upper"],
            "gate_pass":                st["gate_pass"],
            "kernel_weight_threat_intel_before": kw_b["threat_intel_enrichment"],
            "kernel_weight_threat_intel_after":  kw_a["threat_intel_enrichment"],
        }
        if "pattern_history" in pc["degraded_factors"]:
            entry["kernel_weight_pattern_history_before"] = kw_b["pattern_history"]
            entry["kernel_weight_pattern_history_after"]  = kw_a["pattern_history"]
        all_persona_results[pk] = entry

    all_pass = all(v["gate_pass"] for v in all_persona_results.values())
    overall  = "SAFE" if all_pass else "UNSAFE"

    results = {
        "experiment":   "V-ENRICHMENT-NEGATIVE",
        "version":      "v2_gradient_fix_gae_0.7.7_n50",
        "gae_version":  "0.7.7",
        "date":         "2026-03-24",
        "n_seeds":      N_SEEDS,
        "safety_gate":  "max_degradation_ci_upper <= 3pp",
        "bootstrap_contamination_tested": False,
        "previous_result": {
            "p1_ci_upper": 3.0,
            "p1_gate":     "PASS (borderline)",
            "p2_ci_upper": 5.3,
            "p2_gate":     "FAIL",
            "overall":     "UNSAFE",
            "cause":       "DiagonalKernel gradient bug + N=20 wide CI",
        },
        "personas":        all_persona_results,
        "overall_verdict": overall,
        "note":            "GTM implications discussed with roadmap session before writing",
        "runtime_s":       round(time.time() - t0, 1),
    }

    out_path = Path(__file__).parent / "results" / "results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class _NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):   return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_NpEnc)
    print(f"\nResults saved to {out_path}")

    # ── Print verdict ────────────────────────────────────────────────────────────
    p1 = all_persona_results["P1_contradictory_tier1_tier2"]
    p2 = all_persona_results["P2_osint_bulk_tier4"]

    def _pf(b): return "PASS <=3pp" if b else "FAIL >3pp"

    print()
    print("=" * 65)
    print("V-ENRICHMENT-NEGATIVE v2 (GAE 0.7.7, N=50):")
    print("=" * 65)
    print()
    print("P1 (Tier 1-2 contradiction):")
    wb_ti = p1["kernel_weight_threat_intel_before"]
    wa_ti = p1["kernel_weight_threat_intel_after"]
    print(f"  W[threat_intel]: {wb_ti:.1f} -> {wa_ti:.1f} "
          f"({(1 - wa_ti/wb_ti)*100:.0f}% reduction)")
    print(f"  sigma_mean after: {p1['sigma_mean_after']:.3f} "
          f"(AMBER threshold: {AMBER_SIGMA_THRESHOLD})")
    print(f"  AMBER fires: {'yes' if p1['amber_fires'] else 'no'} — "
          f"active mechanism: {p1['active_safety_mechanism']}")
    ci_p1 = p1["ci_95_degradation"]
    print(f"  Day-1 degradation: {p1['day1_accuracy']['delta_pp']:.1f}pp "
          f"(95% CI: [{ci_p1[0]:.1f}, {ci_p1[1]:.1f}]pp)")
    print(f"  Final degradation: {p1['final_accuracy']['delta_pp']:.1f}pp")
    print(f"  Safety gate: [{_pf(p1['gate_pass'])}]")
    print(f"  vs v1 (bugged, N=20): CI upper was 3.0pp -> now {ci_p1[1]:.1f}pp")
    print()
    print("P2 (Tier 4 OSINT bulk):")
    wb_ti2 = p2["kernel_weight_threat_intel_before"]
    wa_ti2 = p2["kernel_weight_threat_intel_after"]
    wb_ph  = p2["kernel_weight_pattern_history_before"]
    wa_ph  = p2["kernel_weight_pattern_history_after"]
    print(f"  W[threat_intel]: {wb_ti2:.1f} -> {wa_ti2:.1f}")
    print(f"  W[pattern_history]: {wb_ph:.1f} -> {wa_ph:.1f}")
    print(f"  sigma_mean after: {p2['sigma_mean_after']:.3f} "
          f"(AMBER threshold: {AMBER_SIGMA_THRESHOLD})")
    print(f"  AMBER fires: {'yes' if p2['amber_fires'] else 'no'} — "
          f"active mechanism: {p2['active_safety_mechanism']}")
    ci_p2 = p2["ci_95_degradation"]
    print(f"  Day-1 degradation: {p2['day1_accuracy']['delta_pp']:.1f}pp "
          f"(95% CI: [{ci_p2[0]:.1f}, {ci_p2[1]:.1f}]pp)")
    print(f"  Final degradation: {p2['final_accuracy']['delta_pp']:.1f}pp")
    print(f"  Safety gate: [{_pf(p2['gate_pass'])}]")
    print(f"  vs v1 (bugged, N=20): CI upper was 5.3pp -> now {ci_p2[1]:.1f}pp")
    print()
    print(f"OVERALL: {overall}")
    print("bootstrap_contamination_tested: false")
    print("Raw numbers for roadmap session review.")
    print("=" * 65)

if __name__ == "__main__":
    main()
