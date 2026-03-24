"""
V-ENRICHMENT-NEGATIVE — adversarial enrichment safety experiment (GAE 0.7.6)
=============================================================================
SAFETY EXPERIMENT. Results for roadmap session only.
Do not draw GTM conclusions from this script.

Tests whether bad enrichment (low-trust or contradictory sources) can hurt
accuracy after the DiagonalKernel self-correction mechanism is applied.

Safety gate: max degradation CI_upper(C0 - T_bad) ≤ 3pp for both personas.

Two adversarial personas:
  P1: CISA KEV vs vendor feed contradiction → threat_intel σ 0.13 → 0.22
  P2: Tier 4 OSINT bulk → threat_intel σ 0.13 → 0.25, pattern_history 0.10 → 0.20

Kernel self-correction: bad enrichment → higher σ → W = 1/σ² drops automatically.
AMBER auto-pause: fires if σ_mean > 0.157. P1≈0.120, P2≈0.142 — below threshold.

Run:
    PYTHONUTF8=1 python experiments/v_enrichment_negative/run.py
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
N_SEEDS          = 20
N_BOOTSTRAP      = 1200
N_POST_BOOTSTRAP = 500
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80
AMBER_SIGMA_THRESHOLD = 0.157   # σ_mean threshold for AMBER auto-pause check

N_CATEGORIES = 6
N_ACTIONS    = 4
N_FACTORS    = 6

FACTOR_NAMES = [
    "travel_match",            # dim 0
    "asset_criticality",       # dim 1
    "threat_intel_enrichment", # dim 2
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

# ── Sigma profiles ──────────────────────────────────────────────────────────────
SIGMA_BASELINE = {
    "travel_match":            0.18,
    "asset_criticality":       0.06,
    "threat_intel_enrichment": 0.13,
    "time_anomaly":            0.07,
    "pattern_history":         0.10,
    "device_trust":            0.09,
}
# P1: Tier 1-2 contradiction — threat_intel degrades
SIGMA_P1_BAD = {**SIGMA_BASELINE, "threat_intel_enrichment": 0.22}
# P2: Tier 4 OSINT — threat_intel and pattern_history both degrade
SIGMA_P2_BAD = {**SIGMA_BASELINE, "threat_intel_enrichment": 0.25, "pattern_history": 0.20}

def _sv(d):
    return np.array([d[f] for f in FACTOR_NAMES])

def _kernel_weights(sigma_dict):
    """Raw DiagonalKernel weights: W = 1/σ² (un-normalized)."""
    sv = _sv(sigma_dict)
    return 1.0 / sv**2

def _kernel_weights_normalized(sigma_dict):
    """Normalized weights: W/mean(W), mean=1."""
    w = _kernel_weights(sigma_dict)
    return w / w.mean()

# ── Structured A1×B1 μ* (same as v5/v6/v7/vceiling) ───────────────────────────
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

def run_one_seed(seed: int, sigma_c0: dict, sigma_bad: dict) -> dict:
    """
    C0:    DiagonalKernel(1/σ_baseline²), bootstrap from σ_baseline
    T_bad: DiagonalKernel(1/σ_bad²),     bootstrap from σ_bad
    Same μ*, same post-bootstrap learning seed.
    """
    sv_c0  = _sv(sigma_c0)
    sv_bad = _sv(sigma_bad)
    dk_c0  = DiagonalKernel(weights=_kernel_weights(sigma_c0))
    dk_bad = DiagonalKernel(weights=_kernel_weights(sigma_bad))

    hist_rng_c0  = np.random.RandomState(seed + 10000)
    hist_rng_bad = np.random.RandomState(seed + 20000)
    learn_rng_c0 = np.random.RandomState(seed + 30000)
    learn_rng_bad = np.random.RandomState(seed + 30000)  # identical sequence

    hist_c0  = [sample_alert(hist_rng_c0,  sv_c0)  for _ in range(N_BOOTSTRAP)]
    hist_bad = [sample_alert(hist_rng_bad, sv_bad) for _ in range(N_BOOTSTRAP)]

    mu0_c0  = standard_bootstrap(hist_c0)
    mu0_bad = standard_bootstrap(hist_bad)

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, dk, sv, lr in [
        ("C0",    mu0_c0,  dk_c0,  sv_c0,  learn_rng_c0),
        ("T_bad", mu0_bad, dk_bad, sv_bad, learn_rng_bad),
    ]:
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
            c, gt_a, f = sample_alert(lr, sv)
            res    = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "day1_acc":  day1_correct / 50.0,
            "post_accs": post_accs,
        }
    return out

# ── Analysis ────────────────────────────────────────────────────────────────────
def analyse_persona(seed_results: list, safety_gate_pp: float = 3.0) -> dict:
    n = len(seed_results)

    d1_c0  = np.array([r["C0"]["day1_acc"]    for r in seed_results])
    d1_bad = np.array([r["T_bad"]["day1_acc"] for r in seed_results])

    fa_c0  = np.array([np.mean(r["C0"]["post_accs"][-100:])    for r in seed_results])
    fa_bad = np.array([np.mean(r["T_bad"]["post_accs"][-100:]) for r in seed_results])

    # Degradation: positive = C0 better than T_bad (C0 - T_bad)
    deg_fa = (fa_c0 - fa_bad) * 100.0   # in pp, positive = degradation
    _, p_fa = scipy_stats.ttest_rel(fa_c0, fa_bad)

    # 95% CI on degradation (final accuracy)
    ci_fa = scipy_stats.t.interval(0.95, n - 1,
                                   loc=deg_fa.mean(),
                                   scale=scipy_stats.sem(deg_fa))

    # CI on day-1 degradation
    deg_d1 = (d1_c0 - d1_bad) * 100.0
    ci_d1  = scipy_stats.t.interval(0.95, n - 1,
                                    loc=deg_d1.mean(),
                                    scale=scipy_stats.sem(deg_d1))

    max_ci_upper = float(ci_fa[1])
    gate_pass = bool(max_ci_upper <= safety_gate_pp)

    return {
        "day1": {
            "c0_mean":      round(float(d1_c0.mean()), 4),
            "tbad_mean":    round(float(d1_bad.mean()), 4),
            "delta_pp":     round(float(deg_d1.mean()), 2),
            "ci_95":        [round(ci_d1[0], 2), round(ci_d1[1], 2)],
        },
        "final": {
            "c0_mean":      round(float(fa_c0.mean()), 4),
            "tbad_mean":    round(float(fa_bad.mean()), 4),
            "delta_pp":     round(float(deg_fa.mean()), 2),
            "ci_95":        [round(ci_fa[0], 2), round(ci_fa[1], 2)],
            "p_value":      round(float(p_fa), 4),
        },
        "max_degradation_ci_upper": round(max_ci_upper, 2),
        "gate_pass": gate_pass,
    }

# ── Kernel weight report ────────────────────────────────────────────────────────
def kernel_weight_report(sigma_c0: dict, sigma_bad: dict, factors: list) -> dict:
    """Raw W = 1/σ² for each factor, before and after bad enrichment."""
    w_c0  = {f: round(1.0 / sigma_c0[f]**2, 2) for f in factors}
    w_bad = {f: round(1.0 / sigma_bad[f]**2, 2) for f in factors}
    return {"before": w_c0, "after": w_bad}

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("V-ENRICHMENT-NEGATIVE Safety Experiment (GAE 0.7.6)")
    print("=" * 65)
    print(f"N_SEEDS={N_SEEDS} per persona, N_BOOTSTRAP={N_BOOTSTRAP}, "
          f"N_POST_BOOTSTRAP={N_POST_BOOTSTRAP}")
    print(f"Safety gate: max degradation CI_upper ≤ 3pp")
    print()

    personas = {
        "P1_contradictory_tier1_tier2": {
            "description": "CISA KEV vs vendor feed contradiction — same IOC",
            "sigma_c0":  SIGMA_BASELINE,
            "sigma_bad": SIGMA_P1_BAD,
            "sigma_change": {"threat_intel_enrichment": "0.13→0.22"},
            "degraded_factors": ["threat_intel_enrichment"],
        },
        "P2_osint_bulk_tier4": {
            "description": "High-volume OSINT degrades threat_intel + pattern_history",
            "sigma_c0":  SIGMA_BASELINE,
            "sigma_bad": SIGMA_P2_BAD,
            "sigma_change": {
                "threat_intel_enrichment": "0.13→0.25",
                "pattern_history":         "0.10→0.20",
            },
            "degraded_factors": ["threat_intel_enrichment", "pattern_history"],
        },
    }

    all_persona_results = {}

    for persona_key, persona_cfg in personas.items():
        sigma_c0  = persona_cfg["sigma_c0"]
        sigma_bad = persona_cfg["sigma_bad"]
        label     = "P1" if "P1" in persona_key else "P2"
        print(f"Running {label}: {persona_cfg['description']}")

        t0 = time.time()
        seed_results = []
        for seed in range(N_SEEDS):
            seed_results.append(run_one_seed(seed, sigma_c0, sigma_bad))
        elapsed = time.time() - t0
        print(f"  {label} complete in {elapsed:.1f}s")

        stats = analyse_persona(seed_results)

        # Kernel weight metrics
        kw = kernel_weight_report(sigma_c0, sigma_bad, FACTOR_NAMES)

        # Sigma stats
        sv_c0_arr  = _sv(sigma_c0)
        sv_bad_arr = _sv(sigma_bad)
        sigma_mean_before = float(sv_c0_arr.mean())
        sigma_mean_after  = float(sv_bad_arr.mean())

        # Noise ratio (mean σ after / mean σ before)
        noise_ratio_before = round(sigma_mean_before / sigma_mean_before, 3)  # 1.0×
        noise_ratio_after  = round(sigma_mean_after  / sigma_mean_before, 3)

        # AMBER check: σ_mean > 0.157 threshold
        amber_fires = bool(sigma_mean_after > AMBER_SIGMA_THRESHOLD)
        active_mechanism = "amber_autopause" if amber_fires else "kernel_weight_reduction"

        # Build persona result entry
        entry = {
            "description":  persona_cfg["description"],
            "sigma_change": persona_cfg["sigma_change"],
            "noise_ratio_before": noise_ratio_before,
            "noise_ratio_after":  noise_ratio_after,
            "sigma_mean_after":   round(sigma_mean_after, 4),
            "amber_fires":        amber_fires,
            "active_safety_mechanism": active_mechanism,
            "day1_accuracy": {
                "c0":       stats["day1"]["c0_mean"],
                "t_bad":    stats["day1"]["tbad_mean"],
                "delta_pp": stats["day1"]["delta_pp"],
            },
            "final_accuracy": {
                "c0":       stats["final"]["c0_mean"],
                "t_bad":    stats["final"]["tbad_mean"],
                "delta_pp": stats["final"]["delta_pp"],
                "p_value":  stats["final"]["p_value"],
            },
            "ci_95_degradation":       stats["final"]["ci_95"],
            "max_degradation_ci_upper": stats["max_degradation_ci_upper"],
            "gate_pass":               stats["gate_pass"],
            "kernel_weights": kw,
        }

        # Add per-factor kernel weight fields at top level (for spec JSON format)
        entry["kernel_weight_threat_intel_before"] = kw["before"]["threat_intel_enrichment"]
        entry["kernel_weight_threat_intel_after"]  = kw["after"]["threat_intel_enrichment"]
        if "pattern_history" in persona_cfg["degraded_factors"]:
            entry["kernel_weight_pattern_history_before"] = kw["before"]["pattern_history"]
            entry["kernel_weight_pattern_history_after"]  = kw["after"]["pattern_history"]

        all_persona_results[persona_key] = entry

    # ── Overall verdict ─────────────────────────────────────────────────────────
    all_pass = all(v["gate_pass"] for v in all_persona_results.values())
    overall_verdict = "SAFE" if all_pass else "UNSAFE"

    # ── Save ────────────────────────────────────────────────────────────────────
    results = {
        "experiment":                    "V-ENRICHMENT-NEGATIVE",
        "gae_version":                   "0.7.6",
        "date":                          "2026-03-24",
        "n_seeds":                       N_SEEDS,
        "safety_gate":                   "max_degradation_ci_upper <= 3pp",
        "bootstrap_contamination_tested": False,
        "personas":                      all_persona_results,
        "overall_verdict":               overall_verdict,
        "note":                          "GTM implications discussed with roadmap session before writing",
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
    print(f"\nResults saved to {out_path}")

    # ── Print verdict ────────────────────────────────────────────────────────────
    def _pf(b): return "PASS ≤3pp" if b else "FAIL >3pp"

    p1 = all_persona_results["P1_contradictory_tier1_tier2"]
    p2 = all_persona_results["P2_osint_bulk_tier4"]

    print()
    print("=" * 65)
    print("V-ENRICHMENT-NEGATIVE Safety Results:")
    print("=" * 65)
    print()
    print("P1 (Tier 1-2 contradiction):")
    print(f"  σ_threat_intel: 0.13 → 0.22")
    print(f"  Kernel weight W[threat_intel]: "
          f"{p1['kernel_weight_threat_intel_before']:.1f} → "
          f"{p1['kernel_weight_threat_intel_after']:.1f}")
    print(f"  Noise ratio: {p1['noise_ratio_before']:.2f}× → {p1['noise_ratio_after']:.2f}×")
    print(f"  AMBER auto-pause fires: {'yes' if p1['amber_fires'] else 'no'} "
          f"(σ_mean={p1['sigma_mean_after']:.3f}, threshold={AMBER_SIGMA_THRESHOLD})")
    print(f"  Active safety mechanism: {p1['active_safety_mechanism']}")
    print(f"  Day-1 degradation: {p1['day1_accuracy']['delta_pp']:.1f}pp "
          f"(95% CI upper: {all_persona_results['P1_contradictory_tier1_tier2']['ci_95_degradation'][1]:.1f}pp)")
    print(f"  Final accuracy degradation: {p1['final_accuracy']['delta_pp']:.1f}pp")
    print(f"  Safety gate: [{_pf(p1['gate_pass'])}]")
    print()
    print("P2 (Tier 4 OSINT bulk):")
    print(f"  σ_threat_intel: 0.13→0.25, σ_pattern_history: 0.10→0.20")
    print(f"  Kernel weight W[threat_intel]: "
          f"{p2['kernel_weight_threat_intel_before']:.1f} → "
          f"{p2['kernel_weight_threat_intel_after']:.1f}")
    print(f"  Kernel weight W[pattern_history]: "
          f"{p2['kernel_weight_pattern_history_before']:.1f} → "
          f"{p2['kernel_weight_pattern_history_after']:.1f}")
    print(f"  Noise ratio: {p2['noise_ratio_before']:.2f}× → {p2['noise_ratio_after']:.2f}×")
    print(f"  σ_mean after bad enrichment: {p2['sigma_mean_after']:.3f} "
          f"(AMBER threshold: {AMBER_SIGMA_THRESHOLD})")
    print(f"  AMBER auto-pause fires: {'yes' if p2['amber_fires'] else 'no'}")
    print(f"  Active safety mechanism: {p2['active_safety_mechanism']}")
    print(f"  Day-1 degradation: {p2['day1_accuracy']['delta_pp']:.1f}pp "
          f"(95% CI upper: {p2['ci_95_degradation'][1]:.1f}pp)")
    print(f"  Final accuracy degradation: {p2['final_accuracy']['delta_pp']:.1f}pp")
    print(f"  Safety gate: [{_pf(p2['gate_pass'])}]")
    print()
    print(f"OVERALL: {overall_verdict}")
    print(f"bootstrap_contamination_tested: false")
    print("Raw numbers for roadmap session review.")
    print("=" * 65)

if __name__ == "__main__":
    main()
