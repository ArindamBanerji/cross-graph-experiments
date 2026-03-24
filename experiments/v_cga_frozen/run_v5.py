"""
V-CGA-FROZEN v5 — 2x2x2 factorial: structured mu*, factor discrimination, bootstrap method
============================================================================================
Distinguishes three hypotheses after v4b null result (random mu*, d=0.077):

  H1: Enriched prior helps with STRUCTURED mu* (domain-realistic geometry)
  H2: Benefit depends on correlation between enriched factors and mu*
      (only when threat_intel / pattern_history are high-discriminating)
  H3: Null everywhere (claim is dead regardless of mu* geometry)

Factorial factors:
  A — mu* geometry:     A0=random [0.15,0.85],  A1=SOC-structured
  B — factor discrim:   B0=low (flattened),      B1=high (as-is)
  C — bootstrap method: C0=standard (un-enriched x1.5), C1=enriched prior

Primary comparison in each A×B cell: C0 vs C1.
Gate: N_half(C1) < N_half(C0), p<0.01, d>0.3  (same as v4b).

Run:
    PYTHONUTF8=1 python experiments/v_cga_frozen/run_v5.py
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

# ── Parameters (committed — do not change) ────────────────────────────────────
N_SEEDS          = 100
N_BOOTSTRAP_HIST = 200
N_POST_BOOTSTRAP = 500
THETA_MIN        = 0.467
TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01
Q_BAR            = 0.75
ALPHA            = 0.80

# SOC domain
N_CATEGORIES = 6
N_ACTIONS    = 4
N_FACTORS    = 6

# Factor order — canonical, must match structured mu* spec
FACTOR_NAMES = [
    "travel_match",           # idx 0
    "asset_criticality",      # idx 1
    "threat_intel_enrichment",# idx 2
    "time_anomaly",           # idx 3
    "pattern_history",        # idx 4
    "device_trust",           # idx 5
]
IDX = {f: i for i, f in enumerate(FACTOR_NAMES)}
ENRICHMENT_FACTORS = ["threat_intel_enrichment", "pattern_history", "device_trust"]

ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = [
    "credential_access", "threat_intel_match", "lateral_movement",
    "data_exfiltration",  "insider_threat",    "cloud_infrastructure",
]
# Map category name → index
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# ── Sigma profiles ─────────────────────────────────────────────────────────────
SIGMA_ENRICHED = {
    "threat_intel_enrichment": 0.13,
    "pattern_history":         0.10,
    "device_trust":            0.11,
    "travel_match":            0.18,
    "asset_criticality":       0.06,
    "time_anomaly":            0.07,
}
_UNENRICH_MULT = 1.5
SIGMA_UNENRICHED = {
    name: (SIGMA_ENRICHED[name] * _UNENRICH_MULT if name in ENRICHMENT_FACTORS
           else SIGMA_ENRICHED[name])
    for name in FACTOR_NAMES
}

def _sigma_vec(d): return np.array([d[f] for f in FACTOR_NAMES])
SV_ENRICHED   = _sigma_vec(SIGMA_ENRICHED)
SV_UNENRICHED = _sigma_vec(SIGMA_UNENRICHED)

# ── Domain config (only .factor_names needed by GAE) ──────────────────────────
class _DomainConfig:
    factor_names = FACTOR_NAMES
DOMAIN_CONFIG = _DomainConfig()

# ── Structured mu* — A1 condition ─────────────────────────────────────────────
# Factor order: [travel_match, asset_criticality, threat_intel_enrichment,
#                time_anomaly, pattern_history, device_trust]
_MU_STAR_A1_RAW = {
    # (category, action) → 6-vector
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

def _build_mu_star_a1(flatten_enrichment: bool) -> np.ndarray:
    """
    Build A1 structured mu* tensor shape (N_CATEGORIES, N_ACTIONS, N_FACTORS).
    flatten_enrichment=True → B0: set threat_intel (idx 2) and pattern_history
    (idx 4) to 0.50 across all centroids (low-discriminating).
    flatten_enrichment=False → B1: use values as-is (high-discriminating).
    """
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5)
    for (cat, act), vec in _MU_STAR_A1_RAW.items():
        ci = CAT_IDX[cat]
        ai = ACT_IDX[act]
        mu[ci, ai, :] = vec
    if flatten_enrichment:
        mu[:, :, IDX["threat_intel_enrichment"]] = 0.50
        mu[:, :, IDX["pattern_history"]]         = 0.50
    return mu

# Pre-build structured mu* variants
MU_STAR_A1_B0 = _build_mu_star_a1(flatten_enrichment=True)   # low-discriminating
MU_STAR_A1_B1 = _build_mu_star_a1(flatten_enrichment=False)  # high-discriminating

# ── GT distribution derived from mu* ──────────────────────────────────────────
def _gt_dist_from_mu_star(mu_star: np.ndarray) -> np.ndarray:
    """
    Action distribution: action with highest Frobenius norm in mu_star row gets p=0.7,
    rest share p=0.1. Deterministic — no rng needed.
    """
    gt_dist = np.ones((N_CATEGORIES, N_ACTIONS)) * 0.1
    for c in range(N_CATEGORIES):
        norms = np.linalg.norm(mu_star[c], axis=-1)  # (N_ACTIONS,)
        dominant = int(np.argmax(norms))
        gt_dist[c, dominant] = 0.7
    gt_dist /= gt_dist.sum(axis=1, keepdims=True)
    return gt_dist

# ── Alert / feedback utilities ─────────────────────────────────────────────────
def sample_alert(rng, mu_star, gt_dist, sigma_vec):
    cat_weights = np.ones(N_CATEGORIES) / N_CATEGORIES
    c = int(rng.choice(N_CATEGORIES, p=cat_weights))
    a = int(rng.choice(N_ACTIONS,    p=gt_dist[c]))
    f = np.clip(mu_star[c, a] + rng.randn(N_FACTORS) * sigma_vec, 0.0, 1.0)
    return c, a, f

def analyst_feedback(rng, pred_a, gt_a):
    if rng.rand() < ALPHA:
        return (gt_a if rng.rand() < Q_BAR else int(rng.choice(N_ACTIONS))), True
    return pred_a, False

# ── Bootstrap priors ───────────────────────────────────────────────────────────
def standard_bootstrap(historical_decisions) -> np.ndarray:
    """Unweighted ETA-update bootstrap (C0)."""
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=float)
    for c, a, f in historical_decisions:
        mu[c, a] += ETA_CONFIRM * (f - mu[c, a])
        mu[c, a]  = np.clip(mu[c, a], 0.0, 1.0)
    return mu

# ── N_half ────────────────────────────────────────────────────────────────────
def compute_n_half(post_accs, window=50, gap_pp=2.0):
    arr = np.array(post_accs)
    threshold = (arr[-100:].mean() * 100.0 - gap_pp) / 100.0
    roll = np.convolve(arr, np.ones(window) / window, mode="valid")
    above = np.where(roll >= threshold)[0]
    return int(above[0]) + window if len(above) else N_POST_BOOTSTRAP

# ── Per-seed simulation for one A×B cell ──────────────────────────────────────
def run_one_seed(seed: int, mu_star: np.ndarray, gt_dist: np.ndarray,
                 random_mu_star: bool, rng_gt) -> dict:
    """
    Run C0 and C1 for a single seed within one A×B cell.
    mu_star and gt_dist are passed in (pre-built or per-seed random).
    rng_gt is the per-seed rng for A0 (random mu*); ignored for A1.
    """
    hist_rng_c0 = np.random.RandomState(seed + 10000)
    hist_rng_c1 = np.random.RandomState(seed + 20000)
    learn_rng_c0 = np.random.RandomState(seed + 30000)
    learn_rng_c1 = np.random.RandomState(seed + 30000)  # identical learning sequence
    day1_rng     = np.random.RandomState(seed + 40000)  # identical day-1 probe

    # For A0: override mu_star with per-seed random
    if random_mu_star:
        mu_star = rng_gt.uniform(0.15, 0.85, size=(N_CATEGORIES, N_ACTIONS, N_FACTORS))
        gt_dist = _gt_dist_from_mu_star(mu_star)

    # Historical decisions
    hist_c0 = [sample_alert(hist_rng_c0, mu_star, gt_dist, SV_UNENRICHED)
               for _ in range(N_BOOTSTRAP_HIST)]
    hist_c1 = [sample_alert(hist_rng_c1, mu_star, gt_dist, SV_ENRICHED)
               for _ in range(N_BOOTSTRAP_HIST)]

    # mu_0 for each condition
    mu0_c0 = standard_bootstrap(hist_c0)
    mu0_c1 = compute_enriched_bootstrap_prior(
        hist_c1, SIGMA_ENRICHED, DOMAIN_CONFIG,
        n_cat=N_CATEGORIES, n_act=N_ACTIONS, n_factors=N_FACTORS,
    )

    # Starting error
    err_c0 = float(np.linalg.norm(mu0_c0 - mu_star))
    err_c1 = float(np.linalg.norm(mu0_c1 - mu_star))

    profile = CalibrationProfile(temperature=TAU, learning_rate=ETA_CONFIRM)
    out = {}

    for cond, mu0, lr, h_rng in [
        ("C0", mu0_c0, learn_rng_c0, hist_rng_c0),
        ("C1", mu0_c1, learn_rng_c1, hist_rng_c1),
    ]:
        scorer = ProfileScorer(
            mu0.copy(), actions=ACTIONS, categories=CATEGORIES,
            profile=profile, eta_override=ETA_OVERRIDE,
        )

        # Day-1 accuracy (same 50-alert probe for both conditions per seed)
        d1_rng2 = np.random.RandomState(seed + 40000)
        day1_correct = 0
        for _ in range(50):
            c, gt_a, f = sample_alert(d1_rng2, mu_star, gt_dist, SV_ENRICHED)
            if scorer.score(f, c).action_index == gt_a:
                day1_correct += 1
        day1_acc = day1_correct / 50.0

        # Post-bootstrap learning
        post_accs = []
        for _ in range(N_POST_BOOTSTRAP):
            c, gt_a, f = sample_alert(lr, mu_star, gt_dist, SV_ENRICHED)
            res = scorer.score(f, c)
            pred_a = res.action_index
            final_a, _ = analyst_feedback(lr, pred_a, gt_a)
            scorer.update(f, c, final_a, (final_a == gt_a), gt_action_index=gt_a)
            post_accs.append(float(pred_a == gt_a))

        out[cond] = {
            "starting_error": err_c0 if cond == "C0" else err_c1,
            "day1_acc":       day1_acc,
            "post_accs":      post_accs,
            "n_half":         compute_n_half(post_accs),
        }

    return out

# ── Cell-level statistical analysis ───────────────────────────────────────────
def analyse_cell(seed_results: list) -> dict:
    n  = len(seed_results)
    n_half_c0 = np.array([r["C0"]["n_half"] for r in seed_results])
    n_half_c1 = np.array([r["C1"]["n_half"] for r in seed_results])
    diff = n_half_c0 - n_half_c1          # positive = C1 faster
    t, p = scipy_stats.ttest_rel(n_half_c0, n_half_c1)
    d    = float(diff.mean() / (diff.std() + 1e-9))
    red  = float((n_half_c0.mean() - n_half_c1.mean()) / (n_half_c0.mean() + 1e-9) * 100)
    m2_pass = (n_half_c1.mean() < n_half_c0.mean() and float(p) < 0.01 and abs(d) > 0.3)

    ci0 = scipy_stats.t.interval(0.95, n-1, loc=n_half_c0.mean(), scale=scipy_stats.sem(n_half_c0))
    ci1 = scipy_stats.t.interval(0.95, n-1, loc=n_half_c1.mean(), scale=scipy_stats.sem(n_half_c1))

    err_c0 = np.array([r["C0"]["starting_error"] for r in seed_results])
    err_c1 = np.array([r["C1"]["starting_error"] for r in seed_results])
    err_red = float((err_c0.mean() - err_c1.mean()) / (err_c0.mean() + 1e-9) * 100)
    m4_pass = bool(err_c1.mean() < err_c0.mean())

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
        "m2_pass":       m2_pass,
        "m4": {
            "starting_error_c0": round(float(err_c0.mean()), 4),
            "starting_error_c1": round(float(err_c1.mean()), 4),
            "reduction_pct":     round(err_red, 2),
            "pass":              m4_pass,
        },
        "day1_accuracy": {
            "c0":       round(d1_c0, 4),
            "c1":       round(d1_c1, 4),
            "delta_pp": round((d1_c1 - d1_c0) * 100, 2),
        },
        "final_accuracy": {
            "c0_mean":  round(float(fa_c0.mean()), 4),
            "c1_mean":  round(float(fa_c1.mean()), 4),
            "delta_pp": round(float((fa_c1.mean() - fa_c0.mean()) * 100), 2),
            "p_value":  round(float(p_fa), 6),
        },
    }

# ── Hypothesis verdicts ────────────────────────────────────────────────────────
def hypothesis_verdicts(cells: dict) -> dict:
    """
    H1: Enriched prior helps with structured mu* (A1×B1: C0 vs C1 passes M2+M4?)
    H2: Benefit depends on discrimination (A1×B1 effect > A1×B0 effect?)
    H3: Null everywhere (all 4 cells fail M2?)
    """
    a1b1 = cells["A1_B1"]
    a1b0 = cells["A1_B0"]
    a0b1 = cells["A0_B1"]
    a0b0 = cells["A0_B0"]

    h1_supported = a1b1["m2_pass"] and a1b1["m4"]["pass"]
    h1_verdict   = "SUPPORTED" if h1_supported else "REJECTED"
    h1_evidence  = (f"A1xB1: d={a1b1['cohens_d']:.3f}, p={a1b1['p_value']:.4f}, "
                    f"M4={'PASS' if a1b1['m4']['pass'] else 'FAIL'}")

    # H2: enriched factors must matter — compare A1xB1 vs A1xB0
    d_b1 = a1b1["cohens_d"]
    d_b0 = a1b0["cohens_d"]
    h2_supported = (d_b1 > d_b0) and (d_b1 - d_b0 > 0.1)  # B1 meaningfully larger effect
    h2_verdict   = "SUPPORTED" if h2_supported else "REJECTED"
    h2_evidence  = (f"A1xB1 d={d_b1:.3f} vs A1xB0 d={d_b0:.3f}  "
                    f"(diff={d_b1-d_b0:+.3f}; need B1>>B0)")

    # H3: null everywhere — all cells fail M2
    all_fail = not any(cells[k]["m2_pass"] for k in cells)
    h3_verdict  = "SUPPORTED" if all_fail else "REJECTED"
    h3_evidence = ("All 4 cells fail M2." if all_fail
                   else "At least one cell passes M2: "
                        + ", ".join(k for k in cells if cells[k]["m2_pass"]))

    # Claim recommendation
    if h1_supported:
        rec = "UNCONDITIONAL — structured-domain enriched bootstrap validated."
    elif not all_fail:
        rec = "CONDITIONAL — effect exists in some cells; domain-specificity required."
    else:
        rec = "RETIRE — effect is null across all factorial conditions."

    return {
        "H1": {"verdict": h1_verdict, "evidence": h1_evidence},
        "H2": {"verdict": h2_verdict, "evidence": h2_evidence},
        "H3": {"verdict": h3_verdict, "evidence": h3_evidence},
        "claim_recommendation": rec,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("V-CGA-FROZEN v5 — 2x2x2 Factorial (GAE 0.7.4)")
    print("=" * 70)
    print(f"N_SEEDS={N_SEEDS} per cell (8 cells = {8*N_SEEDS} total runs)")
    print(f"Factors: A=mu*_geometry, B=factor_discrimination, C=bootstrap_method")
    print()

    # Pre-build A1 mu* for B0 and B1 (deterministic, shared across seeds)
    mu_star_a1_b0 = MU_STAR_A1_B0.copy()
    mu_star_a1_b1 = MU_STAR_A1_B1.copy()
    gt_dist_a1_b0 = _gt_dist_from_mu_star(mu_star_a1_b0)
    gt_dist_a1_b1 = _gt_dist_from_mu_star(mu_star_a1_b1)

    # Cell definitions: (cell_name, random_mu_star, mu_star, gt_dist)
    CELLS = [
        ("A0_B0", True,  None,         None),          # A0: random mu* (B level irrelevant)
        ("A0_B1", True,  None,         None),          # A0: random mu* (B level irrelevant)
        ("A1_B0", False, mu_star_a1_b0, gt_dist_a1_b0),
        ("A1_B1", False, mu_star_a1_b1, gt_dist_a1_b1),
    ]

    t0 = time.time()
    all_cell_results = {}

    for cell_name, random_mu_star, mu_star, gt_dist in CELLS:
        print(f"  Running cell {cell_name} ({'random mu*' if random_mu_star else 'structured mu*'}) ...")
        cell_seed_results = []
        for seed in range(N_SEEDS):
            # For A0 cells: per-seed rng for random mu*
            gt_rng = np.random.RandomState(seed) if random_mu_star else None
            result = run_one_seed(
                seed, mu_star, gt_dist, random_mu_star, gt_rng
            )
            cell_seed_results.append(result)

        all_cell_results[cell_name] = cell_seed_results
        elapsed = time.time() - t0
        stats = analyse_cell(cell_seed_results)
        print(f"    done  [{elapsed:.1f}s]  "
              f"N_half C0={stats['n_half_c0']:.1f} C1={stats['n_half_c1']:.1f}  "
              f"d={stats['cohens_d']:.3f}  p={stats['p_value']:.4f}  "
              f"M2={'PASS' if stats['m2_pass'] else 'fail'}  "
              f"M4={'PASS' if stats['m4']['pass'] else 'fail'}")

    elapsed_total = time.time() - t0
    print(f"\nAll cells complete in {elapsed_total:.1f}s")

    # Analyse all cells
    cells_stats = {name: analyse_cell(res) for name, res in all_cell_results.items()}
    hyp = hypothesis_verdicts(cells_stats)

    # ── Build results dict ─────────────────────────────────────────────────────
    overall_verdict = "PASS" if hyp["H1"]["verdict"] == "SUPPORTED" else "FAIL"
    results = {
        "experiment":         "V-CGA-FROZEN-v5",
        "version":            "v5_factorial_2x2x2",
        "date":               "2026-03-23",
        "n_seeds_per_cell":   N_SEEDS,
        "n_cells":            len(CELLS),
        "verdict":            overall_verdict,
        "runtime_s":          round(elapsed_total, 1),
        "parameters": {
            "n_bootstrap_hist": N_BOOTSTRAP_HIST,
            "n_post_bootstrap": N_POST_BOOTSTRAP,
            "theta_min":        THETA_MIN,
            "tau":              TAU,
            "eta_confirm":      ETA_CONFIRM,
            "eta_override":     ETA_OVERRIDE,
            "q_bar":            Q_BAR,
            "alpha":            ALPHA,
        },
        "sigma_profiles": {
            "enriched":            SIGMA_ENRICHED,
            "unenriched":          SIGMA_UNENRICHED,
            "unenrich_multiplier": _UNENRICH_MULT,
        },
        "cells":              cells_stats,
        "hypothesis_verdicts": hyp,
        "claim_recommendation": hyp["claim_recommendation"],
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v5.json"

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
    print()
    print("=" * 70)
    print(f"V-CGA-FROZEN v5 (2x2x2 Factorial): {overall_verdict}")
    print("=" * 70)
    print()
    print(f"{'Cell':<8} {'N_half C0':>10} {'N_half C1':>10} {'Red%':>7} "
          f"{'p':>8} {'d':>7} {'M2':>5} {'M4':>5} {'Day1-Δpp':>9} {'err_C0':>8} {'err_C1':>8}")
    print("-" * 90)
    for cell_name, st in cells_stats.items():
        m4 = st["m4"]
        d1 = st["day1_accuracy"]
        print(f"{cell_name:<8} "
              f"{st['n_half_c0']:>10.1f} {st['n_half_c1']:>10.1f} "
              f"{st['reduction_pct']:>7.1f}% "
              f"{st['p_value']:>8.4f} {st['cohens_d']:>7.3f} "
              f"{'PASS' if st['m2_pass'] else 'fail':>5} "
              f"{'PASS' if m4['pass'] else 'fail':>5} "
              f"{d1['delta_pp']:>+9.1f} "
              f"{m4['starting_error_c0']:>8.3f} {m4['starting_error_c1']:>8.3f}")
    print()

    for h, info in [("H1", hyp["H1"]), ("H2", hyp["H2"]), ("H3", hyp["H3"])]:
        print(f"{h} [{info['verdict']:>9}]:  {info['evidence']}")
    print()
    print(f"Claim recommendation: {hyp['claim_recommendation']}")

    if overall_verdict == "PASS":
        print("\nCLAIM-62/63 -> UNCONDITIONAL. Empirical Bayes bootstrap validated.")
    else:
        a1b1 = cells_stats["A1_B1"]
        a1b0 = cells_stats["A1_B0"]
        print("\nCLAIM-62/63 remains CONDITIONAL.")
        print(f"  Best cell (A1xB1): d={a1b1['cohens_d']:.3f}, "
              f"p={a1b1['p_value']:.4f}, "
              f"M4={'PASS' if a1b1['m4']['pass'] else 'FAIL'}, "
              f"Day-1 delta={a1b1['day1_accuracy']['delta_pp']:+.1f}pp")
        print(f"  H2 discrimination effect: A1xB1 d={a1b1['cohens_d']:.3f} "
              f"vs A1xB0 d={a1b0['cohens_d']:.3f}")
    print("=" * 70)

    return results

if __name__ == "__main__":
    main()
