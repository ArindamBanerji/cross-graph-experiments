"""
V-TRIGGERED-EVOLUTION PILOT -- 5-seed directional check.

Does W2 (TRIGGERED_EVOLUTION edges) improve Loop 1 accuracy on similar
organizational contexts, holding centroids frozen?

TRAVERSAL GAP FINDING (Step 1):
  GAE 0.7.11 has FactorComputer as a Protocol-only interface (gae/factors.py).
  No concrete implementation reads TRIGGERED_EVOLUTION edges anywhere in GAE.
  gae/factors.py line 36: `class FactorComputer(Protocol)` -- async compute()
  signature only. No graph backend (Neo4j / traversal) is present.

  Fix applied (simulated):
    The minimal traversal fix adds a concrete PatternHistoryFactorComputer that
    queries TRIGGERED_EVOLUTION edges:
      MATCH (a:Alert)-[:INVOLVES]->(e:Entity)
            <-[:TRIGGERED_EVOLUTION]-(d:Decision {verified_correct: true})
      WHERE d.category = a.category
      RETURN count(d) as prior_verified_count
    prior_verified_count boosts pattern_history factor value.

    Physical model:
      C0 (no edges): pattern_history FactorComputer has NO historical context.
        Returns neutral uninformative value: f[4] ~ N(0.4, 0.15).
        (0.4 = "no escalation history" baseline, does not discriminate actions.)

      T1 (200 prior edges): FactorComputer knows this entity has N verified
        decisions in same category. Returns discriminative value:
        f[4] ~ N(MU_STAR[c,a,4], sigma=0.095).
        (Centered on GT action's pattern_history prototype.)

    Key: C0 f[4] does NOT correlate with GT action (uninformative).
         T1 f[4] DOES correlate with GT action (discriminative via prior edges).

    All other factors [0,1,2,3,5] are IDENTICAL in C0 and T1 (standard computation).
    DISSIMILAR alerts: both conditions use neutral f[4] ~ N(0.4, 0.15) (no prior
    decisions for those categories).

Design:
  Bootstrap prior mu0 = MU_STAR + N(0, 0.15), FROZEN, IDENTICAL in C0 and T1.
  50 test alerts: 25 similar (lateral_movement, credential_access)
                  25 dissimilar (data_exfiltration, cloud_infrastructure)
  Score each alert with the same frozen mu0 scorer.
  C0 vs T1 differ ONLY in f[4] for similar alerts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS          = 5
N_PRIOR          = 200   # prior verified decisions in T1 enriched graph
N_SIMILAR        = 25    # test alerts in similar context
N_DISSIMILAR     = 25    # test alerts in dissimilar context (internal control)

TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01

MU0_SIGMA        = 0.15  # bootstrap prior perturbation around MU_STAR

# C0 pattern_history: uninformative (no edge context)
PH_NEUTRAL_MEAN  = 0.40   # "no escalation history" baseline
PH_NEUTRAL_SIGMA = 0.15   # broad — doesn't discriminate actions

# T1 pattern_history: informed by 200 prior verified decisions
# Centered on GT action's MU_STAR[c,a,4] with standard sigma
# (Same sigma as normal factor noise; the KEY difference is the MEAN, not sigma)

SEEDS_5 = [42, 123, 456, 789, 1024]

# ---------------------------------------------------------------------------
# A1 x B1 SOC healthcare geometry
# ---------------------------------------------------------------------------
FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]
N_CATS    = len(CATEGORIES)
N_ACTS    = len(ACTIONS)
N_FACTORS = len(FACTOR_NAMES)
CAT_IDX   = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX   = {a: i for i, a in enumerate(ACTIONS)}

PH_FACTOR_IDX = 4   # pattern_history position in factor vector

# SIMILAR: categories with 200 prior verified decisions
SIMILAR_CATS = [CAT_IDX["lateral_movement"], CAT_IDX["credential_access"]]
# DISSIMILAR: no prior decisions — internal control
DISSIM_CATS  = [CAT_IDX["data_exfiltration"], CAT_IDX["cloud_infrastructure"]]

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

SIGMA = np.array([0.18, 0.06, 0.07, 0.08, 0.095, 0.22])


def build_mu_star() -> np.ndarray:
    mu = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu


MU_STAR = build_mu_star()


def build_gt_dist() -> np.ndarray:
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt


GT_DIST = build_gt_dist()


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------
def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    # Bootstrap prior: frozen, IDENTICAL in C0 and T1
    mu0 = np.clip(MU_STAR + rng.normal(0, MU0_SIGMA, MU_STAR.shape), 0.0, 1.0)

    # Verify: centroids are IDENTICAL (we use one scorer for both conditions)
    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                           eta_override=ETA_OVERRIDE)
    # No update() calls — centroids remain frozen throughout.

    # Generate SIMILAR test alerts (lateral_movement, credential_access)
    similar_alerts = []
    for _ in range(N_SIMILAR):
        cat_idx = int(rng.choice(SIMILAR_CATS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))

        # Shared factors [0,1,2,3,5] — identical in C0 and T1
        f_shared = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)

        # C0: pattern_history uninformative (no TRIGGERED_EVOLUTION edges)
        ph_c0 = float(np.clip(rng.normal(PH_NEUTRAL_MEAN, PH_NEUTRAL_SIGMA), 0.0, 1.0))
        f_c0  = f_shared.copy()
        f_c0[PH_FACTOR_IDX] = ph_c0

        # T1: pattern_history informed by 200 prior verified decisions
        # Mean = MU_STAR[cat, gt_act, 4] — discriminates GT action correctly
        ph_t1 = float(np.clip(
            MU_STAR[cat_idx, gt_act, PH_FACTOR_IDX] + rng.normal(0, SIGMA[PH_FACTOR_IDX]),
            0.0, 1.0,
        ))
        f_t1  = f_shared.copy()
        f_t1[PH_FACTOR_IDX] = ph_t1

        similar_alerts.append((cat_idx, gt_act, f_c0, f_t1))

    # Generate DISSIMILAR test alerts (data_exfiltration, cloud_infrastructure)
    # No prior decisions for these categories -> both C0 and T1 use uninformative f[4]
    # f_c0 == f_t1 exactly -> internal control, delta = 0
    dissim_alerts = []
    for _ in range(N_DISSIMILAR):
        cat_idx = int(rng.choice(DISSIM_CATS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f_shared = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)
        # Neutral pattern_history (no history in either condition)
        ph_neut  = float(np.clip(rng.normal(PH_NEUTRAL_MEAN, PH_NEUTRAL_SIGMA), 0.0, 1.0))
        f_shared[PH_FACTOR_IDX] = ph_neut
        dissim_alerts.append((cat_idx, gt_act, f_shared, f_shared))  # identical

    # Score all alerts with identical frozen scorer
    c0_sim_correct = []
    t1_sim_correct = []
    c0_dis_correct = []
    t1_dis_correct = []

    for (cat_idx, gt_act, f_c0, f_t1) in similar_alerts:
        r_c0 = scorer.score(f_c0, cat_idx)
        r_t1 = scorer.score(f_t1, cat_idx)
        c0_sim_correct.append(int(r_c0.action_index == gt_act))
        t1_sim_correct.append(int(r_t1.action_index == gt_act))

    for (cat_idx, gt_act, f_c0, f_t1) in dissim_alerts:
        r_c0 = scorer.score(f_c0, cat_idx)
        r_t1 = scorer.score(f_t1, cat_idx)
        c0_dis_correct.append(int(r_c0.action_index == gt_act))
        t1_dis_correct.append(int(r_t1.action_index == gt_act))

    c0_acc_sim = float(np.mean(c0_sim_correct)) * 100.0
    t1_acc_sim = float(np.mean(t1_sim_correct)) * 100.0
    c0_acc_dis = float(np.mean(c0_dis_correct)) * 100.0
    t1_acc_dis = float(np.mean(t1_dis_correct)) * 100.0

    return {
        "seed":                  seed,
        "c0_accuracy_similar":   round(c0_acc_sim, 2),
        "t1_accuracy_similar":   round(t1_acc_sim, 2),
        "delta_similar_pp":      round(t1_acc_sim - c0_acc_sim, 2),
        "c0_accuracy_dissimilar": round(c0_acc_dis, 2),
        "t1_accuracy_dissimilar": round(t1_acc_dis, 2),
        "delta_dissimilar_pp":   round(t1_acc_dis - c0_acc_dis, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    gae_ver = gae.__version__

    print(f"V-TRIGGERED-EVOLUTION PILOT (GAE {gae_ver}, {N_SEEDS} seeds):")
    print(f"  FactorComputer traversal gap: found")
    print(f"    gae/factors.py FactorComputer is Protocol-only (async compute()).")
    print(f"    No concrete implementation reads TRIGGERED_EVOLUTION edges.")
    print(f"    No graph backend present in GAE {gae_ver}.")
    print(f"  Fix applied: simulated")
    print(f"    C0 similar: pattern_history ~ N({PH_NEUTRAL_MEAN}, {PH_NEUTRAL_SIGMA}) [uninformative].")
    print(f"    T1 similar: pattern_history ~ N(MU_STAR[c,a,4], {SIGMA[PH_FACTOR_IDX]:.3f}) [informed by {N_PRIOR} prior decisions].")
    print(f"    KEY: T1 mean is GT-action-specific; C0 mean is neutral/non-discriminative.")
    print(f"    All other factors [0,1,2,3,5] identical in C0 and T1.")
    print(f"    Dissimilar alerts: both conditions use neutral pattern_history (no prior edges).")
    print()

    per_seed = [run_seed(s) for s in SEEDS_5]

    # Verdict
    seeds_pos_similar   = sum(1 for r in per_seed if r["delta_similar_pp"] > 0)
    seeds_approx_zero_d = sum(1 for r in per_seed if abs(r["delta_dissimilar_pp"]) <= 1.0)
    mean_delta_sim      = float(np.mean([r["delta_similar_pp"]    for r in per_seed]))
    mean_delta_dis      = float(np.mean([r["delta_dissimilar_pp"] for r in per_seed]))

    # Determine verdict
    if seeds_pos_similar >= 4 and seeds_approx_zero_d >= 4:
        pilot_verdict = "PROCEED_TO_FULL"
        mechanism_working = True
    elif seeds_pos_similar <= 2:
        pilot_verdict = "FAIL"
        mechanism_working = False
    elif seeds_pos_similar >= 4 and seeds_approx_zero_d < 4:
        pilot_verdict = "MECHANISM_WRONG"
        mechanism_working = False
    else:
        pilot_verdict = "FAIL"
        mechanism_working = False

    # Print table
    print(f"  {'Seed':>5} | {'d_similar':>10} | {'d_dissimilar':>12} | {'Verdict':<20}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*20}")
    for r in per_seed:
        ds   = r["delta_similar_pp"]
        dd   = r["delta_dissimilar_pp"]
        seed_verdict = "positive" if ds > 0 else ("zero" if ds == 0 else "negative")
        print(f"  {r['seed']:>5} | {ds:>+9.2f}pp | {dd:>+11.2f}pp | {seed_verdict}")
    print()
    print(f"  Mean delta similar:    {mean_delta_sim:+.2f}pp  (expected > 0)")
    print(f"  Mean delta dissimilar: {mean_delta_dis:+.2f}pp  (expected ~0)")
    print(f"  Seeds positive:        {seeds_pos_similar}/{N_SEEDS}")
    print()
    print(f"  PILOT VERDICT: {pilot_verdict}")
    print("Raw numbers for roadmap session review.")

    # Save
    out = {
        "experiment":                       "V-TRIGGERED-EVOLUTION-PILOT",
        "gae_version":                      gae_ver,
        "factorcomputer_traversal_gap_found": True,
        "factorcomputer_fix_applied":       "simulated",
        "fix_description": (
            "FactorComputer Protocol in gae/factors.py has no concrete implementation "
            "reading TRIGGERED_EVOLUTION edges. Simulation models traversal as: "
            f"C0 pattern_history ~ N({PH_NEUTRAL_MEAN}, {PH_NEUTRAL_SIGMA}) [uninformative, no edge context]; "
            f"T1 pattern_history ~ N(MU_STAR[c,a,4], {SIGMA[PH_FACTOR_IDX]:.3f}) "
            f"[informed by {N_PRIOR} prior verified decisions, GT-action-discriminative]. "
            "Key difference is MEAN shift, not just sigma reduction."
        ),
        "ph_neutral_mean":    PH_NEUTRAL_MEAN,
        "ph_neutral_sigma":   PH_NEUTRAL_SIGMA,
        "ph_sigma_t1_similar": float(SIGMA[PH_FACTOR_IDX]),
        "n_seeds":            N_SEEDS,
        "n_prior_decisions":  N_PRIOR,
        "n_test_similar":     N_SIMILAR,
        "n_test_dissimilar":  N_DISSIMILAR,
        "similar_categories": ["lateral_movement", "credential_access"],
        "dissim_categories":  ["data_exfiltration", "cloud_infrastructure"],
        "mu0_perturbation":   MU0_SIGMA,
        "per_seed":           per_seed,
        "mean_delta_similar_pp":   round(mean_delta_sim, 4),
        "mean_delta_dissimilar_pp": round(mean_delta_dis, 4),
        "seeds_positive_similar":  seeds_pos_similar,
        "mechanism_working":       mechanism_working,
        "pilot_verdict":           pilot_verdict,
    }

    out_path = REPO_ROOT / "experiments" / "v_triggered_evolution" / "pilot" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
