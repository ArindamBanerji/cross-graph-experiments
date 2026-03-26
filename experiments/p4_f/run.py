"""
P4-F: Adversarial quality degradation detection.
Validates that conservation monitor detects gradual analyst quality
degradation before >5pp centroid damage accumulates.

q_bar degrades linearly from 0.85 to 0.40 over 500 decisions.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS     = 30
N_DECISIONS = 500
THETA_MIN   = 0.467
TAU         = 0.1
ETA_CONFIRM = 0.05
ETA_OVERRIDE = 0.01
ALPHA       = 0.80
V           = 500          # alert volume proxy

Q_START     = 0.85
Q_END       = 0.40
BASELINE_WINDOW = 50       # decisions used to estimate baseline accuracy
ROLLING_WINDOW  = 50       # rolling window for q_bar estimate
DAMAGE_THRESHOLD = -5.0    # pp drop from baseline = centroid damage
FALSE_POSITIVE_Q = 0.75    # AMBER before q_bar drops below this = false positive

SEEDS_30 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
            17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624]

# ---------------------------------------------------------------------------
# A1×B1 SOC healthcare geometry
# ---------------------------------------------------------------------------
FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]
N_CATS    = len(CATEGORIES)
N_ACTS    = len(ACTIONS)
N_FACTORS = len(FACTOR_NAMES)
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

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

def build_mu_star():
    mu = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = build_mu_star()

def build_gt_dist():
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt

GT_DIST = build_gt_dist()
CAT_WEIGHTS = np.ones(N_CATS) / N_CATS


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------
def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    mu0 = MU_STAR.copy() + rng.uniform(-0.005, 0.005, MU_STAR.shape)
    np.clip(mu0, 0, 1, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                            eta_override=ETA_OVERRIDE)

    correct_flags  = []   # 1/0 per decision
    q_history      = []   # actual effective quality per decision

    t_detect = None  # first AMBER fire
    t_damage = None  # first >5pp centroid damage from baseline
    amber_fired_early = False  # fired before q_bar dropped below FALSE_POSITIVE_Q

    for t in range(N_DECISIONS):
        # Effective quality at this decision (linear degradation)
        q_eff = Q_START - (Q_START - Q_END) * (t / N_DECISIONS)

        # Category and ground-truth action
        cat_idx = int(rng.choice(N_CATS, p=CAT_WEIGHTS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)

        result = scorer.score(f, cat_idx)
        correct_flags.append(int(result.action_index == gt_act))

        # Analyst feedback: adversarial with probability (1 - q_eff)
        q_history.append(q_eff)
        if rng.random() < q_eff:
            # Clean: confirms correct action
            scorer.update(f=f, category_index=cat_idx, action_index=gt_act,
                          correct=True, gt_action_index=gt_act)
        else:
            # Adversarial: confirms wrong action as correct
            wrong_choices = [a for a in range(N_ACTS) if a != gt_act]
            wrong_act = int(rng.choice(wrong_choices))
            scorer.update(f=f, category_index=cat_idx, action_index=wrong_act,
                          correct=True, gt_action_index=None)

        # ---------------------------------------------------------------
        # Conservation check (rolling ROLLING_WINDOW q̄)
        # ---------------------------------------------------------------
        if t >= ROLLING_WINDOW - 1:
            q_rolling = float(np.mean(q_history[t - ROLLING_WINDOW + 1: t + 1]))
            cons_val  = ALPHA * q_rolling * (V / N_DECISIONS)
            if cons_val < THETA_MIN and t_detect is None:
                t_detect = t
                # False-positive check: if q_rolling still above threshold
                if q_rolling >= FALSE_POSITIVE_Q:
                    amber_fired_early = True

        # ---------------------------------------------------------------
        # Damage check: once we have baseline
        # ---------------------------------------------------------------
        if t >= BASELINE_WINDOW:
            baseline_acc = float(np.mean(correct_flags[:BASELINE_WINDOW])) * 100.0
            rolling_acc  = float(np.mean(correct_flags[max(0, t - 49): t + 1])) * 100.0
            damage_pp    = rolling_acc - baseline_acc
            if damage_pp < DAMAGE_THRESHOLD and t_damage is None:
                t_damage = t

    baseline_acc = float(np.mean(correct_flags[:BASELINE_WINDOW])) * 100.0
    final_acc    = float(np.mean(correct_flags[-50:])) * 100.0

    # Accuracy at detection and damage times
    acc_at_detect = None
    acc_at_damage = None
    if t_detect is not None:
        w = correct_flags[max(0, t_detect - 49): t_detect + 1]
        acc_at_detect = float(np.mean(w)) * 100.0
    if t_damage is not None:
        w = correct_flags[max(0, t_damage - 49): t_damage + 1]
        acc_at_damage = float(np.mean(w)) * 100.0

    # Lead time: positive = AMBER before damage, negative = missed
    if t_detect is not None and t_damage is not None:
        lead_time = t_damage - t_detect
    elif t_detect is not None and t_damage is None:
        # AMBER fired but 5pp damage never accumulated
        lead_time = N_DECISIONS - t_detect   # AMBER fired with no damage = big positive
    else:
        # damage occurred without detect (or neither)
        lead_time = None

    return {
        "seed":              seed,
        "baseline_acc":      baseline_acc,
        "final_acc":         final_acc,
        "t_detect":          t_detect,
        "t_damage":          t_damage,
        "lead_time":         lead_time,
        "acc_at_detect":     acc_at_detect,
        "acc_at_damage":     acc_at_damage,
        "amber_fired_early": amber_fired_early,
    }


def main():
    print("P4-F — Adversarial Quality Degradation Detection (N=30, GAE 0.7.8):")
    print("  Running 30 seeds × 500 decisions ...", flush=True)

    results = [run_seed(s) for s in SEEDS_30]

    baseline_accs = [r["baseline_acc"] for r in results]
    mean_baseline = float(np.mean(baseline_accs))

    # Sanity 3: baseline should be 88-92%
    if mean_baseline < 80.0:
        print(f"  SANITY FAIL: Baseline accuracy {mean_baseline:.1f}% < 80%. STOP.")
        sys.exit(1)

    # Sanity 1: AMBER never fires
    amber_never = all(r["t_detect"] is None for r in results)
    if amber_never:
        print("  SANITY FAIL: AMBER never fires in any seed. STOP.")
        sys.exit(1)

    # Sanity 2: AMBER fires in first 50 decisions for > 50% of seeds
    early_fires = sum(1 for r in results if r["t_detect"] is not None and r["t_detect"] < 50)
    if early_fires / N_SEEDS > 0.50:
        print(f"  WARNING: AMBER fires in first 50 decisions in {early_fires/N_SEEDS:.0%} of seeds — too sensitive.")

    # Miss rate: t_detect > t_damage (or t_detect is None but t_damage exists)
    miss_count = 0
    for r in results:
        if r["t_damage"] is not None:
            if r["t_detect"] is None or r["t_detect"] > r["t_damage"]:
                miss_count += 1
    miss_rate = miss_count / N_SEEDS

    # Lead times (only seeds where both events occurred — or detect with no damage)
    lead_times = [r["lead_time"] for r in results if r["lead_time"] is not None]
    if lead_times:
        lt_p10 = float(np.percentile(lead_times, 10))
        lt_p50 = float(np.percentile(lead_times, 50))
        lt_p90 = float(np.percentile(lead_times, 90))
    else:
        lt_p10 = lt_p50 = lt_p90 = float("nan")

    # False positive rate: AMBER fired before q_bar < 0.75
    fp_count = sum(1 for r in results if r["amber_fired_early"])
    fp_rate  = fp_count / N_SEEDS

    # Accuracy at detect / damage
    detect_accs = [r["acc_at_detect"] for r in results if r["acc_at_detect"] is not None]
    damage_accs = [r["acc_at_damage"] for r in results if r["acc_at_damage"] is not None]
    mean_acc_detect = float(np.mean(detect_accs)) if detect_accs else float("nan")
    mean_acc_damage = float(np.mean(damage_accs)) if damage_accs else float("nan")

    # Gate evaluation
    gate_lead = lt_p90 <= 50.0
    gate_miss  = miss_rate <= 0.10
    both_gates = gate_lead and gate_miss

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = {
        "experiment":             "P4-F",
        "gae_version":            "0.7.8",
        "n_seeds":                N_SEEDS,
        "n_decisions":            N_DECISIONS,
        "lead_time_p10":          round(lt_p10, 1),
        "lead_time_p50":          round(lt_p50, 1),
        "lead_time_p90":          round(lt_p90, 1),
        "miss_rate":              round(miss_rate, 4),
        "false_positive_rate":    round(fp_rate, 4),
        "amber_never_fires":      amber_never,
        "baseline_accuracy":      round(mean_baseline, 2),
        "accuracy_at_t_detect_mean": round(mean_acc_detect, 2) if not np.isnan(mean_acc_detect) else None,
        "accuracy_at_t_damage_mean": round(mean_acc_damage, 2) if not np.isnan(mean_acc_damage) else None,
        "gate_lead_time_pass":    gate_lead,
        "gate_miss_rate_pass":    gate_miss,
        "both_gates_pass":        both_gates,
        "n_seeds_amber_fires":    sum(1 for r in results if r["t_detect"] is not None),
        "n_seeds_damage_occurs":  sum(1 for r in results if r["t_damage"] is not None),
        "n_seeds_lead_time_computed": len(lead_times),
    }
    out_path = REPO / "experiments" / "p4_f" / "results" / "results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print()
    print(f"P4-F — Adversarial Quality Degradation Detection (N=30, GAE 0.7.8):")
    print(f"  Baseline accuracy: {mean_baseline:.1f}%")
    print(f"  AMBER fires: {out['n_seeds_amber_fires']}/30 seeds | Damage >5pp: {out['n_seeds_damage_occurs']}/30 seeds")
    print(f"  Lead time (decisions before 5pp damage):")
    print(f"    p10={lt_p10:.0f}  p50={lt_p50:.0f}  p90={lt_p90:.0f}")
    print(f"    [gate: p90 <= 50 decisions] -> {'PASS' if gate_lead else 'FAIL'}")
    print(f"  Miss rate: {miss_rate:.1%} [gate: ≤10%] → {'PASS' if gate_miss else 'FAIL'}")
    print(f"  False positive rate: {fp_rate:.1%}")
    print(f"  AMBER never fires: {'yes' if amber_never else 'no'}")
    print(f"  Both gates pass: {'PASS' if both_gates else 'FAIL'}")
    print(f"  Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
