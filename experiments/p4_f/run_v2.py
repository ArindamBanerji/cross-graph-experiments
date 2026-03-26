"""
P4-F v2 — Layer 2 active (ConservationMonitor, GAE 0.7.9).

Identical design to original P4-F plus ConservationMonitor Layer 2.
Records T_layer1 (conservation law AMBER) and T_layer2 (YELLOW trend)
per seed to compare lead times vs T_damage (first >5pp accuracy drop).
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
from gae.convergence import ConservationMonitor

# ---------------------------------------------------------------------------
# Parameters — identical to original P4-F
# ---------------------------------------------------------------------------
N_SEEDS      = 30
N_DECISIONS  = 500
THETA_MIN    = 0.467
TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
ALPHA        = 0.80
V            = 500       # alert volume proxy

Q_START      = 0.85
Q_END        = 0.40
BASELINE_WINDOW  = 50
ROLLING_WINDOW   = 50   # for accuracy damage tracking
DAMAGE_THRESHOLD = -5.0  # pp drop from baseline
CONS_WINDOW      = 50   # rolling window for Layer 1 q̄ estimate

SEEDS_30 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
            17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624]

# ---------------------------------------------------------------------------
# A1×B1 SOC healthcare geometry — identical to original P4-F
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

GT_DIST     = build_gt_dist()
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

    # ConservationMonitor: Layer 2 auto-runs via record_quality()
    # Layer 1 requires external computation — we call update_conservation_signal()
    monitor = ConservationMonitor()

    correct_flags  = []
    quality_flags  = []   # binary 0/1 per decision (fed to monitor)

    t_layer1 = None   # first AMBER (Layer 1)
    t_layer2 = None   # first YELLOW (Layer 2)
    t_damage = None   # first >5pp accuracy drop from baseline

    for t in range(N_DECISIONS):
        q_eff = Q_START - (Q_START - Q_END) * (t / N_DECISIONS)

        cat_idx = int(rng.choice(N_CATS, p=CAT_WEIGHTS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)

        result = scorer.score(f, cat_idx)
        correct_flags.append(int(result.action_index == gt_act))

        # Quality draw: Bernoulli(q_eff)
        is_clean = rng.random() < q_eff
        quality_flags.append(1 if is_clean else 0)

        # Analyst feedback
        if is_clean:
            scorer.update(f=f, category_index=cat_idx, action_index=gt_act,
                          correct=True, gt_action_index=gt_act)
        else:
            wrong_choices = [a for a in range(N_ACTS) if a != gt_act]
            wrong_act = int(rng.choice(wrong_choices))
            scorer.update(f=f, category_index=cat_idx, action_index=wrong_act,
                          correct=True, gt_action_index=None)

        # Feed binary quality to ConservationMonitor → Layer 2 auto-updates
        monitor.record_quality(float(quality_flags[-1]))

        # Layer 2: capture first YELLOW fire
        if t_layer2 is None and monitor.yellow_warning:
            t_layer2 = t

        # Layer 1: rolling q̄ over CONS_WINDOW, compute α·q̄·(V/N) vs θ_min
        if len(quality_flags) >= CONS_WINDOW:
            q_rolling = float(np.mean(quality_flags[-CONS_WINDOW:]))
            cons_val  = ALPHA * q_rolling * (V / N_DECISIONS)
            if cons_val < THETA_MIN:
                if t_layer1 is None:
                    t_layer1 = t
                monitor.update_conservation_signal('AMBER')
            else:
                monitor.update_conservation_signal('GREEN')

        # T_damage: rolling accuracy drops >5pp below baseline
        if t >= BASELINE_WINDOW:
            baseline_acc = float(np.mean(correct_flags[:BASELINE_WINDOW])) * 100.0
            rolling_acc  = float(np.mean(correct_flags[max(0, t - 49): t + 1])) * 100.0
            if (rolling_acc - baseline_acc) < DAMAGE_THRESHOLD and t_damage is None:
                t_damage = t

    baseline_acc = float(np.mean(correct_flags[:BASELINE_WINDOW])) * 100.0

    # Lead times: T_damage - T_fire (positive = fired before damage)
    def lead(t_fire):
        if t_fire is not None and t_damage is not None:
            return t_damage - t_fire
        elif t_fire is not None and t_damage is None:
            return N_DECISIONS - t_fire   # fired but no damage → large positive
        return None   # never fired (or damage without fire → miss)

    return {
        "seed":         seed,
        "baseline_acc": baseline_acc,
        "t_layer1":     t_layer1,
        "t_layer2":     t_layer2,
        "t_damage":     t_damage,
        "lead_layer1":  lead(t_layer1),
        "lead_layer2":  lead(t_layer2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    print(f"P4-F v2 -- Layer 2 active (N=30, GAE {gae.__version__})")
    print("  Running 30 seeds x 500 decisions ...", flush=True)

    results = [run_seed(s) for s in SEEDS_30]

    baseline_accs = [r["baseline_acc"] for r in results]
    mean_baseline  = float(np.mean(baseline_accs))

    if mean_baseline < 80.0:
        print(f"  SANITY FAIL: baseline {mean_baseline:.1f}% < 80%. STOP.")
        sys.exit(1)

    def layer_stats(lead_key, t_fire_key):
        leads = [r[lead_key] for r in results if r[lead_key] is not None]
        # Miss: damage occurred but fire was None or after damage
        miss_count = sum(
            1 for r in results
            if r["t_damage"] is not None and (
                r[t_fire_key] is None or r[t_fire_key] > r["t_damage"]
            )
        )
        miss_rate = miss_count / N_SEEDS
        p10 = float(np.percentile(leads, 10)) if leads else float("nan")
        p50 = float(np.percentile(leads, 50)) if leads else float("nan")
        p90 = float(np.percentile(leads, 90)) if leads else float("nan")
        return p10, p50, p90, miss_rate

    l1_p10, l1_p50, l1_p90, l1_miss = layer_stats("lead_layer1", "t_layer1")
    l2_p10, l2_p50, l2_p90, l2_miss = layer_stats("lead_layer2", "t_layer2")

    # Layer 2 FP rate: fires before decision 50 (q_eff still ~0.85)
    l2_fp_count = sum(1 for r in results if r["t_layer2"] is not None and r["t_layer2"] < 50)
    l2_fp_rate  = l2_fp_count / N_SEEDS

    l2_gate_lead = not np.isnan(l2_p90) and l2_p90 >= 0.0
    l2_gate_miss = l2_miss <= 0.10
    l2_gate_pass = l2_gate_lead and l2_gate_miss

    # Save
    out = {
        "experiment":             "P4-F-v2",
        "gae_version":            gae.__version__,
        "layer2_active":          True,
        "n_seeds":                N_SEEDS,
        "baseline_accuracy":      round(mean_baseline, 2),
        "layer1_lead_time_p10":   round(l1_p10, 1),
        "layer1_lead_time_p50":   round(l1_p50, 1),
        "layer1_lead_time_p90":   round(l1_p90, 1),
        "layer1_miss_rate":       round(l1_miss, 4),
        "layer2_lead_time_p10":   round(l2_p10, 1),
        "layer2_lead_time_p50":   round(l2_p50, 1),
        "layer2_lead_time_p90":   round(l2_p90, 1),
        "layer2_miss_rate":       round(l2_miss, 4),
        "layer2_false_positive_rate": round(l2_fp_rate, 4),
        "layer2_gate_pass":       l2_gate_pass,
        "n_seeds_layer1_fires":   sum(1 for r in results if r["t_layer1"] is not None),
        "n_seeds_layer2_fires":   sum(1 for r in results if r["t_layer2"] is not None),
        "n_seeds_damage_occurs":  sum(1 for r in results if r["t_damage"] is not None),
    }
    out_path = REPO / "experiments" / "p4_f" / "results" / "results_v2.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {out_path}")

    # Report
    print()
    print(f"P4-F v2 -- Layer 2 active (N=30, GAE {gae.__version__}):")
    print(f"  Baseline accuracy: {mean_baseline:.1f}%")
    print(f"  Layer 1 (conservation law):")
    print(f"    Lead time p50={l1_p50:.0f}d p90={l1_p90:.0f}d miss={l1_miss:.1%} [unchanged from original]")
    print(f"  Layer 2 (YELLOW trend detection):")
    print(f"    Lead time p10={l2_p10:.0f}d p50={l2_p50:.0f}d p90={l2_p90:.0f}d")
    print(f"    Miss rate: {l2_miss:.1%} [gate: <=10%] -> {'PASS' if l2_gate_miss else 'FAIL'}")
    print(f"    False positive rate: {l2_fp_rate:.1%}")
    print(f"    Gate (p90>=0): {'PASS' if l2_gate_lead else 'FAIL'}")
    print("Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
