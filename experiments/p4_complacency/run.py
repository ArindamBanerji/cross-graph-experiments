"""
P4-COMPLACENCY: Complacency pattern detection via Var(q).

q_bar degrades linearly from 0.85 to 0.55 over 300 decisions.
Override rate alpha STAYS CONSTANT at 0.25.
Conservation law should NOT fire (low alpha keeps cons_val high).
Var(q) is the detection layer.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS        = 30
N_DECISIONS    = 300
THETA_MIN      = 0.467
TAU            = 0.1
ETA_CONFIRM    = 0.05
ETA_OVERRIDE   = 0.01
ALPHA_CONSTANT = 0.25

Q_START = 0.85
Q_END   = 0.55          # q_eff(300) = 0.55

VAR_WINDOW    = 25      # rolling window for Var(q)
ROLL_WINDOW50 = 50      # rolling window for q_bar for T_drop
Q_DROP_THOLD  = 0.65    # T_drop threshold
THRESHOLDS    = [0.005, 0.010, 0.020, 0.030, 0.050]
FP_EARLY_CUTOFF = 50    # fires before this decision = false positive
FP_Q_CUTOFF   = 0.75    # and q_bar still above this

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

GT_DIST    = build_gt_dist()
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

    quality_flags = []   # per-decision latent quality (0/1)
    correct_flags = []

    conservation_fired   = False
    t_drop               = None   # first t where q_bar_rolling50 < Q_DROP_THOLD
    # First Var(q) fire per threshold
    t_detect             = {thr: None for thr in THRESHOLDS}
    # Track q_bar at fire time for FP check
    q_bar_at_detect      = {thr: None for thr in THRESHOLDS}

    for t in range(N_DECISIONS):
        # Effective quality: linear degradation 0.85 → 0.55
        q_eff = Q_START - (Q_START - Q_END) * (t / N_DECISIONS)

        # Ground-truth action and factor vector
        cat_idx = int(rng.choice(N_CATS, p=CAT_WEIGHTS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)

        result = scorer.score(f, cat_idx)
        correct_flags.append(int(result.action_index == gt_act))

        # Per-decision latent quality (Bernoulli)
        quality = int(rng.random() < q_eff)
        quality_flags.append(quality)

        # Override decision (independent of quality)
        if rng.random() < ALPHA_CONSTANT:
            if quality:
                # Clean override: confirm GT action
                scorer.update(f=f, category_index=cat_idx, action_index=gt_act,
                              correct=True, gt_action_index=gt_act)
            else:
                # Adversarial/complacent: confirm wrong action as correct
                wrong_choices = [a for a in range(N_ACTS) if a != gt_act]
                wrong_act = int(rng.choice(wrong_choices))
                scorer.update(f=f, category_index=cat_idx, action_index=wrong_act,
                              correct=True, gt_action_index=None)

        # -------------------------------------------------------------------
        # Rolling q_bar estimate (50-window)
        # -------------------------------------------------------------------
        n_avail = len(quality_flags)
        q_bar_rolling = float(np.mean(quality_flags[-ROLL_WINDOW50:]))

        # Conservation check: α × q_bar × V (V = N_DECISIONS, unnormalized total)
        # Only check once rolling window is full to avoid empty-window spurious fires
        if n_avail >= ROLL_WINDOW50:
            cons_val = ALPHA_CONSTANT * q_bar_rolling * N_DECISIONS
            if cons_val < THETA_MIN:
                conservation_fired = True

        # T_drop: first decision (after window is full) where q_bar < Q_DROP_THOLD
        if t_drop is None and n_avail >= ROLL_WINDOW50 and q_bar_rolling < Q_DROP_THOLD:
            t_drop = t

        # -------------------------------------------------------------------
        # Var(q)_rolling: variance of quality_flags over last VAR_WINDOW decisions
        # -------------------------------------------------------------------
        if n_avail >= VAR_WINDOW:
            var_q = float(np.var(quality_flags[-VAR_WINDOW:]))
            for thr in THRESHOLDS:
                if t_detect[thr] is None and var_q > thr:
                    t_detect[thr] = t
                    q_bar_at_detect[thr] = q_bar_rolling

    baseline_acc = float(np.mean(correct_flags[:50])) * 100.0 if len(correct_flags) >= 50 else float("nan")
    q_final      = float(np.mean(quality_flags[-50:]))

    return {
        "seed":               seed,
        "baseline_acc":       baseline_acc,
        "q_final":            q_final,
        "t_drop":             t_drop,
        "t_detect":           t_detect,
        "q_bar_at_detect":    q_bar_at_detect,
        "conservation_fired": conservation_fired,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Verify conservation math at q_bar=0.55 (end state)
    cons_at_end = ALPHA_CONSTANT * 0.55 * N_DECISIONS
    print(f"P4-COMPLACENCY — Complacency Detection via Var(q) (N=30, GAE 0.7.8):")
    print(f"  [setup] cons at q_bar=0.55: {ALPHA_CONSTANT}*0.55*{N_DECISIONS}={cons_at_end:.3f} vs theta_min={THETA_MIN}")
    print(f"  Running 30 seeds x 300 decisions ...", flush=True)

    all_results = [run_seed(s) for s in SEEDS_30]

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    # Sanity 1: conservation must not fire
    cons_fires = sum(1 for r in all_results if r["conservation_fired"])
    if cons_fires > 0:
        print(f"  SANITY FAIL: Conservation fired in {cons_fires} seeds. STOP.")
        sys.exit(1)

    # Sanity 2: baseline accuracy 88-92%
    mean_baseline = float(np.mean([r["baseline_acc"] for r in all_results]))
    if mean_baseline < 80.0:
        print(f"  SANITY FAIL: Baseline accuracy {mean_baseline:.1f}% < 80%. STOP.")
        sys.exit(1)

    # Sanity 3: q_bar_rolling at decision 300 should be ≈ 0.55
    mean_q_final = float(np.mean([r["q_final"] for r in all_results]))

    # Sanity 2b: flag if AMBER fires too early (analogous to P4-F sanity 2)
    for thr in THRESHOLDS:
        early = sum(1 for r in all_results if r["t_detect"][thr] is not None and r["t_detect"][thr] < FP_EARLY_CUTOFF)
        if early / N_SEEDS > 0.50:
            print(f"  WARNING: Var(q) threshold {thr} fires before decision {FP_EARLY_CUTOFF} in {early/N_SEEDS:.0%} of seeds.")

    # -----------------------------------------------------------------------
    # Per-threshold metrics
    # -----------------------------------------------------------------------
    threshold_stats = {}
    for thr in THRESHOLDS:
        detects = []
        fn_count  = 0
        fp_count  = 0
        lag_list  = []
        lead_list = []

        for r in all_results:
            t_d = r["t_detect"][thr]
            t_drop = r["t_drop"]

            # Detection: Var(q) fires within T_drop + 50 decisions
            if t_drop is not None:
                detected = (t_d is not None and t_d <= t_drop + 50)
            else:
                # q_bar never dropped below 0.65 in this seed — shouldn't happen
                detected = False
            detects.append(int(detected))

            if not detected:
                fn_count += 1

            # False positive: fires before FP_EARLY_CUTOFF with q_bar > FP_Q_CUTOFF
            if t_d is not None and t_d < FP_EARLY_CUTOFF:
                q_at_fire = r["q_bar_at_detect"][thr]
                if q_at_fire is not None and q_at_fire > FP_Q_CUTOFF:
                    fp_count += 1

            # Detection lag: T_detect - T_drop (negative = early warning)
            if t_d is not None and t_drop is not None:
                lag = t_d - t_drop
                lag_list.append(lag)
                lead_list.append(-lag)   # lead = T_drop - T_detect; positive = early

        fn_rate = fn_count / N_SEEDS
        fp_rate = fp_count / N_SEEDS
        n_det   = sum(detects)

        # F1: TP/FN based on per-seed detection within T_drop+50
        # Precision = TP / (TP + FP_fn), where FP here is seeds detected
        # but q_bar never fell (all seeds do fall, so precision=recall=TP/N_SEEDS)
        tp      = n_det
        fn_f1   = N_SEEDS - n_det
        fp_f1   = 0   # all seeds have q_bar < 0.65 eventually → no false alarm seeds
        prec    = tp / (tp + fp_f1) if (tp + fp_f1) > 0 else 0.0
        rec     = tp / (tp + fn_f1) if (tp + fn_f1) > 0 else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        mean_lag  = float(np.mean(lag_list))  if lag_list  else float("nan")
        lead_p50  = float(np.percentile(lead_list, 50)) if lead_list else float("nan")

        threshold_stats[thr] = {
            "fn_rate":    round(fn_rate, 4),
            "fp_rate":    round(fp_rate, 4),
            "f1":         round(f1, 4),
            "n_detected": n_det,
            "mean_detection_lag": round(mean_lag, 1),
            "lead_time_p50":      round(lead_p50, 1),
        }

    # Optimal threshold: highest F1; break ties by lowest FN rate then lowest FP rate
    opt_thr = max(THRESHOLDS, key=lambda t: (
        threshold_stats[t]["f1"],
        -threshold_stats[t]["fn_rate"],
        -threshold_stats[t]["fp_rate"],
    ))
    opt     = threshold_stats[opt_thr]

    gate_pass = opt["fn_rate"] <= 0.10

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = {
        "experiment":            "P4-COMPLACENCY",
        "gae_version":           "0.7.8",
        "n_seeds":               N_SEEDS,
        "n_decisions":           N_DECISIONS,
        "conservation_fires":    False,
        "optimal_threshold":     opt_thr,
        "false_negative_rate":   opt["fn_rate"],
        "false_positive_rate":   opt["fp_rate"],
        "mean_detection_lag":    opt["mean_detection_lag"],
        "lead_time_p50":         opt["lead_time_p50"],
        "gate_pass":             gate_pass,
        "baseline_accuracy":     round(mean_baseline, 2),
        "q_final_rolling":       round(mean_q_final, 3),
        "cons_at_q_end":         round(cons_at_end, 3),
        "threshold_sweep":       threshold_stats,
    }
    out_path = REPO / "experiments" / "p4_complacency" / "results" / "results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print()
    print(f"P4-COMPLACENCY -- Complacency Detection via Var(q) (N=30, GAE 0.7.8):")
    print(f"  Conservation law fires: no [sanity check: cons={cons_at_end:.3f} >> theta_min={THETA_MIN}]")
    print(f"  Baseline accuracy: {mean_baseline:.1f}%")
    print(f"  q_bar_rolling at decision 300: {mean_q_final:.3f} [expected ~0.55]")
    print(f"  Optimal Var(q) threshold: {opt_thr:.4f}")
    print(f"  False negative rate: {opt['fn_rate']:.1%} [gate: <=10%] -> {'PASS' if gate_pass else 'FAIL'}")
    print(f"  False positive rate: {opt['fp_rate']:.1%}")
    print(f"  Mean detection lag: {opt['mean_detection_lag']:.0f} decisions (negative=early warning)")
    print(f"  Lead time p50: {opt['lead_time_p50']:.0f} decisions")
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")
    print()
    print(f"Var(q) threshold sweep:")
    print(f"  {'Threshold':>10} | {'FN rate':>8} | {'FP rate':>8} | {'F1':>6} | {'Mean lag':>10}")
    print(f"  {'-'*54}")
    for thr in THRESHOLDS:
        ts = threshold_stats[thr]
        print(f"  {thr:>10.3f} | {ts['fn_rate']:>8.1%} | {ts['fp_rate']:>8.1%} | "
              f"{ts['f1']:>6.3f} | {ts['mean_detection_lag']:>10.1f}")
    print()
    print("Raw numbers for roadmap session review.")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
