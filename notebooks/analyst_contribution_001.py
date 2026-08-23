"""
ANALYST-CONTRIBUTION-001 — per-analyst OLS variance as Level 3 detector.

MAP v4.2 design. N_DECISIONS=1200, N_WARMUP=400, N_ANALYSTS=10, N_SEEDS=30.
VAR_OLS_THRESHOLD=0.01.

Key fix from V-OLS-ANALYST (GATE FAIL): increase N_DECISIONS from 300 to 1200
so each analyst accumulates 20 overrides in measurement window (400-1200),
suppressing sampling noise below threshold.

Expected per-analyst OLS SE ≈ sqrt(0.2/20) / 0.72 ≈ 0.14 — but true bimodal
spread is 0.49 (OLS 1.25 vs 0.76), so signal >> noise for Var detection.
At VAR_OLS_THRESHOLD=0.01: bimodal true Var ≈ 0.059 >> 0.01 ✓.

Save: notebooks/analyst_contribution_001_results.json
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile
from gae.convergence import OLSMonitor

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_DECISIONS       = 1200
N_WARMUP          = 400
N_ANALYSTS        = 10
N_SEEDS           = 30
ALPHA             = 0.25
VAR_OLS_THRESHOLD = 0.01
MIN_OVERRIDES     = 5    # per analyst in measurement window

Q_HIGH    = 0.90
Q_LOW     = 0.55
Q_UNIFORM = 0.65

TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
MU0_SIGMA    = 0.30
OLS_WINDOW   = 30        # rolling window for Level 2 aggregate OLS

SEEDS_30 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
            17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624]

# ---------------------------------------------------------------------------
# SOC geometry
# ---------------------------------------------------------------------------
FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
ACTIONS      = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES   = ["credential_access", "threat_intel_match", "lateral_movement",
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
# Single seed run
# ---------------------------------------------------------------------------
def run_seed(condition: str, seed: int, analyst_qualities: list) -> dict:
    rng = np.random.default_rng(seed)

    mu0 = MU_STAR.copy() + rng.normal(0, MU0_SIGMA, MU_STAR.shape)
    np.clip(mu0, 0.0, 1.0, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                            eta_override=ETA_OVERRIDE)

    noise = np.full(N_FACTORS, 0.15)

    # ---- Level 2: rolling aggregate OLS -> OLSMonitor (full 1200 decisions) ----
    ols_mon   = OLSMonitor()
    ov_q_roll = deque(maxlen=OLS_WINDOW)
    ac_q_roll = deque(maxlen=OLS_WINDOW)

    # ---- Level 3: per-analyst cumulative counts in measurement window ----
    # measurement window = decisions [N_WARMUP, N_DECISIONS)
    analyst_n_ov   = np.zeros(N_ANALYSTS, dtype=int)
    analyst_n_corr = np.zeros(N_ANALYSTS, dtype=int)
    n_accepted_meas      = 0
    n_corr_accepted_meas = 0

    for t in range(N_DECISIONS):
        cat_idx = int(rng.choice(N_CATS, p=CAT_WEIGHTS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, 1, N_FACTORS) * noise,
                    0.0, 1.0)

        result  = scorer.score(f, cat_idx)
        ai_act  = result.action_index
        ai_corr = (ai_act == gt_act)

        analyst_id  = int(rng.integers(0, N_ANALYSTS))
        is_override = (rng.random() < ALPHA)

        if is_override:
            q_analyst    = analyst_qualities[analyst_id]
            analyst_corr = (rng.random() < q_analyst)

            if analyst_corr:
                scorer.update(f=f, category_index=cat_idx,
                              action_index=gt_act, correct=True,
                              gt_action_index=gt_act)
            else:
                wrong       = [a for a in range(N_ACTS) if a != gt_act]
                analyst_act = int(rng.choice(wrong))
                scorer.update(f=f, category_index=cat_idx,
                              action_index=analyst_act, correct=True,
                              gt_action_index=None)

            ov_q_roll.append(1.0 if analyst_corr else 0.0)

            # Level 3: accumulate in measurement window only
            if t >= N_WARMUP:
                analyst_n_ov[analyst_id]   += 1
                if analyst_corr:
                    analyst_n_corr[analyst_id] += 1
        else:
            ac_q_roll.append(1.0 if ai_corr else 0.0)
            if t >= N_WARMUP:
                n_accepted_meas += 1
                if ai_corr:
                    n_corr_accepted_meas += 1

        # Level 2: feed rolling aggregate OLS to OLSMonitor each step
        if len(ov_q_roll) >= 5 and len(ac_q_roll) >= 5:
            agg_ols_roll = (float(np.mean(ov_q_roll)) /
                            max(float(np.mean(ac_q_roll)), 0.01))
            ols_mon.update(agg_ols_roll)

    # ---- End-of-run: compute Level 3 metrics from measurement window ----
    ai_acc_meas = (n_corr_accepted_meas / n_accepted_meas
                   if n_accepted_meas > 0 else float("nan"))

    analyst_ols_list = []
    for i in range(N_ANALYSTS):
        if analyst_n_ov[i] >= MIN_OVERRIDES:
            ols_i = (analyst_n_corr[i] / analyst_n_ov[i]) / max(ai_acc_meas, 0.01)
            analyst_ols_list.append(float(ols_i))

    n_analysts_qualified = len(analyst_ols_list)

    var_ols   = float(np.var(analyst_ols_list)) if n_analysts_qualified >= 2 else float("nan")
    agg_ols   = float(np.mean(analyst_ols_list)) if n_analysts_qualified >= 1 else float("nan")

    # Level 3 fires: Var > threshold AND aggregate looks healthy
    if n_analysts_qualified >= 2 and not np.isnan(var_ols) and not np.isnan(agg_ols):
        l3_fires = (var_ols > VAR_OLS_THRESHOLD) and (agg_ols > 0.95)
    else:
        l3_fires = False

    # Level 2 fires: OLSMonitor yellow_warning at end of run
    l2_fires = bool(ols_mon.yellow_warning)

    return {
        "condition":            condition,
        "seed":                 seed,
        "l3_fires":             bool(l3_fires),
        "l2_fires":             bool(l2_fires),
        "var_ols":              round(var_ols, 6) if not np.isnan(var_ols) else None,
        "agg_ols":              round(agg_ols, 4) if not np.isnan(agg_ols) else None,
        "ai_acc_meas":          round(float(ai_acc_meas), 4) if not np.isnan(ai_acc_meas) else None,
        "n_analysts_qualified": n_analysts_qualified,
        "analyst_n_overrides":  analyst_n_ov.tolist(),
        "plateau_reached":      bool(ols_mon.baseline_frozen),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    assert gae.__version__ == "0.7.18", \
        f"Expected GAE 0.7.18, got {gae.__version__}"

    print(f"ANALYST-CONTRIBUTION-001 (N={N_SEEDS}, N_DECISIONS={N_DECISIONS}, "
          f"GAE {gae.__version__}):")
    print()

    # Pre-calculations
    meas_window   = N_DECISIONS - N_WARMUP
    ov_per_analyst = meas_window * ALPHA / N_ANALYSTS
    se_at_20 = float(np.sqrt(0.65 * 0.35 / 20) / 0.72)  # using q=0.65, AI_acc=0.72
    print("Pre-calculations:")
    print(f"  Measurement window: decisions {N_WARMUP}-{N_DECISIONS} ({meas_window} decisions)")
    print(f"  Overrides/analyst in measurement window: {ov_per_analyst:.1f}")
    print(f"  OLS estimate SE at N=20 overrides: {se_at_20:.3f}")
    print(f"  True bimodal OLS spread: {Q_HIGH/0.72:.2f} vs {Q_LOW/0.72:.2f} "
          f"(spread={Q_HIGH/0.72 - Q_LOW/0.72:.2f})")
    print(f"  Expected Var(OLS_i) bimodal: "
          f"{np.var([Q_HIGH/0.72]*5 + [Q_LOW/0.72]*5):.4f} >> threshold={VAR_OLS_THRESHOLD}")
    print()

    q_bimodal = [Q_HIGH if i < 5 else Q_LOW for i in range(N_ANALYSTS)]
    q_uniform  = [Q_UNIFORM] * N_ANALYSTS

    # ---- Condition A ----
    print("  Running Condition A (bimodal)...", flush=True)
    results_a = [run_seed("A", s, q_bimodal) for s in SEEDS_30]

    # ---- Condition B ----
    print("  Running Condition B (uniform)...", flush=True)
    results_b = [run_seed("B", s, q_uniform) for s in SEEDS_30]

    # ---- Aggregate metrics ----
    a_agg_ols  = [r["agg_ols"]  for r in results_a if r["agg_ols"]  is not None]
    a_var_ols  = [r["var_ols"]  for r in results_a if r["var_ols"]  is not None]
    a_n_qual   = [r["n_analysts_qualified"] for r in results_a]
    b_agg_ols  = [r["agg_ols"]  for r in results_b if r["agg_ols"]  is not None]

    mean_agg_ols_a  = float(np.mean(a_agg_ols))  if a_agg_ols  else float("nan")
    mean_var_ols_a  = float(np.mean(a_var_ols))  if a_var_ols  else float("nan")
    mean_n_qual_a   = float(np.mean(a_n_qual))
    mean_agg_ols_b  = float(np.mean(b_agg_ols))  if b_agg_ols  else float("nan")

    a_l3_dr = float(np.mean([r["l3_fires"] for r in results_a]))
    a_l2_dr = float(np.mean([r["l2_fires"] for r in results_a]))
    b_l2_dr = float(np.mean([r["l2_fires"] for r in results_b]))
    b_l3_dr = float(np.mean([r["l3_fires"] for r in results_b]))

    # ---- Gate evaluation ----
    a_l3_gate    = a_l3_dr >= 0.70
    a_l2_silence = a_l2_dr <= 0.30
    b_l2_gate    = b_l2_dr >= 0.70
    both_gates   = a_l3_gate and a_l2_silence and b_l2_gate

    # ---- Save ----
    out = {
        "experiment":        "ANALYST-CONTRIBUTION-001",
        "gae_version":       gae.__version__,
        "n_decisions":       N_DECISIONS,
        "n_warmup":          N_WARMUP,
        "n_analysts":        N_ANALYSTS,
        "n_seeds":           N_SEEDS,
        "var_ols_threshold": VAR_OLS_THRESHOLD,
        "condition_a": {
            "q_high":                 Q_HIGH,
            "q_low":                  Q_LOW,
            "mean_aggregate_ols":     round(mean_agg_ols_a, 4),
            "mean_analysts_with_5plus": round(mean_n_qual_a, 1),
            "mean_var_ols":           round(mean_var_ols_a, 6),
            "level3_detection_rate":  round(a_l3_dr, 4),
            "level2_detection_rate":  round(a_l2_dr, 4),
            "level3_gate_pass":       bool(a_l3_gate),
            "level2_silence_gate_pass": bool(a_l2_silence),
        },
        "condition_b": {
            "q_uniform":             Q_UNIFORM,
            "mean_aggregate_ols":    round(mean_agg_ols_b, 4),
            "level2_detection_rate": round(b_l2_dr, 4),
            "level3_detection_rate": round(b_l3_dr, 4),
            "level2_gate_pass":      bool(b_l2_gate),
        },
        "both_gates_pass": bool(both_gates),
        "claim_acm_01":    "VALIDATED" if both_gates else "production milestone",
        "seeds": results_a + results_b,
    }

    out_path = REPO_ROOT / "notebooks" / "analyst_contribution_001_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {out_path}")
    print()

    # ---- Print ----
    insufficient = " ← INSUFFICIENT DATA" if mean_n_qual_a < 7 else ""
    print(f"ANALYST-CONTRIBUTION-001 (N={N_SEEDS}, N_DECISIONS={N_DECISIONS}, "
          f"GAE {gae.__version__}):")
    print()
    print("Pre-calculations:")
    print(f"  Overrides/analyst in measurement window: {ov_per_analyst:.1f}")
    print(f"  OLS estimate SE at N=20 overrides: {se_at_20:.3f}")
    print()
    print(f"Condition A — Bimodal (5xq={Q_HIGH}, 5xq={Q_LOW}):")
    print(f"  Mean aggregate OLS: {mean_agg_ols_a:.3f} [sanity: 0.95-1.10]")
    print(f"  Mean analysts with >=5 overrides: {mean_n_qual_a:.1f}{insufficient}")
    print(f"  Mean Var(OLS_i): {mean_var_ols_a:.4f}")
    print(f"  Level 3 detection rate: {a_l3_dr:.0%} [gate: >=70%] -> "
          f"{'PASS' if a_l3_gate else 'FAIL'}")
    print(f"  Level 2 detection rate: {a_l2_dr:.0%} [gate: <=30%] -> "
          f"{'PASS' if a_l2_silence else 'FAIL'}")
    print()
    print(f"Condition B — Uniform (all q={Q_UNIFORM}):")
    print(f"  Mean aggregate OLS: {mean_agg_ols_b:.3f} [sanity: <1.0]")
    print(f"  Level 2 detection rate: {b_l2_dr:.0%} [gate: >=70%] -> "
          f"{'PASS' if b_l2_gate else 'FAIL'}")
    print()
    print(f"Both gates pass: {'YES' if both_gates else 'NO'}")
    print(f"CLAIM-ACM-01: {'VALIDATED' if both_gates else 'production milestone'}")
    print("Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
