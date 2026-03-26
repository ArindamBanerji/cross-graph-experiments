"""
V-MV-CONSERVATION v6 — VarQMonitor threshold sweep (GAE 0.7.14).

Same 48-cell simulation as v2-v5. OLS CUSUM dropped (structurally
incompatible). Sweep VarQMonitor threshold across 8 values to find
threshold achieving P>0.70 AND R>0.80 simultaneously.

v5 VarQMonitor result at threshold=0.05: P=0.583, R=0.902 (precision gap 0.117).

Save: experiments/v_mv_conservation/results/results_v6.json
"""
from __future__ import annotations

import json
import sys
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile
from gae.convergence import compute_normalized_var_q

# ---------------------------------------------------------------------------
# Parameters (unchanged from v2-v5)
# ---------------------------------------------------------------------------
N_SEEDS      = 20
N_DECISIONS  = 350
ALPHA        = 0.25
MU0_SIGMA    = 0.30
Q_HEALTHY    = 0.85

TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01

VARQ_WIN = 30

TP_PRE  = 30
TP_POST = 100

ONSET_TIMES   = [50, 100, 200]
DEG_PROFILES  = ["step", "linear"]
Q_ENDPOINTS   = [0.65, 0.70]
SIGMA_V_PAIRS = [(0.10, 200), (0.15, 200), (0.20, 100), (0.25, 50)]

THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

SEEDS_20 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]

# ---------------------------------------------------------------------------
# SOC geometry (unchanged)
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
# Degradation schedule (unchanged)
# ---------------------------------------------------------------------------
def get_q_eff(t: int, onset_time: int, deg_profile: str,
              q_endpoint: float) -> float:
    if t < onset_time:
        return Q_HEALTHY
    if deg_profile == "step":
        return q_endpoint
    span     = max(1, N_DECISIONS - onset_time)
    progress = min(1.0, (t - onset_time) / span)
    return Q_HEALTHY - progress * (Q_HEALTHY - q_endpoint)


# ---------------------------------------------------------------------------
# Single seed run — returns override quality buffer for post-hoc threshold sweep
# ---------------------------------------------------------------------------
def run_seed(onset_time: int, deg_profile: str, q_endpoint: float,
             sigma_eff: float, seed: int) -> dict:
    """
    Returns the rolling override-quality time series so VarQMonitor can be
    re-evaluated at any threshold without re-running the simulation.
    """
    rng = np.random.default_rng(seed)

    mu0 = MU_STAR.copy() + rng.normal(0, MU0_SIGMA, MU_STAR.shape)
    np.clip(mu0, 0.0, 1.0, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                            eta_override=ETA_OVERRIDE)

    noise = np.full(N_FACTORS, sigma_eff)

    # Collect per-decision override quality values (1.0 or 0.0)
    override_quality_series: list[tuple[int, float]] = []  # (decision_t, quality)

    for t in range(N_DECISIONS):
        q_eff = get_q_eff(t, onset_time, deg_profile, q_endpoint)

        cat_idx = int(rng.choice(N_CATS, p=CAT_WEIGHTS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, 1, N_FACTORS) * noise,
                    0.0, 1.0)

        result  = scorer.score(f, cat_idx)
        ai_act  = result.action_index
        ai_corr = (ai_act == gt_act)

        is_override = (rng.random() < ALPHA)

        if is_override:
            is_quality = (rng.random() < q_eff)
            if is_quality:
                analyst_corr = True
                scorer.update(f=f, category_index=cat_idx,
                              action_index=gt_act, correct=True,
                              gt_action_index=gt_act)
            else:
                wrong       = [a for a in range(N_ACTS) if a != gt_act]
                analyst_act = int(rng.choice(wrong))
                analyst_corr = False
                scorer.update(f=f, category_index=cat_idx,
                              action_index=analyst_act, correct=True,
                              gt_action_index=None)
            override_quality_series.append((t, 1.0 if analyst_corr else 0.0))
        else:
            # accepted — not fed to VarQMonitor
            pass

    tp_low  = max(0, onset_time - TP_PRE)
    tp_high = onset_time + TP_POST

    return {
        "onset_time":              onset_time,
        "deg_profile":             deg_profile,
        "q_endpoint":              q_endpoint,
        "sigma_eff":               sigma_eff,
        "seed":                    seed,
        "override_quality_series": override_quality_series,
        "tp_window":               [tp_low, tp_high],
    }


# ---------------------------------------------------------------------------
# Apply VarQMonitor at a given threshold post-hoc from override quality series
# ---------------------------------------------------------------------------
def apply_varq(seed_result: dict, threshold: float) -> dict:
    buf      = deque(maxlen=VARQ_WIN)
    fired    = False
    fire_t   = None
    tp_low, tp_high = seed_result["tp_window"]

    for decision_t, quality in seed_result["override_quality_series"]:
        buf.append(quality)
        if not fired and len(buf) >= VARQ_WIN:
            var_norm = compute_normalized_var_q(list(buf), Q_HEALTHY)
            if var_norm > threshold:
                fired  = True
                fire_t = decision_t

    if fired:
        outcome = "TP" if (tp_low <= fire_t <= tp_high) else "FP"
    else:
        outcome = "FN"

    onset_time = seed_result["onset_time"]
    lag = (fire_t - onset_time) if fired else None

    return {"outcome": outcome, "fire_t": fire_t, "lag": lag}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(outcomes: list[str], lags: list[int | None]) -> dict:
    tp = outcomes.count("TP")
    fp = outcomes.count("FP")
    fn = outcomes.count("FN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    valid_lags = [l for o, l in zip(outcomes, lags) if o == "TP" and l is not None]
    mean_lag = float(np.mean(valid_lags)) if valid_lags else float("nan")
    return dict(tp=tp, fp=fp, fn=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                mean_lag=round(mean_lag, 1) if not np.isnan(mean_lag) else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    assert gae.__version__ == "0.7.14", f"Need GAE 0.7.14, got {gae.__version__}"
    print(f"V-MV-CONSERVATION v6 — VarQMonitor threshold sweep (GAE {gae.__version__}):")
    print(f"  48 cells x {N_SEEDS} seeds = {48*N_SEEDS} runs | "
          f"{len(THRESHOLDS)} thresholds", flush=True)
    print()

    cells = list(product(ONSET_TIMES, DEG_PROFILES, Q_ENDPOINTS, SIGMA_V_PAIRS))
    assert len(cells) == 48

    # --- Run simulation once, collect override quality series ---
    print("  Running simulation...", flush=True)
    all_seeds: list[dict] = []
    for onset_time, deg_profile, q_endpoint, (sigma_eff, V) in cells:
        for seed in SEEDS_20:
            r = run_seed(onset_time, deg_profile, q_endpoint, sigma_eff, seed)
            r["V"] = V
            all_seeds.append(r)
    print(f"  Done. {len(all_seeds)} seeds collected.", flush=True)
    print()

    # --- Sweep thresholds ---
    sweep_results: dict[str, dict] = {}
    sweep_by_onset: dict[str, dict[str, dict]] = {}

    for thr in THRESHOLDS:
        outcomes_all: list[str] = []
        lags_all: list[int | None] = []

        for r in all_seeds:
            res = apply_varq(r, thr)
            outcomes_all.append(res["outcome"])
            lags_all.append(res["lag"])

        m = metrics(outcomes_all, lags_all)
        gate = (m["precision"] > 0.70 and m["recall"] > 0.80)
        sweep_results[str(thr)] = {
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "tp":        m["tp"],
            "fp":        m["fp"],
            "fn":        m["fn"],
            "mean_lag":  m["mean_lag"],
            "gate_pass": gate,
        }

        # by onset_time
        by_onset: dict[str, dict] = {}
        for ot in ONSET_TIMES:
            sub = [r for r in all_seeds if r["onset_time"] == ot]
            sub_res = [apply_varq(r, thr) for r in sub]
            sub_out = [x["outcome"] for x in sub_res]
            sub_lag = [x["lag"]     for x in sub_res]
            mo = metrics(sub_out, sub_lag)
            by_onset[str(ot)] = {"precision": mo["precision"], "recall": mo["recall"],
                                  "f1": mo["f1"]}
        sweep_by_onset[str(thr)] = by_onset

    # --- Find optimal threshold ---
    # Max F1 subject to P>0.70 AND R>0.80
    gate_passing = [(thr, sweep_results[str(thr)])
                    for thr in THRESHOLDS
                    if sweep_results[str(thr)]["gate_pass"]]

    if gate_passing:
        opt_thr, opt_m = max(gate_passing, key=lambda x: x[1]["f1"])
        gate_pass = True
    else:
        # No threshold passes; report best F1 overall
        opt_thr, opt_m = max(
            [(thr, sweep_results[str(thr)]) for thr in THRESHOLDS],
            key=lambda x: x[1]["f1"]
        )
        gate_pass = False

    opt_by_onset = sweep_by_onset[str(opt_thr)]

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    out = {
        "experiment":        "V-MV-CONSERVATION-v6",
        "gae_version":       "0.7.14",
        "detector":          "VarQMonitor",
        "varq_window":       VARQ_WIN,
        "n_cells":           48,
        "n_seeds":           N_SEEDS,
        "n_decisions":       N_DECISIONS,
        "threshold_sweep":   sweep_results,
        "optimal_threshold": opt_thr,
        "optimal_precision": opt_m["precision"],
        "optimal_recall":    opt_m["recall"],
        "optimal_f1":        opt_m["f1"],
        "gate_pass":         gate_pass,
        "by_onset_time":     {
            ot: {"precision": opt_by_onset[ot]["precision"],
                 "recall":    opt_by_onset[ot]["recall"],
                 "f1":        opt_by_onset[ot]["f1"]}
            for ot in [str(x) for x in ONSET_TIMES]
        },
    }

    out_path = (REPO_ROOT / "experiments" / "v_mv_conservation"
                / "results" / "results_v6.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {out_path}")
    print()

    # ---------------------------------------------------------------------------
    # Print sweep table
    # ---------------------------------------------------------------------------
    print(f"{'Threshold':<10} | {'P':<6} | {'R':<6} | {'F1':<6} | Gate")
    print("-" * 48)
    for thr in THRESHOLDS:
        m   = sweep_results[str(thr)]
        p, r, f1 = m["precision"], m["recall"], m["f1"]
        gate_str = "PASS" if m["gate_pass"] else (
            "FAIL (P<0.70)" if p <= 0.70 else
            "FAIL (R<0.80)" if r <= 0.80 else "FAIL"
        )
        print(f"{thr:<10} | {p:<6.3f} | {r:<6.3f} | {f1:<6.3f} | {gate_str}")

    print()
    if gate_pass:
        print(f"Optimal threshold: {opt_thr} (max F1 with P>0.70 AND R>0.80)")
    else:
        print(f"Optimal threshold: {opt_thr} (best F1 — no threshold passes gate)")
    print(f"  P={opt_m['precision']:.3f} R={opt_m['recall']:.3f} "
          f"F1={opt_m['f1']:.3f} → {'GATE PASS' if gate_pass else 'GATE FAIL'}")
    print()
    print("By onset_time at optimal threshold:")
    for ot in ONSET_TIMES:
        b = opt_by_onset[str(ot)]
        print(f"  onset={ot:3d}: P={b['precision']:.3f} R={b['recall']:.3f}")
    print()
    print("Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
