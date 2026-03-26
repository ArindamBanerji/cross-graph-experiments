"""
V-MV-CONSERVATION v8 — VarQMonitor persistence filter, all 48 cells (GAE 0.7.15).

Full 48-cell design with onset=200 back in scope. Uses GAE 0.7.15 VarQMonitor
with built-in persistence filter (N=3 consecutive Var(q) crossings required).
Single-spike FPs from natural Bernoulli noise suppressed by persistence; real
degradation produces sustained elevation that persists 3+ consecutive windows.

v6 (no persistence): onset=200 P=0.301 (sole blocker)
v7 (scoped onset∈{50,100}): P=0.728/R=0.983

VarQMonitor API (GAE 0.7.15):
  monitor = VarQMonitor()          # threshold=0.05, persistence=3, window=30
  fired = monitor.update(q_t)      # takes raw quality float, returns bool
  First True → fire; track decision_t externally.
  baseline_window=50 — builds _q_baseline from first 50 override observations.

Save: experiments/v_mv_conservation/results/results_v8.json
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile
from gae.convergence import VarQMonitor

# ---------------------------------------------------------------------------
# Parameters (unchanged from v2-v7)
# ---------------------------------------------------------------------------
N_SEEDS      = 20
N_DECISIONS  = 350
ALPHA        = 0.25
MU0_SIGMA    = 0.30
Q_HEALTHY    = 0.85

TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01

TP_PRE  = 30
TP_POST = 100

ONSET_TIMES   = [50, 100, 200]
DEG_PROFILES  = ["step", "linear"]
Q_ENDPOINTS   = [0.65, 0.70]
SIGMA_V_PAIRS = [(0.10, 200), (0.15, 200), (0.20, 100), (0.25, 50)]

SEEDS_20 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]

# ---------------------------------------------------------------------------
# SOC geometry (unchanged from v2-v7)
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
# Degradation schedule
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
# Single seed run
# ---------------------------------------------------------------------------
def run_seed(onset_time: int, deg_profile: str, q_endpoint: float,
             sigma_eff: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    mu0 = MU_STAR.copy() + rng.normal(0, MU0_SIGMA, MU_STAR.shape)
    np.clip(mu0, 0.0, 1.0, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                            eta_override=ETA_OVERRIDE)

    noise   = np.full(N_FACTORS, sigma_eff)
    monitor = VarQMonitor()   # threshold=0.05, persistence=3, window=30

    fire_t: int | None = None

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
                wrong        = [a for a in range(N_ACTS) if a != gt_act]
                analyst_act  = int(rng.choice(wrong))
                analyst_corr = False
                scorer.update(f=f, category_index=cat_idx,
                              action_index=analyst_act, correct=True,
                              gt_action_index=None)

            if fire_t is None:
                fired = monitor.update(1.0 if analyst_corr else 0.0)
                if fired:
                    fire_t = t

    tp_low  = max(0, onset_time - TP_PRE)
    tp_high = onset_time + TP_POST

    if fire_t is not None:
        outcome = "TP" if (tp_low <= fire_t <= tp_high) else "FP"
    else:
        outcome = "FN"

    lag = (fire_t - onset_time) if fire_t is not None else None

    return {
        "onset_time":  onset_time,
        "deg_profile": deg_profile,
        "q_endpoint":  q_endpoint,
        "sigma_eff":   sigma_eff,
        "seed":        seed,
        "outcome":     outcome,
        "fire_t":      fire_t,
        "lag":         lag,
        "tp_window":   [tp_low, tp_high],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(seed_results: list[dict]) -> dict:
    tp = sum(1 for r in seed_results if r["outcome"] == "TP")
    fp = sum(1 for r in seed_results if r["outcome"] == "FP")
    fn = sum(1 for r in seed_results if r["outcome"] == "FN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    lags = [r["lag"] for r in seed_results
            if r["outcome"] == "TP" and r["lag"] is not None]
    mean_lag = float(np.mean(lags)) if lags else float("nan")
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
    assert gae.__version__ == "0.7.15", f"Need GAE 0.7.15, got {gae.__version__}"
    print(f"V-MV-CONSERVATION v8 (GAE {gae.__version__})")
    print(f"  48 cells x {N_SEEDS} seeds = {48*N_SEEDS} runs "
          f"(VarQMonitor persistence=3, threshold=0.05)", flush=True)
    print()

    cells = list(product(ONSET_TIMES, DEG_PROFILES, Q_ENDPOINTS, SIGMA_V_PAIRS))
    assert len(cells) == 48

    all_seeds:    list[dict] = []
    cell_records: list[dict] = []

    for cell_idx, (onset_time, deg_profile, q_endpoint, (sigma_eff, V)) in enumerate(cells):
        cell_sr = []
        for seed in SEEDS_20:
            r = run_seed(onset_time, deg_profile, q_endpoint, sigma_eff, seed)
            r["V"] = V
            all_seeds.append(r)
            cell_sr.append(r)

        m = metrics(cell_sr)
        cell_records.append({
            "cell_idx":    cell_idx,
            "onset_time":  onset_time,
            "deg_profile": deg_profile,
            "q_endpoint":  q_endpoint,
            "sigma_eff":   sigma_eff,
            "V":           V,
            "varq":        m,
        })
        print(f"  [{cell_idx+1:02d}/48] onset={onset_time:3d} {deg_profile:6s} "
              f"q={q_endpoint} σ={sigma_eff:.2f} V={V:3d} | "
              f"VQ P={m['precision']:.2f} R={m['recall']:.2f} "
              f"lag={m['mean_lag'] if m['mean_lag'] is not None else 'nan'}d",
              flush=True)

    # ---------------------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------------------
    m_all = metrics(all_seeds)
    gate  = (m_all["precision"] > 0.70 and m_all["recall"] > 0.80)

    # By onset_time
    by_onset: dict[str, dict] = {}
    for ot in ONSET_TIMES:
        sub = [r for r in all_seeds if r["onset_time"] == ot]
        mo  = metrics(sub)
        by_onset[str(ot)] = {"precision": mo["precision"], "recall": mo["recall"],
                              "f1": mo["f1"], "mean_lag": mo["mean_lag"]}

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    out = {
        "experiment":   "V-MV-CONSERVATION-v8",
        "gae_version":  "0.7.15",
        "detector":     "VarQMonitor",
        "persistence":  3,
        "threshold":    0.05,
        "window":       30,
        "n_cells":      48,
        "n_seeds":      N_SEEDS,
        "n_decisions":  N_DECISIONS,
        "varq": {
            "precision":          m_all["precision"],
            "recall":             m_all["recall"],
            "f1":                 m_all["f1"],
            "mean_lag_decisions": m_all["mean_lag"],
            "gate_pass":          gate,
            "by_onset_time":      by_onset,
        },
        "vs_v7": {
            "v7_precision": 0.728,
            "v7_recall":    0.983,
            "v7_scope":     "onset 50+100 only",
            "v8_precision": m_all["precision"],
            "v8_recall":    m_all["recall"],
            "v8_scope":     "all onset times including 200",
        },
        "cells": cell_records,
    }

    out_path = (REPO_ROOT / "experiments" / "v_mv_conservation"
                / "results" / "results_v8.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Saved: {out_path}")

    # ---------------------------------------------------------------------------
    # Print
    # ---------------------------------------------------------------------------
    print()
    print(f"V-MV-CONSERVATION v8 (48 cells, all onset times, GAE 0.7.15):")
    print(f"  VarQMonitor (persistence=3, threshold=0.05):")
    print(f"    Overall: P={m_all['precision']:.3f} R={m_all['recall']:.3f} "
          f"F1={m_all['f1']:.3f} → {'GATE PASS' if gate else 'GATE FAIL'}")
    for ot in ONSET_TIMES:
        b = by_onset[str(ot)]
        suffix = "  <- critical case" if ot == 200 else ""
        print(f"    onset={ot:3d}: P={b['precision']:.3f} R={b['recall']:.3f}{suffix}")
    print(f"  vs v7 (scoped 32 cells): P=0.728/R=0.983")
    print(f"  vs v6 (no persistence):  onset=200 P=0.301/R=0.759")
    print(f"Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
