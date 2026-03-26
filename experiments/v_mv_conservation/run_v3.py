"""
V-MV-CONSERVATION v3 — plateau-snapshot OLSMonitor.

Identical 48-cell design to v2. Single change: DETECTOR 1 uses
gae.convergence.OLSMonitor (GAE 0.7.12) with plateau-snapshot baseline.
DETECTOR 2 (VarQMonitor) unchanged for side-by-side comparison.

Root cause fixed: OLS learning-phase decline no longer triggers CUSUM.
Baseline frozen after centroid plateau; CUSUM accumulates post-plateau only.

Save: experiments/v_mv_conservation/results/results_v3.json
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
from gae.convergence import OLSMonitor, compute_normalized_var_q

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS      = 20
N_DECISIONS  = 350
ALPHA        = 0.25
MU0_SIGMA    = 0.30
Q_HEALTHY    = 0.85
THETA_MIN    = 0.467

TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01

# OLSMonitor (GAE 0.7.12 plateau-snapshot)
OLS_PLATEAU_WINDOW    = 20
OLS_PLATEAU_THRESHOLD = 0.02
OLS_H                 = 5.0
OLS_K                 = 0.10
OLS_WIN               = 30    # rolling window for OLS computation
OLS_MINC              = 5     # min entries per buffer before OLS valid

# VarQMonitor (unchanged from v2)
VARQ_WIN  = 30
VARQ_THR  = 0.05

# TP window
TP_PRE  = 30
TP_POST = 100

# Factorial axes
ONSET_TIMES   = [50, 100, 200]
DEG_PROFILES  = ["step", "linear"]
Q_ENDPOINTS   = [0.65, 0.70]
SIGMA_V_PAIRS = [(0.10, 200), (0.15, 200), (0.20, 100), (0.25, 50)]

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
# Rolling OLS computation (feeds OLSMonitor.update())
# ---------------------------------------------------------------------------
class RollingOLS:
    """Compute OLS from rolling W-decision buffers; feeds OLSMonitor.update()."""

    def __init__(self, window: int = OLS_WIN, min_count: int = OLS_MINC):
        self.window    = window
        self.min_count = min_count
        self._ov_buf   = deque(maxlen=window)
        self._ac_buf   = deque(maxlen=window)

    def record(self, is_override: bool, was_correct: bool) -> float | None:
        """Returns OLS value if computable, else None."""
        if is_override:
            self._ov_buf.append(1.0 if was_correct else 0.0)
        else:
            self._ac_buf.append(1.0 if was_correct else 0.0)

        if (len(self._ov_buf) >= self.min_count and
                len(self._ac_buf) >= self.min_count):
            p_ov = float(np.mean(list(self._ov_buf)))
            p_ac = float(np.mean(list(self._ac_buf)))
            return p_ov / max(p_ac, 0.01)
        return None


# ---------------------------------------------------------------------------
# VarQ Monitor (unchanged from v2)
# ---------------------------------------------------------------------------
class VarQMonitor:
    def __init__(self, window: int = VARQ_WIN, q_baseline: float = Q_HEALTHY,
                 threshold: float = VARQ_THR):
        self.window     = window
        self.q_baseline = q_baseline
        self.threshold  = threshold
        self._buf       = deque(maxlen=window)
        self.fired      = False
        self.fire_t: int | None = None

    def record(self, quality: float, decision_t: int) -> None:
        self._buf.append(float(quality))
        if not self.fired and len(self._buf) >= self.window:
            var_norm = compute_normalized_var_q(list(self._buf), self.q_baseline)
            if var_norm > self.threshold:
                self.fired  = True
                self.fire_t = decision_t


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

    noise = np.full(N_FACTORS, sigma_eff)

    rolling_ols = RollingOLS(window=OLS_WIN, min_count=OLS_MINC)
    ols_mon     = OLSMonitor(plateau_window=OLS_PLATEAU_WINDOW,
                             plateau_threshold=OLS_PLATEAU_THRESHOLD,
                             h=OLS_H, k=OLS_K)
    varq_mon    = VarQMonitor(window=VARQ_WIN, q_baseline=Q_HEALTHY,
                              threshold=VARQ_THR)

    ols_fire_t: int | None  = None
    varq_fire_t             = None  # tracked via varq_mon.fire_t
    plateau_t: int | None   = None

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
            is_quality   = (rng.random() < q_eff)
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
            varq_mon.record(quality=1.0 if analyst_corr else 0.0, decision_t=t)
        else:
            analyst_corr = ai_corr  # accepted decision quality = model correctness

        # Compute rolling OLS; feed to OLSMonitor
        ols_val = rolling_ols.record(is_override, analyst_corr)
        if ols_val is not None:
            was_pre_plateau = not ols_mon.baseline_frozen
            fired = ols_mon.update(ols_val)
            if was_pre_plateau and ols_mon.baseline_frozen and plateau_t is None:
                plateau_t = t
            if fired and ols_fire_t is None:
                ols_fire_t = t

    tp_low  = max(0, onset_time - TP_PRE)
    tp_high = onset_time + TP_POST

    def classify(fired: bool, fire_t):
        if fired:
            return "TP" if (tp_low <= fire_t <= tp_high) else "FP"
        return "FN"

    ols_outcome  = classify(ols_fire_t is not None, ols_fire_t)
    varq_outcome = classify(varq_mon.fired, varq_mon.fire_t)

    ols_lag  = (ols_fire_t  - onset_time) if ols_fire_t  is not None else None
    varq_lag = (varq_mon.fire_t - onset_time) if varq_mon.fired else None

    return {
        "onset_time":     onset_time,
        "deg_profile":    deg_profile,
        "q_endpoint":     q_endpoint,
        "sigma_eff":      sigma_eff,
        "seed":           seed,
        "ols_outcome":    ols_outcome,
        "varq_outcome":   varq_outcome,
        "ols_fire_t":     ols_fire_t,
        "varq_fire_t":    varq_mon.fire_t,
        "ols_lag":        ols_lag,
        "varq_lag":       varq_lag,
        "plateau_t":      plateau_t,
        "ols_baseline":   round(ols_mon.baseline_ols, 4),
        "tp_window":      [tp_low, tp_high],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(seed_results: list[dict], key: str) -> dict:
    tp = sum(1 for r in seed_results if r[key] == "TP")
    fp = sum(1 for r in seed_results if r[key] == "FP")
    fn = sum(1 for r in seed_results if r[key] == "FN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall / (precision+recall)
                 if (precision+recall) > 0 else 0.0)
    lags = [r[key.replace("_outcome", "_lag")] for r in seed_results
            if r[key] == "TP" and r[key.replace("_outcome", "_lag")] is not None]
    mean_lag = float(np.mean(lags)) if lags else float("nan")
    return dict(tp=tp, fp=fp, fn=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                mean_lag=round(mean_lag, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    assert gae.__version__ == "0.7.12", f"Need GAE 0.7.12, got {gae.__version__}"
    print(f"V-MV-CONSERVATION v3 (GAE {gae.__version__})")
    print(f"  48 cells x {N_SEEDS} seeds = {48*N_SEEDS} runs  "
          f"(OLSMonitor plateau-snapshot)", flush=True)
    print()

    cells = list(product(ONSET_TIMES, DEG_PROFILES, Q_ENDPOINTS, SIGMA_V_PAIRS))
    assert len(cells) == 48

    all_seeds: list[dict]    = []
    cell_records: list[dict] = []

    for cell_idx, (onset_time, deg_profile, q_endpoint, (sigma_eff, V)) in enumerate(cells):
        cell_sr = []
        for seed in SEEDS_20:
            r = run_seed(onset_time, deg_profile, q_endpoint, sigma_eff, seed)
            r["V"] = V
            all_seeds.append(r)
            cell_sr.append(r)

        m_ols  = metrics(cell_sr, "ols_outcome")
        m_varq = metrics(cell_sr, "varq_outcome")
        plateau_mean = float(np.mean([r["plateau_t"] or N_DECISIONS
                                      for r in cell_sr]))

        cell_records.append({
            "cell_idx":       cell_idx,
            "onset_time":     onset_time,
            "deg_profile":    deg_profile,
            "q_endpoint":     q_endpoint,
            "sigma_eff":      sigma_eff,
            "V":              V,
            "ols":            m_ols,
            "varq":           m_varq,
            "plateau_mean_t": round(plateau_mean, 1),
        })

        print(f"  [{cell_idx+1:02d}/48] onset={onset_time:3d} {deg_profile:6s} "
              f"q={q_endpoint} σ={sigma_eff:.2f} V={V:3d} | "
              f"OLS P={m_ols['precision']:.2f} R={m_ols['recall']:.2f} lag={m_ols['mean_lag']:.0f}d | "
              f"VQ  P={m_varq['precision']:.2f} R={m_varq['recall']:.2f} | "
              f"plateau@{plateau_mean:.0f}", flush=True)

    # ---------------------------------------------------------------------------
    # Macro averages
    # ---------------------------------------------------------------------------
    def macro(key: str, field: str):
        return round(float(np.mean([c[key][field] for c in cell_records])), 4)

    ols_mac_p = macro("ols",  "precision")
    ols_mac_r = macro("ols",  "recall")
    vq_mac_p  = macro("varq", "precision")
    vq_mac_r  = macro("varq", "recall")

    # Mean lag (TP only)
    ols_lags  = [r["ols_lag"]  for r in all_seeds
                 if r["ols_outcome"]  == "TP" and r["ols_lag"]  is not None]
    varq_lags = [r["varq_lag"] for r in all_seeds
                 if r["varq_outcome"] == "TP" and r["varq_lag"] is not None]
    ols_mean_lag  = round(float(np.mean(ols_lags)),  1) if ols_lags  else float("nan")
    varq_mean_lag = round(float(np.mean(varq_lags)), 1) if varq_lags else float("nan")

    # By onset_time breakdown (OLSMonitor)
    by_onset: dict[str, dict] = {}
    for ot in ONSET_TIMES:
        sub = [r for r in all_seeds if r["onset_time"] == ot]
        m   = metrics(sub, "ols_outcome")
        by_onset[str(ot)] = {"precision": m["precision"], "recall": m["recall"],
                              "mean_lag": m["mean_lag"]}

    ols_gate = (ols_mac_p > 0.70 and ols_mac_r > 0.80)
    vq_gate  = (vq_mac_p  > 0.70 and vq_mac_r  > 0.80)
    best = ("ols_monitor" if (ols_gate and not vq_gate) else
            "varq"        if (vq_gate  and not ols_gate) else
            "both"        if (ols_gate and vq_gate) else "neither")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    out = {
        "experiment":   "V-MV-CONSERVATION-v3",
        "gae_version":  "0.7.12",
        "n_cells":      48,
        "n_seeds":      N_SEEDS,
        "n_decisions":  N_DECISIONS,
        "ols_monitor": {
            "precision":          ols_mac_p,
            "recall":             ols_mac_r,
            "mean_lag_decisions": ols_mean_lag,
            "gate_pass":          ols_gate,
            "by_onset_time":      by_onset,
        },
        "varq": {
            "precision":          vq_mac_p,
            "recall":             vq_mac_r,
            "mean_lag_decisions": varq_mean_lag,
            "gate_pass":          vq_gate,
        },
        "best_detector": best,
        "vs_v2": {
            "ols_precision_v2": 0.543,
            "ols_precision_v3": ols_mac_p,
            "improvement_pp":   round((ols_mac_p - 0.543) * 100, 1),
        },
        "cells": cell_records,
    }

    out_path = (REPO_ROOT / "experiments" / "v_mv_conservation"
                / "results" / "results_v3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Saved: {out_path}")

    # ---------------------------------------------------------------------------
    # Print
    # ---------------------------------------------------------------------------
    print()
    print(f"V-MV-CONSERVATION v3 (48 cells, 20 seeds, GAE 0.7.12):")
    print(f"  OLSMonitor (plateau-snapshot):")
    print(f"    Overall: P={ols_mac_p:.3f} R={ols_mac_r:.3f} "
          f"lag={ols_mean_lag:.1f}d → {'GATE PASS' if ols_gate else 'GATE FAIL'}")
    print(f"    By onset_time:")
    for ot in ONSET_TIMES:
        b = by_onset[str(ot)]
        print(f"      onset={ot:3d}: P={b['precision']:.3f} R={b['recall']:.3f} "
              f"lag={b['mean_lag']:.1f}d")
    print(f"  Var(q) normalized:")
    print(f"    Overall: P={vq_mac_p:.3f} R={vq_mac_r:.3f} "
          f"lag={varq_mean_lag:.1f}d → {'GATE PASS' if vq_gate else 'GATE FAIL'}")
    print(f"  Best detector: {best}")
    print(f"  vs v2: OLS precision {0.543} → {ols_mac_p:.3f} "
          f"({(ols_mac_p - 0.543)*100:+.1f}pp)")
    print(f"Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
