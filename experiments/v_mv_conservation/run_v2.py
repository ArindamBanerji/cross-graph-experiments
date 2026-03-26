"""
V-MV-CONSERVATION v2 — dedicated 48-cell degradation simulation.

Design:
  48 cells: onset_time × deg_profile × q_endpoint × (sigma_eff, V)
    onset_time  ∈ {50, 100, 200}
    deg_profile ∈ {step, linear}
    q_endpoint  ∈ {0.65, 0.70}
    (sigma_eff, V) ∈ {(0.10,200),(0.15,200),(0.20,100),(0.25,50)}

  N_SEEDS=20 per cell, N_DECISIONS=350, ALPHA=0.25 (constant).
  Warm start: mu0 = MU_STAR + N(0, MU0_SIGMA=0.30), clipped [0,1].

DETECTOR 1 — CUSUM on OLS (OLSMonitor):
  EWMA λ=0.1, h=5.0, k_offset=0.10, window=30 decisions, baseline_n=50.

DETECTOR 2 — Baseline-normalized Var(q):
  compute_normalized_var_q over rolling 30-override window.
  Fixed q_baseline = Q_HEALTHY = 0.85.
  Fires on first exceedance of threshold=0.05.

Level 1: ConservationMonitor.update_conservation_signal() (sanity: never fires).

TP window: [T_onset - 30, T_onset + 100].
Gate: macro-avg Precision > 0.70 AND Recall > 0.80 for each detector.

Save: experiments/v_mv_conservation/results/results_v2.json
"""
from __future__ import annotations

import json
import sys
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile
from gae.convergence import compute_normalized_var_q, ConservationMonitor

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS      = 20
N_DECISIONS  = 350
ALPHA        = 0.25        # override rate (constant)
MU0_SIGMA    = 0.30        # warm-start jitter
Q_HEALTHY    = 0.85        # pre-onset override quality
THETA_MIN    = 0.467

TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01

# Detector 1 (OLS)
OLS_H        = 5.0
OLS_LAM      = 0.1
OLS_K_OFFSET = 0.10
OLS_WIN      = 30
OLS_MINC     = 5
OLS_BASE_N   = 50

# Detector 2 (VarQ)
VARQ_WIN     = 30
VARQ_THR     = 0.05        # compute_normalized_var_q threshold

# TP window
TP_PRE       = 30          # decisions before onset allowed
TP_POST      = 100         # decisions after onset allowed

# Factorial axes
ONSET_TIMES  = [50, 100, 200]
DEG_PROFILES = ["step", "linear"]
Q_ENDPOINTS  = [0.65, 0.70]
SIGMA_V_PAIRS = [(0.10, 200), (0.15, 200), (0.20, 100), (0.25, 50)]

SEEDS_20 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]

# ---------------------------------------------------------------------------
# SOC geometry (shared with V-OLS-DETECT)
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
# OLS Monitor (from V-OLS-DETECT)
# ---------------------------------------------------------------------------
class OLSMonitor:
    """CUSUM on EWMA of Override Lift Score."""

    def __init__(self, h=5.0, lam=0.1, k_offset=0.10,
                 window=30, min_count=5, baseline_n=50):
        self.h = h
        self.lam = lam
        self.k_offset = k_offset
        self.window = window
        self.min_count = min_count
        self.baseline_n = baseline_n

        self._ov_buf  = deque(maxlen=window)
        self._ac_buf  = deque(maxlen=window)
        self._ols_seq = []
        self._ewma    = None
        self._s       = 0.0
        self._baseline = None
        self.fired    = False
        self.fire_t   = None
        self._t       = 0

    def record(self, is_override: bool, was_correct: bool):
        if is_override:
            self._ov_buf.append(1 if was_correct else 0)
        else:
            self._ac_buf.append(1 if was_correct else 0)

        ols = None
        if (len(self._ov_buf) >= self.min_count and
                len(self._ac_buf) >= self.min_count):
            p_ov = float(np.mean(list(self._ov_buf)))
            p_ac = float(np.mean(list(self._ac_buf)))
            ols  = p_ov / max(p_ac, 0.01)

        if ols is not None:
            if self._ewma is None:
                self._ewma = ols
            else:
                self._ewma = self.lam * ols + (1 - self.lam) * self._ewma

            self._ols_seq.append(ols)

            if self._baseline is None and len(self._ols_seq) >= self.baseline_n:
                self._baseline = float(np.mean(self._ols_seq[:self.baseline_n]))
                self._s = 0.0

            if self._baseline is not None:
                self._s = max(0.0, self._s +
                              (self._baseline - self._ewma - self.k_offset))
                if not self.fired and self._s >= self.h:
                    self.fired  = True
                    self.fire_t = self._t

        self._t += 1

    def initial_ols(self, n: int = 50) -> float:
        if not self._ols_seq:
            return float("nan")
        return float(np.mean(self._ols_seq[:n]))


# ---------------------------------------------------------------------------
# VarQ Monitor
# ---------------------------------------------------------------------------
class VarQMonitor:
    """
    Baseline-normalized Var(q) detector.
    Fires on first window where compute_normalized_var_q > threshold.
    Uses fixed q_baseline = Q_HEALTHY (not learned from data).
    """

    def __init__(self, window=30, q_baseline=Q_HEALTHY, threshold=VARQ_THR):
        self.window     = window
        self.q_baseline = q_baseline
        self.threshold  = threshold
        self._buf       = deque(maxlen=window)
        self.fired      = False
        self.fire_t     = None

    def record(self, quality: float, decision_t: int) -> None:
        """Record quality outcome for an override."""
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
    # linear: ramp from Q_HEALTHY to q_endpoint over remaining decisions
    span = max(1, N_DECISIONS - onset_time)
    progress = min(1.0, (t - onset_time) / span)
    return Q_HEALTHY - progress * (Q_HEALTHY - q_endpoint)


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------
def run_seed(onset_time: int, deg_profile: str, q_endpoint: float,
             sigma_eff: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    # Warm start: MU_STAR + jitter
    mu0 = MU_STAR.copy() + rng.normal(0, MU0_SIGMA, MU_STAR.shape)
    np.clip(mu0, 0.0, 1.0, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                            eta_override=ETA_OVERRIDE)

    noise = np.full(N_FACTORS, sigma_eff)

    ols_mon  = OLSMonitor(h=OLS_H, lam=OLS_LAM, k_offset=OLS_K_OFFSET,
                          window=OLS_WIN, min_count=OLS_MINC,
                          baseline_n=OLS_BASE_N)
    varq_mon = VarQMonitor(window=VARQ_WIN, q_baseline=Q_HEALTHY,
                           threshold=VARQ_THR)
    cons_mon = ConservationMonitor()

    level1_fired = False

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
            is_quality  = (rng.random() < q_eff)
            if is_quality:
                analyst_act  = gt_act
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
            ols_mon.record(is_override=True, was_correct=analyst_corr)
            varq_mon.record(quality=1.0 if analyst_corr else 0.0, decision_t=t)
        else:
            ols_mon.record(is_override=False, was_correct=ai_corr)
            # accepted decisions: quality=1 for conservation, not fed to VarQ

        # Level 1 conservation check via ConservationMonitor.record_quality()
        # Feed per-decision quality (1 for accepted, q_eff binary for overrides)
        if is_override:
            cons_mon.record_quality(1.0 if analyst_corr else 0.0)
        else:
            cons_mon.record_quality(1.0)

        # Manual Level 1 check: alpha * q_bar * V < THETA_MIN?
        # With V=50 (min), alpha=0.25, q=0.65: product=8.125 >> 0.467 -> never fires
        if not level1_fired and cons_mon.yellow_warning:
            level1_fired = True  # CUSUM on q (layer 2 fires on Layer 2 YELLOW)

    tp_low  = max(0, onset_time - TP_PRE)
    tp_high = onset_time + TP_POST

    def classify(fired, fire_t):
        if fired:
            return "TP" if (tp_low <= fire_t <= tp_high) else "FP"
        return "FN"

    return {
        "onset_time":     onset_time,
        "deg_profile":    deg_profile,
        "q_endpoint":     q_endpoint,
        "sigma_eff":      sigma_eff,
        "seed":           seed,
        "det1_outcome":   classify(ols_mon.fired, ols_mon.fire_t),
        "det2_outcome":   classify(varq_mon.fired, varq_mon.fire_t),
        "det1_fire_t":    ols_mon.fire_t,
        "det2_fire_t":    varq_mon.fire_t,
        "initial_ols":    round(ols_mon.initial_ols(n=50), 4),
        "level1_fired":   level1_fired,
        "tp_window":      [tp_low, tp_high],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def cell_metrics(seed_results: list[dict], det: str):
    """Precision, recall, F1 for one cell across N_SEEDS seeds."""
    key = f"det{det}_outcome"
    tp = sum(1 for r in seed_results if r[key] == "TP")
    fp = sum(1 for r in seed_results if r[key] == "FP")
    fn = sum(1 for r in seed_results if r[key] == "FN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall / (precision+recall)
                 if (precision+recall) > 0 else 0.0)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4))


def main():
    import gae
    print(f"V-MV-CONSERVATION v2 (GAE {gae.__version__})")
    print(f"  48 cells x {N_SEEDS} seeds = {48*N_SEEDS} runs", flush=True)
    print()

    all_seeds    = []
    cell_records = []

    cells = list(product(ONSET_TIMES, DEG_PROFILES, Q_ENDPOINTS, SIGMA_V_PAIRS))
    assert len(cells) == 48, f"Expected 48 cells, got {len(cells)}"

    for cell_idx, (onset_time, deg_profile, q_endpoint, (sigma_eff, V)) in enumerate(cells):
        cell_seed_results = []
        for seed in SEEDS_20:
            r = run_seed(onset_time, deg_profile, q_endpoint, sigma_eff, seed)
            r["V"] = V
            all_seeds.append(r)
            cell_seed_results.append(r)

        m1 = cell_metrics(cell_seed_results, "1")
        m2 = cell_metrics(cell_seed_results, "2")
        level1_count = sum(1 for r in cell_seed_results if r["level1_fired"])
        init_ols_mean = float(np.mean([r["initial_ols"] for r in cell_seed_results]))

        cell_records.append({
            "cell_idx":    cell_idx,
            "onset_time":  onset_time,
            "deg_profile": deg_profile,
            "q_endpoint":  q_endpoint,
            "sigma_eff":   sigma_eff,
            "V":           V,
            "det1":        m1,
            "det2":        m2,
            "level1_fires": level1_count,
            "init_ols_mean": round(init_ols_mean, 4),
        })

        print(f"  [{cell_idx+1:02d}/48] onset={onset_time:3d} {deg_profile:6s} "
              f"q={q_endpoint} sigma={sigma_eff:.2f} V={V:3d} | "
              f"D1 P={m1['precision']:.2f} R={m1['recall']:.2f} | "
              f"D2 P={m2['precision']:.2f} R={m2['recall']:.2f} | "
              f"L1={level1_count}/20", flush=True)

    # ---------------------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------------------
    init_ols_all = [r["initial_ols"] for r in all_seeds]
    mean_init_ols = float(np.mean(init_ols_all))
    sanity_ols = mean_init_ols > 1.0

    total_level1 = sum(r["level1_fired"] for r in all_seeds)
    # Level 1 is the ConservationMonitor YELLOW (CUSUM on q) — note this might
    # fire for long runs with ALPHA*q_endpoint < threshold; treat as informational
    sanity_l1_note = (f"L1 fires={total_level1}/{len(all_seeds)} "
                      f"(expected: low — NOT the Level 1 alpha*q*V check, "
                      f"this is CUSUM-YELLOW on per-decision quality)")

    if not sanity_ols:
        print(f"\nSANITY FAIL: mean initial OLS={mean_init_ols:.3f} not > 1.0")
        print("Warm start MU0_SIGMA may be insufficient. STOP.")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Macro-averaged P/R per detector
    # ---------------------------------------------------------------------------
    det1_prec = [c["det1"]["precision"] for c in cell_records]
    det1_rec  = [c["det1"]["recall"]    for c in cell_records]
    det2_prec = [c["det2"]["precision"] for c in cell_records]
    det2_rec  = [c["det2"]["recall"]    for c in cell_records]

    macro_p1 = float(np.mean(det1_prec))
    macro_r1 = float(np.mean(det1_rec))
    macro_p2 = float(np.mean(det2_prec))
    macro_r2 = float(np.mean(det2_rec))

    det1_gate_p = macro_p1 > 0.70
    det1_gate_r = macro_r1 > 0.80
    det2_gate_p = macro_p2 > 0.70
    det2_gate_r = macro_r2 > 0.80

    det1_pass = det1_gate_p and det1_gate_r
    det2_pass = det2_gate_p and det2_gate_r
    gate_pass = det1_pass and det2_pass

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    out = {
        "experiment":         "V-MV-CONSERVATION-v2",
        "gae_version":        gae.__version__,
        "design": {
            "n_cells":        48,
            "n_seeds":        N_SEEDS,
            "n_decisions":    N_DECISIONS,
            "alpha":          ALPHA,
            "mu0_sigma":      MU0_SIGMA,
            "q_healthy":      Q_HEALTHY,
            "tp_window":      [-TP_PRE, TP_POST],
            "det1_params":    {"h": OLS_H, "lam": OLS_LAM, "k_offset": OLS_K_OFFSET,
                               "window": OLS_WIN, "baseline_n": OLS_BASE_N},
            "det2_params":    {"window": VARQ_WIN, "threshold": VARQ_THR,
                               "q_baseline": "fixed Q_HEALTHY=0.85"},
        },
        "sanity": {
            "mean_initial_ols": round(mean_init_ols, 4),
            "initial_ols_gt1":  sanity_ols,
            "level1_note":      sanity_l1_note,
        },
        "detector_1_ols": {
            "macro_precision":   round(macro_p1, 4),
            "macro_recall":      round(macro_r1, 4),
            "gate_precision":    det1_gate_p,
            "gate_recall":       det1_gate_r,
            "gate_pass":         det1_pass,
        },
        "detector_2_varq": {
            "macro_precision":   round(macro_p2, 4),
            "macro_recall":      round(macro_r2, 4),
            "gate_precision":    det2_gate_p,
            "gate_recall":       det2_gate_r,
            "gate_pass":         det2_pass,
        },
        "overall_gate_pass":  gate_pass,
        "cells":              cell_records,
        "notes": (
            "D1 (OLSMonitor) validated against V-OLS-DETECT design. "
            "D2 (VarQMonitor) uses single-threshold on compute_normalized_var_q; "
            "precision bounded by Bernoulli noise in healthy-state 30-override windows. "
            "D2 gate failure (if any) indicates CUSUM integration needed for precision. "
            "Level 1 count reports ConservationMonitor CUSUM-YELLOW (per-decision q), "
            "not the alpha*q*V Level 1 check (which never fires at these volumes)."
        ),
    }

    out_path = REPO_ROOT / "experiments" / "v_mv_conservation" / "results" / "results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Saved: {out_path}")

    print()
    print(f"V-MV-CONSERVATION v2 (N={N_SEEDS} seeds/cell, {N_DECISIONS} decisions):")
    print(f"  Sanity: initial OLS={mean_init_ols:.3f} > 1.0 -> {'PASS' if sanity_ols else 'FAIL'}")
    print(f"  {sanity_l1_note}")
    print()
    print(f"  DETECTOR 1 (OLS): P={macro_p1:.3f} [>0.70 -> {'PASS' if det1_gate_p else 'FAIL'}]"
          f"  R={macro_r1:.3f} [>0.80 -> {'PASS' if det1_gate_r else 'FAIL'}]"
          f"  -> {'GATE PASS' if det1_pass else 'GATE FAIL'}")
    print(f"  DETECTOR 2 (VarQ): P={macro_p2:.3f} [>0.70 -> {'PASS' if det2_gate_p else 'FAIL'}]"
          f"  R={macro_r2:.3f} [>0.80 -> {'PASS' if det2_gate_r else 'FAIL'}]"
          f"  -> {'GATE PASS' if det2_pass else 'GATE FAIL'}")
    print()
    print(f"  Overall gate: {'GATE PASS' if gate_pass else 'GATE FAIL'}")


if __name__ == "__main__":
    main()
