"""
EXP-REFER-LEARN: Two-stage referral classifier prototype.

Validates whether a separate learned referral mechanism can recover the
product capabilities lost at A=4 without the 13pp accuracy cost of A=5.

Architecture:
  Stage 1: ProfileScorer A=4 (escalate/investigate/suppress/monitor)
  Stage 2: ReferralClassifier (4 mechanisms)
    BASELINE: refer if Stage 1 confidence < 0.70
    M1: per-category centroid + calibrated distance threshold
    M2: 13-feature logistic regression (factors + confidence + context)
    M3: KNN cosine similarity on referral history buffer

Protocol: 500 decisions per seed (300 warmup + 200 eval)
Should-refer rate: 15% across 4 patterns:
  Pattern 1 (3%):  executive account — high asset_criticality + suppress/monitor
  Pattern 2 (4%):  rapid succession  — temporal, same factor distribution as normal
  Pattern 3 (4%):  compliance insider_threat — category forces human sign-off
  Pattern 4 (4%):  high-value data exfil — data_exfiltration + high asset_criticality

Gates (all must pass):
  referral_detection_rate > 60%
  false_referral_rate < 10%
  stage1_accuracy impact < 0.5pp
  referral_learning > 0 (late eval DR > early eval DR)

Usage:
    python experiments/exp_refer_learn/run_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel

# ── Experiment constants ───────────────────────────────────────────────────────
N_SEEDS      = 15
N_TOTAL      = 500
N_WARMUP     = 300
N_EVAL       = N_TOTAL - N_WARMUP   # 200

ETA          = 0.05
ETA_NEG      = 0.05
ETA_OVERRIDE = 0.01
VERIFY_RATE  = 0.30

CONF_THRESH  = 0.70   # baseline: refer if confidence < this

# Referral pattern probability thresholds (cumulative)
P_EXEC   = 0.03   # pattern 1: executive account        (3%)
P_COMPLY = 0.07   # cumulative threshold for pattern 3  (4%)
P_HIVAL  = 0.11   # cumulative threshold for pattern 4  (4%)
P_SUCC   = 0.15   # cumulative threshold for pattern 2  (4%)
# 85% normal alerts; 15% should-refer

# Factor and category indices in soc_product_v50
IDX_ASSET_CRIT     = 1   # asset_criticality
IDX_DATA_EXFIL     = 3   # data_exfiltration category
IDX_INSIDER_THREAT = 4   # insider_threat category
IDX_SUPPRESS       = 2   # suppress action
IDX_MONITOR        = 3   # monitor action

# Pass gates
PASS_DR     = 0.60   # detection rate > 60%
PASS_FPR    = 0.10   # false referral rate < 10%
PASS_S1_PP  = 0.005  # stage1 accuracy drop < 0.5pp

OUTPUT_DIR = Path(__file__).parent / "results"


# ── Persona definitions (same as EXP-A4-DIAGONAL) ────────────────────────────
def _hetero(sigma: float, ratios: list, d: int) -> np.ndarray:
    r = np.array(ratios[:d], dtype=float)
    raw = sigma * r / r.mean()
    return np.clip(raw, 0.03, 0.40)


_D = 6
PERSONAS = [
    {"id": "P1", "name": "FinServ SOC",
     "noise": np.full(_D, 0.08), "q_bar": 0.82, "apd": 200, "kernel": "l2"},
    {"id": "P2", "name": "Healthcare SOC",
     "noise": _hetero(0.22, [0.8, 1.0, 0.9, 2.0, 1.1, 0.8], _D),
     "q_bar": 0.65, "apd": 150, "kernel": "diagonal"},
    {"id": "P3", "name": "Technology SOC",
     "noise": _hetero(0.12, [0.8, 1.0, 0.9, 1.2, 1.0, 0.9], _D),
     "q_bar": 0.78, "apd": 300, "kernel": "l2"},
    {"id": "P4", "name": "Startup SOC",
     "noise": _hetero(0.18, [0.7, 1.2, 0.8, 2.1, 1.1, 0.7], _D),
     "q_bar": 0.70, "apd": 80, "kernel": "diagonal"},
    {"id": "P5", "name": "Enterprise SOC",
     "noise": np.full(_D, 0.10), "q_bar": 0.85, "apd": 400, "kernel": "l2"},
]


# ── Analyst team ───────────────────────────────────────────────────────────────
def make_team(q_bar: float) -> list:
    return [
        {"override_rate": 0.20, "override_quality": min(q_bar + 0.08, 0.98),
         "fatigue_factor": 0.15},
        {"override_rate": 0.27, "override_quality": q_bar,
         "fatigue_factor": 0.22},
        {"override_rate": 0.35, "override_quality": max(q_bar - 0.08, 0.40),
         "fatigue_factor": 0.32},
    ]


def analyst_eff(a: dict) -> Tuple[float, float]:
    ff = a["fatigue_factor"]
    return (min(1.0, a["override_rate"] * (1 + ff * 0.3)),
            max(0.4, a["override_quality"] * (1 - ff * 0.2)))


def make_kernel(ktype: str, noise: np.ndarray):
    if ktype == "l2":
        return L2Kernel()
    w = 1.0 / np.maximum(noise ** 2, 1e-4)
    return DiagonalKernel(w / w.max())


# ── Succession tracker ─────────────────────────────────────────────────────────
class SuccessionTracker:
    """Tracks last-N decision sources for rapid succession detection."""

    def __init__(self, window: int = 10, n_sources: int = 5):
        self._hist: deque = deque(maxlen=window)
        self.n_sources = n_sources

    def add(self, src: int) -> None:
        self._hist.append(src)

    def count(self, src: int) -> int:
        return self._hist.count(src)

    def hot_source(self, rng: np.random.Generator) -> int:
        """Return a source that appears ≥2 times in window; else random."""
        counts = Counter(self._hist)
        hot = [s for s, c in counts.items() if c >= 2]
        return hot[int(rng.integers(len(hot)))] if hot else int(rng.integers(self.n_sources))

    def reset(self) -> None:
        self._hist.clear()


# ── History buffer ─────────────────────────────────────────────────────────────
class HistoryBuffer:
    """
    Shared buffer for (factors, was_referred, was_overridden) triples.
    Used by M2 (override_rate feature) and M3 (KNN refer count).
    """

    def __init__(self, maxlen: int = 200):
        self._buf: deque = deque(maxlen=maxlen)

    def add(self, f: np.ndarray, referred: bool, overridden: bool) -> None:
        self._buf.append((f.copy(), referred, overridden))

    def override_rate(self, f: np.ndarray, sim_thresh: float = 0.70) -> float:
        """Fraction of similar recent alerts (cos_sim > thresh) that were overridden."""
        fn = f / (np.linalg.norm(f) + 1e-8)
        total = overridden = 0
        for fb, _, ov in self._buf:
            bn = fb / (np.linalg.norm(fb) + 1e-8)
            if float(np.dot(fn, bn)) > sim_thresh:
                total += 1
                if ov:
                    overridden += 1
        return overridden / total if total > 0 else 0.0

    def knn_refer_count(self, f: np.ndarray, sim_thresh: float = 0.85) -> int:
        """Count of referred alerts with cos_sim > thresh."""
        fn = f / (np.linalg.norm(f) + 1e-8)
        count = 0
        for fb, ref, _ in self._buf:
            if ref:
                bn = fb / (np.linalg.norm(fb) + 1e-8)
                if float(np.dot(fn, bn)) > sim_thresh:
                    count += 1
        return count

    def reset(self) -> None:
        self._buf.clear()


# ── Stage 2: BASELINE ─────────────────────────────────────────────────────────
class BaselineReferral:
    """Refer if Stage 1 confidence < threshold. No learning."""

    def classify(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf) -> bool:
        return conf < CONF_THRESH

    def update(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf,
               was_referral: bool, was_overridden: bool) -> None:
        hist_buf.add(f, was_referral, was_overridden)

    def on_warmup_end(self) -> None:
        pass


# ── Stage 2: M1 — Centroid-based ─────────────────────────────────────────────
class M1CentroidReferral:
    """
    Per-category referral centroid. Initialize from first 10 referrals.
    Threshold calibrated at warmup end (70th pct of referral distances).
    Online centroid update continues during eval.
    """
    MIN_EXAMPLES  = 3
    CAL_PCT       = 70
    ETA           = 0.05
    INIT_N        = 10

    def __init__(self, C: int, d: int):
        self.C = C
        self.d = d
        self.mu        = np.full((C, d), 0.5)
        self.threshold = np.full(C, np.inf)
        self._init_buf: List[List] = [[] for _ in range(C)]
        self._cal_buf:  List[List] = [[] for _ in range(C)]
        self.ready     = np.zeros(C, dtype=bool)

    def classify(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf) -> bool:
        if not self.ready[c_idx]:
            return False
        return float(np.linalg.norm(f - self.mu[c_idx])) < self.threshold[c_idx]

    def update(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf,
               was_referral: bool, was_overridden: bool) -> None:
        hist_buf.add(f, was_referral, was_overridden)
        if was_referral:
            self._cal_buf[c_idx].append(f.copy())
            buf = self._init_buf[c_idx]
            if len(buf) < self.INIT_N:
                buf.append(f.copy())
                if len(buf) == self.INIT_N:
                    self.mu[c_idx] = np.mean(buf, axis=0)
            else:
                # Online update once initialized
                self.mu[c_idx] += self.ETA * (f - self.mu[c_idx])

    def on_warmup_end(self) -> None:
        """Calibrate thresholds from all collected warmup referrals."""
        for c in range(self.C):
            cal = self._cal_buf[c]
            if len(cal) < self.MIN_EXAMPLES:
                continue
            if len(self._init_buf[c]) < self.INIT_N:
                self.mu[c] = np.mean(cal, axis=0)
            dists = [float(np.linalg.norm(f - self.mu[c])) for f in cal]
            self.threshold[c] = float(np.percentile(dists, self.CAL_PCT))
            self.ready[c] = True


# ── Logistic regression (numpy-only) ──────────────────────────────────────────
class _LogReg:
    def __init__(self, n: int, lr: float = 0.05, lam: float = 0.01,
                 n_iter: int = 400):
        self.w = np.zeros(n)
        self.b = 0.0
        self.lr = lr
        self.lam = lam
        self.n_iter = n_iter

    @staticmethod
    def _sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(y)
        for _ in range(self.n_iter):
            p = self._sig(X @ self.w + self.b)
            e = p - y
            self.w -= self.lr * (X.T @ e / n + self.lam * self.w)
            self.b -= self.lr * float(e.mean())


# ── Stage 2: M2 — Composite discriminant ─────────────────────────────────────
class M2DiscriminantReferral:
    """
    13-feature logistic regression:
      features 0-5:  factor_vector
      feature  6:    stage1 confidence
      features 7-10: stage1 action one-hot (4 actions)
      feature  11:   sequence_count (normalised to [0,1])
      feature  12:   override_rate (fraction of similar recent alerts overridden)

    Collects training data during all warmup decisions where analyst verifies.
    Fits logistic regression at on_warmup_end (maximises training data).
    Threshold 0.30 to bias toward recall (class imbalance: 15% positive).
    Does NOT retrain during eval (batch method, fixed after warmup).
    """
    N_FEAT  = 13
    THRESH  = 0.30

    def __init__(self, n_actions: int = 4):
        self.A = n_actions
        self._X: List[np.ndarray] = []
        self._y: List[float]      = []
        self._xmean: Optional[np.ndarray] = None
        self._xstd:  Optional[np.ndarray] = None
        self._w:     Optional[np.ndarray] = None
        self._b:     float                = 0.0
        self._fitted = False

    def _feats(self, f, conf, action_idx, seq_cnt, override_rate) -> np.ndarray:
        x = np.zeros(self.N_FEAT)
        x[0:6]                      = f
        x[6]                        = conf
        x[7 + min(action_idx, 3)]   = 1.0
        x[11]                       = min(seq_cnt / 10.0, 1.0)
        x[12]                       = override_rate
        return x

    def _predict(self, x: np.ndarray) -> float:
        if not self._fitted:
            return 0.0
        xs = (x - self._xmean) / self._xstd
        z  = float(xs @ self._w) + self._b
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))

    def classify(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf) -> bool:
        or_rate = hist_buf.override_rate(f)
        return self._predict(self._feats(f, conf, action_idx, seq_cnt, or_rate)) > self.THRESH

    def update(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf,
               was_referral: bool, was_overridden: bool) -> None:
        hist_buf.add(f, was_referral, was_overridden)
        or_rate = hist_buf.override_rate(f)
        self._X.append(self._feats(f, conf, action_idx, seq_cnt, or_rate))
        self._y.append(float(was_referral))

    def on_warmup_end(self) -> None:
        if not self._X:
            return
        X = np.array(self._X)
        y = np.array(self._y)
        n_pos = int(y.sum())
        if n_pos < 3:
            return   # too few positives — stay silent
        self._xmean = X.mean(axis=0)
        self._xstd  = X.std(axis=0) + 1e-8
        Xs = (X - self._xmean) / self._xstd
        m  = _LogReg(self.N_FEAT)
        m.fit(Xs, y)
        self._w      = m.w.copy()
        self._b      = m.b
        self._fitted = True


# ── Stage 2: M3 — KNN referral history ───────────────────────────────────────
class M3KNNReferral:
    """
    Refer if ≥ k buffered referral alerts have cosine_sim > sim_thresh.
    Uses shared HistoryBuffer (last 200 verified decisions).
    No explicit training — learns implicitly from accumulation.
    """

    def __init__(self, k: int = 3, sim_thresh: float = 0.85):
        self.k = k
        self.sim = sim_thresh

    def classify(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf) -> bool:
        return hist_buf.knn_refer_count(f, self.sim) >= self.k

    def update(self, f, c_idx, conf, action_idx, seq_cnt, hist_buf,
               was_referral: bool, was_overridden: bool) -> None:
        hist_buf.add(f, was_referral, was_overridden)

    def on_warmup_end(self) -> None:
        pass


# ── Alert generation ───────────────────────────────────────────────────────────
def generate_alert(
    rng: np.random.Generator,
    mu_true: np.ndarray,
    gt_arr: np.ndarray,
    noise: np.ndarray,
    cat_w: np.ndarray,
    tracker: SuccessionTracker,
):
    """
    Returns (factors, category_idx, gt_action, should_refer, pattern, source_id).
    should_refer is assigned BEFORE Stage 2 classification; Stage 2 does NOT see it.
    """
    C, A, d = mu_true.shape
    p = rng.random()

    if p < P_EXEC:
        # Pattern 1: executive account — high asset_criticality, benign action
        ci   = int(rng.choice(C, p=cat_w))
        a_gt = int(rng.choice([IDX_SUPPRESS, IDX_MONITOR]))
        f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)
        f[IDX_ASSET_CRIT] = rng.uniform(0.82, 0.95)
        src  = int(rng.integers(5))
        tracker.add(src)
        return f, ci, a_gt, True, "executive", src

    elif p < P_COMPLY:
        # Pattern 3: compliance insider_threat — category mandate
        ci   = IDX_INSIDER_THREAT
        a_gt = int(rng.choice(A, p=gt_arr[IDX_INSIDER_THREAT]))
        f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)
        src  = int(rng.integers(5))
        tracker.add(src)
        return f, ci, a_gt, True, "compliance", src

    elif p < P_HIVAL:
        # Pattern 4: high-value data exfil
        ci   = IDX_DATA_EXFIL
        a_gt = IDX_MONITOR
        f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)
        f[IDX_ASSET_CRIT] = rng.uniform(0.87, 0.99)
        src  = int(rng.integers(5))
        tracker.add(src)
        return f, ci, a_gt, True, "high_value_data", src

    elif p < P_SUCC:
        # Pattern 2: rapid succession — temporal pattern, normal factor signature
        # Force a "hot" source (≥2 recent appearances) to simulate succession
        ci   = int(rng.choice(C, p=cat_w))
        a_gt = int(rng.choice(A, p=gt_arr[ci]))
        f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)
        src  = tracker.hot_source(rng)
        tracker.add(src)
        return f, ci, a_gt, True, "succession", src

    else:
        # Normal alert (85%)
        ci   = int(rng.choice(C, p=cat_w))
        a_gt = int(rng.choice(A, p=gt_arr[ci]))
        f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)
        src  = int(rng.integers(5))
        tracker.add(src)
        return f, ci, a_gt, False, "", src


# ── EvalRecord ─────────────────────────────────────────────────────────────────
@dataclass
class EvalRecord:
    should_refer: bool
    did_refer:    bool
    s1_correct:   bool
    pattern:      str
    eval_idx:     int    # 0-indexed within eval window (0-199)


# ── Core simulation ────────────────────────────────────────────────────────────
def make_stage2(mechanism: str, C: int, d: int, A: int):
    if mechanism == "baseline": return BaselineReferral()
    if mechanism == "M1":       return M1CentroidReferral(C, d)
    if mechanism == "M2":       return M2DiscriminantReferral(n_actions=A)
    if mechanism == "M3":       return M3KNNReferral()
    raise ValueError(f"Unknown mechanism: {mechanism}")


def run_one_seed(config: dict, persona: dict, mechanism: str, seed: int) -> List[EvalRecord]:
    rng     = np.random.default_rng(42 + seed)
    mu_true = config["mu"]           # (C, A, d)
    cats    = config["categories"]
    actions = config["actions"]
    C, A, d = mu_true.shape

    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(cats):
        p = np.array(config["gt_distributions"].get(cat, [1/A]*A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()
    cat_w = np.ones(C) / C

    noise   = persona["noise"]
    team    = make_team(persona["q_bar"])
    a_effs  = [analyst_eff(a) for a in team]

    # Stage 1
    offset = rng.uniform(-0.15, 0.15, mu_true.shape)
    scorer = ProfileScorer(
        np.clip(mu_true + offset, 0, 1), actions,
        scoring_kernel=make_kernel(persona["kernel"], noise),
        eta_override=ETA_OVERRIDE,
    )
    scorer.eta     = ETA
    scorer.eta_neg = ETA_NEG

    # Stage 2 + shared buffer + succession tracker
    stage2   = make_stage2(mechanism, C, d, A)
    hist_buf = HistoryBuffer(maxlen=200)
    tracker  = SuccessionTracker()

    records: List[EvalRecord] = []
    eval_idx = 0

    for t in range(N_TOTAL):
        f, ci, a_gt, should_refer, pattern, src = generate_alert(
            rng, mu_true, gt_arr, noise, cat_w, tracker
        )

        # Stage 1 scoring
        res    = scorer.score(f, ci)
        pred_a = res.action_index
        conf   = res.confidence

        # Sequence count (for M2 feature)
        seq_cnt = tracker.count(src)

        # Stage 2 classification — does NOT see should_refer
        did_refer = stage2.classify(f, ci, conf, pred_a, seq_cnt, hist_buf)

        # Record eval-window decision
        if t >= N_WARMUP:
            records.append(EvalRecord(
                should_refer = should_refer,
                did_refer    = did_refer,
                s1_correct   = (pred_a == a_gt),
                pattern      = pattern,
                eval_idx     = eval_idx,
            ))
            eval_idx += 1

        # Analyst verification (single event per alert)
        if rng.random() < VERIFY_RATE:
            ai_idx  = int(rng.integers(len(team)))
            eff_or, eff_q = a_effs[ai_idx]
            was_overridden = False

            if rng.random() < eff_or:
                gt_a = a_gt if rng.random() < eff_q else int(
                    rng.choice([a for a in range(A) if a != a_gt])
                )
                was_overridden = (gt_a != pred_a)
                # Stage 1 learns only on non-referral alerts
                if not should_refer:
                    scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
            else:
                if not should_refer:
                    scorer.update(f, ci, pred_a, True)

            # Stage 2 always gets feedback when analyst verifies
            stage2.update(f, ci, conf, pred_a, seq_cnt, hist_buf,
                          should_refer, was_overridden)

        # Warmup-end hook
        if t == N_WARMUP - 1:
            stage2.on_warmup_end()

    return records


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(records: List[EvalRecord]) -> dict:
    TP = FN = FP = TN = 0
    s1_corr_tn = s1_tot_tn = 0
    comb_corr  = 0

    # Split for learning measurement
    early = [r for r in records if r.eval_idx < 50]
    late  = [r for r in records if r.eval_idx >= 150]

    def dr_subset(recs):
        tp  = sum(1 for r in recs if r.should_refer and r.did_refer)
        tot = sum(1 for r in recs if r.should_refer)
        return tp / tot if tot > 0 else 0.0

    for r in records:
        if r.should_refer:
            if r.did_refer:
                TP += 1
                comb_corr += 1   # correct routing
            else:
                FN += 1          # missed referral — incorrect
        else:
            if r.did_refer:
                FP += 1
                comb_corr += int(r.s1_correct)   # false alarm; credit S1 if right
            else:
                TN += 1
                s1_tot_tn  += 1
                s1_corr_tn += int(r.s1_correct)
                comb_corr  += int(r.s1_correct)

    n = len(records)
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    false_ref_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    s1_accuracy    = s1_corr_tn / s1_tot_tn if s1_tot_tn > 0 else 0.0
    combined_acc   = comb_corr  / n         if n > 0      else 0.0
    early_dr       = dr_subset(early)
    late_dr        = dr_subset(late)

    return {
        "detection_rate":    round(detection_rate,      4),
        "false_ref_rate":    round(false_ref_rate,      4),
        "s1_accuracy":       round(s1_accuracy,         4),
        "combined_acc":      round(combined_acc,        4),
        "early_dr":          round(early_dr,            4),
        "late_dr":           round(late_dr,             4),
        "referral_learning": round(late_dr - early_dr,  4),
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "n_should_refer":    TP + FN,
        "n_total":           n,
    }


def run_cell(config: dict, persona: dict, mechanism: str) -> dict:
    seed_metrics = []
    all_records  = []
    for si in range(N_SEEDS):
        recs = run_one_seed(config, persona, mechanism, si)
        all_records.extend(recs)
        seed_metrics.append(compute_metrics(recs))

    agg = compute_metrics(all_records)

    # ±95% CI across seeds
    for key in ("detection_rate", "false_ref_rate"):
        vals = [m[key] for m in seed_metrics]
        agg[f"{key}_ci"] = round(1.96 * float(np.std(vals)) / N_SEEDS ** 0.5, 4)

    return agg


# ── Gate evaluation ────────────────────────────────────────────────────────────
def gate_result(m: dict, s1_baseline: float) -> Tuple[bool, str]:
    dr_ok  = m["detection_rate"] > PASS_DR
    fpr_ok = m["false_ref_rate"] < PASS_FPR
    s1_ok  = (s1_baseline - m["s1_accuracy"]) < PASS_S1_PP
    lrn_ok = m["referral_learning"] > 0.0
    ok     = dr_ok and fpr_ok and s1_ok and lrn_ok
    detail = (
        f"DR={'ok' if dr_ok else 'FAIL'}({m['detection_rate']:.0%})"
        f" FPR={'ok' if fpr_ok else 'FAIL'}({m['false_ref_rate']:.0%})"
        f" S1={'ok' if s1_ok else 'FAIL'}({s1_baseline-m['s1_accuracy']:+.3f})"
        f" Lrn={'ok' if lrn_ok else 'FAIL'}({m['referral_learning']:+.2f})"
    )
    return ok, detail


# ── Print helpers ──────────────────────────────────────────────────────────────
def print_results(all_rows: list) -> None:
    print()
    print("=" * 110)
    print("EXP-REFER-LEARN: Full results")
    print(f"  Gates: DR>{PASS_DR:.0%}  FPR<{PASS_FPR:.0%}"
          f"  S1-impact<{PASS_S1_PP:.3f}  learning>0")
    print("=" * 110)
    print()
    print(f"  {'Persona':<18} {'Mech':<10} {'Detection':>10} {'FalseRef':>9}"
          f" {'S1 Acc':>8} {'S1 Imp':>8} {'Lrn':>7} {'Result':>7}")
    print("  " + "-" * 88)

    for row in all_rows:
        m    = row["metrics"]
        base = row["s1_baseline"]
        ok, _detail = gate_result(m, base)
        flag = " PASS" if ok else " fail"
        print(f"  {row['persona_name']:<18} {row['mechanism']:<10}"
              f" {m['detection_rate']:>8.1%}±{m['detection_rate_ci']:.1%}"
              f" {m['false_ref_rate']:>8.1%}"
              f" {m['s1_accuracy']:>8.1%}"
              f" {base - m['s1_accuracy']:>+7.3f}"
              f" {m['referral_learning']:>+6.2f}"
              f" {flag}")

    # Per-mechanism aggregate
    print()
    print("  MECHANISM AGGREGATE (mean across 5 personas):")
    print(f"  {'Mech':<10} {'Mean DR':>9} {'Mean FPR':>10} {'Mean S1':>9}"
          f" {'Mean Lrn':>10} {'#PASS':>7}")
    print("  " + "-" * 60)
    for mname in ["baseline", "M1", "M2", "M3"]:
        rows = [r for r in all_rows if r["mechanism"] == mname]
        if not rows:
            continue
        mdr  = float(np.mean([r["metrics"]["detection_rate"]    for r in rows]))
        mfpr = float(np.mean([r["metrics"]["false_ref_rate"]     for r in rows]))
        ms1  = float(np.mean([r["metrics"]["s1_accuracy"]        for r in rows]))
        mlrn = float(np.mean([r["metrics"]["referral_learning"]  for r in rows]))
        npas = sum(1 for r in rows if gate_result(r["metrics"], r["s1_baseline"])[0])
        print(f"  {mname:<10} {mdr:>8.1%} {mfpr:>10.1%} {ms1:>9.1%}"
              f" {mlrn:>+9.2f} {npas:>7}/{len(rows)}")

    # Early vs late learning
    print()
    print("  EARLY (eval 0-49) vs LATE (eval 150-199) DETECTION RATE:")
    print(f"  {'Mech':<10} {'Early DR':>10} {'Late DR':>9} {'Improvement':>13}")
    print("  " + "-" * 46)
    for mname in ["baseline", "M1", "M2", "M3"]:
        rows = [r for r in all_rows if r["mechanism"] == mname]
        if not rows:
            continue
        early = float(np.mean([r["metrics"]["early_dr"] for r in rows]))
        late  = float(np.mean([r["metrics"]["late_dr"]  for r in rows]))
        print(f"  {mname:<10} {early:>10.1%} {late:>9.1%} {late-early:>+12.1%}")

    # Verdict
    n_pass = sum(1 for r in all_rows if gate_result(r["metrics"], r["s1_baseline"])[0])
    passing = [(r["persona_name"], r["mechanism"], r["metrics"])
               for r in all_rows if gate_result(r["metrics"], r["s1_baseline"])[0]]
    mech_counts = Counter(m for _, m, _ in passing)

    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)

    if n_pass == 0:
        print()
        print("  NO mechanism passes all gates.")
        print("  Confidence gate (BASELINE) is sufficient for v6.0.")
        print("  Two-stage referral architecture revisited at v7.0")
        print("  with production referral data and longer training window.")
    else:
        best_mech = mech_counts.most_common(1)[0]
        n_mechs   = len(mech_counts)
        print()
        print(f"  {n_pass} cell(s) pass across {n_mechs} mechanism(s).")
        print(f"  Best mechanism: {best_mech[0]} ({best_mech[1]}/{len(PERSONAS)} personas passing)")
        print()
        print("  Two-stage architecture is VIABLE.")
        print("  → Design ReferralClassifier as GAE module.")
        print("  → Add to v6.5 roadmap with shipping timeline.")
        print()
        print("  Passing cells:")
        for pname, mname, m in passing:
            print(f"    {pname:<18} {mname:<10}"
                  f"  DR={m['detection_rate']:.1%}"
                  f"  FPR={m['false_ref_rate']:.1%}"
                  f"  lrn={m['referral_learning']:+.2f}"
                  f"  TP={m['TP']}  FP={m['FP']}"
                  f"  n_refer={m['n_should_refer']}")

    print("=" * 110)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config = load_domain_config("soc_product_v50")
    C, A, d = config["mu"].shape
    mechanisms = ["baseline", "M1", "M2", "M3"]

    print()
    print("=" * 70)
    print("=== EXP-REFER-LEARN: Two-stage referral classifier prototype ===")
    print("=" * 70)
    print(f"  Config: A={A}  C={C}  d={d}  (soc_product_v50)")
    print(f"  Personas: {len(PERSONAS)}  Mechanisms: {len(mechanisms)}"
          f"  Seeds: {N_SEEDS}")
    print(f"  Protocol: {N_WARMUP} warmup + {N_EVAL} eval decisions per seed")
    print(f"  Should-refer: 15% (exec 3% + comply 4% + hival 4% + succession 4%)")
    print(f"  Total runs: {len(PERSONAS)} × {len(mechanisms)} × {N_SEEDS} = "
          f"{len(PERSONAS)*len(mechanisms)*N_SEEDS}")

    all_rows   = []
    n_cells    = len(PERSONAS) * len(mechanisms)
    n_done     = 0
    t_total    = time.time()

    for persona in PERSONAS:
        noise = persona["noise"]
        ratio = float(noise.max() / max(noise.min(), 0.001))
        print(f"\n  ── {persona['id']}: {persona['name']}"
              f"  σ={noise.mean():.3f}  ratio={ratio:.1f}×"
              f"  q̄={persona['q_bar']}  V={persona['apd']}"
              f"  kernel={persona['kernel']} ──")

        persona_metrics: dict = {}

        for mname in mechanisms:
            n_done += 1
            t0 = time.time()
            m  = run_cell(config, persona, mname)
            elapsed = time.time() - t0

            print(f"    [{n_done:>2}/{n_cells}] {mname:<10}"
                  f"  DR={m['detection_rate']:.1%}±{m['detection_rate_ci']:.1%}"
                  f"  FPR={m['false_ref_rate']:.1%}"
                  f"  S1={m['s1_accuracy']:.1%}"
                  f"  lrn={m['referral_learning']:+.2f}"
                  f"  TP={m['TP']}  FP={m['FP']}"
                  f"  ({elapsed:.1f}s)")
            persona_metrics[mname] = m

        s1_baseline = persona_metrics["baseline"]["s1_accuracy"]
        for mname in mechanisms:
            all_rows.append({
                "persona_id":   persona["id"],
                "persona_name": persona["name"],
                "mechanism":    mname,
                "s1_baseline":  round(s1_baseline, 4),
                "metrics":      persona_metrics[mname],
            })

    total_time = time.time() - t_total
    print(f"\n  Completed {n_cells} cells in {total_time:.1f}s "
          f"({total_time / n_cells:.1f}s/cell  "
          f"{total_time / (n_cells * N_SEEDS):.2f}s/seed)")

    print_results(all_rows)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "n_seeds": N_SEEDS, "n_warmup": N_WARMUP, "n_eval": N_EVAL,
                "should_refer_rate": 0.15,
                "pass_dr": PASS_DR, "pass_fpr": PASS_FPR, "pass_s1_pp": PASS_S1_PP,
                "total_runtime_s": round(total_time, 1),
            },
            "results": all_rows,
        }, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}\n")


if __name__ == "__main__":
    main()
