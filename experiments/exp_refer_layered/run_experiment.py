"""
EXP-REFER-LAYERED: Layered referral architecture head-to-head.

5 layers × 5 personas × 15 seeds = 375 runs.

Layers:
  0: No referral (baseline cost)
  1: Confidence gate only  (current A=4 system)
  2: Policy rules R1-R7 only  (no ML, Day 1 value)
  3: Policy rules + confidence gate  (combined)
  4: Policy rules + override learning  (numpy LR on rule-passing alerts)

Prerequisite: EXP-REFER-COVERAGE established:
  Rules R1-R6 cover 61% of referrals at FPR=12% (precise).
  Conf gate alone: FPR=38.7% (noisy). R7 adds +6.8pp DR at same FPR.
  Emergent (R8+R10): 20.7% of referrals — the learning opportunity.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel

# ── Constants ─────────────────────────────────────────────────────────────────
N_WARMUP       = 1500
N_EVAL         = 500
N_SEEDS        = 15
N_LAYERS       = 5
CONF_THRESHOLD = 0.70
VERIFY_RATE    = 0.30
N_ENTITIES     = 50      # entity pool per persona
NOVELTY_CAP    = 100     # max factor vectors per category in novelty buffer

# Referral reason rates (fraction of ALL alerts)
REFERRAL_REASONS = {
    "R1": 0.020,   # executive/VIP
    "R2": 0.020,   # rapid succession
    "R3": 0.015,   # compliance-mandated category
    "R4": 0.015,   # high-value data movement
    "R5": 0.010,   # active IR
    "R6": 0.015,   # new asset
    "R7": 0.015,   # cross-category correlation (CONTEXT)
    "R8": 0.020,   # gut feel (EMERGENT)
    "R9": 0.005,   # seasonal (CONTEXT)
    "R10": 0.010,  # prior false negative (EMERGENT)
}
TOTAL_REFER_RATE = sum(REFERRAL_REASONS.values())  # 0.15
REASON_TYPES = {
    "R1": "RULE", "R2": "RULE", "R3": "RULE", "R4": "RULE",
    "R5": "RULE", "R6": "RULE", "R7": "CONTEXT",
    "R8": "EMERGENT", "R9": "CONTEXT", "R10": "EMERGENT",
}

# Miss cost (minutes) — analyst hours wasted if referral is missed
MISS_COSTS = {
    "R1": 60, "R2": 30, "R3": 60, "R4": 30,
    "R5": 60, "R6": 30, "R7": 30, "R8": 30, "R9": 30, "R10": 30,
}
FALSE_REF_COST    = 3    # minutes per unnecessary referral
AUTO_APPROVE_SAVE = 15   # minutes saved per TN (correct auto-approve)

LAYER_NAMES = [
    "L0: None",
    "L1: Conf gate",
    "L2: Rules only",
    "L3: Rules+Conf",
    "L4: Rules+Learn",
]

# 18 features for override model
FEATURE_NAMES = [
    "factor_0_travel",      "factor_1_asset_crit",  "factor_2_threat_intel",
    "factor_3_pattern_hist","factor_4_time_anom",   "factor_5_device_trust",
    "confidence",           "conf_margin",           "category_norm",
    "time_of_day",          "day_of_week",
    "alert_velocity",       "entity_risk_tier",
    "entity_override_rate", "cat_override_rate",
    "alert_novelty",        "active_policy_flags",   "business_context_flag",
]
N_FEATURES = len(FEATURE_NAMES)  # 18

OUTPUT_DIR = Path(__file__).parent / "results"

# ── Persona definitions ────────────────────────────────────────────────────────
_D = 6

def _hetero_noise(sigma_mean: float, ratios: list, d: int) -> np.ndarray:
    r = np.array(ratios[:d], dtype=float)
    raw = sigma_mean * r * (sigma_mean / (sigma_mean * r.mean()))
    return np.clip(raw, 0.03, 0.40)

PERSONAS = [
    {"id": "P1", "name": "FinServ SOC",
     "noise": np.full(_D, 0.08),  "q_bar": 0.82, "apd": 200, "kernel": "l2"},
    {"id": "P2", "name": "Healthcare SOC",
     "noise": _hetero_noise(0.22, [0.8, 1.0, 0.9, 2.0, 1.1, 0.8], _D),
     "q_bar": 0.65, "apd": 150, "kernel": "diagonal"},
    {"id": "P3", "name": "Technology SOC",
     "noise": _hetero_noise(0.12, [0.8, 1.0, 0.9, 1.2, 1.0, 0.9], _D),
     "q_bar": 0.78, "apd": 300, "kernel": "l2"},
    {"id": "P4", "name": "Startup SOC",
     "noise": _hetero_noise(0.18, [0.7, 1.2, 0.8, 2.1, 1.1, 0.7], _D),
     "q_bar": 0.70, "apd": 80,  "kernel": "diagonal"},
    {"id": "P5", "name": "Enterprise SOC",
     "noise": np.full(_D, 0.10), "q_bar": 0.85, "apd": 400, "kernel": "l2"},
]


# ── Kernel builder ─────────────────────────────────────────────────────────────
def build_kernel(kernel_type: str, noise: np.ndarray):
    if kernel_type == "l2":
        return L2Kernel()
    inv_var = 1.0 / np.maximum(noise ** 2, 1e-4)
    return DiagonalKernel(inv_var / inv_var.max())


# ── Alert generation ───────────────────────────────────────────────────────────
def generate_alert(
    rng: np.random.Generator,
    config: dict,
    persona: dict,
    insider_idx: int,
    exfil_idx: int,
    step: int,
) -> dict:
    """Generate a single alert with referral metadata."""
    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape
    noise      = persona["noise"]

    # GT distribution array
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0 / A] * A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()
    cat_w = np.ones(C) / C

    entity_id = int(rng.integers(N_ENTITIES))

    # Decide should_refer and reason
    u = rng.random()
    should_refer    = u < TOTAL_REFER_RATE
    referral_reason = None

    if should_refer:
        cum = 0.0
        for reason, rate in REFERRAL_REASONS.items():
            cum += rate
            if u < cum:
                referral_reason = reason
                break

    # Draw category and GT action
    ci   = int(rng.choice(C, p=cat_w))
    a_gt = int(rng.choice(A, p=gt_arr[ci]))

    # Override category for R3/R4 to satisfy rule conditions
    if referral_reason == "R3" and insider_idx >= 0:
        ci   = insider_idx
        a_gt = int(rng.choice(A, p=gt_arr[ci]))
    elif referral_reason == "R4" and exfil_idx >= 0:
        ci   = exfil_idx
        a_gt = int(rng.choice(A, p=gt_arr[ci]))

    f = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0.0, 1.0)

    # Contextual metadata
    is_r1 = referral_reason == "R1"
    is_r2 = referral_reason == "R2"
    is_r3 = referral_reason == "R3"
    is_r4 = referral_reason == "R4"
    is_r5 = referral_reason == "R5"
    is_r6 = referral_reason == "R6"
    is_r7 = referral_reason == "R7"
    is_r9 = referral_reason == "R9"

    identity_tier  = "executive" if is_r1 else rng.choice(
        ["standard", "service_account", "executive"], p=[0.75, 0.20, 0.05]
    )
    sequence_count = int(rng.integers(3, 8)) if is_r2 else int(rng.integers(0, 3))
    compliance_mode = True if is_r3 else bool(rng.random() < 0.10)
    incident_active = True if is_r5 else bool(rng.random() < 0.05)
    asset_age_days  = int(rng.integers(1, 30)) if is_r6 else int(rng.integers(30, 366))
    cross_cat_count = int(rng.integers(2, 5)) if is_r7 else int(rng.integers(0, 2))
    asset_crit      = float(rng.uniform(0.86, 1.0)) if is_r4 else float(rng.uniform(0.0, 1.0))
    business_cycle  = "quarter_end" if is_r9 else rng.choice(
        ["normal", "quarter_end"], p=[0.90, 0.10]
    )

    return {
        "factors":         f,
        "ci":              ci,
        "a_gt":            a_gt,
        "category_name":   categories[ci],
        "should_refer":    should_refer,
        "referral_reason": referral_reason,
        "entity_id":       entity_id,
        "step":            step,
        "meta": {
            "identity_tier":    identity_tier,
            "sequence_count":   sequence_count,
            "compliance_mode":  compliance_mode,
            "incident_active":  incident_active,
            "asset_age_days":   asset_age_days,
            "cross_cat_count":  cross_cat_count,
            "asset_criticality": asset_crit,
            "business_cycle":   business_cycle,
        },
    }


# ── Rule checking ──────────────────────────────────────────────────────────────
def check_rules(alert: dict, action_name: str) -> dict:
    """Returns {rule_id: bool, 'any': bool} for rules R1-R7."""
    m  = alert["meta"]
    cn = alert["category_name"]

    r1 = m["identity_tier"] in ("executive", "board", "c_suite")
    r2 = m["sequence_count"] >= 3
    r3 = (cn == "insider_threat") and m["compliance_mode"]
    r4 = (cn == "data_exfiltration") and (m["asset_criticality"] > 0.85) \
         and (action_name in ("monitor", "suppress"))
    r5 = m["incident_active"]
    r6 = m["asset_age_days"] < 30
    r7 = m["cross_cat_count"] >= 2

    return {
        "R1": r1, "R2": r2, "R3": r3, "R4": r4,
        "R5": r5, "R6": r6, "R7": r7,
        "any": r1 or r2 or r3 or r4 or r5 or r6 or r7,
    }


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(
    alert: dict,
    res,
    step: int,
    entity_override_counts: dict,
    entity_alert_counts: dict,
    cat_override_counts: list,
    cat_alert_counts: list,
    novelty_buffers: list,
    recent_entities: deque,
    C: int,
) -> np.ndarray:
    """Extract 18 features for the override model."""
    f   = alert["factors"]   # (d,)
    m   = alert["meta"]
    ci  = alert["ci"]
    eid = alert["entity_id"]

    probs    = res.probabilities
    sorted_p = np.sort(probs)[::-1]
    margin   = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 0.0

    # Temporal (simulated from step counter)
    time_of_day = (step % 24) / 24.0
    day_of_week = ((step // 24) % 7) / 7.0

    # Alert velocity: fraction of last-20-decisions from this entity
    recent_list = list(recent_entities)
    velocity    = sum(1 for x in recent_list if x == eid) / max(len(recent_list), 1)

    # Entity risk tier
    tier_map    = {"executive": 1.0, "board": 1.0, "c_suite": 1.0,
                   "service_account": 0.3, "standard": 0.5}
    entity_risk = tier_map.get(m["identity_tier"], 0.5)

    # Override rates (with Laplace smoothing)
    entity_override_rate = entity_override_counts.get(eid, 0) / max(entity_alert_counts.get(eid, 1), 1)
    cat_override_rate    = cat_override_counts[ci] / max(cat_alert_counts[ci], 1)

    # Alert novelty: cosine distance to nearest prior vector in same category
    buf = novelty_buffers[ci]
    if len(buf) == 0:
        novelty = 1.0
    else:
        fn = f / (np.linalg.norm(f) + 1e-9)
        dists = []
        for prior in buf[-50:]:
            pn = prior / (np.linalg.norm(prior) + 1e-9)
            dists.append(1.0 - float(np.dot(fn, pn)))
        novelty = float(min(dists))

    active_policy = float(m["compliance_mode"] or m["incident_active"])
    biz_context   = float(m["business_cycle"] == "quarter_end")

    return np.array([
        *f.tolist(),                          # 0-5: factor values
        float(res.confidence),                # 6: confidence
        margin,                               # 7: confidence margin
        ci / max(C - 1, 1),                   # 8: category index (normalised)
        time_of_day,                          # 9
        day_of_week,                          # 10
        velocity,                             # 11: alert velocity
        entity_risk,                          # 12: entity risk tier
        entity_override_rate,                 # 13: entity override history
        cat_override_rate,                    # 14: category override history
        novelty,                              # 15: alert novelty
        active_policy,                        # 16: active policy flag
        biz_context,                          # 17: business context flag
    ], dtype=np.float64)


# ── Numpy logistic regression (override model) ─────────────────────────────────
class _LogReg:
    """Numpy-only logistic regression with class balancing."""

    def __init__(self, n_iter: int = 300, lr: float = 0.05, lam: float = 0.01):
        self.n_iter = n_iter
        self.lr     = lr
        self.lam    = lam
        self.w: np.ndarray = None
        self.b: float      = 0.0
        self.coef_: np.ndarray = None

    @staticmethod
    def _sig(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogReg":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        # Class weights: balance positive/negative
        n_pos = max(int(y.sum()), 1)
        n_neg = max(n - n_pos, 1)
        weights = np.where(y == 1, n_neg / n_pos, 1.0)
        for _ in range(self.n_iter):
            p = self._sig(X @ self.w + self.b)
            e = (p - y) * weights
            self.w -= self.lr * (X.T @ e / n + self.lam * self.w)
            self.b -= self.lr * float(e.mean())
        self.coef_ = np.array([self.w])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._sig(X @ self.w + self.b)
        return np.column_stack([1.0 - p, p])


def train_override_model(
    X_list: list, y_list: list
) -> tuple[Optional[_LogReg], float, dict]:
    """
    Train override model on rule-passing warmup alerts.
    Returns (model, threshold, importances) or (None, 0.5, {}).
    """
    if len(X_list) < 20 or sum(y_list) < 3:
        return None, 0.5, {}

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    if len(np.unique(y)) < 2:
        return None, 0.5, {}

    # 80/20 split for threshold calibration
    n_val   = max(5, len(X) // 5)
    X_train = X[:-n_val]
    y_train = y[:-n_val]
    X_val   = X[-n_val:]
    y_val   = y[-n_val:]

    if len(np.unique(y_train)) < 2:
        return None, 0.5, {}

    model = _LogReg().fit(X_train, y_train)

    # Calibrate: lowest threshold achieving validation FPR < 5%
    best_thresh = 0.7   # conservative default (effectively off unless data is rich)
    probs_val   = model.predict_proba(X_val)[:, 1]
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (probs_val >= thresh).astype(int)
        fp_v  = int(((preds == 1) & (y_val == 0)).sum())
        tn_v  = int(((preds == 0) & (y_val == 0)).sum())
        fpr_v = fp_v / max(fp_v + tn_v, 1)
        if fpr_v < 0.05:
            best_thresh = thresh
            break

    importances = {
        name: float(abs(coef))
        for name, coef in zip(FEATURE_NAMES, model.coef_[0])
    }
    return model, best_thresh, importances


# ── Per-layer result accumulator ───────────────────────────────────────────────
@dataclass
class LayerResult:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    tp_by_reason:    dict = field(default_factory=dict)
    fn_by_reason:    dict = field(default_factory=dict)
    total_by_reason: dict = field(default_factory=dict)
    s1_correct: int = 0
    s1_total:   int = 0

    def record(self, referred: bool, should_refer: bool, reason: Optional[str]):
        self.s1_correct += 0   # filled externally
        self.s1_total   += 1   # placeholder; actual update done in run_seed
        if should_refer:
            self.total_by_reason[reason] = self.total_by_reason.get(reason, 0) + 1
            if referred:
                self.tp += 1
                self.tp_by_reason[reason] = self.tp_by_reason.get(reason, 0) + 1
            else:
                self.fn += 1
                self.fn_by_reason[reason] = self.fn_by_reason.get(reason, 0) + 1
        else:
            if referred:
                self.fp += 1
            else:
                self.tn += 1


# ── Core simulation (single seed) ─────────────────────────────────────────────
def run_seed(config: dict, persona: dict, seed: int) -> tuple[list, dict]:
    """
    Run all 5 layers for a single seed.
    Stage 1 updates are identical for all layers (Layer 0 behavior).
    Returns (list[LayerResult], l4_extras).
    """
    rng = np.random.default_rng(seed)

    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape
    noise      = persona["noise"]
    q_bar      = persona["q_bar"]

    # GT array
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0 / A] * A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()

    # Category indices for R3/R4
    cat_idx     = {c: i for i, c in enumerate(categories)}
    insider_idx = cat_idx.get("insider_threat", -1)
    exfil_idx   = cat_idx.get("data_exfiltration", -1)

    # Stage 1
    kernel = build_kernel(persona["kernel"], noise)
    offset = rng.uniform(-0.15, 0.15, mu_true.shape)
    scorer = ProfileScorer(
        np.clip(mu_true + offset, 0.0, 1.0), actions,
        scoring_kernel=kernel, eta_override=0.01,
    )
    scorer.eta     = 0.05
    scorer.eta_neg = 0.05

    # History tracking for Layer 4 features
    entity_override_counts: dict = defaultdict(int)
    entity_alert_counts:    dict = defaultdict(int)
    cat_override_counts: list    = [0] * C
    cat_alert_counts:    list    = [1] * C    # Laplace smoothing
    novelty_buffers: list        = [[] for _ in range(C)]
    recent_entities              = deque(maxlen=20)

    # Layer 4 training data
    l4_X: list = []
    l4_y: list = []

    # ── Warmup ────────────────────────────────────────────────────────────────
    for step in range(N_WARMUP):
        alert       = generate_alert(rng, config, persona, insider_idx, exfil_idx, step)
        f, ci, a_gt = alert["factors"], alert["ci"], alert["a_gt"]

        res         = scorer.score(f, ci)
        action_name = actions[res.action_index]
        rules       = check_rules(alert, action_name)

        feats = extract_features(
            alert, res, step,
            entity_override_counts, entity_alert_counts,
            cat_override_counts, cat_alert_counts,
            novelty_buffers, recent_entities, C,
        )

        # Layer 4 training data: rule-passing alerts only
        if not rules["any"]:
            if alert["should_refer"]:
                if rng.random() < q_bar:       # analyst catches with probability q̄
                    l4_X.append(feats)
                    l4_y.append(1)
            else:
                l4_X.append(feats)
                l4_y.append(0)

        # Stage 1 update — identical for all layers
        if rng.random() < VERIFY_RATE:
            eff_q = q_bar * (1.0 - 0.2 * rng.random())
            if rng.random() < eff_q:
                if res.action_index == a_gt:
                    scorer.update(f, ci, res.action_index, True)
                else:
                    scorer.update(f, ci, res.action_index, False, gt_action_index=a_gt)

        # Update history
        eid = alert["entity_id"]
        recent_entities.append(eid)
        entity_alert_counts[eid] += 1
        cat_alert_counts[ci]     += 1
        if alert["should_refer"] and rng.random() < q_bar:
            entity_override_counts[eid] += 1
            cat_override_counts[ci]     += 1
        novelty_buffers[ci].append(f.copy())
        if len(novelty_buffers[ci]) > NOVELTY_CAP:
            novelty_buffers[ci].pop(0)

    # ── Train Layer 4 override model ──────────────────────────────────────────
    l4_model, l4_thresh, l4_importances = train_override_model(l4_X, l4_y)

    # ── Eval phase: all 5 layers simultaneously ───────────────────────────────
    layer_results = [LayerResult() for _ in range(N_LAYERS)]
    s1_correct_total = 0

    # Layer 4 deep metrics
    l4_rule_tp    = 0
    l4_model_tp   = 0
    l4_model_fp   = 0
    r8_total  = 0;  r8_detected  = 0
    r10_total = 0;  r10_detected = 0
    early_tp  = 0;  early_n_ref  = 0   # first 100 eval steps
    late_tp   = 0;  late_n_ref   = 0   # last 100 eval steps

    for step_eval in range(N_EVAL):
        step  = N_WARMUP + step_eval
        alert = generate_alert(rng, config, persona, insider_idx, exfil_idx, step)
        f, ci, a_gt = alert["factors"], alert["ci"], alert["a_gt"]
        sr     = alert["should_refer"]
        reason = alert["referral_reason"]

        # Stage 1 — same for all layers
        res         = scorer.score(f, ci)
        action_name = actions[res.action_index]
        s1_correct  = int(res.action_index == a_gt)
        s1_correct_total += s1_correct

        rules = check_rules(alert, action_name)

        feats = extract_features(
            alert, res, step,
            entity_override_counts, entity_alert_counts,
            cat_override_counts, cat_alert_counts,
            novelty_buffers, recent_entities, C,
        )

        # Layer referral decisions
        conf_flag  = float(res.confidence) < CONF_THRESHOLD
        rules_flag = rules["any"]

        # Layer 4 model
        l4_model_flag = False
        if l4_model is not None and not rules_flag:
            prob = float(l4_model.predict_proba(feats.reshape(1, -1))[0, 1])
            l4_model_flag = prob >= l4_thresh

        refers = [
            False,                             # L0: no referral
            conf_flag,                         # L1: confidence gate
            rules_flag,                        # L2: rules only
            rules_flag or conf_flag,           # L3: rules + conf
            rules_flag or l4_model_flag,       # L4: rules + model
        ]

        # Record per layer
        for li, refer in enumerate(refers):
            lr = layer_results[li]
            lr.s1_correct += s1_correct
            lr.s1_total   += 1
            if sr:
                lr.total_by_reason[reason] = lr.total_by_reason.get(reason, 0) + 1
                if refer:
                    lr.tp += 1
                    lr.tp_by_reason[reason] = lr.tp_by_reason.get(reason, 0) + 1
                else:
                    lr.fn += 1
                    lr.fn_by_reason[reason] = lr.fn_by_reason.get(reason, 0) + 1
            else:
                if refer:
                    lr.fp += 1
                else:
                    lr.tn += 1

        # Layer 4 deep tracking
        if sr:
            if rules_flag:
                l4_rule_tp += 1
            elif l4_model_flag:
                l4_model_tp += 1
            if reason == "R8":
                r8_total    += 1
                r8_detected += int(refers[4])
            elif reason == "R10":
                r10_total    += 1
                r10_detected += int(refers[4])
        else:
            if l4_model_flag:
                l4_model_fp += 1

        # Time-to-learn (first and last 100 eval steps)
        if step_eval < 100:
            early_n_ref += int(sr)
            early_tp    += int(sr and refers[4])
        elif step_eval >= 400:
            late_n_ref += int(sr)
            late_tp    += int(sr and refers[4])

        # Stage 1 update (Layer 0 behavior — identical for all layers)
        if rng.random() < VERIFY_RATE:
            eff_q = q_bar * (1.0 - 0.2 * rng.random())
            if rng.random() < eff_q:
                if res.action_index == a_gt:
                    scorer.update(f, ci, res.action_index, True)
                else:
                    scorer.update(f, ci, res.action_index, False, gt_action_index=a_gt)

        # Update history
        eid = alert["entity_id"]
        recent_entities.append(eid)
        entity_alert_counts[eid] += 1
        cat_alert_counts[ci]     += 1
        if sr and rng.random() < q_bar:
            entity_override_counts[eid] += 1
            cat_override_counts[ci]     += 1
        novelty_buffers[ci].append(f.copy())
        if len(novelty_buffers[ci]) > NOVELTY_CAP:
            novelty_buffers[ci].pop(0)

    l4_extras = {
        "model_trained":   l4_model is not None,
        "l4_threshold":    float(l4_thresh),
        "n_train_pos":     int(sum(l4_y)),
        "n_train_neg":     int(len(l4_y) - sum(l4_y)),
        "l4_rule_tp":      l4_rule_tp,
        "l4_model_tp":     l4_model_tp,
        "l4_model_fp":     l4_model_fp,
        "r8_dr":           r8_detected  / max(r8_total,  1),
        "r10_dr":          r10_detected / max(r10_total, 1),
        "early_dr":        early_tp / max(early_n_ref, 1),
        "late_dr":         late_tp  / max(late_n_ref,  1),
        "importances":     l4_importances,
    }
    return layer_results, l4_extras


# ── Aggregation ────────────────────────────────────────────────────────────────
def aggregate_seeds(seed_results: list) -> list[dict]:
    """Aggregate N_SEEDS LayerResult lists into per-layer stats."""
    agg = []
    for li in range(N_LAYERS):
        tp = sum(r[0][li].tp for r in seed_results)
        fp = sum(r[0][li].fp for r in seed_results)
        tn = sum(r[0][li].tn for r in seed_results)
        fn = sum(r[0][li].fn for r in seed_results)
        s1c = sum(r[0][li].s1_correct for r in seed_results)
        s1t = sum(r[0][li].s1_total   for r in seed_results)

        # Per-reason aggregation
        reason_tp    = defaultdict(int)
        reason_total = defaultdict(int)
        reason_fn    = defaultdict(int)
        for r in seed_results:
            lr = r[0][li]
            for rn, cnt in lr.tp_by_reason.items():
                reason_tp[rn]    += cnt
            for rn, cnt in lr.fn_by_reason.items():
                reason_fn[rn]    += cnt
            for rn, cnt in lr.total_by_reason.items():
                reason_total[rn] += cnt

        ref_total    = tp + fn
        nonref_total = fp + tn
        dr    = tp / max(ref_total,    1)
        fpr   = fp / max(nonref_total, 1)
        prec  = tp / max(tp + fp,      1)
        load  = (tp + fp) / max(ref_total + nonref_total, 1) * 100

        agg.append({
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "DR":        round(dr,   4),
            "FPR":       round(fpr,  4),
            "precision": round(prec, 4),
            "load_p100": round(load, 2),
            "s1_accuracy": round(s1c / max(s1t, 1), 4),
            "s1_correct_raw": s1c,
            "s1_total_raw":   s1t,
            "reason_tp":    dict(reason_tp),
            "reason_total": dict(reason_total),
            "reason_fn":    dict(reason_fn),
            "per_reason_dr": {
                r: round(reason_tp.get(r, 0) / max(reason_total.get(r, 1), 1), 4)
                for r in sorted(REFERRAL_REASONS.keys())
            },
        })
    return agg


def micro_average(persona_agg_list: list) -> list[dict]:
    """Micro-average per-layer stats across all personas."""
    agg = []
    for li in range(N_LAYERS):
        tp = sum(d[li]["tp"] for _, d in persona_agg_list)
        fp = sum(d[li]["fp"] for _, d in persona_agg_list)
        tn = sum(d[li]["tn"] for _, d in persona_agg_list)
        fn = sum(d[li]["fn"] for _, d in persona_agg_list)
        s1c = sum(d[li]["s1_correct_raw"] for _, d in persona_agg_list)
        s1t = sum(d[li]["s1_total_raw"]   for _, d in persona_agg_list)

        reason_tp    = defaultdict(int)
        reason_total = defaultdict(int)
        reason_fn    = defaultdict(int)
        for _, d in persona_agg_list:
            for r, v in d[li]["reason_tp"].items():
                reason_tp[r]    += v
            for r, v in d[li]["reason_total"].items():
                reason_total[r] += v
            for r, v in d[li]["reason_fn"].items():
                reason_fn[r]    += v

        ref_total    = tp + fn
        nonref_total = fp + tn
        dr   = tp / max(ref_total,    1)
        fpr  = fp / max(nonref_total, 1)
        prec = tp / max(tp + fp,      1)
        load = (tp + fp) / max(ref_total + nonref_total, 1) * 100

        agg.append({
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "DR":        round(dr,   4),
            "FPR":       round(fpr,  4),
            "precision": round(prec, 4),
            "load_p100": round(load, 2),
            "s1_accuracy": round(s1c / max(s1t, 1), 4),
            "s1_correct_raw": s1c,
            "s1_total_raw":   s1t,
            "reason_tp":    dict(reason_tp),
            "reason_total": dict(reason_total),
            "reason_fn":    dict(reason_fn),
            "per_reason_dr": {
                r: round(reason_tp.get(r, 0) / max(reason_total.get(r, 1), 1), 4)
                for r in sorted(REFERRAL_REASONS.keys())
            },
        })
    return agg


# ── Economic analysis ──────────────────────────────────────────────────────────
def compute_economics(
    persona_seed_results: list,   # list of (persona, list_of_seed_results)
) -> list[dict]:
    """Compute economic metrics per 100 alerts, per layer."""
    economics = []
    for li in range(N_LAYERS):
        total_tp = total_fp = total_tn = total_fn = 0
        fn_cost_raw = 0.0

        for persona, seed_results in persona_seed_results:
            for layer_results, _ in seed_results:
                lr = layer_results[li]
                total_tp += lr.tp
                total_fp += lr.fp
                total_tn += lr.tn
                total_fn += lr.fn
                for reason, cnt in lr.fn_by_reason.items():
                    fn_cost_raw += cnt * MISS_COSTS.get(reason, 30)

        total_alerts = total_tp + total_fp + total_tn + total_fn
        scale = 100.0 / max(total_alerts, 1)

        auto_save = total_tn * AUTO_APPROVE_SAVE * scale
        fp_cost   = total_fp * FALSE_REF_COST    * scale
        fn_cost   = fn_cost_raw                  * scale

        economics.append({
            "auto_approve_save": round(auto_save, 1),
            "fp_cost":           round(fp_cost,   1),
            "fn_cost":           round(fn_cost,   1),
            "net_minutes":       round(auto_save - fp_cost - fn_cost, 1),
        })
    return economics


# ── Layer 4 aggregate extras ───────────────────────────────────────────────────
def aggregate_l4_extras(persona_seed_results: list) -> dict:
    """Aggregate L4 extras across all seeds and personas."""
    fields = ["n_train_pos", "n_train_neg", "l4_rule_tp", "l4_model_tp",
              "l4_model_fp", "r8_dr", "r10_dr", "early_dr", "late_dr"]
    sums   = {k: 0.0 for k in fields}
    cnt    = 0
    all_importances = defaultdict(float)
    n_imp = 0

    for persona, seed_results in persona_seed_results:
        for _, l4_extras in seed_results:
            cnt += 1
            for k in fields:
                sums[k] += l4_extras.get(k, 0.0)
            if l4_extras.get("importances"):
                for feat, val in l4_extras["importances"].items():
                    all_importances[feat] += val
                n_imp += 1

    mean_importances = {k: round(v / max(n_imp, 1), 5) for k, v in all_importances.items()}
    # Sort by importance
    sorted_imp = sorted(mean_importances.items(), key=lambda x: -x[1])

    return {
        "mean_n_train_pos":  round(sums["n_train_pos"]  / max(cnt, 1), 1),
        "mean_n_train_neg":  round(sums["n_train_neg"]  / max(cnt, 1), 1),
        "mean_l4_rule_tp":   round(sums["l4_rule_tp"]   / max(cnt, 1), 1),
        "mean_l4_model_tp":  round(sums["l4_model_tp"]  / max(cnt, 1), 1),
        "mean_l4_model_fp":  round(sums["l4_model_fp"]  / max(cnt, 1), 1),
        "mean_r8_dr":        round(sums["r8_dr"]   / max(cnt, 1), 4),
        "mean_r10_dr":       round(sums["r10_dr"]  / max(cnt, 1), 4),
        "mean_early_dr":     round(sums["early_dr"] / max(cnt, 1), 4),
        "mean_late_dr":      round(sums["late_dr"]  / max(cnt, 1), 4),
        "sorted_importances": sorted_imp,
        "all_importances":    mean_importances,
    }


# ── Printing ───────────────────────────────────────────────────────────────────
def print_table1(micro_agg: list):
    print()
    print("=" * 95)
    print("Table 1 — Layered Architecture Head-to-Head (micro-average, 5 personas × 15 seeds)")
    print("=" * 95)
    l0_s1 = micro_agg[0]["s1_accuracy"]
    print(f"\n  {'Layer':<20} {'DR':>8} {'FPR':>8} {'Precision':>10} {'Load/100':>10} {'S1 Acc':>8} {'S1 Impact':>10}")
    print("  " + "-" * 80)
    for li, (agg, name) in enumerate(zip(micro_agg, LAYER_NAMES)):
        s1_impact = agg["s1_accuracy"] - l0_s1
        print(
            f"  {name:<20}"
            f" {agg['DR']:>8.1%}"
            f" {agg['FPR']:>8.1%}"
            f" {agg['precision']:>10.1%}"
            f" {agg['load_p100']:>10.1f}"
            f" {agg['s1_accuracy']:>8.1%}"
            f" {s1_impact:>+10.2%}"
        )
    print()

    # S1 contamination gate
    max_impact = max(abs(agg["s1_accuracy"] - l0_s1) for agg in micro_agg)
    gate = "PASS" if max_impact < 0.001 else "FAIL"
    print(f"  S1 contamination gate: max impact={max_impact:.3%}  → {gate}")
    print()


def print_table2(micro_agg: list):
    print("=" * 85)
    print("Table 2 — Marginal Value Analysis")
    print("=" * 85)
    transitions = [
        ("0→1: None→Conf",    0, 1),
        ("0→2: None→Rules",   0, 2),
        ("1→2: Conf→Rules",   1, 2),
        ("2→3: Rules→+Conf",  2, 3),
        ("2→4: Rules→+Learn", 2, 4),
    ]
    print(f"\n  {'Transition':<22} {'Marginal DR':>12} {'Marginal FPR':>13} {'To Precision':>13} {'Worth It?':>14}")
    print("  " + "-" * 78)
    for name, from_li, to_li in transitions:
        a_from = micro_agg[from_li]
        a_to   = micro_agg[to_li]
        dDR    = a_to["DR"]  - a_from["DR"]
        dFPR   = a_to["FPR"] - a_from["FPR"]
        prec   = a_to["precision"]
        # Worth-it heuristic
        if dDR > 0 and dFPR <= 0:
            worth = "YES (dominates)"
        elif dDR > 0.08 and dFPR < 0.05:
            worth = "YES"
        elif dDR > 0.10 and dFPR < 0.15:
            worth = "LIKELY"
        elif dDR < 0.02:
            worth = "NO (tiny DR)"
        else:
            worth = "TRADEOFF"
        print(
            f"  {name:<22}"
            f" {dDR:>+12.1%}"
            f" {dFPR:>+13.1%}"
            f" {prec:>13.1%}"
            f" {worth:>14}"
        )
    print()


def print_table3(micro_agg: list):
    print("=" * 110)
    print("Table 3 — Per-Reason Detection Rate by Layer")
    print("=" * 110)
    print(f"\n  {'Reason':<5} {'Type':<10}", end="")
    for name in LAYER_NAMES:
        print(f" {name:>14}", end="")
    print()
    print("  " + "-" * (15 + 14 * N_LAYERS))
    for reason in sorted(REFERRAL_REASONS.keys()):
        rtype = REASON_TYPES[reason]
        print(f"  {reason:<5} {rtype:<10}", end="")
        for li in range(N_LAYERS):
            dr = micro_agg[li]["per_reason_dr"].get(reason, 0.0)
            print(f" {dr:>14.1%}", end="")
        print()
    print()


def print_table4(persona_agg_list: list):
    print("=" * 100)
    print("Table 4 — Per-Persona Layer DR")
    print("=" * 100)
    print(f"\n  {'Persona':<22}", end="")
    for name in LAYER_NAMES:
        print(f" {name:>12}", end="")
    print(f"  {'Best':>6}")
    print("  " + "-" * (22 + 12 * N_LAYERS + 8))
    for p_name, agg_layers in persona_agg_list:
        best_li = max(range(N_LAYERS), key=lambda i: agg_layers[i]["DR"])
        print(f"  {p_name:<22}", end="")
        for li, agg in enumerate(agg_layers):
            marker = "*" if li == best_li else " "
            print(f" {agg['DR']:>11.1%}{marker}", end="")
        print(f"  L{best_li}")
    print()


def print_table5(economics: list):
    print("=" * 90)
    print("Table 5 — Economic Analysis per 100 Alerts (minutes)")
    print("=" * 90)
    print(f"\n  {'Layer':<20} {'AA Saved':>10} {'FP Cost':>10} {'FN Cost':>10} {'Net':>10}")
    print("  " + "-" * 64)
    for name, econ in zip(LAYER_NAMES, economics):
        print(
            f"  {name:<20}"
            f" {econ['auto_approve_save']:>10.1f}"
            f" {-econ['fp_cost']:>10.1f}"
            f" {-econ['fn_cost']:>10.1f}"
            f" {econ['net_minutes']:>10.1f}"
        )
    print()


def print_l4_analysis(l4_agg: dict):
    print("=" * 85)
    print("Layer 4 Deep Analysis")
    print("=" * 85)
    print()
    print(f"  Training (per seed):  pos={l4_agg['mean_n_train_pos']:.0f}  neg={l4_agg['mean_n_train_neg']:.0f}  "
          f"ratio={l4_agg['mean_n_train_neg']/max(l4_agg['mean_n_train_pos'], 1):.0f}:1")
    print(f"  Model contributions:  rule_tp={l4_agg['mean_l4_rule_tp']:.0f}  "
          f"model_tp={l4_agg['mean_l4_model_tp']:.0f}  "
          f"model_fp={l4_agg['mean_l4_model_fp']:.0f}")
    print(f"  R8 detection (emergent): {l4_agg['mean_r8_dr']:.1%}")
    print(f"  R10 detection (emergent): {l4_agg['mean_r10_dr']:.1%}")
    print(f"  Time-to-learn: early DR={l4_agg['mean_early_dr']:.1%}  late DR={l4_agg['mean_late_dr']:.1%}  "
          f"delta={l4_agg['mean_late_dr'] - l4_agg['mean_early_dr']:+.1%}")
    print()
    print("  Top 5 features by |coefficient|:")
    for i, (feat, val) in enumerate(l4_agg["sorted_importances"][:5]):
        print(f"    {i+1}. {feat:<35} {val:.4f}")
    print()
    top5_names = {feat for feat, _ in l4_agg["sorted_importances"][:5]}
    context_feats = {"entity_override_rate", "cat_override_rate", "entity_risk_tier",
                     "alert_velocity", "alert_novelty", "active_policy_flags",
                     "business_context_flag", "time_of_day", "day_of_week"}
    n_ctx = sum(1 for f in top5_names if f in context_feats)
    print(f"  Context features in top 5: {n_ctx}/5 "
          f"({'confirms architecture insight' if n_ctx >= 3 else 'factor features dominate — learning from geometry, not context'})")
    print()


def print_ship_decision(micro_agg: list, l4_agg: dict):
    print("=" * 85)
    print("SHIP DECISION")
    print("=" * 85)
    print()

    l0, l1, l2, l3, l4 = micro_agg

    # Gate evaluations
    l2_dr_ok   = l2["DR"]        > 0.60
    l2_fpr_ok  = l2["FPR"]       < 0.15
    l2_prec_ok = l2["precision"]  > 0.50
    l2_ships   = l2_dr_ok and l2_fpr_ok and l2_prec_ok

    # Layer 3 over Layer 2: marginal DR > 10pp AND precision >= 30%
    l3_dDR = l3["DR"] - l2["DR"]
    l3_over_l2 = (l3_dDR > 0.10) and (l3["precision"] >= 0.30)

    # Layer 4 gates
    l4_dDR   = l4["DR"]  - l2["DR"]
    l4_dFPR  = l4["FPR"] - l2["FPR"]
    emerg_dr  = max(l4_agg["mean_r8_dr"], l4_agg["mean_r10_dr"])   # at least one emergent type
    ttl_delta = l4_agg["mean_late_dr"] - l4_agg["mean_early_dr"]
    ctx_count = sum(1 for f, _ in l4_agg["sorted_importances"][:5]
                    if f in {"entity_override_rate", "cat_override_rate", "entity_risk_tier",
                              "alert_velocity", "alert_novelty", "active_policy_flags",
                              "business_context_flag"})

    l4_marginal_ok = (l4_dDR > 0.08) and (l4_dFPR < 0.05)
    l4_emergent_ok = emerg_dr >= 0.30
    l4_learn_ok    = ttl_delta > 0.05
    l4_ctx_ok      = ctx_count >= 3
    l4_over_l2     = l4_marginal_ok and l4_emergent_ok and l4_learn_ok and l4_ctx_ok

    # Print gate results
    print(f"  Layer 2 gates:")
    print(f"    DR={l2['DR']:.1%} > 60%?         {'PASS' if l2_dr_ok   else 'FAIL'}")
    print(f"    FPR={l2['FPR']:.1%} < 15%?        {'PASS' if l2_fpr_ok  else 'FAIL'}")
    print(f"    Precision={l2['precision']:.1%} > 50%? {'PASS' if l2_prec_ok else 'FAIL'}")
    print(f"  → Layer 2 ships: {'YES' if l2_ships else 'NO'}")
    print()

    print(f"  Layer 3 over Layer 2:")
    print(f"    Marginal DR={l3_dDR:+.1%} > 10pp?    {'PASS' if l3_dDR > 0.10 else 'FAIL'}")
    print(f"    Precision={l3['precision']:.1%} >= 30%?   {'PASS' if l3['precision'] >= 0.30 else 'FAIL'}")
    print(f"  → Layer 3 preferred: {'YES' if l3_over_l2 else 'NO'}")
    print()

    print(f"  Layer 4 over Layer 2:")
    print(f"    Marginal DR={l4_dDR:+.1%} > 8pp?     {'PASS' if l4_dDR > 0.08 else 'FAIL'}")
    print(f"    Marginal FPR={l4_dFPR:+.1%} < 5pp?   {'PASS' if l4_dFPR < 0.05 else 'FAIL'}")
    print(f"    Emergent DR={emerg_dr:.1%} >= 30%?    {'PASS' if l4_emergent_ok else 'FAIL'}")
    print(f"    Time-to-learn delta={ttl_delta:+.1%} > 5pp? {'PASS' if l4_learn_ok else 'FAIL'}")
    print(f"    Context features in top 5={ctx_count}/5 >= 3? {'PASS' if l4_ctx_ok else 'FAIL'}")
    print(f"  → Layer 4 ships: {'YES' if l4_over_l2 else 'NO'}")
    print()

    # Decision
    print("  ─" * 42)
    if l2_ships and l4_over_l2:
        print("  v6.0 SHIPS: Layer 4 (Rules R1-R7 + Override Detector).")
        print(f"  Reason: DR={l4['DR']:.1%} FPR={l4['FPR']:.1%} Precision={l4['precision']:.1%}.")
        print("  Ship ReferralRules as SOC module (deterministic, Day 1).")
        print("  Ship OverrideDetector as GAE module (trains from analyst overrides).")
        print("  The referral system COMPOUNDS: rules handle known patterns (65%),")
        print("  override learning discovers emergent patterns (21%).")
    elif l2_ships and l3_over_l2 and not l4_over_l2:
        print("  v6.0 SHIPS: Layer 3 (Rules R1-R7 + Confidence Gate).")
        print(f"  Reason: marginal DR from rules={l3_dDR:+.1%} (> 10pp threshold).")
        print("  Layer 4 (override learning) deferred to v7.0: marginal value below threshold.")
    elif l2_ships:
        print("  v6.0 SHIPS: Layer 2 (Rules R1-R7 only).")
        print(f"  Reason: DR={l2['DR']:.1%}  FPR={l2['FPR']:.1%}  Precision={l2['precision']:.1%}.")
        print("  Override learning (Layer 4) revisited at v7.0 with production override data.")
        print("  The 20.7% emergent fraction is accepted as residual at v6.0.")
    else:
        # Nothing passes
        if l1["DR"] > l2["DR"]:
            print("  v6.0: Confidence gate only. Rules don't improve net value.")
            print("  Investigate: is 15% FPR threshold too tight for the analyst load?")
        else:
            print("  No layer exceeds gates. Review gate thresholds and referral distribution.")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("=" * 95)
    print("=== EXP-REFER-LAYERED: Layered referral architecture head-to-head ===")
    print("=" * 95)
    print()
    print(f"  Layers: {N_LAYERS}  Personas: {len(PERSONAS)}  Seeds: {N_SEEDS}")
    print(f"  Total runs: {N_LAYERS} × {len(PERSONAS)} × {N_SEEDS} = {N_LAYERS * len(PERSONAS) * N_SEEDS}")
    print(f"  Protocol: {N_WARMUP} warmup + {N_EVAL} eval decisions per seed")
    print(f"  Should-refer: {TOTAL_REFER_RATE:.0%} | 15% referral rate")
    print()

    config = load_domain_config("soc_product_v50")

    persona_agg_list   = []   # (persona_name, list_of_5_layer_agg_dicts)
    persona_seed_data  = []   # (persona, seed_results) for economics

    total_cells = len(PERSONAS) * N_SEEDS
    cell_n = 0

    for pi, persona in enumerate(PERSONAS):
        print(f"  ── {persona['name']}  σ={persona['noise'].mean():.3f}  "
              f"kernel={persona['kernel']}  q̄={persona['q_bar']}  V={persona['apd']}")
        seed_results = []
        for si in range(N_SEEDS):
            t_s = time.time()
            layer_results, l4_extras = run_seed(config, persona, seed=42 + pi * 100 + si)
            seed_results.append((layer_results, l4_extras))
            cell_n += 1
            if si == 0 or (si + 1) % 5 == 0:
                drs = [f"L{li}={lr.tp/(lr.tp+lr.fn+0.001):.0%}" for li, lr in enumerate(layer_results)]
                print(f"    seed {si+1:2d}/{N_SEEDS}  [{cell_n:3d}/{total_cells}]  "
                      f"{' '.join(drs)}  ({time.time()-t_s:.1f}s)")

        agg_layers = aggregate_seeds(seed_results)
        persona_agg_list.append((persona["name"], agg_layers))
        persona_seed_data.append((persona, seed_results))

        # Quick per-persona summary
        drs = [f"L{li}={agg['DR']:.0%}" for li, agg in enumerate(agg_layers)]
        print(f"    Summary: {' '.join(drs)}")
        print()

    elapsed = time.time() - t0
    print(f"  Completed {N_LAYERS * len(PERSONAS) * N_SEEDS} runs in {elapsed:.1f}s")

    # Aggregate across personas
    micro_agg = micro_average(persona_agg_list)
    economics  = compute_economics(persona_seed_data)
    l4_agg     = aggregate_l4_extras(persona_seed_data)

    # Print tables
    print_table1(micro_agg)
    print_table2(micro_agg)
    print_table3(micro_agg)
    print_table4(persona_agg_list)
    print_table5(economics)
    print_l4_analysis(l4_agg)
    print_ship_decision(micro_agg, l4_agg)

    # ── Save results ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _ser(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [_ser(v) for v in obj]
        if isinstance(obj, tuple):          return [_ser(v) for v in obj]
        return obj

    results = {
        "experiment":    "EXP-REFER-LAYERED",
        "n_runs":        N_LAYERS * len(PERSONAS) * N_SEEDS,
        "n_warmup":      N_WARMUP,
        "n_eval":        N_EVAL,
        "n_seeds":       N_SEEDS,
        "layer_names":   LAYER_NAMES,
        "referral_reasons": REFERRAL_REASONS,
        "micro_average": _ser(micro_agg),
        "per_persona":   _ser([{"persona": pn, "layers": agg} for pn, agg in persona_agg_list]),
        "economics":     _ser(economics),
        "l4_analysis":   _ser(l4_agg),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results → {results_path}")

    feat_imp_path = OUTPUT_DIR / "feature_importances.json"
    with open(feat_imp_path, "w", encoding="utf-8") as f:
        json.dump(_ser(l4_agg["sorted_importances"]), f, indent=2)
    print(f"  Saved feature importances → {feat_imp_path}")


if __name__ == "__main__":
    main()
