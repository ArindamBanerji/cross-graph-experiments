"""
EXP-REFER-COVERAGE: Referral taxonomy coverage analysis.

Answers: "What fraction of the referral problem is rule-expressible vs
emergent vs unlearnable?"

5 personas × 1000 alerts each = 5000 alerts total.
Measures detection rate and FPR at each coverage layer:
  (a) Confidence gate (Stage 1 conf < 0.70)
  (b) Rules R1-R6: deterministic rules on contextual metadata
  (c) Rules R1-R7: + cross-category
  (d) Rules R1-R9: + seasonal
  (e) Theoretical max R1-R10: + emergent (R8, R10) — knowable upper bound

Taxonomy:
  R1  RULE      identity_tier in ['executive','board','c_suite']
  R2  RULE      sequence_count >= 3
  R3  RULE      category == 'insider_threat' AND compliance_mode
  R4  RULE      category == 'data_exfiltration' AND asset_criticality > 0.85
                AND stage1_action in ['monitor','suppress']
  R5  RULE      incident_active == True
  R6  RULE      asset_age_days < 30
  R7  CONTEXT   cross_category_count >= 2
  R8  EMERGENT  analyst gut feel (no structural reason)
  R9  CONTEXT   business_cycle == 'quarter_end'
  R10 EMERGENT  prior false negative (historical FN log needed)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel

# ── Constants ─────────────────────────────────────────────────────────────────
N_ALERTS_PER_PERSONA = 1000
CONF_GATE_THRESHOLD  = 0.70    # Stage 1 confidence below this → flag as uncertain

OUTPUT_DIR = Path(__file__).parent / "results"

# Referral reason rates (as fraction of ALL alerts)
REFERRAL_REASONS = {
    "R1":  0.020,   # executive
    "R2":  0.020,   # rapid succession
    "R3":  0.015,   # compliance
    "R4":  0.015,   # high-value data
    "R5":  0.010,   # active IR
    "R6":  0.015,   # new asset
    "R7":  0.015,   # cross-category
    "R8":  0.020,   # gut feel (emergent)
    "R9":  0.005,   # seasonal
    "R10": 0.010,   # prior false negative (emergent)
}
# Total should-refer = 0.15 (15%)
TOTAL_REFER_RATE = sum(REFERRAL_REASONS.values())   # 0.15

REASON_TYPES = {
    "R1": "RULE",
    "R2": "RULE",
    "R3": "RULE",
    "R4": "RULE",
    "R5": "RULE",
    "R6": "RULE",
    "R7": "CONTEXT",
    "R8": "EMERGENT",
    "R9": "CONTEXT",
    "R10": "EMERGENT",
}

# ── Persona definitions ────────────────────────────────────────────────────────
_D = 6

def _hetero_noise(sigma_mean: float, ratios: list, d: int) -> np.ndarray:
    r = np.array(ratios[:d], dtype=float)
    raw = sigma_mean * r * (sigma_mean / (sigma_mean * r.mean()))
    return np.clip(raw, 0.03, 0.40)

PERSONAS = [
    {
        "id":    "P1",
        "name":  "FinServ SOC",
        "noise": np.full(_D, 0.08),
        "q_bar": 0.82,
        "apd":   200,
        "kernel": "l2",
    },
    {
        "id":    "P2",
        "name":  "Healthcare SOC",
        "noise": _hetero_noise(0.22, [0.8, 1.0, 0.9, 2.0, 1.1, 0.8], _D),
        "q_bar": 0.65,
        "apd":   150,
        "kernel": "diagonal",
    },
    {
        "id":    "P3",
        "name":  "Technology SOC",
        "noise": _hetero_noise(0.12, [0.8, 1.0, 0.9, 1.2, 1.0, 0.9], _D),
        "q_bar": 0.78,
        "apd":   300,
        "kernel": "l2",
    },
    {
        "id":    "P4",
        "name":  "Startup SOC",
        "noise": _hetero_noise(0.18, [0.7, 1.2, 0.8, 2.1, 1.1, 0.7], _D),
        "q_bar": 0.70,
        "apd":   80,
        "kernel": "diagonal",
    },
    {
        "id":    "P5",
        "name":  "Enterprise SOC",
        "noise": np.full(_D, 0.10),
        "q_bar": 0.85,
        "apd":   400,
        "kernel": "l2",
    },
]


# ── Kernel builders ────────────────────────────────────────────────────────────
def build_kernel(kernel_type: str, noise: np.ndarray):
    if kernel_type == "l2":
        return L2Kernel()
    inv_var = 1.0 / np.maximum(noise ** 2, 1e-4)
    weights = inv_var / inv_var.max()
    return DiagonalKernel(weights)


# ── Alert generation with referral metadata ────────────────────────────────────
def generate_alerts(
    n: int,
    config: dict,
    persona: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate n alerts. Each alert is a dict with:
      - factors: np.ndarray (d,)
      - category_index: int
      - category_name: str
      - gt_action_index: int
      - should_refer: bool
      - referral_reason: str or None
      - metadata: dict (contextual fields for rules R1-R9)

    85% normal flow (no referral). 15% should-refer.
    Should-refer alerts are drawn from the SAME factor distribution as normal
    alerts — they look like auto-approvable alerts geometrically.
    The referral is about context, not factor geometry.
    """
    mu_true    = config["mu"]           # (C, A, d)
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape
    noise      = persona["noise"]

    # Build GT distribution array
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0 / A] * A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()

    # Category weights: uniform
    cat_w = np.ones(C) / C

    # Find insider_threat and data_exfiltration indices
    cat_idx = {c: i for i, c in enumerate(categories)}
    insider_idx = cat_idx.get("insider_threat", -1)
    exfil_idx   = cat_idx.get("data_exfiltration", -1)

    # Build cumulative reason breakpoints for referral sampling
    reason_names = list(REFERRAL_REASONS.keys())
    reason_rates = [REFERRAL_REASONS[r] for r in reason_names]
    # Normalised so they sum to TOTAL_REFER_RATE
    cum_rates = np.cumsum(reason_rates)   # last element = TOTAL_REFER_RATE

    alerts = []

    for _ in range(n):
        ci      = int(rng.choice(C, p=cat_w))
        a_gt    = int(rng.choice(A, p=gt_arr[ci]))
        f = np.clip(
            mu_true[ci, a_gt] + rng.standard_normal(d) * noise,
            0.0, 1.0
        )

        # Decide should_refer
        u = rng.random()
        should_refer   = u < TOTAL_REFER_RATE
        referral_reason = None

        if should_refer:
            # Pick reason proportionally
            idx = int(np.searchsorted(cum_rates, u))
            idx = min(idx, len(reason_names) - 1)
            referral_reason = reason_names[idx]

            # Override category for R3/R4 to ensure rule conditions are met
            if referral_reason == "R3" and insider_idx >= 0:
                ci   = insider_idx
                a_gt = int(rng.choice(A, p=gt_arr[ci]))
                f    = np.clip(
                    mu_true[ci, a_gt] + rng.standard_normal(d) * noise,
                    0.0, 1.0
                )
            elif referral_reason == "R4" and exfil_idx >= 0:
                ci   = exfil_idx
                a_gt = int(rng.choice(A, p=gt_arr[ci]))
                f    = np.clip(
                    mu_true[ci, a_gt] + rng.standard_normal(d) * noise,
                    0.0, 1.0
                )

        # Generate contextual metadata
        is_r1  = referral_reason == "R1"
        is_r2  = referral_reason == "R2"
        is_r3  = referral_reason == "R3"
        is_r5  = referral_reason == "R5"
        is_r6  = referral_reason == "R6"
        is_r7  = referral_reason == "R7"
        is_r9  = referral_reason == "R9"

        identity_tier = (
            "executive" if is_r1
            else rng.choice(
                ["standard", "standard", "standard", "service_account", "executive"],
                p=[0.60, 0.0, 0.0, 0.20, 0.20]  # 20% executive in non-R1 (baseline noise)
            )
        )
        # Simpler: standard tier for non-R1
        if not is_r1:
            identity_tier = rng.choice(
                ["standard", "service_account", "executive"],
                p=[0.75, 0.20, 0.05]
            )

        sequence_count    = int(rng.integers(3, 8)) if is_r2 else int(rng.integers(0, 3))
        compliance_mode   = True if is_r3 else (rng.random() < 0.10)
        incident_active   = True if is_r5 else (rng.random() < 0.05)
        asset_age_days    = int(rng.integers(1, 30)) if is_r6 else int(rng.integers(30, 366))
        cross_cat_count   = int(rng.integers(2, 5)) if is_r7 else int(rng.integers(0, 2))
        business_cycle    = "quarter_end" if is_r9 else rng.choice(
            ["normal", "normal", "normal", "quarter_end"],
            p=[0.70, 0.0, 0.0, 0.30]
        )
        if not is_r9:
            business_cycle = rng.choice(["normal", "quarter_end"], p=[0.90, 0.10])

        # Asset criticality: high for R4 alerts, random otherwise
        asset_criticality = float(rng.uniform(0.86, 1.0)) if referral_reason == "R4" \
                            else float(rng.uniform(0.0, 1.0))

        alerts.append({
            "factors":          f,
            "category_index":   ci,
            "category_name":    categories[ci],
            "gt_action_index":  a_gt,
            "should_refer":     should_refer,
            "referral_reason":  referral_reason,
            "metadata": {
                "identity_tier":       identity_tier,
                "sequence_count":      sequence_count,
                "compliance_mode":     bool(compliance_mode),
                "incident_active":     bool(incident_active),
                "asset_age_days":      asset_age_days,
                "cross_category_count": cross_cat_count,
                "business_cycle":      business_cycle,
                "asset_criticality":   asset_criticality,
            },
        })

    return alerts


# ── Rule evaluators ────────────────────────────────────────────────────────────
def check_rules(alert: dict, stage1_action: str) -> dict[str, bool]:
    """
    Returns dict of {rule_id: fired} for rules R1-R9.
    R8 and R10 are never auto-detected (require override history / FN log).
    """
    m  = alert["metadata"]
    cn = alert["category_name"]

    r1 = m["identity_tier"] in ("executive", "board", "c_suite")
    r2 = m["sequence_count"] >= 3
    r3 = (cn == "insider_threat") and m["compliance_mode"]
    r4 = (
        cn == "data_exfiltration"
        and m["asset_criticality"] > 0.85
        and stage1_action in ("monitor", "suppress")
    )
    r5 = m["incident_active"]
    r6 = m["asset_age_days"] < 30
    r7 = m["cross_category_count"] >= 2
    r9 = m["business_cycle"] == "quarter_end"

    return {
        "R1": r1, "R2": r2, "R3": r3, "R4": r4,
        "R5": r5, "R6": r6, "R7": r7,
        "R8": False,    # cannot detect without override history
        "R9": r9,
        "R10": False,   # cannot detect without FN log
    }


# ── Layer detection logic ──────────────────────────────────────────────────────
def compute_layers(
    alerts: list[dict],
    confidences: list[float],
    stage1_actions: list[str],
) -> dict:
    """
    For each alert, check each detection layer.

    Layers:
      CONF:   conf < CONF_GATE_THRESHOLD
      R16:    R1 OR R2 OR R3 OR R4 OR R5 OR R6
      R17:    R16 OR R7
      R19:    R17 OR R9
      MAX:    R19 OR R8 OR R10 (theoretical max — R8/R10 known from ground truth)

    Returns per-layer: tp, fp, fn, tn, DR, FPR
    """
    n = len(alerts)
    layers = {
        "CONF":      [],   # confidence gate only
        "R16_ONLY":  [],   # rules R1-R6 only (no conf gate) — measures rule precision
        "R16":       [],   # CONF OR R1-R6
        "R17":       [],   # CONF OR R1-R7
        "R19":       [],   # CONF OR R1-R9
        "MAX":       [],   # CONF OR R1-R10 (theoretical max)
    }

    per_reason_conf_hits = defaultdict(int)    # conf gate hits on referral alerts
    per_reason_rule_hits = defaultdict(int)    # R1-R6 rule hits on referral alerts
    per_reason_total     = defaultdict(int)    # total alerts per referral reason

    for i, alert in enumerate(alerts):
        conf   = confidences[i]
        a_name = stage1_actions[i]
        rules  = check_rules(alert, a_name)

        sr     = alert["should_refer"]
        reason = alert["referral_reason"]

        if sr and reason:
            per_reason_total[reason] += 1
            if conf < CONF_GATE_THRESHOLD:
                per_reason_conf_hits[reason] += 1
            if any(rules[r] for r in ("R1","R2","R3","R4","R5","R6")):
                per_reason_rule_hits[reason] += 1

        conf_flag  = conf < CONF_GATE_THRESHOLD
        r16_flag   = any(rules[r] for r in ("R1","R2","R3","R4","R5","R6"))
        r17_flag   = r16_flag or rules["R7"]
        r19_flag   = r17_flag or rules["R9"]
        # MAX: add R8 and R10 if they are the referral reason (ground-truth oracle)
        r_max_flag = r19_flag or (sr and reason in ("R8", "R10"))

        layers["CONF"].append(conf_flag)
        layers["R16_ONLY"].append(r16_flag)            # rules only, no conf gate
        layers["R16"].append(conf_flag or r16_flag)
        layers["R17"].append(conf_flag or r17_flag)
        layers["R19"].append(conf_flag or r19_flag)
        layers["MAX"].append(conf_flag or r_max_flag)

    def layer_stats(flags: list[bool]) -> dict:
        tp = fp = tn = fn = 0
        for i, flagged in enumerate(flags):
            sr = alerts[i]["should_refer"]
            if flagged and sr:     tp += 1
            elif flagged and not sr: fp += 1
            elif not flagged and not sr: tn += 1
            else:                  fn += 1
        total_refer    = tp + fn
        total_nonrefer = fp + tn
        dr  = tp / total_refer    if total_refer    > 0 else 0.0
        fpr = fp / total_nonrefer if total_nonrefer > 0 else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "DR":  round(dr,  4),
            "FPR": round(fpr, 4),
            "coverage_pct": round(dr * 100, 1),
        }

    return {
        "layer_stats": {name: layer_stats(flags) for name, flags in layers.items()},
        "per_reason_conf_hits":  dict(per_reason_conf_hits),
        "per_reason_rule_hits":  dict(per_reason_rule_hits),
        "per_reason_total":      dict(per_reason_total),
    }


# ── Per-persona simulation ─────────────────────────────────────────────────────
def run_persona(persona: dict, config: dict, seed: int = 42) -> dict:
    """
    Generate N_ALERTS_PER_PERSONA alerts, score with ProfileScorer,
    compute detection layers.
    """
    rng = np.random.default_rng(seed)

    mu_true  = config["mu"]
    actions  = config["actions"]
    C, A, d  = mu_true.shape
    noise    = persona["noise"]
    kernel   = build_kernel(persona["kernel"], noise)

    # Warm ProfileScorer (small random init offset)
    offset = rng.uniform(-0.15, 0.15, mu_true.shape)
    scorer = ProfileScorer(
        np.clip(mu_true + offset, 0.0, 1.0),
        actions,
        scoring_kernel=kernel,
        eta_override=0.01,
    )
    scorer.eta     = 0.05
    scorer.eta_neg = 0.05

    # Generate alerts
    alerts = generate_alerts(N_ALERTS_PER_PERSONA, config, persona, rng)

    # Score with Stage 1
    confidences    = []
    stage1_actions = []

    for alert in alerts:
        res = scorer.score(alert["factors"], alert["category_index"])
        confidences.append(float(res.confidence))
        stage1_actions.append(actions[res.action_index])

        # Lightweight update (verify 30% of non-referred alerts to keep scorer warm)
        if not alert["should_refer"] and rng.random() < 0.30:
            gt_a   = alert["gt_action_index"]
            pred_a = res.action_index
            if pred_a == gt_a:
                scorer.update(alert["factors"], alert["category_index"], pred_a, True)
            else:
                scorer.update(
                    alert["factors"], alert["category_index"],
                    pred_a, False, gt_action_index=gt_a
                )

    # Compute layer coverage
    layer_data = compute_layers(alerts, confidences, stage1_actions)

    # Mean Stage 1 confidence and accuracy on non-referred alerts
    nonref_conf = [confidences[i] for i in range(len(alerts)) if not alerts[i]["should_refer"]]
    correct = sum(
        1 for i, a in enumerate(alerts)
        if not a["should_refer"]
        and stage1_actions[i] == actions[a["gt_action_index"]]
    )
    nonref_total = sum(1 for a in alerts if not a["should_refer"])

    # Count actual reasons in generated set
    reason_counts: dict[str, int] = defaultdict(int)
    for a in alerts:
        if a["should_refer"] and a["referral_reason"]:
            reason_counts[a["referral_reason"]] += 1
    total_refer = sum(a["should_refer"] for a in alerts)

    return {
        "persona_id":         persona["id"],
        "persona_name":       persona["name"],
        "n_alerts":           N_ALERTS_PER_PERSONA,
        "total_refer":        total_refer,
        "total_nonrefer":     N_ALERTS_PER_PERSONA - total_refer,
        "refer_rate":         round(total_refer / N_ALERTS_PER_PERSONA, 4),
        "s1_accuracy_nonref": round(correct / max(nonref_total, 1), 4),
        "mean_conf_nonref":   round(float(np.mean(nonref_conf)) if nonref_conf else 0.0, 4),
        "reason_counts":      dict(reason_counts),
        "layer_stats":        layer_data["layer_stats"],
        "per_reason_conf":    layer_data["per_reason_conf_hits"],
        "per_reason_rules":   layer_data["per_reason_rule_hits"],
        "per_reason_total":   layer_data["per_reason_total"],
    }


# ── Aggregate across personas ──────────────────────────────────────────────────
def aggregate_results(persona_results: list[dict]) -> dict:
    """Aggregate per-layer DR and FPR across all 5 personas (micro-average)."""
    agg_layers: dict[str, dict] = {}
    layer_names = list(persona_results[0]["layer_stats"].keys())

    for layer in layer_names:
        total_tp = sum(r["layer_stats"][layer]["tp"] for r in persona_results)
        total_fp = sum(r["layer_stats"][layer]["fp"] for r in persona_results)
        total_fn = sum(r["layer_stats"][layer]["fn"] for r in persona_results)
        total_tn = sum(r["layer_stats"][layer]["tn"] for r in persona_results)
        total_refer    = total_tp + total_fn
        total_nonrefer = total_fp + total_tn
        dr  = total_tp / total_refer    if total_refer    > 0 else 0.0
        fpr = total_fp / total_nonrefer if total_nonrefer > 0 else 0.0
        agg_layers[layer] = {
            "tp": total_tp, "fp": total_fp,
            "fn": total_fn, "tn": total_tn,
            "DR":  round(dr,  4),
            "FPR": round(fpr, 4),
        }

    # Per-reason detection across all personas
    all_reason_total: dict[str, int] = defaultdict(int)
    all_reason_conf:  dict[str, int] = defaultdict(int)
    all_reason_rules: dict[str, int] = defaultdict(int)

    for r in persona_results:
        for reason, cnt in r["per_reason_total"].items():
            all_reason_total[reason] += cnt
        for reason, cnt in r["per_reason_conf"].items():
            all_reason_conf[reason] += cnt
        for reason, cnt in r["per_reason_rules"].items():
            all_reason_rules[reason] += cnt

    per_reason_dr: dict[str, dict] = {}
    all_reasons = sorted(REFERRAL_REASONS.keys())
    for reason in all_reasons:
        tot = all_reason_total.get(reason, 0)
        per_reason_dr[reason] = {
            "total":     tot,
            "conf_hits": all_reason_conf.get(reason, 0),
            "rule_hits": all_reason_rules.get(reason, 0),
            "conf_dr":   round(all_reason_conf.get(reason, 0)  / max(tot, 1), 4),
            "rule_dr":   round(all_reason_rules.get(reason, 0) / max(tot, 1), 4),
            "type":      REASON_TYPES[reason],
        }

    return {
        "agg_layers": agg_layers,
        "per_reason": per_reason_dr,
    }


# ── Problem decomposition ──────────────────────────────────────────────────────
def compute_decomposition(agg: dict) -> dict:
    """
    Categorise referral volume into:
      RULE       R1-R6  (pure deterministic)
      CONTEXT    R7, R9 (requires graph/calendar context)
      EMERGENT   R8, R10 (requires override history / FN log)
    """
    rule_pct     = sum(REFERRAL_REASONS[r] for r in ("R1","R2","R3","R4","R5","R6"))
    context_pct  = sum(REFERRAL_REASONS[r] for r in ("R7","R9"))
    emergent_pct = sum(REFERRAL_REASONS[r] for r in ("R8","R10"))

    total = rule_pct + context_pct + emergent_pct
    return {
        "rule_pct":     round(rule_pct     / total * 100, 1),
        "context_pct":  round(context_pct  / total * 100, 1),
        "emergent_pct": round(emergent_pct / total * 100, 1),
        "rule_abs":     round(rule_pct     * 100, 1),
        "context_abs":  round(context_pct  * 100, 1),
        "emergent_abs": round(emergent_pct * 100, 1),
    }


# ── Design recommendation ──────────────────────────────────────────────────────
def compute_recommendation(agg_layers: dict, decomp: dict) -> list[str]:
    r16_dr       = agg_layers["R16"]["DR"]
    r16_only_dr  = agg_layers["R16_ONLY"]["DR"]
    r16_only_fpr = agg_layers["R16_ONLY"]["FPR"]   # rules-only FPR (no conf gate)
    conf_fpr     = agg_layers["CONF"]["FPR"]
    emergent_pct = decomp["emergent_pct"]

    recs = []

    # Gate on rules-only FPR (conf gate FPR is a separate deployment concern)
    if r16_only_dr > 0.70 and r16_only_fpr < 0.02:
        recs.append(
            "RULES SUFFICIENT for v6.0. "
            f"R1-R6 cover {r16_only_dr:.0%} of referrals at "
            f"rules-only FPR={r16_only_fpr:.1%}. "
            "Override learning adds value at v7.0."
        )
    elif r16_only_dr < 0.50:
        recs.append(
            "RULES INSUFFICIENT. "
            f"R1-R6 cover only {r16_only_dr:.0%} of referrals "
            f"(rules-only FPR={r16_only_fpr:.1%}). "
            "Override learning needed at v6.0."
        )
    else:
        recs.append(
            f"RULES PARTIAL. R1-R6 cover {r16_only_dr:.0%} of referrals "
            f"(rules-only FPR={r16_only_fpr:.1%}). "
            "Context layer (R7) adds incremental coverage."
        )

    recs.append(
        f"NOTE: Confidence gate alone has FPR={conf_fpr:.1%} — "
        "it is noisy (high recall, low precision) and should NOT be the sole referral trigger. "
        "Rules are the precision instrument; conf gate is a safety net."
    )

    if emergent_pct > 30:
        recs.append(
            f"SIGNIFICANT EMERGENT FRACTION ({emergent_pct:.0f}% of referrals). "
            "Long-warmup learning required."
        )
    elif emergent_pct < 15:
        recs.append(
            f"SMALL EMERGENT FRACTION ({emergent_pct:.0f}% of referrals). "
            "Rules dominate. ML adds marginal value."
        )
    else:
        recs.append(
            f"MODERATE EMERGENT FRACTION ({emergent_pct:.0f}% of referrals). "
            "Override learning useful but not critical."
        )

    return recs


# ── Printing ───────────────────────────────────────────────────────────────────
def print_results(persona_results: list[dict], agg: dict, decomp: dict, recs: list[str]):

    # Table 1 — Layer Coverage
    print()
    print("=" * 90)
    print("Table 1 — Layer Coverage (micro-average across 5 personas)")
    print("=" * 90)
    print()
    LAYER_LABELS = {
        "CONF":     "Confidence gate only      ",
        "R16_ONLY": "Rules v6.0 ONLY (R1-R6)   ",
        "R16":      "CONF + Rules v6.0 (R1-R6)  ",
        "R17":      "CONF + Rules v6.0+ (R1-R7) ",
        "R19":      "CONF + Full context (R1-R9) ",
        "MAX":      "Theoretical max (R1-R10)   ",
    }
    REASONS_COVERED = {
        "CONF":     "uncertainty (conf<0.70) — noisy, high FPR",
        "R16_ONLY": "exec, succession, compliance, data-exfil, IR, new-asset",
        "R16":      "conf gate + R1-R6 rules",
        "R17":      "+ cross-category (R7)",
        "R19":      "+ seasonal (R9)",
        "MAX":      "+ emergent gut feel + prior FN",
    }
    print(f"  {'Layer':<31} {'Reasons Covered':<45} {'DR':>7} {'FPR':>7} {'Cov%':>7}")
    print("  " + "-" * 102)
    for layer in ("CONF", "R16_ONLY", "R16", "R17", "R19", "MAX"):
        s = agg["agg_layers"][layer]
        print(
            f"  {LAYER_LABELS[layer]:<31} {REASONS_COVERED[layer]:<45}"
            f" {s['DR']:>7.1%} {s['FPR']:>7.1%} {s['DR']*100:>6.1f}%"
        )
    print()

    # Per-persona DR breakdown for key layers
    print(f"  Per-persona layer DR:")
    print(f"  {'Persona':<22}  {'CONF':>7}  {'Rules':>7}  {'R16':>7}  {'R17':>7}  {'MAX':>7}  {'Refer%':>8}  {'S1 Acc':>8}")
    print(f"  {'':22}  {'only':>7}  {'only':>7}  {'combined':>7}")
    print("  " + "-" * 92)
    for r in persona_results:
        ls = r["layer_stats"]
        print(
            f"  {r['persona_name']:<22}"
            f"  {ls['CONF']['DR']:>7.1%}"
            f"  {ls['R16_ONLY']['DR']:>7.1%}"
            f"  {ls['R16']['DR']:>7.1%}"
            f"  {ls['R17']['DR']:>7.1%}"
            f"  {ls['MAX']['DR']:>7.1%}"
            f"  {r['refer_rate']:>8.1%}"
            f"  {r['s1_accuracy_nonref']:>8.1%}"
        )
    print()

    # Table 2 — Per-Reason Detection
    print("=" * 90)
    print("Table 2 — Per-Reason Detection Rate")
    print("=" * 90)
    print()
    print(f"  {'ID':<5} {'Reason':<28} {'Type':<10} {'Count':>6} {'Conf Gate':>10} {'Rules R1-6':>11} {'Learning?':>10}")
    print("  " + "-" * 85)

    REASON_NAMES = {
        "R1":  "Executive/VIP account",
        "R2":  "Rapid succession (3+ similar)",
        "R3":  "Compliance-mandated category",
        "R4":  "High-value data movement",
        "R5":  "Active incident response",
        "R6":  "New asset (<30 days)",
        "R7":  "Cross-category correlation",
        "R8":  "Analyst gut feel override",
        "R9":  "Seasonal/business-cycle",
        "R10": "Previous false negative",
    }
    LEARNING_NEEDED = {
        "R1": "NO",  "R2": "NO",  "R3": "NO",  "R4": "NO",
        "R5": "NO",  "R6": "NO",  "R7": "NO (graph query)",
        "R8": "YES", "R9": "NO (calendar)",
        "R10": "YES",
    }

    for reason in sorted(agg["per_reason"].keys()):
        d = agg["per_reason"][reason]
        rule_covered = "YES" if REASON_TYPES[reason] in ("RULE",) else (
            "YES (context)" if REASON_TYPES[reason] == "CONTEXT" else "NO"
        )
        print(
            f"  {reason:<5} {REASON_NAMES[reason]:<28} {d['type']:<10}"
            f" {d['total']:>6}"
            f" {d['conf_dr']:>10.1%}"
            f" {d['rule_dr']:>11.1%}"
            f" {LEARNING_NEEDED[reason]:>10}"
        )
    print()

    # Table 3 — Problem Decomposition
    print("=" * 90)
    print("Table 3 — Problem Decomposition")
    print("=" * 90)
    print()
    total_referrals = sum(r["total_refer"] for r in persona_results)
    print(f"  Total referral alerts simulated: {total_referrals} (target: {int(5000 * TOTAL_REFER_RATE)})")
    print()
    print(f"  {'Category':<25} {'% of Referrals':>16} {'Mechanism':<35}")
    print("  " + "-" * 78)
    print(f"  {'Rule-expressible (R1-R6)':<25} {decomp['rule_pct']:>14.1f}%  {'Policy rules':<35}")
    print(f"  {'Context-dependent (R7,R9)':<25} {decomp['context_pct']:>14.1f}%  {'Graph queries / calendar':<35}")
    print(f"  {'Emergent (R8,R10)':<25} {decomp['emergent_pct']:>14.1f}%  {'Override learning':<35}")
    print()

    # VERDICT
    print("=" * 90)
    print("VERDICT / DESIGN RECOMMENDATION")
    print("=" * 90)
    print()
    for rec in recs:
        print(f"  {rec}")
    print()

    conf     = agg["agg_layers"]["CONF"]
    r16_only = agg["agg_layers"]["R16_ONLY"]
    r16      = agg["agg_layers"]["R16"]
    r17      = agg["agg_layers"]["R17"]
    r19      = agg["agg_layers"]["R19"]
    print(f"  Coverage summary:")
    print(f"    Conf gate only:       DR={conf['DR']:.1%}  FPR={conf['FPR']:.1%}  (noisy — high FPR)")
    print(f"    Rules R1-R6 only:     DR={r16_only['DR']:.1%}  FPR={r16_only['FPR']:.1%}  (precise)")
    print(f"    Conf + Rules R1-R6:   DR={r16['DR']:.1%}  FPR={r16['FPR']:.1%}")
    print(f"    + Context R7:         DR={r17['DR']:.1%}  FPR={r17['FPR']:.1%}  "
          f"(delta DR={r17['DR']-r16['DR']:+.1%})")
    print(f"    + Seasonal R9:        DR={r19['DR']:.1%}  FPR={r19['FPR']:.1%}  "
          f"(delta DR={r19['DR']-r17['DR']:+.1%})")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("=" * 90)
    print("=== EXP-REFER-COVERAGE: Referral taxonomy coverage analysis ===")
    print("=" * 90)
    print()
    print(f"  Config: A=4  C=6  d=6  (soc_product_v50)")
    print(f"  Personas: {len(PERSONAS)}  Alerts per persona: {N_ALERTS_PER_PERSONA}")
    print(f"  Total alerts: {len(PERSONAS) * N_ALERTS_PER_PERSONA}")
    print(f"  Should-refer rate: {TOTAL_REFER_RATE:.0%}")
    print(f"  Confidence gate threshold: {CONF_GATE_THRESHOLD:.2f}")
    print()

    # Load config once
    config = load_domain_config("soc_product_v50")

    persona_results = []
    for i, persona in enumerate(PERSONAS):
        print(f"  [{i+1}/{len(PERSONAS)}] {persona['name']}  "
              f"σ_mean={persona['noise'].mean():.3f}  kernel={persona['kernel']}  "
              f"V={persona['apd']}",
              end=" ... ", flush=True)
        t_p = time.time()
        result = run_persona(persona, config, seed=42 + i * 7)
        elapsed = time.time() - t_p
        ls = result["layer_stats"]
        print(f"refer={result['refer_rate']:.1%}  "
              f"CONF={ls['CONF']['DR']:.0%}  R16={ls['R16']['DR']:.0%}  "
              f"MAX={ls['MAX']['DR']:.0%}  ({elapsed:.1f}s)")
        persona_results.append(result)

    print()
    print(f"  Completed {len(PERSONAS)} personas in {time.time()-t0:.1f}s")

    # Aggregate
    agg   = aggregate_results(persona_results)
    decomp = compute_decomposition(agg)
    recs  = compute_recommendation(agg["agg_layers"], decomp)

    # Print tables
    print_results(persona_results, agg, decomp, recs)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "coverage_analysis.json"

    # Serialise persona results (convert np arrays to lists)
    def _to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serialisable(v) for v in obj]
        return obj

    save_persona = []
    for r in persona_results:
        entry = {k: v for k, v in r.items()}
        save_persona.append(_to_serialisable(entry))

    output = {
        "experiment": "EXP-REFER-COVERAGE",
        "n_alerts_total": len(PERSONAS) * N_ALERTS_PER_PERSONA,
        "n_personas": len(PERSONAS),
        "conf_gate_threshold": CONF_GATE_THRESHOLD,
        "referral_reasons": REFERRAL_REASONS,
        "reason_types": REASON_TYPES,
        "total_refer_rate": TOTAL_REFER_RATE,
        "persona_results": save_persona,
        "aggregate": {
            "layer_stats": _to_serialisable(agg["agg_layers"]),
            "per_reason":  _to_serialisable(agg["per_reason"]),
        },
        "decomposition": decomp,
        "recommendation": recs,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
