"""
claim_generator.py — Simulated claims for synthesis experiments

SYNTH-EXP-0 deliverable. Lives in src/data/ of cross-graph-experiments.

Generates synthetic claims for testing synthesis pipeline.
"Correct" claims align with the alert generator's ground truth.
"Poisoned" claims recommend wrong actions for certain categories.

IMPORTANT: These are SYNTHETIC claims for experiment validation only.
Real claims come from ContextConnectors (F13) at v6.5+.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Rule templates (Layer 1 domain ontology choices)
# Sign convention: negative → action MORE likely, positive → LESS likely
# Matches Eq. 4-synthesis: P(a|f,c,σ) = softmax(−(‖f−μ‖² + λσ) / (τ·τ_mod))
# ---------------------------------------------------------------------------

SOC_SYNTHESIS_RULES: Dict[str, Dict[str, float]] = {
    # Active campaign targeting a category: escalate_incident/enrich more, auto_close less
    # Semantic map: escalate→escalate_incident, investigate→enrich_and_watch,
    #               suppress→auto_close, monitor→escalate_tier2
    "active_campaign": {
        "escalate_incident": -0.4,
        "enrich_and_watch":  -0.1,
        "auto_close":        +0.6,
        "escalate_tier2":    +0.1,
    },
    # CISO explicit directive (medium force, high authority)
    "ciso_risk_directive": {
        "escalate_incident": -0.3,
        "enrich_and_watch":  -0.1,
        "auto_close":        +0.4,
        "escalate_tier2":     0.0,
    },
    # CVE actively exploited in the wild
    "cve_actively_exploited": {
        "escalate_incident": -0.3,
        "enrich_and_watch":  -0.2,
        "auto_close":        +0.5,
        "escalate_tier2":     0.0,
    },
    # Vulnerability patched — alerts related to it can be auto-closed
    "vulnerability_patched": {
        "escalate_incident": +0.2,
        "enrich_and_watch":   0.0,
        "auto_close":        -0.1,
        "escalate_tier2":    -0.1,
    },
    # Known benign change (maintenance window, new employee batch, etc.)
    "known_change": {
        "escalate_incident": +0.1,
        "enrich_and_watch":   0.0,
        "auto_close":        -0.2,
        "escalate_tier2":    -0.1,
    },
    # Known false-positive pattern for this org
    "known_fp_pattern": {
        "escalate_incident": +0.3,
        "enrich_and_watch":  +0.1,
        "auto_close":        -0.4,
        "escalate_tier2":     0.0,
    },
}

# Categories and actions matching CategoryAlertGenerator / bridge_common config
CATEGORIES: List[str] = [
    "credential_access",
    "threat_intel_match",
    "lateral_movement",
    "data_exfiltration",
    "insider_threat",
]

ACTIONS: List[str] = ["auto_close", "escalate_tier2", "enrich_and_watch", "escalate_incident"]

# ---------------------------------------------------------------------------
# Which claim types make semantic sense for which categories
# (Used to generate "correct" claims that align with GT)
# ---------------------------------------------------------------------------
CORRECT_TYPE_CATEGORY_MAP: Dict[str, List[str]] = {
    "active_campaign":        ["credential_access", "threat_intel_match", "lateral_movement"],
    "ciso_risk_directive":   ["insider_threat", "data_exfiltration"],
    "cve_actively_exploited":["data_exfiltration", "credential_access"],
    "vulnerability_patched": ["data_exfiltration", "insider_threat"],
    "known_change":          ["lateral_movement", "data_exfiltration"],
    "known_fp_pattern":      ["lateral_movement", "insider_threat"],
}

# Poison map: wrong categories that create misleading bias
POISON_TYPE_CATEGORY_MAP: Dict[str, List[str]] = {
    "active_campaign":        ["lateral_movement", "insider_threat"],       # campaign pressure on lower-risk cats
    "vulnerability_patched":  ["threat_intel_match", "credential_access"],  # suppress on high-risk cats
    "known_fp_pattern":       ["insider_threat", "data_exfiltration"],      # suppress on real threats
}


def _make_claim(
    claim_type: str,
    categories: List[str],
    rng: random.Random,
    urgency_range: Tuple[float, float] = (0.0, 1.0),
    confidence_range: Tuple[float, float] = (0.7, 1.0),
    extraction_range: Tuple[float, float] = (0.85, 1.0),
    age_range: Tuple[float, float] = (0.0, 30.0),
    decay_classes: List[str] = None,
    poisoned: bool = False,
) -> Dict[str, Any]:
    """Build a single claim dict."""
    if decay_classes is None:
        decay_classes = ["campaign", "standard", "transient"]
    claim = {
        "type": claim_type,
        "categories_affected": categories,
        "confidence":           round(rng.uniform(*confidence_range), 3),
        "extraction_confidence":round(rng.uniform(*extraction_range), 3),
        "urgency":              round(rng.uniform(*urgency_range), 3),
        "age_days":             round(rng.uniform(*age_range), 1),
        "decay_class":          rng.choice(decay_classes),
    }
    if poisoned:
        claim["_poisoned"] = True   # Internal flag for experiment analysis
    return claim


def generate_correct_claims(n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate n claims that are CONSISTENT with the alert generation parameters.

    "Correct" means: if a claim says "active_campaign on credential_access",
    the alert generator DOES produce credential_access alerts where escalation
    is more appropriate than suppression. Claims align with ground truth.

    All claims have confidence ≥ 0.7 and extraction_confidence ≥ 0.85
    (above the default extraction_threshold of 0.8 effective confidence).
    """
    rng = random.Random(seed)
    claim_types = list(CORRECT_TYPE_CATEGORY_MAP.keys())
    claims = []

    for i in range(n):
        claim_type = claim_types[i % len(claim_types)]
        possible_cats = CORRECT_TYPE_CATEGORY_MAP[claim_type]
        n_cats = len(possible_cats)
        selected_cats = rng.sample(possible_cats, n_cats)
        claims.append(_make_claim(
            claim_type=claim_type,
            categories=selected_cats,
            rng=rng,
            urgency_range=(0.1, 0.9),
            confidence_range=(0.7, 1.0),
            extraction_range=(0.85, 1.0),
            age_range=(0.0, 25.0),
        ))
    return claims


def generate_poisoned_claims(
    n_correct: int,
    n_poison: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a mix of correct + poisoned claims.

    Poisoned claims: claim type says one thing but affects categories where
    the recommended action is WRONG.
    Example: "active_campaign" (pushes toward escalation) applied to travel_anomaly
    (which should be suppressed) → the claim actively misleads scoring.

    The correct claims come first, then poisoned. Both have realistic confidence
    values that pass the extraction threshold — they look legitimate.
    """
    correct = generate_correct_claims(n_correct, seed)
    rng = random.Random(seed + 999)

    poison_types = list(POISON_TYPE_CATEGORY_MAP.keys())
    poisoned = []
    for i in range(n_poison):
        claim_type = poison_types[i % len(poison_types)]
        wrong_cats = POISON_TYPE_CATEGORY_MAP[claim_type]
        n_cats = rng.randint(1, min(2, len(wrong_cats)))
        selected_cats = rng.sample(wrong_cats, n_cats)
        poisoned.append(_make_claim(
            claim_type=claim_type,
            categories=selected_cats,
            rng=rng,
            urgency_range=(0.3, 0.8),
            confidence_range=(0.7, 0.9),
            extraction_range=(0.85, 1.0),
            age_range=(0.0, 15.0),
            poisoned=True,
        ))
    return correct + poisoned


def generate_high_urgency_claims(
    n: int = 5,
    seed: int = 42,
    categories: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate high-urgency claims (urgency ≥ 0.8) to test τ_mod adjustment.
    Used in EXP-S1 to verify temperature sharpening under urgency.
    """
    if categories is None:
        categories = ["credential_access", "threat_intel_match"]
    rng = random.Random(seed + 7777)
    claims = []
    for _ in range(n):
        cat = rng.choice(categories)
        claims.append(_make_claim(
            claim_type=rng.choice(["active_campaign", "cve_actively_exploited"]),
            categories=[cat],
            rng=rng,
            urgency_range=(0.8, 1.0),
            confidence_range=(0.85, 1.0),
            extraction_range=(0.9, 1.0),
            age_range=(0.0, 3.0),
            decay_classes=["campaign"],
        ))
    return claims


if __name__ == "__main__":
    # Self-test: verify claim structure and σ projection
    from src.models.rule_projector import RuleBasedProjector
    from src.models.synthesis import SynthesisBias

    # Test 1: correct claims
    correct = generate_correct_claims(n=10, seed=42)
    assert len(correct) == 10
    assert all(set(c.keys()) >= {"type", "categories_affected", "confidence",
                                  "extraction_confidence", "urgency", "age_days",
                                  "decay_class"} for c in correct)
    print(f"Correct claims: {len(correct)} generated, all fields present")

    # Test 2: poisoned mix
    mixed = generate_poisoned_claims(n_correct=8, n_poison=2, seed=42)
    assert len(mixed) == 10
    poison_count = sum(1 for c in mixed if c.get("_poisoned"))
    assert poison_count == 2
    print(f"Poisoned mix: {len(mixed)} total, {poison_count} poisoned")

    # Test 3: projection
    projector = RuleBasedProjector(
        rules=SOC_SYNTHESIS_RULES,
        categories=CATEGORIES,
        actions=ACTIONS,
    )
    bias = projector.project(correct, lambda_coupling=0.1)
    assert isinstance(bias, SynthesisBias)
    assert bias.sigma.shape == (len(CATEGORIES), len(ACTIONS))
    assert bias.lambda_coupling == 0.1
    print(f"Projection OK: {bias.active_claims} claims active, τ_mod={bias.tau_modifier:.3f}")

    # Print σ tensor
    print("\nσ tensor from 10 correct claims (rows=categories, cols=actions):")
    header = f"{'':30s}" + "".join(f"{a:>12s}" for a in ACTIONS)
    print(header)
    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:30s}" + "".join(f"{bias.sigma[i,j]:>12.4f}" for j in range(len(ACTIONS)))
        print(row)

    print("\nAll claim_generator tests passed.")
