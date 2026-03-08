"""
rule_projector.py — RuleBasedProjector implementation of SynthesisProjector

SYNTH-EXP-0 deliverable. Lives in src/models/ of cross-graph-experiments.

Maps a list of claims to a SynthesisBias using domain-configured rule templates.
This is the v5.0 implementation. A learned projector (v6.5+) would calibrate
action_directions from outcome correlations rather than using hard-coded rules.

Eq. S2 (intelligence_layer_design_v1 §2.4):
    σ[c,a] = Σ_k direction_k[a] · confidence_k · decay(c_k)
    for all claims k affecting category c, clipped to [-σ_max, +σ_max]

Eq. S3:
    τ_mod = max(0.5, 1.0 − max_k(urgency_k · confidence_k))
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional

from src.models.synthesis import SynthesisBias


class RuleBasedProjector:
    """
    Maps claims → SynthesisBias using static rule templates.

    Rule templates specify how each claim type shifts action preferences:
        rule["active_campaign"]["escalate"] = -0.4
    means: an active_campaign claim makes escalation 0.4 units MORE likely
    (negative σ → subtract from distance → action more likely in Eq. 4-synthesis).

    The sign convention (negative = more likely) matches the scoring equation:
        P(a|f,c,σ) = softmax(−(‖f − μ[c,a,:]‖² + λ·σ[c,a]) / (τ · τ_mod))
    Adding a positive σ[c,a] increases the effective distance → action less likely.
    Adding a negative σ[c,a] decreases the effective distance → action more likely.
    """

    DEFAULT_DECAY_RATES: Dict[str, float] = {
        "campaign":   0.005,   # KEV campaigns persist weeks
        "standard":   0.001,   # Standard advisories last months
        "transient":  0.020,   # Slack/email context fades in days
    }

    def __init__(
        self,
        rules: Dict[str, Dict[str, float]],
        categories: List[str],
        actions: List[str],
        decay_rates: Optional[Dict[str, float]] = None,
        sigma_max: float = 1.0,
        extraction_threshold: float = 0.8,
    ):
        """
        Args:
            rules:                Mapping claim_type → {action_name: direction_float}
                                  direction ∈ [-0.5, +0.5] per rule entry.
            categories:           Ordered category names (must match ProfileScorer).
            actions:              Ordered action names (must match ProfileScorer).
            decay_rates:          Per decay_class daily decay rates (exponential).
            sigma_max:            Clipping bound for each σ[c,a] cell.
            extraction_threshold: Minimum effective confidence to include a claim.
                                  effective_conf = claim["confidence"] × claim["extraction_confidence"]
        """
        self.rules = rules
        self.categories = categories
        self.actions = actions
        self.cat_index: Dict[str, int] = {c: i for i, c in enumerate(categories)}
        self.act_index: Dict[str, int] = {a: j for j, a in enumerate(actions)}
        self.decay_rates = decay_rates or self.DEFAULT_DECAY_RATES
        self.sigma_max = sigma_max
        self.extraction_threshold = extraction_threshold

    def project(
        self,
        claims: List[Dict[str, Any]],
        lambda_coupling: float = 0.1,
    ) -> SynthesisBias:
        """
        Map claims to SynthesisBias.

        For each claim that passes the extraction threshold:
          1. Look up the rule template by claim["type"]
          2. Compute time decay: exp(-rate × age_days)
          3. weight = confidence × decay
          4. For each affected category and each action in the rule:
               σ[c_idx, a_idx] += direction × weight
        5. Clip σ to [-sigma_max, +sigma_max]
        6. τ_mod = max(0.5, 1.0 - max(urgency × confidence across all active claims))

        Returns SynthesisBias with lambda=lambda_coupling.
        λ=0 caller → SynthesisBias.neutral() behavior preserved.
        """
        n_cat = len(self.categories)
        n_act = len(self.actions)
        sigma = np.zeros((n_cat, n_act), dtype=float)

        active_count = 0
        max_urgency_x_conf = 0.0

        for claim in claims:
            # Effective confidence = source_trust × extraction_confidence
            eff_conf = claim.get("confidence", 0.5) * claim.get("extraction_confidence", 1.0)
            if eff_conf < self.extraction_threshold:
                continue  # Does not meet extraction bar

            claim_type = claim.get("type", "")
            if claim_type not in self.rules:
                continue  # Unknown claim type — skip silently

            rule = self.rules[claim_type]
            decay_class = claim.get("decay_class", "standard")
            age_days = float(claim.get("age_days", 0))
            rate = self.decay_rates.get(decay_class, self.DEFAULT_DECAY_RATES["standard"])
            decay = np.exp(-rate * age_days)
            weight = claim["confidence"] * decay  # Use raw confidence (not × extraction_conf) for weight

            for cat_name in claim.get("categories_affected", []):
                if cat_name not in self.cat_index:
                    continue  # Unknown category — skip silently
                c_idx = self.cat_index[cat_name]
                for act_name, direction in rule.items():
                    if act_name in self.act_index:
                        a_idx = self.act_index[act_name]
                        sigma[c_idx, a_idx] += direction * weight

            active_count += 1
            urgency_x_conf = claim.get("urgency", 0.5) * claim.get("confidence", 1.0)
            max_urgency_x_conf = max(max_urgency_x_conf, urgency_x_conf)

        # Clip σ to [-σ_max, +σ_max]
        sigma = np.clip(sigma, -self.sigma_max, self.sigma_max)

        return SynthesisBias(
            sigma=sigma,
            active_claims=active_count,
            lambda_coupling=lambda_coupling,
        )

    def project_with_trace(
        self,
        claims: List[Dict[str, Any]],
        lambda_coupling: float = 0.1,
    ) -> tuple:
        """
        Same as project() but also returns a trace dict for auditability.
        trace[cat_name][action_name] = list of (claim_type, weight, direction, contribution)
        Used by Tab 5 "Scoring Impact" display.
        """
        n_cat = len(self.categories)
        n_act = len(self.actions)
        sigma = np.zeros((n_cat, n_act), dtype=float)
        trace: Dict[str, Dict[str, list]] = {
            c: {a: [] for a in self.actions} for c in self.categories
        }

        active_count = 0
        max_urgency_x_conf = 0.0

        for claim in claims:
            eff_conf = claim.get("confidence", 0.5) * claim.get("extraction_confidence", 1.0)
            if eff_conf < self.extraction_threshold:
                continue
            claim_type = claim.get("type", "")
            if claim_type not in self.rules:
                continue

            rule = self.rules[claim_type]
            decay_class = claim.get("decay_class", "standard")
            age_days = float(claim.get("age_days", 0))
            rate = self.decay_rates.get(decay_class, self.DEFAULT_DECAY_RATES["standard"])
            decay = np.exp(-rate * age_days)
            weight = claim["confidence"] * decay

            for cat_name in claim.get("categories_affected", []):
                if cat_name not in self.cat_index:
                    continue
                c_idx = self.cat_index[cat_name]
                for act_name, direction in rule.items():
                    if act_name in self.act_index:
                        a_idx = self.act_index[act_name]
                        contribution = direction * weight
                        sigma[c_idx, a_idx] += contribution
                        trace[cat_name][act_name].append({
                            "claim_type": claim_type,
                            "weight": round(weight, 4),
                            "direction": direction,
                            "contribution": round(contribution, 4),
                        })

            active_count += 1
            max_urgency_x_conf = max(
                max_urgency_x_conf,
                claim.get("urgency", 0.5) * claim.get("confidence", 1.0)
            )

        sigma = np.clip(sigma, -self.sigma_max, self.sigma_max)

        bias = SynthesisBias(
            sigma=sigma,
            active_claims=active_count,
            lambda_coupling=lambda_coupling,
        )
        return bias, trace


if __name__ == "__main__":
    from src.data.claim_generator import (
        SOC_SYNTHESIS_RULES, CATEGORIES, ACTIONS, generate_correct_claims
    )

    projector = RuleBasedProjector(
        rules=SOC_SYNTHESIS_RULES,
        categories=CATEGORIES,
        actions=ACTIONS,
    )
    claims = generate_correct_claims(n=10, seed=42)
    bias = projector.project(claims, lambda_coupling=0.1)

    assert isinstance(bias, SynthesisBias)
    assert bias.sigma.shape == (len(CATEGORIES), len(ACTIONS))
    assert bias.lambda_coupling == 0.1
    print(f"Projector OK — {bias.active_claims} claims active")

    # Print σ tensor
    print("\nσ tensor (rows=categories, cols=actions):")
    header = f"{'':30s}" + "".join(f"{a:>12s}" for a in ACTIONS)
    print(header)
    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:30s}" + "".join(f"{bias.sigma[i,j]:>12.4f}" for j in range(len(ACTIONS)))
        print(row)

    # Verify neutral behavior
    neutral = SynthesisBias.neutral(len(CATEGORIES), len(ACTIONS))
    bias_neutral = projector.project([], lambda_coupling=0.1)
    assert bias_neutral.active_claims == 0
    print("\nEmpty claims → no active claims: OK")
    print("RuleBasedProjector all tests passed.")
