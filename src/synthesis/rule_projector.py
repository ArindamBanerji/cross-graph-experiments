"""
rule_projector.py — RuleBasedProjector for src/synthesis/ path.

Projects Claim objects into a SynthesisBias tensor.
Claims carry (category_idx, action_idx, direction, strength, confidence, source_tier).

Eq. S2 (simplified):
    sigma[c,a] = clip(sum_k direction_k * strength_k * tier_weight_k * lambda, -sigma_max, +sigma_max)
    for all claims k affecting (category c, action a) with confidence >= extraction_threshold.

Sign convention:
    sigma < 0 -> action MORE likely (smaller effective distance -> higher probability)
    sigma > 0 -> action LESS likely (larger effective distance -> lower probability)
    direction = +1 -> bias TOWARD action -> sigma < 0 (negated in project())
    direction = -1 -> bias AWAY from action -> sigma > 0

    project() negates direction so that Claim.direction=+1 produces sigma < 0 (more likely).
    This is the correct semantic mapping: "toward" means "make more likely".
"""
from __future__ import annotations

import numpy as np
from typing import List

from src.synthesis.synthesis import SynthesisBias


class RuleBasedProjector:
    """
    Maps Claim objects -> SynthesisBias using tier weights and confidence filtering.

    Can be instantiated with no arguments (uses defaults).
    """

    TIER_WEIGHTS = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.40}

    def __init__(self, sigma_max: float = 1.0, extraction_threshold: float = 0.8):
        """
        Args:
            sigma_max:             Clipping bound for each sigma[c,a] cell.
            extraction_threshold:  Minimum claim confidence to include a claim.
        """
        self.sigma_max = sigma_max
        self.extraction_threshold = extraction_threshold

    def project(
        self,
        claims: List,
        n_categories: int,
        n_actions: int,
        lambda_coupling: float = 0.1,
    ) -> SynthesisBias:
        """
        Returns SynthesisBias tensor.  Claims below extraction_threshold are ignored.
        sigma_max clips maximum sigma magnitude (safety control).

        Args:
            claims:          List of Claim objects (see src/synthesis/claim_generator.py).
            n_categories:    Number of categories (must match ProfileScorer).
            n_actions:       Number of actions (must match ProfileScorer).
            lambda_coupling: Scaling factor applied to each claim's contribution.

        Returns:
            SynthesisBias with sigma[n_categories, n_actions].
        """
        bias = SynthesisBias(n_categories, n_actions)

        for claim in claims:
            if claim.confidence < self.extraction_threshold:
                continue
            tier_w = self.TIER_WEIGHTS.get(claim.source_tier, 0.40)
            # Negate direction: +1 (toward) -> negative sigma (more likely); -1 (away) -> positive sigma
            value = -claim.direction * claim.strength * tier_w * lambda_coupling
            current = bias.get(claim.category_idx, claim.action_idx)
            bias.set(
                claim.category_idx,
                claim.action_idx,
                np.clip(current + value, -self.sigma_max, self.sigma_max),
            )

        return bias


if __name__ == "__main__":
    from src.synthesis.claim_generator import generate_correct_claims

    projector = RuleBasedProjector()
    claims = generate_correct_claims(10, seed=42)
    bias = projector.project(claims, n_categories=6, n_actions=4, lambda_coupling=0.1)

    assert isinstance(bias, SynthesisBias)
    assert bias.sigma.shape == (6, 4)
    nz = np.count_nonzero(bias.tensor())
    print(f"RuleBasedProjector OK — {nz} non-zero sigma entries")
    print(bias)
