"""
claim_generator.py — Claim dataclass + generators for src/synthesis/ path.

Generates synthetic Claim objects for testing the synthesis pipeline.
"Correct" claims are aligned with ground truth (direction biases system toward right action).
"Poisoned" claims recommend misleading directions for wrong actions.

IMPORTANT: These are SYNTHETIC claims for experiment validation only.
Real claims come from ContextConnectors (F13) at v6.5+.

Claim direction convention:
    direction = +1  ->  bias toward this action for this category
    direction = -1  ->  bias away from this action for this category

In Eq. 4-synthesis, sigma < 0 makes an action MORE likely.
RuleBasedProjector maps direction -> sigma via negation:
    sigma[c,a] += -direction * strength * tier_weight * lambda_coupling
So direction=+1 produces sigma < 0 (more likely), direction=-1 produces sigma > 0 (less likely).

GT-alignment:
    generate_correct_claims() requires gt_profiles (n_categories, n_actions, n_factors).
    For each claim, it picks a random category c and finds the GT-preferred action a*:
        a* = argmax_a  L1-norm(gt_profiles[c, a, :])
    Claims are then direction=+1 toward a*, reinforcing what GT profiles already encode.
    Without gt_profiles the claims land on random (cat, action) cells -> pure noise.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Claim:
    category_idx: int
    action_idx: int
    direction: int        # +1 = bias toward action, -1 = bias away from action
    strength: float       # 0.0-1.0
    confidence: float     # 0.0-1.0
    source_tier: int      # 1=authoritative(CISA/NVD), 2=vendor, 3=research, 4=OSINT
    description: str = ""


def generate_correct_claims(
    n_claims: int,
    seed: int,
    gt_profiles: np.ndarray,
    n_categories: int = 6,
    n_actions: int = 4,
    source_tier: int = 1,
    confidence: float = 0.9,
    strength: float = 0.8,
) -> List[Claim]:
    """
    Generates n_claims Claim objects aligned with GT profiles.

    For each claim: picks a random category c, finds the GT-preferred action a*
    (argmax L1-norm of gt_profiles[c, a, :]), sets direction=+1 toward a*.

    These claims reinforce the right action per category, so they help most
    when profiles are stale or cold — the sigma nudge supplements degraded mu.

    Parameters
    ----------
    n_claims : int
        Number of claims to generate.
    seed : int
        RNG seed (independent from alert seed).
    gt_profiles : np.ndarray
        Shape (n_categories, n_actions, n_factors).  Ground-truth centroids.
    n_categories : int
        Number of categories to sample from (must match gt_profiles.shape[0]).
    n_actions : int
        Number of actions (must match gt_profiles.shape[1]).
    source_tier : int
        Claim authority tier (1=authoritative).
    confidence : float
        Claim confidence — must exceed RuleBasedProjector.extraction_threshold (0.8).
    strength : float
        Claim strength (0.0-1.0).

    Used for clean experiments (EXP-S1, EXP-S1b, EXP-S3, EXP-S4).
    """
    rng = np.random.default_rng(seed)
    claims = []

    for i in range(n_claims):
        c = int(rng.integers(0, n_categories))
        # GT-preferred action: action whose centroid has the highest L1 norm
        # (most activated = most distinctive profile for this category)
        action_norms = [
            float(np.sum(np.abs(gt_profiles[c, a, :])))
            for a in range(n_actions)
        ]
        a_star = int(np.argmax(action_norms))
        claims.append(Claim(
            category_idx=c,
            action_idx=a_star,
            direction=+1,
            strength=strength,
            confidence=confidence,
            source_tier=source_tier,
            description=f"GT-aligned claim: cat={c}, action={a_star}",
        ))

    return claims


def generate_poisoned_claims(
    n_correct: int,
    n_poisoned: int,
    seed: int,
    gt_profiles: np.ndarray,
    n_categories: int = 6,
    n_actions: int = 4,
) -> List[Claim]:
    """
    Generate a mix of correct (GT-aligned) and poisoned (wrong action) claims.

    Poisoned claims: direction=+1 toward a random non-optimal action for
    that category, actively misleading scoring away from the right action.
    Confidence values are realistic (0.9) so they pass the extraction threshold.

    Parameters
    ----------
    n_correct : int
        Number of GT-aligned correct claims.
    n_poisoned : int
        Number of poisoned (wrong-action) claims.
    seed : int
        RNG seed.
    gt_profiles : np.ndarray
        Shape (n_categories, n_actions, n_factors). Required to identify a*.

    Used for resilience experiments (EXP-S2).
    """
    claims = generate_correct_claims(
        n_correct, seed, gt_profiles, n_categories, n_actions
    )
    rng = np.random.default_rng(seed + 999)

    for i in range(n_poisoned):
        c = int(rng.integers(0, n_categories))
        action_norms = [
            float(np.sum(np.abs(gt_profiles[c, a, :])))
            for a in range(n_actions)
        ]
        a_star = int(np.argmax(action_norms))
        wrong_actions = [a for a in range(n_actions) if a != a_star]
        a_wrong = int(rng.choice(wrong_actions))
        claims.append(Claim(
            category_idx=c,
            action_idx=a_wrong,
            direction=+1,
            strength=0.8,
            confidence=0.9,
            source_tier=1,
            description=f"POISONED claim: cat={c}, action={a_wrong} (GT={a_star})",
        ))

    return claims


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.data.category_alert_generator import CategoryAlertGenerator
    from src.synthesis.rule_projector import RuleBasedProjector

    gen = CategoryAlertGenerator(seed=42)
    gen.generate(100)   # warm up RNG (consistent with experiment usage)

    # Build gt_profiles array (n_cats, n_acts, n_factors)
    n_cats  = len(gen.categories)
    n_acts  = len(gen.actions)
    n_facts = len(gen.factors)
    gt_profiles = np.zeros((n_cats, n_acts, n_facts), dtype=np.float64)
    for ci, cat in enumerate(gen.categories):
        for ai, act in enumerate(gen.actions):
            gt_profiles[ci, ai, :] = gen.profiles[cat][act]

    claims = generate_correct_claims(10, seed=42, gt_profiles=gt_profiles,
                                     n_categories=n_cats, n_actions=n_acts)
    assert len(claims) == 10
    assert all(isinstance(c, Claim) for c in claims)
    assert all(c.confidence >= 0.85 for c in claims)
    assert all(c.direction == +1 for c in claims)
    print(f"Correct claims: {len(claims)} generated")
    for cl in claims:
        print(f"  cat={cl.category_idx} action={cl.action_idx} "
              f"dir={cl.direction:+d} str={cl.strength:.2f} "
              f"tier={cl.source_tier} -> {cl.description}")

    # Project through RuleBasedProjector
    projector = RuleBasedProjector()
    bias = projector.project(claims, n_categories=n_cats, n_actions=n_acts,
                             lambda_coupling=0.1)
    print(f"\nsigma tensor ({n_cats} cats x {n_acts} actions):")
    print(bias.tensor())
    print("Non-zero entries:", np.count_nonzero(bias.tensor()))
    print("All GT-preferred cells should be negative (sigma<0 -> more likely), others zero.")

    # Poisoned mix
    mixed = generate_poisoned_claims(n_correct=8, n_poisoned=2, seed=42,
                                     gt_profiles=gt_profiles,
                                     n_categories=n_cats, n_actions=n_acts)
    assert len(mixed) == 10
    print(f"\nPoisoned mix: {len(mixed)} total")
    print("\nAll claim_generator tests passed.")
