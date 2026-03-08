"""
bias_generator.py — Analyst bias simulation for FX-2.
experiments/fx2_noise_distributions/bias_generator.py

Three bias patterns that simulate real analyst behavior:
  POST_INCIDENT_ESCALATION — over-escalation for 50 decisions after an incident
  ALERT_FATIGUE            — persistent suppression drift after dec 300
  EXPERTISE_GRADIENT       — expert on first 2 categories, random on rest

The analyst's SUBMITTED action may differ from GT. The model updates from
the SUBMITTED action — this is the key: biased feedback corrupts centroid updates.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from src.models.profile_scorer import ProfileScorer


class BiasPattern(Enum):
    POST_INCIDENT_ESCALATION = "post_incident"
    ALERT_FATIGUE            = "alert_fatigue"
    EXPERTISE_GRADIENT       = "expertise_gradient"


class BiasedFeedbackSimulator:
    """
    Wraps a ProfileScorer to simulate realistic analyst feedback biases.

    For each decision, the GT action is known. The analyst's SUBMITTED
    action may differ from GT based on the active bias pattern.
    The model updates from the SUBMITTED action, not the GT action.
    This is the key: biased feedback corrupts centroid updates.

    Parameters
    ----------
    scorer : ProfileScorer
        The scorer whose centroids will be updated.
    pattern : BiasPattern
        Which bias pattern to simulate.
    seed : int
        RNG seed for stochastic bias decisions.
    """

    POST_INCIDENT_WINDOW = 50    # decisions after incident where analyst over-escalates
    FATIGUE_ONSET        = 300   # decision where fatigue begins
    FATIGUE_RATE         = 0.003 # probability of suppress increases per decision
    FATIGUE_MAX          = 0.70  # maximum suppress bias probability

    def __init__(
        self,
        scorer:  ProfileScorer,
        pattern: BiasPattern,
        seed:    int = 42,
    ) -> None:
        self.scorer           = scorer
        self.pattern          = pattern
        self.rng              = np.random.default_rng(seed)
        self.decision_count   = 0
        self.last_incident_at = -999  # decision index of last simulated incident

    def simulate_incident(self, at_decision: int) -> None:
        """Record that an incident occurred at this decision index."""
        self.last_incident_at = at_decision

    def get_analyst_action(
        self,
        gt_action_index: int,
        category_idx:    int,
        n_actions:       int,
    ) -> int:
        """
        Returns the action the biased analyst SUBMITS (1-indexed by decision_count
        which is incremented here).

        POST_INCIDENT_ESCALATION:
          For POST_INCIDENT_WINDOW decisions after an incident, analyst overrides
          suppress/monitor (indices 2, 3) to escalate (index 0) with P=0.90.
          After the window expires, returns to accurate behavior.

        ALERT_FATIGUE:
          After FATIGUE_ONSET decisions, probability of submitting 'suppress'
          (index 2) increases linearly at FATIGUE_RATE per decision, regardless
          of GT action, capped at FATIGUE_MAX=0.70. Fatigue is persistent.

        EXPERTISE_GRADIENT:
          Expert on first 2 categories (indices 0, 1): 99% accurate.
          Random on remaining categories: 25% accurate (coin flip between actions).
        """
        self.decision_count += 1

        if self.pattern == BiasPattern.POST_INCIDENT_ESCALATION:
            since_incident = self.decision_count - self.last_incident_at
            if 0 <= since_incident <= self.POST_INCIDENT_WINDOW:
                escalate_idx = 0  # escalate
                if gt_action_index in (2, 3):  # suppress or monitor
                    if self.rng.random() < 0.90:
                        return escalate_idx
            return gt_action_index

        elif self.pattern == BiasPattern.ALERT_FATIGUE:
            if self.decision_count > self.FATIGUE_ONSET:
                fatigue_prob = min(
                    self.FATIGUE_MAX,
                    (self.decision_count - self.FATIGUE_ONSET) * self.FATIGUE_RATE,
                )
                suppress_idx = 2  # suppress
                if self.rng.random() < fatigue_prob:
                    return suppress_idx
            return gt_action_index

        elif self.pattern == BiasPattern.EXPERTISE_GRADIENT:
            expert_categories = (0, 1)  # travel_anomaly, credential_access
            if category_idx in expert_categories:
                if self.rng.random() < 0.99:
                    return gt_action_index
                return int(self.rng.integers(0, n_actions))
            else:
                if self.rng.random() < 0.25:
                    return gt_action_index
                return int(self.rng.integers(0, n_actions))

        # Fallback: accurate
        return gt_action_index
