"""
synthesis.py — SynthesisBias dataclass + SynthesisProjector protocol

SYNTH-EXP-0 deliverable. Lives in src/models/ of cross-graph-experiments.

Design: intelligence_layer_design_v1 §2 (Eq. 4-synthesis) and §8 (code specs).

σ[c,a] is a scalar per (category, action) pair that biases action selection.
λ (lambda_coupling) controls coupling strength. λ=0 → exact Eq. 4-final behavior.

CRITICAL: σ is used ONLY in score(). It is NEVER passed to update().
          Centroids (μ) learn from experience. σ encodes current awareness.
          These are different epistemic categories and must never be conflated.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, List, Dict, Any


@dataclass(frozen=True)
class SynthesisBias:
    """
    Immutable synthesis bias state.

    sigma:          shape (n_categories, n_actions). The awareness bias tensor.
                    σ[c,a] < 0 → action a MORE likely for category c
                    σ[c,a] > 0 → action a LESS likely for category c
    active_claims:  Number of claims that passed the extraction threshold.
    lambda_coupling: Current coupling constant λ ∈ [0.0, ∞).
                    λ = 0.0 → exact Eq. 4-final (kill switch).
    """
    sigma: np.ndarray            # shape: (n_categories, n_actions)
    active_claims: int           # 0 = no claims contributed
    lambda_coupling: float       # 0.0 = synthesis disabled

    def __post_init__(self):
        if not isinstance(self.sigma, np.ndarray):
            object.__setattr__(self, 'sigma', np.array(self.sigma, dtype=float))
        if self.sigma.ndim != 2:
            raise ValueError(f"sigma must be 2D, got shape {self.sigma.shape}")
        if self.lambda_coupling < 0:
            raise ValueError(f"lambda_coupling must be non-negative, got {self.lambda_coupling}")

    @staticmethod
    def neutral(n_categories: int, n_actions: int) -> "SynthesisBias":
        """
        Zero bias — equivalent to no synthesis.
        σ = 0 everywhere, τ_mod = 1.0, λ = 0.
        score(f, c, neutral) == score(f, c, None) — guaranteed by Eq. 4-synthesis.
        """
        return SynthesisBias(
            sigma=np.zeros((n_categories, n_actions), dtype=float),
            active_claims=0,
            lambda_coupling=0.0,
        )

    @property
    def is_active(self) -> bool:
        """True if synthesis will actually change scoring (λ > 0 and any σ ≠ 0)."""
        return self.lambda_coupling > 0.0 and self.active_claims > 0

    def effective_shift(self, category_index: int) -> np.ndarray:
        """
        Return λ·σ[c,:] — the additive distance shift for this category.
        This is what Eq. 4-synthesis adds to the L2 distance term.
        """
        return self.lambda_coupling * self.sigma[category_index, :]

    def describe(self, categories: List[str] = None, actions: List[str] = None) -> str:
        """
        Human-readable summary of the synthesis bias.
        Suitable for advisory logging and Tab 5 display.
        """
        lines = [
            f"SynthesisBias: λ={self.lambda_coupling:.3f}, active_claims={self.active_claims}",
        ]
        if not self.is_active:
            lines.append("  (synthesis inactive — no effect on scoring)")
            return "\n".join(lines)

        n_cat, n_act = self.sigma.shape
        cat_labels = categories or [f"cat_{i}" for i in range(n_cat)]
        act_labels = actions or [f"act_{j}" for j in range(n_act)]

        nonzero = [(i, j) for i in range(n_cat) for j in range(n_act)
                   if abs(self.sigma[i, j]) > 1e-6]
        if nonzero:
            lines.append("  Non-zero σ[c,a] entries:")
            for i, j in nonzero:
                direction = "↑ escalation" if self.sigma[i, j] < 0 else "↓ suppression"
                lines.append(
                    f"    [{cat_labels[i]:25s}, {act_labels[j]:12s}] = "
                    f"{self.sigma[i, j]:+.4f}  ({direction} pressure)"
                )
        return "\n".join(lines)


@runtime_checkable
class SynthesisProjector(Protocol):
    """
    Protocol for mapping a list of claims → SynthesisBias.

    Implementations:
      - RuleBasedProjector: domain-configured rule templates (v5.0)
      - LearnedProjector:   calibrated from outcome correlations (v6.5+)

    Claims are dicts with at minimum:
      type: str
      categories_affected: list[str]
      confidence: float ∈ [0, 1]
      extraction_confidence: float ∈ [0, 1]
      urgency: float ∈ [0, 1]
      age_days: float ≥ 0
      decay_class: str ("campaign" | "standard" | "transient")
    """

    def project(
        self,
        claims: List[Dict[str, Any]],
        lambda_coupling: float = 0.1,
    ) -> SynthesisBias:
        ...


if __name__ == "__main__":
    # Quick smoke test
    bias_neutral = SynthesisBias.neutral(6, 4)
    assert bias_neutral.sigma.shape == (6, 4)
    assert bias_neutral.active_claims == 0
    assert not bias_neutral.is_active
    print("SynthesisBias.neutral OK")

    sigma = np.zeros((6, 4))
    sigma[1, 2] = 0.5   # credential_access, suppress: make less likely
    sigma[2, 0] = -0.4  # threat_intel_match, escalate: make more likely
    bias_active = SynthesisBias(sigma=sigma, active_claims=3, lambda_coupling=0.1)
    assert bias_active.is_active
    assert abs(bias_active.effective_shift(1)[2] - 0.05) < 1e-9   # λ * σ[1,2]
    print("SynthesisBias active OK")

    cats = ["travel_anomaly", "credential_access", "threat_intel_match",
            "insider_behavioral", "cloud_infrastructure", "healthcare"]
    acts = ["escalate", "investigate", "suppress", "monitor"]
    print(bias_active.describe(cats, acts))
    print("All SynthesisBias tests passed.")
