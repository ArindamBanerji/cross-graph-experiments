"""
OperatorSpec — a declared perturbation operator on centroid space.

An operator is a domain adapter's declaration of how centroids should shift
during a transition window. The GAE applies it mechanically; the adapter
authors it semantically.

Rank-0 (scalar): sigma_0[c,a] shifts action preferences, no geometry.
Rank-1 (vector): adds a declared direction v_hat[c] with coupling beta[c,a].

Rank-1 is PROPOSED — EXP-OP5 gates whether it adds value over rank-0.

Five structural checks run at registration:
  1. Schema validity  — required fields, types, value ranges
  2. Bounds safety    — mu + delta_mu stays in [0,1]^d
  3. Rank constraint  — k <= k_max (default 2)
  4. Firewall check   — spec touches no update() parameter
  5. Composition      — cumulative ||delta_mu||_F <= 0.3
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

# Fields that update() reads — operators must never declare these
_UPDATE_PARAMS = frozenset({"eta", "eta_neg", "decay_rate", "counts", "mu"})


@dataclass
class OperatorSpec:
    """
    A declared perturbation operator.

    Required:
        operator_id:    Unique string identifier.
        claim_type:     Semantic type ("active_campaign", "ciso_directive", etc.)
        rank:           0 for scalar-only, 1 for rank-1 vector extension.
        sigma_0:        (n_categories, n_actions) signed bias pattern.
                        Negative = encourage action. Positive = discourage.
        lambda_s:       Scalar coupling constant in [0, 0.5].
        ttl_decisions:  Lifetime in decisions.

    Rank-1 only (ignored for rank=0):
        v_hat:          (n_categories, n_factors) unit-norm direction vectors.
        lambda_v:       Vector coupling constant in [0, 0.3].
        beta:           (n_categories, n_actions) vector coupling magnitudes.
                        If None: beta = -sigma_0 (v1 simplification).

    Metadata:
        description, source, confidence (scales effective coupling).
    """
    operator_id:   str
    claim_type:    str
    rank:          Literal[0, 1, 2]
    sigma_0:       np.ndarray      # shape: (n_categories, n_actions)
    lambda_s:      float
    ttl_decisions: int

    v_hat:         np.ndarray | None = None   # shape: (n_categories, n_factors)
    lambda_v:      float = 0.0
    beta:          np.ndarray | None = None   # shape: (n_categories, n_actions)

    description:   str = ""
    source:        str = ""
    confidence:    float = 1.0

    def effective_lambda_s(self) -> float:
        return self.lambda_s * self.confidence

    def effective_lambda_v(self) -> float:
        return self.lambda_v * self.confidence

    def effective_beta(self) -> np.ndarray:
        """Beta with v1 simplification: beta = -sigma_0 if not declared."""
        if self.beta is not None:
            return self.beta
        return -self.sigma_0

    def compute_delta_mu(self, current_mu: np.ndarray) -> np.ndarray:
        """
        Centroid perturbation Delta_mu.

        rank-0: returns zeros (sigma_0 enters scoring path directly, not centroids)
        rank-1: Delta_mu[c,a,:] = lambda_v * beta[c,a] * v_hat[c]
        """
        C, A, d = current_mu.shape
        delta = np.zeros((C, A, d))

        if self.rank >= 1 and self.v_hat is not None:
            beta = self.effective_beta()
            lv   = self.effective_lambda_v()
            for c in range(C):
                for a in range(A):
                    delta[c, a, :] = lv * beta[c, a] * self.v_hat[c]

        return delta

    def compute_mu_tilde(self, current_mu: np.ndarray) -> np.ndarray:
        """Effective centroid mu_tilde = clip(mu + Delta_mu, 0, 1)."""
        return np.clip(current_mu + self.compute_delta_mu(current_mu), 0.0, 1.0)


class OperatorValidationError(ValueError):
    pass


def check_schema_validity(spec: OperatorSpec) -> None:
    if not spec.operator_id or not isinstance(spec.operator_id, str):
        raise OperatorValidationError("operator_id must be a non-empty string")
    if spec.rank not in (0, 1, 2):
        raise OperatorValidationError(f"rank must be 0, 1, or 2; got {spec.rank}")
    if not isinstance(spec.sigma_0, np.ndarray) or spec.sigma_0.ndim != 2:
        raise OperatorValidationError("sigma_0 must be a 2D numpy array (n_categories, n_actions)")
    if not np.all(np.isfinite(spec.sigma_0)):
        raise OperatorValidationError("sigma_0 contains non-finite values")
    if np.any(np.abs(spec.sigma_0) > 1.0):
        raise OperatorValidationError(f"sigma_0 values outside [-1,1]: max abs = {np.abs(spec.sigma_0).max():.3f}")
    if not (0.0 <= spec.lambda_s <= 0.5):
        raise OperatorValidationError(f"lambda_s must be in [0, 0.5]; got {spec.lambda_s}")
    if spec.ttl_decisions <= 0:
        raise OperatorValidationError(f"ttl_decisions must be > 0; got {spec.ttl_decisions}")
    if not (0.0 <= spec.confidence <= 1.0):
        raise OperatorValidationError(f"confidence must be in [0,1]; got {spec.confidence}")
    if spec.rank >= 1:
        if spec.v_hat is None:
            raise OperatorValidationError("rank=1 operator requires v_hat")
        if spec.v_hat.ndim != 2:
            raise OperatorValidationError("v_hat must be 2D (n_categories, n_factors)")
        if not np.all(np.isfinite(spec.v_hat)):
            raise OperatorValidationError("v_hat contains non-finite values")
        norms = np.linalg.norm(spec.v_hat, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise OperatorValidationError(f"v_hat rows must be unit norm; got norms: {norms}")
        if not (0.0 <= spec.lambda_v <= 0.3):
            raise OperatorValidationError(f"lambda_v must be in [0, 0.3]; got {spec.lambda_v}")


def check_bounds_safety(spec: OperatorSpec, current_mu: np.ndarray) -> None:
    delta  = spec.compute_delta_mu(current_mu)
    mu_new = current_mu + delta
    if np.any(mu_new < -0.05) or np.any(mu_new > 1.05):
        raise OperatorValidationError(
            f"Bounds safety: mu + delta_mu out of [0,1]: "
            f"min={mu_new.min():.3f}, max={mu_new.max():.3f}"
        )


def check_rank_constraint(spec: OperatorSpec, k_max: int = 2) -> None:
    if spec.rank > k_max:
        raise OperatorValidationError(f"rank={spec.rank} > k_max={k_max}")


def check_firewall_compliance(spec: OperatorSpec) -> None:
    violations = set(vars(spec).keys()) & _UPDATE_PARAMS
    if violations:
        raise OperatorValidationError(
            f"Firewall violation: spec references update() parameters: {violations}"
        )


def check_composition_stability(
    new_spec:        OperatorSpec,
    active_specs:    list[OperatorSpec],
    current_mu:      np.ndarray,
    stability_bound: float = 0.3,
) -> None:
    cumulative_delta = np.zeros_like(current_mu)
    for spec in (active_specs + [new_spec]):
        cumulative_delta += spec.compute_delta_mu(current_mu)
    norm = float(np.sqrt(np.sum(cumulative_delta ** 2)))
    if norm > stability_bound:
        raise OperatorValidationError(
            f"Composition stability: cumulative ||delta_mu||_F = {norm:.4f} > {stability_bound}"
        )


def validate_operator(
    spec:             OperatorSpec,
    current_mu:       np.ndarray,
    active_specs:     list[OperatorSpec] | None = None,
    k_max:            int   = 2,
    stability_bound:  float = 0.3,
) -> None:
    """Run all five structural checks. Raises OperatorValidationError on failure."""
    check_schema_validity(spec)
    check_bounds_safety(spec, current_mu)
    check_rank_constraint(spec, k_max)
    check_firewall_compliance(spec)
    check_composition_stability(spec, active_specs or [], current_mu, stability_bound)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    C, A, d = 5, 4, 6
    mu = rng.random((C, A, d)) * 0.4 + 0.3

    sigma_0 = np.zeros((C, A))
    sigma_0[0, 2] =  0.4
    sigma_0[0, 0] = -0.3

    spec0 = OperatorSpec(
        operator_id="test_rank0", claim_type="active_campaign",
        rank=0, sigma_0=sigma_0, lambda_s=0.2, ttl_decisions=300,
    )
    validate_operator(spec0, mu)
    print("[PASS] Rank-0 operator validates")

    v_hat = rng.random((C, d))
    v_hat /= np.linalg.norm(v_hat, axis=1, keepdims=True)
    spec1 = OperatorSpec(
        operator_id="test_rank1", claim_type="active_campaign",
        rank=1, sigma_0=sigma_0, lambda_s=0.1, lambda_v=0.1,
        v_hat=v_hat, ttl_decisions=300,
    )
    validate_operator(spec1, mu)
    delta = spec1.compute_delta_mu(mu)
    assert delta.shape == mu.shape
    mu_tilde = spec1.compute_mu_tilde(mu)
    assert np.all(mu_tilde >= 0) and np.all(mu_tilde <= 1)
    print(f"[PASS] Rank-1 operator validates, delta_norm={float(np.sqrt(np.sum(delta**2))):.4f}")

    import copy
    bad_spec = copy.copy(spec0)
    bad_spec.__dict__["eta"] = 0.05
    try:
        check_firewall_compliance(bad_spec)
        assert False, "Should have raised"
    except OperatorValidationError as e:
        print(f"[PASS] Firewall check caught: {e}")

    bad_sigma = sigma_0.copy()
    bad_sigma[0, 0] = 2.0
    bad_spec2 = OperatorSpec(
        operator_id="bad", claim_type="test", rank=0,
        sigma_0=bad_sigma, lambda_s=0.1, ttl_decisions=100,
    )
    try:
        check_schema_validity(bad_spec2)
        assert False, "Should have raised"
    except OperatorValidationError as e:
        print(f"[PASS] Schema check caught sigma out of range: {e}")

    print("\noperator_spec.py: all self-tests passed.")
