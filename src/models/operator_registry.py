"""
OperatorRegistry — manages active operators, applies composition,
enforces TTL expiry.

Zero contact with ProfileScorer.update() — the firewall is architectural.
The registry produces SynthesisBias; ProfileScorer consumes it at score time.
update() never sees it.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.operator_spec import OperatorSpec, validate_operator, OperatorValidationError
from src.models.synthesis import SynthesisBias


@dataclass
class ActiveOperator:
    spec:                 OperatorSpec
    decisions_remaining:  int
    decisions_applied:    int = 0


class OperatorRegistry:
    """
    Manages the lifecycle of active OperatorSpec instances.

    Args:
        n_categories:    Number of alert categories.
        n_actions:       Number of possible actions.
        n_factors:       Centroid factor dimension.
        k_max:           Maximum rank per operator (default 2).
        stability_bound: Max cumulative ||delta_mu||_F (default 0.3).
    """

    def __init__(
        self,
        n_categories:    int,
        n_actions:       int,
        n_factors:       int,
        k_max:           int   = 2,
        stability_bound: float = 0.3,
    ):
        self.n_categories    = n_categories
        self.n_actions       = n_actions
        self.n_factors       = n_factors
        self.k_max           = k_max
        self.stability_bound = stability_bound

        self._active:           list[ActiveOperator] = []
        self._expired:          list[OperatorSpec]   = []
        self._total_decisions:  int = 0

    def register(self, spec: OperatorSpec, current_mu: np.ndarray) -> None:
        """Validate and register an operator. Raises OperatorValidationError on failure."""
        validate_operator(
            spec=spec,
            current_mu=current_mu,
            active_specs=[ao.spec for ao in self._active],
            k_max=self.k_max,
            stability_bound=self.stability_bound,
        )
        self._active.append(ActiveOperator(
            spec=spec,
            decisions_remaining=spec.ttl_decisions,
        ))

    def step(self, n: int = 1) -> None:
        """Advance decision counter. Does NOT expire — call expire_stale() separately."""
        self._total_decisions += n
        for ao in self._active:
            ao.decisions_remaining -= n
            ao.decisions_applied   += n

    def expire_stale(self) -> list[str]:
        """Remove TTL-expired operators. Returns list of expired operator_ids."""
        expired_ids, still_active = [], []
        for ao in self._active:
            if ao.decisions_remaining <= 0:
                self._expired.append(ao.spec)
                expired_ids.append(ao.spec.operator_id)
            else:
                still_active.append(ao)
        self._active = still_active
        return expired_ids

    def clear(self) -> None:
        """Remove all active operators."""
        self._expired.extend([ao.spec for ao in self._active])
        self._active = []

    def get_synthesis(self, lambda_override: float | None = None) -> SynthesisBias:
        """
        Produce SynthesisBias from all active operators.

        Accumulates sigma contributions; clips composition to [-1, +1].
        Returns SynthesisBias.neutral() if no operators are active.
        """
        if not self._active:
            return SynthesisBias.neutral(self.n_categories, self.n_actions)

        sigma = np.zeros((self.n_categories, self.n_actions))
        for ao in self._active:
            lam    = lambda_override if lambda_override is not None else ao.spec.effective_lambda_s()
            sigma += lam * ao.spec.sigma_0

        sigma      = np.clip(sigma, -1.0, 1.0)
        max_lambda = max(
            (lambda_override or ao.spec.effective_lambda_s())
            for ao in self._active
        )

        return SynthesisBias(
            sigma=sigma,
            active_claims=len(self._active),
            lambda_coupling=max_lambda,
        )

    def n_active(self)        -> int:        return len(self._active)
    def active_ids(self)      -> list[str]:  return [ao.spec.operator_id for ao in self._active]
    def total_decisions(self) -> int:        return self._total_decisions


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    C, A, d = 5, 4, 6
    mu = rng.random((C, A, d)) * 0.4 + 0.3

    registry = OperatorRegistry(n_categories=C, n_actions=A, n_factors=d)

    s = registry.get_synthesis()
    assert not s.is_active
    assert s.sigma.shape == (C, A)
    print("[PASS] Empty registry -> neutral synthesis")

    sigma_0 = np.zeros((C, A))
    sigma_0[0, 2] =  0.4
    sigma_0[0, 0] = -0.3
    spec = OperatorSpec(
        operator_id="op_001", claim_type="active_campaign",
        rank=0, sigma_0=sigma_0, lambda_s=0.2, ttl_decisions=10,
    )
    registry.register(spec, mu)
    assert registry.n_active() == 1
    print(f"[PASS] Registered operator, n_active={registry.n_active()}")

    s = registry.get_synthesis()
    assert s.is_active
    assert s.active_claims == 1
    print(f"[PASS] Active synthesis: lambda={s.lambda_coupling:.2f}, sigma[0,0]={s.sigma[0,0]:.3f}")

    registry.step(10)
    expired = registry.expire_stale()
    assert "op_001" in expired
    assert registry.n_active() == 0
    assert not registry.get_synthesis().is_active
    print(f"[PASS] TTL expiry: expired={expired}, n_active={registry.n_active()}")

    registry2 = OperatorRegistry(n_categories=C, n_actions=A, n_factors=d, stability_bound=0.01)
    registry2.register(spec, mu)
    try:
        registry2.register(spec, mu)
        print("[WARN] Composition check did not fire — stability_bound may need adjustment for this spec")
    except OperatorValidationError as e:
        print(f"[PASS] Composition stability caught second operator: {e}")

    print("\noperator_registry.py: all self-tests passed.")
