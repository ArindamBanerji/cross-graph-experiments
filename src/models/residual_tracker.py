"""
ResidualTracker — tracks R[c,a,:](t) = mu_tilde[c,a,:] - mu[c,a,:](t)

The residual is the gap between where the operator declared centroids should
be (mu_tilde, the effective centroid) and where Loop 2 has actually moved
them (mu, the current centroid).

When the operator is correctly declared:
  R -> 0 as Loop 2 learns the post-shift distribution (self-extinguishing)

When the operator is incorrectly declared:
  R plateaus at a nonzero value (semantic misalignment diagnostic)

Used by EXP-OP3. Pure computation — no file I/O, no side effects.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ResidualSnapshot:
    """Residual state at one point in time."""
    decision_index: int
    residual: np.ndarray            # shape: (n_categories, n_actions, n_factors)
    frobenius_norm: float           # ||R||_F
    per_category_norms: np.ndarray  # ||R[c,:,:]||_F per category
    is_decayed: bool                # True if ||R||_F < epsilon_abs


class ResidualTracker:
    """
    Tracks R[c,a,:](t) = mu_tilde[c,a,:] - mu[c,a,:](t) over time.

    Args:
        mu_tilde:         Effective centroid declared by operator.
                          Shape: (n_categories, n_actions, n_factors).
        epsilon_fraction: Decay threshold as fraction of initial ||R||_F.
                          Default 0.05 (5% of initial norm = "absorbed").
        n_consecutive:    Consecutive snapshots below threshold before
                          declaring absorption. Default 10.
    """

    def __init__(
        self,
        mu_tilde: np.ndarray,
        epsilon_fraction: float = 0.05,
        n_consecutive: int = 10,
    ):
        assert mu_tilde.ndim == 3, (
            f"mu_tilde must be (C, A, d), got shape {mu_tilde.shape}"
        )
        self.mu_tilde = mu_tilde.copy()
        self.epsilon_fraction = epsilon_fraction
        self.n_consecutive = n_consecutive

        self._epsilon_abs: float | None = None
        self._consecutive_below: int = 0
        self.history: list[ResidualSnapshot] = []

    def record(self, current_mu: np.ndarray, decision_index: int) -> ResidualSnapshot:
        """
        Compute and record residual at current_mu.

        Args:
            current_mu:     Current centroid tensor from ProfileScorer.
                            Shape: (n_categories, n_actions, n_factors).
            decision_index: Which decision this snapshot corresponds to.
        """
        assert current_mu.shape == self.mu_tilde.shape, (
            f"Shape mismatch: current_mu {current_mu.shape} vs "
            f"mu_tilde {self.mu_tilde.shape}"
        )

        R = self.mu_tilde - current_mu
        frob = float(np.sqrt(np.sum(R ** 2)))
        per_cat = np.array([
            float(np.sqrt(np.sum(R[c, :, :] ** 2)))
            for c in range(R.shape[0])
        ])

        if self._epsilon_abs is None:
            self._epsilon_abs = self.epsilon_fraction * frob
            if self._epsilon_abs < 1e-6:
                self._epsilon_abs = 1e-6

        is_below = frob < self._epsilon_abs
        if is_below:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        snap = ResidualSnapshot(
            decision_index=decision_index,
            residual=R.copy(),
            frobenius_norm=frob,
            per_category_norms=per_cat,
            is_decayed=is_below,
        )
        self.history.append(snap)
        return snap

    def is_absorbed(self) -> bool:
        """True if residual stayed below epsilon_abs for n_consecutive snapshots."""
        return self._consecutive_below >= self.n_consecutive

    def decay_trajectory(self) -> np.ndarray:
        """Array of Frobenius norms over all recorded snapshots."""
        return np.array([s.frobenius_norm for s in self.history])

    def per_category_trajectory(self) -> np.ndarray:
        """(n_snapshots, n_categories) array of per-category norms."""
        if not self.history:
            return np.array([])
        return np.stack([s.per_category_norms for s in self.history])

    def summary(self) -> dict:
        """Summary statistics for the full tracking period."""
        if not self.history:
            return {"error": "no snapshots recorded"}
        norms = self.decay_trajectory()
        return {
            "initial_norm":       float(norms[0]),
            "final_norm":         float(norms[-1]),
            "min_norm":           float(norms.min()),
            "norm_reduction_pct": float(100 * (1 - norms[-1] / norms[0])) if norms[0] > 0 else 0.0,
            "absorbed":           self.is_absorbed(),
            "n_snapshots":        len(self.history),
            "epsilon_abs":        self._epsilon_abs,
        }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    C, A, d = 5, 4, 6

    mu_initial = rng.random((C, A, d)) * 0.4 + 0.3
    delta = np.zeros((C, A, d))
    delta[0, 2, :] = 0.15
    mu_tilde = np.clip(mu_initial + delta, 0, 1)

    # Test 1: self-extinguishing (mu converges toward tilde)
    tracker = ResidualTracker(mu_tilde, epsilon_fraction=0.05, n_consecutive=3)
    current_mu = mu_initial.copy()
    for i in range(40):
        current_mu = current_mu + 0.08 * (mu_tilde - current_mu)
        current_mu = np.clip(current_mu, 0, 1)
        tracker.record(current_mu, decision_index=i * 10)

    assert tracker.is_absorbed(), "Converging centroids should be absorbed"
    traj = tracker.decay_trajectory()
    assert traj[0] > traj[-1], "Norm should decrease over time"
    print(f"[PASS] Self-extinguishing: initial={traj[0]:.4f} -> final={traj[-1]:.4f}, absorbed={tracker.is_absorbed()}")

    # Test 2: persistent residual (mu diverges away from tilde)
    tracker2 = ResidualTracker(mu_tilde, epsilon_fraction=0.05, n_consecutive=3)
    current_mu2 = mu_initial.copy()
    for i in range(40):
        current_mu2 = current_mu2 + 0.08 * (mu_initial - current_mu2)
        tracker2.record(current_mu2, decision_index=i * 10)

    assert not tracker2.is_absorbed(), "Diverging centroids should NOT be absorbed"
    print(f"[PASS] Persistent residual: absorbed={tracker2.is_absorbed()}, final_norm={tracker2.decay_trajectory()[-1]:.4f}")

    print("\nresidual_tracker.py: all self-tests passed.")
