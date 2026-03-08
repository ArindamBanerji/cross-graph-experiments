"""
synthesis.py — Mutable SynthesisBias for SYNTH-EXP-0 infrastructure.

Eq. 4-synthesis: P(a|f,c,sigma) = softmax(-(||f-mu||^2 + lambda*sigma[c,a]) / tau)

Sign convention:
    sigma[c,a] < 0 -> action a MORE likely for category c (smaller effective distance)
    sigma[c,a] > 0 -> action a LESS likely for category c (larger effective distance)

lambda=0 is the kill switch — exact Eq. 4-final behavior restored.

CRITICAL: sigma is used ONLY in score(). It is NEVER passed to update().
          Centroids (mu) learn from experience. sigma encodes current awareness.
          Loop 2 firewall: centroid updates never see sigma.
"""
from __future__ import annotations

import numpy as np


class SynthesisBias:
    """
    Mutable synthesis bias tensor sigma[n_categories, n_actions].

    lambda=0 -> exact Eq. 4-final behavior (no regression).
    mu (centroids) are NEVER updated using sigma.  Loop 2 firewall preserved.
    """

    def __init__(self, n_categories: int, n_actions: int):
        self.sigma = np.zeros((n_categories, n_actions), dtype=np.float32)
        self.n_categories = n_categories
        self.n_actions = n_actions

    def set(self, category_idx: int, action_idx: int, value: float) -> None:
        """Clip to [-1.0, 1.0] — sigma is a bias, not an unbounded weight."""
        self.sigma[category_idx, action_idx] = np.clip(value, -1.0, 1.0)

    def get(self, category_idx: int, action_idx: int) -> float:
        return float(self.sigma[category_idx, action_idx])

    def tensor(self) -> np.ndarray:
        """Return a copy of the sigma matrix."""
        return self.sigma.copy()

    def reset(self) -> None:
        """Zero out all sigma values."""
        self.sigma[:] = 0.0

    def __repr__(self) -> str:
        nz = int(np.count_nonzero(self.sigma))
        return (f"SynthesisBias(n_categories={self.n_categories}, "
                f"n_actions={self.n_actions}, non_zero={nz})")


if __name__ == "__main__":
    b = SynthesisBias(6, 4)
    assert b.sigma.shape == (6, 4)
    assert b.get(0, 0) == 0.0

    b.set(1, 2, 0.5)
    assert abs(b.get(1, 2) - 0.5) < 1e-6

    # Clip enforcement
    b.set(0, 0, 2.5)
    assert b.get(0, 0) == 1.0
    b.set(0, 1, -2.5)
    assert b.get(0, 1) == -1.0

    t = b.tensor()
    assert isinstance(t, np.ndarray)
    assert t.shape == (6, 4)
    assert np.count_nonzero(t) == 3

    b.reset()
    assert np.count_nonzero(b.sigma) == 0

    print("SynthesisBias OK")
    print(b)
