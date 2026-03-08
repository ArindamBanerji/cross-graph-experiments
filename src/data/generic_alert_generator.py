"""
GenericAlertGenerator — parameterized alert generator for any (C, A, d).

CategoryAlertGenerator hardcodes C=5, A=4, d=6 for SOC.
This generator is needed for:
  GE1: d sweep (3, 6, 12, 24, 50)
  GE2: (C, A) sweep
  OP6A: second domain (S2P-like)

Profiles are orthogonal by construction: action a's primary factor region
is (a * stride) % d where stride = d // A. Guaranteed separability for d >= A.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class GenericAlert:
    alert_id:        int
    category_index:  int
    factors:         np.ndarray   # shape: (n_factors,), values in [0,1]
    gt_action_index: int
    is_noisy:        bool = False


class GenericAlertGenerator:
    """
    Parameterized alert generator for any (C, A, d).

    Args:
        n_categories:    Number of alert categories C.
        n_actions:       Number of possible actions A.
        n_factors:       Number of context factors d. Must be >= n_actions.
        seed:            Random seed.
        factor_sigma:    Factor noise around profile mean (default 0.10).
        category_noise:  Profile variation across categories (default 0.05).
        noise_rate:      Fraction of alerts with flipped GT labels (default 0.0).
        gt_distribution: Per-category action probability distribution (C, A).
                         Default: uniform.
        profile_high:    Primary factor value in profile (default 0.85).
        profile_low:     Non-primary factor value (default 0.15).
    """

    def __init__(
        self,
        n_categories:   int,
        n_actions:      int,
        n_factors:      int,
        seed:           int            = 42,
        factor_sigma:   float          = 0.10,
        category_noise: float          = 0.05,
        noise_rate:     float          = 0.0,
        gt_distribution: np.ndarray | None = None,
        profile_high:   float          = 0.85,
        profile_low:    float          = 0.15,
    ):
        assert n_factors >= n_actions, (
            f"n_factors ({n_factors}) must be >= n_actions ({n_actions})"
        )
        self.C            = n_categories
        self.A            = n_actions
        self.d            = n_factors
        self.factor_sigma = factor_sigma
        self.noise_rate   = noise_rate
        self.rng          = np.random.default_rng(seed)

        self._profiles = self._build_profiles(category_noise, profile_high, profile_low)

        if gt_distribution is not None:
            assert gt_distribution.shape == (n_categories, n_actions)
            self._gt_dist = gt_distribution
        else:
            self._gt_dist = np.ones((n_categories, n_actions)) / n_actions

        self._alert_counter = 0

    def _build_profiles(
        self,
        category_noise: float,
        profile_high:   float,
        profile_low:    float,
    ) -> np.ndarray:
        profiles = np.full((self.C, self.A, self.d), profile_low)
        stride   = max(1, self.d // self.A)
        for a in range(self.A):
            start = (a * stride) % self.d
            end   = min(start + stride, self.d)
            profiles[:, a, start:end] = profile_high
        noise    = self.rng.normal(0, category_noise, size=(self.C, self.A, self.d))
        profiles = np.clip(profiles + noise, 0.0, 1.0)
        return profiles

    def get_profiles(self) -> np.ndarray:
        """Return profile centroid tensor (C, A, d). Use as ProfileScorer init."""
        return self._profiles.copy()

    def generate(self, n: int) -> list[GenericAlert]:
        alerts = []
        for _ in range(n):
            cat       = int(self.rng.integers(0, self.C))
            gt_action = int(self.rng.choice(self.A, p=self._gt_dist[cat]))
            mean      = self._profiles[cat, gt_action, :]
            factors   = np.clip(self.rng.normal(mean, self.factor_sigma), 0.0, 1.0)
            is_noisy  = False
            if self.noise_rate > 0 and self.rng.random() < self.noise_rate:
                gt_action = int(self.rng.integers(0, self.A))
                is_noisy  = True
            alerts.append(GenericAlert(
                alert_id=self._alert_counter,
                category_index=cat,
                factors=factors,
                gt_action_index=gt_action,
                is_noisy=is_noisy,
            ))
            self._alert_counter += 1
        return alerts

    @property
    def categories(self) -> list[str]:  return [f"category_{c}" for c in range(self.C)]
    @property
    def actions(self)    -> list[str]:  return [f"action_{a}"   for a in range(self.A)]
    @property
    def factors(self)    -> list[str]:  return [f"factor_{i}"   for i in range(self.d)]


if __name__ == "__main__":
    gen = GenericAlertGenerator(n_categories=5, n_actions=4, n_factors=6, seed=42)
    alerts = gen.generate(100)
    assert len(alerts) == 100
    assert all(0 <= a.category_index < 5 for a in alerts)
    assert all(a.factors.shape == (6,) for a in alerts)
    assert all(0.0 <= a.factors.min() and a.factors.max() <= 1.0 for a in alerts)
    print("[PASS] Basic generation: C=5, A=4, d=6")

    gen_hd = GenericAlertGenerator(n_categories=5, n_actions=4, n_factors=50, seed=42)
    alerts_hd = gen_hd.generate(200)
    assert all(a.factors.shape == (50,) for a in alerts_hd)
    print("[PASS] High-d generation: d=50")

    profiles = gen.get_profiles()
    assert profiles.shape == (5, 4, 6)
    stride = max(1, 6 // 4)
    for a in range(4):
        start        = (a * stride) % 6
        primary_mean = profiles[:, a, start:start+stride].mean()
        other_idx    = [i for i in range(6) if i < start or i >= start + stride]
        if other_idx:
            other_mean = profiles[:, a, other_idx].mean()
            assert primary_mean > other_mean, (
                f"Action {a}: primary {primary_mean:.3f} should exceed other {other_mean:.3f}"
            )
    print("[PASS] Profile orthogonality")

    gen_a = GenericAlertGenerator(n_categories=3, n_actions=3, n_factors=6, seed=99)
    gen_b = GenericAlertGenerator(n_categories=3, n_actions=3, n_factors=6, seed=99)
    for a, b in zip(gen_a.generate(50), gen_b.generate(50)):
        assert np.allclose(a.factors, b.factors)
    print("[PASS] Reproducibility: same seed -> identical alerts")

    print("\ngeneric_alert_generator.py: all self-tests passed.")
