"""
realistic_generator.py — Non-centroidal alert generator for FX-1-PROXY-REAL.
experiments/fx1_proxy_real/realistic_generator.py

Uses canonical SOC taxonomy (synthesis series) with SOC_CATEGORIES / SOC_ACTIONS /
SOC_FACTORS, contrasting with bridge_common which uses a different set of names.

Five distribution modes:
  centroidal  — Gaussian sigma=0.15 around GT centroids (baseline reference)
  heavy_tail  — Beta distributions with heavier tails than Gaussian
  correlated  — Factor correlations matching SOC domain knowledge
  overlapping — Reduced inter-class separation (pulls factors toward 0.5)
  combined    — All three simultaneously (most realistic)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Canonical SOC taxonomy  (synthesis-series names)
# ---------------------------------------------------------------------------

SOC_CATEGORIES: list[str] = [
    "travel_anomaly",
    "credential_access",
    "threat_intel_match",
    "insider_behavioral",
    "cloud_infrastructure",
]

SOC_ACTIONS: list[str] = [
    "escalate",
    "investigate",
    "suppress",
    "monitor",
]

SOC_FACTORS: list[str] = [
    "travel_match",       # index 0
    "asset_criticality",  # index 1
    "threat_intel",       # index 2
    "pattern_history",    # index 3
    "time_anomaly",       # index 4
    "device_trust",       # index 5
]

# ---------------------------------------------------------------------------
# SOC profile centroids (hardcoded for self-contained experiment)
#
# Design principles:
#   escalate / investigate  → high threat_intel(2) + asset_criticality(1)
#                             + time_anomaly(4) + travel_match(0) where relevant;
#                             low  device_trust(5) + pattern_history(3)
#   suppress / monitor      → high device_trust(5) + pattern_history(3);
#                             low  threat_intel(2) + time_anomaly(4)
#
# Each action within a category has ≥2 dominant factors at 0.75-0.90 that
# differ from all other actions → good L2 separability under centroidal data.
#
# Actions order: [escalate, investigate, suppress, monitor]
# Factors order: [travel_match(0), asset_criticality(1), threat_intel(2),
#                 pattern_history(3), time_anomaly(4), device_trust(5)]
# ---------------------------------------------------------------------------

SOC_PROFILES: Dict[str, Dict[str, list]] = {
    "travel_anomaly": {
        # Primary signals: travel_match(0) + time_anomaly(4)
        "escalate":    [0.85, 0.30, 0.25, 0.10, 0.80, 0.10],
        "investigate": [0.75, 0.25, 0.20, 0.35, 0.65, 0.30],
        "suppress":    [0.15, 0.20, 0.10, 0.85, 0.15, 0.85],
        "monitor":     [0.45, 0.20, 0.15, 0.60, 0.40, 0.60],
    },
    "credential_access": {
        # Primary signals: asset_criticality(1) + threat_intel(2) + time_anomaly(4)
        "escalate":    [0.20, 0.85, 0.80, 0.10, 0.70, 0.10],
        "investigate": [0.25, 0.70, 0.65, 0.35, 0.55, 0.30],
        "suppress":    [0.20, 0.20, 0.15, 0.80, 0.20, 0.85],
        "monitor":     [0.20, 0.30, 0.35, 0.60, 0.30, 0.65],
    },
    "threat_intel_match": {
        # Primary signals: threat_intel(2) + asset_criticality(1) + time_anomaly(4)
        "escalate":    [0.25, 0.80, 0.90, 0.10, 0.75, 0.10],
        "investigate": [0.20, 0.55, 0.85, 0.25, 0.55, 0.20],
        "suppress":    [0.15, 0.20, 0.15, 0.85, 0.15, 0.85],
        "monitor":     [0.20, 0.40, 0.55, 0.55, 0.35, 0.55],
    },
    "insider_behavioral": {
        # Primary signals: pattern_history(3) + time_anomaly(4) for escalate;
        #                  device_trust(5) dominant for suppress
        "escalate":    [0.30, 0.75, 0.70, 0.85, 0.80, 0.15],
        "investigate": [0.25, 0.55, 0.50, 0.70, 0.65, 0.30],
        "suppress":    [0.20, 0.20, 0.15, 0.20, 0.20, 0.85],
        "monitor":     [0.25, 0.30, 0.25, 0.45, 0.40, 0.65],
    },
    "cloud_infrastructure": {
        # Primary signals: asset_criticality(1) + threat_intel(2) + time_anomaly(4)
        "escalate":    [0.25, 0.85, 0.85, 0.10, 0.70, 0.10],
        "investigate": [0.20, 0.65, 0.75, 0.25, 0.55, 0.25],
        "suppress":    [0.15, 0.25, 0.15, 0.80, 0.20, 0.85],
        "monitor":     [0.20, 0.45, 0.50, 0.55, 0.35, 0.60],
    },
}

# GT action distributions: P(action | category) = [escalate, investigate, suppress, monitor]
# Most SOC alerts are lower-severity; suppress + monitor dominate.
SOC_GT_DISTRIBUTIONS: Dict[str, list] = {
    "travel_anomaly":       [0.05, 0.15, 0.55, 0.25],
    "credential_access":    [0.10, 0.20, 0.45, 0.25],
    "threat_intel_match":   [0.20, 0.30, 0.25, 0.25],
    "insider_behavioral":   [0.10, 0.30, 0.30, 0.30],
    "cloud_infrastructure": [0.10, 0.25, 0.35, 0.30],
}


# ---------------------------------------------------------------------------
# SOCDomainConfig — returns tensors from SOC_PROFILES
# ---------------------------------------------------------------------------

class SOCDomainConfig:
    """
    Returns GT profile centroids and action distributions as numpy tensors
    for the canonical SOC taxonomy (synthesis-series names).

    Data is defined inline in SOC_PROFILES / SOC_GT_DISTRIBUTIONS above —
    self-contained, no config file dependency.
    """

    @staticmethod
    def get_profile_centroids() -> np.ndarray:
        """
        Return GT centroid tensor, shape (5, 4, 6).
        mu[category_idx, action_idx, factor_idx] = centroid value in [0, 1].
        """
        n_cats  = len(SOC_CATEGORIES)
        n_acts  = len(SOC_ACTIONS)
        n_facts = len(SOC_FACTORS)
        mu = np.zeros((n_cats, n_acts, n_facts), dtype=np.float64)
        for i, cat in enumerate(SOC_CATEGORIES):
            for j, act in enumerate(SOC_ACTIONS):
                mu[i, j, :] = np.array(SOC_PROFILES[cat][act], dtype=np.float64)
        return mu

    @staticmethod
    def get_gt_distributions() -> np.ndarray:
        """
        Return GT action probability distributions, shape (5, 4).
        gt_dists[category_idx, action_idx] = P(action | category).
        """
        dists = np.zeros((len(SOC_CATEGORIES), len(SOC_ACTIONS)), dtype=np.float64)
        for i, cat in enumerate(SOC_CATEGORIES):
            dists[i, :] = np.array(SOC_GT_DISTRIBUTIONS[cat], dtype=np.float64)
        return dists


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class RealisticAlert:
    """A single alert from RealisticAlertGenerator."""
    factors:         np.ndarray  # shape (6,), values in [0, 1]
    category_index:  int         # index into SOC_CATEGORIES
    gt_action_index: int         # index into SOC_ACTIONS


# ---------------------------------------------------------------------------
# RealisticAlertGenerator
# ---------------------------------------------------------------------------

class RealisticAlertGenerator:
    """
    Generates alerts with a configurable factor distribution for FX-1-PROXY-REAL.

    Five modes:
      centroidal  — Gaussian sigma=0.15 around GT centroids (reference baseline)
      heavy_tail  — Beta(alpha, beta) with k=3 (moderate heavy tails)
      correlated  — Factor correlations from SOC domain knowledge
      overlapping — 25% pull of factors toward 0.5 (reduced class separation)
      combined    — All three realistic effects simultaneously

    Always uses the canonical SOC taxonomy (SOC_CATEGORIES / SOC_ACTIONS /
    SOC_FACTORS) and the hardcoded SOC_PROFILES.

    GT action selection: sampled from SOC_GT_DISTRIBUTIONS[category] using the
    instance RNG.  This produces a realistic stream where different alert instances
    from the same category have different correct actions (probabilistic oracle),
    rather than fixing one dominant action per category.

    Parameters
    ----------
    mode : str
        Distribution mode. One of centroidal, heavy_tail, correlated,
        overlapping, combined.
    seed : int
        Master RNG seed for full reproducibility.
    """

    # Factor correlations (SOC domain knowledge).
    # key: (factor_i_name, factor_j_name), value: correlation_strength
    # Positive s: fi above centroid → nudge fj up (same direction).
    # Negative s: fi above centroid → nudge fj down (opposite direction).
    # All names must be present in SOC_FACTORS.
    FACTOR_CORRELATIONS: Dict[Tuple[str, str], float] = {
        ("threat_intel",    "asset_criticality"): 0.65,   # high-value assets attract threats
        ("time_anomaly",    "travel_match"):       0.55,   # travel causes time anomalies
        ("pattern_history", "device_trust"):       0.60,   # known devices have clean history
        ("asset_criticality", "time_anomaly"):     0.40,   # critical assets monitored tightly
        ("threat_intel",    "pattern_history"):   -0.35,   # new threats have no history
    }

    # Missing value rates per factor (fraction of alerts with missing data)
    MISSING_VALUE_RATES: Dict[str, float] = {
        "travel_match":       0.15,   # 15%: no geo data
        "device_trust":       0.10,   # 10%: unknown devices
        "threat_intel":       0.20,   # 20%: no TI match
        "pattern_history":    0.05,   # usually available
        "asset_criticality":  0.02,   # almost always known
        "time_anomaly":       0.03,   # almost always computable
    }

    _VALID_MODES = ("centroidal", "heavy_tail", "correlated", "overlapping", "combined")

    def __init__(self, mode: str = "combined", seed: int = 42) -> None:
        assert mode in self._VALID_MODES, (
            f"Unknown mode: {mode!r}. Must be one of {self._VALID_MODES}"
        )
        # Enforce correct domain taxonomy
        assert SOC_CATEGORIES == [
            "travel_anomaly", "credential_access", "threat_intel_match",
            "insider_behavioral", "cloud_infrastructure",
        ], f"Category mismatch: {SOC_CATEGORIES}"
        assert SOC_FACTORS == [
            "travel_match", "asset_criticality", "threat_intel",
            "pattern_history", "time_anomaly", "device_trust",
        ], f"Factor mismatch: {SOC_FACTORS}"

        # Verify all FACTOR_CORRELATIONS keys are valid SOC_FACTORS names
        for (fi, fj) in self.FACTOR_CORRELATIONS:
            assert fi in SOC_FACTORS, f"Correlation factor {fi!r} not in SOC_FACTORS"
            assert fj in SOC_FACTORS, f"Correlation factor {fj!r} not in SOC_FACTORS"

        self.mode        = mode
        self.rng         = np.random.default_rng(seed)
        self.categories  = SOC_CATEGORIES
        self.gt_profiles = SOCDomainConfig.get_profile_centroids()   # (5, 4, 6)
        self.gt_dists    = SOCDomainConfig.get_gt_distributions()     # (5, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_alerts: int) -> list[RealisticAlert]:
        """Generate n_alerts reproducibly using this generator's mode."""
        n_cats = len(SOC_CATEGORIES)
        cat_indices = self.rng.integers(0, n_cats, size=n_alerts)
        alerts = []
        for i in range(n_alerts):
            cat       = int(cat_indices[i])
            gt_action = int(self.rng.choice(len(SOC_ACTIONS), p=self.gt_dists[cat]))
            f         = self._sample_factors(cat, gt_action)
            alerts.append(RealisticAlert(
                factors=f,
                category_index=cat,
                gt_action_index=gt_action,
            ))
        return alerts

    def get_profiles(self) -> np.ndarray:
        """Return GT centroid tensor, shape (5, 4, 6)."""
        return self.gt_profiles.copy()

    # ------------------------------------------------------------------
    # Factor sampling pipeline
    # ------------------------------------------------------------------

    def _sample_factors(self, cat: int, gt_action: int) -> np.ndarray:
        """Sample one factor vector for (category, gt_action) using self.mode."""
        centroid = self.gt_profiles[cat, gt_action, :].copy()

        if self.mode == "centroidal":
            # Baseline: tight Gaussian around centroid, matching CategoryAlertGenerator
            f = self.rng.normal(loc=centroid, scale=0.15)

        elif self.mode == "heavy_tail":
            f = self._sample_heavy_tail(centroid)
            f = self._apply_missing(f)

        elif self.mode == "correlated":
            f = self._sample_heavy_tail(centroid)
            f = self._apply_correlations(f, centroid)
            f = self._apply_missing(f)

        elif self.mode == "overlapping":
            f = self._sample_heavy_tail(centroid)
            f = self._apply_overlap(f, centroid)
            f = self._apply_missing(f)

        else:  # combined
            f = self._sample_heavy_tail(centroid)
            f = self._apply_correlations(f, centroid)
            f = self._apply_overlap(f, centroid)
            f = self._apply_missing(f)

        return np.clip(f, 0.0, 1.0)

    def _sample_heavy_tail(self, centroid: np.ndarray) -> np.ndarray:
        """
        Beta(alpha, beta) with k=3 → moderate heavy tails.

        Mean = centroid. Lower k = heavier tails.  k=3 vs Gaussian sigma=0.15:
        variance is ~2× larger in the tails, simulating real SOC data spread.

        centroid clipped to [0.05, 0.95] to avoid degenerate Beta parameters.
        """
        k = 3.0
        f = np.zeros(len(centroid), dtype=np.float64)
        for i, c in enumerate(centroid):
            c_safe = float(np.clip(c, 0.05, 0.95))
            alpha  = 2.0 * c_safe * k
            beta_  = 2.0 * (1.0 - c_safe) * k
            f[i]   = float(self.rng.beta(alpha, beta_))
        return f

    def _apply_correlations(
        self,
        f:        np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """
        Nudge factors based on SOC domain correlations.

        For each (fi, fj) pair with strength s:
            delta = (f[i] - centroid[i]) * s * 0.5
            f[j] += delta   (clipped to [0, 1])

        Uses the pre-deviation from centroid so nudges are proportional
        to how far the sample already moved — amplifies realistic co-movement.
        """
        f = f.copy()
        for (fi_name, fj_name), strength in self.FACTOR_CORRELATIONS.items():
            i     = SOC_FACTORS.index(fi_name)
            j     = SOC_FACTORS.index(fj_name)
            delta = (f[i] - centroid[i]) * strength * 0.5
            f[j]  = float(np.clip(f[j] + delta, 0.0, 1.0))
        return f

    def _apply_overlap(
        self,
        f:        np.ndarray,
        centroid: np.ndarray,  # API symmetry; not used in current impl
    ) -> np.ndarray:
        """
        Pull 25% of each factor value toward 0.5 (ambiguity injection).

        Reduces inter-class separation, simulating alerts that are genuinely
        ambiguous between escalate/investigate or suppress/monitor.
        """
        overlap_factor = 0.25
        f = f + overlap_factor * (0.5 - f)
        return np.clip(f, 0.0, 1.0)

    def _apply_missing(self, f: np.ndarray) -> np.ndarray:
        """
        Stochastically replace factor values with 0.5 (uninformative prior).

        Simulates missing observability: no geo data, unknown device, no TI feed.
        0.5 is equidistant from all action centroids — adds uncertainty without bias.
        """
        f = f.copy()
        for i, factor_name in enumerate(SOC_FACTORS):
            rate = self.MISSING_VALUE_RATES.get(factor_name, 0.0)
            if float(self.rng.random()) < rate:
                f[i] = 0.5
        return f
