"""
Synthetic SOC alert generator for Experiment 1 (Scoring Matrix Convergence).

Generates alerts with characteristic factor profiles sampled from Beta
distributions, plus configurable noise that simulates ambiguous feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTOR_NAMES: list[str] = [
    "severity",
    "asset_criticality",
    "user_risk",
    "time_anomaly",
    "pattern_match",
    "context_richness",
]

ACTION_NAMES: list[str] = [
    "auto_close",
    "enrich_and_watch",
    "escalate_tier2",
    "escalate_incident",
]

# Each profile specifies per-factor Beta distribution means and the
# deterministic ground-truth action for non-noisy alerts.
# Beta parameters: alpha = mean * 5,  beta = (1 - mean) * 5
_ALERT_PROFILES: dict[str, dict] = {
    "false_positive": {
        "means": [0.20, 0.20, 0.15, 0.10, 0.70, 0.80],
        "action": "auto_close",
    },
    "routine_alert": {
        "means": [0.40, 0.40, 0.20, 0.20, 0.50, 0.50],
        "action": "enrich_and_watch",
    },
    "suspicious_login": {
        "means": [0.50, 0.50, 0.70, 0.70, 0.30, 0.30],
        "action": "escalate_tier2",
    },
    "data_exfil": {
        "means": [0.80, 0.80, 0.70, 0.50, 0.40, 0.30],
        "action": "escalate_incident",
    },
    "brute_force": {
        "means": [0.50, 0.40, 0.40, 0.80, 0.70, 0.50],
        "action": "escalate_tier2",
    },
    "insider_threat": {
        "means": [0.70, 0.75, 0.80, 0.50, 0.25, 0.70],
        "action": "escalate_incident",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single synthetic SOC alert."""

    alert_id: str
    alert_type: str
    factors: np.ndarray       # shape (6,), dtype float64, values in [0, 1]
    ground_truth_action: str
    is_noisy: bool


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AlertGenerator:
    """
    Generates synthetic SOC alert streams for Experiment 1.

    Parameters
    ----------
    config : dict | str | Path
        A dict containing experiment config keys (e.g. ``noise_rate``), or a
        path to a YAML file whose top-level ``experiment_1`` key is used.
    """

    def __init__(self, config: Union[dict, str, Path]) -> None:
        if isinstance(config, (str, Path)):
            with open(config) as fh:
                raw = yaml.safe_load(fh)
            config = raw.get("experiment_1", raw)

        self.noise_rate: float = float(config.get("noise_rate", 0.10))
        self.alert_types: list[str] = list(_ALERT_PROFILES.keys())
        self.action_names: list[str] = ACTION_NAMES

    def generate(self, n: int, seed: int) -> list[Alert]:
        """
        Generate *n* alerts reproducibly from the given seed.

        Alert types are sampled uniformly.  A ``noise_rate`` fraction of
        alerts receives a randomly-chosen *incorrect* action label to simulate
        ambiguous analyst feedback.

        Parameters
        ----------
        n : int
            Number of alerts to generate.
        seed : int
            NumPy random seed for full reproducibility.

        Returns
        -------
        list[Alert]
        """
        rng = np.random.default_rng(seed)

        # Draw type indices and noise decisions upfront (vectorised)
        type_indices = rng.integers(0, len(self.alert_types), size=n)
        is_noisy_mask = rng.random(n) < self.noise_rate

        alerts: list[Alert] = []
        for i in range(n):
            alert_type = self.alert_types[type_indices[i]]
            profile = _ALERT_PROFILES[alert_type]
            means = np.asarray(profile["means"], dtype=np.float64)
            true_action: str = profile["action"]

            # Beta(alpha = mean*5, beta = (1-mean)*5) keeps values in (0, 1)
            factors: np.ndarray = rng.beta(means * 5.0, (1.0 - means) * 5.0)

            is_noisy = bool(is_noisy_mask[i])
            if is_noisy:
                other_actions = [a for a in self.action_names if a != true_action]
                ground_truth_action = other_actions[
                    int(rng.integers(0, len(other_actions)))
                ]
            else:
                ground_truth_action = true_action

            alerts.append(Alert(
                alert_id=f"alert_{seed:04d}_{i:06d}",
                alert_type=alert_type,
                factors=factors,
                ground_truth_action=ground_truth_action,
                is_noisy=is_noisy,
            ))

        return alerts


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    config = {"noise_rate": 0.10}
    gen = AlertGenerator(config)
    alerts = gen.generate(1000, seed=42)
    n = len(alerts)

    # 1. Alert type distribution
    type_counts = Counter(a.alert_type for a in alerts)
    print("=== Alert type distribution (expected ~16.7% each) ===")
    for t in gen.alert_types:
        c = type_counts[t]
        bar = "#" * (c // 5)
        print(f"  {t:<20}  {c:4d}  ({c / n * 100:5.1f}%)  {bar}")

    # 2. Ground truth action distribution
    action_counts = Counter(a.ground_truth_action for a in alerts)
    print("\n=== Ground truth action distribution ===")
    for a in ACTION_NAMES:
        c = action_counts[a]
        print(f"  {a:<22}  {c:4d}  ({c / n * 100:5.1f}%)")

    # 3. Mean factor values per alert type: observed vs. profile
    short = [f[:6] for f in FACTOR_NAMES]
    header_cols = "  ".join(f"{'obs/exp':>9}")  # placeholder; built below
    col_heads = "  ".join(f"{s:<9}" for s in short)
    print(f"\n=== Mean factor values per alert type (observed vs profile) ===")
    print(f"  {'type':<20}  {col_heads}")
    for alert_type, profile in _ALERT_PROFILES.items():
        type_alerts = [a for a in alerts if a.alert_type == alert_type]
        if not type_alerts:
            continue
        obs = np.mean([a.factors for a in type_alerts], axis=0)
        exp = np.asarray(profile["means"])
        row = "  ".join(f"{o:.2f}({e:.2f})" for o, e in zip(obs, exp))
        print(f"  {alert_type:<20}  {row}")

    # 4. Noise rate
    noisy_count = sum(1 for a in alerts if a.is_noisy)
    obs_noise = noisy_count / n
    print(f"\n=== Noise rate ===")
    print(f"  Observed: {obs_noise:.3f}   Expected: ~0.100")

    # --- Sanity checks ---
    ok = True

    # Each alert type should appear in roughly [10%, 25%] of samples
    lo_type, hi_type = 0.10, 0.25
    for t, c in type_counts.items():
        frac = c / n
        if not (lo_type <= frac <= hi_type):
            print(f"FAIL: type '{t}' fraction {frac:.3f} outside [{lo_type}, {hi_type}]")
            ok = False

    # Noise rate should be within 4 pp of target
    if not (0.06 <= obs_noise <= 0.14):
        print(f"FAIL: noise rate {obs_noise:.3f} outside [0.06, 0.14]")
        ok = False

    # Observed factor means should be within 0.10 of the profile means
    for alert_type, profile in _ALERT_PROFILES.items():
        type_alerts = [a for a in alerts if a.alert_type == alert_type]
        if not type_alerts:
            continue
        obs = np.mean([a.factors for a in type_alerts], axis=0)
        exp = np.asarray(profile["means"])
        max_err = float(np.max(np.abs(obs - exp)))
        if max_err > 0.10:
            print(
                f"FAIL: type '{alert_type}' max factor error {max_err:.3f} > 0.10"
            )
            ok = False

    if ok:
        print("\nAll checks passed")
