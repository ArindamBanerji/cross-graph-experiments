"""
Category-aware SOC alert generator for Bridge Layer Experiments (EXP 5-9).

ACTION-CONDITIONAL DESIGN: each alert's factor profile is conditioned on both
the alert category AND the ground-truth action.  A credential_access alert
whose correct action is auto_close has a DIFFERENT factor signature from one
whose correct action is escalate_incident.  This raises the theoretical
accuracy ceiling from ~46% (category-only profiles) to ~80%.

Compared to the original category-only design:
  - Factors are sampled from N(μ[category][gt_action], factor_sigma)
  - GT distributions are probabilistic (soft), not deterministic
  - All profile data comes from configs/default.yaml — nothing is hardcoded here
  - Noise flips the GT label but NOT the factors (factors reflect the original
    GT action, modelling a mis-labelled oracle rather than a changed incident)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Module-level name lists (structural constants, no profile data)
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "credential_access",
    "threat_intel_match",
    "lateral_movement",
    "data_exfiltration",
    "insider_threat",
]

ACTIONS: list[str] = [
    "auto_close",
    "escalate_tier2",
    "enrich_and_watch",
    "escalate_incident",
]

FACTORS: list[str] = [
    "travel_match",
    "asset_criticality",
    "threat_intel",
    "time_anomaly",
    "device_trust",
    "pattern_history",
]

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CategoryAlert:
    """A single synthetic SOC alert for bridge layer experiments."""

    alert_id: str
    category: str             # one of CATEGORIES
    category_index: int       # index into CATEGORIES
    factors: np.ndarray       # shape (6,), dtype float64, values clipped to [0, 1]
    ground_truth_action: str  # GT action stored on the alert (may be noise-flipped)
    gt_action_index: int      # index into ACTIONS (may be noise-flipped)
    gt_action_name: str       # original GT action that determined the factor profile
    is_noisy: bool            # True if GT label was flipped by noise


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CategoryAlertGenerator:
    """
    Generates synthetic SOC alert streams for Bridge Layer Experiments (EXP 5-9).

    ACTION-CONDITIONAL: each alert's factor vector is sampled from
    N(μ[category][gt_action], factor_sigma), giving the scoring matrix
    enough information to learn the correct action even when categories share
    overlapping action sets.

    Parameters
    ----------
    categories : list[str]
        Ordered list of category names (length 5).
    actions : list[str]
        Ordered list of action names (length 4).
    factors : list[str]
        Ordered list of factor names (length 6).
    action_conditional_profiles : dict[str, dict[str, list[float]]]
        Nested dict: profiles[category][action] → list of 6 factor means.
    gt_distributions : dict[str, list[float]]
        Per-category probability distributions over ACTIONS (length-4 vectors).
    factor_sigma : float
        Shared standard deviation for all factor dimensions.  Default: 0.15.
    noise_rate : float
        Fraction of alerts whose GT action label is corrupted.  Default: 0.0.
        Noise flips the label only — factors still reflect the original GT action.
    seed : int
        Master seed for the internal RNG.  Default: 42.
    """

    def __init__(
        self,
        categories: Optional[list[str]] = None,
        actions: Optional[list[str]] = None,
        factors: Optional[list[str]] = None,
        action_conditional_profiles: Optional[dict[str, dict[str, list[float]]]] = None,
        gt_distributions: Optional[dict[str, list[float]]] = None,
        factor_sigma: float = 0.15,
        noise_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        # When profile data is omitted, load from the project config file.
        # This preserves backward compatibility with callers that only pass
        # noise_rate and seed (e.g. oracle.py, gating.py self-tests).
        if action_conditional_profiles is None or gt_distributions is None:
            with open(_CONFIG_PATH) as fh:
                bc = yaml.safe_load(fh)["bridge_common"]
            categories = categories or bc["categories"]
            actions = actions or bc["actions"]
            factors = factors or bc["factors"]
            action_conditional_profiles = (
                action_conditional_profiles or bc["action_conditional_profiles"]
            )
            gt_distributions = gt_distributions or bc["category_gt_distributions"]

        self.categories = list(categories)
        self.actions = list(actions)
        self.factors = list(factors)
        self.factor_sigma = float(factor_sigma)
        self.noise_rate = float(noise_rate)
        self.seed = seed

        # Pre-convert profiles to numpy arrays
        self.profiles: dict[str, dict[str, np.ndarray]] = {
            cat: {
                act: np.asarray(action_conditional_profiles[cat][act], dtype=np.float64)
                for act in self.actions
            }
            for cat in self.categories
        }
        self.gt_distributions: dict[str, np.ndarray] = {
            cat: np.asarray(gt_distributions[cat], dtype=np.float64)
            for cat in self.categories
        }
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n: int,
        category_weights: Optional[list[float]] = None,
    ) -> list[CategoryAlert]:
        """
        Generate *n* alerts reproducibly, sampling categories according to
        ``category_weights``.

        Parameters
        ----------
        n : int
            Total number of alerts to generate.
        category_weights : list[float] | None
            Per-category sampling probabilities (length = len(categories)).
            If None, categories are sampled uniformly.  Need not sum to 1.

        Returns
        -------
        list[CategoryAlert]
            Alerts in generation order.
        """
        rng = self._rng
        categories = self.categories
        actions = self.actions

        if category_weights is None:
            probs = np.ones(len(categories)) / len(categories)
        else:
            w = np.asarray(category_weights, dtype=np.float64)
            probs = w / w.sum()

        # Draw all category indices and noise decisions upfront (vectorised)
        cat_indices = rng.choice(len(categories), size=n, p=probs)
        is_noisy_mask = rng.random(n) < self.noise_rate

        alerts: list[CategoryAlert] = []
        for i in range(n):
            cat_idx = int(cat_indices[i])
            category = categories[cat_idx]

            # Sample GT action from category distribution
            gt_dist = self.gt_distributions[category]
            original_gt_idx = int(rng.choice(len(actions), p=gt_dist))
            original_gt_action = actions[original_gt_idx]

            # Sample factors from action-conditional profile
            mu = self.profiles[category][original_gt_action]
            factors = rng.normal(loc=mu, scale=self.factor_sigma).clip(0.0, 1.0)

            # Optionally corrupt GT label (factors are unchanged)
            is_noisy = bool(is_noisy_mask[i])
            stored_gt_idx = original_gt_idx
            if is_noisy:
                other_indices = [j for j in range(len(actions)) if j != original_gt_idx]
                stored_gt_idx = int(other_indices[rng.integers(0, len(other_indices))])

            alerts.append(CategoryAlert(
                alert_id=f"cat_alert_{self.seed:04d}_{i:06d}",
                category=category,
                category_index=cat_idx,
                factors=factors,
                ground_truth_action=actions[stored_gt_idx],
                gt_action_index=stored_gt_idx,
                gt_action_name=original_gt_action,
                is_noisy=is_noisy,
            ))

        return alerts

    def generate_batch(self, n_per_category: int) -> list[CategoryAlert]:
        """
        Generate exactly *n_per_category* alerts for each category (balanced).

        Unlike ``generate()``, this guarantees exact per-category counts by
        iterating categories sequentially.

        Parameters
        ----------
        n_per_category : int
            Number of alerts to generate per category.

        Returns
        -------
        list[CategoryAlert]
            All len(categories) × n_per_category alerts, ordered by category.
        """
        rng = self._rng
        categories = self.categories
        actions = self.actions
        alerts: list[CategoryAlert] = []
        global_i = 0

        for cat_idx, category in enumerate(categories):
            gt_dist = self.gt_distributions[category]
            is_noisy_mask = rng.random(n_per_category) < self.noise_rate

            for j in range(n_per_category):
                # Sample GT action
                original_gt_idx = int(rng.choice(len(actions), p=gt_dist))
                original_gt_action = actions[original_gt_idx]

                # Sample factors from action-conditional profile
                mu = self.profiles[category][original_gt_action]
                factors = rng.normal(loc=mu, scale=self.factor_sigma).clip(0.0, 1.0)

                # Optionally corrupt GT label (factors unchanged)
                is_noisy = bool(is_noisy_mask[j])
                stored_gt_idx = original_gt_idx
                if is_noisy:
                    other_indices = [k for k in range(len(actions)) if k != original_gt_idx]
                    stored_gt_idx = int(other_indices[rng.integers(0, len(other_indices))])

                alerts.append(CategoryAlert(
                    alert_id=f"cat_alert_{self.seed:04d}_{global_i:06d}",
                    category=category,
                    category_index=cat_idx,
                    factors=factors,
                    ground_truth_action=actions[stored_gt_idx],
                    gt_action_index=stored_gt_idx,
                    gt_action_name=original_gt_action,
                    is_noisy=is_noisy,
                ))
                global_i += 1

        return alerts

    def generate_alerts(
        self,
        n: int,
        category_weights: Optional[list[float]] = None,
    ) -> list[CategoryAlert]:
        """Alias for generate() for backward compatibility."""
        return self.generate(n, category_weights=category_weights)

    def generate_campaign(self, n: int, campaign: dict) -> list[CategoryAlert]:
        """
        Generate alerts where GT action probabilities shift for specified categories.

        Parameters
        ----------
        n : int
            Total number of alerts to generate.
        campaign : dict
            Mapping category_name -> {action_name: probability}.
            Example::

                {
                    "credential_access":  {"escalate_incident": 0.80},
                    "threat_intel_match": {"escalate_incident": 0.75},
                }

            For categories NOT in campaign: uses normal GT distribution.
            For categories IN campaign: the named action gets the given probability;
            remaining probability is split equally among all other actions.

        Returns
        -------
        list[CategoryAlert]
            Alerts with the same fields as generate().  Uses a deterministic
            RNG seeded from self.seed + 10000 so results are independent of
            how many times generate() has been called.
        """
        # Build overridden GT distributions
        overridden_gt: dict[str, np.ndarray] = {}
        for cat in self.categories:
            if cat in campaign:
                overrides = campaign[cat]
                dist = np.zeros(len(self.actions), dtype=np.float64)
                override_indices: set[int] = set()
                total_override = 0.0
                for act_name, prob in overrides.items():
                    a_idx = self.actions.index(act_name)
                    dist[a_idx] = float(prob)
                    override_indices.add(a_idx)
                    total_override += float(prob)
                remaining = 1.0 - total_override
                non_override = [j for j in range(len(self.actions)) if j not in override_indices]
                if non_override:
                    per_act = remaining / len(non_override)
                    for j in non_override:
                        dist[j] = per_act
                dist = dist / dist.sum()  # normalize for floating-point safety
                overridden_gt[cat] = dist
            else:
                overridden_gt[cat] = self.gt_distributions[cat]

        # Separate RNG — does not consume the main generator's state
        rng = np.random.default_rng(self.seed + 10000)
        categories = self.categories
        actions = self.actions

        probs = np.ones(len(categories)) / len(categories)
        cat_indices = rng.choice(len(categories), size=n, p=probs)
        is_noisy_mask = rng.random(n) < self.noise_rate

        alerts: list[CategoryAlert] = []
        for i in range(n):
            cat_idx = int(cat_indices[i])
            category = categories[cat_idx]

            gt_dist = overridden_gt[category]
            original_gt_idx = int(rng.choice(len(actions), p=gt_dist))
            original_gt_action = actions[original_gt_idx]

            mu = self.profiles[category][original_gt_action]
            factors = rng.normal(loc=mu, scale=self.factor_sigma).clip(0.0, 1.0)

            is_noisy = bool(is_noisy_mask[i])
            stored_gt_idx = original_gt_idx
            if is_noisy:
                other_indices = [j for j in range(len(actions)) if j != original_gt_idx]
                stored_gt_idx = int(other_indices[rng.integers(0, len(other_indices))])

            alerts.append(CategoryAlert(
                alert_id=f"cat_alert_{self.seed:04d}_c{i:06d}",
                category=category,
                category_index=cat_idx,
                factors=factors,
                ground_truth_action=actions[stored_gt_idx],
                gt_action_index=stored_gt_idx,
                gt_action_name=original_gt_action,
                is_noisy=is_noisy,
            ))

        return alerts

    def generate_precampaign(self, n: int, suppressed_actions: dict) -> list[CategoryAlert]:
        """
        Generate alerts where certain (category, action) combinations are historically rare.

        suppressed_actions: dict mapping category_name -> action_name_to_suppress
        For each suppressed (category, action) pair: that action's GT probability is
        reduced to 5% and the excess redistributed proportionally to other actions.

        This simulates a pre-campaign period where the target action was rarely
        the correct answer for these categories, so ProfileScorer never learns it well.

        Uses self.seed + 20000 as RNG seed (independent of generate() state).
        """
        # Build overridden GT distributions
        overridden_gt: dict[str, np.ndarray] = {}
        for cat in self.categories:
            if cat in suppressed_actions:
                act_to_suppress = suppressed_actions[cat]
                a_idx = self.actions.index(act_to_suppress)
                orig_dist = self.gt_distributions[cat].copy()
                suppressed_prob = 0.05
                excess = float(orig_dist[a_idx]) - suppressed_prob
                dist = orig_dist.copy()
                dist[a_idx] = suppressed_prob
                if excess > 0:
                    other_indices = [j for j in range(len(self.actions)) if j != a_idx]
                    other_total = float(sum(orig_dist[j] for j in other_indices))
                    if other_total > 0:
                        for j in other_indices:
                            dist[j] = orig_dist[j] + excess * (float(orig_dist[j]) / other_total)
                dist = dist / dist.sum()
                overridden_gt[cat] = dist
            else:
                overridden_gt[cat] = self.gt_distributions[cat]

        rng = np.random.default_rng(self.seed + 20000)
        categories = self.categories
        actions = self.actions

        probs = np.ones(len(categories)) / len(categories)
        cat_indices = rng.choice(len(categories), size=n, p=probs)
        is_noisy_mask = rng.random(n) < self.noise_rate

        alerts: list[CategoryAlert] = []
        for i in range(n):
            cat_idx = int(cat_indices[i])
            category = categories[cat_idx]

            gt_dist = overridden_gt[category]
            original_gt_idx = int(rng.choice(len(actions), p=gt_dist))
            original_gt_action = actions[original_gt_idx]

            mu = self.profiles[category][original_gt_action]
            factors = rng.normal(loc=mu, scale=self.factor_sigma).clip(0.0, 1.0)

            is_noisy = bool(is_noisy_mask[i])
            stored_gt_idx = original_gt_idx
            if is_noisy:
                other_indices = [j for j in range(len(actions)) if j != original_gt_idx]
                stored_gt_idx = int(other_indices[rng.integers(0, len(other_indices))])

            alerts.append(CategoryAlert(
                alert_id=f"cat_alert_{self.seed:04d}_q{i:06d}",
                category=category,
                category_index=cat_idx,
                factors=factors,
                ground_truth_action=actions[stored_gt_idx],
                gt_action_index=stored_gt_idx,
                gt_action_name=original_gt_action,
                is_noisy=is_noisy,
            ))

        return alerts

    def get_weighted_category_means(self) -> np.ndarray:
        """
        Return a (n_categories × n_factors) matrix of GT-distribution-weighted
        mean factor profiles.

        Row i = Σ_a  P(action=a | category_i) × profile[category_i][a].

        This is the expected factor vector for an alert from category i,
        averaging over which action is appropriate.  Used for backward
        compatibility with code that expects a single profile per category.

        Returns
        -------
        np.ndarray
            Shape (n_categories, n_factors), values in [0, 1].
        """
        n_cats = len(self.categories)
        n_facts = len(self.factors)
        result = np.zeros((n_cats, n_facts), dtype=np.float64)
        for i, cat in enumerate(self.categories):
            gt_dist = self.gt_distributions[cat]
            for j, act in enumerate(self.actions):
                result[i] += float(gt_dist[j]) * self.profiles[cat][act]
        return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # a. Load config
    with open(_CONFIG_PATH) as fh:
        raw_cfg = yaml.safe_load(fh)
    bc = raw_cfg["bridge_common"]

    # b. Create generator with noise_rate=0.0, seed=42
    gen = CategoryAlertGenerator(
        categories=bc["categories"],
        actions=bc["actions"],
        factors=bc["factors"],
        action_conditional_profiles=bc["action_conditional_profiles"],
        gt_distributions=bc["category_gt_distributions"],
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=0.0,
        seed=42,
    )

    # c. Generate 2000 balanced alerts (400 per category)
    alerts = gen.generate_batch(n_per_category=400)
    assert len(alerts) == 2000, f"FAIL: expected 2000 alerts, got {len(alerts)}"

    categories = bc["categories"]
    actions    = bc["actions"]

    # d. Per-category: GT distribution and per-action factor means
    print("=== Action-conditional factor profiles: observed vs. configured ===")
    for cat in categories:
        cat_alerts = [a for a in alerts if a.category == cat]
        assert len(cat_alerts) == 400, f"FAIL: category '{cat}' count {len(cat_alerts)} != 400"

        # Check GT action distribution using gt_action_name (pre-noise original)
        obs_dist = np.zeros(len(actions))
        for a in cat_alerts:
            obs_dist[actions.index(a.gt_action_name)] += 1
        obs_dist /= len(cat_alerts)
        target_dist = np.asarray(bc["category_gt_distributions"][cat])
        max_dist_err = float(np.max(np.abs(obs_dist - target_dist)))
        assert max_dist_err <= 0.07, (
            f"FAIL: category '{cat}' GT dist max error {max_dist_err:.4f} > 0.07"
        )

        # Per-action factor means
        for act in actions:
            act_alerts = [a for a in cat_alerts if a.gt_action_name == act]
            if len(act_alerts) < 10:
                continue
            obs_mean = np.mean([a.factors for a in act_alerts], axis=0)
            cfg_mean = np.asarray(bc["action_conditional_profiles"][cat][act])
            max_err = float(np.max(np.abs(obs_mean - cfg_mean)))
            print(
                f"  {cat} action={act} n={len(act_alerts)} "
                f"mean_factors=[{', '.join(f'{v:.2f}' for v in obs_mean)}]"
            )
            assert max_err <= 0.08, (
                f"FAIL: {cat}/{act} max factor error {max_err:.4f} > 0.08"
            )

    # e. Verify noise_rate=0.0 produces zero noisy alerts
    n_noisy_clean = sum(1 for a in alerts if a.is_noisy)
    assert n_noisy_clean == 0, f"FAIL: noise_rate=0.0 but {n_noisy_clean} noisy alerts found"

    # f. Noisy generator
    gen_noisy = CategoryAlertGenerator(
        categories=bc["categories"],
        actions=bc["actions"],
        factors=bc["factors"],
        action_conditional_profiles=bc["action_conditional_profiles"],
        gt_distributions=bc["category_gt_distributions"],
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=0.15,
        seed=123,
    )
    alerts_noisy = gen_noisy.generate_batch(n_per_category=400)
    obs_noise_rate = sum(1 for a in alerts_noisy if a.is_noisy) / len(alerts_noisy)
    assert 0.12 <= obs_noise_rate <= 0.18, (
        f"FAIL: noise rate {obs_noise_rate:.4f} outside [0.12, 0.18]"
    )
    print(f"\n=== Noise rate ===")
    print(f"  Observed: {obs_noise_rate:.3f}   Expected: ~0.150")

    # Verify noisy alerts: factors reflect the original GT action (gt_action_name),
    # not the flipped label (ground_truth_action).
    # Check in aggregate per (category, original_action) group — individual
    # alerts can randomly be closer to a different profile due to sigma=0.15.
    noisy_alerts = [a for a in alerts_noisy if a.is_noisy]
    assert all(a.gt_action_name != a.ground_truth_action for a in noisy_alerts), (
        "FAIL: noisy alert has gt_action_name == ground_truth_action"
    )
    # Group by (category, original_action) and check aggregate mean
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for a in noisy_alerts:
        groups[(a.category, a.gt_action_name)].append(a)
    for (cat, orig_act), group in groups.items():
        if len(group) < 5:
            continue
        mean_factors = np.mean([a.factors for a in group], axis=0)
        cfg_orig = np.asarray(bc["action_conditional_profiles"][cat][orig_act])
        # Pick any OTHER action as the comparison flipped profile
        other_acts = [act for act in actions if act != orig_act]
        for flip_act in other_acts:
            cfg_flip = np.asarray(bc["action_conditional_profiles"][cat][flip_act])
            if np.max(np.abs(cfg_orig - cfg_flip)) > 0.25:
                err_orig = float(np.mean(np.abs(mean_factors - cfg_orig)))
                err_flip = float(np.mean(np.abs(mean_factors - cfg_flip)))
                assert err_orig < err_flip, (
                    f"FAIL: {cat}/{orig_act} aggregate factors closer to "
                    f"flipped profile {flip_act} ({err_flip:.3f}) than "
                    f"to original ({err_orig:.3f})"
                )

    # g. get_weighted_category_means: shape (5, 6), all values in [0, 1]
    wm = gen.get_weighted_category_means()
    assert wm.shape == (len(categories), len(bc["factors"])), (
        f"FAIL: weighted means shape {wm.shape}"
    )
    assert float(wm.min()) >= 0.0, "FAIL: weighted means contain values < 0"
    assert float(wm.max()) <= 1.0, "FAIL: weighted means contain values > 1"
    print(f"\n=== Weighted category means (5×6) ===")
    for i, cat in enumerate(categories):
        print(f"  {cat:<22}: [{', '.join(f'{v:.2f}' for v in wm[i])}]")

    print("\nAll checks passed")
