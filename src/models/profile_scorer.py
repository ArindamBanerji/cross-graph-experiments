"""
Profile-based scoring using L2 distance (nearest centroid).

Eq. 4'' (revised): P(a | f, c) = softmax(-||f - mu[c,a,:]||^2 / tau)

mu[c, a, :] = learned profile centroid for (category c, action a)
Initialized from configured profiles (warm start) or uniform 0.5 (cold start).
Updated online via asymmetric centroid pull/push.
"""
from __future__ import annotations

import numpy as np


class ProfileScorer:
    """
    L2-distance profile scorer with online centroid update.

    Parameters
    ----------
    n_categories : int
        Number of alert categories (5).
    n_actions : int
        Number of possible actions (4).
    n_factors : int
        Dimensionality of the factor vector (6).
    tau : float
        Softmax temperature.  Lower = sharper distribution (more greedy).
    eta : float
        Learning rate for correct decisions (pull centroid toward f).
    eta_neg : float
        Learning rate for incorrect decisions (push centroid away from f).
    seed : int
        Master seed for the internal RNG (used by score_stochastic).
    """

    def __init__(
        self,
        n_categories: int,
        n_actions: int,
        n_factors: int,
        tau: float = 0.25,
        eta: float = 0.05,
        eta_neg: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_categories = n_categories
        self.n_actions    = n_actions
        self.n_factors    = n_factors
        self.tau          = tau
        self.eta          = eta
        self.eta_neg      = eta_neg
        self.rng          = np.random.default_rng(seed)

        # Profile centroids: mu[c, a, :] = expected factor vector for (category c, action a)
        self.mu = np.full((n_categories, n_actions, n_factors), 0.5, dtype=np.float64)

        # Decision counts per (category, action) for count-based learning-rate decay
        self.counts = np.zeros((n_categories, n_actions), dtype=np.int64)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_profiles(
        self,
        profiles: dict,
        categories: list[str],
        actions: list[str],
    ) -> None:
        """
        Initialize mu from configured profiles dict.

        Parameters
        ----------
        profiles : dict
            Nested dict: profiles[category_name][action_name] → list of n_factors floats.
        categories : list[str]
            Ordered category names.
        actions : list[str]
            Ordered action names.
        """
        for c_idx, cat in enumerate(categories):
            for a_idx, act in enumerate(actions):
                self.mu[c_idx, a_idx, :] = np.array(profiles[cat][act], dtype=np.float64)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        factors: np.ndarray,
        category_index: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Score all actions for this alert using L2 distance.

        Parameters
        ----------
        factors : np.ndarray
            Factor vector, shape (n_factors,) or (n_factors, 1).
        category_index : int
            Category index of the alert.

        Returns
        -------
        action_idx : int
            Argmax (greedy) action.
        probs : np.ndarray, shape (n_actions,)
            Softmax probabilities.
        distances : np.ndarray, shape (n_actions,)
            Raw squared L2 distances to each centroid.
        """
        f    = factors.flatten()
        mu_c = self.mu[category_index]        # (n_actions, n_factors)

        diffs     = mu_c - f                  # (n_actions, n_factors)
        distances = np.sum(diffs ** 2, axis=1)  # (n_actions,)

        neg_dist = -distances / self.tau
        neg_dist -= neg_dist.max()            # numerically stable softmax
        exp_vals = np.exp(neg_dist)
        probs    = exp_vals / (exp_vals.sum() + 1e-10)

        action_idx = int(np.argmax(probs))
        return action_idx, probs, distances

    def score_stochastic(
        self,
        factors: np.ndarray,
        category_index: int,
        rng: np.random.Generator,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Sample action from softmax distribution (for exploration).

        Parameters
        ----------
        factors : np.ndarray
            Factor vector.
        category_index : int
            Category index.
        rng : np.random.Generator
            External RNG to use for sampling.

        Returns
        -------
        action_idx : int
            Sampled action index.
        probs : np.ndarray, shape (n_actions,)
            Softmax probabilities.
        distances : np.ndarray, shape (n_actions,)
            Raw squared L2 distances.
        """
        _, probs, distances = self.score(factors, category_index)
        action_idx = int(rng.choice(self.n_actions, p=probs))
        return action_idx, probs, distances

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        factors: np.ndarray,
        category_index: int,
        action_idx: int,
        correct: bool,
    ) -> None:
        """
        Update profile centroid based on outcome.

        If correct:   pull mu[c, action_taken, :] toward f.
        If incorrect: push mu[c, action_taken, :] away from f.

        Uses count-based decay: effective_eta = eta / (1 + count * 0.001)
        Early updates are large (fast learning); late updates are small (stable).

        Parameters
        ----------
        factors : np.ndarray
            Factor vector of the alert.
        category_index : int
            Category index of the alert.
        action_idx : int
            Action that was taken.
        correct : bool
            Whether the oracle signalled the action was correct.
        """
        f     = factors.flatten()
        c, a  = category_index, action_idx
        count = int(self.counts[c, a])

        if correct:
            effective_eta = self.eta / (1.0 + count * 0.001)
            self.mu[c, a] += effective_eta * (f - self.mu[c, a])
        else:
            effective_eta = self.eta_neg / (1.0 + count * 0.001)
            self.mu[c, a] -= effective_eta * (f - self.mu[c, a])

        np.clip(self.mu[c, a], 0.0, 1.0, out=self.mu[c, a])
        self.counts[c, a] += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_profile_snapshot(self) -> np.ndarray:
        """Return a copy of the current mu matrix (shape n_categories × n_actions × n_factors)."""
        return self.mu.copy()

    def get_profile_drift(self, initial_mu: np.ndarray) -> np.ndarray:
        """
        Compute per-(category, action) L2 drift from initial profiles.

        Parameters
        ----------
        initial_mu : np.ndarray
            Shape (n_categories, n_actions, n_factors).

        Returns
        -------
        np.ndarray
            Shape (n_categories, n_actions), each entry = ||mu[c,a] - initial_mu[c,a]||.
        """
        diff  = self.mu - initial_mu          # (n_cats, n_actions, n_factors)
        drift = np.linalg.norm(diff, axis=2)  # (n_cats, n_actions)
        return drift


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))

    import yaml
    from src.data.category_alert_generator import CategoryAlertGenerator

    print("--- ProfileScorer self-test ---")

    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    bc                = raw["bridge_common"]
    realistic_profiles = raw["realistic_profiles"]

    # -----------------------------------------------------------------------
    # Test 1: Initialize from profiles
    # -----------------------------------------------------------------------
    scorer = ProfileScorer(5, 4, 6, tau=0.25, eta=0.05, eta_neg=0.01)
    scorer.init_from_profiles(
        realistic_profiles["action_conditional_profiles"],
        bc["categories"],
        bc["actions"],
    )
    assert np.std(scorer.mu) > 0.1, "mu should not be uniform after init"
    print("  init_from_profiles: OK")

    # -----------------------------------------------------------------------
    # Test 2: Score an alert
    # -----------------------------------------------------------------------
    # Factor vector close to credential_access / auto_close profile
    f_test = np.array([0.15, 0.20, 0.70, 0.15, 0.85, 0.85])
    action_idx, probs, distances = scorer.score(f_test, category_index=0)
    print(f"  score test: predicted action {action_idx}, probs={[f'{p:.2f}' for p in probs]}")
    assert action_idx == 0, f"Expected action 0 (auto_close), got {action_idx}"
    print("  score: OK")

    # -----------------------------------------------------------------------
    # Test 3: Update pulls centroid toward correct example
    # -----------------------------------------------------------------------
    initial_mu = scorer.mu[0, 0, :].copy()
    f_nearby   = np.array([0.14, 0.21, 0.68, 0.16, 0.82, 0.84])
    scorer.update(f_nearby, category_index=0, action_idx=0, correct=True)
    moved = float(np.linalg.norm(scorer.mu[0, 0, :] - initial_mu))
    assert moved > 0, "mu should move after correct update"
    print(f"  update (correct): mu moved {moved:.4f}")

    # -----------------------------------------------------------------------
    # Test 4: Incorrect update pushes centroid away
    # -----------------------------------------------------------------------
    initial_mu2 = scorer.mu[0, 1, :].copy()
    scorer.update(f_nearby, category_index=0, action_idx=1, correct=False)
    moved2 = float(np.linalg.norm(scorer.mu[0, 1, :] - initial_mu2))
    assert moved2 > 0, "mu should move after incorrect update"
    print(f"  update (incorrect): mu moved {moved2:.4f}")

    # -----------------------------------------------------------------------
    # Test 5: L2 vs dot product comparison on 1000 realistic alerts
    # -----------------------------------------------------------------------
    gen = CategoryAlertGenerator(
        categories=bc["categories"],
        actions=bc["actions"],
        factors=bc["factors"],
        action_conditional_profiles=realistic_profiles["action_conditional_profiles"],
        gt_distributions=realistic_profiles["category_gt_distributions"],
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=0.0,
        seed=42,
    )
    alerts = gen.generate(1000)

    l2_correct  = 0
    dot_correct = 0
    for alert in alerts:
        a_l2, _, _ = scorer.score(alert.factors, alert.category_index)
        if a_l2 == alert.gt_action_index:
            l2_correct += 1

        f     = alert.factors.flatten()
        c     = alert.category_index
        dot_s = np.array([
            np.dot(f, scorer.mu[c, a, :]) for a in range(4)
        ])
        a_dot = int(np.argmax(dot_s))
        if a_dot == alert.gt_action_index:
            dot_correct += 1

    print(f"  L2 accuracy: {l2_correct/1000:.1%},  Dot accuracy: {dot_correct/1000:.1%}")
    assert l2_correct > dot_correct + 100, (
        f"L2 ({l2_correct}) should beat dot ({dot_correct}) by >100 points"
    )
    print("  L2 vs dot product: OK")

    print("\nAll ProfileScorer tests passed.")
