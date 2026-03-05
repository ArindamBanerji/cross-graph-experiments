"""
Oracle implementations for Bridge Layer Experiments (EXP 5-9).

Two oracle types model different feedback regimes:

  BernoulliOracle — legacy oracle producing the R1 problem.  Outcome is drawn
      from Bernoulli(category_rate) *independently* of whether the system action
      was correct.  The scoring matrix therefore converges to the oracle's
      category-level bias rather than to ground truth.

  GTAlignedOracle — the R1 fix from bridge_layer_design_v1.  Base outcome is
      +1 iff the system action matches ground truth, else -1.  A ``noise_rate``
      fraction of outcomes is randomly flipped to model analyst feedback noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.category_alert_generator import CATEGORIES, CategoryAlert


# ---------------------------------------------------------------------------
# Default Bernoulli rates per category
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORY_RATES: dict[str, float] = {
    "credential_access":  0.75,
    "threat_intel_match": 0.65,
    "lateral_movement":   0.60,
    "data_exfiltration":  0.55,
    "insider_threat":     0.70,
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OracleResult:
    """Result of a single oracle evaluation."""

    outcome: int          # +1 (positive feedback) or -1 (negative feedback)
    source: str           # "bernoulli" | "gt_aligned"
    gt_action: str        # ground truth action (always available in experiments)
    gt_aligned: bool      # whether outcome agrees with ground truth
    noise_flipped: bool   # whether this specific outcome was flipped by noise


# ---------------------------------------------------------------------------
# BernoulliOracle
# ---------------------------------------------------------------------------

class BernoulliOracle:
    """
    Legacy oracle producing the R1 problem (convergence ≠ correctness).

    Outcome is Bernoulli(category_rate) *independently* of whether the system
    action matches ground truth.  This causes the scoring matrix to converge
    toward the oracle's category-level bias rather than toward correct actions.

    ``gt_aligned`` is True only when the coin-flip outcome *happens* to match
    what correct feedback would look like — i.e., (+1 and correct) or
    (-1 and incorrect).  With a biased coin, this alignment is decoupled from
    actual performance.

    Parameters
    ----------
    category_rates : dict[str, float] | None
        Per-category Bernoulli p(+1) rates.  If None, uses hard-coded defaults
        ranging from 0.55 (data_exfiltration) to 0.75 (credential_access).
    seed : int
        Master seed for ``np.random.default_rng(seed)``.
    """

    def __init__(
        self,
        category_rates: Optional[dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        self._category_rates: dict[str, float] = (
            category_rates if category_rates is not None
            else dict(_DEFAULT_CATEGORY_RATES)
        )
        self._rng = np.random.default_rng(seed)

    def evaluate(self, system_action: str, alert: CategoryAlert) -> OracleResult:
        """
        Evaluate system action using an independent Bernoulli coin flip.

        Parameters
        ----------
        system_action : str
            Action selected by the scoring matrix (one of ACTIONS).
        alert : CategoryAlert
            The alert being processed.

        Returns
        -------
        OracleResult
            ``outcome`` is +1 with probability ``category_rates[alert.category]``
            regardless of whether ``system_action`` is correct.
        """
        rate = self._category_rates[alert.category]
        outcome = 1 if bool(self._rng.random() < rate) else -1

        is_correct = system_action == alert.ground_truth_action
        gt_aligned = (
            (outcome == 1 and is_correct) or
            (outcome == -1 and not is_correct)
        )
        return OracleResult(
            outcome=outcome,
            source="bernoulli",
            gt_action=alert.ground_truth_action,
            gt_aligned=gt_aligned,
            noise_flipped=False,
        )


# ---------------------------------------------------------------------------
# GTAlignedOracle
# ---------------------------------------------------------------------------

class GTAlignedOracle:
    """
    GT-aligned oracle: the R1 fix from bridge_layer_design_v1.

    Base outcome is +1 iff the system action matches ground truth, else -1.
    With probability ``noise_rate`` the base outcome is flipped, modelling
    analyst feedback noise (typos, disagreements, delayed corrections).

    Unlike BernoulliOracle, convergence of the scoring matrix is directly
    tied to decision correctness.  Noise only partially corrupts the signal.

    Parameters
    ----------
    noise_rate : float
        Probability of flipping the base outcome.  0.0 = perfect oracle,
        1.0 = always wrong oracle.
    seed : int
        Master seed for ``np.random.default_rng(seed)``.
    """

    def __init__(self, noise_rate: float = 0.0, seed: int = 42) -> None:
        self.noise_rate: float = float(noise_rate)
        self._rng = np.random.default_rng(seed)

    def evaluate(self, system_action: str, alert: CategoryAlert) -> OracleResult:
        """
        Evaluate system action against ground truth with optional noise flip.

        Parameters
        ----------
        system_action : str
            Action selected by the scoring matrix.
        alert : CategoryAlert
            The alert being processed.

        Returns
        -------
        OracleResult
            ``gt_aligned`` is True iff the outcome was not flipped.
            ``noise_flipped`` is True iff the flip occurred.
        """
        base_outcome = 1 if system_action == alert.ground_truth_action else -1
        flip = bool(self._rng.random() < self.noise_rate)
        outcome = -base_outcome if flip else base_outcome
        return OracleResult(
            outcome=outcome,
            source="gt_aligned",
            gt_action=alert.ground_truth_action,
            gt_aligned=not flip,
            noise_flipped=flip,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.category_alert_generator import ACTIONS, CategoryAlertGenerator

    # Generate 500 balanced alerts (100 per category, no generation noise)
    gen = CategoryAlertGenerator(noise_rate=0.0, seed=42)
    alerts = gen.generate_batch(n_per_category=100)
    n = len(alerts)

    # -------------------------------------------------------------------
    # b. BernoulliOracle: use GT action as system action
    #    gt_aligned rate ≈ mean(category_rates) ≈ 0.65
    #    It is NOT 1.0 even though we always act correctly — that's the R1 problem
    # -------------------------------------------------------------------
    bern = BernoulliOracle(seed=42)
    bern_results = [bern.evaluate(a.ground_truth_action, a) for a in alerts]
    gt_aligned_b = sum(r.gt_aligned for r in bern_results) / n
    print(f"=== BernoulliOracle (GT action as system action) ===")
    print(f"  gt_aligned rate: {gt_aligned_b:.3f}  (expected ~0.65, range [0.40, 0.80])")
    print(f"  NOTE: system was correct 100% of the time, yet gt_aligned != 1.0 -- R1 problem")
    assert 0.40 <= gt_aligned_b <= 0.80, (
        f"FAIL: BernoulliOracle gt_aligned {gt_aligned_b:.3f} outside [0.40, 0.80]"
    )

    # -------------------------------------------------------------------
    # c. GTAlignedOracle(noise=0.0): perfect oracle, GT action → all +1
    # -------------------------------------------------------------------
    gta_0 = GTAlignedOracle(noise_rate=0.0, seed=42)
    res_c = [gta_0.evaluate(a.ground_truth_action, a) for a in alerts]
    all_pos = all(r.outcome == 1 for r in res_c)
    all_aligned = all(r.gt_aligned for r in res_c)
    print(f"\n=== GTAlignedOracle(noise=0.0), GT action as system ===")
    print(f"  All outcomes == +1:  {all_pos}")
    print(f"  All gt_aligned:      {all_aligned}")
    assert all_pos,    "FAIL: expected all outcomes == +1 with perfect system and no noise"
    assert all_aligned, "FAIL: expected all gt_aligned with no noise"

    # -------------------------------------------------------------------
    # d. GTAlignedOracle(noise=0.15): 85% aligned, 15% flipped
    # -------------------------------------------------------------------
    gta_15 = GTAlignedOracle(noise_rate=0.15, seed=42)
    res_d = [gta_15.evaluate(a.ground_truth_action, a) for a in alerts]
    gt_aligned_d  = sum(r.gt_aligned     for r in res_d) / n
    noise_flipped_d = sum(r.noise_flipped for r in res_d) / n
    print(f"\n=== GTAlignedOracle(noise=0.15), GT action as system ===")
    print(f"  gt_aligned rate:    {gt_aligned_d:.3f}   (expected ~0.85, range [0.80, 0.90])")
    print(f"  noise_flipped rate: {noise_flipped_d:.3f}  (expected ~0.15, range [0.10, 0.20])")
    assert 0.80 <= gt_aligned_d  <= 0.90, (
        f"FAIL: gt_aligned {gt_aligned_d:.3f} outside [0.80, 0.90]"
    )
    assert 0.10 <= noise_flipped_d <= 0.20, (
        f"FAIL: noise_flipped {noise_flipped_d:.3f} outside [0.10, 0.20]"
    )

    # -------------------------------------------------------------------
    # e. GTAlignedOracle(noise=0.30): 70% aligned, 30% flipped
    # -------------------------------------------------------------------
    gta_30 = GTAlignedOracle(noise_rate=0.30, seed=42)
    res_e = [gta_30.evaluate(a.ground_truth_action, a) for a in alerts]
    gt_aligned_e  = sum(r.gt_aligned     for r in res_e) / n
    noise_flipped_e = sum(r.noise_flipped for r in res_e) / n
    print(f"\n=== GTAlignedOracle(noise=0.30), GT action as system ===")
    print(f"  gt_aligned rate:    {gt_aligned_e:.3f}   (expected ~0.70, range [0.65, 0.75])")
    print(f"  noise_flipped rate: {noise_flipped_e:.3f}  (expected ~0.30, range [0.25, 0.35])")
    assert 0.65 <= gt_aligned_e  <= 0.75, (
        f"FAIL: gt_aligned {gt_aligned_e:.3f} outside [0.65, 0.75]"
    )
    assert 0.25 <= noise_flipped_e <= 0.35, (
        f"FAIL: noise_flipped {noise_flipped_e:.3f} outside [0.25, 0.35]"
    )

    # -------------------------------------------------------------------
    # f. GTAlignedOracle(noise=0.0): random actions → ~25% correct
    # -------------------------------------------------------------------
    rng_rand = np.random.default_rng(99)
    gta_rand = GTAlignedOracle(noise_rate=0.0, seed=77)
    res_f = [
        gta_rand.evaluate(ACTIONS[int(rng_rand.integers(0, len(ACTIONS)))], a)
        for a in alerts
    ]
    correct_rate_f = sum(r.outcome == 1 for r in res_f) / n
    print(f"\n=== GTAlignedOracle(noise=0.0), RANDOM action ===")
    print(f"  outcome==+1 rate: {correct_rate_f:.3f}  (expected ~0.25, range [0.15, 0.35])")
    assert 0.15 <= correct_rate_f <= 0.35, (
        f"FAIL: correct rate {correct_rate_f:.3f} outside [0.15, 0.35]"
    )

    print("\nAll checks passed")
