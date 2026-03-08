"""
OPHarness — Loop-2-running execution harness for OP experiments.

The critical difference from S1-S4:
  S-series: frozen profiles, synthesis applied to static centroids.
  OP-series: Loop 2 runs simultaneously with operator application.

Per-decision sequence:
  1. registry.get_synthesis()        -> current SynthesisBias
  2. scorer.score(f, c, synthesis)   -> action selection
  3. oracle.evaluate(action, gt)     -> outcome
  4. scorer.update(f, c, a, outcome) -> centroid update (NO sigma)
  5. registry.step(1)                -> decrement TTLs
  6. registry.expire_stale()         -> remove expired operators

This interleaving is the product claim: synthesis helps during the
transition window; Loop 2 learns from the operator-biased distribution;
centroids eventually converge to the post-shift equilibrium.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle
from src.models.operator_registry import OperatorRegistry
from src.models.synthesis import SynthesisBias
from src.eval.auac import compute_auac, AUACResult


@dataclass
class HarnessConfig:
    n_decisions:       int   = 500
    snapshot_interval: int   = 50
    use_synthesis:     bool  = True
    lambda_override:   float | None = None
    oracle_noise_rate: float = 0.0
    window_size:       int   = 50


@dataclass
class HarnessResult:
    correct_flags:         list[bool]
    auac_result:           AUACResult
    centroid_snapshots:    dict          # {decision_index: mu_copy}
    n_operators_expired:   int
    synthesis_was_active:  list[bool]
    final_mu:              np.ndarray


class OPHarness:
    """
    Loop-2-running harness.

    Args:
        scorer:   ProfileScorer (warm or cold start).
        oracle:   GTAlignedOracle (used only for noise_rate; evaluation is
                  done by index comparison for compatibility with GenericAlert
                  and CategoryAlert).
        registry: OperatorRegistry (may be empty for baseline runs).
        config:   HarnessConfig.
    """

    def __init__(
        self,
        scorer:   ProfileScorer,
        oracle:   GTAlignedOracle,
        registry: OperatorRegistry,
        config:   HarnessConfig | None = None,
    ):
        self.scorer   = scorer
        self.oracle   = oracle
        self.registry = registry
        self.config   = config or HarnessConfig()
        self._rng     = np.random.default_rng(42)

    def run(self, alerts: list) -> HarnessResult:
        """
        Execute the full decision loop for all alerts.

        Each alert must have:
            .factors (np.ndarray, shape (d,))
            .category_index (int)
            .gt_action_index (int)
        """
        cfg                 = self.config
        correct_flags:      list[bool] = []
        synthesis_active:   list[bool] = []
        centroid_snapshots: dict       = {}
        n_expired_total                = 0

        # Determine n_categories, n_actions from scorer
        n_cat = self.scorer.mu.shape[0]
        n_act = self.scorer.mu.shape[1]

        for i, alert in enumerate(alerts):
            # 1. Get synthesis
            if cfg.use_synthesis:
                synthesis = self.registry.get_synthesis(lambda_override=cfg.lambda_override)
            else:
                synthesis = SynthesisBias.neutral(n_cat, n_act)

            synthesis_active.append(synthesis.is_active)

            # 2. Score -> action
            result        = self.scorer.score(alert.factors, alert.category_index, synthesis=synthesis)
            chosen_action = result.action_index

            # 3. Evaluate correctness by index (compatible with GenericAlert and CategoryAlert)
            is_correct = bool(chosen_action == alert.gt_action_index)
            if cfg.oracle_noise_rate > 0 and self._rng.random() < cfg.oracle_noise_rate:
                is_correct = not is_correct
            correct_flags.append(is_correct)

            # 4. Centroid update — NO synthesis
            self.scorer.update(
                factors=alert.factors,
                category_index=alert.category_index,
                action_idx=chosen_action,
                correct=is_correct,
            )

            # 5. Advance TTL
            self.registry.step(1)

            # 6. Expire stale
            expired          = self.registry.expire_stale()
            n_expired_total += len(expired)

            # 7. Snapshot
            if i % cfg.snapshot_interval == 0:
                centroid_snapshots[i] = self.scorer.mu.copy()

        centroid_snapshots[len(alerts) - 1] = self.scorer.mu.copy()

        return HarnessResult(
            correct_flags=correct_flags,
            auac_result=compute_auac(correct_flags, window_size=cfg.window_size),
            centroid_snapshots=centroid_snapshots,
            n_operators_expired=n_expired_total,
            synthesis_was_active=synthesis_active,
            final_mu=self.scorer.mu.copy(),
        )


def run_paired_comparison(
    alerts:          list,
    scorer_with:     ProfileScorer,
    scorer_without:  ProfileScorer,
    oracle:          GTAlignedOracle,
    registry_with:   OperatorRegistry,
    config:          HarnessConfig,
) -> tuple[HarnessResult, HarnessResult]:
    """
    Run with-operator and without-operator on the SAME alert sequence.
    Required for paired permutation test in OP1-OP4.
    Both scorers must be initialized identically (same mu, same tau/eta).
    """
    empty_registry = OperatorRegistry(
        n_categories=registry_with.n_categories,
        n_actions=registry_with.n_actions,
        n_factors=registry_with.n_factors,
    )
    config_without = HarnessConfig(
        n_decisions=config.n_decisions,
        snapshot_interval=config.snapshot_interval,
        use_synthesis=False,
        oracle_noise_rate=config.oracle_noise_rate,
        window_size=config.window_size,
    )
    result_with    = OPHarness(scorer_with,    oracle, registry_with,  config        ).run(alerts)
    result_without = OPHarness(scorer_without, oracle, empty_registry, config_without).run(alerts)
    return result_with, result_without


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.data.generic_alert_generator import GenericAlertGenerator
    from src.models.operator_spec import OperatorSpec

    C, A, d, N = 5, 4, 6, 300
    gen      = GenericAlertGenerator(n_categories=C, n_actions=A, n_factors=d, seed=42)
    alerts   = gen.generate(N)
    profiles = gen.get_profiles()

    scorer_w  = ProfileScorer(profiles.copy(), gen.actions, tau=0.1, eta=0.05, eta_neg=1.0)
    scorer_wo = ProfileScorer(profiles.copy(), gen.actions, tau=0.1, eta=0.05, eta_neg=1.0)
    oracle    = GTAlignedOracle(noise_rate=0.0)

    sigma_0      = np.zeros((C, A))
    sigma_0[0,0] = -0.4
    spec = OperatorSpec(
        operator_id="test_op", claim_type="test",
        rank=0, sigma_0=sigma_0, lambda_s=0.2, ttl_decisions=N,
    )
    registry = OperatorRegistry(n_categories=C, n_actions=A, n_factors=d)
    registry.register(spec, profiles)

    cfg = HarnessConfig(n_decisions=N, snapshot_interval=50, window_size=50)
    result_with, result_without = run_paired_comparison(
        alerts=alerts,
        scorer_with=scorer_w,
        scorer_without=scorer_wo,
        oracle=oracle,
        registry_with=registry,
        config=cfg,
    )

    assert 0.0 < result_with.auac_result.auac    <= 1.0
    assert 0.0 < result_without.auac_result.auac <= 1.0
    assert len(result_with.correct_flags)    == N
    assert len(result_without.correct_flags) == N
    assert len(result_with.centroid_snapshots) > 0

    print(f"[PASS] Paired comparison:")
    print(f"       AUAC with-operator:    {result_with.auac_result.auac:.4f}")
    print(f"       AUAC without-operator: {result_without.auac_result.auac:.4f}")
    print(f"       Delta:                 {result_with.auac_result.auac - result_without.auac_result.auac:+.4f}")
    print(f"       T70 with:              {result_with.auac_result.t70}")
    print(f"       T70 without:           {result_without.auac_result.t70}")
    print(f"       Operators expired:     {result_with.n_operators_expired}")

    print("\nop_harness.py: all self-tests passed.")
