"""
AUAC — Area Under Accuracy Curve.

Primary metric for all OP and GE experiments. Measures the full accuracy
trajectory, not just the endpoint. Captures recovery speed, convergence
behavior, and the cost of incorrect operators.

All functions are pure — no side effects, no file I/O.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class AUACResult:
    """Complete accuracy trajectory metrics for one experimental run."""
    auac: float               # Area under accuracy curve, normalized to [0,1]
    t70: int | None           # Decision index at which accuracy first reaches 70%
    t90: int | None           # Decision index at which accuracy first reaches 90%
    cumulative_regret: float  # Sum of (1 - accuracy[t]) over all decisions
    final_accuracy: float     # Accuracy at last decision window
    accuracy_curve: np.ndarray  # Full per-window accuracy trajectory


def compute_auac(
    correct_flags: list[bool] | np.ndarray,
    window_size: int = 50,
) -> AUACResult:
    """
    Compute AUAC from a sequence of correct/incorrect decision flags.

    Args:
        correct_flags: Boolean sequence of length n_decisions.
                       True = correct decision, False = incorrect.
        window_size:   Rolling window for accuracy smoothing (default 50).
                       Must be <= len(correct_flags).

    Returns:
        AUACResult with auac, t70, t90, cumulative_regret, final_accuracy,
        and the full accuracy_curve array.
    """
    flags = np.asarray(correct_flags, dtype=float)
    n = len(flags)
    assert window_size <= n, f"window_size {window_size} > n_decisions {n}"

    n_windows = n - window_size + 1
    curve = np.array([
        flags[i:i + window_size].mean()
        for i in range(n_windows)
    ])

    auac = float(np.trapz(curve) / (len(curve) - 1)) if len(curve) > 1 else float(curve[0])

    t70_idx = next((i for i, v in enumerate(curve) if v >= 0.70), None)
    t90_idx = next((i for i, v in enumerate(curve) if v >= 0.90), None)

    t70 = int(t70_idx + window_size // 2) if t70_idx is not None else None
    t90 = int(t90_idx + window_size // 2) if t90_idx is not None else None

    cumulative_regret = float(np.sum(1.0 - flags))

    return AUACResult(
        auac=auac,
        t70=t70,
        t90=t90,
        cumulative_regret=cumulative_regret,
        final_accuracy=float(curve[-1]),
        accuracy_curve=curve,
    )


def compare_auac(
    with_operator: AUACResult,
    without_operator: AUACResult,
) -> dict:
    """
    Compute comparison metrics between two AUAC results.

    Returns dict with:
      auac_delta:      with.auac - without.auac  (positive = operator helps)
      t70_speedup:     without.t70 - with.t70    (positive = faster)
      t90_speedup:     without.t90 - with.t90    (positive = faster)
      regret_delta:    without.regret - with.regret (positive = less regret)
      final_acc_delta: with.final - without.final
    """
    def _t_speedup(a, b):
        if a is None and b is None:
            return 0
        if a is None:
            return None
        if b is None:
            return None
        return int(b - a)

    return {
        "auac_delta": with_operator.auac - without_operator.auac,
        "t70_speedup": _t_speedup(with_operator.t70, without_operator.t70),
        "t90_speedup": _t_speedup(with_operator.t90, without_operator.t90),
        "regret_delta": without_operator.cumulative_regret - with_operator.cumulative_regret,
        "final_acc_delta": with_operator.final_accuracy - without_operator.final_accuracy,
    }


def auac_from_seeds(
    all_correct_flags: list[list[bool]],
    window_size: int = 50,
) -> dict:
    """
    Compute AUAC statistics across multiple seeds.

    Args:
        all_correct_flags: List of per-seed correct_flags lists.
        window_size: Rolling window size.

    Returns:
        dict with: mean_auac, std_auac, mean_t70, std_t70,
                   mean_t90, std_t90, mean_regret, per_seed_results
    """
    results = [compute_auac(flags, window_size) for flags in all_correct_flags]

    auacs   = np.array([r.auac for r in results])
    t70s    = np.array([r.t70 if r.t70 is not None else np.nan for r in results])
    t90s    = np.array([r.t90 if r.t90 is not None else np.nan for r in results])
    regrets = np.array([r.cumulative_regret for r in results])

    return {
        "mean_auac":          float(np.mean(auacs)),
        "std_auac":           float(np.std(auacs)),
        "mean_t70":           float(np.nanmean(t70s)),
        "std_t70":            float(np.nanstd(t70s)),
        "mean_t90":           float(np.nanmean(t90s)),
        "std_t90":            float(np.nanstd(t90s)),
        "mean_regret":        float(np.mean(regrets)),
        "per_seed_results":   results,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    flags_perfect = [True] * 200
    r = compute_auac(flags_perfect, window_size=50)
    assert abs(r.auac - 1.0) < 1e-6, f"Perfect flags: expected AUAC=1.0, got {r.auac}"
    assert r.t70 is not None
    assert r.t90 is not None
    print(f"[PASS] Perfect accuracy: AUAC={r.auac:.4f}, T70={r.t70}, T90={r.t90}")

    flags_random = (rng.random(500) > 0.5).tolist()
    r = compute_auac(flags_random, window_size=50)
    assert 0.4 < r.auac < 0.6, f"Random flags: expected AUAC near 0.5, got {r.auac}"
    print(f"[PASS] Random accuracy: AUAC={r.auac:.4f}")

    flags_good = (rng.random(300) > 0.1).tolist()
    flags_bad  = (rng.random(300) > 0.4).tolist()
    r_good = compute_auac(flags_good, window_size=50)
    r_bad  = compute_auac(flags_bad,  window_size=50)
    cmp = compare_auac(r_good, r_bad)
    assert cmp["auac_delta"] > 0, f"Expected good > bad: delta={cmp['auac_delta']:.4f}"
    print(f"[PASS] compare_auac: auac_delta={cmp['auac_delta']:.4f}")

    all_flags = [(rng.random(300) > 0.2).tolist() for _ in range(10)]
    stats = auac_from_seeds(all_flags, window_size=50)
    assert 0.0 < stats["mean_auac"] < 1.0
    assert len(stats["per_seed_results"]) == 10
    print(f"[PASS] auac_from_seeds: mean_auac={stats['mean_auac']:.4f}, std={stats['std_auac']:.4f}")

    print("\nauac.py: all self-tests passed.")
