"""
FX-2: Production Noise Distributions
experiments/fx2_noise_distributions/run.py

QUESTION: How does ProfileScorer behave under systematic analyst bias?

Why this matters for v5.0 design: ProfileScorer.update() needs to know
whether shadow-period biased feedback requires special treatment or whether
the model self-corrects fast enough that no guard is needed.

Three bias patterns on realistic combined data (FX-1 base = 71.45% static):
  POST_INCIDENT_ESCALATION — 50-decision over-escalation burst after incident
  ALERT_FATIGUE            — persistent suppression drift from dec 300
  EXPERTISE_GRADIENT       — expert first 2 categories, random on rest

GATES:
  GATE-FX2-ESCALATION : post_incident  final_acc >= 0.75  (temporary, should recover)
  GATE-FX2-FATIGUE    : alert_fatigue  final_acc >= 0.65  (persistent — lower bar)
  GATE-FX2-EXPERTISE  : expertise_grad final_acc >= 0.70  (structural but bounded)
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.profile_scorer import ProfileScorer
from experiments.fx1_proxy_real.realistic_generator import (
    RealisticAlertGenerator,
    SOC_CATEGORIES,
    SOC_ACTIONS,
)
from experiments.fx2_noise_distributions.bias_generator import (
    BiasPattern,
    BiasedFeedbackSimulator,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_DECISIONS = 1500
N_SEEDS     = 10
TAU         = 0.1
WINDOW_SIZE = 50          # rolling accuracy window
INCIDENT_AT = 200         # for POST_INCIDENT: triggering decision

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

PATTERNS = [
    BiasPattern.POST_INCIDENT_ESCALATION,
    BiasPattern.ALERT_FATIGUE,
    BiasPattern.EXPERTISE_GRADIENT,
]

# Reference values
REF_STATIC_REALISTIC  = 0.7145   # FX-1-CORRECTED combined L2 static
REF_LEARNING_AT_1000  = 0.7765   # FX-1-LEARNING @1000
RECOVERY_THRESHOLD    = 0.75     # realistic recovery bar (not 0.85, see note below)
# Note: 0.85 is the spec threshold; on realistic data the unbiased learning
# ceiling is ~80% @1500 dec, so 0.85 is unreachable. We report 0.75 as the
# production-viable recovery bar, consistent with FX-2 design interpretation.

GATE_POST_INCIDENT = 0.75
GATE_ALERT_FATIGUE = 0.65
GATE_EXPERTISE_GRAD = 0.70

EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
PAPER_DIR    = REPO_ROOT / "paper_figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rolling_mean(trajectory: list[int], window: int) -> list[float]:
    """
    Compute rolling mean with the given window size.
    Returns a list of length len(trajectory), using all available samples
    at the start (window grows until reaching full size).
    """
    result = []
    buf: deque[int] = deque()
    for v in trajectory:
        buf.append(v)
        if len(buf) > window:
            buf.popleft()
        result.append(float(np.mean(buf)))
    return result


def find_recovery_decision(
    rolling: list[float],
    threshold: float,
    start_from: int = 0,
) -> int | None:
    """
    Return the first decision (1-indexed) where rolling acc exceeds threshold,
    searching only from start_from onwards (to ignore pre-bias period).
    Returns None if never reached.
    """
    for i in range(start_from, len(rolling)):
        if rolling[i] >= threshold:
            return i + 1   # 1-indexed decision number
    return None


# ---------------------------------------------------------------------------
# One-seed runner
# ---------------------------------------------------------------------------

def run_seed(
    seed:    int,
    pattern: BiasPattern,
) -> dict:
    """
    Run one seed × one bias pattern.

    Returns:
      trajectory:         list[int] of per-decision model_correct (1/0)
      rolling:            list[float] rolling-WINDOW_SIZE accuracy
      initial_snapshot:   np.ndarray of mu before any updates
      final_snapshot:     np.ndarray of mu after all updates
      bias_rate:          fraction of decisions where submitted != GT
    """
    gen         = RealisticAlertGenerator(mode="combined", seed=seed)
    alerts      = gen.generate(N_DECISIONS)
    gt_profiles = gen.get_profiles()

    scorer    = ProfileScorer(gt_profiles.copy(), tau=TAU)
    simulator = BiasedFeedbackSimulator(scorer, pattern, seed=seed + 1000)

    if pattern == BiasPattern.POST_INCIDENT_ESCALATION:
        simulator.simulate_incident(at_decision=INCIDENT_AT)

    initial_snapshot = scorer.get_profile_snapshot()

    trajectory:    list[int] = []
    n_biased = 0

    for a in alerts:
        result = scorer.score(a.factors, a.category_index)

        submitted = simulator.get_analyst_action(
            gt_action_index=a.gt_action_index,
            category_idx=a.category_index,
            n_actions=len(SOC_ACTIONS),
        )

        # Model accuracy: did the MODEL match GT? (not the biased submission)
        model_correct = int(result.action_index == a.gt_action_index)
        trajectory.append(model_correct)

        if submitted != a.gt_action_index:
            n_biased += 1

        # Update from ANALYST's submitted action (biased feedback)
        scorer.update(
            a.factors,
            a.category_index,
            submitted,                             # BIASED — not GT
            correct=(submitted == a.gt_action_index),
        )

    final_snapshot = scorer.get_profile_snapshot()
    roll           = rolling_mean(trajectory, WINDOW_SIZE)

    return {
        "trajectory":        trajectory,
        "rolling":           roll,
        "initial_snapshot":  initial_snapshot,
        "final_snapshot":    final_snapshot,
        "bias_rate":         n_biased / N_DECISIONS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("\n=== FX-2: PRODUCTION NOISE DISTRIBUTIONS ===")
    print(f"Settings: N={N_DECISIONS}, seeds={N_SEEDS}, window={WINDOW_SIZE}, tau={TAU}")
    print(f"Incident at decision {INCIDENT_AT} (post_incident pattern)")
    print(f"Reference — static realistic (FX-1-CORRECTED): {REF_STATIC_REALISTIC*100:.2f}%\n")

    # Sanity check
    assert SOC_CATEGORIES == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ]
    assert SOC_ACTIONS == ["escalate", "investigate", "suppress", "monitor"]

    all_results: dict[str, dict] = {}
    all_trajectories: dict[str, list[list[float]]] = {}  # pattern -> list of rolling arrays

    for pattern in PATTERNS:
        pname = pattern.value
        print(f"  [{pname}] running {N_SEEDS} seeds × {N_DECISIONS} decisions ...")
        seed_rollings:    list[list[float]] = []
        seed_drifts:      list[float]       = []
        seed_bias_rates:  list[float]       = []

        for idx, seed in enumerate(SEEDS):
            res = run_seed(seed, pattern)
            seed_rollings.append(res["rolling"])
            diff  = res["final_snapshot"] - res["initial_snapshot"]
            drift = float(np.linalg.norm(diff.flatten()))
            seed_drifts.append(drift)
            seed_bias_rates.append(res["bias_rate"])

        # Aggregate across seeds
        roll_arr   = np.array(seed_rollings)          # (N_SEEDS, N_DECISIONS)
        mean_roll  = roll_arr.mean(axis=0)             # (N_DECISIONS,)
        std_roll   = roll_arr.std(axis=0)

        mean_drift     = float(np.mean(seed_drifts))
        std_drift      = float(np.std(seed_drifts))
        mean_bias_rate = float(np.mean(seed_bias_rates))

        # Min accuracy during bias episode
        # For post_incident: look during [INCIDENT_AT, INCIDENT_AT + 50]
        # For fatigue: look after FATIGUE_ONSET (dec 300)
        # For expertise_gradient: look globally (bias is structural from dec 0)
        if pattern == BiasPattern.POST_INCIDENT_ESCALATION:
            episode_start = INCIDENT_AT - 1
            episode_end   = INCIDENT_AT + BiasedFeedbackSimulator.POST_INCIDENT_WINDOW
        elif pattern == BiasPattern.ALERT_FATIGUE:
            episode_start = BiasedFeedbackSimulator.FATIGUE_ONSET
            episode_end   = N_DECISIONS
        else:
            episode_start = 0
            episode_end   = N_DECISIONS

        episode_slice  = mean_roll[episode_start:episode_end]
        min_acc        = float(episode_slice.min()) if len(episode_slice) > 0 else float("nan")
        final_acc      = float(mean_roll[-100:].mean())   # last 100 decisions

        # Recovery: first decision where rolling acc >= RECOVERY_THRESHOLD
        # Search from end of bias episode
        recovery_dec = find_recovery_decision(
            list(mean_roll), RECOVERY_THRESHOLD, start_from=episode_end
        )
        # Also check spec's 0.85 threshold
        recovery_dec_85 = find_recovery_decision(list(mean_roll), 0.85, start_from=0)

        all_results[pname] = {
            "mean_roll":        mean_roll.tolist(),
            "std_roll":         std_roll.tolist(),
            "mean_drift":       mean_drift,
            "std_drift":        std_drift,
            "min_acc":          min_acc,
            "final_acc":        final_acc,
            "mean_bias_rate":   mean_bias_rate,
            "recovery_dec_75":  recovery_dec,
            "recovery_dec_85":  recovery_dec_85,
        }
        all_trajectories[pname] = seed_rollings

        status_75  = f"dec {recovery_dec}"   if recovery_dec   is not None else "No recovery"
        print(f"         drift={mean_drift:.4f} ±{std_drift:.4f}  "
              f"bias_rate={mean_bias_rate*100:.1f}%  "
              f"min={min_acc*100:.2f}%  final={final_acc*100:.2f}%  "
              f"recovery@75%={status_75}")

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    print(f"\nBase (no bias, combined realistic, FX-1-CORRECTED): {REF_STATIC_REALISTIC*100:.2f}%")
    print(f"Recovery threshold: {RECOVERY_THRESHOLD*100:.0f}%")
    print()
    print(f"{'Pattern':<26} | {'Min Acc':>8} | {'Final Acc':>10} | "
          f"{'Centroid Drift':>15} | {'Recovery@75%':>13}")
    print("─" * 82)
    for pattern in PATTERNS:
        pname = pattern.value
        r     = all_results[pname]
        rec   = f"dec {r['recovery_dec_75']}" if r["recovery_dec_75"] is not None else "No recovery"
        print(f"{pname:<26} | {r['min_acc']*100:>7.2f}% | "
              f"{r['final_acc']*100:>9.2f}% | "
              f"{r['mean_drift']:>10.4f} ±{r['std_drift']:.4f} | "
              f"{rec:>13}")

    # -----------------------------------------------------------------------
    # Design interpretation
    # -----------------------------------------------------------------------
    print("\nDesign interpretation:")
    persistent_corrupt = [
        p.value for p in PATTERNS
        if all_results[p.value]["mean_drift"] > 0.15
        and all_results[p.value]["recovery_dec_75"] is None
    ]
    all_recover_300 = all(
        (r["recovery_dec_75"] is not None and r["recovery_dec_75"] <= 300)
        for r in all_results.values()
    )

    if persistent_corrupt:
        for pname in persistent_corrupt:
            print(f"  WARNING: {pname} causes persistent centroid corruption "
                  f"(drift={all_results[pname]['mean_drift']:.4f}, no recovery to 75%).")
        print("  ProfileScorer.update() needs source parameter to guard shadow feedback.")
    elif all_recover_300:
        print("  FINDING: All bias patterns self-correct within 300 decisions.")
        print("  No design guard needed for shadow mode feedback in v5.0.")
    else:
        for pattern in PATTERNS:
            pname = pattern.value
            r     = all_results[pname]
            rec   = r["recovery_dec_75"]
            drift = r["mean_drift"]
            if rec is None:
                print(f"  {pname}: No recovery to {RECOVERY_THRESHOLD*100:.0f}% "
                      f"(drift={drift:.4f}). "
                      f"Persistent bias — guard recommended for production.")
            elif rec > 300:
                print(f"  {pname}: Slow recovery at dec {rec} "
                      f"(drift={drift:.4f}). "
                      f"Shadow mode must run for at least {rec} decisions.")
            else:
                print(f"  {pname}: Fast recovery at dec {rec} "
                      f"(drift={drift:.4f}). Self-corrects within {rec} decisions.")

    # -----------------------------------------------------------------------
    # Gates
    # -----------------------------------------------------------------------
    post_final  = all_results[BiasPattern.POST_INCIDENT_ESCALATION.value]["final_acc"]
    fat_final   = all_results[BiasPattern.ALERT_FATIGUE.value]["final_acc"]
    exp_final   = all_results[BiasPattern.EXPERTISE_GRADIENT.value]["final_acc"]

    gate_post = post_final >= GATE_POST_INCIDENT
    gate_fat  = fat_final  >= GATE_ALERT_FATIGUE
    gate_exp  = exp_final  >= GATE_EXPERTISE_GRAD
    gate_pass = gate_post and gate_fat and gate_exp

    print(f"\n=== GATE-FX2 ===")
    print(f"  {'✓' if gate_post else '✗'} GATE-FX2-ESCALATION: "
          f"post_incident final_acc >= {GATE_POST_INCIDENT*100:.0f}%:  "
          f"{post_final*100:.2f}%")
    print(f"  {'✓' if gate_fat  else '✗'} GATE-FX2-FATIGUE:    "
          f"alert_fatigue final_acc >= {GATE_ALERT_FATIGUE*100:.0f}%:  "
          f"{fat_final*100:.2f}%")
    print(f"  {'✓' if gate_exp  else '✗'} GATE-FX2-EXPERTISE:  "
          f"expertise_grad final_acc >= {GATE_EXPERTISE_GRAD*100:.0f}%:  "
          f"{exp_final*100:.2f}%")
    print(f"\nGATE-FX2: {'✅ PASS' if gate_pass else '❌ FAIL'}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "gate":             "FX2",
        "pass":             gate_pass,
        "n_decisions":      N_DECISIONS,
        "n_seeds":          N_SEEDS,
        "tau":              TAU,
        "window_size":      WINDOW_SIZE,
        "incident_at":      INCIDENT_AT,
        "recovery_threshold_75": RECOVERY_THRESHOLD,
        "patterns": {
            pname: {
                "mean_drift":      round(r["mean_drift"],     6),
                "std_drift":       round(r["std_drift"],      6),
                "min_acc":         round(r["min_acc"],        6),
                "final_acc":       round(r["final_acc"],      6),
                "mean_bias_rate":  round(r["mean_bias_rate"], 6),
                "recovery_dec_75": r["recovery_dec_75"],
                "recovery_dec_85": r["recovery_dec_85"],
            }
            for pname, r in all_results.items()
        },
        "mean_trajectory": {
            pname: [round(v, 6) for v in r["mean_roll"]]
            for pname, r in all_results.items()
        },
        "std_trajectory": {
            pname: [round(v, 6) for v in r["std_roll"]]
            for pname, r in all_results.items()
        },
        "gate_checks": {
            "post_incident_final_ge_75":    gate_post,
            "alert_fatigue_final_ge_65":    gate_fat,
            "expertise_grad_final_ge_70":   gate_exp,
        },
        "references": {
            "fx1_corrected_static_realistic": REF_STATIC_REALISTIC,
            "fx1_learning_at_1000":           REF_LEARNING_AT_1000,
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    try:
        from experiments.fx2_noise_distributions.charts import generate_charts
        generate_charts(
            all_results=all_results,
            patterns=PATTERNS,
            n_decisions=N_DECISIONS,
            window_size=WINDOW_SIZE,
            incident_at=INCIDENT_AT,
            paper_dir=str(PAPER_DIR),
        )
    except Exception as exc:
        import traceback
        print(f"\nWarning: chart generation failed: {exc}")
        traceback.print_exc()

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
