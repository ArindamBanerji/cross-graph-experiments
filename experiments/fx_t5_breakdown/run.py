"""
FX-T5-BREAKDOWN: Per-action analysis of the auto-approve confidence band.
experiments/fx_t5_breakdown/run.py

CONTEXT:
Statistical pass confirmed auto-approve accuracy at 90.7% [90.1%, 91.2%] —
a marginal pass at p=0.024. The 0.90 threshold is symmetric across all four
actions, but suppress/monitor errors (missed threats) are 20× more costly than
escalate errors.

Q1: What action types are being auto-approved?
Q2: Within each auto-approved action, what is accuracy?
Q3: When auto-approve is wrong, what is the error direction?
    Are suppress/monitor decisions being incorrectly auto-approved?

COST_RATIO = 20.0 — incorrect suppress/monitor is 20× worse than escalate error.

Design verdict thresholds:
  dangerous_error_rate < 0.002 AND cost_weighted_acc > 0 → RESULT A (symmetric threshold ok)
  dangerous_error_rate < 0.005 AND cost_weighted_acc > 0 → RESULT B (asymmetric thresholds)
  otherwise                                               → RESULT C (LLM judge panel needed)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_SEEDS              = 50
N_ALERTS             = 2000
TAU                  = 0.1
MODE                 = "combined"
AUTO_APPROVE_THRESH  = 0.90
COST_RATIO           = 20.0

THREAT_ACTIONS   = {0, 1}   # escalate, investigate
CAUTION_ACTIONS  = {2, 3}   # suppress, monitor

THRESHOLD_SCAN   = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
TARGET_ACC_CAUTION = 0.99   # target accuracy for suppress/monitor auto-approve
TARGET_ACC_THREAT  = 0.90   # target accuracy for escalate/investigate auto-approve

EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
PAPER_DIR    = REPO_ROOT / "paper_figures"


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------

@dataclass
class BandDecision:
    predicted:  int    # action index
    gt:         int    # GT action index
    correct:    bool
    conf:       float
    category:   int    # category index
    error_type: str    # "correct", "dangerous", "safe", "over_escalation"


def classify_error(predicted: int, gt: int) -> str:
    if predicted == gt:
        return "correct"
    pred_is_caution = predicted in CAUTION_ACTIONS
    gt_is_threat    = gt in THREAT_ACTIONS
    if pred_is_caution and gt_is_threat:
        return "dangerous"       # said "fine", was actually a threat
    pred_is_threat  = predicted in THREAT_ACTIONS
    gt_is_caution   = gt in CAUTION_ACTIONS
    if pred_is_threat and gt_is_caution:
        return "over_escalation" # said "threat", was actually fine
    return "safe"                # within-tier confusion (both threat or both caution)


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> list[BandDecision]:
    """
    Score all alerts, return BandDecision records for every auto-approve decision.
    Static scorer (no learning) — measuring the threshold, not learning dynamics.
    """
    gen         = RealisticAlertGenerator(mode=MODE, seed=seed)
    alerts      = gen.generate(N_ALERTS)
    gt_profiles = gen.get_profiles()

    assert gen.categories == SOC_CATEGORIES, \
        f"Category mismatch at seed {seed}: {gen.categories}"

    scorer    = ProfileScorer(gt_profiles.copy(), tau=TAU)
    decisions: list[BandDecision] = []

    for a in alerts:
        result = scorer.score(a.factors, a.category_index)
        conf   = float(result.probabilities[result.action_index])
        if conf >= AUTO_APPROVE_THRESH:
            predicted = result.action_index
            gt        = a.gt_action_index
            correct   = (predicted == gt)
            decisions.append(BandDecision(
                predicted  = predicted,
                gt         = gt,
                correct    = correct,
                conf       = conf,
                category   = a.category_index,
                error_type = classify_error(predicted, gt),
            ))

    return decisions


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    seed:   int = 0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.array([
        float(rng.choice(arr, size=len(arr), replace=True).mean())
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def threshold_accuracy(
    decisions: list[BandDecision],
    action_idx: int,
    threshold: float,
) -> tuple[float, int]:
    """Return (accuracy, count) for action_idx decisions at conf >= threshold."""
    subset = [d for d in decisions if d.predicted == action_idx and d.conf >= threshold]
    if not subset:
        return float("nan"), 0
    acc = float(np.mean([d.correct for d in subset]))
    return acc, len(subset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== FX-T5-BREAKDOWN: Auto-Approve Band Action Analysis ===")
    print(f"N={N_SEEDS} seeds, {N_ALERTS} alerts/seed, τ={TAU}, {MODE} realistic")
    print(f"Auto-approve band: confidence ≥ {AUTO_APPROVE_THRESH}")
    print(f"Cost asymmetry: suppress/monitor error = {COST_RATIO:.0f}× escalate error\n")

    assert SOC_CATEGORIES == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ]
    assert SOC_ACTIONS == ["escalate", "investigate", "suppress", "monitor"]

    # -----------------------------------------------------------------------
    # Collect all decisions across 50 seeds
    # -----------------------------------------------------------------------
    all_decisions: list[BandDecision] = []
    # Per-seed, per-action accuracy (for bootstrapping CIs across seeds)
    per_seed_action_acc: dict[int, list[float]] = {i: [] for i in range(4)}
    overall_n_per_seed: list[int] = []

    for seed in range(N_SEEDS):
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  seed {seed+1}/{N_SEEDS} ...", flush=True)
        decs = run_seed(seed)
        all_decisions.extend(decs)
        overall_n_per_seed.append(len(decs))

        # Per-action accuracy this seed
        for act_idx in range(4):
            act_decs = [d for d in decs if d.predicted == act_idx]
            if act_decs:
                per_seed_action_acc[act_idx].append(
                    float(np.mean([d.correct for d in act_decs]))
                )

    total_band = len(all_decisions)
    print(f"\n  Total auto-approve decisions: {total_band:,} "
          f"(mean {total_band/N_SEEDS:.0f}/seed)")

    # -----------------------------------------------------------------------
    # Q1: Action distribution
    # -----------------------------------------------------------------------
    action_count = {i: sum(1 for d in all_decisions if d.predicted == i) for i in range(4)}
    action_pct   = {i: action_count[i] / total_band for i in range(4)}

    # -----------------------------------------------------------------------
    # Q2: Per-action accuracy with bootstrap CIs across seeds
    # -----------------------------------------------------------------------
    per_action_stats: dict[int, dict] = {}
    for act_idx in range(4):
        seed_accs = per_seed_action_acc[act_idx]
        if seed_accs:
            mean_acc = float(np.mean(seed_accs))
            ci_lo, ci_hi = bootstrap_ci(seed_accs, seed=act_idx)
        else:
            mean_acc = float("nan")
            ci_lo = ci_hi = float("nan")
        per_action_stats[act_idx] = {
            "count":      action_count[act_idx],
            "pct_band":   action_pct[act_idx],
            "accuracy":   mean_acc,
            "ci_lo":      ci_lo,
            "ci_hi":      ci_hi,
            "n_seeds":    len(seed_accs),
        }

    # -----------------------------------------------------------------------
    # Q3: Error direction
    # -----------------------------------------------------------------------
    errors         = [d for d in all_decisions if not d.correct]
    dangerous      = [d for d in errors if d.error_type == "dangerous"]
    safe_errs      = [d for d in errors if d.error_type == "safe"]
    over_esc       = [d for d in errors if d.error_type == "over_escalation"]

    total_errors         = len(errors)
    n_dangerous          = len(dangerous)
    n_safe               = len(safe_errs)
    n_over_esc           = len(over_esc)
    dangerous_error_rate = n_dangerous / total_band    # fraction of ALL band decisions

    # -----------------------------------------------------------------------
    # Cost-weighted analysis
    # -----------------------------------------------------------------------
    correct_decisions = total_band - total_errors
    other_errors      = n_safe + n_over_esc

    # Score: correct × 1, dangerous × -20, other errors × -1
    cost_weighted_score = (
        correct_decisions * 1.0
        - n_dangerous * COST_RATIO
        - other_errors * 1.0
    )
    cost_weighted_acc = cost_weighted_score / total_band
    nominal_acc       = correct_decisions / total_band

    # -----------------------------------------------------------------------
    # Recommended per-action thresholds
    # -----------------------------------------------------------------------
    recommended_thresholds: dict[int, float | str] = {}
    threshold_coverage: dict[int, dict[float, float]] = {i: {} for i in range(4)}

    for act_idx in range(4):
        act_all = [d for d in all_decisions if d.predicted == act_idx]
        target  = TARGET_ACC_CAUTION if act_idx in CAUTION_ACTIONS else TARGET_ACC_THREAT
        found   = False
        for thr in THRESHOLD_SCAN:
            subset = [d for d in act_all if d.conf >= thr]
            n_act_total = len(act_all)
            cov = len(subset) / n_act_total if n_act_total > 0 else 0.0
            threshold_coverage[act_idx][thr] = cov
            if len(subset) < 10:
                recommended_thresholds[act_idx] = "insufficient_data"
                found = True
                break
            acc = float(np.mean([d.correct for d in subset]))
            if acc >= target:
                recommended_thresholds[act_idx] = thr
                found = True
                break
        if not found:
            recommended_thresholds[act_idx] = "never_reaches_target"

    # -----------------------------------------------------------------------
    # Dangerous errors per category (which categories drive risk)
    # -----------------------------------------------------------------------
    dangerous_by_cat: dict[int, int] = {i: 0 for i in range(5)}
    for d in dangerous:
        dangerous_by_cat[d.category] += 1

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n--- Q1: Action Distribution in Auto-Approve Band ---")
    print(f"  (vs GT distribution in overall data)")
    overall_gt = {i: sum(1 for d in all_decisions if d.gt == i) for i in range(4)}
    for act_idx, act_name in enumerate(SOC_ACTIONS):
        flag = " ◄ high-cost" if act_idx in CAUTION_ACTIONS else ""
        print(f"  {act_name:<15} {action_pct[act_idx]*100:>5.1f}% of band"
              f"  ({action_count[act_idx]:>6,} decisions){flag}")

    print(f"\n--- Q2: Per-Action Accuracy in Auto-Approve Band ---")
    print(f"  {'Action':<15} {'Accuracy':>10} {'95% CI':>22} {'Count':>8}  {'Target':>8}")
    print("  " + "─" * 68)
    for act_idx, act_name in enumerate(SOC_ACTIONS):
        s      = per_action_stats[act_idx]
        target = TARGET_ACC_CAUTION if act_idx in CAUTION_ACTIONS else TARGET_ACC_THREAT
        flag   = ""
        if act_idx in CAUTION_ACTIONS and not np.isnan(s["accuracy"]):
            flag = " ⚠" if s["accuracy"] < 0.99 else " ✓"
        ci_str = (f"[{s['ci_lo']*100:.1f}%, {s['ci_hi']*100:.1f}%]"
                  if not np.isnan(s["ci_lo"]) else "   n/a   ")
        print(f"  {act_name:<15} {s['accuracy']*100:>9.2f}%  {ci_str:>22}"
              f"  {s['count']:>7,}  {target*100:>7.0f}%{flag}")

    print(f"\n--- Q3: Error Direction Analysis ---")
    print(f"  Total errors in band: {total_errors:,} "
          f"({total_errors/total_band*100:.2f}% of {total_band:,} band decisions)")
    if total_errors > 0:
        print(f"  Dangerous (suppress/monitor predicted, GT=escalate/investigate):")
        print(f"    Count:             {n_dangerous:,}")
        print(f"    Rate (of band):    {dangerous_error_rate*100:.3f}%")
        print(f"    Share of errors:   {n_dangerous/total_errors*100:.1f}%")
        print(f"  Safe (within-tier confusion):")
        print(f"    Count:             {n_safe:,}  ({n_safe/total_errors*100:.1f}% of errors)")
        print(f"  Over-escalation (escalate, GT=suppress/monitor):")
        print(f"    Count:             {n_over_esc:,}  ({n_over_esc/total_errors*100:.1f}% of errors)")

        if n_dangerous > 0:
            print(f"\n  Dangerous errors by category:")
            for cat_idx, cat_name in enumerate(SOC_CATEGORIES):
                n = dangerous_by_cat[cat_idx]
                if n > 0:
                    print(f"    {cat_name:<28} {n:>4} ({n/n_dangerous*100:.1f}%)")

    print(f"\n--- Cost-Weighted Analysis (COST_RATIO={COST_RATIO:.0f}:1) ---")
    print(f"  Correct decisions:      {correct_decisions:>8,}  × +1.0 = {correct_decisions:>+12,.1f}")
    print(f"  Dangerous errors:       {n_dangerous:>8,}  × -{COST_RATIO:.0f}  = {-n_dangerous*COST_RATIO:>+12,.1f}")
    print(f"  Other errors:           {other_errors:>8,}  × -1.0 = {-other_errors:>+12,.1f}")
    print(f"  Net score:              {cost_weighted_score:>+12,.1f}")
    print(f"  Nominal accuracy:       {nominal_acc*100:.2f}%")
    print(f"  Cost-weighted score:    {cost_weighted_acc:>+.4f}")
    if cost_weighted_acc < 0:
        print(f"  ⚠ NEGATIVE: auto-approve is net-harmful under {COST_RATIO:.0f}:1 cost ratio")
    elif cost_weighted_acc < 0.50:
        print(f"  ⚠ MARGINAL: cost-weighted score < 0.5 — review thresholds")
    else:
        print(f"  ✓ Positive cost-weighted score — auto-approve viable under {COST_RATIO:.0f}:1")

    print(f"\n--- Recommended Per-Action Thresholds ---")
    print(f"  {'Action':<15} {'Recommended':>14}  {'Coverage at thr':>18}  {'Target acc':>11}")
    print("  " + "─" * 64)
    for act_idx, act_name in enumerate(SOC_ACTIONS):
        thr     = recommended_thresholds[act_idx]
        target  = TARGET_ACC_CAUTION if act_idx in CAUTION_ACTIONS else TARGET_ACC_THREAT
        current = ("SAME as 0.90" if thr == 0.90
                   else f"RAISE to {thr}" if isinstance(thr, float)
                   else thr)
        if isinstance(thr, float):
            cov = threshold_coverage[act_idx].get(thr, float("nan"))
            cov_str = f"{cov*100:.1f}% of {act_name} decisions"
        else:
            cov_str = "—"
        print(f"  {act_name:<15} {current:>14}  {cov_str:>22}  {target*100:.0f}%")

    # -----------------------------------------------------------------------
    # Design verdict
    # -----------------------------------------------------------------------
    print(f"\n=== DESIGN VERDICT ===")
    print(f"  Dangerous error rate: {dangerous_error_rate*100:.3f}% of band decisions")
    print(f"  Cost-weighted score:  {cost_weighted_acc:+.4f}")

    if dangerous_error_rate < 0.002 and cost_weighted_acc > 0:
        verdict = "A"
        print("RESULT A: Symmetric threshold viable.")
        print("  Dangerous error rate < 0.2%. Cost-weighted score positive.")
        print("  Proceed with 0.90 global threshold + credential_access 0.95 override.")
        print("  No LLM judge panel needed.")
    elif dangerous_error_rate < 0.005 and cost_weighted_acc > 0:
        verdict = "B"
        print("RESULT B: Asymmetric thresholds needed but no judge panel.")
        print("  Dangerous error rate 0.2–0.5%. Use recommended per-action thresholds.")
        print("  No LLM judge panel needed — data is sufficient.")
    else:
        verdict = "C"
        print("RESULT C: LLM judge panel required.")
        print("  Dangerous error rate > 0.5% OR cost-weighted score ≤ 0.")
        print("  Auto-approve design needs architectural review before SOC-PROF-2.")
        print("  Convene: Opus + GPT-5 review of threshold design.")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):  return bool(obj)
            if isinstance(obj, np.integer):   return int(obj)
            if isinstance(obj, np.floating):  return float(obj)
            return super().default(obj)

    output = {
        "experiment":         "FX-T5-BREAKDOWN",
        "n_seeds":            N_SEEDS,
        "n_alerts":           N_ALERTS,
        "tau":                TAU,
        "mode":               MODE,
        "auto_approve_threshold": AUTO_APPROVE_THRESH,
        "cost_ratio":         COST_RATIO,
        "total_band_decisions": total_band,
        "mean_band_per_seed": total_band / N_SEEDS,
        "q1_action_distribution": {
            SOC_ACTIONS[i]: {
                "count":    action_count[i],
                "pct_band": round(action_pct[i], 6),
            }
            for i in range(4)
        },
        "q2_per_action_accuracy": {
            SOC_ACTIONS[i]: {
                "count":    per_action_stats[i]["count"],
                "pct_band": round(per_action_stats[i]["pct_band"], 6),
                "accuracy": round(per_action_stats[i]["accuracy"], 6),
                "ci_lo":    round(per_action_stats[i]["ci_lo"], 6),
                "ci_hi":    round(per_action_stats[i]["ci_hi"], 6),
            }
            for i in range(4)
        },
        "q3_error_direction": {
            "total_errors":           total_errors,
            "dangerous":              n_dangerous,
            "safe":                   n_safe,
            "over_escalation":        n_over_esc,
            "dangerous_error_rate":   round(dangerous_error_rate, 8),
            "dangerous_pct_of_errors": round(n_dangerous / total_errors, 6)
                                       if total_errors > 0 else 0.0,
            "dangerous_by_category": {
                SOC_CATEGORIES[i]: dangerous_by_cat[i] for i in range(5)
            },
        },
        "cost_analysis": {
            "correct_decisions":    correct_decisions,
            "dangerous_errors":     n_dangerous,
            "other_errors":         other_errors,
            "cost_weighted_score":  round(cost_weighted_score, 4),
            "cost_weighted_acc":    round(cost_weighted_acc, 6),
            "nominal_acc":          round(nominal_acc, 6),
        },
        "recommended_thresholds": {
            SOC_ACTIONS[i]: (float(recommended_thresholds[i])
                             if isinstance(recommended_thresholds[i], float)
                             else recommended_thresholds[i])
            for i in range(4)
        },
        "threshold_scan": {
            SOC_ACTIONS[i]: {
                str(thr): {
                    "acc":      round(threshold_accuracy(all_decisions, i, thr)[0], 6),
                    "count":    threshold_accuracy(all_decisions, i, thr)[1],
                    "coverage": round(threshold_coverage[i].get(thr, float("nan")), 6),
                }
                for thr in THRESHOLD_SCAN
            }
            for i in range(4)
        },
        "design_verdict": verdict,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    try:
        from experiments.fx_t5_breakdown.charts import generate_charts
        generate_charts(
            action_count=action_count,
            action_pct=action_pct,
            per_action_stats=per_action_stats,
            total_band=total_band,
            n_dangerous=n_dangerous,
            n_safe=n_safe,
            n_over_esc=n_over_esc,
            total_errors=total_errors,
            correct_decisions=correct_decisions,
            dangerous_error_rate=dangerous_error_rate,
            cost_weighted_acc=cost_weighted_acc,
            cost_weighted_score=cost_weighted_score,
            recommended_thresholds=recommended_thresholds,
            threshold_coverage=threshold_coverage,
            all_decisions=all_decisions,
            paper_dir=str(PAPER_DIR),
        )
    except Exception as exc:
        import traceback
        print(f"\nWarning: chart generation failed: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
    sys.exit(0)
