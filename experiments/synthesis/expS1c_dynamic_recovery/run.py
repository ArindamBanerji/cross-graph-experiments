"""
EXP-S1c: Synthesis Bias — Dynamic Cold-Start Recovery Test
experiments/synthesis/expS1c_dynamic_recovery/run.py

QUESTION: Does sigma accelerate cold-start recovery when Loop 2 (centroid learning) is active?

CONTEXT:
EXP-S1 (warm-start) and EXP-S1b (degraded static) both showed sigma has no effect
on static accuracy because:
- Warm start: profiles are near-optimal; no room to improve.
- Cold start: random centroid L2 distances have high variance; small sigma is drowned out.

The real operating context for sigma is Loop 2 — sigma biases each decision TOWARD
the GT-preferred action, yielding better oracle feedback, which accelerates centroid
learning. This tests SPEED of recovery rather than static accuracy.

DESIGN:
  - 4 lambda values x 10 seeds x 2000 decisions per run
  - Cold start: random centroids (uniform [0,1])
  - Loop 2: ProfileScorer.update() called after each decision (oracle_rate=0.85)
  - Oracle: noisy GT-conditioned (85% accurate — signals correctly 85% of the time)
  - Recovery metric: decisions needed for rolling-50 accuracy to first exceed 90%

GATE-S1c:
  sigma reaches 90% rolling accuracy >= 50 decisions faster than baseline,
  at p < 0.05 (paired t-test across 10 seeds).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.category_alert_generator import CategoryAlertGenerator, CATEGORIES, ACTIONS
from src.models.profile_scorer import ProfileScorer
from src.synthesis.claim_generator import generate_correct_claims
from src.synthesis.rule_projector import RuleBasedProjector

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
GATE_PATH    = EXP_DIR / "gate_result.json"
CSV_PATH     = EXP_DIR / "results.csv"
PAPER_DIR    = REPO_ROOT / "paper_figures"

with open(REPO_ROOT / "configs" / "default.yaml") as _f:
    _cfg = yaml.safe_load(_f)

SEEDS            = _cfg["synthesis"]["seeds"]        # 10 seeds
LAMBDA_VALUES    = [0.0, 0.1, 0.2, 0.5]
ACCURACY_TARGET  = 0.90
WINDOW_SIZE      = 50
MAX_DECISIONS    = 2000
N_CATS           = len(CATEGORIES)    # 5
N_ACTS           = len(ACTIONS)       # 4
ORACLE_RATE      = 0.85               # oracle accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_gt_array(gen: CategoryAlertGenerator) -> np.ndarray:
    """Extract (n_cats, n_acts, n_factors) numpy array from gen.profiles dict."""
    n_cats  = len(gen.categories)
    n_acts  = len(gen.actions)
    n_facts = len(gen.factors)
    arr = np.zeros((n_cats, n_acts, n_facts), dtype=np.float64)
    for ci, cat in enumerate(gen.categories):
        for ai, act in enumerate(gen.actions):
            arr[ci, ai, :] = gen.profiles[cat][act]
    return arr


def run_one_seed(
    seed: int,
    lambda_val: float,
    projector: RuleBasedProjector,
) -> Dict:
    """
    Simulate Loop 2 dynamic scoring from cold start.

    Returns
    -------
    dict with keys:
        recovery_decision  int or None — first decision where rolling-50 >= 90%
        final_acc          float — mean accuracy over last 100 decisions
        curve              list[float] — rolling-50 accuracy (%) at every WINDOW_SIZE steps
        never_recovered    bool
    """
    gen         = CategoryAlertGenerator(seed=seed)
    alerts      = gen.generate(MAX_DECISIONS)
    gt_profiles = build_gt_array(gen)

    # Cold start: random centroids
    rng_init = np.random.default_rng(seed + 1000)
    random_mu = rng_init.uniform(0, 1, gt_profiles.shape).astype(np.float64)
    scorer = ProfileScorer(random_mu, tau=0.1)

    # GT-aligned claims -> synthesis bias
    claims = generate_correct_claims(
        10, seed=seed, gt_profiles=gt_profiles,
        n_categories=N_CATS, n_actions=N_ACTS,
    )
    bias = projector.project(
        claims, n_categories=N_CATS, n_actions=N_ACTS,
        lambda_coupling=lambda_val,
    )

    # Noisy oracle RNG — independent from alert/init seeds
    rng_oracle = np.random.default_rng(seed * 10000)

    rolling_correct: List[int] = []
    recovery_decision: Optional[int] = None
    curve: List[float] = []            # rolling-50 accuracy at checkpoints

    for i, alert in enumerate(alerts):
        result = scorer.score(
            alert.factors,
            alert.category_index,
            synthesis=bias if lambda_val > 0.0 else None,
            lambda_coupling=lambda_val,
        )

        hit = (result.action_index == alert.gt_action_index)
        rolling_correct.append(int(hit))

        # Noisy GT-conditioned oracle (85% accurate)
        oracle_flip    = rng_oracle.random() < (1.0 - ORACLE_RATE)
        oracle_correct = hit ^ oracle_flip   # bool XOR bool

        # Loop 2: update centroid based on oracle signal
        scorer.update(alert.factors, alert.category_index,
                      result.action_index, oracle_correct)

        # Recovery check (requires at least WINDOW_SIZE decisions)
        if len(rolling_correct) >= WINDOW_SIZE:
            window_acc = float(np.mean(rolling_correct[-WINDOW_SIZE:]))
            if recovery_decision is None and window_acc >= ACCURACY_TARGET:
                recovery_decision = i + 1

        # Accuracy checkpoint every WINDOW_SIZE decisions
        if (i + 1) % WINDOW_SIZE == 0:
            start = max(0, i + 1 - WINDOW_SIZE)
            window_acc = float(np.mean(rolling_correct[start:i + 1]))
            curve.append(round(window_acc * 100.0, 4))

    final_acc = float(np.mean(rolling_correct[-100:])) * 100.0 if len(rolling_correct) >= 100 \
                else float(np.mean(rolling_correct)) * 100.0

    return {
        "recovery_decision": recovery_decision,
        "final_acc":         final_acc,
        "curve":             curve,
        "never_recovered":   recovery_decision is None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("EXP-S1c: Dynamic Cold-Start Recovery Test")
    print(f"  lambda values:   {LAMBDA_VALUES}")
    print(f"  seeds:           {len(SEEDS)}")
    print(f"  max_decisions:   {MAX_DECISIONS}")
    print(f"  window_size:     {WINDOW_SIZE}")
    print(f"  accuracy_target: {ACCURACY_TARGET:.0%}")
    print(f"  oracle_rate:     {ORACLE_RATE:.0%}")
    print()

    projector = RuleBasedProjector()

    # results[lam_key][seed_idx] = {recovery_decision, final_acc, curve}
    all_results: Dict[str, List[Dict]] = {}
    csv_rows:    List[Dict]            = []

    for lam in LAMBDA_VALUES:
        lam_key = f"lambda_{lam:.3f}"
        all_results[lam_key] = []
        print(f"--- lambda={lam:.2f} ---")
        for seed in SEEDS:
            r = run_one_seed(seed, lam, projector)
            all_results[lam_key].append(r)
            rd_str = str(r["recovery_decision"]) if r["recovery_decision"] else "NEVER"
            print(f"  seed={seed:4d}  recovery={rd_str:>5}  "
                  f"final_acc={r['final_acc']:.2f}%  "
                  f"never_recovered={r['never_recovered']}")

            csv_rows.append({
                "lambda":            lam,
                "seed":              seed,
                "recovery_decision": r["recovery_decision"] if r["recovery_decision"] else MAX_DECISIONS,
                "never_recovered":   int(r["never_recovered"]),
                "final_acc":         round(r["final_acc"], 4),
            })
        print()

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    aggregated: Dict[str, Dict] = {}
    for lam in LAMBDA_VALUES:
        lam_key  = f"lambda_{lam:.3f}"
        res_list = all_results[lam_key]
        # never-recovered → MAX_DECISIONS for statistics
        rds    = [r["recovery_decision"] if r["recovery_decision"] else MAX_DECISIONS
                  for r in res_list]
        faccs  = [r["final_acc"] for r in res_list]
        n_never = sum(1 for r in res_list if r["never_recovered"])
        aggregated[lam_key] = {
            "mean_recovery":   float(np.mean(rds)),
            "std_recovery":    float(np.std(rds)),
            "mean_final_acc":  float(np.mean(faccs)),
            "std_final_acc":   float(np.std(faccs)),
            "never_recovered": n_never,
            "recovery_decisions_per_seed": rds,
            "final_accs_per_seed":         faccs,
        }

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------
    baseline_key = "lambda_0.000"
    baseline_rds = np.array(aggregated[baseline_key]["recovery_decisions_per_seed"])
    baseline_mean = float(np.mean(baseline_rds))

    stat_results: Dict[str, Dict] = {}
    print("\n=== EXP-S1c RESULTS ===")
    print(f"{'lambda':<10}  {'mean_recovery':>14}  {'improvement':>12}  {'p_value':>10}  {'never_rec':>10}  {'final_acc':>10}")
    print("-" * 80)
    print(f"{'0.00 (base)':<10}  {baseline_mean:>14.1f}  {'—':>12}  {'—':>10}  "
          f"{aggregated[baseline_key]['never_recovered']:>10}  "
          f"{aggregated[baseline_key]['mean_final_acc']:>9.2f}%")

    gate_pass   = False
    best_result: Optional[Dict] = None

    for lam in LAMBDA_VALUES[1:]:
        lam_key    = f"lambda_{lam:.3f}"
        treat_rds  = np.array(aggregated[lam_key]["recovery_decisions_per_seed"])
        improvement = baseline_mean - float(np.mean(treat_rds))   # positive = faster
        t_stat, p_val = stats.ttest_rel(baseline_rds, treat_rds)
        if np.isnan(p_val):
            p_val = 1.0
        p_val = float(p_val)
        sig = "  *** SIG" if p_val < 0.05 else ""

        stat_results[lam_key] = {
            "improvement_decisions": improvement,
            "p_value":               p_val,
            "t_stat":                float(t_stat),
            "baseline_mean":         baseline_mean,
            "treatment_mean":        float(np.mean(treat_rds)),
        }

        print(f"lambda={lam:.2f}    {float(np.mean(treat_rds)):>14.1f}  "
              f"{improvement:>+11.1f}d  {p_val:>10.4f}  "
              f"{aggregated[lam_key]['never_recovered']:>10}  "
              f"{aggregated[lam_key]['mean_final_acc']:>9.2f}%{sig}")

        if improvement >= 50.0 and p_val < 0.05:
            gate_pass = True
            if best_result is None or improvement > best_result["improvement_decisions"]:
                best_result = {
                    "lambda":                lam,
                    "improvement_decisions": improvement,
                    "p_value":               p_val,
                    "baseline_mean_recovery": baseline_mean,
                    "treatment_mean_recovery": float(np.mean(treat_rds)),
                }

    print()
    print("=== GATE-S1c ===")
    print("Gate: sigma reaches 90% accuracy >= 50 decisions faster, p<0.05")
    if gate_pass and best_result:
        print("GATE-S1c: PASS")
        print(f"  Best: lambda={best_result['lambda']:.2f}")
        print(f"  Improvement: {best_result['improvement_decisions']:+.1f} decisions faster")
        print(f"  p={best_result['p_value']:.4f}")
        print(f"  Baseline mean recovery: {best_result['baseline_mean_recovery']:.1f} decisions")
        print(f"  With sigma:             {best_result['treatment_mean_recovery']:.1f} decisions")
        interp = (
            f"sigma (lambda={best_result['lambda']:.2f}) reduces cold-start recovery time "
            f"by {best_result['improvement_decisions']:.0f} decisions (p={best_result['p_value']:.4f}). "
            f"Confirms sigma's role as Loop 2 accelerator during cold-start."
        )
    else:
        print("GATE-S1c: FAIL")
        print("  sigma does not significantly reduce recovery time.")
        print("  The sigma bias is too small relative to random-centroid distance variance.")
        interp = (
            "sigma (all lambdas) does not accelerate cold-start recovery by >=50 decisions at p<0.05. "
            "Combined with EXP-S1 and EXP-S1b: sigma's benefit is in confidence shaping "
            "and briefing display, not raw accuracy or recovery speed."
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Drop per-seed curves from aggregated (keep in per_lambda)
    results = {
        "experiment":  "EXP-S1c",
        "design":      "dynamic_cold_start_recovery",
        "per_lambda":  all_results,
        "aggregated":  aggregated,
        "statistical": stat_results,
        "gate": {
            "S1c_passed": gate_pass,
            "best_result": best_result,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {RESULTS_PATH}")

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV saved to {CSV_PATH}  ({len(csv_rows)} rows)")

    gate_result = {
        "gate":             "S1c",
        "pass":             gate_pass,
        "best_result":      best_result,
        "interpretation":   interp,
        "s1_finding":       "warm_start_ceiling",
        "s1b_finding":      "degraded_no_improvement",
        "s1c_finding":      "sigma_helps_loop2" if gate_pass else "sigma_no_loop2_benefit",
    }
    with open(GATE_PATH, "w") as f:
        json.dump(gate_result, f, indent=2, default=float)
    print(f"Gate result saved to {GATE_PATH}")

    # Charts
    try:
        import importlib
        charts_mod = importlib.import_module(
            "experiments.synthesis.expS1c_dynamic_recovery.charts"
        )
        charts_mod.make_all_charts(results, str(EXP_DIR), str(PAPER_DIR))
        print("Charts saved to paper_figures/")
    except Exception as e:
        print(f"Warning: chart generation failed: {e}")
        import traceback; traceback.print_exc()

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
