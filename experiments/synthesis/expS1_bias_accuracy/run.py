"""
EXP-S1: Synthesis Bias Accuracy (warm-start design)
experiments/synthesis/expS1_bias_accuracy/run.py

QUESTION: Does Eq. 4-synthesis (σ bias) improve scoring accuracy over Eq. 4-final?

DESIGN (warm-start):
  - ProfileScorer initialized with GT profiles (tau=0.1) — near-optimal start.
  - 500 alerts scored per seed per lambda.
  - Synthesis bias from 10 correct claims projected through RuleBasedProjector.
  - Compare accuracy and ECE across lambda in [0.0, 0.05, 0.1, 0.2].

GATE-S1:
  improvement >= 3pp AND p < 0.05 AND ECE degradation <= 0.02

Uses src/synthesis/ modules created in SYNTH-EXP-0.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
# Paths
# ---------------------------------------------------------------------------
EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
GATE_PATH    = EXP_DIR / "gate_result.json"
CSV_PATH     = EXP_DIR / "results.csv"
PAPER_DIR    = REPO_ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(REPO_ROOT / "configs" / "default.yaml") as _f:
    _cfg = yaml.safe_load(_f)

LAMBDA_VALUES  = [0.0, 0.05, 0.1, 0.2]
SEEDS          = _cfg["synthesis"]["seeds"]           # 10 seeds
N_ALERTS       = _cfg["synthesis"]["n_decisions"]     # 500
N_CLAIM_CATS   = 6    # generate_correct_claims default (n_categories=6)
N_CATS         = len(CATEGORIES)                      # 5
N_ACTS         = len(ACTIONS)                         # 4


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    correct_flags: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Standard equal-width 10-bin Expected Calibration Error."""
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        conf = float(confidences[mask].mean())
        acc  = float(correct_flags[mask].astype(float).mean())
        ece += mask.sum() * abs(acc - conf)
    return ece / n


# ---------------------------------------------------------------------------
# Single-seed runner
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


def run_one_seed(seed: int, lambda_val: float, projector: RuleBasedProjector) -> Dict:
    """
    Warm-start design:
      1. Initialize ProfileScorer with GT profiles from config (tau=0.1).
      2. Generate 10 GT-aligned correct claims; project to SynthesisBias at lambda_val.
      3. Score N_ALERTS alerts.  No online updates — scorer is read-only.
      4. Return accuracy, ECE, per-category accuracy, action distribution.
    """
    gen    = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_ALERTS)

    n_factors   = len(gen.factors)
    gt_profiles = build_gt_array(gen)

    # Warm-start: init from GT profiles (near-optimal starting point)
    scorer = ProfileScorer(N_CATS, N_ACTS, n_factors, tau=0.1)
    scorer.init_from_profiles(gen.profiles, gen.categories, gen.actions)

    # Build synthesis bias using GT-aligned claims
    claims = generate_correct_claims(10, seed=seed, gt_profiles=gt_profiles,
                                     n_categories=N_CATS, n_actions=N_ACTS)
    bias   = projector.project(
        claims,
        n_categories=N_CATS,
        n_actions=N_ACTS,
        lambda_coupling=lambda_val,
    )

    # Score
    confs   = []
    correct = []
    per_cat_correct: Dict[int, int] = defaultdict(int)
    per_cat_total:   Dict[int, int] = defaultdict(int)
    act_counts:      Dict[int, int] = defaultdict(int)

    for alert in alerts:
        result = scorer.score(
            alert.factors,
            alert.category_index,
            synthesis=bias if lambda_val > 0.0 else None,
            lambda_coupling=lambda_val,
        )
        is_correct = int(result.action_index == alert.gt_action_index)
        confs.append(float(result.probabilities[result.action_index]))
        correct.append(is_correct)
        per_cat_correct[alert.category_index] += is_correct
        per_cat_total[alert.category_index]   += 1
        act_counts[result.action_index]        += 1

    confs_arr   = np.array(confs)
    correct_arr = np.array(correct, dtype=bool)

    accuracy = float(np.mean(correct_arr)) * 100.0
    ece      = compute_ece(confs_arr, correct_arr)

    per_cat_acc: Dict[str, float] = {}
    for cat_name in CATEGORIES:
        cat_idx = gen.categories.index(cat_name)
        total   = per_cat_total[cat_idx]
        per_cat_acc[cat_name] = (
            per_cat_correct[cat_idx] / total * 100.0 if total > 0 else 0.0
        )

    action_dist: Dict[str, float] = {
        act: act_counts[j] / N_ALERTS
        for j, act in enumerate(ACTIONS)
    }

    return {
        "accuracy":              accuracy,
        "ece":                   ece,
        "per_category_accuracy": per_cat_acc,
        "action_distribution":   action_dist,
        "n_alerts":              N_ALERTS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("EXP-S1: Synthesis Bias Accuracy (warm-start design)")
    print(f"  lambda values: {LAMBDA_VALUES}")
    print(f"  seeds:         {len(SEEDS)}")
    print(f"  n_alerts:      {N_ALERTS}")
    print()

    projector = RuleBasedProjector()

    all_results: Dict[str, List[Dict]] = {}
    csv_rows: List[Dict] = []

    for lam in LAMBDA_VALUES:
        lam_key = f"lambda_{lam:.3f}"
        all_results[lam_key] = []
        for seed in SEEDS:
            r = run_one_seed(seed, lam, projector)
            all_results[lam_key].append(r)
            csv_rows.append({
                "lambda": lam,
                "seed":   seed,
                "accuracy": round(r["accuracy"], 4),
                "ece":      round(r["ece"], 6),
                **{f"acc_{cat}": round(r["per_category_accuracy"][cat], 4)
                   for cat in CATEGORIES},
                **{f"act_{act}": round(r["action_distribution"][act], 4)
                   for act in ACTIONS},
            })
            print(f"  lambda={lam:.3f}  seed={seed:4d}  "
                  f"acc={r['accuracy']:.2f}%  ece={r['ece']:.4f}")
        print()

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    aggregated: Dict[str, Dict] = {}
    for lam in LAMBDA_VALUES:
        lam_key     = f"lambda_{lam:.3f}"
        seed_res    = all_results[lam_key]
        accs        = [r["accuracy"] for r in seed_res]
        eces        = [r["ece"]      for r in seed_res]

        per_cat_mean: Dict[str, float] = {}
        for cat in CATEGORIES:
            per_cat_mean[cat] = float(
                np.mean([r["per_category_accuracy"][cat] for r in seed_res])
            )

        # Mean action distribution across seeds
        mean_act_dist: Dict[str, float] = {}
        for act in ACTIONS:
            mean_act_dist[act] = float(
                np.mean([r["action_distribution"][act] for r in seed_res])
            )

        aggregated[lam_key] = {
            "mean_accuracy":              float(np.mean(accs)),
            "std_accuracy":               float(np.std(accs)),
            "mean_ece":                   float(np.mean(eces)),
            "std_ece":                    float(np.std(eces)),
            "per_seed_accuracies":        accs,
            "per_category_mean_accuracy": per_cat_mean,
            "mean_action_distribution":   mean_act_dist,
        }

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------
    baseline_key  = "lambda_0.000"
    baseline_accs = np.array(aggregated[baseline_key]["per_seed_accuracies"])
    baseline_ece  = aggregated[baseline_key]["mean_ece"]

    best_lam        = None
    best_improvement = -1e9
    best_p          = 1.0

    print("Statistical analysis:")
    for lam in LAMBDA_VALUES[1:]:
        lam_key       = f"lambda_{lam:.3f}"
        treat_accs    = np.array(aggregated[lam_key]["per_seed_accuracies"])
        improvement   = float(np.mean(treat_accs) - np.mean(baseline_accs))
        t_stat, p_val = stats.ttest_rel(treat_accs, baseline_accs)
        if np.isnan(p_val):
            p_val = 1.0
        p_val = float(p_val)
        print(f"  lambda={lam:.3f}: improvement={improvement:+.2f}pp, "
              f"t={t_stat:.3f}, p={p_val:.4f}")
        if improvement > best_improvement:
            best_improvement = improvement
            best_lam         = lam
            best_p           = p_val

    if best_lam is None:
        best_lam = LAMBDA_VALUES[1]

    best_ece  = aggregated[f"lambda_{best_lam:.3f}"]["mean_ece"]
    ece_delta = best_ece - baseline_ece

    gate_pass = (
        best_improvement >= 3.0
        and best_p < 0.05
        and ece_delta <= 0.02
    )

    print(f"\n=== GATE-S1 ===")
    print(f"Best lambda: {best_lam}")
    print(f"Improvement: {best_improvement:+.2f}pp (gate: >=3pp)")
    print(f"p-value: {best_p:.4f} (gate: <0.05)")
    print(f"ECE delta: {ece_delta:+.4f} (gate: <=0.02)")
    if gate_pass:
        print("GATE-S1: PASS")
    else:
        print("GATE-S1: FAIL")
        if best_improvement < 3.0:
            print(f"  REASON: improvement {best_improvement:+.2f}pp < 3pp threshold")
        if best_p >= 0.05:
            print(f"  REASON: p={best_p:.4f} not significant")
        if ece_delta > 0.02:
            print(f"  REASON: ECE degraded {ece_delta:+.4f} > 0.02")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "experiment": "EXP-S1",
        "design":     "warm_start",
        "per_lambda": all_results,
        "aggregated": aggregated,
        "statistical_test": {
            "best_lambda":            best_lam,
            "improvement_pp":         best_improvement,
            "p_value":                best_p,
            "ece_delta":              ece_delta,
            "baseline_mean_accuracy": aggregated[baseline_key]["mean_accuracy"],
            "best_mean_accuracy":     aggregated[f"lambda_{best_lam:.3f}"]["mean_accuracy"],
        },
        "gate": {
            "S1_passed":   gate_pass,
            "improvement_pp": best_improvement,
            "p_value":     best_p,
            "ece_delta":   ece_delta,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {RESULTS_PATH}")

    # CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV saved to {CSV_PATH}")

    # Gate result
    gate_result = {
        "gate":               "S1",
        "pass":               gate_pass,
        "best_lambda":        best_lam,
        "improvement_pp":     best_improvement,
        "p_value":            best_p,
        "ece_delta":          ece_delta,
        "baseline_accuracy":  aggregated[baseline_key]["mean_accuracy"],
        "best_accuracy":      aggregated[f"lambda_{best_lam:.3f}"]["mean_accuracy"],
    }
    with open(GATE_PATH, "w") as f:
        json.dump(gate_result, f, indent=2, default=float)
    print(f"Gate result saved to {GATE_PATH}")

    # Charts
    try:
        import importlib
        charts_mod = importlib.import_module(
            "experiments.synthesis.expS1_bias_accuracy.charts"
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
