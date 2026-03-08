"""
EXP-S3: Loop Independence
experiments/synthesis/expS3_loop_independence/run.py

QUESTION: Does σ contaminate μ through the indirect action-selection path?
GATE-S3: relative_frobenius <= 5% AND accuracy_diff <= 1pp

This is the most architecturally critical experiment. Even though
ProfileScorer.update() has no σ parameter (direct firewall), there is
an indirect path: σ biases action selection → different outcomes occur
→ different (f, action, correct) tuples fed to update() → centroid drift.

If S3 FAILS (>5% Frobenius divergence):
  - Loop 2 centroid updates need a counterfactual-aware mode
  - ProfileScorer.update() API will need to change
  - This changes v5.0 code — better to know before writing it

Design: intelligence_layer_design_v1 §4 EXP-S3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.synthesis import SynthesisBias
from src.models.rule_projector import RuleBasedProjector
from src.data.claim_generator import (
    SOC_SYNTHESIS_RULES, CATEGORIES, ACTIONS, generate_correct_claims
)
from src.viz.synthesis_common import save_results, print_gate_result

RESULTS_PATH = Path(__file__).parent / "results.json"
PAPER_FIG_DIR = REPO_ROOT / "paper_figures"
EXP_DIR = Path(__file__).parent

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
N_DECISIONS = 300    # Training decisions (with/without synthesis)
N_TEST = 200         # Test decisions (centroids-alone, no synthesis)
SNAPSHOT_STEPS = [50, 100, 150, 200, 250, 300]


def _load_best_lambda() -> float:
    s1_path = REPO_ROOT / "experiments" / "synthesis" / "expS1_bias_accuracy" / "results.json"
    if s1_path.exists():
        with open(s1_path) as f:
            s1 = json.load(f)
        best = s1.get("statistical_test", {}).get("best_lambda", 0.1)
        return float(best) if best is not None else 0.1
    return 0.1


def run_one_seed(
    seed: int,
    lambda_val: float,
    projector: RuleBasedProjector,
) -> Dict:
    """
    Run CONDITION A (with synthesis) and CONDITION B (without) on same alert sequence.
    Compare final centroids via Frobenius norm.
    """
    from src.data.category_alert_generator import CategoryAlertGenerator
    from src.models.profile_scorer import ProfileScorer

    gen = CategoryAlertGenerator(seed=seed)
    # Use SAME alert sequence for both conditions
    train_alerts = gen.generate(N_DECISIONS)
    test_alerts  = gen.generate(N_TEST)            # Fresh test set
    mu_initial   = np.array(
        [[gen.profiles[cat][act] for act in gen.actions] for cat in gen.categories]
    ).copy()

    claims = generate_correct_claims(n=10, seed=seed)

    # ----------------------------------------------------------------
    # CONDITION A: WITH synthesis (λ > 0, σ active)
    # ----------------------------------------------------------------
    scorer_A = ProfileScorer(mu_initial.copy(), ACTIONS)
    bias = projector.project(claims, lambda_coupling=lambda_val)

    snapshots_A = {}
    for step, alert in enumerate(train_alerts):
        result_A = scorer_A.score(alert.factors, alert.category_index, bias)
        is_correct_A = (result_A.action_index == alert.gt_action_index)
        scorer_A.update(
            alert.factors, alert.category_index,
            result_A.action_index, is_correct_A
        )
        if (step + 1) in SNAPSHOT_STEPS:
            snapshots_A[step + 1] = scorer_A.mu.copy()

    mu_final_A = scorer_A.mu.copy()

    # ----------------------------------------------------------------
    # CONDITION B: WITHOUT synthesis (λ = 0)
    # ----------------------------------------------------------------
    scorer_B = ProfileScorer(mu_initial.copy(), ACTIONS)  # Fresh copy — SAME initial μ

    snapshots_B = {}
    for step, alert in enumerate(train_alerts):    # SAME alert sequence
        result_B = scorer_B.score(alert.factors, alert.category_index, None)
        is_correct_B = (result_B.action_index == alert.gt_action_index)
        scorer_B.update(
            alert.factors, alert.category_index,
            result_B.action_index, is_correct_B
        )
        if (step + 1) in SNAPSHOT_STEPS:
            snapshots_B[step + 1] = scorer_B.mu.copy()

    mu_final_B = scorer_B.mu.copy()

    # ----------------------------------------------------------------
    # Frobenius divergence: ||μ_A - μ_B||_F / ||μ_B||_F
    # ----------------------------------------------------------------
    frobenius_diff = float(np.linalg.norm(mu_final_A - mu_final_B))
    frobenius_norm = float(np.linalg.norm(mu_final_B))
    relative_diff  = frobenius_diff / frobenius_norm if frobenius_norm > 0 else 0.0

    # Per-category Frobenius diff
    per_cat_diff = {}
    for i, cat in enumerate(CATEGORIES):
        if i < mu_final_A.shape[0]:
            diff_cat = float(np.linalg.norm(mu_final_A[i] - mu_final_B[i]))
            per_cat_diff[cat] = diff_cat

    # Snapshot relative diffs over time
    snapshot_relative_diffs = {}
    for step in SNAPSHOT_STEPS:
        mu_a = snapshots_A.get(step)
        mu_b = snapshots_B.get(step)
        if mu_a is not None and mu_b is not None:
            f_diff = float(np.linalg.norm(mu_a - mu_b))
            f_norm = float(np.linalg.norm(mu_b))
            snapshot_relative_diffs[step] = f_diff / f_norm if f_norm > 0 else 0.0

    # ----------------------------------------------------------------
    # Centroids-alone test: score 200 test alerts with λ=0, using each set of centroids
    # ----------------------------------------------------------------
    scorer_test_A = ProfileScorer(mu_final_A.copy(), ACTIONS)
    scorer_test_B = ProfileScorer(mu_final_B.copy(), ACTIONS)

    correct_A_test, correct_B_test = [], []
    for alert in test_alerts:
        r_A = scorer_test_A.score(alert.factors, alert.category_index, None)
        r_B = scorer_test_B.score(alert.factors, alert.category_index, None)
        correct_A_test.append(r_A.action_index == alert.gt_action_index)
        correct_B_test.append(r_B.action_index == alert.gt_action_index)

    acc_A_test = float(np.mean(correct_A_test)) * 100
    acc_B_test = float(np.mean(correct_B_test)) * 100

    return {
        "frobenius_diff":         frobenius_diff,
        "frobenius_norm":         frobenius_norm,
        "relative_diff":          relative_diff,
        "per_category_diff":      per_cat_diff,
        "snapshot_relative_diffs": snapshot_relative_diffs,
        "accuracy_with_synthesis_centroids":    acc_A_test,
        "accuracy_without_synthesis_centroids": acc_B_test,
        "accuracy_diff":          abs(acc_A_test - acc_B_test),
    }


def main():
    print("EXP-S3: Loop Independence — Centroid Contamination Test")
    print(f"  Training decisions: {N_DECISIONS}")
    print(f"  Test alerts: {N_TEST}")
    print(f"  Seeds: {len(SEEDS)}")

    best_lambda = _load_best_lambda()
    print(f"  Lambda: {best_lambda}")
    print()

    projector = RuleBasedProjector(
        rules=SOC_SYNTHESIS_RULES, categories=CATEGORIES, actions=ACTIONS
    )

    seed_results = []
    for seed in SEEDS:
        r = run_one_seed(seed, best_lambda, projector)
        seed_results.append(r)
        sys.stdout.write(
            f"\r  seed={seed} relative_diff={r['relative_diff']:.4f} "
            f"acc_diff={r['accuracy_diff']:.2f}pp    "
        )
        sys.stdout.flush()
    print()

    # Aggregate
    rel_diffs  = [r["relative_diff"] for r in seed_results]
    acc_diffs  = [r["accuracy_diff"] for r in seed_results]
    acc_with   = [r["accuracy_with_synthesis_centroids"]    for r in seed_results]
    acc_without= [r["accuracy_without_synthesis_centroids"] for r in seed_results]

    mean_rel_diff  = float(np.mean(rel_diffs))
    std_rel_diff   = float(np.std(rel_diffs))
    mean_acc_diff  = float(np.mean(acc_diffs))

    # Snapshot trajectory (mean across seeds)
    snapshot_means = {}
    for step in SNAPSHOT_STEPS:
        vals = [r["snapshot_relative_diffs"].get(step, 0.0) for r in seed_results]
        snapshot_means[step] = float(np.mean(vals))

    gate_pass = (
        mean_rel_diff <= 0.05 and
        mean_acc_diff <= 1.0
    )

    results = {
        "experiment": "EXP-S3",
        "lambda_used": best_lambda,
        "per_seed": seed_results,
        "aggregated": {
            "mean_relative_frobenius": mean_rel_diff,
            "std_relative_frobenius":  std_rel_diff,
            "mean_accuracy_diff_pp":   mean_acc_diff,
            "mean_acc_with_synthesis_centroids":    float(np.mean(acc_with)),
            "mean_acc_without_synthesis_centroids": float(np.mean(acc_without)),
            "snapshot_mean_relative_diffs": snapshot_means,
            "per_seed_relative_diffs": rel_diffs,
            "per_seed_accuracy_diffs": acc_diffs,
        },
        "gate": {
            "S3_passed": gate_pass,
            "mean_relative_frobenius": mean_rel_diff,
            "mean_accuracy_diff_pp":   mean_acc_diff,
            "threshold_frobenius": 0.05,
            "threshold_accuracy_diff_pp": 1.0,
            "interpretation": (
                "PASS: σ does not significantly contaminate μ through indirect path" if gate_pass
                else "FAIL: consider counterfactual-aware update rule in ProfileScorer.update()"
            ),
        },
    }

    save_results(results, str(RESULTS_PATH))
    print(f"\nResults saved to {RESULTS_PATH}")

    details = (
        f"mean_relative_frobenius={mean_rel_diff:.4f} (need ≤0.05) | "
        f"mean_acc_diff={mean_acc_diff:.3f}pp (need ≤1pp)"
    )
    print_gate_result("S3", gate_pass, details)

    print("\nTrajectory (mean relative Frobenius across seeds):")
    for step, val in snapshot_means.items():
        bar = "█" * int(val * 200)
        print(f"  Step {step:3d}: {val:.4f}  {bar}")

    if not gate_pass:
        print("\n⚠ GATE-S3 FAILED — ARCHITECTURAL IMPLICATION:")
        print("  σ is contaminating μ through the action-selection path.")
        print("  ProfileScorer.update() needs a counterfactual-aware mode:")
        print("    Update centroids based on what the system WOULD have decided")
        print("    WITHOUT σ, not what it decided WITH σ active.")
        print("  This changes the update() API — inform v5.0 Phase 1 (GAE-PROF-1).")

    try:
        import importlib
        charts = importlib.import_module(
            "experiments.synthesis.expS3_loop_independence.charts"
        )
        charts.make_all_charts(results, str(EXP_DIR), str(PAPER_FIG_DIR))
        print("\nCharts saved.")
    except Exception as e:
        print(f"\nWarning: chart generation failed: {e}")

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
