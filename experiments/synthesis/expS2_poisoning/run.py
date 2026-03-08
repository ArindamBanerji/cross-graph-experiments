"""
EXP-S2: Poisoning Resilience
experiments/synthesis/expS2_poisoning/run.py

QUESTION: How much does 20% / 40% poisoned claims degrade accuracy?
GATE-S2: poison_20pct_degradation <= 2pp AND safety_effectiveness >= 0.50

Depends on SYNTH-EXP-0 and SYNTH-EXP-1 (best_lambda from S1).
If S1 results not available, uses lambda=0.1 as default.

Design: intelligence_layer_design_v1 §4 EXP-S2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.synthesis import SynthesisBias
from src.models.rule_projector import RuleBasedProjector
from src.data.claim_generator import (
    SOC_SYNTHESIS_RULES, CATEGORIES, ACTIONS,
    generate_correct_claims, generate_poisoned_claims
)
from src.viz.synthesis_common import save_results, print_gate_result

RESULTS_PATH = Path(__file__).parent / "results.json"
PAPER_FIG_DIR = REPO_ROOT / "paper_figures"
EXP_DIR = Path(__file__).parent

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
N_ALERTS = 500

CAMPAIGN = {
    "credential_access":  {"escalate_incident": 0.80},
    "threat_intel_match": {"escalate_incident": 0.75},
    "lateral_movement":   {"escalate_incident": 0.80},
    "data_exfiltration":  {"escalate_incident": 0.70},
    "insider_threat":     {"escalate_tier2":    0.70},
}

# Poison conditions: (n_correct, n_poison, label)
POISON_CONDITIONS = [
    (10, 0,  "clean_0pct"),
    (8,  2,  "poison_20pct"),
    (6,  4,  "poison_40pct"),
]


def _load_best_lambda() -> float:
    """Load best λ from S1 results, default to 0.1 if not available."""
    s1_path = REPO_ROOT / "experiments" / "synthesis" / "expS1_bias_accuracy" / "results.json"
    if s1_path.exists():
        with open(s1_path) as f:
            s1 = json.load(f)
        best_lam = s1.get("statistical_test", {}).get("best_lambda", 0.1)
        print(f"  Using best λ from S1: {best_lam}")
        return float(best_lam) if best_lam is not None else 0.1
    print("  S1 results not found — using λ=0.1 as default")
    return 0.1


def run_one_seed_condition(
    seed: int,
    n_correct: int,
    n_poison: int,
    lambda_val: float,
    projector: RuleBasedProjector,
) -> Dict:
    """Run 500 alerts under one poison condition for one seed."""
    from src.data.category_alert_generator import CategoryAlertGenerator
    from src.models.profile_scorer import ProfileScorer

    gen = CategoryAlertGenerator(seed=seed)
    mu_initial = np.array(
        [[gen.profiles[cat][act] for act in gen.actions] for cat in gen.categories]
    )
    scorer = ProfileScorer(mu_initial.copy(), ACTIONS)
    # Score campaign-period alerts so poisoning resilience is measured where sigma is active
    gen_camp = CategoryAlertGenerator(seed=seed + 9999)
    alerts = gen_camp.generate_campaign(N_ALERTS, CAMPAIGN)

    if n_poison == 0:
        claims = generate_correct_claims(n_correct, seed)
    else:
        claims = generate_poisoned_claims(n_correct, n_poison, seed)

    bias = projector.project(claims, lambda_coupling=lambda_val)

    correct_flags = []
    for alert in alerts:
        result = scorer.score(alert.factors, alert.category_index, bias)
        correct_flags.append(result.action_index == alert.gt_action_index)

    return {
        "accuracy": float(np.mean(correct_flags)) * 100,
        "n_correct_claims": n_correct,
        "n_poison_claims": n_poison,
        "active_claims": bias.active_claims,
    }


def main():
    print("EXP-S2: Poisoning Resilience")
    best_lambda = _load_best_lambda()
    print(f"  Lambda: {best_lambda}")

    projector = RuleBasedProjector(
        rules=SOC_SYNTHESIS_RULES,
        categories=CATEGORIES,
        actions=ACTIONS,
    )

    all_results = {}

    for n_correct, n_poison, label in POISON_CONDITIONS:
        condition_results = []
        for seed in SEEDS:
            r = run_one_seed_condition(seed, n_correct, n_poison, best_lambda, projector)
            condition_results.append(r)
            sys.stdout.write(
                f"\r  {label} seed={seed} acc={r['accuracy']:.2f}%    "
            )
            sys.stdout.flush()
        print()
        accs = [r["accuracy"] for r in condition_results]
        all_results[label] = {
            "per_seed": condition_results,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy":  float(np.std(accs)),
            "n_correct": n_correct,
            "n_poison":  n_poison,
        }

    clean_mean = all_results["clean_0pct"]["mean_accuracy"]
    poison_20_mean = all_results["poison_20pct"]["mean_accuracy"]
    poison_40_mean = all_results["poison_40pct"]["mean_accuracy"]

    degradation_20 = clean_mean - poison_20_mean
    degradation_40 = clean_mean - poison_40_mean

    # Safety effectiveness: how much of the poison impact is caught?
    # Approximated as: 1 - (degradation_20 / maximum_possible_degradation)
    # Max degradation = baseline - random (rough bound)
    from src.data.category_alert_generator import CategoryAlertGenerator
    gen0 = CategoryAlertGenerator(seed=42)
    n_actions = len(ACTIONS)
    baseline_acc = all_results["clean_0pct"]["mean_accuracy"]
    # Simple bound: if all claims were adversarial, accuracy approaches random (~25%)
    max_degradation = max(baseline_acc - 25.0, 1.0)
    safety_effectiveness = 1.0 - (degradation_20 / max_degradation) if max_degradation > 0 else 1.0

    gate_pass = (
        degradation_20 <= 2.0 and
        safety_effectiveness >= 0.50
    )

    results = {
        "experiment": "EXP-S2",
        "lambda_used": best_lambda,
        "conditions": all_results,
        "summary": {
            "clean_mean_accuracy":    clean_mean,
            "poison_20pct_mean_accuracy": poison_20_mean,
            "poison_40pct_mean_accuracy": poison_40_mean,
            "degradation_20pct":  degradation_20,
            "degradation_40pct":  degradation_40,
            "safety_effectiveness": safety_effectiveness,
        },
        "gate": {
            "S2_passed": gate_pass,
            "poison_20pct_degradation": degradation_20,
            "safety_effectiveness": safety_effectiveness,
            "threshold_degradation_pp": 2.0,
            "threshold_safety_effectiveness": 0.50,
        },
    }

    save_results(results, str(RESULTS_PATH))
    print(f"\nResults saved to {RESULTS_PATH}")

    details = (
        f"20%-poison degradation={degradation_20:.2f}pp (need ≤2pp) | "
        f"40%-poison degradation={degradation_40:.2f}pp | "
        f"safety_eff={safety_effectiveness:.3f} (need ≥0.50)"
    )
    print_gate_result("S2", gate_pass, details)

    print("\nCondition summary:")
    for label, res in all_results.items():
        deg = clean_mean - res["mean_accuracy"]
        print(f"  {label:18s}  mean={res['mean_accuracy']:.3f}%  "
              f"std={res['std_accuracy']:.3f}  degradation={deg:+.3f}pp")

    try:
        import importlib
        charts = importlib.import_module(
            "experiments.synthesis.expS2_poisoning.charts"
        )
        charts.make_all_charts(results, str(EXP_DIR), str(PAPER_FIG_DIR))
        print("\nCharts saved.")
    except Exception as e:
        print(f"\nWarning: chart generation failed: {e}")

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
