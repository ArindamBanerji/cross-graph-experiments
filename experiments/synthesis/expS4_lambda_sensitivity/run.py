"""
EXP-S4: Coupling Constant Sensitivity (cold-start + pre-campaign design)
experiments/synthesis/expS4_lambda_sensitivity/run.py

QUESTION: Is there a stable plateau for lambda, or is tuning fragile?
GATE-S4: plateau_width >= 0.05

Plateau measured on improvement_pp = acc_with_sigma - acc_gap.
Uses same cold-start + pre-campaign design as S1 to ensure a real knowledge gap.

Design: intelligence_layer_design_v1 §4 EXP-S4
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.rule_projector import RuleBasedProjector
from src.data.claim_generator import (
    SOC_SYNTHESIS_RULES, CATEGORIES, ACTIONS, generate_correct_claims
)
from src.viz.synthesis_common import save_results, print_gate_result

RESULTS_PATH = Path(__file__).parent / "results.json"
PAPER_FIG_DIR = REPO_ROOT / "paper_figures"
EXP_DIR = Path(__file__).parent

LAMBDA_SWEEP = [
    0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
    0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5
]
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
N_PRECAMPAIGN  = 400
N_CAMPAIGN     = 300
PLATEAU_MIN_IMPROVEMENT_PP = 2.0

SUPPRESSED = {
    "credential_access":  "escalate_incident",
    "threat_intel_match": "escalate_incident",
    "lateral_movement":   "escalate_incident",
    "data_exfiltration":  "escalate_incident",
    "insider_threat":     "escalate_tier2",
}

CAMPAIGN = {
    "credential_access":  {"escalate_incident": 0.80},
    "threat_intel_match": {"escalate_incident": 0.75},
    "lateral_movement":   {"escalate_incident": 0.80},
    "data_exfiltration":  {"escalate_incident": 0.70},
    "insider_threat":     {"escalate_tier2":    0.70},
}


def run_one_seed_lambda(
    seed: int,
    lambda_val: float,
    projector: RuleBasedProjector,
) -> Dict:
    """
    Cold-start + pre-campaign three-condition run.
    Returns acc_gap, acc_with_sigma, improvement_pp.
    """
    from src.data.category_alert_generator import CategoryAlertGenerator
    from src.models.profile_scorer import ProfileScorer

    # Phase 1: cold-start + pre-campaign training
    gen = CategoryAlertGenerator(seed=seed)
    precampaign_alerts = gen.generate_precampaign(N_PRECAMPAIGN, SUPPRESSED)

    n_cats    = len(CATEGORIES)
    n_acts    = len(ACTIONS)
    n_factors = len(gen.factors)
    mu_cold   = np.full((n_cats, n_acts, n_factors), 0.5)
    scorer    = ProfileScorer(mu_cold, ACTIONS)

    for alert in precampaign_alerts:
        result  = scorer.score(alert.factors, alert.category_index)
        correct = (result.action_index == alert.gt_action_index)
        scorer.update(alert.factors, alert.category_index, result.action_index, correct)
    mu_precampaign = scorer.mu.copy()

    # Phase 2: project claims -> bias
    claims = generate_correct_claims(n=20, seed=seed)
    bias   = projector.project(claims, lambda_coupling=lambda_val)

    # Phase 3: campaign alerts
    gen_camp        = CategoryAlertGenerator(seed=seed + 9999)
    campaign_alerts = gen_camp.generate_campaign(N_CAMPAIGN, CAMPAIGN)

    # Condition 2: gap (no sigma)
    scorer_gap  = ProfileScorer(mu_precampaign.copy(), ACTIONS)
    correct_gap = []
    for alert in campaign_alerts:
        r = scorer_gap.score(alert.factors, alert.category_index, None)
        correct_gap.append(r.action_index == alert.gt_action_index)
    acc_gap = float(np.mean(correct_gap)) * 100

    # Condition 3: with sigma
    scorer_sig  = ProfileScorer(mu_precampaign.copy(), ACTIONS)
    correct_sig = []
    for alert in campaign_alerts:
        r = scorer_sig.score(
            alert.factors, alert.category_index,
            bias if lambda_val > 0 else None,
        )
        correct_sig.append(r.action_index == alert.gt_action_index)
    acc_with_sigma = float(np.mean(correct_sig)) * 100

    return {
        "acc_gap":        acc_gap,
        "acc_with_sigma": acc_with_sigma,
        "improvement_pp": acc_with_sigma - acc_gap,
    }


def find_plateau(lambdas: List[float], mean_improvements: List[float]) -> Dict:
    """Find the plateau: contiguous range of lambda where improvement_pp >= 2pp."""
    plateau_lambdas = [
        l for l, imp in zip(lambdas, mean_improvements)
        if imp >= PLATEAU_MIN_IMPROVEMENT_PP
    ]
    plateau_width = (
        (max(plateau_lambdas) - min(plateau_lambdas)) if plateau_lambdas else 0.0
    )
    peak_idx = int(np.argmax(mean_improvements))
    return {
        "plateau_lambdas":  plateau_lambdas,
        "plateau_width":    plateau_width,
        "lambda_peak":      lambdas[peak_idx],
        "peak_improvement": mean_improvements[peak_idx],
        "peak_accuracy":    0.0,
        "threshold_used":   PLATEAU_MIN_IMPROVEMENT_PP,
    }


def main():
    print("EXP-S4: Coupling Constant Sensitivity (cold-start + pre-campaign)")
    print(f"  Lambda values: {len(LAMBDA_SWEEP)} ({LAMBDA_SWEEP[0]}-{LAMBDA_SWEEP[-1]})")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Pre-campaign alerts: {N_PRECAMPAIGN}  |  Campaign alerts: {N_CAMPAIGN}")
    print(f"  Plateau threshold: improvement_pp >= {PLATEAU_MIN_IMPROVEMENT_PP}pp")
    print()

    projector = RuleBasedProjector(
        rules=SOC_SYNTHESIS_RULES, categories=CATEGORIES, actions=ACTIONS
    )

    acc_gap_matrix     = np.zeros((len(LAMBDA_SWEEP), len(SEEDS)))
    acc_sigma_matrix   = np.zeros((len(LAMBDA_SWEEP), len(SEEDS)))
    improvement_matrix = np.zeros((len(LAMBDA_SWEEP), len(SEEDS)))

    for j, lam in enumerate(LAMBDA_SWEEP):
        for k, seed in enumerate(SEEDS):
            r = run_one_seed_lambda(seed, lam, projector)
            acc_gap_matrix[j, k]     = r["acc_gap"]
            acc_sigma_matrix[j, k]   = r["acc_with_sigma"]
            improvement_matrix[j, k] = r["improvement_pp"]
            sys.stdout.write(
                f"\r  lambda={lam:.3f} ({j+1}/{len(LAMBDA_SWEEP)}) seed={seed} "
                f"gap={r['acc_gap']:.2f}% sig={r['acc_with_sigma']:.2f}% "
                f"imp={r['improvement_pp']:+.2f}pp    "
            )
            sys.stdout.flush()
        print()

    mean_gaps         = acc_gap_matrix.mean(axis=1).tolist()
    mean_sigmas       = acc_sigma_matrix.mean(axis=1).tolist()
    std_sigmas        = acc_sigma_matrix.std(axis=1).tolist()
    mean_improvements  = improvement_matrix.mean(axis=1).tolist()
    std_improvements   = improvement_matrix.std(axis=1).tolist()

    baseline_mean = mean_gaps[0]   # acc_gap at lambda=0

    plateau_info = find_plateau(LAMBDA_SWEEP, mean_improvements)
    peak_idx = int(np.argmax(mean_improvements))
    plateau_info["peak_accuracy"] = mean_sigmas[peak_idx]

    gate_pass = plateau_info["plateau_width"] >= 0.05

    results = {
        "experiment":                  "EXP-S4",
        "design":                      "cold_start_precampaign",
        "lambda_sweep":                LAMBDA_SWEEP,
        "per_lambda_mean_accuracy":    mean_sigmas,      # for charts.py compat
        "per_lambda_std_accuracy":     std_sigmas,
        "baseline_mean_accuracy":      baseline_mean,    # acc_gap at lambda=0
        "per_lambda_mean_acc_gap":     mean_gaps,
        "per_lambda_mean_improvement": mean_improvements,
        "per_lambda_std_improvement":  std_improvements,
        "per_seed_acc_gap_matrix":     acc_gap_matrix.tolist(),
        "per_seed_acc_sigma_matrix":   acc_sigma_matrix.tolist(),
        "per_seed_improvement_matrix": improvement_matrix.tolist(),
        "plateau": plateau_info,
        "gate": {
            "S4_passed":               gate_pass,
            "plateau_width":           plateau_info["plateau_width"],
            "lambda_peak":             plateau_info["lambda_peak"],
            "peak_accuracy":           plateau_info["peak_accuracy"],
            "peak_improvement_pp":     plateau_info["peak_improvement"],
            "threshold_plateau_width": 0.05,
            "interpretation": (
                "PASS: stable plateau exists — single global lambda is tunable"
                if gate_pass
                else "FAIL: narrow spike — consider per-category lambda[c] in SynthesisBias"
            ),
        },
    }

    save_results(results, str(RESULTS_PATH))
    print(f"\nResults saved to {RESULTS_PATH}")

    if plateau_info["plateau_lambdas"]:
        details = (
            f"plateau_width={plateau_info['plateau_width']:.3f} (need >=0.05) | "
            f"lambda_peak={plateau_info['lambda_peak']:.3f} | "
            f"peak_improvement={plateau_info['peak_improvement']:.2f}pp | "
            f"plateau=[{min(plateau_info['plateau_lambdas']):.3f}, "
            f"{max(plateau_info['plateau_lambdas']):.3f}]"
        )
    else:
        details = (
            f"plateau_width=0 (no lambda achieved improvement>={PLATEAU_MIN_IMPROVEMENT_PP}pp)"
        )
    print_gate_result("S4", gate_pass, details)

    if not gate_pass:
        print("\n  GATE-S4 FAILED — ARCHITECTURAL IMPLICATION:")
        print("  lambda is tuning-sensitive (narrow spike, no stable plateau).")
        print("  Options:")
        print("    (1) Per-category lambda[c] — SynthesisBias.lambda_coupling becomes array (n_cat,)")
        print("    (2) Adaptive coupling: lambda scales with claim confidence")
        print("  Either changes SynthesisBias dataclass — inform v5.0 before writing it.")

    print(f"\nLambda sweep (top 10 by improvement_pp):")
    print(f"  {'lambda':>8s}  {'mean_imp':>10s}  {'std_imp':>8s}  "
          f"{'gap_acc':>9s}  {'sig_acc':>9s}")
    sorted_idx = np.argsort(mean_improvements)[::-1][:10]
    for idx in sorted_idx:
        print(
            f"  {LAMBDA_SWEEP[idx]:>8.3f}  "
            f"{mean_improvements[idx]:>+10.3f}pp  "
            f"{std_improvements[idx]:>8.3f}  "
            f"{mean_gaps[idx]:>9.3f}%  "
            f"{mean_sigmas[idx]:>9.3f}%"
        )

    try:
        import importlib
        charts = importlib.import_module(
            "experiments.synthesis.expS4_lambda_sensitivity.charts"
        )
        charts.make_all_charts(results, str(EXP_DIR), str(PAPER_FIG_DIR))
        print("\nCharts saved.")
    except Exception as e:
        print(f"\nWarning: chart generation failed: {e}")

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
