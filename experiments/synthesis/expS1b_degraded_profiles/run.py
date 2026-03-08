"""
EXP-S1b: Synthesis Bias Under Degraded Profile Conditions
experiments/synthesis/expS1b_degraded_profiles/run.py

QUESTION: Does sigma help when profiles are stale or cold?

EXP-S1 showed +0.20pp (n.s.) with warm-start GT profiles.
Root cause: near-optimal profiles leave no gap for sigma to bridge.
This experiment tests: cold_start, 50% noise, 25% noise, and warm_start (reproduced).

"mu is what you've learned. sigma is what you know right now."

GATE-S1b: sigma provides >=3pp improvement at p<0.05 under ANY degraded condition.

Design:
  - 4 degradation conditions x 6 lambda values x 10 seeds x 500 alerts
  - ProfileScorer initialized with degraded centroids (tau=0.1), no online learning
  - Claims: 10 correct claims from generate_correct_claims (direction=-1 in sigma after fix)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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

SEEDS          = _cfg["synthesis"]["seeds"]     # 10 seeds
N_ALERTS       = _cfg["synthesis"]["n_decisions"]  # 500
LAMBDA_VALUES  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
N_CLAIM_CATS   = 6    # generate_correct_claims default
N_CATS         = len(CATEGORIES)   # 5
N_ACTS         = len(ACTIONS)      # 4

# (name, noise_level, condition_type)
DEGRADATION_CONDITIONS: List[Tuple[str, float, str]] = [
    ("cold_start",   0.0,  "random"),
    ("noise_50pct",  0.5,  "noise"),
    ("noise_25pct",  0.25, "noise"),
    ("warm_start",   0.0,  "gt"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_gt_array(gen: CategoryAlertGenerator) -> np.ndarray:
    """Extract (n_cats, n_acts, n_factors) numpy array from gen.profiles dict."""
    n_cats  = len(gen.categories)
    n_acts  = len(gen.actions)
    n_facts = len(gen.factors)
    arr = np.zeros((n_cats, n_acts, n_facts), dtype=np.float32)
    for c_idx, cat in enumerate(gen.categories):
        for a_idx, act in enumerate(gen.actions):
            arr[c_idx, a_idx, :] = gen.profiles[cat][act]
    return arr


def degrade_profiles(
    gt_profiles: np.ndarray,
    condition: str,
    noise_level: float,
    seed: int,
) -> np.ndarray:
    """
    gt_profiles: shape (n_categories, n_actions, n_factors), values in [0,1]
    Returns degraded centroids of same shape, clipped to [0, 1].
    """
    rng = np.random.default_rng(seed + 9999)   # independent from alert seed
    if condition == "gt":
        return gt_profiles.copy().astype(np.float64)
    elif condition == "random":
        return rng.uniform(0, 1, gt_profiles.shape).astype(np.float64)
    elif condition == "noise":
        noise = rng.normal(0, noise_level, gt_profiles.shape)
        return np.clip(gt_profiles.astype(np.float64) + noise, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def compute_ece(
    confs: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    n = len(confs)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confs >= lo) & (confs < hi)
        if b == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(
            float(confs[mask].mean()) - float(correct[mask].astype(float).mean())
        )
    return ece / n


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_one_seed(
    seed: int,
    lambda_val: float,
    cond_name: str,
    noise_level: float,
    condition: str,
    projector: RuleBasedProjector,
) -> Dict:
    """Score N_ALERTS with degraded profiles + optional synthesis bias."""
    gen    = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_ALERTS)

    gt_profiles = build_gt_array(gen)
    degraded    = degrade_profiles(gt_profiles, condition, noise_level, seed)
    scorer      = ProfileScorer(degraded, tau=0.1)

    claims = generate_correct_claims(10, seed=seed, gt_profiles=gt_profiles,
                                     n_categories=N_CATS, n_actions=N_ACTS)
    bias   = projector.project(
        claims,
        n_categories=N_CATS,
        n_actions=N_ACTS,
        lambda_coupling=lambda_val,
    )

    confs_list:   List[float] = []
    correct_list: List[int]   = []
    per_cat: Dict[int, List[int]] = defaultdict(lambda: [0, 0])

    for alert in alerts:
        result = scorer.score(
            alert.factors,
            alert.category_index,
            synthesis=bias if lambda_val > 0.0 else None,
            lambda_coupling=lambda_val,
        )
        hit = int(result.action_index == alert.gt_action_index)
        confs_list.append(float(result.probabilities[result.action_index]))
        correct_list.append(hit)
        per_cat[alert.category_index][0] += hit
        per_cat[alert.category_index][1] += 1

    confs_arr   = np.array(confs_list)
    correct_arr = np.array(correct_list, dtype=bool)
    accuracy    = float(np.mean(correct_arr)) * 100.0

    per_cat_acc: Dict[str, float] = {}
    for cat in CATEGORIES:
        cat_idx = gen.categories.index(cat)
        tot     = per_cat[cat_idx][1]
        per_cat_acc[cat] = per_cat[cat_idx][0] / tot * 100.0 if tot > 0 else 0.0

    return {
        "accuracy":              accuracy,
        "ece":                   compute_ece(confs_arr, correct_arr),
        "per_category_accuracy": per_cat_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("EXP-S1b: Synthesis Bias Under Degraded Profile Conditions")
    print(f"  conditions:    {[c[0] for c in DEGRADATION_CONDITIONS]}")
    print(f"  lambda values: {LAMBDA_VALUES}")
    print(f"  seeds:         {len(SEEDS)}")
    print(f"  n_alerts:      {N_ALERTS}")
    print()

    projector = RuleBasedProjector()

    # Structure: all_results[cond_name][lam_key] = list of seed results
    all_results: Dict[str, Dict[str, List[Dict]]] = {}
    csv_rows:    List[Dict]                        = []

    for cond_name, noise_level, condition in DEGRADATION_CONDITIONS:
        print(f"--- {cond_name} ---")
        all_results[cond_name] = {}

        for lam in LAMBDA_VALUES:
            lam_key = f"lambda_{lam:.3f}"
            all_results[cond_name][lam_key] = []

            for seed in SEEDS:
                r = run_one_seed(
                    seed, lam, cond_name, noise_level, condition, projector
                )
                all_results[cond_name][lam_key].append(r)
                csv_rows.append({
                    "condition":  cond_name,
                    "noise_level": noise_level,
                    "lambda":     lam,
                    "seed":       seed,
                    "accuracy":   round(r["accuracy"], 4),
                    "ece":        round(r["ece"], 6),
                    **{f"acc_{cat}": round(r["per_category_accuracy"][cat], 4)
                       for cat in CATEGORIES},
                })

            seed_accs = [r["accuracy"] for r in all_results[cond_name][lam_key]]
            print(f"  lambda={lam:.2f}: mean={np.mean(seed_accs):.2f}%  "
                  f"std={np.std(seed_accs):.2f}%")

        print()

    # ------------------------------------------------------------------
    # Aggregate per condition × lambda
    # ------------------------------------------------------------------
    aggregated: Dict[str, Dict[str, Dict]] = {}
    for cond_name, _, _ in DEGRADATION_CONDITIONS:
        aggregated[cond_name] = {}
        for lam in LAMBDA_VALUES:
            lam_key  = f"lambda_{lam:.3f}"
            seed_res = all_results[cond_name][lam_key]
            accs     = [r["accuracy"] for r in seed_res]
            eces     = [r["ece"]      for r in seed_res]
            per_cat_mean: Dict[str, float] = {
                cat: float(np.mean([r["per_category_accuracy"][cat] for r in seed_res]))
                for cat in CATEGORIES
            }
            aggregated[cond_name][lam_key] = {
                "mean_accuracy":              float(np.mean(accs)),
                "std_accuracy":               float(np.std(accs)),
                "mean_ece":                   float(np.mean(eces)),
                "std_ece":                    float(np.std(eces)),
                "per_seed_accuracies":        accs,
                "per_category_mean_accuracy": per_cat_mean,
            }

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------
    stat_results: Dict[str, Dict] = {}
    print("\n=== EXP-S1b RESULTS ===")

    for cond_name, _, _ in DEGRADATION_CONDITIONS:
        baseline_key  = "lambda_0.000"
        baseline_accs = np.array(aggregated[cond_name][baseline_key]["per_seed_accuracies"])
        baseline_mean = float(np.mean(baseline_accs))
        print(f"\n{cond_name}  baseline: {baseline_mean:.2f}%")

        stat_results[cond_name] = {}
        for lam in LAMBDA_VALUES[1:]:
            lam_key       = f"lambda_{lam:.3f}"
            treat_accs    = np.array(aggregated[cond_name][lam_key]["per_seed_accuracies"])
            improvement   = float(np.mean(treat_accs)) - baseline_mean
            t_stat, p_val = stats.ttest_rel(treat_accs, baseline_accs)
            if np.isnan(p_val):
                p_val = 1.0
            p_val = float(p_val)
            sig   = "  *** SIG" if p_val < 0.05 else ""
            print(f"  lambda={lam:.2f}: {float(np.mean(treat_accs)):.2f}%  "
                  f"({improvement:+.2f}pp, p={p_val:.4f}){sig}")
            stat_results[cond_name][lam_key] = {
                "improvement_pp": improvement,
                "p_value":        p_val,
                "t_stat":         float(t_stat),
                "baseline_mean":  baseline_mean,
                "treatment_mean": float(np.mean(treat_accs)),
            }

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------
    gate_pass   = False
    best_result = None

    for cond_name, _, _ in DEGRADATION_CONDITIONS:
        if cond_name == "warm_start":
            continue   # warm_start excluded from gate (confirmed ceiling in EXP-S1)
        for lam in LAMBDA_VALUES[1:]:
            lam_key = f"lambda_{lam:.3f}"
            sr      = stat_results[cond_name][lam_key]
            if sr["improvement_pp"] >= 3.0 and sr["p_value"] < 0.05:
                gate_pass = True
                if (best_result is None
                        or sr["improvement_pp"] > best_result["improvement_pp"]):
                    best_result = {
                        "degradation":       cond_name,
                        "lambda":            lam,
                        "improvement_pp":    sr["improvement_pp"],
                        "p_value":           sr["p_value"],
                        "baseline_accuracy": sr["baseline_mean"],
                        "treatment_accuracy": sr["treatment_mean"],
                    }

    print(f"\n=== GATE-S1b ===")
    if gate_pass and best_result is not None:
        print("GATE-S1b: PASS")
        print(f"  Best: {best_result['degradation']} + lambda={best_result['lambda']:.2f}")
        print(f"  Improvement: {best_result['improvement_pp']:+.2f}pp")
        print(f"  p={best_result['p_value']:.4f}")
        print(f"  Baseline: {best_result['baseline_accuracy']:.2f}%  "
              f"-> With sigma: {best_result['treatment_accuracy']:.2f}%")
        print(f"  Interpretation: sigma compensates for profile staleness.")
        print(f"  Product implication: sigma is most valuable at cold-start and")
        print(f"  during profile drift periods -- not when profiles are optimal.")
        finding = "sigma_helps_degraded"
        interp  = (
            "Synthesis bias sigma provides statistically significant improvement "
            f"(+{best_result['improvement_pp']:.2f}pp, p={best_result['p_value']:.4f}) "
            f"under {best_result['degradation']} conditions. "
            "Combined with EXP-S1 (warm-start ceiling), this confirms sigma is most "
            "valuable as a bridge during cold-start and profile drift periods."
        )
    else:
        print("GATE-S1b: FAIL")
        print("  sigma does not improve accuracy even with degraded profiles.")
        print("  Implication: sigma effect is too small to gate on accuracy alone.")
        print("  Tab 5 Panel A (briefing display) remains valuable independently.")
        finding = "sigma_no_effect"
        interp  = (
            "Synthesis bias sigma does not provide >=3pp statistically significant "
            "improvement even under degraded profile conditions. "
            "Combined with EXP-S1 (warm-start ceiling), this suggests sigma's value "
            "lies in non-accuracy dimensions (confidence shaping, briefing display)."
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "experiment": "EXP-S1b",
        "design":     "degraded_profiles",
        "conditions": [c[0] for c in DEGRADATION_CONDITIONS],
        "per_condition": all_results,
        "aggregated":    aggregated,
        "statistical":   stat_results,
        "gate": {
            "S1b_passed": gate_pass,
            "best_result": best_result,
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
        print(f"CSV saved to {CSV_PATH}  ({len(csv_rows)} rows)")

    # Gate result
    gate_result = {
        "gate":                    "S1b",
        "pass":                    gate_pass,
        "finding":                 finding,
        "best_result":             best_result,
        "s1_finding":              "warm_start_ceiling_confirmed",
        "combined_interpretation": interp,
    }
    with open(GATE_PATH, "w") as f:
        json.dump(gate_result, f, indent=2, default=float)
    print(f"Gate result saved to {GATE_PATH}")

    # Charts
    try:
        import importlib
        charts_mod = importlib.import_module(
            "experiments.synthesis.expS1b_degraded_profiles.charts"
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
