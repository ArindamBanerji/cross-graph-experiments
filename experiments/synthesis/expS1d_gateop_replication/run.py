"""
EXP-S1d: GATE-OP Replication — Synthesis via Claim Chain
experiments/synthesis/expS1d_gateop_replication/run.py

QUESTION: Does sigma (via the claim_generator → rule_projector chain) replicate
the GATE-OP δ=+0.0041, p=0.0008 result?

CONTEXT:
GATE-OP found δ=+0.0041 accuracy improvement with Loop 2 running and σ set
directly via OperatorRegistry. S1/S1b/S1c found no benefit in static or cold-start
contexts. This experiment replicates GATE-OP conditions — warm profiles, Loop 2
running — but tests the full claim_generator → rule_projector chain.

DESIGN: Two phases per (seed, lambda):
  Phase 1 (warmup, 500 decisions): Loop 2 active, σ=0. Warms profiles from GT start.
  Phase 2 (eval, 500 decisions): Loop 2 active, σ applied at test lambda.
  Accuracy measured on eval phase only.

Oracle: GT-conditioned, 85% accurate.
  P(oracle says correct | action correct) = 0.85
  P(oracle says correct | action wrong)   = 0.15

Claims: generate_correct_claims() on post-warmup scorer.mu (current knowledge state).
  This mimics the real scenario: operator creates claims based on current intelligence.

GATE-S1d (DI-02 RESOLUTION):
  delta >= 0.3pp AND p < 0.05 at any lambda in [0.1, 0.5]
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

SEEDS             = _cfg["synthesis"]["seeds"]     # 10 seeds
LAMBDA_VALUES     = [0.0, 0.1, 0.2, 0.3, 0.5]
WARMUP_DECISIONS  = 500
EVAL_DECISIONS    = 500
N_CATS            = len(CATEGORIES)                # 5
N_ACTS            = len(ACTIONS)                   # 4
ORACLE_RATE       = 0.85
GATEOP_DELTA_REF  = 0.0041                         # GATE-OP reference result

# Learning rates matching GATE-OP design intent (spec lr=0.01, lr_neg=0.005)
ETA     = 0.01
ETA_NEG = 0.005


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


def make_oracle_signal(
    hit: bool,
    rng: np.random.Generator,
) -> bool:
    """
    GT-conditioned noisy oracle (ORACLE_RATE accuracy).
    P(oracle correct | hit) = ORACLE_RATE
    P(oracle correct | miss) = 1 - ORACLE_RATE
    """
    flip = rng.random() < (1.0 - ORACLE_RATE)
    return hit ^ flip


def run_warmup(
    scorer: ProfileScorer,
    alerts: list,
    seed: int,
) -> None:
    """
    Phase 1: run WARMUP_DECISIONS with Loop 2, no synthesis.
    Mutates scorer.mu in place.
    Uses per-decision RNG seeded from seed * 100000 + i for reproducibility.
    """
    for i, alert in enumerate(alerts):
        result = scorer.score(alert.factors, alert.category_index,
                              synthesis=None, lambda_coupling=0.0)
        hit    = (result.action_index == alert.gt_action_index)
        rng_i  = np.random.default_rng(seed * 100000 + i)
        oracle = make_oracle_signal(hit, rng_i)
        scorer.update(alert.factors, alert.category_index,
                      result.action_index, oracle)


def run_eval(
    scorer: ProfileScorer,
    alerts: list,
    seed: int,
    lambda_val: float,
    bias,
) -> Tuple[float, Dict[str, float]]:
    """
    Phase 2: run EVAL_DECISIONS with synthesis at lambda_val, Loop 2 still active.
    Returns (accuracy, per_category_accuracy).
    Uses per-decision RNG seeded from seed * 100000 + WARMUP_DECISIONS + i.
    """
    correct = 0
    per_cat: Dict[int, List[int]] = defaultdict(lambda: [0, 0])

    for i, alert in enumerate(alerts):
        result = scorer.score(
            alert.factors,
            alert.category_index,
            synthesis=bias if lambda_val > 0.0 else None,
            lambda_coupling=lambda_val,
        )
        hit = (result.action_index == alert.gt_action_index)
        correct += int(hit)
        per_cat[alert.category_index][0] += int(hit)
        per_cat[alert.category_index][1] += 1

        # Loop 2 continues during eval
        rng_i  = np.random.default_rng(seed * 100000 + WARMUP_DECISIONS + i)
        oracle = make_oracle_signal(hit, rng_i)
        scorer.update(alert.factors, alert.category_index,
                      result.action_index, oracle)

    accuracy = correct / len(alerts)

    per_cat_acc: Dict[str, float] = {}
    for ci, cat in enumerate(CATEGORIES):
        total  = per_cat[ci][1]
        per_cat_acc[cat] = per_cat[ci][0] / total if total > 0 else 0.0

    return accuracy, per_cat_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("EXP-S1d: GATE-OP Replication — Synthesis via Claim Chain (DI-02)")
    print(f"  lambda values:    {LAMBDA_VALUES}")
    print(f"  seeds:            {len(SEEDS)}")
    print(f"  warmup decisions: {WARMUP_DECISIONS}  (Loop 2, no sigma)")
    print(f"  eval decisions:   {EVAL_DECISIONS}    (Loop 2 + sigma)")
    print(f"  oracle rate:      {ORACLE_RATE:.0%}")
    print(f"  GATE-OP ref:      delta={GATEOP_DELTA_REF:+.4f} (pp={GATEOP_DELTA_REF*100:+.2f}pp)")
    print()

    projector = RuleBasedProjector()

    # results[lam_key][seed_idx] = {accuracy, per_cat_acc}
    all_results: Dict[str, List[Dict]] = {}
    csv_rows:    List[Dict]            = []

    for seed in SEEDS:
        gen          = CategoryAlertGenerator(seed=seed)
        all_alerts   = gen.generate(WARMUP_DECISIONS + EVAL_DECISIONS)
        gt_profiles  = build_gt_array(gen)
        base_mu      = gt_profiles.copy()            # warm-start baseline

        warmup_alerts = all_alerts[:WARMUP_DECISIONS]
        eval_alerts   = all_alerts[WARMUP_DECISIONS:]

        print(f"=== seed={seed} ===")

        # --- Run all lambdas for this seed ---
        # Each lambda gets an independent copy of base_mu, then runs warmup,
        # then generates claims on post-warmup mu, then runs eval.
        for lam in LAMBDA_VALUES:
            lam_key = f"lambda_{lam:.3f}"
            if lam_key not in all_results:
                all_results[lam_key] = []

            # Fresh scorer reset to GT for this (seed, lambda)
            scorer = ProfileScorer(base_mu.copy(), tau=0.1,
                                   eta=ETA, eta_neg=ETA_NEG)

            # Phase 1: warmup (no synthesis)
            run_warmup(scorer, warmup_alerts, seed)

            # Generate claims from post-warmup centroids (current knowledge state)
            claims = generate_correct_claims(
                10, seed=seed,
                gt_profiles=scorer.mu,          # post-warmup, not original GT
                n_categories=N_CATS,
                n_actions=N_ACTS,
            )
            bias = projector.project(
                claims,
                n_categories=N_CATS,
                n_actions=N_ACTS,
                lambda_coupling=lam,
            )

            # Phase 2: eval (with synthesis)
            accuracy, per_cat_acc = run_eval(
                scorer, eval_alerts, seed, lam, bias
            )

            all_results[lam_key].append({
                "accuracy":     accuracy,
                "per_cat_acc":  per_cat_acc,
            })

            print(f"  lambda={lam:.2f}: acc={accuracy*100:.2f}%")

            csv_rows.append({
                "seed":    seed,
                "lambda":  lam,
                "accuracy": round(accuracy, 6),
                **{f"acc_{cat}": round(per_cat_acc.get(cat, 0.0), 6)
                   for cat in CATEGORIES},
            })
        print()

    # ------------------------------------------------------------------
    # Aggregate & statistics
    # ------------------------------------------------------------------
    baseline_key  = "lambda_0.000"
    baseline_accs = np.array([r["accuracy"] for r in all_results[baseline_key]])
    baseline_mean = float(np.mean(baseline_accs))

    aggregated: Dict[str, Dict] = {}
    stat_results: Dict[str, Dict] = {}

    for lam in LAMBDA_VALUES:
        lam_key = f"lambda_{lam:.3f}"
        accs    = np.array([r["accuracy"] for r in all_results[lam_key]])
        per_cat_means: Dict[str, float] = {
            cat: float(np.mean([r["per_cat_acc"].get(cat, 0.0)
                                for r in all_results[lam_key]]))
            for cat in CATEGORIES
        }
        aggregated[lam_key] = {
            "mean_accuracy":    float(np.mean(accs)),
            "std_accuracy":     float(np.std(accs)),
            "per_cat_means":    per_cat_means,
            "per_seed_accs":    accs.tolist(),
        }

    print("\n=== EXP-S1d RESULTS ===")
    print(f"Warmup: {WARMUP_DECISIONS} decisions (Loop 2, no sigma)")
    print(f"Eval:   {EVAL_DECISIONS} decisions (Loop 2 + sigma)")
    print(f"Baseline accuracy post-warmup (lambda=0): "
          f"{baseline_mean*100:.2f}%  +/-{float(np.std(baseline_accs))*100:.2f}%")
    print()
    print(f"{'lambda':<10}  {'mean_acc':>10}  {'std':>7}  {'delta_pp':>10}  "
          f"{'p_value':>10}  {'sig':>6}")
    print("-" * 65)
    print(f"{'0.00 (base)':<10}  {baseline_mean*100:>10.2f}%  "
          f"{float(np.std(baseline_accs))*100:>6.2f}%  {'—':>10}  {'—':>10}")

    gate_pass   = False
    best_result: Optional[Dict] = None

    for lam in LAMBDA_VALUES[1:]:
        lam_key     = f"lambda_{lam:.3f}"
        treat_accs  = np.array(aggregated[lam_key]["per_seed_accs"])
        delta       = float(np.mean(treat_accs)) - baseline_mean
        t_stat, p_val = stats.ttest_rel(treat_accs, baseline_accs)
        if np.isnan(p_val):
            p_val = 1.0
        p_val    = float(p_val)
        sig_mark = "*** SIG" if p_val < 0.05 else ""

        stat_results[lam_key] = {
            "delta":          delta,
            "delta_pp":       delta * 100.0,
            "p_value":        p_val,
            "t_stat":         float(t_stat),
            "baseline_mean":  baseline_mean,
            "treatment_mean": float(np.mean(treat_accs)),
        }

        print(f"lambda={lam:.2f}   {float(np.mean(treat_accs))*100:>10.2f}%  "
              f"{float(np.std(treat_accs))*100:>6.2f}%  "
              f"{delta*100:>+9.3f}pp  {p_val:>10.4f}  {sig_mark}")

        if delta * 100.0 >= 0.3 and p_val < 0.05:
            gate_pass = True
            if best_result is None or delta > best_result["delta"]:
                best_result = {
                    "lambda":            lam,
                    "delta":             delta,
                    "delta_pp":          delta * 100.0,
                    "p_value":           p_val,
                    "baseline_accuracy": baseline_mean,
                    "treatment_accuracy": float(np.mean(treat_accs)),
                }

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    print()
    print("=== GATE-S1d (DI-02 RESOLUTION GATE) ===")
    print("Gate: delta >= 0.3pp AND p < 0.05 (replication of GATE-OP conditions)")

    # Best result by delta regardless of significance (for reporting)
    best_delta_lam = max(LAMBDA_VALUES[1:],
                         key=lambda l: stat_results[f"lambda_{l:.3f}"]["delta"])
    best_sr = stat_results[f"lambda_{best_delta_lam:.3f}"]
    print(f"Best: lambda={best_delta_lam:.2f}, "
          f"delta={best_sr['delta_pp']:+.3f}pp, "
          f"p={best_sr['p_value']:.4f}")

    if gate_pass and best_result:
        design_resolution = "option_a"
        print("GATE-S1d: PASS")
        print(f"  lambda={best_result['lambda']:.2f}  "
              f"delta={best_result['delta_pp']:+.3f}pp  "
              f"p={best_result['p_value']:.4f}")
        print("  -> OPTION A: sigma is marginal-decision enhancer for warm profiles.")
        print("  -> Tab 5 sigma scoring proceeds. GATE-M criterion to be updated.")
        print("  -> EXP-S2/S3/S4 may proceed with warm-profile design.")
        interp = (
            f"sigma (lambda={best_result['lambda']:.2f}) improves accuracy by "
            f"{best_result['delta_pp']:+.3f}pp (p={best_result['p_value']:.4f}) "
            f"in GATE-OP-matched conditions (warm profiles + Loop 2). "
            f"GATE-OP reference: delta=+0.41pp. "
            f"DI-02 resolution: OPTION A — sigma scoring ships."
        )
    else:
        design_resolution = "option_b"
        print("GATE-S1d: FAIL")
        print("  -> OPTION B: sigma scoring does not ship.")
        print("  -> Tab 5 ships as display-only (Panel A briefing + Panel B Ask the Graph).")
        print("  -> lambda=0 always. No scoring regression possible.")
        print("  -> F15 (SynthesisNode) deferred.")
        interp = (
            f"sigma does not achieve >=0.3pp improvement at p<0.05 in GATE-OP-matched conditions. "
            f"Best: delta={best_sr['delta_pp']:+.3f}pp (lambda={best_delta_lam:.2f}, "
            f"p={best_sr['p_value']:.4f}). "
            f"GATE-OP reference: delta=+0.41pp. "
            f"DI-02 resolution: OPTION B — display-only."
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "experiment":      "EXP-S1d",
        "design":          "gateop_replication",
        "warmup_decisions": WARMUP_DECISIONS,
        "eval_decisions":   EVAL_DECISIONS,
        "oracle_rate":      ORACLE_RATE,
        "gateop_ref_delta": GATEOP_DELTA_REF,
        "per_lambda":       all_results,
        "aggregated":       aggregated,
        "statistical":      stat_results,
        "gate": {
            "S1d_passed":   gate_pass,
            "best_result":  best_result,
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

    all_stat_list = [
        {
            "lambda":    lam,
            "delta_pp":  stat_results[f"lambda_{lam:.3f}"]["delta_pp"],
            "p_value":   stat_results[f"lambda_{lam:.3f}"]["p_value"],
        }
        for lam in LAMBDA_VALUES[1:]
    ]
    gate_result = {
        "gate":                  "S1d",
        "pass":                  gate_pass,
        "design_resolution":     design_resolution,
        "warmup_decisions":      WARMUP_DECISIONS,
        "eval_decisions":        EVAL_DECISIONS,
        "baseline_accuracy":     baseline_mean,
        "best_lambda":           best_result["lambda"] if best_result else best_delta_lam,
        "best_delta_pp":         best_result["delta_pp"] if best_result else best_sr["delta_pp"],
        "best_p_value":          best_result["p_value"] if best_result else best_sr["p_value"],
        "gateop_ref_delta_pp":   GATEOP_DELTA_REF * 100.0,
        "interpretation":        interp,
        "all_results":           all_stat_list,
    }
    with open(GATE_PATH, "w") as f:
        json.dump(gate_result, f, indent=2, default=float)
    print(f"Gate result saved to {GATE_PATH}")

    # Charts
    try:
        import importlib
        charts_mod = importlib.import_module(
            "experiments.synthesis.expS1d_gateop_replication.charts"
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
