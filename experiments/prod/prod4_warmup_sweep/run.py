"""
PROD-4-WARMUP — Coverage Growth vs Warmup Length
Answers: "how many verified decisions until meaningful auto-approve coverage?"

Fixed: η_neg=0.05, τ=0.1, noise=0.10. Only warmup length varies.
Regime: centroidal synthetic. Ontology: SOC product v5.0+refer (C=6, A=5, d=6).

WARMUP_LEVELS = [200, 500, 1000, 2000]
N_EVAL = 1000 (post-warmup evaluation window, same across all warmup levels)

PROD-4 final (warmup=200) showed 4-7% coverage per category.
Four-model judge panel identified warmup insufficiency as most likely bottleneck.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

_REPO_ROOT  = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Add this experiment's directory so `from charts import generate_charts` works
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP_LEVELS    = [200, 500, 1000, 2000]
N_SEEDS          = 50
N_EVAL           = 1000       # post-warmup evaluation window (fixed)
TAU              = 0.1
ETA              = 0.05
ETA_NEG          = 0.05       # η_neg=1.0 is FORBIDDEN
NOISE_RATE       = 0.10
ACCURACY_GATE    = 0.85
THRESHOLD_SWEEP  = list(np.arange(0.50, 1.00, 0.01))
MARGIN_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG    = "soc_product_v50"

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

print(f"Domain config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Categories: {CATEGORIES}")
print(f"η_neg={ETA_NEG}, τ={TAU}, η={ETA}, noise={NOISE_RATE}")
print(f"Seeds: {N_SEEDS}, N_eval: {N_EVAL}")
print(f"Warmup levels: {WARMUP_LEVELS}")
print()

RESULTS_PATH = Path(__file__).parent / "prod4_warmup_results.json"

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

all_results: dict[int, dict] = {}

for n_warmup in WARMUP_LEVELS:
    print(f"\n{'='*60}")
    print(f"=== WARMUP = {n_warmup} ===")
    print(f"{'='*60}")

    all_seed_records: list[list[dict]] = []

    for seed_idx in range(N_SEEDS):
        if seed_idx % 10 == 0:
            print(f"  Warmup={n_warmup}, Seed {seed_idx+1}/{N_SEEDS}", flush=True)

        gen = CategoryAlertGenerator(
            **config["generator_kwargs"],
            noise_rate=NOISE_RATE,
            seed=RANDOM_SEED_BASE + seed_idx,
        )
        scorer = ProfileScorer(
            config["mu"].copy(),
            config["actions"],
            tau=TAU,
            eta=ETA,
            eta_neg=ETA_NEG,
        )

        # ---- Warmup phase (variable length, learning ON, not recorded) ----
        warmup_alerts = gen.generate(n_warmup)
        for alert in warmup_alerts:
            scorer.update(
                alert.factors,
                alert.category_index,
                alert.gt_action_index,
                correct=True,
            )

        # ---- Evaluation phase (fixed N_EVAL, learning ON, recorded) ----
        eval_alerts = gen.generate(N_EVAL)
        seed_records: list[dict] = []
        for alert in eval_alerts:
            result     = scorer.score(alert.factors, alert.category_index)
            is_correct = result.action_index == alert.gt_action_index
            sorted_probs = np.sort(result.probabilities)[::-1]
            margin       = float(sorted_probs[0] - sorted_probs[1])
            scorer.update(
                alert.factors,
                alert.category_index,
                result.action_index,
                correct=is_correct,
                gt_action_index=alert.gt_action_index,
            )
            seed_records.append({
                "category":     alert.category,
                "category_idx": alert.category_index,
                "confidence":   float(result.confidence),
                "margin":       margin,
                "is_correct":   bool(is_correct),
            })
        all_seed_records.append(seed_records)

    # ----------------------------------------------------------------
    # Analysis for this warmup level
    # ----------------------------------------------------------------

    flat_records = [r for seed_recs in all_seed_records for r in seed_recs]
    overall_acc  = float(np.mean([r["is_correct"] for r in flat_records]))

    # Per-category confidence threshold analysis
    confidence_results: dict[str, dict] = {}
    for cat in CATEGORIES:
        best_t = None
        best_acc = None
        best_cov = None

        for t in THRESHOLD_SWEEP:
            seed_accs: list[float] = []
            seed_covs: list[float] = []
            for seed_recs in all_seed_records:
                above = [r for r in seed_recs
                         if r["category"] == cat and r["confidence"] >= t]
                if above:
                    seed_accs.append(float(np.mean([r["is_correct"] for r in above])))
                    seed_covs.append(len(above) / N_EVAL)
                else:
                    seed_covs.append(0.0)

            acc_mean = float(np.nanmean(seed_accs)) if seed_accs else float("nan")
            cov_mean = float(np.mean(seed_covs))

            if not np.isnan(acc_mean) and acc_mean >= ACCURACY_GATE and best_t is None:
                best_t   = float(t)
                best_acc = acc_mean
                best_cov = cov_mean

        confidence_results[cat] = {
            "threshold_star":    best_t,
            "accuracy_at_star":  round(best_acc, 4) if best_acc is not None else None,
            "coverage_at_star":  round(best_cov, 4) if best_cov is not None else None,
        }

    # Per-category margin analysis
    margin_results: dict[str, dict] = {}
    for cat in CATEGORIES:
        best_m   = None
        best_acc = None
        best_cov = None

        for m in MARGIN_THRESHOLDS:
            seed_accs = []
            seed_covs = []
            for seed_recs in all_seed_records:
                above = [r for r in seed_recs
                         if r["category"] == cat and r["margin"] >= m]
                if above:
                    seed_accs.append(float(np.mean([r["is_correct"] for r in above])))
                    seed_covs.append(len(above) / N_EVAL)
                else:
                    seed_covs.append(0.0)

            acc_mean = float(np.nanmean(seed_accs)) if seed_accs else float("nan")
            cov_mean = float(np.mean(seed_covs))

            if not np.isnan(acc_mean) and acc_mean >= ACCURACY_GATE and best_m is None:
                best_m   = float(m)
                best_acc = acc_mean
                best_cov = cov_mean

        margin_results[cat] = {
            "margin_star":      best_m,
            "accuracy_at_star": round(best_acc, 4) if best_acc is not None else None,
            "coverage_at_star": round(best_cov, 4) if best_cov is not None else None,
        }

    all_results[n_warmup] = {
        "n_warmup":              n_warmup,
        "overall_accuracy":      round(overall_acc, 4),
        "confidence_thresholds": confidence_results,
        "margin_thresholds":     margin_results,
    }

    # ---- Per-warmup summary table ----
    print(f"\n--- Warmup={n_warmup} Summary ---")
    print(f"Overall accuracy: {overall_acc:.1%}")
    print(f"{'Category':<24} | {'conf t*':>7} | {'conf cov':>8} | {'margin*':>7} | {'margin cov':>10}")
    print(f"{'-'*24}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}")
    for cat in CATEGORIES:
        ct = confidence_results[cat]
        mt = margin_results[cat]
        ct_str = f"{ct['threshold_star']:.3f}" if ct["threshold_star"] is not None else "  ---  "
        cc_str = f"{ct['coverage_at_star']:.1%}" if ct["coverage_at_star"] is not None else "  ---  "
        mt_str = f"{mt['margin_star']:.2f}"      if mt["margin_star"]    is not None else "  ---  "
        mc_str = f"{mt['coverage_at_star']:.1%}" if mt["coverage_at_star"] is not None else "  ---  "
        print(f"  {cat:<22} | {ct_str:>7} | {cc_str:>8} | {mt_str:>7} | {mc_str:>10}")

# ---------------------------------------------------------------------------
# Cross-warmup summary
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== COVERAGE GROWTH CURVE (confidence threshold) ===")
print(f"{'='*70}")
header_vals = " | ".join(f"w={w:>4}" for w in WARMUP_LEVELS)
print(f"{'Category':<24} | {header_vals}")
print(f"{'-'*24}-+-" + "-+-".join(["-"*7] * len(WARMUP_LEVELS)))
for cat in CATEGORIES:
    vals = []
    for w in WARMUP_LEVELS:
        cov = all_results[w]["confidence_thresholds"][cat]["coverage_at_star"]
        vals.append(f"{cov:.1%}" if cov is not None else "  ---  ")
    print(f"  {cat:<22} | {' | '.join(vals)}")

print()
print("  (threshold* values)")
print(f"{'Category':<24} | {header_vals}")
print(f"{'-'*24}-+-" + "-+-".join(["-"*7] * len(WARMUP_LEVELS)))
for cat in CATEGORIES:
    vals = []
    for w in WARMUP_LEVELS:
        t = all_results[w]["confidence_thresholds"][cat]["threshold_star"]
        vals.append(f"{t:.3f}" if t is not None else "  ---  ")
    print(f"  {cat:<22} | {' | '.join(vals)}")

print(f"\nOverall accuracy by warmup:")
for w in WARMUP_LEVELS:
    print(f"  warmup={w:>4}: {all_results[w]['overall_accuracy']:.1%}")

print(f"\nMean coverage across categories by warmup (confidence gate):")
for w in WARMUP_LEVELS:
    covs = [
        all_results[w]["confidence_thresholds"][cat]["coverage_at_star"]
        for cat in CATEGORIES
        if all_results[w]["confidence_thresholds"][cat]["coverage_at_star"] is not None
    ]
    mean_cov = float(np.mean(covs)) if covs else 0.0
    n_cats   = len(covs)
    print(f"  warmup={w:>4}: mean coverage = {mean_cov:.1%}  ({n_cats}/6 categories with threshold*)")

# ---------------------------------------------------------------------------
# Write results JSON
# ---------------------------------------------------------------------------

json_results = {str(k): v for k, v in all_results.items()}
json_results["meta"] = {
    "experiment":    "PROD-4-WARMUP",
    "regime":        "centroidal_synthetic",
    "domain_config": DOMAIN_CONFIG,
    "warmup_levels": WARMUP_LEVELS,
    "n_seeds":       N_SEEDS,
    "n_eval":        N_EVAL,
    "tau":           TAU,
    "eta":           ETA,
    "eta_neg":       ETA_NEG,
    "noise_rate":    NOISE_RATE,
    "accuracy_gate": ACCURACY_GATE,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(json_results, fh, indent=2)

assert RESULTS_PATH.exists()
print(f"\nResults written to: {RESULTS_PATH.resolve()}")

# ---------------------------------------------------------------------------
# Launch charts
# ---------------------------------------------------------------------------

from charts import generate_charts
generate_charts(json_results, CATEGORIES, WARMUP_LEVELS)
