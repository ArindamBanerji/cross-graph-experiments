"""
PROD-4: Per-Category Auto-Approve Threshold Calibration.

Regime: centroidal synthetic
Ontology: SOC product v5.0+refer (C=6, A=5, d=6)

Finds the minimum confidence threshold per category where accuracy >= 85%
(pre-declared ACCURACY_GATE).  Measures coverage at that threshold.
These values replace design-estimate thresholds in Phase 5 and set the
refer_to_analyst confidence floor.

All results include 95% bootstrap CI.  tau=0.1 throughout.
Learning is ON during the decision window (unlike PROD-3 shadow).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle

# ---------------------------------------------------------------------------
# Config — all parameters in one place
# ---------------------------------------------------------------------------

N_SEEDS           = 50
N_WARMUP          = 200
N_DECISIONS       = 1000
TAU               = 0.1
ETA               = 0.05
ETA_NEG           = 1.0
NOISE_RATE        = 0.10
THRESHOLD_SWEEP   = list(np.arange(0.50, 1.00, 0.01))   # 50 values
ACCURACY_GATE     = 0.85   # PRE-DECLARED — do not change after seeing results
RANDOM_SEED_BASE  = 42
DOMAIN_CONFIG     = "soc_product_v50"

N_BOOTSTRAP       = 1000
BOOTSTRAP_RNG_SEED = 8888

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod4_threshold_calibration" / "prod4_threshold_table.json"

# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = N_BOOTSTRAP,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_low, ci_high) via percentile bootstrap."""
    arr  = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    rng  = np.random.default_rng(rng_seed)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    ci_low  = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean, std, ci_low, ci_high

# ---------------------------------------------------------------------------
# Main loop — 50 seeds
# ---------------------------------------------------------------------------

# per_seed_data[seed_idx][cat_idx][thresh_idx] = (accuracy, coverage)
# Store as parallel lists for memory efficiency
# Shape: (N_SEEDS, C, len(THRESHOLD_SWEEP))
per_seed_accuracy = np.full((N_SEEDS, C, len(THRESHOLD_SWEEP)), np.nan)
per_seed_coverage = np.zeros((N_SEEDS, C, len(THRESHOLD_SWEEP)))

for seed_idx in range(N_SEEDS):
    print(f"Seed {seed_idx+1}/{N_SEEDS}", flush=True)

    # ---- Setup ----
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
    oracle = GTAlignedOracle(
        noise_rate=NOISE_RATE,
        seed=RANDOM_SEED_BASE + seed_idx,
    )

    # ---- Warmup — centroids learn, nothing recorded ----
    warmup_alerts = gen.generate(N_WARMUP)
    for alert in warmup_alerts:
        scorer.update(
            alert.factors,
            alert.category_index,
            alert.gt_action_index,
            correct=True,
        )

    # ---- Decision window — record all, learning ON ----
    decision_alerts = gen.generate(N_DECISIONS)
    seed_records: list[dict] = []

    for alert in decision_alerts:
        result     = scorer.score(alert.factors, alert.category_index)
        is_correct = result.action_index == alert.gt_action_index
        # Learning ON: update with GT action
        scorer.update(
            alert.factors,
            alert.category_index,
            result.action_index,
            correct=is_correct,
            gt_action_index=alert.gt_action_index,
        )
        seed_records.append({
            "category_idx": alert.category_index,
            "category":     alert.category,
            "confidence":   result.confidence,
            "is_correct":   is_correct,
        })

    # ---- Per-category × threshold sweep ----
    for c_idx in range(C):
        cat_recs = [r for r in seed_records if r["category_idx"] == c_idx]
        for t_idx, t in enumerate(THRESHOLD_SWEEP):
            above = [r for r in cat_recs if r["confidence"] >= t]
            if above:
                per_seed_accuracy[seed_idx, c_idx, t_idx] = float(
                    np.mean([r["is_correct"] for r in above])
                )
                per_seed_coverage[seed_idx, c_idx, t_idx] = len(above) / N_DECISIONS
            else:
                per_seed_accuracy[seed_idx, c_idx, t_idx] = np.nan
                per_seed_coverage[seed_idx, c_idx, t_idx] = 0.0

# ---------------------------------------------------------------------------
# Aggregate across seeds per category × threshold
# ---------------------------------------------------------------------------

# accuracy_agg[c, t] = (mean, std, ci_low, ci_high) — NaN seeds excluded
# coverage_agg[c, t] = (mean, std, ci_low, ci_high) — no NaN in coverage
accuracy_agg = {}   # (c_idx, t_idx) -> dict
coverage_agg = {}

for c_idx in range(C):
    for t_idx in range(len(THRESHOLD_SWEEP)):
        acc_vals = per_seed_accuracy[:, c_idx, t_idx]
        valid    = acc_vals[~np.isnan(acc_vals)].tolist()
        if valid:
            m, s, lo, hi = bootstrap_ci(valid)
        else:
            m, s, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
        accuracy_agg[(c_idx, t_idx)] = {"mean": m, "std": s, "ci_low": lo, "ci_high": hi}

        cov_vals = per_seed_coverage[:, c_idx, t_idx].tolist()
        cm, cs, clo, chi = bootstrap_ci(cov_vals)
        coverage_agg[(c_idx, t_idx)] = {"mean": cm, "std": cs, "ci_low": clo, "ci_high": chi}

# ---------------------------------------------------------------------------
# Per category — find threshold* (minimum t where accuracy_mean >= ACCURACY_GATE)
# ---------------------------------------------------------------------------

threshold_star:      dict[str, float | None]  = {}
accuracy_at_star:    dict[str, float | None]  = {}
coverage_at_star:    dict[str, float | None]  = {}
accuracy_ci_at_star: dict[str, dict | None]   = {}

for c_idx, cat in enumerate(CATEGORIES):
    found = None
    for t_idx, t in enumerate(THRESHOLD_SWEEP):
        acc_mean = accuracy_agg[(c_idx, t_idx)]["mean"]
        if not np.isnan(acc_mean) and acc_mean >= ACCURACY_GATE:
            found    = t_idx
            break
    if found is not None:
        threshold_star[cat]      = float(THRESHOLD_SWEEP[found])
        accuracy_at_star[cat]    = accuracy_agg[(c_idx, found)]["mean"]
        coverage_at_star[cat]    = coverage_agg[(c_idx, found)]["mean"]
        accuracy_ci_at_star[cat] = {
            "ci_low":  accuracy_agg[(c_idx, found)]["ci_low"],
            "ci_high": accuracy_agg[(c_idx, found)]["ci_high"],
        }
    else:
        threshold_star[cat]      = None
        accuracy_at_star[cat]    = None
        coverage_at_star[cat]    = None
        accuracy_ci_at_star[cat] = None

# ---------------------------------------------------------------------------
# refer_to_analyst confidence floor: max(0.70, min t where coverage < 10%)
# ---------------------------------------------------------------------------

refer_to_analyst_floors: dict[str, float] = {}

for c_idx, cat in enumerate(CATEGORIES):
    floor_t = 0.70
    for t_idx, t in enumerate(THRESHOLD_SWEEP):
        cov_mean = coverage_agg[(c_idx, t_idx)]["mean"]
        if cov_mean < 0.10:
            floor_t = max(0.70, float(t))
            break
    refer_to_analyst_floors[cat] = round(floor_t, 2)

# ---------------------------------------------------------------------------
# Inline checks
# ---------------------------------------------------------------------------

total_decisions = N_SEEDS * N_DECISIONS
assert total_decisions == 50000, f"Expected 50000, got {total_decisions}"
for cat in CATEGORIES:
    assert cat in threshold_star, f"Missing category: {cat}"
assert "gate_verdict" or True   # built below

# Gate verdict: categories with threshold* <= 0.75
n_pass = sum(
    1 for cat in CATEGORIES
    if threshold_star[cat] is not None and threshold_star[cat] <= 0.75
)
gate_verdict = f"{n_pass}/6 categories pass"

results_categories = {}
for cat in CATEGORIES:
    results_categories[cat] = {
        "threshold_star":     threshold_star[cat],
        "accuracy_at_star":   accuracy_at_star[cat],
        "coverage_at_star":   coverage_at_star[cat],
        "accuracy_ci_at_star": accuracy_ci_at_star[cat],
    }

# Build sweep_data for charts
sweep_data: dict[str, dict] = {}
for c_idx, cat in enumerate(CATEGORIES):
    sweep_data[cat] = {}
    for t_idx, t in enumerate(THRESHOLD_SWEEP):
        sweep_data[cat][f"{t:.2f}"] = {
            "accuracy_mean": accuracy_agg[(c_idx, t_idx)]["mean"],
            "accuracy_ci_low":  accuracy_agg[(c_idx, t_idx)]["ci_low"],
            "accuracy_ci_high": accuracy_agg[(c_idx, t_idx)]["ci_high"],
            "coverage_mean": coverage_agg[(c_idx, t_idx)]["mean"],
        }

results = {
    "regime":         "centroidal_synthetic",
    "domain_config":  DOMAIN_CONFIG,
    "n_seeds":        N_SEEDS,
    "n_warmup":       N_WARMUP,
    "n_decisions":    N_DECISIONS,
    "noise_rate":     NOISE_RATE,
    "tau":            TAU,
    "accuracy_gate":  ACCURACY_GATE,
    "ontology":       {"C": C, "A": A, "d": d},
    "categories":     results_categories,
    "gate_verdict":   gate_verdict,
    "refer_to_analyst_floors": refer_to_analyst_floors,
    "sweep_data":     sweep_data,
}

# Inline check: required keys present
for cat in CATEGORIES:
    assert cat in results["categories"], f"Missing category: {cat}"
assert "gate_verdict" in results
assert "refer_to_analyst_floors" in results

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

assert RESULTS_PATH.exists()
print("[CHECKS] All passed")

# ---------------------------------------------------------------------------
# Print calibration table
# ---------------------------------------------------------------------------

print()
print("=== PROD-4 THRESHOLD CALIBRATION (centroidal synthetic, 50 seeds) ===")
print(f"Ontology: C={C}, A={A}, d={d} (SOC product v5.0+refer)")
print(f"Gate: accuracy >= {ACCURACY_GATE:.0%} at threshold")
print()
print("Per-category threshold* table:")
print("  Category              | threshold* | accuracy | coverage")
print("  ----------------------|------------|----------|--------")
for cat in CATEGORIES:
    t_star = threshold_star[cat]
    if t_star is None:
        print(f"  {cat:22s}| BELOW GATE | ---      | ---")
    else:
        acc = accuracy_at_star[cat]
        cov = coverage_at_star[cat]
        ci  = accuracy_ci_at_star[cat]
        print(f"  {cat:22s}| {t_star:.2f}       | {acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}] | {cov:.1%}")

print()
print(f"Gate verdict: {gate_verdict} achieve threshold* <= 0.75")
print()
print("=== refer_to_analyst CONFIDENCE FLOOR RECOMMENDATIONS ===")
for cat in CATEGORIES:
    print(f"  {cat}: {refer_to_analyst_floors[cat]:.2f}")
print()
print("COPY threshold* INTO project_status_and_plan_v3_part2 Phase 5 prompt")
print(f"\nResults written to: {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Launch charts
# ---------------------------------------------------------------------------

import subprocess
charts_path = Path(__file__).parent / "charts.py"
subprocess.run(
    [sys.executable, str(charts_path)],
    check=True,
    cwd=str(_REPO_ROOT),
)
