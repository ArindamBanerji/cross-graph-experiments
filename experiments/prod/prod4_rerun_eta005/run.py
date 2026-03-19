"""
PROD-4 RERUN — η_neg=0.05 (product canonical)
Original PROD-4 used η_neg=1.0 which produced catastrophic miscalibration.
PROD-4b (March 14) confirmed η_neg=0.05 passes the 85% gate for all 6 categories.
This re-run produces the actual threshold* table for Phase 5 sprint prompt.

Regime: centroidal synthetic
Ontology: SOC product v5.0+refer (C=6, A=5, d=6)

All results include 95% bootstrap CI.  tau=0.1, eta_neg=0.05 throughout.
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
ETA_NEG           = 0.05       # ← THE FIX — was 1.0 in original PROD-4
NOISE_RATE        = 0.10
THRESHOLD_SWEEP   = list(np.arange(0.50, 1.00, 0.01))   # 50 values
MARGIN_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ACCURACY_GATE     = 0.85   # PRE-DECLARED — do not change after seeing results
RANDOM_SEED_BASE  = 42
DOMAIN_CONFIG     = "soc_product_v50"

N_BOOTSTRAP        = 1000
BOOTSTRAP_RNG_SEED = 8888

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod4_rerun_eta005" / "prod4_rerun_results.json"

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

# Confidence sweep arrays: (N_SEEDS, C, len(THRESHOLD_SWEEP))
per_seed_conf_accuracy = np.full((N_SEEDS, C, len(THRESHOLD_SWEEP)), np.nan)
per_seed_conf_coverage = np.zeros((N_SEEDS, C, len(THRESHOLD_SWEEP)))

# Margin sweep arrays: (N_SEEDS, C, len(MARGIN_THRESHOLDS))
per_seed_margin_accuracy = np.full((N_SEEDS, C, len(MARGIN_THRESHOLDS)), np.nan)
per_seed_margin_coverage = np.zeros((N_SEEDS, C, len(MARGIN_THRESHOLDS)))

for seed_idx in range(N_SEEDS):
    print(f"Seed {seed_idx+1}/{N_SEEDS}", flush=True)

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
        scorer.update(
            alert.factors,
            alert.category_index,
            result.action_index,
            correct=is_correct,
            gt_action_index=alert.gt_action_index,
        )
        probs_sorted = np.sort(result.probabilities)
        margin       = float(probs_sorted[-1] - probs_sorted[-2])
        seed_records.append({
            "category_idx": alert.category_index,
            "category":     alert.category,
            "confidence":   float(result.confidence),
            "margin":       margin,
            "is_correct":   bool(is_correct),
        })

    # ---- Per-category × confidence threshold sweep ----
    for c_idx in range(C):
        cat_recs = [r for r in seed_records if r["category_idx"] == c_idx]
        for t_idx, t in enumerate(THRESHOLD_SWEEP):
            above = [r for r in cat_recs if r["confidence"] >= t]
            if above:
                per_seed_conf_accuracy[seed_idx, c_idx, t_idx] = float(
                    np.mean([r["is_correct"] for r in above])
                )
                per_seed_conf_coverage[seed_idx, c_idx, t_idx] = len(above) / N_DECISIONS
            else:
                per_seed_conf_accuracy[seed_idx, c_idx, t_idx] = np.nan
                per_seed_conf_coverage[seed_idx, c_idx, t_idx] = 0.0

        # ---- Per-category × margin threshold sweep ----
        for m_idx, m in enumerate(MARGIN_THRESHOLDS):
            above_m = [r for r in cat_recs if r["margin"] >= m]
            if above_m:
                per_seed_margin_accuracy[seed_idx, c_idx, m_idx] = float(
                    np.mean([r["is_correct"] for r in above_m])
                )
                per_seed_margin_coverage[seed_idx, c_idx, m_idx] = len(above_m) / N_DECISIONS
            else:
                per_seed_margin_accuracy[seed_idx, c_idx, m_idx] = np.nan
                per_seed_margin_coverage[seed_idx, c_idx, m_idx] = 0.0

# ---------------------------------------------------------------------------
# Aggregate: confidence sweep
# ---------------------------------------------------------------------------

conf_accuracy_agg: dict[tuple, dict] = {}
conf_coverage_agg: dict[tuple, dict] = {}

for c_idx in range(C):
    for t_idx in range(len(THRESHOLD_SWEEP)):
        acc_vals = per_seed_conf_accuracy[:, c_idx, t_idx]
        valid    = acc_vals[~np.isnan(acc_vals)].tolist()
        if valid:
            m, s, lo, hi = bootstrap_ci(valid)
        else:
            m, s, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
        conf_accuracy_agg[(c_idx, t_idx)] = {"mean": m, "std": s, "ci_low": lo, "ci_high": hi}

        cov_vals = per_seed_conf_coverage[:, c_idx, t_idx].tolist()
        cm, cs, clo, chi = bootstrap_ci(cov_vals)
        conf_coverage_agg[(c_idx, t_idx)] = {"mean": cm, "std": cs, "ci_low": clo, "ci_high": chi}

# ---------------------------------------------------------------------------
# Aggregate: margin sweep
# ---------------------------------------------------------------------------

margin_accuracy_agg: dict[tuple, dict] = {}
margin_coverage_agg: dict[tuple, dict] = {}

for c_idx in range(C):
    for m_idx in range(len(MARGIN_THRESHOLDS)):
        acc_vals = per_seed_margin_accuracy[:, c_idx, m_idx]
        valid    = acc_vals[~np.isnan(acc_vals)].tolist()
        if valid:
            m, s, lo, hi = bootstrap_ci(valid)
        else:
            m, s, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
        margin_accuracy_agg[(c_idx, m_idx)] = {"mean": m, "std": s, "ci_low": lo, "ci_high": hi}

        cov_vals = per_seed_margin_coverage[:, c_idx, m_idx].tolist()
        cm, cs, clo, chi = bootstrap_ci(cov_vals)
        margin_coverage_agg[(c_idx, m_idx)] = {"mean": cm, "std": cs, "ci_low": clo, "ci_high": chi}

# ---------------------------------------------------------------------------
# Per category — find confidence threshold* and margin*
# ---------------------------------------------------------------------------

threshold_star:       dict[str, float | None] = {}
accuracy_at_star:     dict[str, float | None] = {}
coverage_at_star:     dict[str, float | None] = {}
accuracy_ci_at_star:  dict[str, dict | None]  = {}

margin_star:          dict[str, float | None] = {}
accuracy_at_mstar:    dict[str, float | None] = {}
coverage_at_mstar:    dict[str, float | None] = {}

for c_idx, cat in enumerate(CATEGORIES):
    # Confidence threshold*
    found_t = None
    for t_idx, t in enumerate(THRESHOLD_SWEEP):
        acc_mean = conf_accuracy_agg[(c_idx, t_idx)]["mean"]
        if not np.isnan(acc_mean) and acc_mean >= ACCURACY_GATE:
            found_t = t_idx
            break
    if found_t is not None:
        threshold_star[cat]      = float(THRESHOLD_SWEEP[found_t])
        accuracy_at_star[cat]    = conf_accuracy_agg[(c_idx, found_t)]["mean"]
        coverage_at_star[cat]    = conf_coverage_agg[(c_idx, found_t)]["mean"]
        accuracy_ci_at_star[cat] = {
            "ci_low":  conf_accuracy_agg[(c_idx, found_t)]["ci_low"],
            "ci_high": conf_accuracy_agg[(c_idx, found_t)]["ci_high"],
        }
    else:
        threshold_star[cat]      = None
        accuracy_at_star[cat]    = None
        coverage_at_star[cat]    = None
        accuracy_ci_at_star[cat] = None

    # Margin*
    found_m = None
    for m_idx, m in enumerate(MARGIN_THRESHOLDS):
        acc_mean = margin_accuracy_agg[(c_idx, m_idx)]["mean"]
        if not np.isnan(acc_mean) and acc_mean >= ACCURACY_GATE:
            found_m = m_idx
            break
    if found_m is not None:
        margin_star[cat]       = float(MARGIN_THRESHOLDS[found_m])
        accuracy_at_mstar[cat] = margin_accuracy_agg[(c_idx, found_m)]["mean"]
        coverage_at_mstar[cat] = margin_coverage_agg[(c_idx, found_m)]["mean"]
    else:
        margin_star[cat]       = None
        accuracy_at_mstar[cat] = None
        coverage_at_mstar[cat] = None

# ---------------------------------------------------------------------------
# refer_to_analyst confidence floor: max(0.70, min t where coverage < 10%)
# ---------------------------------------------------------------------------

refer_to_analyst_floors: dict[str, float] = {}
for c_idx, cat in enumerate(CATEGORIES):
    floor_t = 0.70
    for t_idx, t in enumerate(THRESHOLD_SWEEP):
        if conf_coverage_agg[(c_idx, t_idx)]["mean"] < 0.10:
            floor_t = max(0.70, float(t))
            break
    refer_to_analyst_floors[cat] = round(floor_t, 2)

# ---------------------------------------------------------------------------
# Inline checks
# ---------------------------------------------------------------------------

total_decisions = N_SEEDS * N_DECISIONS
assert total_decisions == 50000, f"Expected 50000, got {total_decisions}"
for cat in CATEGORIES:
    assert cat in threshold_star, f"Missing category in threshold_star: {cat}"
    assert cat in margin_star,    f"Missing category in margin_star: {cat}"

n_pass_conf = sum(
    1 for cat in CATEGORIES
    if threshold_star[cat] is not None and threshold_star[cat] <= 0.75
)
gate_verdict = f"{n_pass_conf}/6 categories pass (confidence threshold* <= 0.75)"

n_pass_margin = sum(1 for cat in CATEGORIES if margin_star[cat] is not None)
margin_gate_verdict = f"{n_pass_margin}/6 categories have a margin*"

# Build results dicts
results_categories: dict[str, dict] = {}
for cat in CATEGORIES:
    results_categories[cat] = {
        "threshold_star":      threshold_star[cat],
        "accuracy_at_star":    accuracy_at_star[cat],
        "coverage_at_star":    coverage_at_star[cat],
        "accuracy_ci_at_star": accuracy_ci_at_star[cat],
        "margin_star":         margin_star[cat],
        "accuracy_at_mstar":   accuracy_at_mstar[cat],
        "coverage_at_mstar":   coverage_at_mstar[cat],
    }

conf_sweep_data: dict[str, dict] = {}
for c_idx, cat in enumerate(CATEGORIES):
    conf_sweep_data[cat] = {}
    for t_idx, t in enumerate(THRESHOLD_SWEEP):
        conf_sweep_data[cat][f"{t:.2f}"] = {
            "accuracy_mean":    conf_accuracy_agg[(c_idx, t_idx)]["mean"],
            "accuracy_ci_low":  conf_accuracy_agg[(c_idx, t_idx)]["ci_low"],
            "accuracy_ci_high": conf_accuracy_agg[(c_idx, t_idx)]["ci_high"],
            "coverage_mean":    conf_coverage_agg[(c_idx, t_idx)]["mean"],
        }

margin_sweep_data: dict[str, dict] = {}
for c_idx, cat in enumerate(CATEGORIES):
    margin_sweep_data[cat] = {}
    for m_idx, m in enumerate(MARGIN_THRESHOLDS):
        margin_sweep_data[cat][f"{m:.2f}"] = {
            "accuracy_mean":    margin_accuracy_agg[(c_idx, m_idx)]["mean"],
            "accuracy_ci_low":  margin_accuracy_agg[(c_idx, m_idx)]["ci_low"],
            "accuracy_ci_high": margin_accuracy_agg[(c_idx, m_idx)]["ci_high"],
            "coverage_mean":    margin_coverage_agg[(c_idx, m_idx)]["mean"],
        }

results = {
    "regime":          "centroidal_synthetic",
    "eta_neg":         ETA_NEG,
    "domain_config":   DOMAIN_CONFIG,
    "n_seeds":         N_SEEDS,
    "n_warmup":        N_WARMUP,
    "n_decisions":     N_DECISIONS,
    "noise_rate":      NOISE_RATE,
    "tau":             TAU,
    "accuracy_gate":   ACCURACY_GATE,
    "ontology":        {"C": C, "A": A, "d": d},
    "categories":      results_categories,
    "gate_verdict":    gate_verdict,
    "margin_gate_verdict": margin_gate_verdict,
    "refer_to_analyst_floors": refer_to_analyst_floors,
    "sweep_data":      conf_sweep_data,
    "margin_sweep_data": margin_sweep_data,
}

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
# Print confidence threshold table
# ---------------------------------------------------------------------------

print()
print("=== PROD-4 RERUN — η_neg=0.05 (product canonical) ===")
print(f"THRESHOLD CALIBRATION (centroidal synthetic, {N_SEEDS} seeds)")
print(f"Ontology: C={C}, A={A}, d={d} (SOC product v5.0+refer)")
print(f"Gate: accuracy >= {ACCURACY_GATE:.0%} at threshold")
print()
print("Per-category threshold* table (confidence gating):")
print("  Category              | threshold* | accuracy              | coverage")
print("  ----------------------|------------|----------------------|--------")
for cat in CATEGORIES:
    t_star = threshold_star[cat]
    if t_star is None:
        print(f"  {cat:22s}| BELOW GATE | ---                  | ---")
    else:
        acc = accuracy_at_star[cat]
        cov = coverage_at_star[cat]
        ci  = accuracy_ci_at_star[cat]
        print(f"  {cat:22s}| {t_star:.2f}       | {acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}] | {cov:.1%}")

print()
print(f"Gate verdict: {gate_verdict}")

# ---------------------------------------------------------------------------
# Print margin analysis
# ---------------------------------------------------------------------------

print()
print("=== MARGIN-BASED THRESHOLD ALTERNATIVE ===")
print("Category              | margin* (≥85% acc) | accuracy | coverage")
print("----------------------|--------------------|----------|--------")
for cat in CATEGORIES:
    m_star = margin_star[cat]
    if m_star is None:
        print(f"  {cat:22s}| BELOW GATE         | ---      | ---")
    else:
        acc_m = accuracy_at_mstar[cat]
        cov_m = coverage_at_mstar[cat]
        print(f"  {cat:22s}| {m_star:.2f}               | {acc_m:.1%}    | {cov_m:.1%}")

print()
print(f"Margin gate verdict: {margin_gate_verdict}")

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
