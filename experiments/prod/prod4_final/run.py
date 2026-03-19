"""
PROD-4 FINAL — η_neg=0.05 (product canonical)
Definitive threshold calibration for Phase 5 sprint prompt.

Regime: centroidal synthetic
Ontology: SOC product v5.0+refer (C=6, A=5, d=6)

Prior runs for reference (not overwritten):
  prod4_threshold_calibration/ — original PROD-4 with η_neg=1.0 (wrong)
  prod4_rerun_eta005/          — crash-session rerun (valid, same results expected)

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

# ---------------------------------------------------------------------------
# Config — all parameters in one place
# ---------------------------------------------------------------------------

N_SEEDS           = 50
N_WARMUP          = 200
N_DECISIONS       = 1000
TAU               = 0.1
ETA               = 0.05
ETA_NEG           = 0.05        # ← THE FIX — η_neg=1.0 is FORBIDDEN (ECE=0.49)
NOISE_RATE        = 0.10
THRESHOLD_SWEEP   = list(np.arange(0.50, 1.00, 0.01))   # 50 values
MARGIN_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ACCURACY_GATE     = 0.85        # PRE-DECLARED — do not change after seeing results
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

print(f"Domain config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Categories: {CATEGORIES}")
print(f"Actions:    {ACTIONS}")
print(f"η_neg={ETA_NEG}, τ={TAU}, η={ETA}, noise={NOISE_RATE}")
print(f"Seeds: {N_SEEDS}, warmup: {N_WARMUP}, decisions: {N_DECISIONS}")
print()

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod4_final" / "prod4_final_results.json"

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
        # Compute margin (top1 - top2 probability)
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
    # Confidence threshold*: first t where accuracy_mean >= ACCURACY_GATE
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

    # Margin*: first m where accuracy_mean >= ACCURACY_GATE
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
# refer_to_analyst confidence floor: max(0.70, first t where coverage < 10%)
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
# Recommendation: confidence vs margin vs hybrid
# ---------------------------------------------------------------------------

n_conf_pass  = sum(1 for cat in CATEGORIES if threshold_star[cat] is not None)
n_margin_ok  = sum(1 for cat in CATEGORIES if margin_star[cat] is not None)

# Compare mean coverage across categories where both exist
cats_both = [cat for cat in CATEGORIES
             if threshold_star[cat] is not None and margin_star[cat] is not None]
if cats_both:
    mean_conf_cov   = float(np.mean([coverage_at_star[cat]  for cat in cats_both]))
    mean_margin_cov = float(np.mean([coverage_at_mstar[cat] for cat in cats_both]))
else:
    mean_conf_cov   = 0.0
    mean_margin_cov = 0.0

if n_conf_pass > n_margin_ok:
    # Confidence covers more categories
    recommendation = "confidence"
elif n_margin_ok > n_conf_pass:
    recommendation = "margin"
elif mean_conf_cov >= mean_margin_cov * 1.10:
    # Confidence gives ≥10% more coverage for same accuracy — prefer confidence
    recommendation = "confidence"
elif mean_margin_cov > mean_conf_cov and n_margin_ok == n_conf_pass:
    recommendation = "hybrid (confidence for missing-margin categories, margin elsewhere)"
else:
    recommendation = "confidence"

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
gate_verdict      = f"{n_pass_conf}/6 categories pass (confidence threshold* <= 0.75)"
n_pass_margin_any = sum(1 for cat in CATEGORIES if margin_star[cat] is not None)
margin_gate_verdict = f"{n_pass_margin_any}/6 categories have a margin*"

# ---------------------------------------------------------------------------
# Build result structures
# ---------------------------------------------------------------------------

confidence_thresholds: dict[str, dict] = {}
for cat in CATEGORIES:
    confidence_thresholds[cat] = {
        "threshold_star":      threshold_star[cat],
        "accuracy_at_star":    accuracy_at_star[cat],
        "coverage_at_star":    coverage_at_star[cat],
        "accuracy_ci_at_star": accuracy_ci_at_star[cat],
    }

margin_thresholds: dict[str, dict] = {}
for cat in CATEGORIES:
    margin_thresholds[cat] = {
        "margin_star":      margin_star[cat],
        "accuracy_at_star": accuracy_at_mstar[cat],
        "coverage_at_star": coverage_at_mstar[cat],
    }

# Sweep data for charts (confidence)
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

# Sweep data for charts (margin)
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
    "experiment":    "PROD-4-FINAL",
    "regime":        "centroidal_synthetic",
    "domain_config": DOMAIN_CONFIG,
    "n_seeds":       N_SEEDS,
    "n_warmup":      N_WARMUP,
    "n_decisions":   N_DECISIONS,
    "noise_rate":    NOISE_RATE,
    "tau":           TAU,
    "eta":           ETA,
    "eta_neg":       ETA_NEG,
    "accuracy_gate": ACCURACY_GATE,
    "ontology":      {"C": C, "A": A, "d": d},
    "confidence_thresholds":    confidence_thresholds,
    "margin_thresholds":        margin_thresholds,
    "gate_verdict":             gate_verdict,
    "gate_verdict_075":         f"{n_pass_conf}/6 with threshold* <= 0.75",
    "margin_gate_verdict":      margin_gate_verdict,
    "recommendation":           recommendation,
    "refer_to_analyst_floors":  refer_to_analyst_floors,
    "sweep_data":               conf_sweep_data,
    "margin_sweep_data":        margin_sweep_data,
}

for cat in CATEGORIES:
    assert cat in results["confidence_thresholds"], f"Missing category: {cat}"
    assert cat in results["margin_thresholds"],     f"Missing category: {cat}"
assert ACCURACY_GATE == 0.85

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

assert RESULTS_PATH.exists()
print("[CHECKS] All passed")

# ---------------------------------------------------------------------------
# Print confidence threshold table
# ---------------------------------------------------------------------------

print()
print("=== PROD-4 FINAL — CONFIDENCE THRESHOLD TABLE (η_neg=0.05) ===")
print(f"Regime: centroidal synthetic, {N_SEEDS} seeds, C={C}, A={A}, τ={TAU}, η_neg={ETA_NEG}")
print()
print(f"{'Category':24s}| threshold* | accuracy                  | coverage")
print(f"{'------------------------':24s}|------------|--------------------------|--------")
for cat in CATEGORIES:
    t_star = threshold_star[cat]
    if t_star is None:
        print(f"  {cat:24s}| BELOW GATE | ---                       | ---")
    else:
        acc = accuracy_at_star[cat]
        cov = coverage_at_star[cat]
        ci  = accuracy_ci_at_star[cat]
        print(f"  {cat:22s}| {t_star:.3f}      | {acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}] | {cov:.1%}")

print()
print(f"Gate verdict: {gate_verdict}")
print(f"Gate verdict (≤0.75): {results['gate_verdict_075']}")

# ---------------------------------------------------------------------------
# Print margin threshold table
# ---------------------------------------------------------------------------

print()
print("=== MARGIN-BASED THRESHOLD TABLE ===")
print(f"{'Category':24s}| margin*  | accuracy | coverage")
print(f"{'------------------------':24s}|----------|----------|--------")
for cat in CATEGORIES:
    m_star = margin_star[cat]
    if m_star is None:
        print(f"  {cat:24s}| BELOW GATE | ---      | ---")
    else:
        acc_m = accuracy_at_mstar[cat]
        cov_m = coverage_at_mstar[cat]
        print(f"  {cat:22s}| {m_star:.2f}     | {acc_m:.1%}    | {cov_m:.1%}")

print()
print(f"Margin gate verdict: {margin_gate_verdict}")

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

print()
print("=== COMPARISON: CONFIDENCE vs MARGIN GATING ===")
print(f"{'Category':24s}| conf threshold* | conf coverage | margin* | margin coverage")
print(f"{'------------------------':24s}|-----------------|---------------|---------|----------------")
for cat in CATEGORIES:
    t_s   = threshold_star[cat]
    cov_c = coverage_at_star[cat]
    m_s   = margin_star[cat]
    cov_m = coverage_at_mstar[cat]
    t_str   = f"{t_s:.3f}          " if t_s   is not None else "BELOW GATE      "
    cc_str  = f"{cov_c:.1%}        " if cov_c is not None else "---            "
    m_str   = f"{m_s:.2f}    "       if m_s   is not None else "---     "
    cm_str  = f"{cov_m:.1%}"         if cov_m is not None else "---"
    print(f"  {cat:22s}| {t_str[:15]} | {cc_str[:13]} | {m_str[:7]} | {cm_str}")

print()
print(f"RECOMMENDATION: {recommendation}")
print("  (Based on coverage breadth and number of categories with a valid threshold)")

print()
print("=== refer_to_analyst CONFIDENCE FLOOR RECOMMENDATIONS ===")
for cat in CATEGORIES:
    print(f"  {cat}: {refer_to_analyst_floors[cat]:.2f}")

print()
print(f"Results written to: {RESULTS_PATH}")
print()
print("COPY threshold* values into Phase 5 sprint prompt.")

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
