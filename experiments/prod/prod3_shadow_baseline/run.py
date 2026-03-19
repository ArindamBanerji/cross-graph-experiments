"""
PROD-3: Shadow Mode Agreement Rate Baseline.

Regime: centroidal synthetic
Ontology: SOC product v5.0+refer (C=6, A=5, d=6)

Measures system-vs-oracle agreement rate after warmup, under shadow conditions
(observe but do not learn).  Produces:
  - Calibration table for CISO shadow deployment interpretation
  - Per-category theta (θ) values for similar-past-cases sidebar (§23.4)

All results include 95% bootstrap CI.  tau=0.1 throughout.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle

# ---------------------------------------------------------------------------
# Config — all parameters in one place, nothing scattered
# ---------------------------------------------------------------------------

N_SEEDS               = 50
N_WARMUP              = 200
N_SHADOW              = 200
TAU                   = 0.1
ETA                   = 0.05
ETA_NEG               = 1.0
NOISE_RATE            = 0.10
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
RANDOM_SEED_BASE      = 42
DOMAIN_CONFIG         = "soc_product_v50"

N_BOOTSTRAP           = 1000
BOOTSTRAP_RNG_SEED    = 7777

# ---------------------------------------------------------------------------
# Load domain config — derive all ontology constants
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod3_shadow_baseline" / "prod3_calibration_table.json"

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

all_seed_records_raw: list[list[dict]] = []   # [seed_idx][decision_idx]
all_seed_records:     list[dict]       = []   # per-seed aggregates

# Cosine similarity accumulator: category -> list[float] (across all seeds)
cosine_sims_all: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}

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

    # ---- Warmup phase — centroids learn, nothing recorded ----
    warmup_alerts = gen.generate(N_WARMUP)
    for alert in warmup_alerts:
        result  = scorer.score(alert.factors, alert.category_index)
        correct = result.action_index == alert.gt_action_index
        # Oracle-aligned training: update on GT action, always correct=True
        scorer.update(
            alert.factors,
            alert.category_index,
            alert.gt_action_index,
            correct=True,
        )

    # ---- Shadow phase — record everything, NO centroid updates ----
    shadow_alerts = gen.generate(N_SHADOW)
    seed_records: list[dict] = []

    for dec_idx, alert in enumerate(shadow_alerts):
        result       = scorer.score(alert.factors, alert.category_index)
        system_action = result.action_index
        gt_action    = alert.gt_action_index
        confidence   = result.confidence
        agreed       = int(system_action == gt_action)

        seed_records.append({
            "seed":         seed_idx,
            "decision_idx": dec_idx,
            "category_idx": alert.category_index,
            "category":     alert.category,
            "factors":      alert.factors.tolist(),
            "system_action": system_action,
            "gt_action":    gt_action,
            "confidence":   confidence,
            "agreed":       agreed,
        })
        # CRITICAL: No scorer.update() during shadow phase

    all_seed_records_raw.append(seed_records)

    # ---- Per-seed aggregation ----
    overall_agreement = float(np.mean([r["agreed"] for r in seed_records]))

    per_cat_agreement: dict[str, float] = {}
    for cat in CATEGORIES:
        cat_recs = [r for r in seed_records if r["category"] == cat]
        per_cat_agreement[cat] = float(np.mean([r["agreed"] for r in cat_recs])) if cat_recs else float("nan")

    high_conf: dict[float, dict] = {}
    for t in CONFIDENCE_THRESHOLDS:
        filtered = [r for r in seed_records if r["confidence"] >= t]
        if filtered:
            agreement = float(np.mean([r["agreed"] for r in filtered]))
            coverage  = len(filtered) / N_SHADOW
        else:
            agreement = float("nan")
            coverage  = 0.0
        high_conf[t] = {"agreement": agreement, "coverage": coverage}

    # ---- Per-category cosine similarity accumulation ----
    for c_idx, cat in enumerate(CATEGORIES):
        cat_vecs = [r["factors"] for r in seed_records if r["category_idx"] == c_idx]
        if len(cat_vecs) >= 2:
            for i in range(len(cat_vecs)):
                for j in range(i + 1, len(cat_vecs)):
                    sim = 1.0 - cosine_dist(cat_vecs[i], cat_vecs[j])
                    cosine_sims_all[cat].append(sim)

    all_seed_records.append({
        "overall_agreement": overall_agreement,
        "per_category":      per_cat_agreement,
        "high_conf":         high_conf,
    })

# ---------------------------------------------------------------------------
# Aggregate across seeds — bootstrap CI
# ---------------------------------------------------------------------------

overall_values = [s["overall_agreement"] for s in all_seed_records]
ov_mean, ov_std, ov_ci_low, ov_ci_high = bootstrap_ci(overall_values)

per_cat_results: dict[str, dict] = {}
for cat in CATEGORIES:
    cat_values = [s["per_category"][cat] for s in all_seed_records
                  if not np.isnan(s["per_category"][cat])]
    m, s, lo, hi = bootstrap_ci(cat_values)
    per_cat_results[cat] = {"mean": m, "std": s, "ci_low": lo, "ci_high": hi}

high_conf_results: dict[str, dict] = {}
for t in CONFIDENCE_THRESHOLDS:
    agr_vals = [s["high_conf"][t]["agreement"] for s in all_seed_records
                if not np.isnan(s["high_conf"][t]["agreement"])]
    cov_vals = [s["high_conf"][t]["coverage"] for s in all_seed_records]
    if agr_vals:
        am, astd, alo, ahi = bootstrap_ci(agr_vals)
    else:
        am, astd, alo, ahi = float("nan"), float("nan"), float("nan"), float("nan")
    cm, cstd, clo, chi = bootstrap_ci(cov_vals)
    high_conf_results[str(t)] = {
        "agreement_mean": am, "agreement_std": astd,
        "agreement_ci_low": alo, "agreement_ci_high": ahi,
        "coverage_mean": cm,   "coverage_std": cstd,
        "coverage_ci_low": clo, "coverage_ci_high": chi,
    }

# Theta per category: p25 of all pairwise cosine sims
theta_results: dict[str, dict] = {}
for cat in CATEGORIES:
    sims = cosine_sims_all[cat]
    if sims:
        p25     = float(np.percentile(sims, 25))
        n_pairs = len(sims)
    else:
        p25     = float("nan")
        n_pairs = 0
    theta_results[cat] = {"p25": p25, "n_pairs": n_pairs}

# ---------------------------------------------------------------------------
# Inline checks
# ---------------------------------------------------------------------------

assert len(all_seed_records) == N_SEEDS, f"Expected {N_SEEDS} seed results"
assert all(len(CATEGORIES) == C for _ in [1])
total_records = sum(len(sr) for sr in all_seed_records_raw)
assert total_records == N_SEEDS * N_SHADOW, (
    f"Expected {N_SEEDS * N_SHADOW} shadow records, got {total_records}"
)
for cat in CATEGORIES:
    assert cat in per_cat_results, f"Missing category: {cat}"
    assert not np.isnan(per_cat_results[cat]["mean"]), f"NaN for {cat}"
for cat in CATEGORIES:
    assert cat in theta_results, f"Missing θ for {cat}"
print(f"[CHECKS] All {len(CATEGORIES)} categories present, no NaN values")

# ---------------------------------------------------------------------------
# Print calibration table
# ---------------------------------------------------------------------------

print()
print("=== SHADOW MODE CALIBRATION TABLE (v5.5) ===")
print(f"Regime: centroidal synthetic, {N_SEEDS} seeds, noise_rate={NOISE_RATE}")
print(f"Ontology: C={C}, A={A}, d={d} (SOC product v5.0+refer)")
print(f"Overall agreement rate: {ov_mean:.1%} ± {ov_std:.1%} [{ov_ci_low:.1%}, {ov_ci_high:.1%}]")
print("Per-category:")
for cat in CATEGORIES:
    r = per_cat_results[cat]
    print(f"  {cat}: {r['mean']:.1%} [{r['ci_low']:.1%}, {r['ci_high']:.1%}]")

r90 = high_conf_results["0.9"]
print(f"At P>=0.90: agreement={r90['agreement_mean']:.1%} ± {r90['agreement_std']:.1%}, "
      f"coverage={r90['coverage_mean']:.1%} ± {r90['coverage_std']:.1%}")

print()
print("=== §23.4 SIMILARITY THRESHOLD (θ) RECOMMENDATIONS ===")
print("Method: p25 of pairwise cosine similarity within category")
for cat in CATEGORIES:
    tr = theta_results[cat]
    print(f"  {cat}: θ = {tr['p25']:.3f} (N={tr['n_pairs']} pairs)")
print()
print("COPY THESE θ VALUES INTO soc_copilot_design_v5_4 §23.4.1")

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

results = {
    "regime":        "centroidal_synthetic",
    "domain_config": DOMAIN_CONFIG,
    "n_seeds":       N_SEEDS,
    "n_warmup":      N_WARMUP,
    "n_shadow":      N_SHADOW,
    "noise_rate":    NOISE_RATE,
    "tau":           TAU,
    "ontology":      {"C": C, "A": A, "d": d},
    "overall":       {"mean": ov_mean, "std": ov_std, "ci_low": ov_ci_low, "ci_high": ov_ci_high},
    "per_category":  per_cat_results,
    "high_conf":     high_conf_results,
    "theta_recommendations": theta_results,
    "per_seed_overall": overall_values,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nResults written to: {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Launch charts
# ---------------------------------------------------------------------------

import importlib.util, subprocess
charts_path = Path(__file__).parent / "charts.py"
subprocess.run(
    [sys.executable, str(charts_path)],
    check=True,
    cwd=str(_REPO_ROOT),
)
