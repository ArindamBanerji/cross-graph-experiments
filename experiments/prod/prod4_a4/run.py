"""
PROD-4-A4 — Threshold calibration on A=4 geometry (Phase 0a: refer_to_analyst removed).

The A=5 thresholds from prod4_final (theta*=0.72-0.87, coverage 2-7%) were computed
with refer_to_analyst still present, which compressed confidence distributions.
This run measures the effect of the A=4 migration on auto-approve coverage.

Config: soc_product_v50 (C=6, A=4, d=6)
Prior run (superseded for A=4 geometry): prod4_final/ (A=5, coverage 2-7%)
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

N_SEEDS         = 50
N_WARMUP        = 500
N_DECISIONS     = 500
TAU             = 0.1
ETA             = 0.05
ETA_NEG         = 0.05
NOISE_RATE      = 0.10
THRESHOLD_SWEEP = list(np.arange(0.50, 0.99, 0.01))   # 49 values
ACCURACY_GATE   = 0.85
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG   = "soc_product_v50"

N_BOOTSTRAP        = 1000
BOOTSTRAP_RNG_SEED = 8888

# A=5 reference values from prod4_final (for comparison table)
A5_REFERENCE = {
    "credential_access":    {"threshold_star": 0.72, "coverage": 0.067},
    "data_exfiltration":    {"threshold_star": 0.73, "coverage": 0.072},
    "lateral_movement":     {"threshold_star": 0.79, "coverage": 0.043},
    "threat_intel_match":   {"threshold_star": 0.81, "coverage": 0.033},
    "cloud_infrastructure": {"threshold_star": 0.82, "coverage": 0.029},
    "insider_threat":       {"threshold_star": 0.87, "coverage": 0.020},
}

RESULTS_PATH = _REPO_ROOT / "experiments" / "prod" / "prod4_a4" / "prod4_a4_results.json"

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
print(f"Threshold sweep: {THRESHOLD_SWEEP[0]:.2f}–{THRESHOLD_SWEEP[-1]:.2f} "
      f"({len(THRESHOLD_SWEEP)} values)")
print()

assert A == 4, f"Expected A=4, got A={A}. Check soc_product_v50 config."

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

# ---------------------------------------------------------------------------
# Aggregate
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
# Find threshold* per category
# ---------------------------------------------------------------------------

threshold_star:      dict[str, float | None] = {}
accuracy_at_star:    dict[str, float | None] = {}
coverage_at_star:    dict[str, float | None] = {}
accuracy_ci_at_star: dict[str, dict | None]  = {}

for c_idx, cat in enumerate(CATEGORIES):
    # threshold* = first (lowest) t where accuracy_mean >= ACCURACY_GATE
    # = highest-coverage t that still meets accuracy gate
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

# ---------------------------------------------------------------------------
# Build sweep data for charts
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Gate verdict
# ---------------------------------------------------------------------------

n_pass_conf = sum(
    1 for cat in CATEGORIES
    if threshold_star[cat] is not None and threshold_star[cat] <= 0.75
)
gate_verdict = (
    f"{n_pass_conf}/6 categories pass (confidence threshold* <= 0.75)"
)
n_cats_with_star = sum(1 for cat in CATEGORIES if threshold_star[cat] is not None)

# ---------------------------------------------------------------------------
# Inline assertions
# ---------------------------------------------------------------------------

assert N_SEEDS * N_DECISIONS == 25000, (
    f"Expected 25000 total decisions, got {N_SEEDS * N_DECISIONS}"
)
for cat in CATEGORIES:
    assert cat in threshold_star,    f"Missing category in threshold_star: {cat}"
assert ACCURACY_GATE == 0.85

print("[CHECKS] All passed")

# ---------------------------------------------------------------------------
# Build result dict
# ---------------------------------------------------------------------------

confidence_thresholds: dict[str, dict] = {}
for cat in CATEGORIES:
    confidence_thresholds[cat] = {
        "threshold_star":      threshold_star[cat],
        "accuracy_at_star":    accuracy_at_star[cat],
        "coverage_at_star":    coverage_at_star[cat],
        "accuracy_ci_at_star": accuracy_ci_at_star[cat],
    }

results = {
    "experiment":    "PROD-4-A4",
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
    "confidence_thresholds": confidence_thresholds,
    "gate_verdict":          gate_verdict,
    "sweep_data":            conf_sweep_data,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

assert RESULTS_PATH.exists()

# ---------------------------------------------------------------------------
# Print A=4 threshold table
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("=== PROD-4-A4: THRESHOLD CALIBRATION (A=4 geometry) ===")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"{N_SEEDS} seeds, warmup={N_WARMUP}, decisions={N_DECISIONS}, "
      f"tau={TAU}, eta={ETA}, eta_neg={ETA_NEG}, noise={NOISE_RATE}")
print()
print(f"{'Category':24s}| threshold* | accuracy                  | coverage")
print(f"{'':24s}|------------|---------------------------|--------")
for cat in CATEGORIES:
    t_star = threshold_star[cat]
    if t_star is None:
        print(f"  {cat:22s}| BELOW GATE | ---                       | ---")
    else:
        acc = accuracy_at_star[cat]
        cov = coverage_at_star[cat]
        ci  = accuracy_ci_at_star[cat]
        print(f"  {cat:22s}| {t_star:.3f}      | "
              f"{acc:.1%} [{ci['ci_low']:.1%},{ci['ci_high']:.1%}] | {cov:.1%}")
print()
print(f"Gate verdict: {gate_verdict}")

# ---------------------------------------------------------------------------
# Print comparison table (A=5 vs A=4)
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print(
    f"  {'Category':22s}  {'theta*(A=5)':>10}  {'Cov(A=5)':>9}  "
    f"{'theta*(A=4)':>10}  {'Cov(A=4)':>9}  {'delta_Cov':>10}"
)
print("  " + "-" * 74)
for cat in CATEGORIES:
    ref5  = A5_REFERENCE.get(cat, {})
    t5    = ref5.get("threshold_star")
    cov5  = ref5.get("coverage")
    t4    = threshold_star[cat]
    cov4  = coverage_at_star[cat]

    t5_s   = f"{t5:.2f}"  if t5   is not None else "N/A"
    cov5_s = f"{cov5:.1%}" if cov5 is not None else "N/A"
    t4_s   = f"{t4:.2f}"  if t4   is not None else "BELOW GATE"
    cov4_s = f"{cov4:.1%}" if cov4 is not None else "---"

    if cov4 is not None and cov5 is not None:
        delta  = cov4 - cov5
        delta_s = f"{delta:+.1%}"
        flag    = " ↑↑" if delta >= 0.10 else (" ↑" if delta >= 0.02 else "")
    else:
        delta_s = "N/A"
        flag    = ""

    print(f"  {cat:22s}  {t5_s:>10}  {cov5_s:>9}  {t4_s:>10}  {cov4_s:>9}  "
          f"{delta_s:>10}{flag}")

print()

# Summarize coverage change
covered = [(cat, coverage_at_star[cat]) for cat in CATEGORIES
           if coverage_at_star[cat] is not None]
if covered:
    mean_cov4 = float(np.mean([v for _, v in covered]))
    mean_cov5 = float(np.mean([A5_REFERENCE[cat]["coverage"]
                               for cat in CATEGORIES]))
    print(f"Mean coverage A=5: {mean_cov5:.1%}")
    print(f"Mean coverage A=4: {mean_cov4:.1%}  (delta: {mean_cov4 - mean_cov5:+.1%})")
    if mean_cov4 >= 0.20:
        print("  => Coverage >= 20%: '40% auto-approve' CISO story is now plausible.")
    elif mean_cov4 >= 0.10:
        print("  => Coverage >= 10%: meaningful improvement; further tuning possible.")
    else:
        print("  => Coverage still below 10%: ceiling is structural, not refer-collision.")

print()
print(f"Results written to: {RESULTS_PATH}")

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
