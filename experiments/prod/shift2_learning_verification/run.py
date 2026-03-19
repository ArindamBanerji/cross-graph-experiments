"""
SHIFT-2 — Rigorous Learning Impact Verification

Addresses all five methodological concerns from SHIFT-1:

  (a) FIXED: scorer is FROZEN after warmup. Eval is stationary — no updates.
  (b) FIXED: coverage computed via sklearn precision_recall_curve (same as DISC-1).
  (c) VERIFIED: update() is purely deterministic (no RNG). Score is deterministic.
       RNG alignment: n_warmup=0 produces first N_EVAL alerts; n_warmup=N produces
       alerts N+1..N+N_EVAL. Eval alerts differ across warmup levels (intentional —
       we test scorer STATE, not specific alerts; averaged over 50 seeds).
  (d) FIXED: noise_rate=0.00 control added.
  (e) FIXED: per-category breakdown at noise=0.10, delta=0.00.

Key interpretation chart: zero-noise diagnostic (noise=0.00).
  If lift > 0 at noise=0, delta>0   → learning works, noise was the problem
  If lift ≈ 0 at noise=0, delta=0   → circularity confirmed (expected)
  If lift < 0 at noise=0            → update rule is architecturally misaligned

Regime: centroidal synthetic. Ontology: SOC product v5.0+refer (C=6, A=5, d=6).
Shift direction: same as SHIFT-1 (seed=12345) for comparability.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

_REPO_ROOT  = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SEEDS         = 50
N_WARMUP_LEVELS = [0, 200, 500, 1000]
N_EVAL          = 1000
TAU             = 0.1
ETA             = 0.05
ETA_NEG         = 0.05
NOISE_RATES     = [0.0, 0.05, 0.10]
DELTA_LEVELS    = [0.0, 0.10]
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG   = "soc_product_v50"
ACCURACY_GATE   = 0.85

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

mu_zero = config["mu"].copy()    # shape (C, A, d) — expert prior / warm start

print(f"Domain config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Warmup levels: {N_WARMUP_LEVELS}")
print(f"Noise rates:   {NOISE_RATES}")
print(f"Delta levels:  {DELTA_LEVELS}")
print(f"Seeds: {N_SEEDS}, N_eval: {N_EVAL}")
print(f"Total conditions: {len(N_WARMUP_LEVELS)*len(NOISE_RATES)*len(DELTA_LEVELS)}")
print(f"Total runs: {len(N_WARMUP_LEVELS)*len(NOISE_RATES)*len(DELTA_LEVELS)*N_SEEDS}")
print()

RESULTS_PATH = _SCRIPT_DIR / "shift2_results.json"

# ---------------------------------------------------------------------------
# Fixed shift direction (SAME seed as SHIFT-1 for comparability)
# ---------------------------------------------------------------------------

rng_direction = np.random.RandomState(12345)
raw_direction = rng_direction.randn(C, A, d)
norms = np.linalg.norm(raw_direction, axis=-1, keepdims=True)
norms[norms == 0] = 1.0
unit_direction = raw_direction / norms    # shape (C, A, d), unit norm per (c,a) cell

print(f"Shift direction: seed=12345, unit-normalised per cell (matches SHIFT-1)")
print()

# ---------------------------------------------------------------------------
# Helper: compute coverage at a precision gate via sklearn
# ---------------------------------------------------------------------------

def coverage_at_precision(y_true: np.ndarray, y_score: np.ndarray,
                           gate: float) -> float:
    """Highest recall achievable at precision >= gate.  Returns 0.0 if none."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    best = 0.0
    for p, r in zip(prec, rec):
        if p >= gate and r > best:
            best = r
    return float(best)

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

all_results: dict[str, dict] = {}

for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:

        # Compute shifted truth: mu_true = clip(mu_zero + delta * direction, 0, 1)
        mu_true      = np.clip(mu_zero + delta * unit_direction, 0.0, 1.0)
        actual_shift = float(np.linalg.norm(mu_true - mu_zero, axis=-1).mean())

        # Build shifted generator config (option b: constructor kwarg injection)
        shifted_profiles: dict[str, dict[str, list[float]]] = {
            cat: {
                act: mu_true[ci, ai, :].tolist()
                for ai, act in enumerate(ACTIONS)
            }
            for ci, cat in enumerate(CATEGORIES)
        }
        gen_config = dict(config["generator_kwargs"])
        gen_config["action_conditional_profiles"] = shifted_profiles

        for n_warmup in N_WARMUP_LEVELS:
            condition_key = (
                f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup{n_warmup}"
            )
            print(f"\n--- {condition_key} ---", flush=True)

            all_eval_records: list[dict] = []
            centroid_drifts:  list[float] = []

            for seed in range(N_SEEDS):
                if seed % 25 == 0:
                    print(f"  Seed {seed+1}/{N_SEEDS}", flush=True)

                gen = CategoryAlertGenerator(
                    **gen_config,
                    noise_rate=noise_rate,
                    seed=RANDOM_SEED_BASE + seed,
                )
                scorer = ProfileScorer(
                    mu_zero.copy(),
                    config["actions"],
                    tau=TAU,
                    eta=ETA,
                    eta_neg=ETA_NEG,
                )

                # ===== WARMUP PHASE — learning ON =====
                if n_warmup > 0:
                    warmup_alerts = gen.generate(n_warmup)
                    for alert in warmup_alerts:
                        result     = scorer.score(alert.factors, alert.category_index)
                        is_correct = result.action_index == alert.gt_action_index
                        scorer.update(
                            alert.factors,
                            alert.category_index,
                            result.action_index,
                            correct=is_correct,
                            gt_action_index=alert.gt_action_index,
                        )
                # n_warmup == 0: no warmup, scorer stays at mu_zero exactly

                # ===== FREEZE: measure drift, then NO MORE UPDATES =====
                drift = float(np.linalg.norm(scorer.mu - mu_zero, axis=-1).mean())
                centroid_drifts.append(drift)

                # ===== EVAL PHASE — FROZEN scorer, no updates =====
                eval_alerts = gen.generate(N_EVAL)
                for alert in eval_alerts:
                    result     = scorer.score(alert.factors, alert.category_index)
                    is_correct = result.action_index == alert.gt_action_index

                    probs        = result.probabilities
                    sorted_probs = np.sort(probs)[::-1]
                    confidence   = float(sorted_probs[0])
                    margin       = float(sorted_probs[0] - sorted_probs[1])

                    # NO scorer.update() here — frozen measurement
                    all_eval_records.append({
                        "seed":       seed,
                        "category":   alert.category,
                        "confidence": confidence,
                        "margin":     margin,
                        "is_correct": bool(is_correct),
                    })

            # ===== AGGREGATE METRICS =====
            y_true       = np.array([r["is_correct"]  for r in all_eval_records]).astype(int)
            y_conf       = np.array([r["confidence"]   for r in all_eval_records])
            cats_arr     = np.array([r["category"]     for r in all_eval_records])

            overall_acc  = float(y_true.mean())
            mean_drift   = float(np.mean(centroid_drifts))
            cov_85       = coverage_at_precision(y_true, y_conf, ACCURACY_GATE)
            cov_90       = coverage_at_precision(y_true, y_conf, 0.90)

            # Per-category breakdown
            per_cat: dict[str, dict] = {}
            for cat in CATEGORIES:
                mask   = cats_arr == cat
                if mask.sum() == 0:
                    continue
                y_c    = y_true[mask]
                conf_c = y_conf[mask]
                per_cat[cat] = {
                    "accuracy":    round(float(y_c.mean()), 4),
                    "coverage_85": round(coverage_at_precision(y_c, conf_c, ACCURACY_GATE), 4),
                    "n":           int(mask.sum()),
                }

            all_results[condition_key] = {
                "noise_rate":         noise_rate,
                "delta":              delta,
                "n_warmup":           n_warmup,
                "overall_accuracy":   round(overall_acc, 4),
                "coverage_85":        round(cov_85, 4),
                "coverage_90":        round(cov_90, 4),
                "mean_centroid_drift": round(mean_drift, 6),
                "mean_shift":         round(actual_shift, 4),
                "per_category":       per_cat,
            }

            print(f"  acc={overall_acc:.1%}  cov@85={cov_85:.1%}  "
                  f"cov@90={cov_90:.1%}  drift={mean_drift:.4f}")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

assert len(all_results) == len(N_WARMUP_LEVELS) * len(NOISE_RATES) * len(DELTA_LEVELS), (
    f"Expected {len(N_WARMUP_LEVELS)*len(NOISE_RATES)*len(DELTA_LEVELS)} conditions, "
    f"got {len(all_results)}"
)
# Frozen baseline (n_warmup=0) must have zero drift
for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:
        key = f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup0"
        drift = all_results[key]["mean_centroid_drift"]
        assert drift == 0.0, f"n_warmup=0 should have zero drift, got {drift} for {key}"
print("\n[CHECKS] All conditions present, frozen drift == 0.0 ✓")

# ---------------------------------------------------------------------------
# Summary Table 1: Accuracy lift
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== TABLE 1: LEARNING LIFT (accuracy: warmup=N vs warmup=0) ===")
print(f"{'='*70}")
print(f"{'noise':>6} | {'delta':>6} | {'w=0 (frozen)':>12} | "
      f"{'w=200':>7} | {'w=500':>7} | {'w=1000':>7}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:
        frozen_key = f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup0"
        frozen_acc = all_results[frozen_key]["overall_accuracy"]
        vals = [f"{frozen_acc:.1%}"]
        for w in [200, 500, 1000]:
            key   = f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup{w}"
            lift  = all_results[key]["overall_accuracy"] - frozen_acc
            vals.append(f"{lift:+.1%}")
        print(f"{noise_rate:6.2f} | {delta:6.2f} | {vals[0]:>12} | "
              f"{vals[1]:>7} | {vals[2]:>7} | {vals[3]:>7}")

# ---------------------------------------------------------------------------
# Summary Table 2: Coverage lift
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== TABLE 2: COVERAGE LIFT (coverage@85%: warmup=N vs warmup=0) ===")
print(f"{'='*70}")
print(f"{'noise':>6} | {'delta':>6} | {'w=0 (frozen)':>12} | "
      f"{'w=200':>7} | {'w=500':>7} | {'w=1000':>7}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:
        frozen_key = f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup0"
        frozen_cov = all_results[frozen_key]["coverage_85"]
        vals = [f"{frozen_cov:.1%}"]
        for w in [200, 500, 1000]:
            key  = f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup{w}"
            lift = all_results[key]["coverage_85"] - frozen_cov
            vals.append(f"{lift:+.1%}")
        print(f"{noise_rate:6.2f} | {delta:6.2f} | {vals[0]:>12} | "
              f"{vals[1]:>7} | {vals[2]:>7} | {vals[3]:>7}")

# ---------------------------------------------------------------------------
# Summary Table 3: Centroid drift
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== TABLE 3: CENTROID DRIFT FROM μ₀ ===")
print(f"{'='*70}")
print(f"{'noise':>6} | {'delta':>6} | {'w=0':>6} | "
      f"{'w=200':>7} | {'w=500':>7} | {'w=1000':>7}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:
        vals = []
        for w in N_WARMUP_LEVELS:
            key   = f"noise{noise_rate:.2f}_delta{delta:.2f}_warmup{w}"
            drift = all_results[key]["mean_centroid_drift"]
            vals.append(f"{drift:.4f}")
        print(f"{noise_rate:6.2f} | {delta:6.2f} | {vals[0]:>6} | "
              f"{vals[1]:>7} | {vals[2]:>7} | {vals[3]:>7}")

# ---------------------------------------------------------------------------
# Summary Table 4: Per-category accuracy at noise=0.10, delta=0.00
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== TABLE 4: PER-CATEGORY ACCURACY (noise=0.10, delta=0.00) ===")
print(f"{'='*70}")
print(f"{'category':>24} | {'w=0':>7} | {'w=200':>7} | {'w=500':>7} | {'w=1000':>7}")
print(f"{'-'*24}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
for cat in CATEGORIES:
    vals = []
    for w in N_WARMUP_LEVELS:
        key = f"noise0.10_delta0.00_warmup{w}"
        cat_acc = all_results[key]["per_category"][cat]["accuracy"]
        vals.append(f"{cat_acc:.1%}")
    print(f"{cat:>24} | {vals[0]:>7} | {vals[1]:>7} | {vals[2]:>7} | {vals[3]:>7}")

# ---------------------------------------------------------------------------
# Zero-noise diagnostic
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== INTERPRETATION ===")
print(f"{'='*70}")
print()
print("CRITICAL DIAGNOSTIC — noise=0.00 conditions:")
for delta in DELTA_LEVELS:
    frozen_key  = f"noise0.00_delta{delta:.2f}_warmup0"
    learned_key = f"noise0.00_delta{delta:.2f}_warmup1000"
    frozen_acc  = all_results[frozen_key]["overall_accuracy"]
    learned_acc = all_results[learned_key]["overall_accuracy"]
    lift        = learned_acc - frozen_acc
    print(f"  delta={delta:.2f}: frozen={frozen_acc:.1%}  "
          f"learned(w=1000)={learned_acc:.1%}  lift={lift:+.1%}")

print()
print("If lift is ~0 at noise=0.00, delta=0.00:")
print("  → Circularity confirmed (expected). Nothing to learn.")
print("If lift is POSITIVE at noise=0.00, delta>0:")
print("  → Learning works with clean labels. Noise tolerance is the problem.")
print("  → Fix: confidence-weighted update, batch update, or lower η at eval time.")
print("If lift is ~0 at noise=0.00, delta>0:")
print("  → Update rule cannot track shifted truth even with perfect labels.")
print("  → Architecture is the ceiling (not noise).")
print("If lift is NEGATIVE even at noise=0.00:")
print("  → Update rule is architecturally misaligned with classification.")
print("  → Not a noise problem. Deeper issue with correct/incorrect update direction.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

with open(RESULTS_PATH, "w") as fh:
    json.dump(all_results, fh, indent=2, default=str)

assert RESULTS_PATH.exists()
print(f"\nResults written to: {RESULTS_PATH.resolve()}")

# ---------------------------------------------------------------------------
# Launch charts
# ---------------------------------------------------------------------------

exec(open(str(_SCRIPT_DIR / "charts.py")).read())
