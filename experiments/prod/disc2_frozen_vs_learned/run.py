"""
DISC-2 — Frozen vs Learned Composite Discriminant (Post-Fix)

Closes the gap between DISC-1 (composite coverage, frozen scorer only) and
SHIFT-2 (raw accuracy lift from learning, confidence-only coverage).

Question: does enabling centroid learning (with corrected gt_action_index
update rule) improve composite discriminant coverage at 85% precision?

Two conditions per (noise, delta):
  FROZEN  — scorer stays at mu_zero throughout (no warmup, no updates)
  LEARNED — N_WARMUP decisions with learning, then FREEZE before eval

Both conditions evaluated through the same 13-feature composite discriminant.

Regime: centroidal synthetic.  N_SEEDS=50, N_WARMUP=500, N_EVAL=1000.
Ontology: soc_product_v50 (C=6, A=5, d=6).
Shift direction: same seed=12345 as SHIFT-1/2 for comparability.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

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

N_SEEDS          = 50
N_WARMUP         = 500       # post-fix warmup to show learning benefit
N_EVAL           = 1000
TAU              = 0.1
ETA              = 0.05
ETA_NEG          = 0.05
NOISE_RATES      = [0.0, 0.05, 0.10]
DELTA_LEVELS     = [0.0, 0.10, 0.20]
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG    = "soc_product_v50"
ACCURACY_GATE    = 0.85

CONDITIONS       = ["frozen", "learned"]

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

mu_zero = config["mu"].copy()    # shape (C, A, d)

print(f"Domain config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"NOISE_RATES:   {NOISE_RATES}")
print(f"DELTA_LEVELS:  {DELTA_LEVELS}")
print(f"N_WARMUP={N_WARMUP}, N_EVAL={N_EVAL}, N_SEEDS={N_SEEDS}")
print(f"Conditions: {CONDITIONS}")
print(f"Total conditions: {len(CONDITIONS)*len(NOISE_RATES)*len(DELTA_LEVELS)}")
print(f"Total runs: {len(CONDITIONS)*len(NOISE_RATES)*len(DELTA_LEVELS)*N_SEEDS}")
print()

RESULTS_PATH = _SCRIPT_DIR / "disc2_results.json"

# ---------------------------------------------------------------------------
# Fixed shift direction (same seed as SHIFT-1/2 for comparability)
# ---------------------------------------------------------------------------

rng_direction = np.random.RandomState(12345)
raw_direction = rng_direction.randn(C, A, d)
norms = np.linalg.norm(raw_direction, axis=-1, keepdims=True)
norms[norms == 0] = 1.0
unit_direction = raw_direction / norms    # shape (C, A, d), unit norm per (c,a) cell

print(f"Shift direction: seed=12345, unit-normalised per cell (matches SHIFT-1/2)")
print()

# ---------------------------------------------------------------------------
# Feature names (identical to DISC-1)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "confidence",          # 0  — probability-space: top1
    "margin",              # 1  — probability-space: top1 - top2
    "entropy",             # 2  — probability-space: distribution spread
    "top3_mass",           # 3  — probability-space: top3 cumulative mass
    "prob_std",            # 4  — probability-space: std of probs
    "dist_ratio",          # 5  — distance-space: d_top1 / d_top2
    "dist_gap",            # 6  — distance-space: d_top2 - d_top1
    "factor_extremity",    # 7  — factor-space: range of factor vector
    "factor_norm",         # 8  — factor-space: ||f||
    "factor_center_dist",  # 9  — factor-space: distance from 0.5
    "cat_count",           # 10 — context: decisions for this category
    "rolling_accuracy",    # 11 — context: running accuracy for this category
    "decision_position",   # 12 — context: position in eval window
]

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

all_results: dict[str, dict] = {}

for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:

        # Shifted truth: mu_true = clip(mu_zero + delta * direction, 0, 1)
        mu_true = np.clip(mu_zero + delta * unit_direction, 0.0, 1.0)

        # Build shifted generator config (constructor kwarg injection, option b)
        shifted_profiles: dict[str, dict[str, list[float]]] = {
            cat: {
                act: mu_true[ci, ai, :].tolist()
                for ai, act in enumerate(ACTIONS)
            }
            for ci, cat in enumerate(CATEGORIES)
        }
        gen_config = dict(config["generator_kwargs"])
        gen_config["action_conditional_profiles"] = shifted_profiles

        for condition in CONDITIONS:
            key = f"noise{noise_rate:.2f}_delta{delta:.2f}_{condition}"
            print(f"\n--- {key} ---", flush=True)

            all_records: list[dict] = []
            seed_drifts: list[float] = []

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

                # ===== WARMUP PHASE =====
                if condition == "learned":
                    warmup_alerts = gen.generate(N_WARMUP)
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
                else:
                    # Frozen: advance generator RNG to same state, don't learn
                    _ = gen.generate(N_WARMUP)

                # ===== FREEZE: measure drift, NO MORE UPDATES =====
                drift = float(np.linalg.norm(scorer.mu - mu_zero, axis=-1).mean())
                seed_drifts.append(drift)

                # ===== EVAL PHASE — frozen scorer, no updates, record features =====
                cat_counts  = {c: 0 for c in CATEGORIES}
                cat_correct = {c: 0 for c in CATEGORIES}

                eval_alerts = gen.generate(N_EVAL)
                for i, alert in enumerate(eval_alerts):
                    result     = scorer.score(alert.factors, alert.category_index)
                    is_correct = result.action_index == alert.gt_action_index

                    probs        = result.probabilities           # shape (A,)
                    dists        = result.distances               # shape (A,) — raw ||f-mu||^2
                    sorted_probs = np.sort(probs)[::-1]           # descending
                    sorted_dists = np.sort(dists)                 # ascending (best = smallest)

                    cat = alert.category
                    cat_counts[cat] += 1
                    if is_correct:
                        cat_correct[cat] += 1

                    rolling_acc = (
                        cat_correct[cat] / cat_counts[cat]
                        if cat_counts[cat] > 0 else 0.5
                    )

                    all_records.append({
                        "seed":               seed,
                        "category":           cat,
                        "is_correct":         bool(is_correct),
                        "confidence":         float(sorted_probs[0]),
                        "margin":             float(sorted_probs[0] - sorted_probs[1]),
                        "entropy":            float(-np.sum(probs * np.log(probs + 1e-10))),
                        "top3_mass":          float(sorted_probs[0] + sorted_probs[1] + sorted_probs[2]),
                        "prob_std":           float(np.std(probs)),
                        "dist_ratio":         float(sorted_dists[0] / (sorted_dists[1] + 1e-10)),
                        "dist_gap":           float(sorted_dists[1] - sorted_dists[0]),
                        "factor_extremity":   float(np.max(alert.factors) - np.min(alert.factors)),
                        "factor_norm":        float(np.linalg.norm(alert.factors)),
                        "factor_center_dist": float(np.linalg.norm(alert.factors - 0.5)),
                        "cat_count":          cat_counts[cat],
                        "rolling_accuracy":   rolling_acc,
                        "decision_position":  i / N_EVAL,
                    })

            # ===== COMPUTE METRICS =====
            assert len(all_records) == N_SEEDS * N_EVAL, (
                f"Expected {N_SEEDS * N_EVAL} records, got {len(all_records)}"
            )

            X      = np.array([[r[f] for f in FEATURE_NAMES] for r in all_records])
            y      = np.array([r["is_correct"] for r in all_records]).astype(int)
            y_conf = X[:, 0]    # confidence column (feature index 0)

            overall_acc = float(y.mean())
            mean_drift  = float(np.mean(seed_drifts))

            # Confidence-only coverage at 85%
            prec_c, rec_c, _ = precision_recall_curve(y, y_conf)
            conf_cov_85 = 0.0
            for p, r in zip(prec_c, rec_c):
                if p >= ACCURACY_GATE and r > conf_cov_85:
                    conf_cov_85 = r

            # Composite discriminant coverage at 85% (5-fold CV)
            lr_model = LogisticRegression(max_iter=2000, C=1.0, penalty="l2")
            y_disc   = cross_val_predict(lr_model, X, y, cv=5, method="predict_proba")[:, 1]
            prec_d, rec_d, _ = precision_recall_curve(y, y_disc)
            disc_cov_85 = 0.0
            for p, r in zip(prec_d, rec_d):
                if p >= ACCURACY_GATE and r > disc_cov_85:
                    disc_cov_85 = r

            all_results[key] = {
                "condition":              condition,
                "noise_rate":             noise_rate,
                "delta":                  delta,
                "overall_accuracy":       round(overall_acc, 4),
                "centroid_drift":         round(mean_drift, 6),
                "confidence_coverage_85": round(conf_cov_85, 4),
                "composite_coverage_85":  round(disc_cov_85, 4),
                "composite_lift":         round(disc_cov_85 - conf_cov_85, 4),
            }

            print(f"  acc={overall_acc:.1%}  conf_cov={conf_cov_85:.1%}  "
                  f"disc_cov={disc_cov_85:.1%}  "
                  f"disc_lift={disc_cov_85-conf_cov_85:+.1%}  "
                  f"drift={mean_drift:.4f}")

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

expected_n = len(CONDITIONS) * len(NOISE_RATES) * len(DELTA_LEVELS)
assert len(all_results) == expected_n, (
    f"Expected {expected_n} conditions, got {len(all_results)}"
)
# Frozen condition must have zero drift
for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:
        fk    = f"noise{noise_rate:.2f}_delta{delta:.2f}_frozen"
        drift = all_results[fk]["centroid_drift"]
        assert drift == 0.0, f"Frozen condition should have zero drift, got {drift} for {fk}"
print("\n[CHECKS] All conditions present, frozen drift == 0.0 for all ✓")

# ---------------------------------------------------------------------------
# Deployment decision table
# ---------------------------------------------------------------------------

print(f"\n{'='*80}")
print("=== THE DEPLOYMENT QUESTION: DOES LEARNING IMPROVE COMPOSITE COVERAGE? ===")
print(f"{'='*80}")
header = (
    f"{'noise':>6} | {'delta':>5} | "
    f"{'frz conf':>8} | {'frz disc':>8} | "
    f"{'lrn conf':>8} | {'lrn disc':>8} | "
    f"{'disc lift':>9}"
)
sep = (
    f"{'-'*6}-+-{'-'*5}-+-"
    f"{'-'*8}-+-{'-'*8}-+-"
    f"{'-'*8}-+-{'-'*8}-+-"
    f"{'-'*9}"
)
print(header)
print(sep)

for noise_rate in NOISE_RATES:
    for delta in DELTA_LEVELS:
        fk = f"noise{noise_rate:.2f}_delta{delta:.2f}_frozen"
        lk = f"noise{noise_rate:.2f}_delta{delta:.2f}_learned"
        fr = all_results[fk]
        lr = all_results[lk]
        disc_lift = lr["composite_coverage_85"] - fr["composite_coverage_85"]
        print(
            f"{noise_rate:6.2f} | {delta:5.2f} | "
            f"{fr['confidence_coverage_85']:7.1%} | {fr['composite_coverage_85']:7.1%} | "
            f"{lr['confidence_coverage_85']:7.1%} | {lr['composite_coverage_85']:7.1%} | "
            f"{disc_lift:+8.1%}"
        )

# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

print(f"\n=== DECISION RULE ===")
print("If learned_disc > frozen_disc at noise=0.10, delta>=0.10:")
print("  -> Learning IMPROVES composite coverage in realistic conditions.")
print("  -> Ship with LEARNING_ENABLED=True (after shadow validation).")
print("If learned_disc <= frozen_disc at noise=0.10:")
print("  -> Learning does NOT help the composite discriminant.")
print("  -> Ship FROZEN. Composite gate compounds via rolling_accuracy alone.")
print()

# Evaluate the rule
key_noises  = [0.10]
key_deltas  = [0.10, 0.20]
any_improve = False
for noise_rate in key_noises:
    for delta in key_deltas:
        fk        = f"noise{noise_rate:.2f}_delta{delta:.2f}_frozen"
        lk        = f"noise{noise_rate:.2f}_delta{delta:.2f}_learned"
        lift      = all_results[lk]["composite_coverage_85"] - all_results[fk]["composite_coverage_85"]
        direction = "IMPROVES" if lift > 0.0 else "DOES NOT IMPROVE"
        print(f"  noise={noise_rate:.2f} delta={delta:.2f}: disc_lift={lift:+.1%}  -> {direction}")
        if lift > 0.0:
            any_improve = True

print()
if any_improve:
    print("VERDICT: LEARNING IMPROVES composite discriminant coverage (at least one key condition).")
    print("         -> LEARNING_ENABLED=True is appropriate with shadow validation.")
else:
    print("VERDICT: Learning does NOT improve composite coverage under realistic conditions.")
    print("         -> Ship FROZEN. Focus on composite features over learning.")

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
