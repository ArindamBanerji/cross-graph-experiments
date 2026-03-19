"""
DISC-1 — Composite Discriminant on Frozen Scorer

PART A: Does a composite discriminant on frozen scorer outputs achieve
higher coverage at 85% precision than confidence alone?

PART B: IKS reverse engineering — what does a "good" IKS trajectory look
like as decisions accumulate (simulating graph enrichment)?

Regime: centroidal synthetic. Frozen scorer only — no centroid updates.
η_neg=0.05, τ=0.1, noise=0.10.

Note on distances: ScoringResult.distances contains raw squared L2 distances
‖f − μ[c,a,:]‖² per action. Used directly (no log-prob approximation needed).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
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
N_DECISIONS      = 1000
TAU              = 0.1
ETA              = 0.05       # not used (frozen), required for constructor
ETA_NEG          = 0.05       # not used (frozen), required for constructor
NOISE_RATE       = 0.10
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG    = "soc_product_v50"

ACCURACY_GATE    = 0.85
WINDOWS          = [50, 100, 200, 500, 1000]

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

mu_zero = config["mu"].copy()    # shape (C, A, d)

print(f"Domain config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Categories: {CATEGORIES}")
print(f"τ={TAU}, noise={NOISE_RATE}, seeds={N_SEEDS}, decisions/seed={N_DECISIONS}")
print(f"Frozen scorer (no centroid updates)")
print()

RESULTS_PATH = _SCRIPT_DIR / "disc1_results.json"

# ---------------------------------------------------------------------------
# PART A: GENERATE FEATURES FROM FROZEN SCORER
# ---------------------------------------------------------------------------

all_records: list[dict] = []

for seed in range(N_SEEDS):
    if seed % 10 == 0:
        print(f"Seed {seed+1}/{N_SEEDS}", flush=True)

    gen = CategoryAlertGenerator(
        **config["generator_kwargs"],
        noise_rate=NOISE_RATE,
        seed=RANDOM_SEED_BASE + seed,
    )

    # FROZEN scorer — initialized at μ₀, NEVER updated
    scorer = ProfileScorer(
        mu_zero.copy(),
        config["actions"],
        tau=TAU,
        eta=ETA,
        eta_neg=ETA_NEG,
    )

    alerts = gen.generate(N_DECISIONS)

    # Running per-category counts (simulates accumulating graph context)
    cat_counts:         dict[str, int] = {c: 0 for c in CATEGORIES}
    cat_correct_counts: dict[str, int] = {c: 0 for c in CATEGORIES}

    for i, alert in enumerate(alerts):
        result     = scorer.score(alert.factors, alert.category_index)
        is_correct = result.action_index == alert.gt_action_index

        probs       = result.probabilities                # shape (A,)
        dists       = result.distances                    # shape (A,) — raw ‖f−μ‖²

        sorted_probs = np.sort(probs)[::-1]              # descending
        sorted_dists = np.sort(dists)                    # ascending (smallest = best)

        cat = alert.category
        cat_counts[cat] += 1
        if is_correct:
            cat_correct_counts[cat] += 1

        # Probability-space features
        confidence  = float(sorted_probs[0])
        margin      = float(sorted_probs[0] - sorted_probs[1])
        entropy     = float(-np.sum(probs * np.log(probs + 1e-10)))
        top3_mass   = float(sorted_probs[0] + sorted_probs[1] + sorted_probs[2])
        prob_std    = float(np.std(probs))

        # Distance-space features (real squared L2, not log-prob approximation)
        d_top1      = float(sorted_dists[0])              # smallest = best action
        d_top2      = float(sorted_dists[1])
        dist_ratio  = d_top1 / (d_top2 + 1e-10)          # near 0 = sharp minimum
        dist_gap    = d_top2 - d_top1                     # larger = cleaner separation

        # Factor-space features
        f                  = alert.factors
        factor_extremity   = float(np.max(f) - np.min(f))
        factor_norm        = float(np.linalg.norm(f))
        factor_center_dist = float(np.linalg.norm(f - 0.5))

        # Context features (simulating graph enrichment over time)
        n_cat              = cat_counts[cat]
        rolling_accuracy   = (cat_correct_counts[cat] / n_cat) if n_cat > 0 else 0.5
        decision_position  = i / N_DECISIONS    # 0.0 → 1.0

        all_records.append({
            "seed":               seed,
            "decision_idx":       i,
            "category":           cat,
            "category_idx":       alert.category_index,
            "is_correct":         bool(is_correct),
            # Scorer output features
            "confidence":         confidence,
            "margin":             margin,
            "entropy":            entropy,
            "top3_mass":          top3_mass,
            "prob_std":           prob_std,
            "dist_ratio":         dist_ratio,
            "dist_gap":           dist_gap,
            # Factor-space features
            "factor_extremity":   factor_extremity,
            "factor_norm":        factor_norm,
            "factor_center_dist": factor_center_dist,
            # Context features
            "cat_count":          n_cat,
            "rolling_accuracy":   rolling_accuracy,
            "decision_position":  decision_position,
        })

# Verify no scorer.update() was called (scorer is frozen)
assert np.allclose(scorer.mu, mu_zero, atol=1e-10), "Scorer was NOT frozen — mu changed!"

print(f"\nTotal records: {len(all_records)}")
assert len(all_records) == N_SEEDS * N_DECISIONS, (
    f"Expected {N_SEEDS * N_DECISIONS}, got {len(all_records)}"
)
print(f"Overall accuracy: {np.mean([r['is_correct'] for r in all_records]):.1%}")
print("[CHECKS] Record count and frozen scorer verified")

# ---------------------------------------------------------------------------
# PART A: FIT COMPOSITE DISCRIMINANT — 5 models
# ---------------------------------------------------------------------------

# Feature ordering must match model index slices below
FEATURE_NAMES = [
    "confidence",         # 0  — probability-space: top1
    "margin",             # 1  — probability-space: top1 − top2
    "entropy",            # 2  — probability-space: distribution spread
    "top3_mass",          # 3  — probability-space: top3 cumulative mass
    "prob_std",           # 4  — probability-space: std of probs
    "dist_ratio",         # 5  — distance-space: d_top1 / d_top2
    "dist_gap",           # 6  — distance-space: d_top2 − d_top1
    "factor_extremity",   # 7  — factor-space: range of factor vector
    "factor_norm",        # 8  — factor-space: ‖f‖
    "factor_center_dist", # 9  — factor-space: distance from 0.5
    "cat_count",          # 10 — context: decisions accumulated for this category
    "rolling_accuracy",   # 11 — context: running accuracy for this category
    "decision_position",  # 12 — context: position in session (proxy for graph age)
]

X          = np.array([[r[f] for f in FEATURE_NAMES] for r in all_records])
y          = np.array([r["is_correct"] for r in all_records]).astype(int)
categories = np.array([r["category"] for r in all_records])

print(f"\nFeature matrix: {X.shape}")
print(f"Positive rate (overall accuracy): {y.mean():.1%}")

# Model definitions by feature index slice
MODELS: dict[str, list[int]] = {
    "A_confidence_only":    [0],
    "B_confidence_margin":  [0, 1],
    "C_scorer_features":    [0, 1, 2, 3, 4, 5, 6],
    "D_scorer_factor":      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "E_all_features":       list(range(len(FEATURE_NAMES))),
}

model_results: dict[str, dict] = {}

print()
for model_name, feat_indices in MODELS.items():
    X_model        = X[:, feat_indices]
    feat_names_m   = [FEATURE_NAMES[i] for i in feat_indices]
    print(f"--- Model {model_name} ({X_model.shape[1]} features) ---")

    lr = LogisticRegression(max_iter=1000, C=1.0, penalty="l2")

    # 5-fold cross-validated probability predictions
    y_prob = cross_val_predict(lr, X_model, y, cv=5, method="predict_proba")[:, 1]

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y, y_prob)

    # Coverage at 85% precision — highest recall where precision >= gate
    coverage_at_85 = 0.0
    threshold_at_85 = None
    for p, r, t in zip(prec_arr, rec_arr, thr_arr):
        if p >= 0.85 and r > coverage_at_85:
            coverage_at_85 = r
            threshold_at_85 = float(t)

    # Coverage at 90% precision
    coverage_at_90 = 0.0
    for p, r, t in zip(prec_arr, rec_arr, thr_arr):
        if p >= 0.90 and r > coverage_at_90:
            coverage_at_90 = r

    auc = roc_auc_score(y, y_prob)

    # Refit on full dataset for coefficients
    lr_final = LogisticRegression(max_iter=1000, C=1.0, penalty="l2")
    lr_final.fit(X_model, y)
    coefficients = dict(zip(feat_names_m, lr_final.coef_[0].tolist()))

    model_results[model_name] = {
        "n_features":              X_model.shape[1],
        "features":                feat_names_m,
        "auc":                     round(auc, 4),
        "coverage_at_85_precision": round(coverage_at_85, 4),
        "coverage_at_90_precision": round(coverage_at_90, 4),
        "threshold_at_85":         round(threshold_at_85, 4) if threshold_at_85 else None,
        "coefficients":            {k: round(v, 4) for k, v in coefficients.items()},
    }

    print(f"  AUC:                  {auc:.4f}")
    print(f"  Coverage at 85% prec: {coverage_at_85:.1%}")
    print(f"  Coverage at 90% prec: {coverage_at_90:.1%}")
    print(f"  Coefs: {coefficients}")
    print()

# ---------------------------------------------------------------------------
# Per-category breakdown for Model E
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("=== PER-CATEGORY COVERAGE (Model E, all features) ===")
print(f"{'='*60}")

X_all  = X[:, MODELS["E_all_features"]]
lr_cat = LogisticRegression(max_iter=1000, C=1.0, penalty="l2")
y_prob_full = cross_val_predict(lr_cat, X_all, y, cv=5, method="predict_proba")[:, 1]

per_cat_results: dict[str, dict] = {}
for cat in CATEGORIES:
    mask   = categories == cat
    y_cat  = y[mask]
    p_cat  = y_prob_full[mask]

    prec_c, rec_c, _ = precision_recall_curve(y_cat, p_cat)
    cov_85 = 0.0
    for p, r in zip(prec_c, rec_c):
        if p >= 0.85 and r > cov_85:
            cov_85 = r

    per_cat_results[cat] = {
        "base_accuracy":            round(float(y_cat.mean()), 4),
        "composite_coverage_85":    round(cov_85, 4),
        "n_decisions":              int(mask.sum()),
    }
    print(f"  {cat:24s}: base_acc={y_cat.mean():.1%}  "
          f"composite_cov_85={cov_85:.1%}  n={mask.sum()}")

# ---------------------------------------------------------------------------
# Feature correlations with is_correct
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("=== FEATURE CORRELATION WITH is_correct ===")
print(f"{'='*60}")
feat_corrs_with_y: dict[str, float] = {}
for i, fname in enumerate(FEATURE_NAMES):
    corr = float(np.corrcoef(X[:, i], y)[0, 1])
    feat_corrs_with_y[fname] = round(corr, 4)
    print(f"  {fname:24s}: r = {corr:+.4f}")

# Pairwise key correlations
print(f"\n{'='*60}")
print("=== PAIRWISE FEATURE CORRELATIONS (key pairs) ===")
print(f"{'='*60}")
key_pairs = [
    ("confidence", "margin"),
    ("confidence", "entropy"),
    ("confidence", "dist_ratio"),
    ("confidence", "dist_gap"),
    ("confidence", "cat_count"),
    ("confidence", "rolling_accuracy"),
    ("margin", "entropy"),
    ("margin", "cat_count"),
    ("margin", "dist_gap"),
    ("factor_extremity", "confidence"),
    ("rolling_accuracy", "cat_count"),
    ("dist_gap", "dist_ratio"),
]
pairwise_corrs: dict[str, float] = {}
for f1, f2 in key_pairs:
    i1 = FEATURE_NAMES.index(f1)
    i2 = FEATURE_NAMES.index(f2)
    corr = float(np.corrcoef(X[:, i1], X[:, i2])[0, 1])
    key = f"{f1}×{f2}"
    pairwise_corrs[key] = round(corr, 4)
    print(f"  {f1:20s} × {f2:20s}: r = {corr:+.4f}")

# Full pairwise correlation matrix (for heatmap)
corr_matrix = np.corrcoef(X.T)   # (13, 13)

# ---------------------------------------------------------------------------
# PART B: IKS REVERSE ENGINEERING
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("=== PART B: IKS AS DISCRIMINANT INPUT QUALITY ===")
print(f"{'='*60}")

# Train Model E on the full dataset for the IKS analysis
lr_iks = LogisticRegression(max_iter=1000, C=1.0, penalty="l2")
lr_iks.fit(X_all, y)

window_results: dict[int, list[dict]] = {w: [] for w in WINDOWS}

for seed in range(N_SEEDS):
    seed_mask = np.array([r["seed"] == seed for r in all_records])
    seed_X    = X_all[seed_mask]
    seed_y    = y[seed_mask]

    for w in WINDOWS:
        if w > seed_X.shape[0]:
            continue

        X_w = seed_X[:w]
        y_w = seed_y[:w]

        y_prob_w = lr_iks.predict_proba(X_w)[:, 1]

        prec_w, rec_w, _ = precision_recall_curve(y_w, y_prob_w)
        cov_85 = 0.0
        for p, r in zip(prec_w, rec_w):
            if p >= 0.85 and r > cov_85:
                cov_85 = r

        mean_conf    = float(X_w[:, 0].mean())
        cc_idx       = FEATURE_NAMES.index("cat_count")
        roll_idx     = FEATURE_NAMES.index("rolling_accuracy")
        mean_cat_cnt = float(X_w[:, cc_idx].mean())
        mean_rolling = float(X_w[:, roll_idx].mean())
        window_acc   = float(y_w.mean())

        window_results[w].append({
            "seed":                 seed,
            "coverage_at_85":       cov_85,
            "accuracy":             window_acc,
            "mean_confidence":      mean_conf,
            "mean_cat_count":       mean_cat_cnt,
            "mean_rolling_accuracy": mean_rolling,
        })

print(f"\n{'Decisions':>10} | {'Coverage@85%':>12} | {'Accuracy':>8} | "
      f"{'Mean CatCount':>13} | {'Mean Rolling':>12}")
print(f"{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*13}-+-{'-'*12}")
for w in WINDOWS:
    if not window_results[w]:
        continue
    wr = window_results[w]
    m_cov  = float(np.mean([r["coverage_at_85"]        for r in wr]))
    m_acc  = float(np.mean([r["accuracy"]               for r in wr]))
    m_cc   = float(np.mean([r["mean_cat_count"]         for r in wr]))
    m_roll = float(np.mean([r["mean_rolling_accuracy"]  for r in wr]))
    print(f"{w:>10} | {m_cov:>11.1%} | {m_acc:>7.1%} | {m_cc:>13.1f} | {m_roll:>11.1%}")

# IKS v2 trajectory
print(f"\n{'='*60}")
print("=== PROPOSED IKS v2 TRAJECTORY ===")
print(f"{'='*60}")
print("IKS v2 = equal-weight composite (each 0-100):")
print("  graph_richness    = min(decisions/1000, 1) × 100")
print("  decision_maturity = min(mean_cat_count/100, 1) × 100")
print("  trust_coverage    = coverage@85% × 100")
print("  factor_quality    = mean_rolling_accuracy × 100")
print()

iks_trajectory: dict[int, float] = {}
iks_components: dict[int, dict]  = {}
for w in WINDOWS:
    if not window_results[w]:
        continue
    wr     = window_results[w]
    m_cov  = float(np.mean([r["coverage_at_85"]       for r in wr]))
    m_cc   = float(np.mean([r["mean_cat_count"]        for r in wr]))
    m_roll = float(np.mean([r["mean_rolling_accuracy"] for r in wr]))

    graph_richness    = min(w / 1000.0, 1.0) * 100.0
    decision_maturity = min(m_cc / 100.0,  1.0) * 100.0
    trust_coverage    = m_cov  * 100.0
    factor_quality    = m_roll * 100.0

    iks_v2 = 0.25 * (graph_richness + decision_maturity + trust_coverage + factor_quality)
    iks_trajectory[w]  = round(iks_v2, 2)
    iks_components[w]  = {
        "graph_richness":    round(graph_richness, 2),
        "decision_maturity": round(decision_maturity, 2),
        "trust_coverage":    round(trust_coverage, 2),
        "factor_quality":    round(factor_quality, 2),
    }

    print(f"  Decisions={w:>5}:  IKS_v2={iks_v2:5.1f}  "
          f"[graph={graph_richness:5.1f}  maturity={decision_maturity:5.1f}  "
          f"trust={trust_coverage:5.1f}  quality={factor_quality:5.1f}]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("=== COMPOSITE DISCRIMINANT SUMMARY ===")
print(f"{'='*60}")
print()
print("Model comparison (coverage at 85% precision):")
for name, r in model_results.items():
    print(f"  {name:30s}: {r['coverage_at_85_precision']:6.1%}  "
          f"(AUC={r['auc']:.4f}, {r['n_features']} features)")

print()
baseline_cov = model_results["A_confidence_only"]["coverage_at_85_precision"]
best_cov     = max(r["coverage_at_85_precision"] for r in model_results.values())
best_name    = next(k for k, r in model_results.items()
                    if r["coverage_at_85_precision"] == best_cov)

print(f"Baseline (confidence only): {baseline_cov:.1%}")
print(f"Best composite ({best_name}): {best_cov:.1%}")
if best_cov > baseline_cov:
    lift     = best_cov - baseline_cov
    lift_pct = lift / (baseline_cov + 1e-10) * 100.0
    print(f"LIFT: +{lift:.1%} absolute ({lift_pct:.0f}% relative)")
    print(f"VERDICT: Composite discriminant IMPROVES coverage.")
else:
    print(f"VERDICT: Composite discriminant does NOT improve coverage.")
    print(f"  → Single-feature gating may be sufficient.")

print()
print("Key diagnostic — features orthogonal to confidence:")
conf_col = X[:, 0]
for i, fname in enumerate(FEATURE_NAMES):
    if fname == "confidence":
        continue
    corr_y    = float(np.corrcoef(X[:, i], y)[0, 1])
    corr_conf = float(np.corrcoef(X[:, i], conf_col)[0, 1])
    if abs(corr_y) > 0.05 and abs(corr_conf) < 0.70:
        print(f"  {fname:24s}: corr(y)={corr_y:+.3f}  "
              f"corr(conf)={corr_conf:+.3f}  ← ORTHOGONAL SIGNAL")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "experiment":    "DISC-1",
    "regime":        "frozen_centroidal_synthetic",
    "domain_config": DOMAIN_CONFIG,
    "n_seeds":       N_SEEDS,
    "n_decisions":   N_DECISIONS,
    "tau":           TAU,
    "noise_rate":    NOISE_RATE,
    "feature_names": FEATURE_NAMES,
    "model_results": model_results,
    "per_category_model_e": per_cat_results,
    "window_coverage": {
        str(w): {
            "mean_coverage_85": round(float(np.mean([r["coverage_at_85"]
                                 for r in window_results[w]])), 4),
            "mean_accuracy":    round(float(np.mean([r["accuracy"]
                                 for r in window_results[w]])), 4),
            "iks_v2":           iks_trajectory.get(w),
            "iks_components":   iks_components.get(w),
        }
        for w in WINDOWS if window_results[w]
    },
    "feature_correlations_with_correct": feat_corrs_with_y,
    "pairwise_correlations": pairwise_corrs,
    "full_correlation_matrix": corr_matrix.tolist(),   # (13,13) for heatmap
}

with open(RESULTS_PATH, "w") as fh:
    json.dump(output, fh, indent=2)

assert RESULTS_PATH.exists()
print(f"\nResults written to: {RESULTS_PATH.resolve()}")

# ---------------------------------------------------------------------------
# Launch charts
# ---------------------------------------------------------------------------

exec(open(str(_SCRIPT_DIR / "charts.py")).read())
