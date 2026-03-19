"""
VAL-3A-3: Data Efficiency Ratio

ProfileScorer (warm-start from DomainConfig, N_warmup unlabeled decisions)
vs ML baselines (XGBoost, RandomForest) trained on growing labeled sets.

Key question: how many labeled samples does an ML baseline need to match
a ProfileScorer that uses zero labels but has a compiled ontology?

Config: default.yaml bridge_common (C=5, A=4, d=6)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(REPO_ROOT / "configs" / "default.yaml") as _f:
    _CFG = yaml.safe_load(_f)["bridge_common"]

CATEGORIES = _CFG["categories"]
ACTIONS    = _CFG["actions"]
FACTORS    = _CFG["factors"]
PROFILES   = _CFG["action_conditional_profiles"]
GT_DISTS   = _CFG["category_gt_distributions"]

C_DIM = len(CATEGORIES)   # 5
A_DIM = len(ACTIONS)      # 4
D_DIM = len(FACTORS)      # 6

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TAU          = 0.1
ETA          = 0.05
ETA_NEG      = 0.05   # CANONICAL — never 1.0
N_SEEDS      = 10
N_WARMUP     = 200    # ProfileScorer warm-up decisions (no labels needed)
N_EVAL       = 500    # held-out evaluation set
SAMPLE_SIZES = [0, 25, 50, 100, 200, 400, 800, 1000, 1300, 1600, 2000]
N_MAX        = max(SAMPLE_SIZES)

RANDOM_BASELINE = 1.0 / (C_DIM * A_DIM)  # 1/20 = 5%

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Warm mu from profiles
# ---------------------------------------------------------------------------
def build_mu() -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, D_DIM), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = PROFILES[cat][act]
    return mu

MU_WARM = build_mu()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=== VAL-3A-3: Data Efficiency Ratio ===")
print(f"Config: default.yaml  C={C_DIM} A={A_DIM} d={D_DIM}")
print(f"N_SEEDS={N_SEEDS}  N_WARMUP={N_WARMUP}  N_EVAL={N_EVAL}  N_MAX={N_MAX}")
print(f"Random baseline: {RANDOM_BASELINE:.1%}")
print()

all_profile_acc: list[float] = []
all_xgb_acc:     list[list[float]] = []
all_rf_acc:      list[list[float]] = []

for seed in range(N_SEEDS):
    gen_train = CategoryAlertGenerator(
        categories=CATEGORIES, actions=ACTIONS, factors=FACTORS,
        action_conditional_profiles=PROFILES, gt_distributions=GT_DISTS,
        seed=seed,
    )
    gen_eval = CategoryAlertGenerator(
        categories=CATEGORIES, actions=ACTIONS, factors=FACTORS,
        action_conditional_profiles=PROFILES, gt_distributions=GT_DISTS,
        seed=seed + 5000,
    )

    # Fixed evaluation set
    eval_alerts = gen_eval.generate(N_EVAL)
    X_eval = np.array([a.factors for a in eval_alerts])
    # Label: flattened category × action index
    y_eval = np.array([a.category_index * A_DIM + a.gt_action_index for a in eval_alerts])

    # --- ProfileScorer: warm-start, N_WARMUP unlabeled decisions ---
    scorer = ProfileScorer(
        MU_WARM.copy(), A_DIM,
        tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed,
    )
    warmup_alerts = gen_train.generate(N_WARMUP)
    for alert in warmup_alerts:
        result     = scorer.score(alert.factors, alert.category_index)
        is_correct = result.action_index == alert.gt_action_index
        scorer.update(
            alert.factors, alert.category_index,
            result.action_index, is_correct,
            gt_action_index=alert.gt_action_index,
        )

    profile_correct = sum(
        int(scorer.score(a.factors, a.category_index).action_index == a.gt_action_index)
        for a in eval_alerts
    )
    profile_acc_seed = profile_correct / N_EVAL
    all_profile_acc.append(profile_acc_seed)

    # --- ML baselines: train on growing labeled sets ---
    # Pull N_MAX training samples (labels from gt_action_index, no oracle noise)
    train_alerts = gen_train.generate(N_MAX)
    X_train_full = np.array([a.factors for a in train_alerts])
    y_train_full = np.array([a.category_index * A_DIM + a.gt_action_index
                             for a in train_alerts])

    xgb_accs_seed: list[float] = []
    rf_accs_seed:  list[float] = []

    for N in SAMPLE_SIZES:
        if N == 0:
            xgb_accs_seed.append(RANDOM_BASELINE)
            rf_accs_seed.append(RANDOM_BASELINE)
            continue

        X_tr = X_train_full[:N].copy()
        y_tr = y_train_full[:N].copy()

        # XGBoost requires all classes present in training data.
        # For small N, add one profile-centroid pseudo-sample per missing class.
        all_classes = np.arange(C_DIM * A_DIM)
        missing_cls = np.setdiff1d(all_classes, np.unique(y_tr))
        if len(missing_cls) > 0:
            extra_X = np.array([MU_WARM[cls // A_DIM, cls % A_DIM, :]
                                 for cls in missing_cls])
            extra_y = missing_cls
            X_tr = np.vstack([X_tr, extra_X])
            y_tr = np.concatenate([y_tr, extra_y])

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="mlogloss",
            n_jobs=1, random_state=seed, verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_accs_seed.append(float((xgb_model.predict(X_eval) == y_eval).mean()))

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=seed, n_jobs=1,
        )
        rf_model.fit(X_tr, y_tr)
        rf_accs_seed.append(float((rf_model.predict(X_eval) == y_eval).mean()))

    all_xgb_acc.append(xgb_accs_seed)
    all_rf_acc.append(rf_accs_seed)

    xgb_1300 = xgb_accs_seed[SAMPLE_SIZES.index(1300)]
    rf_800   = rf_accs_seed[SAMPLE_SIZES.index(800)]
    print(f"  seed={seed}  profile={profile_acc_seed:.1%}  "
          f"xgb@1300={xgb_1300:.1%}  rf@800={rf_800:.1%}")

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
profile_arr = np.array(all_profile_acc)
xgb_arr     = np.array(all_xgb_acc)    # (N_seeds, len(SAMPLE_SIZES))
rf_arr      = np.array(all_rf_acc)

profile_mean = float(profile_arr.mean())
profile_std  = float(profile_arr.std())
xgb_mean     = xgb_arr.mean(axis=0)
xgb_std      = xgb_arr.std(axis=0)
rf_mean      = rf_arr.mean(axis=0)
rf_std       = rf_arr.std(axis=0)

# Crossover: first N where ML baseline reaches 90% of ProfileScorer
target_90 = 0.90 * profile_mean

def find_crossover(curve, target, sizes):
    for n, acc in zip(sizes, curve):
        if acc >= target:
            return n
    return ">2000"

xgb_crossover = find_crossover(xgb_mean, target_90, SAMPLE_SIZES)
rf_crossover  = find_crossover(rf_mean,  target_90, SAMPLE_SIZES)

# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
print()
print("=== VAL-3A-3 DATA EFFICIENCY RESULTS ===")
print(f"ProfileScorer ({N_WARMUP} warmup decisions, 0 labels): "
      f"{profile_mean:.1%} ± {profile_std:.1%}")
print(f"XGBoost at N=1300: {xgb_mean[SAMPLE_SIZES.index(1300)]:.1%}")
print(f"RF      at N=800:  {rf_mean[SAMPLE_SIZES.index(800)]:.1%}")
print(f"XGBoost reaches 90% of ProfileScorer at N={xgb_crossover} samples")
print(f"RF      reaches 90% of ProfileScorer at N={rf_crossover} samples")
print(f"Random baseline (1/C×A = 1/{C_DIM*A_DIM}): {RANDOM_BASELINE:.1%}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _to_json(obj):
    if isinstance(obj, np.ndarray):         return obj.tolist()
    if isinstance(obj, (np.floating,)):     return float(obj)
    if isinstance(obj, (np.integer,)):      return int(obj)
    if isinstance(obj, dict):               return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):               return [_to_json(v) for v in obj]
    return obj

payload = {
    "profile_mean":     profile_mean,
    "profile_std":      profile_std,
    "xgb_mean":         xgb_mean,
    "xgb_std":          xgb_std,
    "rf_mean":          rf_mean,
    "rf_std":           rf_std,
    "sample_sizes":     SAMPLE_SIZES,
    "xgb_crossover":    xgb_crossover,
    "rf_crossover":     rf_crossover,
    "random_baseline":  RANDOM_BASELINE,
    "C":                C_DIM,
    "A":                A_DIM,
    "d":                D_DIM,
    "n_warmup":         N_WARMUP,
    "n_seeds":          N_SEEDS,
}

np.save(str(OUT_DIR / "results.npy"), payload, allow_pickle=True)
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(_to_json(payload), f, indent=2)
print(f"Results saved → {OUT_DIR / 'results.npy'}")
print("Calling charts.py ...")

from experiments.val3a_data_efficiency.charts import make_charts
make_charts(payload)

print()
print("=== VAL-3A-3 COMPLETE ===")
