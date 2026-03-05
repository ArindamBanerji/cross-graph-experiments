"""
VALIDATION-3: Baseline Comparison (Task A).

Compares L2 centroid scorer against standard ML baselines on the SAME
synthetic SOC alert data used in the bridge experiments.

Static  (1000 train / 500 test):
  Methods: l2_centroid, l2_centroid_online, logistic_regression,
           xgboost (GradientBoosting), random_forest, knn, random

Online  (200 train / 1300 test):
  L2 centroid: update after every alert (pull/push + clip)
  ML baselines: periodic retrain every 100 test alerts (add new data)
  Checkpoints (total seen = train + test): [400, 600, 800, 1000, 1200, 1500]

Features for ML baselines: 6 factors + 5-dim one-hot category = 11 dims.
Target: gt_action_index (the oracle's answer, includes 3% noise).
"""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.category_alert_generator import CategoryAlertGenerator, CATEGORIES, ACTIONS
from src.models.profile_scorer import ProfileScorer

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS           = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
N_TRAIN_STATIC  = 1000
N_TEST_STATIC   = 500
N_TRAIN_ONLINE  = 200          # limited-data scenario
N_TOTAL_ONLINE  = 1500         # 200 train + 1300 test
NOISE_RATE      = 0.03
TAU             = 0.25
ETA             = 0.05
ETA_NEG         = 0.025        # half of eta (from profile_scorer default)

N_CATEGORIES = len(CATEGORIES)  # 5
N_ACTIONS    = len(ACTIONS)     # 4
N_FACTORS    = 6

# Checkpoints = total alerts seen (train + test); starts after N_TRAIN_ONLINE
ONLINE_CHECKPOINTS = [400, 600, 800, 1000, 1200, 1500]


# ---------------------------------------------------------------------------
# Config / data
# ---------------------------------------------------------------------------
def _load_config():
    with open(ROOT / "configs" / "default.yaml") as fh:
        cfg = yaml.safe_load(fh)
    return cfg["bridge_common"], cfg["realistic_profiles"]


def _generate(seed: int, n: int, bc: dict, rp: dict):
    gen = CategoryAlertGenerator(
        categories=bc["categories"],
        actions=bc["actions"],
        factors=bc["factors"],
        action_conditional_profiles=rp["action_conditional_profiles"],
        gt_distributions=rp["category_gt_distributions"],
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=NOISE_RATE,
        seed=seed,
    )
    return gen.generate(n)


# ---------------------------------------------------------------------------
# Feature building for sklearn (11-dim)
# ---------------------------------------------------------------------------
def _features(alerts) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(alerts), N_FACTORS + N_CATEGORIES), dtype=np.float64)
    y = np.zeros(len(alerts), dtype=np.int64)
    for i, a in enumerate(alerts):
        X[i, :N_FACTORS] = a.factors
        X[i, N_FACTORS + a.category_index] = 1.0
        y[i] = a.gt_action_index
    return X, y


def _feat_single(alert) -> np.ndarray:
    x = np.zeros(N_FACTORS + N_CATEGORIES, dtype=np.float64)
    x[:N_FACTORS] = alert.factors
    x[N_FACTORS + alert.category_index] = 1.0
    return x.reshape(1, -1)


# ---------------------------------------------------------------------------
# L2 centroid helpers
# ---------------------------------------------------------------------------
def _build_mu_from_data(train_alerts) -> np.ndarray:
    """Compute mu[c,a,:] = mean factor vector for training alerts in (c,a)."""
    sums   = np.zeros((N_CATEGORIES, N_ACTIONS, N_FACTORS), dtype=np.float64)
    counts = np.zeros((N_CATEGORIES, N_ACTIONS), dtype=np.int64)
    for a in train_alerts:
        c, act = a.category_index, a.gt_action_index
        sums[c, act]  += a.factors
        counts[c, act] += 1
    mu = np.full((N_CATEGORIES, N_ACTIONS, N_FACTORS), 0.5, dtype=np.float64)
    for c in range(N_CATEGORIES):
        for act in range(N_ACTIONS):
            if counts[c, act] > 0:
                mu[c, act] = sums[c, act] / counts[c, act]
    return mu


def _l2_predict_batch(mu: np.ndarray, alerts) -> np.ndarray:
    preds = np.empty(len(alerts), dtype=np.int64)
    for i, a in enumerate(alerts):
        dists = np.sum((mu[a.category_index] - a.factors) ** 2, axis=1)
        preds[i] = int(np.argmin(dists))
    return preds


# ---------------------------------------------------------------------------
# Static comparison
# ---------------------------------------------------------------------------
def run_static(seed: int, bc: dict, rp: dict) -> list[dict]:
    alerts = _generate(seed, N_TRAIN_STATIC + N_TEST_STATIC, bc, rp)
    train  = alerts[:N_TRAIN_STATIC]
    test   = alerts[N_TRAIN_STATIC:]
    y_true = np.array([a.gt_action_index for a in test])

    preds: dict[str, np.ndarray] = {}

    # --- L2 centroid static (from training data means) ---
    mu_train = _build_mu_from_data(train)
    preds["l2_centroid"] = _l2_predict_batch(mu_train, test)

    # --- L2 centroid + online learning during test ---
    scorer = ProfileScorer(N_CATEGORIES, N_ACTIONS, N_FACTORS,
                           tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed)
    scorer.mu[:] = mu_train
    y_online = np.empty(len(test), dtype=np.int64)
    for i, a in enumerate(test):
        a_pred, _, _ = scorer.score(a.factors, a.category_index)
        y_online[i] = a_pred
        scorer.update(a.factors, a.category_index, a_pred, a_pred == a.gt_action_index)
    preds["l2_centroid_online"] = y_online

    # --- ML baselines ---
    X_tr, y_tr = _features(train)
    X_te, _    = _features(test)
    rs = int(seed)  # deterministic sklearn

    lr = LogisticRegression(max_iter=500, random_state=rs)
    lr.fit(X_tr, y_tr)
    preds["logistic_regression"] = lr.predict(X_te)

    gbt = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=rs)
    gbt.fit(X_tr, y_tr)
    preds["xgboost"] = gbt.predict(X_te)

    rf = RandomForestClassifier(n_estimators=100, random_state=rs)
    rf.fit(X_tr, y_tr)
    preds["random_forest"] = rf.predict(X_te)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    preds["knn"] = knn.predict(X_te)

    preds["random"] = np.random.default_rng(seed + 9999).integers(0, N_ACTIONS, size=len(test))

    # --- Compute stats ---
    rows = []
    for method, y_pred in preds.items():
        acc = float(np.mean(y_pred == y_true))
        mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_cat = {}
        for ci, cat in enumerate(CATEGORIES):
            mask = np.array([a.category_index == ci for a in test])
            if mask.sum() > 0:
                per_cat[cat] = round(float(np.mean(y_pred[mask] == y_true[mask])), 4)
        rows.append({
            "seed": seed, "method": method,
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "per_category_accuracy": json.dumps(per_cat),
        })
    return rows


# ---------------------------------------------------------------------------
# Online comparison
# ---------------------------------------------------------------------------
def run_online(seed: int, bc: dict, rp: dict) -> list[dict]:
    """
    Online comparison: 200 train / 1300 test.
    Checkpoints at total_seen = 400, 600, 800, 1000, 1200, 1500.
    ML baselines retrain every 100 test alerts on all available data.
    """
    alerts = _generate(seed, N_TOTAL_ONLINE, bc, rp)
    train  = alerts[:N_TRAIN_ONLINE]   # 200
    test   = alerts[N_TRAIN_ONLINE:]   # 1300
    y_true = np.array([a.gt_action_index for a in test])

    # Initial training
    X_train0, y_train0 = _features(train)
    mu_train = _build_mu_from_data(train)

    # L2 centroid online
    scorer = ProfileScorer(N_CATEGORIES, N_ACTIONS, N_FACTORS,
                           tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed)
    scorer.mu[:] = mu_train

    # ML baselines (n_estimators=50 for speed in retrain loop)
    rs = int(seed)
    lr_m   = LogisticRegression(max_iter=500, random_state=rs)
    gbt_m  = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=rs)
    rf_m   = RandomForestClassifier(n_estimators=50, random_state=rs)
    for m in (lr_m, gbt_m, rf_m):
        m.fit(X_train0, y_train0)

    # Growing data pool for retrain
    X_pool = X_train0.copy()
    y_pool = y_train0.copy()

    # Per-method correct lists
    correct: dict[str, list[int]] = {
        "l2_centroid": [],
        "logistic_regression_retrain": [],
        "xgboost_retrain": [],
        "random_forest_retrain": [],
    }

    rows = []

    for i, alert in enumerate(test):
        true_a = alert.gt_action_index
        X_i    = _feat_single(alert)

        # L2 centroid
        a_l2, _, _ = scorer.score(alert.factors, alert.category_index)
        correct["l2_centroid"].append(int(a_l2 == true_a))
        scorer.update(alert.factors, alert.category_index, a_l2, a_l2 == true_a)

        # ML
        correct["logistic_regression_retrain"].append(int(lr_m.predict(X_i)[0]  == true_a))
        correct["xgboost_retrain"].append(            int(gbt_m.predict(X_i)[0] == true_a))
        correct["random_forest_retrain"].append(      int(rf_m.predict(X_i)[0]  == true_a))

        # Accumulate data and retrain every 100 test alerts
        X_pool = np.vstack([X_pool, X_i])
        y_pool = np.append(y_pool, true_a)

        if (i + 1) % 100 == 0:
            lr_m  = LogisticRegression(max_iter=500, random_state=rs)
            gbt_m = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=rs)
            rf_m  = RandomForestClassifier(n_estimators=50, random_state=rs)
            for m in (lr_m, gbt_m, rf_m):
                m.fit(X_pool, y_pool)

        # Record at checkpoints
        total_seen = N_TRAIN_ONLINE + i + 1
        if total_seen in ONLINE_CHECKPOINTS:
            for method, c_list in correct.items():
                arr = c_list
                cum_acc = float(np.mean(arr))
                win_acc = float(np.mean(arr[-100:])) if len(arr) >= 100 else float(np.mean(arr))
                rows.append({
                    "seed": seed, "method": method,
                    "checkpoint": total_seen,
                    "cumulative_accuracy": round(cum_acc, 4),
                    "window_accuracy_last_100": round(win_acc, 4),
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_baseline_comparison() -> None:
    t0 = time.time()
    bc, rp = _load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    static_rows: list[dict] = []
    online_rows: list[dict] = []

    for seed in SEEDS:
        print(f"  seed {seed} ...", end="", flush=True)
        static_rows.extend(run_static(seed, bc, rp))
        online_rows.extend(run_online(seed, bc, rp))
        print(" done")

    # -----------------------------------------------------------------------
    # Write static CSV
    # -----------------------------------------------------------------------
    static_csv = OUT_DIR / "baseline_static_results.csv"
    with open(static_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "method", "accuracy",
                                            "macro_f1", "per_category_accuracy"])
        w.writeheader()
        w.writerows(static_rows)
    print(f"Wrote {len(static_rows)} rows -> {static_csv}")

    # -----------------------------------------------------------------------
    # Write online CSV
    # -----------------------------------------------------------------------
    online_csv = OUT_DIR / "baseline_online_results.csv"
    with open(online_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "method", "checkpoint",
                                            "cumulative_accuracy",
                                            "window_accuracy_last_100"])
        w.writeheader()
        w.writerows(online_rows)
    print(f"Wrote {len(online_rows)} rows -> {online_csv}")

    # -----------------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------------
    STATIC_METHODS = ["l2_centroid", "l2_centroid_online", "logistic_regression",
                      "xgboost", "random_forest", "knn", "random"]
    ONLINE_METHODS = ["l2_centroid", "logistic_regression_retrain",
                      "xgboost_retrain", "random_forest_retrain"]

    static_comp = {}
    for m in STATIC_METHODS:
        accs = [r["accuracy"] for r in static_rows if r["method"] == m]
        static_comp[m] = {
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std":           round(float(np.std(accs)),  4),
        }

    final_cp = ONLINE_CHECKPOINTS[-1]  # 1500
    online_comp = {}
    for m in ONLINE_METHODS:
        lc = []
        for cp in ONLINE_CHECKPOINTS:
            cp_rows = [r for r in online_rows if r["method"] == m and r["checkpoint"] == cp]
            if cp_rows:
                lc.append({
                    "checkpoint": cp,
                    "mean_cumulative_accuracy": round(
                        float(np.mean([r["cumulative_accuracy"] for r in cp_rows])), 4),
                    "mean_window_accuracy": round(
                        float(np.mean([r["window_accuracy_last_100"] for r in cp_rows])), 4),
                })
        fin = [r for r in online_rows if r["method"] == m and r["checkpoint"] == final_cp]
        online_comp[m] = {
            "accuracy_at_1500": round(float(np.mean([r["cumulative_accuracy"] for r in fin])), 4),
            "learning_curve":   lc,
        }

    # Verdict based on static comparison
    baselines = [m for m in STATIC_METHODS
                 if m not in ("l2_centroid", "l2_centroid_online", "random")]
    best_bl   = max(baselines, key=lambda m: static_comp[m]["mean_accuracy"])
    l2_acc    = static_comp["l2_centroid"]["mean_accuracy"]
    bl_acc    = static_comp[best_bl]["mean_accuracy"]
    gap_pp    = (l2_acc - bl_acc) * 100

    if gap_pp > 2.0:
        verdict = (f"l2_centroid_superior: leads best baseline ({best_bl}) "
                   f"by {gap_pp:.1f}pp ({l2_acc*100:.1f}% vs {bl_acc*100:.1f}%)")
    elif gap_pp >= -2.0:
        verdict = (f"competitive: l2_centroid within 2pp of {best_bl} "
                   f"({l2_acc*100:.1f}% vs {bl_acc*100:.1f}%)")
    else:
        verdict = (f"baselines_superior: {best_bl} leads l2_centroid "
                   f"by {-gap_pp:.1f}pp ({bl_acc*100:.1f}% vs {l2_acc*100:.1f}%)")

    summary = {
        "static_comparison":  static_comp,
        "online_comparison":  online_comp,
        "verdict":            verdict,
    }

    json_path = OUT_DIR / "baseline_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("VALIDATION-3A: Baseline Comparison")
    print("=" * 72)

    print(f"\n  Static:  {N_TRAIN_STATIC} train / {N_TEST_STATIC} test / "
          f"{len(SEEDS)} seeds / noise={NOISE_RATE}")
    print(f"  {'Method':<32}  {'Accuracy':>10}  {'std':>6}  {'macro_F1':>10}")
    print(f"  {'-'*62}")
    for m in STATIC_METHODS:
        acc  = static_comp[m]["mean_accuracy"]
        std  = static_comp[m]["std"]
        f1s  = [r["macro_f1"] for r in static_rows if r["method"] == m]
        mf1  = float(np.mean(f1s))
        mark = "**" if m == "l2_centroid" else "  "
        print(f"{mark} {m:<32}  {acc*100:>9.2f}%  {std*100:>5.2f}%  {mf1*100:>9.2f}%")

    print(f"\n  VERDICT: {verdict}")

    print(f"\n  Online: {N_TRAIN_ONLINE} train / 1300 test (ML retrains every 100 test alerts)")
    cp_disp = [400, 600, 1000, 1500]
    hdr = "  ".join(f"@{cp:>4d}" for cp in cp_disp)
    print(f"  {'Method':<37}  {hdr}")
    print(f"  {'-'*72}")
    for m in ONLINE_METHODS:
        vals = []
        for cp in cp_disp:
            cp_rows = [r for r in online_rows if r["method"] == m and r["checkpoint"] == cp]
            v = float(np.mean([r["cumulative_accuracy"] for r in cp_rows])) if cp_rows else 0.0
            vals.append(f"{v*100:>5.1f}%")
        print(f"  {m:<37}  {'  '.join(vals)}")

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    run_baseline_comparison()
