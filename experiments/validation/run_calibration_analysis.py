"""
VALIDATION-3B: Calibration Analysis (ECE).

Measures whether softmax confidence scores from the L2 centroid scorer
are well-calibrated, and compares against sklearn baseline calibration.

Reuses data generation and centroid logic from run_baseline_comparison.py.

Computes:
  - ECE (Expected Calibration Error) per method
  - Per-category ECE for L2 centroid
  - Temperature sensitivity: ECE at tau = [0.1, 0.25, 0.5, 1.0]
  - Confidence distribution (overconfidence check)

Calibration thresholds (standard):
  ECE < 0.05  -> well_calibrated
  ECE < 0.15  -> moderately_calibrated
  ECE >= 0.15 -> poorly_calibrated

Outputs
-------
experiments/validation/calibration_results.csv
experiments/validation/calibration_summary.json
"""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Reuse helpers from run_baseline_comparison (same data/model logic)
from experiments.validation.run_baseline_comparison import (
    _load_config, _generate, _features, _build_mu_from_data,
    SEEDS, N_TRAIN_STATIC, N_TEST_STATIC, N_CATEGORIES, N_ACTIONS, N_FACTORS,
)
from src.data.category_alert_generator import CATEGORIES, ACTIONS

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TAUS     = [0.1, 0.25, 0.5, 1.0]
N_BINS   = 10
ECE_WELL = 0.05
ECE_MOD  = 0.15

CAL_METHODS = [
    "l2_centroid",
    "logistic_regression",
    "xgboost",
    "random_forest",
]


# ---------------------------------------------------------------------------
# L2 centroid probability vector
# ---------------------------------------------------------------------------
def _l2_probs(mu: np.ndarray, alert, tau: float) -> np.ndarray:
    """Return softmax(-||f - mu[c,a]||^2 / tau) over all actions."""
    f    = alert.factors
    c    = alert.category_index
    dists = np.sum((mu[c] - f) ** 2, axis=1)   # (n_actions,)
    neg   = -dists / tau
    neg  -= neg.max()                            # numerically stable
    e     = np.exp(neg)
    return e / e.sum()


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------
def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[float, list[dict]]:
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece      = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(mask.sum())
        if count == 0:
            bin_data.append({"bin": i, "count": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        bin_acc  = float((predictions[mask] == ground_truth[mask]).mean())
        bin_conf = float(confidences[mask].mean())
        ece     += (count / len(confidences)) * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin": i, "count": count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
        })

    return ece, bin_data


# ---------------------------------------------------------------------------
# Per-seed calibration
# ---------------------------------------------------------------------------
def run_calibration_seed(seed: int, bc: dict, rp: dict) -> dict:
    """
    Returns dict with keys: rows (for CSV), bin_data, per_cat_ece, tau_ece.
    """
    alerts = _generate(seed, N_TRAIN_STATIC + N_TEST_STATIC, bc, rp)
    train  = alerts[:N_TRAIN_STATIC]
    test   = alerts[N_TEST_STATIC:]      # last 500
    y_true = np.array([a.gt_action_index for a in test])

    mu_train = _build_mu_from_data(train)

    # -----------------------------------------------------------------------
    # L2 centroid at default tau + temperature sweep
    # -----------------------------------------------------------------------
    tau_results: dict[float, dict] = {}
    for tau in TAUS:
        probs_all = np.vstack([_l2_probs(mu_train, a, tau) for a in test])
        confs  = probs_all.max(axis=1)
        preds  = probs_all.argmax(axis=1)
        ece, bdata = compute_ece(confs, preds, y_true)
        acc = float((preds == y_true).mean())
        tau_results[tau] = {
            "ece": round(ece, 6),
            "mean_confidence": round(float(confs.mean()), 4),
            "accuracy": round(acc, 4),
            "bin_data": bdata,
            "confs": confs,
            "preds": preds,
        }

    # -----------------------------------------------------------------------
    # Per-category ECE at default tau (0.25)
    # -----------------------------------------------------------------------
    default_res  = tau_results[0.25]
    confs_def    = default_res["confs"]
    preds_def    = default_res["preds"]
    per_cat_ece: dict[str, float] = {}
    for ci, cat_name in enumerate(CATEGORIES):
        mask = np.array([a.category_index == ci for a in test])
        if mask.sum() == 0:
            per_cat_ece[cat_name] = None
            continue
        ece_c, _ = compute_ece(confs_def[mask], preds_def[mask], y_true[mask])
        per_cat_ece[cat_name] = round(ece_c, 6)

    # -----------------------------------------------------------------------
    # ML baselines with predict_proba
    # -----------------------------------------------------------------------
    X_tr, y_tr = _features(train)
    X_te, _    = _features(test)
    rs         = int(seed)

    ml_results: dict[str, dict] = {}
    for name, clf in [
        ("logistic_regression",
         LogisticRegression(max_iter=500, random_state=rs)),
        ("xgboost",
         GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=rs)),
        ("random_forest",
         RandomForestClassifier(n_estimators=100, random_state=rs)),
    ]:
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)    # shape (n, n_classes)
        # predict_proba columns are ordered by clf.classes_
        # Remap to full action index space
        full_proba = np.zeros((len(test), N_ACTIONS), dtype=np.float64)
        for col_idx, class_label in enumerate(clf.classes_):
            full_proba[:, class_label] = proba[:, col_idx]
        confs = full_proba.max(axis=1)
        preds = full_proba.argmax(axis=1)
        ece, bdata = compute_ece(confs, preds, y_true)
        ml_results[name] = {
            "ece": round(ece, 6),
            "mean_confidence": round(float(confs.mean()), 4),
            "accuracy": round(float((preds == y_true).mean()), 4),
            "bin_data": bdata,
            "confs": confs,
            "preds": preds,
        }

    return {
        "seed": seed,
        "tau_results": tau_results,
        "per_cat_ece": per_cat_ece,
        "ml_results": ml_results,
        "y_true": y_true,
        "test": test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_calibration() -> None:
    t0 = time.time()
    bc, rp = _load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in SEEDS:
        print(f"  seed {seed} ...", end="", flush=True)
        seed_results.append(run_calibration_seed(seed, bc, rp))
        print(" done")

    # -----------------------------------------------------------------------
    # Build CSV rows
    # -----------------------------------------------------------------------
    csv_rows: list[dict] = []
    for sr in seed_results:
        seed = sr["seed"]
        for tau in TAUS:
            res = sr["tau_results"][tau]
            csv_rows.append({
                "seed": seed, "method": "l2_centroid",
                "tau": tau, "ece": res["ece"],
                "mean_confidence": res["mean_confidence"],
                "accuracy": res["accuracy"],
            })
        for name, res in sr["ml_results"].items():
            csv_rows.append({
                "seed": seed, "method": name,
                "tau": None, "ece": res["ece"],
                "mean_confidence": res["mean_confidence"],
                "accuracy": res["accuracy"],
            })

    csv_path = OUT_DIR / "calibration_results.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "method", "tau", "ece",
                                            "mean_confidence", "accuracy"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Wrote {len(csv_rows)} rows -> {csv_path}")

    # -----------------------------------------------------------------------
    # Build summary JSON
    # -----------------------------------------------------------------------
    def _agg(vals: list[float]) -> tuple[float, float]:
        return round(float(np.mean(vals)), 6), round(float(np.std(vals)), 6)

    # -- Aggregate bin data across seeds (for reliability diagram) --
    def _agg_bins(bin_lists: list[list[dict]]) -> list[dict]:
        """Average bin accuracy and confidence across seeds (weight by count)."""
        n_b = N_BINS
        agg = []
        for b in range(n_b):
            total_count = sum(bl[b]["count"] for bl in bin_lists)
            if total_count == 0:
                agg.append({"bin": b, "count": 0, "mean_accuracy": 0.0,
                             "mean_confidence": 0.0})
                continue
            w_acc  = sum(bl[b]["accuracy"]   * bl[b]["count"] for bl in bin_lists)
            w_conf = sum(bl[b]["confidence"] * bl[b]["count"] for bl in bin_lists)
            agg.append({
                "bin":             b,
                "count":           total_count,
                "mean_accuracy":   round(w_acc  / total_count, 4),
                "mean_confidence": round(w_conf / total_count, 4),
            })
        return agg

    # -- L2 centroid at default tau --
    l2_eces  = [sr["tau_results"][0.25]["ece"]              for sr in seed_results]
    l2_confs = [sr["tau_results"][0.25]["mean_confidence"]  for sr in seed_results]
    l2_bins  = [sr["tau_results"][0.25]["bin_data"]         for sr in seed_results]
    l2_mean_ece, l2_std_ece = _agg(l2_eces)

    # Confidence distribution: fraction of test preds with conf > 0.95
    l2_high_conf = float(np.mean([
        (sr["tau_results"][0.25]["confs"] > 0.95).mean()
        for sr in seed_results
    ]))

    l2_entry = {
        "mean_ece":          l2_mean_ece,
        "std_ece":           l2_std_ece,
        "mean_confidence":   round(float(np.mean(l2_confs)), 4),
        "pct_confidence_gt_095": round(l2_high_conf * 100, 1),
        "reliability_bins":  _agg_bins(l2_bins),
    }

    # -- ML baselines --
    ml_entries: dict[str, dict] = {}
    for name in ["logistic_regression", "xgboost", "random_forest"]:
        eces  = [sr["ml_results"][name]["ece"]             for sr in seed_results]
        confs = [sr["ml_results"][name]["mean_confidence"] for sr in seed_results]
        bins  = [sr["ml_results"][name]["bin_data"]        for sr in seed_results]
        high  = float(np.mean([
            (sr["ml_results"][name]["confs"] > 0.95).mean()
            for sr in seed_results
        ]))
        me, se = _agg(eces)
        ml_entries[name] = {
            "mean_ece":              me,
            "std_ece":               se,
            "mean_confidence":       round(float(np.mean(confs)), 4),
            "pct_confidence_gt_095": round(high * 100, 1),
            "reliability_bins":      _agg_bins(bins),
        }

    # -- Per-category ECE (L2, default tau) --
    per_cat_agg: dict[str, float | None] = {}
    for cat_name in CATEGORIES:
        vals = [sr["per_cat_ece"][cat_name]
                for sr in seed_results
                if sr["per_cat_ece"][cat_name] is not None]
        per_cat_agg[cat_name] = round(float(np.mean(vals)), 6) if vals else None

    # -- Temperature sensitivity --
    temp_sens: dict[str, dict] = {}
    for tau in TAUS:
        eces  = [sr["tau_results"][tau]["ece"]            for sr in seed_results]
        confs = [sr["tau_results"][tau]["mean_confidence"] for sr in seed_results]
        me, _ = _agg(eces)
        temp_sens[str(tau)] = {
            "mean_ece":        me,
            "mean_confidence": round(float(np.mean(confs)), 4),
        }

    # -- Verdict --
    best_tau_str = min(temp_sens, key=lambda t: temp_sens[t]["mean_ece"])
    best_ece     = temp_sens[best_tau_str]["mean_ece"]

    if l2_mean_ece < ECE_WELL:
        cal_grade = "well_calibrated"
    elif l2_mean_ece < ECE_MOD:
        cal_grade = "moderately_calibrated"
    else:
        cal_grade = "poorly_calibrated"

    best_ml_name = min(ml_entries, key=lambda m: ml_entries[m]["mean_ece"])
    best_ml_ece  = ml_entries[best_ml_name]["mean_ece"]

    verdict = (
        f"L2 centroid (tau=0.25) is {cal_grade} "
        f"(ECE={l2_mean_ece:.4f}). "
        f"Best ML baseline: {best_ml_name} ECE={best_ml_ece:.4f}. "
        f"Optimal tau={best_tau_str} gives ECE={best_ece:.4f}."
    )

    summary = {
        "l2_centroid":         l2_entry,
        "logistic_regression": ml_entries["logistic_regression"],
        "xgboost":             ml_entries["xgboost"],
        "random_forest":       ml_entries["random_forest"],
        "per_category_ece":    per_cat_agg,
        "temperature_sensitivity": temp_sens,
        "verdict":             verdict,
    }

    json_path = OUT_DIR / "calibration_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("VALIDATION-3B: Calibration Analysis (ECE)")
    print("=" * 72)

    print(f"\n  Method comparison (ECE, confidence, accuracy — 10 seeds):")
    print(f"  {'Method':<28}  {'ECE':>8}  {'std':>6}  "
          f"{'MeanConf':>9}  {'>0.95':>7}  {'Accuracy':>9}")
    print(f"  {'-'*70}")

    # L2 centroid
    l2_acc_vals = [sr["tau_results"][0.25]["accuracy"] for sr in seed_results]
    print(f"  {'l2_centroid (tau=0.25)':<28}  "
          f"{l2_mean_ece:>8.4f}  "
          f"{l2_std_ece:>6.4f}  "
          f"{np.mean(l2_confs):>8.3f}  "
          f"{l2_high_conf*100:>6.1f}%  "
          f"{np.mean(l2_acc_vals)*100:>8.2f}%")

    for name in ["logistic_regression", "xgboost", "random_forest"]:
        e   = ml_entries[name]
        acc_vals = [sr["ml_results"][name]["accuracy"] for sr in seed_results]
        print(f"  {name:<28}  "
              f"{e['mean_ece']:>8.4f}  "
              f"{e['std_ece']:>6.4f}  "
              f"{e['mean_confidence']:>8.3f}  "
              f"{e['pct_confidence_gt_095']:>6.1f}%  "
              f"{np.mean(acc_vals)*100:>8.2f}%")

    print(f"\n  Temperature sensitivity (L2 centroid, tau sweep):")
    print(f"  {'tau':<8}  {'mean ECE':>10}  {'mean confidence':>17}  "
          f"{'calibration':>14}")
    print(f"  {'-'*54}")
    for tau in TAUS:
        ts    = temp_sens[str(tau)]
        grade = ("well" if ts["mean_ece"] < ECE_WELL
                 else "moderate" if ts["mean_ece"] < ECE_MOD
                 else "poor")
        marker = " *" if str(tau) == best_tau_str else "  "
        print(f"{marker} {str(tau):<8}  "
              f"{ts['mean_ece']:>10.4f}  "
              f"{ts['mean_confidence']:>17.4f}  "
              f"{grade:>14}")
    print(f"  (* = lowest ECE)")

    print(f"\n  Per-category ECE (L2 centroid, tau=0.25):")
    print(f"  {'Category':<25}  {'ECE':>8}  {'Grade':>15}")
    print(f"  {'-'*50}")
    for cat_name in CATEGORIES:
        ece_c = per_cat_agg[cat_name]
        if ece_c is None:
            print(f"  {cat_name:<25}  {'N/A':>8}")
            continue
        grade = ("well" if ece_c < ECE_WELL
                 else "moderate" if ece_c < ECE_MOD
                 else "poor")
        print(f"  {cat_name:<25}  {ece_c:>8.4f}  {grade:>15}")

    print(f"\n  Reliability bins (L2 centroid, tau=0.25):")
    print(f"  {'Bin':>4}  {'Range':>12}  {'Count':>7}  "
          f"{'Mean Acc':>9}  {'Mean Conf':>10}  {'Gap':>7}")
    print(f"  {'-'*56}")
    boundaries = np.linspace(0.0, 1.0, N_BINS + 1)
    for b_data in l2_entry["reliability_bins"]:
        b   = b_data["bin"]
        lo  = boundaries[b];  hi = boundaries[b + 1]
        cnt = b_data["count"]
        if cnt == 0:
            continue
        acc_b  = b_data["mean_accuracy"]
        conf_b = b_data["mean_confidence"]
        gap    = acc_b - conf_b
        print(f"  {b:>4}  ({lo:.1f},{hi:.1f}]  "
              f"{cnt:>7}  {acc_b:>9.3f}  {conf_b:>10.3f}  "
              f"{gap:>+7.3f}")

    print(f"\n  VERDICT: {verdict}")
    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    run_calibration()
