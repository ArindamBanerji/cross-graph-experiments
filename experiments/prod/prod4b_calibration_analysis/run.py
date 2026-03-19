"""
PROD-4b: Per-Category Reliability Diagrams and ECE Analysis.

Regime: centroidal synthetic
Ontology: SOC product v5.0+refer (C=6, A=5, d=6)

Diagnostic experiment recommended by 4-model judge panel before the factorial
calibration sweep.  Saves raw (confidence, is_correct, category) triples for
binned calibration analysis, runs at both eta_neg=0.05 and eta_neg=1.0 to
surface the product-vs-experiments mismatch.

tau=0.1 throughout (current production value — this is what we are diagnosing).
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
# Config
# ---------------------------------------------------------------------------

N_SEEDS           = 20
N_WARMUP          = 200
N_EVAL            = 1000
TAU               = 0.1
ETA               = 0.05
NOISE_RATE        = 0.10
DOMAIN_CONFIG     = "soc_product_v50"
RANDOM_SEED_BASE  = 42

ETA_NEG_VALUES    = [0.05, 1.0]   # product vs experiments

N_BINS   = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

# ---------------------------------------------------------------------------
# Main loop — one pass per eta_neg value
# ---------------------------------------------------------------------------

for eta_neg_val in ETA_NEG_VALUES:
    print(f"\n=== Running with eta_neg={eta_neg_val} ===")

    all_raw_records: list[dict] = []   # accumulate across seeds

    for seed in range(N_SEEDS):
        print(f"  Seed {seed+1}/{N_SEEDS}", flush=True)

        gen = CategoryAlertGenerator(
            **config["generator_kwargs"],
            noise_rate=NOISE_RATE,
            seed=RANDOM_SEED_BASE + seed,
        )
        scorer = ProfileScorer(
            config["mu"].copy(),
            config["actions"],
            tau=TAU,
            eta=ETA,
            eta_neg=eta_neg_val,
        )

        # ---- Warmup — learning ON, not recorded ----
        warmup_alerts = gen.generate(N_WARMUP)
        for alert in warmup_alerts:
            scorer.update(
                alert.factors,
                alert.category_index,
                alert.gt_action_index,
                correct=True,
            )

        # ---- Evaluation — learning ON, raw records saved ----
        eval_alerts = gen.generate(N_EVAL)
        for alert in eval_alerts:
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
            all_raw_records.append({
                "seed":          seed,
                "category":      alert.category,
                "category_idx":  alert.category_index,
                "confidence":    float(result.confidence),
                "max_prob":      float(result.probabilities.max()),
                "margin":        float(probs_sorted[-1] - probs_sorted[-2]),
                "is_correct":    bool(is_correct),
                "action_idx":    int(result.action_index),
                "gt_action_idx": int(alert.gt_action_index),
            })

    records = all_raw_records   # N_SEEDS * N_EVAL records

    # ---- 1. Global reliability diagram ----
    global_bins = []
    for i in range(N_BINS):
        lo, hi   = float(BIN_EDGES[i]), float(BIN_EDGES[i + 1])
        in_bin   = [r for r in records if lo <= r["confidence"] < hi]
        if in_bin:
            avg_conf = float(np.mean([r["confidence"] for r in in_bin]))
            avg_acc  = float(np.mean([r["is_correct"]  for r in in_bin]))
            count    = len(in_bin)
        else:
            avg_conf = (lo + hi) / 2.0
            avg_acc  = float("nan")
            count    = 0
        global_bins.append({
            "bin_lo":         lo,
            "bin_hi":         hi,
            "avg_confidence": avg_conf,
            "avg_accuracy":   avg_acc,
            "count":          count,
        })

    total      = len(records)
    global_ece = sum(
        b["count"] / total * abs(b["avg_accuracy"] - b["avg_confidence"])
        for b in global_bins
        if b["count"] > 0 and not np.isnan(b["avg_accuracy"])
    )

    # ---- 2. Per-category reliability diagrams + ECE ----
    per_cat_results: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_records = [r for r in records if r["category"] == cat]
        if not cat_records:
            per_cat_results[cat] = {
                "bins": [], "ece": float("nan"), "n": 0, "accuracy": float("nan"),
            }
            continue

        cat_bins = []
        for i in range(N_BINS):
            lo, hi  = float(BIN_EDGES[i]), float(BIN_EDGES[i + 1])
            in_bin  = [r for r in cat_records if lo <= r["confidence"] < hi]
            if in_bin:
                avg_conf = float(np.mean([r["confidence"] for r in in_bin]))
                avg_acc  = float(np.mean([r["is_correct"]  for r in in_bin]))
                count    = len(in_bin)
            else:
                avg_conf = (lo + hi) / 2.0
                avg_acc  = float("nan")
                count    = 0
            cat_bins.append({
                "bin_lo":         lo,
                "bin_hi":         hi,
                "avg_confidence": avg_conf,
                "avg_accuracy":   avg_acc,
                "count":          count,
            })

        cat_total = len(cat_records)
        cat_ece   = sum(
            b["count"] / cat_total * abs(b["avg_accuracy"] - b["avg_confidence"])
            for b in cat_bins
            if b["count"] > 0 and not np.isnan(b["avg_accuracy"])
        )
        cat_accuracy = float(np.mean([r["is_correct"] for r in cat_records]))

        per_cat_results[cat] = {
            "bins":     cat_bins,
            "ece":      cat_ece,
            "n":        cat_total,
            "accuracy": cat_accuracy,
        }

    # ---- 3. Confidence distribution ----
    confs         = [r["confidence"] for r in records]
    conf_hist, _  = np.histogram(confs, bins=BIN_EDGES)
    pct_above_90  = float(np.mean([c >= 0.90 for c in confs]))

    # ---- 4. Margin analysis ----
    margins    = [r["margin"]     for r in records]
    correctness = [r["is_correct"] for r in records]
    margin_bins = np.linspace(0.0, 1.0, 11)
    margin_accuracy: list[dict] = []
    for i in range(10):
        lo, hi  = float(margin_bins[i]), float(margin_bins[i + 1])
        in_bin  = [(m, c) for m, c in zip(margins, correctness) if lo <= m < hi]
        if in_bin:
            margin_accuracy.append({
                "margin_lo": lo,
                "margin_hi": hi,
                "accuracy":  float(np.mean([c for _, c in in_bin])),
                "count":     len(in_bin),
            })

    # ---- Print results ----
    print(f"\n=== CALIBRATION ANALYSIS (eta_neg={eta_neg_val}) ===")
    print(f"Regime: centroidal synthetic, {N_SEEDS} seeds, tau={TAU}")
    print(f"Total records: {len(records)}")
    print(f"Predictions with P>=0.90: {pct_above_90:.1%}")
    print(f"\nGlobal ECE: {global_ece:.4f}")
    print(f"\nPer-category ECE and accuracy:")
    for cat in CATEGORIES:
        r = per_cat_results[cat]
        if r["n"] > 0:
            print(f"  {cat:24s}: ECE={r['ece']:.4f}  acc={r['accuracy']:.1%}  n={r['n']}")

    print(f"\nPer-category reliability (P>=0.90 bin):")
    for cat in CATEGORIES:
        r        = per_cat_results[cat]
        high_bin = [b for b in r["bins"] if b["bin_lo"] >= 0.90 and b["count"] > 0]
        if high_bin:
            b = high_bin[0]
            print(
                f"  {cat:24s}: P_avg={b['avg_confidence']:.3f}  "
                f"acc={b['avg_accuracy']:.1%}  n={b['count']}  "
                f"miscal={b['avg_confidence'] - b['avg_accuracy']:.3f}"
            )
        else:
            print(f"  {cat:24s}: no decisions in P>=0.90 bin")

    print(f"\nMargin as accuracy predictor:")
    for mb in margin_accuracy:
        print(
            f"  margin [{mb['margin_lo']:.1f}-{mb['margin_hi']:.1f}]: "
            f"accuracy={mb['accuracy']:.1%}  n={mb['count']}"
        )

    # ---- Save JSON ----
    results_path = (
        _REPO_ROOT
        / "experiments" / "prod" / "prod4b_calibration_analysis"
        / f"prod4b_eta_neg_{eta_neg_val:.2f}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "eta_neg":                eta_neg_val,
        "tau":                    TAU,
        "n_seeds":                N_SEEDS,
        "n_warmup":               N_WARMUP,
        "n_eval":                 N_EVAL,
        "noise_rate":             NOISE_RATE,
        "regime":                 "centroidal_synthetic",
        "domain_config":          DOMAIN_CONFIG,
        "global_ece":             global_ece,
        "pct_above_90":           pct_above_90,
        "per_category":           per_cat_results,
        "global_bins":            global_bins,
        "margin_accuracy":        margin_accuracy,
        "confidence_histogram":   conf_hist.tolist(),
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {results_path}")

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
