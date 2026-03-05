"""
EXP-C1: Centroid Oracle Diagnostic.

Pure classification — no ScoringMatrix, no gating, no learning, no oracle.
For each alert: which configured profile centroid is closest? Is that correct?

Answers whether the factor data contains enough signal for profile-based
classification WITHOUT any learning.

Generates
---------
experiments/expC1_centroid_oracle/results/classification_results.csv
experiments/expC1_centroid_oracle/results/confusion_matrices.json
experiments/expC1_centroid_oracle/results/summary.json
paper_figures/expC1_*.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator


# ---------------------------------------------------------------------------
# Config load
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw["realistic_profiles"]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run() -> None:
    bc, realistic = _load_config()

    categories   = bc["categories"]
    actions      = bc["actions"]
    factors      = bc["factors"]
    seeds        = bc["seeds"]
    factor_sigma = float(bc["factor_sigma"])

    profiles = realistic["action_conditional_profiles"]
    gt_dists = realistic["category_gt_distributions"]

    n_categories = len(categories)
    n_actions    = len(actions)
    n_factors    = len(factors)
    n_per_cat    = 2000   # 2000 per category = 10000 total per seed

    # Build centroid matrix: mu[c, a, :] = configured mean factor vector
    mu = np.zeros((n_categories, n_actions, n_factors), dtype=np.float64)
    for c_idx, cat in enumerate(categories):
        for a_idx, act in enumerate(actions):
            mu[c_idx, a_idx, :] = np.array(profiles[cat][act], dtype=np.float64)

    results_dir = ROOT / "experiments" / "expC1_centroid_oracle" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # Per-seed confusion matrices: [method][category] = 4x4 matrix (row=pred, col=true)
    confusion: dict[str, dict[str, np.ndarray]] = {
        m: {cat: np.zeros((n_actions, n_actions), dtype=np.float64) for cat in categories}
        for m in ("dot", "l2", "cos")
    }

    seed_accs: dict[str, list[float]] = {"dot": [], "l2": [], "cos": []}

    for seed in seeds:
        gen = CategoryAlertGenerator(
            categories=categories,
            actions=actions,
            factors=factors,
            action_conditional_profiles=profiles,
            gt_distributions=gt_dists,
            factor_sigma=factor_sigma,
            noise_rate=0.0,
            seed=seed,
        )
        alerts = gen.generate_batch(n_per_category=n_per_cat)

        n_correct = {"dot": 0, "l2": 0, "cos": 0}

        for alert in alerts:
            f  = alert.factors          # shape (6,)
            c  = alert.category_index
            gt = alert.gt_action_index

            mu_c = mu[c]                # shape (4, 6)

            # Method 1: Dot product similarity
            dot_scores = mu_c @ f       # (4,)
            dot_pred   = int(np.argmax(dot_scores))
            dot_correct = (dot_pred == gt)

            # Method 2: Negative L2 distance (nearest centroid)
            l2_scores  = -np.linalg.norm(mu_c - f, axis=1)   # (4,)
            l2_pred    = int(np.argmax(l2_scores))
            l2_correct = (l2_pred == gt)

            # Method 3: Cosine similarity
            f_norm     = np.linalg.norm(f)
            mu_norms   = np.linalg.norm(mu_c, axis=1)         # (4,)
            cos_scores = (mu_c @ f) / (f_norm * mu_norms + 1e-10)
            cos_pred   = int(np.argmax(cos_scores))
            cos_correct = (cos_pred == gt)

            # Accumulate confusion (row=pred, col=true)
            confusion["dot"][categories[c]][dot_pred, gt] += 1
            confusion["l2"][categories[c]][l2_pred,  gt] += 1
            confusion["cos"][categories[c]][cos_pred, gt] += 1

            n_correct["dot"] += int(dot_correct)
            n_correct["l2"]  += int(l2_correct)
            n_correct["cos"] += int(cos_correct)

            all_rows.append({
                "seed": seed,
                "category_index": c,
                "category": categories[c],
                "gt_action_index": gt,
                "gt_action": actions[gt],
                "dot_pred": dot_pred,
                "dot_correct": dot_correct,
                "l2_pred": l2_pred,
                "l2_correct": l2_correct,
                "cos_pred": cos_pred,
                "cos_correct": cos_correct,
            })

        n_total = len(alerts)
        dot_acc = n_correct["dot"] / n_total
        l2_acc  = n_correct["l2"]  / n_total
        cos_acc = n_correct["cos"] / n_total

        for m in ("dot", "l2", "cos"):
            seed_accs[m].append(n_correct[m] / n_total)

        print(f"Seed {seed}: dot={dot_acc:.1%} l2={l2_acc:.1%} cos={cos_acc:.1%}")

    # -----------------------------------------------------------------------
    # Save classification_results.csv
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = results_dir / "classification_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Build summary stats
    # -----------------------------------------------------------------------
    n_seeds = len(seeds)

    overall_acc: dict[str, float] = {}
    std_acc:     dict[str, float] = {}
    per_cat_acc: dict[str, dict[str, float]] = {}

    for m in ("dot", "l2", "cos"):
        overall_acc[m] = float(np.mean(seed_accs[m]))
        std_acc[m]     = float(np.std(seed_accs[m]))

        per_cat_acc[m] = {}
        for cat in categories:
            cm    = confusion[m][cat]
            total = float(cm.sum())
            per_cat_acc[m][cat] = float(np.trace(cm)) / total if total > 0 else 0.0

    best_method   = max(("dot", "l2", "cos"), key=lambda m: overall_acc[m])
    best_accuracy = overall_acc[best_method]

    # Per-category per-action accuracy (recall: correctly predicted / total true)
    per_cat_per_action_acc: dict[str, dict[str, dict[str, float]]] = {}
    for m in ("dot", "l2", "cos"):
        per_cat_per_action_acc[m] = {}
        for cat in categories:
            per_cat_per_action_acc[m][cat] = {}
            cm = confusion[m][cat]
            for a_idx, act in enumerate(actions):
                col_total = float(cm[:, a_idx].sum())
                correct   = float(cm[a_idx, a_idx])
                per_cat_per_action_acc[m][cat][act] = (
                    correct / col_total if col_total > 0 else 0.0
                )

    summary = {
        "methods": {
            m: {
                "overall_accuracy":               overall_acc[m],
                "std_accuracy":                   std_acc[m],
                "per_category_accuracy":          per_cat_acc[m],
                "per_category_per_action_accuracy": per_cat_per_action_acc[m],
            }
            for m in ("dot", "l2", "cos")
        },
        "best_method":        best_method,
        "best_accuracy":      best_accuracy,
        "n_seeds":            n_seeds,
        "n_per_category":     n_per_cat,
        "n_total_per_seed":   n_per_cat * n_categories,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {summary_path}")

    # -----------------------------------------------------------------------
    # Save confusion_matrices.json (averaged across seeds)
    # -----------------------------------------------------------------------
    cm_out: dict = {}
    for m in ("dot", "l2", "cos"):
        cm_out[m] = {}
        for cat in categories:
            cm_mat = confusion[m][cat] / n_seeds
            cm_out[m][cat] = cm_mat.tolist()

    cm_path = results_dir / "confusion_matrices.json"
    with open(cm_path, "w") as fh:
        json.dump(cm_out, fh, indent=2)
    print(f"Saved: {cm_path}")

    # -----------------------------------------------------------------------
    # Validation report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXP-C1: CENTROID ORACLE DIAGNOSTIC")
    print("=" * 60)

    for m in ("dot", "l2", "cos"):
        print(f"  {m:>3}: {overall_acc[m]:.4f} +/- {std_acc[m]:.4f}")
        for cat in categories:
            print(f"    {cat}: {per_cat_acc[m][cat]:.4f}")

    print(f"\n  Best method: {best_method} at {best_accuracy:.4f}")

    print(f"\n  COMPARISON TO EXP-A:")
    print(f"    Shared W (Hebbian, 1000 decisions):      49.26%")
    print(f"    Per-category W (Hebbian, 200/cat):        51.61%")
    print(f"    Centroid oracle (NO learning):            {best_accuracy:.1%}")
    print(f"    Gap: centroid oracle - shared W =         {(best_accuracy - 0.4926)*100:+.1f}pp")

    if best_accuracy > 0.75:
        print(f"\n  >>> CEILING CONFIRMED at {best_accuracy:.1%}.")
        print(f"  >>> Factor data contains sufficient signal.")
        print(f"  >>> The problem is LEARNING, not DATA.")
        print(f"  >>> Profile-based scoring (start from config, refine) is the path forward.")
    elif best_accuracy > 0.60:
        print(f"\n  >>> Partial ceiling: {best_accuracy:.1%}.")
        print(f"  >>> Factor data has signal but profiles may need tightening.")
    else:
        print(f"\n  >>> LOW ceiling: {best_accuracy:.1%}.")
        print(f"  >>> Factor profiles don't contain enough signal for classification.")
        print(f"  >>> Need to redesign realistic_profiles or add factors.")

    print(f"\n  PER-CATEGORY DIAGNOSIS (best method: {best_method}):")
    for cat in categories:
        acc = per_cat_acc[best_method][cat]
        if acc > 0.80:
            label = "EASY (well-separated profiles)"
        elif acc > 0.60:
            label = "MODERATE (some profile overlap)"
        else:
            label = "HARD (significant profile overlap)"
        print(f"    {cat}: {acc:.1%} -- {label}")

    # Most confused action pairs per category
    print(f"\n  MOST CONFUSED ACTION PAIRS (best method: {best_method}):")
    for cat in categories:
        cm = confusion[best_method][cat]
        best_conf_rate = 0.0
        best_pair      = ("", "")
        for true_a in range(n_actions):
            col_total = float(cm[:, true_a].sum())
            if col_total == 0:
                continue
            for pred_a in range(n_actions):
                if pred_a == true_a:
                    continue
                rate = float(cm[pred_a, true_a]) / col_total
                if rate > best_conf_rate:
                    best_conf_rate = rate
                    best_pair = (actions[true_a], actions[pred_a])
        if best_pair[0]:
            print(f"    {cat}: {best_pair[0]} confused with {best_pair[1]} ({best_conf_rate:.1%})")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    from src.viz.expC1_charts import generate_all_charts
    generate_all_charts(str(results_dir))


if __name__ == "__main__":
    run()
