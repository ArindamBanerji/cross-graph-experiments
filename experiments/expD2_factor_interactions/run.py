"""
EXP-D2: Factor Interaction Discovery.

Tests whether pairwise factor interactions (f_i × f_j) contain additional
discriminative information beyond individual factors.

Phase 1: Compute single-factor and interaction MI across seeds (75 pairs × 5 categories)
Phase 2: If interactions have gain > 1.5, test augmented ProfileScorer

Generates
---------
experiments/expD2_factor_interactions/results/mi_single.csv
experiments/expD2_factor_interactions/results/mi_interaction.csv
experiments/expD2_factor_interactions/results/top_interactions.json
experiments/expD2_factor_interactions/results/augmentation_results.csv (if Phase 2 runs)
experiments/expD2_factor_interactions/results/summary.json
paper_figures/expD2_*.{pdf,png}
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.oracle import GTAlignedOracle
from src.models.profile_scorer import ProfileScorer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw["realistic_profiles"]


# ---------------------------------------------------------------------------
# MI helpers
# ---------------------------------------------------------------------------

def mutual_information(x_binned: np.ndarray, y_labels: np.ndarray) -> float:
    """Compute MI(X; Y) from binned X values and discrete Y labels."""
    n = len(x_binned)
    if n == 0:
        return 0.0
    joint    = Counter(zip(x_binned.tolist(), y_labels.tolist()))
    x_counts = Counter(x_binned.tolist())
    y_counts = Counter(y_labels.tolist())
    mi = 0.0
    for (x, y), n_xy in joint.items():
        p_xy = n_xy / n
        p_x  = x_counts[x] / n
        p_y  = y_counts[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return float(mi)


def bin_values(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin continuous values into n_bins equal-width bins."""
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax - vmin < 1e-10:
        return np.zeros(len(values), dtype=np.int64)
    edges = np.linspace(vmin, vmax, n_bins + 1)
    binned = np.digitize(values, edges[1:-1])          # 0..n_bins-1
    return np.clip(binned, 0, n_bins - 1).astype(np.int64)


# ---------------------------------------------------------------------------
# Phase 1: MI computation
# ---------------------------------------------------------------------------

def _phase1_compute_mi(bc: dict, realistic: dict, seeds: list) -> dict:
    categories    = bc["categories"]
    actions       = bc["actions"]
    factors       = bc["factors"]
    n_categories  = len(categories)
    n_factors     = len(factors)
    pair_indices  = [(i, j) for i in range(n_factors) for j in range(i + 1, n_factors)]
    n_pairs       = len(pair_indices)                  # 15

    all_mi_single      = np.zeros((len(seeds), n_categories, n_factors))
    all_mi_interaction = np.zeros((len(seeds), n_categories, n_pairs))

    for seed_idx, seed in enumerate(seeds):
        gen = CategoryAlertGenerator(
            categories=categories,
            actions=actions,
            factors=factors,
            action_conditional_profiles=realistic["action_conditional_profiles"],
            gt_distributions=realistic["category_gt_distributions"],
            factor_sigma=float(bc["factor_sigma"]),
            noise_rate=0.0,
            seed=seed,
        )
        alerts = gen.generate_batch(n_per_category=2000)   # 10 000 total

        for c_idx in range(n_categories):
            cat_alerts = [a for a in alerts if a.category_index == c_idx]
            if len(cat_alerts) < 100:
                continue

            F = np.array([a.factors.flatten() for a in cat_alerts])   # (n, 6)
            Y = np.array([a.gt_action_index    for a in cat_alerts])   # (n,)

            for f_idx in range(n_factors):
                binned = bin_values(F[:, f_idx])
                all_mi_single[seed_idx, c_idx, f_idx] = mutual_information(binned, Y)

            for pair_idx, (i, j) in enumerate(pair_indices):
                f_ij   = F[:, i] * F[:, j]
                binned = bin_values(f_ij)
                all_mi_interaction[seed_idx, c_idx, pair_idx] = mutual_information(binned, Y)

        print(f"  MI seed {seed}: done")

    mi_single      = np.mean(all_mi_single,      axis=0)   # (5, 6)
    mi_interaction = np.mean(all_mi_interaction, axis=0)   # (5, 15)

    return {
        "mi_single":      mi_single,
        "mi_interaction": mi_interaction,
        "pair_indices":   pair_indices,
    }


# ---------------------------------------------------------------------------
# Phase 2: Augmentation test
# ---------------------------------------------------------------------------

def _phase2_augmentation(
    top_pairs: list[tuple[int, int]],
    bc: dict,
    realistic: dict,
    seeds: list,
) -> list[dict]:
    categories   = bc["categories"]
    actions      = bc["actions"]
    factors      = bc["factors"]
    n_categories = len(categories)
    n_actions    = len(actions)
    n_factors    = len(factors)
    n_aug        = n_factors + len(top_pairs)
    tau          = float(bc["scoring"]["temperature"])

    aug_results: list[dict] = []

    for seed in seeds:
        # ----- test alert stream (shared by base and augmented) -----
        gen = CategoryAlertGenerator(
            categories=categories, actions=actions, factors=factors,
            action_conditional_profiles=realistic["action_conditional_profiles"],
            gt_distributions=realistic["category_gt_distributions"],
            factor_sigma=float(bc["factor_sigma"]),
            noise_rate=0.0, seed=seed,
        )
        alerts = gen.generate(1000)

        oracle = GTAlignedOracle(noise_rate=0.0, seed=seed + 1000)

        # ----- baseline (6 factors, config centroids) -----
        scorer_base = ProfileScorer(n_categories, n_actions, n_factors,
                                    tau=tau, eta=0.05, eta_neg=0.05, seed=seed)
        scorer_base.init_from_profiles(
            realistic["action_conditional_profiles"], categories, actions,
        )
        correct_base = 0
        for alert in alerts:
            action_idx, _, _ = scorer_base.score(alert.factors, alert.category_index)
            if action_idx == alert.gt_action_index:
                correct_base += 1
            result = oracle.evaluate(actions[action_idx], alert)
            scorer_base.update(alert.factors, alert.category_index,
                               action_idx, result.outcome > 0)
        base_acc = correct_base / len(alerts)

        # ----- build augmented profile centroids from data -----
        gen2 = CategoryAlertGenerator(
            categories=categories, actions=actions, factors=factors,
            action_conditional_profiles=realistic["action_conditional_profiles"],
            gt_distributions=realistic["category_gt_distributions"],
            factor_sigma=float(bc["factor_sigma"]),
            noise_rate=0.0, seed=seed,
        )
        data_alerts = gen2.generate_batch(n_per_category=2000)   # 10 000 total

        aug_mu     = np.zeros((n_categories, n_actions, n_aug), dtype=np.float64)
        aug_counts = np.zeros((n_categories, n_actions),         dtype=np.float64)

        for a in data_alerts:
            c  = a.category_index
            gt = a.gt_action_index
            f  = a.factors.flatten()
            f_aug = np.empty(n_aug)
            f_aug[:n_factors] = f
            for k, (pi, pj) in enumerate(top_pairs):
                f_aug[n_factors + k] = f[pi] * f[pj]
            aug_mu[c, gt]     += f_aug
            aug_counts[c, gt] += 1

        for c in range(n_categories):
            for a in range(n_actions):
                if aug_counts[c, a] > 0:
                    aug_mu[c, a] /= aug_counts[c, a]

        # ----- augmented scorer -----
        scorer_aug = ProfileScorer(n_categories, n_actions, n_aug,
                                   tau=tau, eta=0.05, eta_neg=0.05, seed=seed)
        scorer_aug.mu = aug_mu.copy()

        # fresh oracle for augmented (noise=0 so order doesn't matter, but clean separation)
        oracle2 = GTAlignedOracle(noise_rate=0.0, seed=seed + 2000)

        correct_aug = 0
        for alert in alerts:      # reuse same alerts for fair comparison
            f = alert.factors.flatten()
            f_aug = np.empty(n_aug)
            f_aug[:n_factors] = f
            for k, (pi, pj) in enumerate(top_pairs):
                f_aug[n_factors + k] = f[pi] * f[pj]

            action_idx, _, _ = scorer_aug.score(f_aug, alert.category_index)
            if action_idx == alert.gt_action_index:
                correct_aug += 1
            result = oracle2.evaluate(actions[action_idx], alert)
            scorer_aug.update(f_aug, alert.category_index,
                              action_idx, result.outcome > 0)
        aug_acc = correct_aug / len(alerts)

        aug_results.append({
            "seed":                 seed,
            "baseline_accuracy":   base_acc,
            "augmented_accuracy":  aug_acc,
            "delta":               aug_acc - base_acc,
        })

    return aug_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    bc, realistic = _load_config()
    categories   = bc["categories"]
    actions      = bc["actions"]
    factors      = bc["factors"]
    seeds        = bc["seeds"]
    n_categories = len(categories)
    n_factors    = len(factors)
    pair_indices = [(i, j) for i in range(n_factors) for j in range(i + 1, n_factors)]
    n_pairs      = len(pair_indices)   # 15

    results_dir = ROOT / "experiments" / "expD2_factor_interactions" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    print("Phase 1: Computing mutual information...")
    mi_data        = _phase1_compute_mi(bc, realistic, seeds)
    mi_single      = mi_data["mi_single"]       # (5, 6)
    mi_interaction = mi_data["mi_interaction"]  # (5, 15)

    # Interaction gain
    interaction_gain = np.zeros((n_categories, n_pairs), dtype=np.float64)
    for c_idx in range(n_categories):
        for pair_idx, (i, j) in enumerate(pair_indices):
            mi_sum = mi_single[c_idx, i] + mi_single[c_idx, j]
            interaction_gain[c_idx, pair_idx] = (
                mi_interaction[c_idx, pair_idx] / (mi_sum + 1e-10)
            )

    # Build sorted interaction list
    all_interactions: list[dict] = []
    for c_idx, cat in enumerate(categories):
        for pair_idx, (i, j) in enumerate(pair_indices):
            mi_sum = float(mi_single[c_idx, i] + mi_single[c_idx, j])
            all_interactions.append({
                "category":        cat,
                "category_idx":    c_idx,
                "factor_i":        factors[i],
                "factor_j":        factors[j],
                "factor_i_idx":    i,
                "factor_j_idx":    j,
                "mi_interaction":  float(mi_interaction[c_idx, pair_idx]),
                "mi_sum":          mi_sum,
                "gain":            float(interaction_gain[c_idx, pair_idx]),
            })
    all_interactions.sort(key=lambda x: x["gain"], reverse=True)

    significant = [x for x in all_interactions if x["gain"] > 1.5]
    print(f"  Significant interactions (gain > 1.5): {len(significant)}")

    # -----------------------------------------------------------------------
    # Save mi_single.csv
    # -----------------------------------------------------------------------
    mi_single_rows = []
    for c_idx, cat in enumerate(categories):
        row = {"category": cat}
        for f_idx, fname in enumerate(factors):
            row[fname] = float(mi_single[c_idx, f_idx])
        mi_single_rows.append(row)
    mi_single_df = pd.DataFrame(mi_single_rows)
    mi_single_df.to_csv(results_dir / "mi_single.csv", index=False)
    print(f"Saved: mi_single.csv")

    # -----------------------------------------------------------------------
    # Save mi_interaction.csv
    # -----------------------------------------------------------------------
    mi_inter_rows = []
    for c_idx, cat in enumerate(categories):
        for pair_idx, (i, j) in enumerate(pair_indices):
            mi_sum = float(mi_single[c_idx, i] + mi_single[c_idx, j])
            mi_inter_rows.append({
                "category":        cat,
                "factor_i":        factors[i],
                "factor_j":        factors[j],
                "mi_interaction":  float(mi_interaction[c_idx, pair_idx]),
                "mi_sum":          mi_sum,
                "interaction_gain": float(interaction_gain[c_idx, pair_idx]),
            })
    mi_inter_df = pd.DataFrame(mi_inter_rows)
    mi_inter_df.to_csv(results_dir / "mi_interaction.csv", index=False)
    print(f"Saved: mi_interaction.csv")

    # -----------------------------------------------------------------------
    # Save top_interactions.json
    # -----------------------------------------------------------------------
    top_json = [
        {**x, "significant": x["gain"] > 1.5}
        for x in all_interactions
    ]
    with open(results_dir / "top_interactions.json", "w") as fh:
        json.dump(top_json, fh, indent=2)
    print(f"Saved: top_interactions.json")

    # -----------------------------------------------------------------------
    # Phase 2: Augmentation (conditional)
    # -----------------------------------------------------------------------
    augmentation_ran  = False
    aug_results_list: list[dict] = []
    top_pairs: list[tuple[int, int]] = []
    mean_base = mean_aug = aug_lift = 0.0

    if len(significant) == 0:
        print("\nNo significant interactions. Factors are independently informative.")
        print("Skipping augmentation test.")
    else:
        augmentation_ran = True
        # Top-3 unique factor pairs with gain > 1.5
        seen_pairs: set = set()
        for inter in all_interactions:
            pk = (inter["factor_i_idx"], inter["factor_j_idx"])
            if pk not in seen_pairs and inter["gain"] > 1.5:
                seen_pairs.add(pk)
                top_pairs.append(pk)
                if len(top_pairs) >= 3:
                    break

        print(f"\nPhase 2: Augmentation test with top pairs: "
              f"{[(factors[i], factors[j]) for i,j in top_pairs]}")

        aug_results_list = _phase2_augmentation(top_pairs, bc, realistic, seeds)

        mean_base = float(np.mean([r["baseline_accuracy"]  for r in aug_results_list]))
        mean_aug  = float(np.mean([r["augmented_accuracy"] for r in aug_results_list]))
        aug_lift  = mean_aug - mean_base

        aug_df = pd.DataFrame(aug_results_list)
        aug_df.to_csv(results_dir / "augmentation_results.csv", index=False)
        print(f"Saved: augmentation_results.csv")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary = {
        "n_significant_interactions": len(significant),
        "top_5_interactions": [
            {k: v for k, v in x.items() if k != "category_idx" and k != "factor_i_idx" and k != "factor_j_idx"}
            for x in all_interactions[:5]
        ],
        "augmentation_ran":  augmentation_ran,
        "augmentation_lift": float(aug_lift),
        "mi_single_matrix": {
            cat: {factors[f_idx]: float(mi_single[c_idx, f_idx]) for f_idx in range(n_factors)}
            for c_idx, cat in enumerate(categories)
        },
        "mean_baseline_accuracy":  float(mean_base),
        "mean_augmented_accuracy": float(mean_aug),
    }
    with open(results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: summary.json")

    # -----------------------------------------------------------------------
    # Validation report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXP-D2: FACTOR INTERACTION DISCOVERY")
    print("=" * 60)

    # VD2.1: Single-factor MI table
    print("\nVD2.1: Single-Factor MI (averaged across seeds)")
    hdr = f"  {'Category':25s}" + "".join(f"{f:>16s}" for f in factors)
    print(hdr)
    for c_idx, cat in enumerate(categories):
        vals  = mi_single[c_idx]
        best  = int(np.argmax(vals))
        cells = []
        for f_idx, v in enumerate(vals):
            mark = " *" if f_idx == best else "  "
            cells.append(f"{v:>14.4f}{mark}")
        print(f"  {cat:25s}" + "".join(cells))

    # VD2.2: Interaction summary
    n_gt20 = len([x for x in all_interactions if x["gain"] > 2.0])
    print(f"\nVD2.2: Interaction Summary")
    print(f"  Total interactions analyzed:      {n_categories * n_pairs}")
    print(f"  Interactions with gain > 1.5:     {len(significant)}")
    print(f"  Interactions with gain > 2.0:     {n_gt20}")

    # VD2.3: Top-10 interactions
    print("\nVD2.3: Top-10 Interactions")
    for inter in all_interactions[:10]:
        sig_flag = "  *** SIGNIFICANT" if inter["gain"] > 1.5 else ""
        print(f"  {inter['category']:25s}  {inter['factor_i']:20s} x {inter['factor_j']:20s}"
              f"  MI={inter['mi_interaction']:.4f}  sum={inter['mi_sum']:.4f}"
              f"  gain={inter['gain']:.3f}{sig_flag}")

    # VD2.4: Augmentation
    n_aug_feat = n_factors + len(top_pairs) if augmentation_ran else n_factors
    if augmentation_ran:
        print(f"\nVD2.4: Augmentation Results")
        print(f"  Top pairs used: {[(factors[i], factors[j]) for i,j in top_pairs]}")
        print(f"  Baseline ({n_factors} factors):      {mean_base:.1%}")
        print(f"  Augmented ({n_aug_feat} features): {mean_aug:.1%}")
        print(f"  Lift: {aug_lift*100:+.2f}pp")
        if aug_lift >= 0.01:
            print("  >>> PASS: Factor interactions improve scoring.")
        elif aug_lift >= 0.0:
            print("  >>> MARGINAL: Interactions exist but lift is < 1pp at 98% baseline.")
        else:
            print("  >>> Augmentation HURTS. Interaction features add noise.")
    else:
        print("\nVD2.4: Augmentation not run (no significant interactions).")

    print(f"\n  OVERALL ASSESSMENT:")
    if len(significant) == 0:
        print("  Factors are independently informative. No interaction structure.")
        print("  This is GOOD -- the factor space is well-designed.")
    elif augmentation_ran and aug_lift < 0.01:
        print("  Interactions exist statistically but don't improve the 98% baseline.")
        print("  The profile centroids already capture sufficient structure.")
    elif augmentation_ran and aug_lift >= 0.01:
        print("  Interactions provide meaningful lift. Consider augmenting profiles.")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    from src.viz.expD2_charts import generate_all_charts
    generate_all_charts(
        str(results_dir),
        augmentation_ran=augmentation_ran,
        top_pairs=top_pairs,
        factors=factors,
        mean_base=mean_base,
        mean_aug=mean_aug,
        aug_lift=aug_lift,
    )


if __name__ == "__main__":
    run()
