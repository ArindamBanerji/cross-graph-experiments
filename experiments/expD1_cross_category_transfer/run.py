"""
EXP-D1: Cross-Category Transfer.

Tests whether one category's learned profiles can accelerate another
category's cold-start learning (meta-graph cross-category transfer).

Phase 1: Learn source profiles (5 categories × 10 seeds × 500 decisions)
Phase 2: Compute 5×5 transfer similarity matrix
Phase 3: Test transfer initialization (5 targets × 3 conditions × 10 seeds × 200 decisions)

Generates
---------
experiments/expD1_cross_category_transfer/results/transfer_matrix.csv
experiments/expD1_cross_category_transfer/results/accuracy_trajectories.csv
experiments/expD1_cross_category_transfer/results/convergence_speed.csv
experiments/expD1_cross_category_transfer/results/summary.json
paper_figures/expD1_*.{pdf,png}
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
from src.models.oracle import GTAlignedOracle
from src.models.profile_scorer import ProfileScorer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINTS_PHASE3 = [25, 50, 100, 150, 200]
N_SOURCE_DECISIONS = 500
N_TARGET_DECISIONS = 200
CONDITIONS         = ["cold", "config", "transfer"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw["realistic_profiles"]


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def _get_category_alerts(seed: int, cat_idx: int, n: int, bc: dict, realistic: dict) -> list:
    """Get exactly n alerts for cat_idx via balanced generate_batch."""
    gen = CategoryAlertGenerator(
        categories=bc["categories"],
        actions=bc["actions"],
        factors=bc["factors"],
        action_conditional_profiles=realistic["action_conditional_profiles"],
        gt_distributions=realistic["category_gt_distributions"],
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=0.0,
        seed=seed,
    )
    batch = gen.generate_batch(n_per_category=n)
    return [a for a in batch if a.category_index == cat_idx]


# ---------------------------------------------------------------------------
# Phase 1: Learn source profiles
# ---------------------------------------------------------------------------

def _phase1_learn_sources(
    bc: dict,
    realistic: dict,
    seeds: list,
    tau: float,
    eta: float,
    eta_neg: float,
) -> dict:
    """
    For each (c_source, seed): warm-start ProfileScorer, feed 500 source alerts,
    record learned mu snapshot.

    Returns
    -------
    dict[c_src_idx][seed] -> np.ndarray shape (5, 4, 6)
    """
    categories   = bc["categories"]
    actions      = bc["actions"]
    n_categories = len(categories)
    n_actions    = len(actions)
    n_factors    = int(bc["n_factors"])

    snapshots: dict = {c: {} for c in range(n_categories)}

    for c_src_idx, c_src_name in enumerate(categories):
        for seed in seeds:
            scorer = ProfileScorer(
                n_categories, n_actions, n_factors,
                tau=tau, eta=eta, eta_neg=eta_neg, seed=seed,
            )
            scorer.init_from_profiles(
                realistic["action_conditional_profiles"], categories, actions,
            )

            alerts = _get_category_alerts(seed, c_src_idx, N_SOURCE_DECISIONS, bc, realistic)
            # Shuffle so action ordering doesn't bias early learning
            rng_shuffle = np.random.default_rng(seed + 3000)
            rng_shuffle.shuffle(alerts)

            oracle = GTAlignedOracle(noise_rate=0.0, seed=seed + 1000)

            for alert in alerts:
                action_idx, _, _ = scorer.score(alert.factors, alert.category_index)
                result = oracle.evaluate(actions[action_idx], alert)
                scorer.update(alert.factors, alert.category_index,
                              action_idx, result.outcome > 0)

            snapshots[c_src_idx][seed] = scorer.get_profile_snapshot()

        print(f"  Phase 1: source '{c_src_name}' done ({len(seeds)} seeds)")

    return snapshots


# ---------------------------------------------------------------------------
# Phase 2: Compute transfer matrix
# ---------------------------------------------------------------------------

def _phase2_transfer_matrix(
    snapshots: dict,
    bc: dict,
    realistic: dict,
    seeds: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 5×5 transfer_score matrix.
    transfer_score[src, tgt] = mean cosine sim between learned(src, a) and config(tgt, a)
    over all 4 actions.

    Returns
    -------
    transfer_score : np.ndarray (5, 5)
    mean_source_mu : np.ndarray (5, 5, 4, 6) — [c_src, category, action, factor]
    """
    categories = bc["categories"]
    actions    = bc["actions"]
    n_cats     = len(categories)
    n_acts     = len(actions)

    # Config centroid matrix: config_mu[c, a, :] = configured profile
    config_mu = np.zeros((n_cats, n_acts, 6), dtype=np.float64)
    for c_idx, cat in enumerate(categories):
        for a_idx, act in enumerate(actions):
            config_mu[c_idx, a_idx, :] = np.array(
                realistic["action_conditional_profiles"][cat][act],
                dtype=np.float64,
            )

    # Mean learned mu per source (averaged over seeds)
    mean_source_mu = np.zeros((n_cats, n_cats, n_acts, 6), dtype=np.float64)
    for c_src in range(n_cats):
        stack = np.stack([snapshots[c_src][seed] for seed in seeds], axis=0)  # (10, 5, 4, 6)
        mean_source_mu[c_src] = stack.mean(axis=0)  # (5, 4, 6)

    transfer_score = np.zeros((n_cats, n_cats), dtype=np.float64)
    for c_src in range(n_cats):
        for c_tgt in range(n_cats):
            if c_src == c_tgt:
                continue
            sims = []
            for a_idx in range(n_acts):
                learned_vec = mean_source_mu[c_src][c_src, a_idx, :]   # learned for c_src
                config_vec  = config_mu[c_tgt, a_idx, :]               # configured for c_tgt
                cos_sim = (
                    np.dot(learned_vec, config_vec) /
                    (np.linalg.norm(learned_vec) * np.linalg.norm(config_vec) + 1e-10)
                )
                sims.append(float(cos_sim))
            transfer_score[c_src, c_tgt] = float(np.mean(sims))

    return transfer_score, mean_source_mu


# ---------------------------------------------------------------------------
# Phase 3: Test transfer initialization
# ---------------------------------------------------------------------------

def _phase3_run_one(
    condition: str,
    c_tgt_idx: int,
    c_best_src_idx: int,
    seed: int,
    mean_source_mu: np.ndarray,
    bc: dict,
    realistic: dict,
    tau: float,
    eta: float,
    eta_neg: float,
) -> list[dict]:
    categories = bc["categories"]
    actions    = bc["actions"]
    n_categories = len(categories)
    n_actions    = len(actions)
    n_factors    = int(bc["n_factors"])

    scorer = ProfileScorer(
        n_categories, n_actions, n_factors,
        tau=tau, eta=eta, eta_neg=eta_neg, seed=seed,
    )

    if condition == "cold":
        rng_sym = np.random.default_rng(seed + 20000)
        scorer.mu += rng_sym.uniform(-0.01, 0.01, scorer.mu.shape)
        np.clip(scorer.mu, 0.0, 1.0, out=scorer.mu)

    elif condition == "config":
        scorer.init_from_profiles(
            realistic["action_conditional_profiles"], categories, actions,
        )

    elif condition == "transfer":
        scorer.init_from_profiles(
            realistic["action_conditional_profiles"], categories, actions,
        )
        # Replace target's profiles with best source's learned profiles
        scorer.mu[c_tgt_idx, :, :] = mean_source_mu[c_best_src_idx, c_best_src_idx, :, :]
        np.clip(scorer.mu[c_tgt_idx], 0.0, 1.0, out=scorer.mu[c_tgt_idx])

    # Generate target-category-only alerts (phase 3 seed offset = +5000)
    alerts = _get_category_alerts(seed + 5000, c_tgt_idx, N_TARGET_DECISIONS, bc, realistic)
    rng_shuffle = np.random.default_rng(seed + 7000)
    rng_shuffle.shuffle(alerts)

    oracle = GTAlignedOracle(noise_rate=0.0, seed=seed + 1000)

    cp_set   = set(CHECKPOINTS_PHASE3)
    n_correct = 0
    rows: list[dict] = []

    for t, alert in enumerate(alerts):
        t1 = t + 1
        action_idx, _, _ = scorer.score(alert.factors, alert.category_index)
        gt_correct        = (action_idx == alert.gt_action_index)
        n_correct        += int(gt_correct)

        result = oracle.evaluate(actions[action_idx], alert)
        scorer.update(alert.factors, alert.category_index,
                      action_idx, result.outcome > 0)

        if t1 in cp_set:
            rows.append({
                "target_category":      categories[c_tgt_idx],
                "condition":            condition,
                "seed":                 seed,
                "checkpoint":           t1,
                "cumulative_gt_accuracy": n_correct / t1,
                "best_source_category": categories[c_best_src_idx],
            })

    return rows


# ---------------------------------------------------------------------------
# Convergence speed helper
# ---------------------------------------------------------------------------

def _convergence_speed(df: pd.DataFrame, categories: list, seeds: list) -> pd.DataFrame:
    thresholds = [
        ("decisions_to_80pct", 0.80),
        ("decisions_to_90pct", 0.90),
        ("decisions_to_95pct", 0.95),
    ]
    rows = []
    for cat in categories:
        for cond in CONDITIONS:
            for seed in seeds:
                sub = df[
                    (df["target_category"] == cat) &
                    (df["condition"]        == cond) &
                    (df["seed"]             == seed)
                ].sort_values("checkpoint")

                row = {"target_category": cat, "condition": cond, "seed": seed}
                for col, thr in thresholds:
                    reached = sub[sub["cumulative_gt_accuracy"] >= thr]
                    row[col] = int(reached["checkpoint"].min()) if not reached.empty else np.nan
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    bc, realistic = _load_config()
    categories = bc["categories"]
    actions    = bc["actions"]
    seeds      = bc["seeds"]
    tau        = float(bc["scoring"]["temperature"])
    eta        = 0.05
    eta_neg    = 0.05

    results_dir = ROOT / "experiments" / "expD1_cross_category_transfer" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    print("Phase 1: Learning source profiles...")
    snapshots = _phase1_learn_sources(bc, realistic, seeds, tau, eta, eta_neg)

    # -----------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------
    print("\nPhase 2: Computing transfer matrix...")
    transfer_score, mean_source_mu = _phase2_transfer_matrix(snapshots, bc, realistic, seeds)

    # Best source per target
    best_source_per_target = {}
    for c_tgt in range(len(categories)):
        # Exclude self
        scores_no_self = transfer_score[:, c_tgt].copy()
        scores_no_self[c_tgt] = -np.inf
        best_source_per_target[c_tgt] = int(np.argmax(scores_no_self))

    # Save transfer_matrix.csv
    tm_df = pd.DataFrame(
        transfer_score,
        index=categories,
        columns=categories,
    )
    tm_path = results_dir / "transfer_matrix.csv"
    tm_df.to_csv(tm_path)
    print(f"Saved: {tm_path}")

    # -----------------------------------------------------------------------
    # Phase 3
    # -----------------------------------------------------------------------
    print("\nPhase 3: Testing transfer initialization...")
    all_rows: list[dict] = []

    for c_tgt_idx, c_tgt_name in enumerate(categories):
        c_best = best_source_per_target[c_tgt_idx]
        for condition in CONDITIONS:
            for seed in seeds:
                rows = _phase3_run_one(
                    condition, c_tgt_idx, c_best,
                    seed, mean_source_mu,
                    bc, realistic, tau, eta, eta_neg,
                )
                all_rows.extend(rows)
        print(f"  Phase 3: target '{c_tgt_name}' done ({len(CONDITIONS)} conditions × {len(seeds)} seeds)")

    # -----------------------------------------------------------------------
    # Save accuracy_trajectories.csv
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = results_dir / "accuracy_trajectories.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Convergence speed
    # -----------------------------------------------------------------------
    speed_df  = _convergence_speed(df, categories, seeds)
    speed_path = results_dir / "convergence_speed.csv"
    speed_df.to_csv(speed_path, index=False)
    print(f"Saved: {speed_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    per_cond: dict = {}
    for cat in categories:
        per_cond[cat] = {}
        for cond in CONDITIONS:
            sub = df[
                (df["target_category"] == cat) &
                (df["condition"]        == cond) &
                (df["checkpoint"]       == 200)
            ]
            per_cond[cat][cond] = {
                "mean": float(sub["cumulative_gt_accuracy"].mean()) if not sub.empty else 0.0,
                "std":  float(sub["cumulative_gt_accuracy"].std())  if not sub.empty else 0.0,
            }

    transfer_vs_config = {
        cat: per_cond[cat]["transfer"]["mean"] - per_cond[cat]["config"]["mean"]
        for cat in categories
    }
    transfer_vs_cold = {
        cat: per_cond[cat]["transfer"]["mean"] - per_cond[cat]["cold"]["mean"]
        for cat in categories
    }

    summary = {
        "transfer_matrix": {
            categories[src]: {categories[tgt]: float(transfer_score[src, tgt]) for tgt in range(5)}
            for src in range(5)
        },
        "best_source_per_target": {
            categories[tgt]: categories[best_source_per_target[tgt]] for tgt in range(5)
        },
        "per_condition_accuracy_t200": per_cond,
        "transfer_vs_config_delta": transfer_vs_config,
        "transfer_vs_cold_delta": transfer_vs_cold,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {summary_path}")

    # -----------------------------------------------------------------------
    # Validation report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXP-D1: CROSS-CATEGORY TRANSFER")
    print("=" * 60)

    print("\nVD1.1: Transfer Matrix (source=row, target=col)")
    header = f"{'':22s}" + "  ".join(f"{c[:8]:>8s}" for c in categories)
    print(f"  {header}")
    for src_idx, src in enumerate(categories):
        row_str = "  ".join(
            f"{'---':>8s}" if src_idx == tgt_idx
            else f"{transfer_score[src_idx, tgt_idx]:8.3f}"
            for tgt_idx in range(len(categories))
        )
        print(f"  {src:<22s}{row_str}")

    print("\nVD1.2: Best Source per Target")
    for c_tgt_idx, cat in enumerate(categories):
        c_best = best_source_per_target[c_tgt_idx]
        score  = float(transfer_score[c_best, c_tgt_idx])
        print(f"  {cat} <- {categories[c_best]} (score={score:.3f})")

    print("\nVD1.3: Accuracy at t=200 by Condition")
    for cat in categories:
        cold_m     = per_cond[cat]["cold"]["mean"]
        cold_s     = per_cond[cat]["cold"]["std"]
        config_m   = per_cond[cat]["config"]["mean"]
        config_s   = per_cond[cat]["config"]["std"]
        transfer_m = per_cond[cat]["transfer"]["mean"]
        transfer_s = per_cond[cat]["transfer"]["std"]
        print(f"  {cat}:")
        print(f"    cold:     {cold_m:.1%} +/- {cold_s:.1%}")
        print(f"    config:   {config_m:.1%} +/- {config_s:.1%}")
        print(f"    transfer: {transfer_m:.1%} +/- {transfer_s:.1%}")

    print("\nVD1.4: Transfer Assessment")
    n_competitive = sum(
        1 for cat in categories
        if per_cond[cat]["transfer"]["mean"] >= per_cond[cat]["config"]["mean"] - 0.02
    )
    n_beats_cold = sum(
        1 for cat in categories
        if per_cond[cat]["transfer"]["mean"] > per_cond[cat]["cold"]["mean"] + 0.10
    )
    print(f"  Transfer competitive with config (>= config - 2pp): {n_competitive}/5 categories")
    print(f"  Transfer beats cold by >10pp:                        {n_beats_cold}/5 categories")

    if n_competitive >= 3:
        print("  >>> PASS: Transfer initialization is competitive with config.")
    else:
        print("  >>> Transfer is NOT competitive with config (config profiles are already good).")

    if n_beats_cold >= 4:
        print("  >>> TRANSFER OVER COLD: Transfer substantially beats cold start.")
        print("  >>> Useful when config profiles are unavailable for a new category.")

    print("\n  INTERPRETATION:")
    print("  Transfer is most valuable when:")
    print("    - A new category appears with no expert profiles")
    print("    - A similar category's operational data can bootstrap it")
    print("  Transfer is redundant when:")
    print("    - Expert profiles are available (config init ~98%)")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    from src.viz.expD1_charts import generate_all_charts
    generate_all_charts(str(results_dir))


if __name__ == "__main__":
    run()
