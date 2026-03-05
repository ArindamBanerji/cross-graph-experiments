"""
EXP-E2: Scale Test.

Does profile-based L2 scoring generalise beyond 5x4x6?
Tests 4 scale configurations (small / medium / large / xlarge).

Phase 1 : centroid oracle accuracy at each scale.
Phase 2 : warm-start and cold-start learning convergence.
Phase 3 : profile-separation analysis.

Generates
---------
experiments/expE2_scale_test/results/phase1_oracle.csv
experiments/expE2_scale_test/results/phase2_learning.csv
experiments/expE2_scale_test/results/phase3_separation.csv
experiments/expE2_scale_test/results/summary.json
paper_figures/expE2_*.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator


# ---------------------------------------------------------------------------
# Scale configurations
# ---------------------------------------------------------------------------

SCALE_CONFIGS = {
    "small":  dict(n_categories=5,  n_actions=4,  n_factors=6),
    "medium": dict(n_categories=10, n_actions=6,  n_factors=10),
    "large":  dict(n_categories=15, n_actions=8,  n_factors=15),
    "xlarge": dict(n_categories=20, n_actions=10, n_factors=20),
}

CHECKPOINT_PCTS = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw["realistic_profiles"]


# ---------------------------------------------------------------------------
# Synthetic profile + alert generation (medium / large / xlarge)
# ---------------------------------------------------------------------------

def generate_synthetic_profiles(
    n_categories: int,
    n_actions: int,
    n_factors: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic action-conditional profiles.

    Returns
    -------
    profiles     : (n_categories, n_actions, n_factors), values in [0.05, 0.95]
    gt_dists     : (n_categories, n_actions), sums to 1 per category
    """
    rng = np.random.default_rng(seed)

    profiles     = np.zeros((n_categories, n_actions, n_factors))
    gt_dists     = np.zeros((n_categories, n_actions))

    for c in range(n_categories):
        mode_action = c % n_actions
        for a in range(n_actions):
            if a == mode_action:
                gt_dists[c, a] = 0.40
            else:
                gt_dists[c, a] = 0.60 / (n_actions - 1)

        for a in range(n_actions):
            base = rng.uniform(0.2, 0.8, size=n_factors)

            n_dominant = min(3, n_factors)
            dominant_indices = [(a * 2 + d) % n_factors for d in range(n_dominant)]
            for d_idx in dominant_indices:
                base[d_idx] = rng.uniform(0.70, 0.95)

            cat_shift_indices = [(c * 3 + s) % n_factors for s in range(2)]
            for s_idx in cat_shift_indices:
                base[s_idx] += rng.uniform(-0.15, 0.15)

            profiles[c, a, :] = np.clip(base, 0.05, 0.95)

    return profiles, gt_dists


def generate_synthetic_alerts(
    profiles: np.ndarray,
    gt_distributions: np.ndarray,
    n_alerts: int,
    factor_sigma: float,
    noise_rate: float = 0.0,
    seed: int = 42,
) -> list:
    rng = np.random.default_rng(seed)
    n_categories, n_actions, _ = profiles.shape
    alerts = []

    for _ in range(n_alerts):
        c = int(rng.integers(0, n_categories))
        a = int(rng.choice(n_actions, p=gt_distributions[c]))
        f = rng.normal(profiles[c, a, :], factor_sigma)
        f = np.clip(f, 0.0, 1.0)
        if noise_rate > 0 and rng.random() < noise_rate:
            a = int(rng.integers(0, n_actions))
        alerts.append(SimpleNamespace(
            category_index=c,
            gt_action_index=a,
            factors=f,
        ))
    return alerts


# ---------------------------------------------------------------------------
# L2 oracle score
# ---------------------------------------------------------------------------

def l2_predict(f: np.ndarray, mu_c: np.ndarray) -> int:
    """mu_c: (n_actions, n_factors). Returns argmin distance."""
    diffs = mu_c - f          # (n_actions, n_factors)
    dists = np.einsum("af,af->a", diffs, diffs)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Phase 1: Centroid oracle
# ---------------------------------------------------------------------------

def run_phase1(
    bc: dict,
    realistic: dict,
    seeds: list,
    factor_sigma: float,
) -> list[dict]:
    rows: list[dict] = []

    for config_name, dims in SCALE_CONFIGS.items():
        nc = dims["n_categories"]
        na = dims["n_actions"]
        nf = dims["n_factors"]
        n_alerts = 5000 if config_name == "xlarge" else 10000

        # Build centroid matrix
        if config_name == "small":
            profiles_raw = realistic["action_conditional_profiles"]
            categories   = bc["categories"]
            actions      = bc["actions"]
            mu = np.zeros((nc, na, nf), dtype=np.float64)
            for c, cat in enumerate(categories):
                for a, act in enumerate(actions):
                    mu[c, a, :] = np.array(profiles_raw[cat][act], dtype=np.float64)
        else:
            mu, gt_dists = generate_synthetic_profiles(nc, na, nf, seed=0)

        print(f"\n  [{config_name}] {nc}x{na}x{nf}  ({nc*na} centroids, {nc*na*nf} params)")

        for seed in seeds:
            if config_name == "small":
                categories = bc["categories"]
                actions    = bc["actions"]
                gen = CategoryAlertGenerator(
                    categories=categories,
                    actions=actions,
                    factors=bc["factors"],
                    action_conditional_profiles=realistic["action_conditional_profiles"],
                    gt_distributions=realistic["category_gt_distributions"],
                    factor_sigma=factor_sigma,
                    noise_rate=0.0,
                    seed=seed,
                )
                per_cat = n_alerts // nc
                alerts = gen.generate_batch(n_per_category=per_cat)
            else:
                alerts = generate_synthetic_alerts(
                    mu, gt_dists, n_alerts, factor_sigma, seed=seed)

            correct = 0
            for alert in alerts:
                f   = alert.factors
                c   = alert.category_index
                pred = l2_predict(f, mu[c])
                if pred == alert.gt_action_index:
                    correct += 1

            oracle_acc = correct / len(alerts)
            print(f"    seed={seed}: {oracle_acc:.1%}")

            rows.append({
                "scale_config":    config_name,
                "seed":            seed,
                "oracle_accuracy": oracle_acc,
                "n_categories":    nc,
                "n_actions":       na,
                "n_factors":       nf,
                "total_centroids": nc * na,
                "total_params":    nc * na * nf,
            })

    return rows


# ---------------------------------------------------------------------------
# Phase 2: Learning convergence (warm + cold)
# ---------------------------------------------------------------------------

def run_phase2(
    bc: dict,
    realistic: dict,
    seeds: list,
    factor_sigma: float,
) -> list[dict]:
    rows: list[dict] = []

    for config_name, dims in SCALE_CONFIGS.items():
        nc = dims["n_categories"]
        na = dims["n_actions"]
        nf = dims["n_factors"]

        # Build reference centroid matrix
        if config_name == "small":
            categories  = bc["categories"]
            actions     = bc["actions"]
            profiles_raw = realistic["action_conditional_profiles"]
            mu_ref = np.zeros((nc, na, nf), dtype=np.float64)
            for c, cat in enumerate(categories):
                for a, act in enumerate(actions):
                    mu_ref[c, a, :] = np.array(profiles_raw[cat][act], dtype=np.float64)
            gt_dists = None   # use CategoryAlertGenerator
        else:
            mu_ref, gt_dists = generate_synthetic_profiles(nc, na, nf, seed=0)

        n_decisions_raw = nc * na * 50      # ~50 decisions per centroid
        n_decisions = min(max(n_decisions_raw, 1000), 10000)
        if config_name == "xlarge":
            n_decisions = min(n_decisions, 5000)

        print(f"\n  [{config_name}] n_decisions={n_decisions}")

        for condition in ("warm", "cold"):
            for seed in seeds:
                if config_name == "small":
                    gen = CategoryAlertGenerator(
                        categories=bc["categories"],
                        actions=bc["actions"],
                        factors=bc["factors"],
                        action_conditional_profiles=realistic["action_conditional_profiles"],
                        gt_distributions=realistic["category_gt_distributions"],
                        factor_sigma=factor_sigma,
                        noise_rate=0.0,
                        seed=seed,
                    )
                    alerts = gen.generate(n=n_decisions)
                else:
                    alerts = generate_synthetic_alerts(
                        mu_ref, gt_dists, n_decisions, factor_sigma, seed=seed)

                if condition == "warm":
                    mu_local = mu_ref.copy()
                else:
                    mu_local = np.full_like(mu_ref, 0.5)

                counts = np.zeros((nc, na), dtype=np.float64)
                gt_correct_history: list[bool] = []

                for t, alert in enumerate(alerts):
                    f    = alert.factors
                    c    = alert.category_index
                    gt   = alert.gt_action_index
                    pred = l2_predict(f, mu_local[c])
                    gt_correct = (pred == gt)
                    gt_correct_history.append(gt_correct)

                    eta = 0.05 / (1.0 + counts[c, pred] * 0.001)
                    if gt_correct:
                        mu_local[c, pred, :] += eta * (f - mu_local[c, pred, :])
                    else:
                        mu_local[c, pred, :] -= eta * (f - mu_local[c, pred, :])
                    mu_local[c, pred, :] = np.clip(mu_local[c, pred, :], 0.0, 1.0)
                    counts[c, pred] += 1

                for pct in CHECKPOINT_PCTS:
                    t_check = int(pct * n_decisions)
                    if t_check < 1:
                        t_check = 1
                    acc = sum(gt_correct_history[:t_check]) / t_check
                    rows.append({
                        "scale_config":        config_name,
                        "condition":           condition,
                        "seed":                seed,
                        "checkpoint_pct":      pct,
                        "checkpoint_t":        t_check,
                        "cumulative_accuracy": acc,
                        "n_decisions":         n_decisions,
                    })

    return rows


# ---------------------------------------------------------------------------
# Phase 3: Profile separation
# ---------------------------------------------------------------------------

def run_phase3(bc: dict, realistic: dict) -> list[dict]:
    rows: list[dict] = []

    for config_name, dims in SCALE_CONFIGS.items():
        nc = dims["n_categories"]
        na = dims["n_actions"]
        nf = dims["n_factors"]

        if config_name == "small":
            categories   = bc["categories"]
            actions      = bc["actions"]
            profiles_raw = realistic["action_conditional_profiles"]
            mu = np.zeros((nc, na, nf), dtype=np.float64)
            for c, cat in enumerate(categories):
                for a, act in enumerate(actions):
                    mu[c, a, :] = np.array(profiles_raw[cat][act], dtype=np.float64)
        else:
            mu, _ = generate_synthetic_profiles(nc, na, nf, seed=0)

        for c in range(nc):
            pairwise = []
            for a1 in range(na):
                for a2 in range(a1 + 1, na):
                    dist = float(np.linalg.norm(mu[c, a1, :] - mu[c, a2, :]))
                    pairwise.append(dist)
            mean_sep = float(np.mean(pairwise)) if pairwise else 0.0
            min_sep  = float(np.min(pairwise))  if pairwise else 0.0
            rows.append({
                "scale_config":    config_name,
                "category_index":  c,
                "mean_separation": mean_sep,
                "min_separation":  min_sep,
            })

    return rows


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    oracle_rows: list[dict],
    learning_rows: list[dict],
    sep_rows: list[dict],
) -> dict:
    df1 = pd.DataFrame(oracle_rows)
    df2 = pd.DataFrame(learning_rows)
    df3 = pd.DataFrame(sep_rows)

    summary: dict = {}

    for config_name, dims in SCALE_CONFIGS.items():
        nc = dims["n_categories"]
        na = dims["n_actions"]
        nf = dims["n_factors"]

        # Oracle
        sub1 = df1[df1["scale_config"] == config_name]
        oracle_mean = float(sub1["oracle_accuracy"].mean())
        oracle_std  = float(sub1["oracle_accuracy"].std())

        # Learning — warm @ t=100%
        sub_warm = df2[(df2["scale_config"] == config_name) &
                       (df2["condition"] == "warm") &
                       (df2["checkpoint_pct"] == 1.00)]
        warm_mean = float(sub_warm["cumulative_accuracy"].mean()) if not sub_warm.empty else 0.0
        warm_std  = float(sub_warm["cumulative_accuracy"].std())  if not sub_warm.empty else 0.0

        # Learning — cold @ t=100%
        sub_cold = df2[(df2["scale_config"] == config_name) &
                       (df2["condition"] == "cold") &
                       (df2["checkpoint_pct"] == 1.00)]
        cold_mean = float(sub_cold["cumulative_accuracy"].mean()) if not sub_cold.empty else 0.0
        cold_std  = float(sub_cold["cumulative_accuracy"].std())  if not sub_cold.empty else 0.0

        n_decisions = int(sub_warm["n_decisions"].iloc[0]) if not sub_warm.empty else 0

        # Separation
        sub3 = df3[df3["scale_config"] == config_name]
        mean_sep = float(sub3["mean_separation"].mean()) if not sub3.empty else 0.0
        min_sep  = float(sub3["min_separation"].min())   if not sub3.empty else 0.0

        summary[config_name] = {
            "oracle_accuracy":   {"mean": oracle_mean, "std": oracle_std},
            "warm_t100pct":      {"mean": warm_mean,   "std": warm_std},
            "cold_t100pct":      {"mean": cold_mean,   "std": cold_std},
            "n_categories":      nc,
            "n_actions":         na,
            "n_factors":         nf,
            "total_centroids":   nc * na,
            "total_params":      nc * na * nf,
            "n_decisions":       n_decisions,
            "mean_separation":   mean_sep,
            "min_separation":    min_sep,
        }

    # Scaling trend
    configs = list(SCALE_CONFIGS.keys())
    oracle_accs = [summary[c]["oracle_accuracy"]["mean"] for c in configs]
    small_acc   = oracle_accs[0]
    xlarge_acc  = oracle_accs[-1]
    summary["scaling_trend"] = {
        "small_oracle":  small_acc,
        "xlarge_oracle": xlarge_acc,
        "degradation_pp": (small_acc - xlarge_acc) * 100,
    }

    return summary


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _print_report(summary: dict) -> None:
    configs = list(SCALE_CONFIGS.keys())

    print("\n" + "=" * 70)
    print("EXP-E2: SCALE TEST -- PROFILE SCORING AT HIGHER DIMENSIONS")
    print("=" * 70)

    print("\n--- PHASE 1: CENTROID ORACLE ---\n")
    print(f"  {'Config':<10s} {'Cat':>4s} {'Act':>4s} {'Fac':>4s} "
          f"{'Centroids':>10s} {'Params':>7s} {'Oracle Acc':>11s} {'+/-std':>7s}")
    print(f"  {'-'*64}")
    for c in configs:
        s = summary[c]
        nc = s["n_categories"]
        na = s["n_actions"]
        nf = s["n_factors"]
        print(f"  {c:<10s} {nc:>4d} {na:>4d} {nf:>4d} "
              f"{nc*na:>10d} {nc*na*nf:>7d} "
              f"{s['oracle_accuracy']['mean']:>10.1%} "
              f"{s['oracle_accuracy']['std']:>6.1%}")

    print("\n--- PHASE 2: LEARNING CONVERGENCE (warm start) ---\n")
    for c in configs:
        s = summary[c]
        print(f"  {c}: {s['warm_t100pct']['mean']:.1%} at t={s['n_decisions']}")

    print("\n--- PHASE 2b: COLD START RECOVERY ---\n")
    for c in configs:
        s = summary[c]
        print(f"  {c}: {s['cold_t100pct']['mean']:.1%} at t={s['n_decisions']}")

    print("\n--- PHASE 3: PROFILE SEPARATION ---\n")
    for c in configs:
        s = summary[c]
        print(f"  {c}: mean sep={s['mean_separation']:.3f}, "
              f"min sep={s['min_separation']:.3f}")

    print("\n--- SCALING ASSESSMENT ---\n")

    trend = summary["scaling_trend"]
    small_acc  = trend["small_oracle"]
    xlarge_acc = trend["xlarge_oracle"]
    degradation = small_acc - xlarge_acc

    print(f"  Oracle accuracy: small={small_acc:.1%} -> xlarge={xlarge_acc:.1%}")
    print(f"  Degradation: {degradation*100:.1f}pp")

    if degradation < 0.05:
        print("  >>> PASS: Profile scoring scales with < 5pp degradation.")
        print("  >>> Architecture is sound for larger deployments.")
    elif degradation < 0.15:
        print("  >>> MARGINAL: 5-15pp degradation. Check profile separation.")
        print("  >>> May need tighter factor_sigma or more factors.")
    else:
        print("  >>> CONCERN: >15pp degradation. Profile overlap increases with scale.")
        print("  >>> May need: (a) more factors, (b) tighter sigma, or")
        print("  >>> (c) hierarchical categories instead of flat structure.")

    cold_small  = summary["small"]["cold_t100pct"]["mean"]
    cold_xlarge = summary["xlarge"]["cold_t100pct"]["mean"]
    cold_deg    = cold_small - cold_xlarge

    print(f"\n  Cold start: small={cold_small:.1%} -> xlarge={cold_xlarge:.1%}")
    print(f"  Cold degradation: {cold_deg*100:.1f}pp")

    if cold_deg < 0.10:
        print("  >>> Cold start scales acceptably.")
    else:
        print("  >>> Cold start degrades significantly at scale.")
        print("  >>> Warm start (expert profiles) becomes essential at scale.")

    print("\n  Mean separation: ", end="")
    for c in configs:
        print(f"{c}={summary[c]['mean_separation']:.3f}  ", end="")
    print()

    if summary["xlarge"]["min_separation"] < 0.05:
        print("  >>> WARNING: Minimum separation very low at xlarge.")
        print("  >>> Some centroids are nearly overlapping.")

    print("\n--- OVERALL ---\n")
    xlarge_warm = summary["xlarge"]["warm_t100pct"]["mean"]
    print(f"  Profile-based scoring with L2 kernel:")
    print(f"    Tested up to: 20 categories x 10 actions x 20 factors")
    print(f"    = 200 centroids, 4000 parameters")
    print(f"    Oracle accuracy at xlarge: {xlarge_acc:.1%}")
    print(f"    Warm learning at xlarge:   {xlarge_warm:.1%}")

    if xlarge_acc > 0.85 and xlarge_warm > 0.80:
        print("  >>> Architecture scales. Ready for GAE generalization.")
    elif xlarge_acc > 0.70:
        print("  >>> Architecture has moderate scaling limits. Document constraints.")
    else:
        print("  >>> Architecture has scaling limits. Profile quality is critical.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    bc, realistic = _load_config()
    seeds        = bc["seeds"]
    factor_sigma = float(bc["factor_sigma"])

    results_dir = ROOT / "experiments" / "expE2_scale_test" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 1: Centroid Oracle ===")
    oracle_rows = run_phase1(bc, realistic, seeds, factor_sigma)

    print("\n=== PHASE 2: Learning Convergence ===")
    learning_rows = run_phase2(bc, realistic, seeds, factor_sigma)

    print("\n=== PHASE 3: Profile Separation ===")
    sep_rows = run_phase3(bc, realistic)

    # Build summary
    summary = _build_summary(oracle_rows, learning_rows, sep_rows)

    # Save CSVs
    cols1 = ["scale_config", "seed", "oracle_accuracy", "n_categories",
             "n_actions", "n_factors", "total_centroids", "total_params"]
    df1 = pd.DataFrame(oracle_rows)[cols1]
    p1  = results_dir / "phase1_oracle.csv"
    df1.to_csv(p1, index=False)
    print(f"\nSaved: {p1} ({len(df1)} rows)")

    cols2 = ["scale_config", "condition", "seed", "checkpoint_pct",
             "checkpoint_t", "cumulative_accuracy", "n_decisions"]
    df2 = pd.DataFrame(learning_rows)[cols2]
    p2  = results_dir / "phase2_learning.csv"
    df2.to_csv(p2, index=False)
    print(f"Saved: {p2} ({len(df2)} rows)")

    cols3 = ["scale_config", "category_index", "mean_separation", "min_separation"]
    df3 = pd.DataFrame(sep_rows)[cols3]
    p3  = results_dir / "phase3_separation.csv"
    df3.to_csv(p3, index=False)
    print(f"Saved: {p3} ({len(df3)} rows)")

    with open(results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {results_dir / 'summary.json'}")

    _print_report(summary)

    from src.viz.expE2_charts import generate_all_charts
    generate_all_charts(str(results_dir))


if __name__ == "__main__":
    run()
