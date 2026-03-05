"""
EXP-E1: Kernel Generalization.

Answers: should the GAE hardcode L2, or support pluggable kernels?

Tests 4 similarity kernels x 3 factor distributions x 10 seeds x 10,000 alerts.
Phase 1: Pure centroid oracle (no learning) - isolates kernel choice.
Phase 2: Learning validation on best and worst kernel per distribution.

Generates
---------
experiments/expE1_kernel_generalization/results/phase1_oracle.csv
experiments/expE1_kernel_generalization/results/phase2_learning.csv
experiments/expE1_kernel_generalization/results/covariance_stats.csv
experiments/expE1_kernel_generalization/results/summary.json
paper_figures/expE1_*.{pdf,png}
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
# Config
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw["realistic_profiles"]


# ---------------------------------------------------------------------------
# Factor distribution transforms
# ---------------------------------------------------------------------------

MIXED_SCALE = np.array([1.0, 1.0, 100.0, 100.0, 0.01, 0.01])


def _build_scaler(profiles_dict: dict, categories: list, actions: list) -> dict:
    """Compute per-factor mean and std from all profile centroids (for 'normalized')."""
    all_vecs = []
    for cat in categories:
        for act in actions:
            all_vecs.append(profiles_dict[cat][act])
    arr = np.array(all_vecs, dtype=np.float64)   # (20, 6)
    return {
        "mean": np.mean(arr, axis=0),
        "std":  np.std(arr,  axis=0),
    }


def transform_factors(f: np.ndarray, distribution: str, scaler: dict | None = None) -> np.ndarray:
    if distribution == "original":
        return f
    elif distribution == "normalized":
        return (f - scaler["mean"]) / (scaler["std"] + 1e-10)
    elif distribution == "mixed_scale":
        return f * MIXED_SCALE
    else:
        raise ValueError(distribution)


def transform_profiles(mu: np.ndarray, distribution: str, scaler: dict | None = None) -> np.ndarray:
    """mu: (n_categories, n_actions, n_factors)."""
    mu_t = mu.copy()
    for c in range(mu.shape[0]):
        for a in range(mu.shape[1]):
            mu_t[c, a, :] = transform_factors(mu[c, a, :], distribution, scaler)
    return mu_t


# ---------------------------------------------------------------------------
# Similarity kernels
# ---------------------------------------------------------------------------

def score_l2(f: np.ndarray, mu_c: np.ndarray) -> np.ndarray:
    """Negative squared L2. mu_c: (n_actions, n_factors)."""
    return np.array([-np.sum((f - mu_c[a]) ** 2) for a in range(mu_c.shape[0])])


def score_cosine(f: np.ndarray, mu_c: np.ndarray) -> np.ndarray:
    f_norm = np.linalg.norm(f) + 1e-10
    return np.array([
        np.dot(f, mu_c[a]) / (f_norm * (np.linalg.norm(mu_c[a]) + 1e-10))
        for a in range(mu_c.shape[0])
    ])


def score_dot(f: np.ndarray, mu_c: np.ndarray) -> np.ndarray:
    return mu_c @ f


def score_mahalanobis(f: np.ndarray, mu_c: np.ndarray, cov_inv_c: np.ndarray) -> np.ndarray:
    scores = []
    for a in range(mu_c.shape[0]):
        diff = f - mu_c[a]
        scores.append(-diff @ cov_inv_c[a] @ diff)
    return np.array(scores)


def apply_kernel(
    kernel: str,
    f: np.ndarray,
    mu_c: np.ndarray,
    cov_inv_c: np.ndarray | None = None,
) -> np.ndarray:
    if kernel == "l2":
        return score_l2(f, mu_c)
    elif kernel == "cosine":
        return score_cosine(f, mu_c)
    elif kernel == "dot":
        return score_dot(f, mu_c)
    elif kernel == "mahalanobis":
        return score_mahalanobis(f, mu_c, cov_inv_c)
    else:
        raise ValueError(kernel)


# ---------------------------------------------------------------------------
# Covariance estimation for Mahalanobis
# ---------------------------------------------------------------------------

def _estimate_covariances(
    categories: list,
    actions: list,
    factors: list,
    profiles: dict,
    gt_dists: dict,
    factor_sigma: float,
    seeds: list,
    distribution: str,
    scaler: dict | None,
    n_per_cat: int = 2000,
) -> np.ndarray:
    """
    Return cov_inv[n_categories, n_actions, n_factors, n_factors].
    Estimated from generated data, transformed per distribution.
    """
    n_cat = len(categories)
    n_act = len(actions)
    n_fac = len(factors)

    # Accumulate factor vectors per (cat, action)
    collected: list[list[list]] = [[[] for _ in range(n_act)] for _ in range(n_cat)]

    for seed in seeds[:3]:   # 3 seeds is enough for stable covariance
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
        for alert in alerts:
            f_t = transform_factors(alert.factors, distribution, scaler)
            collected[alert.category_index][alert.gt_action_index].append(f_t)

    cov_inv = np.zeros((n_cat, n_act, n_fac, n_fac), dtype=np.float64)
    for c in range(n_cat):
        for a in range(n_act):
            vecs = np.array(collected[c][a], dtype=np.float64)   # (N, 6)
            if len(vecs) < n_fac + 1:
                cov_inv[c, a] = np.eye(n_fac)
                continue
            cov = np.cov(vecs, rowvar=False)
            cov_reg = cov + 0.01 * np.eye(n_fac)
            try:
                cov_inv[c, a] = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv[c, a] = np.eye(n_fac)

    return cov_inv


# ---------------------------------------------------------------------------
# Covariance stats for covariance_stats.csv
# ---------------------------------------------------------------------------

def _covariance_stats(
    cov_inv: np.ndarray,
    categories: list,
    actions: list,
    distribution: str,
) -> list[dict]:
    rows = []
    n_fac = cov_inv.shape[-1]
    for c, cat in enumerate(categories):
        for a, act in enumerate(actions):
            ci = cov_inv[c, a]
            # Recover approx cov from inv (for stats display; small numeric error OK)
            try:
                cov = np.linalg.inv(ci)
            except np.linalg.LinAlgError:
                cov = np.eye(n_fac)
            trace = float(np.trace(cov))
            det   = float(np.linalg.det(cov))
            eigs  = np.linalg.eigvalsh(cov)
            eigs  = np.maximum(eigs, 1e-12)
            cond  = float(eigs.max() / eigs.min())
            rows.append({
                "distribution": distribution,
                "category":     cat,
                "action":       act,
                "cov_trace":    trace,
                "cov_det":      det,
                "cov_condition_number": cond,
            })
    return rows


# ---------------------------------------------------------------------------
# Phase 1: Centroid oracle (no learning)
# ---------------------------------------------------------------------------

DISTRIBUTIONS = ["original", "normalized", "mixed_scale"]
KERNELS = ["l2", "cosine", "dot", "mahalanobis"]


def _run_phase1(
    categories: list,
    actions: list,
    factors: list,
    profiles: dict,
    gt_dists: dict,
    factor_sigma: float,
    seeds: list,
    mu: np.ndarray,
    cov_inv_by_dist: dict,
    scaler: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (oracle_rows, cov_stat_rows).
    oracle_rows: one row per (distribution, kernel, seed).
    cov_stat_rows: one row per (distribution, category, action).
    """
    n_per_cat = 2000   # 2000 per category x 5 = 10,000 per seed
    cat_names = categories

    oracle_rows: list[dict] = []
    cov_stat_rows: list[dict] = []

    for distribution in DISTRIBUTIONS:
        scaler_d = scaler if distribution == "normalized" else None
        mu_t = transform_profiles(mu, distribution, scaler_d)
        cov_inv = cov_inv_by_dist[distribution]

        # Covariance stats (once per distribution)
        cov_stat_rows.extend(_covariance_stats(cov_inv, cat_names, actions, distribution))

        for kernel in KERNELS:
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

                correct = 0
                per_cat_correct = np.zeros(len(categories), dtype=int)
                per_cat_total   = np.zeros(len(categories), dtype=int)

                for alert in alerts:
                    f_t = transform_factors(alert.factors, distribution, scaler_d)
                    c   = alert.category_index
                    gt  = alert.gt_action_index
                    mu_c = mu_t[c]

                    scores = apply_kernel(kernel, f_t, mu_c, cov_inv[c] if kernel == "mahalanobis" else None)
                    pred = int(np.argmax(scores))

                    if pred == gt:
                        correct += 1
                        per_cat_correct[c] += 1
                    per_cat_total[c] += 1

                overall_acc = correct / len(alerts)
                print(f"  {distribution}/{kernel} seed={seed}: {overall_acc:.1%}")

                row: dict = {
                    "distribution": distribution,
                    "kernel":       kernel,
                    "seed":         seed,
                    "overall_accuracy": overall_acc,
                }
                for ci, cat in enumerate(cat_names):
                    short_key = f"acc_{cat.split('_')[0][:8]}"   # acc_credenti, etc.
                    # Use stable column names
                    col = f"acc_{ci}"
                    row[col] = per_cat_correct[ci] / per_cat_total[ci] if per_cat_total[ci] > 0 else 0.0

                # Also add named columns for the 5 standard categories
                row["acc_credential"] = per_cat_correct[0] / per_cat_total[0] if per_cat_total[0] > 0 else 0.0
                row["acc_threat"]     = per_cat_correct[1] / per_cat_total[1] if per_cat_total[1] > 0 else 0.0
                row["acc_lateral"]    = per_cat_correct[2] / per_cat_total[2] if per_cat_total[2] > 0 else 0.0
                row["acc_exfil"]      = per_cat_correct[3] / per_cat_total[3] if per_cat_total[3] > 0 else 0.0
                row["acc_insider"]    = per_cat_correct[4] / per_cat_total[4] if per_cat_total[4] > 0 else 0.0
                oracle_rows.append(row)

    return oracle_rows, cov_stat_rows


# ---------------------------------------------------------------------------
# Phase 2: Learning validation
# ---------------------------------------------------------------------------

CHECKPOINTS = [100, 200, 400, 700, 1000]


def _run_phase2(
    categories: list,
    actions: list,
    factors: list,
    profiles: dict,
    gt_dists: dict,
    factor_sigma: float,
    seeds: list,
    mu: np.ndarray,
    cov_inv_by_dist: dict,
    scaler: dict,
    best_worst: dict,
) -> list[dict]:
    """
    best_worst: {distribution: {"best": kernel, "worst": kernel}}
    Returns learning rows: (distribution, kernel, kernel_rank, seed, checkpoint, cumulative_acc).
    """
    rows: list[dict] = []
    n_decisions = 1000

    for distribution in DISTRIBUTIONS:
        scaler_d = scaler if distribution == "normalized" else None
        mu_t_init = transform_profiles(mu, distribution, scaler_d)
        cov_inv = cov_inv_by_dist[distribution]

        bw = best_worst[distribution]
        kernels_to_test = []
        if bw["best"] != bw["worst"]:
            kernels_to_test = [bw["best"], bw["worst"]]
        else:
            kernels_to_test = [bw["best"]]

        for kernel in kernels_to_test:
            rank = "best" if kernel == bw["best"] else "worst"

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
                alerts = gen.generate(n=n_decisions)

                mu_local = mu_t_init.copy()
                counts   = np.zeros((len(categories), len(actions)), dtype=np.float64)
                n_correct_so_far = 0

                for t, alert in enumerate(alerts):
                    f_t = transform_factors(alert.factors, distribution, scaler_d)
                    c   = alert.category_index
                    gt  = alert.gt_action_index

                    scores = apply_kernel(kernel, f_t, mu_local[c], cov_inv[c] if kernel == "mahalanobis" else None)
                    pred = int(np.argmax(scores))
                    gt_correct = (pred == gt)

                    # Centroid update (L2-style pull/push)
                    eta = 0.05 / (1.0 + counts[c, pred] * 0.001)
                    if gt_correct:
                        mu_local[c, pred] += eta * (f_t - mu_local[c, pred])
                    else:
                        eta_neg = 0.05 / (1.0 + counts[c, pred] * 0.001)
                        mu_local[c, pred] -= eta_neg * (f_t - mu_local[c, pred])
                    counts[c, pred] += 1

                    if gt_correct:
                        n_correct_so_far += 1

                    step = t + 1
                    if step in CHECKPOINTS:
                        cum_acc = n_correct_so_far / step
                        rows.append({
                            "distribution":          distribution,
                            "kernel":                kernel,
                            "kernel_rank":           rank,
                            "seed":                  seed,
                            "checkpoint":            step,
                            "cumulative_gt_accuracy": cum_acc,
                        })

    return rows


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    oracle_rows: list[dict],
    learning_rows: list[dict],
) -> dict:
    df = pd.DataFrame(oracle_rows)

    summary: dict = {"phase1": {}, "phase2": {}, "recommendations": {}, "gae_guidance": ""}

    best_worst: dict = {}

    for dist in DISTRIBUTIONS:
        sub = df[df["distribution"] == dist]
        summary["phase1"][dist] = {}
        kernel_means: dict[str, float] = {}

        for kernel in KERNELS:
            ksub = sub[sub["kernel"] == kernel]
            mean_acc = float(ksub["overall_accuracy"].mean())
            std_acc  = float(ksub["overall_accuracy"].std())
            summary["phase1"][dist][kernel] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}
            kernel_means[kernel] = mean_acc

        ranked = sorted(kernel_means.items(), key=lambda x: x[1], reverse=True)
        best_k  = ranked[0][0]
        worst_k = ranked[-1][0]

        summary["phase1"][dist]["best_kernel"]    = best_k
        summary["phase1"][dist]["worst_kernel"]   = worst_k
        summary["phase1"][dist]["best_accuracy"]  = kernel_means[best_k]
        summary["phase1"][dist]["worst_accuracy"] = kernel_means[worst_k]
        summary["phase1"][dist]["kernel_ranking"] = [k for k, _ in ranked]

        best_worst[dist] = {"best": best_k, "worst": worst_k}

    # Phase 2
    if learning_rows:
        df2 = pd.DataFrame(learning_rows)
        for dist in DISTRIBUTIONS:
            for kernel in KERNELS:
                sub2 = df2[(df2["distribution"] == dist) & (df2["kernel"] == kernel) & (df2["checkpoint"] == 1000)]
                if sub2.empty:
                    continue
                key = f"{dist}/{kernel}"
                summary["phase2"][key] = {
                    "accuracy_at_t1000_mean": float(sub2["cumulative_gt_accuracy"].mean()),
                    "accuracy_at_t1000_std":  float(sub2["cumulative_gt_accuracy"].std()),
                }

    # Recommendations
    for dist in DISTRIBUTIONS:
        best_k = summary["phase1"][dist]["best_kernel"]
        best_a = summary["phase1"][dist]["best_accuracy"]
        summary["recommendations"][dist] = {
            "recommended_kernel": best_k,
            "reason": f"{best_k} achieved highest accuracy ({best_a:.3f}) on {dist} distribution",
        }

    # GAE guidance
    l2_wins = sum(1 for d in DISTRIBUTIONS if summary["phase1"][d]["best_kernel"] == "l2")
    maha_best = max(summary["phase1"][d]["mahalanobis"]["mean_accuracy"] for d in DISTRIBUTIONS)
    l2_best   = max(summary["phase1"][d]["l2"]["mean_accuracy"] for d in DISTRIBUTIONS)

    if l2_wins == 3:
        summary["gae_guidance"] = "hardcode_l2"
    elif l2_wins >= 2:
        summary["gae_guidance"] = "l2_default_with_cosine_option"
    else:
        summary["gae_guidance"] = "pluggable_kernels"

    return summary, best_worst


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _print_report(summary: dict, oracle_rows: list[dict]) -> None:
    df = pd.DataFrame(oracle_rows)

    def get_accuracy(dist: str, kernel: str) -> float:
        return summary["phase1"][dist][kernel]["mean_accuracy"]

    print("\n" + "=" * 70)
    print("EXP-E1: KERNEL GENERALIZATION")
    print("=" * 70)

    print("\n--- PHASE 1: CENTROID ORACLE (no learning) ---\n")

    for dist in DISTRIBUTIONS:
        ranked = summary["phase1"][dist]["kernel_ranking"]
        print(f"  Distribution: {dist}")
        print(f"  {'Kernel':<15s} {'Accuracy':>10s} {'+/-std':>8s} {'':>6s}")
        print(f"  {'-'*44}")
        for i, kernel in enumerate(ranked):
            mean_acc = summary["phase1"][dist][kernel]["mean_accuracy"]
            std_acc  = summary["phase1"][dist][kernel]["std_accuracy"]
            marker = " <<<" if i == 0 else ""
            print(f"  {kernel:<15s} {mean_acc:>9.1%} {std_acc:>7.1%}{marker}")
        print()

    print("\n--- KEY COMPARISONS ---\n")

    l2_wins = 0
    for dist in DISTRIBUTIONS:
        best = summary["phase1"][dist]["best_kernel"]
        print(f"  {dist}: best = {best} ({summary['phase1'][dist]['best_accuracy']:.1%})")
        if best == "l2":
            l2_wins += 1

    print(f"\n  L2 wins {l2_wins}/3 distributions.")
    if l2_wins == 3:
        print("  >>> L2 is universally best. GAE can hardcode L2.")
    elif l2_wins >= 2:
        print("  >>> L2 is usually best. GAE should default to L2 with kernel option.")
    else:
        print("  >>> Kernel choice is data-dependent. GAE MUST support pluggable kernels.")

    # Dot product: does normalization help?
    dot_original   = get_accuracy("original",   "dot")
    dot_normalized = get_accuracy("normalized", "dot")
    l2_original    = get_accuracy("original",   "l2")
    l2_normalized  = get_accuracy("normalized", "l2")

    print(f"\n  Dot product: original={dot_original:.1%} -> normalized={dot_normalized:.1%} "
          f"(delta={dot_normalized-dot_original:+.1f}pp)")
    print(f"  L2:          original={l2_original:.1%} -> normalized={l2_normalized:.1%} "
          f"(delta={l2_normalized-l2_original:+.1f}pp)")

    if dot_normalized > 0.90:
        print("  >>> Normalization fixes dot product. Magnitude was the only problem.")
    else:
        print("  >>> Normalization does NOT fully fix dot product. Shape matters too.")

    # Mahalanobis vs L2
    maha_best = max(get_accuracy(d, "mahalanobis") for d in DISTRIBUTIONS)
    l2_best   = max(get_accuracy(d, "l2") for d in DISTRIBUTIONS)
    print(f"\n  Best Mahalanobis: {maha_best:.1%}")
    print(f"  Best L2:          {l2_best:.1%}")
    if maha_best > l2_best + 0.01:
        print("  >>> Mahalanobis beats L2. Cluster shape matters. Consider for GAE.")
    else:
        print("  >>> Mahalanobis ~= L2. Clusters are roughly spherical. L2 sufficient.")

    # Mixed scale test
    print()
    for kernel in KERNELS:
        mixed_acc = get_accuracy("mixed_scale", kernel)
        orig_acc  = get_accuracy("original",    kernel)
        delta_pp  = (mixed_acc - orig_acc) * 100
        print(f"  {kernel:<15}: original={orig_acc:.1%} -> mixed_scale={mixed_acc:.1%} "
              f"(delta={delta_pp:+.1f}pp)")

    print("\n--- GAE RECOMMENDATION ---\n")

    guidance = summary["gae_guidance"]
    maha_vs_l2 = maha_best - l2_best

    if guidance == "hardcode_l2" and maha_vs_l2 <= 0.01:
        print("  RECOMMENDATION: L2 as default kernel.")
        print("  Cosine as optional alternative for pre-normalized data.")
        print("  Mahalanobis not needed (clusters are roughly spherical).")
        print("  Dot product should NOT be offered (magnitude confounding risk).")
    elif guidance in ("hardcode_l2", "l2_default_with_cosine_option"):
        print("  RECOMMENDATION: Pluggable kernels with L2 default.")
        print("  Supported: L2 (default), cosine, mahalanobis.")
        print("  Dot product: supported but warned (only for pre-normalized data).")
    else:
        print("  RECOMMENDATION: Pluggable kernels, NO default.")
        print("  User must choose based on data characteristics.")
        print("  Provide guidance: L2 for [0,1] data, cosine for normalized,")
        print("  mahalanobis for non-spherical clusters.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
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

    n_cat = len(categories)
    n_act = len(actions)
    n_fac = len(factors)

    # Build centroid matrix from configured profiles
    mu = np.zeros((n_cat, n_act, n_fac), dtype=np.float64)
    for c, cat in enumerate(categories):
        for a, act in enumerate(actions):
            mu[c, a, :] = np.array(profiles[cat][act], dtype=np.float64)

    # Scaler for normalized distribution
    scaler = _build_scaler(profiles, categories, actions)

    results_dir = ROOT / "experiments" / "expE1_kernel_generalization" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute covariances for Mahalanobis per distribution
    print("Pre-computing covariances for Mahalanobis...")
    cov_inv_by_dist: dict = {}
    for distribution in DISTRIBUTIONS:
        scaler_d = scaler if distribution == "normalized" else None
        print(f"  {distribution}...")
        cov_inv_by_dist[distribution] = _estimate_covariances(
            categories=categories,
            actions=actions,
            factors=factors,
            profiles=profiles,
            gt_dists=gt_dists,
            factor_sigma=factor_sigma,
            seeds=seeds,
            distribution=distribution,
            scaler=scaler_d,
        )
    print("Done.\n")

    # Phase 1
    print("=== PHASE 1: Centroid Oracle ===")
    oracle_rows, cov_stat_rows = _run_phase1(
        categories=categories,
        actions=actions,
        factors=factors,
        profiles=profiles,
        gt_dists=gt_dists,
        factor_sigma=factor_sigma,
        seeds=seeds,
        mu=mu,
        cov_inv_by_dist=cov_inv_by_dist,
        scaler=scaler,
    )

    # Build partial summary to identify best/worst kernels
    summary, best_worst = _build_summary(oracle_rows, [])

    # Phase 2
    print("\n=== PHASE 2: Learning Validation ===")
    learning_rows = _run_phase2(
        categories=categories,
        actions=actions,
        factors=factors,
        profiles=profiles,
        gt_dists=gt_dists,
        factor_sigma=factor_sigma,
        seeds=seeds,
        mu=mu,
        cov_inv_by_dist=cov_inv_by_dist,
        scaler=scaler,
        best_worst=best_worst,
    )

    # Final summary with phase 2 data
    summary, _ = _build_summary(oracle_rows, learning_rows)

    # Save CSVs
    cols_oracle = [
        "distribution", "kernel", "seed", "overall_accuracy",
        "acc_credential", "acc_threat", "acc_lateral", "acc_exfil", "acc_insider",
    ]
    df_oracle = pd.DataFrame(oracle_rows)[cols_oracle]
    oracle_path = results_dir / "phase1_oracle.csv"
    df_oracle.to_csv(oracle_path, index=False)
    print(f"\nSaved: {oracle_path} ({len(df_oracle)} rows)")

    if learning_rows:
        df_learning = pd.DataFrame(learning_rows)
        learning_path = results_dir / "phase2_learning.csv"
        df_learning.to_csv(learning_path, index=False)
        print(f"Saved: {learning_path} ({len(df_learning)} rows)")
    else:
        df_learning = pd.DataFrame()

    df_cov = pd.DataFrame(cov_stat_rows)
    cov_path = results_dir / "covariance_stats.csv"
    df_cov.to_csv(cov_path, index=False)
    print(f"Saved: {cov_path} ({len(df_cov)} rows)")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {summary_path}")

    # Validation report
    _print_report(summary, oracle_rows)

    # Charts
    from src.viz.expE1_charts import generate_all_charts
    generate_all_charts(str(results_dir))


if __name__ == "__main__":
    run()
