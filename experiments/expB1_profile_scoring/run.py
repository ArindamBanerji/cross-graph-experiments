"""
EXP-B1: Profile-Based Scoring with Online Learning.

Tests profile-based scoring (L2 distance, Eq. 4'') with three conditions:
  profile_warm   — mu initialized from config, updated online
  profile_cold   — mu initialized to uniform 0.5, updated online
  centroid_only  — mu initialized from config, NO updates (EXP-C1 baseline)

Sweeps:
  eta      (correct pull):   [0.01, 0.05, 0.10]
  eta_neg  (incorrect push): [0.005, 0.01, 0.05]
  noise_rate:                [0.0, 0.15, 0.30]

Total: 570 runs × 1000 decisions.

Generates
---------
experiments/expB1_profile_scoring/results/accuracy_trajectories.csv
experiments/expB1_profile_scoring/results/summary.json
paper_figures/expB1_*.{pdf,png}
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
# Experiment constants
# ---------------------------------------------------------------------------

NOISE_RATES   = [0.0, 0.15, 0.30]
ETA_VALUES    = [0.01, 0.05, 0.10]
ETA_NEG_VALUES = [0.005, 0.01, 0.05]
CHECKPOINTS   = [50, 100, 200, 400, 700, 1000]
N_DECISIONS   = 1000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw["realistic_profiles"]


# ---------------------------------------------------------------------------
# Mean pairwise L2 separation between action centroids within each category
# ---------------------------------------------------------------------------

def _mean_separation(mu: np.ndarray) -> float:
    """mu shape: (n_categories, n_actions, n_factors)."""
    n_cats, n_actions, _ = mu.shape
    seps = []
    for c in range(n_cats):
        for a1 in range(n_actions):
            for a2 in range(a1 + 1, n_actions):
                seps.append(float(np.linalg.norm(mu[c, a1] - mu[c, a2])))
    return float(np.mean(seps))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one(
    condition: str,
    eta: float,
    eta_neg: float,
    noise_rate: float,
    seed: int,
    bc: dict,
    realistic: dict,
    tau: float,
) -> list[dict]:
    categories   = bc["categories"]
    actions      = bc["actions"]
    n_categories = len(categories)
    n_actions    = len(actions)
    n_factors    = int(bc["n_factors"])
    gt_dists     = realistic["category_gt_distributions"]

    scorer = ProfileScorer(
        n_categories, n_actions, n_factors,
        tau=tau, eta=eta, eta_neg=eta_neg, seed=seed,
    )

    if condition in ("profile_warm", "centroid_only"):
        scorer.init_from_profiles(
            realistic["action_conditional_profiles"],
            categories, actions,
        )
    else:
        # profile_cold: uniform 0.5 + tiny symmetry-breaking noise so
        # greedy argmax doesn't always collapse to action 0
        rng_sym = np.random.default_rng(seed + 10000)
        scorer.mu += rng_sym.uniform(-0.01, 0.01, scorer.mu.shape)
        np.clip(scorer.mu, 0.0, 1.0, out=scorer.mu)

    initial_mu = scorer.get_profile_snapshot()

    oracle = GTAlignedOracle(noise_rate=noise_rate, seed=seed + 1000)
    gen    = CategoryAlertGenerator(
        categories=categories,
        actions=actions,
        factors=bc["factors"],
        action_conditional_profiles=realistic["action_conditional_profiles"],
        gt_distributions=gt_dists,
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=0.0,
        seed=seed,
    )
    alerts = gen.generate(N_DECISIONS)
    rng_shuffle = np.random.default_rng(seed + 2000)
    rng_shuffle.shuffle(alerts)

    # Running totals
    n_correct_total = 0
    n_correct_cat   = np.zeros(n_categories, dtype=np.int64)
    n_total_cat     = np.zeros(n_categories, dtype=np.int64)

    cp_set = set(CHECKPOINTS)
    rows: list[dict] = []

    for t, alert in enumerate(alerts):
        t1 = t + 1

        action_idx, probs, distances = scorer.score(alert.factors, alert.category_index)

        gt_correct = (action_idx == alert.gt_action_index)
        n_correct_total += int(gt_correct)
        c = alert.category_index
        n_correct_cat[c] += int(gt_correct)
        n_total_cat[c]   += 1

        if condition != "centroid_only":
            result         = oracle.evaluate(actions[action_idx], alert)
            outcome_correct = result.outcome > 0
            scorer.update(alert.factors, c, action_idx, outcome_correct)

        if t1 in cp_set:
            drift   = float(scorer.get_profile_drift(initial_mu).mean())
            mean_sep = _mean_separation(scorer.mu)

            rows.append({
                "condition":            condition,
                "eta":                  eta,
                "eta_neg":              eta_neg,
                "noise_rate":           noise_rate,
                "seed":                 seed,
                "checkpoint":           t1,
                "cumulative_gt_acc":    n_correct_total / t1,
                "gt_acc_credential":    n_correct_cat[0] / max(1, n_total_cat[0]),
                "gt_acc_threat":        n_correct_cat[1] / max(1, n_total_cat[1]),
                "gt_acc_lateral":       n_correct_cat[2] / max(1, n_total_cat[2]),
                "gt_acc_exfil":         n_correct_cat[3] / max(1, n_total_cat[3]),
                "gt_acc_insider":       n_correct_cat[4] / max(1, n_total_cat[4]),
                "mean_profile_drift":   drift,
                "mean_profile_separation": mean_sep,
            })

    return rows


# ---------------------------------------------------------------------------
# Helpers for summary stats
# ---------------------------------------------------------------------------

def _mean_acc_at_t1000(df: pd.DataFrame, condition: str, eta: float,
                        eta_neg: float, noise_rate: float) -> float:
    sub = df[
        (df["condition"]  == condition) &
        (df["checkpoint"] == 1000) &
        (np.abs(df["eta"]          - eta)      < 1e-9) &
        (np.abs(df["eta_neg"]      - eta_neg)  < 1e-9) &
        (np.abs(df["noise_rate"]   - noise_rate) < 1e-6)
    ]
    if sub.empty:
        return 0.0
    return float(sub["cumulative_gt_acc"].mean())


def _best_lr(df: pd.DataFrame, condition: str, noise_rate: float) -> tuple[float, float, float]:
    """Return (best_eta, best_eta_neg, best_acc) for given condition and noise_rate."""
    best_acc, best_eta, best_eta_neg = -1.0, 0.0, 0.0
    for eta in ETA_VALUES:
        for eta_neg in ETA_NEG_VALUES:
            acc = _mean_acc_at_t1000(df, condition, eta, eta_neg, noise_rate)
            if acc > best_acc:
                best_acc, best_eta, best_eta_neg = acc, eta, eta_neg
    return best_eta, best_eta_neg, best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    bc, realistic = _load_config()
    categories = bc["categories"]
    actions    = bc["actions"]
    seeds      = bc["seeds"]
    tau        = float(bc["scoring"]["temperature"])

    results_dir = ROOT / "experiments" / "expB1_profile_scoring" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # centroid_only: 3 noise × 10 seeds = 30 runs
    # -----------------------------------------------------------------------
    print("Running centroid_only (30 runs)...")
    for noise_rate in NOISE_RATES:
        for seed in seeds:
            rows = _run_one("centroid_only", 0.0, 0.0,
                            noise_rate, seed, bc, realistic, tau)
            all_rows.extend(rows)
    print("  centroid_only: done")

    # -----------------------------------------------------------------------
    # profile_warm and profile_cold: 9 lr × 3 noise × 10 seeds = 270 each
    # -----------------------------------------------------------------------
    for condition in ("profile_warm", "profile_cold"):
        n_total = len(ETA_VALUES) * len(ETA_NEG_VALUES) * len(NOISE_RATES) * len(seeds)
        print(f"Running {condition} ({n_total} runs)...")
        done = 0
        for noise_rate in NOISE_RATES:
            for eta in ETA_VALUES:
                for eta_neg in ETA_NEG_VALUES:
                    for seed in seeds:
                        rows = _run_one(condition, eta, eta_neg,
                                        noise_rate, seed, bc, realistic, tau)
                        all_rows.extend(rows)
                    done += len(seeds)
            print(f"  {condition}: {done}/{n_total} runs (noise={noise_rate})")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = results_dir / "accuracy_trajectories.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------------
    # centroid_only baseline (noise=0)
    co_sub = df[
        (df["condition"] == "centroid_only") &
        (df["checkpoint"] == 1000) &
        (df["noise_rate"] == 0.0)
    ]
    centroid_baseline = float(co_sub["cumulative_gt_acc"].mean())

    # Best warm config at noise=0
    best_warm_eta, best_warm_eta_neg, best_warm_acc = _best_lr(df, "profile_warm", 0.0)

    # Best warm config at noise=0.15
    best_warm_eta_15, best_warm_eta_neg_15, best_warm_acc_15 = _best_lr(df, "profile_warm", 0.15)

    # Best warm config at noise=0.30
    best_warm_eta_30, best_warm_eta_neg_30, best_warm_acc_30 = _best_lr(df, "profile_warm", 0.30)

    # Best cold config at noise=0
    best_cold_eta, best_cold_eta_neg, best_cold_acc = _best_lr(df, "profile_cold", 0.0)

    # All config stats
    config_stats: dict = {}
    for condition in ("profile_warm", "profile_cold"):
        for eta in ETA_VALUES:
            for eta_neg in ETA_NEG_VALUES:
                for noise_rate in NOISE_RATES:
                    key = f"{condition}_{eta}_{eta_neg}_{noise_rate}"
                    sub = df[
                        (df["condition"]  == condition) &
                        (df["checkpoint"] == 1000) &
                        (np.abs(df["eta"]        - eta)      < 1e-9) &
                        (np.abs(df["eta_neg"]    - eta_neg)  < 1e-9) &
                        (np.abs(df["noise_rate"] - noise_rate) < 1e-6)
                    ]
                    config_stats[key] = {
                        "mean_accuracy_t1000": float(sub["cumulative_gt_acc"].mean()) if not sub.empty else 0.0,
                        "std_accuracy_t1000":  float(sub["cumulative_gt_acc"].std())  if not sub.empty else 0.0,
                    }
    # centroid_only across noise rates
    for noise_rate in NOISE_RATES:
        key = f"centroid_only_0.0_0.0_{noise_rate}"
        sub = df[
            (df["condition"] == "centroid_only") &
            (df["checkpoint"] == 1000) &
            (np.abs(df["noise_rate"] - noise_rate) < 1e-6)
        ]
        config_stats[key] = {
            "mean_accuracy_t1000": float(sub["cumulative_gt_acc"].mean()) if not sub.empty else 0.0,
            "std_accuracy_t1000":  float(sub["cumulative_gt_acc"].std())  if not sub.empty else 0.0,
        }

    summary = {
        "configs": config_stats,
        "best_config": {
            "condition": "profile_warm",
            "eta":       best_warm_eta,
            "eta_neg":   best_warm_eta_neg,
            "accuracy":  best_warm_acc,
        },
        "best_noisy_config": {
            "condition":  "profile_warm",
            "eta":        best_warm_eta_15,
            "eta_neg":    best_warm_eta_neg_15,
            "accuracy":   best_warm_acc_15,
        },
        "cold_best_config": {
            "condition": "profile_cold",
            "eta":       best_cold_eta,
            "eta_neg":   best_cold_eta_neg,
            "accuracy":  best_cold_acc,
        },
        "centroid_baseline":    centroid_baseline,
        "best_warm_acc_noise0": best_warm_acc,
        "best_warm_acc_noise15": best_warm_acc_15,
        "best_warm_acc_noise30": best_warm_acc_30,
        "best_cold_acc_noise0": best_cold_acc,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {summary_path}")

    # -----------------------------------------------------------------------
    # Validation report
    # -----------------------------------------------------------------------
    vb1_1 = centroid_baseline
    vb1_2 = best_warm_acc
    vb1_3 = best_cold_acc
    vb1_4 = best_warm_acc_15
    vb1_5 = best_warm_acc_30

    # VB1.7: cold start early vs late convergence
    cold_df_t100  = df[
        (df["condition"]  == "profile_cold") &
        (df["noise_rate"] == 0.0) &
        (df["checkpoint"] == 100) &
        (np.abs(df["eta"]        - best_cold_eta)     < 1e-9) &
        (np.abs(df["eta_neg"]    - best_cold_eta_neg) < 1e-9)
    ]
    cold_df_t1000 = df[
        (df["condition"]  == "profile_cold") &
        (df["noise_rate"] == 0.0) &
        (df["checkpoint"] == 1000) &
        (np.abs(df["eta"]        - best_cold_eta)     < 1e-9) &
        (np.abs(df["eta_neg"]    - best_cold_eta_neg) < 1e-9)
    ]
    cold_t100  = float(cold_df_t100["cumulative_gt_acc"].mean())  if not cold_df_t100.empty  else 0.0
    cold_t1000 = float(cold_df_t1000["cumulative_gt_acc"].mean()) if not cold_df_t1000.empty else 0.0

    print("\n" + "=" * 60)
    print("EXP-B1: PROFILE-BASED SCORING VALIDATION")
    print("=" * 60)

    def _gate(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"\n  VB1.1  centroid_only noise=0.0:             {vb1_1:.1%}  "
          f"[expected ~97.89%]  {_gate(abs(vb1_1 - 0.9789) < 0.05)}")
    print(f"  VB1.2  profile_warm best (noise=0):         {vb1_2:.1%}  "
          f"[expected >= 95%]   {_gate(vb1_2 >= 0.95)}")
    print(f"  VB1.3  profile_cold best (noise=0):         {vb1_3:.1%}  "
          f"[expected >= 80%]   {_gate(vb1_3 >= 0.80)}")
    print(f"  VB1.4  profile_warm best (noise=0.15):      {vb1_4:.1%}  "
          f"[expected >= 85%]   {_gate(vb1_4 >= 0.85)}")
    print(f"  VB1.5  profile_warm best (noise=0.30):      {vb1_5:.1%}  "
          f"[expected >= 70%]   {_gate(vb1_5 >= 0.70)}")

    delta_learning = vb1_2 - vb1_1
    print(f"\n  VB1.6  Learning effect (warm vs centroid_only):")
    print(f"    centroid_only (no learning)          = {vb1_1:.1%}")
    print(f"    profile_warm (best lr, noise=0)      = {vb1_2:.1%}")
    print(f"    Delta from learning                  = {delta_learning*100:+.1f}pp")
    if delta_learning < -0.02:
        print(f"    WARNING: Learning HURTS. Centroid-only is better.")
    else:
        print(f"    Learning helps or is neutral.")

    print(f"\n  VB1.7  profile_cold convergence (best lr, noise=0):")
    print(f"    t=100:  {cold_t100:.1%}")
    print(f"    t=1000: {cold_t1000:.1%}")
    converges = cold_t1000 > cold_t100
    print(f"    Cold start converges from t=100 to t=1000: {_gate(converges)}")

    print(f"\n  VB1.8  Best learning rates:")
    print(f"    profile_warm noise=0:    eta={best_warm_eta}, eta_neg={best_warm_eta_neg}"
          f"  -> {vb1_2:.1%}")
    print(f"    profile_warm noise=0.15: eta={best_warm_eta_15}, eta_neg={best_warm_eta_neg_15}"
          f"  -> {vb1_4:.1%}")
    print(f"    profile_cold noise=0:    eta={best_cold_eta}, eta_neg={best_cold_eta_neg}"
          f"  -> {vb1_3:.1%}")

    pass_condition = (vb1_2 >= 0.95 and vb1_4 >= 0.85)
    marginal_condition = (vb1_2 >= 0.90 or vb1_4 >= 0.75)

    print("\n  OVERALL GATE:", end="  ")
    if pass_condition:
        print("PASS  (VB1.2 >= 95% AND VB1.4 >= 85%)")
    elif marginal_condition:
        print("MARGINAL  (VB1.2 >= 90% OR VB1.4 >= 75%)")
    else:
        print("FAIL  (VB1.2 < 90% AND VB1.4 < 75%)")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    from src.viz.expB1_charts import generate_all_charts
    generate_all_charts(str(results_dir))


if __name__ == "__main__":
    run()
