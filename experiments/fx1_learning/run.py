"""
FX-1-LEARNING: Does Learning Close the Realistic Distribution Gap?
experiments/fx1_learning/run.py

QUESTION: Does ProfileScorer recover to production-viable accuracy under
realistic data when learning is active (warm start + oracle feedback)?

FX-1-CORRECTED showed 71.45% static accuracy on combined realistic distributions.
All prior high-accuracy results (97.89%, 98.2%) used centroidal synthetic data.
This experiment measures the critical missing number: learning lift under realistic data.

Design outcomes:
  >= 82% by decision 1000 → production-viable as-is
  75-81%                  → per-category kernel config needed
  <  75%                  → shadow mode is a hard prerequisite

GATE-FX1-LEARNING:
  acc_at_1000 >= 0.82     (production-viable with learning)
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.profile_scorer import ProfileScorer
from experiments.fx1_proxy_real.realistic_generator import (
    RealisticAlertGenerator,
    SOC_CATEGORIES,
    SOC_ACTIONS,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_DECISIONS = 2000
N_SEEDS     = 10
TAU         = 0.1
MODE        = "combined"
CHECKPOINTS = [100, 200, 500, 1000, 1500, 2000]
ROLLING_WIN = 200          # window size for rolling accuracy

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# Reference values from prior experiments
REF_STATIC_CENTROIDAL  = 0.9181   # FX-1-CORRECTED centroidal L2
REF_STATIC_REALISTIC   = 0.7145   # FX-1-CORRECTED combined L2 (our static baseline)
REF_LEARNING_CENTROID  = 0.9820   # EXP-B1 warm start + learning

EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
PAPER_DIR    = REPO_ROOT / "paper_figures"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> dict:
    """
    Run one seed. Returns:
      static_acc: float
      learn_checkpoints: {cp: rolling_200_acc}
      learn_checkpoints_per_cat: {cat_name: {cp: rolling_200_acc}}
      static_per_cat: {cat_name: acc}
    """
    gen         = RealisticAlertGenerator(mode=MODE, seed=seed)
    alerts      = gen.generate(N_DECISIONS)
    gt_profiles = gen.get_profiles()

    # ------------------------------------------------------------------
    # Condition 1: Static (no learning)
    # ------------------------------------------------------------------
    scorer_static = ProfileScorer(gt_profiles.copy(), tau=TAU)
    static_correct = 0
    static_per_cat_correct = {cat: 0 for cat in SOC_CATEGORIES}
    static_per_cat_total   = {cat: 0 for cat in SOC_CATEGORIES}

    for a in alerts:
        result     = scorer_static.score(a.factors, a.category_index)
        is_correct = (result.action_index == a.gt_action_index)
        if is_correct:
            static_correct += 1
        cat_name = SOC_CATEGORIES[a.category_index]
        static_per_cat_total[cat_name]   += 1
        if is_correct:
            static_per_cat_correct[cat_name] += 1

    static_acc     = static_correct / N_DECISIONS
    static_per_cat = {
        cat: static_per_cat_correct[cat] / max(static_per_cat_total[cat], 1)
        for cat in SOC_CATEGORIES
    }

    # ------------------------------------------------------------------
    # Condition 2: Learning (warm start + oracle feedback)
    # ------------------------------------------------------------------
    scorer_learn = ProfileScorer(gt_profiles.copy(), tau=TAU)

    # Overall rolling window
    rolling: deque[int] = deque()
    checkpoints_overall: dict[int, float] = {}

    # Per-category rolling windows
    rolling_cat: dict[str, deque[int]] = {cat: deque() for cat in SOC_CATEGORIES}
    checkpoints_per_cat: dict[str, dict[int, float]] = {
        cat: {} for cat in SOC_CATEGORIES
    }

    for i, a in enumerate(alerts):
        result     = scorer_learn.score(a.factors, a.category_index)
        is_correct = int(result.action_index == a.gt_action_index)
        cat_name   = SOC_CATEGORIES[a.category_index]

        # Track rolling overall
        rolling.append(is_correct)
        if len(rolling) > ROLLING_WIN:
            rolling.popleft()

        # Track rolling per-category
        rolling_cat[cat_name].append(is_correct)
        if len(rolling_cat[cat_name]) > ROLLING_WIN:
            rolling_cat[cat_name].popleft()

        # Oracle feedback: update from GT action
        scorer_learn.update(
            a.factors,
            a.category_index,
            a.gt_action_index,
            correct=(result.action_index == a.gt_action_index),
        )

        decision = i + 1
        if decision in CHECKPOINTS:
            checkpoints_overall[decision] = float(np.mean(rolling))
            for cat in SOC_CATEGORIES:
                w = rolling_cat[cat]
                checkpoints_per_cat[cat][decision] = (
                    float(np.mean(w)) if len(w) >= 5 else float("nan")
                )

    return {
        "static_acc":              static_acc,
        "static_per_cat":          static_per_cat,
        "learn_checkpoints":       checkpoints_overall,
        "learn_checkpoints_per_cat": checkpoints_per_cat,
    }


def main() -> bool:
    print("\n=== FX-1-LEARNING: Does Learning Close the Realistic Distribution Gap? ===")
    print(f"Settings: mode={MODE}, N={N_DECISIONS}, seeds={N_SEEDS}, tau={TAU}")
    print(f"Checkpoints: {CHECKPOINTS}, rolling window: {ROLLING_WIN}\n")

    # Sanity-check taxonomy
    assert SOC_CATEGORIES == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ], f"Category mismatch: {SOC_CATEGORIES}"
    assert SOC_ACTIONS == ["escalate", "investigate", "suppress", "monitor"], \
        f"Action mismatch: {SOC_ACTIONS}"

    # Also validate RealisticAlertGenerator produces correct categories
    _gen_check = RealisticAlertGenerator(mode="combined", seed=42)
    assert _gen_check.categories == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ], f"Generator category mismatch: {_gen_check.categories}"

    # -----------------------------------------------------------------------
    # Run all seeds
    # -----------------------------------------------------------------------
    all_static_accs:   list[float] = []
    all_checkpoints:   list[dict]  = []      # list of {cp: acc} per seed
    all_static_per_cat: dict[str, list[float]] = {cat: [] for cat in SOC_CATEGORIES}
    # per-cat checkpoints: {cat: {cp: [acc_seed0, acc_seed1, ...]}}
    all_cp_per_cat: dict[str, dict[int, list[float]]] = {
        cat: {cp: [] for cp in CHECKPOINTS} for cat in SOC_CATEGORIES
    }

    for idx, seed in enumerate(SEEDS):
        print(f"  seed {seed} ({idx+1}/{N_SEEDS}) ...", end=" ", flush=True)
        res = run_seed(seed)
        all_static_accs.append(res["static_acc"])
        all_checkpoints.append(res["learn_checkpoints"])

        for cat in SOC_CATEGORIES:
            all_static_per_cat[cat].append(res["static_per_cat"][cat])
            for cp in CHECKPOINTS:
                v = res["learn_checkpoints_per_cat"][cat].get(cp, float("nan"))
                all_cp_per_cat[cat][cp].append(v)

        acc_1000 = res["learn_checkpoints"].get(1000, float("nan"))
        print(f"static={res['static_acc']*100:.2f}%  learn@1000={acc_1000*100:.2f}%")

    mean_static = float(np.mean(all_static_accs))
    std_static  = float(np.std(all_static_accs))

    # Aggregate checkpoint stats
    checkpoint_stats: dict[int, dict] = {}
    for cp in CHECKPOINTS:
        vals = [d[cp] for d in all_checkpoints if cp in d]
        checkpoint_stats[cp] = {
            "mean_acc": float(np.mean(vals)),
            "std_acc":  float(np.std(vals)),
            "accs":     [round(v, 6) for v in vals],
        }

    # Aggregate per-category at each checkpoint (ignore NaN)
    per_cat_cp_mean: dict[str, dict[int, float]] = {cat: {} for cat in SOC_CATEGORIES}
    for cat in SOC_CATEGORIES:
        for cp in CHECKPOINTS:
            vals = [v for v in all_cp_per_cat[cat][cp] if not np.isnan(v)]
            per_cat_cp_mean[cat][cp] = float(np.mean(vals)) if vals else float("nan")

    per_cat_static_mean = {
        cat: float(np.mean(all_static_per_cat[cat])) for cat in SOC_CATEGORIES
    }

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\nStatic baseline (no learning):            {mean_static*100:.2f}% ±{std_static*100:.2f}pp")
    print(f"Reference — centroidal static  (FX-1):    {REF_STATIC_CENTROIDAL*100:.2f}%")
    print(f"Reference — centroidal learning (EXP-B1): {REF_LEARNING_CENTROID*100:.2f}%")

    print(f"\nLearning trajectory (combined realistic, rolling-{ROLLING_WIN} accuracy):")
    print(f"{'Decision':>10} | {'Mean Acc':>9} | {'±std':>7} | {'vs static':>10} | {'vs 82% gate':>12}")
    print("─" * 62)
    for cp in CHECKPOINTS:
        s = checkpoint_stats[cp]
        delta_vs_static = s["mean_acc"] - mean_static
        gate_gap        = s["mean_acc"] - 0.82
        print(f"{cp:>10} | {s['mean_acc']*100:>8.2f}% | {s['std_acc']*100:>6.2f}% "
              f"| {delta_vs_static*100:>+9.2f}pp | {gate_gap*100:>+11.2f}pp")

    print(f"\nPer-category accuracy at decision 1000 (learning condition, mean over {N_SEEDS} seeds):")
    print(f"  {'Category':<28} {'Learn@1000':>10}  {'Static':>8}  {'Lift':>8}")
    print("  " + "─" * 60)
    for cat in SOC_CATEGORIES:
        learn_acc  = per_cat_cp_mean[cat].get(1000, float("nan"))
        static_acc = per_cat_static_mean[cat]
        lift       = (learn_acc - static_acc) * 100 if not np.isnan(learn_acc) else float("nan")
        learn_str  = f"{learn_acc*100:.2f}%" if not np.isnan(learn_acc) else "  n/a  "
        lift_str   = f"{lift:+.2f}pp"        if not np.isnan(lift)      else "  n/a  "
        print(f"  {cat:<28} {learn_str:>10}  {static_acc*100:>7.2f}%  {lift_str:>8}")

    # -----------------------------------------------------------------------
    # Gate
    # -----------------------------------------------------------------------
    acc_at_1000 = checkpoint_stats[1000]["mean_acc"]
    gate_pass   = acc_at_1000 >= 0.82

    print(f"\n=== GATE-FX1-LEARNING ===")
    print(f"Accuracy at decision 1000: {acc_at_1000*100:.2f}% (gate: >=82%)")
    print(f"GATE-FX1-LEARNING: {'✅ PASS' if gate_pass else '❌ FAIL'}")

    print(f"\nDesign interpretation:")
    if acc_at_1000 >= 0.82:
        print("  Architecture is production-viable with learning.")
        print("  Static 71.45% is the cold-start floor, not the operating point.")
        print("  Warm start + 1000 decisions brings accuracy to production range.")
    elif 0.75 <= acc_at_1000 < 0.82:
        print("  Learning helps but not enough for auto-approve on all categories.")
        print("  travel_anomaly and credential_access need lower confidence thresholds.")
        print("  Shadow mode is a hard prerequisite before production deployment.")
    else:
        print("  Learning cannot compensate for realistic distribution difficulty.")
        print("  Architecture requires kernel adaptation or feature engineering.")
        print("  Do not make production accuracy claims without real SOC data.")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "gate":              "FX1-LEARNING",
        "pass":              gate_pass,
        "mode":              MODE,
        "n_decisions":       N_DECISIONS,
        "n_seeds":           N_SEEDS,
        "tau":               TAU,
        "rolling_window":    ROLLING_WIN,
        "checkpoints":       CHECKPOINTS,
        "mean_static_acc":   round(mean_static, 6),
        "std_static_acc":    round(std_static, 6),
        "acc_at_1000":       round(acc_at_1000, 6),
        "checkpoint_stats":  {
            str(cp): {
                "mean_acc": round(s["mean_acc"], 6),
                "std_acc":  round(s["std_acc"],  6),
            }
            for cp, s in checkpoint_stats.items()
        },
        "checkpoint_stats_per_seed": {
            str(cp): s["accs"] for cp, s in checkpoint_stats.items()
        },
        "per_cat_static_mean": {
            cat: round(v, 6) for cat, v in per_cat_static_mean.items()
        },
        "per_cat_learn_at_1000": {
            cat: round(per_cat_cp_mean[cat].get(1000, float("nan")), 6)
            for cat in SOC_CATEGORIES
        },
        "per_cat_learn_trajectory": {
            cat: {
                str(cp): round(per_cat_cp_mean[cat][cp], 6)
                if not np.isnan(per_cat_cp_mean[cat][cp]) else None
                for cp in CHECKPOINTS
            }
            for cat in SOC_CATEGORIES
        },
        "references": {
            "fx1_corrected_static_centroidal": REF_STATIC_CENTROIDAL,
            "fx1_corrected_static_realistic":  REF_STATIC_REALISTIC,
            "expB1_learning_centroidal":       REF_LEARNING_CENTROID,
        },
        "gate_checks": {
            "acc_at_1000_ge_82pct": gate_pass,
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    try:
        from experiments.fx1_learning.charts import generate_charts
        generate_charts(
            checkpoint_stats=checkpoint_stats,
            mean_static=mean_static,
            per_cat_cp_mean=per_cat_cp_mean,
            per_cat_static_mean=per_cat_static_mean,
            n_decisions=N_DECISIONS,
            n_seeds=N_SEEDS,
            rolling_win=ROLLING_WIN,
            paper_dir=str(PAPER_DIR),
        )
    except Exception as exc:
        import traceback
        print(f"\nWarning: chart generation failed: {exc}")
        traceback.print_exc()

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
