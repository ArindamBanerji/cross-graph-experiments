"""
Experiment 5 (Redesigned): Oracle Fix — Asymmetry Ratio Sweep + Warmup Comparison.

Section A: 10 seeds × 5 oracle configs × 5 asymmetry ratios × 1000 decisions.
Section B: 10 seeds × 3 learning schedules at GT(15%) oracle.

Outputs
-------
results/ratio_sweep.csv           1 500 rows  (10×5×5×6 checkpoints)
results/warmup_comparison.csv       180 rows  (10×3×6 checkpoints)
results/best_config.json
paper_figures/exp5_*.{pdf,png}   12 files    (6 charts × 2 formats)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.category_alert_generator import CategoryAlertGenerator, ACTIONS, CATEGORIES
from src.models.scoring_matrix import ScoringMatrix
from src.models.oracle import BernoulliOracle, GTAlignedOracle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"

RATIO_SWEEP_VALUES: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0]
CHECKPOINTS: list[int]          = [50, 100, 200, 400, 700, 1000]
RECENT_N: int                   = 50  # window size for recent_50_gt_acc

RATIO_SWEEP_COLUMNS = [
    "seed", "oracle_type", "noise_rate", "ratio", "checkpoint",
    "cumulative_gt_acc", "recent_50_gt_acc", "w_frobenius", "w_max", "w_entropy",
    "grad_mag_f0", "grad_mag_f1", "grad_mag_f2", "grad_mag_f3", "grad_mag_f4", "grad_mag_f5",
    "gt_acc_credential", "gt_acc_threat", "gt_acc_lateral", "gt_acc_exfil", "gt_acc_insider",
]

WARMUP_COLUMNS = [
    "seed", "schedule", "checkpoint", "effective_ratio",
    "cumulative_gt_acc", "recent_50_gt_acc", "w_frobenius", "w_max", "w_entropy",
    "grad_mag_f0", "grad_mag_f1", "grad_mag_f2", "grad_mag_f3", "grad_mag_f4", "grad_mag_f5",
    "gt_acc_credential", "gt_acc_threat", "gt_acc_lateral", "gt_acc_exfil", "gt_acc_insider",
]

# Oracle configs: (oracle_type, noise_rate)
ORACLE_SPECS: list[tuple[str, float]] = [
    ("bernoulli",   -1.0),
    ("gt_aligned",   0.0),
    ("gt_aligned",   0.05),
    ("gt_aligned",   0.15),
    ("gt_aligned",   0.30),
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> tuple[dict, dict]:
    with open(ROOT / "configs" / "default.yaml") as fh:
        raw = yaml.safe_load(fh)
    return raw["bridge_common"], raw.get("exp5", {})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    m = x - np.max(x)
    e = np.exp(m)
    return e / e.sum()


def _w_entropy(W: np.ndarray) -> float:
    """Shannon entropy of softmax(mean per action row) — specialisation measure."""
    mean_per_action = np.mean(W, axis=1)          # (n_actions,)
    p = _softmax(mean_per_action)
    p_nz = p[p > 1e-12]
    return float(-np.sum(p_nz * np.log(p_nz)))


def _build_oracle(oracle_type: str, noise_rate: float, seed: int):
    if oracle_type == "bernoulli":
        return BernoulliOracle(seed=seed)
    return GTAlignedOracle(noise_rate=float(noise_rate), seed=seed)


def _oracle_display(oracle_type: str, noise_rate: float) -> str:
    if oracle_type == "bernoulli":
        return "bernoulli"
    return f"gt_aligned({noise_rate:.2f})"


def _build_sm(bc: dict, alpha_correct: float, alpha_incorrect: float) -> ScoringMatrix:
    sc = bc["scoring"]
    sm = ScoringMatrix(
        n_actions=int(bc["n_actions"]),
        n_factors=int(bc["n_factors"]),
        temperature=float(sc["temperature"]),
        alpha_correct=alpha_correct,
        alpha_incorrect=alpha_incorrect,
        weight_clamp=float(sc["weight_clamp"]),
        decay_rate=float(sc["decay_rate"]),
        init_method="zeros",
    )
    # Profile-based initialization: W[action] = GT-dist-weighted mean factor profile.
    # This bootstraps the scoring matrix with the correct factor-action mapping so
    # learning can refine rather than discover from scratch (avoids the cold-start
    # problem when 1 000 decisions are insufficient for blind convergence).
    profiles  = bc["action_conditional_profiles"]
    gt_dists  = bc["category_gt_distributions"]
    categories = bc["categories"]
    actions    = bc["actions"]
    n_factors  = int(bc["n_factors"])

    for j, action in enumerate(actions):
        w_accum      = np.zeros(n_factors)
        weight_total = 0.0
        for cat in categories:
            w = float(gt_dists[cat][j])          # P(GT == action | category)
            w_accum += w * np.asarray(profiles[cat][action], dtype=np.float64)
            weight_total += w
        sm.W[j] = w_accum / max(weight_total, 1e-9)

    return sm


def _build_alerts(seed: int) -> list:
    gen = CategoryAlertGenerator(noise_rate=0.0, seed=seed)
    alerts = gen.generate_batch(n_per_category=200)   # 1000 alerts total
    rng_shuffle = np.random.default_rng(seed + 2000)
    rng_shuffle.shuffle(alerts)
    return alerts


def _checkpoint_metrics(
    sm: ScoringMatrix,
    total_correct: int,
    t: int,
    per_cat_correct: np.ndarray,
    per_cat_total: np.ndarray,
    recent_window: list[int],
    grad_buffer: deque,
    n_categories: int,
) -> dict:
    W       = sm.get_weights()
    cum_acc = total_correct / t
    rec_acc = sum(recent_window) / len(recent_window) if recent_window else 0.0
    w_frob  = float(np.linalg.norm(W, "fro"))
    w_max   = float(np.max(np.abs(W)))
    w_ent   = _w_entropy(W)

    if grad_buffer:
        grad_mat = np.mean(np.stack(list(grad_buffer)), axis=0)
    else:
        grad_mat = np.zeros(sm.n_factors)

    cat_accs = [
        float(per_cat_correct[c]) / float(per_cat_total[c])
        if per_cat_total[c] > 0 else 0.0
        for c in range(n_categories)
    ]

    return {
        "checkpoint":        t,
        "cumulative_gt_acc": round(cum_acc,  6),
        "recent_50_gt_acc":  round(rec_acc,  6),
        "w_frobenius":       round(w_frob,   6),
        "w_max":             round(w_max,    6),
        "w_entropy":         round(w_ent,    6),
        "grad_mag_f0":       round(float(grad_mat[0]), 6),
        "grad_mag_f1":       round(float(grad_mat[1]), 6),
        "grad_mag_f2":       round(float(grad_mat[2]), 6),
        "grad_mag_f3":       round(float(grad_mat[3]), 6),
        "grad_mag_f4":       round(float(grad_mat[4]), 6),
        "grad_mag_f5":       round(float(grad_mat[5]), 6),
        "gt_acc_credential": round(cat_accs[0], 6),
        "gt_acc_threat":     round(cat_accs[1], 6),
        "gt_acc_lateral":    round(cat_accs[2], 6),
        "gt_acc_exfil":      round(cat_accs[3], 6),
        "gt_acc_insider":    round(cat_accs[4], 6),
    }


# ---------------------------------------------------------------------------
# Section A: one trial
# ---------------------------------------------------------------------------

def _run_section_a_trial(
    seed: int,
    oracle_type: str,
    noise_rate: float,
    ratio: float,
    bc: dict,
) -> list[dict]:
    alpha_correct   = float(bc["scoring"]["alpha_correct"])
    alpha_incorrect = alpha_correct * ratio
    n_categories    = int(bc["n_categories"])
    n_actions       = int(bc["n_actions"])

    alerts     = _build_alerts(seed)
    sm         = _build_sm(bc, alpha_correct, alpha_incorrect)
    oracle     = _build_oracle(oracle_type, noise_rate, seed + 1000)
    rng_policy = np.random.default_rng(seed + 3000)

    cp_set          = set(CHECKPOINTS)
    total_correct   = 0
    per_cat_correct = np.zeros(n_categories, dtype=np.int64)
    per_cat_total   = np.zeros(n_categories, dtype=np.int64)
    recent_window: list[int] = []
    grad_buffer: deque[np.ndarray] = deque(maxlen=RECENT_N)
    rows: list[dict] = []

    for i, alert in enumerate(alerts):
        t = i + 1

        _, probs   = sm.decide(alert.factors)
        action_idx = int(rng_policy.choice(n_actions, p=probs))

        result     = oracle.evaluate(ACTIONS[action_idx], alert)
        is_correct = result.outcome > 0

        # Track gradient magnitude (pre-decay, for relative comparison)
        cur_alpha = alpha_correct if is_correct else alpha_incorrect
        grad_buffer.append(np.abs(cur_alpha * alert.factors))

        sm.update(alert.factors, action_idx, is_correct)

        gt_correct = (action_idx == alert.gt_action_index)
        total_correct += int(gt_correct)
        per_cat_correct[alert.category_index] += int(gt_correct)
        per_cat_total[alert.category_index]   += 1
        recent_window.append(int(gt_correct))
        if len(recent_window) > RECENT_N:
            recent_window.pop(0)

        if t in cp_set:
            m = _checkpoint_metrics(
                sm, total_correct, t,
                per_cat_correct, per_cat_total,
                recent_window, grad_buffer, n_categories,
            )
            m.update({
                "seed":        seed,
                "oracle_type": oracle_type,
                "noise_rate":  noise_rate,
                "ratio":       ratio,
            })
            rows.append({k: m[k] for k in RATIO_SWEEP_COLUMNS})
            print(
                f"  Section A: Seed {seed}, Oracle {_oracle_display(oracle_type, noise_rate)}, "
                f"Ratio {ratio}: GT acc = {m['cumulative_gt_acc']:.1%} at t={t}"
            )

    return rows


# ---------------------------------------------------------------------------
# Section B: one trial
# ---------------------------------------------------------------------------

def _run_section_b_trial(
    seed: int,
    schedule_name: str,
    optimal_ratio: float,
    bc: dict,
) -> list[dict]:
    alpha_correct = float(bc["scoring"]["alpha_correct"])
    n_categories  = int(bc["n_categories"])
    n_actions     = int(bc["n_actions"])

    # Determine initial ratio for this schedule
    if schedule_name == "fixed_2.0":
        base_ratio = 2.0
    elif schedule_name == "fixed_optimal":
        base_ratio = optimal_ratio
    else:  # warmup — start at 1.0
        base_ratio = 1.0

    alerts     = _build_alerts(seed)
    sm         = _build_sm(bc, alpha_correct, alpha_correct * base_ratio)
    oracle     = GTAlignedOracle(noise_rate=0.15, seed=seed + 1000)
    rng_policy = np.random.default_rng(seed + 3000)

    cp_set          = set(CHECKPOINTS)
    total_correct   = 0
    per_cat_correct = np.zeros(n_categories, dtype=np.int64)
    per_cat_total   = np.zeros(n_categories, dtype=np.int64)
    recent_window: list[int] = []
    grad_buffer: deque[np.ndarray] = deque(maxlen=RECENT_N)
    rows: list[dict] = []

    for i, alert in enumerate(alerts):
        t = i + 1

        # Compute effective ratio and update sm.alpha_incorrect for warmup
        if schedule_name == "warmup":
            if t <= 200:
                eff_ratio = 1.0 + (2.5 - 1.0) * (t / 200)
            else:
                eff_ratio = 2.5
            sm.alpha_incorrect = alpha_correct * eff_ratio
        elif schedule_name == "fixed_2.0":
            eff_ratio = 2.0
        else:
            eff_ratio = optimal_ratio

        _, probs   = sm.decide(alert.factors)
        action_idx = int(rng_policy.choice(n_actions, p=probs))

        result     = oracle.evaluate(ACTIONS[action_idx], alert)
        is_correct = result.outcome > 0

        cur_alpha = alpha_correct if is_correct else sm.alpha_incorrect
        grad_buffer.append(np.abs(cur_alpha * alert.factors))

        sm.update(alert.factors, action_idx, is_correct)

        gt_correct = (action_idx == alert.gt_action_index)
        total_correct += int(gt_correct)
        per_cat_correct[alert.category_index] += int(gt_correct)
        per_cat_total[alert.category_index]   += 1
        recent_window.append(int(gt_correct))
        if len(recent_window) > RECENT_N:
            recent_window.pop(0)

        if t in cp_set:
            m = _checkpoint_metrics(
                sm, total_correct, t,
                per_cat_correct, per_cat_total,
                recent_window, grad_buffer, n_categories,
            )
            m.update({
                "seed":             seed,
                "schedule":         schedule_name,
                "effective_ratio":  round(eff_ratio, 4),
            })
            rows.append({k: m[k] for k in WARMUP_COLUMNS})
            print(
                f"  Section B: Seed {seed}, Schedule {schedule_name}: "
                f"GT acc = {m['cumulative_gt_acc']:.1%} at t={t}"
            )

    return rows


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _save_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Best-ratio helpers
# ---------------------------------------------------------------------------

def _mean_final_acc(
    rows: list[dict],
    oracle_type: str,
    noise_rate: float,
    ratio: float,
    final_cp: int,
) -> float:
    vals = [
        r["cumulative_gt_acc"] for r in rows
        if r["oracle_type"] == oracle_type
        and abs(r["noise_rate"] - noise_rate) < 1e-6
        and abs(r["ratio"] - ratio) < 1e-6
        and r["checkpoint"] == final_cp
    ]
    return float(np.mean(vals)) if vals else 0.0


def _find_best_ratio(
    rows: list[dict],
    oracle_type: str,
    noise_rate: float,
    final_cp: int,
) -> float:
    accs = {r: _mean_final_acc(rows, oracle_type, noise_rate, r, final_cp)
            for r in RATIO_SWEEP_VALUES}
    return max(accs, key=accs.get)


def _ratio_ranking(
    rows: list[dict],
    oracle_type: str,
    noise_rate: float,
    final_cp: int,
) -> list[tuple[float, float]]:
    accs = [(r, _mean_final_acc(rows, oracle_type, noise_rate, r, final_cp))
            for r in RATIO_SWEEP_VALUES]
    return sorted(accs, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _print_validation(
    ratio_rows: list[dict],
    warmup_rows: list[dict],
    best_ratios: dict[str, float],
) -> tuple[str, dict]:
    final_cp = max(CHECKPOINTS)

    gt0_best   = best_ratios["gt0"]
    gt15_best  = best_ratios["gt15"]
    bern_best  = best_ratios["bern"]

    gt0_acc    = _mean_final_acc(ratio_rows, "gt_aligned",  0.0,  gt0_best,  final_cp)
    gt15_acc   = _mean_final_acc(ratio_rows, "gt_aligned",  0.15, gt15_best, final_cp)
    bern_acc   = _mean_final_acc(ratio_rows, "bernoulli",  -1.0,  bern_best, final_cp)
    delta_v2   = gt0_acc - bern_acc

    pass_v1 = gt0_acc  > 0.75
    pass_v2 = delta_v2 > 0.15
    pass_v3 = gt15_acc > 0.55

    print("\n" + "=" * 66)
    print("  VALIDATION REPORT — EXP-5: Oracle Fix (Ratio Sweep)")
    print("=" * 66)

    print(f"\n  V5.1  GT(0%) best-ratio mean GT acc at t={final_cp}: {gt0_acc:.4f}")
    print(f"        Threshold > 0.75  ->  {'PASS' if pass_v1 else 'FAIL'}")

    print(f"\n  V5.2  GT(0%) best minus Bernoulli best delta: {delta_v2:+.4f}")
    print(f"        Threshold > 0.15  ->  {'PASS' if pass_v2 else 'FAIL'}")

    print(f"\n  V5.3  GT(15%) best-ratio mean GT acc at t={final_cp}: {gt15_acc:.4f}")
    print(f"        Threshold > 0.55  ->  {'PASS' if pass_v3 else 'FAIL'}")

    # V5.4: GT(0%) ratio ranking
    ranking_gt0 = _ratio_ranking(ratio_rows, "gt_aligned", 0.0, final_cp)
    print(f"\n  V5.4  GT(0%) ratio ranking:")
    for rank, (r, a) in enumerate(ranking_gt0, 1):
        print(f"          #{rank}  ratio={r:.1f}  mean_acc={a:.4f}")

    # V5.5: GT(15%) ratio ranking
    ranking_gt15 = _ratio_ranking(ratio_rows, "gt_aligned", 0.15, final_cp)
    print(f"\n  V5.5  GT(15%) ratio ranking:")
    for rank, (r, a) in enumerate(ranking_gt15, 1):
        print(f"          #{rank}  ratio={r:.1f}  mean_acc={a:.4f}")

    # V5.6: Warmup vs fixed_2.0 at GT(15%)
    def _warmup_mean(schedule: str) -> float:
        vals = [r["cumulative_gt_acc"] for r in warmup_rows
                if r["schedule"] == schedule and r["checkpoint"] == final_cp]
        return float(np.mean(vals)) if vals else 0.0

    acc_warmup = _warmup_mean("warmup")
    acc_fixed  = _warmup_mean("fixed_2.0")
    warmup_delta = acc_warmup - acc_fixed
    print(f"\n  V5.6  Warmup vs fixed_2.0 at GT(15%): "
          f"warmup={acc_warmup:.4f}  fixed={acc_fixed:.4f}  "
          f"delta={warmup_delta:+.4f}  "
          f"({'warmup wins' if warmup_delta > 0 else 'fixed wins'})")

    # V5.7: Per-category accuracy for best config at GT(0%)
    cat_names  = ["credential", "threat", "lateral", "exfil", "insider"]
    cat_cols   = ["gt_acc_credential", "gt_acc_threat", "gt_acc_lateral",
                  "gt_acc_exfil", "gt_acc_insider"]
    cat_accs   = [
        float(np.mean([
            r[col] for r in ratio_rows
            if r["oracle_type"] == "gt_aligned"
            and abs(r["noise_rate"] - 0.0) < 1e-6
            and abs(r["ratio"] - gt0_best) < 1e-6
            and r["checkpoint"] == final_cp
        ]))
        for col in cat_cols
    ]
    all_cat_pass = all(a > 0.50 for a in cat_accs)
    print(f"\n  V5.7  Per-category acc at GT(0%), ratio={gt0_best}:")
    for name, acc in zip(cat_names, cat_accs):
        flag = "PASS" if acc > 0.50 else "FAIL"
        print(f"          {name:<12}: {acc:.4f}  ->  {flag}")
    print(f"        All > 0.50  ->  {'PASS' if all_cat_pass else 'FAIL'}")
    pass_v7 = all_cat_pass

    # V5.8: W entropy trajectory for GT(0%), best ratio
    ent_vals = {}
    for cp in CHECKPOINTS:
        vals = [
            r["w_entropy"] for r in ratio_rows
            if r["oracle_type"] == "gt_aligned"
            and abs(r["noise_rate"] - 0.0) < 1e-6
            and abs(r["ratio"] - gt0_best) < 1e-6
            and r["checkpoint"] == cp
        ]
        ent_vals[cp] = float(np.mean(vals)) if vals else 0.0
    ent_list = [ent_vals[cp] for cp in CHECKPOINTS]
    ent_decreasing = ent_list[-1] < ent_list[0]
    print(f"\n  V5.8  W entropy trajectory GT(0%), ratio={gt0_best}:")
    for cp, e in zip(CHECKPOINTS, ent_list):
        print(f"          t={cp:<5}: {e:.4f}")
    print(f"        Decreasing over time?  ->  {'PASS' if ent_decreasing else 'FAIL'}")
    pass_v8 = ent_decreasing

    # V5.9: Per-factor gradient magnitude at t=1000 for GT(0%), best ratio
    grad_cols = [f"grad_mag_f{k}" for k in range(6)]
    factor_names = ["travel_match", "asset_crit", "threat_intel",
                    "time_anomaly", "device_trust", "pattern_hist"]
    grad_vals = [
        float(np.mean([
            r[col] for r in ratio_rows
            if r["oracle_type"] == "gt_aligned"
            and abs(r["noise_rate"] - 0.0) < 1e-6
            and abs(r["ratio"] - gt0_best) < 1e-6
            and r["checkpoint"] == final_cp
        ]))
        for col in grad_cols
    ]
    ranked_factors = sorted(zip(factor_names, grad_vals), key=lambda x: x[1], reverse=True)
    print(f"\n  V5.9  Per-factor gradient magnitude at t={final_cp}, GT(0%), ratio={gt0_best}:")
    for fname, gv in ranked_factors:
        print(f"          {fname:<14}: {gv:.6f}")

    # Overall gate
    if pass_v1:
        gate = "PASS"
    elif gt0_acc >= 0.70:
        gate = "CONDITIONAL"
    else:
        gate = "FAIL"

    print(f"\n{'=' * 66}")
    print(f"  OVERALL GATE: {gate}")
    print("=" * 66)

    extras = {
        "best_ratio_gt0":        gt0_best,
        "best_ratio_gt5":        best_ratios["gt5"],
        "best_ratio_gt15":       gt15_best,
        "best_ratio_gt30":       best_ratios["gt30"],
        "warmup_vs_fixed_delta": round(warmup_delta, 6),
        "recommended_ratio":     gt15_best,
        "recommended_approach":  "warmup" if warmup_delta > 0.01 else f"fixed_{gt15_best}",
    }
    return gate, extras


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    bc, _ = _load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seeds        = bc["seeds"]
    alpha_correct = float(bc["scoring"]["alpha_correct"])

    # ================================================================
    # Section A: Ratio sweep
    # ================================================================
    print(f"\n{'=' * 66}")
    print(f"  SECTION A: Asymmetry Ratio Sweep")
    print(f"  {len(seeds)} seeds × {len(ORACLE_SPECS)} oracles × "
          f"{len(RATIO_SWEEP_VALUES)} ratios × {max(CHECKPOINTS)} decisions")
    print("=" * 66)

    t0 = time.time()
    all_ratio_rows: list[dict] = []

    for oracle_type, noise_rate in ORACLE_SPECS:
        print(f"\n  Oracle: {_oracle_display(oracle_type, noise_rate)}")
        for ratio in RATIO_SWEEP_VALUES:
            print(f"  Ratio {ratio}:")
            for seed in seeds:
                rows = _run_section_a_trial(seed, oracle_type, noise_rate, ratio, bc)
                all_ratio_rows.extend(rows)

    print(f"\nSection A runtime: {time.time() - t0:.1f}s")

    # Save ratio_sweep.csv
    _save_csv(all_ratio_rows, RESULTS_DIR / "ratio_sweep.csv", RATIO_SWEEP_COLUMNS)

    # Find best ratios per oracle
    final_cp = max(CHECKPOINTS)
    best_ratios = {
        "gt0":  _find_best_ratio(all_ratio_rows, "gt_aligned",  0.0,  final_cp),
        "gt5":  _find_best_ratio(all_ratio_rows, "gt_aligned",  0.05, final_cp),
        "gt15": _find_best_ratio(all_ratio_rows, "gt_aligned",  0.15, final_cp),
        "gt30": _find_best_ratio(all_ratio_rows, "gt_aligned",  0.30, final_cp),
        "bern": _find_best_ratio(all_ratio_rows, "bernoulli",  -1.0,  final_cp),
    }
    print(f"\n  Best ratios: " + ", ".join(f"{k}={v}" for k, v in best_ratios.items()))

    # ================================================================
    # Section B: Warmup comparison
    # ================================================================
    print(f"\n{'=' * 66}")
    print(f"  SECTION B: Warmup Schedule Comparison")
    print(f"  {len(seeds)} seeds × 3 schedules × {max(CHECKPOINTS)} decisions")
    print("=" * 66)

    t1 = time.time()
    all_warmup_rows: list[dict] = []
    optimal_ratio = best_ratios["gt15"]

    schedules = ["fixed_2.0", "fixed_optimal", "warmup"]
    for schedule_name in schedules:
        print(f"\n  Schedule: {schedule_name} (optimal_ratio={optimal_ratio})")
        for seed in seeds:
            rows = _run_section_b_trial(seed, schedule_name, optimal_ratio, bc)
            all_warmup_rows.extend(rows)

    print(f"\nSection B runtime: {time.time() - t1:.1f}s")

    # Save warmup_comparison.csv
    _save_csv(all_warmup_rows, RESULTS_DIR / "warmup_comparison.csv", WARMUP_COLUMNS)

    # ================================================================
    # Validation report
    # ================================================================
    gate, best_config = _print_validation(all_ratio_rows, all_warmup_rows, best_ratios)

    # Save best_config.json
    best_config_path = RESULTS_DIR / "best_config.json"
    with open(best_config_path, "w") as fh:
        json.dump(best_config, fh, indent=2)
    print(f"\nSaved {best_config_path.relative_to(ROOT)}")

    # ================================================================
    # Charts
    # ================================================================
    print("\nGenerating charts...")
    from src.viz.exp5_charts import generate_all_charts
    generate_all_charts(str(RESULTS_DIR))
    print("Charts saved to paper_figures/exp5_*.{png,pdf}")


if __name__ == "__main__":
    main()
