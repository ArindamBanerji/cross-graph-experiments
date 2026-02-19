"""
Experiment 1: Scoring Matrix Convergence

Validates Eq. 4: P(action | alert) = softmax(f @ W.T / tau)

Compares asymmetric compounding learning against four baselines across
10 random seeds, measuring how quickly the scoring matrix specialises to
the correct action for each alert type.

Outputs
-------
results/convergence_data.csv   — per-checkpoint accuracy for all methods/seeds
results/weight_evolution.npz   — W snapshots at each checkpoint (compounding only)
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup — make src/ importable from anywhere
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.alert_generator import AlertGenerator, ACTION_NAMES
from src.models.scoring_matrix import ScoringMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CSV_COLUMNS = [
    "method", "seed", "checkpoint",
    "cumulative_accuracy", "window_accuracy",
    "action_0_accuracy", "action_1_accuracy",
    "action_2_accuracy", "action_3_accuracy",
]

# Ordered for consistent summary output
BASELINE_ORDER = [
    "compounding",
    "fixed_weight",
    "random_policy",
    "periodic_retrain",
    "symmetric",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = ROOT / "configs" / "default.yaml"
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["experiment_1"]


# ---------------------------------------------------------------------------
# Matrix factory
# ---------------------------------------------------------------------------

def _build_matrix(baseline_name: str, baseline_cfg: dict, cfg: dict) -> ScoringMatrix | None:
    """
    Return a configured ScoringMatrix for the given baseline.
    Returns None for random_policy (no matrix needed).
    """
    if baseline_name == "random_policy":
        return None

    # Symmetric baseline overrides both learning rates
    alpha_c = float(baseline_cfg.get("alpha_correct",  cfg["alpha_correct"]))
    alpha_i = float(baseline_cfg.get("alpha_incorrect", cfg["alpha_incorrect"]))

    return ScoringMatrix(
        n_actions=cfg["n_actions"],
        n_factors=cfg["n_factors"],
        temperature=cfg["temperature"],
        alpha_correct=alpha_c,
        alpha_incorrect=alpha_i,
        weight_clamp=cfg["weight_clamp"],
        init_method=cfg.get("init_method", "uniform"),
        decay_rate=float(cfg.get("decay_rate", 0.0)),
    )


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def _run_one(
    baseline_name: str,
    baseline_cfg: dict,
    alerts: list,
    seed: int,
    cfg: dict,
    checkpoints: list[int],
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """
    Process all alerts for one (baseline, seed) trial.

    Returns
    -------
    rows : list[dict]
        One row per checkpoint for the CSV.
    weight_snapshots : dict[int, np.ndarray]
        {checkpoint -> W copy}  — populated only for 'compounding'.
    """
    np.random.seed(seed)   # reproducible np.random.choice for random_policy

    n_actions   = cfg["n_actions"]
    sm          = _build_matrix(baseline_name, baseline_cfg, cfg)
    reset_every = int(baseline_cfg.get("reset_interval", 500))
    cp_set      = set(checkpoints)

    do_update = baseline_name not in ("fixed_weight", "random_policy")

    # Running accumulators
    total_correct         = 0
    window_correct        = 0
    per_action_correct    = np.zeros(n_actions, dtype=np.int64)
    per_action_total      = np.zeros(n_actions, dtype=np.int64)
    prev_cp               = 0

    rows: list[dict] = []
    weight_snapshots: dict[int, np.ndarray] = {}

    for i, alert in enumerate(alerts):
        factors  = alert.factors
        true_idx = ACTION_TO_IDX[alert.ground_truth_action]

        # 1. Greedy decision — used only for accuracy tracking (evaluation signal)
        if baseline_name == "random_policy":
            greedy_idx = int(np.random.choice(n_actions))
        else:
            greedy_idx, _ = sm.decide(factors)

        # 2. Track accuracy using the greedy (argmax) recommendation
        correct_greedy = greedy_idx == true_idx
        total_correct      += int(correct_greedy)
        window_correct     += int(correct_greedy)
        per_action_total[true_idx]   += 1
        per_action_correct[true_idx] += int(correct_greedy)

        # 3. Stochastic decision — used for execution and weight update (exploration)
        #    Separates exploration from evaluation: the policy explores during
        #    learning but is assessed on its greedy recommendation.
        if do_update:
            stochastic_idx, _ = sm.decide_stochastic(factors)
            correct_stochastic = stochastic_idx == true_idx
            sm.update(factors, stochastic_idx, correct_stochastic)

        current_count = i + 1

        # 4. Checkpoint recording (before any periodic reset)
        if current_count in cp_set:
            window_size    = current_count - prev_cp
            window_acc     = window_correct / window_size
            cumulative_acc = total_correct  / current_count

            action_accs = [
                (per_action_correct[a] / per_action_total[a])
                if per_action_total[a] > 0 else 0.0
                for a in range(n_actions)
            ]

            rows.append({
                "method":              baseline_name,
                "seed":                seed,
                "checkpoint":          current_count,
                "cumulative_accuracy": round(cumulative_acc,    6),
                "window_accuracy":     round(window_acc,        6),
                "action_0_accuracy":   round(action_accs[0],    6),
                "action_1_accuracy":   round(action_accs[1],    6),
                "action_2_accuracy":   round(action_accs[2],    6),
                "action_3_accuracy":   round(action_accs[3],    6),
            })

            if baseline_name == "compounding":
                weight_snapshots[current_count] = sm.get_weights()

            print(
                f"  [{baseline_name}] seed={seed} "
                f"checkpoint={current_count} "
                f"accuracy={cumulative_acc * 100:.1f}%"
            )

            prev_cp        = current_count
            window_correct = 0

        # 5. Periodic reset (after checkpoint recording, not before)
        if baseline_name == "periodic_retrain" and current_count % reset_every == 0:
            sm.reset()

    return rows, weight_snapshots


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_csv(all_rows: list[dict]) -> None:
    out_path = RESULTS_DIR / "convergence_data.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved  {out_path.relative_to(ROOT)}")


def _save_weights(
    weight_data: dict[tuple[int, int], np.ndarray],
    checkpoints: list[int],
    seeds: list[int],
) -> None:
    """
    Save compounding W snapshots as a compressed NPZ.

    Array shape: (n_seeds, n_checkpoints, n_actions, n_factors)
    Keys in NPZ: 'weights', 'seeds', 'checkpoints'
    """
    if not weight_data:
        print("No compounding weight data to save — skipping NPZ.")
        return

    sample_W = next(iter(weight_data.values()))
    n_actions, n_factors = sample_W.shape
    arr = np.zeros(
        (len(seeds), len(checkpoints), n_actions, n_factors),
        dtype=np.float64,
    )
    for si, seed in enumerate(seeds):
        for ci, cp in enumerate(checkpoints):
            W = weight_data.get((seed, cp))
            if W is not None:
                arr[si, ci] = W

    out_path = RESULTS_DIR / "weight_evolution.npz"
    np.savez_compressed(
        out_path,
        weights=arr,
        seeds=np.array(seeds),
        checkpoints=np.array(checkpoints),
    )
    print(f"Saved  {out_path.relative_to(ROOT)}")


def _print_summary(all_rows: list[dict]) -> None:
    """Print method -> accuracy at 5000 decisions (mean +/- std across seeds)."""
    final_acc: dict[str, list[float]] = defaultdict(list)
    for row in all_rows:
        if row["checkpoint"] == 5000:
            final_acc[row["method"]].append(row["cumulative_accuracy"])

    print("\n" + "=" * 62)
    print("  FINAL SUMMARY - cumulative accuracy at checkpoint = 5000")
    print("=" * 62)
    print(f"  {'Method':<22}  {'Mean':>7}  {'Std':>6}  {'Min':>7}  {'Max':>7}")
    print("  " + "-" * 58)
    for method in BASELINE_ORDER:
        vals = final_acc.get(method)
        if not vals:
            continue
        a = np.array(vals)
        print(
            f"  {method:<22}  "
            f"{a.mean() * 100:6.2f}%  "
            f"{a.std()  * 100:5.2f}%  "
            f"{a.min()  * 100:6.2f}%  "
            f"{a.max()  * 100:6.2f}%"
        )
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints: list[int] = cfg["checkpoints"]
    seeds:       list[int] = cfg["seeds"]
    n_alerts:    int       = cfg["n_alerts"]
    baselines:   list      = cfg["baselines"]

    all_rows:           list[dict]                             = []
    compounding_weights: dict[tuple[int, int], np.ndarray]    = {}

    gen = AlertGenerator(cfg)
    t0  = time.time()

    for baseline in baselines:
        name = baseline["name"]
        desc = baseline.get("description", "")

        print(f"\n{'='*62}")
        print(f"  Baseline : {name}")
        print(f"  Desc     : {desc}")
        print(f"{'='*62}")

        for seed in seeds:
            alerts = gen.generate(n_alerts, seed=seed)

            rows, snaps = _run_one(
                baseline_name=name,
                baseline_cfg=baseline,
                alerts=alerts,
                seed=seed,
                cfg=cfg,
                checkpoints=checkpoints,
            )

            all_rows.extend(rows)
            for cp, W in snaps.items():
                compounding_weights[(seed, cp)] = W

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print()

    _save_csv(all_rows)
    _save_weights(compounding_weights, checkpoints, seeds)
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
