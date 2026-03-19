"""
EXP-OP2-ISOLATE: 2×2 Factorial — Isolating Bug Effects from Config Effects

Context: The recheck (expOP2_recheck) showed NR rates of 86-99%, far higher
than the original OP2's 35-38%. This experiment answers WHY by holding the
config constant (soc_product_v50, C=6, A=5, d=6) and varying only the two bugs.

2×2 factorial:
  Factor 1: η_neg   (0.05 = fixed  vs  1.0 = legacy)
  Factor 2: update  (gt_action_index passed = fixed  vs  None = legacy)

Arms:
  A: η_neg=0.05, gt_passed=True   (both fixed)        ← matches recheck
  B: η_neg=0.05, gt_passed=False  (η_neg fixed only)
  C: η_neg=1.0,  gt_passed=True   (update fixed only)
  D: η_neg=1.0,  gt_passed=False  (both bugs present)  ← approximates original

Operator model (different from recheck):
  - Scoring always uses "correct" SynthesisBias during injection window (TTL_HALF)
  - op_accuracy controls DECISION QUALITY: with prob op_accuracy, use GT action;
    otherwise use sigma-biased action for the update step.
  - This cleanly controls centroid corruption rate independent of sigma quality.

NR metric: fraction of (c,a) centroid cells with L2 drift > 0.05 from
  pre-injection baseline (measured after all N_POST decisions).
  This directly quantifies centroid corruption, not accuracy recovery.

Post-accuracy: mean accuracy during post-expiry phase (dec TTL_HALF..N_POST).
  Used as proxy for c_exp.
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias

# ---------------------------------------------------------------------------
# Load soc_product_v50
# ---------------------------------------------------------------------------
with open(REPO_ROOT / "configs" / "soc_product_v50.yaml") as _f:
    _CFG = yaml.safe_load(_f)

CATEGORIES = _CFG["categories"]
ACTIONS    = _CFG["actions"]
FACTORS    = _CFG["factors"]
PROFILES   = _CFG["action_conditional_profiles"]
GT_DISTS   = _CFG["category_gt_distributions"]

C_DIM     = len(CATEGORIES)   # 6
A_DIM     = len(ACTIONS)      # 5
N_FACTORS = len(FACTORS)      # 6

ESC_IDX  = ACTIONS.index("escalate")   # 0
SUPP_IDX = ACTIONS.index("suppress")  # 2

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ARMS = {
    "A_both_fixed":        {"eta_neg": 0.05, "pass_gt": True},
    "B_eta_fixed_only":    {"eta_neg": 0.05, "pass_gt": False},
    "C_update_fixed_only": {"eta_neg": 1.0,  "pass_gt": True},
    "D_both_legacy":       {"eta_neg": 1.0,  "pass_gt": False},
}

N_SEEDS             = 50
OPERATOR_ACCURACIES = [0.0, 0.50, 0.75, 1.00]
N_PRE               = 200
N_POST              = 400
TTL_HALF            = 150
N_POST_EXP          = N_POST - TTL_HALF   # 250  post-expiry decisions
TAU                 = 0.1
ETA                 = 0.05
LAMBDA_S            = 0.5
SIGMA_VALUE         = 0.4
SEED_BASE           = 42000
DRIFT_THRESHOLD     = 0.05   # centroid L2 drift threshold for NR cell count

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Warm-start mu
# ---------------------------------------------------------------------------
def build_mu() -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = PROFILES[cat][act]
    return mu

MU_WARM = build_mu()

# ---------------------------------------------------------------------------
# Sigma — correct sigma used during injection (same dual-action as recheck)
# ---------------------------------------------------------------------------
SIGMA_CORRECT = np.zeros((C_DIM, A_DIM), dtype=np.float64)
SIGMA_CORRECT[:, SUPP_IDX] = +SIGMA_VALUE   # suppress less likely
SIGMA_CORRECT[:, ESC_IDX]  = -SIGMA_VALUE   # escalate more likely

SYNTHESIS_CORRECT = SynthesisBias(
    sigma=SIGMA_CORRECT, active_claims=1, lambda_coupling=LAMBDA_S,
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# results[key] = dict with nr_rate, nr_ci, post_accuracy, etc.
results: dict[str, dict] = {}

# Full per-seed arrays for charts
raw: dict[str, dict] = {
    arm: {
        f"op{int(op*100)}": {"nr_cells": [], "post_acc": []}
        for op in OPERATOR_ACCURACIES
    }
    for arm in ARMS
}

# ---------------------------------------------------------------------------
# Suppress DeprecationWarnings from legacy update() arms (B and D)
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="update\\(\\) called with correct=False")

# ---------------------------------------------------------------------------
# Main loop: arm × op_accuracy × seed
# ---------------------------------------------------------------------------
print("=== EXP-OP2-ISOLATE: 2×2 FACTORIAL ===")
print(f"Config: soc_product_v50  C={C_DIM} A={A_DIM} d={N_FACTORS}")
print(f"N_SEEDS={N_SEEDS}  N_PRE={N_PRE}  TTL_HALF={TTL_HALF}  N_POST={N_POST}")
print(f"Operator accuracies: {OPERATOR_ACCURACIES}")
print()

for arm_name, arm_cfg in ARMS.items():
    eta_neg = arm_cfg["eta_neg"]
    pass_gt = arm_cfg["pass_gt"]
    print(f"--- Arm {arm_name} (η_neg={eta_neg}, pass_gt={pass_gt}) ---")

    for op_acc in OPERATOR_ACCURACIES:
        key = f"{arm_name}_op{int(op_acc * 100)}"
        op_key = f"op{int(op_acc * 100)}"

        nr_cells_list: list[float] = []
        post_acc_list: list[float] = []

        for seed_idx in range(N_SEEDS):
            seed = SEED_BASE + seed_idx
            rng  = np.random.default_rng(seed + 99999)   # for op_accuracy decisions

            gen = CategoryAlertGenerator(
                categories=CATEGORIES, actions=ACTIONS, factors=FACTORS,
                action_conditional_profiles=PROFILES, gt_distributions=GT_DISTS,
                seed=seed,
            )

            scorer = ProfileScorer(
                MU_WARM.copy(), A_DIM,
                tau=TAU, eta=ETA, eta_neg=eta_neg, seed=seed,
            )

            # --- Phase 1: Pre-injection warmup (N_PRE decisions, no operator) ---
            pre_alerts = gen.generate(N_PRE)
            for alert in pre_alerts:
                result     = scorer.score(alert.factors, alert.category_index)
                is_correct = result.action_index == alert.gt_action_index
                scorer.update(
                    alert.factors, alert.category_index,
                    result.action_index, is_correct,
                    gt_action_index=alert.gt_action_index if pass_gt else None,
                )

            # Record baseline centroids after warmup
            baseline_mu = scorer.mu.copy()

            # --- Phase 2: Injection (TTL_HALF decisions with correct sigma active) ---
            inject_alerts = gen.generate(TTL_HALF)
            for alert in inject_alerts:
                # Score with correct synthesis bias
                result = scorer.score(
                    alert.factors, alert.category_index,
                    synthesis=SYNTHESIS_CORRECT,
                )

                # Operator decision quality
                if rng.random() < op_acc:
                    chosen_action = alert.gt_action_index   # correct feedback
                else:
                    chosen_action = result.action_index     # sigma-biased (may be wrong)

                is_correct = (chosen_action == alert.gt_action_index)
                scorer.update(
                    alert.factors, alert.category_index,
                    chosen_action, is_correct,
                    gt_action_index=alert.gt_action_index if pass_gt else None,
                )

            # --- Phase 3: Post-expiry (N_POST_EXP decisions, no operator) ---
            post_alerts = gen.generate(N_POST_EXP)
            post_correct = 0
            for alert in post_alerts:
                result     = scorer.score(alert.factors, alert.category_index)
                is_correct = result.action_index == alert.gt_action_index
                post_correct += int(is_correct)
                scorer.update(
                    alert.factors, alert.category_index,
                    result.action_index, is_correct,
                    gt_action_index=alert.gt_action_index if pass_gt else None,
                )

            post_acc = post_correct / N_POST_EXP

            # --- NR metric: fraction of (c,a) cells with drift > threshold ---
            n_cells = C_DIM * A_DIM
            drift_count = 0
            for c in range(C_DIM):
                for a in range(A_DIM):
                    dist = float(np.linalg.norm(scorer.mu[c, a] - baseline_mu[c, a]))
                    if dist > DRIFT_THRESHOLD:
                        drift_count += 1
            nr_cell_rate = drift_count / n_cells

            nr_cells_list.append(nr_cell_rate)
            post_acc_list.append(post_acc)

        # --- Aggregate ---
        nr_arr  = np.array(nr_cells_list)
        acc_arr = np.array(post_acc_list)

        mean_nr = float(nr_arr.mean())
        std_nr  = float(nr_arr.std())
        ci_nr   = 1.96 * std_nr / np.sqrt(N_SEEDS)

        mean_acc = float(acc_arr.mean())
        std_acc  = float(acc_arr.std())
        ci_acc   = 1.96 * std_acc / np.sqrt(N_SEEDS)

        results[key] = {
            "arm":          arm_name,
            "op_accuracy":  op_acc,
            "eta_neg":      eta_neg,
            "pass_gt":      pass_gt,
            "nr_rate":      round(mean_nr, 4),
            "nr_ci":        round(ci_nr,  4),
            "nr_std":       round(std_nr, 4),
            "post_accuracy": round(mean_acc, 4),
            "post_acc_ci":  round(ci_acc, 4),
        }
        raw[arm_name][op_key]["nr_cells"] = nr_cells_list
        raw[arm_name][op_key]["post_acc"] = post_acc_list

        print(f"  op={op_acc:.2f}  NR={mean_nr:.1%} ±{ci_nr:.1%}  "
              f"post_acc={mean_acc:.3f} ±{ci_acc:.3f}")

    print()

# ---------------------------------------------------------------------------
# Factorial analysis table
# ---------------------------------------------------------------------------
print("=" * 80)
print("=== 2×2 FACTORIAL: ISOLATING BUG EFFECTS ===")
print("=" * 80)
print(f"\n{'Arm':<25} | {'η_neg':>6} | {'GT':>5} | "
      f"{'NR(0%)':>7} | {'NR(50%)':>7} | {'NR(75%)':>7} | {'NR(100%)':>8}")
print("-" * 80)
for arm_name, arm_cfg in ARMS.items():
    nr_0   = results[f"{arm_name}_op0"]["nr_rate"]
    nr_50  = results[f"{arm_name}_op50"]["nr_rate"]
    nr_75  = results[f"{arm_name}_op75"]["nr_rate"]
    nr_100 = results[f"{arm_name}_op100"]["nr_rate"]
    print(f"  {arm_name:<23} | {arm_cfg['eta_neg']:>6} | "
          f"{'yes' if arm_cfg['pass_gt'] else 'no':>5} | "
          f"{nr_0:>6.1%} | {nr_50:>6.1%} | {nr_75:>6.1%} | {nr_100:>7.1%}")

# ---------------------------------------------------------------------------
# Effect decomposition at each operator accuracy level
# ---------------------------------------------------------------------------
print()
print("=== EFFECT DECOMPOSITION ===")
print(f"{'Op acc':>6}  {'η effect':>10}  {'update bug':>10}  {'interaction':>12}  {'total':>8}")
print("-" * 55)

decomp: dict[str, dict] = {}
for op_acc in OPERATOR_ACCURACIES:
    op_key = f"op{int(op_acc * 100)}"
    nr_A = results[f"A_both_fixed_{op_key}"]["nr_rate"]
    nr_B = results[f"B_eta_fixed_only_{op_key}"]["nr_rate"]
    nr_C = results[f"C_update_fixed_only_{op_key}"]["nr_rate"]
    nr_D = results[f"D_both_legacy_{op_key}"]["nr_rate"]

    eta_effect    = nr_C - nr_A          # η_neg 0.05→1.0, gt fixed
    update_effect = nr_B - nr_A          # gt None vs passed, η_neg=0.05
    interaction   = nr_D - nr_A - eta_effect - update_effect
    total         = nr_D - nr_A

    decomp[op_key] = {
        "eta_effect": eta_effect, "update_effect": update_effect,
        "interaction": interaction, "total": total,
        "nr_A": nr_A, "nr_B": nr_B, "nr_C": nr_C, "nr_D": nr_D,
    }
    print(f"  {op_acc:>4.0%}  {eta_effect:>+10.1%}  {update_effect:>+10.1%}  "
          f"{interaction:>+12.1%}  {total:>+8.1%}")

# Dominant effect at 0% operator accuracy (worst case)
print()
d0 = decomp["op0"]
print(f"At 0% operator accuracy (worst case, N={N_SEEDS}):")
print(f"  η_neg effect:      {d0['eta_effect']:+.1%}  (NR goes from {d0['nr_A']:.1%} to {d0['nr_C']:.1%})")
print(f"  Update bug effect: {d0['update_effect']:+.1%}  (NR goes from {d0['nr_A']:.1%} to {d0['nr_B']:.1%})")
print(f"  Interaction:       {d0['interaction']:+.1%}")
print(f"  Total (D - A):     {d0['total']:+.1%}  ({d0['nr_A']:.1%} → {d0['nr_D']:.1%})")

if abs(d0["eta_effect"]) > abs(d0["update_effect"]):
    print("\n  → η_neg is the DOMINANT effect")
    print("    The 38%→86% NR increase in recheck was primarily driven by")
    print("    the soc_product_v50 config change (different action distribution),")
    print("    not by the η_neg fix alone.")
else:
    print("\n  → Update bug is the DOMINANT effect")
    print("    Missing gt_action_index was the primary driver of NR rates.")

# Sanity check: op_acc=1.0 should show near-zero NR for all arms
print()
print("Sanity check (op_acc=100%, all feedback correct — all arms should match):")
for arm_name in ARMS:
    nr_100 = results[f"{arm_name}_op100"]["nr_rate"]
    print(f"  {arm_name}: NR={nr_100:.1%}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _to_json(obj):
    if isinstance(obj, float): return obj
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):  return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_to_json(v) for v in obj]
    return obj

with open(OUT_DIR / "isolate_results.json", "w") as f:
    json.dump(_to_json(results), f, indent=2)
with open(OUT_DIR / "decomp.json", "w") as f:
    json.dump(_to_json(decomp), f, indent=2)

print()
print(f"Results saved → {OUT_DIR / 'isolate_results.json'}")
print(f"Decomp saved  → {OUT_DIR / 'decomp.json'}")

from experiments.operator.expOP2_isolate.charts import make_charts
make_charts(results, decomp, raw, N_SEEDS)

print()
print("=== EXP-OP2-ISOLATE COMPLETE ===")
