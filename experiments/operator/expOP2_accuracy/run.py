"""
EXP-OP2-ACCURACY: Accuracy Recovery vs Centroid Displacement

Motivation: EXP-OP2-ISOLATE found high centroid NR rates (51-84%) even
for Arm A (both bugs fixed). But centroid displacement != scoring failure.
A centroid can drift from its pre-injection position and still score
correctly if it remains in the correct Voronoi cell.

This experiment measures POST-EXPIRY ACCURACY (does the system make
correct decisions?) rather than centroid displacement, for the same
Arm A conditions as expOP2_isolate.

Key comparison:
  centroid_nr_isolate[op_key]  (hardcoded from expOP2_isolate)
  vs
  accuracy_nr_rate             (measured here: fraction seeds where
                                W4 accuracy < baseline - 2pp)

If accuracy_nr << centroid_nr:
  → Centroid displacement is over-sensitive
  → System WORKS even though centroids moved
  → CLAIM-17 severity is lower than displacement suggests

Parameters match expOP2_isolate Arm A:
  η_neg=0.05, gt_action_index always passed
  N_SEEDS=50, N_PRE=200, TTL_HALF=150, N_POST=400
  4 operator accuracies: [0.0, 0.50, 0.75, 1.00]
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
SUPP_IDX = ACTIONS.index("suppress")   # 2

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS             = 50
OPERATOR_ACCURACIES = [0.0, 0.50, 0.75, 1.00]
N_PRE               = 200
TTL_HALF            = 150    # injection window
N_POST              = 400    # post-expiry decisions (longer than isolate's 250)
TAU                 = 0.1
ETA                 = 0.05
ETA_NEG             = 0.05   # Arm A: both bugs fixed
LAMBDA_S            = 0.5
SIGMA_VALUE         = 0.4
SEED_BASE           = 42000  # consistent with expOP2_isolate

# Accuracy windows (offsets within N_POST phase)
W1 = (0,   50)    # dec 1-50
W2 = (50,  100)   # dec 51-100
W3 = (100, 200)   # dec 101-200
W4 = (200, 400)   # dec 201-400

# Recovery threshold: W4 accuracy >= baseline - 2pp → recovered
RECOVERY_MARGIN_PP = 2.0

# Centroid NR rates from expOP2_isolate Arm A (hardcoded for comparison)
CENTROID_NR_ISOLATE = {
    "op0":   0.584,
    "op50":  0.539,
    "op75":  0.529,
    "op100": 0.515,
}

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Warm-start mu from soc_product_v50 profiles
# ---------------------------------------------------------------------------
def build_mu() -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = PROFILES[cat][act]
    return mu

MU_WARM = build_mu()

# ---------------------------------------------------------------------------
# Sigma — correct synthesis bias (same dual-action as isolate/recheck)
# ---------------------------------------------------------------------------
SIGMA_CORRECT = np.zeros((C_DIM, A_DIM), dtype=np.float64)
SIGMA_CORRECT[:, SUPP_IDX] = +SIGMA_VALUE   # suppress less likely
SIGMA_CORRECT[:, ESC_IDX]  = -SIGMA_VALUE   # escalate more likely

SYNTHESIS_CORRECT = SynthesisBias(
    sigma=SIGMA_CORRECT, active_claims=1, lambda_coupling=LAMBDA_S,
)

# Suppress DeprecationWarnings (not expected here, but defensive)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# results[op_key] = aggregated stats
results: dict[str, dict] = {}

# Per-seed raw trajectories for chart (list of per-seed window accuracy arrays)
# Shape per op_key: (N_SEEDS, 4) — W1/W2/W3/W4 mean accuracy per seed
raw_windows: dict[str, list] = {f"op{int(op*100)}": [] for op in OPERATOR_ACCURACIES}

# Per-seed pre-injection baseline accuracy (to compute relative recovery)
raw_baselines: dict[str, list] = {f"op{int(op*100)}": [] for op in OPERATOR_ACCURACIES}

# Per-seed post-expiry full trajectory (list of length N_POST booleans per seed)
raw_post_traj: dict[str, list] = {f"op{int(op*100)}": [] for op in OPERATOR_ACCURACIES}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=== EXP-OP2-ACCURACY: Accuracy Recovery vs Centroid Displacement ===")
print(f"Config: soc_product_v50  C={C_DIM} A={A_DIM} d={N_FACTORS}")
print(f"Arm A only: η_neg={ETA_NEG}, gt_action_index always passed")
print(f"N_SEEDS={N_SEEDS}  N_PRE={N_PRE}  TTL_HALF={TTL_HALF}  N_POST={N_POST}")
print(f"Operator accuracies: {OPERATOR_ACCURACIES}")
print()

for op_acc in OPERATOR_ACCURACIES:
    op_key = f"op{int(op_acc * 100)}"
    print(f"--- op_accuracy={op_acc:.2f} ---")

    seed_baselines    = []
    seed_window_accs  = []
    seed_post_trajs   = []
    seed_recovered    = []

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
            tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed,
        )

        # --- Phase 1: Pre-injection warmup (N_PRE decisions) ---
        pre_alerts = gen.generate(N_PRE)
        pre_correct = 0
        for alert in pre_alerts:
            result     = scorer.score(alert.factors, alert.category_index)
            is_correct = result.action_index == alert.gt_action_index
            pre_correct += int(is_correct)
            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, is_correct,
                gt_action_index=alert.gt_action_index,
            )
        baseline_acc = pre_correct / N_PRE
        seed_baselines.append(baseline_acc)

        # --- Phase 2: Injection (TTL_HALF decisions with correct sigma) ---
        inject_alerts = gen.generate(TTL_HALF)
        for alert in inject_alerts:
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
                gt_action_index=alert.gt_action_index,
            )

        # --- Phase 3: Post-expiry (N_POST decisions, no operator, no sigma) ---
        post_alerts = gen.generate(N_POST)
        post_correct_seq = []
        for alert in post_alerts:
            result     = scorer.score(alert.factors, alert.category_index)
            is_correct = result.action_index == alert.gt_action_index
            post_correct_seq.append(int(is_correct))
            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, is_correct,
                gt_action_index=alert.gt_action_index,
            )

        seed_post_trajs.append(post_correct_seq)

        # Window accuracies
        def window_acc(seq, lo, hi):
            return float(np.mean(seq[lo:hi]))

        w_accs = [
            window_acc(post_correct_seq, W1[0], W1[1]),
            window_acc(post_correct_seq, W2[0], W2[1]),
            window_acc(post_correct_seq, W3[0], W3[1]),
            window_acc(post_correct_seq, W4[0], W4[1]),
        ]
        seed_window_accs.append(w_accs)

        # Recovery: W4 accuracy >= baseline - 2pp
        w4_acc = w_accs[3]
        recovered = w4_acc >= (baseline_acc - RECOVERY_MARGIN_PP / 100.0)
        seed_recovered.append(int(recovered))

    # --- Aggregate ---
    baselines_arr = np.array(seed_baselines)
    windows_arr   = np.array(seed_window_accs)   # (N_SEEDS, 4)
    recovered_arr = np.array(seed_recovered)

    mean_baseline = float(baselines_arr.mean())

    mean_w = [float(windows_arr[:, i].mean()) for i in range(4)]
    ci_w   = [1.96 * float(windows_arr[:, i].std()) / np.sqrt(N_SEEDS) for i in range(4)]

    recovery_rate    = float(recovered_arr.mean())
    accuracy_nr_rate = 1.0 - recovery_rate
    ci_nr = 1.96 * float(recovered_arr.std()) / np.sqrt(N_SEEDS)

    centroid_nr = CENTROID_NR_ISOLATE.get(op_key, float("nan"))
    gap         = centroid_nr - accuracy_nr_rate

    results[op_key] = {
        "op_accuracy":       op_acc,
        "mean_baseline":     round(mean_baseline, 4),
        "mean_w1":           round(mean_w[0], 4),
        "mean_w2":           round(mean_w[1], 4),
        "mean_w3":           round(mean_w[2], 4),
        "mean_w4":           round(mean_w[3], 4),
        "ci_w1":             round(ci_w[0], 4),
        "ci_w2":             round(ci_w[1], 4),
        "ci_w3":             round(ci_w[2], 4),
        "ci_w4":             round(ci_w[3], 4),
        "recovery_rate":     round(recovery_rate, 4),
        "accuracy_nr_rate":  round(accuracy_nr_rate, 4),
        "accuracy_nr_ci":    round(ci_nr, 4),
        "centroid_nr_rate":  centroid_nr,
        "sensitivity_gap":   round(gap, 4),
        "n_seeds":           N_SEEDS,
    }

    raw_windows[op_key]   = seed_window_accs
    raw_baselines[op_key] = seed_baselines
    raw_post_traj[op_key] = seed_post_trajs

    print(f"  baseline={mean_baseline:.1%}  recovery={recovery_rate:.1%}")
    print(f"  W1={mean_w[0]:.1%}  W2={mean_w[1]:.1%}  W3={mean_w[2]:.1%}  W4={mean_w[3]:.1%}")
    print(f"  accuracy_nr={accuracy_nr_rate:.1%} ±{ci_nr:.1%}  centroid_nr={centroid_nr:.1%}  gap={gap:+.1%}")
    print()

# ---------------------------------------------------------------------------
# Key comparison table
# ---------------------------------------------------------------------------
print("=" * 70)
print("=== ACCURACY NR vs CENTROID NR COMPARISON ===")
print("=" * 70)
print(f"\n{'Op acc':>6}  {'Acc NR':>8}  {'Cent NR':>8}  {'Gap (cent-acc)':>15}  {'Interpretation'}")
print("-" * 70)
for op_acc in OPERATOR_ACCURACIES:
    op_key = f"op{int(op_acc * 100)}"
    r = results[op_key]
    acc_nr  = r["accuracy_nr_rate"]
    cent_nr = r["centroid_nr_rate"]
    gap     = r["sensitivity_gap"]
    interp  = "centroid over-sensitive" if gap > 0.10 else (
              "metrics agree" if abs(gap) < 0.05 else "moderate gap"
    )
    print(f"  {op_acc:>4.0%}  {acc_nr:>8.1%}  {cent_nr:>8.1%}  {gap:>+15.1%}  {interp}")

# Conclusion
print()
avg_gap = float(np.mean([results[f"op{int(op*100)}"]["sensitivity_gap"] for op in OPERATOR_ACCURACIES]))
if avg_gap > 0.10:
    print(f"→ CONCLUSION: Centroid NR is OVER-SENSITIVE (avg gap={avg_gap:+.1%})")
    print("  System scores correctly even where centroids have drifted.")
    print("  CLAIM-17 severity is lower than centroid displacement suggests.")
elif avg_gap > 0.0:
    print(f"→ CONCLUSION: Centroid NR is SLIGHTLY pessimistic (avg gap={avg_gap:+.1%})")
    print("  Minor over-counting of failure cases.")
else:
    print(f"→ CONCLUSION: Centroid NR and accuracy NR AGREE (avg gap={avg_gap:+.1%})")
    print("  Centroid displacement reliably predicts scoring failure.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _to_json(obj):
    if isinstance(obj, float):            return obj
    if isinstance(obj, (np.floating,)):   return float(obj)
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, np.ndarray):       return obj.tolist()
    if isinstance(obj, dict):             return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):             return [_to_json(v) for v in obj]
    return obj

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(_to_json(results), f, indent=2)
print(f"\nResults saved → {OUT_DIR / 'results.json'}")

np.save(str(OUT_DIR / "raw_windows.npy"),   {k: np.array(v) for k, v in raw_windows.items()})
np.save(str(OUT_DIR / "raw_baselines.npy"), {k: np.array(v) for k, v in raw_baselines.items()})
# Post-trajectory too large as 2D; save as object array
np.save(str(OUT_DIR / "raw_post_traj.npy"), raw_post_traj, allow_pickle=True)
print(f"Raw data saved → {OUT_DIR}/raw_*.npy")

from experiments.operator.expOP2_accuracy.charts import make_charts
make_charts(results, raw_windows, raw_baselines, N_SEEDS)

print()
print("=== EXP-OP2-ACCURACY COMPLETE ===")
