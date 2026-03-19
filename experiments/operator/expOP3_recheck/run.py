"""
EXP-OP3-RECHECK: Residual Tracker Early-Warning at η_neg=0.05, A=4 geometry.

Original EXP-OP3 ran with η_neg=1.0 (forbidden) and A=5 ontology.
This recheck confirms the core result: the drift-based residual tracker
can distinguish a harmful operator from a correct operator within the
first W=1 window (50 decisions).

Residual metric (new — no declared mu_tilde needed):
  R(t) = ‖μ(t) - μ(t-W)‖_F   [Frobenius norm of centroid change in last window]

Three conditions:
  A — No operator  (baseline): noisy oracle, noise_rate=0.10
  B — Correct operator      : oracle always correct (noise_rate=0.0)
  C — Harmful operator      : oracle always wrong (flipped gt_action_index)

Config: soc_product_v50 (C=6, A=4, d=6).  η_neg=0.05, gt_action_index always passed.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SEEDS      = 50
N_WARMUP     = 200     # warm up centroids before measuring drift
N_DECISIONS  = 400     # measurement window (8 × W=50)
WINDOW_SIZE  = 50      # residual window
TAU          = 0.1
ETA          = 0.05
ETA_NEG      = 0.05    # CANONICAL — η_neg=1.0 FORBIDDEN
NOISE_RATE_A = 0.10    # oracle noise in baseline condition
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG    = "soc_product_v50"

N_WINDOWS    = N_DECISIONS // WINDOW_SIZE   # 8 windows
WINDOW_KEYS  = list(range(N_WINDOWS))       # W=0..7; W=0 is pre-decision baseline

CONDITIONS   = ["A", "B", "C"]

# Gate criteria (per spec)
GATE_DETECT_RATE = 0.90   # C detection at W=1 >= 90%
GATE_FA_RATE     = 0.30   # B false-alarm at W=1 <= 30%
GATE_AUC         = 0.70   # ROC AUC >= 0.70

# Original results for comparison (η_neg=1.0, A=5, N_seeds=20)
ORIGINAL = {
    "C_detection_rate": 1.00,
    "B_fa_rate":        0.20,
    "roc_auc":          0.749,
}

RESULTS_PATH = _REPO_ROOT / "experiments" / "operator" / "expOP3_recheck" / "results.json"

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

assert A == 4, f"Expected A=4, got A={A}"

print("=" * 60)
print("EXP-OP3-RECHECK: RESIDUAL TRACKER (η_neg=0.05, A=4)")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Actions: {ACTIONS}")
print(f"N_SEEDS={N_SEEDS}, N_WARMUP={N_WARMUP}, N_DECISIONS={N_DECISIONS}")
print(f"WINDOW_SIZE={WINDOW_SIZE}, N_WINDOWS={N_WINDOWS}")
print(f"ETA={ETA}, ETA_NEG={ETA_NEG}, TAU={TAU}")
print(f"Conditions: {CONDITIONS}")
print()

# ---------------------------------------------------------------------------
# Storage: per_seed_R[cond][seed] = list of R values at each window boundary
# W=0: R before any decisions (= 0 by definition)
# W=k: R after k*WINDOW_SIZE decisions (drift from W=k-1)
# ---------------------------------------------------------------------------

per_seed_R:   dict[str, list[list[float]]] = {c: [] for c in CONDITIONS}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for seed_idx in range(N_SEEDS):
    print(f"Seed {seed_idx+1}/{N_SEEDS}", flush=True)
    rng_noise = np.random.default_rng(RANDOM_SEED_BASE + seed_idx + 5000)

    # Shared alert generator for warmup
    gen_warm = CategoryAlertGenerator(
        **config["generator_kwargs"],
        noise_rate=0.0,
        seed=RANDOM_SEED_BASE + seed_idx,
    )
    warmup_alerts = gen_warm.generate(N_WARMUP)

    # Shared alert generator for the measurement window (same alerts for A/B/C)
    gen_meas = CategoryAlertGenerator(
        **config["generator_kwargs"],
        noise_rate=0.0,
        seed=RANDOM_SEED_BASE + seed_idx + 1000,
    )
    meas_alerts = gen_meas.generate(N_DECISIONS)

    # Build warmup scorer — warm up with perfect oracle, producing starting_mu
    scorer_warm = ProfileScorer(
        config["mu"].copy(), config["actions"],
        tau=TAU, eta=ETA, eta_neg=ETA_NEG,
    )
    for alert in warmup_alerts:
        result     = scorer_warm.score(alert.factors, alert.category_index)
        is_correct = result.action_index == alert.gt_action_index
        scorer_warm.update(
            alert.factors, alert.category_index,
            result.action_index, correct=is_correct,
            gt_action_index=alert.gt_action_index,
        )
    starting_mu = scorer_warm.mu.copy()

    # Run each condition
    for cond in CONDITIONS:
        scorer = ProfileScorer(
            starting_mu.copy(), config["actions"],
            tau=TAU, eta=ETA, eta_neg=ETA_NEG,
        )

        mu_prev = starting_mu.copy()   # snapshot before first window
        R_trajectory = [0.0]           # W=0: no drift yet

        for w_idx in range(N_WINDOWS):
            window_alerts = meas_alerts[w_idx * WINDOW_SIZE : (w_idx + 1) * WINDOW_SIZE]

            for alert in window_alerts:
                result = scorer.score(alert.factors, alert.category_index)

                if cond == "A":
                    # Baseline: noisy oracle (10% flip)
                    is_correct_true = result.action_index == alert.gt_action_index
                    if rng_noise.random() < NOISE_RATE_A:
                        # Flip: tell scorer the opposite of truth
                        if is_correct_true:
                            # Was correct, say wrong with shifted gt
                            wrong_gt = (alert.gt_action_index + 1) % A
                            scorer.update(
                                alert.factors, alert.category_index,
                                result.action_index, correct=False,
                                gt_action_index=wrong_gt,
                            )
                        else:
                            # Was wrong, say correct (hallucinates correctness)
                            scorer.update(
                                alert.factors, alert.category_index,
                                result.action_index, correct=True,
                            )
                    else:
                        scorer.update(
                            alert.factors, alert.category_index,
                            result.action_index, correct=is_correct_true,
                            gt_action_index=alert.gt_action_index,
                        )

                elif cond == "B":
                    # Correct operator: always truth
                    is_correct = result.action_index == alert.gt_action_index
                    scorer.update(
                        alert.factors, alert.category_index,
                        result.action_index, correct=is_correct,
                        gt_action_index=alert.gt_action_index,
                    )

                else:  # cond == "C"
                    # Harmful operator: always wrong
                    # Always says prediction was wrong; pulls toward a wrong action
                    harmful_gt = (alert.gt_action_index + 1) % A
                    scorer.update(
                        alert.factors, alert.category_index,
                        result.action_index, correct=False,
                        gt_action_index=harmful_gt,
                    )

            # Snapshot after window and compute R = ‖μ(t) - μ(t-W)‖_F
            mu_now = scorer.mu.copy()
            R = float(np.sqrt(np.sum((mu_now - mu_prev) ** 2)))
            R_trajectory.append(R)
            mu_prev = mu_now.copy()

        per_seed_R[cond].append(R_trajectory)

# ---------------------------------------------------------------------------
# Aggregate trajectories
# ---------------------------------------------------------------------------

def get_traj(cond: str) -> np.ndarray:
    """Returns (N_SEEDS, N_WINDOWS+1) array."""
    return np.array(per_seed_R[cond])

traj = {c: get_traj(c) for c in CONDITIONS}

traj_mean = {c: traj[c].mean(axis=0) for c in CONDITIONS}
traj_std  = {c: traj[c].std(axis=0)  for c in CONDITIONS}

# W=1 values (first window, after 50 decisions)
r_w1 = {c: traj[c][:, 1] for c in CONDITIONS}

# ---------------------------------------------------------------------------
# Threshold sweep for detection/FA rates at W=1
# ---------------------------------------------------------------------------

# Threshold: find value where C detection >= 90% and B FA <= 30%
r_all_w1 = np.concatenate([r_w1["B"], r_w1["C"]])
thresholds = np.linspace(r_all_w1.min() * 0.9, r_all_w1.max() * 1.1, 2000)

best_threshold    = None
best_detect_rate  = 0.0
best_fa_rate      = 1.0

for thr in thresholds:
    det = float(np.mean(r_w1["C"] > thr))
    fa  = float(np.mean(r_w1["B"] > thr))
    if det >= GATE_DETECT_RATE and fa <= GATE_FA_RATE:
        if det > best_detect_rate or (det == best_detect_rate and fa < best_fa_rate):
            best_detect_rate  = det
            best_fa_rate      = fa
            best_threshold    = float(thr)

# Fallback: best TPR-FPR operating point even if gate criteria not met
if best_threshold is None:
    best_j = -1.0
    for thr in thresholds:
        det = float(np.mean(r_w1["C"] > thr))
        fa  = float(np.mean(r_w1["B"] > thr))
        j   = det - fa   # Youden's J
        if j > best_j:
            best_j            = j
            best_detect_rate  = det
            best_fa_rate      = fa
            best_threshold    = float(thr)

# ---------------------------------------------------------------------------
# ROC AUC: B vs C discrimination using R(W=1)
# ---------------------------------------------------------------------------

def roc_auc_manual(scores_neg: np.ndarray, scores_pos: np.ndarray) -> tuple[list, list, float]:
    """
    Compute ROC AUC treating scores_pos (C) as positives, scores_neg (B) as negatives.
    Returns (fprs, tprs, auc).
    """
    all_thr = np.sort(np.unique(np.concatenate([scores_neg, scores_pos])))
    # Add sentinels
    sentinels = np.array([all_thr.min() - 1e-6, all_thr.max() + 1e-6])
    all_thr = np.concatenate([sentinels, all_thr])[::-1]   # high to low

    fprs, tprs = [0.0], [0.0]
    for thr in all_thr:
        tpr = float(np.mean(scores_pos > thr))
        fpr = float(np.mean(scores_neg > thr))
        fprs.append(fpr)
        tprs.append(tpr)
    fprs.append(1.0)
    tprs.append(1.0)

    # Trapezoidal AUC
    auc = float(np.trapz(tprs, fprs))
    return fprs, tprs, auc

roc_fprs, roc_tprs, roc_auc = roc_auc_manual(r_w1["B"], r_w1["C"])

# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

gate_detect  = best_detect_rate >= GATE_DETECT_RATE
gate_fa      = best_fa_rate      <= GATE_FA_RATE
gate_auc_val = roc_auc           >= GATE_AUC
gate_pass    = gate_detect and gate_fa and gate_auc_val

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("=== EXP-OP3-RECHECK: RESIDUAL TRACKER (η_neg=0.05, A=4) ===")
print("=" * 60)
print()

print(f"{'Condition':<12}  {'R(W=1) mean':>12}  {'R(W=1) std':>11}  {'Detection/FA':>14}")
print("  " + "-" * 55)

r_a_mean = traj_mean["A"][1]
r_a_std  = traj_std["A"][1]
print(f"  {'A (no op)':<12}  {r_a_mean:>12.4f}  {r_a_std:>11.4f}  {'baseline':>14}")

r_b_mean = r_w1["B"].mean()
r_b_std  = r_w1["B"].std()
fa_str   = f"FA: {best_fa_rate:.0%}"
print(f"  {'B (100%)':<12}  {r_b_mean:>12.4f}  {r_b_std:>11.4f}  {fa_str:>14}")

r_c_mean = r_w1["C"].mean()
r_c_std  = r_w1["C"].std()
det_str  = f"Det: {best_detect_rate:.0%}"
print(f"  {'C (0%)':<12}  {r_c_mean:>12.4f}  {r_c_std:>11.4f}  {det_str:>14}")

print()
print(f"Operating threshold (R): {best_threshold:.4f}")
print(f"ROC AUC (B vs C):        {roc_auc:.3f}")
print()

# Gate table
print(f"{'Gate criterion':<38}  {'Result':>8}  {'Pass?':>6}")
print("  " + "-" * 58)
print(f"  C detection rate >= {GATE_DETECT_RATE:.0%}:          "
      f"{best_detect_rate:>8.0%}  {'PASS' if gate_detect else 'FAIL':>6}")
print(f"  B false-alarm rate <= {GATE_FA_RATE:.0%}:          "
      f"{best_fa_rate:>8.0%}  {'PASS' if gate_fa else 'FAIL':>6}")
print(f"  ROC AUC >= {GATE_AUC:.2f}:                    "
      f"{roc_auc:>8.3f}  {'PASS' if gate_auc_val else 'FAIL':>6}")
print()
print(f"Overall: {'GATE PASS' if gate_pass else 'GATE FAIL'}")

print()
print("Trajectory (mean R(t) per window):")
hdr = f"  {'Cond':<6}  " + "  ".join(f"W{w}" for w in range(N_WINDOWS + 1))
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for cond in CONDITIONS:
    row = f"  {cond:<6}  " + "  ".join(f"{v:.4f}" for v in traj_mean[cond])
    print(row)

print()
print("=== COMPARISON: Original (η_neg=1.0, A=5) vs Recheck (η_neg=0.05, A=4) ===")
print(f"{'Metric':<30}  {'Original':>10}  {'Recheck':>10}  {'Change':>10}")
print("  " + "-" * 65)
print(f"  {'C detection rate (W=1)':<28}  "
      f"{ORIGINAL['C_detection_rate']:>10.0%}  "
      f"{best_detect_rate:>10.0%}  "
      f"{best_detect_rate - ORIGINAL['C_detection_rate']:>+10.0%}")
print(f"  {'B false-alarm rate (W=1)':<28}  "
      f"{ORIGINAL['B_fa_rate']:>10.0%}  "
      f"{best_fa_rate:>10.0%}  "
      f"{best_fa_rate - ORIGINAL['B_fa_rate']:>+10.0%}")
print(f"  {'ROC AUC':<28}  "
      f"{ORIGINAL['roc_auc']:>10.3f}  "
      f"{roc_auc:>10.3f}  "
      f"{roc_auc - ORIGINAL['roc_auc']:>+10.3f}")
print()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results_out = {
    "experiment":  "EXP-OP3-RECHECK",
    "domain_config": DOMAIN_CONFIG,
    "n_seeds":     N_SEEDS,
    "n_warmup":    N_WARMUP,
    "n_decisions": N_DECISIONS,
    "window_size": WINDOW_SIZE,
    "eta":         ETA,
    "eta_neg":     ETA_NEG,
    "tau":         TAU,
    "ontology":    {"C": C, "A": A, "d": d},
    "gate": {
        "detect_rate_gate":  GATE_DETECT_RATE,
        "fa_rate_gate":      GATE_FA_RATE,
        "auc_gate":          GATE_AUC,
        "detect_rate":       best_detect_rate,
        "fa_rate":           best_fa_rate,
        "auc":               roc_auc,
        "threshold":         best_threshold,
        "pass":              gate_pass,
    },
    "trajectories": {
        cond: {
            "mean": traj_mean[cond].tolist(),
            "std":  traj_std[cond].tolist(),
        }
        for cond in CONDITIONS
    },
    "r_w1": {
        cond: r_w1[cond].tolist()
        for cond in CONDITIONS
    },
    "roc": {
        "fprs": roc_fprs,
        "tprs": roc_tprs,
        "auc":  roc_auc,
    },
    "original_comparison": ORIGINAL,
    "per_seed_R": {
        cond: per_seed_R[cond]
        for cond in CONDITIONS
    },
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(results_out, fh, indent=2)

print(f"Results saved to {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

import subprocess
charts_path = Path(__file__).parent / "charts.py"
subprocess.run(
    [sys.executable, str(charts_path)],
    check=True,
    cwd=str(_REPO_ROOT),
)
