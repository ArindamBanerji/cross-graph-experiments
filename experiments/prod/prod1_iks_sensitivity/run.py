"""
PROD-1: Institutional Knowledge Score (IKS) Sensitivity Analysis

Calibrate the IKS normalization constant κ.
Formula: IKS(t) = 100 × min(mean_L2_distance(μ(t), μ(0)) / κ, 1.0)
Find κ* such that IKS(200) ∈ [15, 40] in ≥90% of seeds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open(REPO_ROOT / "configs" / "default.yaml") as _f:
    _CFG = yaml.safe_load(_f)["bridge_common"]

CATEGORIES = _CFG["categories"]
ACTIONS    = _CFG["actions"]
FACTORS    = _CFG["factors"]
PROFILES   = _CFG["action_conditional_profiles"]
GT_DISTS   = _CFG["category_gt_distributions"]

C_DIM = len(CATEGORIES)   # 5
A_DIM = len(ACTIONS)      # 4
D_DIM = len(FACTORS)      # 6

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TAU         = 0.1
ETA         = 0.05
ETA_NEG     = 0.05        # CANONICAL — never 1.0
N_SEEDS     = 50
N_DECISIONS = 200
NOISE_RATE  = 0.10
KAPPA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Warm-start mu
# ---------------------------------------------------------------------------
def build_mu() -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, D_DIM), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = PROFILES[cat][act]
    return mu

MU_WARM = build_mu()

# ---------------------------------------------------------------------------
# IKS formula
# ---------------------------------------------------------------------------
def compute_iks(mu_t: np.ndarray, mu_0: np.ndarray, kappa: float) -> float:
    """IKS = 100 × min(mean_L2_norm_of_drift / kappa, 1.0)"""
    per_cell   = np.linalg.norm(mu_t - mu_0, axis=-1)   # shape (C, A)
    mean_drift = float(per_cell.mean())
    return min(100.0 * mean_drift / kappa, 100.0)

# ---------------------------------------------------------------------------
# Storage: results[kappa] = array shape (N_seeds, N_decisions)
# ---------------------------------------------------------------------------
results = {k: np.zeros((N_SEEDS, N_DECISIONS)) for k in KAPPA_VALUES}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=== PROD-1: IKS SENSITIVITY ANALYSIS ===")
print(f"Config: default.yaml  C={C_DIM} A={A_DIM} d={D_DIM}")
print(f"N_SEEDS={N_SEEDS}  N_DECISIONS={N_DECISIONS}  noise_rate={NOISE_RATE}")
print(f"κ values: {KAPPA_VALUES}")
print()

for seed in range(N_SEEDS):
    gen    = CategoryAlertGenerator(
        categories=CATEGORIES, actions=ACTIONS, factors=FACTORS,
        action_conditional_profiles=PROFILES, gt_distributions=GT_DISTS,
        noise_rate=NOISE_RATE, seed=seed,
    )
    oracle = GTAlignedOracle(noise_rate=0.0, seed=seed + 10000)
    # Note: noise injected via gen (label noise) AND oracle (feedback noise).
    # Oracle noise_rate=0.10 means oracle feedback may flip even a correct action.
    # We use noise_rate in oracle only; gen uses clean GT distributions.
    # Re-instantiate oracle with correct noise:
    oracle = GTAlignedOracle(noise_rate=NOISE_RATE, seed=seed + 10000)

    scorer = ProfileScorer(
        MU_WARM.copy(), A_DIM,
        tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed,
    )
    mu_0 = scorer.mu.copy()

    alerts = gen.generate(N_DECISIONS)
    for decision, alert in enumerate(alerts):
        result = scorer.score(alert.factors, alert.category_index)

        # Oracle evaluates system action vs ground truth (with 10% noise flip)
        oracle_result = oracle.evaluate(ACTIONS[result.action_index], alert)
        is_correct    = oracle_result.outcome > 0

        scorer.update(
            alert.factors, alert.category_index,
            result.action_index, is_correct,
            gt_action_index=alert.gt_action_index,
        )

        mu_t = scorer.mu
        for kappa in KAPPA_VALUES:
            results[kappa][seed, decision] = compute_iks(mu_t, mu_0, kappa)

    if (seed + 1) % 10 == 0:
        print(f"  {seed + 1}/{N_SEEDS} seeds done")

# ---------------------------------------------------------------------------
# NaN check
# ---------------------------------------------------------------------------
for kappa in KAPPA_VALUES:
    assert not np.isnan(results[kappa]).any(), f"NaN in results[{kappa}]"
total_values = len(KAPPA_VALUES) * N_SEEDS * N_DECISIONS
print(f"\n[OK] {total_values:,} IKS values computed — no NaN")

# ---------------------------------------------------------------------------
# Analysis per kappa
# ---------------------------------------------------------------------------
print()
print("=== PROD-1 IKS SENSITIVITY RESULTS ===")
kappa_star = None
summary: dict[float, dict] = {}

for kappa in KAPPA_VALUES:
    iks_at_200   = results[kappa][:, -1]
    mean_200     = float(iks_at_200.mean())
    std_200      = float(iks_at_200.std())
    pct_above_15 = float((iks_at_200 >= 15).mean())
    pct_above_40 = float((iks_at_200 >= 40).mean())
    pct_in_range = float(((iks_at_200 >= 15) & (iks_at_200 <= 40)).mean())

    summary[kappa] = dict(
        mean=mean_200, std=std_200,
        pct_in_range=pct_in_range,
        pct_above_15=pct_above_15,
        pct_above_40=pct_above_40,
    )

    print(f"  κ={kappa:.2f}:  IKS(200)={mean_200:5.1f}±{std_200:4.1f}  "
          f"in[15,40]={pct_in_range:.0%}  "
          f"≥15={pct_above_15:.0%}  ≥40={pct_above_40:.0%}")

    if kappa_star is None and pct_in_range >= 0.90:
        kappa_star = kappa

print()
if kappa_star is not None:
    s = summary[kappa_star]
    print(f"Recommended κ* = {kappa_star}")
    print(f"  IKS(200) at κ*: {s['mean']:.1f} ± {s['std']:.1f}")
    print(f"  Seeds with IKS(200) in [15,40]: {s['pct_in_range']:.0%}")
    print(f"  Demo outcome: IKS ≥ 15 in {s['pct_above_15']:.0%} of seeds after 200 decisions")
else:
    print("FLAG: No κ in [0.05, 0.30] achieves IKS(200) ∈ [15,40] in ≥90% of seeds.")
    print("  Diagnosis:")
    for kappa in KAPPA_VALUES:
        s = summary[kappa]
        if s["pct_above_40"] > 0.50:
            print(f"  κ={kappa:.2f}: IKS saturates (>{s['pct_above_40']:.0%} seeds above 40)")
        elif s["pct_above_15"] < 0.50:
            print(f"  κ={kappa:.2f}: IKS too low (<{s['pct_above_15']:.0%} seeds reach 15)")
    print("  Recommendation: extend kappa_values range or revisit IKS formula normalization.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _serializable(obj):
    """Convert numpy types for safe npy storage."""
    if isinstance(obj, dict):
        return {_serializable(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating,)):   return float(obj)
    if isinstance(obj, (np.integer,)):    return int(obj)
    if isinstance(obj, np.ndarray):       return obj
    return obj

np.save(str(OUT_DIR / "iks_results.npy"), {
    "results":      results,
    "summary":      summary,
    "kappa_star":   kappa_star,
    "kappa_values": KAPPA_VALUES,
    "C":            C_DIM,
    "A":            A_DIM,
    "d":            D_DIM,
    "N_seeds":      N_SEEDS,
    "N_decisions":  N_DECISIONS,
}, allow_pickle=True)
print(f"\nResults saved → {OUT_DIR / 'iks_results.npy'}")
print("Calling charts.py ...")

from experiments.prod.prod1_iks_sensitivity.charts import make_charts
make_charts()

print()
print("=== PROD-1 COMPLETE ===")
