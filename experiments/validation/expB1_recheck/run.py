"""
EXP-B1 RECHECK: Warm-Start Accuracy at Canonical Settings

Re-runs EXP-B1 with both bugs fixed:
  Fix 1: eta_neg = 0.05  (was 1.0 in original)
  Fix 2: gt_action_index always passed to update()  (was never passed)

Config: soc_product_v50 (C=6, A=5, d=6)  — different from original default.yaml

Three conditions per noise rate:
  STATIC       — warm init, score only, no update
  LEARNING     — warm init, score then update with gt_action_index
  CENTROID-ONLY — warm init, learning run, then re-score all alerts with final mu

Published pre-fix values (default.yaml, C=5, A=4):
  Static noise=0:    97.89%  (EXP-C1)
  Learning noise=0:  98.2%   (CLAIM-02)
  Learning noise=0.30: 98.1% (CLAIM-03)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.category_alert_generator import CategoryAlertGenerator
from src.data.domain_config import load_domain_config
from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS     = 50
N_DECISIONS = 1000
NOISE_RATES = [0.0, 0.10, 0.30]
TAU         = 0.1
ETA         = 0.05
ETA_NEG     = 0.05   # CANONICAL — never 1.0

DOMAIN_CONFIG = "soc_product_v50"

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load soc_product_v50
# ---------------------------------------------------------------------------
cfg        = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = cfg["categories"]   # 6
ACTIONS    = cfg["actions"]      # 5
C_DIM      = cfg["C"]            # 6
A_DIM      = cfg["A"]            # 5
D_DIM      = cfg["d"]            # 6

# mu_true: warm-start centroids from compiled profiles (shape C×A×d)
MU_TRUE = cfg["mu"].copy()

# generator_kwargs contains action_conditional_profiles and gt_distributions
GEN_KWARGS = cfg["generator_kwargs"]

# ---------------------------------------------------------------------------
# Storage: acc[condition][noise_rate] = list of per-seed accuracy values
# ---------------------------------------------------------------------------
CONDITIONS = ["static", "learning", "centroid_only"]
acc: dict[str, dict[float, list[float]]] = {
    cond: {nr: [] for nr in NOISE_RATES}
    for cond in CONDITIONS
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"EXP-B1 RECHECK: Warm-Start Accuracy (post-fix)")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C_DIM} A={A_DIM} d={D_DIM}")
print(f"N_SEEDS={N_SEEDS}  N_DECISIONS={N_DECISIONS}")
print(f"ETA={ETA}  ETA_NEG={ETA_NEG}  TAU={TAU}")
print(f"NOISE_RATES={NOISE_RATES}")
print()

for noise_rate in NOISE_RATES:
    print(f"--- noise_rate={noise_rate:.2f} ---")

    for seed in range(N_SEEDS):
        # Shared alert stream: same alerts for all conditions this (seed, noise)
        gen = CategoryAlertGenerator(
            **GEN_KWARGS,
            noise_rate=0.0,   # factor noise off; oracle carries feedback noise
            seed=seed,
        )
        oracle = GTAlignedOracle(noise_rate=noise_rate, seed=seed + 10000)
        alerts = gen.generate(N_DECISIONS)

        # ── CONDITION 1: STATIC (no learning) ─────────────────────────────
        scorer_static = ProfileScorer(
            MU_TRUE.copy(), A_DIM,
            tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed,
        )
        n_correct_static = sum(
            int(scorer_static.score(a.factors, a.category_index).action_index
                == a.gt_action_index)
            for a in alerts
        )
        acc["static"][noise_rate].append(n_correct_static / N_DECISIONS)

        # ── CONDITION 2: LEARNING ──────────────────────────────────────────
        scorer_learn = ProfileScorer(
            MU_TRUE.copy(), A_DIM,
            tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed,
        )
        n_correct_learn = 0
        for alert in alerts:
            result     = scorer_learn.score(alert.factors, alert.category_index)
            predicted  = result.action_index
            gt_correct = (predicted == alert.gt_action_index)
            n_correct_learn += int(gt_correct)

            # Oracle gives noisy feedback on the predicted action
            oracle_result = oracle.evaluate(ACTIONS[predicted], alert)
            is_correct    = oracle_result.outcome > 0

            scorer_learn.update(
                alert.factors, alert.category_index,
                predicted, is_correct,
                gt_action_index=alert.gt_action_index,   # FIX: dual push/pull
            )
        acc["learning"][noise_rate].append(n_correct_learn / N_DECISIONS)

        # ── CONDITION 3: CENTROID-ONLY (re-score with final learned mu) ────
        # Use the final mu from the learning run; re-score ALL alerts as static
        n_correct_co = sum(
            int(scorer_learn.score(a.factors, a.category_index).action_index
                == a.gt_action_index)
            for a in alerts
        )
        acc["centroid_only"][noise_rate].append(n_correct_co / N_DECISIONS)

    # Per noise_rate summary
    for cond in CONDITIONS:
        vals = np.array(acc[cond][noise_rate])
        print(f"  {cond:<14} noise={noise_rate:.2f}: "
              f"{vals.mean():.1%} ± {1.96*vals.std()/np.sqrt(N_SEEDS):.1%} 95CI")
    print()

# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------
summary: dict = {}
for cond in CONDITIONS:
    summary[cond] = {}
    for nr in NOISE_RATES:
        vals = np.array(acc[cond][nr])
        summary[cond][nr] = {
            "mean":  float(vals.mean()),
            "std":   float(vals.std()),
            "ci95":  float(1.96 * vals.std() / np.sqrt(N_SEEDS)),
        }

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
print("=" * 62)
print("=== EXP-B1 RECHECK: WARM-START ACCURACY (post-fix) ===")
print("=" * 62)

# Original published values (pre-fix, default.yaml C=5 A=4)
ORIGINALS = {
    ("static",   0.0):  ("97.89%",   "EXP-C1 centroid_only"),
    ("learning", 0.0):  ("98.2%",    "CLAIM-02"),
    ("static",   0.10): ("~97.5% (est)", "estimated"),
    ("learning", 0.10): ("~98.0% (est)", "estimated"),
    ("static",   0.30): ("~97.0% (est)", "estimated"),
    ("learning", 0.30): ("98.1%",    "CLAIM-03"),
}

header = f"{'Condition':<16} {'Noise':>5}  {'Original (pre-fix)':<22} {'Recheck (post-fix)':>20}  {'Changed?'}"
print(f"\n{header}")
print("-" * len(header))

flags = []
for cond in ["static", "learning"]:
    for nr in NOISE_RATES:
        s         = summary[cond][nr]
        recheck   = s["mean"]
        ci        = s["ci95"]
        orig_str, orig_source = ORIGINALS.get((cond, nr), ("N/A", ""))
        recheck_str = f"{recheck:.1%} ±{ci:.1%}"

        # Parse original numeric if possible for delta check
        try:
            orig_num = float(orig_str.replace("%", "").replace("~", "").replace(" (est)", "")) / 100
            delta_pp = (recheck - orig_num) * 100
            changed  = f"{delta_pp:+.1f}pp"
            if abs(delta_pp) > 0.5:
                changed += " ⚠"
                flags.append((cond, nr, orig_num, recheck, delta_pp))
        except ValueError:
            changed = "N/A"

        print(f"  {cond:<14} {nr:>5.2f}  {orig_str:<22} {recheck_str:>20}  {changed}")

print()

# CLAIM-02 and CLAIM-03 checks
claim02_new = summary["learning"][0.0]["mean"]
claim03_new = summary["learning"][0.30]["mean"]
claim02_orig = 0.982
claim03_orig = 0.981

print(f"CLAIM-02 CHECK: Learning noise=0.0 — original 98.2% → recheck {claim02_new:.1%}")
delta02 = (claim02_new - claim02_orig) * 100
if abs(delta02) > 0.5:
    print(f"  ⚠  CHANGED by {delta02:+.1f}pp — flag for claims update")
else:
    print(f"  ✓  Within 0.5pp tolerance ({delta02:+.1f}pp) — claim stands")

print()
print(f"CLAIM-03 CHECK: Learning noise=0.30 — original 98.1% → recheck {claim03_new:.1%}")
delta03 = (claim03_new - claim03_orig) * 100
if abs(delta03) > 0.5:
    print(f"  ⚠  CHANGED by {delta03:+.1f}pp — flag for claims update")
else:
    print(f"  ✓  Within 0.5pp tolerance ({delta03:+.1f}pp) — claim stands")

print()
static0_new = summary["static"][0.0]["mean"]
print(f"STATIC noise=0 check: recheck {static0_new:.1%} vs original 97.89%")
delta_static = (static0_new - 0.9789) * 100
if abs(delta_static) > 0.5:
    print(f"  ⚠  CHANGED by {delta_static:+.1f}pp — config change (default.yaml→soc_product_v50) explains this")
else:
    print(f"  ✓  Within 0.5pp of original ({delta_static:+.1f}pp)")

print()
note = (
    "NOTE: Config changed from default.yaml (C=5, A=4, d=6) to soc_product_v50 "
    "(C=6, A=5, d=6) per spec. Direct numerical comparison is indicative only."
)
print(note)
print("=" * 62)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump({
        "summary": {
            cond: {str(nr): v for nr, v in d.items()}
            for cond, d in summary.items()
        },
        "claim02_delta_pp": delta02,
        "claim03_delta_pp": delta03,
        "flags": [
            {"cond": c, "noise": nr, "orig": o, "recheck": r, "delta_pp": d}
            for c, nr, o, r, d in flags
        ],
    }, f, indent=2)
np.save(str(OUT_DIR / "acc_data.npy"), acc, allow_pickle=True)
print(f"\nResults saved → {OUT_DIR}/summary.json + acc_data.npy")
print("Calling charts.py ...")

from experiments.validation.expB1_recheck.charts import make_charts
make_charts(summary, N_SEEDS)

print()
print("=== EXP-B1 RECHECK COMPLETE ===")
