"""
EXP-OP2-N100-RECHECK (N=100)

Identical to expOP2_recheck/run.py but with N=100 seeds for tighter CIs.
Both bugs fixed: eta_neg=0.05, gt_action_index always passed.
Config: soc_product_v50 (C=6, A=5, d=6).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.oracle import GTAlignedOracle
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias

# ---------------------------------------------------------------------------
# Load soc_product_v50 config
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
SEEDS = [
    # Original 20 OP2 seeds
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
    # Additional 80 seeds (base 100000, step 1000 — no overlap with seed+10000 offsets)
    100000, 101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000,
    110000, 111000, 112000, 113000, 114000, 115000, 116000, 117000, 118000, 119000,
    120000, 121000, 122000, 123000, 124000, 125000, 126000, 127000, 128000, 129000,
    130000, 131000, 132000, 133000, 134000, 135000, 136000, 137000, 138000, 139000,
    140000, 141000, 142000, 143000, 144000, 145000, 146000, 147000, 148000, 149000,
    150000, 151000, 152000, 153000, 154000, 155000, 156000, 157000, 158000, 159000,
    160000, 161000, 162000, 163000, 164000, 165000, 166000, 167000, 168000, 169000,
    170000, 171000, 172000, 173000, 174000, 175000, 176000, 177000, 178000, 179000,
]
assert len(SEEDS) == 100, f"Expected 100 seeds, got {len(SEEDS)}"

LAMBDA_S         = 0.5
SIGMA_VALUE      = 0.4
N_PRE            = 200
N_POST           = 400
WINDOW_SIZE      = 50
N_WINDOWS        = N_POST // WINDOW_SIZE    # 8
TAU              = 0.1
ETA              = 0.05
ETA_NEG          = 0.05    # FIX #1: was 1.0
TTL_FULL         = N_POST
TTL_HALF         = 150
RECOVERY_THRESH  = 1.0
HOLD_WINDOWS     = 2
SENTINEL         = N_POST + 1   # 401
PARTIAL_RNG_SEED = 99

CONDITIONS_ALL = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]

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
# Sigma constructors
# ---------------------------------------------------------------------------
def _correct_sigma() -> np.ndarray:
    s = np.zeros((C_DIM, A_DIM), dtype=np.float64)
    s[:, SUPP_IDX] = +SIGMA_VALUE
    s[:, ESC_IDX]  = -SIGMA_VALUE
    return s

def _harmful_sigma() -> np.ndarray:
    return -_correct_sigma()

def _partial_sigma(fraction_correct: float, rng: np.random.Generator) -> np.ndarray:
    s = _correct_sigma()
    nonzero = [(c, SUPP_IDX) for c in range(C_DIM)] + [(c, ESC_IDX) for c in range(C_DIM)]
    n_flip  = int(len(nonzero) * (1.0 - fraction_correct))
    if n_flip > 0:
        flip_idx = rng.choice(len(nonzero), size=n_flip, replace=False)
        for fi in flip_idx:
            r, col = nonzero[fi]
            s[r, col] = -s[r, col]
    return s

_PARTIAL_RNG = np.random.default_rng(seed=PARTIAL_RNG_SEED)
SIGMA_P100 = _correct_sigma()
SIGMA_P75  = _partial_sigma(0.75, _PARTIAL_RNG)
SIGMA_P50  = _partial_sigma(0.50, _PARTIAL_RNG)
SIGMA_P25  = _partial_sigma(0.25, _PARTIAL_RNG)
SIGMA_P0   = _harmful_sigma()

COND_SPEC: dict[str, tuple] = {
    "A":     (None,        TTL_FULL),
    "B":     (SIGMA_P100,  TTL_FULL),
    "B-exp": (SIGMA_P100,  TTL_HALF),
    "C":     (SIGMA_P0,    TTL_FULL),
    "C-exp": (SIGMA_P0,    TTL_HALF),
    "P-75":  (SIGMA_P75,   TTL_FULL),
    "P-50":  (SIGMA_P50,   TTL_FULL),
    "P-25":  (SIGMA_P25,   TTL_FULL),
    "P-0":   (SIGMA_P0,    TTL_FULL),
}

# ---------------------------------------------------------------------------
# Recovery helper
# ---------------------------------------------------------------------------
def compute_t_recovery(
    acc_curve: list[float],
    baseline: float,
    thresh_pp: float = RECOVERY_THRESH,
    hold: int = HOLD_WINDOWS,
    sentinel: int = SENTINEL,
) -> int:
    threshold = baseline - thresh_pp / 100.0
    n = len(acc_curve)
    for i in range(n - hold + 1):
        if all(acc_curve[i + j] >= threshold for j in range(hold)):
            return i * WINDOW_SIZE
    return sentinel

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
per_cond_acc_curves: dict[str, list] = {c: [] for c in CONDITIONS_ALL}
per_cond_t_rec:      dict[str, list] = {c: [] for c in CONDITIONS_ALL}
per_cond_baseline:   dict[str, list] = {c: [] for c in CONDITIONS_ALL}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=== EXP-OP2-N100-RECHECK (N=100) ===")
print(f"Config: soc_product_v50  C={C_DIM} A={A_DIM} d={N_FACTORS}")
print(f"ETA={ETA}  ETA_NEG={ETA_NEG}  TAU={TAU}  LAMBDA={LAMBDA_S}")
print(f"N_PRE={N_PRE}  N_POST={N_POST}  TTL_HALF={TTL_HALF}")
print(f"Fixes: eta_neg=0.05 ✓   gt_action_index always passed ✓")
print()

for s_idx, seed in enumerate(SEEDS):
    gen_pre  = CategoryAlertGenerator(
        categories=CATEGORIES, actions=ACTIONS, factors=FACTORS,
        action_conditional_profiles=PROFILES, gt_distributions=GT_DISTS,
        seed=seed,
    )
    gen_post = CategoryAlertGenerator(
        categories=CATEGORIES, actions=ACTIONS, factors=FACTORS,
        action_conditional_profiles=PROFILES, gt_distributions=GT_DISTS,
        seed=seed + 10000,
    )

    pre_alerts  = gen_pre.generate(N_PRE)
    post_alerts = gen_post.generate(N_POST)

    for cond in CONDITIONS_ALL:
        sigma, ttl = COND_SPEC[cond]

        scorer = ProfileScorer(
            MU_WARM.copy(), A_DIM, tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed,
        )

        pre_correct: list[float] = []
        for alert in pre_alerts:
            result     = scorer.score(alert.factors, alert.category_index)
            is_correct = result.action_index == alert.gt_action_index
            pre_correct.append(float(is_correct))
            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, is_correct,
                gt_action_index=alert.gt_action_index,
            )
        baseline_acc = float(np.mean(pre_correct[-100:]))

        post_correct: list[float] = []
        for d, alert in enumerate(post_alerts):
            if sigma is not None and d < ttl:
                synthesis = SynthesisBias(
                    sigma=sigma, active_claims=1, lambda_coupling=LAMBDA_S,
                )
            else:
                synthesis = None

            result     = scorer.score(alert.factors, alert.category_index, synthesis=synthesis)
            is_correct = result.action_index == alert.gt_action_index
            post_correct.append(float(is_correct))

            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, is_correct,
                gt_action_index=alert.gt_action_index,
            )

        acc_curve = [
            float(np.mean(post_correct[w * WINDOW_SIZE:(w + 1) * WINDOW_SIZE]))
            for w in range(N_WINDOWS)
        ]
        t_rec = compute_t_recovery(acc_curve, baseline_acc)

        per_cond_acc_curves[cond].append(acc_curve)
        per_cond_t_rec[cond].append(t_rec)
        per_cond_baseline[cond].append(baseline_acc)

    if (s_idx + 1) % 10 == 0 or s_idx == 0:
        print(f"  [{s_idx+1:3d}/100] seed={seed}  "
              f"B={np.mean(per_cond_acc_curves['B'][-1]):.3f}  "
              f"C={np.mean(per_cond_acc_curves['C'][-1]):.3f}  "
              f"C-exp={np.mean(per_cond_acc_curves['C-exp'][-1]):.3f}")

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
results: dict[str, dict] = {}
N_SEEDS = len(SEEDS)

for cond in CONDITIONS_ALL:
    curves    = np.array(per_cond_acc_curves[cond])
    t_recs    = np.array(per_cond_t_rec[cond])
    baselines = np.array(per_cond_baseline[cond])

    mean_curve     = curves.mean(axis=0)
    auac           = float(mean_curve.mean())
    auac_delta     = float((mean_curve - baselines.mean()).mean())
    post_exp_idx   = list(range(3, N_WINDOWS))
    post_exp_delta = float((curves[:, post_exp_idx].mean(axis=1) - baselines).mean())

    nr_count = int((t_recs == SENTINEL).sum())
    nr_rate  = nr_count / N_SEEDS

    z      = 1.96
    p      = nr_rate
    n      = N_SEEDS
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    nr_ci_lo = max(0.0, center - margin)
    nr_ci_hi = min(1.0, center + margin)

    rec_times  = t_recs[t_recs < SENTINEL]
    t_rec_mean = float(rec_times.mean()) if len(rec_times) > 0 else float("nan")
    t_rec_std  = float(rec_times.std())  if len(rec_times) > 0 else float("nan")

    results[cond] = {
        "auac":             auac,
        "auac_delta":       auac_delta,
        "post_exp_delta":   post_exp_delta,
        "nr_rate":          nr_rate,
        "nr_rate_pct":      nr_rate * 100,
        "nr_ci_lo":         nr_ci_lo * 100,
        "nr_ci_hi":         nr_ci_hi * 100,
        "t_rec_mean":       t_rec_mean,
        "t_rec_std":        t_rec_std,
        "nr_count":         nr_count,
        "mean_acc_curve":   mean_curve.tolist(),
        "t_recovery_list":  t_recs.tolist(),
        "baseline_mean":    float(baselines.mean()),
    }

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
print()
print("=== EXP-OP2-N100-RECHECK RESULTS ===")
print(f"{'Cond':>6}  {'AUAC':>7}  {'AUAC Δ':>8}  {'PostExpΔ':>10}  "
      f"{'NR%':>6}  {'NR CI':>15}  {'T_rec':>7}  {'T_std':>7}")
for cond in CONDITIONS_ALL:
    r = results[cond]
    t = f"{r['t_rec_mean']:6.0f}" if r['t_rec_mean'] == r['t_rec_mean'] else "   NaN"
    s = f"{r['t_rec_std']:6.0f}"  if r['t_rec_std']  == r['t_rec_std']  else "   NaN"
    print(f"  {cond:>6}  {r['auac']:.4f}  {r['auac_delta']:+.4f}  "
          f"{r['post_exp_delta']:+.8f}  "
          f"{r['nr_rate_pct']:5.1f}%  "
          f"[{r['nr_ci_lo']:4.1f}%,{r['nr_ci_hi']:4.1f}%]  {t}  {s}")

# ---------------------------------------------------------------------------
# SIDE-BY-SIDE COMPARISON TABLE (with N=100 reference values)
# ---------------------------------------------------------------------------
ORIGINAL_N100 = {
    "c_exp_100_post_expiry":  -0.0124,
    "nr_rate_0op":             38.0,
    "nr_ci_lo_0op":            29.0,
    "nr_ci_hi_0op":            48.0,
    "nr_rate_75op":            28.0,
    "nr_rate_baseline":        24.0,
    "nr_ci_lo_baseline":       17.0,
    "nr_ci_hi_baseline":       33.0,
    "p75_paradox":             True,
    "zero_crossing":           True,
}

r_bexp = results["B-exp"]
r_cexp = results["C-exp"]
r_c    = results["C"]
r_a    = results["A"]
r_p75  = results["P-75"]
r_p25  = results["P-25"]
r_p50  = results["P-50"]

p75_paradox   = r_p75["nr_rate"] > r_a["nr_rate"]
zero_crossing = (r_p50["auac_delta"] > 0) and (r_p25["auac_delta"] < 0)

print()
print("=" * 82)
print("=== EXP-OP2 RECHECK: SIDE-BY-SIDE (N=100) ===")
print("=" * 82)
print()
print(f"{'Metric':<42}  {'Original (bugs)':>16}  {'Recheck (fixed)':>16}  {'Changed?':>8}")
print("-" * 82)

def _fmt(val, fmt):
    return f"{val:{fmt}}" if val is not None else "   N/A"

def _pct_str(pct, lo=None, hi=None):
    if lo is not None:
        return f"{pct:.1f}% [{lo:.0f}%,{hi:.0f}%]"
    return f"{pct:.1f}%"

rows = [
    ("c_exp (100% op) post-expiry",    f"{ORIGINAL_N100['c_exp_100_post_expiry']:+.4f}",
     f"{r_bexp['post_exp_delta']:+.4f}",
     "YES" if abs(r_bexp["post_exp_delta"] - ORIGINAL_N100["c_exp_100_post_expiry"]) > 0.005 else "NO"),

    ("NR rate (0% op, N=100)",
     _pct_str(ORIGINAL_N100["nr_rate_0op"], ORIGINAL_N100["nr_ci_lo_0op"], ORIGINAL_N100["nr_ci_hi_0op"]),
     _pct_str(r_c["nr_rate_pct"], r_c["nr_ci_lo"], r_c["nr_ci_hi"]),
     "YES" if abs(r_c["nr_rate_pct"] - ORIGINAL_N100["nr_rate_0op"]) > 2.0 else "NO"),

    ("NR rate (75% op, N=100)",
     _pct_str(ORIGINAL_N100["nr_rate_75op"]),
     _pct_str(r_p75["nr_rate_pct"], r_p75["nr_ci_lo"], r_p75["nr_ci_hi"]),
     "YES" if abs(r_p75["nr_rate_pct"] - ORIGINAL_N100["nr_rate_75op"]) > 2.0 else "NO"),

    ("NR rate (baseline/no op, N=100)",
     _pct_str(ORIGINAL_N100["nr_rate_baseline"], ORIGINAL_N100["nr_ci_lo_baseline"], ORIGINAL_N100["nr_ci_hi_baseline"]),
     _pct_str(r_a["nr_rate_pct"], r_a["nr_ci_lo"], r_a["nr_ci_hi"]),
     "YES" if abs(r_a["nr_rate_pct"] - ORIGINAL_N100["nr_rate_baseline"]) > 2.0 else "NO"),

    ("P-75 paradox (NR > baseline?)",
     "YES (28%>24%)" if ORIGINAL_N100["p75_paradox"] else "NO",
     f"YES ({r_p75['nr_rate_pct']:.0f}%>{r_a['nr_rate_pct']:.0f}%)" if p75_paradox else f"NO ({r_p75['nr_rate_pct']:.0f}%≤{r_a['nr_rate_pct']:.0f}%)",
     "YES" if p75_paradox != ORIGINAL_N100["p75_paradox"] else "NO"),

    ("Zero-crossing (P-25 to P-50)",
     "YES" if ORIGINAL_N100["zero_crossing"] else "NO",
     f"YES (P-50:{r_p50['auac_delta']:+.4f})" if zero_crossing else f"NO (P-50:{r_p50['auac_delta']:+.4f})",
     "YES" if zero_crossing != ORIGINAL_N100["zero_crossing"] else "NO"),
]

for name, orig_s, new_s, changed in rows:
    print(f"  {name:<42}  {orig_s:>16}  {new_s:>16}  {changed:>8}")

print()
print("=== INTERPRETATION ===")
print()

c_exp_post = r_cexp["post_exp_delta"]
nr_0op     = r_c["nr_rate_pct"]
nr_base    = r_a["nr_rate_pct"]

if c_exp_post > 0:
    print("  c_exp post-expiry > 0:")
    print("    → TTL expiry IS sufficient at eta_neg=0.05. Centroids self-heal.")
    print("    → CLAIM-17 severity should be DOWNGRADED. Checkpoint is nice-to-have.")
else:
    print("  c_exp post-expiry < 0:")
    print("    → Damage persists even at eta_neg=0.05. Checkpoint remains CRITICAL.")

if nr_0op < 10.0:
    print("  NR rate (0% op) < 10%:")
    print("    → The 38% finding was an ARTIFACT of eta_neg=1.0 + push-all bug.")
    print("    → CLAIM-17 must be RESTATED with corrected numbers.")
else:
    print(f"  NR rate (0% op) = {nr_0op:.1f}% (original: 38%):")
    if nr_0op < ORIGINAL_N100["nr_rate_0op"] - 5:
        print("    → Rate dropped significantly. Magnitude was bug-inflated.")
        print("    → Update CLAIM-17 with new value + confidence interval.")
    else:
        print("    → Rate robust. Harmful operator risk confirmed.")

if not p75_paradox:
    print("  P-75 paradox DISAPPEARS:")
    print("    → 'Bad operator worse than no operator' was BUG-DEPENDENT.")
else:
    print("  P-75 paradox SURVIVES (robust to both bug fixes).")

if zero_crossing:
    print("  Zero-crossing SURVIVES: kernel property confirmed, independent of eta_neg.")
else:
    print("  Zero-crossing ABSENT: was bug-dependent.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _to_json(obj):
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, (np.integer, np.int64)):    return int(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, dict):  return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_to_json(v) for v in obj]
    return obj

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(_to_json(results), f, indent=2)

np.save(str(OUT_DIR / "acc_curves.npy"),
        {c: np.array(per_cond_acc_curves[c]) for c in CONDITIONS_ALL}, allow_pickle=True)
np.save(str(OUT_DIR / "t_recovery.npy"),
        {c: np.array(per_cond_t_rec[c]) for c in CONDITIONS_ALL},      allow_pickle=True)
print()
print(f"Results saved → {OUT_DIR / 'results.json'}")

from experiments.operator.expOP2_n100_recheck.charts import make_charts
make_charts(results, per_cond_acc_curves, per_cond_t_rec, N_SEEDS, tag="expOP2n_recheck")

print()
print("=== EXP-OP2-N100-RECHECK COMPLETE ===")
