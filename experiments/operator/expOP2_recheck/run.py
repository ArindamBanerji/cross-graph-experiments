"""
EXP-OP2-RECHECK (N=20)

Re-runs EXP-OP2 with both bugs fixed:
  Bug #1: eta_neg was 1.0  → now 0.05 (canonical)
  Bug #2: scorer.update() did not pass gt_action_index → now always passed

Config: soc_product_v50 (C=6, A=5, d=6)
Seeds:  same first 20 as expOP2_n100

Conditions (9):
  A      — no operator
  B      — correct sigma, TTL=400 (full post)
  B-exp  — correct sigma, TTL=150 (expires mid-run)
  C      — harmful sigma, TTL=400
  C-exp  — harmful sigma, TTL=150
  P-75   — 75% correct sigma, TTL=400
  P-50   — 50% correct sigma, TTL=400
  P-25   — 25% correct sigma, TTL=400
  P-0    — 0% correct (all inverted), TTL=400

Metrics:
  AUAC (mean window accuracy over all post-shift windows)
  c_exp: AUAC delta for post-expiry windows (w=3..7, decisions 150-400)
  T_recovery: decision # when rolling accuracy >= baseline-1pp for 2 consecutive windows
  NR_rate: fraction of seeds with T_recovery == SENTINEL
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

CATEGORIES = _CFG["categories"]                         # 6
ACTIONS    = _CFG["actions"]                            # 5
FACTORS    = _CFG["factors"]                            # 6
PROFILES   = _CFG["action_conditional_profiles"]        # dict
GT_DISTS   = _CFG["category_gt_distributions"]         # dict

C_DIM     = len(CATEGORIES)   # 6
A_DIM     = len(ACTIONS)      # 5
N_FACTORS = len(FACTORS)      # 6

ESC_IDX  = ACTIONS.index("escalate")    # 0  — campaign action to promote
SUPP_IDX = ACTIONS.index("suppress")   # 2  — action to suppress

# ---------------------------------------------------------------------------
# Parameters  (match OP2 original except the two bug fixes)
# ---------------------------------------------------------------------------
SEEDS_N20 = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
]
SEEDS = SEEDS_N20

LAMBDA_S         = 0.5
SIGMA_VALUE      = 0.4
N_PRE            = 200
N_POST           = 400
WINDOW_SIZE      = 50
N_WINDOWS        = N_POST // WINDOW_SIZE    # 8
TAU              = 0.1
ETA              = 0.05
ETA_NEG          = 0.05    # FIX #1: was 1.0
TTL_FULL         = N_POST  # 400 — active for all post-shift decisions
TTL_HALF         = 150     # expires mid-run
RECOVERY_THRESH  = 1.0     # pp below baseline to count as recovered
HOLD_WINDOWS     = 2       # consecutive windows needed
SENTINEL         = N_POST + 1   # 401 — never-recovered marker
PARTIAL_RNG_SEED = 99      # IDENTICAL to OP2 — deterministic flip pattern

CONDITIONS_ALL = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Warm-start mu from config profiles
# ---------------------------------------------------------------------------
def build_mu() -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = PROFILES[cat][act]
    return mu

MU_WARM = build_mu()

# ---------------------------------------------------------------------------
# Sigma constructors  (IDENTICAL to OP2 — only indices differ for v50)
# ---------------------------------------------------------------------------
def _correct_sigma() -> np.ndarray:
    s = np.zeros((C_DIM, A_DIM), dtype=np.float64)
    s[:, SUPP_IDX] = +SIGMA_VALUE   # suppress less likely
    s[:, ESC_IDX]  = -SIGMA_VALUE   # escalate more likely
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

# Build partial sigmas with deterministic RNG — SAME seed as OP2
_PARTIAL_RNG = np.random.default_rng(seed=PARTIAL_RNG_SEED)
SIGMA_P100 = _correct_sigma()
SIGMA_P75  = _partial_sigma(0.75, _PARTIAL_RNG)
SIGMA_P50  = _partial_sigma(0.50, _PARTIAL_RNG)
SIGMA_P25  = _partial_sigma(0.25, _PARTIAL_RNG)
SIGMA_P0   = _harmful_sigma()

# Map condition → (sigma, ttl)
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
    """Return decision # of recovery, or sentinel if never recovered."""
    threshold = baseline - thresh_pp / 100.0
    n = len(acc_curve)
    for i in range(n - hold + 1):
        if all(acc_curve[i + j] >= threshold for j in range(hold)):
            return i * WINDOW_SIZE
    return sentinel

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# per_cond_acc_curves[cond]   = list of (N_seeds, N_WINDOWS) rows
# per_cond_t_rec[cond]        = list of t_recovery values (one per seed)
# per_cond_baseline[cond]     = list of baseline_acc values
per_cond_acc_curves: dict[str, list] = {c: [] for c in CONDITIONS_ALL}
per_cond_t_rec:      dict[str, list] = {c: [] for c in CONDITIONS_ALL}
per_cond_baseline:   dict[str, list] = {c: [] for c in CONDITIONS_ALL}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=== EXP-OP2-RECHECK (N=20) ===")
print(f"Config: soc_product_v50  C={C_DIM} A={A_DIM} d={N_FACTORS}")
print(f"ETA={ETA}  ETA_NEG={ETA_NEG}  TAU={TAU}  LAMBDA={LAMBDA_S}")
print(f"N_PRE={N_PRE}  N_POST={N_POST}  TTL_HALF={TTL_HALF}")
print(f"Fixes: eta_neg=0.05 ✓   gt_action_index always passed ✓")
print()

for seed in SEEDS:
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

        # --- Pre-shift warmup (no operator) ---
        pre_correct: list[float] = []
        for alert in pre_alerts:
            result     = scorer.score(alert.factors, alert.category_index)
            is_correct = result.action_index == alert.gt_action_index
            pre_correct.append(float(is_correct))
            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, is_correct,
                gt_action_index=alert.gt_action_index,   # FIX #2
            )
        baseline_acc = float(np.mean(pre_correct[-100:]))

        # --- Post-shift with operator ---
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
                gt_action_index=alert.gt_action_index,   # FIX #2
            )

        # --- Per-window accuracy curve ---
        acc_curve = [
            float(np.mean(post_correct[w * WINDOW_SIZE:(w + 1) * WINDOW_SIZE]))
            for w in range(N_WINDOWS)
        ]
        t_rec = compute_t_recovery(acc_curve, baseline_acc)

        per_cond_acc_curves[cond].append(acc_curve)
        per_cond_t_rec[cond].append(t_rec)
        per_cond_baseline[cond].append(baseline_acc)

    seed_c_acc    = np.mean(per_cond_acc_curves["C"][-1])
    seed_cexp_acc = np.mean(per_cond_acc_curves["C-exp"][-1])
    print(f"  seed={seed:6d}  base={per_cond_baseline['A'][-1]:.3f}  "
          f"B={np.mean(per_cond_acc_curves['B'][-1]):.3f}  "
          f"C={seed_c_acc:.3f}  C-exp={seed_cexp_acc:.3f}")

# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
results: dict[str, dict] = {}
N_SEEDS = len(SEEDS)

for cond in CONDITIONS_ALL:
    curves    = np.array(per_cond_acc_curves[cond])      # (N_SEEDS, N_WINDOWS)
    t_recs    = np.array(per_cond_t_rec[cond])
    baselines = np.array(per_cond_baseline[cond])

    mean_curve   = curves.mean(axis=0)
    auac         = float(mean_curve.mean())
    auac_delta   = float((mean_curve - baselines.mean()).mean())

    # Post-expiry AUAC delta (windows 3-7, decisions 150-400)
    post_exp_idx   = list(range(3, N_WINDOWS))
    post_exp_delta = float((curves[:, post_exp_idx].mean(axis=1) - baselines).mean())

    nr_count = int((t_recs == SENTINEL).sum())
    nr_rate  = nr_count / N_SEEDS

    # 95% Wilson CI for NR rate
    z = 1.96
    p = nr_rate
    n = N_SEEDS
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    nr_ci_lo = max(0.0, center - margin)
    nr_ci_hi = min(1.0, center + margin)

    # T_recovery stats (exclude sentinels for mean)
    rec_times = t_recs[t_recs < SENTINEL]
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
print("=== EXP-OP2-RECHECK RESULTS ===")
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
# SIDE-BY-SIDE COMPARISON TABLE
# ---------------------------------------------------------------------------
ORIGINAL = {
    "c_exp_100_post_expiry": -0.0124,
    "c_exp_0_post_expiry":   None,
    "nr_rate_0op_n20":       35.0,
    "nr_ci_lo_0op_n20":      15.0,
    "nr_ci_hi_0op_n20":      59.0,
    "nr_rate_0op_n100":      38.0,
    "nr_ci_lo_0op_n100":     29.0,
    "nr_ci_hi_0op_n100":     48.0,
    "nr_rate_75op_n100":     28.0,
    "nr_rate_baseline_n20":  20.0,
    "nr_ci_lo_baseline":     None,
    "nr_ci_hi_baseline":     None,
    "nr_rate_baseline_n100": 24.0,
    "nr_ci_lo_baseline_n100":17.0,
    "nr_ci_hi_baseline_n100":33.0,
    "p75_paradox":           True,
    "zero_crossing":         True,
}

r_bexp = results["B-exp"]
r_cexp = results["C-exp"]
r_c    = results["C"]
r_a    = results["A"]
r_p75  = results["P-75"]
r_p25  = results["P-25"]
r_p50  = results["P-50"]

p75_paradox    = r_p75["nr_rate"] > r_a["nr_rate"]
zero_crossing  = (r_p50["auac_delta"] > 0) and (r_p25["auac_delta"] < 0)

print()
print("=" * 78)
print("=== EXP-OP2 RECHECK: SIDE-BY-SIDE (N=20) ===")
print("=" * 78)
print()
print(f"{'Metric':<38}  {'Original (bugs)':>16}  {'Recheck (fixed)':>16}  {'Changed?':>8}")
print("-" * 78)

def _row(name, orig, new_val, fmt="+.4f"):
    orig_s = f"{orig:{fmt}}" if orig is not None else "  N/A"
    new_s  = f"{new_val:{fmt}}"
    if orig is not None:
        changed = "YES" if abs(new_val - orig) > 0.005 else "NO "
    else:
        changed = " ? "
    print(f"  {name:<36}  {orig_s:>16}  {new_s:>16}  {changed:>8}")

def _row_pct(name, orig, new_val, orig_lo=None, orig_hi=None, new_lo=None, new_hi=None):
    orig_s = f"{orig:.1f}% [{orig_lo:.0f}%,{orig_hi:.0f}%]" if orig_lo is not None else (f"{orig:.1f}%" if orig is not None else "N/A")
    new_s  = f"{new_val:.1f}% [{new_lo:.1f}%,{new_hi:.1f}%]" if new_lo is not None else f"{new_val:.1f}%"
    changed = "YES" if orig is not None and abs(new_val - orig) > 1.0 else "NO "
    print(f"  {name:<36}  {orig_s:>16}  {new_s:>16}  {changed:>8}")

_row("c_exp (100% op) post-expiry", ORIGINAL["c_exp_100_post_expiry"],
     r_bexp["post_exp_delta"])
_row("c_exp (0%  op) post-expiry",  ORIGINAL["c_exp_0_post_expiry"],
     r_cexp["post_exp_delta"])
_row_pct("NR rate (0% op, N=20)",
         ORIGINAL["nr_rate_0op_n20"],  r_c["nr_rate_pct"],
         ORIGINAL["nr_ci_lo_0op_n20"], ORIGINAL["nr_ci_hi_0op_n20"],
         r_c["nr_ci_lo"],              r_c["nr_ci_hi"])
_row_pct("NR rate (75% op, N=20)",
         None, r_p75["nr_rate_pct"],
         None, None, r_p75["nr_ci_lo"], r_p75["nr_ci_hi"])
_row_pct("NR rate (baseline/no op, N=20)",
         ORIGINAL["nr_rate_baseline_n20"], r_a["nr_rate_pct"],
         None, None, r_a["nr_ci_lo"], r_a["nr_ci_hi"])

orig_paradox_s = "YES (28%>24%)" if ORIGINAL["p75_paradox"] else "NO"
new_paradox_s  = f"YES (P-75:{r_p75['nr_rate_pct']:.0f}%>A:{r_a['nr_rate_pct']:.0f}%)" if p75_paradox else f"NO (P-75:{r_p75['nr_rate_pct']:.0f}%,A:{r_a['nr_rate_pct']:.0f}%)"
orig_zc_s = "YES" if ORIGINAL["zero_crossing"] else "NO"
new_zc_s  = f"YES (P-50:{r_p50['auac_delta']:+.4f},P-25:{r_p25['auac_delta']:+.4f})" if zero_crossing else f"NO (P-50:{r_p50['auac_delta']:+.4f},P-25:{r_p25['auac_delta']:+.4f})"
print(f"  {'P-75 paradox (NR > baseline?)':<36}  {orig_paradox_s:>16}  {new_paradox_s:>16}")
print(f"  {'Zero-crossing (P-25 to P-50)':<36}  {orig_zc_s:>16}  {new_zc_s:>16}")

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
    print("    → The original finding was an ARTIFACT of eta_neg=1.0 + push-all bug.")
    print("    → CLAIM-17 must be RESTATED with corrected numbers.")
else:
    print(f"  NR rate (0% op) = {nr_0op:.1f}%:")
    print("    → Harmful operator risk survives both bug fixes.")
    print("    → Magnitude may differ from original — check CI overlap.")

if not p75_paradox:
    print("  P-75 paradox DISAPPEARS:")
    print("    → The 'bad operator worse than no operator' finding was BUG-DEPENDENT.")
else:
    print("  P-75 paradox SURVIVES:")
    print("    → Finding is robust to eta_neg and gt_action_index fixes.")

if zero_crossing:
    print("  Zero-crossing SURVIVES:")
    print("    → Kernel property confirmed — independent of eta_neg.")
else:
    print("  Zero-crossing ABSENT:")
    print("    → Zero-crossing was bug-dependent.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
# Convert numpy types to python for JSON
def _to_json(obj):
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, (np.integer, np.int64)):    return int(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, dict):  return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_to_json(v) for v in obj]
    return obj

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(_to_json(results), f, indent=2)
print()
print(f"Results saved → {OUT_DIR / 'results.json'}")

# ---------------------------------------------------------------------------
# Save raw arrays for charts
# ---------------------------------------------------------------------------
np.save(str(OUT_DIR / "acc_curves.npy"),
        {c: np.array(per_cond_acc_curves[c]) for c in CONDITIONS_ALL}, allow_pickle=True)
np.save(str(OUT_DIR / "t_recovery.npy"),
        {c: np.array(per_cond_t_rec[c]) for c in CONDITIONS_ALL},      allow_pickle=True)
print(f"Arrays saved  → {OUT_DIR / 'acc_curves.npy'}, t_recovery.npy")

from experiments.operator.expOP2_recheck.charts import make_charts
make_charts(results, per_cond_acc_curves, per_cond_t_rec, N_SEEDS, tag="expOP2_recheck")

print()
print("=== EXP-OP2-RECHECK COMPLETE ===")
