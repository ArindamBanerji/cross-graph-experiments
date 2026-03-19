"""
EXP-OP2-N100: EXP-OP2 replicated at N=100 seeds for tighter CI on never-recover rate.

Identical to EXP-OP2 (experiments/synthesis/expOP2_harmful/run.py) in every
parameter, sigma construction, harness pattern, and condition set.
THE ONLY CHANGE: N_seeds = 100 (was 20).

9 conditions:
  A      — no operator (baseline)
  B      — correct sigma, TTL=400 (full post-shift)
  B-exp  — correct sigma, TTL=150 (expires mid-run; tests indirect path)
  C      — harmful sigma, TTL=400
  C-exp  — harmful sigma, TTL=150
  P-75   — 75% of σ cells correct
  P-50   — 50% of σ cells correct
  P-25   — 25% of σ cells correct
  P-0    — 0% correct (all inverted; alias for C at module level)

Outputs:
  experiments/synthesis/expOP2_n100/results.json
  paper_figures/expOP2n_*.{pdf,png}  (3 charts × 2 formats = 6 files)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.category_alert_generator import (
    CategoryAlertGenerator, CATEGORIES, ACTIONS,
)
from src.models.profile_scorer import ProfileScorer
from src.models.oracle import GTAlignedOracle
from src.models.operator_spec import OperatorSpec
from src.models.operator_registry import OperatorRegistry
from src.eval.op_harness import OPHarness, HarnessConfig, HarnessResult

# ---------------------------------------------------------------------------
# Constants — IDENTICAL to EXP-OP2 except N_seeds
# ---------------------------------------------------------------------------

# N=100 seeds: first 20 are the original OP2 seeds, plus 80 additional
SEEDS = [
    # Original OP2 seeds (20)
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

LAMBDA_S          = 0.5
SIGMA_VALUE       = 0.4
N_PRE_SHIFT       = 200
N_POST_SHIFT      = 400
WINDOW_SIZE       = 50
TAU               = 0.1
ETA               = 0.05
ETA_NEG           = 1.0
TTL_FULL          = 400
TTL_HALF          = 150
PARTIAL_RNG_SEED  = 99        # IDENTICAL to OP2

CAMPAIGN = {cat: {"escalate_incident": 0.90} for cat in CATEGORIES}

RECOVERY_THRESHOLD_PP = 1.0
RECOVERY_HOLD_WINDOWS = 2
SENTINEL              = N_POST_SHIFT + 1    # 401 — never-recover marker

RESULTS_PATH = Path("experiments/synthesis/expOP2_n100/results.json")

N_FACTORS = 6
C_DIM = len(CATEGORIES)    # 5
A_DIM = len(ACTIONS)       # 4
AC_IDX  = ACTIONS.index("auto_close")          # 0
ESC_IDX = ACTIONS.index("escalate_incident")   # 3

conditions = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]

# N=20 OP2 reference values (hardcoded for side-by-side comparison)
OP2_N20_REFERENCE = {
    "A":    {"never_recover_pct": 20.0, "t_rec_mean": 178.0, "t_rec_std": 356.0},
    "B":    {"never_recover_pct":  5.0, "t_rec_mean":  55.0, "t_rec_std": 240.0},
    "B-exp":{"never_recover_pct": None, "t_rec_mean": None,  "t_rec_std": None},
    "C":    {"never_recover_pct": 35.0, "t_rec_mean": 425.0, "t_rec_std": 561.0},
    "C-exp":{"never_recover_pct": 35.0, "t_rec_mean": None,  "t_rec_std": None},
    "P-75": {"never_recover_pct": 20.0, "t_rec_mean": 228.0, "t_rec_std": 445.0},
    "P-50": {"never_recover_pct": None, "t_rec_mean": None,  "t_rec_std": None},
    "P-25": {"never_recover_pct": None, "t_rec_mean": None,  "t_rec_std": None},
    "P-0":  {"never_recover_pct": None, "t_rec_mean": None,  "t_rec_std": None},
}


# ---------------------------------------------------------------------------
# Helpers — IDENTICAL to EXP-OP2
# ---------------------------------------------------------------------------

def _build_profiles_tensor(gen: CategoryAlertGenerator) -> np.ndarray:
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(CATEGORIES):
        for a_idx, act in enumerate(ACTIONS):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def _evaluate(result_action_index: int, gt_action_index: int) -> bool:
    return bool(result_action_index == gt_action_index)


def compute_t_recovery(
    accuracy_curve: list,
    baseline_pre_shift: float,
    threshold_pp: float = RECOVERY_THRESHOLD_PP,
    hold_windows: int = RECOVERY_HOLD_WINDOWS,
    sentinel: int = SENTINEL,
) -> int:
    threshold = baseline_pre_shift - threshold_pp / 100.0
    n = len(accuracy_curve)
    for i in range(n - hold_windows + 1):
        if all(accuracy_curve[i + j] >= threshold for j in range(hold_windows)):
            return i * WINDOW_SIZE
    return sentinel


# ---------------------------------------------------------------------------
# Sigma constructors — IDENTICAL to EXP-OP2
# ---------------------------------------------------------------------------

def build_correct_sigma(C: int, A: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    s = np.zeros((C, A), dtype=np.float64)
    s[:, ac_idx]  = +SIGMA_VALUE
    s[:, esc_idx] = -SIGMA_VALUE
    return s


def build_harmful_sigma(C: int, A: int, ac_idx: int, esc_idx: int) -> np.ndarray:
    return -build_correct_sigma(C, A, ac_idx, esc_idx)


def build_partial_sigma(
    C: int, A: int, ac_idx: int, esc_idx: int, fraction_correct: float
) -> np.ndarray:
    """
    Start from correct sigma, flip (1-fraction_correct) of non-zero cells.
    Uses module-level PARTIAL_RNG for deterministic flip pattern — SAME as OP2.
    """
    s = build_correct_sigma(C, A, ac_idx, esc_idx)
    nonzero_indices = (
        [(c, ac_idx)  for c in range(C)] +
        [(c, esc_idx) for c in range(C)]
    )
    n_flip = int(len(nonzero_indices) * (1.0 - fraction_correct))
    if n_flip > 0:
        flip_choices = PARTIAL_RNG.choice(len(nonzero_indices), size=n_flip, replace=False)
        for idx in flip_choices:
            r, col = nonzero_indices[idx]
            s[r, col] = -s[r, col]
    return s


# Pre-build partial sigmas at module level — IDENTICAL RNG seed + call order to OP2
PARTIAL_RNG = np.random.default_rng(seed=PARTIAL_RNG_SEED)

sigma_P100 = build_correct_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX)
sigma_P75  = build_partial_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, 0.75)
sigma_P50  = build_partial_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, 0.50)
sigma_P25  = build_partial_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX, 0.25)
sigma_P0   = build_harmful_sigma(C_DIM, A_DIM, AC_IDX, ESC_IDX)


# ---------------------------------------------------------------------------
# Core helpers — IDENTICAL to EXP-OP2
# ---------------------------------------------------------------------------

def run_pre_shift(seed: int, profiles: np.ndarray) -> tuple:
    scorer = ProfileScorer(profiles.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    gen    = CategoryAlertGenerator(seed=seed)
    alerts = gen.generate(N_PRE_SHIFT)
    last100_correct = []
    for i, alert in enumerate(alerts):
        result     = scorer.score(alert.factors, alert.category_index, synthesis=None)
        is_correct = _evaluate(result.action_index, alert.gt_action_index)
        if i >= N_PRE_SHIFT - 100:
            last100_correct.append(is_correct)
        scorer.update(
            factors=alert.factors,
            category_index=alert.category_index,
            action_idx=result.action_index,
            correct=is_correct,
        )
    baseline_acc = float(np.mean(last100_correct))
    return scorer.mu.copy(), baseline_acc


def run_post_shift(
    starting_mu: np.ndarray,
    alerts: list,
    registry: OperatorRegistry,
    lambda_override: float = LAMBDA_S,
) -> HarnessResult:
    scorer = ProfileScorer(starting_mu.copy(), ACTIONS, tau=TAU, eta=ETA, eta_neg=ETA_NEG)
    oracle = GTAlignedOracle(noise_rate=0.0)
    config = HarnessConfig(
        n_decisions=N_POST_SHIFT,
        snapshot_interval=WINDOW_SIZE,
        use_synthesis=True,
        lambda_override=lambda_override,
        window_size=WINDOW_SIZE,
    )
    return OPHarness(scorer, oracle, registry, config).run(alerts)


def make_registry(spec: OperatorSpec | None, mu: np.ndarray) -> OperatorRegistry:
    reg = OperatorRegistry(n_categories=C_DIM, n_actions=A_DIM, n_factors=N_FACTORS)
    if spec is not None:
        reg.register(spec, mu)
    return reg


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion k/n. Returns (lo, hi) in [0, 1]."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom     = 1 + z**2 / n
    center    = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)


def bimodality_verdict(t_recovery_vals: list[int]) -> tuple[str, dict]:
    """
    Test B-exp T_recovery for bimodality.
    Primary: std > 0.5 * mean → BIMODAL.
    Secondary: try Hartigan's dip test if scipy available.
    Returns (verdict_str, evidence_dict).
    """
    arr = np.array(t_recovery_vals, dtype=float)
    mean_t = float(np.mean(arr))
    std_t  = float(np.std(arr))
    ratio  = std_t / mean_t if mean_t > 0 else 0.0

    simple_verdict = "BIMODAL" if ratio > 0.5 else "UNIMODAL"
    evidence = {"mean": mean_t, "std": std_t, "std_over_mean_ratio": ratio,
                "threshold": 0.5, "test": "std/mean > 0.5"}

    # Try Hartigan's dip test
    try:
        from scipy.stats import dip_test   # type: ignore[attr-defined]
        stat, p_val = dip_test(arr)
        dip_verdict = "BIMODAL" if p_val < 0.05 else "UNIMODAL"
        evidence["dip_stat"] = float(stat)
        evidence["dip_p_value"] = float(p_val)
        evidence["dip_verdict"] = dip_verdict
        evidence["test"] = "Hartigan dip test (p<0.05)"
        return dip_verdict, evidence
    except Exception:
        pass

    return simple_verdict, evidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    t_start = time.time()

    print("=" * 70)
    print("EXP-OP2-N100: HARMFUL CLAIM RESILIENCE (N=100 SEEDS)")
    print("=" * 70)
    print(f"Seeds:   {len(SEEDS)}")
    print(f"Conds:   {conditions}")
    print(f"lambda={LAMBDA_S}, N_pre={N_PRE_SHIFT}, N_post={N_POST_SHIFT}, tau={TAU}")
    print(f"Total runs: {len(SEEDS) * len(conditions)}")
    print()

    # results[cond] = list of per-seed dicts
    results: dict = {cond: [] for cond in conditions}

    # Condition → OperatorSpec factory (lambda over mu, seed) — IDENTICAL to OP2
    cond_specs: dict = {
        "A":     None,
        "B":     lambda mu, seed: OperatorSpec(
                     f"B_{seed}", "active_campaign", 0,
                     sigma_P100, LAMBDA_S, TTL_FULL),
        "B-exp": lambda mu, seed: OperatorSpec(
                     f"Bexp_{seed}", "active_campaign", 0,
                     sigma_P100, LAMBDA_S, TTL_HALF),
        "C":     lambda mu, seed: OperatorSpec(
                     f"C_{seed}", "active_campaign", 0,
                     sigma_P0, LAMBDA_S, TTL_FULL),
        "C-exp": lambda mu, seed: OperatorSpec(
                     f"Cexp_{seed}", "active_campaign", 0,
                     sigma_P0, LAMBDA_S, TTL_HALF),
        "P-75":  lambda mu, seed: OperatorSpec(
                     f"P75_{seed}", "active_campaign", 0,
                     sigma_P75, LAMBDA_S, TTL_FULL),
        "P-50":  lambda mu, seed: OperatorSpec(
                     f"P50_{seed}", "active_campaign", 0,
                     sigma_P50, LAMBDA_S, TTL_FULL),
        "P-25":  lambda mu, seed: OperatorSpec(
                     f"P25_{seed}", "active_campaign", 0,
                     sigma_P25, LAMBDA_S, TTL_FULL),
        "P-0":   lambda mu, seed: OperatorSpec(
                     f"P0_{seed}", "active_campaign", 0,
                     sigma_P0, LAMBDA_S, TTL_FULL),
    }

    total_runs = 0

    for seed_idx, seed in enumerate(SEEDS):
        # Pre-shift (shared across all conditions for this seed)
        gen_pre     = CategoryAlertGenerator(seed=seed)
        gt_profiles = _build_profiles_tensor(gen_pre)
        pre_mu, baseline_acc = run_pre_shift(seed, gt_profiles)

        # Post-shift alerts (shared across conditions — same campaign)
        post_gen    = CategoryAlertGenerator(seed=seed + 10000)
        post_alerts = post_gen.generate_campaign(N_POST_SHIFT, CAMPAIGN)

        for cond in conditions:
            spec_fn = cond_specs[cond]
            spec    = spec_fn(pre_mu, seed) if spec_fn is not None else None
            reg     = make_registry(spec, pre_mu)
            res     = run_post_shift(pre_mu, post_alerts, reg)

            acc_curve = [float(v) for v in res.auac_result.accuracy_curve]
            t_rec     = compute_t_recovery(acc_curve, baseline_acc)

            results[cond].append({
                "auac":               float(res.auac_result.auac),
                "accuracy_curve":     acc_curve,
                "t_recovery":         int(t_rec),
                "baseline_pre_shift": float(baseline_acc),
                "n_expired":          int(res.n_operators_expired),
                "never_recover":      bool(t_rec >= SENTINEL),
            })

            total_runs += 1

        if (seed_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate    = total_runs / elapsed
            remaining = (len(SEEDS) * len(conditions) - total_runs) / rate
            print(f"  [{total_runs:4d}/{len(SEEDS)*len(conditions)}]  "
                  f"seed {seed:6d} done  "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                  flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nAll {total_runs} runs complete in {elapsed_total/60:.2f} min.")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------

    n_seeds = len(SEEDS)

    def get_auacs(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["auac"] for s in range(n_seeds)])

    def get_t_recs(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["t_recovery"] for s in range(n_seeds)])

    def get_never(cond: str) -> np.ndarray:
        return np.array([results[cond][s]["never_recover"] for s in range(n_seeds)])

    # AUAC stats + t-test vs A
    from scipy import stats as _stats   # noqa: F401 (optional)

    auacs_A = get_auacs("A")
    auac_stats = {}
    for cond in conditions:
        arr = get_auacs(cond)
        d   = arr - auacs_A
        try:
            _, p = _stats.ttest_rel(arr, auacs_A)
        except Exception:
            p = float("nan")
        auac_stats[cond] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "delta_vs_A": float(d.mean()), "p_value": float(p),
        }

    # T_recovery stats
    t_rec_stats = {}
    for cond in conditions:
        arr   = get_t_recs(cond)
        t_rec_stats[cond] = {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "p25":    float(np.percentile(arr, 25)),
            "p50":    float(np.percentile(arr, 50)),
            "p75":    float(np.percentile(arr, 75)),
            "p90":    float(np.percentile(arr, 90)),
        }

    # Never-recover stats + Wilson CI
    never_stats = {}
    for cond in conditions:
        arr  = get_never(cond)
        k    = int(arr.sum())
        rate = float(k / n_seeds)
        lo, hi = wilson_ci(k, n_seeds)
        never_stats[cond] = {
            "count": k, "rate": rate,
            "pct":   rate * 100,
            "ci_lo": lo, "ci_hi": hi,
            "ci_lo_pct": lo * 100, "ci_hi_pct": hi * 100,
        }

    # B-exp bimodality
    bexp_t_recs   = [results["B-exp"][s]["t_recovery"] for s in range(n_seeds)]
    bimod_verdict, bimod_evidence = bimodality_verdict(bexp_t_recs)

    # ---------------------------------------------------------------------------
    # Print: Full table
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FULL RESULTS TABLE (N=100 seeds)")
    print("=" * 80)
    print(f"{'Cond':>6}  {'AUAC':>7}  {'delta_A':>8}  {'p_val':>7}  "
          f"{'T_rec_mean':>11}  {'T_rec_p90':>10}  "
          f"{'NR%':>5}  {'95% CI':>17}  {'N=20 NR%':>10}")
    print("-" * 95)
    for cond in conditions:
        a  = auac_stats[cond]
        t  = t_rec_stats[cond]
        nr = never_stats[cond]
        ref_nr = OP2_N20_REFERENCE[cond]["never_recover_pct"]
        ref_str = f"{ref_nr:.0f}%" if ref_nr is not None else "  n/a"
        sig = "*" if a["p_value"] < 0.05 else " "
        print(f"{cond:>6}  {a['mean']:.4f}  {a['delta_vs_A']:+.4f}{sig}  "
              f"{a['p_value']:7.4f}  "
              f"{t['mean']:11.1f}  {t['p90']:10.1f}  "
              f"{nr['pct']:5.1f}%  "
              f"[{nr['ci_lo_pct']:5.1f}%, {nr['ci_hi_pct']:5.1f}%]  "
              f"{ref_str:>10}")
    print("  * = p<0.05 vs condition A")

    # ---------------------------------------------------------------------------
    # Print: Side-by-side N=20 vs N=100
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("N=20 vs N=100 COMPARISON (never-recover rate)")
    print("=" * 70)
    print(f"{'Cond':>6}  {'N=20 NR%':>10}  {'N=100 NR%':>11}  "
          f"{'N=100 95% CI':>20}  {'Change':>8}")
    print("-" * 65)
    for cond in conditions:
        nr  = never_stats[cond]
        ref = OP2_N20_REFERENCE[cond]["never_recover_pct"]
        if ref is not None:
            change = nr["pct"] - ref
            sign   = "+" if change >= 0 else ""
            ref_s  = f"{ref:.0f}%"
            chg_s  = f"{sign}{change:.1f}pp"
        else:
            ref_s  = "  n/a"
            chg_s  = "  n/a"
        print(f"{cond:>6}  {ref_s:>10}  {nr['pct']:10.1f}%  "
              f"[{nr['ci_lo_pct']:5.1f}%, {nr['ci_hi_pct']:5.1f}%]  {chg_s:>8}")

    # ---------------------------------------------------------------------------
    # Print: B-exp bimodality verdict
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("B-EXP BIMODALITY CHECK")
    print("=" * 70)
    print(f"  Verdict:    {bimod_verdict}")
    print(f"  Mean T_rec: {bimod_evidence['mean']:.1f} decisions")
    print(f"  Std T_rec:  {bimod_evidence['std']:.1f} decisions")
    print(f"  std/mean:   {bimod_evidence['std_over_mean_ratio']:.3f}  "
          f"(threshold = {bimod_evidence['threshold']})")
    if "dip_p_value" in bimod_evidence:
        print(f"  Dip test:   stat={bimod_evidence['dip_stat']:.4f}, "
              f"p={bimod_evidence['dip_p_value']:.4f}  [{bimod_evidence['dip_verdict']}]")
    print(f"  Test used:  {bimod_evidence['test']}")
    nr_bexp = never_stats["B-exp"]
    print(f"  Never-recover: {nr_bexp['count']}/{n_seeds} ({nr_bexp['pct']:.1f}%)  "
          f"95% CI [{nr_bexp['ci_lo_pct']:.1f}%, {nr_bexp['ci_hi_pct']:.1f}%]")

    # ---------------------------------------------------------------------------
    # Print: Safety policy table
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAFETY POLICY TABLE")
    print("=" * 70)
    print(f"  {'Condition':>8}  {'NR%':>6}  {'95% CI':>20}  Policy")
    print("  " + "-" * 60)
    for cond in conditions:
        nr = never_stats[cond]
        if nr["pct"] > 10.0:
            policy = "REQUIRES CHECKPOINT+ROLLBACK"
        elif nr["pct"] <= 5.0:
            policy = "SAFE FOR DEPLOYMENT"
        else:
            policy = "MONITORING REQUIRED"
        print(f"  {cond:>8}  {nr['pct']:5.1f}%  "
              f"[{nr['ci_lo_pct']:5.1f}%, {nr['ci_hi_pct']:5.1f}%]  {policy}")

    print("=" * 70)
    print(f"\nTotal runtime: {elapsed_total/60:.2f} minutes")

    # ---------------------------------------------------------------------------
    # Save results.json
    # ---------------------------------------------------------------------------

    results_data = {
        "config": {
            "n_seeds": n_seeds,
            "lambda_s": LAMBDA_S,
            "tau": TAU,
            "n_pre": N_PRE_SHIFT,
            "n_post": N_POST_SHIFT,
            "sentinel": SENTINEL,
            "window_size": WINDOW_SIZE,
        },
        "per_seed_results": {
            cond: [
                {
                    "seed":               SEEDS[s],
                    "auac":               float(results[cond][s]["auac"]),
                    "accuracy_curve":     [float(v) for v in results[cond][s]["accuracy_curve"]],
                    "t_recovery":         int(results[cond][s]["t_recovery"]),
                    "never_recover":      bool(results[cond][s]["never_recover"]),
                    "baseline_pre_shift": float(results[cond][s]["baseline_pre_shift"]),
                }
                for s in range(n_seeds)
            ]
            for cond in conditions
        },
        "summary": {
            "auac_stats":   auac_stats,
            "t_rec_stats":  t_rec_stats,
            "never_recover": never_stats,
            "bimodality": {
                "condition": "B-exp",
                "verdict": bimod_verdict,
                "evidence": bimod_evidence,
            },
        },
        "t_recovery_arrays": {
            cond: [int(results[cond][s]["t_recovery"]) for s in range(n_seeds)]
            for cond in conditions
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fout:
        json.dump(results_data, fout, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

    # ---------------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------------

    from experiments.synthesis.expOP2_n100.charts import generate_charts
    generate_charts(json.load(open(RESULTS_PATH)))
