"""
FX-T5-A4: Auto-Approve Band Action Analysis on A=4 Geometry.

Context: FX-T5-BREAKDOWN found monitor accuracy=85.95% at A=5 with ≥0.90 threshold,
making it the dominant source of dangerous auto-approve errors (2.03% of band).
The hypothesis: monitor/refer_to_analyst probability mass collision at A=5 compressed
confidence and caused systematic monitor confusion. At A=4, monitor Voronoi space is
cleaner — the 5th action's probability mass redistributed to the remaining 4.

Config: soc_product_v50 (C=6, A=4, d=6). Static warm-start scorer (matching FX-T5
intent: measure threshold quality, not learning dynamics).

Actions: escalate(0), investigate(1), suppress(2), monitor(3)
THREAT_ACTIONS:  {escalate, investigate}   — under-response is dangerous
CAUTION_ACTIONS: {suppress, monitor}       — over-caution is expensive but not dangerous

Dangerous error: predicted CAUTION (suppress/monitor) but GT = THREAT (escalate/investigate).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_SEEDS             = 50
N_DECISIONS         = 1000
N_WARMUP            = 500       # warm-start centroid drift before evaluation
TAU                 = 0.1
ETA                 = 0.05
ETA_NEG             = 0.05
NOISE_RATE          = 0.10
BAND_THRESHOLD      = 0.90
COST_RATIO          = 20.0
DOMAIN_CONFIG       = "soc_product_v50"
RANDOM_SEED_BASE    = 42

THREAT_ACTIONS      = {0, 1}    # escalate, investigate
CAUTION_ACTIONS     = {2, 3}    # suppress, monitor

THRESHOLD_SCAN      = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
TARGET_ACC_CAUTION  = 0.99
TARGET_ACC_THREAT   = 0.90

# A=5 reference values from FX-T5-BREAKDOWN (realistic generator, mode=combined)
A5_REF = {
    "escalate":    {"accuracy": 1.0000, "pct_band": 0.0020},
    "investigate": {"accuracy": 0.9218, "pct_band": 0.0878},
    "suppress":    {"accuracy": 0.9986, "pct_band": 0.3011},
    "monitor":     {"accuracy": 0.8595, "pct_band": 0.6091},
    "total_band":  11502,
    "mean_band_per_seed": 230.04,
    "dangerous_count": 234,
    "dangerous_rate":   0.0203,
}

RESULTS_PATH = _REPO_ROOT / "experiments" / "fx_t5_a4" / "results.json"

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

assert A == 4, f"Expected A=4, got A={A}"
assert ACTIONS == ["escalate", "investigate", "suppress", "monitor"], \
    f"Unexpected action order: {ACTIONS}"

print("=" * 60)
print("FX-T5-A4: AUTO-APPROVE BAND ANALYSIS (A=4 geometry)")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Actions: {ACTIONS}")
print(f"Categories: {CATEGORIES}")
print(f"N_SEEDS={N_SEEDS}, N_WARMUP={N_WARMUP}, N_DECISIONS={N_DECISIONS}")
print(f"BAND_THRESHOLD={BAND_THRESHOLD}, NOISE_RATE={NOISE_RATE}")
print()


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------

@dataclass
class BandDecision:
    predicted:  int
    gt:         int
    correct:    bool
    conf:       float
    category:   int
    error_type: str   # "correct", "dangerous", "safe", "over_escalation"


def classify_error(predicted: int, gt: int) -> str:
    if predicted == gt:
        return "correct"
    if predicted in CAUTION_ACTIONS and gt in THREAT_ACTIONS:
        return "dangerous"        # said "fine", was a threat
    if predicted in THREAT_ACTIONS and gt in CAUTION_ACTIONS:
        return "over_escalation"  # over-reacted
    return "safe"                 # within-tier confusion


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    seed:   int = 0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.array([
        float(rng.choice(arr, size=len(arr), replace=True).mean())
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def threshold_accuracy(
    decisions: list[BandDecision], action_idx: int, threshold: float,
) -> tuple[float, int]:
    subset = [d for d in decisions if d.predicted == action_idx and d.conf >= threshold]
    if not subset:
        return float("nan"), 0
    return float(np.mean([d.correct for d in subset])), len(subset)


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> list[BandDecision]:
    """
    Warm-start scorer (N_WARMUP decisions with learning), then evaluate
    N_DECISIONS decisions with learning ON but NOISE_RATE=0.10 oracle.
    Record band decisions (conf >= BAND_THRESHOLD) from evaluation window only.
    """
    rng_oracle = np.random.default_rng(seed + 99999)

    gen = CategoryAlertGenerator(
        **config["generator_kwargs"],
        noise_rate=0.0,   # factor noise off; oracle carries feedback noise
        seed=RANDOM_SEED_BASE + seed,
    )

    scorer = ProfileScorer(
        config["mu"].copy(), config["actions"],
        tau=TAU, eta=ETA, eta_neg=ETA_NEG,
    )

    # --- Warmup: learning on, no recording ---
    warmup_alerts = gen.generate(N_WARMUP)
    for alert in warmup_alerts:
        result     = scorer.score(alert.factors, alert.category_index)
        is_correct = result.action_index == alert.gt_action_index
        scorer.update(
            alert.factors, alert.category_index,
            result.action_index, correct=is_correct,
            gt_action_index=alert.gt_action_index,
        )

    # --- Evaluation: learning on, record band decisions ---
    eval_alerts = gen.generate(N_DECISIONS)
    decisions: list[BandDecision] = []

    for alert in eval_alerts:
        result = scorer.score(alert.factors, alert.category_index)
        conf   = float(result.confidence)

        # Noisy oracle feedback for update (doesn't affect what we record)
        is_correct_true = result.action_index == alert.gt_action_index
        if rng_oracle.random() < NOISE_RATE:
            if is_correct_true:
                wrong_gt = (alert.gt_action_index + 1) % A
                scorer.update(
                    alert.factors, alert.category_index,
                    result.action_index, correct=False,
                    gt_action_index=wrong_gt,
                )
            else:
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

        # Record if in band (ground truth correctness, not noisy feedback)
        if conf >= BAND_THRESHOLD:
            predicted = result.action_index
            gt        = alert.gt_action_index
            decisions.append(BandDecision(
                predicted  = predicted,
                gt         = gt,
                correct    = bool(predicted == gt),
                conf       = conf,
                category   = alert.category_index,
                error_type = classify_error(predicted, gt),
            ))

    return decisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_decisions: list[BandDecision] = []
per_seed_action_acc: dict[int, list[float]] = {i: [] for i in range(A)}
overall_n_per_seed: list[int] = []

for seed in range(N_SEEDS):
    if (seed + 1) % 10 == 0 or seed == 0:
        print(f"  seed {seed+1}/{N_SEEDS} ...", flush=True)
    decs = run_seed(seed)
    all_decisions.extend(decs)
    overall_n_per_seed.append(len(decs))
    for act_idx in range(A):
        act_decs = [d for d in decs if d.predicted == act_idx]
        if act_decs:
            per_seed_action_acc[act_idx].append(
                float(np.mean([d.correct for d in act_decs]))
            )

total_band = len(all_decisions)
print(f"\n  Total band decisions: {total_band:,}  "
      f"(mean {total_band / N_SEEDS:.0f}/seed, "
      f"{total_band / (N_SEEDS * N_DECISIONS) * 100:.1f}% of all)")

# ---------------------------------------------------------------------------
# Q1: Action distribution
# ---------------------------------------------------------------------------

action_count = {i: sum(1 for d in all_decisions if d.predicted == i) for i in range(A)}
action_pct   = {i: (action_count[i] / total_band if total_band > 0 else 0.0) for i in range(A)}

# ---------------------------------------------------------------------------
# Q2: Per-action accuracy
# ---------------------------------------------------------------------------

per_action_stats: dict[int, dict] = {}
for act_idx in range(A):
    seed_accs = per_seed_action_acc[act_idx]
    if seed_accs:
        mean_acc = float(np.mean(seed_accs))
        ci_lo, ci_hi = bootstrap_ci(seed_accs, seed=act_idx)
    else:
        mean_acc = ci_lo = ci_hi = float("nan")
    per_action_stats[act_idx] = {
        "count":    action_count[act_idx],
        "pct_band": action_pct[act_idx],
        "accuracy": mean_acc,
        "ci_lo":    ci_lo,
        "ci_hi":    ci_hi,
        "n_seeds":  len(seed_accs),
    }

# ---------------------------------------------------------------------------
# Q3: Error direction
# ---------------------------------------------------------------------------

errors     = [d for d in all_decisions if not d.correct]
dangerous  = [d for d in errors if d.error_type == "dangerous"]
safe_errs  = [d for d in errors if d.error_type == "safe"]
over_esc   = [d for d in errors if d.error_type == "over_escalation"]

n_dangerous          = len(dangerous)
n_safe               = len(safe_errs)
n_over_esc           = len(over_esc)
total_errors         = len(errors)
dangerous_error_rate = n_dangerous / total_band if total_band > 0 else 0.0

# ---------------------------------------------------------------------------
# Cost-weighted analysis
# ---------------------------------------------------------------------------

correct_decisions = total_band - total_errors
other_errors      = n_safe + n_over_esc
cost_weighted_score = (
    correct_decisions * 1.0
    - n_dangerous * COST_RATIO
    - other_errors * 1.0
)
cost_weighted_acc = cost_weighted_score / total_band if total_band > 0 else 0.0
nominal_acc       = correct_decisions / total_band if total_band > 0 else 0.0

# ---------------------------------------------------------------------------
# Recommended thresholds
# ---------------------------------------------------------------------------

recommended_thresholds: dict[int, float | str] = {}
threshold_coverage:     dict[int, dict[float, float]] = {i: {} for i in range(A)}

for act_idx in range(A):
    act_all = [d for d in all_decisions if d.predicted == act_idx]
    target  = TARGET_ACC_CAUTION if act_idx in CAUTION_ACTIONS else TARGET_ACC_THREAT
    found   = False
    for thr in THRESHOLD_SCAN:
        subset   = [d for d in act_all if d.conf >= thr]
        n_all    = len(act_all)
        cov      = len(subset) / n_all if n_all > 0 else 0.0
        threshold_coverage[act_idx][thr] = cov
        if len(subset) < 10:
            recommended_thresholds[act_idx] = "insufficient_data"
            found = True
            break
        acc = float(np.mean([d.correct for d in subset]))
        if acc >= target:
            recommended_thresholds[act_idx] = thr
            found = True
            break
    if not found:
        recommended_thresholds[act_idx] = "never_reaches_target"

# ---------------------------------------------------------------------------
# Dangerous errors by category
# ---------------------------------------------------------------------------

dangerous_by_cat: dict[int, int] = {i: 0 for i in range(C)}
for d in dangerous:
    dangerous_by_cat[d.category] += 1

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("=== FX-T5-A4: AUTO-APPROVE BAND ANALYSIS (A=4) ===")
print("=" * 60)
print()
print(f"  {'Action':<13}  {'Acc(A=5)':>9}  {'Acc(A=4)':>9}  "
      f"{'Delta':>8}  {'Band%(A=5)':>11}  {'Band%(A=4)':>11}")
print("  " + "-" * 68)
for act_idx, act_name in enumerate(ACTIONS):
    ref5   = A5_REF[act_name]
    acc5   = ref5["accuracy"]
    pct5   = ref5["pct_band"]
    acc4   = per_action_stats[act_idx]["accuracy"]
    pct4   = per_action_stats[act_idx]["pct_band"]
    delta  = acc4 - acc5 if not np.isnan(acc4) else float("nan")
    flag   = ""
    if act_idx in CAUTION_ACTIONS and not np.isnan(acc4):
        flag = " ✓" if acc4 >= 0.99 else (" ↑" if delta > 0.02 else " ⚠")
    delta_s = f"{delta:+.1%}" if not np.isnan(delta) else "N/A"
    print(f"  {act_name:<13}  {acc5:>9.1%}  {acc4:>9.1%}  "
          f"{delta_s:>8}  {pct5:>11.1%}  {pct4:>11.1%}{flag}")

print()
print(f"  Dangerous errors: {n_dangerous}/{total_band} "
      f"({dangerous_error_rate:.2%})")
print(f"  (Was {A5_REF['dangerous_count']}/{A5_REF['total_band']} "
      f"= {A5_REF['dangerous_rate']:.2%} at A=5)")
print()

# Monitor accuracy verdict
monitor_acc4 = per_action_stats[3]["accuracy"]
if not np.isnan(monitor_acc4):
    if monitor_acc4 >= 0.95:
        print("  VERDICT: monitor accuracy > 95% at A=4.")
        print("    => refer/monitor collision WAS the dominant cause.")
        print("    => Symmetric threshold adequate at A=4.")
    elif monitor_acc4 >= 0.90:
        print("  VERDICT: monitor accuracy 90–95% at A=4 (improved but not resolved).")
        print("    => Collision was a contributing factor but not the only one.")
        print("    => Asymmetric threshold may still help.")
    else:
        print("  VERDICT: monitor accuracy still < 90% at A=4.")
        print("    => Problem is structural, not caused by refer collision.")
        print("    => Per-action threshold matrix still needed.")

# Q1 distribution
print()
print("--- Q1: Action Distribution in Auto-Approve Band ---")
for act_idx, act_name in enumerate(ACTIONS):
    flag = " ◄ high-cost" if act_idx in CAUTION_ACTIONS else ""
    print(f"  {act_name:<15}  {action_pct[act_idx]*100:>5.1f}% of band  "
          f"({action_count[act_idx]:>6,} decisions){flag}")

# Q2 detailed accuracy
print()
print("--- Q2: Per-Action Accuracy (A=4) ---")
print(f"  {'Action':<15} {'Accuracy':>10} {'95% CI':>22} {'Count':>8}  {'Target':>8}")
print("  " + "─" * 68)
for act_idx, act_name in enumerate(ACTIONS):
    s      = per_action_stats[act_idx]
    target = TARGET_ACC_CAUTION if act_idx in CAUTION_ACTIONS else TARGET_ACC_THREAT
    flag   = ""
    if act_idx in CAUTION_ACTIONS and not np.isnan(s["accuracy"]):
        flag = " ⚠" if s["accuracy"] < 0.99 else " ✓"
    ci_str = (f"[{s['ci_lo']*100:.1f}%, {s['ci_hi']*100:.1f}%]"
              if not np.isnan(s["ci_lo"]) else "n/a")
    print(f"  {act_name:<15} {s['accuracy']*100:>9.2f}%  {ci_str:>22}"
          f"  {s['count']:>7,}  {target*100:>7.0f}%{flag}")

# Q3 error direction
print()
print("--- Q3: Error Direction ---")
print(f"  Total errors: {total_errors:,} "
      f"({total_errors / total_band * 100:.2f}% of {total_band:,} band decisions)")
if total_errors > 0:
    print(f"  Dangerous:        {n_dangerous:>6,}  "
          f"({dangerous_error_rate * 100:.3f}% of band, "
          f"{n_dangerous / total_errors * 100:.1f}% of errors)")
    print(f"  Safe:             {n_safe:>6,}  "
          f"({n_safe / total_errors * 100:.1f}% of errors)")
    print(f"  Over-escalation:  {n_over_esc:>6,}  "
          f"({n_over_esc / total_errors * 100:.1f}% of errors)")
    if n_dangerous > 0:
        print(f"\n  Dangerous by category:")
        for cat_idx, cat_name in enumerate(CATEGORIES):
            n = dangerous_by_cat.get(cat_idx, 0)
            if n > 0:
                print(f"    {cat_name:<28} {n:>4}  "
                      f"({n / n_dangerous * 100:.1f}%)")

# Cost analysis
print()
print(f"--- Cost-Weighted Analysis (COST_RATIO={COST_RATIO:.0f}:1) ---")
print(f"  Correct:           {correct_decisions:>8,}  × +1.0 = {correct_decisions:>+12,.1f}")
print(f"  Dangerous errors:  {n_dangerous:>8,}  × -{COST_RATIO:.0f}  = {-n_dangerous * COST_RATIO:>+12,.1f}")
print(f"  Other errors:      {other_errors:>8,}  × -1.0 = {-other_errors:>+12,.1f}")
print(f"  Net score:         {cost_weighted_score:>+12,.1f}")
print(f"  Nominal accuracy:  {nominal_acc * 100:.2f}%")
print(f"  Cost-weighted:     {cost_weighted_acc:>+.4f}")

# Recommended thresholds
print()
print("--- Recommended Per-Action Thresholds ---")
print(f"  {'Action':<15} {'Recommended':>16}  {'Target acc':>11}")
print("  " + "─" * 48)
for act_idx, act_name in enumerate(ACTIONS):
    thr    = recommended_thresholds[act_idx]
    target = TARGET_ACC_CAUTION if act_idx in CAUTION_ACTIONS else TARGET_ACC_THREAT
    current = ("SAME as 0.90" if thr == 0.90
               else f"RAISE to {thr}" if isinstance(thr, float)
               else thr)
    print(f"  {act_name:<15} {current:>16}  {target * 100:.0f}%")

# Design verdict
print()
print("=== DESIGN VERDICT ===")
print(f"  Dangerous error rate: {dangerous_error_rate * 100:.3f}%")
print(f"  Cost-weighted score:  {cost_weighted_acc:+.4f}")

if dangerous_error_rate < 0.002 and cost_weighted_acc > 0:
    verdict = "A"
    print("RESULT A: Symmetric threshold adequate at A=4.")
    print("  Dangerous error rate < 0.2%. Cost-weighted score positive.")
elif dangerous_error_rate < 0.005 and cost_weighted_acc > 0:
    verdict = "B"
    print("RESULT B: Asymmetric thresholds still recommended.")
    print("  Dangerous error rate 0.2–0.5%. Proceed with per-action thresholds.")
else:
    verdict = "C"
    print("RESULT C: LLM judge panel still required.")
    print("  Dangerous error rate > 0.5% or cost-weighted ≤ 0.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)

output = {
    "experiment":             "FX-T5-A4",
    "domain_config":          DOMAIN_CONFIG,
    "n_seeds":                N_SEEDS,
    "n_warmup":               N_WARMUP,
    "n_decisions":            N_DECISIONS,
    "tau":                    TAU,
    "eta":                    ETA,
    "eta_neg":                ETA_NEG,
    "noise_rate":             NOISE_RATE,
    "band_threshold":         BAND_THRESHOLD,
    "cost_ratio":             COST_RATIO,
    "ontology":               {"C": C, "A": A, "d": d},
    "total_band_decisions":   total_band,
    "mean_band_per_seed":     total_band / N_SEEDS,
    "band_pct_of_all":        total_band / (N_SEEDS * N_DECISIONS),
    "q1_action_distribution": {
        ACTIONS[i]: {"count": action_count[i], "pct_band": round(action_pct[i], 6)}
        for i in range(A)
    },
    "q2_per_action_accuracy": {
        ACTIONS[i]: {
            "count":    per_action_stats[i]["count"],
            "pct_band": round(per_action_stats[i]["pct_band"], 6),
            "accuracy": round(per_action_stats[i]["accuracy"], 6),
            "ci_lo":    round(per_action_stats[i]["ci_lo"], 6),
            "ci_hi":    round(per_action_stats[i]["ci_hi"], 6),
        }
        for i in range(A)
    },
    "q3_error_direction": {
        "total_errors":           total_errors,
        "dangerous":              n_dangerous,
        "safe":                   n_safe,
        "over_escalation":        n_over_esc,
        "dangerous_error_rate":   round(dangerous_error_rate, 8),
        "dangerous_by_category":  {
            CATEGORIES[i]: dangerous_by_cat.get(i, 0) for i in range(C)
        },
    },
    "cost_analysis": {
        "correct_decisions":   correct_decisions,
        "dangerous_errors":    n_dangerous,
        "other_errors":        other_errors,
        "cost_weighted_score": round(cost_weighted_score, 4),
        "cost_weighted_acc":   round(cost_weighted_acc, 6),
        "nominal_acc":         round(nominal_acc, 6),
    },
    "recommended_thresholds": {
        ACTIONS[i]: (float(recommended_thresholds[i])
                     if isinstance(recommended_thresholds[i], float)
                     else recommended_thresholds[i])
        for i in range(A)
    },
    "design_verdict":  verdict,
    "a5_reference":    A5_REF,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
    json.dump(output, fh, indent=2, cls=_NumpyEncoder)
print(f"\nResults saved to {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

import subprocess
charts_path = Path(__file__).parent / "charts.py"
subprocess.run(
    [sys.executable, str(charts_path)],
    check=True, cwd=str(_REPO_ROOT),
)
