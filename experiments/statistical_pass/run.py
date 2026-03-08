"""
Statistical Pass: 50-Seed Published Claims Validation
experiments/statistical_pass/run.py

Standing rule (2026-03-08-AM): any number appearing in a published claim
requires >=50 seeds + reported 95% CI + one-sample t-test at gate boundary.
10-seed results from FX-1-CORRECTED and FX-1-LEARNING are directional only.

Five published-claim numbers hardened here:
  T1: static accuracy (no learning)           — FX-1-CORRECTED combined
  T2: learning accuracy at decision 1000      — FX-1-LEARNING checkpoint
  T3: learning accuracy at decision 1500      — FX-1-LEARNING checkpoint
  T4: credential_access accuracy at dec 1000  — per-category weakest link
  T5: auto-approve band accuracy (conf >=0.90, static)

DI-06: Is the dec-1500 > dec-1000 improvement real, or 10-seed noise?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.profile_scorer import ProfileScorer
from experiments.fx1_proxy_real.realistic_generator import (
    RealisticAlertGenerator,
    SOC_CATEGORIES,
    SOC_ACTIONS,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_SEEDS  = 50
N_ALERTS = 2000
TAU      = 0.1
MODE     = "combined"
ROLLING_WIN = 200        # window for learning trajectory accuracy

CRED_ACCESS_IDX = 1      # credential_access is category index 1
AUTO_APPROVE_CONF = 0.90 # confidence threshold for auto-approve band

EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
PAPER_DIR    = REPO_ROOT / "paper_figures"

# Reference values from 10-seed runs
REF_T1 = 0.7145   # FX-1-CORRECTED static combined
REF_T2 = 0.7765   # FX-1-LEARNING @1000
REF_T3 = 0.8000   # FX-1-LEARNING @1500
REF_T4 = 0.6640   # FX-1-LEARNING credential_access @1000
REF_T5 = 0.9147   # FX-1-CORRECTED auto-approve accuracy


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(
    values:   list[float],
    gate:     float | None,
    n_boot:   int = 10_000,
    rng_seed: int = 0,
) -> dict:
    """
    Full statistical summary for a list of per-seed values.

    Returns dict with: mean, std, se, ci95_lo, ci95_hi,
    boot_lo, boot_hi, t_stat, p_value, n.
    """
    arr  = np.array(values, dtype=np.float64)
    n    = len(arr)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1))
    se   = std / np.sqrt(n)
    ci95_lo = mean - 1.96 * se
    ci95_hi = mean + 1.96 * se

    # Bootstrap CI
    rng  = np.random.default_rng(rng_seed)
    boot = np.array([
        float(rng.choice(arr, size=n, replace=True).mean())
        for _ in range(n_boot)
    ])
    boot_lo = float(np.percentile(boot, 2.5))
    boot_hi = float(np.percentile(boot, 97.5))

    t_stat  = None
    p_value = None
    if gate is not None:
        t_stat, p_value = stats.ttest_1samp(arr, gate)
        t_stat  = float(t_stat)
        p_value = float(p_value)

    return {
        "n":       n,
        "mean":    mean,
        "std":     std,
        "se":      se,
        "ci95_lo": ci95_lo,
        "ci95_hi": ci95_hi,
        "boot_lo": boot_lo,
        "boot_hi": boot_hi,
        "t_stat":  t_stat,
        "p_value": p_value,
    }


def print_target(
    label:  str,
    s:      dict,
    gate:   float | None,
    ref_10: float | None = None,
) -> None:
    """Print formatted statistics block for one target number."""
    print(f"\n{label}")
    print(f"  n={s['n']} seeds, N={N_ALERTS} alerts/seed")
    print(f"  Mean:            {s['mean']*100:.3f}%")
    print(f"  Std:             {s['std']*100:.3f}%")
    print(f"  95% CI (normal): [{s['ci95_lo']*100:.2f}%, {s['ci95_hi']*100:.2f}%]")
    print(f"  95% CI (boot):   [{s['boot_lo']*100:.2f}%, {s['boot_hi']*100:.2f}%]")
    if ref_10 is not None:
        print(f"  10-seed ref:     {ref_10*100:.2f}%  "
              f"(delta: {(s['mean']-ref_10)*100:+.2f}pp)")
    if gate is not None:
        direction = "above" if s["mean"] > gate else "below"
        print(f"  Gate:            {gate*100:.1f}%  ({direction})")
        t, p = s["t_stat"], s["p_value"]
        if p < 0.001:
            sig = "p<0.001"
        elif p < 0.01:
            sig = f"p={p:.4f}"
        elif p < 0.05:
            sig = f"p={p:.3f}"
        else:
            sig = f"p={p:.3f} (not significant at α=0.05)"
        print(f"  t-test vs gate:  t={t:.3f}, {sig}")
        lo, hi = s["ci95_lo"], s["ci95_hi"]
        if hi < gate:
            print(f"  ✓ CI entirely below gate — FAIL is statistically confirmed")
        elif lo > gate:
            print(f"  ✓ CI entirely above gate — PASS is statistically confirmed")
        else:
            print(f"  ⚠ CI crosses gate — result is statistically ambiguous")


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> dict:
    """
    Run one seed for both static and learning conditions.

    Returns per-seed values for all five targets.
    """
    gen         = RealisticAlertGenerator(mode=MODE, seed=seed)
    alerts      = gen.generate(N_ALERTS)
    gt_profiles = gen.get_profiles()

    # Mandatory taxonomy assertion
    assert gen.categories == SOC_CATEGORIES, \
        f"Category mismatch at seed {seed}: {gen.categories}"

    # ------------------------------------------------------------------
    # T1 + T5: Static condition
    # ------------------------------------------------------------------
    scorer_static = ProfileScorer(gt_profiles.copy(), tau=TAU)
    auto_correct: list[int]   = []
    t1_correct:   list[int]   = []

    for a in alerts:
        result     = scorer_static.score(a.factors, a.category_index)
        is_correct = int(result.action_index == a.gt_action_index)
        t1_correct.append(is_correct)
        conf = float(result.probabilities[result.action_index])
        if conf >= AUTO_APPROVE_CONF:
            auto_correct.append(is_correct)

    t1_acc    = float(np.mean(t1_correct))
    t5_acc    = float(np.mean(auto_correct)) if auto_correct else float("nan")
    t5_pct    = len(auto_correct) / N_ALERTS

    # ------------------------------------------------------------------
    # T2, T3, T4: Learning condition
    # ------------------------------------------------------------------
    scorer_learn = ProfileScorer(gt_profiles.copy(), tau=TAU)
    recent: list[int]     = []
    cred_decisions: list[int] = []

    t2_acc = float("nan")
    t3_acc = float("nan")
    t4_acc = float("nan")

    for i, a in enumerate(alerts):
        result     = scorer_learn.score(a.factors, a.category_index)
        is_correct = int(result.action_index == a.gt_action_index)

        recent.append(is_correct)
        if len(recent) > ROLLING_WIN:
            recent.pop(0)

        if a.category_index == CRED_ACCESS_IDX:
            cred_decisions.append(is_correct)

        # Oracle update
        scorer_learn.update(
            a.factors,
            a.category_index,
            a.gt_action_index,
            correct=(result.action_index == a.gt_action_index),
        )

        decision = i + 1
        if decision == 1000:
            t2_acc = float(np.mean(recent))
            t4_acc = float(np.mean(cred_decisions)) if cred_decisions else float("nan")
        if decision == 1500:
            t3_acc = float(np.mean(recent))

    return {
        "t1": t1_acc,
        "t2": t2_acc,
        "t3": t3_acc,
        "t4": t4_acc,
        "t5": t5_acc,
        "t5_pct": t5_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("\n=== STATISTICAL PASS: 50-SEED PUBLISHED CLAIMS VALIDATION ===")
    print(f"Standing rule: >=50 seeds + 95% CI for any published number.")
    print(f"Mode: {MODE} realistic. tau={TAU}. N={N_ALERTS}/seed.")
    print(f"Running {N_SEEDS} seeds...\n")

    # Sanity check taxonomy
    assert SOC_CATEGORIES == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ]
    assert SOC_ACTIONS == ["escalate", "investigate", "suppress", "monitor"]

    # -----------------------------------------------------------------------
    # Collect per-seed values
    # -----------------------------------------------------------------------
    t1_vals: list[float] = []
    t2_vals: list[float] = []
    t3_vals: list[float] = []
    t4_vals: list[float] = []
    t5_vals: list[float] = []
    t5_pcts: list[float] = []

    for seed in range(N_SEEDS):
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  seed {seed+1}/{N_SEEDS} ...", flush=True)
        res = run_seed(seed)
        t1_vals.append(res["t1"])
        t2_vals.append(res["t2"])
        t3_vals.append(res["t3"])
        t4_vals.append(res["t4"])
        if not np.isnan(res["t5"]):
            t5_vals.append(res["t5"])
        t5_pcts.append(res["t5_pct"])

    # -----------------------------------------------------------------------
    # Compute statistics for each target
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TARGET STATISTICS")
    print("=" * 70)

    s_t1 = compute_stats(t1_vals, gate=0.80, rng_seed=1)
    s_t2 = compute_stats(t2_vals, gate=0.82, rng_seed=2)
    s_t3 = compute_stats(t3_vals, gate=0.82, rng_seed=3)
    s_t4 = compute_stats(t4_vals, gate=None, rng_seed=4)
    s_t5 = compute_stats(t5_vals, gate=0.90, rng_seed=5)

    print_target("T1: Static accuracy (no learning)", s_t1, gate=0.80, ref_10=REF_T1)
    print_target("T2: Learning accuracy at dec 1000 (rolling-200)", s_t2, gate=0.82, ref_10=REF_T2)
    print_target("T3: Learning accuracy at dec 1500 (rolling-200)", s_t3, gate=0.82, ref_10=REF_T3)
    print_target("T4: credential_access accuracy at dec 1000 (cumulative)", s_t4, gate=None, ref_10=REF_T4)
    print_target("T5: Auto-approve band accuracy (conf >=0.90, static)", s_t5, gate=0.90, ref_10=REF_T5)

    # Auto-approve coverage
    mean_t5_pct = float(np.mean(t5_pcts))
    std_t5_pct  = float(np.std(t5_pcts, ddof=1))
    print(f"\n  Auto-approve band coverage: {mean_t5_pct*100:.2f}% ±{std_t5_pct*100:.2f}% of decisions")

    # -----------------------------------------------------------------------
    # DI-06: dec 1500 vs dec 1000 comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DI-06: dec 1500 vs dec 1000 — is the improvement real?")
    print("=" * 70)

    deltas = [t3_vals[i] - t2_vals[i] for i in range(N_SEEDS)]
    mean_d = float(np.mean(deltas))
    std_d  = float(np.std(deltas, ddof=1))
    se_d   = std_d / np.sqrt(N_SEEDS)
    ci_lo  = mean_d - 1.96 * se_d
    ci_hi  = mean_d + 1.96 * se_d
    t_d, p_d = stats.ttest_1samp(deltas, 0.0)

    print(f"\n  Mean delta (dec1500 - dec1000):  {mean_d*100:+.3f}pp")
    print(f"  Std:                             {std_d*100:.3f}pp")
    print(f"  95% CI:                          [{ci_lo*100:.3f}pp, {ci_hi*100:.3f}pp]")
    print(f"  t-test vs 0:                     t={float(t_d):.3f}, p={float(p_d):.4f}")
    if ci_lo > 0:
        print("  ✓ CI entirely positive — dec 1500 genuinely better than dec 1000")
        print("  DI-06: RESOLVED — reversal in 10-seed run was noise")
    elif ci_hi < 0:
        print("  ✗ CI entirely negative — dec 1500 genuinely worse (oscillation confirmed)")
        print("  DI-06: CONFIRMED — centroid oscillation is real, requires design fix")
    else:
        print("  ⚠ CI crosses zero — statistically ambiguous at 50 seeds")
        print(f"  DI-06: UNRESOLVED — need >{int(np.ceil((1.96*std_d/0.005)**2))} seeds"
              f" to detect ±0.5pp difference at 95%")

    # -----------------------------------------------------------------------
    # Publishable numbers summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PUBLISHABLE NUMBERS (use these, not 10-seed results)")
    print("=" * 70)
    rows = [
        ("Static realistic acc",          s_t1),
        ("Learning dec 1000",             s_t2),
        ("Learning dec 1500",             s_t3),
        ("credential_access dec 1000",    s_t4),
        ("Auto-approve accuracy (≥0.90)", s_t5),
    ]
    for name, s in rows:
        print(f"  {name:<34}: {s['mean']*100:.1f}% "
              f"[{s['boot_lo']*100:.1f}%, {s['boot_hi']*100:.1f}%] "
              f"(95% CI, n={s['n']})")

    # -----------------------------------------------------------------------
    # Optional: forest plot
    # -----------------------------------------------------------------------
    try:
        _generate_forest_plot(rows, PAPER_DIR)
    except Exception as exc:
        import traceback
        print(f"\nWarning: forest plot failed: {exc}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    per_seed = {
        "t1": [round(v, 6) for v in t1_vals],
        "t2": [round(v, 6) for v in t2_vals],
        "t3": [round(v, 6) for v in t3_vals],
        "t4": [round(v, 6) for v in t4_vals],
        "t5": [round(v, 6) for v in t5_vals],
        "t5_pct": [round(v, 6) for v in t5_pcts],
        "delta_t3_t2": [round(v, 6) for v in deltas],
    }

    def _s(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.floating, float)) and v is not None:
                out[k] = round(float(v), 6)
            elif isinstance(v, (np.bool_, bool)):
                out[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    output = {
        "n_seeds":    N_SEEDS,
        "n_alerts":   N_ALERTS,
        "tau":        TAU,
        "mode":       MODE,
        "targets": {
            "T1_static_acc":             _s(s_t1),
            "T2_learning_dec1000":       _s(s_t2),
            "T3_learning_dec1500":       _s(s_t3),
            "T4_cred_access_dec1000":    _s(s_t4),
            "T5_auto_approve_acc":       _s(s_t5),
        },
        "t5_coverage_mean": round(mean_t5_pct, 6),
        "t5_coverage_std":  round(std_t5_pct,  6),
        "di06": {
            "mean_delta":  round(mean_d, 6),
            "std_delta":   round(std_d,  6),
            "ci95_lo":     round(ci_lo,  6),
            "ci95_hi":     round(ci_hi,  6),
            "t_stat":      round(float(t_d), 6),
            "p_value":     round(float(p_d), 6),
            "resolved":    ci_lo > 0 or ci_hi < 0,
        },
        "per_seed": per_seed,
        "references_10seed": {
            "T1": REF_T1, "T2": REF_T2, "T3": REF_T3,
            "T4": REF_T4, "T5": REF_T5,
        },
    }
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Always exit 0 — this is a measurement pass, not a gate experiment
    return True


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def _generate_forest_plot(
    rows:      list[tuple[str, dict]],
    paper_dir: Path,
) -> None:
    """
    Simple forest plot: 5 rows showing mean ± 95% CI (normal) with gate lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz.bridge_common import VIZ_DEFAULTS, save_figure

    gates = {
        "Static realistic acc":          0.80,
        "Learning dec 1000":             0.82,
        "Learning dec 1500":             0.82,
        "credential_access dec 1000":    None,
        "Auto-approve accuracy (≥0.90)": 0.90,
    }
    gate_colors = {
        "Static realistic acc":          "#DC2626",
        "Learning dec 1000":             "#D97706",
        "Learning dec 1500":             "#D97706",
        "Auto-approve accuracy (≥0.90)": "#7C3AED",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_positions = list(range(len(rows)))
    labels      = [name for name, _ in rows]

    for yi, (name, s) in zip(y_positions, rows):
        mean = s["mean"] * 100
        lo   = s["ci95_lo"] * 100
        hi   = s["ci95_hi"] * 100
        boot_lo = s["boot_lo"] * 100
        boot_hi = s["boot_hi"] * 100

        # Bootstrap CI (thick inner bar)
        ax.plot([boot_lo, boot_hi], [yi, yi],
                color="#1E40AF", linewidth=4, alpha=0.35, solid_capstyle="butt", zorder=3)
        # Normal CI (thin outer bar)
        ax.plot([lo, hi], [yi, yi],
                color="#1E40AF", linewidth=1.5, alpha=0.7, solid_capstyle="butt", zorder=3)
        # Mean dot
        ax.scatter([mean], [yi], color="#1E40AF", s=60, zorder=5)
        # Value label
        ax.text(hi + 0.15, yi,
                f"{mean:.1f}% [{lo:.1f}%, {hi:.1f}%]",
                va="center", ha="left", fontsize=7.5, color="#111827")

        # Gate line for this row
        gate = gates.get(name)
        if gate is not None:
            gate_pct = gate * 100
            color    = gate_colors.get(name, "#DC2626")
            ax.plot([gate_pct, gate_pct], [yi - 0.35, yi + 0.35],
                    color=color, linewidth=2.0, linestyle="--", alpha=0.8, zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Statistical Pass: 50-Seed Published Claims Validation\n"
        "Thick bar = 95% CI (bootstrap). Thin = 95% CI (normal). "
        "Dashed = gate threshold.",
        fontsize=VIZ_DEFAULTS["title_fontsize"],
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # Set x-range to show all CIs with annotation room
    all_means = [s["mean"] * 100 for _, s in rows]
    all_hi    = [s["ci95_hi"] * 100 for _, s in rows]
    x_lo = max(0, min(all_means) - 5)
    x_hi = min(100, max(all_hi) + 12)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.7, len(rows) - 0.3)

    fig.tight_layout()
    save_figure(fig, "statistical_pass_forest", str(paper_dir))
    print("  Forest plot saved.")


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
