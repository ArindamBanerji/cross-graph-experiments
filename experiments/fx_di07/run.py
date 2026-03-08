"""
FX-DI07: shadow_discount Parameter Validation
experiments/fx_di07/run.py

Closes DI-07 by finding the experimentally correct shadow_discount value
rather than assuming 0.3.

CONTEXT:
FX-2 showed persistent analyst bias produces centroid drift 0.71–0.77 with
no accuracy recovery. The proposed fix: shadow_discount on update() reduces
learning rate for unconfirmed shadow-period submissions. 0.3 was assumed.

Q1: Does discounting actually limit drift?
Q2: What discount produces drift < 0.10 under both bias patterns?
Q3: Does low discount impair post-incident recovery?
Q4: Is 0.3 correct, too aggressive, or not aggressive enough?

Design: LOCAL ProfileScorer subclass with shadow_discount parameter.
Do NOT modify src/models/profile_scorer.py.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.profile_scorer import ProfileScorer
from experiments.fx1_proxy_real.realistic_generator import (
    RealisticAlertGenerator,
    SOC_CATEGORIES,
    SOC_ACTIONS,
)
from experiments.fx2_noise_distributions.bias_generator import (
    BiasPattern,
    BiasedFeedbackSimulator,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DISCOUNT_VALUES  = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SEEDS          = 30
N_ALERTS         = 1500
TAU              = 0.1
MODE             = "combined"
ROLLING_WIN      = 200
INCIDENT_AT      = 200
BIAS_PATTERNS    = [BiasPattern.ALERT_FATIGUE, BiasPattern.EXPERTISE_GRADIENT]

DRIFT_GATE       = 0.10    # drift must be below this
ACC_GATE         = {
    BiasPattern.ALERT_FATIGUE:            0.65,
    BiasPattern.EXPERTISE_GRADIENT:       0.70,
}
RECOVERY_IMPAIR_PP = 0.03   # >3pp slower = recovery impaired

EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
PAPER_DIR    = REPO_ROOT / "paper_figures"

# FX-2 reference: drift at discount=1.0 (no protection)
REF_DRIFT_FATIGUE    = 0.7120
REF_DRIFT_EXPERTISE  = 0.7729


# ---------------------------------------------------------------------------
# Local ProfileScorer with shadow_discount  (does NOT modify src/)
# ---------------------------------------------------------------------------

class DiscountedProfileScorer(ProfileScorer):
    """
    ProfileScorer subclass that applies shadow_discount to update().

    source="shadow"          → effective_eta *= shadow_discount
    source="human_confirmed" → effective_eta unchanged (discount ignored)

    All other behaviour (scoring, get_profile_snapshot, etc.) is inherited.
    """

    def update(
        self,
        factors:         np.ndarray,
        category_index:  int,
        action_idx:      int,
        correct:         bool,
        source:          str   = "human_confirmed",
        shadow_discount: float = 1.0,
    ) -> None:
        """
        Update with optional shadow_discount multiplier.

        shadow_discount: multiplier on effective learning rate.
          1.0 = full weight (human_confirmed or oracle)
          0.0 = freeze (no learning from this submission)
          0.3 = proposed production default for shadow submissions
        """
        discount = shadow_discount if source == "shadow" else 1.0

        f     = factors.flatten()
        c, a  = category_index, action_idx
        count = int(self.counts[c, a])

        if correct:
            effective_eta = self.eta     / (1.0 + count * 0.001) * discount
            self.mu[c, a] += effective_eta * (f - self.mu[c, a])
        else:
            effective_eta = self.eta_neg / (1.0 + count * 0.001) * discount
            self.mu[c, a] -= effective_eta * (f - self.mu[c, a])

        np.clip(self.mu[c, a], 0.0, 1.0, out=self.mu[c, a])
        self.counts[c, a] += 1


# ---------------------------------------------------------------------------
# Persistent bias runner  (Q1, Q2, Q4)
# ---------------------------------------------------------------------------

def run_persistent_bias(
    seed:     int,
    pattern:  BiasPattern,
    discount: float,
) -> dict:
    """
    One seed of persistent bias (alert_fatigue or expertise_gradient).

    Biased submissions → source="shadow", discounted.
    Clean submissions  → source="human_confirmed", full weight.

    Returns: drift (Frobenius), final_acc (rolling-200 last 200 decisions).
    """
    gen         = RealisticAlertGenerator(mode=MODE, seed=seed)
    alerts      = gen.generate(N_ALERTS)
    gt_profiles = gen.get_profiles()

    scorer    = DiscountedProfileScorer(gt_profiles.copy(), tau=TAU)
    simulator = BiasedFeedbackSimulator(scorer, pattern, seed=seed + 1000)

    initial_mu = scorer.get_profile_snapshot()
    recent:  list[int] = []

    for a in alerts:
        result = scorer.score(a.factors, a.category_index)

        submitted = simulator.get_analyst_action(
            gt_action_index=a.gt_action_index,
            category_idx=a.category_index,
            n_actions=len(SOC_ACTIONS),
        )

        model_correct = int(result.action_index == a.gt_action_index)
        recent.append(model_correct)
        if len(recent) > ROLLING_WIN:
            recent.pop(0)

        is_biased = (submitted != a.gt_action_index)

        if is_biased:
            # Biased submission: apply shadow discount, update from wrong action
            scorer.update(
                a.factors,
                a.category_index,
                submitted,          # the wrong submitted action
                correct=False,
                source="shadow",
                shadow_discount=discount,
            )
        else:
            # Clean submission: full weight, update from GT
            scorer.update(
                a.factors,
                a.category_index,
                a.gt_action_index,
                correct=(result.action_index == a.gt_action_index),
                source="human_confirmed",
                shadow_discount=discount,
            )

    final_mu   = scorer.get_profile_snapshot()
    drift      = float(np.linalg.norm((final_mu - initial_mu).flatten()))
    final_acc  = float(np.mean(recent[-ROLLING_WIN:]))

    return {"drift": drift, "final_acc": final_acc}


# ---------------------------------------------------------------------------
# Post-incident recovery runner  (Q3)
# ---------------------------------------------------------------------------

def run_recovery(
    seed:     int,
    discount: float,
) -> dict:
    """
    Post-incident burst bias: ALL updates in dec 1–250 are shadow (discounted).
    From dec 251 onwards: human_confirmed (full weight).

    Returns: acc@dec500 and acc@dec1000 (rolling-200).
    """
    gen         = RealisticAlertGenerator(mode=MODE, seed=seed)
    alerts      = gen.generate(N_ALERTS)
    gt_profiles = gen.get_profiles()

    scorer    = DiscountedProfileScorer(gt_profiles.copy(), tau=TAU)
    simulator = BiasedFeedbackSimulator(
        scorer, BiasPattern.POST_INCIDENT_ESCALATION, seed=seed + 2000
    )
    simulator.simulate_incident(at_decision=INCIDENT_AT)

    recent: list[int] = []
    acc_500  = float("nan")
    acc_1000 = float("nan")

    for i, a in enumerate(alerts):
        result    = scorer.score(a.factors, a.category_index)
        correct   = int(result.action_index == a.gt_action_index)

        submitted = simulator.get_analyst_action(
            gt_action_index=a.gt_action_index,
            category_idx=a.category_index,
            n_actions=len(SOC_ACTIONS),
        )

        recent.append(correct)
        if len(recent) > ROLLING_WIN:
            recent.pop(0)

        decision = i + 1
        if decision <= 251:
            # Shadow period: all submissions discounted (production can't distinguish)
            scorer.update(
                a.factors,
                a.category_index,
                submitted,
                correct=(submitted == a.gt_action_index),
                source="shadow",
                shadow_discount=discount,
            )
        else:
            # Recovery period: human confirmed, full weight
            scorer.update(
                a.factors,
                a.category_index,
                a.gt_action_index,
                correct=correct,
                source="human_confirmed",
                shadow_discount=discount,
            )

        if decision == 500:
            acc_500  = float(np.mean(recent))
        if decision == 1000:
            acc_1000 = float(np.mean(recent))

    return {"acc_500": acc_500, "acc_1000": acc_1000}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== FX-DI07: shadow_discount Parameter Validation ===")
    print(f"Discounts: {DISCOUNT_VALUES}")
    print(f"Patterns: {[p.value for p in BIAS_PATTERNS]}")
    print(f"N_SEEDS={N_SEEDS}, N_ALERTS={N_ALERTS}, tau={TAU}")
    print(f"Drift gate: <{DRIFT_GATE}. Acc gates: fatigue≥65%, expertise≥70%\n")

    assert SOC_CATEGORIES == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ]

    # -----------------------------------------------------------------------
    # Persistent bias sweep  (Q1, Q2, Q4)
    # -----------------------------------------------------------------------
    # results[discount][pattern_value] = {drift: [float×30], final_acc: [float×30]}
    results: dict = {
        d: {p.value: {"drift": [], "final_acc": []} for p in BIAS_PATTERNS}
        for d in DISCOUNT_VALUES
    }

    total_runs = len(DISCOUNT_VALUES) * len(BIAS_PATTERNS) * N_SEEDS
    run_idx    = 0

    for discount in DISCOUNT_VALUES:
        for pattern in BIAS_PATTERNS:
            pname = pattern.value
            for seed in range(N_SEEDS):
                run_idx += 1
                if run_idx % (N_SEEDS * len(BIAS_PATTERNS)) == 1:
                    print(f"  discount={discount} [{run_idx}/{total_runs}] ...",
                          flush=True)
                res = run_persistent_bias(seed, pattern, discount)
                results[discount][pname]["drift"].append(res["drift"])
                results[discount][pname]["final_acc"].append(res["final_acc"])

    # -----------------------------------------------------------------------
    # Recovery sweep  (Q3)
    # -----------------------------------------------------------------------
    print(f"\n  Recovery sweep (post_incident, {N_SEEDS} seeds × {len(DISCOUNT_VALUES)} discounts)...")
    recovery: dict = {d: {"acc_500": [], "acc_1000": []} for d in DISCOUNT_VALUES}

    for discount in DISCOUNT_VALUES:
        for seed in range(N_SEEDS):
            res = run_recovery(seed, discount)
            recovery[discount]["acc_500"].append(res["acc_500"])
            recovery[discount]["acc_1000"].append(res["acc_1000"])

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    agg: dict = {}   # agg[discount][pattern] = {mean_drift, std_drift, mean_acc, std_acc}
    for discount in DISCOUNT_VALUES:
        agg[discount] = {}
        for pattern in BIAS_PATTERNS:
            pname = pattern.value
            d_arr  = np.array(results[discount][pname]["drift"])
            a_arr  = np.array(results[discount][pname]["final_acc"])
            agg[discount][pname] = {
                "mean_drift": float(d_arr.mean()),
                "std_drift":  float(d_arr.std(ddof=1)),
                "mean_acc":   float(a_arr.mean()),
                "std_acc":    float(a_arr.std(ddof=1)),
            }

    rec_agg: dict = {}  # rec_agg[discount] = {mean_500, mean_1000}
    baseline_500 = float(np.mean(recovery[1.0]["acc_500"]))
    for discount in DISCOUNT_VALUES:
        rec_agg[discount] = {
            "mean_500":  float(np.mean(recovery[discount]["acc_500"])),
            "std_500":   float(np.std(recovery[discount]["acc_500"], ddof=1)),
            "mean_1000": float(np.mean(recovery[discount]["acc_1000"])),
            "std_1000":  float(np.std(recovery[discount]["acc_1000"], ddof=1)),
        }

    # -----------------------------------------------------------------------
    # Print persistent bias results table
    # -----------------------------------------------------------------------
    print(f"\n--- Persistent Bias Results ---")
    print(f"FX-2 reference drift (discount=1.0): "
          f"fatigue={REF_DRIFT_FATIGUE:.4f}, expertise={REF_DRIFT_EXPERTISE:.4f}")
    print()

    w = 12
    print(f"{'Discount':>{w}} | "
          f"{'alert_fatigue':^42} | "
          f"{'expertise_gradient':^42}")
    print(f"{'':>{w}} | "
          f"{'drift':>10} {'acc':>10} {'drift<0.10':>10} {'pass':>8} | "
          f"{'drift':>10} {'acc':>10} {'drift<0.10':>10} {'pass':>8}")
    print("─" * (w + 2 + 44 + 2 + 44))

    for discount in DISCOUNT_VALUES:
        row_parts = [f"{discount:>{w}.1f}"]
        for pattern in BIAS_PATTERNS:
            pname     = pattern.value
            a         = agg[discount][pname]
            drift_ok  = a["mean_drift"] < DRIFT_GATE
            acc_gate  = ACC_GATE[pattern]
            acc_ok    = a["mean_acc"] >= acc_gate
            both_ok   = drift_ok and acc_ok
            row_parts.append(
                f" | {a['mean_drift']:>10.4f} {a['mean_acc']*100:>9.2f}%"
                f" {'✓' if drift_ok else '✗':>10}"
                f" {'✓' if both_ok else '✗':>8}"
            )
        print("".join(row_parts))

    # -----------------------------------------------------------------------
    # Print recovery table (Q3)
    # -----------------------------------------------------------------------
    print(f"\n--- Post-Incident Recovery (Q3: does low discount slow recovery?) ---")
    print(f"  Baseline acc@500 (discount=1.0, no protection): {baseline_500*100:.2f}%")
    print(f"  Recovery-impaired threshold: baseline − {RECOVERY_IMPAIR_PP*100:.0f}pp"
          f" = {(baseline_500 - RECOVERY_IMPAIR_PP)*100:.2f}%")
    print()
    print(f"  {'Discount':>10} | {'Acc@dec500':>12} | {'Acc@dec1000':>13} | {'Recovery impaired?':>20}")
    print("  " + "─" * 64)
    for discount in DISCOUNT_VALUES:
        r = rec_agg[discount]
        impaired = r["mean_500"] < baseline_500 - RECOVERY_IMPAIR_PP
        flag     = "YES ⚠" if impaired else "no"
        print(f"  {discount:>10.1f} | {r['mean_500']*100:>11.2f}% | "
              f"{r['mean_1000']*100:>12.2f}% | {flag:>20}")

    # -----------------------------------------------------------------------
    # Find recommended discount value
    # -----------------------------------------------------------------------
    print(f"\n=== VALIDATED shadow_discount VALUE ===")
    print(f"Selection criteria:")
    print(f"  1. drift < {DRIFT_GATE} for both bias patterns")
    print(f"  2. final_acc >= gate for both patterns (fatigue≥65%, expertise≥70%)")
    print(f"  3. recovery NOT impaired (acc@500 within {RECOVERY_IMPAIR_PP*100:.0f}pp of 1.0 baseline)")
    print()

    candidates = []
    for discount in DISCOUNT_VALUES:
        drift_ok_both = all(
            agg[discount][p.value]["mean_drift"] < DRIFT_GATE
            for p in BIAS_PATTERNS
        )
        acc_ok_both = all(
            agg[discount][p.value]["mean_acc"] >= ACC_GATE[p]
            for p in BIAS_PATTERNS
        )
        rec_ok = rec_agg[discount]["mean_500"] >= baseline_500 - RECOVERY_IMPAIR_PP
        if drift_ok_both and acc_ok_both and rec_ok:
            candidates.append(discount)

    if candidates:
        recommended = max(candidates)   # highest discount = most learning preserved
        print(f"  Candidates (all criteria met): {candidates}")
        print(f"  Recommended (max learning): {recommended}")
        print()
        if recommended == 0.3:
            print("  ✓ 0.3 confirmed — use as CalibrationProfile default")
        elif recommended > 0.3:
            print(f"  ⚠ 0.3 was too aggressive — {recommended} preserves more learning")
            print(f"    At discount=0.3 the drift gate is met but learning is unnecessarily slow.")
            print(f"    Raise CalibrationProfile.shadow_discount to {recommended}.")
        else:
            print(f"  ⚠ 0.3 was not aggressive enough — use {recommended}")
            print(f"    At discount=0.3 one or more criteria fail. "
                  f"More aggressive discounting required.")
            print(f"    Lower CalibrationProfile.shadow_discount to {recommended}.")
    else:
        # Check what fails at each discount
        print("  No discount value meets all three criteria simultaneously.")
        print("  Diagnosing failure modes:")
        for discount in DISCOUNT_VALUES:
            drift_ok = all(
                agg[discount][p.value]["mean_drift"] < DRIFT_GATE for p in BIAS_PATTERNS
            )
            acc_ok = all(
                agg[discount][p.value]["mean_acc"] >= ACC_GATE[p] for p in BIAS_PATTERNS
            )
            rec_ok = rec_agg[discount]["mean_500"] >= baseline_500 - RECOVERY_IMPAIR_PP
            issues = []
            if not drift_ok:
                issues.append(f"drift>0.10")
            if not acc_ok:
                issues.append(f"acc<gate")
            if not rec_ok:
                issues.append(f"recovery impaired")
            print(f"    discount={discount}: {', '.join(issues) if issues else 'all pass'}")
        print()
        print("  Recommendation: discount=0.0 (freeze) + SHADOW_MIN_DAYS extended to 21+")
        recommended = 0.0

    # -----------------------------------------------------------------------
    # Generate chart
    # -----------------------------------------------------------------------
    try:
        _generate_chart(agg, rec_agg, DISCOUNT_VALUES, BIAS_PATTERNS,
                        baseline_500, candidates, recommended, str(PAPER_DIR))
    except Exception as exc:
        import traceback
        print(f"\nWarning: chart failed: {exc}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.bool_):    return bool(o)
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            return super().default(o)

    output = {
        "experiment":         "FX-DI07",
        "n_seeds":            N_SEEDS,
        "n_alerts":           N_ALERTS,
        "tau":                TAU,
        "mode":               MODE,
        "discount_values":    DISCOUNT_VALUES,
        "drift_gate":         DRIFT_GATE,
        "acc_gates":          {p.value: ACC_GATE[p] for p in BIAS_PATTERNS},
        "recommended_discount": recommended,
        "candidates":         candidates,
        "agg": {
            str(d): {
                p.value: {
                    k: round(v, 6) for k, v in agg[d][p.value].items()
                }
                for p in BIAS_PATTERNS
            }
            for d in DISCOUNT_VALUES
        },
        "recovery_agg": {
            str(d): {k: round(v, 6) for k, v in rec_agg[d].items()}
            for d in DISCOUNT_VALUES
        },
        "baseline_recovery_500": round(baseline_500, 6),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, cls=_Enc)
    print(f"\nResults saved to {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _generate_chart(
    agg:            dict,
    rec_agg:        dict,
    discounts:      list,
    patterns:       list,
    baseline_500:   float,
    candidates:     list,
    recommended:    float,
    paper_dir:      str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz.bridge_common import VIZ_DEFAULTS, save_figure

    PAT_COLORS  = {
        BiasPattern.ALERT_FATIGUE.value:        "#D97706",
        BiasPattern.EXPERTISE_GRADIENT.value:   "#7C3AED",
    }
    PAT_LABELS  = {
        BiasPattern.ALERT_FATIGUE.value:        "Alert Fatigue",
        BiasPattern.EXPERTISE_GRADIENT.value:   "Expertise Gradient",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    x = np.array(discounts)

    # ── Left: Centroid drift vs discount ────────────────────────────────
    ax = axes[0]
    for pattern in patterns:
        pname = pattern.value
        means = [agg[d][pname]["mean_drift"] for d in discounts]
        stds  = [agg[d][pname]["std_drift"]  for d in discounts]
        m_arr = np.array(means)
        s_arr = np.array(stds)
        ax.plot(x, m_arr, "o-", color=PAT_COLORS[pname], linewidth=2.2,
                markersize=6, label=PAT_LABELS[pname], zorder=4)
        ax.fill_between(x, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.15, color=PAT_COLORS[pname], zorder=3)

    ax.axhline(DRIFT_GATE, color="#DC2626", linewidth=1.5, linestyle="--",
               alpha=0.85, zorder=2, label=f"Drift gate: {DRIFT_GATE}")
    # Mark recommended
    if recommended in discounts:
        ax.axvline(recommended, color="#059669", linewidth=1.5,
                   linestyle=":", alpha=0.8, label=f"Recommended: {recommended}")

    ax.set_xlabel("shadow_discount", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Centroid Drift (Frobenius)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("Q1/Q2: Drift vs Discount\n(gate: <0.10)",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in discounts],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])

    # ── Center: Final accuracy vs discount ──────────────────────────────
    ax = axes[1]
    for pattern in patterns:
        pname     = pattern.value
        means     = [agg[d][pname]["mean_acc"] * 100 for d in discounts]
        stds      = [agg[d][pname]["std_acc"]  * 100 for d in discounts]
        m_arr     = np.array(means)
        s_arr     = np.array(stds)
        gate_pct  = ACC_GATE[pattern] * 100
        ax.plot(x, m_arr, "o-", color=PAT_COLORS[pname], linewidth=2.2,
                markersize=6, label=PAT_LABELS[pname], zorder=4)
        ax.fill_between(x, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.15, color=PAT_COLORS[pname], zorder=3)
        ax.axhline(gate_pct, color=PAT_COLORS[pname], linewidth=1.0,
                   linestyle=":", alpha=0.6, zorder=2,
                   label=f"{PAT_LABELS[pname]} gate: {gate_pct:.0f}%")

    if recommended in discounts:
        ax.axvline(recommended, color="#059669", linewidth=1.5,
                   linestyle=":", alpha=0.8, label=f"Recommended: {recommended}")

    ax.set_xlabel("shadow_discount", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Final Accuracy (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("Q4: Accuracy vs Discount\n(gates: fatigue≥65%, expertise≥70%)",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in discounts],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])

    # ── Right: Recovery acc@500 vs discount (Q3) ─────────────────────
    ax = axes[2]
    means_500 = [rec_agg[d]["mean_500"] * 100 for d in discounts]
    stds_500  = [rec_agg[d]["std_500"]  * 100 for d in discounts]
    m_arr     = np.array(means_500)
    s_arr     = np.array(stds_500)

    ax.plot(x, m_arr, "o-", color="#2563EB", linewidth=2.2,
            markersize=6, label="Acc @ dec 500", zorder=4)
    ax.fill_between(x, m_arr - s_arr, m_arr + s_arr,
                    alpha=0.15, color="#2563EB", zorder=3)

    impair_line = (baseline_500 - RECOVERY_IMPAIR_PP) * 100
    ax.axhline(baseline_500 * 100, color="#94A3B8", linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=2,
               label=f"Baseline (discount=1.0): {baseline_500*100:.1f}%")
    ax.axhline(impair_line, color="#DC2626", linewidth=1.2,
               linestyle=":", alpha=0.75, zorder=2,
               label=f"Impaired threshold: {impair_line:.1f}%")

    if recommended in discounts:
        ax.axvline(recommended, color="#059669", linewidth=1.5,
                   linestyle=":", alpha=0.8, label=f"Recommended: {recommended}")

    ax.set_xlabel("shadow_discount", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Rolling Acc at Dec 500 (%)", fontsize=VIZ_DEFAULTS["label_fontsize"])
    ax.set_title("Q3: Post-Incident Recovery\n(does low discount slow learning?)",
                 fontsize=VIZ_DEFAULTS["title_fontsize"])
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in discounts],
                       fontsize=VIZ_DEFAULTS["tick_fontsize"])

    fig.suptitle(
        f"FX-DI07: shadow_discount Validation — "
        f"Recommended value: {recommended}  "
        f"({'Confirmed' if recommended == 0.3 else 'Revised from 0.3'})\n"
        f"N={N_SEEDS} seeds, {N_ALERTS} alerts/seed, combined realistic, τ={TAU}",
        fontsize=VIZ_DEFAULTS["title_fontsize"] + 0.5,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "fx_di07_discount_sweep", paper_dir)
    print("  Chart saved.")


if __name__ == "__main__":
    main()
    sys.exit(0)
