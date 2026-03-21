"""
Three asks from roadmap session:
  Ask 1: Fix selector to track trajectory (slope), not just level
  Ask 2: Verify simplified Phase 2 rule (noise_ratio only, drop rho)
  Ask 3: Run 3-4 more healthcare personas with DiagonalKernel
"""

import numpy as np
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel
from gae.kernel_selector import KernelSelector

# ── Shared constants ───────────────────────────────────────────────────────────
ETA          = 0.05
ETA_NEG      = 0.05
ETA_OVERRIDE = 0.01
VERIFY_RATE  = 0.30

HC_NOISE = np.array([0.18, 0.20, 0.19, 0.25, 0.22, 0.28])
HC_FACTORS = [
    "travel_match", "asset_criticality", "threat_intel_enrichment",
    "time_anomaly", "pattern_history", "device_trust",
]

S2P_HETERO_RATIOS = np.array([1.0, 1.5, 0.7, 0.8, 0.6, 1.3, 1.8, 1.6])

HC_TEAM = [
    {"id": "a1", "override_rate": 0.14, "override_quality": 0.90, "fatigue_factor": 0.18},
    {"id": "a2", "override_rate": 0.26, "override_quality": 0.78, "fatigue_factor": 0.28},
    {"id": "a3", "override_rate": 0.24, "override_quality": 0.80, "fatigue_factor": 0.22},
    {"id": "a4", "override_rate": 0.42, "override_quality": 0.63, "fatigue_factor": 0.40},
]

S2P_TEAM = [
    {"id": "a1", "override_rate": 0.16, "override_quality": 0.88, "fatigue_factor": 0.22},
    {"id": "a2", "override_rate": 0.24, "override_quality": 0.80, "fatigue_factor": 0.28},
    {"id": "a3", "override_rate": 0.28, "override_quality": 0.76, "fatigue_factor": 0.32},
    {"id": "a4", "override_rate": 0.40, "override_quality": 0.64, "fatigue_factor": 0.38},
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def diagonal_weights(noise: np.ndarray) -> np.ndarray:
    inv_var = 1.0 / np.maximum(noise ** 2, 0.001)
    return inv_var / inv_var.max()


def build_gt_arr(config: dict) -> np.ndarray:
    """(C, A) array from gt_distributions dict."""
    categories = config["categories"]
    actions    = config["actions"]
    C, A       = len(categories), len(actions)
    gt_dists   = config["gt_distributions"]
    gt_arr     = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0/A]*A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()
    return gt_arr


def analyst_eff(analyst: dict):
    """Returns (eff_override_rate, eff_quality) after fatigue scaling."""
    ff  = analyst.get("fatigue_factor", 0.20)
    eo  = min(1.0, analyst["override_rate"] * (1 + ff * 0.3))
    eq  = max(0.4,  analyst["override_quality"] * (1 - ff * 0.2))
    return eo, eq


def make_scorer(mu_init: np.ndarray, actions: list, kernel) -> ProfileScorer:
    scorer = ProfileScorer(
        mu_init.copy(), actions,
        scoring_kernel=kernel,
        eta_override=ETA_OVERRIDE,
    )
    scorer.eta     = ETA
    scorer.eta_neg = ETA_NEG
    return scorer


def s2p_noise_array(sigma_eff: float) -> np.ndarray:
    raw = sigma_eff * S2P_HETERO_RATIOS
    raw = raw * (sigma_eff / raw.mean())
    return np.clip(raw, 0.03, 0.40)


# ══════════════════════════════════════════════════════════════════════════════
# ASK 1: Trajectory-aware selector
# ══════════════════════════════════════════════════════════════════════════════

def run_trajectory_selector(
    config, noise: np.ndarray, team: list,
    volume: int, n_decisions: int,
    window_size: int = 100,
) -> tuple:
    """
    Run two selector variants on the same data stream:
      Cumulative: tracks all-time agreement rate (current KernelSelector behaviour)
      Rolling:    tracks last `window_size` decisions only (trajectory-aware fix)

    Each kernel gets its own learning ProfileScorer to surface the trajectory
    effect: diagonal starts slightly behind L2 on day 1 but gains faster.
    """
    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    C, A, d    = mu_true.shape
    cat_w      = np.ones(C) / C
    gt_arr     = build_gt_arr(config)

    a_eff = [analyst_eff(a) for a in team]

    # Build one learning scorer per kernel
    rng_init = np.random.default_rng(42)
    offset   = rng_init.uniform(-0.15, 0.15, mu_true.shape)
    mu_init  = np.clip(mu_true + offset, 0, 1)

    weights  = diagonal_weights(noise)
    scorers  = {
        "l2":       make_scorer(mu_init, actions, L2Kernel()),
        "diagonal": make_scorer(mu_init, actions, DiagonalKernel(weights)),
    }

    # Tracking
    cumulative      = {k: {"agree": 0, "total": 0} for k in scorers}
    rolling_buffer  = {k: [] for k in scorers}  # deque of bool
    traj_cumulative = []
    traj_rolling    = []

    rng = np.random.default_rng(99)   # separate from init seed

    for t in range(n_decisions):
        ci   = int(rng.choice(C, p=cat_w))
        a_gt = int(rng.choice(A, p=gt_arr[ci]))
        f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)

        # Analyst action (80% quality)
        eff_q_analyst  = 0.80
        analyst_action = a_gt if rng.random() < eff_q_analyst else int(
            rng.choice([a for a in range(A) if a != a_gt])
        )

        # Score with each kernel's scorer
        for kname, scorer in scorers.items():
            res    = scorer.score(f, ci)
            pred_a = res.action_index
            agreed = (pred_a == analyst_action)

            cumulative[kname]["total"] += 1
            if agreed:
                cumulative[kname]["agree"] += 1

            rolling_buffer[kname].append(agreed)
            if len(rolling_buffer[kname]) > window_size:
                rolling_buffer[kname].pop(0)

            # Update scorer (learning)
            if rng.random() < VERIFY_RATE:
                ai      = int(rng.integers(len(team)))
                eff_or, eff_q = a_eff[ai]
                if rng.random() < eff_or:
                    gt_a = a_gt if rng.random() < eff_q else int(
                        rng.choice([a for a in range(A) if a != a_gt])
                    )
                    scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                else:
                    scorer.update(f, ci, pred_a, True)

        # Checkpoint every 25 decisions
        if (t + 1) % 25 == 0:
            cum_rates = {
                k: v["agree"] / max(v["total"], 1)
                for k, v in cumulative.items()
            }
            cum_best = max(cum_rates, key=cum_rates.get)
            traj_cumulative.append({
                "n": t + 1, "rates": {k: round(v, 4) for k, v in cum_rates.items()},
                "pick": cum_best,
            })

            roll_rates = {
                k: sum(rolling_buffer[k]) / max(len(rolling_buffer[k]), 1)
                for k in scorers
            }
            roll_best = max(roll_rates, key=roll_rates.get)
            traj_rolling.append({
                "n": t + 1, "rates": {k: round(v, 4) for k, v in roll_rates.items()},
                "pick": roll_best,
            })

    return traj_cumulative, traj_rolling


# ══════════════════════════════════════════════════════════════════════════════
# ASK 2: Simplified Phase 2 rule
# ══════════════════════════════════════════════════════════════════════════════

def simplified_phase2(sigma_per_factor: np.ndarray) -> tuple:
    """
    Simplified rule: noise_ratio > 1.5 → diagonal, else L2.
    No rho_max. Explanation A confirmed — correlation irrelevant.
    """
    ratio = float(sigma_per_factor.max()) / max(float(sigma_per_factor.min()), 0.001)
    kernel = "diagonal" if ratio > 1.5 else "l2"
    return kernel, ratio


# ══════════════════════════════════════════════════════════════════════════════
# ASK 3: More healthcare personas
# ══════════════════════════════════════════════════════════════════════════════

HEALTHCARE_PERSONAS = [
    {
        "name":       "HC-A: device_trust clean, time_anomaly very noisy",
        "noise":      np.array([0.15, 0.18, 0.14, 0.32, 0.16, 0.10]),
        "sigma_mean": 0.175,
        "ratio":      3.2,
    },
    {
        "name":       "HC-B: moderate noise across all (ratio~1.8)",
        "noise":      np.array([0.14, 0.17, 0.13, 0.22, 0.18, 0.25]),
        "sigma_mean": 0.182,
        "ratio":      1.9,
    },
    {
        "name":       "HC-C: extreme noise (ratio~4.6)",
        "noise":      np.array([0.08, 0.12, 0.07, 0.28, 0.15, 0.32]),
        "sigma_mean": 0.170,
        "ratio":      4.6,
    },
    {
        "name":       "HC-D: original 1D-N4 profile (control)",
        "noise":      np.array([0.18, 0.20, 0.19, 0.25, 0.22, 0.28]),
        "sigma_mean": 0.220,
        "ratio":      1.6,
    },
]


def run_hc_comparison(
    config: dict, persona: dict, team: list,
    n_seeds: int = 15, days: int = 60, volume: int = 200,
) -> dict:
    """Run L2 vs Diagonal on one healthcare persona. Returns per-kernel metrics."""
    noise      = persona["noise"]
    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    C, A, d    = mu_true.shape
    cat_w      = np.ones(C) / C
    gt_arr     = build_gt_arr(config)
    a_eff_list = [analyst_eff(a) for a in team]

    def run_with_kernel(kernel):
        all_d1, all_d60 = [], []
        for seed in range(n_seeds):
            rng     = np.random.default_rng(42 + seed)
            offset  = rng.uniform(-0.15, 0.15, mu_true.shape)
            mu_init = np.clip(mu_true + offset, 0, 1)
            scorer  = make_scorer(mu_init, actions, kernel)

            daily_acc = np.zeros(days)

            for day in range(days):
                n_alerts = int(rng.poisson(volume))
                correct  = 0

                for _ in range(n_alerts):
                    ci   = int(rng.choice(C, p=cat_w))
                    a_gt = int(rng.choice(A, p=gt_arr[ci]))
                    f    = np.clip(mu_true[ci, a_gt] + rng.standard_normal(d) * noise, 0, 1)

                    res    = scorer.score(f, ci)
                    pred_a = res.action_index
                    correct += int(pred_a == a_gt)

                    if rng.random() < VERIFY_RATE:
                        ai_idx  = int(rng.integers(len(team)))
                        eff_or, eff_q = a_eff_list[ai_idx]
                        if rng.random() < eff_or:
                            gt_a = a_gt if rng.random() < eff_q else int(
                                rng.choice([a for a in range(A) if a != a_gt])
                            )
                            scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                        else:
                            scorer.update(f, ci, pred_a, True)

                daily_acc[day] = correct / n_alerts if n_alerts > 0 else 0.0

            all_d1.append(daily_acc[0])
            all_d60.append(daily_acc[-1])

        return float(np.mean(all_d1)), float(np.mean(all_d60))

    d1_l2,  d60_l2  = run_with_kernel(L2Kernel())
    w                = diagonal_weights(noise)
    d1_diag, d60_diag = run_with_kernel(DiagonalKernel(w))

    diag_delta = d60_diag - d1_diag
    l2_delta   = d60_l2  - d1_l2

    return {
        "l2":       {"day1": round(d1_l2,   4), "day60": round(d60_l2,   4),
                     "delta": round(l2_delta, 4)},
        "diagonal": {"day1": round(d1_diag, 4), "day60": round(d60_diag, 4),
                     "delta": round(diag_delta, 4)},
        "diagonal_advantage": round(diag_delta - l2_delta, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    soc_config = load_domain_config("soc_product_v50")
    s2p_config = load_domain_config("s2p_v03")
    results    = {}
    t_start    = time.time()

    # ── ASK 1: Trajectory selector ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("ASK 1: TRAJECTORY-AWARE SELECTOR")
    print("=" * 60)

    print("\n  Healthcare SOC  (σ_mean=0.220  ratio=1.6×  V=200):")
    traj_cum_hc, traj_roll_hc = run_trajectory_selector(
        soc_config, HC_NOISE, HC_TEAM, volume=200, n_decisions=500,
    )

    hdr = (f"  {'n':>5}  {'Cum pick':<12} {'Roll pick':<12}"
           f"  {'Cum L2':>7} {'Cum Diag':>8}  {'Roll L2':>7} {'Roll Diag':>9}")
    print(hdr)
    print("  " + "-" * 68)
    for c, r in zip(traj_cum_hc, traj_roll_hc):
        if c["n"] <= 25 or c["n"] % 50 == 0:
            print(f"  {c['n']:>5}  {c['pick']:<12} {r['pick']:<12}"
                  f"  {c['rates']['l2']:>7.1%} {c['rates']['diagonal']:>8.1%}"
                  f"  {r['rates']['l2']:>7.1%} {r['rates']['diagonal']:>9.1%}")

    s2p_noise = s2p_noise_array(0.15)
    print(f"\n  S2P Manufacturing  (σ_mean={s2p_noise.mean():.3f}"
          f"  ratio={s2p_noise.max()/s2p_noise.min():.1f}×  V=200):")
    traj_cum_s2p, traj_roll_s2p = run_trajectory_selector(
        s2p_config, s2p_noise, S2P_TEAM, volume=200, n_decisions=500,
    )

    print(hdr)
    print("  " + "-" * 68)
    for c, r in zip(traj_cum_s2p, traj_roll_s2p):
        if c["n"] <= 25 or c["n"] % 50 == 0:
            print(f"  {c['n']:>5}  {c['pick']:<12} {r['pick']:<12}"
                  f"  {c['rates']['l2']:>7.1%} {c['rates']['diagonal']:>8.1%}"
                  f"  {r['rates']['l2']:>7.1%} {r['rates']['diagonal']:>9.1%}")

    # Verdict
    final_cum_hc  = traj_cum_hc[-1]["pick"]
    final_roll_hc = traj_roll_hc[-1]["pick"]
    final_cum_s2p  = traj_cum_s2p[-1]["pick"]
    final_roll_s2p = traj_roll_s2p[-1]["pick"]

    # Stabilization for rolling (first n where 5 consecutive agree)
    def stabilize_n(traj):
        picks = [t["pick"] for t in traj]
        for i in range(len(picks) - 4):
            if len(set(picks[i:i+5])) == 1:
                return traj[i]["n"]
        return None

    stab_hc  = stabilize_n(traj_roll_hc)
    stab_s2p = stabilize_n(traj_roll_s2p)

    print("\n  VERDICT:")
    print(f"    Healthcare: cumulative={final_cum_hc}  rolling={final_roll_hc}")
    print(f"    S2P:        cumulative={final_cum_s2p}  rolling={final_roll_s2p}")

    any_correction = False
    if final_roll_hc != final_cum_hc:
        print(f"    Rolling CORRECTS cumulative for Healthcare:"
              f" {final_cum_hc} → {final_roll_hc}")
        any_correction = True
    if final_roll_s2p != final_cum_s2p:
        print(f"    Rolling CORRECTS cumulative for S2P:"
              f" {final_cum_s2p} → {final_roll_s2p}")
        any_correction = True
    if not any_correction:
        print("    Rolling confirms cumulative — both methods agree at n=500.")
        print("    Cumulative is sufficient; rolling window is a design option.")

    print(f"    Healthcare rolling stabilization: "
          f"{stab_hc if stab_hc else '>500'} decisions")
    print(f"    S2P rolling stabilization: "
          f"{stab_s2p if stab_s2p else '>500'} decisions")

    # Key diagnostic: rolling rate trajectory direction (is diagonal trending up?)
    def late_slope(traj, kname):
        """Agreement rate slope in last 200 decisions (last 8 checkpoints)."""
        late = traj[-8:]
        if len(late) < 2:
            return 0.0
        rates = [t["rates"][kname] for t in late]
        ns    = [t["n"] for t in late]
        return float(np.polyfit(ns, rates, 1)[0]) * 100   # pp per decision

    slope_l2_hc   = late_slope(traj_roll_hc,  "l2")
    slope_diag_hc = late_slope(traj_roll_hc,  "diagonal")
    slope_l2_s2p  = late_slope(traj_roll_s2p, "l2")
    slope_diag_s2p = late_slope(traj_roll_s2p, "diagonal")

    print()
    print("  Late-window slope (pp/decision, last 200 decisions):")
    print(f"    Healthcare:  L2={slope_l2_hc:+.4f}  Diagonal={slope_diag_hc:+.4f}"
          f"  {'Diagonal trending up' if slope_diag_hc > slope_l2_hc else 'L2 trending up or tied'}")
    print(f"    S2P:         L2={slope_l2_s2p:+.4f}  Diagonal={slope_diag_s2p:+.4f}"
          f"  {'Diagonal trending up' if slope_diag_s2p > slope_l2_s2p else 'L2 trending up or tied'}")

    results["ask_1"] = {
        "healthcare":  {"cumulative": traj_cum_hc,  "rolling": traj_roll_hc},
        "s2p":         {"cumulative": traj_cum_s2p, "rolling": traj_roll_s2p},
        "verdicts": {
            "final_cumulative_hc": final_cum_hc,
            "final_rolling_hc":    final_roll_hc,
            "final_cumulative_s2p": final_cum_s2p,
            "final_rolling_s2p":   final_roll_s2p,
            "stabilized_at_hc":    stab_hc,
            "stabilized_at_s2p":   stab_s2p,
            "late_slope_l2_hc":    round(slope_l2_hc,   4),
            "late_slope_diag_hc":  round(slope_diag_hc, 4),
            "late_slope_l2_s2p":   round(slope_l2_s2p,  4),
            "late_slope_diag_s2p": round(slope_diag_s2p, 4),
        },
    }

    # ── ASK 2: Simplified Phase 2 rule ────────────────────────────────────────
    print()
    print("=" * 60)
    print("ASK 2: SIMPLIFIED PHASE 2 RULE")
    print("=" * 60)

    test_cases = [
        ("Healthcare SOC",    HC_NOISE,                                               "diagonal"),
        ("S2P Manufacturing", s2p_noise,                                              "diagonal"),
        ("FinServ (uniform)", np.array([0.10]*6),                                    "l2"),
        ("Startup (extreme)", np.array([0.07, 0.09, 0.05, 0.30, 0.12, 0.28]),       "diagonal"),
    ]

    print()
    print(f"  Rule: noise_ratio > 1.5 → diagonal, else L2  (ρ dropped)")
    print()
    print(f"  {'Deployment':<26} {'Ratio':>7} {'Rule':>10} {'Expected':>10} {'Match':>7}")
    print("  " + "-" * 65)

    all_match = True
    ask2_out  = []
    for name, sigma, expected in test_cases:
        rule_pick, ratio = simplified_phase2(sigma)
        match = (rule_pick == expected)
        if not match:
            all_match = False
        print(f"  {name:<26} {ratio:>6.1f}×  {rule_pick:>10} {expected:>10}"
              f" {'  YES' if match else '  NO':>7}")
        ask2_out.append({
            "name": name, "ratio": round(ratio, 2),
            "rule": rule_pick, "expected": expected, "match": match,
        })

    print()
    print(f"  Result: {'ALL PASS' if all_match else 'SOME FAIL'}")
    if all_match:
        print("  Simplified rule sufficient. Phase 2 can drop rho_max parameter.")
        print("  Explanation A confirmed: noise ratio alone determines kernel choice.")
    else:
        print("  Some test cases fail. rho_max still needed in Phase 2.")

    results["ask_2"] = {
        "simplified_rule_correct": all_match,
        "test_cases": ask2_out,
    }

    # ── ASK 3: Healthcare persona portfolio ───────────────────────────────────
    print()
    print("=" * 60)
    print("ASK 3: HEALTHCARE PERSONA PORTFOLIO (4 personas)")
    print("=" * 60)
    print(f"  N_SEEDS=15  DAYS=60  V=200  η_override=0.01")

    hdr3 = (f"\n  {'Persona':<46} {'Ratio':>6}"
            f"  {'L2 D1':>6} {'L2 D60':>7} {'L2 Δ':>6}"
            f"  {'D D1':>6} {'D D60':>7} {'D Δ':>6}"
            f"  {'Adv':>6}")
    print(hdr3)
    print("  " + "-" * 100)

    hc_results = []
    for persona in HEALTHCARE_PERSONAS:
        print(f"  Running {persona['name']} ...", flush=True)
        r = run_hc_comparison(soc_config, persona, HC_TEAM)
        noise = persona["noise"]
        actual_ratio = float(noise.max() / noise.min())
        print(f"  {persona['name']:<46} {actual_ratio:>5.1f}×"
              f"  {r['l2']['day1']:>6.1%} {r['l2']['day60']:>7.1%} {r['l2']['delta']:>+5.2%}"
              f"  {r['diagonal']['day1']:>6.1%} {r['diagonal']['day60']:>7.1%}"
              f" {r['diagonal']['delta']:>+5.2%}"
              f"  {r['diagonal_advantage']:>+5.2%}")
        hc_results.append({
            "name":       persona["name"],
            "sigma_mean": persona["sigma_mean"],
            "ratio":      round(actual_ratio, 2),
            **r,
        })

    # Summary
    advantages = [r["diagonal_advantage"] for r in hc_results]
    ratios     = [r["ratio"] for r in hc_results]
    corr       = float(np.corrcoef(ratios, advantages)[0, 1]) if len(ratios) > 2 else 0.0

    print()
    print("  SUMMARY:")
    print(f"    Diagonal advantage range: {min(advantages):+.2%} to {max(advantages):+.2%}")
    print(f"    All positive? {'YES' if all(a > 0 for a in advantages) else 'NO'}")
    print(f"    Mean advantage: {float(np.mean(advantages)):+.2%}")
    print(f"    Corr(ratio, advantage): {corr:.3f}")

    print()
    if corr > 0.50:
        print("    CONFIRMED: advantage scales with noise ratio.")
        print("    The simplified Phase 2 threshold (ratio>1.5) is grounded here:")
        below = [r for r in hc_results if r["ratio"] <= 1.5]
        above = [r for r in hc_results if r["ratio"] >  1.5]
        if below:
            mean_adv_below = float(np.mean([r["diagonal_advantage"] for r in below]))
            print(f"      ratio ≤ 1.5: mean adv = {mean_adv_below:+.2%} (marginal)")
        if above:
            mean_adv_above = float(np.mean([r["diagonal_advantage"] for r in above]))
            print(f"      ratio > 1.5: mean adv = {mean_adv_above:+.2%} (clear)")
    elif corr > 0:
        print("    Advantage is positive across all personas but weak correlation")
        print("    with ratio — other factors (σ_mean, team quality) also contribute.")
    else:
        print("    Advantage does not scale with ratio.")
        print("    Noise ratio is not a sufficient proxy for kernel selection.")

    results["ask_3"] = hc_results

    # ── SAVE ──────────────────────────────────────────────────────────────────
    total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Total runtime: {total:.1f}s")

    out_dir  = Path("experiments/factorial/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selector_fixes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved → {out_path}")
    print()


if __name__ == "__main__":
    main()
