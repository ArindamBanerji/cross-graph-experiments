"""
V-HC-CONFIG Re-run: L2 vs Mask vs DiagonalKernel on healthcare persona 1D-N4.

Three conditions on the same persona (σ_mean ≈ 0.220):
  A: L2Kernel, all 6 factors        — control, reproduces -7.4pp
  B: L2Kernel + binary mask 4/6     — reproduces -3.5pp
  C: DiagonalKernel, all 6 factors  — THE TEST

Usage:
    python experiments/factorial/run_vhc_diagonal.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel

# ── Constants ──────────────────────────────────────────────────────────────────
N_SEEDS      = 15
DAYS         = 60
ETA          = 0.05
ETA_NEG      = 0.05
ETA_OVERRIDE = 0.01
VERIFY_RATE  = 0.30
CONV_EPS     = 0.10     # max-action L2 error for convergence

OUTPUT_DIR   = Path("experiments/factorial/results/vhc_diagonal")

# ── Healthcare persona 1D-N4 ───────────────────────────────────────────────────
FACTOR_NAMES = [
    "travel_match",
    "asset_criticality",
    "threat_intel_enrichment",
    "time_anomaly",
    "pattern_history",
    "device_trust",
]

FACTOR_NOISE = np.array([0.18, 0.20, 0.19, 0.25, 0.22, 0.28])   # d=6

APD          = 200   # alerts per day

# 4-analyst team (senior + 2 mid + junior)
ANALYST_TEAM = [
    {"override_rate": 0.25, "override_quality": 0.90, "fatigue_factor": 0.10},  # senior
    {"override_rate": 0.28, "override_quality": 0.78, "fatigue_factor": 0.20},  # mid
    {"override_rate": 0.28, "override_quality": 0.80, "fatigue_factor": 0.18},  # mid
    {"override_rate": 0.35, "override_quality": 0.63, "fatigue_factor": 0.35},  # junior
]

# Condition B: mask time_anomaly (index 3 in FACTOR_NAMES) + device_trust (index 5)
# FACTOR_NAMES = [travel_match(0), asset_criticality(1), threat_intel(2),
#                 time_anomaly(3), pattern_history(4), device_trust(5)]
MASK_B = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])   # zero out time_anomaly + device_trust


# ── Analyst params ─────────────────────────────────────────────────────────────
def build_analyst_params():
    params = []
    for a in ANALYST_TEAM:
        ff  = a["fatigue_factor"]
        eo  = min(1.0, a["override_rate"] * (1 + ff * 0.3))
        eq  = max(0.4, a["override_quality"] * (1 - ff * 0.2))
        params.append((eo, eq))
    return params


# ── Core simulation ────────────────────────────────────────────────────────────
def run_condition(
    mu_true:      np.ndarray,      # (C, A, d)
    categories:   list,
    actions:      list,
    gt_arr:       np.ndarray,      # (C, A)
    cat_w:        np.ndarray,      # (C,)
    kernel,                        # L2Kernel or DiagonalKernel
    factor_mask:  np.ndarray | None,  # (d,) or None
    noise_array:  np.ndarray,      # (d,) per-factor σ — for generating alerts
    label:        str,
) -> dict:
    """Run N_SEEDS × DAYS simulation for one condition."""
    C, A, d = mu_true.shape
    a_params   = build_analyst_params()
    n_analysts = len(a_params)

    all_daily_acc = []
    all_conv      = {cat: [] for cat in categories}

    for si in range(N_SEEDS):
        rng = np.random.default_rng(42 + si)

        offset  = rng.uniform(-0.15, 0.15, mu_true.shape)
        mu_init = np.clip(mu_true + offset, 0, 1)

        # Apply mask to initial centroids (condition B only)
        if factor_mask is not None:
            mu_init = mu_init * factor_mask[np.newaxis, np.newaxis, :]

        scorer = ProfileScorer(
            mu_init.copy(), actions,
            scoring_kernel=kernel,
            eta_override=ETA_OVERRIDE,
        )
        scorer.eta     = ETA
        scorer.eta_neg = ETA_NEG

        daily_acc = np.zeros(DAYS)
        conv_day  = {cat: None for cat in categories}

        for day in range(DAYS):
            n_alerts = int(rng.poisson(APD))
            correct  = 0

            for _ in range(n_alerts):
                ci   = int(rng.choice(C, p=cat_w))
                a_gt = int(rng.choice(A, p=gt_arr[ci]))

                # Generate factor vector with per-factor noise
                z = rng.standard_normal(d)
                f = np.clip(mu_true[ci, a_gt] + noise_array * z, 0, 1)

                # Apply mask if condition B
                if factor_mask is not None:
                    f = f * factor_mask

                res    = scorer.score(f, ci)
                pred_a = res.action_index
                correct += int(pred_a == a_gt)

                # Analyst verification
                if rng.random() < VERIFY_RATE:
                    ai_idx = rng.integers(n_analysts)
                    eff_o, eff_q = a_params[ai_idx]

                    if rng.random() < eff_o:
                        gt_a = a_gt if rng.random() < eff_q else int(
                            rng.choice([a for a in range(A) if a != a_gt])
                        )
                        scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                    else:
                        scorer.update(f, ci, pred_a, True)

            daily_acc[day] = correct / n_alerts if n_alerts > 0 else 0.0

            # Convergence check
            for ci2, cat in enumerate(categories):
                if conv_day[cat] is None:
                    err = max(
                        np.linalg.norm(scorer.centroids[ci2, a] - mu_true[ci2, a])
                        for a in range(A)
                    )
                    if err < CONV_EPS:
                        conv_day[cat] = day + 1

        all_daily_acc.append(daily_acc)
        for cat in categories:
            all_conv[cat].append(conv_day[cat] if conv_day[cat] is not None else DAYS + 1)

    acc_arr = np.stack(all_daily_acc)   # (N_SEEDS, DAYS)
    mean_acc = acc_arr.mean(axis=0)     # (DAYS,)

    conv_summary = {}
    for cat in categories:
        days_list = all_conv[cat]
        pct_conv  = float(np.mean([v <= DAYS for v in days_list]))
        converged = [v for v in days_list if v <= DAYS]
        conv_summary[cat] = {
            "mean_conv_day": round(float(np.mean(converged)), 1) if converged else None,
            "pct_converged": round(pct_conv, 3),
        }

    day1_acc  = float(mean_acc[0])
    day30_acc = float(mean_acc[29])
    day60_acc = float(mean_acc[-1])
    delta     = day60_acc - day1_acc

    return {
        "label":        label,
        "day1_accuracy":  round(day1_acc,  4),
        "day30_accuracy": round(day30_acc, 4),
        "day60_accuracy": round(day60_acc, 4),
        "delta_d1_d60":   round(delta,     4),
        "convergence":    conv_summary,
        "daily_acc":      [round(float(x), 4) for x in mean_acc],
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config     = load_domain_config("soc_product_v50")
    mu_true    = config["mu"]            # (C, A, d)
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]

    C, A, d = mu_true.shape

    # GT distributions
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        probs = np.array(gt_dists.get(cat, [1.0 / A] * A), dtype=float)[:A]
        gt_arr[ci] = probs / probs.sum()

    cat_w = np.ones(C) / C   # uniform category weights

    # Build DiagonalKernel weights for condition C
    weights_raw  = 1.0 / (FACTOR_NOISE ** 2)
    weights_norm = weights_raw / weights_raw.max()

    # ── Print header ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("=== V-HC-CONFIG RE-RUN: L2 vs Mask vs DiagonalKernel ===")
    print("=" * 60)
    print(f"  Persona:  1D-N4 healthcare  σ_mean={FACTOR_NOISE.mean():.3f}"
          f"  APD={APD}  Seeds={N_SEEDS}  Days={DAYS}")
    print(f"  η_confirm={ETA}  η_override={ETA_OVERRIDE}")
    print()
    print("  Diagonal weight vector:")
    print(f"  {'Factor':<26} {'σ':>6}  {'1/σ²':>8}  {'Weight':>8}  Role")
    print("  " + "-" * 70)
    roles = ["Full signal", "Strong signal", "Full signal",
             "Weak signal", "Moderate", "Weak signal"]
    for i, fn in enumerate(FACTOR_NAMES):
        print(f"  {fn:<26} {FACTOR_NOISE[i]:>6.2f}  {weights_raw[i]:>8.2f}"
              f"  {weights_norm[i]:>8.4f}  {roles[i]}")
    print()

    # ── Run all three conditions ───────────────────────────────────────────────
    conditions = [
        ("A: L2 only",   L2Kernel(),                    None,   "L2",       "6/6"),
        ("B: L2 + mask", L2Kernel(),                    MASK_B, "L2",       "4/6"),
        ("C: Diagonal",  DiagonalKernel(weights_norm),  None,   "Diagonal", "6/6"),
    ]

    results = []
    t_total = time.time()

    for label, kernel, mask, kernel_name, factor_str in conditions:
        mask_desc = "mask(time_anomaly,device_trust)" if mask is not None else "none"
        print(f"  Running {label}  kernel={kernel_name}  mask={mask_desc} ...", flush=True)
        t0 = time.time()
        result = run_condition(
            mu_true, categories, actions, gt_arr, cat_w,
            kernel, mask, FACTOR_NOISE, label,
        )
        elapsed = time.time() - t0
        result["kernel_name"]  = kernel_name
        result["factor_count"] = factor_str
        d60   = result["day60_accuracy"]
        delta = result["delta_d1_d60"]
        sign  = "+" if delta >= 0 else ""
        print(f"    → Day60={d60:.1%}  Δ={sign}{delta:.2%}  ({elapsed:.1f}s)")
        results.append(result)

    total_time = time.time() - t_total

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("=== V-HC-CONFIG RE-RUN: L2 vs Mask vs DiagonalKernel ===")
    print("=" * 60)
    print()
    print(f"  {'Condition':<14} {'Kernel':<10} {'Factors':<8}"
          f" {'Day 1':>7} {'Day 30':>7} {'Day 60':>7} {'Δ(60-1)':>9}  Gate")
    print("  " + "-" * 75)

    GATE_THRESH = 0.0   # Day60 >= Day1 → passes

    for r in results:
        d1   = r["day1_accuracy"]
        d30  = r["day30_accuracy"]
        d60  = r["day60_accuracy"]
        delt = r["delta_d1_d60"]
        sign = "+" if delt >= 0 else ""
        gate = "PASS" if delt >= GATE_THRESH else "FAIL"
        print(f"  {r['label']:<14} {r['kernel_name']:<10} {r['factor_count']:<8}"
              f" {d1:>7.1%} {d30:>7.1%} {d60:>7.1%} {sign}{delt:>8.2%}  {gate}")

    # ── Per-category convergence ───────────────────────────────────────────────
    print()
    print("  Per-category convergence (mean day | pct converged):")
    print(f"  {'Category':<26} {'L2 only':>14} {'L2+mask':>14} {'Diagonal':>14}")
    print("  " + "-" * 72)

    def fmt_conv(cv):
        d = cv["mean_conv_day"]
        p = cv["pct_converged"]
        if d is not None:
            return f"Day {d:.0f} ({p:.0%})"
        return f"NC     ({p:.0%})"

    for cat in categories:
        l2_cv   = results[0]["convergence"][cat]
        mask_cv = results[1]["convergence"][cat]
        diag_cv = results[2]["convergence"][cat]
        print(f"  {cat:<26} {fmt_conv(l2_cv):>14} {fmt_conv(mask_cv):>14}"
              f" {fmt_conv(diag_cv):>14}")

    # ── Diagonal weight vector recap ──────────────────────────────────────────
    print()
    print("  Diagonal weight vector:")
    print(f"  {'Factor':<26} {'σ':>6}  {'Weight':>8}  Role")
    print("  " + "-" * 60)
    for i, fn in enumerate(FACTOR_NAMES):
        print(f"  {fn:<26} {FACTOR_NOISE[i]:>6.2f}  {weights_norm[i]:>8.4f}  {roles[i]}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    diag_delta = results[2]["delta_d1_d60"]
    diag_d1    = results[2]["day1_accuracy"]
    diag_d60   = results[2]["day60_accuracy"]
    l2_delta   = results[0]["delta_d1_d60"]
    mask_delta = results[1]["delta_d1_d60"]

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    if diag_delta >= 0.0:
        print()
        print("  DiagonalKernel RESCUES healthcare at σ>0.20.")
        print("  Segment opens at v6.0. Factor mask DEPRECATED.")
        print(f"  Noise ceiling moves: σ=0.157 (L2) → σ≈0.25 (Diagonal).")
        print()
        print(f"  Diagonal: {diag_d1:.1%} → {diag_d60:.1%}  Δ={diag_delta:+.2%}")
        print(f"  L2 only:  Δ={l2_delta:+.2%}  (baseline degradation)")
        print(f"  L2+mask:  Δ={mask_delta:+.2%}  (partial mitigation — obsolete)")
        print()
        print("  Recommendation: deploy DiagonalKernel(weights=1/σ²) as default")
        print("  for any domain where per-factor σ_range/σ_mean > 0.40.")
    elif diag_delta > -0.01:
        print()
        print("  DiagonalKernel STABILIZES healthcare but does not improve.")
        print("  Healthcare opens with monitoring. Better than mask"
              f" ({mask_delta:+.2%}).")
        print(f"  Diagonal Δ={diag_delta:+.2%} vs L2 Δ={l2_delta:+.2%}.")
        print()
        print("  Recommendation: deploy Diagonal as noise mitigation layer.")
        print("  Per-category accuracy monitoring required at σ>0.25.")
    else:
        print()
        print("  DiagonalKernel INSUFFICIENT at this noise level.")
        print("  Factor mask was the right approach.")
        print(f"  Diagonal Δ={diag_delta:+.2%} vs mask Δ={mask_delta:+.2%}.")
        print()
        print("  Healthcare still requires noise remediation before v6.0.")
        print("  Defer segment opening to v6.5 (ShrinkageKernel with full covariance).")

    print()
    print(f"  Completed in {total_time:.1f}s  ({N_SEEDS} seeds × {DAYS} days"
          f" × 3 conditions × {APD} alerts/day)")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment":  "V-HC-CONFIG re-run: DiagonalKernel",
        "persona":     "1D-N4 healthcare",
        "sigma_mean":  round(float(FACTOR_NOISE.mean()), 4),
        "n_seeds":     N_SEEDS,
        "days":        DAYS,
        "apd":         APD,
        "eta":         ETA,
        "eta_override": ETA_OVERRIDE,
        "weight_vector": {
            fn: {"sigma": float(FACTOR_NOISE[i]), "weight": round(float(weights_norm[i]), 4)}
            for i, fn in enumerate(FACTOR_NAMES)
        },
        "conditions":  [
            {k: v for k, v in r.items() if k != "daily_acc"}
            for r in results
        ],
        "verdict": (
            "rescue" if diag_delta >= 0.0 else
            "stabilize" if diag_delta > -0.01 else
            "insufficient"
        ),
    }
    out_path = OUTPUT_DIR / "vhc_diagonal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")
    print()


if __name__ == "__main__":
    main()
