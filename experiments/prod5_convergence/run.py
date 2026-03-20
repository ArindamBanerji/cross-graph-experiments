"""
PROD-5: Production Convergence Rate Validation (P31)
Simulates 60 days of production (200 alerts/day, 30% verification).
Compares per-category convergence to L-08 calendar predictions.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.models.profile_scorer import ProfileScorer

# ── Parameters ────────────────────────────────────────────────────────────────
ALERTS_PER_DAY    = 200
VERIFICATION_RATE = 0.30
N_SEEDS           = 30
DAYS              = 60
SIGMA_F           = 0.15   # factor noise std
E0                = 0.15   # initial centroid offset magnitude (uniform ±E0)
EPS_CONV          = 0.05   # convergence threshold (max per-action L2 error)
TAU               = 0.10
ETA               = 0.05
ETA_NEG           = 0.05
N_ACTIONS         = 4
N_FACTORS         = 6

# Alert category weights (from BOOTSTRAP_CATEGORY_WEIGHTS)
CATEGORY_WEIGHTS = {
    "credential_access":    0.30,
    "lateral_movement":     0.20,
    "data_exfiltration":    0.15,
    "insider_threat":       0.15,
    "cloud_infrastructure": 0.10,
    "threat_intel_match":   0.10,
}

SEEDS       = list(range(N_SEEDS))
EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ── L-08 Analytical Prediction ────────────────────────────────────────────────
def predict_category_convergence_weeks(category, effective_alerts_per_period,
                                       verification_rate, graph_level="G1"):
    """
    L-08 onboarding calendar analytical model.

    effective_alerts_per_period = alerts_per_day * category_weight * n_factors
      (factor of n_factors accounts for d-dimensional convergence scaling)

    Geometric decay model: after N pull-updates, centroid offset decays as
      e_N = E0 * (1-eta)^N
    Target: e_N < EPS_CONV  =>  N = ceil(log(EPS/E0) / log(1-eta))

    G1 calibration (1.8) accounts for count-based eta decay and Poisson variance.
    """
    CALIBRATION = {"G1": 1.8, "G2": 2.5, "G3": 3.5}

    # Divide by d to recover daily category alerts
    daily_alerts = effective_alerts_per_period / N_FACTORS
    # Verified pulls per action per week
    verified_per_week_per_action = daily_alerts * verification_rate * 7 / N_ACTIONS

    n_per_action = math.ceil(math.log(EPS_CONV / E0) / math.log(1.0 - ETA))  # = 22
    calib = CALIBRATION.get(graph_level, 1.8)
    weeks = (n_per_action / verified_per_week_per_action) * calib

    return {"weeks": weeks, "n_per_action": n_per_action, "calibration": calib}


# ── Single-seed Simulation ────────────────────────────────────────────────────
def simulate_seed(mu_true, categories, weights_arr, gt_dists_arr, seed):
    C, A, d = mu_true.shape
    rng = np.random.RandomState(seed)

    # Cold-start: offset each centroid element by Uniform(−E0, +E0)
    offset = rng.uniform(-E0, E0, mu_true.shape)
    mu_init = np.clip(mu_true + offset, 0.0, 1.0)

    scorer = ProfileScorer(mu_init.copy(), tau=TAU, eta=ETA, eta_neg=ETA_NEG, seed=seed)

    # Tracking: (C, DAYS)
    daily_mean_err = np.zeros((C, DAYS))
    daily_max_err  = np.zeros((C, DAYS))
    daily_acc      = np.zeros(DAYS)
    daily_cat_acc  = np.full((C, DAYS), np.nan)
    verify_counts  = np.zeros(C, dtype=int)
    converge_day   = np.full(C, -1, dtype=int)  # -1 = not converged by day 60

    cat_indices = np.arange(C)

    for day in range(DAYS):
        n_alerts = rng.poisson(ALERTS_PER_DAY)

        n_correct = 0
        cat_correct_sum   = np.zeros(C)
        cat_alert_count   = np.zeros(C, dtype=int)

        for _ in range(n_alerts):
            c_idx = rng.choice(cat_indices, p=weights_arr)
            gt_a  = rng.choice(A, p=gt_dists_arr[c_idx])
            factors = np.clip(
                mu_true[c_idx, gt_a, :] + rng.randn(d) * SIGMA_F, 0.0, 1.0
            )

            result  = scorer.score(factors, c_idx)
            pred_a  = result.action_index
            correct = int(pred_a == gt_a)

            n_correct += correct
            cat_correct_sum[c_idx] += correct
            cat_alert_count[c_idx] += 1

            if rng.random() < VERIFICATION_RATE:
                scorer.update(factors, c_idx, pred_a, bool(correct),
                              gt_action_index=gt_a)
                verify_counts[c_idx] += 1

        # End-of-day error
        mu_now = scorer.mu
        for c in range(C):
            per_action_err = np.array([
                np.linalg.norm(mu_now[c, a, :] - mu_true[c, a, :])
                for a in range(A)
            ])
            daily_mean_err[c, day] = per_action_err.mean()
            daily_max_err[c, day]  = per_action_err.max()

            if cat_alert_count[c] > 0:
                daily_cat_acc[c, day] = cat_correct_sum[c] / cat_alert_count[c]

        if n_alerts > 0:
            daily_acc[day] = n_correct / n_alerts

        # Convergence check: first day max error < EPS_CONV for all actions
        for c in range(C):
            if converge_day[c] == -1 and daily_max_err[c, day] < EPS_CONV:
                converge_day[c] = day + 1   # 1-indexed day label

    return {
        "converge_day":    converge_day,         # shape (C,), -1 if not converged
        "daily_mean_err":  daily_mean_err,        # (C, DAYS)
        "daily_acc":       daily_acc,             # (DAYS,)
        "daily_cat_acc":   daily_cat_acc,         # (C, DAYS)
        "verify_counts":   verify_counts,         # (C,)
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config     = load_domain_config("soc_product_v50")
    mu_true    = config["mu"].copy().astype(np.float64)   # (C, A, d)
    categories = config["categories"]
    C, A, d    = mu_true.shape

    # Weights and gt distributions aligned to config category order
    weights_arr  = np.array([CATEGORY_WEIGHTS[c] for c in categories], dtype=float)
    weights_arr /= weights_arr.sum()

    gt_dist_raw  = config["gt_distributions"]
    gt_dists_arr = np.array([gt_dist_raw[c] for c in categories], dtype=float)
    gt_dists_arr = gt_dists_arr / gt_dists_arr.sum(axis=1, keepdims=True)

    # ── L-08 predictions ──────────────────────────────────────────────────────
    l08_preds = {}
    for cat in categories:
        w         = CATEGORY_WEIGHTS[cat]
        effective = ALERTS_PER_DAY * w * N_FACTORS
        l08_preds[cat] = predict_category_convergence_weeks(
            cat, effective, VERIFICATION_RATE, "G1"
        )

    # ── Simulation ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("=== PROD-5: CONVERGENCE RATE VALIDATION (60-day sim) ===")
    print("=" * 60)
    print(f"\nC={C}, A={A}, d={d}  |  seeds={N_SEEDS}, days={DAYS}")
    print(f"alerts/day={ALERTS_PER_DAY}, verify_rate={VERIFICATION_RATE}, "
          f"σ_f={SIGMA_F}, e₀={E0}, ε_conv={EPS_CONV}\n")
    print(f"Running {N_SEEDS} seeds...", flush=True)

    # Storage: (N_SEEDS, C, DAYS) and (N_SEEDS, C)
    all_mean_err    = np.zeros((N_SEEDS, C, DAYS))
    all_conv_day    = np.full((N_SEEDS, C), -1, dtype=int)
    all_daily_acc   = np.zeros((N_SEEDS, DAYS))
    all_cat_acc     = np.full((N_SEEDS, C, DAYS), np.nan)
    all_verify_cnt  = np.zeros((N_SEEDS, C), dtype=int)

    for si, seed in enumerate(SEEDS):
        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1}/{N_SEEDS}] seed={seed}", flush=True)
        res = simulate_seed(mu_true, categories, weights_arr, gt_dists_arr, seed)
        all_mean_err[si]   = res["daily_mean_err"]
        all_conv_day[si]   = res["converge_day"]
        all_daily_acc[si]  = res["daily_acc"]
        all_cat_acc[si]    = res["daily_cat_acc"]
        all_verify_cnt[si] = res["verify_counts"]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    simulated  = {}
    comparison = {}

    for ci, cat in enumerate(categories):
        days_all   = all_conv_day[:, ci]                     # (N_SEEDS,)
        valid_mask = days_all > 0
        valid_days = days_all[valid_mask].astype(float)
        n_not_conv = int((~valid_mask).sum())

        mean_days  = float(valid_days.mean()) if len(valid_days) > 0 else None
        std_days   = float(valid_days.std())  if len(valid_days) > 1 else 0.0
        mean_weeks = mean_days / 7.0          if mean_days is not None else None

        daily_err_mean = all_mean_err[:, ci, :].mean(axis=0)   # (DAYS,)

        simulated[cat] = {
            "converge_day_mean":     round(mean_days, 1)  if mean_days  else None,
            "converge_day_std":      round(std_days, 1),
            "converge_weeks_mean":   round(mean_weeks, 2) if mean_weeks else None,
            "converge_days":         [int(d) if d > 0 else None for d in days_all],
            "not_converged_seeds":   n_not_conv,
            "verify_count_mean":     float(all_verify_cnt[:, ci].mean()),
            "daily_mean_error_mean": daily_err_mean.tolist(),
        }

        pred_w   = l08_preds[cat]["weeks"]
        ratio    = (mean_weeks / pred_w) if mean_weeks is not None else None
        on_track = (ratio is not None) and (ratio <= 1.5)

        comparison[cat] = {
            "predicted_weeks": round(pred_w, 2),
            "simulated_weeks": round(mean_weeks, 2) if mean_weeks else None,
            "ratio":           round(ratio, 2)      if ratio       else None,
            "on_track":        on_track,
        }

    # Gate
    off_track = [c for c in categories if not comparison[c]["on_track"]]
    gate_pass = len(off_track) == 0

    # Accuracy trajectory at days 1, 30, 60
    mean_daily_acc = all_daily_acc.mean(axis=0)
    acc_trajectory = {}
    for label, idx in [("day_1", 0), ("day_30", 29), ("day_60", 59)]:
        per_cat = {}
        for ci, cat in enumerate(categories):
            col   = all_cat_acc[:, ci, idx]
            valid = col[~np.isnan(col)]
            per_cat[cat] = round(float(valid.mean()), 4) if len(valid) > 0 else None
        acc_trajectory[label] = {
            "overall":      round(float(mean_daily_acc[idx]), 4),
            "per_category": per_cat,
        }

    # ── Print Table ───────────────────────────────────────────────────────────
    print()
    hdr = (f"| {'Category':<22} | {'Predicted (L-08)':>16} | "
           f"{'Simulated':>9} | {'Ratio':>5} | {'On Track?':>9} |")
    sep = "|" + "-"*24 + "|" + "-"*18 + "|" + "-"*11 + "|" + "-"*7 + "|" + "-"*11 + "|"
    print(hdr)
    print(sep)
    for cat in categories:
        c   = comparison[cat]
        ps  = f"{c['predicted_weeks']:.1f} weeks"   if c["predicted_weeks"] else "N/A"
        ss  = f"{c['simulated_weeks']:.1f} weeks"   if c["simulated_weeks"] else ">60 days"
        rs  = f"{c['ratio']:.2f}"                   if c["ratio"]           else "N/A"
        ts  = "YES" if c["on_track"] else "NO"
        print(f"| {cat:<22} | {ps:>16} | {ss:>9} | {rs:>5} | {ts:>9} |")
    print()

    if gate_pass:
        print("Gate: PASS")
        print("L-08 onboarding calendar validated on synthetic production sim.")
    else:
        print("Gate: FAIL")
        print(f"L-08 predictions need adjustment. Categories off track: {off_track}.")

    print()
    print("Accuracy trajectory (mean over seeds):")
    for label, idx in [("Day  1", 0), ("Day 30", 29), ("Day 60", 59)]:
        print(f"  {label}: {mean_daily_acc[idx]*100:.1f}%")

    print()
    print("Verification count per category at day 60 (mean over seeds):")
    for ci, cat in enumerate(categories):
        s  = simulated[cat]
        nc = s["not_converged_seeds"]
        nc_str = f"  [{nc}/{N_SEEDS} did not converge within {DAYS} days]" if nc > 0 else ""
        print(f"  {cat:<22}: {s['verify_count_mean']:6.0f} verified{nc_str}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "config": {
            "alerts_per_day":    ALERTS_PER_DAY,
            "verification_rate": VERIFICATION_RATE,
            "n_seeds":           N_SEEDS,
            "days":              DAYS,
            "sigma_f":           SIGMA_F,
            "e0":                E0,
            "eps_conv":          EPS_CONV,
            "tau":               TAU,
            "eta":               ETA,
            "eta_neg":           ETA_NEG,
            "domain":            "soc_product_v50",
        },
        "categories":    categories,
        "weights":       CATEGORY_WEIGHTS,
        "l08_predictions": {
            cat: {
                "weeks": round(l08_preds[cat]["weeks"], 3),
                "days":  round(l08_preds[cat]["weeks"] * 7, 1),
            }
            for cat in categories
        },
        "simulated":           simulated,
        "comparison":          comparison,
        "gate_pass":           gate_pass,
        "off_track_categories": off_track,
        "accuracy_trajectory": acc_trajectory,
    }

    out_path = RESULTS_DIR / "prod5_convergence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print("Run charts.py to generate paper figures.")
    return output


if __name__ == "__main__":
    main()
