"""
V-HC-CONFIG: Factor Quarantine Mask on Healthcare Persona (1D-N4 equivalent).

Validates whether masking the 2 noisiest factors (device_trust, time_anomaly)
rescues accuracy at σ=0.220 — a regime where the full 6-factor scorer degrades.

Condition A (control):  No mask, 6/6 factors. Expected: Day60 < Day1.
Condition B (treatment): Mask time_anomaly (idx 3) + device_trust (idx 5).
                          Score and update on 4 active factors only.

NOTE: ProfileScorer does not have a factor_mask parameter in the current
codebase. Masking is implemented via AsymmetricScorer: masked dimensions
are zeroed in both the input vector and the initial centroid, so they
contribute 0 to L2 distance in all epochs. This is mathematically
equivalent to projecting to the 4-factor subspace.

Usage:
    python experiments/persona_sweeps/priority1/run_vhc_config.py \
        --output experiments/persona_sweeps/results/priority1_vhc_config/
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
HARNESS_DIR = REPO_ROOT / "experiments" / "persona_sweeps"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HARNESS_DIR))

from run_harness import (
    AsymmetricScorer,
    ETA_CONFIRM, ETA_OVERRIDE,
    P5_E0, P5_EPS, P5_TAU,
)
from src.data.domain_config import load_domain_config

# ── Constants ─────────────────────────────────────────────────────────────────
N_SEEDS = 15
DAYS    = 60

# Factor mask built by name — avoids hardcoded index assumptions.
# Indices are resolved at runtime against cfg["factors"].
MASKED_FACTOR_NAMES = ["time_anomaly", "device_trust"]

# ── Inline 1D-N4 persona ──────────────────────────────────────────────────────
PERSONA = {
    "persona_id": "1D-N4",
    "name": "Healthcare Noisy (sigma=0.220)",
    "alerts_per_day": 200,
    "category_distribution": {
        "credential_access":    0.25,
        "lateral_movement":     0.15,
        "data_exfiltration":    0.15,
        "insider_threat":       0.15,
        "cloud_infrastructure": 0.15,
        "threat_intel_match":   0.15,
    },
    "analyst_team": [
        {"id": "a1", "override_rate": 0.14, "override_quality": 0.90, "fatigue_factor": 0.18},
        {"id": "a2", "override_rate": 0.26, "override_quality": 0.78, "fatigue_factor": 0.28},
        {"id": "a3", "override_rate": 0.24, "override_quality": 0.80, "fatigue_factor": 0.22},
        {"id": "a4", "override_rate": 0.42, "override_quality": 0.63, "fatigue_factor": 0.40},
    ],
    "factor_noise_profile": {
        "travel_match":             {"base_noise": 0.18},
        "asset_criticality":        {"base_noise": 0.20},
        "threat_intel_enrichment":  {"base_noise": 0.19},
        "time_anomaly":             {"base_noise": 0.25},
        "pattern_history":          {"base_noise": 0.22},
        "device_trust":             {"base_noise": 0.28},
    },
    "environment_shifts": [],
}


def _noise_vec(persona, factor_names):
    return np.array([persona["factor_noise_profile"][f]["base_noise"]
                     for f in factor_names])


def _analyst_params(team):
    params = []
    for a in team:
        eff_o = min(1.0, a["override_rate"] * (1 + a["fatigue_factor"] * 0.3))
        eff_q = max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2))
        params.append((eff_o, eff_q))
    return params


def _cat_dist(persona, categories):
    cd = persona["category_distribution"]
    w  = np.array([cd[c] for c in categories], dtype=float)
    return w / w.sum()


def run_condition(persona, mu_true, categories, cat_to_idx, gt_dists_arr,
                  factor_names, factor_mask=None):
    """
    60-day PROD-5 simulation.

    factor_mask : None  → no masking (Condition A)
                  array → zero out masked dimensions in inputs and centroids (Condition B)
    """
    C, A, d   = mu_true.shape
    noise     = _noise_vec(persona, factor_names)
    a_params  = _analyst_params(persona["analyst_team"])
    n_an      = len(a_params)
    apd       = persona["alerts_per_day"]
    cat_w     = _cat_dist(persona, categories)
    cat_idx   = np.arange(C)

    mask = factor_mask if factor_mask is not None else np.ones(d)

    acc_day1_all  = []
    acc_day30_all = []
    acc_day60_all = []
    conv_days_all = {cat: [] for cat in categories}   # first day err<P5_EPS

    for si in range(N_SEEDS):
        rng    = np.random.RandomState(si + 9000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        mu_init = np.clip(mu_true + offset, 0, 1) * mask[np.newaxis, np.newaxis, :]

        scorer = AsymmetricScorer(
            mu_init, tau=P5_TAU,
            eta_confirm=ETA_CONFIRM, eta_override=ETA_OVERRIDE,
        )

        # Per-category convergence tracking (first day max-action err < P5_EPS)
        conv_day = {cat: None for cat in categories}
        daily_acc = []

        for day in range(DAYS):
            day_num  = day + 1
            n_alerts = int(rng.poisson(apd))
            correct  = 0

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=cat_w))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f_raw   = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise, 0, 1)
                f       = f_raw * mask          # zero out quarantined dims

                pred_a, _ = scorer.score(f, c)
                correct  += int(pred_a == true_gt)

                ai             = rng.randint(n_an)
                eff_over, eff_q = a_params[ai]

                if rng.random() < eff_over:
                    if rng.random() < eff_q:
                        gt_a = true_gt
                    else:
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])
                    scorer.update_override(f, c, gt_a)
                else:
                    scorer.update_confirm(f, c, pred_a)

            daily_acc.append(correct / n_alerts if n_alerts > 0 else 0.0)

            # Check convergence per category (max-action L2 in active dims)
            for ci, cat in enumerate(categories):
                if conv_day[cat] is None:
                    err = max(
                        np.linalg.norm((scorer.mu[ci, a] - mu_true[ci, a]) * mask)
                        for a in range(A)
                    )
                    if err < P5_EPS:
                        conv_day[cat] = day_num

        acc_day1_all.append(daily_acc[0])
        acc_day30_all.append(daily_acc[29])
        acc_day60_all.append(daily_acc[59])

        for cat in categories:
            conv_days_all[cat].append(conv_day[cat])

    # Summarise convergence: mean conv day (only converged seeds)
    conv_summary = {}
    for cat in categories:
        days_conv = [d for d in conv_days_all[cat] if d is not None]
        n_nc      = N_SEEDS - len(days_conv)
        conv_summary[cat] = {
            "mean_conv_day": round(float(np.mean(days_conv)), 1) if days_conv else None,
            "n_converged":   len(days_conv),
            "not_converged": n_nc,
        }

    return {
        "acc_day1":  round(float(np.mean(acc_day1_all)),  4),
        "acc_day30": round(float(np.mean(acc_day30_all)), 4),
        "acc_day60": round(float(np.mean(acc_day60_all)), 4),
        "delta":     round(float(np.mean(acc_day60_all)) - float(np.mean(acc_day1_all)), 4),
        "n_cats_converged": sum(1 for c in conv_summary.values() if c["not_converged"] == 0),
        "convergence": conv_summary,
    }


def print_results(res_a, res_b, categories, factor_names):
    print()
    print("=" * 62)
    print("=== V-HC-CONFIG: Factor Quarantine on Healthcare Persona ===")
    print(f"    σ_mean≈0.220  |  Masked: {', '.join(MASKED_FACTOR_NAMES)}")
    print(f"    N={N_SEEDS} seeds  |  {DAYS} days  |  "
          f"η_confirm={ETA_CONFIRM}  η_override={ETA_OVERRIDE}")
    print("=" * 62)
    print()

    def _gate(r):
        return "PASS" if r["acc_day60"] >= r["acc_day1"] else "FAIL"

    hdr = (f"| {'Condition':<11} | {'Factors':7} | {'Day 1':6} | "
           f"{'Day 30':6} | {'Day 60':6} | {'Δ(60-1)':8} | {'Gate':6} |")
    sep = ("|" + "-"*13 + "|" + "-"*9 + "|" + "-"*8 + "|"
           + "-"*8 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*8 + "|")
    print(hdr)
    print(sep)

    for label, factors, r in [
        ("A: No mask",  "6/6", res_a),
        ("B: Masked",   "4/6", res_b),
    ]:
        g = _gate(r)
        print(f"| {label:<11} | {factors:7} | {r['acc_day1']:6.1%} | "
              f"{r['acc_day30']:6.1%} | {r['acc_day60']:6.1%} | "
              f"{r['delta']:+8.3%} | {g:<6} |")

    print()
    print("Per-category convergence comparison (mean days to err<0.10):")
    print()
    hdr2 = (f"| {'Category':<24} | {'No mask':10} | {'Masked':10} | {'Δ days':8} |")
    sep2 = "|" + "-"*26 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*10 + "|"
    print(hdr2)
    print(sep2)

    for cat in categories:
        ca = res_a["convergence"][cat]
        cb = res_b["convergence"][cat]

        def _fmt(c):
            if c["mean_conv_day"] is None:
                return f"NC({c['not_converged']}/{N_SEEDS})"
            return f"{c['mean_conv_day']:.1f}d"

        fa = _fmt(ca)
        fb = _fmt(cb)

        # Delta: only if both converged
        if ca["mean_conv_day"] is not None and cb["mean_conv_day"] is not None:
            ddelta = f"{cb['mean_conv_day'] - ca['mean_conv_day']:+.1f}d"
        elif ca["mean_conv_day"] is None and cb["mean_conv_day"] is not None:
            ddelta = "rescued"
        elif ca["mean_conv_day"] is not None and cb["mean_conv_day"] is None:
            ddelta = "worse"
        else:
            ddelta = "—"

        print(f"| {cat:<24} | {fa:>10} | {fb:>10} | {ddelta:>8} |")

    print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 62)
    print("VERDICT:")
    b_pass = res_b["acc_day60"] >= res_b["acc_day1"]
    a_pass = res_a["acc_day60"] >= res_a["acc_day1"]
    lift   = res_b["acc_day60"] - res_a["acc_day60"]

    if b_pass and not a_pass:
        print(f"  Factor quarantine RESCUES healthcare (Day60 lift: {lift:+.3%}).")
        print(f"  Deploy with mask on {' + '.join(MASKED_FACTOR_NAMES)}.")
        print(f"  Day60 B={res_b['acc_day60']:.1%} vs A={res_a['acc_day60']:.1%}  "
              f"(+{lift:.3%}).")
    elif b_pass and a_pass:
        print(f"  Both conditions pass. Mask adds {lift:+.3%} Day60 lift.")
        print(f"  Optional: quarantine provides marginal improvement.")
    elif not b_pass and not a_pass:
        print(f"  4-factor scoring INSUFFICIENT. Both conditions degrade.")
        print(f"  Healthcare needs noise remediation before deployment.")
        print(f"  Δ(B-A) Day60 = {lift:+.3%}  "
              f"({'improvement' if lift > 0 else 'no improvement'} over no-mask).")
    else:
        # b_pass=False, a_pass=True — masking made things worse
        print(f"  UNEXPECTED: mask degrades vs no-mask (Δ Day60 = {lift:+.3%}).")
        print(f"  Investigate: removing factors may drop information needed for convergence.")

    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(description="V-HC-CONFIG Factor Quarantine")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)

    cfg          = load_domain_config("soc_product_v50")
    categories   = cfg["categories"]
    factor_names = cfg["factors"]
    mu_true      = np.array(cfg["mu"], dtype=float)
    gt_dists_raw = cfg["gt_distributions"]
    cat_to_idx   = {c: i for i, c in enumerate(categories)}
    gt_dists_arr = np.array([gt_dists_raw[c] for c in categories], dtype=float)
    gt_dists_arr /= gt_dists_arr.sum(axis=1, keepdims=True)

    # Build mask from names — resolve indices against actual config ordering
    FACTOR_MASK = np.array(
        [0.0 if fn in MASKED_FACTOR_NAMES else 1.0 for fn in factor_names]
    )

    # Confirm factor index alignment
    for i, fn in enumerate(factor_names):
        mask_val = FACTOR_MASK[i]
        status   = "ACTIVE" if mask_val == 1.0 else "MASKED"
        noise    = PERSONA["factor_noise_profile"].get(fn, {}).get("base_noise", "?")
        print(f"  Factor {i}: {fn:<28}  σ={noise}  [{status}]")
    print()

    print(f"Running Condition A (no mask, 6 factors)  N={N_SEEDS} seeds...")
    t0    = time.time()
    res_a = run_condition(PERSONA, mu_true, categories, cat_to_idx,
                          gt_dists_arr, factor_names, factor_mask=None)
    print(f"  Done ({time.time()-t0:.1f}s)  "
          f"Day1={res_a['acc_day1']:.1%}  Day60={res_a['acc_day60']:.1%}  "
          f"Δ={res_a['delta']:+.3%}")

    print(f"Running Condition B (masked: {', '.join(MASKED_FACTOR_NAMES)})  "
          f"N={N_SEEDS} seeds...")
    t0    = time.time()
    res_b = run_condition(PERSONA, mu_true, categories, cat_to_idx,
                          gt_dists_arr, factor_names, factor_mask=FACTOR_MASK)
    print(f"  Done ({time.time()-t0:.1f}s)  "
          f"Day1={res_b['acc_day1']:.1%}  Day60={res_b['acc_day60']:.1%}  "
          f"Δ={res_b['delta']:+.3%}")

    print_results(res_a, res_b, categories, factor_names)
    factor_mask_list = list(float(x) for x in FACTOR_MASK)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "persona_id":     PERSONA["persona_id"],
        "sigma_mean":     round(float(np.mean([
            v["base_noise"] for v in PERSONA["factor_noise_profile"].values()
        ])), 4),
        "factor_mask":    factor_mask_list,
        "masked_factors": MASKED_FACTOR_NAMES,
        "condition_a":    res_a,
        "condition_b":    res_b,
    }
    out_path = output_dir / "vhc_config_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
