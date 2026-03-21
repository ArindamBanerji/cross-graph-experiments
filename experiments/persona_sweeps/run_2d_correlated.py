"""
2D Correlated Error Sweep — standard harness + specialized CE1/CE2 analysis.

Extends run_harness.py with:
  CE1 (Phishing Campaign):
    - Centroid drift at Days 19/25/30 (80% correlated wrong overrides)
    - Conservation signal during campaign vs baseline
    - Relative threshold (0.70× baseline) detection
    - Post-campaign recovery decisions (gate: ≤28)
  CE2 (Ransomware Volume Spike):
    - Per-category centroid error: pre-spike vs during-spike
    - Starvation analysis for non-spiking categories

Usage:
    python experiments/persona_sweeps/run_2d_correlated.py \
        --personas experiments/persona_sweeps/personas_sweep_2d_correlated.json \
        --output   experiments/persona_sweeps/results/sweep_2d_correlated/
"""

import sys
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
HARNESS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HARNESS_DIR))

from run_harness import (
    AsymmetricScorer,
    build_persona_noise, build_persona_weights, precompute_day_weights,
    precompute_analyst_params, persona_q_bar,
    run_td034, run_prod5, run_ba,
    print_tables, save_results,
    ETA_CONFIRM, ETA_OVERRIDE, THETA_MIN,
    P5_E0, P5_EPS, P5_TAU,
)
from src.data.domain_config import load_domain_config

# ── 2D constants ──────────────────────────────────────────────────────────────
N_2D_SEEDS     = 15
CORR_ERR_RATE  = 0.80    # fraction of campaign cred_access overrides that are WRONG
RECOVERY_GATE  = 28      # 2× N_half decisions
REL_THRESHOLD  = 0.70    # AMBER fires when campaign signal < 0.70 × baseline mean


# ── CE1: Phishing campaign correlated-error analysis ──────────────────────────
def run_ce1_analysis(persona, mu_true, categories, cat_to_idx, gt_dists_arr,
                     factor_names):
    """
    Full 60-day simulation with correlated campaign errors on credential_access.
    Campaign: Days 20–30 (inclusive). 80% of cred_access overrides → wrong action.
    Tracks:
      - Centroid max-action L2 error at Days 19, 25, 30
      - Daily α·q·V signal (Days 1–19 baseline vs Days 20–30 campaign)
      - Post-campaign recovery: cred_access decisions until error ≤ pre + ε
    """
    C, A, d    = mu_true.shape
    cred_idx   = cat_to_idx["credential_access"]
    noise      = build_persona_noise(persona, factor_names)
    base_w     = build_persona_weights(persona, categories)
    shifts     = persona.get("environment_shifts", [])
    a_params   = precompute_analyst_params(persona["analyst_team"])
    n_analysts = len(a_params)
    apd        = persona["alerts_per_day"]
    cat_idx    = np.arange(C)
    n_days     = 60
    day_wts    = precompute_day_weights(base_w, categories, cat_to_idx, shifts, n_days)

    CAMP_START = 20   # day number, 1-indexed, inclusive
    CAMP_END   = 30

    err_day19_all = []
    err_day25_all = []
    err_day30_all = []
    sig_base_all  = []   # mean daily signal Days 1–19
    sig_camp_all  = []   # mean daily signal Days 20–30
    recovery_all  = []   # decisions-to-recovery (-1 = not recovered by Day 60)

    for si in range(N_2D_SEEDS):
        rng    = np.random.RandomState(si + 3000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        scorer = AsymmetricScorer(
            np.clip(mu_true + offset, 0, 1),
            tau=P5_TAU, eta_confirm=ETA_CONFIRM, eta_override=ETA_OVERRIDE,
        )

        daily_sig      = np.zeros(n_days)
        daily_cred_err = np.zeros(n_days)
        pre_camp_err   = None
        post_camp_decs = 0
        recovered_at   = None

        for day in range(n_days):
            day_num       = day + 1
            dw            = day_wts[day]
            n_alerts      = int(rng.poisson(apd))
            in_campaign   = CAMP_START <= day_num <= CAMP_END
            post_campaign = day_num > CAMP_END

            day_ov        = 0
            day_ovc       = 0
            cred_decs_day = 0

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f       = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise, 0, 1)
                pred_a, _ = scorer.score(f, c)

                ai             = rng.randint(n_analysts)
                eff_over, eff_q = a_params[ai]

                if rng.random() < eff_over:
                    # Correlated campaign error for credential_access
                    if in_campaign and c == cred_idx and rng.random() < CORR_ERR_RATE:
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])
                        ovc    = False
                    elif rng.random() < eff_q:
                        gt_a = true_gt
                        ovc  = True
                    else:
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])
                        ovc    = False
                    scorer.update_override(f, c, gt_a)
                    day_ov  += 1
                    day_ovc += int(ovc)
                else:
                    scorer.update_confirm(f, c, pred_a)

                if post_campaign and c == cred_idx:
                    cred_decs_day += 1

            # Daily conservation signal
            if day_ov > 0 and n_alerts > 0:
                daily_sig[day] = (day_ov / n_alerts) * (day_ovc / day_ov) * n_alerts

            # Centroid max-action L2 error for credential_access
            daily_cred_err[day] = max(
                np.linalg.norm(scorer.mu[cred_idx, a] - mu_true[cred_idx, a])
                for a in range(A)
            )

            # Pre-campaign baseline
            if day_num == 19:
                pre_camp_err = daily_cred_err[day]

            # Post-campaign recovery counting
            if post_campaign:
                post_camp_decs += cred_decs_day
                if recovered_at is None and pre_camp_err is not None:
                    if daily_cred_err[day] <= pre_camp_err + P5_EPS:
                        recovered_at = post_camp_decs

        err_day19_all.append(daily_cred_err[18])   # Day 19 = index 18
        err_day25_all.append(daily_cred_err[24])   # Day 25 = index 24
        err_day30_all.append(daily_cred_err[29])   # Day 30 = index 29
        sig_base_all.append(float(np.mean(daily_sig[0:19])))    # Days 1–19
        sig_camp_all.append(float(np.mean(daily_sig[19:30])))   # Days 20–30
        recovery_all.append(recovered_at if recovered_at is not None else -1)

    mean_base  = float(np.mean(sig_base_all))
    mean_camp  = float(np.mean(sig_camp_all))
    rel_fired  = mean_camp < REL_THRESHOLD * mean_base

    valid_rec  = [r for r in recovery_all if r >= 0]
    mean_rec   = float(np.mean(valid_rec)) if valid_rec else float("nan")

    return {
        "err_day19":                round(float(np.mean(err_day19_all)), 4),
        "err_day25":                round(float(np.mean(err_day25_all)), 4),
        "err_day30":                round(float(np.mean(err_day30_all)), 4),
        "peak_drift":               round(float(np.mean(err_day30_all)) - float(np.mean(err_day19_all)), 4),
        "sig_baseline_mean":        round(mean_base, 3),
        "sig_campaign_mean":        round(mean_camp, 3),
        "relative_threshold_fired": rel_fired,
        "recovery_decs_mean":       round(mean_rec, 1) if not math.isnan(mean_rec) else None,
        "recovery_within_gate":     (mean_rec <= RECOVERY_GATE) if not math.isnan(mean_rec) else False,
        "n_seeds_recovered":        len(valid_rec),
        "n_seeds_total":            N_2D_SEEDS,
    }


# ── CE2: Ransomware volume spike — starvation analysis ────────────────────────
def run_ce2_analysis(persona, mu_true, categories, cat_to_idx, gt_dists_arr,
                     factor_names):
    """
    Standard 60-day simulation (no correlated errors). Tracks per-category
    centroid max-action L2 error daily. Reports pre-spike vs during-spike for
    all categories, highlighting starved (low-volume) ones.
    """
    C, A, d    = mu_true.shape
    noise      = build_persona_noise(persona, factor_names)
    base_w     = build_persona_weights(persona, categories)
    shifts     = persona.get("environment_shifts", [])
    a_params   = precompute_analyst_params(persona["analyst_team"])
    n_analysts = len(a_params)
    apd        = persona["alerts_per_day"]
    cat_idx    = np.arange(C)
    n_days     = 60
    day_wts    = precompute_day_weights(base_w, categories, cat_to_idx, shifts, n_days)

    spike_shift  = persona["environment_shifts"][0]
    sd           = spike_shift["day"]             # 15
    dur          = spike_shift["duration_days"]   # 14
    cat_roles    = spike_shift["category_impact"]

    # Index arithmetic (0-indexed)
    pre_end   = sd - 1          # exclusive → Days 1..(sd-1)
    spike_s   = sd - 1          # inclusive start (Day sd = index sd-1)
    spike_e   = sd + dur - 1    # exclusive end  (Day sd+dur-1 → index sd+dur-2 included)
    # mean_daily[0:pre_end]  = Days 1..14  (14 items)
    # mean_daily[spike_s:spike_e] = Days 15..28 (14 items)

    all_cat_err = np.zeros((N_2D_SEEDS, n_days, C))

    for si in range(N_2D_SEEDS):
        rng    = np.random.RandomState(si + 4000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        scorer = AsymmetricScorer(
            np.clip(mu_true + offset, 0, 1),
            tau=P5_TAU, eta_confirm=ETA_CONFIRM, eta_override=ETA_OVERRIDE,
        )

        for day in range(n_days):
            dw       = day_wts[day]
            n_alerts = int(rng.poisson(apd))

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f       = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise, 0, 1)
                pred_a, _ = scorer.score(f, c)

                ai              = rng.randint(n_analysts)
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

            for ci in range(C):
                all_cat_err[si, day, ci] = max(
                    np.linalg.norm(scorer.mu[ci, a] - mu_true[ci, a])
                    for a in range(A)
                )

    mean_daily = np.mean(all_cat_err, axis=0)   # (n_days, C)
    pre_mean   = np.mean(mean_daily[0:pre_end],    axis=0)   # (C,)
    spike_mean = np.mean(mean_daily[spike_s:spike_e], axis=0)  # (C,)

    results = {}
    for ci, cat in enumerate(categories):
        mult = cat_roles.get(cat, 1.0)
        role = "spiking" if mult > 1.0 else ("starved" if mult < 1.0 else "neutral")
        results[cat] = {
            "pre_spike_err": round(float(pre_mean[ci]),   4),
            "spike_err":     round(float(spike_mean[ci]), 4),
            "delta":         round(float(spike_mean[ci] - pre_mean[ci]), 4),
            "role":          role,
            "volume_mult":   mult,
        }

    return {
        "categories":  results,
        "spike_start": sd,
        "spike_end":   sd + dur - 1,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_2d_analysis(ce1_res, ce2_res, categories):
    # ── CE1 ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("=== 2D-CE1: PHISHING CAMPAIGN ANALYSIS ===")
    print(f"    Campaign Days 20–30  |  corr_err_rate={CORR_ERR_RATE:.0%}  |  "
          f"N={ce1_res['n_seeds_total']} seeds")
    print("=" * 62)
    print()

    fired_str = "YES (AMBER)" if ce1_res["relative_threshold_fired"] else "no"
    rec_v     = ce1_res["recovery_decs_mean"]
    rec_str   = f"{rec_v:.1f}" if rec_v is not None else "N/A (not recovered)"
    gate_str  = "PASS" if ce1_res["recovery_within_gate"] else "FAIL"

    rows = [
        ("Pre-campaign cred_acc error (Day 19)",        f"{ce1_res['err_day19']:.4f}"),
        ("Mid-campaign cred_acc error (Day 25)",        f"{ce1_res['err_day25']:.4f}"),
        ("End-campaign cred_acc error (Day 30)",        f"{ce1_res['err_day30']:.4f}"),
        ("Peak drift magnitude (Day30 - Day19)",        f"{ce1_res['peak_drift']:+.4f}"),
        ("Conservation signal Days 1-19 (mean)",        f"{ce1_res['sig_baseline_mean']:.3f}"),
        ("Conservation signal Days 20-30 (mean)",       f"{ce1_res['sig_campaign_mean']:.3f}"),
        ("Relative threshold fired? (< 0.70x base)",    fired_str),
        ("Post-campaign recovery decisions (mean)",     rec_str),
        (f"Recovery within {RECOVERY_GATE} decisions?", gate_str),
    ]
    W = 44
    print(f"| {'Metric':{W}} | {'Value':>14} |")
    print("|" + "-" * (W + 2) + "|" + "-" * 16 + "|")
    for label, val in rows:
        print(f"| {label:{W}} | {val:>14} |")
    print()

    # ── CE2 ───────────────────────────────────────────────────────────────────
    sp   = ce2_res
    ss   = sp["spike_start"]
    se   = sp["spike_end"]
    print("=" * 62)
    print("=== 2D-CE2: RANSOMWARE VOLUME SPIKE ANALYSIS ===")
    print(f"    Spike Days {ss}–{se}  |  N={N_2D_SEEDS} seeds")
    print("=" * 62)
    print()

    hdr = (f"| {'Category':<22} | {'Role':8} | {'Pre-spike':9} | "
           f"{'Spike':9} | {'Delta':7} |")
    sep = "|" + "-"*24 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*9 + "|"
    print(hdr)
    print(sep)
    for cat in categories:
        cr    = sp["categories"][cat]
        dstr  = f"{cr['delta']:+.4f}"
        flag  = " << DEGRADE" if (cr["role"] == "starved" and cr["delta"] > 0.010) else ""
        print(f"| {cat:<22} | {cr['role']:8} | {cr['pre_spike_err']:9.4f} | "
              f"{cr['spike_err']:9.4f} | {dstr:7} |{flag}")
    print()


def print_gates_2d(ce1_res, ce2_res, all_results, personas, categories):
    print("=" * 62)
    print("=== GATE EVALUATION — 2D ===")
    print("=" * 62)

    # G1: Day60 >= Day1
    print("\nG1: Day60 accuracy >= Day1 accuracy:")
    all_g1 = True
    for p in personas:
        pid    = p["persona_id"]
        p5     = all_results[pid]["prod5"]
        passes = p5["acc_day60"] >= p5["acc_day1"]
        if not passes:
            all_g1 = False
        print(f"  {pid}: {p5['acc_day1']:.1%} → {p5['acc_day60']:.1%}  "
              f"[{'PASS' if passes else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_g1 else 'FAIL'}")

    # G2: Post-campaign recovery within 28 decisions
    rec_v = ce1_res["recovery_decs_mean"]
    rec_s = f"{rec_v:.1f}" if rec_v is not None else "N/A"
    g2    = ce1_res["recovery_within_gate"]
    print(f"\nG2: Post-campaign recovery ≤ {RECOVERY_GATE} decisions:")
    print(f"  Mean recovery = {rec_s} decisions  [{'PASS' if g2 else 'FAIL'}]")
    if not g2 and rec_v is not None:
        print(f"  FAIL: campaign drift exceeds η_override=0.01 attenuation capacity.")
        print(f"  Implication: source-tagged update rejection needed for campaign period.")

    # G3: Conservation monitor fires (relative threshold)
    base  = ce1_res["sig_baseline_mean"]
    camp  = ce1_res["sig_campaign_mean"]
    ratio = camp / base if base > 0 else float("nan")
    g3    = ce1_res["relative_threshold_fired"]
    print(f"\nG3: Conservation monitor detects campaign (< {REL_THRESHOLD:.0%}× baseline):")
    print(f"  Baseline signal = {base:.3f}  |  Campaign signal = {camp:.3f}  "
          f"|  Ratio = {ratio:.3f}")
    print(f"  Threshold: < {REL_THRESHOLD * base:.3f}  [{'PASS' if g3 else 'FAIL'}]")

    # G4: Non-spiking categories don't degrade
    starved = {cat: d for cat, d in ce2_res["categories"].items()
               if d["role"] == "starved"}
    g4      = all(d["delta"] <= 0.010 for d in starved.values())
    print(f"\nG4: Starved categories don't degrade (delta ≤ 0.010):")
    for cat, d in starved.items():
        mark = "OK" if d["delta"] <= 0.010 else "DEGRADE"
        print(f"  {cat}: pre={d['pre_spike_err']:.4f} → spike={d['spike_err']:.4f}  "
              f"Δ={d['delta']:+.4f}  [{mark}]")
    print(f"  Overall: {'PASS' if g4 else 'FAIL'}")

    print()
    all_gates = [all_g1, g2, g3, g4]
    n_pass    = sum(all_gates)
    print(f"Gates passed: {n_pass}/4  "
          f"({'ALL PASS' if n_pass == 4 else str(n_pass) + ' of 4'})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="2D Correlated Error Sweep")
    parser.add_argument("--personas", required=True)
    parser.add_argument("--output",   required=True)
    args = parser.parse_args()

    personas_path = Path(args.personas)
    output_dir    = Path(args.output)

    if not personas_path.exists():
        raise FileNotFoundError(f"Personas file not found: {personas_path}")

    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)

    cfg          = load_domain_config("soc_product_v50")
    categories   = cfg["categories"]
    factor_names = cfg["factors"]
    mu_true      = np.array(cfg["mu"], dtype=float)
    gt_dists_raw = cfg["gt_distributions"]
    cat_to_idx   = {c: i for i, c in enumerate(categories)}
    gt_dists_arr = np.array([gt_dists_raw[c] for c in categories], dtype=float)
    gt_dists_arr /= gt_dists_arr.sum(axis=1, keepdims=True)

    print()
    print("=" * 62)
    print("=== 2D CORRELATED ERROR SWEEP RESULTS ===")
    print("=" * 62)
    print(f"  Personas file: {personas_path}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Config:        soc_product_v50  "
          f"C={len(categories)} A={len(cfg['actions'])} d={mu_true.shape[2]}")
    print(f"  eta_confirm={ETA_CONFIRM}  eta_override={ETA_OVERRIDE}  "
          f"corr_err_rate={CORR_ERR_RATE:.0%}")
    print(f"  N personas: {len(personas)}")

    # ── Standard harness ──────────────────────────────────────────────────────
    all_results = {}
    t_total     = time.time()

    for persona in personas:
        pid      = persona["persona_id"]
        name     = persona["name"]
        industry = persona["industry"]
        apd      = persona["alerts_per_day"]
        n_an     = len(persona["analyst_team"])
        qb       = persona_q_bar(persona["analyst_team"])

        print(f"\n{'─'*62}")
        print(f"[{pid}] {name}")
        print(f"       {industry}  |  {apd} alerts/day  |  {n_an} analysts  |  q_bar={qb:.3f}")

        t0     = time.time()
        td_res = run_td034(persona, mu_true, categories, gt_dists_arr, factor_names)
        print(f"  TD-034 : optimal τ={td_res['optimal_tau']:.2f}  "
              f"ECE@opt={td_res['ece_at_opt']:.4f}  "
              f"ECE@0.10={td_res['ece_at_010']:.4f}  ({time.time()-t0:.1f}s)")

        t0     = time.time()
        p5_res = run_prod5(persona, mu_true, categories, cat_to_idx,
                           gt_dists_arr, factor_names)
        nc     = p5_res["n_cats_converged"]
        print(f"  PROD-5 : {nc}/6 cats converge  "
              f"acc {p5_res['acc_day1']:.0%}→{p5_res['acc_day60']:.0%}  "
              f"gate={p5_res['gate_stats']['override_pct']:.1%}  ({time.time()-t0:.1f}s)")

        t0     = time.time()
        ba_res = run_ba(persona)
        print(f"  B-A    : delta={ba_res['delta_reward']:+.4f}  "
              f"promoted={ba_res['promo_rate']*100:.0f}%  "
              f"CL breach={ba_res['breach_rate']*100:.0f}%  ({time.time()-t0:.1f}s)")

        all_results[pid] = {
            "persona_id": pid,
            "name":       name,
            "industry":   industry,
            "q_bar":      round(qb, 4),
            "td034":      td_res,
            "prod5":      p5_res,
            "ba":         ba_res,
        }

    print(f"\n{'='*62}")
    print_tables(all_results, personas, categories)

    print("\nSaving standard results...")
    save_results(all_results, personas, output_dir)

    # ── 2D specialized analyses ───────────────────────────────────────────────
    ce1_persona = next(p for p in personas if p["persona_id"] == "2D-CE1")
    ce2_persona = next(p for p in personas if p["persona_id"] == "2D-CE2")

    print(f"\n{'─'*62}")
    print(f"Running CE1 phishing campaign analysis  "
          f"(corr_err_rate={CORR_ERR_RATE:.0%}, N={N_2D_SEEDS} seeds)...")
    t0     = time.time()
    ce1_res = run_ce1_analysis(ce1_persona, mu_true, categories, cat_to_idx,
                               gt_dists_arr, factor_names)
    print(f"  Done ({time.time()-t0:.1f}s)")

    print(f"Running CE2 ransomware starvation analysis  "
          f"(N={N_2D_SEEDS} seeds)...")
    t0     = time.time()
    ce2_res = run_ce2_analysis(ce2_persona, mu_true, categories, cat_to_idx,
                               gt_dists_arr, factor_names)
    print(f"  Done ({time.time()-t0:.1f}s)")

    print_2d_analysis(ce1_res, ce2_res, categories)
    print_gates_2d(ce1_res, ce2_res, all_results, personas, categories)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "correlated_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump({"ce1": ce1_res, "ce2": ce2_res}, f, indent=2, default=str)
    print(f"\n  Saved → {analysis_path}")

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
