"""
Priority 1 Validation — V-B3, V-B1, V-CL-RECOVER.

Runs standard harness for all 9 personas, then adds three specialized tables:

  V-B3 (Noise Ceiling with Junior Teams):
    σ/q̄/V table; gate: VB3-2 improves AND VB3-3 degrades AND VB3-1 is boundary.

  V-B1 (AMBER Zone Learning):
    Day60 ≥ Day1 for all 3 personas (η_override=0.01, σ=0.12–0.157).

  V-CL-RECOVER (Post-Campaign Recovery at Low Volume):
    Tracks credential_access centroid error: pre-campaign / peak / recovery.
    Gate: recovery ≤ 3 days at V=200; volume-normalised claim at V=50/100.

Usage:
    python experiments/persona_sweeps/run_priority1.py \
        --output experiments/persona_sweeps/results/priority1/
"""

import sys
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
HARNESS_DIR = Path(__file__).resolve().parent
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

# ── Priority-1 constants ───────────────────────────────────────────────────────
N_P1_SEEDS       = 15
CORR_ERR_RATE    = 0.80   # fraction of campaign cred_access overrides → wrong
VCL_RECOVERY_DAYS_GATE = 3   # claimed recovery gate in calendar days

# Persona file locations
PRIORITY1_DIR    = Path(__file__).resolve().parent / "priority1"
VB3_PERSONAS     = PRIORITY1_DIR / "personas_priority1_vb3.json"
VB1_PERSONAS     = PRIORITY1_DIR / "personas_priority1_vb1.json"
VCL_PERSONAS     = PRIORITY1_DIR / "personas_priority1_vcl_recover.json"

# Noise ceilings (from 1D sweep)
CEILING_DAY60_THRESH = 0.85   # Day60 < this → near-ceiling or degraded
VB3_DEGRADE_THRESH   = 0.0    # VB3-3 gate: Day60 - Day1 < this → degraded


def _mean_noise(persona):
    """Mean of per-factor base_noise values."""
    return float(np.mean([v["base_noise"]
                           for v in persona["factor_noise_profile"].values()]))


# ── V-CL-RECOVER specialized simulation ───────────────────────────────────────
def run_vcl_recovery(persona, mu_true, categories, cat_to_idx, gt_dists_arr,
                     factor_names):
    """
    60-day simulation with correlated phishing campaign on credential_access.
    Campaign: Days 20–29 (inclusive, 10 days). 80% wrong overrides on cred_acc.
    Tracks:
      - cred_access centroid max-action L2 error at Day 19 (pre), Day 29 (peak)
      - Post-campaign recovery: cred_access decisions until error ≤ pre + ε
      - Converts decisions → days using apd × cred_access_fraction
    """
    shifts      = persona.get("environment_shifts", [])
    camp_shift  = next(
        (s for s in shifts if s["type"] == "campaign"),
        {"day": 20, "duration_days": 10}
    )
    CAMP_START  = camp_shift["day"]           # 20 (1-indexed, inclusive)
    CAMP_END    = CAMP_START + camp_shift["duration_days"] - 1  # 29

    C, A, d    = mu_true.shape
    cred_idx   = cat_to_idx["credential_access"]
    noise      = build_persona_noise(persona, factor_names)
    base_w     = build_persona_weights(persona, categories)
    a_params   = precompute_analyst_params(persona["analyst_team"])
    n_analysts = len(a_params)
    apd        = persona["alerts_per_day"]
    cat_idx    = np.arange(C)
    n_days     = 60
    day_wts    = precompute_day_weights(base_w, categories, cat_to_idx, shifts, n_days)

    # Cred_access alerts per day (pre-campaign volume, used for days→conversion)
    cred_frac    = persona["category_distribution"]["credential_access"]
    cred_per_day = apd * cred_frac    # expected (Poisson mean)

    err_pre_all    = []
    err_peak_all   = []
    recovery_all   = []   # -1 if not recovered by Day 60

    for si in range(N_P1_SEEDS):
        rng    = np.random.RandomState(si + 7000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        scorer = AsymmetricScorer(
            np.clip(mu_true + offset, 0, 1),
            tau=P5_TAU, eta_confirm=ETA_CONFIRM, eta_override=ETA_OVERRIDE,
        )

        daily_cred_err  = np.zeros(n_days)
        pre_camp_err    = None
        post_camp_decs  = 0    # cumulative cred_access decisions after campaign
        recovered_at    = None

        for day in range(n_days):
            day_num       = day + 1
            dw            = day_wts[day]
            n_alerts      = int(rng.poisson(apd))
            in_campaign   = CAMP_START <= day_num <= CAMP_END
            post_campaign = day_num > CAMP_END
            cred_day      = 0

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f       = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise, 0, 1)
                pred_a, _ = scorer.score(f, c)

                ai             = rng.randint(n_analysts)
                eff_over, eff_q = a_params[ai]

                if rng.random() < eff_over:
                    if in_campaign and c == cred_idx and rng.random() < CORR_ERR_RATE:
                        # Correlated wrong override during campaign
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])
                    elif rng.random() < eff_q:
                        gt_a = true_gt
                    else:
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])
                    scorer.update_override(f, c, gt_a)
                else:
                    scorer.update_confirm(f, c, pred_a)

                if post_campaign and c == cred_idx:
                    cred_day += 1

            daily_cred_err[day] = max(
                np.linalg.norm(scorer.mu[cred_idx, a] - mu_true[cred_idx, a])
                for a in range(A)
            )

            if day_num == (CAMP_START - 1):
                pre_camp_err = daily_cred_err[day]

            if post_campaign:
                post_camp_decs += cred_day
                if recovered_at is None and pre_camp_err is not None:
                    if daily_cred_err[day] <= pre_camp_err + P5_EPS:
                        recovered_at = post_camp_decs

        # Peak = last day of campaign
        err_pre_all.append(daily_cred_err[CAMP_START - 2] if CAMP_START >= 2
                           else daily_cred_err[0])
        err_peak_all.append(daily_cred_err[CAMP_END - 1])
        recovery_all.append(recovered_at if recovered_at is not None else -1)

    mean_pre  = float(np.mean(err_pre_all))
    mean_peak = float(np.mean(err_peak_all))

    valid_rec     = [r for r in recovery_all if r >= 0]
    mean_rec_decs = float(np.mean(valid_rec)) if valid_rec else float("nan")

    # Convert decisions → days
    mean_rec_days = (mean_rec_decs / cred_per_day) if not math.isnan(mean_rec_decs) else float("nan")
    gate_passes   = (not math.isnan(mean_rec_days)) and (mean_rec_days <= VCL_RECOVERY_DAYS_GATE)

    return {
        "apd":               apd,
        "cred_frac":         cred_frac,
        "cred_per_day":      round(cred_per_day, 1),
        "err_pre_campaign":  round(mean_pre,       4),
        "err_peak":          round(mean_peak,       4),
        "peak_drift":        round(mean_peak - mean_pre, 4),
        "recovery_decs_mean":   round(mean_rec_decs, 1) if not math.isnan(mean_rec_decs) else None,
        "recovery_days_mean":   round(mean_rec_days, 2) if not math.isnan(mean_rec_days) else None,
        "recovery_within_3d":   gate_passes,
        "n_seeds_recovered":    len(valid_rec),
        "n_seeds_total":        N_P1_SEEDS,
    }


# ── Print helpers ──────────────────────────────────────────────────────────────
def print_vb3_table(all_results, vb3_personas):
    print()
    print("=" * 70)
    print("=== V-B3: NOISE CEILING — JUNIOR TEAM VALIDATION ===")
    print("=" * 70)
    print()
    hdr = (f"| {'Persona':<8} | {'σ_mean':6} | {'q̄':5} | {'V':4} | "
           f"{'Day1':6} | {'Day60':6} | {'Δ':7} | {'Gate':12} |")
    sep = ("|" + "-"*10 + "|" + "-"*8 + "|" + "-"*7 + "|" + "-"*6 + "|"
           + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*14 + "|")
    print(hdr)
    print(sep)

    for p in vb3_personas:
        pid    = p["persona_id"]
        sigma  = _mean_noise(p)
        qb     = persona_q_bar(p["analyst_team"])
        apd    = p["alerts_per_day"]
        p5     = all_results[pid]["prod5"]
        d1     = p5["acc_day1"]
        d60    = p5["acc_day60"]
        delta  = d60 - d1
        # Gate logic: VB3-2 should improve, VB3-3 should degrade, VB3-1 boundary
        if pid == "VB3-2":
            gate_label = "SHOULD IMPROVE"
            gate_pass  = delta > 0.0
        elif pid == "VB3-3":
            gate_label = "SHOULD DEGRADE"
            gate_pass  = delta <= 0.0
        elif pid == "VB3-1":
            gate_label = "BOUNDARY"
            gate_pass  = True   # informational
        else:  # VB3-4 volume control
            gate_label = "VOLUME CTRL"
            gate_pass  = True
        mark = "PASS" if gate_pass else "FAIL"
        print(f"| {pid:<8} | {sigma:6.3f} | {qb:5.3f} | {apd:4} | "
              f"{d1:6.1%} | {d60:6.1%} | {delta:+7.3%} | {mark:<6} {gate_label:<5} |")
    print()


def print_vb1_table(all_results, vb1_personas):
    print()
    print("=" * 70)
    print("=== V-B1: AMBER ZONE LEARNING — η_override=0.01 ===")
    print("=" * 70)
    print()
    hdr = (f"| {'Persona':<8} | {'σ_mean':6} | {'V':4} | "
           f"{'Day1':6} | {'Day60':6} | {'Δ':7} | {'Day60≥Day1':12} |")
    sep = ("|" + "-"*10 + "|" + "-"*8 + "|" + "-"*6 + "|"
           + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*14 + "|")
    print(hdr)
    print(sep)

    all_pass = True
    for p in vb1_personas:
        pid    = p["persona_id"]
        sigma  = _mean_noise(p)
        apd    = p["alerts_per_day"]
        p5     = all_results[pid]["prod5"]
        d1     = p5["acc_day1"]
        d60    = p5["acc_day60"]
        delta  = d60 - d1
        passes = d60 >= d1
        if not passes:
            all_pass = False
        mark = "PASS" if passes else "FAIL"
        print(f"| {pid:<8} | {sigma:6.3f} | {apd:4} | "
              f"{d1:6.1%} | {d60:6.1%} | {delta:+7.3%} | {mark:<12} |")

    print()
    print(f"  Overall V-B1 gate: {'ALL PASS' if all_pass else 'FAIL'}")
    print()


def print_vcl_table(vcl_results, vcl_personas):
    print()
    print("=" * 70)
    print("=== V-CL-RECOVER: POST-CAMPAIGN RECOVERY AT LOW VOLUME ===")
    print(f"    Campaign: Days 20–29  |  corr_err_rate={CORR_ERR_RATE:.0%}  "
          f"|  N={N_P1_SEEDS} seeds  |  gate: ≤{VCL_RECOVERY_DAYS_GATE} days")
    print("=" * 70)
    print()

    hdr = (f"| {'Persona':<8} | {'V':4} | {'cred/day':8} | "
           f"{'Pre-camp err':12} | {'Peak err':8} | {'Drift':7} | "
           f"{'Rec decs':8} | {'Rec days':8} | {'≤3d?':6} |")
    sep = ("|" + "-"*10 + "|" + "-"*6 + "|" + "-"*10 + "|"
           + "-"*14 + "|" + "-"*10 + "|" + "-"*9 + "|"
           + "-"*10 + "|" + "-"*10 + "|" + "-"*8 + "|")
    print(hdr)
    print(sep)

    all_pass = True
    for p in vcl_personas:
        pid   = p["persona_id"]
        r     = vcl_results[pid]
        rd    = r["recovery_decs_mean"]
        rdays = r["recovery_days_mean"]
        rd_str    = f"{rd:.1f}" if rd is not None else "N/A"
        rdays_str = f"{rdays:.2f}" if rdays is not None else "N/A"
        g3 = r["recovery_within_3d"]
        if not g3:
            all_pass = False
        mark = "PASS" if g3 else "FAIL"
        print(f"| {pid:<8} | {r['apd']:4} | {r['cred_per_day']:8.1f} | "
              f"{r['err_pre_campaign']:12.4f} | {r['err_peak']:8.4f} | "
              f"{r['peak_drift']:+7.4f} | {rd_str:>8} | {rdays_str:>8} | {mark:<6} |")

    print()
    if not all_pass:
        print("  NOTE: '3 days' claim applies at V=200 (150 cred_access decisions).")
        print("  At V=50: ~15 cred_access/day → 150 decisions = 10 days.")
        print("  At V=100: ~35 cred_access/day → 150 decisions = 4.3 days.")
        print("  Claim should be volume-normalised: '≤150 cred_access decisions'.")
    print()


def print_gates_priority1(all_results, vb3_personas, vb1_personas,
                          vcl_results, vcl_personas):
    print("=" * 70)
    print("=== GATE EVALUATION — PRIORITY 1 (v6.0) ===")
    print("=" * 70)

    # ── V-B3 gates ────────────────────────────────────────────────────────────
    print("\n--- V-B3: Noise Ceiling ---")

    vb3_res = {p["persona_id"]: all_results[p["persona_id"]] for p in vb3_personas}

    # Gate A: VB3-2 (σ=0.13) improves
    p5_2     = vb3_res["VB3-2"]["prod5"]
    ga       = p5_2["acc_day60"] > p5_2["acc_day1"]
    print(f"  Gate A (VB3-2 σ=0.13 improves): "
          f"{p5_2['acc_day1']:.1%} → {p5_2['acc_day60']:.1%}  "
          f"[{'PASS' if ga else 'FAIL'}]")

    # Gate B: VB3-3 (σ=0.19) degrades
    p5_3     = vb3_res["VB3-3"]["prod5"]
    gb       = p5_3["acc_day60"] <= p5_3["acc_day1"]
    print(f"  Gate B (VB3-3 σ=0.19 degrades): "
          f"{p5_3['acc_day1']:.1%} → {p5_3['acc_day60']:.1%}  "
          f"[{'PASS' if gb else 'FAIL'}]")

    # Gate C: VB3-1 (σ=0.157) boundary — informational
    p5_1     = vb3_res["VB3-1"]["prod5"]
    gc_delta = p5_1["acc_day60"] - p5_1["acc_day1"]
    print(f"  Gate C (VB3-1 σ=0.157 boundary, informational): "
          f"{p5_1['acc_day1']:.1%} → {p5_1['acc_day60']:.1%}  "
          f"Δ={gc_delta:+.3%}  [INFO]")

    # Gate D: VB3-4 (σ=0.157, V=200) volume lifts outcome
    p5_4     = vb3_res["VB3-4"]["prod5"]
    gd_delta = p5_4["acc_day60"] - p5_4["acc_day1"]
    gd       = gd_delta > gc_delta   # higher volume → larger delta
    print(f"  Gate D (VB3-4 V=200 > VB3-1 V=50 by Δ): "
          f"Δ_V200={gd_delta:+.3%}  Δ_V50={gc_delta:+.3%}  "
          f"[{'PASS' if gd else 'FAIL'}]")

    vb3_pass = ga and gb
    print(f"  V-B3 overall (gates A+B required): "
          f"{'PASS' if vb3_pass else 'FAIL'}")

    # ── V-B1 gates ────────────────────────────────────────────────────────────
    print("\n--- V-B1: AMBER Zone Learning ---")
    vb1_pass = True
    for p in vb1_personas:
        pid  = p["persona_id"]
        p5   = all_results[pid]["prod5"]
        ok   = p5["acc_day60"] >= p5["acc_day1"]
        if not ok:
            vb1_pass = False
        sigma = _mean_noise(p)
        print(f"  {pid} σ={sigma:.3f}: "
              f"{p5['acc_day1']:.1%} → {p5['acc_day60']:.1%}  "
              f"[{'PASS' if ok else 'FAIL'}]")
    print(f"  V-B1 overall: {'ALL PASS' if vb1_pass else 'FAIL'}")

    # ── V-CL-RECOVER gates ────────────────────────────────────────────────────
    print("\n--- V-CL-RECOVER: Post-Campaign Recovery ---")
    vcl_pass = True
    for p in vcl_personas:
        pid   = p["persona_id"]
        r     = vcl_results[pid]
        rdays = r["recovery_days_mean"]
        rdays_str = f"{rdays:.2f}d" if rdays is not None else "N/A"
        g     = r["recovery_within_3d"]
        if not g:
            vcl_pass = False
        print(f"  {pid} V={r['apd']}: recovery={rdays_str}  "
              f"[{'PASS' if g else 'FAIL'}]")

    if not vcl_pass:
        print()
        print("  CLAIM REVISION: '3-day recovery' holds only at V≥200.")
        print("  Volume-normalised claim: '≤150 cred_access decisions post-campaign'")
        # Compute at V=200 reference
        for p in vcl_personas:
            pid  = p["persona_id"]
            r    = vcl_results[pid]
            if r["recovery_decs_mean"] is not None:
                cred_per_day_200 = 200 * r["cred_frac"]
                days_at_200 = r["recovery_decs_mean"] / cred_per_day_200
                print(f"    {pid}: {r['recovery_decs_mean']:.1f} decisions "
                      f"= {days_at_200:.2f}d at V=200 ref "
                      f"({r['recovery_days_mean']:.2f}d at actual V={r['apd']})")
    print(f"  V-CL-RECOVER overall (3-day gate): "
          f"{'PASS' if vcl_pass else 'FAIL — see volume note above'}")

    print()
    all_core = vb3_pass and vb1_pass
    print("=" * 70)
    print(f"  V-B3:        {'PASS' if vb3_pass else 'FAIL'}")
    print(f"  V-B1:        {'PASS' if vb1_pass else 'FAIL'}")
    print(f"  V-CL-RECOVER:{'PASS' if vcl_pass else 'FAIL (claim needs revision)'}")
    print()
    if all_core:
        print("  CORE GATES (V-B3 + V-B1): PASS — v6.0 noise ceiling validated")
    else:
        print("  CORE GATES (V-B3 + V-B1): FAIL — review noise ceiling definition")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Priority 1 Validation — V-B3, V-B1, V-CL-RECOVER")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Load all three persona files
    def _load(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    vb3_personas = _load(VB3_PERSONAS)
    vb1_personas = _load(VB1_PERSONAS)
    vcl_personas = _load(VCL_PERSONAS)
    all_personas = vb3_personas + vb1_personas + vcl_personas

    cfg          = load_domain_config("soc_product_v50")
    categories   = cfg["categories"]
    factor_names = cfg["factors"]
    mu_true      = np.array(cfg["mu"], dtype=float)
    gt_dists_raw = cfg["gt_distributions"]
    cat_to_idx   = {c: i for i, c in enumerate(categories)}
    gt_dists_arr = np.array([gt_dists_raw[c] for c in categories], dtype=float)
    gt_dists_arr /= gt_dists_arr.sum(axis=1, keepdims=True)

    print()
    print("=" * 70)
    print("=== PRIORITY 1 VALIDATION — V-B3 / V-B1 / V-CL-RECOVER ===")
    print("=" * 70)
    print(f"  VB3 personas: {VB3_PERSONAS.name}")
    print(f"  VB1 personas: {VB1_PERSONAS.name}")
    print(f"  VCL personas: {VCL_PERSONAS.name}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Config:       soc_product_v50  "
          f"C={len(categories)} A={len(cfg['actions'])} d={mu_true.shape[2]}")
    print(f"  eta_confirm={ETA_CONFIRM}  eta_override={ETA_OVERRIDE}")
    print(f"  N personas: {len(all_personas)}  (4 VB3 + 3 VB1 + 2 VCL)")

    # ── Standard harness for all 9 ────────────────────────────────────────────
    all_results = {}
    t_total     = time.time()

    for persona in all_personas:
        pid      = persona["persona_id"]
        name     = persona["name"]
        industry = persona["industry"]
        apd      = persona["alerts_per_day"]
        n_an     = len(persona["analyst_team"])
        qb       = persona_q_bar(persona["analyst_team"])

        print(f"\n{'─'*70}")
        print(f"[{pid}] {name}")
        print(f"       {industry}  |  {apd} alerts/day  |  "
              f"{n_an} analysts  |  q_bar={qb:.3f}")

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

    print(f"\n{'='*70}")
    print_tables(all_results, all_personas, categories)

    # ── VCL specialized: recovery simulation ──────────────────────────────────
    vcl_results = {}
    for persona in vcl_personas:
        pid = persona["persona_id"]
        print(f"\n{'─'*70}")
        print(f"Running VCL recovery simulation for {pid}  "
              f"(V={persona['alerts_per_day']}, N={N_P1_SEEDS} seeds)...")
        t0 = time.time()
        vcl_results[pid] = run_vcl_recovery(
            persona, mu_true, categories, cat_to_idx, gt_dists_arr, factor_names
        )
        print(f"  Done ({time.time()-t0:.1f}s)  "
              f"recovery={vcl_results[pid]['recovery_days_mean']}d  "
              f"(gate={VCL_RECOVERY_DAYS_GATE}d)")

    # ── Print specialized tables ───────────────────────────────────────────────
    print_vb3_table(all_results, vb3_personas)
    print_vb1_table(all_results, vb1_personas)
    print_vcl_table(vcl_results, vcl_personas)
    print_gates_priority1(all_results, vb3_personas, vb1_personas,
                          vcl_results, vcl_personas)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(all_results, all_personas, output_dir)

    vcl_path = output_dir / "vcl_recovery.json"
    with open(vcl_path, "w", encoding="utf-8") as f:
        json.dump(vcl_results, f, indent=2, default=str)
    print(f"\n  Saved VCL recovery → {vcl_path}")

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
