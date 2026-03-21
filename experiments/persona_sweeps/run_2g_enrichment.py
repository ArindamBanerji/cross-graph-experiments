"""
2G Enrichment Shock Sweep — standard harness + enrichment-aware simulation.

The standard harness (run_harness.py) uses FIXED factor noise throughout
the simulation. 2G personas have a mid-pilot noise DROP when a second data
source connects (Sentinel, Defender for Endpoint). This script:

  1. Runs the standard harness for all standard tables.
  2. Runs a specialized 2G simulation that switches factor noise at
     the enrichment day and tracks:
       - Pre/post enrichment accuracy (mean accuracy before vs after)
       - Per-category centroid error at Day enrich-1 and Day enrich+5
       - Convergence rate (error-per-day) before vs after enrichment

Post-enrichment noise is extracted from the persona descriptions and
encoded in POST_ENRICHMENT_NOISE below.

Usage:
    python experiments/persona_sweeps/run_2g_enrichment.py \
        --personas experiments/persona_sweeps/personas_sweep_2g_enrichment.json \
        --output   experiments/persona_sweeps/results/sweep_2g_enrichment/
"""

import sys
import json
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

N_2G_SEEDS  = 15
DIP_GATE    = 0.03    # max acceptable accuracy dip at enrichment boundary

# ── Post-enrichment factor noise (extracted from description text) ─────────────
# Factor order in soc_product_v50: travel_match, asset_criticality,
#   threat_intel_enrichment, time_anomaly, pattern_history, device_trust
# Only changed factors are listed; unchanged take persona base_noise value.
POST_ENRICHMENT_NOISE = {
    "2G-GE1": {
        # Sentinel added at Day 45: Defender TI feeds + Defender for Endpoint
        "threat_intel_enrichment": 0.08,   # was 0.15
        "device_trust":            0.12,   # was 0.20
    },
    "2G-GE2": {
        # Defender for Endpoint at Day 30: real-time device health + patch data
        "asset_criticality": 0.13,         # was 0.22
        "device_trust":      0.10,         # was 0.28
    },
}

# Post-enrichment alerts/day (GE1 volume increases 180→250; GE2 unchanged)
POST_ENRICHMENT_APD = {
    "2G-GE1": 250,
    "2G-GE2": 160,
}


# ── Build noise arrays ─────────────────────────────────────────────────────────
def build_post_noise(persona: dict, factor_names: list, pid: str) -> np.ndarray:
    """Return post-enrichment noise array; unchanged factors keep pre value."""
    overrides = POST_ENRICHMENT_NOISE.get(pid, {})
    return np.array([
        overrides.get(f, persona["factor_noise_profile"][f]["base_noise"])
        for f in factor_names
    ])


# ── 2G specialized simulation ─────────────────────────────────────────────────
def run_2g_analysis(persona, mu_true, categories, cat_to_idx, gt_dists_arr,
                    factor_names, n_seeds=N_2G_SEEDS):
    """
    60-day simulation with factor noise switch at persona's enrichment day.
    Tracks daily accuracy and per-category centroid error throughout.
    """
    C, A, d    = mu_true.shape
    pid        = persona["persona_id"]
    enrich_day = persona["environment_shifts"][0]["day"]   # 1-indexed
    enrich_idx = enrich_day - 1                            # 0-indexed

    pre_noise  = build_persona_noise(persona, factor_names)
    post_noise = build_post_noise(persona, factor_names, pid)
    pre_apd    = persona["alerts_per_day"]
    post_apd   = POST_ENRICHMENT_APD.get(pid, pre_apd)

    base_w     = build_persona_weights(persona, categories)
    shifts     = persona.get("environment_shifts", [])
    a_params   = precompute_analyst_params(persona["analyst_team"])
    n_analysts = len(a_params)
    cat_idx    = np.arange(C)
    n_days     = 60
    day_wts    = precompute_day_weights(base_w, categories, cat_to_idx, shifts, n_days)

    all_daily_acc = np.full((n_seeds, n_days), np.nan)
    all_cat_err   = np.zeros((n_seeds, n_days, C))

    for si in range(n_seeds):
        rng    = np.random.RandomState(si + 5000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        scorer = AsymmetricScorer(
            np.clip(mu_true + offset, 0, 1),
            tau=P5_TAU, eta_confirm=ETA_CONFIRM, eta_override=ETA_OVERRIDE,
        )

        for day in range(n_days):
            day_num    = day + 1
            post       = day_num >= enrich_day
            noise      = post_noise if post else pre_noise
            apd        = post_apd   if post else pre_apd
            dw         = day_wts[day]
            n_alerts   = int(rng.poisson(apd))
            n_correct  = 0

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f       = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise, 0, 1)
                pred_a, _ = scorer.score(f, c)
                n_correct += int(pred_a == true_gt)

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

            if n_alerts > 0:
                all_daily_acc[si, day] = n_correct / n_alerts

            for ci in range(C):
                all_cat_err[si, day, ci] = max(
                    np.linalg.norm(scorer.mu[ci, a] - mu_true[ci, a])
                    for a in range(A)
                )

    mean_acc     = np.nanmean(all_daily_acc, axis=0)   # (n_days,)
    mean_cat_err = np.mean(all_cat_err, axis=0)         # (n_days, C)

    # ── Pre/post enrichment accuracy ─────────────────────────────────────────
    pre_window  = mean_acc[0:enrich_idx]
    post_window = mean_acc[enrich_idx:]
    pre_acc     = float(np.mean(pre_window))  if len(pre_window)  > 0 else float("nan")
    post_acc    = float(np.mean(post_window)) if len(post_window) > 0 else float("nan")

    # ── Accuracy dip at enrichment boundary (window: enrich-3 to enrich+5) ──
    w_s       = max(0, enrich_idx - 3)
    w_e       = min(n_days, enrich_idx + 5)
    ref_acc   = float(np.mean(mean_acc[max(0, enrich_idx - 5):enrich_idx])) \
                if enrich_idx > 0 else pre_acc
    dip_min   = float(np.min(mean_acc[w_s:w_e])) if w_e > w_s else ref_acc
    dip_mag   = max(0.0, ref_acc - dip_min)

    # ── Per-category error at boundary ───────────────────────────────────────
    # pre_day:  day before enrichment  (index enrich_idx - 1)
    # post_day: 5 days after enrichment (index enrich_idx + 4)
    pre_err_idx  = max(0, enrich_idx - 1)
    post_err_idx = min(n_days - 1, enrich_idx + 4)
    boundary_pre  = {categories[ci]: round(float(mean_cat_err[pre_err_idx,  ci]), 4)
                     for ci in range(C)}
    boundary_post = {categories[ci]: round(float(mean_cat_err[post_err_idx, ci]), 4)
                     for ci in range(C)}

    # ── Convergence rate: slope of mean-across-categories error vs day ───────
    # More negative slope = faster convergence; rate = -slope > 0 is improving
    mean_err_by_day = np.mean(mean_cat_err, axis=1)  # (n_days,)

    def slope_rate(segment):
        """Return convergence rate (-slope) over segment (array of errors)."""
        if len(segment) < 2:
            return 0.0
        xs = np.arange(len(segment), dtype=float)
        return float(-np.polyfit(xs, segment, 1)[0])

    pre_rate  = slope_rate(mean_err_by_day[0:enrich_idx])
    post_rate = slope_rate(mean_err_by_day[enrich_idx:])

    return {
        "enrich_day":         enrich_day,
        "pre_noise_mean":     round(float(np.mean(pre_noise)),  4),
        "post_noise_mean":    round(float(np.mean(post_noise)), 4),
        "noise_drop":         round(float(np.mean(pre_noise) - np.mean(post_noise)), 4),
        "pre_apd":            pre_apd,
        "post_apd":           post_apd,
        "pre_acc":            round(pre_acc,  4),
        "post_acc":           round(post_acc, 4),
        "delta_acc":          round(post_acc - pre_acc, 4),
        "dip_magnitude":      round(dip_mag, 4),
        "dip_within_gate":    dip_mag <= DIP_GATE,
        "boundary_pre":       boundary_pre,
        "boundary_post":      boundary_post,
        "pre_rate":           round(pre_rate,  6),
        "post_rate":          round(post_rate, 6),
        "rate_accelerated":   post_rate > pre_rate,
        "mean_daily_acc":     mean_acc.tolist(),
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_2g_analysis(results_map: dict, personas: list, categories: list):
    """results_map: {pid: 2g_result_dict}"""

    print()
    print("=" * 66)
    print("=== 2G ENRICHMENT SHOCK ANALYSIS ===")
    print("=" * 66)

    # ── Table 1: Pre/post enrichment accuracy ─────────────────────────────
    print()
    print("Table 1: Pre/Post Enrichment Accuracy")
    print("  (simulation with noise-aware switching at enrichment day)")
    hdr = (f"| {'Persona':8} | {'Enrich Day':10} | {'σ pre':7} | {'σ post':7} | "
           f"{'APD pre':7} | {'APD post':8} | {'Pre Acc':8} | {'Post Acc':9} | {'Delta':7} | Verdict |")
    sep = ("|" + "-"*10 + "|" + "-"*12 + "|" + "-"*9 + "|" + "-"*9 + "|"
           + "-"*9 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*9 + "|" + "-"*9 + "|")
    print(hdr); print(sep)

    for persona in personas:
        pid = persona["persona_id"]
        r   = results_map[pid]
        delta_str = f"{r['delta_acc']:+.1%}"
        verdict = "JUMP" if r["delta_acc"] > 0.02 else \
                  ("DIP" if r["delta_acc"] < -0.02 else "stable")
        print(f"| {pid:8} | Day {r['enrich_day']:>5}      | {r['pre_noise_mean']:7.3f} | "
              f"{r['post_noise_mean']:7.3f} | {r['pre_apd']:>7} | {r['post_apd']:>8} | "
              f"{r['pre_acc']:7.1%} | {r['post_acc']:8.1%}  | {delta_str:7} | {verdict:7} |")

    # ── Table 2: Per-category error at enrichment boundary ────────────────
    print()
    print("Table 2: Per-Category Centroid Error at Enrichment Boundary")
    print(f"  (pre = Day enrich-1, post = Day enrich+5)")

    for persona in personas:
        pid = persona["persona_id"]
        r   = results_map[pid]
        print(f"\n  {pid} — enrichment at Day {r['enrich_day']}  "
              f"(noise {r['pre_noise_mean']:.3f} → {r['post_noise_mean']:.3f})")
        hdr = (f"  | {'Category':<22} | {'Pre err':8} | {'Post err':9} | "
               f"{'Delta':8} | Benefit? |")
        sep2 = ("  |" + "-"*24 + "|" + "-"*10 + "|" + "-"*11 + "|"
                + "-"*10 + "|" + "-"*10 + "|")
        print(hdr); print(sep2)
        for cat in categories:
            pre_e  = r["boundary_pre"][cat]
            post_e = r["boundary_post"][cat]
            delta  = post_e - pre_e
            flag   = "YES" if delta < -0.005 else ("~" if abs(delta) <= 0.005 else "harm")
            print(f"  | {cat:<22} | {pre_e:8.4f} | {post_e:9.4f} | "
                  f"{delta:+8.4f} | {flag:8} |")

    # ── Table 3: Convergence rate ─────────────────────────────────────────
    print()
    print("Table 3: Convergence Rate Before vs After Enrichment")
    print("  (rate = mean error reduction per day across all categories;")
    print("   higher = faster convergence)")
    hdr = (f"| {'Persona':8} | {'Pre rate':10} | {'Post rate':10} | "
           f"{'Δ rate':8} | {'Accelerated?':14} |")
    sep = ("|" + "-"*10 + "|" + "-"*12 + "|" + "-"*12 + "|"
           + "-"*10 + "|" + "-"*16 + "|")
    print(hdr); print(sep)
    for persona in personas:
        pid = persona["persona_id"]
        r   = results_map[pid]
        pr  = r["pre_rate"]
        po  = r["post_rate"]
        acc = "YES" if r["rate_accelerated"] else "no"
        print(f"| {pid:8} | {pr:10.5f}  | {po:10.5f}  | "
              f"{po-pr:+8.5f} | {acc:14} |")


def print_2g_gates(all_results: dict, g_results: dict, personas: list):
    print()
    print("=" * 66)
    print("=== GATE EVALUATION — 2G ===")
    print("=" * 66)

    # G1: Day60 >= Day1 (standard harness)
    print("\nG1: Day60 accuracy >= Day1 accuracy (standard harness):")
    g1_all = True
    for persona in personas:
        pid    = persona["persona_id"]
        p5     = all_results[pid]["prod5"]
        passes = p5["acc_day60"] >= p5["acc_day1"]
        if not passes:
            g1_all = False
        print(f"  {pid}: {p5['acc_day1']:.1%} → {p5['acc_day60']:.1%}  "
              f"[{'PASS' if passes else 'FAIL'}]")
    print(f"  Overall: {'PASS' if g1_all else 'FAIL'}")

    # G2: No accuracy dip > 3pp at enrichment boundary
    print(f"\nG2: No accuracy dip > {DIP_GATE:.0%} at enrichment boundary:")
    g2_all = True
    for persona in personas:
        pid    = persona["persona_id"]
        r      = g_results[pid]
        passes = r["dip_within_gate"]
        if not passes:
            g2_all = False
        print(f"  {pid}: dip={r['dip_magnitude']:.2%}  "
              f"[{'PASS' if passes else 'FAIL'}]")
    print(f"  Overall: {'PASS' if g2_all else 'FAIL'}")

    # G3: Post-enrichment convergence rate >= pre-enrichment rate
    print("\nG3: Post-enrichment convergence rate >= pre-enrichment rate:")
    g3_all = True
    for persona in personas:
        pid    = persona["persona_id"]
        r      = g_results[pid]
        passes = r["rate_accelerated"]
        if not passes:
            g3_all = False
        print(f"  {pid}: pre={r['pre_rate']:.5f}  post={r['post_rate']:.5f}  "
              f"[{'PASS' if passes else 'FAIL'}]")
    print(f"  Overall: {'PASS' if g3_all else 'FAIL'}")

    n_pass = sum([g1_all, g2_all, g3_all])
    print(f"\nGates passed: {n_pass}/3  ({'ALL PASS' if n_pass == 3 else str(n_pass)+' of 3'})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="2G Enrichment Shock Sweep")
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

    fname = personas_path.stem
    print()
    print("=" * 66)
    print(f"=== SWEEP RESULTS: {fname} ===")
    print("=" * 66)
    print(f"  Personas file: {personas_path}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Config:        soc_product_v50  "
          f"C={len(categories)} A={len(cfg['actions'])} d={mu_true.shape[2]}")
    print(f"  eta_confirm={ETA_CONFIRM}  eta_override={ETA_OVERRIDE}")
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

        pre_n  = build_persona_noise(persona, factor_names)
        post_n = build_post_noise(persona, factor_names, pid)
        enr    = persona["environment_shifts"][0]["day"]

        print(f"\n{'─'*66}")
        print(f"[{pid}] {name}")
        print(f"       {industry}  |  {apd} alerts/day  |  {n_an} analysts  |  q_bar={qb:.3f}")
        print(f"       Enrichment: Day {enr}  |  "
              f"noise {np.mean(pre_n):.3f} → {np.mean(post_n):.3f}  |  "
              f"APD {apd} → {POST_ENRICHMENT_APD.get(pid, apd)}")

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

    print(f"\n{'='*66}")
    print_tables(all_results, personas, categories)

    print("\nSaving standard results...")
    save_results(all_results, personas, output_dir)

    # ── 2G specialized simulation ─────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("Running 2G enrichment-aware simulations (noise switch at enrich_day)...")

    g_results = {}
    for persona in personas:
        pid = persona["persona_id"]
        print(f"  [{pid}] enrichment Day {persona['environment_shifts'][0]['day']}  "
              f"(N={N_2G_SEEDS} seeds)...", end="", flush=True)
        t0  = time.time()
        res = run_2g_analysis(persona, mu_true, categories, cat_to_idx,
                              gt_dists_arr, factor_names)
        g_results[pid] = res
        print(f"  done ({time.time()-t0:.1f}s)")

    print_2g_analysis(g_results, personas, categories)
    print_2g_gates(all_results, g_results, personas)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    enrich_path = output_dir / "enrichment_analysis.json"
    save_out = {}
    for pid, r in g_results.items():
        save_out[pid] = {k: v for k, v in r.items() if k != "mean_daily_acc"}
    with open(enrich_path, "w", encoding="utf-8") as fh:
        json.dump(save_out, fh, indent=2)
    print(f"\n  Saved → {enrich_path}")

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
