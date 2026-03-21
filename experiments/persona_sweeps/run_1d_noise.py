"""
1D Noise Profile Sweep — standard harness + noise-specific analysis.

Additional outputs (derived from harness data, no extra simulation):
  1. τ vs noise relationship (does optimal τ rise with noise?)
  2. Accuracy ceiling vs noise (Day60 < 85% = scorer near-random)
  3. Per-category convergence at high noise (N4/N5)

Usage:
    python experiments/persona_sweeps/run_1d_noise.py \
        --personas experiments/persona_sweeps/personas_sweep_1d_noise.json \
        --output   experiments/persona_sweeps/results/sweep_1d_noise/
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
    build_persona_noise, build_persona_weights, precompute_day_weights,
    precompute_analyst_params, persona_q_bar,
    run_td034, run_prod5, run_ba,
    print_tables, save_results,
    ETA_CONFIRM, ETA_OVERRIDE,
)
from src.data.domain_config import load_domain_config

ACCURACY_CEILING = 0.85   # Day60 below this → scorer near-random, product boundary


# ── Helpers ───────────────────────────────────────────────────────────────────
def mean_noise(persona: dict) -> float:
    return float(np.mean([
        v["base_noise"] for v in persona["factor_noise_profile"].values()
    ]))


def noise_label(persona: dict) -> str:
    """Extract noise tier label from persona name (first word)."""
    return persona["name"].split()[0]   # Ultra-Clean / Clean / Moderate / Noisy / Chaotic


# ── 1D analysis tables ────────────────────────────────────────────────────────
def print_1d_analysis(all_results: dict, personas: list, categories: list):

    # ── Table 1: τ vs noise ──────────────────────────────────────────────────
    print("\n1D Analysis 1: τ vs Noise Relationship")
    print("  (expected: τ* increases with noise — wider distribution needed)")
    hdr = (f"| {'Persona':8} | {'Noise Tier':12} | {'Mean σ':8} | "
           f"{'Opt τ':7} | {'ECE@τ*':8} | {'ECE@0.10':8} | {'Recal?':6} |")
    sep = ("|" + "-"*10 + "|" + "-"*14 + "|" + "-"*10 + "|"
           + "-"*9 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*8 + "|")
    print(hdr); print(sep)
    prev_tau = None
    for persona in personas:
        pid = persona["persona_id"]
        mn  = mean_noise(persona)
        nl  = noise_label(persona)
        td  = all_results[pid]["td034"]
        tau = td["optimal_tau"]
        # Mark if tau increased vs previous tier
        trend = " ↑" if (prev_tau is not None and tau > prev_tau) else \
                (" =" if (prev_tau is not None and tau == prev_tau) else "  ")
        prev_tau = tau
        rec = "YES" if td["recalibrate"] else "no"
        print(f"| {pid:8} | {nl:12} | {mn:8.3f} | "
              f"{str(tau)+trend:7} | {td['ece_at_opt']:8.5f} | "
              f"{td['ece_at_010']:8.5f} | {rec:6} |")

    # ── Table 2: Accuracy ceiling ────────────────────────────────────────────
    print("\n1D Analysis 2: Accuracy Ceiling vs Noise")
    print(f"  Ceiling threshold: Day60 < {ACCURACY_CEILING:.0%}")
    hdr = (f"| {'Persona':8} | {'Noise Tier':12} | {'Mean σ':8} | "
           f"{'Day1':7} | {'Day30':7} | {'Day60':7} | {'Δ(60-1)':8} | {'Ceiling?':8} |")
    sep = ("|" + "-"*10 + "|" + "-"*14 + "|" + "-"*10 + "|"
           + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*10 + "|" + "-"*10 + "|")
    print(hdr); print(sep)
    ceiling_pids = []
    for persona in personas:
        pid  = persona["persona_id"]
        mn   = mean_noise(persona)
        nl   = noise_label(persona)
        p5   = all_results[pid]["prod5"]
        d1   = p5["acc_day1"]
        d30  = p5["acc_day30"]
        d60  = p5["acc_day60"]
        delta = d60 - d1
        ceil  = d60 < ACCURACY_CEILING
        if ceil:
            ceiling_pids.append(pid)
        ceil_str = "YES ⚠" if ceil else "no"
        print(f"| {pid:8} | {nl:12} | {mn:8.3f} | "
              f"{d1:6.1%} | {d30:6.1%} | {d60:6.1%} | "
              f"{delta:+7.2%} | {ceil_str:8} |")

    if ceiling_pids:
        print(f"\n  Product boundary: noise ceiling reached at "
              f"{', '.join(ceiling_pids)}")
        for pid in ceiling_pids:
            mn  = mean_noise(next(p for p in personas if p["persona_id"] == pid))
            d60 = all_results[pid]["prod5"]["acc_day60"]
            print(f"    {pid} (σ={mn:.3f}): Day60={d60:.1%} — "
                  f"signal quality insufficient for reliable scoring")
    else:
        print("\n  No ceiling reached (all personas Day60 ≥ 85%)")

    # ── Table 3: Per-category convergence at high noise ──────────────────────
    print("\n1D Analysis 3: Per-Category Convergence at High Noise")
    high_noise_ids = [p["persona_id"] for p in personas
                      if mean_noise(p) >= 0.20]
    if not high_noise_ids:
        high_noise_ids = [personas[-2]["persona_id"], personas[-1]["persona_id"]]

    for persona in personas:
        pid = persona["persona_id"]
        if pid not in high_noise_ids:
            continue
        mn   = mean_noise(persona)
        nl   = noise_label(persona)
        p5   = all_results[pid]["prod5"]
        nc   = p5["n_cats_converged"]

        print(f"\n  {pid} — {nl} (mean σ={mn:.3f})  |  {nc}/6 categories converged")
        print(f"  {'Category':<22}  {'Conv%':6}  {'Mean Day':8}  {'Not Conv':8}  Notes")
        print(f"  {'-'*22}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*15}")
        for cat in categories:
            cr       = p5["categories"][cat]
            cp       = cr["converge_pct"]
            md_raw   = cr["mean_conv_day"]
            md_str   = f"{md_raw:.1f}d" if md_raw is not None else "—"
            nconv    = cr["not_converged"]
            if cp == 0.0:
                note = "NO convergence"
            elif cp < 80.0:
                note = f"partial ({nconv}/15 fail)"
            else:
                note = "OK"
            flag = " *" if cp < 80.0 else ""
            print(f"  {cat:<22}  {cp:5.1f}%  {md_str:>8}  {nconv:>8}  {note}{flag}")

    if not high_noise_ids:
        print("  (No high-noise personas found)")

    # ── Noise ceiling summary ────────────────────────────────────────────────
    print("\n1D Noise Ceiling Summary:")
    print(f"  {'Persona':8}  {'σ mean':8}  {'τ*':6}  "
          f"{'ECE@τ*':8}  {'Day60':7}  {'Cats Conv':10}  Status")
    print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*15}")
    for persona in personas:
        pid  = persona["persona_id"]
        mn   = mean_noise(persona)
        td   = all_results[pid]["td034"]
        p5   = all_results[pid]["prod5"]
        d60  = p5["acc_day60"]
        nc   = p5["n_cats_converged"]
        ceil = "CEILING" if d60 < ACCURACY_CEILING else ("degraded" if d60 < d60 else "OK")
        # Check if Day60 < Day1
        if p5["acc_day60"] < p5["acc_day1"]:
            status = "DEGRADES"
        elif d60 < ACCURACY_CEILING:
            status = "CEILING"
        else:
            status = "OK"
        print(f"  {pid:8}  {mn:8.3f}  {td['optimal_tau']:6.3f}  "
              f"{td['ece_at_opt']:8.5f}  {d60:6.1%}  {nc:>2}/6 conv    {status}")


def print_1d_gates(all_results: dict, personas: list):
    print("\n" + "=" * 62)
    print("=== GATE EVALUATION — 1D ===")
    print("=" * 62)

    # Gate 1: ECE@opt < 0.10 for all noise levels
    print("\nGate 1: ECE@optimal < 0.10 for all noise levels:")
    g1_all = True
    for persona in personas:
        pid    = persona["persona_id"]
        mn     = mean_noise(persona)
        ece    = all_results[pid]["td034"]["ece_at_opt"]
        passes = ece < 0.10
        if not passes:
            g1_all = False
        print(f"  {pid} (σ={mn:.3f}): ECE@opt={ece:.5f}  "
              f"[{'PASS' if passes else 'FAIL'}]")
    print(f"  Overall: {'PASS' if g1_all else 'FAIL'}")
    if not g1_all:
        print("  FAIL: at chaotic noise, τ sweep cannot find well-calibrated τ.")
        print("  Product recommendation: report 'signal quality insufficient'.")

    # Gate 2: Day60 ≥ Day1 for all noise levels
    print("\nGate 2: Day60 accuracy >= Day1 accuracy:")
    g2_all = True
    fail_pids = []
    for persona in personas:
        pid    = persona["persona_id"]
        p5     = all_results[pid]["prod5"]
        passes = p5["acc_day60"] >= p5["acc_day1"]
        if not passes:
            g2_all = False
            fail_pids.append(pid)
        print(f"  {pid}: {p5['acc_day1']:.1%} → {p5['acc_day60']:.1%}  "
              f"[{'PASS' if passes else 'FAIL'}]")
    print(f"  Overall: {'PASS' if g2_all else 'FAIL'}")
    if not g2_all:
        print(f"  Honest product boundary: {', '.join(fail_pids)} fail due to noise floor.")
        print("  At chaotic σ, learning cannot overcome signal overlap — "
              "scorer degrades over time as centroid drift from noisy overrides "
              "exceeds convergence pull from clean signals.")

    # Noise ceiling threshold
    print("\nNoise ceiling detection:")
    ceiling_found = False
    prev_ok = True
    for persona in personas:
        pid  = persona["persona_id"]
        mn   = mean_noise(persona)
        d60  = all_results[pid]["prod5"]["acc_day60"]
        if d60 < ACCURACY_CEILING:
            if prev_ok:
                print(f"  Ceiling onset: {pid} (σ={mn:.3f}, Day60={d60:.1%})")
                print(f"  Product boundary: σ > {mn:.3f} requires signal remediation")
                print(f"  Recommended action: data quality program before deployment")
            ceiling_found = True
            prev_ok = False
    if not ceiling_found:
        print("  No ceiling reached — scorer viable across entire tested noise range")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="1D Noise Profile Sweep")
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
    print("=" * 62)
    print(f"=== SWEEP RESULTS: {fname} ===")
    print("=" * 62)
    print(f"  Personas file: {personas_path}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Config:        soc_product_v50  "
          f"C={len(categories)} A={len(cfg['actions'])} d={mu_true.shape[2]}")
    print(f"  eta_confirm={ETA_CONFIRM}  eta_override={ETA_OVERRIDE}  (asymmetric, Q5 fix)")
    print(f"  N personas: {len(personas)}")

    all_results = {}
    t_total = time.time()

    for persona in personas:
        pid      = persona["persona_id"]
        name     = persona["name"]
        industry = persona["industry"]
        apd      = persona["alerts_per_day"]
        n_an     = len(persona["analyst_team"])
        qb       = persona_q_bar(persona["analyst_team"])
        mn       = mean_noise(persona)

        print(f"\n{'─'*62}")
        print(f"[{pid}] {name}")
        print(f"       {industry}  |  {apd} alerts/day  |  {n_an} analysts  |  "
              f"q_bar={qb:.3f}  |  mean_noise={mn:.3f}")

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
            "mean_noise": round(mn, 4),
            "td034":      td_res,
            "prod5":      p5_res,
            "ba":         ba_res,
        }

    print(f"\n{'='*62}")
    print_tables(all_results, personas, categories)
    print_1d_analysis(all_results, personas, categories)
    print_1d_gates(all_results, personas)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(all_results, personas, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    noise_analysis = {}
    for persona in personas:
        pid = persona["persona_id"]
        r   = all_results[pid]
        noise_analysis[pid] = {
            "mean_noise":       round(mean_noise(persona), 4),
            "noise_label":      noise_label(persona),
            "noise_profile":    {k: v["base_noise"]
                                 for k, v in persona["factor_noise_profile"].items()},
            "optimal_tau":      r["td034"]["optimal_tau"],
            "ece_at_opt":       r["td034"]["ece_at_opt"],
            "ece_at_010":       r["td034"]["ece_at_010"],
            "acc_day1":         r["prod5"]["acc_day1"],
            "acc_day30":        r["prod5"]["acc_day30"],
            "acc_day60":        r["prod5"]["acc_day60"],
            "n_cats_converged": r["prod5"]["n_cats_converged"],
            "ceiling":          r["prod5"]["acc_day60"] < ACCURACY_CEILING,
        }
    analysis_path = output_dir / "noise_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(noise_analysis, fh, indent=2)
    print(f"  Saved → {analysis_path}")

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
