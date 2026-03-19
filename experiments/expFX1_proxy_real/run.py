"""
FX1-PROXY-REAL: Real IOC Factor Distribution Characterization.

Pulls public threat intelligence (CISA KEV, NVD CVE, MITRE ATT&CK),
maps to 3 SOC factors, computes KL divergence vs synthetic centroidal reference,
and characterizes the distribution gap.

Outputs:
  experiments/expFX1_proxy_real/results.json
  paper_figures/fx1r_*.{pdf,png}  (3 charts × 2 formats = 6 files)

Runtime: <5 minutes (30s timeout per API call; caches on first run).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local modules
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from data_pull import fetch_cisa_kev, fetch_nvd_cves, fetch_mitre_attack
from factor_mapper import map_to_factors
from distribution_analysis import fit_gaussian, compute_kl_divergence, distribution_stats

RESULTS_PATH = _DIR / "results.json"

# Synthetic centroidal reference parameters
SYNTHETIC_REFS = {
    "threat_intel":      {"mean": 0.5, "std": 0.20},
    "asset_criticality": {"mean": 0.5, "std": 0.25},
    "pattern_history":   {"mean": 0.3, "std": 0.20},
}

FACTOR_DISPLAY = {
    "threat_intel":      "Threat Intel Score",
    "asset_criticality": "Asset Criticality Proxy",
    "pattern_history":   "Pattern History Proxy",
}


if __name__ == "__main__":
    t_start = time.time()

    print("=" * 65)
    print("FX1-PROXY-REAL: REAL IOC FACTOR DISTRIBUTION CHARACTERIZATION")
    print("=" * 65)
    print()

    # -------------------------------------------------------------------
    # Step 1: Pull data
    # -------------------------------------------------------------------
    print("[STEP 1] Pulling public threat intelligence data ...")
    kev_records   = fetch_cisa_kev()
    nvd_records   = fetch_nvd_cves(max_results=200)
    mitre_records = fetch_mitre_attack()

    total_pulled = len(kev_records) + len(nvd_records) + len(mitre_records)
    print(f"\n[DATA] Pulled {len(kev_records)} CISA KEV records, "
          f"{len(nvd_records)} NVD CVEs, "
          f"{len(mitre_records)} ATT&CK techniques")
    print(f"[DATA] Total: {total_pulled} records")

    # -------------------------------------------------------------------
    # Step 2: Map to factor space
    # -------------------------------------------------------------------
    print("\n[STEP 2] Mapping to SOC factor space ...")
    factors = map_to_factors(kev_records, nvd_records, mitre_records)

    for fname, vals in factors.items():
        print(f"  {FACTOR_DISPLAY[fname]:30s}: {len(vals)} values  "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]" if len(vals) > 0
              else f"  {FACTOR_DISPLAY[fname]:30s}: 0 values (API failed)")

    # Minimum data check
    min_n = min(len(v) for v in factors.values())
    if min_n < 50:
        print(f"\n  WARNING: Minimum factor has only {min_n} values (<50). "
              f"API may have partially failed. Proceeding with available data.")

    # -------------------------------------------------------------------
    # Step 3: Distribution stats + KL divergence
    # -------------------------------------------------------------------
    print("\n[STEP 3] Computing distribution statistics and KL divergence ...")
    stats_dict  = {}
    kl_dict     = {}
    fit_dict    = {}

    for fname in ["threat_intel", "asset_criticality", "pattern_history"]:
        vals = factors[fname]
        ref  = SYNTHETIC_REFS[fname]

        st   = distribution_stats(vals)
        mu, sigma = fit_gaussian(vals)
        kl   = compute_kl_divergence(vals, ref["mean"], ref["std"])

        stats_dict[fname] = st
        kl_dict[fname]    = float(kl) if not (kl != kl) else 0.0
        fit_dict[fname]   = {"mean": mu, "std": sigma}

    # -------------------------------------------------------------------
    # Print results table
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    print(f"{'Factor':<30}  {'N':>6}  {'Mean':>6}  {'Std':>6}  "
          f"{'Skew':>7}  {'Kurt':>7}  {'KL div':>8}  {'Synthetic ref':>15}")
    print("-" * 90)
    for fname in ["threat_intel", "asset_criticality", "pattern_history"]:
        st  = stats_dict[fname]
        kl  = kl_dict[fname]
        ref = SYNTHETIC_REFS[fname]
        print(f"{FACTOR_DISPLAY[fname]:<30}  {st['n']:>6}  {st['mean']:>6.3f}  "
              f"{st['std']:>6.3f}  {st['skewness']:>7.3f}  {st['kurtosis']:>7.3f}  "
              f"{kl:>8.4f}  μ={ref['mean']:.2f} σ={ref['std']:.2f}")

    # -------------------------------------------------------------------
    # Overall assessment
    # -------------------------------------------------------------------
    kl_values = list(kl_dict.values())
    max_kl    = max(kl_values) if kl_values else 0.0
    all_kl_ok = all(kl < 0.3  for kl in kl_values)
    any_kl_hi = any(kl > 0.5  for kl in kl_values)

    if all_kl_ok:
        assessment = "CENTROIDAL ASSUMPTION HOLDS"
        assess_color = "green"
    elif any_kl_hi:
        assessment = "DISTRIBUTION GAP DETECTED"
        assess_color = "red"
    else:
        assessment = "MODERATE GAP"
        assess_color = "orange"

    print("\n" + "=" * 65)
    print(f"OVERALL ASSESSMENT: {assessment}")
    print(f"  Max KL divergence: {max_kl:.4f}")
    print(f"  KL by factor: " +
          ", ".join(f"{FACTOR_DISPLAY[f]}={kl_dict[f]:.4f}"
                    for f in ["threat_intel", "asset_criticality", "pattern_history"]))
    print("=" * 65)

    # Per-factor verdict
    for fname in ["threat_intel", "asset_criticality", "pattern_history"]:
        kl = kl_dict[fname]
        if kl < 0.3:
            verdict = "LOW divergence — synthetic approximation adequate"
        elif kl < 0.5:
            verdict = "MODERATE divergence — consider distribution-aware sampling"
        else:
            verdict = "HIGH divergence — centroidal assumption INVALID for this factor"
        print(f"  {FACTOR_DISPLAY[fname]}: KL={kl:.4f}  [{verdict}]")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f} seconds")

    # -------------------------------------------------------------------
    # Save results.json
    # -------------------------------------------------------------------
    results = {
        "n_records": {
            "cisa_kev":    len(kev_records),
            "nvd_cves":    len(nvd_records),
            "mitre_attack": len(mitre_records),
        },
        "factors": {k: v.tolist() for k, v in factors.items()},
        "stats":         {k: {sk: (float(sv) if isinstance(sv, (int, float, np.floating))
                                   else sv)
                               for sk, sv in v.items()}
                          for k, v in stats_dict.items()},
        "kl_divergence": kl_dict,
        "gaussian_fit":  fit_dict,
        "synthetic_refs": SYNTHETIC_REFS,
        "assessment":    assessment,
        "runtime_seconds": elapsed,
    }

    with open(RESULTS_PATH, "w") as fout:
        json.dump(results, fout, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

    # -------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------
    from charts import generate_charts
    generate_charts(factors, stats_dict, kl_dict)

    # -------------------------------------------------------------------
    # Final file check
    # -------------------------------------------------------------------
    chart_files = [
        "fx1r_factor_distributions.png",
        "fx1r_factor_distributions.pdf",
        "fx1r_kl_divergence_from_synthetic.png",
        "fx1r_kl_divergence_from_synthetic.pdf",
        "fx1r_distribution_statistics.png",
        "fx1r_distribution_statistics.pdf",
    ]
    paper_figs = _REPO_ROOT / "paper_figures"
    print("\n[FILES]")
    all_ok = True
    for fname in chart_files:
        p = paper_figs / fname
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  OK  {fname}  ({size_kb} KB)")
        else:
            print(f"  MISSING  {fname}")
            all_ok = False

    if not all_ok:
        print("  WARNING: Some chart files missing.")
    else:
        print("\n  All 6 chart files confirmed.")

    print(f"\nFINAL ASSESSMENT: {assessment}")
    print(f"Total runtime: {elapsed:.1f}s")
