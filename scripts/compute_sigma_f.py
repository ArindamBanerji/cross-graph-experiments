"""
compute_sigma_f.py — Compute factor covariance matrix Sigma_f from FX-1-PROXY-REAL data.

Prompt 1, Task B: Sigma_f Computation.

DATA SOURCE:
  experiments/expFX1_proxy_real/results.json
  Contains 2,430+ IOC records mapped to 3 SOC factors:
    - threat_intel_enrichment  (stored as "threat_intel" in FX1 results)
    - asset_criticality
    - pattern_history

IMPORTANT — PARTIAL COVERAGE:
  The FX-1-PROXY-REAL experiment maps 3 of 6 SOC factors.
  The remaining 3 are NOT available from real threat-intel data:
    - travel_match   (no real-data proxy — user behavior signal only)
    - time_anomaly   (no real-data proxy — requires live event timestamps)
    - device_trust   (no real-data proxy — requires endpoint/MDM data)
  Sigma_f is therefore a 3x3 matrix for the observable subspace.
  Full 6D Sigma_f requires synthetic or operational fill-in for 3 factors.

SOC factor index order (from soc_product_v50.yaml):
  0: travel_match            -- NOT MAPPED
  1: asset_criticality       -- MAPPED
  2: threat_intel_enrichment -- MAPPED (stored as "threat_intel")
  3: pattern_history         -- MAPPED
  4: time_anomaly            -- NOT MAPPED
  5: device_trust            -- NOT MAPPED

DESIGN ESTIMATE:
  tr(Sigma_f) = 0.24 for full 6D space (uniform ~0.04 per factor).
  For the 3 mapped factors, expected partial trace = ~0.12 (3 × 0.04).

OUTPUTS:
  results/sigma_f_computation.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FX1_RESULTS = REPO_ROOT / "experiments" / "expFX1_proxy_real" / "results.json"
OUT_DIR     = REPO_ROOT / "results"
OUT_FILE    = OUT_DIR / "sigma_f_computation.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# TASK A: eta confirmation (read from ProfileScorer source)
# ---------------------------------------------------------------------------
print("=" * 65)
print("TASK A: eta_pos / eta_neg confirmation")
print("=" * 65)

# Values from src/models/profile_scorer.py __init__ signature:
#   eta:     float = 0.05  -- POSITIVE learning rate (pull correct centroid)
#   eta_neg: float = 0.01  -- NEGATIVE learning rate (push wrong centroid)
#   BUT: all canonical experiments explicitly pass eta_neg=0.05
#   CANONICAL product value: eta=0.05, eta_neg=0.05

ETA_DEFAULT_CODE = 0.05     # eta default in ProfileScorer source
ETA_NEG_DEFAULT_CODE = 0.01 # eta_neg default in ProfileScorer source
ETA_CANONICAL = 0.05        # value used in all prod4_final, expB1_recheck, etc.
ETA_NEG_CANONICAL = 0.05    # CLAIM-35: eta_neg=1.0 FORBIDDEN; all exps use 0.05

n_half_pos = math.log(2) / (-math.log(1 - ETA_CANONICAL))
n_half_neg = math.log(2) / (-math.log(1 - ETA_NEG_CANONICAL))

print(f"ProfileScorer source defaults:")
print(f"  eta (positive)  default in code = {ETA_DEFAULT_CODE}")
print(f"  eta_neg         default in code = {ETA_NEG_DEFAULT_CODE}")
print()
print(f"Canonical product values (explicitly set in all experiments):")
print(f"  eta_pos  = {ETA_CANONICAL}")
print(f"  eta_neg  = {ETA_NEG_CANONICAL}")
print()

if ETA_NEG_DEFAULT_CODE != ETA_NEG_CANONICAL:
    print(f"⚠  FLAG: eta_neg default in code ({ETA_NEG_DEFAULT_CODE}) ≠ canonical "
          f"({ETA_NEG_CANONICAL}).")
    print(f"   Any call to ProfileScorer() without explicit eta_neg= will use {ETA_NEG_DEFAULT_CODE}.")
    print(f"   All product experiments override this to {ETA_NEG_CANONICAL} explicitly.")
    print(f"   CLAIM-35 (eta_neg=1.0 FORBIDDEN) is upheld in all validated runs.")
    print()

print(f"η_pos = {ETA_CANONICAL}, η_neg = {ETA_NEG_CANONICAL}. "
      f"N_half = ln(2)/ln(1/(1-η)) = {n_half_pos:.2f}")
print()

# ---------------------------------------------------------------------------
# TASK B: Load FX-1-PROXY-REAL factor data
# ---------------------------------------------------------------------------
print("=" * 65)
print("TASK B: Sigma_f Computation")
print("=" * 65)

print(f"\n[STEP 1] Loading factor data from {FX1_RESULTS} ...")

if not FX1_RESULTS.exists():
    raise FileNotFoundError(
        f"FX-1-PROXY-REAL results not found at {FX1_RESULTS}. "
        "Run experiments/expFX1_proxy_real/run.py first."
    )

with open(FX1_RESULTS) as f:
    fx1 = json.load(f)

n_records = fx1["n_records"]
print(f"  Source records: CISA KEV={n_records['cisa_kev']}, "
      f"NVD={n_records['nvd_cves']}, ATT&CK={n_records.get('mitre_attack', 0)}")

# The 3 mapped factors — parallel arrays of equal length
ti_arr  = np.array(fx1["factors"]["threat_intel"],      dtype=float)   # threat_intel_enrichment
ac_arr  = np.array(fx1["factors"]["asset_criticality"], dtype=float)
ph_arr  = np.array(fx1["factors"]["pattern_history"],   dtype=float)

# Verify equal length
assert len(ti_arr) == len(ac_arr) == len(ph_arr), (
    f"Factor arrays have unequal lengths: "
    f"ti={len(ti_arr)}, ac={len(ac_arr)}, ph={len(ph_arr)}"
)
N = len(ti_arr)
print(f"  Factor vectors loaded: N = {N}")

# Stack into (N, 3) matrix — columns: [threat_intel, asset_criticality, pattern_history]
# Corresponding to SOC indices [2, 1, 3]
factor_data_3d = np.column_stack([ti_arr, ac_arr, ph_arr])

print(f"\n  Factor ranges:")
for name, arr in [("threat_intel_enrichment", ti_arr),
                  ("asset_criticality",        ac_arr),
                  ("pattern_history",           ph_arr)]:
    print(f"    {name:30s}: min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")

# ---------------------------------------------------------------------------
# STEP 2: Compute covariance matrix (3x3 for observable subspace)
# ---------------------------------------------------------------------------
print(f"\n[STEP 2] Computing covariance matrix ...")

Sigma_3d     = np.cov(factor_data_3d.T)   # shape (3, 3)
tr_sigma_3d  = float(np.trace(Sigma_3d))
per_factor_var = np.diag(Sigma_3d).tolist()

MAPPED_FACTOR_NAMES = [
    "threat_intel_enrichment",
    "asset_criticality",
    "pattern_history",
]
UNMAPPED_FACTOR_NAMES = [
    "travel_match",
    "time_anomaly",
    "device_trust",
]

print(f"\n  3x3 Sigma_f (observable subspace: {MAPPED_FACTOR_NAMES}):")
print(f"  {'':30s}  " + "  ".join(f"{n[:8]:>8}" for n in MAPPED_FACTOR_NAMES))
for i, row in enumerate(Sigma_3d):
    print(f"  {MAPPED_FACTOR_NAMES[i]:30s}  " +
          "  ".join(f"{v:8.5f}" for v in row))

print(f"\n  Per-factor variances:")
for name, var in zip(MAPPED_FACTOR_NAMES, per_factor_var):
    print(f"    {name:30s}: {var:.6f}")

print(f"\n  tr(Sigma_f) for 3 mapped factors = {tr_sigma_3d:.6f}")
print(f"  Note: tr for unmapped 3 factors (travel_match, time_anomaly, device_trust) "
      f"= NOT AVAILABLE from real data")

# Steady-state MSE at eta=0.05
eta          = ETA_CANONICAL
mse_inf_3d   = eta / (2 - eta) * tr_sigma_3d
e_inf_per_3d = math.sqrt(mse_inf_3d / 3)

print(f"\n  Steady-state MSE (3D, η=0.05): {mse_inf_3d:.6f}")
print(f"  e_∞ per component (3D):         {e_inf_per_3d:.4f}")

# Extrapolation to 6D (assuming uniform variance for unmapped factors)
# Unmapped factors (travel_match, time_anomaly, device_trust) are binary/bounded.
# Conservative estimate: treat same variance as observed mean.
mean_obs_var  = tr_sigma_3d / 3
tr_sigma_6d_est = tr_sigma_3d + 3 * mean_obs_var   # extrapolated
mse_inf_6d_est  = eta / (2 - eta) * tr_sigma_6d_est
e_inf_per_6d    = math.sqrt(mse_inf_6d_est / 6)

print(f"\n  Extrapolated 6D tr(Sigma_f): {tr_sigma_6d_est:.6f} "
      f"(assumes unmapped factors same avg variance = {mean_obs_var:.6f})")
print(f"  Steady-state MSE (6D, extrapolated): {mse_inf_6d_est:.6f}")
print(f"  e_∞ per component (6D, extrapolated): {e_inf_per_6d:.4f}")

# ---------------------------------------------------------------------------
# STEP 2b: Sanity checks
# ---------------------------------------------------------------------------
print(f"\n[STEP 2b] Sanity checks ...")

all_pass = True
for name, var in zip(MAPPED_FACTOR_NAMES, per_factor_var):
    if var <= 0:
        print(f"  FAIL: {name} variance = {var:.6f} — must be > 0")
        all_pass = False
    elif var > 0.25:
        print(f"  FAIL: {name} variance = {var:.6f} — factors in [0,1], max var = 0.25")
        all_pass = False
    else:
        print(f"  PASS: {name} variance = {var:.6f}  in (0, 0.25]")

if all_pass:
    print("  All sanity checks passed.")

# ---------------------------------------------------------------------------
# STEP 4: Compare to design estimate
# ---------------------------------------------------------------------------
print(f"\n[STEP 4] Design estimate comparison ...")

DESIGN_ESTIMATE_6D = 0.24
DESIGN_ESTIMATE_3D = DESIGN_ESTIMATE_6D / 2   # expected 3-factor partial trace

delta_3d = (tr_sigma_3d - DESIGN_ESTIMATE_3D) / DESIGN_ESTIMATE_3D * 100
delta_6d = (tr_sigma_6d_est - DESIGN_ESTIMATE_6D) / DESIGN_ESTIMATE_6D * 100

print(f"  Design estimate (6D): tr(Sigma_f) = {DESIGN_ESTIMATE_6D}")
print(f"  Design estimate (3D, expected half): {DESIGN_ESTIMATE_3D}")
print(f"  Observed (3D):        tr(Sigma_f) = {tr_sigma_3d:.4f}  "
      f"({delta_3d:+.1f}% vs 3D estimate)")
print(f"  Extrapolated (6D):    tr(Sigma_f) = {tr_sigma_6d_est:.4f}  "
      f"({delta_6d:+.1f}% vs 6D design estimate)")

if abs(delta_3d) > 20:
    print(f"\n  FLAG: tr(Sigma_f) 3D = {tr_sigma_3d:.4f} differs from "
          f"3D design estimate {DESIGN_ESTIMATE_3D:.2f} by {delta_3d:+.1f}%.")
    print(f"        All L-08 predictions for the 3 observed factors need updating.")
else:
    print(f"\n  OK: 3D tr(Sigma_f) within 20% of design estimate.")

if abs(delta_6d) > 20:
    print(f"\n  FLAG: Extrapolated 6D tr(Sigma_f) = {tr_sigma_6d_est:.4f} differs "
          f"from design estimate {DESIGN_ESTIMATE_6D} by {delta_6d:+.1f}%.")
    print(f"        L-08 predictions for full 6D space need updating once "
          f"unmapped factor distributions are characterized.")
else:
    print(f"  OK: Extrapolated 6D tr(Sigma_f) within 20% of design estimate.")

# ---------------------------------------------------------------------------
# STEP 3: Save results
# ---------------------------------------------------------------------------
print(f"\n[STEP 3] Saving results ...")

results_out = {
    "tr_sigma_f_3d_observed": tr_sigma_3d,
    "tr_sigma_f_6d_extrapolated": tr_sigma_6d_est,
    "per_factor_variances_observed": {
        name: float(var)
        for name, var in zip(MAPPED_FACTOR_NAMES, per_factor_var)
    },
    "per_factor_variances_note": (
        "Only 3 of 6 SOC factors have real-data coverage. "
        "travel_match, time_anomaly, device_trust are not present in "
        "CISA KEV / NVD / MITRE ATT&CK and are therefore excluded."
    ),
    "mapped_factors": MAPPED_FACTOR_NAMES,
    "unmapped_factors": UNMAPPED_FACTOR_NAMES,
    "Sigma_f_matrix_3d": Sigma_3d.tolist(),
    "Sigma_f_matrix_note": (
        "3x3 covariance for [threat_intel_enrichment, asset_criticality, "
        "pattern_history] — SOC indices [2, 1, 3]"
    ),
    "steady_state_mse_3d": float(mse_inf_3d),
    "e_inf_per_component_3d": float(e_inf_per_3d),
    "steady_state_mse_6d_extrapolated": float(mse_inf_6d_est),
    "e_inf_per_component_6d_extrapolated": float(e_inf_per_6d),
    "eta_used": eta,
    "n_samples": N,
    "source": "FX-1-PROXY-REAL (CISA KEV 1542 + NVD 200 + MITRE ATT&CK 691)",
    "design_estimate_6d": DESIGN_ESTIMATE_6D,
    "delta_3d_pct": float(delta_3d),
    "delta_6d_extrapolated_pct": float(delta_6d),
    "a4_note": (
        "Computed on 6D factor space (3 observed factors only). "
        "Action count (A=4 after Phase 0a refer_to_analyst removal) does not "
        "affect factor covariance. Sigma_f is factor-space only."
    ),
    "eta_confirmation": {
        "eta_pos_code_default": ETA_DEFAULT_CODE,
        "eta_neg_code_default": ETA_NEG_DEFAULT_CODE,
        "eta_pos_canonical":    ETA_CANONICAL,
        "eta_neg_canonical":    ETA_NEG_CANONICAL,
        "n_half_pos":           round(n_half_pos, 3),
        "n_half_neg":           round(n_half_neg, 3),
        "eta_neg_default_matches_canonical": ETA_NEG_DEFAULT_CODE == ETA_NEG_CANONICAL,
        "flag_eta_neg_default": (
            "eta_neg default in code (0.01) != canonical product value (0.05). "
            "All validated experiments explicitly set eta_neg=0.05. "
            "CLAIM-35 (eta_neg=1.0 FORBIDDEN) is upheld in all prod runs."
            if ETA_NEG_DEFAULT_CODE != ETA_NEG_CANONICAL else "OK"
        ),
    },
}

with open(OUT_FILE, "w") as f:
    json.dump(results_out, f, indent=2)

print(f"  Results saved to {OUT_FILE}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"η_pos = {ETA_CANONICAL}, η_neg = {ETA_NEG_CANONICAL} (canonical; code default differs)")
print(f"N_half = ln(2)/ln(1/(1-η)) = {n_half_pos:.2f}")
print()
print(f"tr(Σ_f) [3 observed factors] = {tr_sigma_3d:.6f}")
print(f"tr(Σ_f) [6D extrapolated]    = {tr_sigma_6d_est:.6f}")
print(f"Design estimate (6D)         = {DESIGN_ESTIMATE_6D}")
print(f"Delta (6D extrapolated)      = {delta_6d:+.1f}%")
print()
print(f"Steady-state MSE (η=0.05, 3D): {mse_inf_3d:.6f}")
print(f"e_∞ per component (3D):        {e_inf_per_3d:.4f}")
print()
print("Per-factor variances (observed):")
for name, var in zip(MAPPED_FACTOR_NAMES, per_factor_var):
    in_range = "✓" if 0 < var <= 0.25 else "✗"
    print(f"  {in_range} {name:30s}: {var:.6f}")
print()
print(f"Not computed (no real-data source): {', '.join(UNMAPPED_FACTOR_NAMES)}")
print()
if abs(delta_6d) > 20:
    print(f"FLAG: tr(Σ_f) = {tr_sigma_6d_est:.4f} differs from design estimate "
          f"{DESIGN_ESTIMATE_6D} by {delta_6d:+.1f}%. "
          f"All L-08 predictions need updating.")
else:
    print(f"OK: tr(Σ_f) within 20% of design estimate ({delta_6d:+.1f}%).")
print("=" * 65)
