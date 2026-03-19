"""
Bridge B Phase C: Temporal Compounding — Does convergence accelerate as graph enriches?

Simulates a customer's first 5000 verified decisions with graph maturity
increasing over time. Extracts effective learning rate η_eff per 500-decision
window by fitting error trajectory to exponential decay.

Timeline:
  Decisions 0-500:     G₁ (single SIEM, day 1)
  Decisions 500-1500:  G₂ (second SIEM, ~week 2)
  Decisions 1500-3000: G₃ (entity resolution, ~week 6)
  Decisions 3000-5000: G₄ (ThreatIndicators, ~week 12)

Window → G-level mapping (10 windows × 500 decisions):
  W0 (0-500):    G₁
  W1 (500-1000): G₂
  W2 (1000-1500):G₂
  W3 (1500-2000):G₃
  W4 (2000-2500):G₃
  W5 (2500-3000):G₃
  W6 (3000-3500):G₄
  W7 (3500-4000):G₄
  W8 (4000-4500):G₄
  W9 (4500-5000):G₄

Depends on: results/sigma_f_computation.json (P1), results/bridge_b_phase_b.json (P2)
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import linregress, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SEEDS        = 50
WINDOW_SIZE    = 500
N_WINDOWS      = 10
N_DECISIONS    = N_WINDOWS * WINDOW_SIZE   # 5000
ETA            = 0.05
E0_BASE        = 0.15
EPS            = 0.05
RHO            = 0.8
N_EFF          = 2.0 / (1.0 + RHO)        # 1.1111
THREAT_IDX     = 2
DOMAIN_CONFIG  = "soc_product_v50"
RANDOM_SEED_BASE = 42

# Window → G level
WINDOW_G = ["G1", "G2", "G2", "G3", "G3", "G3", "G4", "G4", "G4", "G4"]

assert len(WINDOW_G) == N_WINDOWS

WINDOW_DECISIONS = [
    (w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE) for w in range(N_WINDOWS)
]

P1_PATH      = _REPO_ROOT / "results" / "sigma_f_computation.json"
P2_PATH      = _REPO_ROOT / "results" / "bridge_b_phase_b.json"
RESULTS_PATH = _REPO_ROOT / "results" / "bridge_b_phase_c.json"

# ---------------------------------------------------------------------------
# Load P1 — per-factor variances
# ---------------------------------------------------------------------------

with open(P1_PATH) as f:
    p1 = json.load(f)

obs = p1["per_factor_variances_observed"]
var_mean = (obs["asset_criticality"] +
            obs["threat_intel_enrichment"] +
            obs["pattern_history"]) / 3.0

sigma_f_base = np.array([
    var_mean,                        # 0: travel_match (extrapolated)
    obs["asset_criticality"],        # 1: asset_criticality
    obs["threat_intel_enrichment"],  # 2: threat_intel_enrichment
    obs["pattern_history"],          # 3: pattern_history
    var_mean,                        # 4: time_anomaly (extrapolated)
    var_mean,                        # 5: device_trust (extrapolated)
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Load P2 — reference N_converge values
# ---------------------------------------------------------------------------

with open(P2_PATH) as f:
    p2 = json.load(f)

p2_n_converge = {g: p2["levels"][g]["mean_n_converge"] for g in ["G1","G2","G3","G4"]}

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config    = load_domain_config(DOMAIN_CONFIG)
C, A, d   = config["C"], config["A"], config["d"]
mu_config = config["mu"].copy()   # (C, A, d)

assert (C, A, d) == (6, 4, 6)

# ---------------------------------------------------------------------------
# Noise model (same as P2)
# ---------------------------------------------------------------------------

def sigma_f_for(g_level: str) -> np.ndarray:
    if g_level == "G1":
        return sigma_f_base.copy()
    sf = sigma_f_base / N_EFF
    if g_level == "G4":
        sf = sf.copy()
        sf[THREAT_IDX] *= 0.70
    return sf

def e0_for(g_level: str) -> float:
    return E0_BASE if g_level in ("G1", "G2") else E0_BASE * 0.85

# ---------------------------------------------------------------------------
# Print setup
# ---------------------------------------------------------------------------

print("=" * 60)
print("BRIDGE B PHASE C: TEMPORAL COMPOUNDING (A=4)")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"N_SEEDS={N_SEEDS}, N_DECISIONS={N_DECISIONS}, WINDOW_SIZE={WINDOW_SIZE}")
print(f"ETA={ETA}, EPS={EPS}, E0_BASE={E0_BASE}, RHO={RHO}")
print(f"Total cells: {N_SEEDS}×{C}×{A} = {N_SEEDS*C*A}")
print()
print("P2 reference N_converge:")
for g, nc in p2_n_converge.items():
    print(f"  {g}: {nc:.1f}")
print()
print("Window plan:")
for wi, (g_level, (t0, t1)) in enumerate(zip(WINDOW_G, WINDOW_DECISIONS)):
    print(f"  W{wi} [{t0:4d}-{t1:4d}]: {g_level}")
print()

# ---------------------------------------------------------------------------
# Build mu_true_batch: (n_cells, d)
# ---------------------------------------------------------------------------

n_cells = N_SEEDS * C * A  # 1200

mu_true_batch = np.zeros((n_cells, d), dtype=np.float64)
for s in range(N_SEEDS):
    for c in range(C):
        for a in range(A):
            mu_true_batch[s * C * A + c * A + a] = mu_config[c, a, :]

# ---------------------------------------------------------------------------
# η_eff fitting: exponential decay → slope → η_eff
# ---------------------------------------------------------------------------

def fit_eta_eff(errors: np.ndarray) -> tuple[float, float]:
    """
    Fit error(t) = A × exp(slope × t). Returns (eta_eff, r_squared).
    eta_eff = 1 - exp(slope).
    If already flat/converged: return (0.0, 0.0).
    """
    if len(errors) < 5:
        return 0.0, 0.0
    if errors[0] < 1e-8:
        return 0.0, 0.0
    t_vals     = np.arange(len(errors), dtype=float)
    log_errors = np.log(np.maximum(errors, 1e-10))
    slope, intercept, r, p, se = linregress(t_vals, log_errors)
    r2       = r ** 2
    eta_eff  = float(1.0 - np.exp(slope))
    return eta_eff, float(r2)

# ---------------------------------------------------------------------------
# Main simulation — single pass over 5000 decisions for all cells
# ---------------------------------------------------------------------------

print("Running 5000-decision simulation across 1200 cells × 50 seeds...", flush=True)

# We need per-window, per-cell error trajectories
# Memory: 10 windows × 500 steps × 1200 cells → 6M floats = ~48 MB: fine
# Store mean error per step per window aggregated on the fly

# per_window_errors[w, t] = mean max-component error across all cells at step t within window w
per_window_errors_sum = np.zeros((N_WINDOWS, WINDOW_SIZE), dtype=np.float64)

# per_cell_init_error[w, cell] = initial max-component error at start of window w
per_cell_init_error = np.zeros((N_WINDOWS, n_cells), dtype=np.float64)

# per_cell_window_errors[w, t, cell] — too large; instead collect per-cell mean errors
# per_cell_mean_error_in_window[w, cell] — used to compute per-cell η_eff
# We store the full trajectory for each cell × window only for η_eff fitting
# That's 10 × 500 × 1200 = 6M — store float32 to keep it ~24MB
cell_traj = np.zeros((N_WINDOWS, WINDOW_SIZE, n_cells), dtype=np.float32)

# Initialize with G1 e0
rng = np.random.default_rng(RANDOM_SEED_BASE)
sf_current = sigma_f_for("G1")
offset = rng.standard_normal((n_cells, d)) * E0_BASE
mu = np.clip(mu_true_batch + offset, 0.0, 1.0)

for wi in range(N_WINDOWS):
    g_level  = WINDOW_G[wi]
    sf       = sigma_f_for(g_level)
    sqrt_sf  = np.sqrt(sf)
    t0, t1   = WINDOW_DECISIONS[wi]

    # At window start: record initial error per cell
    init_err = np.max(np.abs(mu - mu_true_batch), axis=1)  # (n_cells,)
    per_cell_init_error[wi] = init_err

    for t in range(WINDOW_SIZE):
        noise = rng.standard_normal((n_cells, d)) * sqrt_sf
        f     = np.clip(mu_true_batch + noise, 0.0, 1.0)
        err   = np.max(np.abs(mu - mu_true_batch), axis=1)   # (n_cells,)

        cell_traj[wi, t] = err.astype(np.float32)
        per_window_errors_sum[wi, t] += err.mean()

        mu += ETA * (f - mu)
        np.clip(mu, 0.0, 1.0, out=mu)

    print(f"  W{wi} [{t0:4d}-{t1:4d}] {g_level}: "
          f"init_err={init_err.mean():.4f}  "
          f"final_err={cell_traj[wi,-1].mean():.4f}", flush=True)

# Mean error trajectory per window (across all cells)
per_window_mean_errors = per_window_errors_sum / 1.0   # already mean (we divided by nothing; it's .sum()/1, but actually it's already mean b/c we called err.mean())
# Actually per_window_errors_sum[wi, t] = sum of err.mean() over... no, we called err.mean() once per step and added it. So it IS the step-level cross-cell mean. No division needed.
per_window_mean_errors = per_window_errors_sum   # shape (10, 500): cross-cell mean error at each step

# ---------------------------------------------------------------------------
# Compute η_eff per window, aggregated across pre-convergence cells
# ---------------------------------------------------------------------------

print()
print("Fitting η_eff per window...")

# For each window and each cell, fit η_eff if init_error > EPS
# Then aggregate: mean η_eff across pre-convergence cells for each window

window_eta_eff   = np.zeros(N_WINDOWS, dtype=np.float64)
window_eta_std   = np.zeros(N_WINDOWS, dtype=np.float64)
window_r2_mean   = np.zeros(N_WINDOWS, dtype=np.float64)
window_n_active  = np.zeros(N_WINDOWS, dtype=int)    # cells with init_err > EPS

for wi in range(N_WINDOWS):
    cell_etas  = []
    cell_r2s   = []
    for ci in range(n_cells):
        if per_cell_init_error[wi, ci] <= EPS:
            continue   # already converged — skip
        errors = cell_traj[wi, :, ci].astype(np.float64)
        eta_eff, r2 = fit_eta_eff(errors)
        cell_etas.append(eta_eff)
        cell_r2s.append(r2)

    if cell_etas:
        window_eta_eff[wi]  = float(np.mean(cell_etas))
        window_eta_std[wi]  = float(np.std(cell_etas))
        window_r2_mean[wi]  = float(np.mean(cell_r2s))
        window_n_active[wi] = len(cell_etas)
    else:
        window_eta_eff[wi]  = 0.0
        window_eta_std[wi]  = 0.0
        window_r2_mean[wi]  = 0.0
        window_n_active[wi] = 0

    print(f"  W{wi} {WINDOW_G[wi]:2s}: η_eff={window_eta_eff[wi]:.5f} "
          f"±{window_eta_std[wi]:.5f}  R²={window_r2_mean[wi]:.3f}  "
          f"n_active={window_n_active[wi]}", flush=True)

# ---------------------------------------------------------------------------
# Spearman correlation (pre-convergence windows only)
# ---------------------------------------------------------------------------

active_mask     = window_n_active > 0
active_indices  = np.where(active_mask)[0]
active_eta_eff  = window_eta_eff[active_mask]

n_pre_conv = int(active_mask.sum())

if n_pre_conv >= 3:
    spear_rho, spear_p = spearmanr(active_indices.tolist(), active_eta_eff.tolist())
else:
    spear_rho, spear_p = float("nan"), float("nan")

# ---------------------------------------------------------------------------
# Per-transition analysis
# ---------------------------------------------------------------------------

# G₁→G₂: mean η_eff of W0 (G1) vs mean η_eff of W1-W2 (G2)
def window_mean_eta(windows: list[int]) -> float:
    vals = [window_eta_eff[w] for w in windows if window_n_active[w] > 0]
    return float(np.mean(vals)) if vals else float("nan")

g1_eta   = window_mean_eta([0])
g2_eta   = window_mean_eta([1, 2])
g3_eta   = window_mean_eta([3, 4, 5])
g4_eta   = window_mean_eta([6, 7, 8, 9])

g1_to_g2 = g2_eta - g1_eta
g2_to_g3 = g3_eta - g2_eta
g3_to_g4 = g4_eta - g3_eta

transitions_monotonic = sum([
    g1_to_g2 > 0,
    g2_to_g3 > 0,
    g3_to_g4 > 0,
])

# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

spearman_pass    = (not np.isnan(spear_rho)) and (spear_rho > 0) and (spear_p < 0.05)
monotonic_pass   = transitions_monotonic >= 3
overall_evidence = spearman_pass and monotonic_pass

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("=== BRIDGE B PHASE C: TEMPORAL COMPOUNDING (A=4) ===")
print("=" * 70)
print()

hdr = (f"  {'Window':<6} | {'Decisions':<12} | {'G Level':<7} | "
       f"{'η_eff mean':>11} | {'η_eff std':>9} | {'Delta vs prev':>14} | "
       f"{'n_active':>8}")
sep = "  " + "-" * (len(hdr) - 2)
print(hdr)
print(sep)

prev_eta = None
for wi in range(N_WINDOWS):
    t0, t1  = WINDOW_DECISIONS[wi]
    g_level = WINDOW_G[wi]
    eta_m   = window_eta_eff[wi]
    eta_s   = window_eta_std[wi]
    n_act   = window_n_active[wi]

    if prev_eta is None or np.isnan(prev_eta):
        delta_str = "  baseline"
    elif n_act == 0:
        delta_str = "  (no active)"
    else:
        delta = eta_m - prev_eta
        delta_str = f"{delta:+.5f}"

    # Mark G-level transitions
    is_transition = wi > 0 and WINDOW_G[wi] != WINDOW_G[wi - 1]
    trans_marker  = " ←" if is_transition else "  "

    print(f"  W{wi:<5} | {t0:4d}-{t1:4d}     | {g_level:<7} | "
          f"{eta_m:>11.5f} | {eta_s:>9.5f} | {delta_str:>14}{trans_marker} | "
          f"{n_act:>8}")

    if n_act > 0:
        prev_eta = eta_m

print()
print(f"  G₁ mean η_eff: {g1_eta:.5f}")
print(f"  G₂ mean η_eff: {g2_eta:.5f}  (Δ from G₁: {g1_to_g2:+.5f}  "
      f"{'↑' if g1_to_g2 > 0 else '↓'})")
print(f"  G₃ mean η_eff: {g3_eta:.5f}  (Δ from G₂: {g2_to_g3:+.5f}  "
      f"{'↑' if g2_to_g3 > 0 else '↓'})")
print(f"  G₄ mean η_eff: {g4_eta:.5f}  (Δ from G₃: {g3_to_g4:+.5f}  "
      f"{'↑' if g3_to_g4 > 0 else '↓'})")
print()
print(f"  Pre-convergence windows used for Spearman: {n_pre_conv}/{N_WINDOWS}")
print(f"  Spearman ρ: {spear_rho:.4f}  p-value: {spear_p:.4f}")
print(f"    {'PASS ✓' if spearman_pass else 'FAIL ✗'}  "
      f"(ρ>0 AND p<0.05 required)")
print(f"  Monotonic transitions: {transitions_monotonic}/3 "
      f"(G₁→G₂, G₂→G₃, G₃→G₄)")
print(f"    {'PASS ✓' if monotonic_pass else 'FAIL ✗'}  (≥3 required)")
print()

if overall_evidence:
    verdict = (
        "Simulation evidence consistent with γ>1: effective learning rate "
        "increases monotonically with graph enrichment, and Spearman correlation "
        f"ρ={spear_rho:.4f} (p={spear_p:.4f}) confirms the trend across windows. "
        "This is mechanism-level evidence, not direct γ measurement. "
        "EXP-G1 with real multi-domain data is the measurement experiment."
    )
    verdict_short = "MECHANISM EVIDENCE CONSISTENT WITH γ>1"
elif spearman_pass:
    verdict = (
        f"Partial evidence: Spearman ρ={spear_rho:.4f} (p={spear_p:.4f}) supports "
        f"trend, but only {transitions_monotonic}/3 transitions are monotonic. "
        "Enrichment effect is present but not uniformly monotonic across transitions."
    )
    verdict_short = "PARTIAL EVIDENCE — Spearman pass, monotonic partial"
elif monotonic_pass:
    verdict = (
        f"Partial evidence: {transitions_monotonic}/3 transitions monotonic, "
        f"but Spearman ρ={spear_rho:.4f} (p={spear_p:.4f}) does not reach significance. "
        "Convergence rate does not increase significantly with graph maturity."
    )
    verdict_short = "PARTIAL EVIDENCE — monotonic pass, Spearman fail"
else:
    verdict = (
        "No evidence: convergence rate does not increase with graph maturity. "
        f"Spearman ρ={spear_rho:.4f} (p={spear_p:.4f}), "
        f"monotonic transitions={transitions_monotonic}/3. "
        "The mechanism assumed by γ>1 is not confirmed in simulation."
    )
    verdict_short = "NO EVIDENCE — convergence rate does not increase with graph maturity"

print(f"  VERDICT: {verdict_short}")
print()
print(f"  {verdict}")

# ---------------------------------------------------------------------------
# Build full error trajectory (across all 5000 decisions)
# ---------------------------------------------------------------------------

full_error_traj = np.concatenate([
    per_window_mean_errors[wi] for wi in range(N_WINDOWS)
])   # shape (5000,)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "experiment":      "BRIDGE-B-PHASE-C",
    "domain_config":   DOMAIN_CONFIG,
    "ontology":        {"C": C, "A": A, "d": d},
    "n_seeds":         N_SEEDS,
    "n_decisions":     N_DECISIONS,
    "window_size":     WINDOW_SIZE,
    "n_windows":       N_WINDOWS,
    "eta":             ETA,
    "eps":             EPS,
    "e0_base":         E0_BASE,
    "rho":             RHO,
    "n_eff":           N_EFF,
    "window_g_levels": WINDOW_G,
    "window_decisions": WINDOW_DECISIONS,
    "windows": {
        f"W{wi}": {
            "g_level":         WINDOW_G[wi],
            "t_start":         WINDOW_DECISIONS[wi][0],
            "t_end":           WINDOW_DECISIONS[wi][1],
            "eta_eff_mean":    float(window_eta_eff[wi]),
            "eta_eff_std":     float(window_eta_std[wi]),
            "r2_mean":         float(window_r2_mean[wi]),
            "n_active_cells":  int(window_n_active[wi].item()),
            "init_error_mean": float(per_cell_init_error[wi].mean()),
            "mean_error_traj": per_window_mean_errors[wi].tolist(),
        }
        for wi in range(N_WINDOWS)
    },
    "g_level_summary": {
        "G1": {"mean_eta_eff": g1_eta, "windows": [0]},
        "G2": {"mean_eta_eff": g2_eta, "windows": [1, 2]},
        "G3": {"mean_eta_eff": g3_eta, "windows": [3, 4, 5]},
        "G4": {"mean_eta_eff": g4_eta, "windows": [6, 7, 8, 9]},
    },
    "transitions": {
        "G1_to_G2": g1_to_g2,
        "G2_to_G3": g2_to_g3,
        "G3_to_G4": g3_to_g4,
        "monotonic_count": transitions_monotonic,
    },
    "spearman": {
        "rho": float(spear_rho) if not np.isnan(spear_rho) else None,
        "p_value": float(spear_p) if not np.isnan(spear_p) else None,
        "n_windows_used": int(n_pre_conv),
        "pass": bool(spearman_pass),
    },
    "gates": {
        "spearman_pass":  bool(spearman_pass),
        "monotonic_pass": bool(monotonic_pass),
        "overall_evidence": bool(overall_evidence),
    },
    "verdict": verdict,
    "verdict_short": verdict_short,
    "full_error_traj_downsampled": full_error_traj[::10].tolist(),   # every 10th
    "p2_reference": p2_n_converge,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"\nResults saved → {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

import subprocess
charts_path = Path(__file__).parent / "charts.py"
subprocess.run(
    [sys.executable, str(charts_path)],
    check=True, cwd=str(_REPO_ROOT),
    env={**os.environ, "PYTHONUTF8": "1"},
)
