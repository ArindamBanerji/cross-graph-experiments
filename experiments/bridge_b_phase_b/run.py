"""
Bridge B Phase B: Graph Maturity G₁→G₄ with ρ-adjusted noise.

Measures whether richer graphs accelerate centroid convergence.
This is a single-cell convergence simulation — each (c, a) centroid is
simulated independently to isolate convergence rate from scoring interactions.

Depends on: P1 results in results/sigma_f_computation.json
  - Per-factor variances (3 measured, 3 extrapolated at mean)
  - η=0.05 confirmed

Factor index map (soc_product_v50):
  0: travel_match         (unmapped — extrapolated at mean variance)
  1: asset_criticality    (measured: 0.05045)
  2: threat_intel_enrichment (measured: 0.09687)  ← G4 applies 0.70× here
  3: pattern_history      (measured: 0.02511)
  4: time_anomaly         (unmapped — extrapolated at mean variance)
  5: device_trust         (unmapped — extrapolated at mean variance)

Four graph maturity levels:
  G1: Single SIEM, no enrichment — base noise
  G2: Two SIEMs (ρ=0.8) — noise / N_eff where N_eff = 2/(1+0.8) = 1.111
  G3: G2 noise + entity resolution — same noise, e₀ reduced by 15%
  G4: G2 noise + ThreatIndicators — threat_intel noise × 0.70, e₀ reduced by 15%
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SEEDS        = 50
N_DECISIONS    = 2000
ETA            = 0.05
EPS            = 0.05       # convergence threshold: max|μ_t − μ*| per component
E0_BASE        = 0.15       # initial error per component
RHO            = 0.8        # cross-source correlation
N_EFF          = 2.0 / (1.0 + RHO)   # = 1.111
THREAT_INTEL_IDX = 2        # index of threat_intel_enrichment in factor vector
DOMAIN_CONFIG  = "soc_product_v50"
RANDOM_SEED_BASE = 42

P1_RESULTS_PATH = _REPO_ROOT / "results" / "sigma_f_computation.json"
RESULTS_PATH    = _REPO_ROOT / "results" / "bridge_b_phase_b.json"

G_LEVELS     = ["G1", "G2", "G3", "G4"]
G_LABELS     = {
    "G1": "Single SIEM, no enrichment",
    "G2": "Two SIEMs (ρ=0.8)",
    "G3": "Two SIEMs + entity resolution",
    "G4": "Full enrichment + ThreatIndicators",
}

# ---------------------------------------------------------------------------
# Load P1 results — per-factor variances
# ---------------------------------------------------------------------------

with open(P1_RESULTS_PATH) as f:
    p1 = json.load(f)

# Build 6D per-factor variance vector
# Index order: [travel_match, asset_criticality, threat_intel_enrichment,
#               pattern_history, time_anomaly, device_trust]
observed = p1["per_factor_variances_observed"]
var_asset_crit  = observed["asset_criticality"]
var_threat_intel = observed["threat_intel_enrichment"]
var_pattern_hist = observed["pattern_history"]
var_mean_observed = (var_asset_crit + var_threat_intel + var_pattern_hist) / 3.0

sigma_f_base = np.array([
    var_mean_observed,    # 0: travel_match (extrapolated)
    var_asset_crit,       # 1: asset_criticality
    var_threat_intel,     # 2: threat_intel_enrichment
    var_pattern_hist,     # 3: pattern_history
    var_mean_observed,    # 4: time_anomaly (extrapolated)
    var_mean_observed,    # 5: device_trust (extrapolated)
], dtype=np.float64)

print("=" * 60)
print("BRIDGE B PHASE B: GRAPH MATURITY G₁→G₄ (A=4)")
print("=" * 60)
print(f"P1 source: {P1_RESULTS_PATH}")
print(f"tr(Σ_f) 6D extrapolated: {p1['tr_sigma_f_6d_extrapolated']:.4f}")
print(f"Mean observed variance: {var_mean_observed:.6f}")
print(f"Per-factor variance base vector (6D):")
factor_names = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "pattern_history", "time_anomaly", "device_trust"]
for i, (name, v) in enumerate(zip(factor_names, sigma_f_base)):
    src = "measured" if name in observed else "extrapolated"
    print(f"  [{i}] {name:<28}: {v:.6f}  ({src})")
print(f"N_EFF = 2/(1+ρ) = 2/(1+{RHO}) = {N_EFF:.4f}")
print()

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]
mu_config  = config["mu"].copy()    # shape (C, A, d)

assert mu_config.shape == (6, 4, 6)
assert A == 4

print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"N_SEEDS={N_SEEDS}, N_DECISIONS={N_DECISIONS}")
print(f"η={ETA}, ε={EPS}, e₀_base={E0_BASE}")
print(f"Total cells per G level: {N_SEEDS} × {C} × {A} = {N_SEEDS*C*A}")
print()

# ---------------------------------------------------------------------------
# Noise model functions
# ---------------------------------------------------------------------------

def compute_factor_noise(g_level: str) -> np.ndarray:
    """Per-factor variance vector for each G level."""
    if g_level == "G1":
        return sigma_f_base.copy()
    sigma_f = sigma_f_base / N_EFF     # G2 base reduction
    if g_level == "G4":
        sigma_f = sigma_f.copy()
        sigma_f[THREAT_INTEL_IDX] *= 0.70   # ThreatIndicator accumulation
    return sigma_f


def compute_e0(g_level: str) -> float:
    if g_level in ("G1", "G2"):
        return E0_BASE
    return E0_BASE * 0.85   # entity resolution: 15% closer to true profile


# ---------------------------------------------------------------------------
# Pre-compute batch arrays
# ---------------------------------------------------------------------------
# Cells: N_SEEDS × C × A = 1200, indexed as seed*C*A + c*A + a
# mu_true_batch[i] = mu_config[c, a, :] for cell i
# This avoids Python loops over cells — only the time loop remains.

n_cells = N_SEEDS * C * A

mu_true_batch = np.zeros((n_cells, d), dtype=np.float64)
for seed in range(N_SEEDS):
    for c_idx in range(C):
        for a_idx in range(A):
            cell_i = seed * (C * A) + c_idx * A + a_idx
            mu_true_batch[cell_i] = mu_config[c_idx, a_idx, :]

# ---------------------------------------------------------------------------
# Simulate one G level — vectorized over all cells simultaneously
# ---------------------------------------------------------------------------

def simulate_g_level(g_level: str) -> dict:
    """
    Runs the single-cell convergence simulation for all 1200 cells simultaneously.

    Returns:
        n_converge: (n_cells,) array of first-passage decision index (1-based), or
                    N_DECISIONS if never converged within the window.
        mean_error_traj: (N_DECISIONS,) mean of max-component error across all cells.
    """
    sigma_f = compute_factor_noise(g_level)   # (d,)
    e0      = compute_e0(g_level)
    sqrt_sf = np.sqrt(sigma_f)                # (d,) std dev per factor

    rng = np.random.default_rng(RANDOM_SEED_BASE + G_LEVELS.index(g_level) * 1000)

    # Initialize: mu_0 = clip(mu_true + offset, 0, 1)
    offset = rng.standard_normal((n_cells, d)) * e0    # (n_cells, d)
    mu     = np.clip(mu_true_batch + offset, 0.0, 1.0)  # (n_cells, d)

    n_converge = np.full(n_cells, N_DECISIONS, dtype=np.int64)  # sentinel = N_DECISIONS
    converged  = np.zeros(n_cells, dtype=bool)

    mean_error_traj = np.zeros(N_DECISIONS, dtype=np.float64)

    for t in range(N_DECISIONS):
        # Draw factor vectors from true profile + noise
        noise = rng.standard_normal((n_cells, d)) * sqrt_sf    # (n_cells, d)
        f     = np.clip(mu_true_batch + noise, 0.0, 1.0)       # (n_cells, d)

        # Max-component error for each cell
        err = np.max(np.abs(mu - mu_true_batch), axis=1)       # (n_cells,)
        mean_error_traj[t] = float(err.mean())

        # Update convergence tracker: first t where err < EPS
        newly_converged = (~converged) & (err < EPS)
        n_converge[newly_converged] = t + 1   # 1-based decision index
        converged |= newly_converged

        # Pull update: mu += η × (f − μ)  (always correct in single-cell sim)
        mu += ETA * (f - mu)
        np.clip(mu, 0.0, 1.0, out=mu)

    return {
        "n_converge":        n_converge,
        "mean_error_traj":   mean_error_traj,
        "sigma_f":           sigma_f.tolist(),
        "e0":                e0,
    }


# ---------------------------------------------------------------------------
# Run all G levels
# ---------------------------------------------------------------------------

results_per_level: dict[str, dict] = {}

for g_level in G_LEVELS:
    print(f"Running {g_level}: {G_LABELS[g_level]} ...", flush=True)
    res = simulate_g_level(g_level)
    nc  = res["n_converge"]

    mean_nc   = float(nc.mean())
    std_nc    = float(nc.std())
    median_nc = float(np.median(nc))
    p2_5      = float(np.percentile(nc, 2.5))
    p97_5     = float(np.percentile(nc, 97.5))
    frac_conv = float(np.mean(nc < N_DECISIONS))

    results_per_level[g_level] = {
        "label":           G_LABELS[g_level],
        "sigma_f":         res["sigma_f"],
        "e0":              res["e0"],
        "mean_n_converge": mean_nc,
        "std_n_converge":  std_nc,
        "median_n_converge": median_nc,
        "p2_5":            p2_5,
        "p97_5":           p97_5,
        "frac_converged":  frac_conv,
        "mean_error_traj": res["mean_error_traj"].tolist(),
    }
    print(f"  mean N_conv={mean_nc:.1f}  median={median_nc:.1f}  "
          f"[{p2_5:.0f}, {p97_5:.0f}]  conv={frac_conv:.1%}")

# ---------------------------------------------------------------------------
# Cross-level metrics
# ---------------------------------------------------------------------------

mean_g1 = results_per_level["G1"]["mean_n_converge"]

reductions = {}
for g_level in ["G2", "G3", "G4"]:
    r = (mean_g1 - results_per_level[g_level]["mean_n_converge"]) / mean_g1 * 100.0
    reductions[g_level] = float(r)

# Monotonic check: G1 ≥ G2 ≥ G3 ≥ G4 (mean)
means = [results_per_level[g]["mean_n_converge"] for g in G_LEVELS]
monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

# Success criteria
g4_reduction     = reductions["G4"]
g2_reduction     = reductions["G2"]
gate_g4          = g4_reduction >= 30.0
gate_g2_range    = 5.0 <= g2_reduction <= 20.0
gate_monotonic   = monotonic
overall_pass     = gate_g4 and gate_monotonic

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

print()
print("=" * 85)
print("=== BRIDGE B PHASE B: GRAPH MATURITY G₁→G₄ (A=4) ===")
print("=" * 85)
print()
print(f"  {'Level':<4}  {'Description':<30}  {'N_conv mean':>12}  {'median':>7}  "
      f"{'95% CI':>14}  {'% conv':>8}  {'Reduction':>10}")
print("  " + "-" * 82)

for g_level in G_LEVELS:
    s = results_per_level[g_level]
    red_str = ("baseline" if g_level == "G1"
               else f"{reductions[g_level]:+.1f}%")
    ci_str = f"[{s['p2_5']:.0f}, {s['p97_5']:.0f}]"
    print(f"  {g_level:<4}  {s['label']:<30}  {s['mean_n_converge']:>12.1f}  "
          f"{s['median_n_converge']:>7.1f}  {ci_str:>14}  "
          f"{s['frac_converged']:>7.1%}  {red_str:>10}")

print()
print(f"  Monotonic (G₁ ≥ G₂ ≥ G₃ ≥ G₄):  {'YES ✓' if monotonic else 'NO ✗'}")
print(f"  G₄/G₁ reduction: {g4_reduction:.1f}%  (gate: ≥30%)  "
      f"{'PASS ✓' if gate_g4 else 'FAIL ✗'}")
print(f"  G₂/G₁ reduction: {g2_reduction:.1f}%  (validates ρ=0.8 model: 5–20%)  "
      f"{'PASS ✓' if gate_g2_range else 'OUTSIDE RANGE'}")
print(f"  OVERALL: {'PASS ✓' if overall_pass else 'FAIL ✗'}")

# Per-level noise summary
print()
print("  Noise model summary:")
print(f"  {'Level':<4}  {'e₀':>6}  {'σ²_mean':>10}  {'σ²_threat_intel':>18}")
for g_level in G_LEVELS:
    s = results_per_level[g_level]
    sf = np.array(s["sigma_f"])
    print(f"  {g_level:<4}  {s['e0']:>6.4f}  "
          f"{sf.mean():>10.6f}  {sf[THREAT_INTEL_IDX]:>18.6f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "experiment":       "BRIDGE-B-PHASE-B",
    "domain_config":    DOMAIN_CONFIG,
    "ontology":         {"C": C, "A": A, "d": d},
    "n_seeds":          N_SEEDS,
    "n_decisions":      N_DECISIONS,
    "n_cells_per_level": n_cells,
    "eta":              ETA,
    "eps":              EPS,
    "e0_base":          E0_BASE,
    "rho":              RHO,
    "n_eff":            N_EFF,
    "threat_intel_idx": THREAT_INTEL_IDX,
    "p1_source":        str(P1_RESULTS_PATH),
    "sigma_f_base":     sigma_f_base.tolist(),
    "factor_names":     factor_names,
    "levels":           results_per_level,
    "reductions":       reductions,
    "monotonic":        monotonic,
    "gates": {
        "g4_reduction_pct":   g4_reduction,
        "g4_reduction_pass":  gate_g4,
        "g2_reduction_pct":   g2_reduction,
        "g2_range_pass":      gate_g2_range,
        "monotonic_pass":     gate_monotonic,
        "overall_pass":       overall_pass,
    },
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as fh:
    # Truncate mean_error_traj to save space (keep every 10th point for charts)
    out_light = {k: v for k, v in output.items() if k != "levels"}
    out_light["levels"] = {}
    for g_level, s in results_per_level.items():
        out_light["levels"][g_level] = {
            k: (v[::10] if k == "mean_error_traj" else v)
            for k, v in s.items()
        }
    json.dump(out_light, fh, indent=2)

print(f"\nResults saved → {RESULTS_PATH}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

import subprocess
charts_path = Path(__file__).parent / "charts.py"
subprocess.run(
    [sys.executable, str(charts_path)],
    check=True, cwd=str(_REPO_ROOT),
    env={**__import__("os").environ, "PYTHONUTF8": "1"},
)
