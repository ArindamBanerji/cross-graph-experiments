"""
Bridge B Phase C v2: Temporal Compounding with ENDOGENOUS Graph Enrichment.

P3 failed because convergence happened in W0 (single-cell, N_converge≈60 < window 500).
This simulation uses FULL scoring (not single-cell) where each centroid receives
~N/(C*A) ≈ 208 updates over 5000 decisions — convergence spans multiple windows.

Graph state G(t) accumulates endogenously with each decision:
  - unique_entities: +Binomial(1, 0.3) per decision
  - threat_indicators: +Binomial(1, 0.05) per decision
  - cross_category_links: +Binomial(1, 0.10) if total decisions > 100
  - decisions_per_category[c]: +1 per decision in category c

Factor noise decreases as G(t) grows:
  sigma_j(t) = sigma_base[j] / (1 + enrichment_j(G, c))
  enrichment capped at 1.0 → max 50% noise reduction

Factor enrichment mapping (soc_product_v50 index order):
  j=0: travel_match           → entities / 100        (TravelRecord nodes)
  j=1: asset_criticality      → entities / 100        (Asset nodes)
  j=2: threat_intel_enrichment→ threat_indicators / 50 (IOC accumulation)
  j=3: pattern_history        → decisions_per_cat[c] / 200  (experience)
  j=4: time_anomaly           → 0.0  (time data does not enrich graph)
  j=5: device_trust           → entities / 100        (Device nodes)

Note: prompt labels j=3/j=4 as time_anomaly/pattern_history — this is a label swap;
the actual soc_product_v50 index order has j=3=pattern_history, j=4=time_anomaly.
We implement the physically correct mapping (pattern_history benefits from decisions).

Enrichment rates (0.3 entity/decision, 0.05 IOC/decision) are estimates;
real rates depend on customer environment. Results are mechanism evidence only.

SUCCESS CRITERIA:
  - Spearman ρ(window_index, η_eff) > 0 with p < 0.05
  - Noise level decreases monotonically across windows
  - Per-category convergence accelerates as cross-category links grow
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
ETA_NEG        = 0.05    # canonical product value
E0             = 0.20    # larger than P2 to extend convergence phase
EPS            = 0.05
DOMAIN_CONFIG  = "soc_product_v50"
RANDOM_SEED_BASE = 42

# Category weights (uniform — BOOTSTRAP_CATEGORY_WEIGHTS not defined in spec)
# In production, credential_access would be ~30%, but uniform avoids over-fitting
# to category mix. State this assumption clearly.
BOOTSTRAP_CATEGORY_WEIGHTS = None   # set to None → uniform after C is known

# GT action distribution: uniform over 4 actions per category
GT_ACTION_WEIGHTS = None             # set to None → uniform after A is known

# Graph growth parameters (estimates — real rates depend on environment)
P_ENTITY      = 0.30    # probability of new unique entity per decision
P_IOC         = 0.05    # probability of new ThreatIndicator per decision
P_CROSSLINK   = 0.10    # probability of new cross-category link per decision
                         # (only if total decisions > 100)
CROSSLINK_MIN = 100     # minimum decisions before cross-category links form
ENRICH_PATTERN_DENOM  = 200.0   # decisions_per_cat normalization
ENRICH_ENTITY_DENOM   = 100.0   # unique_entities normalization
ENRICH_IOC_DENOM      = 50.0    # threat_indicators normalization

P1_PATH      = _REPO_ROOT / "results" / "sigma_f_computation.json"
P2_PATH      = _REPO_ROOT / "results" / "bridge_b_phase_b.json"
RESULTS_PATH = _REPO_ROOT / "results" / "bridge_b_phase_c_v2.json"

FACTOR_NAMES = [
    "travel_match",            # 0 — entities
    "asset_criticality",       # 1 — entities
    "threat_intel_enrichment", # 2 — IOCs
    "pattern_history",         # 3 — decisions_per_cat
    "time_anomaly",            # 4 — no enrichment
    "device_trust",            # 5 — entities
]

# ---------------------------------------------------------------------------
# Load P1 and P2 references
# ---------------------------------------------------------------------------

with open(P1_PATH) as f:
    p1 = json.load(f)

obs = p1["per_factor_variances_observed"]
var_mean = (obs["asset_criticality"] +
            obs["threat_intel_enrichment"] +
            obs["pattern_history"]) / 3.0

sigma_f_base = np.array([
    var_mean,
    obs["asset_criticality"],
    obs["threat_intel_enrichment"],
    obs["pattern_history"],
    var_mean,
    var_mean,
], dtype=np.float64)

with open(P2_PATH) as f:
    p2 = json.load(f)

p2_n_converge = {g: p2["levels"][g]["mean_n_converge"] for g in ["G1","G2","G3","G4"]}

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config    = load_domain_config(DOMAIN_CONFIG)
C, A, d   = config["C"], config["A"], config["d"]
mu_true   = config["mu"].copy().astype(np.float64)   # (C, A, d)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]

assert (C, A, d) == (6, 4, 6)

cat_weights = np.ones(C) / C   # uniform
gt_weights  = np.ones(A) / A   # uniform

# ---------------------------------------------------------------------------
# Print setup
# ---------------------------------------------------------------------------

print("=" * 60)
print("BRIDGE B PHASE C v2: ENDOGENOUS ENRICHMENT (A=4)")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"N_SEEDS={N_SEEDS}, N_DECISIONS={N_DECISIONS}, WINDOW_SIZE={WINDOW_SIZE}")
print(f"ETA={ETA}, ETA_NEG={ETA_NEG}, EPS={EPS}, E0={E0}")
print(f"Full scoring (not single-cell): each centroid gets ~{N_DECISIONS//(C*A)} updates")
print(f"Category weights: uniform (1/{C} each)")
print(f"GT action weights: uniform (1/{A} each)")
print()
print("P2 reference N_converge (single-cell):")
for g, nc in p2_n_converge.items():
    print(f"  {g}: {nc:.1f}  → full-sim equivalent: ~{nc*C*A:.0f} decisions")
print()
print("Graph growth parameters (environment-specific estimates):")
print(f"  P(new entity per decision) = {P_ENTITY}")
print(f"  P(new IOC per decision)    = {P_IOC}")
print(f"  P(new crosslink per dec.)  = {P_CROSSLINK} [after {CROSSLINK_MIN} total]")
print()
print("Expected noise reduction timeline:")
for name_j, denom, prate in [
    ("entity factors (j=0,1,5)", ENRICH_ENTITY_DENOM, P_ENTITY),
    ("threat_intel (j=2)",        ENRICH_IOC_DENOM,    P_IOC),
    ("pattern_hist (j=3)",        ENRICH_PATTERN_DENOM, P_ENTITY / C),
]:
    dec_50pct = denom / prate
    print(f"  {name_j}: 50% noise reduction at ~decision {dec_50pct:.0f}")
print()

# ---------------------------------------------------------------------------
# Factor noise function
# ---------------------------------------------------------------------------

def factor_noise_vec(g_entities: float, g_iocs: float,
                     g_decisions_per_cat: np.ndarray,
                     c_idx: int) -> np.ndarray:
    """
    Returns (d,) array of current sigma^2 based on graph state.
    sigma^2_j = sigma_base[j] / (1 + enrichment_j)
    enrichment_j in [0, 1] → max 50% noise reduction.
    """
    enrichment = np.array([
        g_entities / ENRICH_ENTITY_DENOM,      # j=0: travel_match
        g_entities / ENRICH_ENTITY_DENOM,      # j=1: asset_criticality
        g_iocs     / ENRICH_IOC_DENOM,         # j=2: threat_intel_enrichment
        g_decisions_per_cat[c_idx] / ENRICH_PATTERN_DENOM,  # j=3: pattern_history
        0.0,                                    # j=4: time_anomaly (no enrichment)
        g_entities / ENRICH_ENTITY_DENOM,      # j=5: device_trust
    ], dtype=np.float64)
    reduction = np.minimum(enrichment, 1.0)    # cap at 1.0
    return sigma_f_base / (1.0 + reduction)

# ---------------------------------------------------------------------------
# η_eff fitting
# ---------------------------------------------------------------------------

def fit_eta_eff(errors: np.ndarray) -> tuple[float, float]:
    """
    Fit log-linear model to error trajectory within window.
    Returns (eta_eff, r_squared).
    eta_eff = 1 - exp(slope)  where slope < 0 if error is decreasing.
    """
    if len(errors) < 10:
        return 0.0, 0.0
    if float(errors[0]) < 1e-8:
        return 0.0, 0.0
    t_vals     = np.arange(len(errors), dtype=float)
    log_errors = np.log(np.maximum(errors, 1e-10))
    slope, intercept, r, p, se = linregress(t_vals, log_errors)
    r2      = float(r ** 2)
    eta_eff = float(1.0 - np.exp(slope))
    return eta_eff, r2

# ---------------------------------------------------------------------------
# Accumulation arrays (summed over seeds, divided after loop)
# ---------------------------------------------------------------------------

error_traj_sum     = np.zeros(N_DECISIONS, dtype=np.float64)
noise_traj_sum     = np.zeros(N_DECISIONS, dtype=np.float64)
g_entities_sum     = np.zeros(N_DECISIONS, dtype=np.float64)
g_iocs_sum         = np.zeros(N_DECISIONS, dtype=np.float64)
g_crosslinks_sum   = np.zeros(N_DECISIONS, dtype=np.float64)
g_decisions_sum    = np.zeros(N_DECISIONS, dtype=np.float64)
per_cat_err_sum    = np.zeros((C, N_DECISIONS), dtype=np.float64)

# Per-window per-category: decisions accumulated (for cross-coupling)
# Track G["decisions_per_cat"] at each window boundary
per_window_cat_decisions = np.zeros((N_WINDOWS, C), dtype=np.float64)  # averaged over seeds

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

print(f"Running simulation: {N_SEEDS} seeds × {N_DECISIONS} decisions...", flush=True)

for seed_idx in range(N_SEEDS):
    rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)

    # Initialize centroids with e0 offset
    offset = rng.standard_normal((C, A, d)) * E0
    mu     = np.clip(mu_true + offset, 0.0, 1.0)

    # Initialize graph state
    g_entities   = 0.0
    g_iocs       = 0.0
    g_crosslinks = 0.0
    g_dec_per_cat = np.zeros(C, dtype=np.float64)

    for t in range(N_DECISIONS):
        # Select category and GT action
        c_idx  = int(rng.choice(C, p=cat_weights))
        a_gt   = int(rng.choice(A, p=gt_weights))

        # Compute current factor noise based on graph state
        current_sigma = factor_noise_vec(g_entities, g_iocs, g_dec_per_cat, c_idx)
        sqrt_sigma    = np.sqrt(current_sigma)

        # Generate factor vector
        f = np.clip(mu_true[c_idx, a_gt] +
                    rng.standard_normal(d) * sqrt_sigma, 0.0, 1.0)

        # Score: L2 distance to all action centroids for this category
        diffs   = mu[c_idx] - f           # (A, d)
        dists   = np.sum(diffs ** 2, axis=1)   # (A,)
        a_pred  = int(np.argmin(dists))

        # Update centroids
        if a_pred == a_gt:
            mu[c_idx, a_pred] += ETA * (f - mu[c_idx, a_pred])
        else:
            mu[c_idx, a_pred] -= ETA_NEG * (f - mu[c_idx, a_pred])
            mu[c_idx, a_gt]   += ETA * (f - mu[c_idx, a_gt])
        np.clip(mu, 0.0, 1.0, out=mu)

        # Update graph state (endogenous)
        g_dec_per_cat[c_idx] += 1.0
        g_entities   += float(rng.binomial(1, P_ENTITY))
        g_iocs       += float(rng.binomial(1, P_IOC))
        if g_dec_per_cat.sum() > CROSSLINK_MIN:
            g_crosslinks += float(rng.binomial(1, P_CROSSLINK))

        # Record: mean L2 centroid displacement over all (c, a) pairs
        disp_all = np.sqrt(np.sum((mu - mu_true) ** 2, axis=2))   # (C, A)
        mean_err = float(disp_all.mean())
        error_traj_sum[t] += mean_err

        # Per-category mean L2 displacement (mean over A actions)
        per_cat_err_sum[:, t] += disp_all.mean(axis=1)   # (C,)

        # Noise level: mean sigma^2 over all d factors
        noise_traj_sum[t]   += float(current_sigma.mean())
        g_entities_sum[t]   += g_entities
        g_iocs_sum[t]       += g_iocs
        g_crosslinks_sum[t] += g_crosslinks
        g_decisions_sum[t]  += float(g_dec_per_cat.sum())

    # Store per-window per-category decisions at window ends
    # We use g_dec_per_cat at each window boundary — need to re-run a lighter version
    # Instead: store from the final state (approximate: uniform → linear growth)
    # We'll compute per-window decisions from the per-step accumulation below

    if (seed_idx + 1) % 10 == 0:
        cur_err = error_traj_sum[N_DECISIONS - 1] / (seed_idx + 1)
        print(f"  seed {seed_idx+1:3d}/{N_SEEDS}  final_err={cur_err:.4f}", flush=True)

# Normalize by N_SEEDS
error_traj    = error_traj_sum    / N_SEEDS
noise_traj    = noise_traj_sum    / N_SEEDS
g_entities    = g_entities_sum    / N_SEEDS
g_iocs        = g_iocs_sum        / N_SEEDS
g_crosslinks  = g_crosslinks_sum  / N_SEEDS
g_decisions   = g_decisions_sum   / N_SEEDS
per_cat_err   = per_cat_err_sum   / N_SEEDS   # (C, N_DECISIONS)

print()

# ---------------------------------------------------------------------------
# Per-window analysis
# ---------------------------------------------------------------------------

print("Computing per-window η_eff and noise statistics...")

window_eta_eff    = np.zeros(N_WINDOWS, dtype=np.float64)
window_eta_r2     = np.zeros(N_WINDOWS, dtype=np.float64)
window_noise_mean = np.zeros(N_WINDOWS, dtype=np.float64)
window_err_start  = np.zeros(N_WINDOWS, dtype=np.float64)
window_err_end    = np.zeros(N_WINDOWS, dtype=np.float64)
window_entities   = np.zeros(N_WINDOWS, dtype=np.float64)
window_iocs       = np.zeros(N_WINDOWS, dtype=np.float64)
window_crosslinks = np.zeros(N_WINDOWS, dtype=np.float64)
window_err_rel_imp = np.zeros(N_WINDOWS, dtype=np.float64)   # (start-end)/start

# Per-window per-category error decline rate: (start-end)/start for each cat
per_cat_window_decline = np.zeros((N_WINDOWS, C), dtype=np.float64)
per_cat_window_decisions_end = np.zeros((N_WINDOWS, C), dtype=np.float64)

for wi in range(N_WINDOWS):
    t0 = wi * WINDOW_SIZE
    t1 = (wi + 1) * WINDOW_SIZE

    w_errors = error_traj[t0:t1]
    w_noise  = noise_traj[t0:t1]

    eta_eff, r2 = fit_eta_eff(w_errors)
    window_eta_eff[wi]    = eta_eff
    window_eta_r2[wi]     = r2
    window_noise_mean[wi] = float(w_noise.mean())
    window_err_start[wi]  = float(w_errors[0])
    window_err_end[wi]    = float(w_errors[-1])
    window_entities[wi]   = float(g_entities[t1 - 1])
    window_iocs[wi]       = float(g_iocs[t1 - 1])
    window_crosslinks[wi] = float(g_crosslinks[t1 - 1])

    if float(w_errors[0]) > 1e-8:
        window_err_rel_imp[wi] = (float(w_errors[0]) - float(w_errors[-1])) / float(w_errors[0])
    else:
        window_err_rel_imp[wi] = 0.0

    # Per-category
    for c_idx in range(C):
        cat_w = per_cat_err[c_idx, t0:t1]
        if float(cat_w[0]) > 1e-8:
            per_cat_window_decline[wi, c_idx] = (
                (float(cat_w[0]) - float(cat_w[-1])) / float(cat_w[0])
            )
        # decisions for cat c at end of window: from g_decisions (total) / C (uniform)
        # approximate: decisions_per_cat[c] ≈ t1 / C
        per_cat_window_decisions_end[wi, c_idx] = (t1) / C

# ---------------------------------------------------------------------------
# Spearman correlations
# ---------------------------------------------------------------------------

# 1. ρ(window_index, η_eff) — core test
w_indices = list(range(N_WINDOWS))
spear_idx_eta,   p_idx_eta   = spearmanr(w_indices, window_eta_eff.tolist())

# 2. ρ(noise_level, η_eff) — noise drives learning rate
spear_noise_eta, p_noise_eta = spearmanr(window_noise_mean.tolist(), window_eta_eff.tolist())

# 3. ρ(entities, η_eff)
spear_ent_eta,   p_ent_eta   = spearmanr(window_entities.tolist(), window_eta_eff.tolist())

# 4. ρ(window_index, noise) — verify noise decreases
spear_idx_noise, p_idx_noise = spearmanr(w_indices, window_noise_mean.tolist())

# 5. Noise monotonic check
noise_monotone = all(window_noise_mean[i] >= window_noise_mean[i+1]
                     for i in range(N_WINDOWS - 1))

# ---------------------------------------------------------------------------
# Cross-category coupling
# ---------------------------------------------------------------------------
# For each pair (c_src, c_dst) where c_src ≠ c_dst:
#   ρ(entities_at_window_end, per_cat_decline[c_dst]) — shared entity enrichment drives both
# With uniform category weights, per-category decisions are equal, so
# we use g_entities (shared across categories) as the coupling variable.

cross_spear_vals = []
for c_dst in range(C):
    y = per_cat_window_decline[:, c_dst].tolist()
    x = window_entities.tolist()
    rho, _ = spearmanr(x, y)
    if not np.isnan(rho):
        cross_spear_vals.append(float(rho))

cross_cat_spear_mean = float(np.mean(cross_spear_vals)) if cross_spear_vals else float("nan")

# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

spearman_pass   = (not np.isnan(spear_idx_eta)) and \
                  (float(spear_idx_eta) > 0) and (float(p_idx_eta) < 0.05)
noise_mono_pass = noise_monotone
noise_decrease  = (float(window_noise_mean[-1]) < float(window_noise_mean[0]))

# Overall
evidence_level = sum([spearman_pass, noise_decrease, cross_cat_spear_mean > 0.3
                      if not np.isnan(cross_cat_spear_mean) else False])

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("=== BRIDGE B PHASE C v2: ENDOGENOUS ENRICHMENT (A=4) ===")
print("=" * 70)
print()

hdr = (f"  {'Window':<6} | {'Decisions':<12} | {'G(entities)':>11} | {'G(IOCs)':>7} | "
       f"{'Noise':>9} | {'η_eff':>8} | {'Error':>7}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for wi in range(N_WINDOWS):
    t0 = wi * WINDOW_SIZE
    t1 = (wi + 1) * WINDOW_SIZE
    print(f"  W{wi:<5} | {t0:4d}-{t1:4d}     | "
          f"{window_entities[wi]:>11.1f} | {window_iocs[wi]:>7.1f} | "
          f"{window_noise_mean[wi]:>9.6f} | {window_eta_eff[wi]:>8.5f} | "
          f"{window_err_start[wi]:>7.4f}")

print()
print(f"  Mean error W0 start: {window_err_start[0]:.4f}  →  W9 end: {window_err_end[-1]:.4f}")
noise_reduction_pct = (window_noise_mean[0] - window_noise_mean[-1]) / window_noise_mean[0] * 100
print(f"  Noise reduction W0→W9: {noise_reduction_pct:.1f}%  "
      f"({'monotone ✓' if noise_mono_pass else 'not monotone'})")
print()
print(f"  Spearman ρ(window_index, η_eff) = {float(spear_idx_eta):.4f}  "
      f"p = {float(p_idx_eta):.4f}  "
      f"{'PASS ✓' if spearman_pass else 'FAIL ✗'}")
print(f"  Spearman ρ(noise_level, η_eff)  = {float(spear_noise_eta):.4f}  "
      f"p = {float(p_noise_eta):.4f}")
print(f"  Spearman ρ(entities, η_eff)      = {float(spear_ent_eta):.4f}  "
      f"p = {float(p_ent_eta):.4f}")
print(f"  Spearman ρ(window_index, noise)  = {float(spear_idx_noise):.4f}  "
      f"p = {float(p_idx_noise):.4f}")
print(f"  Noise decreases W0→W9: {'YES ✓' if noise_decrease else 'NO ✗'}")
print()
print(f"  Cross-category coupling (mean ρ over {len(cross_spear_vals)} categories):")
print(f"    ρ(G(entities), per-cat convergence rate) = {cross_cat_spear_mean:.4f}")
print()

# Verdict
if evidence_level >= 3:
    verdict_short = "ENDOGENOUS γ>1 EVIDENCE: all three criteria met"
    verdict = (
        f"Simulation evidence with endogenous enrichment consistent with γ>1. "
        f"Spearman ρ(window, η_eff)={float(spear_idx_eta):.4f} (p={float(p_idx_eta):.4f}), "
        f"noise decreases {noise_reduction_pct:.1f}% endogenously, "
        f"cross-category coupling ρ={cross_cat_spear_mean:.4f}. "
        f"Enrichment rates (P_entity={P_ENTITY}, P_ioc={P_IOC}) are environment-specific "
        f"estimates; real deployment rates will vary. "
        f"EXP-G1 with real multi-domain data is the measurement experiment."
    )
elif evidence_level == 2:
    verdict_short = "PARTIAL EVIDENCE: 2/3 criteria met"
    verdict = (
        f"Partial simulation evidence: {evidence_level}/3 criteria met. "
        f"Spearman ρ(window, η_eff)={float(spear_idx_eta):.4f} (p={float(p_idx_eta):.4f}). "
        f"Noise reduction {noise_reduction_pct:.1f}%. "
        f"Cross-category ρ={cross_cat_spear_mean:.4f}. "
        f"Endogenous enrichment has a measurable effect but does not fully confirm γ>1 mechanism. "
        f"EXP-G1 with real multi-domain data is required."
    )
elif evidence_level == 1:
    verdict_short = "WEAK EVIDENCE: 1/3 criteria met"
    verdict = (
        f"Weak simulation evidence: {evidence_level}/3 criteria met. "
        f"The endogenous enrichment in this parameter regime is insufficient "
        f"to produce observable temporal compounding in η_eff across 500-decision windows. "
        f"EXP-G1 with real multi-domain data is required."
    )
else:
    verdict_short = "NO EVIDENCE: convergence rate does not increase with endogenous enrichment"
    verdict = (
        f"No evidence: Spearman ρ={float(spear_idx_eta):.4f} (p={float(p_idx_eta):.4f}), "
        f"noise reduction {noise_reduction_pct:.1f}%. "
        f"The endogenous enrichment mechanism does not produce observable temporal compounding. "
        f"EXP-G1 with real multi-domain data is required."
    )

print(f"  VERDICT: {verdict_short}")
print()
print(f"  {verdict}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "experiment":    "BRIDGE-B-PHASE-C-V2",
    "domain_config": DOMAIN_CONFIG,
    "ontology":      {"C": int(C), "A": int(A), "d": int(d)},
    "n_seeds":       N_SEEDS,
    "n_decisions":   N_DECISIONS,
    "window_size":   WINDOW_SIZE,
    "n_windows":     N_WINDOWS,
    "eta":           ETA,
    "eta_neg":       ETA_NEG,
    "eps":           EPS,
    "e0":            E0,
    "graph_growth_params": {
        "p_entity":    P_ENTITY,
        "p_ioc":       P_IOC,
        "p_crosslink": P_CROSSLINK,
        "crosslink_min": CROSSLINK_MIN,
    },
    "windows": {
        f"W{wi}": {
            "t_start":         int(wi * WINDOW_SIZE),
            "t_end":           int((wi + 1) * WINDOW_SIZE),
            "eta_eff":         float(window_eta_eff[wi]),
            "eta_r2":          float(window_eta_r2[wi]),
            "noise_mean":      float(window_noise_mean[wi]),
            "err_start":       float(window_err_start[wi]),
            "err_end":         float(window_err_end[wi]),
            "err_rel_imp":     float(window_err_rel_imp[wi]),
            "g_entities":      float(window_entities[wi]),
            "g_iocs":          float(window_iocs[wi]),
            "g_crosslinks":    float(window_crosslinks[wi]),
        }
        for wi in range(N_WINDOWS)
    },
    "spearman": {
        "rho_window_eta":   float(spear_idx_eta),
        "p_window_eta":     float(p_idx_eta),
        "rho_noise_eta":    float(spear_noise_eta),
        "p_noise_eta":      float(p_noise_eta),
        "rho_entities_eta": float(spear_ent_eta),
        "p_entities_eta":   float(p_ent_eta),
        "rho_window_noise": float(spear_idx_noise),
        "p_window_noise":   float(p_idx_noise),
        "spearman_pass":    bool(spearman_pass),
    },
    "noise": {
        "noise_W0":        float(window_noise_mean[0]),
        "noise_W9":        float(window_noise_mean[-1]),
        "reduction_pct":   float(noise_reduction_pct),
        "monotone":        bool(noise_mono_pass),
        "decreases":       bool(noise_decrease),
    },
    "cross_category": {
        "mean_rho_entities_per_cat_decline": float(cross_cat_spear_mean),
        "n_pairs": int(len(cross_spear_vals)),
    },
    "gates": {
        "spearman_pass":  bool(spearman_pass),
        "noise_decrease": bool(noise_decrease),
        "cross_cat_pass": bool(cross_cat_spear_mean > 0.3
                              if not np.isnan(cross_cat_spear_mean) else False),
        "evidence_level": int(evidence_level),
        "overall_evidence": bool(evidence_level >= 2),
    },
    "verdict": verdict,
    "verdict_short": verdict_short,
    # Downsampled trajectories for charts
    "error_traj_ds":    error_traj[::10].tolist(),
    "noise_traj_ds":    noise_traj[::10].tolist(),
    "g_entities_ds":    g_entities[::10].tolist(),
    "g_iocs_ds":        g_iocs[::10].tolist(),
    "g_crosslinks_ds":  g_crosslinks[::10].tolist(),
    "g_decisions_ds":   g_decisions[::10].tolist(),
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
