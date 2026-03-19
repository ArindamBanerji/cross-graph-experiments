"""
Bridge B Phase C v3: Temporal Compounding via RE-CONVERGENCE Episodes.

γ>1 predicts that each distribution shift triggers a re-convergence episode
that is FASTER than the previous one, because the graph is richer.

Three phases × 1500 decisions each (total 4500):
  Phase 1 (0–1500):    Initial convergence from e₀=0.20. Graph sparse.
  Shift 1 at t=1500:   Random perturbation δ=0.10 per unit vector. Graph at medium level.
  Phase 2 (1500–3000): Re-convergence 1. Medium graph.
  Shift 2 at t=3000:   Another random perturbation δ=0.10. Graph at rich level.
  Phase 3 (3000–4500): Re-convergence 2. Richest graph.

γ>1 EVIDENCE if: N_reconv_2 < N_reconv_1 < N_initial

NOISE MODEL (progressive, no cap):
  σ²_j(t) = σ²_base_j / (1 + enrichment_j(G(t)))

  Denominators chosen for continuous noise reduction across all 3 phases:
    entity factors (j=0,1,5): ENTITY_DENOM = 1500  (50% reduction at 1500 entities)
    threat_intel  (j=2):      IOC_DENOM    = 500   (50% reduction at 500 IOCs)
    pattern_hist  (j=3):      PATTERN_DENOM= 1000  (50% reduction at 1000 dec/cat)
    time_anomaly  (j=4):      no enrichment
  No cap → enrichment > 1 allowed (noise keeps decreasing).

Expected noise at each shift:
  Shift 1 (t=1500): entities≈450, IOCs≈75   → mean σ² ≈ 0.04817 (12% below base)
  Shift 2 (t=3000): entities≈900, IOCs≈150  → mean σ² ≈ 0.04202 (23% below base)

EPS_RECONVERGE = 0.10 (mean L2 centroid displacement, achievable above floor ~0.082)

Enrichment rates (0.3 entity/decision, 0.05 IOC/decision) are environment-specific
estimates. Results are mechanism evidence only.

Key difference from P3/P3-REVISED:
  - Uses RE-CONVERGENCE episodes (each phase starts from known displacement)
  - Avoids η_eff fitting (compares first-passage times directly)
  - Controls for shift magnitude (δ=0.10 constant): Phase 2 vs 3 is a fair comparison
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SEEDS            = 50
PHASE_LEN          = 1500             # decisions per phase
N_PHASES           = 3
N_TOTAL            = N_PHASES * PHASE_LEN   # 4500
ETA                = 0.05
ETA_NEG            = 0.05
E0                 = 0.20             # initial offset per component
DELTA_SHIFT        = 0.10             # shift magnitude per unit vector in R^d
EPS_RECONVERGE     = 0.10            # mean L2 threshold for convergence detection
DOMAIN_CONFIG      = "soc_product_v50"
RANDOM_SEED_BASE   = 42

# Graph growth (same rates as P3-REVISED, different denominators for progressive reduction)
P_ENTITY   = 0.30
P_IOC      = 0.05
P_CROSSLINK= 0.10
CROSSLINK_MIN = 100

# Enrichment denominators (larger → slower reduction → keeps decreasing across all phases)
ENTITY_DENOM  = 1500.0
IOC_DENOM     = 500.0
PATTERN_DENOM = 1000.0
# No cap: enrichment can exceed 1.0 → noise keeps decreasing

# T_eval: evaluate error at fixed horizon after each shift (robust secondary metric)
T_EVAL = 300   # decisions after shift start

PHASE_STARTS = [0, PHASE_LEN, 2 * PHASE_LEN]   # [0, 1500, 3000]

P1_PATH      = _REPO_ROOT / "results" / "sigma_f_computation.json"
P2_PATH      = _REPO_ROOT / "results" / "bridge_b_phase_b.json"
RESULTS_PATH = _REPO_ROOT / "results" / "bridge_b_phase_c_v3.json"

FACTOR_NAMES = [
    "travel_match",            # 0: entity
    "asset_criticality",       # 1: entity
    "threat_intel_enrichment", # 2: IOC
    "pattern_history",         # 3: decisions/cat
    "time_anomaly",            # 4: no enrichment
    "device_trust",            # 5: entity
]

# ---------------------------------------------------------------------------
# Load P1 and P2
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

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config    = load_domain_config(DOMAIN_CONFIG)
C, A, d   = config["C"], config["A"], config["d"]
mu_true_0 = config["mu"].copy().astype(np.float64)   # (C, A, d) — base profiles

assert (C, A, d) == (6, 4, 6)

cat_weights = np.ones(C, dtype=np.float64) / C
gt_weights  = np.ones(A, dtype=np.float64) / A

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def factor_noise_vec(g_ent: float, g_ioc: float,
                     g_dec: np.ndarray, c_idx: int) -> np.ndarray:
    """
    Returns (d,) variance vector based on current graph state.
    No cap: enrichment can exceed 1.0.
    """
    enr = np.array([
        g_ent / ENTITY_DENOM,          # j=0: travel_match
        g_ent / ENTITY_DENOM,          # j=1: asset_criticality
        g_ioc / IOC_DENOM,             # j=2: threat_intel_enrichment
        g_dec[c_idx] / PATTERN_DENOM,  # j=3: pattern_history
        0.0,                            # j=4: time_anomaly (no enrichment)
        g_ent / ENTITY_DENOM,          # j=5: device_trust
    ], dtype=np.float64)
    return sigma_f_base / (1.0 + enr)


def mean_noise_all_cats(g_ent: float, g_ioc: float,
                        g_dec: np.ndarray) -> float:
    """Mean sigma² over all factors and all categories."""
    total = 0.0
    for c in range(C):
        total += factor_noise_vec(g_ent, g_ioc, g_dec, c).mean()
    return total / C


def apply_shift(mu_true: np.ndarray, delta: float, rng) -> np.ndarray:
    """
    Perturb each (c,a) centroid by delta along a random unit vector.
    Returns new mu_true. Centroids mu (learned) are NOT reset.
    """
    mu_new = mu_true.copy()
    for c in range(C):
        for a in range(A):
            direction = rng.standard_normal(d)
            direction /= np.linalg.norm(direction)
            mu_new[c, a] = np.clip(mu_true[c, a] + delta * direction, 0.0, 1.0)
    return mu_new


def mean_l2_error(mu: np.ndarray, mu_true: np.ndarray) -> float:
    """Mean L2 displacement over all (C, A) centroid pairs."""
    return float(np.sqrt(np.sum((mu - mu_true) ** 2, axis=2)).mean())

# ---------------------------------------------------------------------------
# Print setup
# ---------------------------------------------------------------------------

print("=" * 62)
print("BRIDGE B PHASE C v3: RE-CONVERGENCE COMPOUNDING (A=4)")
print("=" * 62)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"N_SEEDS={N_SEEDS}, PHASE_LEN={PHASE_LEN}, N_TOTAL={N_TOTAL}")
print(f"ETA={ETA}, ETA_NEG={ETA_NEG}, E0={E0}, DELTA={DELTA_SHIFT}")
print(f"EPS_RECONVERGE={EPS_RECONVERGE}  T_EVAL={T_EVAL}")
print()
print("Graph growth: P_entity=0.30, P_ioc=0.05  (environment-specific estimates)")
print(f"Enrichment denominators: entity={ENTITY_DENOM:.0f}, ioc={IOC_DENOM:.0f}, "
      f"pattern={PATTERN_DENOM:.0f}  (no cap)")
print()
print("Expected noise progression (mean σ² across all factors):")
for phase, (t, ent, ioc, dec_per_cat) in enumerate([
    (1500, 450, 75,  250),
    (3000, 900, 150, 500),
    (4500, 1350, 225, 750),
]):
    enrs = [ent/ENTITY_DENOM, ent/ENTITY_DENOM, ioc/IOC_DENOM,
            dec_per_cat/PATTERN_DENOM, 0.0, ent/ENTITY_DENOM]
    sigs = [sigma_f_base[j]/(1+enrs[j]) for j in range(6)]
    mns = float(np.mean(sigs))
    red = (1 - mns / float(sigma_f_base.mean())) * 100
    print(f"  Phase {phase+1} end (t={t:4d}): entities≈{ent:4d}, IOCs≈{ioc:3d}  "
          f"→ mean σ²={mns:.5f}  ({red:.1f}% reduction)")
print()

# ---------------------------------------------------------------------------
# Accumulation arrays (summed over seeds, normalized after)
# ---------------------------------------------------------------------------

error_traj_sum  = np.zeros(N_TOTAL, dtype=np.float64)
noise_traj_sum  = np.zeros(N_TOTAL, dtype=np.float64)
g_entities_sum  = np.zeros(N_TOTAL, dtype=np.float64)
g_iocs_sum      = np.zeros(N_TOTAL, dtype=np.float64)

# Per-phase sub-trajectories (from phase start) for Chart 2
phase_error_sum = np.zeros((N_PHASES, PHASE_LEN), dtype=np.float64)

# Per-seed first-passage times (sentinel = PHASE_LEN if not reached)
n_init_arr  = np.full(N_SEEDS, PHASE_LEN, dtype=np.int64)
n_rc1_arr   = np.full(N_SEEDS, PHASE_LEN, dtype=np.int64)
n_rc2_arr   = np.full(N_SEEDS, PHASE_LEN, dtype=np.int64)

# Per-seed error at T_EVAL decisions after each shift
err_init_at_T  = np.zeros(N_SEEDS, dtype=np.float64)
err_rc1_at_T   = np.zeros(N_SEEDS, dtype=np.float64)
err_rc2_at_T   = np.zeros(N_SEEDS, dtype=np.float64)

# Noise level at shift points (per seed)
noise_shift1_arr = np.zeros(N_SEEDS, dtype=np.float64)
noise_shift2_arr = np.zeros(N_SEEDS, dtype=np.float64)

# Accuracy per phase per seed
accuracy_arr = np.zeros((N_SEEDS, N_PHASES), dtype=np.float64)

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

print(f"Running {N_SEEDS} seeds × {N_TOTAL} decisions...", flush=True)

for seed_idx in range(N_SEEDS):
    rng = np.random.default_rng(RANDOM_SEED_BASE + seed_idx)

    # Initialize
    mu_true = mu_true_0.copy()
    offset  = rng.standard_normal((C, A, d)) * E0
    mu      = np.clip(mu_true + offset, 0.0, 1.0)

    g_ent     = 0.0
    g_ioc     = 0.0
    g_cross   = 0.0
    g_dec     = np.zeros(C, dtype=np.float64)

    # Convergence flags and times
    found_init = found_rc1 = found_rc2 = False

    # Per-phase accuracy
    phase_correct = np.zeros(N_PHASES, dtype=np.int64)
    phase_total   = np.zeros(N_PHASES, dtype=np.int64)

    for t in range(N_TOTAL):
        phase_idx = t // PHASE_LEN

        # ---- Distribution shifts ----
        if t == PHASE_LEN:
            noise_shift1_arr[seed_idx] = mean_noise_all_cats(g_ent, g_ioc, g_dec)
            mu_true = apply_shift(mu_true, DELTA_SHIFT, rng)

        if t == 2 * PHASE_LEN:
            noise_shift2_arr[seed_idx] = mean_noise_all_cats(g_ent, g_ioc, g_dec)
            mu_true = apply_shift(mu_true, DELTA_SHIFT, rng)

        # ---- Decision ----
        c_idx = int(rng.choice(C, p=cat_weights))
        a_gt  = int(rng.choice(A, p=gt_weights))

        sigma   = factor_noise_vec(g_ent, g_ioc, g_dec, c_idx)
        f       = np.clip(mu_true[c_idx, a_gt] +
                          rng.standard_normal(d) * np.sqrt(sigma), 0.0, 1.0)

        diffs   = mu[c_idx] - f
        dists   = np.sum(diffs ** 2, axis=1)
        a_pred  = int(np.argmin(dists))

        if a_pred == a_gt:
            mu[c_idx, a_pred] += ETA * (f - mu[c_idx, a_pred])
            phase_correct[phase_idx] += 1
        else:
            mu[c_idx, a_pred] -= ETA_NEG * (f - mu[c_idx, a_pred])
            mu[c_idx, a_gt]   += ETA * (f - mu[c_idx, a_gt])
        np.clip(mu, 0.0, 1.0, out=mu)

        # ---- Graph state ----
        g_dec[c_idx] += 1.0
        g_ent  += float(rng.binomial(1, P_ENTITY))
        g_ioc  += float(rng.binomial(1, P_IOC))
        if g_dec.sum() > CROSSLINK_MIN:
            g_cross += float(rng.binomial(1, P_CROSSLINK))

        # ---- Record ----
        phase_total[phase_idx] += 1
        err  = mean_l2_error(mu, mu_true)
        mn   = float(sigma.mean())

        error_traj_sum[t]  += err
        noise_traj_sum[t]  += mn
        g_entities_sum[t]  += g_ent
        g_iocs_sum[t]      += g_ioc

        t_in_phase = t - phase_idx * PHASE_LEN
        phase_error_sum[phase_idx, t_in_phase] += err

        # T_EVAL snapshots
        if t_in_phase == T_EVAL - 1:
            if phase_idx == 0:
                err_init_at_T[seed_idx] = err
            elif phase_idx == 1:
                err_rc1_at_T[seed_idx] = err
            elif phase_idx == 2:
                err_rc2_at_T[seed_idx] = err

        # ---- First-passage detection ----
        if not found_init and phase_idx == 0 and err < EPS_RECONVERGE:
            n_init_arr[seed_idx] = t_in_phase
            found_init = True
        if not found_rc1 and phase_idx == 1 and err < EPS_RECONVERGE:
            n_rc1_arr[seed_idx] = t_in_phase
            found_rc1 = True
        if not found_rc2 and phase_idx == 2 and err < EPS_RECONVERGE:
            n_rc2_arr[seed_idx] = t_in_phase
            found_rc2 = True

    # Accuracy per phase
    for ph in range(N_PHASES):
        accuracy_arr[seed_idx, ph] = (
            float(phase_correct[ph]) / max(float(phase_total[ph]), 1)
        )

    if (seed_idx + 1) % 10 == 0:
        print(f"  seed {seed_idx+1:3d}/{N_SEEDS}  "
              f"n_init={n_init_arr[seed_idx]}  "
              f"n_rc1={n_rc1_arr[seed_idx]}  "
              f"n_rc2={n_rc2_arr[seed_idx]}", flush=True)

print()

# Normalize trajectories
error_traj  = error_traj_sum  / N_SEEDS
noise_traj  = noise_traj_sum  / N_SEEDS
g_entities  = g_entities_sum  / N_SEEDS
g_iocs      = g_iocs_sum      / N_SEEDS
phase_error = phase_error_sum / N_SEEDS    # (3, 1500)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

# First-passage times
n_init_mean = float(n_init_arr.mean())
n_rc1_mean  = float(n_rc1_arr.mean())
n_rc2_mean  = float(n_rc2_arr.mean())
n_init_std  = float(n_init_arr.std())
n_rc1_std   = float(n_rc1_arr.std())
n_rc2_std   = float(n_rc2_arr.std())

# Fraction of seeds that converged (didn't hit sentinel)
frac_init = float(np.mean(n_init_arr < PHASE_LEN))
frac_rc1  = float(np.mean(n_rc1_arr  < PHASE_LEN))
frac_rc2  = float(np.mean(n_rc2_arr  < PHASE_LEN))

# Acceleration ratios (using mean; only meaningful when both converged)
# N_reconv_1 / N_reconv_2 > 1 → Phase 3 faster than Phase 2
rc_ratio = n_rc1_mean / max(n_rc2_mean, 1.0)   # > 1.0 if Phase 3 faster

# N_initial / N_reconv_1 for context
init_rc1_ratio = n_init_mean / max(n_rc1_mean, 1.0)

# Error at T_EVAL
err_init_T_mean = float(err_init_at_T.mean())
err_rc1_T_mean  = float(err_rc1_at_T.mean())
err_rc2_T_mean  = float(err_rc2_at_T.mean())

# Noise at shift points
noise_s1_mean = float(noise_shift1_arr.mean())
noise_s2_mean = float(noise_shift2_arr.mean())
noise_reduction = (noise_s1_mean - noise_s2_mean) / noise_s1_mean * 100

# Accuracy per phase
acc_ph = [float(accuracy_arr[:, ph].mean()) for ph in range(N_PHASES)]

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

# Paired t-test: N_rc1 vs N_rc2 (H1: N_rc1 > N_rc2, i.e., Phase 3 faster)
t_stat, p_ttest = ttest_rel(n_rc1_arr.astype(float), n_rc2_arr.astype(float),
                            alternative='greater')

# Spearman on pooled data: x=shift_index, y=N_converge
x_pool = np.array([0]*N_SEEDS + [1]*N_SEEDS + [2]*N_SEEDS, dtype=float)
y_pool = np.concatenate([n_init_arr, n_rc1_arr, n_rc2_arr]).astype(float)
spear_rho, spear_p = spearmanr(x_pool, y_pool)

# Fraction of seeds with strict ordering: N_reconv_2 < N_reconv_1 < N_initial
strict_order = float(np.mean(
    (n_rc2_arr < n_rc1_arr) & (n_rc1_arr < n_init_arr)
))
rc_order_only = float(np.mean(n_rc2_arr < n_rc1_arr))   # Phase 3 faster than Phase 2

# Error at T_EVAL ordering
t_eval_order = (err_rc2_T_mean < err_rc1_T_mean < err_init_T_mean)

# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------

spearman_pass  = float(spear_rho) < 0 and float(spear_p) < 0.05
ttest_pass     = float(p_ttest) < 0.05
ordering_pass  = rc_order_only >= 0.60   # at least 60% of seeds show Phase 3 < Phase 2
t_eval_pass    = err_rc2_T_mean < err_rc1_T_mean

evidence_count = sum([spearman_pass, ttest_pass, ordering_pass])

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

print("=" * 66)
print("=== BRIDGE B PHASE C v3: RE-CONVERGENCE COMPOUNDING (A=4) ===")
print("=" * 66)
print()

def nc_str(mean, std, frac):
    if frac < 0.50:
        return f">{PHASE_LEN} (only {frac:.0%} converged)"
    return f"{mean:.0f} ±{std:.0f}  ({frac:.0%} conv)"

print(f"  {'Phase':<8} | {'Decisions':<12} | {'G(entities)':>13} | {'G(IOCs)':>9} | "
      f"{'Noise σ²':>9} | {'N_converge':>20} | {'Acc':>5}")
print("  " + "-" * 87)

g_at_shift = [
    (float(g_entities[PHASE_LEN - 1]),   float(g_iocs[PHASE_LEN - 1])),
    (float(g_entities[2*PHASE_LEN - 1]), float(g_iocs[2*PHASE_LEN - 1])),
    (float(g_entities[N_TOTAL - 1]),     float(g_iocs[N_TOTAL - 1])),
]

phase_info = [
    ("Init",    0,          PHASE_LEN,       float(noise_traj[:PHASE_LEN].mean()),
     n_init_mean, n_init_std, frac_init, acc_ph[0]),
    ("Shift 1", PHASE_LEN,  2*PHASE_LEN,     float(noise_traj[PHASE_LEN:2*PHASE_LEN].mean()),
     n_rc1_mean,  n_rc1_std,  frac_rc1,  acc_ph[1]),
    ("Shift 2", 2*PHASE_LEN, N_TOTAL,        float(noise_traj[2*PHASE_LEN:].mean()),
     n_rc2_mean,  n_rc2_std,  frac_rc2,  acc_ph[2]),
]

for i, (pname, t0, t1, noise_m, nc_m, nc_s, frac, acc) in enumerate(phase_info):
    ent_rng = f"{g_at_shift[max(i-1,0)][0]:.0f}→{g_at_shift[i][0]:.0f}" if i > 0 else f"0→{g_at_shift[0][0]:.0f}"
    ioc_rng = f"{g_at_shift[max(i-1,0)][1]:.0f}→{g_at_shift[i][1]:.0f}" if i > 0 else f"0→{g_at_shift[0][1]:.0f}"
    nc_s_str = nc_str(nc_m, nc_s, frac)
    print(f"  {pname:<8} | {t0:4d}–{t1:<8d} | {ent_rng:>13} | {ioc_rng:>9} | "
          f"{noise_m:>9.5f} | {nc_s_str:>20} | {acc:>5.1%}")

print()
print(f"  Error at T={T_EVAL} decisions after phase start:")
print(f"    Phase 1 (init):     {err_init_T_mean:.5f}")
print(f"    Phase 2 (shift 1):  {err_rc1_T_mean:.5f}  {'↓' if err_rc1_T_mean < err_init_T_mean else '↑'}")
print(f"    Phase 3 (shift 2):  {err_rc2_T_mean:.5f}  {'↓' if err_rc2_T_mean < err_rc1_T_mean else '↑'}")
print()
print(f"  Noise at shift 1: {noise_s1_mean:.5f}")
print(f"  Noise at shift 2: {noise_s2_mean:.5f}  "
      f"(additional {noise_reduction:.1f}% reduction shift 1→shift 2)")
print()
print(f"  Acceleration ratios:")
print(f"    N_initial / N_reconv_1 = {init_rc1_ratio:.3f}  "
      f"({'Phase 2 faster than Phase 1' if init_rc1_ratio > 1 else 'Phase 1 faster'})")
print(f"    N_reconv_1 / N_reconv_2 = {rc_ratio:.3f}  "
      f"({'Phase 3 faster ✓' if rc_ratio > 1 else 'Phase 3 slower ✗'})")
print()
print(f"  Ordering: N_reconv_2 < N_reconv_1 in {rc_order_only:.0%} of seeds  "
      f"({'PASS ✓' if ordering_pass else 'FAIL ✗'} ≥60% required)")
print(f"  Strict order (all 3): {strict_order:.0%} of seeds")
print()
print(f"  Paired t-test (N_rc1 > N_rc2):  t={t_stat:.3f}  p={float(p_ttest):.4f}  "
      f"{'PASS ✓' if ttest_pass else 'FAIL ✗'}")
print(f"  Spearman ρ(shift_idx, N_conv):   ρ={float(spear_rho):.4f}  "
      f"p={float(spear_p):.4f}  "
      f"{'PASS ✓' if spearman_pass else 'FAIL ✗'}")
print()

# Verdict
if evidence_count >= 3:
    verdict_short = "RE-CONVERGENCE COMPOUNDING CONFIRMED — simulation evidence consistent with γ>1"
    verdict = (
        f"Simulation evidence for re-convergence compounding consistent with γ>1. "
        f"N_reconv_1/N_reconv_2 ratio={rc_ratio:.3f} ({rc_order_only:.0%} of seeds show "
        f"Phase 3 faster). Paired t-test p={float(p_ttest):.4f}. "
        f"Spearman ρ={float(spear_rho):.4f} (p={float(spear_p):.4f}). "
        f"Each re-convergence episode is faster than the last because the graph is richer "
        f"→ factor noise is lower → same η produces faster centroid tracking. "
        f"Enrichment rates (P_entity={P_ENTITY}, P_ioc={P_IOC}) and shift magnitude "
        f"(δ={DELTA_SHIFT}) are design choices; real deployments will vary. "
        f"EXP-G1 with real multi-domain data is the definitive measurement."
    )
elif evidence_count == 2:
    verdict_short = "PARTIAL EVIDENCE — 2/3 criteria met, simulation suggests γ>1 mechanism"
    verdict = (
        f"Partial simulation evidence: {evidence_count}/3 criteria met. "
        f"N_rc1/N_rc2={rc_ratio:.3f}, paired t-test p={float(p_ttest):.4f}, "
        f"Spearman ρ={float(spear_rho):.4f} (p={float(spear_p):.4f}). "
        f"Re-convergence shows directional improvement but not all criteria significant. "
        f"EXP-G1 with real multi-domain data is required."
    )
elif evidence_count == 1:
    verdict_short = "WEAK EVIDENCE — 1/3 criteria met"
    verdict = (
        f"Weak simulation evidence: {evidence_count}/3 criteria met. "
        f"N_rc1/N_rc2={rc_ratio:.3f}, paired t-test p={float(p_ttest):.4f}, "
        f"Spearman ρ={float(spear_rho):.4f} (p={float(spear_p):.4f}). "
        f"The endogenous noise reduction ({noise_reduction:.1f}% between shifts) "
        f"is insufficient to produce statistically significant re-convergence acceleration "
        f"in this parameter regime. EXP-G1 with real data is required."
    )
else:
    verdict_short = "NO EVIDENCE — re-convergence does not accelerate with graph maturity"
    verdict = (
        f"No evidence: {evidence_count}/3 criteria met. "
        f"N_rc1/N_rc2={rc_ratio:.3f} (Phase 3 {'faster' if rc_ratio>1 else 'slower'}). "
        f"Noise reduction between shifts: {noise_reduction:.1f}%. "
        f"This simulation design does not confirm the γ>1 mechanism. "
        f"EXP-G1 with real multi-domain data is required."
    )

print(f"  VERDICT: {verdict_short}")
print()
print(f"  {verdict}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "experiment":    "BRIDGE-B-PHASE-C-V3",
    "domain_config": DOMAIN_CONFIG,
    "ontology":      {"C": int(C), "A": int(A), "d": int(d)},
    "n_seeds":       N_SEEDS,
    "n_total":       N_TOTAL,
    "phase_len":     PHASE_LEN,
    "eta":           ETA,
    "eta_neg":       ETA_NEG,
    "e0":            E0,
    "delta_shift":   DELTA_SHIFT,
    "eps_reconverge":EPS_RECONVERGE,
    "t_eval":        T_EVAL,
    "graph_growth_params": {
        "p_entity":      P_ENTITY,
        "p_ioc":         P_IOC,
        "entity_denom":  ENTITY_DENOM,
        "ioc_denom":     IOC_DENOM,
        "pattern_denom": PATTERN_DENOM,
        "no_cap":        True,
    },
    "n_converge": {
        "n_init_mean":  n_init_mean,
        "n_init_std":   n_init_std,
        "n_init_frac":  frac_init,
        "n_rc1_mean":   n_rc1_mean,
        "n_rc1_std":    n_rc1_std,
        "n_rc1_frac":   frac_rc1,
        "n_rc2_mean":   n_rc2_mean,
        "n_rc2_std":    n_rc2_std,
        "n_rc2_frac":   frac_rc2,
    },
    "acceleration_ratios": {
        "init_rc1": float(init_rc1_ratio),
        "rc1_rc2":  float(rc_ratio),
    },
    "error_at_T_eval": {
        "t_eval":     T_EVAL,
        "phase1":     float(err_init_T_mean),
        "phase2":     float(err_rc1_T_mean),
        "phase3":     float(err_rc2_T_mean),
        "decreasing": bool(t_eval_order),
    },
    "noise_at_shifts": {
        "shift1": float(noise_s1_mean),
        "shift2": float(noise_s2_mean),
        "reduction_pct": float(noise_reduction),
    },
    "accuracy_per_phase": {
        "phase1": float(acc_ph[0]),
        "phase2": float(acc_ph[1]),
        "phase3": float(acc_ph[2]),
    },
    "statistics": {
        "paired_ttest_t":    float(t_stat),
        "paired_ttest_p":    float(p_ttest),
        "paired_ttest_pass": bool(ttest_pass),
        "spearman_rho":      float(spear_rho),
        "spearman_p":        float(spear_p),
        "spearman_pass":     bool(spearman_pass),
        "rc_order_frac":     float(rc_order_only),
        "strict_order_frac": float(strict_order),
        "ordering_pass":     bool(ordering_pass),
        "evidence_count":    int(evidence_count),
        "overall_evidence":  bool(evidence_count >= 2),
    },
    "verdict": verdict,
    "verdict_short": verdict_short,
    # Downsampled trajectories for charts (every 5th step)
    "error_traj_ds":        error_traj[::5].tolist(),
    "noise_traj_ds":        noise_traj[::5].tolist(),
    "g_entities_ds":        g_entities[::5].tolist(),
    "g_iocs_ds":            g_iocs[::5].tolist(),
    "phase_error_ds": {
        f"phase{ph+1}": phase_error[ph, ::5].tolist()
        for ph in range(N_PHASES)
    },
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
