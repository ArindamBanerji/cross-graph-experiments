"""
V-MV-CONVERGENCE v2 — log-spaced checkpoints, curve_fit TRF, MAE gate.

Fix for v1 failure (R²=0.305): v1 estimated N_half from 3 time points
(day1/day30/day60) which all measured the asymptote — 92% of exponential
decay occurs by decision 50.

v2 design:
  Checkpoints: n = [1,3,7,14,21,30,42,56,70,84]  (dense near decision 1)
  curve_fit TRF on accuracy(n) = A_final - (A_final - A_0)*exp(-n/tau)
  A_0 fixed to observed accuracy at n=1; free params A_final and tau.
  N_half = tau * ln(2)

  "n" (decision number) = decisions per category: at checkpoint n, each
  of the N_CATS=6 categories has received exactly n verified updates.
  Each simulation step = N_CATS updates (one per category, action from GT_DIST).
  This ensures each centroid receives proportional updates and convergence
  is visible within the 84-decision window (each centroid gets ~n×0.70
  updates for the dominant action, matching the IKS N_half≈14 model).

  Factorial:
    sigma ∈ {0.08, 0.12, 0.15, 0.20, 0.25}
    V     ∈ {50, 100, 200, 500}
    q_bar ∈ {0.65, 0.75, 0.85}
    kernel ∈ {l2, diagonal}
    Skip infeasible: sigma > 0.157 with l2 (RED zone) → skip sigma=0.20,0.25 for l2.
    Total: 36 l2 cells + 60 diagonal cells = 96 cells.

  N_SEEDS = 15 per cell. Cold start (mu_init ~ Uniform(0.2, 0.8)).
  Each step: N_CATS updates (one per category, analyst quality q_bar).
  Accuracy evaluated on N_TEST=200 fresh test alerts at each checkpoint.

  Diagonal kernel: heterogeneous per-factor noise (SOC_HETERO_RATIOS),
    weights = 1/sigma_per_factor^2 (normalised). l2: uniform sigma_eff, L2Kernel.

  Calendar conversion: days = N_half * N_CATS / V
    (N_half = decisions per category; V = verified decisions per day, all categories)

Regression on log(N_half):
  Features: sigma_eff, log(V), q_bar, kernel_enc, sigma*logV, sigma*kernel, q_bar*logV
  5-fold CV, LinearRegression, MAE in decisions (back-transformed).
  80% PI via per-fold training residual std + t-quantile.

Gate: CV MAE <= 3 decisions AND PI coverage >= 88%.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Parameters — do not change
# ---------------------------------------------------------------------------
THETA_MIN   = 0.467
N_SEEDS     = 15
CHECKPOINTS = [1, 3, 7, 14, 21, 30, 42, 56, 70, 84]
N_TEST      = 200
TEST_SEED   = 9999

ETA         = 0.05
ETA_NEG     = 0.05
ETA_OVERRIDE = 0.01
TAU_SCORE   = 0.1

SIGMA_LEVELS = [0.08, 0.12, 0.15, 0.20, 0.25]
V_LEVELS     = [50, 100, 200, 500]
QBAR_LEVELS  = [0.65, 0.75, 0.85]
L2_MAX_SIGMA = 0.157    # RED zone: skip l2 for sigma > this

# SOC heterogeneous noise ratios (for diagonal kernel, same as V-MV-KERNEL rerun)
SOC_HETERO_RATIOS = np.array([0.7, 0.6, 0.5, 1.5, 1.0, 2.0])

# ---------------------------------------------------------------------------
# A1×B1 SOC healthcare geometry
# ---------------------------------------------------------------------------
FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]
N_CATS    = len(CATEGORIES)
N_ACTS    = len(ACTIONS)
N_FACTORS = len(FACTOR_NAMES)
CAT_IDX   = {c: i for i, c in enumerate(CATEGORIES)}

_MU_STAR_RAW = {
    ("lateral_movement",     "escalate"):    [0.30, 0.50, 0.75, 0.35, 0.80, 0.65],
    ("lateral_movement",     "investigate"): [0.30, 0.43, 0.55, 0.35, 0.60, 0.55],
    ("lateral_movement",     "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("lateral_movement",     "monitor"):     [0.30, 0.43, 0.40, 0.35, 0.35, 0.45],
    ("insider_threat",       "escalate"):    [0.25, 0.55, 0.70, 0.30, 0.75, 0.65],
    ("insider_threat",       "investigate"): [0.25, 0.46, 0.50, 0.30, 0.55, 0.55],
    ("insider_threat",       "suppress"):    [0.25, 0.40, 0.20, 0.30, 0.20, 0.35],
    ("insider_threat",       "monitor"):     [0.25, 0.42, 0.38, 0.30, 0.32, 0.45],
    ("credential_access",    "escalate"):    [0.35, 0.50, 0.80, 0.40, 0.75, 0.65],
    ("credential_access",    "investigate"): [0.35, 0.43, 0.60, 0.40, 0.58, 0.55],
    ("credential_access",    "suppress"):    [0.35, 0.40, 0.20, 0.40, 0.22, 0.35],
    ("credential_access",    "monitor"):     [0.35, 0.42, 0.42, 0.40, 0.33, 0.45],
    ("data_exfiltration",    "escalate"):    [0.30, 0.52, 0.78, 0.35, 0.82, 0.65],
    ("data_exfiltration",    "investigate"): [0.30, 0.44, 0.58, 0.35, 0.62, 0.55],
    ("data_exfiltration",    "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("data_exfiltration",    "monitor"):     [0.30, 0.42, 0.40, 0.35, 0.32, 0.45],
    ("cloud_infrastructure", "escalate"):    [0.28, 0.45, 0.72, 0.38, 0.70, 0.65],
    ("cloud_infrastructure", "investigate"): [0.28, 0.41, 0.52, 0.38, 0.52, 0.55],
    ("cloud_infrastructure", "suppress"):    [0.28, 0.40, 0.20, 0.38, 0.20, 0.35],
    ("cloud_infrastructure", "monitor"):     [0.28, 0.41, 0.38, 0.38, 0.30, 0.45],
    ("threat_intel_match",   "escalate"):    [0.32, 0.52, 0.82, 0.36, 0.78, 0.65],
    ("threat_intel_match",   "investigate"): [0.32, 0.44, 0.62, 0.36, 0.58, 0.55],
    ("threat_intel_match",   "suppress"):    [0.32, 0.40, 0.20, 0.36, 0.20, 0.35],
    ("threat_intel_match",   "monitor"):     [0.32, 0.42, 0.44, 0.36, 0.33, 0.45],
}


def build_mu_star() -> np.ndarray:
    mu = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    for (cat, act), vec in _MU_STAR_RAW.items():
        ai = ACTIONS.index(act)
        mu[CAT_IDX[cat], ai, :] = vec
    return mu


MU_STAR = build_mu_star()


def build_gt_dist() -> np.ndarray:
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt


GT_DIST = build_gt_dist()
CAT_W   = np.ones(N_CATS) / N_CATS


# ---------------------------------------------------------------------------
# Kernel and noise construction
# ---------------------------------------------------------------------------
def make_kernel_and_noise(kernel_type: str, sigma_eff: float):
    """
    Returns (kernel_object, noise_array).
    l2:       L2Kernel, uniform noise sigma_eff.
    diagonal: DiagonalKernel(1/sigma_per_factor^2), heterogeneous noise via SOC_HETERO_RATIOS.
    """
    if kernel_type == "l2":
        noise = np.full(N_FACTORS, sigma_eff)
        return L2Kernel(), noise

    # Diagonal: heterogeneous noise, rescaled so mean == sigma_eff
    raw_ratios = SOC_HETERO_RATIOS.copy()
    noise = sigma_eff * raw_ratios
    noise = noise * (sigma_eff / noise.mean())
    noise = np.clip(noise, 0.03, 0.40)

    weights = 1.0 / np.maximum(noise ** 2, 1e-4)
    weights /= weights.max()
    return DiagonalKernel(weights), noise


# ---------------------------------------------------------------------------
# Single seed simulation
# ---------------------------------------------------------------------------
def run_seed(
    sigma_eff: float,
    q_bar: float,
    kernel_obj,
    noise_array: np.ndarray,
    seed: int,
) -> list:
    """
    Returns list of accuracy values at each checkpoint (len == len(CHECKPOINTS)).
    Uses cold start. Each step = N_CATS updates (one per category, action from GT_DIST).
    Checkpoint n = n steps completed = n decisions per category.
    """
    rng = np.random.default_rng(seed)

    # Cold start
    mu_init = rng.uniform(0.2, 0.8, MU_STAR.shape)
    profile = CalibrationProfile(learning_rate=ETA, penalty_ratio=1.0, temperature=TAU_SCORE)
    scorer  = ProfileScorer(mu_init.copy(), ACTIONS, scoring_kernel=kernel_obj,
                            profile=profile, eta_override=ETA_OVERRIDE)
    scorer.eta     = ETA
    scorer.eta_neg = ETA_NEG

    cp_set = set(CHECKPOINTS)
    accs: list = []

    for step in range(1, CHECKPOINTS[-1] + 1):
        # N_CATS updates this step (one per category)
        for ci in range(N_CATS):
            gt_a = int(rng.choice(N_ACTS, p=GT_DIST[ci]))
            f    = np.clip(MU_STAR[ci, gt_a] + rng.standard_normal(N_FACTORS) * noise_array,
                           0.0, 1.0)
            # Analyst label: correct with prob q_bar
            if rng.random() < q_bar:
                label_a = gt_a
            else:
                choices = [a for a in range(N_ACTS) if a != gt_a]
                label_a = int(rng.choice(choices))
            scorer.update(f, ci, label_a, correct=True)

        # Checkpoint evaluation
        if step in cp_set:
            test_rng  = np.random.default_rng(TEST_SEED)
            n_correct = 0
            for _ in range(N_TEST):
                tc = int(test_rng.choice(N_CATS, p=CAT_W))
                ta = int(test_rng.choice(N_ACTS, p=GT_DIST[tc]))
                tf = np.clip(MU_STAR[tc, ta] + test_rng.standard_normal(N_FACTORS) * noise_array,
                             0.0, 1.0)
                r  = scorer.score(tf, tc)
                n_correct += int(r.action_index == ta)
            accs.append(n_correct / N_TEST)

    return accs


# ---------------------------------------------------------------------------
# Cell-level estimation: average across seeds, then curve_fit
# ---------------------------------------------------------------------------
def estimate_n_half(mean_acc: np.ndarray) -> tuple:
    """
    Fit: acc(n) = A_final - (A_final - A_0) * exp(-n / tau)
    A_0 fixed to mean_acc[0] (checkpoint n=1).
    Returns (N_half, converged: bool).
    """
    A_0 = float(mean_acc[0])
    ns  = np.array(CHECKPOINTS, dtype=float)

    def model(n, A_final, tau):
        return A_final - (A_final - A_0) * np.exp(-n / tau)

    try:
        popt, _ = curve_fit(
            model, ns, mean_acc,
            p0=[0.90, 20.0],
            bounds=([0.5, 1.0], [1.0, 200.0]),
            method="trf",
            maxfev=5000,
        )
        A_final, tau = popt
        n_half = tau * math.log(2)
        return n_half, True
    except RuntimeError:
        return 84.0, False


# ---------------------------------------------------------------------------
# Build cell list
# ---------------------------------------------------------------------------
def build_cells() -> list:
    cells = []
    for sigma, V, q_bar in product(SIGMA_LEVELS, V_LEVELS, QBAR_LEVELS):
        for kernel_type in ("l2", "diagonal"):
            if kernel_type == "l2" and sigma > L2_MAX_SIGMA:
                continue  # RED zone
            cells.append({
                "sigma_eff":   sigma,
                "volume":      V,
                "q_bar":       q_bar,
                "kernel_type": kernel_type,
            })
    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    gae_ver = gae.__version__

    cells = build_cells()
    n_cells_attempted = len(cells)

    print(f"V-MV-CONVERGENCE v2 (log-spaced, curve_fit TRF, GAE {gae_ver}):")
    print(f"  Checkpoints: {CHECKPOINTS}")
    print(f"  Cells to run: {n_cells_attempted}  (l2: {sum(1 for c in cells if c['kernel_type']=='l2')}, "
          f"diagonal: {sum(1 for c in cells if c['kernel_type']=='diagonal')})")
    print(f"  Seeds per cell: {N_SEEDS}")
    print()

    # ---- Run all cells ----
    cell_results = []
    n_cells_converged = 0

    for cell_i, cell in enumerate(cells):
        sigma  = cell["sigma_eff"]
        V      = cell["volume"]
        q_bar  = cell["q_bar"]
        ktype  = cell["kernel_type"]

        kernel_obj, noise_array = make_kernel_and_noise(ktype, sigma)

        # Average across N_SEEDS
        seed_accs = []
        for si in range(N_SEEDS):
            accs = run_seed(sigma, q_bar, kernel_obj, noise_array, seed=42 + si)
            seed_accs.append(accs)

        mean_acc = np.mean(seed_accs, axis=0)  # shape (len(CHECKPOINTS),)

        n_half, converged = estimate_n_half(mean_acc)
        if converged:
            n_cells_converged += 1

        cell_results.append({
            "sigma_eff":    sigma,
            "volume":       V,
            "q_bar":        q_bar,
            "kernel_type":  ktype,
            "kernel_enc":   0.0 if ktype == "l2" else 1.0,
            "mean_acc_by_checkpoint": [round(float(a), 4) for a in mean_acc],
            "n_half":       round(float(n_half), 3),
            "converged":    converged,
        })

        if (cell_i + 1) % 10 == 0 or cell_i == 0:
            print(f"  [{cell_i+1:3d}/{n_cells_attempted}] "
                  f"σ={sigma:.2f} V={V:4d} q̄={q_bar:.2f} {ktype:<9} "
                  f"N_half={n_half:6.1f}d  {'ok' if converged else 'CENSORED'}")

    convergence_rate = n_cells_converged / n_cells_attempted

    # ---- Regression ----
    print()
    print(f"  Fitting log(N_half) regression on {len(cell_results)} cells ...")

    sigma_arr   = np.array([c["sigma_eff"]  for c in cell_results])
    logV_arr    = np.log(np.array([c["volume"]     for c in cell_results]))
    qbar_arr    = np.array([c["q_bar"]       for c in cell_results])
    kernel_arr  = np.array([c["kernel_enc"]  for c in cell_results])
    n_half_arr  = np.array([c["n_half"]      for c in cell_results])
    log_nhalf   = np.log(n_half_arr)

    X = np.column_stack([
        sigma_arr,
        logV_arr,
        qbar_arr,
        kernel_arr,
        sigma_arr * logV_arr,
        sigma_arr * kernel_arr,
        qbar_arr  * logV_arr,
    ])
    feature_names = ["sigma_eff", "log_V", "q_bar", "kernel_enc",
                     "sigma_x_logV", "sigma_x_kernel", "q_bar_x_logV"]

    # 5-fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_mae_dec  = []
    all_covered  = []

    for fold_i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_tr, y_tr = X[train_idx], log_nhalf[train_idx]
        X_te, y_te = X[test_idx],  log_nhalf[test_idx]

        lr = LinearRegression()
        lr.fit(X_tr, y_tr)

        # Training residuals for PI width (residual std with correct df)
        y_tr_pred = lr.predict(X_tr)
        n_p = len(feature_names) + 1   # features + intercept
        resid_std = float(np.sqrt(np.sum((y_tr - y_tr_pred) ** 2) / (len(train_idx) - n_p)))

        y_te_pred = lr.predict(X_te)

        # MAE in decisions (back-transform)
        mae_dec = float(np.mean(np.abs(np.exp(y_te_pred) - np.exp(y_te))))
        all_mae_dec.append(mae_dec)

        # 80% PI coverage with proper leverage correction: t * s * sqrt(1 + hat_h)
        # Augment X with intercept column for hat_h computation
        ones_tr = np.ones((len(train_idx), 1))
        ones_te = np.ones((len(test_idx),  1))
        Xa_tr = np.column_stack([ones_tr, X_tr])
        Xa_te = np.column_stack([ones_te, X_te])
        XtXinv = np.linalg.pinv(Xa_tr.T @ Xa_tr)

        df = len(train_idx) - n_p
        t_q = float(scipy_stats.t.ppf(0.90, df=max(df, 1)))

        for pred, true, xa in zip(y_te_pred, y_te, Xa_te):
            hat_h   = float(xa @ XtXinv @ xa)
            pi_half = t_q * resid_std * math.sqrt(1.0 + hat_h)
            covered = (true >= pred - pi_half) and (true <= pred + pi_half)
            all_covered.append(covered)

    cv_mae = float(np.mean(all_mae_dec))
    pi_coverage = float(np.mean(all_covered))

    # Full-data fit for top coefficients
    lr_full = LinearRegression()
    lr_full.fit(X, log_nhalf)
    coef_pairs = list(zip(feature_names, lr_full.coef_))
    coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_coefficients = {name: round(float(coef), 4)
                        for name, coef in coef_pairs[:5]}

    # Gates
    gate_mae = cv_mae <= 3.0
    gate_pi  = pi_coverage >= 0.88
    both_pass = gate_mae and gate_pi

    # ---- Calendar predictions (only if both gates pass) ----
    calendar_predictions: dict = {}
    if both_pass:
        profiles = {
            "A_sigma015_V200_q075_diagonal": (0.15, 200, 0.75, 1.0),
            "B_sigma010_V100_q085_l2":       (0.10, 100, 0.85, 0.0),
            "C_sigma020_V50_q070_diagonal":  (0.20,  50, 0.70, 1.0),
        }
        for prof_name, (s, v, q, k) in profiles.items():
            x_prof = np.array([[s, math.log(v), q, k,
                                 s * math.log(v), s * k, q * math.log(v)]])
            log_nh_pred = float(lr_full.predict(x_prof)[0])
            nh_pred = math.exp(log_nh_pred)
            # days = N_half * N_CATS / V
            # (N_half = decisions/category; V = verified decisions/day all categories)
            days_pred = nh_pred * N_CATS / v
            calendar_predictions[prof_name] = {
                "n_half_decisions": round(nh_pred, 1),
                "days":             round(days_pred, 2),
            }

    # ---- Print results ----
    print()
    print(f"  Cells attempted/converged: {n_cells_attempted}/{n_cells_converged} "
          f"({convergence_rate:.1%} convergence rate)")
    print(f"  Cross-validated MAE: {cv_mae:.2f} decisions [gate: <=3] -> "
          f"{'PASS' if gate_mae else 'FAIL'}")
    print(f"  PI coverage:         {pi_coverage:.1%}         [gate: >=88%] -> "
          f"{'PASS' if gate_pi else 'FAIL'}")
    print(f"  Both gates: {'PASS' if both_pass else 'FAIL'}")

    if both_pass:
        print()
        print("  Calendar predictions:")
        q070_key = "C_sigma020_V50_q070_diagonal"
        for lab, (s, v, q, k), pkey in [
            ("Profile A (sigma=0.15,V=200,q=0.75,Diagonal)", (0.15,200,0.75,1.0),
             "A_sigma015_V200_q075_diagonal"),
            ("Profile B (sigma=0.10,V=100,q=0.85,L2)      ", (0.10,100,0.85,0.0),
             "B_sigma010_V100_q085_l2"),
            ("Profile C (sigma=0.20,V=50, q=0.70,Diagonal) ", (0.20,50,0.70,1.0),
             "C_sigma020_V50_q070_diagonal"),
        ]:
            cp = calendar_predictions[pkey]
            print(f"    {lab}: N_half={cp['n_half_decisions']:.0f}d, "
                  f"~{cp['days']:.1f} days")

    print()
    print("Raw numbers for roadmap session review.")

    # ---- Save ----
    out = {
        "experiment":           "V-MV-CONVERGENCE-v2",
        "gae_version":          gae_ver,
        "checkpoints":          CHECKPOINTS,
        "n_seeds":              N_SEEDS,
        "n_cells_attempted":    n_cells_attempted,
        "n_cells_converged":    n_cells_converged,
        "convergence_rate":     round(convergence_rate, 4),
        "cv_mae_decisions":     round(cv_mae, 4),
        "pi_coverage_pct":      round(pi_coverage * 100, 2),
        "gate_mae_pass":        gate_mae,
        "gate_pi_pass":         gate_pi,
        "both_gates_pass":      both_pass,
        "top_coefficients":     top_coefficients,
        "calendar_predictions": calendar_predictions,
        "cell_results":         cell_results,
    }

    out_path = REPO_ROOT / "experiments" / "v_mv_convergence" / "results" / "results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
