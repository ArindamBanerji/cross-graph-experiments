"""
Roadmap session deliverables 2-4: kernel experiments.

Deliverable 2: V-HC-CONFIG-SHRINKAGE
    Healthcare persona — Diagonal vs Shrinkage-proxy(rho=0.45) vs Shrinkage-proxy(rho=0)
    Does the device_trust↔time_anomaly correlation (rho=0.45) add benefit over
    noise-ratio weighting alone?

Deliverable 3: V-S2P-HETERO
    18 S2P cells: 3 kernels × 3 sigma_eff × 2 q_bar, all heterogeneous noise.
    Shrinkage-proxy from Regime A correlation matrix. Measures correlation density benefit.

Deliverable 4: KERNEL SELECTOR SELF-TEST
    KernelSelector in shadow mode on healthcare SOC and S2P Manufacturing deployments.
    Tracks per-kernel analyst-agreement, stabilization point, and Phase 2/4 recommendations.

Usage:
    python experiments/factorial/run_kernel_deliverables.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel
from gae.kernel_selector import KernelSelector

# ── Shared constants ───────────────────────────────────────────────────────────
ETA          = 0.05
ETA_NEG      = 0.05
ETA_OVERRIDE = 0.01
VERIFY_RATE  = 0.30
CONV_EPS     = 0.10

# ── Healthcare persona 1D-N4 ───────────────────────────────────────────────────
HC_FACTOR_NAMES = [
    "travel_match",            # 0
    "asset_criticality",       # 1
    "threat_intel_enrichment", # 2
    "time_anomaly",            # 3  ← correlated with device_trust
    "pattern_history",         # 4
    "device_trust",            # 5  ← correlated with time_anomaly
]

HC_FACTOR_NOISE = np.array([0.18, 0.20, 0.19, 0.25, 0.22, 0.28])   # d=6
HC_APD          = 200
HC_N_SEEDS      = 15
HC_DAYS         = 60

HC_ANALYST_TEAM = [
    {"override_rate": 0.25, "override_quality": 0.90, "fatigue_factor": 0.10},
    {"override_rate": 0.28, "override_quality": 0.78, "fatigue_factor": 0.20},
    {"override_rate": 0.28, "override_quality": 0.80, "fatigue_factor": 0.18},
    {"override_rate": 0.35, "override_quality": 0.63, "fatigue_factor": 0.35},
]

# S2P constants
S2P_FACTOR_NAMES = [
    "supplier_risk", "logistics_risk", "demand_risk", "inventory_risk",
    "regulatory_risk", "geopolitical_risk", "financial_risk", "environmental_risk",
]
S2P_HETERO_RATIOS = [1.0, 1.5, 0.7, 0.8, 0.6, 1.3, 1.8, 1.6]

S2P_REGIME_A_CORR = np.array([
    [1.00, 0.48, 0.35, 0.62, 0.45, 0.58, 0.70, 0.50],
    [0.48, 1.00, 0.42, 0.60, 0.40, 0.55, 0.38, 0.65],
    [0.35, 0.42, 1.00, 0.65, 0.18, 0.28, 0.40, 0.25],
    [0.62, 0.60, 0.65, 1.00, 0.30, 0.30, 0.48, 0.35],
    [0.45, 0.40, 0.18, 0.30, 1.00, 0.72, 0.52, 0.30],
    [0.58, 0.55, 0.28, 0.30, 0.72, 1.00, 0.68, 0.35],
    [0.70, 0.38, 0.40, 0.48, 0.52, 0.68, 1.00, 0.35],
    [0.50, 0.65, 0.25, 0.35, 0.30, 0.35, 0.35, 1.00],
], dtype=float)

S2P_N_SEEDS  = 10
S2P_DAYS     = 60
S2P_APD      = 200


# ── Weight construction helpers ────────────────────────────────────────────────

def diagonal_weights(sigma: np.ndarray) -> np.ndarray:
    """1/σ² normalised to max=1."""
    inv_var = 1.0 / np.maximum(sigma ** 2, 1e-4)
    return inv_var / inv_var.max()


def shrinkage_proxy_weights(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """
    Compute diagonal of Σ⁻¹ as kernel weights.
    Σ = diag(σ) @ corr @ diag(σ). When corr=I this reduces to 1/σ².
    Normalised to max=1.
    """
    cov = np.outer(sigma, sigma) * corr
    # Ensure PD
    cov = (cov + cov.T) / 2.0
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-8:
        cov += (abs(eigvals.min()) + 1e-6) * np.eye(len(sigma))
    cov_inv = np.linalg.inv(cov)
    w = np.diag(cov_inv)
    w = np.maximum(w, 0.0)          # numerical guard
    return w / w.max()


def make_hc_corr(rho: float) -> np.ndarray:
    """
    Healthcare 6×6 correlation: ρ on (time_anomaly, device_trust) = (3, 5).
    All other off-diagonals = 0.
    """
    corr = np.eye(6)
    corr[3, 5] = corr[5, 3] = rho
    return corr


def make_noise_array(sigma_eff: float, hetero_ratios: list, d: int) -> np.ndarray:
    """Heterogeneous noise array with mean = sigma_eff."""
    ratios = np.array(hetero_ratios[:d], dtype=float)
    raw    = sigma_eff * ratios
    raw    = raw * (sigma_eff / raw.mean())
    return np.clip(raw, 0.03, 0.40)


# ── Analyst params ─────────────────────────────────────────────────────────────

def build_analyst_params(team: list) -> list:
    params = []
    for a in team:
        ff  = a["fatigue_factor"]
        eo  = min(1.0, a["override_rate"] * (1 + ff * 0.3))
        eq  = max(0.4, a["override_quality"] * (1 - ff * 0.2))
        params.append((eo, eq))
    return params if params else [(0.25, 0.75)]


# ── Core simulation (single condition) ────────────────────────────────────────

def run_condition(
    mu_true:     np.ndarray,      # (C, A, d)
    categories:  list,
    actions:     list,
    gt_arr:      np.ndarray,      # (C, A)
    cat_w:       np.ndarray,      # (C,)
    kernel,
    noise_array: np.ndarray,      # (d,) per-factor σ
    apd:         int,
    n_seeds:     int,
    days:        int,
    analyst_team: list,
) -> dict:
    C, A, d    = mu_true.shape
    a_params   = build_analyst_params(analyst_team)
    n_analysts = len(a_params)

    all_daily = []
    all_conv  = {cat: [] for cat in categories}

    for si in range(n_seeds):
        rng = np.random.default_rng(42 + si)

        offset  = rng.uniform(-0.15, 0.15, mu_true.shape)
        mu_init = np.clip(mu_true + offset, 0, 1)

        scorer = ProfileScorer(
            mu_init.copy(), actions,
            scoring_kernel=kernel,
            eta_override=ETA_OVERRIDE,
        )
        scorer.eta     = ETA
        scorer.eta_neg = ETA_NEG

        daily_acc = np.zeros(days)
        conv_day  = {cat: None for cat in categories}

        for day in range(days):
            n_alerts = int(rng.poisson(apd))
            correct  = 0

            for _ in range(n_alerts):
                ci   = int(rng.choice(C, p=cat_w))
                a_gt = int(rng.choice(A, p=gt_arr[ci]))

                z = rng.standard_normal(d)
                f = np.clip(mu_true[ci, a_gt] + noise_array * z, 0, 1)

                res    = scorer.score(f, ci)
                pred_a = res.action_index
                correct += int(pred_a == a_gt)

                if rng.random() < VERIFY_RATE:
                    ai_idx = rng.integers(n_analysts)
                    eff_o, eff_q = a_params[ai_idx]
                    if rng.random() < eff_o:
                        gt_a = a_gt if rng.random() < eff_q else int(
                            rng.choice([a for a in range(A) if a != a_gt])
                        )
                        scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                    else:
                        scorer.update(f, ci, pred_a, True)

            daily_acc[day] = correct / n_alerts if n_alerts > 0 else 0.0

            for ci2, cat in enumerate(categories):
                if conv_day[cat] is None:
                    err = max(
                        np.linalg.norm(scorer.centroids[ci2, a] - mu_true[ci2, a])
                        for a in range(A)
                    )
                    if err < CONV_EPS:
                        conv_day[cat] = day + 1

        all_daily.append(daily_acc)
        for cat in categories:
            all_conv[cat].append(conv_day[cat] if conv_day[cat] is not None else days + 1)

    mean_acc = np.stack(all_daily).mean(axis=0)

    conv_summary = {}
    for cat in categories:
        days_list = all_conv[cat]
        pct_conv  = float(np.mean([v <= days for v in days_list]))
        converged = [v for v in days_list if v <= days]
        conv_summary[cat] = {
            "mean_conv_day": round(float(np.mean(converged)), 1) if converged else None,
            "pct_converged": round(pct_conv, 3),
        }

    d1  = float(mean_acc[0])
    d30 = float(mean_acc[29] if days >= 30 else mean_acc[-1])
    d60 = float(mean_acc[-1])

    return {
        "day1_accuracy":  round(d1,  4),
        "day30_accuracy": round(d30, 4),
        "day60_accuracy": round(d60, 4),
        "delta_d1_d60":   round(d60 - d1, 4),
        "convergence":    conv_summary,
    }


# ════════════════════════════════════════════════════════════════════════════════
# DELIVERABLE 2: V-HC-CONFIG-SHRINKAGE
# ════════════════════════════════════════════════════════════════════════════════

def run_deliverable2(config: dict) -> dict:
    print()
    print("=" * 66)
    print("DELIVERABLE 2: V-HC-CONFIG-SHRINKAGE")
    print("=" * 66)
    print(f"  Healthcare 1D-N4  σ_mean={HC_FACTOR_NOISE.mean():.3f}"
          f"  APD={HC_APD}  Seeds={HC_N_SEEDS}  Days={HC_DAYS}")

    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape

    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0/A]*A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()
    cat_w = np.ones(C) / C

    # Build three weight vectors
    noise = HC_FACTOR_NOISE

    wA = diagonal_weights(noise)                                        # Condition A
    wB = shrinkage_proxy_weights(noise, make_hc_corr(rho=0.45))        # Condition B
    wC = shrinkage_proxy_weights(noise, make_hc_corr(rho=0.0))         # Condition C

    conditions = [
        ("A: Diagonal",           DiagonalKernel(wA), wA, "Diagonal",           "rho=0.00"),
        ("B: Shrinkage rho=0.45", DiagonalKernel(wB), wB, "Shrinkage_proxy",    "rho=0.45"),
        ("C: Shrinkage rho=0",    DiagonalKernel(wC), wC, "Shrinkage_proxy",    "rho=0.00"),
    ]

    # Print weight vectors
    print()
    print(f"  {'Factor':<26} {'σ':>5}  {'wA':>7}  {'wB':>7}  {'wC':>7}")
    print("  " + "-" * 60)
    for i, fn in enumerate(HC_FACTOR_NAMES):
        marker = " ← corr" if i in (3, 5) else ""
        print(f"  {fn:<26} {noise[i]:>5.2f}  {wA[i]:>7.4f}  {wB[i]:>7.4f}  {wC[i]:>7.4f}{marker}")

    results = []
    t0_total = time.time()

    for label, kernel, weights, kname, rtag in conditions:
        print(f"\n  Running {label} ...", flush=True)
        t0 = time.time()
        r  = run_condition(
            mu_true, categories, actions, gt_arr, cat_w,
            kernel, noise, HC_APD, HC_N_SEEDS, HC_DAYS, HC_ANALYST_TEAM,
        )
        elapsed = time.time() - t0
        sign = "+" if r["delta_d1_d60"] >= 0 else ""
        print(f"    → Day60={r['day60_accuracy']:.1%}  Δ={sign}{r['delta_d1_d60']:.2%}"
              f"  ({elapsed:.1f}s)")
        r.update({"label": label, "kernel_name": kname, "rho_tag": rtag,
                  "weights": [round(float(x), 4) for x in weights]})
        results.append(r)

    # Summary table
    print()
    print(f"  {'Condition':<24} {'Day 1':>7} {'Day 30':>7} {'Day 60':>7} {'Δ(60-1)':>9}")
    print("  " + "-" * 58)
    for r in results:
        sign = "+" if r["delta_d1_d60"] >= 0 else ""
        print(f"  {r['label']:<24} {r['day1_accuracy']:>7.1%}"
              f" {r['day30_accuracy']:>7.1%} {r['day60_accuracy']:>7.1%}"
              f" {sign}{r['delta_d1_d60']:>8.2%}")

    # B-C comparison
    delta_B = results[1]["delta_d1_d60"]
    delta_C = results[2]["delta_d1_d60"]
    d60_B   = results[1]["day60_accuracy"]
    d60_C   = results[2]["day60_accuracy"]
    bc_gap  = abs(d60_B - d60_C)

    print()
    print("  B-C analysis (correlation value vs noise-ratio only):")
    print(f"    B (rho=0.45) Day60 = {d60_B:.1%}   Δ = {delta_B:+.2%}")
    print(f"    C (rho=0.00) Day60 = {d60_C:.1%}   Δ = {delta_C:+.2%}")
    print(f"    |B-C| Day60 gap = {bc_gap:.2%}")

    print()
    print("  VERDICT (Deliverable 2):")
    if bc_gap > 0.01:
        expl = "B"
        print(f"    |B-C| = {bc_gap:.2%} > 1pp → Correlation structure MATTERS.")
        print(f"    Shrinkage-proxy with ρ=0.45 adds value beyond noise-ratio weighting.")
        print(f"    Explanation B confirmed: off-diagonal covariance improves kernel.")
        print(f"    → Priority for full ShrinkageKernel (v6.5) elevated.")
    else:
        expl = "A"
        print(f"    |B-C| = {bc_gap:.2%} ≤ 1pp → Noise ratio ALONE is sufficient.")
        print(f"    Correlation structure provides negligible benefit.")
        print(f"    Explanation A confirmed: DiagonalKernel(1/σ²) captures the gain.")
        print(f"    → Full ShrinkageKernel lower priority; Diagonal sufficient for v6.0.")

    total_time = time.time() - t0_total
    print(f"\n  D2 completed in {total_time:.1f}s")

    return {
        "deliverable": "D2_vhc_shrinkage",
        "bc_gap_d60":  round(bc_gap, 4),
        "explanation": expl,
        "conditions":  [
            {k: v for k, v in r.items() if k != "convergence"} for r in results
        ],
        "convergence": {r["label"]: r["convergence"] for r in results},
    }


# ════════════════════════════════════════════════════════════════════════════════
# DELIVERABLE 3: V-S2P-HETERO
# ════════════════════════════════════════════════════════════════════════════════

def run_deliverable3(config: dict) -> dict:
    print()
    print("=" * 66)
    print("DELIVERABLE 3: V-S2P-HETERO (18 cells)")
    print("=" * 66)
    print(f"  S2P  APD={S2P_APD}  Seeds={S2P_N_SEEDS}  Days={S2P_DAYS}"
          f"  heterogeneous noise")

    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape

    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0/A]*A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()
    cat_w = np.ones(C) / C

    sigma_levels = [0.08, 0.15, 0.22]
    q_bar_levels = [0.60, 0.80]
    kernel_types = ["l2", "diagonal", "shrinkage"]

    # Regime-A shrinkage weights differ per sigma_eff because Σ = diag(σ) @ R @ diag(σ)
    def get_kernel(kt: str, noise_arr: np.ndarray):
        if kt == "l2":
            return L2Kernel(), np.ones(d)
        if kt == "diagonal":
            w = diagonal_weights(noise_arr)
            return DiagonalKernel(w), w
        # shrinkage: full Regime A covariance
        w = shrinkage_proxy_weights(noise_arr, S2P_REGIME_A_CORR)
        return DiagonalKernel(w), w

    # Build a simple analyst team for S2P
    def analyst_team_for_q(q_bar: float):
        q = q_bar
        return [
            {"override_rate": 0.25, "override_quality": min(q + 0.05, 1.0), "fatigue_factor": 0.15},
            {"override_rate": 0.28, "override_quality": q,                   "fatigue_factor": 0.20},
        ]

    results = []
    total = len(sigma_levels) * len(q_bar_levels) * len(kernel_types)
    idx   = 0

    for sig in sigma_levels:
        noise_arr = make_noise_array(sig, S2P_HETERO_RATIOS, d)
        for qb in q_bar_levels:
            team = analyst_team_for_q(qb)
            for kt in kernel_types:
                idx += 1
                kernel, weights = get_kernel(kt, noise_arr)
                cell_id = f"s2p-{kt[:4]}-s{int(sig*100):03d}-q{int(qb*100):02d}-hete"
                t0 = time.time()
                r  = run_condition(
                    mu_true, categories, actions, gt_arr, cat_w,
                    kernel, noise_arr, S2P_APD, S2P_N_SEEDS, S2P_DAYS, team,
                )
                elapsed = time.time() - t0
                sign = "+" if r["delta_d1_d60"] >= 0 else ""
                print(f"  [{idx:>2}/{total}] {cell_id:<38}"
                      f"  Day60={r['day60_accuracy']:.1%}  Δ={sign}{r['delta_d1_d60']:.2%}"
                      f"  ({elapsed:.1f}s)")
                r.update({"cell_id": cell_id, "kernel_type": kt,
                          "sigma_eff": sig, "q_bar": qb})
                results.append(r)

    # Summary by kernel
    print()
    print(f"  {'Kernel':<12} {'Mean D60':>10} {'Mean Delta':>11} {'Cells':>6}")
    print("  " + "-" * 44)
    for kt in kernel_types:
        cells = [r for r in results if r["kernel_type"] == kt]
        mean_d60   = float(np.mean([c["day60_accuracy"] for c in cells]))
        mean_delta = float(np.mean([c["delta_d1_d60"]   for c in cells]))
        print(f"  {kt:<12} {mean_d60:>10.1%} {mean_delta:>+10.2%} {len(cells):>6}")

    # Shrinkage-vs-diagonal gap
    dia_mean = float(np.mean([r["day60_accuracy"] for r in results if r["kernel_type"] == "diagonal"]))
    shr_mean = float(np.mean([r["day60_accuracy"] for r in results if r["kernel_type"] == "shrinkage"]))
    shr_dia_gap = shr_mean - dia_mean

    print()
    print(f"  Shrinkage vs Diagonal Day60 gap: {shr_dia_gap:+.2%}")
    print()
    print("  VERDICT (Deliverable 3):")
    if shr_dia_gap > 0.03:
        print(f"    Gap = {shr_dia_gap:+.2%} > 3pp → S2P correlation DENSITY MATTERS.")
        print(f"    Regime A off-diagonal structure (avg corr ~0.43) provides real benefit.")
        print(f"    → Full ShrinkageKernel prioritised for S2P domains.")
    elif shr_dia_gap > 0.01:
        print(f"    Gap = {shr_dia_gap:+.2%} → Marginal correlation benefit (1-3pp).")
        print(f"    Noise ratio weighting dominates; correlation adds incremental value.")
        print(f"    → DiagonalKernel sufficient for S2P v6.0. Shrinkage for v6.5.")
    else:
        print(f"    Gap = {shr_dia_gap:+.2%} ≤ 1pp → Correlation density DOES NOT help.")
        print(f"    Noise heterogeneity is the dominant signal; Diagonal already optimal.")
        print(f"    → Full ShrinkageKernel low priority for S2P domain.")

    return {
        "deliverable":  "D3_s2p_hetero",
        "shr_dia_gap":  round(shr_dia_gap, 4),
        "cells":        [{k: v for k, v in r.items() if k != "convergence"}
                         for r in results],
    }


# ════════════════════════════════════════════════════════════════════════════════
# DELIVERABLE 4: KERNEL SELECTOR SELF-TEST
# ════════════════════════════════════════════════════════════════════════════════

def run_selector_shadow(
    label:         str,
    sigma:         np.ndarray,
    rho_max:       float,
    config:        dict,
    noise_array:   np.ndarray,
    n_decisions:   int = 500,
    checkpoint_n:  int = 10,
    apd:           int = 100,
) -> dict:
    """
    Run KernelSelector shadow mode for one deployment.
    Uses a live ProfileScorer (L2) for scoring; tracks all kernels via record_comparison.
    """
    mu_true    = config["mu"]
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape

    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0/A]*A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()
    cat_w = np.ones(C) / C

    # Selector instantiation (Phase 2)
    selector = KernelSelector(d=d, sigma_per_factor=sigma, correlation_max=rho_max)
    phase2_rec = selector.preliminary_recommendation()

    # Live scorer (L2 — baseline; selector is independent)
    rng    = np.random.default_rng(42)
    offset = rng.uniform(-0.15, 0.15, mu_true.shape)
    scorer = ProfileScorer(
        np.clip(mu_true + offset, 0, 1), actions,
        scoring_kernel=L2Kernel(),
        eta_override=ETA_OVERRIDE,
    )
    scorer.eta     = ETA
    scorer.eta_neg = ETA_NEG

    trajectory = []    # list of (n_verified, recommended_kernel)
    n_verified  = 0
    decision_count = 0
    last_checkpoint_rec = None

    # Stability tracking: consecutive checkpoints with same recommendation
    stable_window  = []           # last 5 checkpoint recommendations
    stabilized_at  = None         # decision number when stability first achieved

    while n_verified < n_decisions:
        # Generate one day's worth (apd alerts)
        n_alerts = int(rng.poisson(apd))
        for _ in range(n_alerts):
            ci   = int(rng.choice(C, p=cat_w))
            a_gt = int(rng.choice(A, p=gt_arr[ci]))

            z = rng.standard_normal(d)
            f = np.clip(mu_true[ci, a_gt] + noise_array * z, 0, 1)

            res    = scorer.score(f, ci)
            pred_a = res.action_index
            decision_count += 1

            # Analyst verification (~30% of alerts)
            if rng.random() < VERIFY_RATE:
                # Analyst agrees 80% of the time with GT
                analyst_action = a_gt if rng.random() < 0.80 else int(
                    rng.choice([a for a in range(A) if a != a_gt])
                )

                # Shadow comparison: score with ALL kernels
                selector.record_comparison(
                    factors=f,
                    category_index=ci,
                    mu=scorer.centroids,
                    analyst_action_index=analyst_action,
                    actions=actions,
                )

                # Live scorer update
                if analyst_action != pred_a:
                    scorer.update(f, ci, pred_a, False, gt_action_index=analyst_action)
                else:
                    scorer.update(f, ci, pred_a, True)

                n_verified += 1

                # Checkpoint every checkpoint_n verified decisions
                if n_verified % checkpoint_n == 0:
                    rec = selector.recommend()
                    k   = rec.recommended_kernel
                    trajectory.append({
                        "n_verified": n_verified,
                        "recommended": k,
                        "method": rec.method,
                        "sufficient_data": rec.sufficient_data,
                    })

                    # Stability check: 5 consecutive same recommendation
                    stable_window.append(k)
                    if len(stable_window) > 5:
                        stable_window.pop(0)
                    if (len(stable_window) == 5
                            and len(set(stable_window)) == 1
                            and stabilized_at is None):
                        stabilized_at = n_verified - 4 * checkpoint_n

    # Phase 4 final recommendation
    phase4_rec = selector.recommend()
    summary    = selector.get_comparison_summary()

    return {
        "label":             label,
        "phase2_kernel":     phase2_rec.recommended_kernel,
        "phase2_reason":     phase2_rec.reason,
        "phase4_kernel":     phase4_rec.recommended_kernel,
        "phase4_method":     phase4_rec.method,
        "phase4_reason":     phase4_rec.reason,
        "phase4_margin":     round(phase4_rec.confidence, 4),
        "sufficient_data":   phase4_rec.sufficient_data,
        "stabilized_at":     stabilized_at,
        "n_verified":        n_verified,
        "agreement_rates":   {k: round(v["agreement_rate"], 4) for k, v in summary.items()},
        "total_decisions":   {k: v["total_decisions"] for k, v in summary.items()},
        "mean_confidence":   {k: round(v["mean_confidence"], 4) for k, v in summary.items()},
        "trajectory":        trajectory,
    }


def run_deliverable4(soc_config: dict, s2p_config: dict) -> dict:
    print()
    print("=" * 66)
    print("DELIVERABLE 4: KERNEL SELECTOR SELF-TEST")
    print("=" * 66)

    d_soc = HC_FACTOR_NOISE.shape[0]
    d_s2p = len(S2P_FACTOR_NAMES)

    # Deployment A: Healthcare SOC
    sigma_hc  = HC_FACTOR_NOISE
    rho_hc    = 0.15
    noise_hc  = HC_FACTOR_NOISE   # use actual per-factor σ directly

    print("\n  Deployment A: Healthcare SOC  (σ_mean={:.3f}  ρ_max={:.2f})".format(
        sigma_hc.mean(), rho_hc))
    dA = run_selector_shadow(
        label       = "Healthcare SOC",
        sigma       = sigma_hc,
        rho_max     = rho_hc,
        config      = soc_config,
        noise_array = noise_hc,
        n_decisions = 500,
        checkpoint_n = 10,
        apd         = 100,
    )

    # Deployment B: S2P Manufacturing
    sigma_s2p  = make_noise_array(0.15, S2P_HETERO_RATIOS, d_s2p)
    rho_s2p    = 0.60
    noise_s2p  = sigma_s2p

    print("\n  Deployment B: S2P Manufacturing  (σ_mean={:.3f}  ρ_max={:.2f})".format(
        sigma_s2p.mean(), rho_s2p))
    dB = run_selector_shadow(
        label       = "S2P Manufacturing",
        sigma       = sigma_s2p,
        rho_max     = rho_s2p,
        config      = s2p_config,
        noise_array = noise_s2p,
        n_decisions = 500,
        checkpoint_n = 10,
        apd         = 100,
    )

    # Print per-deployment report
    for dep, fn in [(dA, HC_FACTOR_NAMES), (dB, S2P_FACTOR_NAMES)]:
        print()
        print(f"  ── {dep['label']} ──")
        print(f"  Phase 2 (preliminary) → {dep['phase2_kernel'].upper()}")
        print(f"    Reason: {dep['phase2_reason']}")
        print(f"  Phase 4 (empirical)   → {dep['phase4_kernel'].upper()}"
              f"  [{dep['phase4_method']}]")
        print(f"    Reason: {dep['phase4_reason']}")
        print(f"  Stabilized at:  {dep['stabilized_at']} verified decisions"
              if dep['stabilized_at'] else "  Stabilized at:  not reached in 500 decisions")
        print()
        print(f"  Agreement rates:")
        for kname, rate in sorted(dep["agreement_rates"].items()):
            n    = dep["total_decisions"][kname]
            conf = dep["mean_confidence"][kname]
            print(f"    {kname:<12}: {rate:.1%}  ({n} decisions"
                  f"  mean_confidence={conf:.3f})")

        # Trajectory (every 5th checkpoint to keep output short)
        traj = dep["trajectory"]
        print()
        print(f"  Trajectory (every 50 verified decisions):")
        for t in traj:
            if t["n_verified"] % 50 == 0:
                flag = " ← locked" if (dep["stabilized_at"] is not None
                                        and t["n_verified"] >= dep["stabilized_at"]
                                        and t["n_verified"] <= dep["stabilized_at"] + 50) else ""
                print(f"    n={t['n_verified']:>4}: {t['recommended']:<12}"
                      f" [{t['method']}]{flag}")

    print()
    print("  VERDICT (Deliverable 4):")
    p2_hc = dA["phase2_kernel"]
    p4_hc = dA["phase4_kernel"]
    p2_s2p = dB["phase2_kernel"]
    p4_s2p = dB["phase4_kernel"]
    agree_hc  = (p2_hc  == p4_hc)
    agree_s2p = (p2_s2p == p4_s2p)

    print(f"    Healthcare: Phase2={p2_hc}  Phase4={p4_hc}"
          f"  {'AGREE' if agree_hc else 'DIFFER'}")
    print(f"    S2P:        Phase2={p2_s2p}  Phase4={p4_s2p}"
          f"  {'AGREE' if agree_s2p else 'DIFFER'}")

    if agree_hc and agree_s2p:
        print("    Rule-based heuristic (Phase 2) confirmed empirically (Phase 4).")
        print("    KernelSelector adds validation but rules are sufficient.")
    elif not agree_hc or not agree_s2p:
        print("    Empirical Phase 4 OVERRIDES rule-based Phase 2 for at least one deployment.")
        print("    Shadow comparison is NECESSARY — rules alone are insufficient.")
        print("    → KernelSelector shadow mode is a required component, not optional.")

    return {
        "deliverable":     "D4_kernel_selector_self_test",
        "deployment_A":    {k: v for k, v in dA.items() if k != "trajectory"},
        "deployment_A_trajectory": dA["trajectory"],
        "deployment_B":    {k: v for k, v in dB.items() if k != "trajectory"},
        "deployment_B_trajectory": dB["trajectory"],
    }


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    soc_config = load_domain_config("soc_product_v50")
    s2p_config = load_domain_config("s2p_v03")

    print()
    print("=" * 66)
    print("=== KERNEL DELIVERABLES: D2 + D3 + D4 ===")
    print("=" * 66)

    t_start = time.time()

    d2 = run_deliverable2(soc_config)
    d3 = run_deliverable3(s2p_config)
    d4 = run_deliverable4(soc_config, s2p_config)

    total_time = time.time() - t_start
    print()
    print(f"  Total runtime: {total_time:.1f}s")

    # Save
    output_dir  = Path("experiments/factorial/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kernel_deliverables.json"

    all_results = {
        "meta": {
            "n_seeds_d2": HC_N_SEEDS,
            "n_seeds_d3": S2P_N_SEEDS,
            "days": HC_DAYS,
            "eta": ETA,
            "eta_override": ETA_OVERRIDE,
            "total_runtime_s": round(total_time, 1),
        },
        "D2": d2,
        "D3": d3,
        "D4": d4,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved → {output_path}")
    print()


if __name__ == "__main__":
    main()
