"""
V-SIM v2 — Phase 4 accuracy instrumented.
Records per-kernel shadow accuracy to answer KERNELSEL-001 urgency.

For each stream: both L2 and DiagonalKernel are run in parallel over the
250-decision shadow window. Accuracy of each is compared against
the kernel that KernelSelector locked at decision 250.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gae

# ---------------------------------------------------------------------------
# Constants — identical to original V-SIM
# ---------------------------------------------------------------------------
N_SHADOW     = 250
N_POST       = 50
THETA_MIN    = 0.467
TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
Q_BAR        = 0.75
ALPHA        = 0.80

FACTOR_NAMES = [
    "travel_match", "asset_criticality", "threat_intel_enrichment",
    "time_anomaly", "pattern_history", "device_trust",
]
N_FACTORS = len(FACTOR_NAMES)

CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat"]
ACTIONS    = ["auto_close", "escalate_tier2", "enrich_and_watch", "escalate_incident"]
N_CATS = len(CATEGORIES)
N_ACTS = len(ACTIONS)

THRESHOLDS = {
    "l2":       {"GREEN": 0.105, "AMBER": 0.157},
    "diagonal": {"GREEN": 0.157, "AMBER": 0.25},
}

SIGMA_PROFILES = {
    "A_healthcare":    np.array([0.180, 0.060, 0.070, 0.080, 0.095, 0.220]),
    "B_finserv":       np.array([0.105, 0.085, 0.090, 0.080, 0.095, 0.110]),
    "C_manufacturing": np.array([0.150, 0.070, 0.090, 0.075, 0.100, 0.160]),
}

EXPECTED_KERNEL = {
    "A_healthcare":    "diagonal",
    "B_finserv":       "l2",
    "C_manufacturing": "diagonal",
}
EXPECTED_GATE = {
    "A_healthcare":    "GREEN_or_AMBER",
    "B_finserv":       "GREEN",
    "C_manufacturing": "GREEN_or_AMBER",
}

JUDGE_MIXES = {
    "A": [0.25, 0.25, 0.20, 0.15, 0.15],
    "B": [0.40, 0.20, 0.15, 0.15, 0.10],
    "C": [0.15, 0.35, 0.20, 0.15, 0.15],
}


# ---------------------------------------------------------------------------
# Helpers (identical to original)
# ---------------------------------------------------------------------------
def make_mu_warm(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = np.zeros((N_CATS, N_ACTS, N_FACTORS))
    for c in range(N_CATS):
        for a in range(N_ACTS):
            base = 0.3 + 0.1 * a
            mu[c, a, :] = np.clip(base + rng.normal(0, 0.05, N_FACTORS), 0.1, 0.9)
    return mu


def generate_stream(sigma_profile: np.ndarray, judge: str,
                    n_decisions: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cat_weights = np.array(JUDGE_MIXES[judge])
    cat_weights /= cat_weights.sum()

    mu_gt = np.zeros((N_CATS, N_ACTS, N_FACTORS))
    rng2 = np.random.default_rng(seed + 1000)
    for c in range(N_CATS):
        for a in range(N_ACTS):
            base = 0.3 + 0.1 * a
            mu_gt[c, a, :] = np.clip(base + rng2.normal(0, 0.05, N_FACTORS), 0.1, 0.9)

    alerts = []
    for _ in range(n_decisions):
        cat_idx = int(rng.choice(N_CATS, p=cat_weights))
        gt_act  = int(rng.choice(N_ACTS, p=np.array([0.30, 0.30, 0.25, 0.15])))
        f = mu_gt[cat_idx, gt_act, :] + rng.normal(0, sigma_profile)
        f = np.clip(f, 0.0, 1.0)
        alerts.append({
            "factors":         f,
            "category_index":  cat_idx,
            "gt_action_index": gt_act,
        })
    return alerts


def classify_gate(sigma_mean: float, kernel: str) -> str:
    thr = THRESHOLDS[kernel]
    if sigma_mean <= thr["GREEN"]:
        return "GREEN"
    elif sigma_mean <= thr["AMBER"]:
        return "AMBER"
    return "RED"


def gate_correct(gate: str, industry: str) -> bool:
    if industry == "B_finserv":
        return gate == "GREEN"
    return gate in ("GREEN", "AMBER")


# ---------------------------------------------------------------------------
# Single stream — instrumented
# ---------------------------------------------------------------------------
def run_stream(industry: str, judge: str, seed: int) -> dict:
    sigma_profile = SIGMA_PROFILES[industry]
    noise_ratio   = sigma_profile.max() / sigma_profile.min()
    sigma_mean    = sigma_profile.mean()

    # Sanity checks (same as original)
    if industry == "B_finserv" and noise_ratio >= 1.5:
        raise RuntimeError(f"STOP: FinServ noise_ratio={noise_ratio:.3f} >= 1.5")
    if industry in ("A_healthcare", "C_manufacturing") and noise_ratio <= 1.5:
        raise RuntimeError(f"STOP: {industry} noise_ratio={noise_ratio:.3f} <= 1.5")

    # KernelSelector (same as original)
    ks = gae.KernelSelector(
        d=N_FACTORS,
        sigma_per_factor=sigma_profile,
        correlation_max=0.0,
        window_size=100,
    )

    # Warm-start centroids — shared initial state
    mu_init = make_mu_warm(seed)

    # DiagonalKernel weights: 1/sigma^2 per factor (precision weighting)
    diag_weights = 1.0 / (sigma_profile ** 2)
    diag_weights = diag_weights / diag_weights.sum() * N_FACTORS  # normalise scale

    # Two independent ProfileScorers: L2 and Diagonal
    profile = gae.CalibrationProfile(
        learning_rate=ETA_CONFIRM, penalty_ratio=1.0, temperature=TAU
    )
    scorer_l2 = gae.ProfileScorer(
        mu=mu_init.copy(), actions=ACTIONS,
        scoring_kernel=gae.L2Kernel(),
        profile=profile, eta_override=ETA_OVERRIDE,
    )
    scorer_diag = gae.ProfileScorer(
        mu=mu_init.copy(), actions=ACTIONS,
        scoring_kernel=gae.DiagonalKernel(diag_weights),
        profile=profile, eta_override=ETA_OVERRIDE,
    )

    # Stream
    alerts = generate_stream(sigma_profile, judge, N_SHADOW + N_POST, seed)

    # Phase 4: shadow run — 250 decisions
    correct_l2   = []
    correct_diag = []
    rollover_switches = 0
    last_rec = None
    stable_by_200 = True

    for i, alert in enumerate(alerts[:N_SHADOW]):
        f          = alert["factors"]
        cat_idx    = alert["category_index"]
        gt_act_idx = alert["gt_action_index"]

        # KernelSelector comparison recording (for lock decision)
        ks.record_comparison(
            factors=f,
            category_index=cat_idx,
            mu=mu_init,
            analyst_action_index=gt_act_idx,
            actions=ACTIONS,
        )

        # Rollover stability tracking
        current_rec = ks.recommend()
        if i >= 100:
            if last_rec is not None and current_rec.recommended_kernel != last_rec:
                rollover_switches += 1
                if i >= 200:
                    stable_by_200 = False
            last_rec = current_rec.recommended_kernel

        # L2 scorer: score and update
        res_l2 = scorer_l2.score(f, cat_idx)
        correct_l2.append(int(res_l2.action_index == gt_act_idx))
        scorer_l2.update(
            f=f, category_index=cat_idx,
            action_index=gt_act_idx, correct=True,
            gt_action_index=gt_act_idx,
        )

        # Diagonal scorer: score and update
        res_diag = scorer_diag.score(f, cat_idx)
        correct_diag.append(int(res_diag.action_index == gt_act_idx))
        scorer_diag.update(
            f=f, category_index=cat_idx,
            action_index=gt_act_idx, correct=True,
            gt_action_index=gt_act_idx,
        )

    # Per-kernel shadow accuracy
    acc_l2   = float(np.mean(correct_l2))   * 100.0
    acc_diag = float(np.mean(correct_diag)) * 100.0
    margin   = acc_diag - acc_l2
    shadow_winner = "diagonal" if margin > 0 else ("l2" if margin < 0 else "tie")

    # Phase 5: lock
    final_rec     = ks.recommend()
    kernel_selected = final_rec.recommended_kernel

    # Phase 6: DeploymentQualifier gate
    gate   = classify_gate(sigma_mean, kernel_selected)
    k_corr = (kernel_selected == EXPECTED_KERNEL[industry])
    g_corr = gate_correct(gate, industry)
    lock_agrees = (kernel_selected == shadow_winner)

    return {
        "industry":                  industry,
        "judge":                     judge,
        "noise_ratio":               round(float(noise_ratio), 4),
        "accuracy_l2_shadow":        round(acc_l2, 2),
        "accuracy_diagonal_shadow":  round(acc_diag, 2),
        "shadow_winner":             shadow_winner,
        "shadow_margin_pp":          round(abs(margin), 2),
        "kernel_selected":           kernel_selected,
        "kernel_correct":            k_corr,
        "lock_agrees_with_shadow":   lock_agrees,
        "gate_classification":       gate,
        "gate_correct":              g_corr,
        "rollover_switches_200_250": rollover_switches,
        "rollover_stable_by_200":    stable_by_200,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    industries = ["A_healthcare", "B_finserv", "C_manufacturing"]
    judges     = ["A", "B", "C"]
    seeds      = [42, 123, 456]

    streams = []
    for ind in industries:
        for j_idx, judge in enumerate(judges):
            streams.append(run_stream(ind, judge, seeds[j_idx]))

    k_correct_count    = sum(1 for s in streams if s["kernel_correct"])
    g_correct_count    = sum(1 for s in streams if s["gate_correct"])
    lock_agree_count   = sum(1 for s in streams if s["lock_agrees_with_shadow"])
    g1_pass  = k_correct_count  >= 8
    g2_pass  = g_correct_count  >= 8
    g3_pass  = lock_agree_count >= 8

    # KERNELSEL-001 verdict from C_manufacturing-B
    cmb = next(s for s in streams
               if s["industry"] == "C_manufacturing" and s["judge"] == "B")
    margin_cmb = cmb["accuracy_diagonal_shadow"] - cmb["accuracy_l2_shadow"]
    if abs(margin_cmb) < 1.0:
        verdict = "ambiguous"
    elif cmb["shadow_winner"] == "diagonal" and cmb["kernel_selected"] != "diagonal":
        verdict = "pre-ship"
    elif cmb["shadow_winner"] == "l2":
        verdict = "post-mvp"
    else:
        verdict = "ambiguous"

    # Save
    out = {
        "experiment":          "V-SIM-v2",
        "gae_version":         gae.__version__,
        "streams":             streams,
        "kernelselector_accuracy": f"{k_correct_count}/9",
        "gate_accuracy":           f"{g_correct_count}/9",
        "lock_shadow_agreement":   f"{lock_agree_count}/9",
        "g1_pass":  g1_pass,
        "g2_pass":  g2_pass,
        "g3_pass":  g3_pass,
        "kernelsel_001_verdict": verdict,
    }
    out_path = REPO_ROOT / "experiments" / "v_sim" / "results" / "results_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    gae_ver = gae.__version__
    print(f"V-SIM v2 -- Phase 4 accuracy instrumented (GAE {gae_ver}):")
    if gae_ver != "0.7.8":
        print(f"  NOTE: running on GAE {gae_ver} (requested 0.7.8).")
    print()
    hdr = (f"{'Stream':<22} | {'ratio':>6} | {'L2_acc':>7} | {'Diag_acc':>9} | "
           f"{'Shadow_winner':<14} | {'Selected':<9} | {'Lock=Shadow':>11} | {'Gate':<5}")
    print(hdr)
    print("-" * len(hdr))
    for s in streams:
        name = f"{s['industry']}-{s['judge']}"
        print(
            f"{name:<22} | {s['noise_ratio']:>5.2f}x | "
            f"{s['accuracy_l2_shadow']:>6.2f}% | "
            f"{s['accuracy_diagonal_shadow']:>8.2f}% | "
            f"{s['shadow_winner']:<14} | "
            f"{s['kernel_selected']:<9} | "
            f"{'Y' if s['lock_agrees_with_shadow'] else 'N':>11} | "
            f"{s['gate_classification']:<5}"
        )

    print()
    print(f"G1 KernelSelector accuracy:  {k_correct_count}/9  -> {'PASS' if g1_pass else 'FAIL'}")
    print(f"G2 Deployment gate accuracy: {g_correct_count}/9  -> {'PASS' if g2_pass else 'FAIL'}")
    print(f"G3 Lock agrees with shadow:  {lock_agree_count}/9  -> {'PASS' if g3_pass else 'FAIL'}")

    print()
    print(f"KEY -- C_manufacturing-B (ratio={cmb['noise_ratio']:.2f}x, expected diagonal):")
    print(f"  L2 shadow accuracy:       {cmb['accuracy_l2_shadow']:.2f}%")
    print(f"  Diagonal shadow accuracy: {cmb['accuracy_diagonal_shadow']:.2f}%")
    print(f"  Shadow winner: {cmb['shadow_winner']} (margin: {cmb['shadow_margin_pp']:.2f}pp)")
    print(f"  Kernel selected (lock):   {cmb['kernel_selected']}")
    print(f"  Lock agrees with shadow:  {'Y' if cmb['lock_agrees_with_shadow'] else 'N'}")
    if cmb["shadow_winner"] == "diagonal" and cmb["kernel_selected"] != "diagonal":
        print("  VERDICT: Diagonal won shadow but L2 was locked -- PRE-SHIP FIX REQUIRED")
    elif abs(margin_cmb) < 1.0:
        print("  VERDICT: Margin < 1pp -- AMBIGUOUS")
    else:
        print("  VERDICT: L2 genuinely won shadow -- POST-MVP")

    print()
    print(f"KERNELSEL-001 urgency: {verdict.upper()}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
