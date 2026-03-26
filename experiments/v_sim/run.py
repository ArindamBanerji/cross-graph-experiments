"""
V-SIM — P28 Pipeline End-to-End Validation on 9 Synthetic Streams.

3 industries × 3 judge variants = 9 streams.
Each stream: 250 shadow decisions + 50 post-lock decisions = 300 total.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gae

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SHADOW    = 250
N_POST      = 50
THETA_MIN   = 0.467
TAU         = 0.1
ETA_CONFIRM = 0.05
ETA_OVERRIDE = 0.01
Q_BAR       = 0.75
ALPHA       = 0.80

# Factor order: [travel_match(0), asset_criticality(1),
#                threat_intel_enrichment(2), time_anomaly(3),
#                pattern_history(4), device_trust(5)]
FACTOR_NAMES = [
    "travel_match",
    "asset_criticality",
    "threat_intel_enrichment",
    "time_anomaly",
    "pattern_history",
    "device_trust",
]
N_FACTORS = len(FACTOR_NAMES)

CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat"]
ACTIONS    = ["auto_close", "escalate_tier2", "enrich_and_watch", "escalate_incident"]
N_CATS = len(CATEGORIES)
N_ACTS = len(ACTIONS)

# DeploymentQualifier thresholds (sigma_mean based)
THRESHOLDS = {
    "l2":       {"GREEN": 0.105, "AMBER": 0.157},
    "diagonal": {"GREEN": 0.157, "AMBER": 0.25},
}

# ---------------------------------------------------------------------------
# Industry sigma profiles
# Factor order: travel_match, asset_criticality, threat_intel_enrichment,
#               time_anomaly, pattern_history, device_trust
# ---------------------------------------------------------------------------
SIGMA_PROFILES = {
    "A_healthcare": np.array([0.180, 0.060, 0.070, 0.080, 0.095, 0.220]),
    "B_finserv":    np.array([0.105, 0.085, 0.090, 0.080, 0.095, 0.110]),
    "C_manufacturing": np.array([0.150, 0.070, 0.090, 0.075, 0.100, 0.160]),
}

# Expected outcomes
EXPECTED_KERNEL = {
    "A_healthcare":   "diagonal",
    "B_finserv":      "l2",
    "C_manufacturing": "diagonal",
}
EXPECTED_GATE = {
    "A_healthcare":    "GREEN_or_AMBER",
    "B_finserv":       "GREEN",
    "C_manufacturing": "GREEN_or_AMBER",
}

# Judge category mix variants (3 judges per industry — vary category weights)
JUDGE_MIXES = {
    "A": [0.25, 0.25, 0.20, 0.15, 0.15],  # balanced
    "B": [0.40, 0.20, 0.15, 0.15, 0.10],  # credential_access heavy
    "C": [0.15, 0.35, 0.20, 0.15, 0.15],  # threat_intel heavy
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mu_warm(seed: int) -> np.ndarray:
    """Warm-start centroids: shape (n_cats, n_acts, n_factors)."""
    rng = np.random.default_rng(seed)
    # Spread centroids so actions are distinguishable
    mu = np.zeros((N_CATS, N_ACTS, N_FACTORS))
    for c in range(N_CATS):
        for a in range(N_ACTS):
            # Each action gets a distinct centroid offset
            base = 0.3 + 0.1 * a
            mu[c, a, :] = np.clip(base + rng.normal(0, 0.05, N_FACTORS), 0.1, 0.9)
    return mu


def generate_stream(sigma_profile: np.ndarray, judge: str,
                    n_decisions: int, seed: int) -> list[dict]:
    """Generate alert stream for one industry+judge combination.

    Each alert has:
      - factors: np.ndarray shape (N_FACTORS,)
      - category_index: int
      - gt_action_index: int
    """
    rng = np.random.default_rng(seed)
    cat_weights = np.array(JUDGE_MIXES[judge])
    cat_weights /= cat_weights.sum()

    # Construct ground-truth centroids (fixed per stream)
    mu_gt = np.zeros((N_CATS, N_ACTS, N_FACTORS))
    rng2 = np.random.default_rng(seed + 1000)
    for c in range(N_CATS):
        for a in range(N_ACTS):
            base = 0.3 + 0.1 * a
            mu_gt[c, a, :] = np.clip(base + rng2.normal(0, 0.05, N_FACTORS), 0.1, 0.9)

    alerts = []
    for _ in range(n_decisions):
        cat_idx = int(rng.choice(N_CATS, p=cat_weights))
        # Ground-truth action: slightly prefer higher actions for higher cats
        gt_act = int(rng.choice(N_ACTS, p=np.array([0.30, 0.30, 0.25, 0.15])))
        # Factor vector: centroid + per-factor noise using sigma_profile
        f = mu_gt[cat_idx, gt_act, :] + rng.normal(0, sigma_profile)
        f = np.clip(f, 0.0, 1.0)
        alerts.append({
            "factors": f,
            "category_index": cat_idx,
            "gt_action_index": gt_act,
        })
    return alerts


def classify_gate(sigma_mean: float, kernel: str) -> str:
    thr = THRESHOLDS[kernel]
    if sigma_mean <= thr["GREEN"]:
        return "GREEN"
    elif sigma_mean <= thr["AMBER"]:
        return "AMBER"
    else:
        return "RED"


def gate_correct(gate: str, industry: str) -> bool:
    """Check if gate classification is acceptable for this industry."""
    if industry == "B_finserv":
        return gate == "GREEN"
    else:
        # Healthcare and Manufacturing: GREEN or AMBER acceptable, not RED
        return gate in ("GREEN", "AMBER")


def kernel_correct(kernel: str, industry: str) -> bool:
    return kernel == EXPECTED_KERNEL[industry]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stream(industry: str, judge: str, seed: int) -> dict:
    sigma_profile = SIGMA_PROFILES[industry]
    noise_ratio = sigma_profile.max() / sigma_profile.min()
    sigma_mean = sigma_profile.mean()

    print(f"Stream {industry}-{judge}: sigma_mean={sigma_mean:.3f}, "
          f"noise_ratio={noise_ratio:.2f}x")
    print(f"  Expected kernel: {EXPECTED_KERNEL[industry]}, "
          f"Expected gate: {EXPECTED_GATE[industry]}")

    # Sanity check
    if industry == "B_finserv" and noise_ratio >= 1.5:
        raise RuntimeError(f"STOP: FinServ noise_ratio={noise_ratio:.3f} >= 1.5 — stream wrong")
    if industry in ("A_healthcare", "C_manufacturing") and noise_ratio <= 1.5:
        raise RuntimeError(f"STOP: {industry} noise_ratio={noise_ratio:.3f} <= 1.5")

    # Phase 2: KernelSelector initialized with known sigma (from stream design)
    ks = gae.KernelSelector(
        d=N_FACTORS,
        sigma_per_factor=sigma_profile,
        correlation_max=0.0,
        window_size=100,
    )

    # Phase 3: preliminary recommendation
    prelim = ks.preliminary_recommendation()

    # Warm-start centroids
    mu = make_mu_warm(seed)

    # Generate stream: N_SHADOW + N_POST decisions
    alerts = generate_stream(sigma_profile, judge, N_SHADOW + N_POST, seed)

    # Phase 4: shadow run — 250 decisions, record KernelSelector comparisons
    rollover_switches = 0
    last_rec = None
    stable_by_200 = True

    for i, alert in enumerate(alerts[:N_SHADOW]):
        f          = alert["factors"]
        cat_idx    = alert["category_index"]
        gt_act_idx = alert["gt_action_index"]

        # Record comparison in shadow mode
        ks.record_comparison(
            factors=f,
            category_index=cat_idx,
            mu=mu,
            analyst_action_index=gt_act_idx,
            actions=ACTIONS,
        )

        # Track rollover stability at decision 200+
        current_rec = ks.recommend()
        if i >= 100:  # after first window fills
            if last_rec is not None and current_rec.recommended_kernel != last_rec:
                rollover_switches += 1
                if i >= 200:
                    stable_by_200 = False
            last_rec = current_rec.recommended_kernel

    # Phase 5: Lock KernelSelector at decision 250
    final_rec = ks.recommend()
    kernel_selected = final_rec.recommended_kernel

    # Phase 6: DeploymentQualifier gate using locked kernel + sigma_mean
    gate = classify_gate(sigma_mean, kernel_selected)

    k_correct = kernel_correct(kernel_selected, industry)
    g_correct = gate_correct(gate, industry)

    print(f"  -> kernel={kernel_selected} ({'OK' if k_correct else 'WRONG'}), "
          f"gate={gate} ({'OK' if g_correct else 'WRONG'}), "
          f"stable_by_200={stable_by_200}")

    return {
        "industry": industry,
        "judge": judge,
        "sigma_mean": round(float(sigma_mean), 4),
        "noise_ratio": round(float(noise_ratio), 4),
        "kernel_selected": kernel_selected,
        "kernel_correct": k_correct,
        "gate_classification": gate,
        "gate_correct": g_correct,
        "rollover_stable_by_200": stable_by_200,
        "rollover_switches_200_250": rollover_switches,
    }


def main():
    print("V-SIM — P28 Pipeline Validation (9 streams, GAE 0.7.8):")
    print()

    industries = ["A_healthcare", "B_finserv", "C_manufacturing"]
    judges     = ["A", "B", "C"]
    seeds      = [42, 123, 456]  # one per judge variant

    streams = []
    for ind in industries:
        for j_idx, judge in enumerate(judges):
            seed = seeds[j_idx]
            result = run_stream(ind, judge, seed)
            streams.append(result)

    # Summary table
    print()
    header = (f"{'Stream':<20} | {'sigma_mean':>9} | {'ratio':>7} | "
              f"{'Kernel':<8} | {'Correct':>7} | {'Gate':<5} | {'Correct':>7}")
    print(header)
    print("-" * len(header))
    for s in streams:
        stream_name = f"{s['industry']}-{s['judge']}"
        print(f"{stream_name:<20} | {s['sigma_mean']:>9.3f} | "
              f"{s['noise_ratio']:>6.2f}x | "
              f"{s['kernel_selected']:<8} | "
              f"{'Y' if s['kernel_correct'] else 'N':>7} | "
              f"{s['gate_classification']:<5} | "
              f"{'Y' if s['gate_correct'] else 'N':>7}")

    # Gate results
    k_correct_count = sum(1 for s in streams if s["kernel_correct"])
    g_correct_count = sum(1 for s in streams if s["gate_correct"])
    g1_pass = k_correct_count >= 8
    g2_pass = g_correct_count >= 8
    both_pass = g1_pass and g2_pass

    print()
    print(f"KernelSelector accuracy: {k_correct_count}/9 [gate: >=8/9]  -> {'PASS' if g1_pass else 'FAIL'}")
    print(f"Deployment gate accuracy: {g_correct_count}/9 [gate: >=8/9] -> {'PASS' if g2_pass else 'FAIL'}")
    print(f"Overall: {'PASS' if both_pass else 'FAIL'}")
    print("Raw numbers for roadmap session review.")

    # Save results
    results = {
        "experiment": "V-SIM",
        "gae_version": gae.__version__,
        "streams": streams,
        "kernelselector_accuracy": f"{k_correct_count}/9",
        "gate_accuracy": f"{g_correct_count}/9",
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "both_pass": both_pass,
    }

    out_path = REPO_ROOT / "experiments" / "v_sim" / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
