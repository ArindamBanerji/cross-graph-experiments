"""
TD-034 — tau calibration validation on realistic alert distributions.

tau sweep [0.05, 0.08, 0.10, 0.12, 0.15] across 3 SOC streams.
10 seeds x 5 tau values x 3 streams = 150 runs x 200 decisions = 30,000 decisions.
Metric: ECE (Expected Calibration Error, 10 bins).
Gate: tau=0.10 achieves ECE <= 0.05 on >= 2/3 streams.

GENERATION DESIGN:
  Perturbed warm start: mu_init = MU_STAR + N(0, 0.20)
  Simulates new deployment with imperfect transferred profiles.
  Creates genuine overconfidence at low tau (pre-calibration regime).
  Alert generation: action-conditional centroidal with per-stream sigma.
  Online learning during the 200-decision window.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile
from gae.evaluation import compute_ece

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TAU_VALUES  = [0.05, 0.08, 0.10, 0.12, 0.15]
N_SEEDS     = 10
N_DECISIONS = 200

PERTURB_SIGMA = 0.20   # centroid init noise — simulates imperfect profile transfer

THETA_MIN    = 0.467
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
Q_BAR        = 0.75
ALPHA        = 0.80

SEEDS_10 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# ---------------------------------------------------------------------------
# A1xB1 SOC Healthcare Geometry (canonical)
# Factor order: [travel_match(0), asset_criticality(1),
#                threat_intel_enrichment(2), time_anomaly(3),
#                pattern_history(4), device_trust(5)]
# ---------------------------------------------------------------------------
FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]

N_CATS    = len(CATEGORIES)    # 6
N_ACTS    = len(ACTIONS)       # 4
N_FACTORS = len(FACTOR_NAMES)  # 6

CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

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
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu


MU_STAR = build_mu_star()

# GT distribution: biased toward action with highest L2-norm centroid per category
def build_gt_dist() -> np.ndarray:
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt


GT_DIST = build_gt_dist()

# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------
STREAMS = {
    "healthcare": {
        # sigma: [travel_match, asset_criticality, threat_intel, time_anomaly,
        #         pattern_history, device_trust]
        "sigma": np.array([0.18, 0.06, 0.07, 0.08, 0.095, 0.22]),
        # biased toward high-threat: 40% lateral_movement, 30% credential_access, 30% other
        "cat_weights": np.array([0.30, 0.0667, 0.40, 0.0667, 0.0667, 0.10]),
    },
    "finserv": {
        "sigma": np.array([0.105, 0.085, 0.090, 0.080, 0.095, 0.110]),
        # balanced across categories
        "cat_weights": np.ones(N_CATS) / N_CATS,
    },
    "enterprise": {
        "sigma": np.array([0.15, 0.07, 0.12, 0.09, 0.14, 0.18]),
        # uniform across all 6 categories
        "cat_weights": np.ones(N_CATS) / N_CATS,
    },
}
for s in STREAMS.values():
    s["cat_weights"] = s["cat_weights"] / s["cat_weights"].sum()


# ---------------------------------------------------------------------------
# Single run: one seed, one tau, one stream
#
# DESIGN:
#   Perturbed warm start: mu = MU_STAR + N(0, PERTURB_SIGMA)
#   This simulates a new deployment where profiles were transferred from
#   a similar SOC (approximately correct but not perfectly tuned). This
#   creates the pre-calibration regime where tau=0.05 is overconfident.
#
#   Alert generation: action-conditional centroidal
#     f = MU_STAR[c, gt_act] + sigma_noise
#   Online learning during the 200-decision evaluation window.
# ---------------------------------------------------------------------------
def run_one(stream_name: str, tau: float, seed: int) -> dict:
    cfg   = STREAMS[stream_name]
    sigma = cfg["sigma"]
    cat_w = cfg["cat_weights"]

    rng = np.random.default_rng(seed)

    profile = CalibrationProfile(
        learning_rate=ETA_CONFIRM,
        penalty_ratio=1.0,
        temperature=tau,
    )

    # Perturbed warm start: imperfect profile transfer
    mu0 = np.clip(MU_STAR + rng.normal(0, PERTURB_SIGMA, MU_STAR.shape), 0.0, 1.0)

    scorer = ProfileScorer(
        mu=mu0,
        actions=ACTIONS,
        profile=profile,
        eta_override=ETA_OVERRIDE,
    )

    confidences   = []
    correct_flags = []

    for _t in range(N_DECISIONS):
        cat_idx = int(rng.choice(N_CATS, p=cat_w))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f       = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, sigma), 0.0, 1.0)

        result = scorer.score(f, cat_idx)

        confidences.append(float(result.confidence))
        correct_flags.append(bool(result.action_index == gt_act))

        # Clean analyst feedback (Q_BAR quality)
        if rng.random() < Q_BAR:
            label_act = gt_act
        else:
            label_act = int(rng.choice(N_ACTS))

        scorer.update(
            f=f,
            category_index=cat_idx,
            action_index=label_act,
            correct=(label_act == gt_act),
            gt_action_index=gt_act,
        )

    ece      = compute_ece(confidences, correct_flags, n_bins=10)
    accuracy = float(np.mean(correct_flags))

    return {"ece": ece, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("TD-034 running (150 runs x 200 decisions) ...", flush=True)

    # Results store: stream -> tau_str -> list of per-seed dicts
    all_results: dict[str, dict[str, list[dict]]] = {
        s: {str(t): [] for t in TAU_VALUES} for s in STREAMS
    }

    for stream_name in STREAMS:
        print(f"  Stream: {stream_name}", flush=True)
        for tau in TAU_VALUES:
            tau_str = str(tau)
            for seed in SEEDS_10:
                r = run_one(stream_name, tau, seed)
                all_results[stream_name][tau_str].append(r)
            mean_ece = float(np.mean([r["ece"] for r in all_results[stream_name][tau_str]]))
            print(f"    tau={tau:.2f}  mean_ECE={mean_ece:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # Aggregate per (stream, tau)
    # -------------------------------------------------------------------------
    def agg(stream: str, tau: float) -> tuple[float, float]:
        rows = all_results[stream][str(tau)]
        return (float(np.mean([r["ece"]      for r in rows])),
                float(np.mean([r["accuracy"] for r in rows])))

    stream_data = {}
    for stream_name in STREAMS:
        tau_sweep = {}
        for tau in TAU_VALUES:
            ece, acc = agg(stream_name, tau)
            tau_sweep[str(tau)] = {"ece": round(ece, 6), "accuracy": round(acc, 4)}

        # Optimal tau (minimum ECE)
        eces      = {tau: tau_sweep[str(tau)]["ece"] for tau in TAU_VALUES}
        optimal_tau = min(eces, key=eces.get)
        ece_at_010  = tau_sweep["0.1"]["ece"]
        gate_pass   = ece_at_010 <= 0.05

        stream_data[stream_name] = {
            "tau_sweep":     tau_sweep,
            "optimal_tau":   optimal_tau,
            "ece_at_tau_01": ece_at_010,
            "gate_pass":     gate_pass,
        }

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    # 1. ECE at tau=0.05 > ECE at tau=0.10 for at least 2/3 streams
    s1_count = sum(
        1 for s in STREAMS
        if stream_data[s]["tau_sweep"]["0.05"]["ece"]
           > stream_data[s]["tau_sweep"]["0.1"]["ece"]
    )
    if s1_count < 2:
        print(f"STOP (sanity 1): ECE at tau=0.05 not higher than tau=0.10 in "
              f"{s1_count}/3 streams — ECE monotonically decreases with tau. "
              f"Underconfident regime at all tested tau values.")
        sys.exit(1)

    # 2. ECE at tau=0.15 > ECE at tau=0.10 for at least 2/3 streams (warning)
    s2_count = sum(
        1 for s in STREAMS
        if stream_data[s]["tau_sweep"]["0.15"]["ece"]
           > stream_data[s]["tau_sweep"]["0.1"]["ece"]
    )

    # 3. Optimal tau range check
    opt_taus    = [stream_data[s]["optimal_tau"] for s in STREAMS]
    flag_range  = all(t < 0.06 or t > 0.14 for t in opt_taus)

    # -------------------------------------------------------------------------
    # Overall gate
    # -------------------------------------------------------------------------
    gate_streams_pass   = sum(1 for s in STREAMS if stream_data[s]["gate_pass"])
    overall_gate_pass   = gate_streams_pass >= 2
    per_deploy_required = not overall_gate_pass or any(
        stream_data[s]["optimal_tau"] != 0.1 for s in STREAMS
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out = {
        "experiment":                    "TD-034",
        "gae_version":                   "0.7.8",
        "n_seeds":                       N_SEEDS,
        "perturb_sigma":                 PERTURB_SIGMA,
        "streams":                       stream_data,
        "tau_01_gate_streams_pass":      gate_streams_pass,
        "overall_gate_pass":             overall_gate_pass,
        "per_deployment_sweep_required": per_deploy_required,
        "sanity_1_tau005_gt_tau010_count": s1_count,
        "sanity_2_tau015_gt_tau010_count": s2_count,
        "flag_optimal_tau_out_of_range":   flag_range,
    }

    results_path = REPO_ROOT / "experiments" / "td_034" / "results" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {results_path}")
    print()

    # -------------------------------------------------------------------------
    # Print report
    # -------------------------------------------------------------------------
    print("TD-034 — tau calibration sweep (3 streams, GAE 0.7.8):")
    print()

    header = (f"{'Stream':<12} | {'t=0.05':>6} | {'t=0.08':>6} | "
              f"{'t=0.10':>6} | {'t=0.12':>6} | {'t=0.15':>6} | {'optimal_t':>9}")
    print(header)
    print("-" * len(header))
    for stream_name, label in [("healthcare", "Healthcare"),
                                ("finserv",    "FinServ"),
                                ("enterprise", "Enterprise")]:
        ts  = stream_data[stream_name]["tau_sweep"]
        opt = stream_data[stream_name]["optimal_tau"]
        row = (f"{label:<12} | "
               f"{ts['0.05']['ece']:>6.4f} | "
               f"{ts['0.08']['ece']:>6.4f} | "
               f"{ts['0.1']['ece']:>6.4f} | "
               f"{ts['0.12']['ece']:>6.4f} | "
               f"{ts['0.15']['ece']:>6.4f} | "
               f"{opt:>9}")
        print(row)

    print()
    hc_ece = stream_data["healthcare"]["ece_at_tau_01"]
    fs_ece = stream_data["finserv"]["ece_at_tau_01"]
    en_ece = stream_data["enterprise"]["ece_at_tau_01"]
    print(f"ECE at tau=0.10: Healthcare={hc_ece:.4f} "
          f"FinServ={fs_ece:.4f} Enterprise={en_ece:.4f}")
    print(f"Gate (ECE<=0.05 at tau=0.10, >=2/3 streams): "
          f"{gate_streams_pass}/3 streams pass -> "
          f"{'PASS' if overall_gate_pass else 'FAIL'}")
    print(f"Per-deployment tau sweep required: "
          f"{'yes' if per_deploy_required else 'no'}")
    if flag_range:
        print("FLAG: All streams optimal tau outside 0.06-0.14 range.")
    print("Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
