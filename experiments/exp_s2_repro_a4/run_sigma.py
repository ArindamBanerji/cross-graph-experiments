"""
EXP-S2-REPRO-A4-SIGMA — Poisoning resilience via sigma-perturbation at A=4.

Attack: Gaussian noise injected into factor vectors.
Labels remain CORRECT — only the factor vector presented to update() is noisy.
is_override=False (indistinguishable from clean).

3 arms x 20 seeds x 500 decisions.
  Arm 0: clean (0%)
  Arm 1: 10% adversarial rate
  Arm 2: 20% adversarial rate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS        = 20
N_DECISIONS    = 500
WINDOW_LAST    = 100
CONS_WINDOW    = 50

THETA_MIN      = 0.467
TAU            = 0.1
ETA_CONFIRM    = 0.05
ETA_OVERRIDE   = 0.01
ETA_NEG        = 0.05
Q_BAR          = 0.75
ALPHA          = 0.80
SIGMA_PERTURB  = 0.20   # noise std added to factor vectors in adversarial decisions

SEEDS_20 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]

# ---------------------------------------------------------------------------
# A1xB1 SOC Healthcare Geometry — identical to EXP-S2-REPRO-A4
# ---------------------------------------------------------------------------
FACTOR_NAMES = ["travel_match", "asset_criticality", "threat_intel_enrichment",
                "time_anomaly", "pattern_history", "device_trust"]
ACTIONS    = ["monitor", "investigate", "suppress", "escalate"]
CATEGORIES = ["credential_access", "threat_intel_match", "lateral_movement",
              "data_exfiltration", "insider_threat", "cloud_infrastructure"]

N_CATS    = len(CATEGORIES)
N_ACTS    = len(ACTIONS)
N_FACTORS = len(FACTOR_NAMES)

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


def build_gt_dist() -> np.ndarray:
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt


GT_DIST      = build_gt_dist()
SIGMA_PROFILE = np.array([0.165, 0.090, 0.090, 0.070, 0.095, 0.200])


# ---------------------------------------------------------------------------
# Single-arm, single-seed run
# ---------------------------------------------------------------------------
def run_one(adv_rate: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    profile = CalibrationProfile(
        learning_rate=ETA_CONFIRM,
        penalty_ratio=ETA_NEG / ETA_CONFIRM,
        temperature=TAU,
    )

    mu0 = MU_STAR.copy()
    mu0 += rng.uniform(-0.005, 0.005, mu0.shape)
    np.clip(mu0, 0.0, 1.0, out=mu0)

    scorer = ProfileScorer(
        mu=mu0,
        actions=ACTIONS,
        profile=profile,
        eta_override=ETA_OVERRIDE,
    )

    correct_flags = []
    # For conservation: labels are ALWAYS correct in sigma attack
    # so conservation signal = ALPHA * 1.0 = 0.80 > THETA_MIN throughout.
    # Track anyway for completeness.
    label_clean = []

    for _t in range(N_DECISIONS):
        cat_idx = int(rng.integers(0, N_CATS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))

        # Clean factor vector (used for scoring)
        f_clean = np.clip(
            MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA_PROFILE),
            0.0, 1.0,
        )

        # Score always on clean f
        result   = scorer.score(f_clean, cat_idx)
        pred_act = result.action_index
        correct_flags.append(int(pred_act == gt_act))

        # Determine factor vector for update
        is_adversarial = rng.random() < adv_rate
        if is_adversarial:
            # Add Gaussian noise to factor vector — label stays CORRECT
            eps      = rng.normal(0.0, SIGMA_PERTURB, N_FACTORS)
            f_update = np.clip(f_clean + eps, 0.0, 1.0)
        else:
            f_update = f_clean

        # Label is always correct regardless of adversarial flag
        scorer.update(
            f=f_update,
            category_index=cat_idx,
            action_index=gt_act,
            correct=True,
            gt_action_index=gt_act,
        )
        label_clean.append(1)   # labels always clean for conservation tracking

    final_acc = float(np.mean(correct_flags[-WINDOW_LAST:]))

    # Conservation check (expected: never fires — labels always correct)
    conservation_fired = False
    for i in range(CONS_WINDOW, N_DECISIONS):
        window_q = float(np.mean(label_clean[i - CONS_WINDOW:i]))
        if ALPHA * window_q < THETA_MIN:
            conservation_fired = True
            break

    return {
        "adv_rate":           adv_rate,
        "seed":               seed,
        "final_acc":          final_acc,
        "conservation_fired": conservation_fired,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    gae_ver = gae.__version__

    print(f"EXP-S2-REPRO-A4-SIGMA running ... (GAE {gae_ver})")
    print(f"  Tensor: {N_CATS}x{N_ACTS}x{N_FACTORS}  "
          f"sigma_perturb={SIGMA_PERTURB}  N_seeds={N_SEEDS}  N_dec={N_DECISIONS}")
    print()

    arm_results: dict[str, list[dict]] = {}
    for adv_rate, arm_name in [(0.0, "arm_0"), (0.10, "arm_1"), (0.20, "arm_2")]:
        print(f"  {arm_name} (adv={adv_rate:.0%}) ...", flush=True)
        arm_results[arm_name] = [run_one(adv_rate, s) for s in SEEDS_20]

    def arm_acc(name: str) -> float:
        return float(np.mean([r["final_acc"] for r in arm_results[name]]))

    acc_0 = arm_acc("arm_0")
    acc_1 = arm_acc("arm_1")
    acc_2 = arm_acc("arm_2")

    # Sanity 1
    if acc_0 < 0.80:
        print(f"STOP: Arm 0 accuracy={acc_0:.1%} < 80% — geometry wrong.")
        sys.exit(1)

    degradation_10 = (acc_0 - acc_1) * 100.0
    degradation_20 = (acc_0 - acc_2) * 100.0

    # Sanity 2
    if degradation_20 < degradation_10:
        print(f"STOP: Arm 2 degradation ({degradation_20:.3f}pp) < "
              f"Arm 1 ({degradation_10:.3f}pp) — perturbation not applied correctly.")
        sys.exit(1)

    # Sanity 3
    if degradation_20 > 1.0:
        print(f"  WARNING: Arm 2 degradation={degradation_20:.3f}pp > 1.0pp (unexpected).")

    # 95% CI on per-seed degradation at 20%
    diffs_20 = [
        (arm_results["arm_0"][i]["final_acc"] - arm_results["arm_2"][i]["final_acc"]) * 100.0
        for i in range(N_SEEDS)
    ]
    ci_lo, ci_hi = scipy_stats.t.interval(
        0.95,
        df=N_SEEDS - 1,
        loc=float(np.mean(diffs_20)),
        scale=float(scipy_stats.sem(diffs_20)),
    )
    ci_lo, ci_hi = float(ci_lo), float(ci_hi)

    gate_pass = ci_hi <= 0.20

    # Save
    out = {
        "experiment":              "EXP-S2-REPRO-A4-SIGMA",
        "attack_vector":           "sigma_perturbation",
        "sigma_perturb":           SIGMA_PERTURB,
        "gae_version":             gae_ver,
        "tensor_shape":            "6x4x6",
        "n_seeds":                 N_SEEDS,
        "arm_0_accuracy":          round(acc_0 * 100, 3),
        "arm_1_accuracy":          round(acc_1 * 100, 3),
        "arm_2_accuracy":          round(acc_2 * 100, 3),
        "degradation_10pct":       round(degradation_10, 4),
        "degradation_20pct":       round(degradation_20, 4),
        "ci_95_degradation_20pct": [round(ci_lo, 4), round(ci_hi, 4)],
        "ci_upper_20pct":          round(ci_hi, 4),
        "gate_pass":               gate_pass,
        "vs_label_poisoning": {
            "label_poisoning_ci_upper":    4.88,
            "sigma_perturbation_ci_upper": round(ci_hi, 4),
        },
    }

    results_path = (REPO_ROOT / "experiments" / "exp_s2_repro_a4"
                    / "results" / "results_sigma.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"  Saved: {results_path}")
    print()
    print(f"EXP-S2-REPRO-A4-SIGMA (sigma-perturbation, N=20, GAE {gae_ver}):")
    print(f"  Attack: eps~N(0,{SIGMA_PERTURB}) added to factor vectors. Labels CORRECT.")
    print(f"  Arm 0 (clean):   accuracy={acc_0*100:.1f}%")
    print(f"  Arm 1 (10% adv): accuracy={acc_1*100:.1f}%, degradation={degradation_10:.3f}pp")
    print(f"  Arm 2 (20% adv): accuracy={acc_2*100:.1f}%, degradation={degradation_20:.3f}pp")
    print(f"  CI upper (20%): {ci_hi:.3f}pp [gate: <=0.20pp] -> {'PASS' if gate_pass else 'FAIL'}")
    print(f"  vs label poisoning CI upper: 4.88pp")
    print("Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
