"""
V-G4-INTERACTION: 8-cell mini-factorial on team quality × size × campaign.
(q̄ ∈ {0.65, 0.85}) × (team ∈ {5, 15}) × (campaign ∈ {no, yes})
20 seeds × 8 cells × 500 decisions.

μ*: A1×B1 SOC healthcare geometry.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS     = 20
N_DECISIONS = 500
WINDOW_LAST = 100

THETA_MIN    = 0.467
TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
ALPHA        = 0.80

SEEDS_20 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]

# ---------------------------------------------------------------------------
# A1×B1 SOC healthcare geometry (canonical)
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

def build_mu_star():
    mu = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu

MU_STAR = build_mu_star()

def build_gt_dist():
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt

GT_DIST = build_gt_dist()
SIGMA   = np.array([0.18, 0.06, 0.07, 0.08, 0.095, 0.22])  # healthcare noise

# Campaign category weights: 70% lateral_movement + credential_access, 30% rest
CAT_WEIGHTS_UNIFORM  = np.ones(N_CATS) / N_CATS
CAT_WEIGHTS_CAMPAIGN = np.array([0.35, 0.05, 0.35, 0.083, 0.083, 0.084])
CAT_WEIGHTS_CAMPAIGN /= CAT_WEIGHTS_CAMPAIGN.sum()


def run_one(q_bar: float, team: int, campaign: bool, seed: int) -> float:
    """Return final accuracy (last WINDOW_LAST decisions)."""
    rng = np.random.default_rng(seed)
    cat_w = CAT_WEIGHTS_CAMPAIGN if campaign else CAT_WEIGHTS_UNIFORM

    mu0 = MU_STAR.copy() + rng.uniform(-0.005, 0.005, MU_STAR.shape)
    np.clip(mu0, 0, 1, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0, temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile, eta_override=ETA_OVERRIDE)

    correct_flags = []
    for _t in range(N_DECISIONS):
        cat_idx = int(rng.choice(N_CATS, p=cat_w))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f       = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)

        result  = scorer.score(f, cat_idx)
        correct_flags.append(int(result.action_index == gt_act))

        # Analyst feedback — team size affects noise variance but not q_bar mean
        # team=5: ±0.05 quality noise, team=15: ±0.02 quality noise
        team_noise = 0.05 if team == 5 else 0.02
        eff_q = float(np.clip(q_bar + rng.normal(0, team_noise), 0.0, 1.0))
        if rng.random() < eff_q:
            label_act = gt_act
        else:
            label_act = int(rng.choice(N_ACTS))

        scorer.update(f=f, category_index=cat_idx, action_index=label_act,
                      correct=(label_act == gt_act), gt_action_index=gt_act)

    return float(np.mean(correct_flags[-WINDOW_LAST:]))


def main():
    CELLS = [
        {"q_bar": q, "team": t, "campaign": c}
        for q in [0.65, 0.85]
        for t in [5, 15]
        for c in [False, True]
    ]

    print("V-G4-INTERACTION (8 cells × 20 seeds) running ...", flush=True)
    cell_results = []
    for cell in CELLS:
        accs = [run_one(cell["q_bar"], cell["team"], cell["campaign"], s) for s in SEEDS_20]
        mean_acc = float(np.mean(accs))
        ci = scipy_stats.t.interval(0.95, df=N_SEEDS-1,
                                    loc=mean_acc, scale=scipy_stats.sem(accs))
        cell_results.append({
            "q_bar":         cell["q_bar"],
            "team":          cell["team"],
            "campaign":      cell["campaign"],
            "accuracy_mean": round(mean_acc * 100, 2),
            "accuracy_ci":   [round(ci[0]*100, 2), round(ci[1]*100, 2)],
            "_accs":         [round(a*100, 2) for a in accs],
        })
        print(f"  q={cell['q_bar']} team={cell['team']:2d} camp={'Y' if cell['campaign'] else 'N'}: "
              f"{mean_acc*100:.1f}%  [{ci[0]*100:.1f},{ci[1]*100:.1f}]", flush=True)

    # -----------------------------------------------------------------------
    # Two-way and three-way interactions (ANOVA-style marginal differences)
    # -----------------------------------------------------------------------
    def cell_acc(q=None, team=None, campaign=None):
        """Mean accuracy of cells matching all specified factors."""
        matching = [r["accuracy_mean"] for r in cell_results
                    if (q is None or r["q_bar"] == q)
                    and (team is None or r["team"] == team)
                    and (campaign is None or r["campaign"] == campaign)]
        return float(np.mean(matching)) if matching else float("nan")

    # q̄ × team: (q=0.85,t=15) - (q=0.65,t=5) main effect of interaction
    # Interaction = (acc[q=0.85,t=15] - acc[q=0.65,t=15]) - (acc[q=0.85,t=5] - acc[q=0.65,t=5])
    # Averaged over campaign levels
    q_by_team = (
        (cell_acc(q=0.85, team=15) - cell_acc(q=0.65, team=15)) -
        (cell_acc(q=0.85, team=5)  - cell_acc(q=0.65, team=5))
    )
    # q̄ × campaign
    q_by_camp = (
        (cell_acc(q=0.85, campaign=True)  - cell_acc(q=0.65, campaign=True)) -
        (cell_acc(q=0.85, campaign=False) - cell_acc(q=0.65, campaign=False))
    )
    # team × campaign
    team_by_camp = (
        (cell_acc(team=15, campaign=True)  - cell_acc(team=5, campaign=True)) -
        (cell_acc(team=15, campaign=False) - cell_acc(team=5, campaign=False))
    )
    # Three-way: how much does team size modulate q×campaign interaction
    three_way = (
        (
            (cell_acc(q=0.85, team=15, campaign=True)  - cell_acc(q=0.65, team=15, campaign=True)) -
            (cell_acc(q=0.85, team=15, campaign=False) - cell_acc(q=0.65, team=15, campaign=False))
        ) - (
            (cell_acc(q=0.85, team=5, campaign=True)  - cell_acc(q=0.65, team=5, campaign=True)) -
            (cell_acc(q=0.85, team=5, campaign=False) - cell_acc(q=0.65, team=5, campaign=False))
        )
    )

    max_interaction = max(abs(q_by_team), abs(q_by_camp), abs(team_by_camp), abs(three_way))
    any_below_75 = any(r["accuracy_mean"] < 75.0 for r in cell_results)
    below_75_cells = [
        f"q={r['q_bar']},team={r['team']},camp={'Y' if r['campaign'] else 'N'}={r['accuracy_mean']}%"
        for r in cell_results if r["accuracy_mean"] < 75.0
    ]
    product_boundaries = below_75_cells if any_below_75 else []

    interactions = {
        "q_by_team":      round(q_by_team, 2),
        "q_by_campaign":  round(q_by_camp, 2),
        "team_by_campaign": round(team_by_camp, 2),
        "three_way":      round(three_way, 2),
    }
    large_interactions = {k: v for k, v in interactions.items() if abs(v) > 5.0}

    # Strip _accs from saved output
    cells_clean = [{k: v for k, v in r.items() if k != "_accs"} for r in cell_results]

    out = {
        "experiment":           "V-G4-INTERACTION",
        "n_cells":              8,
        "n_seeds":              N_SEEDS,
        "n_decisions":          N_DECISIONS,
        "cells":                cells_clean,
        "interactions":         interactions,
        "max_interaction_pp":   round(max_interaction, 2),
        "product_boundaries":   product_boundaries,
        "any_cell_below_75pct": any_below_75,
        "large_interactions_over_5pp": large_interactions,
    }
    out_path = REPO / "experiments" / "v_g4_interaction" / "results" / "results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print()
    print("V-G4-INTERACTION (8 cells × 20 seeds):")
    print()
    print(f"{'q_bar':<5} | {'team':<4} | {'campaign':<8} | {'accuracy':>8} | CI")
    print("-" * 50)
    for r in cell_results:
        print(f"{r['q_bar']:<5} | {r['team']:<4} | "
              f"{'yes' if r['campaign'] else 'no':<8} | "
              f"{r['accuracy_mean']:>7.1f}% | "
              f"[{r['accuracy_ci'][0]:.1f},{r['accuracy_ci'][1]:.1f}]")
    print()
    print("Interactions:")
    print(f"  q_bar × team:     {q_by_team:.1f}pp")
    print(f"  q_bar × campaign: {q_by_camp:.1f}pp")
    print(f"  team × campaign:  {team_by_camp:.1f}pp")
    print(f"  three-way:        {three_way:.1f}pp")
    print(f"  Max interaction: {max_interaction:.1f}pp [boundary threshold: 5pp]")
    print()
    if product_boundaries:
        print(f"Product boundaries (cells < 75%): {product_boundaries}")
    else:
        print("Product boundaries (any cell < 75%): NONE")
    if large_interactions:
        print(f"Any interaction > 5pp: yes → {large_interactions}")
    else:
        print("Any interaction > 5pp: no")


if __name__ == "__main__":
    main()
