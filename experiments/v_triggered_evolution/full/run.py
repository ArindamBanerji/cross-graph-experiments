"""
V-TRIGGERED-EVOLUTION FULL -- N=30 powered experiment.

Validates the W2 compounding flywheel using the real PatternHistoryFactorComputer
from the SOC backend with a MockNeo4j supplying TRIGGERED_EVOLUTION edges.

Hypothesis:
  T1 (200 W2 edges for similar categories) achieves higher Loop 1 accuracy
  on similar organizational contexts than C0 (no W2 edges, fallback=0.40).
  Dissimilar categories (no W2 edges in either condition) should show zero delta
  — specificity control.

Design:
  N=30 seeds. Per seed:
    - Frozen centroids mu0 = MU_STAR + N(0, 0.15).
    - 25 similar test alerts: lateral_movement, credential_access
      (W2 edges exist for these categories in T1's graph).
    - 25 dissimilar test alerts: data_exfiltration, cloud_infrastructure
      (no W2 edges in either condition -> PhFC returns 0.40 for both -> delta=0).
    - C0: PatternHistoryFactorComputer with EMPTY MockNeo4j -> f[4] = 0.40.
    - T1: PatternHistoryFactorComputer with populated MockNeo4j -> recency-weighted mean.
    - Both score with the SAME frozen mu0 scorer (no update() calls).

W2 edge generation (per seed, same for all T1 test alerts in that seed):
  100 edges for lateral_movement, 100 for credential_access.
  For each edge: gt_act ~ GT_DIST[cat], decision_num = i+1 (1..100),
  pattern_value = clip(MU_STAR[cat, gt_act, 4] + N(0, PH_W2_SIGMA), 0, 1).
  Decision_num range [1..100] -> max recency gap=99, recency range [2^(-99/30)..1.0].

Gate:
  CI_lower > 1.0pp  AND  p-value < 0.05  AND  mean_delta_dissimilar < 0.5pp
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parents[3]
SOC_BACKEND = REPO_ROOT.parent / "gen-ai-roi-demo-v4-v50" / "backend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOC_BACKEND))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile
from app.domains.soc.factors import PatternHistoryFactorComputer

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS          = 30
N_W2_PER_CAT     = 100   # W2 edges per similar category (100 lat_mv + 100 cred_acc = 200 total)
PH_W2_SIGMA      = 0.12  # noise on W2 pattern_value around MU_STAR[cat, gt_act, 4]
N_SIMILAR        = 25    # similar test alerts per seed
N_DISSIMILAR     = 25    # dissimilar test alerts per seed (specificity control)
MU0_SIGMA        = 0.15  # frozen centroid bootstrap perturbation

TAU              = 0.1
ETA_CONFIRM      = 0.05
ETA_OVERRIDE     = 0.01

PH_FACTOR_IDX    = 4    # pattern_history in factor vector

SEEDS_30 = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
    17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624,
]
assert len(SEEDS_30) == N_SEEDS

# ---------------------------------------------------------------------------
# A1 x B1 SOC healthcare geometry  (identical to pilot)
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

SIMILAR_CATS = [CAT_IDX["lateral_movement"], CAT_IDX["credential_access"]]
DISSIM_CATS  = [CAT_IDX["data_exfiltration"], CAT_IDX["cloud_infrastructure"]]

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

SIGMA = np.array([0.18, 0.06, 0.07, 0.08, 0.095, 0.22])


def build_mu_star() -> np.ndarray:
    mu = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], [a for a, n in enumerate(ACTIONS) if n == act][0], :] = vec
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
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ---------------------------------------------------------------------------
# MockNeo4j
# ---------------------------------------------------------------------------
class MockNeo4j:
    """
    Simulates Neo4j query returning TRIGGERED_EVOLUTION edge rows.
    Filters by category and optionally action_index.
    Returns top-50 by decision_num DESC, format: {pattern_value, decision_num}.
    """

    def __init__(self, edges: List[Dict]) -> None:
        self._edges = edges

    async def run_query(self, query: str, params: Dict) -> List[Dict]:
        category     = params.get("category", "")
        action_index = params.get("action_index", None)

        rows = [e for e in self._edges if e["category"] == category]
        if action_index is not None:
            rows = [e for e in rows if e["action_index"] == action_index]

        rows = sorted(rows, key=lambda r: r["decision_num"], reverse=True)[:50]
        return [{"pattern_value": r["pattern_value"], "decision_num": r["decision_num"]}
                for r in rows]


EMPTY_NEO4J = MockNeo4j([])
PHFC = PatternHistoryFactorComputer()


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------
def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    # Frozen centroids -- identical for C0 and T1
    mu0 = np.clip(MU_STAR + rng.normal(0, MU0_SIGMA, MU_STAR.shape), 0.0, 1.0)
    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                           eta_override=ETA_OVERRIDE)
    # No update() calls -- centroids remain frozen.

    # W2 edges for T1 (similar categories only)
    w2_edges: List[Dict] = []
    for cat_name in ("lateral_movement", "credential_access"):
        c = CAT_IDX[cat_name]
        for i in range(N_W2_PER_CAT):
            gt_act = int(rng.choice(N_ACTS, p=GT_DIST[c]))
            pv = float(np.clip(
                MU_STAR[c, gt_act, PH_FACTOR_IDX] + rng.normal(0, PH_W2_SIGMA),
                0.0, 1.0,
            ))
            w2_edges.append({
                "category":     cat_name,
                "action_index": gt_act,
                "pattern_value": pv,
                "decision_num": i + 1,   # 1..100 per category
            })
    mock_neo4j = MockNeo4j(w2_edges)

    # Generate test alerts
    similar_alerts: List[tuple] = []
    for _ in range(N_SIMILAR):
        c = int(rng.choice(SIMILAR_CATS))
        a = int(rng.choice(N_ACTS, p=GT_DIST[c]))
        f_base = np.clip(MU_STAR[c, a] + rng.normal(0, SIGMA), 0.0, 1.0)
        similar_alerts.append((c, CATEGORIES[c], a, f_base))

    dissim_alerts: List[tuple] = []
    for _ in range(N_DISSIMILAR):
        c = int(rng.choice(DISSIM_CATS))
        a = int(rng.choice(N_ACTS, p=GT_DIST[c]))
        f_base = np.clip(MU_STAR[c, a] + rng.normal(0, SIGMA), 0.0, 1.0)
        dissim_alerts.append((c, CATEGORIES[c], a, f_base))

    # Async scoring loop
    async def _score_all():
        c0_sim, t1_sim, c0_dis, t1_dis = [], [], [], []

        for (cat_idx, cat_name, gt_act, f_base) in similar_alerts:
            alert = {"category": cat_name}

            ph_c0 = await PHFC.compute(alert, EMPTY_NEO4J)   # 0.40 fallback
            f_c0 = f_base.copy()
            f_c0[PH_FACTOR_IDX] = ph_c0
            r_c0 = scorer.score(f_c0, cat_idx)
            c0_sim.append(int(r_c0.action_index == gt_act))

            ph_t1 = await PHFC.compute(alert, mock_neo4j)    # recency-weighted mean
            f_t1 = f_base.copy()
            f_t1[PH_FACTOR_IDX] = ph_t1
            r_t1 = scorer.score(f_t1, cat_idx)
            t1_sim.append(int(r_t1.action_index == gt_act))

        for (cat_idx, cat_name, gt_act, f_base) in dissim_alerts:
            alert = {"category": cat_name}

            ph_c0 = await PHFC.compute(alert, EMPTY_NEO4J)   # 0.40
            f_c0 = f_base.copy()
            f_c0[PH_FACTOR_IDX] = ph_c0
            r_c0 = scorer.score(f_c0, cat_idx)
            c0_dis.append(int(r_c0.action_index == gt_act))

            ph_t1 = await PHFC.compute(alert, mock_neo4j)    # no edges -> 0.40
            f_t1 = f_base.copy()
            f_t1[PH_FACTOR_IDX] = ph_t1
            r_t1 = scorer.score(f_t1, cat_idx)
            t1_dis.append(int(r_t1.action_index == gt_act))

        return c0_sim, t1_sim, c0_dis, t1_dis

    c0_sim, t1_sim, c0_dis, t1_dis = asyncio.run(_score_all())

    c0_acc_sim = float(np.mean(c0_sim)) * 100.0
    t1_acc_sim = float(np.mean(t1_sim)) * 100.0
    c0_acc_dis = float(np.mean(c0_dis)) * 100.0
    t1_acc_dis = float(np.mean(t1_dis)) * 100.0

    return {
        "seed":                   seed,
        "c0_accuracy_similar":    round(c0_acc_sim, 2),
        "t1_accuracy_similar":    round(t1_acc_sim, 2),
        "delta_similar_pp":       round(t1_acc_sim - c0_acc_sim, 2),
        "c0_accuracy_dissimilar": round(c0_acc_dis, 2),
        "t1_accuracy_dissimilar": round(t1_acc_dis, 2),
        "delta_dissimilar_pp":    round(t1_acc_dis - c0_acc_dis, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    assert gae.__version__ == "0.7.18", \
        f"Expected GAE 0.7.18, got {gae.__version__}. Re-run with correct version."
    gae_ver = gae.__version__

    print(f"V-TRIGGERED-EVOLUTION FULL (GAE {gae_ver}, N={N_SEEDS} seeds):")
    print(f"  PatternHistoryFactorComputer: real (SOC backend, HALF_LIFE=30)")
    print(f"  MockNeo4j: 200 W2 edges ({N_W2_PER_CAT} lateral_movement + {N_W2_PER_CAT} credential_access)")
    print(f"  C0: EMPTY MockNeo4j -> PHFC fallback = 0.40")
    print(f"  T1: populated MockNeo4j -> PHFC recency-weighted mean")
    print(f"  Gate: CI_lower > 1.0pp AND p < 0.05 AND mean_delta_dissimilar < 0.5pp")
    print()

    per_seed = []
    for i, seed in enumerate(SEEDS_30, 1):
        r = run_seed(seed)
        per_seed.append(r)
        status = f"+{r['delta_similar_pp']:.1f}pp" if r['delta_similar_pp'] >= 0 else f"{r['delta_similar_pp']:.1f}pp"
        print(f"  [{i:2d}/{N_SEEDS}] seed={seed:6d}  delta_sim={status:>8}  delta_dis={r['delta_dissimilar_pp']:+.1f}pp")

    # ---------------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------------
    deltas_sim = np.array([r["delta_similar_pp"]    for r in per_seed])
    deltas_dis = np.array([r["delta_dissimilar_pp"] for r in per_seed])

    mean_sim = float(np.mean(deltas_sim))
    mean_dis = float(np.mean(deltas_dis))

    t_stat, p_val = scipy_stats.ttest_1samp(deltas_sim, 0.0)
    ci = scipy_stats.t.interval(
        0.95, df=N_SEEDS - 1,
        loc=mean_sim,
        scale=scipy_stats.sem(deltas_sim),
    )
    ci_lower = float(ci[0])
    ci_upper = float(ci[1])

    seeds_positive = int(np.sum(deltas_sim > 0))

    # Gates
    gate_ci    = ci_lower > 1.0
    gate_p     = float(p_val) < 0.05
    gate_spec  = mean_dis < 0.5
    gate_pass  = gate_ci and gate_p and gate_spec

    verdict = "GATE PASS" if gate_pass else "GATE FAIL"

    print()
    print("Results:")
    print(f"  mean delta_similar:    {mean_sim:+.2f}pp")
    print(f"  95% CI:                [{ci_lower:+.2f}pp, {ci_upper:+.2f}pp]")
    print(f"  t-statistic:           {float(t_stat):.3f}")
    print(f"  p-value:               {float(p_val):.4f}")
    print(f"  seeds positive:        {seeds_positive}/{N_SEEDS}")
    print(f"  mean delta_dissimilar: {mean_dis:+.2f}pp  (specificity control)")
    print()
    print("Gate checks:")
    print(f"  CI lower > 1.0pp:      {ci_lower:+.3f}pp  -> {'PASS' if gate_ci else 'FAIL'}")
    print(f"  p-value < 0.05:        {float(p_val):.4f}        -> {'PASS' if gate_p else 'FAIL'}")
    print(f"  delta_dissim < 0.5pp:  {mean_dis:+.3f}pp   -> {'PASS' if gate_spec else 'FAIL'}")
    print()
    print(f"  VERDICT: {verdict}")
    print("Raw numbers for roadmap session review.")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    out = {
        "experiment":          "V-TRIGGERED-EVOLUTION-FULL",
        "gae_version":         gae_ver,
        "n_seeds":             N_SEEDS,
        "n_w2_per_category":   N_W2_PER_CAT,
        "similar_categories":  ["lateral_movement", "credential_access"],
        "dissim_categories":   ["data_exfiltration", "cloud_infrastructure"],
        "n_test_similar":      N_SIMILAR,
        "n_test_dissimilar":   N_DISSIMILAR,
        "ph_w2_sigma":         PH_W2_SIGMA,
        "mu0_perturbation":    MU0_SIGMA,
        "phfc_class":          "PatternHistoryFactorComputer (SOC backend, HALF_LIFE=30)",
        "per_seed":            per_seed,
        "mean_delta_similar_pp":    round(mean_sim, 4),
        "mean_delta_dissimilar_pp": round(mean_dis, 4),
        "ci_95_lower":              round(ci_lower, 4),
        "ci_95_upper":              round(ci_upper, 4),
        "t_statistic":              round(float(t_stat), 4),
        "p_value":                  round(float(p_val), 6),
        "seeds_positive_similar":   seeds_positive,
        "gate_ci_lower_1pp":        gate_ci,
        "gate_p_lt_005":            gate_p,
        "gate_specificity":         gate_spec,
        "gate_pass":                gate_pass,
        "verdict":                  verdict,
    }

    out_path = REPO_ROOT / "experiments" / "v_triggered_evolution" / "full" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
