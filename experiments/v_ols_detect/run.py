"""
V-OLS-DETECT -- Validate CUSUM on Override Lift Score detects gradual degradation
before 5pp centroid damage in two conditions.

  Condition A (Adversarial):  q 0.85->0.40, alpha=0.80, N=500 decisions
  Condition B (Complacency):  q 0.85->0.55, alpha=0.25, N=300 decisions

Cold start (mu0=0.5+jitter): sanity check initial OLS > 1.0.
OLS = P(correct | analyst override) / P(correct | AI accepted)

T_damage: plateau-snapshot approach -- rolling acc drops DAMAGE_PP below
the snapshot taken at PLATEAU_FRAC*N_DEC. Avoids spurious fires from
learning-phase oscillations.

Gate: OLS lead time p90 >= 50 AND miss rate <= 10% for BOTH conditions.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gae.profile_scorer import ProfileScorer
from gae.calibration import CalibrationProfile

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SEEDS      = 30
THETA_MIN    = 0.467
TAU          = 0.1
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
ALPHA_QUAL   = 0.80      # conservation quality multiplier

# Condition A
Q_START_A = 0.85
Q_END_A   = 0.40
N_DEC_A   = 500
ALPHA_A   = 0.80         # override rate

# Condition B
Q_START_B = 0.85
Q_END_B   = 0.55
N_DEC_B   = 300
ALPHA_B   = 0.25         # override rate (complacency)

# OLS Monitor
OLS_H       = 5.0        # CUSUM threshold (tuned for this OLS scale)
OLS_LAM     = 0.1
OLS_K       = 0.10
OLS_WIN     = 30
OLS_MINC    = 5
OLS_BASE_N  = 50

# T_damage: plateau-snapshot
# Record rolling acc at t = PLATEAU_FRAC * N_DEC (after learning, before peak corruption)
# T_damage fires when rolling drops DAMAGE_PP below that snapshot
PLATEAU_FRAC = 0.40
ROLL_WIN     = 50
DAMAGE_PP    = 5.0

CONS_WINDOW  = 50

SEEDS_30 = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
            7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
            17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624]

# ---------------------------------------------------------------------------
# A1 x B1 SOC healthcare geometry
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
ACT_IDX   = {a: i for i, a in enumerate(ACTIONS)}

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


def build_mu_star():
    mu = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    for (cat, act), vec in _MU_STAR_RAW.items():
        mu[CAT_IDX[cat], ACT_IDX[act], :] = vec
    return mu


MU_STAR     = build_mu_star()


def build_gt_dist():
    gt = np.ones((N_CATS, N_ACTS)) * 0.1
    for c in range(N_CATS):
        norms = np.linalg.norm(MU_STAR[c], axis=-1)
        gt[c, int(np.argmax(norms))] = 0.70
    gt /= gt.sum(axis=1, keepdims=True)
    return gt


GT_DIST     = build_gt_dist()
CAT_WEIGHTS = np.ones(N_CATS) / N_CATS


# ---------------------------------------------------------------------------
# OLS Monitor -- EWMA + one-sided downward CUSUM
# ---------------------------------------------------------------------------
class OLSMonitor:
    """
    Tracks Override Lift Score = P(correct|override) / P(correct|accepted).
    Fires when EWMA-smoothed OLS drops h cumulative units below its baseline.
    """

    def __init__(self, h=5.0, lam=0.1, k_offset=0.10,
                 window=30, min_count=5, baseline_n=50):
        self.h = h
        self.lam = lam
        self.k_offset = k_offset
        self.window = window
        self.min_count = min_count
        self.baseline_n = baseline_n

        self._ov_buf  = deque(maxlen=window)   # override outcome history
        self._ac_buf  = deque(maxlen=window)   # accepted outcome history
        self._ols_seq = []                     # all valid OLS values
        self._ewma    = None
        self._s       = 0.0
        self._baseline = None
        self.fired    = False
        self.fire_t   = None
        self._t       = 0

    def record(self, is_override: bool, was_correct: bool):
        if is_override:
            self._ov_buf.append(1 if was_correct else 0)
        else:
            self._ac_buf.append(1 if was_correct else 0)

        ols = None
        if (len(self._ov_buf) >= self.min_count and
                len(self._ac_buf) >= self.min_count):
            p_ov = float(np.mean(list(self._ov_buf)))
            p_ac = float(np.mean(list(self._ac_buf)))
            ols  = p_ov / max(p_ac, 0.01)

        if ols is not None:
            if self._ewma is None:
                self._ewma = ols
            else:
                self._ewma = self.lam * ols + (1 - self.lam) * self._ewma

            self._ols_seq.append(ols)

            if self._baseline is None and len(self._ols_seq) >= self.baseline_n:
                self._baseline = float(np.mean(self._ols_seq[:self.baseline_n]))
                self._s = 0.0

            if self._baseline is not None:
                self._s = max(0.0, self._s +
                              (self._baseline - self._ewma - self.k_offset))
                if not self.fired and self._s >= self.h:
                    self.fired  = True
                    self.fire_t = self._t

        self._t += 1

    def initial_ols(self, n: int = 50) -> float:
        if not self._ols_seq:
            return float("nan")
        return float(np.mean(self._ols_seq[:n]))


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------
def run_seed(condition: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    if condition == "A":
        q_start, q_end = Q_START_A, Q_END_A
        n_dec           = N_DEC_A
        alpha           = ALPHA_A
    else:
        q_start, q_end = Q_START_B, Q_END_B
        n_dec           = N_DEC_B
        alpha           = ALPHA_B

    # Cold start
    mu0 = np.full((N_CATS, N_ACTS, N_FACTORS), 0.5, dtype=float)
    mu0 += rng.uniform(-0.005, 0.005, mu0.shape)
    np.clip(mu0, 0, 1, out=mu0)

    profile = CalibrationProfile(learning_rate=ETA_CONFIRM, penalty_ratio=1.0,
                                 temperature=TAU)
    scorer  = ProfileScorer(mu=mu0, actions=ACTIONS, profile=profile,
                            eta_override=ETA_OVERRIDE)

    monitor = OLSMonitor(h=OLS_H, lam=OLS_LAM, k_offset=OLS_K,
                         window=OLS_WIN, min_count=OLS_MINC,
                         baseline_n=OLS_BASE_N)

    correct_flags = []
    # quality_flags: accepted=1; override=Bernoulli(q_eff)
    # rolling mean = (1-alpha)*1 + alpha*q_eff per window
    quality_flags = []

    t_ols       = None
    t_damage    = None
    t_conserv   = None
    cons_fire   = False

    # Plateau-snapshot for T_damage
    plateau_t   = int(n_dec * PLATEAU_FRAC)   # decision where we snapshot acc
    plateau_acc = None                         # set once when t == plateau_t

    for t in range(n_dec):
        q_eff = q_start - (q_start - q_end) * (t / n_dec)

        cat_idx = int(rng.choice(N_CATS, p=CAT_WEIGHTS))
        gt_act  = int(rng.choice(N_ACTS, p=GT_DIST[cat_idx]))
        f = np.clip(MU_STAR[cat_idx, gt_act] + rng.normal(0, SIGMA), 0.0, 1.0)

        result  = scorer.score(f, cat_idx)
        ai_act  = result.action_index
        ai_corr = (ai_act == gt_act)
        correct_flags.append(int(ai_corr))

        is_override = (rng.random() < alpha)

        if is_override:
            is_quality  = (rng.random() < q_eff)
            if is_quality:
                analyst_act  = gt_act
                analyst_corr = True
                scorer.update(f=f, category_index=cat_idx,
                              action_index=gt_act, correct=True,
                              gt_action_index=gt_act)
            else:
                wrong        = [a for a in range(N_ACTS) if a != gt_act]
                analyst_act  = int(rng.choice(wrong))
                analyst_corr = False
                scorer.update(f=f, category_index=cat_idx,
                              action_index=analyst_act, correct=True,
                              gt_action_index=None)
            monitor.record(is_override=True, was_correct=analyst_corr)
            quality_flags.append(1 if analyst_corr else 0)
        else:
            # Accepted: no scorer update; quality = 1 (no corruption risk)
            monitor.record(is_override=False, was_correct=ai_corr)
            quality_flags.append(1)

        # OLS fire time
        if t_ols is None and monitor.fired:
            t_ols = t

        # Plateau snapshot at plateau_t
        if t == plateau_t and plateau_acc is None:
            lo = max(0, t - ROLL_WIN + 1)
            w  = correct_flags[lo: t + 1]
            if len(w) >= ROLL_WIN:
                plateau_acc = float(np.mean(w)) * 100.0

        # T_damage: first rolling drop DAMAGE_PP below plateau snapshot
        # Only active after plateau_t + ROLL_WIN (full window after plateau)
        if (plateau_acc is not None and
                t > plateau_t + ROLL_WIN and
                t_damage is None):
            lo  = max(0, t - ROLL_WIN + 1)
            w   = correct_flags[lo: t + 1]
            if len(w) >= ROLL_WIN:
                rolling_acc = float(np.mean(w)) * 100.0
                if plateau_acc - rolling_acc > DAMAGE_PP:
                    t_damage = t

        # Conservation check: fires when alpha_qual * mean_quality < theta_min
        if not cons_fire and len(quality_flags) >= CONS_WINDOW:
            q_bar    = float(np.mean(quality_flags[-CONS_WINDOW:]))
            cons_val = ALPHA_QUAL * q_bar
            if cons_val < THETA_MIN:
                cons_fire = True
                t_conserv = t

    init_ols = monitor.initial_ols(n=50)

    def lead(t_fire):
        if t_fire is not None and t_damage is not None:
            return t_damage - t_fire
        elif t_fire is not None and t_damage is None:
            return n_dec - t_fire   # fired, no damage -> large positive
        return None   # never fired

    return {
        "condition":         condition,
        "seed":              seed,
        "t_ols":             t_ols,
        "t_damage":          t_damage,
        "t_conservation":    t_conserv,
        "conservation_fire": cons_fire,
        "lead_ols":          lead(t_ols),
        "initial_ols":       round(init_ols, 4) if not np.isnan(init_ols) else None,
        "plateau_acc":       round(plateau_acc, 2) if plateau_acc is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    print(f"V-OLS-DETECT running ... (GAE {gae.__version__})")
    print(f"  N={N_SEEDS} seeds x (A={N_DEC_A}dec + B={N_DEC_B}dec)", flush=True)
    print()

    results_a = [run_seed("A", s) for s in SEEDS_30]
    results_b = [run_seed("B", s) for s in SEEDS_30]

    # Sanity: initial OLS > 1.0 (cold start validates analyst superiority at start)
    ols_a_init = [r["initial_ols"] for r in results_a if r["initial_ols"] is not None]
    ols_b_init = [r["initial_ols"] for r in results_b if r["initial_ols"] is not None]
    mean_init_a = float(np.mean(ols_a_init)) if ols_a_init else float("nan")
    mean_init_b = float(np.mean(ols_b_init)) if ols_b_init else float("nan")

    if mean_init_a <= 1.0 or mean_init_b <= 1.0:
        print(f"SANITY FAIL: Initial OLS not >1.0 (A={mean_init_a:.3f}, B={mean_init_b:.3f}). STOP.")
        sys.exit(1)

    def gate_stats(results):
        leads = [r["lead_ols"] for r in results if r["lead_ols"] is not None]
        miss_count = sum(
            1 for r in results
            if r["t_damage"] is not None and (
                r["t_ols"] is None or r["t_ols"] > r["t_damage"]
            )
        )
        miss_rate  = miss_count / N_SEEDS
        p10 = float(np.percentile(leads, 10)) if leads else float("nan")
        p50 = float(np.percentile(leads, 50)) if leads else float("nan")
        p90 = float(np.percentile(leads, 90)) if leads else float("nan")
        n_ols  = sum(1 for r in results if r["t_ols"]     is not None)
        n_dam  = sum(1 for r in results if r["t_damage"]  is not None)
        n_cons = sum(1 for r in results if r["conservation_fire"])
        return p10, p50, p90, miss_rate, n_ols, n_dam, n_cons

    a_p10, a_p50, a_p90, a_miss, a_ols, a_dam, a_cons = gate_stats(results_a)
    b_p10, b_p50, b_p90, b_miss, b_ols, b_dam, b_cons = gate_stats(results_b)

    a_gate_lead = (not np.isnan(a_p90)) and a_p90 >= 50.0
    a_gate_miss = a_miss <= 0.10
    b_gate_lead = (not np.isnan(b_p90)) and b_p90 >= 50.0
    b_gate_miss = b_miss <= 0.10
    gate_pass   = a_gate_lead and a_gate_miss and b_gate_lead and b_gate_miss

    out = {
        "experiment":         "V-OLS-DETECT",
        "gae_version":        gae.__version__,
        "n_seeds":            N_SEEDS,
        "initial_ols_mean_A": round(mean_init_a, 4),
        "initial_ols_mean_B": round(mean_init_b, 4),
        "condition_A": {
            "q_range":              f"{Q_START_A}->{Q_END_A}",
            "alpha_override":       ALPHA_A,
            "n_decisions":          N_DEC_A,
            "plateau_frac":         PLATEAU_FRAC,
            "lead_p10":             round(a_p10, 1) if not np.isnan(a_p10) else None,
            "lead_p50":             round(a_p50, 1) if not np.isnan(a_p50) else None,
            "lead_p90":             round(a_p90, 1) if not np.isnan(a_p90) else None,
            "miss_rate":            round(a_miss, 4),
            "n_ols_fires":          a_ols,
            "n_damage_fires":       a_dam,
            "n_conservation_fires": a_cons,
            "gate_lead_pass":       a_gate_lead,
            "gate_miss_pass":       a_gate_miss,
        },
        "condition_B": {
            "q_range":              f"{Q_START_B}->{Q_END_B}",
            "alpha_override":       ALPHA_B,
            "n_decisions":          N_DEC_B,
            "plateau_frac":         PLATEAU_FRAC,
            "lead_p10":             round(b_p10, 1) if not np.isnan(b_p10) else None,
            "lead_p50":             round(b_p50, 1) if not np.isnan(b_p50) else None,
            "lead_p90":             round(b_p90, 1) if not np.isnan(b_p90) else None,
            "miss_rate":            round(b_miss, 4),
            "n_ols_fires":          b_ols,
            "n_damage_fires":       b_dam,
            "n_conservation_fires": b_cons,
            "gate_lead_pass":       b_gate_lead,
            "gate_miss_pass":       b_gate_miss,
        },
        "gate_pass": gate_pass,
    }

    out_path = REPO_ROOT / "experiments" / "v_ols_detect" / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved: {out_path}")

    print()
    print(f"V-OLS-DETECT (N={N_SEEDS}, GAE {gae.__version__}):")
    print(f"  Initial OLS: A={mean_init_a:.3f}, B={mean_init_b:.3f} [sanity: >1.0 -> PASS]")
    print()
    print(f"  Condition A (adversarial, q 0.85->0.40, alpha={ALPHA_A}):")
    print(f"    Lead time p10={a_p10:.0f}d p50={a_p50:.0f}d p90={a_p90:.0f}d [gate: >=50]"
          f" -> {'PASS' if a_gate_lead else 'FAIL'}")
    print(f"    Miss rate: {a_miss:.1%} [gate: <=10%] -> {'PASS' if a_gate_miss else 'FAIL'}")
    print(f"    OLS fires: {a_ols}/30, Damage fires: {a_dam}/30, Conservation fires: {a_cons}/30")
    print()
    print(f"  Condition B (complacency, q 0.85->0.55, alpha={ALPHA_B}):")
    print(f"    Lead time p10={b_p10:.0f}d p50={b_p50:.0f}d p90={b_p90:.0f}d [gate: >=50]"
          f" -> {'PASS' if b_gate_lead else 'FAIL'}")
    print(f"    Miss rate: {b_miss:.1%} [gate: <=10%] -> {'PASS' if b_gate_miss else 'FAIL'}")
    print(f"    OLS fires: {b_ols}/30, Damage fires: {b_dam}/30, Conservation fires: {b_cons}/30")
    print()
    print(f"Overall gate: {'PASS' if gate_pass else 'FAIL'}")
    print("Raw numbers for roadmap session review.")


if __name__ == "__main__":
    main()
