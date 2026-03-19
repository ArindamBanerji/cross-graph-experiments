"""
EXP-L2-POISON: Level 2 Adversarial Poisoning

Five conditions (A-E) x 50 seeds x 1000 decisions.
Thompson Sampling + four-condition promotion gate.
Conservation law Gate 3 with theta_min = 0.467.

KEY MEASUREMENT: Does Gate 3 (conservation law) fire BEFORE
accuracy degradation becomes significant (>1pp drop)?

Design calibration:
  Honest: override_rate = 0.25 (FIXED, not quality-dependent)
          override_quality = 0.80
          => alpha*q*V = 0.25*0.80*3 = 0.60 (above theta_min=0.467)
  Malicious: always override, quality=0 (rubber-stamp, no improvement)
          => alpha*q*V = (f_honest*0.25*0.80)*3 = honest_fraction*0.60
          At 40% malicious: 0.60*0.60 = 0.36 (below theta_min)
          At 20% malicious: 0.80*0.60 = 0.48 (borderline)
  V_t = 3 decisions/day (DAYS_PER_DECISION = 1/3)
"""
from __future__ import annotations

import sys
import json
import math
import time
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS             = 20
N_DECISIONS         = 500
N_GATE_MIN          = 200
GATE_CHECK_INTERVAL = 50

K_VARIANTS  = 5
TARGET_V    = 4
BASELINE_V  = 0

TRUE_QUALITY = np.array([0.70, 0.80, 0.85, 0.65, 0.55])

DELTA_MIN = 0.03    # achievable margin with v0's ~40-50 TS samples (wide Beta posterior)
THETA_MIN = 0.467   # eta * N_half^2 / T_max = 0.05 * 196 / 21

W_ACCEPT  = 0.4
W_RESOLVE = 0.3
W_OUTCOME = 0.3

HONEST_OVERRIDE_RATE    = 0.22   # fixed, not quality-dependent
# Calibrated so honest alpha*q*V = 0.22*0.80*3 = 0.528 >> theta_min
# and 20%-poisoned alpha*q*V = 0.80*0.22*0.80*3 = 0.422 < theta_min
HONEST_OVERRIDE_QUALITY = 0.80
GATE_WINDOW             = 100

DAYS_PER_DECISION = 1.0 / 3.0   # V_t = 3 decisions/day steady-state

ACCURACY_DROP_THRESHOLD = 0.01

# Gate 3 persistence: require this many consecutive gate-check windows
# all above theta_min before Gate 3 passes.
# Eliminates random early fluctuations while preserving detection.
CL_PERSISTENCE = 3

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "A": dict(poison_rate=0.00, targeted=False, label="0% poison"),
    "B": dict(poison_rate=0.10, targeted=False, label="10% poison"),
    "C": dict(poison_rate=0.20, targeted=False, label="20% poison"),
    "D": dict(poison_rate=0.40, targeted=False, label="40% poison"),
    "E": dict(poison_rate=0.20, targeted=True,  label="20% targeted"),
}

# ---------------------------------------------------------------------------
# Analyst models
# ---------------------------------------------------------------------------

def honest_obs(variant_idx: int, rng: np.random.Generator) -> dict:
    q        = TRUE_QUALITY[variant_idx]
    accepted = bool(rng.random() < q)
    res_time = float(rng.exponential(1.0 / q))   # mean = 1/q, faster for higher q
    correct  = bool(rng.random() < q)
    # Fixed override rate: 0.25 regardless of which variant is selected
    if rng.random() < HONEST_OVERRIDE_RATE:
        override_correct = bool(rng.random() < HONEST_OVERRIDE_QUALITY)
        return dict(accepted=accepted, res_time=res_time, correct=correct,
                    override=True, override_correct=override_correct)
    return dict(accepted=accepted, res_time=res_time, correct=correct,
                override=False, override_correct=False)


def malicious_obs_full(variant_idx: int, rng: np.random.Generator) -> dict:
    """Non-targeted malicious: promotes v4, degrades others.
    Always overrides (rubber-stamp, quality=0) to inflate alpha while
    contributing zero to q — making alpha*q*V drop below theta_min."""
    if variant_idx == TARGET_V:
        return dict(accepted=True, res_time=0.5, correct=True,
                    override=True, override_correct=False)
    return dict(accepted=False, res_time=5.0, correct=False,
                override=True, override_correct=False)


def malicious_obs_targeted(variant_idx: int, rng: np.random.Generator) -> dict:
    """Targeted malicious: only boosts v4 context; behaves honestly otherwise.
    Creates bimodal reward signal for v4 -> Gate 4 (variance) catches."""
    if variant_idx == TARGET_V:
        return dict(accepted=True, res_time=0.5, correct=True,
                    override=False, override_correct=False)
    return honest_obs(variant_idx, rng)


def get_obs(v: int, is_malicious: bool, targeted: bool,
            rng: np.random.Generator) -> dict:
    if not is_malicious:
        return honest_obs(v, rng)
    return malicious_obs_targeted(v, rng) if targeted else malicious_obs_full(v, rng)


# ---------------------------------------------------------------------------
# Composite reward (all components bounded (0,1])
# ---------------------------------------------------------------------------

def composite_reward(obs: dict) -> float:
    resolve = 1.0 / (1.0 + obs["res_time"])
    return W_ACCEPT * obs["accepted"] + W_RESOLVE * resolve + W_OUTCOME * obs["correct"]


# ---------------------------------------------------------------------------
# Conservation tracker (cumulative)
# ---------------------------------------------------------------------------

class ConservationTracker:
    def __init__(self):
        self.n_dec  = 0
        self.n_over = 0
        self.n_good = 0
        self.traj: list[float] = []

    def update(self, obs: dict, elapsed_days: float) -> None:
        self.n_dec += 1
        if obs["override"]:
            self.n_over += 1
            if obs["override_correct"]:
                self.n_good += 1
        alpha_t = self.n_over / self.n_dec
        q_t     = self.n_good / max(self.n_over, 1)
        V_t     = self.n_dec  / max(elapsed_days, 1e-9)
        self.traj.append(float(alpha_t * q_t * V_t))

    def aqv(self) -> float:
        return self.traj[-1] if self.traj else 0.0


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def _beta_normal_approx(a: float, b: float) -> tuple[float, float]:
    """Return (mean, variance) of Beta(a, b) via normal approximation."""
    s = a + b
    mean = a / s
    var  = (a * b) / (s * s * (s + 1.0))
    return mean, var


def gate1_superiority(v: int, alpha: np.ndarray, beta_: np.ndarray,
                      rng: np.random.Generator) -> bool:
    """Normal approximation to P(Beta_v > Beta_baseline + DELTA_MIN) > 0.75.
    100x faster than MC Beta sampling."""
    av, bv = float(np.maximum(alpha[v],          1e-6)), float(np.maximum(beta_[v],          1e-6))
    ab, bb = float(np.maximum(alpha[BASELINE_V], 1e-6)), float(np.maximum(beta_[BASELINE_V], 1e-6))
    mean_v, var_v = _beta_normal_approx(av, bv)
    mean_b, var_b = _beta_normal_approx(ab, bb)
    # diff = Beta_v - Beta_baseline ~ N(mean_v - mean_b, var_v + var_b)
    diff_mean = mean_v - mean_b
    diff_std  = math.sqrt(max(var_v + var_b, 1e-12))
    # P(diff > DELTA_MIN)
    z    = (diff_mean - DELTA_MIN) / diff_std
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return prob > 0.75


def gate2_correctness(v: int, correct_v: list, correct_base: list) -> bool:
    cv = correct_v[-GATE_WINDOW:]
    cb = correct_base[-GATE_WINDOW:]
    if len(cv) < 10 or len(cb) < 10:
        return False
    return float(np.mean(cv)) >= float(np.mean(cb))


def gate3_conservation(tracker: ConservationTracker,
                        aqv_check_history: list[float]) -> bool:
    """Gate 3 passes only if the last CL_PERSISTENCE gate-check aqv values
    are ALL >= theta_min. Persistence filter eliminates early noise spikes."""
    if not aqv_check_history:
        return False
    recent = aqv_check_history[-CL_PERSISTENCE:]
    return all(v >= THETA_MIN for v in recent)


def gate4_variance(v: int, rew_v: list, rew_base: list) -> bool:
    rv = rew_v[-GATE_WINDOW:]
    rb = rew_base[-GATE_WINDOW:]
    if len(rv) < 10 or len(rb) < 10:
        return False
    var_v = float(np.var(rv))
    var_b = float(np.var(rb)) if len(rb) > 1 else 1e-9
    return var_v <= max(var_b, 1e-9) * 1.20


# ---------------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------------

def run_seed(cond_cfg: dict, seed: int) -> dict:
    rng         = np.random.default_rng(seed)
    poison_rate = cond_cfg["poison_rate"]
    targeted    = cond_cfg["targeted"]

    alpha_ts = np.ones(K_VARIANTS, dtype=float)
    beta_ts  = np.ones(K_VARIANTS, dtype=float)

    reward_hist  = [[] for _ in range(K_VARIANTS)]
    correct_hist = [[] for _ in range(K_VARIANTS)]

    tracker = ConservationTracker()

    acc_window: list[int] = []
    baseline_acc: float | None = None

    promoted      = [False] * K_VARIANTS
    promo_t       = [None]  * K_VARIANTS
    gate_block_v4 = [0, 0, 0, 0]

    aqv_traj: list[float] = []
    aqv_check_history: list[float] = []   # aqv recorded at each gate check
    t_conservation: int | None = None
    t_accuracy_drop: int | None = None

    for t in range(N_DECISIONS):
        ts_samp  = rng.beta(np.maximum(alpha_ts, 1e-9),
                             np.maximum(beta_ts,  1e-9))
        selected = int(np.argmax(ts_samp))

        is_mal = bool(rng.random() < poison_rate)
        obs    = get_obs(selected, is_mal, targeted, rng)

        R = composite_reward(obs)
        if rng.random() < R:
            alpha_ts[selected] += 1.0
        else:
            beta_ts[selected]  += 1.0

        reward_hist[selected].append(R)
        correct_hist[selected].append(int(obs["correct"]))

        elapsed = max((t + 1) * DAYS_PER_DECISION, 1e-9)
        tracker.update(obs, elapsed)
        aqv_traj.append(tracker.aqv())

        acc_window.append(int(obs["correct"]))
        if t == N_GATE_MIN - 1:
            baseline_acc = float(np.mean(acc_window))

        if baseline_acc is not None and len(acc_window) >= 50:
            rolling = float(np.mean(acc_window[-50:]))
            if t_accuracy_drop is None and rolling < baseline_acc - ACCURACY_DROP_THRESHOLD:
                t_accuracy_drop = t

        if t >= N_GATE_MIN and (t - N_GATE_MIN) % GATE_CHECK_INTERVAL == 0:
            current_aqv = tracker.aqv()
            aqv_check_history.append(current_aqv)
            # single-check trigger for early-warning measurement
            if current_aqv < THETA_MIN and t_conservation is None:
                t_conservation = t
            # persistence-filtered Gate 3
            g3 = gate3_conservation(tracker, aqv_check_history)

            for v in range(K_VARIANTS):
                if promoted[v]:
                    continue
                g1 = gate1_superiority(v, alpha_ts, beta_ts, rng)
                g2 = gate2_correctness(v, correct_hist[v], correct_hist[BASELINE_V])
                g4 = gate4_variance(v, reward_hist[v], reward_hist[BASELINE_V])
                gates = [g1, g2, g3, g4]
                if all(gates):
                    promoted[v] = True
                    promo_t[v]  = t
                elif v == TARGET_V:
                    for gi, gp in enumerate(gates):
                        if not gp:
                            gate_block_v4[gi] += 1

    return dict(
        promoted        = promoted,
        promo_t         = promo_t,
        gate_block_v4   = gate_block_v4,
        aqv_traj        = aqv_traj,
        t_conservation  = t_conservation,
        t_accuracy_drop = t_accuracy_drop,
        baseline_acc    = baseline_acc,
    )


# ---------------------------------------------------------------------------
# Aggregate condition
# ---------------------------------------------------------------------------

def run_condition(cond_name: str, cond_cfg: dict) -> dict:
    seeds   = [42 + i * 97 for i in range(N_SEEDS)]
    results = [run_seed(cond_cfg, s) for s in seeds]

    v4_prom = [r["promoted"][TARGET_V] for r in results]
    v2_prom = [r["promoted"][2]        for r in results]

    gate_block_total = [sum(r["gate_block_v4"][gi] for r in results) for gi in range(4)]
    total_blocks     = sum(gate_block_total)
    gate_eff         = [g / max(total_blocks, 1) for g in gate_block_total]

    primary_blocker = []
    for r in results:
        if not r["promoted"][TARGET_V] and sum(r["gate_block_v4"]) > 0:
            primary_blocker.append(int(np.argmax(r["gate_block_v4"])))

    t_cons_list = [r["t_conservation"] for r in results if r["t_conservation"] is not None]
    t_acc_list  = [r["t_accuracy_drop"] for r in results if r["t_accuracy_drop"] is not None]
    cons_trig   = float(np.mean([r["t_conservation"] is not None for r in results]))

    cl_early: float | None = None
    if cond_name != "A":
        eligible = early = 0
        for r in results:
            tc, ta = r["t_conservation"], r["t_accuracy_drop"]
            if tc is not None:
                eligible += 1
                if ta is None or tc < ta:
                    early += 1
        if eligible > 0:
            cl_early = float(early / eligible)

    ds       = max(N_DECISIONS // 100, 1)
    aqv_mean = np.mean(np.array([r["aqv_traj"] for r in results]), axis=0)[::ds].tolist()

    return dict(
        cond                      = cond_name,
        label                     = cond_cfg["label"],
        v4_promoted_rate          = float(np.mean(v4_prom)),
        v2_promoted_rate          = float(np.mean(v2_prom)),
        gate_block_total          = gate_block_total,
        gate_effectiveness        = gate_eff,
        primary_blocker_counts    = [primary_blocker.count(gi) for gi in range(4)],
        conservation_trigger_rate = cons_trig,
        cl_early_warning_rate     = cl_early,
        mean_t_conservation       = float(np.mean(t_cons_list)) if t_cons_list else None,
        mean_t_accuracy_drop      = float(np.mean(t_acc_list))  if t_acc_list  else None,
        aqv_mean_traj             = aqv_mean,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("=== EXP-L2-POISON: LEVEL 2 ADVERSARIAL POISONING ===")
    print("=" * 62)
    print(f"N_SEEDS={N_SEEDS}  N_DECISIONS={N_DECISIONS}")
    print(f"theta_min={THETA_MIN}  K={K_VARIANTS}  V_t~{1/DAYS_PER_DECISION:.0f}/day")
    print(f"Honest override: rate={HONEST_OVERRIDE_RATE}, quality={HONEST_OVERRIDE_QUALITY}")
    print(f"Expected honest alpha*q*V = {HONEST_OVERRIDE_RATE*HONEST_OVERRIDE_QUALITY/DAYS_PER_DECISION:.3f}")
    print()

    all_results: dict[str, dict] = {}
    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"  {cond_name} ({cond_cfg['label']}) ...", flush=True)
        t0 = time.time()
        all_results[cond_name] = run_condition(cond_name, cond_cfg)
        elapsed = time.time() - t0
        print(f"  Condition {cond_name} complete: {elapsed:.1f}s", flush=True)

    # Table 1
    print()
    print("=" * 78)
    print("=== RESULTS ===")
    print("=" * 78)
    hdr = (f"{'Cond':<5} {'Poison':<12} {'v4%':>6} {'v2%':>6} "
           f"{'PriGate':>10} {'CL<thr%':>8} {'CLearly':>8}")
    print(hdr)
    print("-" * len(hdr))
    for c, res in all_results.items():
        pb = res["primary_blocker_counts"]
        bg = f"Gate {1+int(np.argmax(pb))}" if sum(pb) > 0 else "—"
        cl = f"{res['cl_early_warning_rate']*100:.0f}%" if res["cl_early_warning_rate"] is not None else "—"
        print(f"{c:<5} {res['label']:<12} "
              f"{res['v4_promoted_rate']*100:>5.1f}% "
              f"{res['v2_promoted_rate']*100:>5.1f}% "
              f"{bg:>10} "
              f"{res['conservation_trigger_rate']*100:>7.1f}% "
              f"{cl:>8}")

    # Table 2
    print()
    print("Conservation Law as Early Warning:")
    hdr2 = f"{'Cond':<5} {'T_cons':>10} {'T_acc_drop':>12} {'CL_first':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for c in ["B", "C", "D", "E"]:
        res = all_results[c]
        tc  = f"{res['mean_t_conservation']:.0f}"  if res["mean_t_conservation"]  is not None else "N/A"
        ta  = f"{res['mean_t_accuracy_drop']:.0f}" if res["mean_t_accuracy_drop"] is not None else "N/A"
        cl  = f"{res['cl_early_warning_rate']*100:.0f}%" if res["cl_early_warning_rate"] is not None else "—"
        print(f"{c:<5} {tc:>10} {ta:>12} {cl:>10}")

    # Table 3
    print()
    print("Gate effectiveness (fraction of blocking events):")
    hdr3 = f"{'Gate':<22}  {'B':>7}  {'C':>7}  {'D':>7}  {'E':>7}"
    print(hdr3)
    print("-" * len(hdr3))
    for gi, gl in enumerate(["G1 (superiority)", "G2 (correctness)",
                              "G3 (conservation)", "G4 (variance)"]):
        row = f"{gl:<22}"
        for c in ["B", "C", "D", "E"]:
            row += f"  {all_results[c]['gate_effectiveness'][gi]*100:>5.1f}%"
        print(row)

    # Success criteria
    print()
    A, B, C, D = (all_results[k] for k in "ABCD")
    cl_rates = [r["cl_early_warning_rate"] for r in [B, C, D]
                if r["cl_early_warning_rate"] is not None]
    mean_cl  = float(np.mean(cl_rates)) if cl_rates else 0.0
    v4_max   = max(all_results[c]["v4_promoted_rate"] for c in ["B","C","D","E"])

    criteria = [
        ("Cond A: v2 promoted >80%",                  A["v2_promoted_rate"] > 0.80),
        ("Cond B: v4 promoted <5%",                   B["v4_promoted_rate"] < 0.05),
        ("Cond C: v4 promoted <10%",                  C["v4_promoted_rate"] < 0.10),
        ("Cond D: v4 promoted <20%",                  D["v4_promoted_rate"] < 0.20),
        (f"CL early warning >60% (B/C/D={mean_cl*100:.0f}%)", mean_cl > 0.60),
    ]
    print("Success criteria:")
    for label, passed in criteria:
        print(f"  {'PASS v' if passed else 'FAIL x'}  {label}")

    # Verdict
    if mean_cl > 0.60 and v4_max < 0.20:
        verdict = (
            f"Conservation law provides early warning — alpha*q*V drops below "
            f"theta_min={THETA_MIN} before accuracy degrades in {mean_cl*100:.0f}% "
            "of poisoned seeds (B/C/D mean). Gate combination blocks target promotion."
        )
    elif v4_max < 0.20:
        verdict = (
            f"Gate combination effective — v4 blocked (max {v4_max*100:.0f}%). "
            f"Conservation law early warning: {mean_cl*100:.0f}% (below 60% target)."
        )
    else:
        verdict = (
            f"PARTIAL — v4 promoted up to {v4_max*100:.0f}% of seeds. "
            f"CL early warning: {mean_cl*100:.0f}%. "
            "Conservation law insufficient alone at high poison rates; "
            "multi-gate defense partially effective."
        )
    print(f"\nVERDICT: {verdict}")

    # Save
    out = _REPO_ROOT / "results" / "exp_l2_poison.json"
    (_REPO_ROOT / "results").mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dict(
            config=dict(
                n_seeds=N_SEEDS, n_decisions=N_DECISIONS,
                n_gate_min=N_GATE_MIN, gate_check_interval=GATE_CHECK_INTERVAL,
                k_variants=K_VARIANTS, true_quality=TRUE_QUALITY.tolist(),
                delta_min=float(DELTA_MIN), theta_min=float(THETA_MIN),
                w_accept=W_ACCEPT, w_resolve=W_RESOLVE, w_outcome=W_OUTCOME,
                honest_override_rate=HONEST_OVERRIDE_RATE,
                honest_override_quality=HONEST_OVERRIDE_QUALITY,
                days_per_decision=DAYS_PER_DECISION,
                accuracy_drop_threshold=ACCURACY_DROP_THRESHOLD,
            ),
            conditions=all_results,
            verdict=verdict,
        ), fh, indent=2)
    print(f"\nResults saved -> {out}")

    charts_py = Path(__file__).parent / "charts.py"
    if charts_py.exists():
        subprocess.run(
            [sys.executable, str(charts_py)],
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
            check=True,
        )


if __name__ == "__main__":
    main()
