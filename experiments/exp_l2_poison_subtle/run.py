"""
P4-F: Subtle Poisoning — Condition F (40pp acceptance inflation) and
Condition G (10pp acceptance inflation).

Confirms whether the full four-condition gate (same as EXP-L2-POISON)
catches a subtle attacker who only inflates the ACCEPTANCE signal for v4
while behaving genuinely on all other metrics.

Key design difference from EXP-L2-POISON:
  CRUDE attacker: sets accepted=True, res_time=0.5, correct=True for v4
                  AND suppresses all other variants (no acceptance, high res_time)
                  AND rubber-stamps overrides (inflates alpha, kills q -> Gate 3)
  SUBTLE attacker: inflates ONLY the acceptance probability for v4
                   All other metrics (res_time, correct) are genuine
                   Other variants are fully unaffected
                   NO overrides (avoids conservation law detection)

Variants (same as EXP-L2-POISON):
  v0 quality = 0.70  (baseline)
  v1 quality = 0.80
  v2 quality = 0.85  (best)
  v3 quality = 0.65
  v4 quality = 0.55  (attacker target)

Conditions:
  A: 0% malicious (control)
  F: 20% malicious, v4 accept inflated 0.55 -> 0.95 (+40pp)
  G: 20% malicious, v4 accept inflated 0.55 -> 0.65 (+10pp)

All gate constants match EXP-L2-POISON exactly.
"""
from __future__ import annotations

import sys
import json
import math
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Constants — identical to EXP-L2-POISON
# ---------------------------------------------------------------------------
N_SEEDS             = 30
N_DECISIONS         = 1000
N_GATE_MIN          = 200
GATE_CHECK_INTERVAL = 50

K_VARIANTS  = 5
TARGET_V    = 4
BASELINE_V  = 0

TRUE_QUALITY = np.array([0.70, 0.80, 0.85, 0.65, 0.55])

DELTA_MIN = 0.03
THETA_MIN = 0.467

W_ACCEPT  = 0.4
W_RESOLVE = 0.3
W_OUTCOME = 0.3

HONEST_OVERRIDE_RATE    = 0.22
HONEST_OVERRIDE_QUALITY = 0.80
GATE_WINDOW             = 100
DAYS_PER_DECISION       = 1.0 / 3.0
ACCURACY_DROP_THRESHOLD = 0.01
CL_PERSISTENCE          = 3

RESULTS_FILE = _REPO_ROOT / "results" / "exp_l2_poison_subtle.json"

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "A": dict(poison_frac=0.00, v4_accept_inflated=None,  label="0% (control)"),
    "F": dict(poison_frac=0.20, v4_accept_inflated=0.95,  label="subtle 40pp"),
    "G": dict(poison_frac=0.20, v4_accept_inflated=0.65,  label="subtle 10pp"),
}

# P4-POWER predictions for comparison (from exp_l2_power results)
# At N~100 per variant (N_DECISIONS=1000, ~200/K):
#   30pp inflation -> ~73% detection  (F most similar: 40pp acceptance)
#   10pp inflation -> ~13% detection  (G: 10pp acceptance)
# But subtle attack inflates only acceptance (weight 0.4), 20% malicious
# => effective composite reward delta much smaller than raw acceptance inflation
POWER_PREDICTION = {"A": 0.0, "F": "~73%* (crude 30pp ref)", "G": "~13%* (crude 10pp ref)"}

# ---------------------------------------------------------------------------
# Analyst models
# ---------------------------------------------------------------------------

def honest_obs(variant_idx: int, rng: np.random.Generator) -> dict:
    q        = float(TRUE_QUALITY[variant_idx])
    accepted = bool(rng.random() < q)
    res_time = float(rng.exponential(1.0 / q))
    correct  = bool(rng.random() < q)
    if rng.random() < HONEST_OVERRIDE_RATE:
        override_correct = bool(rng.random() < HONEST_OVERRIDE_QUALITY)
        return dict(accepted=accepted, res_time=res_time, correct=correct,
                    override=True, override_correct=override_correct)
    return dict(accepted=accepted, res_time=res_time, correct=correct,
                override=False, override_correct=False)


def subtle_obs(variant_idx: int, v4_accept_inflated: float,
               rng: np.random.Generator) -> dict:
    """Subtle attacker: inflates ONLY acceptance for v4. Everything else genuine.
    No overrides — attacker stays below conservation radar.
    Other variants: completely honest.
    """
    q        = float(TRUE_QUALITY[variant_idx])
    res_time = float(rng.exponential(1.0 / q))   # genuine for all
    correct  = bool(rng.random() < q)             # genuine for all

    if variant_idx == TARGET_V:
        accepted = bool(rng.random() < v4_accept_inflated)
    else:
        accepted = bool(rng.random() < q)         # genuine for non-targets

    # No override — subtle attacker avoids conservation law detection
    return dict(accepted=accepted, res_time=res_time, correct=correct,
                override=False, override_correct=False)


def get_obs(v: int, is_malicious: bool, v4_accept: float | None,
            rng: np.random.Generator) -> dict:
    if not is_malicious or v4_accept is None:
        return honest_obs(v, rng)
    return subtle_obs(v, v4_accept, rng)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

def composite_reward(obs: dict) -> float:
    resolve = 1.0 / (1.0 + obs["res_time"])
    return W_ACCEPT * obs["accepted"] + W_RESOLVE * resolve + W_OUTCOME * obs["correct"]


# ---------------------------------------------------------------------------
# Conservation tracker
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
# Gates (normal approximation — NO scipy)
# ---------------------------------------------------------------------------

def _beta_normal_approx(a: float, b: float) -> tuple[float, float]:
    s    = a + b
    mean = a / s
    var  = (a * b) / (s * s * (s + 1.0))
    return mean, var


def gate1_superiority(v: int, alpha: np.ndarray, beta_: np.ndarray) -> bool:
    """P(Beta_v > Beta_baseline + DELTA_MIN) > 0.75 via normal approximation."""
    av = float(max(alpha[v],          1e-6));  bv = float(max(beta_[v],          1e-6))
    ab = float(max(alpha[BASELINE_V], 1e-6));  bb = float(max(beta_[BASELINE_V], 1e-6))
    mean_v, var_v = _beta_normal_approx(av, bv)
    mean_b, var_b = _beta_normal_approx(ab, bb)
    diff_mean = mean_v - mean_b
    diff_std  = math.sqrt(max(var_v + var_b, 1e-12))
    z    = (diff_mean - DELTA_MIN) / diff_std
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return prob > 0.75


def gate2_correctness(v: int, correct_v: list, correct_base: list) -> bool:
    cv = correct_v[-GATE_WINDOW:];   cb = correct_base[-GATE_WINDOW:]
    if len(cv) < 10 or len(cb) < 10:
        return False
    return float(np.mean(cv)) >= float(np.mean(cb))


def gate3_conservation(aqv_history: list[float]) -> bool:
    """All last CL_PERSISTENCE aqv values must be >= THETA_MIN."""
    if not aqv_history:
        return False
    return all(v >= THETA_MIN for v in aqv_history[-CL_PERSISTENCE:])


def gate4_variance(v: int, rew_v: list, rew_base: list) -> bool:
    rv = rew_v[-GATE_WINDOW:];   rb = rew_base[-GATE_WINDOW:]
    if len(rv) < 10 or len(rb) < 10:
        return False
    var_v = float(np.var(rv))
    var_b = float(np.var(rb)) if len(rb) > 1 else 1e-9
    return var_v <= max(var_b, 1e-9) * 1.20


# ---------------------------------------------------------------------------
# Expected reward analysis (deterministic, for interpretation)
# ---------------------------------------------------------------------------

def expected_composite_reward(q: float, accept_override: float | None = None) -> float:
    """Expected composite reward given true quality q.
    Optionally override acceptance probability (for subtle attacker model).
    """
    accept = accept_override if accept_override is not None else q
    resolve_mean = 1.0 / (1.0 + 1.0 / q)   # E[1/(1+Exp(1/q))] ≈ 1/(1+E[Exp(1/q)]) = 1/(1+1/q)
    return W_ACCEPT * accept + W_RESOLVE * resolve_mean + W_OUTCOME * q


# ---------------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------------

def run_seed(cond_cfg: dict, seed: int) -> dict:
    rng          = np.random.default_rng(seed)
    poison_frac  = cond_cfg["poison_frac"]
    v4_accept    = cond_cfg["v4_accept_inflated"]

    alpha_ts = np.ones(K_VARIANTS, dtype=float)
    beta_ts  = np.ones(K_VARIANTS, dtype=float)

    reward_hist  = [[] for _ in range(K_VARIANTS)]
    correct_hist = [[] for _ in range(K_VARIANTS)]

    tracker = ConservationTracker()

    acc_window:   list[int]   = []
    baseline_acc: float | None = None

    promoted      = [False] * K_VARIANTS
    promo_t       = [None]  * K_VARIANTS
    gate_block_v4 = [0, 0, 0, 0]

    aqv_traj:         list[float] = []
    aqv_check_history: list[float] = []
    t_conservation:  int | None   = None
    t_accuracy_drop: int | None   = None

    for t in range(N_DECISIONS):
        ts_samp  = rng.beta(np.maximum(alpha_ts, 1e-9),
                            np.maximum(beta_ts,  1e-9))
        selected = int(np.argmax(ts_samp))

        is_mal = bool(rng.random() < poison_frac)
        obs    = get_obs(selected, is_mal, v4_accept, rng)

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
            if current_aqv < THETA_MIN and t_conservation is None:
                t_conservation = t

            g3 = gate3_conservation(aqv_check_history)

            for v in range(K_VARIANTS):
                if promoted[v]:
                    continue
                g1 = gate1_superiority(v, alpha_ts, beta_ts)
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

    cons_trig = float(np.mean([r["t_conservation"] is not None for r in results]))

    t_cons_list = [r["t_conservation"] for r in results if r["t_conservation"] is not None]
    t_acc_list  = [r["t_accuracy_drop"] for r in results if r["t_accuracy_drop"] is not None]

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

    ds = max(N_DECISIONS // 100, 1)
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
# Expected reward pre-analysis
# ---------------------------------------------------------------------------

def print_expected_rewards(cond_name: str, cond_cfg: dict) -> None:
    frac = cond_cfg["poison_frac"]
    v4a  = cond_cfg["v4_accept_inflated"]

    print(f"  Expected composite rewards for condition {cond_name} ({cond_cfg['label']}):")
    for vi in range(K_VARIANTS):
        q = float(TRUE_QUALITY[vi])
        # Honest expected reward
        er_honest = expected_composite_reward(q)
        if vi == TARGET_V and v4a is not None:
            er_mal = expected_composite_reward(q, accept_override=v4a)
            er_mix = (1.0 - frac) * er_honest + frac * er_mal
            print(f"    v{vi} (q={q:.2f}): honest={er_honest:.3f}  "
                  f"malicious={er_mal:.3f}  mixed={er_mix:.3f}")
        else:
            print(f"    v{vi} (q={q:.2f}): {er_honest:.3f} (unaffected)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("=== P4-F: SUBTLE POISONING TEST ===")
    print("=" * 62)
    print(f"N_SEEDS={N_SEEDS}  N_DECISIONS={N_DECISIONS}")
    print(f"theta_min={THETA_MIN}  DELTA_MIN={DELTA_MIN}")
    print(f"Honest override: rate={HONEST_OVERRIDE_RATE}, quality={HONEST_OVERRIDE_QUALITY}")
    print()

    # Pre-analysis: show expected rewards so attacker viability is clear
    print("── Expected reward analysis (before simulation) ──")
    for cn, cc in CONDITIONS.items():
        print_expected_rewards(cn, cc)
    print()
    # Compare v4 to v0 baseline
    er_v0 = expected_composite_reward(0.70)
    for cn, cc in CONDITIONS.items():
        frac = cc["poison_frac"]; v4a = cc["v4_accept_inflated"]
        if v4a is not None:
            er_v4_mix = ((1-frac)*expected_composite_reward(0.55)
                         + frac*expected_composite_reward(0.55, accept_override=v4a))
            gap = er_v0 - er_v4_mix
            print(f"  [{cn}] v0={er_v0:.3f}  v4_mixed={er_v4_mix:.3f}  "
                  f"gap(v0-v4)={gap:+.3f}  "
                  f"=> v4 {'CANNOT' if gap > 0 else 'CAN'} appear superior to v0 via Gate 1")
    print()

    # Conservation: expected aqv with subtle (no-override) attacker
    frac_mal = 0.20
    honest_alpha_q = HONEST_OVERRIDE_RATE * HONEST_OVERRIDE_QUALITY
    mixed_aqv_per_dec = (1 - frac_mal) * honest_alpha_q  # malicious contributes 0 overrides
    aqv_steady = mixed_aqv_per_dec / DAYS_PER_DECISION
    print(f"  Conservation (F/G): honest_alpha_q={honest_alpha_q:.3f}  "
          f"mixed_per_dec={(1-frac_mal)*honest_alpha_q:.3f}  "
          f"V_t={1/DAYS_PER_DECISION:.0f}/day  "
          f"expected_aqv={aqv_steady:.3f}  "
          f"theta_min={THETA_MIN}  "
          f"=> Gate 3 {'TRIGGERS' if aqv_steady < THETA_MIN else 'holds'}")
    print()

    # Run conditions
    all_results: dict[str, dict] = {}
    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"  Running {cond_name} ({cond_cfg['label']}) ...", flush=True)
        t0 = time.time()
        all_results[cond_name] = run_condition(cond_name, cond_cfg)
        print(f"  Condition {cond_name} complete: {time.time()-t0:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print()
    print("=" * 82)
    print("=== RESULTS ===")
    print("=" * 82)
    print(f"{'Cond':<5} {'Inflation':<12} {'v4%':>7} {'v2%':>7} "
          f"{'Primary gate':>13} {'CL trig%':>9} {'CL early':>9}")
    print("-" * 82)
    for c, res in all_results.items():
        pb  = res["primary_blocker_counts"]
        bg  = f"Gate {1+int(np.argmax(pb))}" if sum(pb) > 0 else "—"
        cl  = f"{res['conservation_trigger_rate']*100:.0f}%"
        cle = (f"{res['cl_early_warning_rate']*100:.0f}%"
               if res["cl_early_warning_rate"] is not None else "—")
        print(f"{c:<5} {res['label']:<12} "
              f"{res['v4_promoted_rate']*100:>6.1f}% "
              f"{res['v2_promoted_rate']*100:>6.1f}% "
              f"{bg:>13} {cl:>9} {cle:>9}")

    # -----------------------------------------------------------------------
    # Gate breakdown
    # -----------------------------------------------------------------------
    print()
    print("Gate effectiveness (fraction of v4 blocking events):")
    hdr = f"  {'Gate':<26}" + "".join(f"  {c:>8}" for c in CONDITIONS)
    print(hdr)
    print("  " + "-" * (26 + 11 * len(CONDITIONS)))
    gate_names = ["G1 (superiority)", "G2 (correctness)",
                  "G3 (conservation)", "G4 (variance)"]
    for gi, gn in enumerate(gate_names):
        row = f"  {gn:<26}"
        for c, res in all_results.items():
            row += f"  {res['gate_effectiveness'][gi]*100:>7.1f}%"
        print(row)

    # -----------------------------------------------------------------------
    # Comparison with power analysis predictions
    # -----------------------------------------------------------------------
    print()
    print("Comparison to P4-POWER predictions:")
    print(f"  {'Cond':<5} {'Inflation':<15} {'Predicted*':>12} {'Actual':>8} {'Match?':>8}")
    print("  " + "-" * 52)
    power_refs = {
        "A": (0.00, None),
        "F": (0.30, 0.73),  # 30pp total quality inflation → 73% at N~100
        "G": (0.10, 0.13),  # 10pp total quality inflation → 13% at N~100
    }
    for c, res in all_results.items():
        infl_ref, pred = power_refs[c]
        actual = res["v4_promoted_rate"]
        if pred is None:
            print(f"  {c:<5} {'0pp (clean)':<15} {'—':>12} {actual*100:>7.1f}% {'N/A':>8}")
        else:
            match = "YES" if abs(actual - pred) < 0.15 else "NO"
            note  = "(accept-only)" if c in ("F","G") else ""
            print(f"  {c:<5} {f'accept +{int(infl_ref*100)}pp':<15} "
                  f"{pred*100:>11.0f}% {actual*100:>7.1f}% {match:>8}")
    print("  * P4-POWER: total quality inflation at N~100 per variant. NOT comparable")
    print("    directly: subtle attacker inflates only acceptance (weight=0.4), 20% malicious.")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    v4_f = all_results["F"]["v4_promoted_rate"]
    v4_g = all_results["G"]["v4_promoted_rate"]
    cl_f = all_results["F"]["conservation_trigger_rate"]

    print()
    print("VERDICT:")
    if v4_f < 0.05 and v4_g < 0.05:
        print("  FULL GATE blocks subtle poisoning in both F (40pp) and G (10pp).")
        print("  Protection mechanism: Gate 1 (v4 mixed reward still < v0 baseline).")
        if cl_f > 0.30:
            print(f"  Gate 3 also contributes: CL triggers in {cl_f*100:.0f}% of F seeds")
            print("  (subtle attacker's absence from overrides reduces alpha*q*V below theta_min).")
        print("  Power analysis mismatch EXPECTED: crude attacker SUPPRESSES other variants,")
        print("  making v4 appear relatively superior. Subtle attacker cannot achieve this.")
        print("  => Subtle poisoning confirmed UNDETECTABLE by Gate 1 ALONE (v4 never appears")
        print("     superior), and UNNECESSARY to detect — it cannot cause promotion.")
    elif v4_f > 0.05:
        print(f"  WARNING: Cond F v4 promoted {v4_f*100:.1f}% — gate not fully protective.")
        print("  Gates 2-4 provide additional defense beyond Gate 1.")
    if v4_g < 0.05:
        print(f"  [G] 10pp subtle poisoning: 0% promotion — consistent with power analysis")
        print("      (undetectable AND harmless: cannot overcome quality gap).")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    result = dict(
        config = dict(
            n_seeds=N_SEEDS, n_decisions=N_DECISIONS, theta_min=THETA_MIN,
            delta_min=DELTA_MIN, honest_override_rate=HONEST_OVERRIDE_RATE,
            honest_override_quality=HONEST_OVERRIDE_QUALITY,
            days_per_decision=DAYS_PER_DECISION, cl_persistence=CL_PERSISTENCE,
        ),
        conditions = all_results,
        expected_aqv_f_g = aqv_steady,
        power_analysis_refs = {k: {"infl_pp": int(v[0]*100), "pred": v[1]}
                               for k, v in power_refs.items()},
    )
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"Results saved -> {RESULTS_FILE}")

    # Charts
    from experiments.exp_l2_poison_subtle.charts import chart1_comparison
    chart1_comparison(result)


if __name__ == "__main__":
    main()
