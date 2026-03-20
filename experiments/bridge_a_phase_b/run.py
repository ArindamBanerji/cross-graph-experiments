"""
B-A Phase B: Production A/B Harness (P29)
90-day A/B study.
Group A: fixed variant 0 (quality=0.70, no adaptation).
Group B: two-variant Thompson Sampling + four-condition promotion gate.
K=2 variants. N_ANALYSTS=10. N_SEEDS=30.
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Parameters ────────────────────────────────────────────────────────────────
N_ANALYSTS            = 10
DAYS                  = 90
K                     = 2                    # number of variants
DECISIONS_PER_DAY     = 0.85                 # Poisson mean per analyst per day
QUALITIES             = [0.70, 0.80]         # variant 0 baseline, variant 1 improved
THETA_MIN             = 0.434               # conservation law threshold
GATE_INTERVAL         = 50                  # gate check every 50 Group B decisions
GATE_MIN_B_DECISIONS  = 100                 # first gate check at 100 decisions
GATE_MIN_V_EACH       = 20                  # minimum per-variant for gate validity
GATE_MIN_V1_COVERAGE  = 50                  # Gate 4: min variant 1 decisions
REWARD_THRESHOLD      = 0.5                 # Thompson update boundary
POWER_TARGET          = 380                 # required decisions per group
N_SEEDS               = 30
SEEDS                 = list(range(N_SEEDS))

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ── Analyst assignment ────────────────────────────────────────────────────────
def _hash_analyst(analyst_id: str) -> int:
    """Deterministic hash (not affected by PYTHONHASHSEED)."""
    return int(hashlib.md5((analyst_id + "salt_ba_phase_b").encode()).hexdigest(), 16)


def build_group_assignments(n_analysts: int) -> dict:
    """
    Sort analysts by deterministic hash, assign first half to A, second half to B.
    Guarantees exactly n_analysts//2 per group; reproducible across sessions.
    """
    analysts = [f"analyst_{i:02d}" for i in range(n_analysts)]
    ranked   = sorted(analysts, key=_hash_analyst)
    mid      = n_analysts // 2
    return {a: "A" if i < mid else "B" for i, a in enumerate(ranked)}


ANALYSTS = [f"analyst_{i:02d}" for i in range(N_ANALYSTS)]
GROUPS   = build_group_assignments(N_ANALYSTS)
GROUP_A_ANALYSTS = [a for a in ANALYSTS if GROUPS[a] == "A"]
GROUP_B_ANALYSTS = [a for a in ANALYSTS if GROUPS[a] == "B"]


# ── Four-condition promotion gate ────────────────────────────────────────────
def check_four_condition_gate(log_B: list) -> bool:
    """
    All four conditions must hold:
      Gate 1: mean reward v1 > mean reward v0
      Gate 2: one-sided Mann-Whitney p < 0.10 (v1 > v0)
      Gate 3: conservation signal α_v1·accuracy_v1·(n_v1/days_elapsed) > θ_min
      Gate 4: n_v1 ≥ GATE_MIN_V1_COVERAGE
    """
    v0 = [d for d in log_B if d["variant"] == 0]
    v1 = [d for d in log_B if d["variant"] == 1]

    if len(v0) < GATE_MIN_V_EACH or len(v1) < GATE_MIN_V_EACH:
        return False

    r0 = np.array([d["R"] for d in v0])
    r1 = np.array([d["R"] for d in v1])

    # Gate 1
    gate1 = float(r1.mean()) > float(r0.mean())

    # Gate 2
    _, p_gate = scipy_stats.mannwhitneyu(r1, r0, alternative="greater")
    gate2 = p_gate < 0.10

    # Gate 3: conservation signal at check time
    days_elapsed = max(1, log_B[-1]["day"])
    alpha_v1     = float(np.mean([d["accepted"] for d in v1]))
    acc_v1       = float(np.mean([d["correct"]  for d in v1]))
    V_daily_v1   = len(v1) / days_elapsed
    gate3        = alpha_v1 * acc_v1 * V_daily_v1 > THETA_MIN

    # Gate 4: minimum v1 coverage
    gate4 = len(v1) >= GATE_MIN_V1_COVERAGE

    return gate1 and gate2 and gate3 and gate4


# ── Single-seed simulation ────────────────────────────────────────────────────
def simulate_seed(seed: int) -> dict:
    rng = np.random.RandomState(seed)

    # Thompson Sampling priors: Beta(1,1) for each variant
    alpha_ts = np.ones(K, dtype=float)
    beta_ts  = np.ones(K, dtype=float)

    log_A = []
    log_B = []

    group_b_count   = 0
    promoted        = False
    promotion_dec   = None   # group_b_count at first promotion

    # Running sums for efficient conservation signal
    b_accept_sum = 0.0
    b_correct_sum = 0.0
    b_total       = 0

    daily_signal = []

    for day in range(DAYS):
        b_today = 0    # Group B decisions this day

        for analyst in ANALYSTS:
            n_dec = rng.poisson(DECISIONS_PER_DAY)
            for _ in range(n_dec):
                group = GROUPS[analyst]

                if group == "A":
                    variant = 0
                    quality = QUALITIES[0]
                else:
                    if promoted:
                        variant = 1          # always use promoted variant
                    else:
                        ts_draws = np.array([
                            rng.beta(alpha_ts[v], beta_ts[v]) for v in range(K)
                        ])
                        variant = int(np.argmax(ts_draws))
                    quality = QUALITIES[variant]

                # Decision metrics
                accepted = int(rng.random() < quality)
                res_time = float(rng.exponential(1.0 / quality))
                correct  = int(rng.random() < quality)

                rt_score = min(1.0, 1.0 / max(res_time, 0.1))
                R = 0.4 * accepted + 0.3 * rt_score + 0.3 * correct

                rec = {
                    "day":      day + 1,
                    "variant":  variant,
                    "accepted": accepted,
                    "res_time": res_time,
                    "correct":  correct,
                    "R":        float(R),
                }

                if group == "A":
                    log_A.append(rec)
                else:
                    log_B.append(rec)
                    group_b_count += 1
                    b_today       += 1
                    b_accept_sum  += accepted
                    b_correct_sum += correct
                    b_total       += 1

                    # Thompson update (only while not yet promoted)
                    if not promoted:
                        if R > REWARD_THRESHOLD:
                            alpha_ts[variant] += 1.0
                        else:
                            beta_ts[variant]  += 1.0

                        # Gate check trigger
                        if (group_b_count >= GATE_MIN_B_DECISIONS
                                and group_b_count % GATE_INTERVAL == 0):
                            if check_four_condition_gate(log_B):
                                promoted      = True
                                promotion_dec = group_b_count

        # Daily conservation signal: α_running · q_running · V_today
        if b_total > 0 and b_today > 0:
            signal = (b_accept_sum / b_total) * (b_correct_sum / b_total) * b_today
        else:
            signal = 0.0
        daily_signal.append(float(signal))

    return {
        "log_A":           log_A,
        "log_B":           log_B,
        "promoted":        promoted,
        "promotion_dec":   promotion_dec,
        "daily_signal":    daily_signal,
        "ts_alpha_final":  alpha_ts.tolist(),
        "ts_beta_final":   beta_ts.tolist(),
    }


def compute_metrics(log: list) -> dict:
    if not log:
        return {"n": 0, "acceptance_rate": 0.0, "mean_resolution": 0.0,
                "accuracy": 0.0, "mean_reward": 0.0}
    return {
        "n":                len(log),
        "acceptance_rate":  float(np.mean([d["accepted"] for d in log])),
        "mean_resolution":  float(np.mean([d["res_time"]  for d in log])),
        "accuracy":         float(np.mean([d["correct"]   for d in log])),
        "mean_reward":      float(np.mean([d["R"]         for d in log])),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("=== B-A PHASE B: PRODUCTION A/B (K=2, 90 days synthetic) ===")
    print("=" * 62)
    print(f"\nN_ANALYSTS={N_ANALYSTS} ({len(GROUP_A_ANALYSTS)} A / {len(GROUP_B_ANALYSTS)} B) "
          f"| DAYS={DAYS} | N_SEEDS={N_SEEDS}")
    print(f"Group A analysts: {GROUP_A_ANALYSTS}")
    print(f"Group B analysts: {GROUP_B_ANALYSTS}")
    print(f"Qualities: v0={QUALITIES[0]:.2f}, v1={QUALITIES[1]:.2f}  |  θ_min={THETA_MIN}\n")
    print(f"Running {N_SEEDS} seeds...", flush=True)

    # Per-seed accumulators
    seed_acc_A  = np.zeros(N_SEEDS)
    seed_acc_B  = np.zeros(N_SEEDS)
    seed_res_A  = np.zeros(N_SEEDS)
    seed_res_B  = np.zeros(N_SEEDS)
    seed_cor_A  = np.zeros(N_SEEDS)
    seed_cor_B  = np.zeros(N_SEEDS)
    seed_rew_A  = np.zeros(N_SEEDS)
    seed_rew_B  = np.zeros(N_SEEDS)
    seed_n_A    = np.zeros(N_SEEDS, dtype=int)
    seed_n_B    = np.zeros(N_SEEDS, dtype=int)
    seed_promo  = np.zeros(N_SEEDS, dtype=bool)
    seed_pdec   = np.full(N_SEEDS, np.nan)

    seed_daily_sig  = np.zeros((N_SEEDS, DAYS))

    # Daily trajectory arrays (sum over analysts, then normalise)
    daily_acc_A  = np.zeros((N_SEEDS, DAYS))
    daily_n_A    = np.zeros((N_SEEDS, DAYS), dtype=int)
    daily_acc_B  = np.zeros((N_SEEDS, DAYS))
    daily_n_B    = np.zeros((N_SEEDS, DAYS), dtype=int)

    all_R_A = []
    all_R_B = []

    for si, seed in enumerate(SEEDS):
        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1}/{N_SEEDS}] seed={seed}", flush=True)

        res = simulate_seed(seed)
        mA  = compute_metrics(res["log_A"])
        mB  = compute_metrics(res["log_B"])

        seed_acc_A[si] = mA["acceptance_rate"]
        seed_acc_B[si] = mB["acceptance_rate"]
        seed_res_A[si] = mA["mean_resolution"]
        seed_res_B[si] = mB["mean_resolution"]
        seed_cor_A[si] = mA["accuracy"]
        seed_cor_B[si] = mB["accuracy"]
        seed_rew_A[si] = mA["mean_reward"]
        seed_rew_B[si] = mB["mean_reward"]
        seed_n_A[si]   = mA["n"]
        seed_n_B[si]   = mB["n"]
        seed_promo[si] = res["promoted"]
        if res["promotion_dec"] is not None:
            seed_pdec[si] = res["promotion_dec"]

        seed_daily_sig[si] = res["daily_signal"]

        for d in res["log_A"]:
            di = d["day"] - 1
            daily_acc_A[si, di] += d["accepted"]
            daily_n_A[si, di]   += 1
        for d in res["log_B"]:
            di = d["day"] - 1
            daily_acc_B[si, di] += d["accepted"]
            daily_n_B[si, di]   += 1

        all_R_A.extend([d["R"] for d in res["log_A"]])
        all_R_B.extend([d["R"] for d in res["log_B"]])

    # ── Statistical analysis ──────────────────────────────────────────────────
    def mean_ci(arr):
        m   = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        return m, m - 1.96 * sem, m + 1.96 * sem

    acc_A_m, acc_A_lo, acc_A_hi = mean_ci(seed_acc_A)
    acc_B_m, acc_B_lo, acc_B_hi = mean_ci(seed_acc_B)

    u_stat, p_mw = scipy_stats.mannwhitneyu(seed_acc_B, seed_acc_A, alternative="greater")

    delta_acc = acc_B_m - acc_A_m
    pool_std  = float(np.sqrt(
        (seed_acc_A.std(ddof=1)**2 + seed_acc_B.std(ddof=1)**2) / 2
    ))
    cohens_d = delta_acc / pool_std if pool_std > 0 else float("nan")

    promo_rate   = float(seed_promo.mean())
    valid_pdec   = seed_pdec[~np.isnan(seed_pdec)]
    mean_pdec    = float(valid_pdec.mean()) if len(valid_pdec) > 0 else None

    n_A_mean = float(seed_n_A.mean())
    n_B_mean = float(seed_n_B.mean())
    power_ok = (n_A_mean >= POWER_TARGET) and (n_B_mean >= POWER_TARGET)

    # Conservation breach: any seed where a day with >0 decisions had signal < θ_min
    breach_seeds = 0
    for si in range(N_SEEDS):
        sig = seed_daily_sig[si]
        if np.any((sig > 0) & (sig < THETA_MIN)):
            breach_seeds += 1
    breach_rate = breach_seeds / N_SEEDS
    consv_mean  = seed_daily_sig.mean(axis=0)   # (DAYS,) — for chart

    # Daily acceptance trajectories (summed over all seeds)
    sum_n_A = daily_n_A.sum(axis=0)
    sum_n_B = daily_n_B.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        traj_A = np.where(sum_n_A > 0, daily_acc_A.sum(axis=0) / sum_n_A, np.nan)
        traj_B = np.where(sum_n_B > 0, daily_acc_B.sum(axis=0) / sum_n_B, np.nan)

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    hdr = (f"| {'Metric':<16} | {'Group A (fixed)':>15} | "
           f"{'Group B (adaptive)':>18} | {'Δ':>8} | {'p-value':>8} |")
    sep = "|" + "-"*18 + "|" + "-"*17 + "|" + "-"*20 + "|" + "-"*10 + "|" + "-"*10 + "|"
    print(hdr)
    print(sep)
    rows = [
        ("Acceptance rate",
         f"{acc_A_m*100:.1f}%",
         f"{acc_B_m*100:.1f}%",
         f"{delta_acc*100:+.1f}pp",
         f"{p_mw:.4f}"),
        ("Resolution time",
         f"{seed_res_A.mean():.3f} min",
         f"{seed_res_B.mean():.3f} min",
         f"{seed_res_B.mean()-seed_res_A.mean():+.4f}",
         "—"),
        ("Accuracy",
         f"{seed_cor_A.mean()*100:.1f}%",
         f"{seed_cor_B.mean()*100:.1f}%",
         f"{(seed_cor_B.mean()-seed_cor_A.mean())*100:+.1f}pp",
         "—"),
        ("Composite reward",
         f"{seed_rew_A.mean():.4f}",
         f"{seed_rew_B.mean():.4f}",
         f"{seed_rew_B.mean()-seed_rew_A.mean():+.4f}",
         "—"),
    ]
    for label, vA, vB, delta, pv in rows:
        print(f"| {label:<16} | {vA:>15} | {vB:>18} | {delta:>8} | {pv:>8} |")

    print()
    print(f"Mann-Whitney U: {u_stat:.0f}  p={p_mw:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"95% CI (Group A): [{acc_A_lo:.4f}, {acc_A_hi:.4f}]")
    print(f"95% CI (Group B): [{acc_B_lo:.4f}, {acc_B_hi:.4f}]")
    print()
    print(f"Variant 1 promoted: {promo_rate*100:.1f}% of seeds")
    if mean_pdec is not None:
        print(f"Mean decisions to promotion: {mean_pdec:.0f}")
    else:
        print("Mean decisions to promotion: N/A (no promotions observed)")

    if breach_rate == 0:
        print("Conservation law status: never breached")
    else:
        print(f"Conservation law status: breached in {breach_rate*100:.1f}% of seeds")

    print()
    print(f"Power check: {n_A_mean:.0f} decisions (Group A), "
          f"{n_B_mean:.0f} decisions (Group B)  (target ≥{POWER_TARGET})")
    print(f"VERDICT: {'sufficient power' if power_ok else 'underpowered'}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "config": {
            "n_analysts":          N_ANALYSTS,
            "days":                DAYS,
            "k_variants":          K,
            "decisions_per_day":   DECISIONS_PER_DAY,
            "qualities":           QUALITIES,
            "theta_min":           THETA_MIN,
            "gate_interval":       GATE_INTERVAL,
            "gate_min_decisions":  GATE_MIN_B_DECISIONS,
            "gate_min_v1":         GATE_MIN_V1_COVERAGE,
            "n_seeds":             N_SEEDS,
            "power_target":        POWER_TARGET,
        },
        "group_assignments": GROUPS,
        "group_a_analysts":  GROUP_A_ANALYSTS,
        "group_b_analysts":  GROUP_B_ANALYSTS,
        "metrics": {
            "group_a": {
                "acceptance_rate_mean": round(acc_A_m, 4),
                "acceptance_rate_ci":   [round(acc_A_lo, 4), round(acc_A_hi, 4)],
                "resolution_mean":      round(float(seed_res_A.mean()), 4),
                "accuracy_mean":        round(float(seed_cor_A.mean()), 4),
                "reward_mean":          round(float(seed_rew_A.mean()), 4),
                "n_decisions_mean":     round(n_A_mean, 1),
                "seed_values":          seed_acc_A.tolist(),
            },
            "group_b": {
                "acceptance_rate_mean": round(acc_B_m, 4),
                "acceptance_rate_ci":   [round(acc_B_lo, 4), round(acc_B_hi, 4)],
                "resolution_mean":      round(float(seed_res_B.mean()), 4),
                "accuracy_mean":        round(float(seed_cor_B.mean()), 4),
                "reward_mean":          round(float(seed_rew_B.mean()), 4),
                "n_decisions_mean":     round(n_B_mean, 1),
                "seed_values":          seed_acc_B.tolist(),
            },
        },
        "statistical_tests": {
            "mann_whitney_u":    float(u_stat),
            "mann_whitney_p":    round(float(p_mw), 6),
            "cohens_d":          round(cohens_d, 3),
            "delta_acceptance":  round(delta_acc, 4),
        },
        "promotion": {
            "promotion_rate":              round(promo_rate, 3),
            "mean_decisions_to_promotion": round(mean_pdec, 1) if mean_pdec else None,
            "n_seeds_promoted":            int(seed_promo.sum()),
            "seed_promotion_decisions":    [
                int(x) if not np.isnan(x) else None for x in seed_pdec
            ],
        },
        "conservation": {
            "theta_min":    THETA_MIN,
            "breach_rate":  round(breach_rate, 3),
            "breach_seeds": breach_seeds,
            "mean_signal_by_day": consv_mean.tolist(),
        },
        "power": {
            "n_decisions_A": round(n_A_mean, 1),
            "n_decisions_B": round(n_B_mean, 1),
            "target":        POWER_TARGET,
            "sufficient":    bool(power_ok),
        },
        "trajectories": {
            "daily_acceptance_A":       traj_A.tolist(),
            "daily_acceptance_B":       traj_B.tolist(),
            "conservation_signal_mean": consv_mean.tolist(),
        },
        "distributions": {
            "reward_A_sample": [float(r) for r in all_R_A[:8000]],
            "reward_B_sample": [float(r) for r in all_R_B[:8000]],
        },
    }

    out_path = RESULTS_DIR / "bridge_a_phase_b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print("Run charts.py to generate paper figures.")
    return output


if __name__ == "__main__":
    main()
