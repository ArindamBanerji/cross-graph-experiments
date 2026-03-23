"""
Reusable persona harness: TD-034 + PROD-5 + B-A Phase B.

Usage:
    python experiments/persona_sweeps/run_harness.py \
        --personas experiments/persona_sweeps/personas_sweep_1c_quality.json \
        --output   experiments/persona_sweeps/results/sweep_1c_quality/

Asymmetric learning rates (Q5 validated):
    η_confirm  = 0.05  (analyst confirms model prediction)
    η_override = 0.01  (analyst changes recommendation)

Design: lower η_override reduces centroid corruption from low-quality
overrides while preserving the signal from senior analyst corrections.
"""

import sys
import json
import time
import hashlib
import argparse
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config

# ── Harness parameters ────────────────────────────────────────────────────────
ETA_CONFIRM  = 0.05
ETA_OVERRIDE = 0.01
THETA_MIN    = 0.467

# TD-034
TD_TAU_VALUES  = [0.05, 0.08, 0.10, 0.12, 0.15]
TD_N_SEEDS     = 20
TD_N_DECISIONS = 300

# PROD-5
P5_N_SEEDS = 15
P5_DAYS    = 60
P5_E0      = 0.15
P5_EPS     = 0.10
P5_TAU     = 0.10

# B-A Phase B
BA_N_SEEDS            = 15
BA_DAYS               = 90
BA_K                  = 2
BA_QUALITIES          = [0.70, 0.80]
BA_DECISIONS_PER_DAY  = 0.85
BA_GATE_MIN_TOTAL     = 100
BA_GATE_INTERVAL      = 50
BA_GATE_MIN_V_EACH    = 15
BA_GATE_MIN_V1        = 30
POWER_TARGET          = 380


# ── Asymmetric scorer ─────────────────────────────────────────────────────────
class AsymmetricScorer:
    """
    Centroid scorer with dual learning rates:
      - update_confirm:  η = 0.05 (analyst accepts model)
      - update_override: η = 0.01 (analyst changes recommendation)
    No push-away step (avoids compounding override noise).
    """

    def __init__(self, mu: np.ndarray, tau: float = 0.10,
                 eta_confirm: float = 0.05, eta_override: float = 0.01):
        self.mu           = mu.copy().astype(float)
        self.tau          = tau
        self.eta_confirm  = eta_confirm
        self.eta_override = eta_override
        self.counts       = np.zeros(mu.shape[:2], dtype=int)

    def score(self, factors: np.ndarray, c_idx: int):
        dists  = np.sum((self.mu[c_idx] - factors) ** 2, axis=1)
        logits = -dists / self.tau
        logits -= logits.max()
        probs  = np.exp(logits); probs /= probs.sum()
        pred_a = int(np.argmax(probs))
        return pred_a, float(probs[pred_a])

    def _update(self, factors: np.ndarray, c_idx: int, gt_a: int, eta: float):
        cnt     = self.counts[c_idx, gt_a]
        eta_eff = eta / (1 + cnt * 0.001)
        self.mu[c_idx, gt_a] += eta_eff * (factors - self.mu[c_idx, gt_a])
        np.clip(self.mu[c_idx, gt_a], 0, 1, out=self.mu[c_idx, gt_a])
        self.counts[c_idx, gt_a] += 1

    def update_confirm(self, factors: np.ndarray, c_idx: int, gt_a: int):
        self._update(factors, c_idx, gt_a, self.eta_confirm)

    def update_override(self, factors: np.ndarray, c_idx: int, gt_a: int):
        self._update(factors, c_idx, gt_a, self.eta_override)


# ── Shared persona utilities ───────────────────────────────────────────────────
def build_persona_noise(persona: dict, factor_names: list) -> np.ndarray:
    return np.array([persona["factor_noise_profile"][f]["base_noise"]
                     for f in factor_names])


def build_persona_weights(persona: dict, categories: list) -> np.ndarray:
    w = np.array([persona["category_distribution"][c] for c in categories],
                 dtype=float)
    return w / w.sum()


def precompute_day_weights(base_w: np.ndarray, categories: list,
                           cat_to_idx: dict, shifts: list,
                           n_days: int) -> np.ndarray:
    dw = np.zeros((n_days, len(categories)))
    for day in range(n_days):
        cd = day + 1
        w  = base_w.copy()
        for s in shifts:
            if s["day"] <= cd < s["day"] + s["duration_days"]:
                for cat, mult in s["category_impact"].items():
                    if cat in cat_to_idx:
                        w[cat_to_idx[cat]] *= mult
        dw[day] = w / w.sum()
    return dw


def precompute_analyst_params(team: list) -> list:
    return [(min(1.0, a["override_rate"] * (1 + a["fatigue_factor"] * 0.3)),
             max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2)))
            for a in team]


def persona_q_bar(team: list) -> float:
    qs = [max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2))
          for a in team]
    return float(np.mean(qs))


# ── Harness 1: TD-034 ─────────────────────────────────────────────────────────
def run_td034(persona, mu_true, categories, gt_dists_arr, factor_names):
    C, A, d  = mu_true.shape
    noise    = build_persona_noise(persona, factor_names)
    weights  = build_persona_weights(persona, categories)
    cat_idx  = np.arange(C)

    def compute_ece(confs, corrects, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        ece, n = 0.0, len(confs)
        for i in range(n_bins):
            m = (confs >= edges[i]) & (confs < edges[i + 1])
            if m.sum():
                ece += m.sum() / n * abs(corrects[m].mean() - confs[m].mean())
        return float(ece)

    tau_results = {}
    for tau in TD_TAU_VALUES:
        eces, accs = [], []
        for seed in range(TD_N_SEEDS):
            rng    = np.random.RandomState(seed)
            scorer = AsymmetricScorer(mu_true, tau=tau,
                                      eta_confirm=ETA_CONFIRM,
                                      eta_override=ETA_OVERRIDE)
            confs    = np.empty(TD_N_DECISIONS)
            corrects = np.empty(TD_N_DECISIONS)
            for j in range(TD_N_DECISIONS):
                c  = int(rng.choice(cat_idx, p=weights))
                gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f  = np.clip(mu_true[c, gt] + rng.randn(d) * noise, 0, 1)
                pa, conf = scorer.score(f, c)
                confs[j] = conf; corrects[j] = float(pa == gt)
            eces.append(compute_ece(confs, corrects))
            accs.append(float(corrects.mean()))
        tau_results[tau] = {"ece_mean": float(np.mean(eces)),
                            "ece_std":  float(np.std(eces)),
                            "acc_mean": float(np.mean(accs))}

    opt_tau    = min(tau_results, key=lambda t: tau_results[t]["ece_mean"])
    ece_at_010 = tau_results.get(0.10, {}).get("ece_mean")
    return {
        "tau_results": {str(k): v for k, v in tau_results.items()},
        "optimal_tau": float(opt_tau),
        "ece_at_opt":  round(tau_results[opt_tau]["ece_mean"], 5),
        "ece_at_010":  round(float(ece_at_010), 5) if ece_at_010 is not None else None,
        "recalibrate": bool(abs(opt_tau - 0.10) > 1e-9),
    }


# ── Harness 2: PROD-5 ─────────────────────────────────────────────────────────
def run_prod5(persona, mu_true, categories, cat_to_idx, gt_dists_arr,
              factor_names):
    C, A, d    = mu_true.shape
    team       = persona["analyst_team"]
    apd        = persona["alerts_per_day"]
    noise      = build_persona_noise(persona, factor_names)
    base_w     = build_persona_weights(persona, categories)
    shifts     = persona.get("environment_shifts", [])
    a_params   = precompute_analyst_params(team)
    n_analysts = len(a_params)
    cat_idx    = np.arange(C)

    day_wts = precompute_day_weights(base_w, categories, cat_to_idx, shifts, P5_DAYS)

    all_daily_acc       = np.full((P5_N_SEEDS, P5_DAYS), np.nan)
    all_conv_day        = np.full((P5_N_SEEDS, C), -1, dtype=int)
    all_daily_signal    = np.zeros((P5_N_SEEDS, P5_DAYS))
    total_override      = np.zeros(P5_N_SEEDS, dtype=int)
    total_confirm       = np.zeros(P5_N_SEEDS, dtype=int)
    total_override_corr = np.zeros(P5_N_SEEDS, dtype=int)

    for si in range(P5_N_SEEDS):
        rng    = np.random.RandomState(si + 1000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        scorer = AsymmetricScorer(np.clip(mu_true + offset, 0, 1),
                                  tau=P5_TAU,
                                  eta_confirm=ETA_CONFIRM,
                                  eta_override=ETA_OVERRIDE)

        for day in range(P5_DAYS):
            dw        = day_wts[day]
            n_alerts  = int(rng.poisson(apd))
            n_correct = 0
            day_ov, day_ov_corr = 0, 0

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f       = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise, 0, 1)
                pred_a, _ = scorer.score(f, c)
                n_correct += int(pred_a == true_gt)

                ai = rng.randint(n_analysts)
                eff_over, eff_qual = a_params[ai]

                if rng.random() < eff_over:
                    # Override path — use η_override
                    if rng.random() < eff_qual:
                        gt_a    = true_gt
                        ov_corr = True
                    else:
                        others  = [a for a in range(A) if a != true_gt]
                        gt_a    = int(others[rng.randint(len(others))])
                        ov_corr = False
                    scorer.update_override(f, c, gt_a)
                    day_ov      += 1
                    day_ov_corr += int(ov_corr)
                    total_override[si]      += 1
                    total_override_corr[si] += int(ov_corr)
                else:
                    # Confirm path — use η_confirm
                    scorer.update_confirm(f, c, pred_a)
                    total_confirm[si] += 1

            if n_alerts > 0:
                all_daily_acc[si, day] = n_correct / n_alerts

            # Conservation signal: α·q·V = n_correct_overrides today
            if day_ov > 0 and n_alerts > 0:
                alpha_d  = day_ov / n_alerts
                q_d      = day_ov_corr / day_ov
                all_daily_signal[si, day] = alpha_d * q_d * n_alerts
            # else 0 (already initialized)

            # Convergence check
            for ci in range(C):
                if all_conv_day[si, ci] == -1:
                    per_a = np.array([np.linalg.norm(scorer.mu[ci, a] - mu_true[ci, a])
                                      for a in range(A)])
                    if per_a.max() < P5_EPS:
                        all_conv_day[si, ci] = day + 1

    # Aggregate
    cat_results = {}
    for ci, cat in enumerate(categories):
        days  = all_conv_day[:, ci]
        valid = days[days > 0].astype(float)
        conv_pct = 100 * len(valid) / P5_N_SEEDS
        cat_results[cat] = {
            "converge_pct":      round(conv_pct, 1),
            "mean_conv_day":     round(float(valid.mean()), 1) if len(valid) else None,
            "not_converged":     int((days < 0).sum()),
        }

    mean_acc = np.nanmean(all_daily_acc, axis=0)

    # Conservation signal stats
    sig_flat   = all_daily_signal.flatten()
    sig_active = sig_flat[sig_flat > 0]   # days with at least one override
    breach_days_per_seed = np.sum(
        (all_daily_signal > 0) & (all_daily_signal < THETA_MIN), axis=1)

    # Gate stats
    ov_pct   = float(total_override.sum()) / max(1, total_override.sum() + total_confirm.sum())
    ov_corr_pct = float(total_override_corr.sum()) / max(1, total_override.sum())

    return {
        "categories":         cat_results,
        "acc_day1":           round(float(np.nanmean(all_daily_acc[:, 0])),  4),
        "acc_day30":          round(float(np.nanmean(all_daily_acc[:, 29])), 4),
        "acc_day60":          round(float(np.nanmean(all_daily_acc[:, 59])), 4),
        "daily_acc_mean":     mean_acc.tolist(),
        "n_cats_converged":   int(sum(1 for r in cat_results.values()
                                      if r["converge_pct"] >= 80)),
        "conservation": {
            "min_signal":          round(float(sig_active.min()), 3) if len(sig_active) else 0.0,
            "mean_signal":         round(float(sig_active.mean()), 3) if len(sig_active) else 0.0,
            "mean_breach_days":    round(float(breach_days_per_seed.mean()), 2),
            "any_breach":          bool(breach_days_per_seed.max() > 0),
        },
        "gate_stats": {
            "override_pct":        round(ov_pct, 4),
            "override_correct_pct": round(ov_corr_pct, 4),
        },
        "all_daily_acc":      all_daily_acc.tolist(),   # (N_SEEDS, DAYS) for prod5_detailed.json
    }


# ── Harness 3: B-A Phase B ────────────────────────────────────────────────────
def _hash_analyst(analyst_id: str) -> int:
    return int(hashlib.md5((analyst_id + "salt_ba_persona").encode()).hexdigest(), 16)


def assign_ba_groups(team: list) -> dict:
    groups = {a["id"]: ("A" if _hash_analyst(a["id"]) % 2 == 0 else "B")
              for a in team}
    n_a = sum(1 for g in groups.values() if g == "A")
    if n_a == 0 or n_a == len(team):
        groups = {a["id"]: ("A" if i % 2 == 0 else "B")
                  for i, a in enumerate(team)}
    return groups


def check_ba_gate(log_B):
    v0 = [d for d in log_B if d["variant"] == 0]
    v1 = [d for d in log_B if d["variant"] == 1]
    if len(v0) < BA_GATE_MIN_V_EACH or len(v1) < BA_GATE_MIN_V_EACH:
        return False
    r0 = np.array([d["R"] for d in v0])
    r1 = np.array([d["R"] for d in v1])
    gate1 = float(r1.mean()) > float(r0.mean())
    gate4 = len(v1) >= BA_GATE_MIN_V1
    if not (gate1 and gate4):
        return False
    try:
        _, p = scipy_stats.mannwhitneyu(r1, r0, alternative="greater")
        gate2 = p < 0.10
    except Exception:
        gate2 = False
    days_el = max(1, log_B[-1]["day"])
    a_v1    = float(np.mean([d["accepted"] for d in v1]))
    q_v1    = float(np.mean([d["correct"]  for d in v1]))
    gate3   = a_v1 * q_v1 * (len(v1) / days_el) > THETA_MIN
    return gate1 and gate2 and gate3 and gate4


def run_ba(persona):
    team     = persona["analyst_team"]
    groups   = assign_ba_groups(team)
    a_params = {a["id"]: precompute_analyst_params([a])[0] for a in team}
    team_a   = [a for a in team if groups[a["id"]] == "A"]
    team_b   = [a for a in team if groups[a["id"]] == "B"]

    seed_rew_A = np.zeros(BA_N_SEEDS)
    seed_rew_B = np.zeros(BA_N_SEEDS)
    seed_promo = np.zeros(BA_N_SEEDS, dtype=bool)
    seed_pdec  = np.full(BA_N_SEEDS, np.nan)
    seed_n_A   = np.zeros(BA_N_SEEDS, dtype=int)
    seed_n_B   = np.zeros(BA_N_SEEDS, dtype=int)
    seed_daily_sig = np.zeros((BA_N_SEEDS, BA_DAYS))

    for si in range(BA_N_SEEDS):
        rng      = np.random.RandomState(si + 2000)
        alpha_ts = np.ones(BA_K, dtype=float)
        beta_ts  = np.ones(BA_K, dtype=float)
        log_A, log_B   = [], []
        grp_b_count    = 0
        promoted       = False
        promo_dec      = None
        b_accept_sum = b_correct_sum = b_total = 0.0

        for day in range(BA_DAYS):
            b_today = 0
            for analyst in team:
                if rng.random() > BA_DECISIONS_PER_DAY:
                    continue
                grp                = groups[analyst["id"]]
                eff_over, eff_qual = a_params[analyst["id"]]
                if grp == "A":
                    variant = 0
                    base_q  = BA_QUALITIES[0]
                else:
                    variant = (1 if promoted else
                               int(np.argmax([rng.beta(alpha_ts[v], beta_ts[v])
                                              for v in range(BA_K)])))
                    base_q = BA_QUALITIES[variant]

                effective_q = eff_over * eff_qual + (1 - eff_over) * base_q
                accepted    = int(rng.random() < effective_q)
                res_time    = float(rng.exponential(1 / max(effective_q, 0.1)))
                correct     = int(rng.random() < effective_q)
                rt_score    = min(1.0, 1 / max(res_time, 0.1))
                R           = float(0.4 * accepted + 0.3 * rt_score + 0.3 * correct)
                rec         = {"day": day + 1, "variant": variant,
                               "accepted": accepted, "correct": correct, "R": R}

                if grp == "A":
                    log_A.append(rec)
                    seed_n_A[si] += 1
                else:
                    log_B.append(rec)
                    grp_b_count   += 1
                    b_today       += 1
                    b_accept_sum  += accepted
                    b_correct_sum += correct
                    b_total       += 1
                    if not promoted:
                        if R > 0.5: alpha_ts[variant] += 1.0
                        else:       beta_ts[variant]  += 1.0
                        if (grp_b_count >= BA_GATE_MIN_TOTAL and
                                grp_b_count % BA_GATE_INTERVAL == 0):
                            if check_ba_gate(log_B):
                                promoted  = True
                                promo_dec = grp_b_count

            if b_total > 0 and b_today > 0:
                seed_daily_sig[si, day] = (b_accept_sum/b_total) * \
                                           (b_correct_sum/b_total) * b_today
            seed_n_B[si] = grp_b_count

        seed_rew_A[si] = np.mean([d["R"] for d in log_A]) if log_A else 0
        seed_rew_B[si] = np.mean([d["R"] for d in log_B]) if log_B else 0
        seed_promo[si] = promoted
        seed_pdec[si]  = promo_dec if promo_dec is not None else np.nan

    delta_rew  = float(seed_rew_B.mean() - seed_rew_A.mean())
    try:
        _, p_val = scipy_stats.ttest_ind(seed_rew_B, seed_rew_A,
                                         equal_var=False, alternative="greater")
    except Exception:
        p_val = 1.0

    promo_rate = float(seed_promo.mean())
    valid_pdec = seed_pdec[~np.isnan(seed_pdec)]
    breach_seeds = int(sum(
        1 for si in range(BA_N_SEEDS)
        if np.any((seed_daily_sig[si] > 0) & (seed_daily_sig[si] < THETA_MIN))
    ))

    return {
        "delta_reward":             round(delta_rew, 4),
        "p_value":                  round(float(p_val), 5),
        "promo_rate":               round(promo_rate, 3),
        "mean_decisions_to_promo":  round(float(valid_pdec.mean()), 1)
                                    if len(valid_pdec) else None,
        "breach_rate":              round(breach_seeds / BA_N_SEEDS, 3),
        "n_A_mean":                 round(float(seed_n_A.mean()), 1),
        "n_B_mean":                 round(float(seed_n_B.mean()), 1),
        "power_sufficient":         bool(seed_n_A.mean() >= POWER_TARGET and
                                         seed_n_B.mean() >= POWER_TARGET),
        "group_a_analysts":         [a["id"] for a in team_a],
        "group_b_analysts":         [a["id"] for a in team_b],
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
CATS_ABBREV = {
    "credential_access":    "cred_acc",
    "lateral_movement":     "lat_mov",
    "data_exfiltration":    "data_exf",
    "insider_threat":       "insider",
    "cloud_infrastructure": "cloud",
    "threat_intel_match":   "threat_intel",
}


def print_tables(all_results, personas, categories):
    persona_ids = [p["persona_id"] for p in personas]

    # ── PROD-5 summary ────────────────────────────────────────────────────────
    print("\nPROD-5 Summary:")
    hdr = (f"| {'Persona':8} | {'Team q_bar':9} | {'Day 1 Acc':9} | "
           f"{'Day 30 Acc':10} | {'Day 60 Acc':10} | {'D(60-1)':8} | {'Gate%':6} |")
    sep = "|" + "-"*10+"|"+"-"*11+"|"+"-"*11+"|"+"-"*12+"|"+"-"*12+"|"+"-"*10+"|"+"-"*8+"|"
    print(hdr); print(sep)
    for persona in personas:
        pid = persona["persona_id"]
        r   = all_results[pid]
        p5  = r["prod5"]
        qb  = r["q_bar"]
        d60 = p5["acc_day60"]
        d1  = p5["acc_day1"]
        d30 = p5["acc_day30"]
        delta = d60 - d1
        gate_pct = p5["gate_stats"]["override_pct"] * 100
        print(f"| {pid:8} | {qb:9.3f} | {d1:8.1%}  | "
              f"{d30:9.1%}   | {d60:9.1%}   | {delta:+7.2%}  | {gate_pct:5.1f}%  |")

    # ── Per-category convergence ──────────────────────────────────────────────
    print("\nPer-Category Convergence (mean days to err<0.10, or NC=not converged):")
    abbrevs = [CATS_ABBREV.get(c, c[:10]) for c in categories]
    hdr = "| " + f"{'Persona':8}" + " | " + " | ".join(f"{a:10}" for a in abbrevs) + " |"
    sep = "|" + "-"*10 + ("|" + "-"*12) * len(categories) + "|"
    print(hdr); print(sep)
    for persona in personas:
        pid = persona["persona_id"]
        p5  = all_results[pid]["prod5"]
        vals = []
        for cat in categories:
            cr = p5["categories"][cat]
            if cr["mean_conv_day"] is not None:
                vals.append(f"{cr['mean_conv_day']:.0f}d")
            else:
                vals.append("NC")
        row = "| " + f"{pid:8}" + " | " + " | ".join(f"{v:10}" for v in vals) + " |"
        print(row)

    # ── Conservation signal ───────────────────────────────────────────────────
    print("\nConservation Signal (alpha*q*V per day):")
    hdr = (f"| {'Persona':8} | {'Min a*q*V':9} | {'Mean a*q*V':10} | "
           f"{'Breach Days':11} | {'Status':10} |")
    sep = "|"+"-"*10+"|"+"-"*11+"|"+"-"*12+"|"+"-"*13+"|"+"-"*12+"|"
    print(hdr); print(sep)
    for persona in personas:
        pid = persona["persona_id"]
        cs  = all_results[pid]["prod5"]["conservation"]
        status = "BREACH" if cs["any_breach"] else "OK"
        print(f"| {pid:8} | {cs['min_signal']:9.2f} | {cs['mean_signal']:10.2f} | "
              f"{cs['mean_breach_days']:11.2f} | {status:10} |")

    # ── TD-034 ────────────────────────────────────────────────────────────────
    print("\nTD-034 Summary:")
    hdr = (f"| {'Persona':8} | {'Optimal t':9} | {'ECE@optimal':11} | "
           f"{'ECE@0.10':8} | {'Recalibrate?':12} |")
    sep = "|"+"-"*10+"|"+"-"*11+"|"+"-"*13+"|"+"-"*10+"|"+"-"*14+"|"
    print(hdr); print(sep)
    for persona in personas:
        pid = persona["persona_id"]
        td  = all_results[pid]["td034"]
        r   = "YES" if td["recalibrate"] else "no"
        print(f"| {pid:8} | {td['optimal_tau']:9.3f} | {td['ece_at_opt']:11.5f} | "
              f"{td['ece_at_010']:8.5f} | {r:12} |")

    # ── B-A Phase B ───────────────────────────────────────────────────────────
    print("\nB-A Phase B Summary:")
    hdr = (f"| {'Persona':8} | {'D Reward':8} | {'p-value':7} | "
           f"{'Promoted%':9} | {'CL Breach%':10} | {'Power':6} |")
    sep = "|"+"-"*10+"|"+"-"*10+"|"+"-"*9+"|"+"-"*11+"|"+"-"*12+"|"+"-"*8+"|"
    print(hdr); print(sep)
    for persona in personas:
        pid = persona["persona_id"]
        ba  = all_results[pid]["ba"]
        pw  = "OK" if ba["power_sufficient"] else "LOW"
        print(f"| {pid:8} | {ba['delta_reward']:+8.4f} | {ba['p_value']:7.4f} | "
              f"{ba['promo_rate']*100:8.1f}%  | {ba['breach_rate']*100:9.1f}%   | {pw:6} |")

    # ── Gate evaluation ───────────────────────────────────────────────────────
    print("\nGATE EVALUATION (Day 60 accuracy >= Day 1 accuracy?):")
    all_pass = True
    failed   = []
    for persona in personas:
        pid = persona["persona_id"]
        p5  = all_results[pid]["prod5"]
        d1  = p5["acc_day1"]
        d60 = p5["acc_day60"]
        passes = d60 >= d1
        mark   = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False
            failed.append(pid)
        print(f"  {pid}: day1={d1:.1%} → day60={d60:.1%}  [{mark}]")
    print()
    if all_pass:
        print("  Overall: PASS — all personas improve over 60 days")
    else:
        print(f"  Overall: FAIL — degraded personas: {', '.join(failed)}")


# ── Save ──────────────────────────────────────────────────────────────────────
def save_results(all_results, personas, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    summary = {}
    for pid, r in all_results.items():
        p5 = r["prod5"]
        summary[pid] = {
            "persona_id":  pid,
            "name":        r["name"],
            "industry":    r["industry"],
            "q_bar":       r["q_bar"],
            "td034": {
                "optimal_tau": r["td034"]["optimal_tau"],
                "ece_at_opt":  r["td034"]["ece_at_opt"],
                "ece_at_010":  r["td034"]["ece_at_010"],
                "recalibrate": r["td034"]["recalibrate"],
            },
            "prod5": {
                "acc_day1":       p5["acc_day1"],
                "acc_day30":      p5["acc_day30"],
                "acc_day60":      p5["acc_day60"],
                "n_cats_converged": p5["n_cats_converged"],
                "categories":     p5["categories"],
                "conservation":   p5["conservation"],
                "gate_stats":     p5["gate_stats"],
            },
            "ba": r["ba"],
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {summary_path}")

    # prod5_detailed.json — per-seed, per-day accuracy trajectories
    detailed = {}
    for pid, r in all_results.items():
        detailed[pid] = {
            "daily_acc_per_seed":  r["prod5"]["all_daily_acc"],
            "daily_acc_mean":      r["prod5"]["daily_acc_mean"],
        }
    detail_path = output_dir / "prod5_detailed.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    print(f"  Saved → {detail_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run TD-034 + PROD-5 + B-A Phase B on every persona in a JSON file.")
    parser.add_argument("--personas", required=True,
                        help="Path to personas JSON file")
    parser.add_argument("--output",   required=True,
                        help="Output directory for results")
    args = parser.parse_args()

    personas_path = Path(args.personas)
    output_dir    = Path(args.output)

    if not personas_path.exists():
        raise FileNotFoundError(f"Personas file not found: {personas_path}")

    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)

    # Domain config
    cfg          = load_domain_config("soc_product_v50")
    categories   = cfg["categories"]
    factor_names = cfg["factors"]
    mu_true      = np.array(cfg["mu"], dtype=float)
    gt_dists_raw = cfg["gt_distributions"]
    cat_to_idx   = {c: i for i, c in enumerate(categories)}

    gt_dists_arr = np.array([gt_dists_raw[c] for c in categories], dtype=float)
    gt_dists_arr = gt_dists_arr / gt_dists_arr.sum(axis=1, keepdims=True)

    fname = personas_path.stem
    print()
    print("=" * 62)
    print(f"=== SWEEP RESULTS: {fname} ===")
    print("=" * 62)
    print(f"  Personas file: {personas_path}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Config:        soc_product_v50  C={len(categories)} A={len(cfg['actions'])} d={mu_true.shape[2]}")
    print(f"  eta_confirm={ETA_CONFIRM}  eta_override={ETA_OVERRIDE}  (asymmetric, Q5 fix)")
    print(f"  N personas: {len(personas)}")
    print()

    all_results = {}
    t_total = time.time()

    for persona in personas:
        pid      = persona["persona_id"]
        name     = persona["name"]
        industry = persona["industry"]
        apd      = persona["alerts_per_day"]
        n_an     = len(persona["analyst_team"])
        qb       = persona_q_bar(persona["analyst_team"])

        print(f"{'─'*62}")
        print(f"[{pid}] {name}")
        print(f"       {industry}  |  {apd} alerts/day  |  {n_an} analysts  |  q_bar={qb:.3f}")

        t0     = time.time()
        td_res = run_td034(persona, mu_true, categories, gt_dists_arr, factor_names)
        print(f"  TD-034 : optimal τ={td_res['optimal_tau']:.2f}  "
              f"ECE@opt={td_res['ece_at_opt']:.4f}  "
              f"ECE@0.10={td_res['ece_at_010']:.4f}  ({time.time()-t0:.1f}s)")

        t0     = time.time()
        p5_res = run_prod5(persona, mu_true, categories, cat_to_idx,
                           gt_dists_arr, factor_names)
        nc     = p5_res["n_cats_converged"]
        print(f"  PROD-5 : {nc}/6 cats converge  "
              f"acc {p5_res['acc_day1']:.0%}→{p5_res['acc_day60']:.0%}  "
              f"gate={p5_res['gate_stats']['override_pct']:.1%}  ({time.time()-t0:.1f}s)")

        t0     = time.time()
        ba_res = run_ba(persona)
        print(f"  B-A    : delta={ba_res['delta_reward']:+.4f}  "
              f"promoted={ba_res['promo_rate']*100:.0f}%  "
              f"CL breach={ba_res['breach_rate']*100:.0f}%  ({time.time()-t0:.1f}s)")

        all_results[pid] = {
            "persona_id": pid,
            "name":       name,
            "industry":   industry,
            "q_bar":      round(qb, 4),
            "n_analysts": n_an,
            "td034":      td_res,
            "prod5":      p5_res,
            "ba":         ba_res,
        }

    print()
    print("=" * 62)
    print_tables(all_results, personas, categories)

    elapsed = time.time() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("\nSaving results...")
    save_results(all_results, personas, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
