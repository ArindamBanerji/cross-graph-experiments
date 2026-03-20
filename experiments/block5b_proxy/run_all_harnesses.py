"""
BLOCK-5B-PROXY: 9-Persona Harness Runner
Runs TD-034, PROD-5, and B-A Phase B on all 9 LLM-generated customer personas.
27 total harness runs (9 personas × 3 harnesses).
"""

import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config

# ── Harness parameters ────────────────────────────────────────────────────────
TD_TAU_VALUES  = [0.05, 0.08, 0.10, 0.12, 0.15]
TD_N_SEEDS     = 20
TD_N_DECISIONS = 300

P5_N_SEEDS  = 15
P5_DAYS     = 60
P5_E0       = 0.15
P5_EPS      = 0.10    # generous convergence threshold (vs 0.05 in standalone)
P5_ETA      = 0.05
P5_ETA_NEG  = 0.05
P5_TAU      = 0.10

BA_N_SEEDS          = 15
BA_DAYS             = 90
BA_K                = 2
BA_QUALITIES        = [0.70, 0.80]
BA_THETA_MIN        = 0.434
BA_DECISIONS_PER_DAY = 0.85
BA_GATE_MIN_TOTAL   = 100     # first gate check after 100 Group B decisions
BA_GATE_INTERVAL    = 50
BA_GATE_MIN_V_EACH  = 15      # per-variant minimum for gate
BA_GATE_MIN_V1      = 30      # Gate 4 threshold (smaller for persona teams)
POWER_TARGET        = 380

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ── Lightweight scorer (avoids ProfileScorer overhead) ───────────────────────
class FastScorer:
    """Minimal centroid scorer matching ProfileScorer semantics."""

    def __init__(self, mu: np.ndarray, tau: float = 0.10,
                 eta: float = 0.05, eta_neg: float = 0.05):
        self.mu      = mu.copy().astype(float)   # (C, A, d)
        self.tau     = tau
        self.eta     = eta
        self.eta_neg = eta_neg
        self.counts  = np.zeros(mu.shape[:2], dtype=int)

    def score(self, factors: np.ndarray, c_idx: int):
        dists  = np.sum((self.mu[c_idx] - factors) ** 2, axis=1)  # (A,)
        logits = -dists / self.tau
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()
        pred_a  = int(np.argmax(probs))
        return pred_a, float(probs[pred_a])

    def update(self, factors: np.ndarray, c_idx: int,
               pred_a: int, correct: bool, gt_a: int):
        # Pull gt_a toward factors
        cnt     = self.counts[c_idx, gt_a]
        eta_eff = self.eta / (1 + cnt * 0.001)
        self.mu[c_idx, gt_a] += eta_eff * (factors - self.mu[c_idx, gt_a])
        np.clip(self.mu[c_idx, gt_a], 0, 1, out=self.mu[c_idx, gt_a])
        self.counts[c_idx, gt_a] += 1
        # Push pred_a away if wrong
        if (not correct) and pred_a != gt_a:
            cnt_p     = self.counts[c_idx, pred_a]
            eta_p_eff = self.eta_neg / (1 + cnt_p * 0.001)
            self.mu[c_idx, pred_a] -= eta_p_eff * (factors - self.mu[c_idx, pred_a])
            np.clip(self.mu[c_idx, pred_a], 0, 1, out=self.mu[c_idx, pred_a])
            self.counts[c_idx, pred_a] += 1


# ── Shared utilities ──────────────────────────────────────────────────────────
def compute_ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(confs)
    for i in range(n_bins):
        mask = (confs >= edges[i]) & (confs < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(corrects[mask].mean() - confs[mask].mean())
    return float(ece)


def build_persona_noise(persona: dict, factor_names: list) -> np.ndarray:
    return np.array([persona["factor_noise_profile"][f]["base_noise"]
                     for f in factor_names])


def build_persona_weights(persona: dict, categories: list) -> np.ndarray:
    w = np.array([persona["category_distribution"][c] for c in categories], dtype=float)
    return w / w.sum()


def precompute_day_weights(base_w: np.ndarray, categories: list,
                           cat_to_idx: dict, shifts: list, n_days: int) -> np.ndarray:
    """Return (n_days, C) array of normalized category weights per day."""
    day_weights = np.zeros((n_days, len(categories)))
    for day in range(n_days):
        current_day = day + 1
        w = base_w.copy()
        for shift in shifts:
            start = shift["day"]
            end   = start + shift["duration_days"]
            if start <= current_day < end:
                for cat, mult in shift["category_impact"].items():
                    if cat in cat_to_idx:
                        w[cat_to_idx[cat]] *= mult
        day_weights[day] = w / w.sum()
    return day_weights


def precompute_analyst_params(team: list) -> list:
    """Pre-compute (eff_override, eff_quality) for each analyst."""
    params = []
    for a in team:
        eff_over = min(1.0, a["override_rate"]  * (1 + a["fatigue_factor"] * 0.3))
        eff_qual = max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2))
        params.append((eff_over, eff_qual))
    return params


def hash_group(analyst_id: str, salt: str = "salt_ba_persona") -> str:
    h = int(hashlib.md5((analyst_id + salt).encode()).hexdigest(), 16)
    return "A" if h % 2 == 0 else "B"


def assign_ba_groups(team: list) -> dict:
    """Balanced hash-based group assignment. Falls back to alternating if unbalanced."""
    groups = {a["id"]: hash_group(a["id"]) for a in team}
    n_a = sum(1 for g in groups.values() if g == "A")
    n_b = len(team) - n_a
    if n_a == 0 or n_b == 0:
        groups = {a["id"]: "A" if i % 2 == 0 else "B"
                  for i, a in enumerate(team)}
    return groups


def check_ba_gate(log_B: list) -> bool:
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

    days_el  = max(1, log_B[-1]["day"])
    a_v1     = float(np.mean([d["accepted"] for d in v1]))
    q_v1     = float(np.mean([d["correct"]  for d in v1]))
    gate3    = a_v1 * q_v1 * (len(v1) / days_el) > BA_THETA_MIN

    return gate1 and gate2 and gate3 and gate4


# ── Harness 1: TD-034 (τ recalibration) ──────────────────────────────────────
def run_td034(persona: dict, mu_true: np.ndarray, categories: list,
              gt_dists_arr: np.ndarray, factor_names: list) -> dict:
    C, A, d  = mu_true.shape
    noise    = build_persona_noise(persona, factor_names)
    weights  = build_persona_weights(persona, categories)
    cat_idx  = np.arange(C)
    seeds    = list(range(TD_N_SEEDS))

    tau_results = {}
    for tau in TD_TAU_VALUES:
        seed_ece, seed_acc = [], []
        for seed in seeds:
            rng    = np.random.RandomState(seed)
            scorer = FastScorer(mu_true, tau=tau)
            confs    = np.empty(TD_N_DECISIONS)
            corrects = np.empty(TD_N_DECISIONS, dtype=float)

            for j in range(TD_N_DECISIONS):
                c  = int(rng.choice(cat_idx, p=weights))
                gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f  = np.clip(mu_true[c, gt] + rng.randn(d) * noise, 0.0, 1.0)
                pa, conf  = scorer.score(f, c)
                confs[j]    = conf
                corrects[j] = float(pa == gt)

            seed_ece.append(compute_ece(confs, corrects))
            seed_acc.append(float(corrects.mean()))

        tau_results[tau] = {
            "ece_mean": float(np.mean(seed_ece)),
            "ece_std":  float(np.std(seed_ece)),
            "acc_mean": float(np.mean(seed_acc)),
        }

    opt_tau    = min(tau_results, key=lambda t: tau_results[t]["ece_mean"])
    ece_at_010 = tau_results.get(0.10, {}).get("ece_mean")
    return {
        "tau_results": {str(k): v for k, v in tau_results.items()},
        "optimal_tau": float(opt_tau),
        "ece_at_010":  round(float(ece_at_010), 5) if ece_at_010 is not None else None,
        "recalibrate": bool(abs(opt_tau - 0.10) > 1e-9),
    }


# ── Harness 2: PROD-5 (60-day convergence) ────────────────────────────────────
def run_prod5(persona: dict, mu_true: np.ndarray, categories: list,
              cat_to_idx: dict, gt_dists_arr: np.ndarray,
              factor_names: list) -> dict:
    C, A, d   = mu_true.shape
    team      = persona["analyst_team"]
    apd       = persona["alerts_per_day"]
    noise     = build_persona_noise(persona, factor_names)
    base_w    = build_persona_weights(persona, categories)
    shifts    = persona.get("environment_shifts", [])
    a_params  = precompute_analyst_params(team)
    cat_idx   = np.arange(C)
    seeds     = list(range(P5_N_SEEDS))

    # Pre-compute shifted weights for every day (deterministic)
    day_weights = precompute_day_weights(base_w, categories, cat_to_idx, shifts, P5_DAYS)

    all_conv_day  = np.full((P5_N_SEEDS, C), -1, dtype=int)
    all_daily_err = np.zeros((P5_N_SEEDS, C, P5_DAYS))
    all_daily_acc = np.full((P5_N_SEEDS, P5_DAYS), np.nan)

    for si, seed in enumerate(seeds):
        rng    = np.random.RandomState(seed + 1000)
        offset = rng.uniform(-P5_E0, P5_E0, mu_true.shape)
        scorer = FastScorer(np.clip(mu_true + offset, 0, 1),
                            tau=P5_TAU, eta=P5_ETA, eta_neg=P5_ETA_NEG)
        n_analysts = len(a_params)

        for day in range(P5_DAYS):
            dw       = day_weights[day]
            n_alerts = int(rng.poisson(apd))
            n_correct = 0

            for _ in range(n_alerts):
                c  = int(rng.choice(cat_idx, p=dw))
                gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f  = np.clip(mu_true[c, gt] + rng.randn(d) * noise, 0.0, 1.0)
                pa, _ = scorer.score(f, c)
                correct = (pa == gt)
                n_correct += int(correct)

                # Analyst review
                ai = rng.randint(n_analysts)
                eff_over, eff_qual = a_params[ai]
                if rng.random() < eff_over:
                    if rng.random() < eff_qual:
                        scorer.update(f, c, pa, correct, gt)
                    else:
                        others = [a for a in range(A) if a != gt]
                        wrong  = int(others[rng.randint(len(others))])
                        scorer.update(f, c, pa, pa == wrong, wrong)

            # End-of-day: record per-category error and accuracy
            mu_now = scorer.mu
            for ci in range(C):
                per_a = np.array([np.linalg.norm(mu_now[ci, a] - mu_true[ci, a])
                                   for a in range(A)])
                all_daily_err[si, ci, day] = per_a.mean()
                if all_conv_day[si, ci] == -1 and per_a.max() < P5_EPS:
                    all_conv_day[si, ci] = day + 1  # 1-indexed

            if n_alerts > 0:
                all_daily_acc[si, day] = n_correct / n_alerts

    # Aggregate
    cat_results = {}
    for ci, cat in enumerate(categories):
        days  = all_conv_day[:, ci]
        valid = days[days > 0].astype(float)
        cat_results[cat] = {
            "converge_pct":  round(100 * len(valid) / P5_N_SEEDS, 1),
            "mean_weeks":    round(float(valid.mean() / 7), 2) if len(valid) > 0 else None,
            "not_converged": int((days < 0).sum()),
            "daily_err_mean": all_daily_err[:, ci, :].mean(axis=0).tolist(),
        }

    mean_acc = np.nanmean(all_daily_acc, axis=0)  # (DAYS,)
    return {
        "categories":             cat_results,
        "acc_day1":               round(float(np.nanmean(all_daily_acc[:, 0])),  4),
        "acc_day30":              round(float(np.nanmean(all_daily_acc[:, 29])), 4),
        "acc_day60":              round(float(np.nanmean(all_daily_acc[:, 59])), 4),
        "daily_acc_mean":         mean_acc.tolist(),
        "n_categories_converged": int(sum(1 for r in cat_results.values()
                                          if r["converge_pct"] >= 80)),
    }


# ── Harness 3: B-A Phase B (90-day A/B) ──────────────────────────────────────
def run_ba(persona: dict) -> dict:
    team     = persona["analyst_team"]
    groups   = assign_ba_groups(team)
    a_params = {a["id"]: precompute_analyst_params([a])[0] for a in team}
    seeds    = list(range(BA_N_SEEDS))

    team_a = [a for a in team if groups[a["id"]] == "A"]
    team_b = [a for a in team if groups[a["id"]] == "B"]

    seed_rew_A  = np.zeros(BA_N_SEEDS)
    seed_rew_B  = np.zeros(BA_N_SEEDS)
    seed_acc_A  = np.zeros(BA_N_SEEDS)
    seed_acc_B  = np.zeros(BA_N_SEEDS)
    seed_n_A    = np.zeros(BA_N_SEEDS, dtype=int)
    seed_n_B    = np.zeros(BA_N_SEEDS, dtype=int)
    seed_promo  = np.zeros(BA_N_SEEDS, dtype=bool)
    seed_pdec   = np.full(BA_N_SEEDS, np.nan)
    seed_daily_sig = np.zeros((BA_N_SEEDS, BA_DAYS))

    for si, seed in enumerate(seeds):
        rng       = np.random.RandomState(seed + 2000)
        alpha_ts  = np.ones(BA_K, dtype=float)
        beta_ts   = np.ones(BA_K, dtype=float)

        log_A, log_B = [], []
        grp_b_count  = 0
        promoted     = False
        promo_dec    = None

        b_accept_sum = b_correct_sum = b_total = 0.0
        daily_sig = []

        for day in range(BA_DAYS):
            b_today = 0

            for analyst in team:
                if rng.random() > BA_DECISIONS_PER_DAY:
                    continue   # no decision this analyst today

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

                effective_q = eff_over * eff_qual + (1.0 - eff_over) * base_q
                accepted    = int(rng.random() < effective_q)
                res_time    = float(rng.exponential(1.0 / max(effective_q, 0.1)))
                correct     = int(rng.random() < effective_q)
                rt_score    = min(1.0, 1.0 / max(res_time, 0.1))
                R           = float(0.4 * accepted + 0.3 * rt_score + 0.3 * correct)

                rec = {"day": day + 1, "variant": variant,
                       "accepted": accepted, "correct": correct, "R": R}

                if grp == "A":
                    log_A.append(rec)
                else:
                    log_B.append(rec)
                    grp_b_count  += 1
                    b_today      += 1
                    b_accept_sum += accepted
                    b_correct_sum += correct
                    b_total      += 1

                    if not promoted:
                        if R > 0.5:
                            alpha_ts[variant] += 1.0
                        else:
                            beta_ts[variant]  += 1.0
                        if (grp_b_count >= BA_GATE_MIN_TOTAL
                                and grp_b_count % BA_GATE_INTERVAL == 0):
                            if check_ba_gate(log_B):
                                promoted  = True
                                promo_dec = grp_b_count

            # Daily conservation signal
            if b_total > 0 and b_today > 0:
                sig = (b_accept_sum / b_total) * (b_correct_sum / b_total) * b_today
            else:
                sig = 0.0
            daily_sig.append(float(sig))

        # Aggregate seed
        if log_A:
            seed_rew_A[si] = np.mean([d["R"]        for d in log_A])
            seed_acc_A[si] = np.mean([d["accepted"] for d in log_A])
            seed_n_A[si]   = len(log_A)
        if log_B:
            seed_rew_B[si] = np.mean([d["R"]        for d in log_B])
            seed_acc_B[si] = np.mean([d["accepted"] for d in log_B])
            seed_n_B[si]   = len(log_B)
        seed_promo[si]        = promoted
        seed_pdec[si]         = promo_dec if promo_dec is not None else np.nan
        seed_daily_sig[si]    = daily_sig

    # Statistics
    delta_rew = float(seed_rew_B.mean() - seed_rew_A.mean())
    try:
        _, p_val = scipy_stats.ttest_ind(seed_rew_B, seed_rew_A,
                                         equal_var=False, alternative="greater")
    except Exception:
        p_val = 1.0

    promo_rate = float(seed_promo.mean())
    valid_pdec = seed_pdec[~np.isnan(seed_pdec)]
    mean_pdec  = float(valid_pdec.mean()) if len(valid_pdec) > 0 else None

    n_A_mean = float(seed_n_A.mean())
    n_B_mean = float(seed_n_B.mean())

    # Conservation breach (days with decisions but signal < θ_min)
    breach_seeds = int(sum(
        1 for si in range(BA_N_SEEDS)
        if np.any((seed_daily_sig[si] > 0) & (seed_daily_sig[si] < BA_THETA_MIN))
    ))

    return {
        "delta_reward":              round(delta_rew, 4),
        "p_value":                   round(float(p_val), 5),
        "promo_rate":                round(promo_rate, 3),
        "mean_decisions_to_promo":   round(mean_pdec, 1) if mean_pdec else None,
        "breach_rate":               round(breach_seeds / BA_N_SEEDS, 3),
        "breach_seeds":              breach_seeds,
        "n_A_mean":                  round(n_A_mean, 1),
        "n_B_mean":                  round(n_B_mean, 1),
        "power_sufficient":          bool(n_A_mean >= POWER_TARGET
                                          and n_B_mean >= POWER_TARGET),
        "group_a_analysts":          [a["id"] for a in team_a],
        "group_b_analysts":          [a["id"] for a in team_b],
        "acc_A_mean":                round(float(seed_acc_A.mean()), 4),
        "acc_B_mean":                round(float(seed_acc_B.mean()), 4),
    }


# ── Issue detection ───────────────────────────────────────────────────────────
def detect_issues(all_results: dict, categories: list) -> list:
    issues = []
    for pid, res in all_results.items():
        td = res["td034"]
        p5 = res["prod5"]
        ba = res["ba"]
        lbl = f"{pid} ({res['name']}, {res['industry']})"

        if td["recalibrate"]:
            issues.append(
                f"[CALIBRATION] {lbl}: optimal τ={td['optimal_tau']:.2f} "
                f"≠ 0.10. ECE@τ=0.10={td['ece_at_010']:.4f}. "
                f"{'High-noise' if td['ece_at_010'] > 0.10 else 'Low-noise'} "
                f"customer profile shifts optimal temperature."
            )

        for cat, cr in p5["categories"].items():
            if cr["converge_pct"] < 80:
                wks = f"{cr['mean_weeks']:.1f} wk" if cr["mean_weeks"] else ">8.6 wk"
                issues.append(
                    f"[CONVERGENCE] {lbl} — {cat}: only {cr['converge_pct']:.0f}% "
                    f"of seeds converge (mean {wks}). "
                    f"Cause: low category volume or high factor noise."
                )

        if ba["breach_rate"] > 0:
            issues.append(
                f"[CONSERVATION] {lbl}: CL breach in "
                f"{ba['breach_rate']*100:.0f}% of seeds. "
                f"Small analyst team ({res['n_analysts']} analysts) + fatigue "
                f"reduces daily V below θ_min={BA_THETA_MIN}."
            )

        if not ba["power_sufficient"]:
            issues.append(
                f"[POWER] {lbl}: Group A={ba['n_A_mean']:.0f}, "
                f"Group B={ba['n_B_mean']:.0f} decisions "
                f"(target {POWER_TARGET}). "
                f"Analyst team size ({res['n_analysts']}) limits B-A statistical power."
            )

        acc_delta = p5["acc_day60"] - p5["acc_day1"]
        if acc_delta < 0.01:
            issues.append(
                f"[ACCURACY] {lbl}: accuracy flat/declining "
                f"(day1={p5['acc_day1']:.1%} → day60={p5['acc_day60']:.1%}, "
                f"Δ={acc_delta:+.1%}). Learning not taking hold — "
                f"check verification rate and noise floor."
            )

    return issues


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_td_table(all_results):
    print("\nTD-034: τ Recalibration")
    hdr = (f"| {'Persona':7} | {'Industry':20} | {'τ*':4} | "
           f"{'ECE@0.10':8} | {'Recalibrate?':12} |")
    sep = "|" + "-"*9+"|"+"-"*22+"|"+"-"*6+"|"+"-"*10+"|"+"-"*14+"|"
    print(hdr); print(sep)
    for pid, res in all_results.items():
        td = res["td034"]
        r  = "YES" if td["recalibrate"] else "no"
        print(f"| {pid:7} | {res['industry']:20} | {td['optimal_tau']:4.2f} | "
              f"{td['ece_at_010']:8.4f} | {r:12} |")


def print_p5_failures(all_results):
    print("\nPROD-5: Categories NOT converging (< 80% seeds within 60 days)")
    hdr = f"| {'Persona':7} | {'Category':22} | {'% conv':7} | {'Mean wks':8} |"
    sep = "|"+"-"*9+"|"+"-"*24+"|"+"-"*9+"|"+"-"*10+"|"
    print(hdr); print(sep)
    any_fail = False
    for pid, res in all_results.items():
        for cat, cr in res["prod5"]["categories"].items():
            if cr["converge_pct"] < 80:
                wks = f"{cr['mean_weeks']:.1f}" if cr["mean_weeks"] else ">8.6"
                print(f"| {pid:7} | {cat:22} | {cr['converge_pct']:6.0f}% | {wks:8} |")
                any_fail = True
    if not any_fail:
        print("  (none — all categories converge in ≥80% of seeds)")


def print_ba_table(all_results):
    print("\nB-A Phase B: Level 2 Readiness")
    hdr = (f"| {'Persona':7} | {'Promoted':8} | {'CL breach':9} | "
           f"{'Δ reward':8} | {'p-value':7} | {'Power':6} |")
    sep = "|"+"-"*9+"|"+"-"*10+"|"+"-"*11+"|"+"-"*10+"|"+"-"*9+"|"+"-"*8+"|"
    print(hdr); print(sep)
    for pid, res in all_results.items():
        ba = res["ba"]
        pw = "OK" if ba["power_sufficient"] else "LOW"
        print(f"| {pid:7} | {ba['promo_rate']*100:5.0f}%    | "
              f"{ba['breach_rate']*100:6.0f}%     | "
              f"{ba['delta_reward']:+8.4f} | {ba['p_value']:7.4f} | {pw:6} |")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()

    # Config
    config       = load_domain_config("soc_product_v50")
    mu_true      = config["mu"].copy().astype(np.float64)
    categories   = config["categories"]
    factor_names = config["factors"]
    C, A, d      = mu_true.shape
    cat_to_idx   = {c: i for i, c in enumerate(categories)}

    gt_dists_raw = config["gt_distributions"]
    gt_dists_arr = np.array([gt_dists_raw[c] for c in categories], dtype=float)
    gt_dists_arr = gt_dists_arr / gt_dists_arr.sum(axis=1, keepdims=True)

    # Load personas
    personas_path = EXP_DIR / "personas_all.json"
    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)

    ids = [p["persona_id"] for p in personas]
    print(f"Loaded {len(personas)} personas: {ids}")
    print(f"Categories: {categories}")
    print(f"Factors:    {factor_names}")
    print()

    all_results = {}

    for persona in personas:
        pid      = persona["persona_id"]
        name     = persona["name"]
        industry = persona["industry"]
        apd      = persona["alerts_per_day"]
        n_an     = len(persona["analyst_team"])

        print(f"{'─'*62}")
        print(f"[{pid}] {name}")
        print(f"       {industry}  |  {apd} alerts/day  |  {n_an} analysts")

        t0 = time.time()
        td_res = run_td034(persona, mu_true, categories, gt_dists_arr, factor_names)
        t_td   = time.time() - t0
        print(f"  TD-034: optimal τ={td_res['optimal_tau']:.2f}, "
              f"ECE@0.10={td_res['ece_at_010']:.4f}  ({t_td:.1f}s)")

        t0 = time.time()
        p5_res = run_prod5(persona, mu_true, categories, cat_to_idx,
                           gt_dists_arr, factor_names)
        t_p5   = time.time() - t0
        nc     = p5_res["n_categories_converged"]
        print(f"  PROD-5: {nc}/6 converge,  "
              f"acc {p5_res['acc_day1']:.0%}→{p5_res['acc_day60']:.0%}  ({t_p5:.1f}s)")

        t0 = time.time()
        ba_res = run_ba(persona)
        t_ba   = time.time() - t0
        print(f"  B-A:    Δ={ba_res['delta_reward']:+.4f}, "
              f"promoted={ba_res['promo_rate']*100:.0f}%, "
              f"CL breach={ba_res['breach_rate']*100:.0f}%  ({t_ba:.1f}s)")

        all_results[pid] = {
            "persona_id":    pid,
            "name":          name,
            "industry":      industry,
            "alerts_per_day": apd,
            "n_analysts":    n_an,
            "judge":         persona.get("judge", ""),
            "td034":         td_res,
            "prod5":         p5_res,
            "ba":            ba_res,
        }

    # ── Cross-persona summary ─────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("=== BLOCK 5B PROXY: 9-PERSONA HARNESS RESULTS ===")
    print("=" * 62)
    print()
    for pid, res in all_results.items():
        td = res["td034"]; p5 = res["prod5"]; ba = res["ba"]
        print(f"  [{pid}] {res['name']}")
        print(f"         {res['industry']} | {res['alerts_per_day']}/day | "
              f"{res['n_analysts']} analysts")
        print(f"    TD-034: optimal τ={td['optimal_tau']:.2f}, "
              f"ECE@0.10={td['ece_at_010']:.4f}")
        print(f"    PROD-5: {p5['n_categories_converged']}/6 categories converge, "
              f"acc {p5['acc_day1']:.0%}→{p5['acc_day60']:.0%}")
        print(f"    B-A:    Δ reward={ba['delta_reward']:+.4f}, "
              f"promoted {ba['promo_rate']*100:.0f}%, "
              f"CL breach {ba['breach_rate']*100:.0f}%")
        print()

    print_td_table(all_results)
    print_p5_failures(all_results)
    print_ba_table(all_results)

    issues = detect_issues(all_results, categories)
    print()
    print("ISSUES SURFACED (not found in Bernoulli testing):")
    if issues:
        for iss in issues:
            print(f"  {iss}")
    else:
        print("  (none)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    output = {
        "meta": {
            "n_personas":  len(personas),
            "persona_ids": ids,
            "runtime_s":   round(elapsed, 1),
            "harness_config": {
                "td034": {"tau_values": TD_TAU_VALUES,
                          "n_seeds": TD_N_SEEDS, "n_decisions": TD_N_DECISIONS},
                "prod5": {"n_seeds": P5_N_SEEDS, "days": P5_DAYS,
                          "eps_conv": P5_EPS, "e0": P5_E0},
                "ba":    {"n_seeds": BA_N_SEEDS, "days": BA_DAYS,
                          "qualities": BA_QUALITIES, "theta_min": BA_THETA_MIN},
            },
        },
        "results": all_results,
        "issues":  issues,
    }

    out_path = RESULTS_DIR / "all_harness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved → {out_path}")
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
