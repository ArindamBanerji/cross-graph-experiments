"""
BLOCK-5B-PROXY: Min-Confidence Gate Validation
Re-runs the PROD-5 harness on all 9 personas under two conditions:

  Condition A (control):   min_confidence=0.0  — gate OFF, all updates pass
  Condition B (treatment): min_confidence=0.40 — gate ON, suppress low-conf updates

Analyst behaviour model (new for this validation):
  - Every alert goes through scorer.score()
  - If analyst overrides: gt_a = true_gt (with prob eff_qual) or random wrong action
  - Else:                 gt_a = model prediction, is_correct = (pred == true_gt)
  - scorer.update() is ALWAYS called, but internally gated by confidence >= min_confidence
  - No push-away on wrong predictions (only pull toward gt_a)

Motivation: original harness showed 0% convergence across all 9 personas due to
wrong-feedback corruption accumulating faster than correct updates can correct it.
The min_confidence gate is the proposed P0 fix.

Initial-confidence note: at soc_product_v50 (well-separated centroids), initial
confidence ≈ 0.999 (not ~0.25). Gate at 0.40 blocks updates only when centroids
have drifted and the model becomes uncertain — preventing cascading corruption
rather than preventing early learning.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Validation parameters ─────────────────────────────────────────────────────
N_SEEDS   = 10
DAYS      = 60
E0        = 0.15    # cold-start centroid offset
EPS_CONV  = 0.10    # convergence threshold (max per-action L2 distance)
ETA       = 0.05
TAU       = 0.10

MIN_CONF_A = 0.0    # condition A: gate OFF
MIN_CONF_B = 0.40   # condition B: gate ON


# ── Gated scorer ──────────────────────────────────────────────────────────────
class GatedScorer:
    """
    Centroid scorer with a minimum-confidence gate on updates.

    update() signature differs from FastScorer:
      - no pred_a argument
      - no push-away step
      - update fires only when confidence >= min_confidence
    """

    def __init__(self, mu: np.ndarray, tau: float = 0.10,
                 eta: float = 0.05, min_confidence: float = 0.0):
        self.mu             = mu.copy().astype(float)   # (C, A, d)
        self.tau            = tau
        self.eta            = eta
        self.min_confidence = min_confidence
        self.counts         = np.zeros(mu.shape[:2], dtype=int)
        # Telemetry
        self.n_updates_fired    = 0
        self.n_updates_gated    = 0

    def score(self, factors: np.ndarray, c_idx: int):
        """Returns (pred_a, confidence)."""
        dists  = np.sum((self.mu[c_idx] - factors) ** 2, axis=1)
        logits = -dists / self.tau
        logits -= logits.max()
        probs  = np.exp(logits)
        probs /= probs.sum()
        pred_a = int(np.argmax(probs))
        return pred_a, float(probs[pred_a])

    def update(self, factors: np.ndarray, c_idx: int,
               gt_a: int, confidence: float):
        """
        Pull centroid[c_idx, gt_a] toward factors.
        Fires only when confidence >= min_confidence.
        """
        if confidence < self.min_confidence:
            self.n_updates_gated += 1
            return
        self.n_updates_fired += 1
        cnt     = self.counts[c_idx, gt_a]
        eta_eff = self.eta / (1 + cnt * 0.001)
        self.mu[c_idx, gt_a] += eta_eff * (factors - self.mu[c_idx, gt_a])
        np.clip(self.mu[c_idx, gt_a], 0, 1, out=self.mu[c_idx, gt_a])
        self.counts[c_idx, gt_a] += 1


# ── Persona helpers (shared with run_all_harnesses.py) ────────────────────────
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
    params = []
    for a in team:
        eff_over = min(1.0, a["override_rate"] * (1 + a["fatigue_factor"] * 0.3))
        eff_qual = max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2))
        params.append((eff_over, eff_qual))
    return params


# ── Core simulation ───────────────────────────────────────────────────────────
def simulate_persona_condition(persona: dict, mu_true: np.ndarray,
                               categories: list, cat_to_idx: dict,
                               gt_dists_arr: np.ndarray, factor_names: list,
                               min_confidence: float) -> dict:
    """
    Run 60-day PROD-5 simulation for one persona under a single min_confidence.
    Returns per-seed aggregates and summary statistics.
    """
    C, A, d   = mu_true.shape
    team      = persona["analyst_team"]
    apd       = persona["alerts_per_day"]
    noise     = build_persona_noise(persona, factor_names)
    base_w    = build_persona_weights(persona, categories)
    shifts    = persona.get("environment_shifts", [])
    a_params  = precompute_analyst_params(team)
    n_analysts = len(a_params)
    cat_idx   = np.arange(C)

    day_weights = precompute_day_weights(base_w, categories, cat_to_idx,
                                        shifts, DAYS)

    all_conv_day  = np.full((N_SEEDS, C), -1, dtype=int)
    all_daily_acc = np.full((N_SEEDS, DAYS), np.nan)
    all_updates_fired  = np.zeros(N_SEEDS, dtype=int)
    all_updates_gated  = np.zeros(N_SEEDS, dtype=int)

    for si in range(N_SEEDS):
        rng    = np.random.RandomState(si + 2000)
        offset = rng.uniform(-E0, E0, mu_true.shape)
        scorer = GatedScorer(np.clip(mu_true + offset, 0, 1),
                             tau=TAU, eta=ETA, min_confidence=min_confidence)

        for day in range(DAYS):
            dw       = day_weights[day]
            n_alerts = int(rng.poisson(apd))
            n_correct = 0

            for _ in range(n_alerts):
                c      = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f      = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise,
                                 0.0, 1.0)

                pred_a, conf = scorer.score(f, c)
                n_correct += int(pred_a == true_gt)

                # Analyst decision
                ai = rng.randint(n_analysts)
                eff_over, eff_qual = a_params[ai]

                if rng.random() < eff_over:
                    # Analyst overrides model
                    if rng.random() < eff_qual:
                        gt_a = true_gt            # correct override
                    else:
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])  # wrong
                else:
                    # Accept model prediction; reinforce it
                    gt_a = pred_a

                scorer.update(f, c, gt_a, conf)

            # End-of-day
            if n_alerts > 0:
                all_daily_acc[si, day] = n_correct / n_alerts

            mu_now = scorer.mu
            for ci in range(C):
                per_a = np.array([np.linalg.norm(mu_now[ci, a] - mu_true[ci, a])
                                   for a in range(A)])
                if all_conv_day[si, ci] == -1 and per_a.max() < EPS_CONV:
                    all_conv_day[si, ci] = day + 1

        all_updates_fired[si] = scorer.n_updates_fired
        all_updates_gated[si] = scorer.n_updates_gated

    # Aggregate per-category convergence
    cat_results = {}
    for ci, cat in enumerate(categories):
        days  = all_conv_day[:, ci]
        valid = days[days > 0].astype(float)
        cat_results[cat] = {
            "converge_pct":  round(100 * len(valid) / N_SEEDS, 1),
            "mean_weeks":    round(float(valid.mean() / 7), 2) if len(valid) else None,
            "not_converged": int((days < 0).sum()),
        }

    mean_acc = np.nanmean(all_daily_acc, axis=0)

    return {
        "min_confidence": min_confidence,
        "categories":     cat_results,
        "acc_day1":       round(float(np.nanmean(all_daily_acc[:, 0])),  4),
        "acc_day30":      round(float(np.nanmean(all_daily_acc[:, 29])), 4),
        "acc_day60":      round(float(np.nanmean(all_daily_acc[:, 59])), 4),
        "daily_acc_mean": mean_acc.tolist(),
        "n_cats_converged": int(sum(1 for r in cat_results.values()
                                     if r["converge_pct"] >= 80)),
        "updates_fired_mean": round(float(all_updates_fired.mean()), 1),
        "updates_gated_mean": round(float(all_updates_gated.mean()), 1),
        "gate_rate":          round(float(all_updates_gated.mean() /
                                   max(1, all_updates_fired.mean() +
                                          all_updates_gated.mean())), 4),
    }


# ── Entry ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load personas
    personas_path = EXP_DIR / "personas_all.json"
    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)

    # Load domain config
    cfg          = load_domain_config("soc_product_v50")
    categories   = cfg["categories"]
    actions      = cfg["actions"]
    factor_names = cfg["factors"]
    mu_true      = np.array(cfg["mu"], dtype=float)   # (C, A, d)
    gt_dists     = cfg["gt_distributions"]             # dict cat -> list[A]
    cat_to_idx   = {c: i for i, c in enumerate(categories)}

    # Build gt_dists_arr (C, A)
    gt_dists_arr = np.array([gt_dists[c] for c in categories], dtype=float)
    # Normalize rows
    gt_dists_arr = (gt_dists_arr.T / gt_dists_arr.sum(axis=1)).T

    results = {}
    print(f"{'Persona':<6} {'Cond':<5} {'Acc@60':>7} {'Acc@30':>7} "
          f"{'CatConv':>8} {'GateRate':>9}  Time")
    print("-" * 65)

    for persona in personas:
        pid = persona["persona_id"]
        results[pid] = {
            "persona_id": pid,
            "judge":      persona["judge"],
            "industry":   persona["industry"],
        }

        for label, min_conf in [("A_off", MIN_CONF_A), ("B_on", MIN_CONF_B)]:
            t1 = time.time()
            cond = simulate_persona_condition(
                persona, mu_true, categories, cat_to_idx,
                gt_dists_arr, factor_names, min_conf
            )
            elapsed = time.time() - t1
            results[pid][label] = cond

            cond_label = "OFF" if min_conf == 0.0 else f"{min_conf:.2f}"
            print(f"{pid:<6} {cond_label:<5} "
                  f"{cond['acc_day60']:>7.1%} {cond['acc_day30']:>7.1%} "
                  f"{cond['n_cats_converged']:>3}/6     "
                  f"{cond['gate_rate']:>7.1%}    {elapsed:.1f}s")

    print()
    # Summary table: delta acc@60 and delta cats_converged
    print("── Delta Summary (B_on − A_off) ─────────────────────────────────")
    print(f"{'Persona':<6} {'Industry':<20} {'Δacc@60':>9} {'Δcats':>7} "
          f"{'Gate%':>8}")
    print("-" * 55)
    for persona in personas:
        pid = persona["persona_id"]
        a   = results[pid]["A_off"]
        b   = results[pid]["B_on"]
        delta_acc  = b["acc_day60"] - a["acc_day60"]
        delta_cats = b["n_cats_converged"] - a["n_cats_converged"]
        gate_pct   = b["gate_rate"] * 100
        sign = "+" if delta_acc >= 0 else ""
        print(f"{pid:<6} {persona['industry']:<20} "
              f"{sign}{delta_acc:>+.2%}   {delta_cats:>+3}    {gate_pct:>6.1f}%")

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s")

    # Save
    out = {
        "experiment":    "validate_min_confidence_gate",
        "n_seeds":       N_SEEDS,
        "days":          DAYS,
        "eps_conv":      EPS_CONV,
        "eta":           ETA,
        "tau":           TAU,
        "min_conf_A":    MIN_CONF_A,
        "min_conf_B":    MIN_CONF_B,
        "e0":            E0,
        "results":       results,
    }
    out_path = RESULTS_DIR / "min_confidence_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
