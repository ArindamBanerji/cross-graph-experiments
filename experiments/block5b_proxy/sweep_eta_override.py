"""
BLOCK-5B-PROXY: Q5 — η_override Sweep (9 personas × 6 values)

Sweeps η_override ∈ {0.005, 0.01, 0.02, 0.03, 0.04, 0.05}
with η_confirm fixed at 0.05 on the PROD-5 (60-day) harness.

Two-rate learning: override-path updates use η_override; confirm-path
updates use η_confirm (0.05). Lower η_override reduces centroid corruption
from low-quality overrides while preserving the learning signal from
high-quality senior analyst corrections.

Theoretical predictions from roadmap:
  Mixed team  (q̄ ≥ 0.70, σ²_q ≈ 0.03):  η* = 0.023
  Junior-heavy (q̄ < 0.70, σ²_q ≈ 0.05): η* = 0.019
"""

import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# ── Sweep parameters ──────────────────────────────────────────────────────────
N_SEEDS           = 10
DAYS              = 60
E0                = 0.15
TAU               = 0.10
ETA_CONFIRM       = 0.05   # fixed for confirm path
ETA_OVERRIDE_VALS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
EPS_CONV          = 0.10

# Theoretical predictions
THEORY_MIXED   = 0.023   # q̄ ≥ 0.70
THEORY_JUNIOR  = 0.019   # q̄ < 0.70


# ── Dual-rate scorer ──────────────────────────────────────────────────────────
class DualRateScorer:
    """
    Centroid scorer with separate learning rates for confirm vs override paths.
    No push-away (avoids double-penalising the override path).
    """

    def __init__(self, mu: np.ndarray, tau: float = 0.10):
        self.mu     = mu.copy().astype(float)   # (C, A, d)
        self.tau    = tau
        self.counts = np.zeros(mu.shape[:2], dtype=int)

    def score(self, factors: np.ndarray, c_idx: int):
        dists  = np.sum((self.mu[c_idx] - factors) ** 2, axis=1)
        logits = -dists / self.tau
        logits -= logits.max()
        probs  = np.exp(logits)
        probs /= probs.sum()
        pred_a = int(np.argmax(probs))
        return pred_a, float(probs[pred_a])

    def update(self, factors: np.ndarray, c_idx: int, gt_a: int,
               eta_eff: float):
        """Pull centroid[c_idx, gt_a] toward factors using decayed eta_eff."""
        cnt     = self.counts[c_idx, gt_a]
        eta     = eta_eff / (1 + cnt * 0.001)
        self.mu[c_idx, gt_a] += eta * (factors - self.mu[c_idx, gt_a])
        np.clip(self.mu[c_idx, gt_a], 0, 1, out=self.mu[c_idx, gt_a])
        self.counts[c_idx, gt_a] += 1


# ── Persona helpers ────────────────────────────────────────────────────────────
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


def persona_q_bar(team: list) -> float:
    """Mean effective quality across analyst team."""
    qs = [max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2))
          for a in team]
    return float(np.mean(qs))


def persona_q_var(team: list) -> float:
    """Variance of effective quality across analyst team."""
    qs = [max(0.4, a["override_quality"] * (1 - a["fatigue_factor"] * 0.2))
          for a in team]
    return float(np.var(qs))


# ── Core simulation ────────────────────────────────────────────────────────────
def simulate_persona_eta(persona: dict, mu_true: np.ndarray,
                          categories: list, cat_to_idx: dict,
                          gt_dists_arr: np.ndarray, factor_names: list,
                          eta_override: float) -> dict:
    """
    60-day PROD-5 sim for one persona at one η_override.
    Returns acc_day60, acc_day30, daily_acc_mean, n_cats_converged.
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

    all_daily_acc = np.full((N_SEEDS, DAYS), np.nan)
    all_conv_day  = np.full((N_SEEDS, C), -1, dtype=int)

    for si in range(N_SEEDS):
        rng    = np.random.RandomState(si + 3000)
        offset = rng.uniform(-E0, E0, mu_true.shape)
        scorer = DualRateScorer(np.clip(mu_true + offset, 0, 1), tau=TAU)

        for day in range(DAYS):
            dw       = day_weights[day]
            n_alerts = int(rng.poisson(apd))
            n_correct = 0

            for _ in range(n_alerts):
                c       = int(rng.choice(cat_idx, p=dw))
                true_gt = int(rng.choice(A, p=gt_dists_arr[c]))
                f       = np.clip(mu_true[c, true_gt] + rng.randn(d) * noise,
                                  0.0, 1.0)

                pred_a, _ = scorer.score(f, c)
                n_correct += int(pred_a == true_gt)

                # Analyst decision
                ai = rng.randint(n_analysts)
                eff_over, eff_qual = a_params[ai]

                if rng.random() < eff_over:
                    # Override path — use η_override
                    if rng.random() < eff_qual:
                        gt_a = true_gt
                    else:
                        others = [a for a in range(A) if a != true_gt]
                        gt_a   = int(others[rng.randint(len(others))])
                    scorer.update(f, c, gt_a, eta_override)
                else:
                    # Confirm path — use η_confirm (0.05)
                    scorer.update(f, c, pred_a, ETA_CONFIRM)

            if n_alerts > 0:
                all_daily_acc[si, day] = n_correct / n_alerts

            # Convergence check
            mu_now = scorer.mu
            for ci in range(C):
                per_a = np.array([np.linalg.norm(mu_now[ci, a] - mu_true[ci, a])
                                   for a in range(A)])
                if all_conv_day[si, ci] == -1 and per_a.max() < EPS_CONV:
                    all_conv_day[si, ci] = day + 1

    mean_acc = np.nanmean(all_daily_acc, axis=0)  # (DAYS,)
    n_cats_conv = int(sum(
        1 for ci in range(C)
        if (all_conv_day[:, ci] > 0).mean() >= 0.80
    ))

    return {
        "eta_override":     eta_override,
        "acc_day60":        round(float(np.nanmean(all_daily_acc[:, 59])), 4),
        "acc_day30":        round(float(np.nanmean(all_daily_acc[:, 29])), 4),
        "acc_day1":         round(float(np.nanmean(all_daily_acc[:, 0])),  4),
        "daily_acc_mean":   mean_acc.tolist(),
        "n_cats_converged": n_cats_conv,
    }


# ── Charts ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def save_fig(fig, stem: str):
    for ext in ("pdf", "png"):
        p = PAPER_FIGS / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


def chart_heatmap(results: dict, personas: list):
    """Heatmap: 9 personas (rows) × 6 η values (cols). Color = Day 60 acc."""
    persona_ids = [p["persona_id"] for p in personas]
    n_p         = len(persona_ids)
    n_eta       = len(ETA_OVERRIDE_VALS)

    matrix = np.zeros((n_p, n_eta))
    for ri, pid in enumerate(persona_ids):
        for ci, eta in enumerate(ETA_OVERRIDE_VALS):
            matrix[ri, ci] = results[pid][eta]["acc_day60"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=matrix.min() - 0.005,
                   vmax=matrix.max() + 0.005, aspect="auto")

    ax.set_xticks(range(n_eta))
    ax.set_xticklabels([f"η={v}" for v in ETA_OVERRIDE_VALS], fontsize=9)
    ax.set_yticks(range(n_p))

    # Row labels: persona + industry initial
    ind_init = {"Financial Services": "F", "Healthcare": "H", "Technology": "T"}
    ylabels = []
    for p in personas:
        ind = p["industry"]
        ylabels.append(f"{p['persona_id']} [{ind_init.get(ind,'?')}]")
    ax.set_yticklabels(ylabels, fontsize=9)

    # Annotate each cell; star best η per row
    best_cols = matrix.argmax(axis=1)
    for ri in range(n_p):
        for ci in range(n_eta):
            val = matrix[ri, ci]
            txt = f"{val:.1%}"
            col = "white" if val < (matrix.min() + 0.6 * (matrix.max() - matrix.min())) \
                  else "black"
            weight = "bold" if ci == best_cols[ri] else "normal"
            label  = f"[*]\n{txt}" if ci == best_cols[ri] else txt
            ax.text(ci, ri, label, ha="center", va="center",
                    fontsize=7, color=col, fontweight=weight)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Day 60 Accuracy", fontsize=9)

    ax.set_title("Day 60 Accuracy by eta_override (9 Personas, [*] = best per persona)")
    fig.tight_layout()
    save_fig(fig, "eta_override_sweep_heatmap")


def chart_curves(results: dict, personas: list):
    """6 curves: mean accuracy across personas over 60 days, one per η."""
    days = np.arange(1, DAYS + 1)
    persona_ids = [p["persona_id"] for p in personas]

    cmap   = plt.get_cmap("plasma", len(ETA_OVERRIDE_VALS))
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, eta in enumerate(ETA_OVERRIDE_VALS):
        # Mean daily accuracy across all 9 personas
        all_traj = np.array([results[pid][eta]["daily_acc_mean"]
                             for pid in persona_ids])   # (9, 60)
        mean_traj = np.nanmean(all_traj, axis=0)

        # 7-day rolling smooth
        smooth = np.convolve(mean_traj, np.ones(7) / 7, mode="same")
        smooth[:3]  = np.nanmean(mean_traj[:7])
        smooth[-3:] = np.nanmean(mean_traj[-7:])

        ax.plot(days, smooth, color=cmap(i), linewidth=1.8,
                label=f"η_override={eta}")

    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Accuracy (7-day rolling, 9 personas)")
    ax.set_title("Accuracy Trajectories by Override Learning Rate")
    ax.set_xlim(1, DAYS)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", ncol=2, fontsize=9, framealpha=0.85)
    fig.tight_layout()
    save_fig(fig, "eta_override_sweep_curves")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load personas
    personas_path = EXP_DIR / "personas_all.json"
    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)

    # Load domain config
    cfg          = load_domain_config("soc_product_v50")
    categories   = cfg["categories"]
    factor_names = cfg["factors"]
    mu_true      = np.array(cfg["mu"], dtype=float)
    gt_dists     = cfg["gt_distributions"]
    cat_to_idx   = {c: i for i, c in enumerate(categories)}
    A            = len(cfg["actions"])

    gt_dists_arr = np.array([gt_dists[c] for c in categories], dtype=float)
    gt_dists_arr = (gt_dists_arr.T / gt_dists_arr.sum(axis=1)).T

    # ── Run sweep ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("=== Q5: η_override SWEEP (9 personas × 6 values) ===")
    print("=" * 60)
    print()

    # results[pid][eta_float] = {acc_day60, daily_acc_mean, ...}
    results = {}

    total_runs = len(personas) * len(ETA_OVERRIDE_VALS)
    run_idx    = 0
    for persona in personas:
        pid = persona["persona_id"]
        results[pid] = {}
        for eta in ETA_OVERRIDE_VALS:
            run_idx += 1
            t1   = time.time()
            cond = simulate_persona_eta(
                persona, mu_true, categories, cat_to_idx,
                gt_dists_arr, factor_names, eta
            )
            elapsed = time.time() - t1
            results[pid][eta] = cond
            print(f"  [{run_idx:>2}/{total_runs}] {pid} η={eta:.3f} → "
                  f"acc@60={cond['acc_day60']:.1%}  ({elapsed:.1f}s)")

    print()

    # ── Table 1: Day 60 accuracy by η_override ────────────────────────────────
    print("Day 60 Accuracy by η_override:")
    print()
    hdr = (f"| {'Persona':<7} | "
           + " | ".join(f"{v:>6}" for v in ETA_OVERRIDE_VALS)
           + f" | {'Best η':>6} | {'Δ vs 0.05':>9} |")
    sep = "|" + "-" * 9 + "|" + ("-" * 8 + "|") * 6 + "-" * 8 + "|" + "-" * 11 + "|"
    print(hdr)
    print(sep)

    best_etas = {}
    for persona in personas:
        pid = persona["persona_id"]
        accs     = [results[pid][eta]["acc_day60"] for eta in ETA_OVERRIDE_VALS]
        best_eta = ETA_OVERRIDE_VALS[int(np.argmax(accs))]
        best_acc = max(accs)
        acc_05   = results[pid][0.05]["acc_day60"]
        delta    = (best_acc - acc_05) * 100
        best_etas[pid] = best_eta

        row = (f"| {pid:<7} | "
               + " | ".join(f"{a:>5.1%}" for a in accs)
               + f" | {best_eta:>6.3f} | {delta:>+8.2f}pp |")
        print(row)

    print()

    # ── Cross-persona optimal ─────────────────────────────────────────────────
    eta_vals = list(best_etas.values())
    print("Cross-persona optimal:")
    print(f"  Mean best η_override:   {np.mean(eta_vals):.4f}")
    print(f"  Median best η_override: {np.median(eta_vals):.4f}")
    from collections import Counter
    cnt = Counter(eta_vals)
    top_eta, top_n = cnt.most_common(1)[0]
    if top_n == len(personas):
        print(f"  All personas agree on: {top_eta}")
    else:
        print(f"  Most common best η: {top_eta}  ({top_n}/{len(personas)} personas)")

    print()

    # ── Theoretical prediction comparison ─────────────────────────────────────
    print("Theoretical prediction (from roadmap formula):")
    print(f"  Mixed team   (q̄≥0.70, σ²_q≈0.03): η* = {THEORY_MIXED}")
    print(f"  Junior-heavy (q̄<0.70, σ²_q≈0.05): η* = {THEORY_JUNIOR}")
    print()
    print(f"  {'Persona':<7} {'q̄':>6} {'σ²_q':>8} {'Theory η*':>10} {'Empirical':>10} {'Match?':>7}")
    print("  " + "-" * 48)
    matches = 0
    for persona in personas:
        pid    = persona["persona_id"]
        qbar   = persona_q_bar(persona["analyst_team"])
        qvar   = persona_q_var(persona["analyst_team"])
        theory = THEORY_MIXED if qbar >= 0.70 else THEORY_JUNIOR
        emp    = best_etas[pid]
        # "match" if within one step of closest ETA_OVERRIDE_VALS
        closest = min(ETA_OVERRIDE_VALS, key=lambda x: abs(x - theory))
        match  = (emp == closest or
                  abs(ETA_OVERRIDE_VALS.index(emp) -
                      ETA_OVERRIDE_VALS.index(closest)) <= 1)
        matches += int(match)
        flag = "✓" if match else "✗"
        print(f"  {pid:<7} {qbar:>6.3f} {qvar:>8.4f} {theory:>10.3f} "
              f"{emp:>10.3f} {flag:>7}")
    print(f"\n  Theory match rate: {matches}/{len(personas)}")

    print()

    # ── Improvement over baseline (η_override=0.05) ───────────────────────────
    print("IMPROVEMENT over no-asymmetry (η_override=0.05):")
    print(f"| {'Persona':<7} | {'Acc@0.05':>9} | {'Acc@best':>9} | {'Improvement':>12} |")
    print("|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 14 + "|")

    improvements = []
    for persona in personas:
        pid      = persona["persona_id"]
        acc_05   = results[pid][0.05]["acc_day60"]
        best_eta = best_etas[pid]
        acc_best = results[pid][best_eta]["acc_day60"]
        imp      = (acc_best - acc_05) * 100
        improvements.append((pid, imp))
        print(f"| {pid:<7} | {acc_05:>8.1%}  | {acc_best:>8.1%}  | "
              f"{imp:>+11.2f}pp |")

    imps = [x[1] for x in improvements]
    print()
    print(f"  Mean improvement:  {np.mean(imps):+.2f}pp")
    worst = min(improvements, key=lambda x: x[1])
    best  = max(improvements, key=lambda x: x[1])
    print(f"  Worst case: {worst[1]:+.2f}pp at persona {worst[0]}")
    print(f"  Best case:  {best[1]:+.2f}pp at persona {best[0]}")

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    # JSON-serializable: convert float keys to strings
    out_results = {}
    for pid, eta_map in results.items():
        out_results[pid] = {str(eta): v for eta, v in eta_map.items()}

    out = {
        "experiment":        "q5_eta_override_sweep",
        "n_seeds":           N_SEEDS,
        "days":              DAYS,
        "eta_confirm":       ETA_CONFIRM,
        "eta_override_vals": ETA_OVERRIDE_VALS,
        "eps_conv":          EPS_CONV,
        "tau":               TAU,
        "e0":                E0,
        "theory_mixed":      THEORY_MIXED,
        "theory_junior":     THEORY_JUNIOR,
        "best_etas":         best_etas,
        "results":           out_results,
    }
    out_path = RESULTS_DIR / "eta_override_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating charts...")
    chart_heatmap(results, personas)
    chart_curves(results, personas)
    print("Done. 2 charts × 2 formats = 4 files in paper_figures/.")


if __name__ == "__main__":
    main()
