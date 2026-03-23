"""
EXP-A4-DIAGONAL: Does DiagonalKernel recover the A=4→A=5 action-space gap?

5 personas × 4 conditions × 15 seeds = 300 runs.

Conditions per persona:
  A: A=4, L2Kernel     (control — expected ~90.6%)
  B: A=4, DiagonalKernel
  C: A=5, L2Kernel     (control — expected ~80.4%)
  D: A=5, DiagonalKernel (THE QUESTION)

Protocol: 500 decisions per seed (300 warmup, 200 evaluation window).
Accuracy = mean over evaluation window (steady-state, not cold-start).

Personas:
  P1: FinServ SOC      (sigma_mean=0.08, uniform,      q_bar=0.82, V=200)
  P2: Healthcare SOC   (sigma_mean=0.22, hetero 2.5x,  q_bar=0.65, V=150)
  P3: Technology SOC   (sigma_mean=0.12, hetero 1.5x,  q_bar=0.78, V=300)
  P4: Startup SOC      (sigma_mean=0.18, hetero 3.0x,  q_bar=0.70, V=80)
  P5: Enterprise SOC   (sigma_mean=0.10, uniform,      q_bar=0.85, V=400)

Configs:
  A=4: soc_product_v50          (C=6, A=4, d=6)
  A=5: soc_product_v50_A5_backup (C=6, A=5, d=6)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel

# ── Experiment constants ───────────────────────────────────────────────────────
N_SEEDS   = 15
N_TOTAL   = 500
N_WARMUP  = 300
N_EVAL    = N_TOTAL - N_WARMUP   # 200 decisions for accuracy measurement

ETA          = 0.05
ETA_NEG      = 0.05
ETA_OVERRIDE = 0.01
VERIFY_RATE  = 0.30

OUTPUT_DIR = Path(__file__).parent / "results"

# ── Persona definitions ────────────────────────────────────────────────────────
# Noise ratios: rescaled so mean = sigma_mean, max/min ≈ target ratio
# Uniform personas: all factors get sigma_mean directly

def _hetero_noise(sigma_mean: float, ratios: list, d: int) -> np.ndarray:
    """Scale ratios so mean = sigma_mean, clip [0.03, 0.40]."""
    r = np.array(ratios[:d], dtype=float)
    raw = sigma_mean * r * (sigma_mean / (sigma_mean * r.mean()))
    return np.clip(raw, 0.03, 0.40)


# d=6 for all SOC personas
_D = 6

PERSONAS = [
    {
        "id":    "P1",
        "name":  "FinServ SOC",
        "noise": np.full(_D, 0.08),       # uniform
        "q_bar": 0.82,
        "apd":   200,
    },
    {
        "id":    "P2",
        "name":  "Healthcare SOC",
        # ratios [0.8,1.0,0.9,2.0,1.1,0.8] → max/min=2.5, rescaled to mean=0.22
        "noise": _hetero_noise(0.22, [0.8, 1.0, 0.9, 2.0, 1.1, 0.8], _D),
        "q_bar": 0.65,
        "apd":   150,
    },
    {
        "id":    "P3",
        "name":  "Technology SOC",
        # ratios [0.8,1.0,0.9,1.2,1.0,0.9] → max/min=1.5, rescaled to mean=0.12
        "noise": _hetero_noise(0.12, [0.8, 1.0, 0.9, 1.2, 1.0, 0.9], _D),
        "q_bar": 0.78,
        "apd":   300,
    },
    {
        "id":    "P4",
        "name":  "Startup SOC",
        # ratios [0.7,1.2,0.8,2.1,1.1,0.7] → max/min=3.0, rescaled to mean=0.18
        "noise": _hetero_noise(0.18, [0.7, 1.2, 0.8, 2.1, 1.1, 0.7], _D),
        "q_bar": 0.70,
        "apd":   80,
    },
    {
        "id":    "P5",
        "name":  "Enterprise SOC",
        "noise": np.full(_D, 0.10),       # uniform
        "q_bar": 0.85,
        "apd":   400,
    },
]


# ── Analyst team from q_bar ────────────────────────────────────────────────────
def make_team(q_bar: float) -> list:
    """
    Three-analyst team whose mean quality ≈ q_bar.
    Senior: q_bar+0.08, mid: q_bar, junior: q_bar-0.08.
    Override rates scale mildly with quality.
    """
    return [
        {"override_rate": 0.20, "override_quality": min(q_bar + 0.08, 0.98), "fatigue_factor": 0.15},
        {"override_rate": 0.27, "override_quality": q_bar,                    "fatigue_factor": 0.22},
        {"override_rate": 0.35, "override_quality": max(q_bar - 0.08, 0.40), "fatigue_factor": 0.32},
    ]


def analyst_eff(a: dict):
    ff  = a["fatigue_factor"]
    eo  = min(1.0, a["override_rate"] * (1 + ff * 0.3))
    eq  = max(0.4,  a["override_quality"] * (1 - ff * 0.2))
    return eo, eq


# ── Kernel builders ────────────────────────────────────────────────────────────
def diagonal_weights(noise: np.ndarray) -> np.ndarray:
    inv_var = 1.0 / np.maximum(noise ** 2, 1e-4)
    return inv_var / inv_var.max()


def build_kernel(kernel_type: str, noise: np.ndarray):
    if kernel_type == "l2":
        return L2Kernel()
    w = diagonal_weights(noise)
    return DiagonalKernel(w)


# ── Core cell simulation ───────────────────────────────────────────────────────
def run_cell(
    config:      dict,
    persona:     dict,
    kernel_type: str,
) -> dict:
    """
    Run N_SEEDS simulations of N_TOTAL decisions.
    Returns mean accuracy over the evaluation window (decisions 301-500)
    and per-seed standard deviation.
    """
    mu_true    = config["mu"]           # (C, A, d)
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]
    C, A, d    = mu_true.shape

    # GT array
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        p = np.array(gt_dists.get(cat, [1.0/A]*A), dtype=float)[:A]
        gt_arr[ci] = p / p.sum()

    cat_w      = np.ones(C) / C
    noise      = persona["noise"]
    apd        = persona["apd"]
    team       = make_team(persona["q_bar"])
    a_eff_list = [analyst_eff(a) for a in team]
    n_analysts = len(team)

    kernel = build_kernel(kernel_type, noise)

    eval_accs   = []   # one entry per seed: mean accuracy over eval window
    warmup_accs = []   # mean accuracy over warmup window

    for si in range(N_SEEDS):
        rng    = np.random.default_rng(42 + si)
        offset = rng.uniform(-0.15, 0.15, mu_true.shape)

        scorer = ProfileScorer(
            np.clip(mu_true + offset, 0, 1), actions,
            scoring_kernel=kernel,
            eta_override=ETA_OVERRIDE,
        )
        scorer.eta     = ETA
        scorer.eta_neg = ETA_NEG

        warmup_correct = 0
        warmup_total   = 0
        eval_correct   = 0
        eval_total     = 0

        # Simulate decisions one at a time (not day-based — count-based)
        decision_count = 0
        while decision_count < N_TOTAL:
            # Draw a burst of ~apd/10 decisions per iteration to avoid
            # per-decision Poisson overhead while keeping volume realistic
            burst = max(1, int(rng.poisson(apd // 10)))
            for _ in range(burst):
                if decision_count >= N_TOTAL:
                    break

                ci   = int(rng.choice(C, p=cat_w))
                a_gt = int(rng.choice(A, p=gt_arr[ci]))
                f    = np.clip(
                    mu_true[ci, a_gt] + rng.standard_normal(d) * noise,
                    0, 1
                )

                res    = scorer.score(f, ci)
                pred_a = res.action_index

                in_eval = decision_count >= N_WARMUP
                if in_eval:
                    eval_correct += int(pred_a == a_gt)
                    eval_total   += 1
                else:
                    warmup_correct += int(pred_a == a_gt)
                    warmup_total   += 1

                # Analyst verification
                if rng.random() < VERIFY_RATE:
                    ai_idx  = int(rng.integers(n_analysts))
                    eff_or, eff_q = a_eff_list[ai_idx]
                    if rng.random() < eff_or:
                        gt_a = a_gt if rng.random() < eff_q else int(
                            rng.choice([a for a in range(A) if a != a_gt])
                        )
                        scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                    else:
                        scorer.update(f, ci, pred_a, True)

                decision_count += 1

        eval_accs.append(eval_correct   / max(eval_total,   1))
        warmup_accs.append(warmup_correct / max(warmup_total, 1))

    mean_eval   = float(np.mean(eval_accs))
    std_eval    = float(np.std(eval_accs))
    mean_warmup = float(np.mean(warmup_accs))

    return {
        "mean_acc":    round(mean_eval,   4),
        "std_acc":     round(std_eval,    4),
        "warmup_acc":  round(mean_warmup, 4),
        "ci_half":     round(1.96 * std_eval / (N_SEEDS ** 0.5), 4),
        "per_seed":    [round(x, 4) for x in eval_accs],
    }


# ── Print helpers ──────────────────────────────────────────────────────────────
def fmt_cell(r: dict) -> str:
    return f"{r['mean_acc']:.1%} ±{r['ci_half']:.1%}"


def print_results(all_results: list):
    """Print the main comparison table."""

    print()
    print("=" * 80)
    print("EXP-A4-DIAGONAL: 20-cell results")
    print("=" * 80)
    print()
    print(f"  Protocol: {N_WARMUP} warmup + {N_EVAL} eval decisions, "
          f"{N_SEEDS} seeds, ±95% CI")
    print()

    # Header
    print(f"  {'Persona':<22} {'A=4 L2 (A)':>12} {'A=4 Diag (B)':>14}"
          f" {'A=5 L2 (C)':>12} {'A=5 Diag (D)':>14}"
          f" {'C→D recovery':>14} {'B-A':>8} {'D-C':>8}")
    print("  " + "-" * 110)

    for prow in all_results:
        rA = prow["A"]
        rB = prow["B"]
        rC = prow["C"]
        rD = prow["D"]

        # Gap C→D: how much of the A=4 L2 ceiling does D recover?
        a4_ceiling = rA["mean_acc"]
        gap_cd     = rD["mean_acc"] - rC["mean_acc"]
        a4_to_a5_gap = rA["mean_acc"] - rC["mean_acc"]   # baseline gap
        recovery_pp  = gap_cd  # pp recovered by diagonal vs L2 at A=5
        recovery_pct = (gap_cd / a4_to_a5_gap * 100) if a4_to_a5_gap > 0.001 else 0.0

        b_minus_a = rB["mean_acc"] - rA["mean_acc"]
        d_minus_c = rD["mean_acc"] - rC["mean_acc"]

        print(f"  {prow['persona_name']:<22}"
              f" {fmt_cell(rA):>12} {fmt_cell(rB):>14}"
              f" {fmt_cell(rC):>12} {fmt_cell(rD):>14}"
              f" {recovery_pp:>+12.2%}({recovery_pct:>+5.0f}%)"
              f" {b_minus_a:>+7.2%} {d_minus_c:>+7.2%}")

    # Aggregate
    print()
    print("  AGGREGATE:")
    mean_A = np.mean([r["A"]["mean_acc"] for r in all_results])
    mean_B = np.mean([r["B"]["mean_acc"] for r in all_results])
    mean_C = np.mean([r["C"]["mean_acc"] for r in all_results])
    mean_D = np.mean([r["D"]["mean_acc"] for r in all_results])

    print(f"    A=4 L2:       {mean_A:.1%}")
    print(f"    A=4 Diagonal: {mean_B:.1%}  (B-A = {mean_B-mean_A:+.2%})")
    print(f"    A=5 L2:       {mean_C:.1%}")
    print(f"    A=5 Diagonal: {mean_D:.1%}  (D-C = {mean_D-mean_C:+.2%})")
    print()

    a4_l2_baseline = mean_A
    a5_gap         = mean_A - mean_C    # gap from A=4 L2 to A=5 L2
    diag_recovery  = mean_D - mean_C    # Diagonal's uplift at A=5

    print(f"    A=4→A=5 gap (L2 baseline): {-a5_gap:+.2%}")
    print(f"    Diagonal recovery at A=5:  {diag_recovery:+.2%}")
    if a5_gap > 0.001:
        pct_recovered = diag_recovery / a5_gap * 100
        print(f"    Fraction recovered:        {pct_recovered:.0f}%  of A=4→A=5 gap")

    # Verdict
    print()
    print("  VERDICT:")
    if diag_recovery >= a5_gap * 0.80:
        print("    DiagonalKernel FULLY RECOVERS the A=4→A=5 gap (≥80%).")
        print("    A=5 deployment viable with DiagonalKernel at v6.0.")
        print("    Noise-ratio weighting compensates for action-space expansion.")
    elif diag_recovery >= a5_gap * 0.40:
        frac = diag_recovery / a5_gap
        print(f"    DiagonalKernel PARTIALLY recovers the gap ({frac:.0%}).")
        print("    A=5 deployment viable with monitoring; full recovery needs more data.")
    elif diag_recovery > 0.005:
        print("    DiagonalKernel adds marginal value at A=5 but gap persists.")
        print("    A=5 deployment requires noise remediation first.")
    else:
        print("    DiagonalKernel does NOT recover the A=4→A=5 gap.")
        print("    Action-space expansion effect is independent of kernel choice.")
        print("    Investigate action-prior calibration or centroid initialisation.")

    # Per-persona: does advantage scale with noise ratio?
    print()
    print("  PER-PERSONA D-C ADVANTAGE vs NOISE RATIO:")
    print(f"  {'Persona':<22} {'Ratio':>7} {'σ_mean':>7} {'B-A':>8} {'D-C':>8}"
          f" {'D-C > 1pp':>10}")
    print("  " + "-" * 72)
    for prow in all_results:
        noise = np.array(prow["noise"])
        ratio = float(noise.max() / max(noise.min(), 0.001))
        sig   = float(noise.mean())
        ba    = prow["B"]["mean_acc"] - prow["A"]["mean_acc"]
        dc    = prow["D"]["mean_acc"] - prow["C"]["mean_acc"]
        flag  = "YES" if dc > 0.01 else "no"
        print(f"  {prow['persona_name']:<22} {ratio:>6.1f}x {sig:>7.3f}"
              f" {ba:>+7.2%} {dc:>+7.2%} {flag:>10}")

    # Does advantage scale with ratio?
    ratios = []
    dc_vals = []
    for prow in all_results:
        noise = np.array(prow["noise"])
        ratios.append(float(noise.max() / max(noise.min(), 0.001)))
        dc_vals.append(prow["D"]["mean_acc"] - prow["C"]["mean_acc"])
    corr = float(np.corrcoef(ratios, dc_vals)[0, 1]) if len(ratios) > 2 else 0.0
    print()
    print(f"  Corr(noise_ratio, D-C advantage): {corr:.3f}")
    if corr > 0.6:
        print("  Diagonal benefit at A=5 scales with noise heterogeneity — as expected.")
    elif corr > 0.2:
        print("  Weak correlation: noise ratio partly predicts Diagonal benefit at A=5.")
    else:
        print("  No correlation: benefit is not driven by noise heterogeneity alone.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config_a4 = load_domain_config("soc_product_v50")
    config_a5 = load_domain_config("soc_product_v50_A5_backup")

    # Verify configs
    C4, A4, d4 = config_a4["mu"].shape
    C5, A5, d5 = config_a5["mu"].shape
    assert A4 == 4,  f"Expected A=4, got {A4}"
    assert A5 == 5,  f"Expected A=5, got {A5}"
    assert d4 == d5, f"d mismatch: {d4} vs {d5}"

    print()
    print("=" * 80)
    print("=== EXP-A4-DIAGONAL: DiagonalKernel vs L2 across A=4 and A=5 ===")
    print("=" * 80)
    print(f"  A=4 config: {C4} categories, {A4} actions, {d4} factors")
    print(f"  A=5 config: {C5} categories, {A5} actions, {d5} factors")
    print(f"  Protocol: {N_WARMUP} warmup + {N_EVAL} eval decisions, "
          f"{N_SEEDS} seeds per cell")
    print(f"  Personas: {len(PERSONAS)}, Conditions: 4, Total cells: "
          f"{len(PERSONAS) * 4}")

    all_results = []
    t_total     = time.time()
    cell_n      = 0
    n_cells     = len(PERSONAS) * 4

    for persona in PERSONAS:
        noise = persona["noise"]
        ratio = float(noise.max() / max(noise.min(), 0.001))

        print()
        print(f"  ── {persona['id']}: {persona['name']}"
              f"  σ_mean={noise.mean():.3f}  ratio={ratio:.1f}x"
              f"  q̄={persona['q_bar']}  V={persona['apd']} ──")

        prow = {
            "persona_id":   persona["id"],
            "persona_name": persona["name"],
            "sigma_mean":   round(float(noise.mean()), 4),
            "noise_ratio":  round(ratio, 2),
            "q_bar":        persona["q_bar"],
            "apd":          persona["apd"],
            "noise":        [round(float(x), 4) for x in noise],
        }

        for cond_label, config, ktype in [
            ("A", config_a4, "l2"),
            ("B", config_a4, "diagonal"),
            ("C", config_a5, "l2"),
            ("D", config_a5, "diagonal"),
        ]:
            cell_n += 1
            t0     = time.time()
            result = run_cell(config, persona, ktype)
            elapsed = time.time() - t0

            sign = "+" if (result["mean_acc"] - prow.get("_last_acc", result["mean_acc"])) >= 0 else ""
            print(f"    [{cell_n:>2}/{n_cells}] {persona['id']}-{cond_label}"
                  f"  A={'4' if cond_label in ('A','B') else '5'}"
                  f"  kernel={ktype:<8}"
                  f"  eval_acc={result['mean_acc']:.1%} ±{result['ci_half']:.1%}"
                  f"  warmup={result['warmup_acc']:.1%}"
                  f"  ({elapsed:.1f}s)")
            prow[cond_label] = result

        all_results.append(prow)

    total_time = time.time() - t_total
    print(f"\n  Completed {n_cells} cells in {total_time:.1f}s"
          f" ({total_time/n_cells:.1f}s/cell  {total_time/(n_cells*N_SEEDS):.1f}s/seed)")

    print_results(all_results)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results.json"
    save_obj = {
        "meta": {
            "n_seeds":   N_SEEDS,
            "n_warmup":  N_WARMUP,
            "n_eval":    N_EVAL,
            "eta":       ETA,
            "eta_override": ETA_OVERRIDE,
            "total_runtime_s": round(total_time, 1),
        },
        "results": [
            {k: v for k, v in r.items()}
            for r in all_results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")
    print()


if __name__ == "__main__":
    main()
