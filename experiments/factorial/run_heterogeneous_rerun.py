"""
V-MV-KERNEL Heterogeneous Noise Re-run.

The original factorial used uniform σ_eff per cell, which makes all kernels
identical (1/σ² * ones normalised → uniform weights = L2).

This re-run tests L2 vs DiagonalKernel under HETEROGENEOUS per-factor noise:
  - SOC (d=6):  ratios=[0.7, 0.6, 0.5, 1.5, 1.0, 2.0]
  - S2P (d=8):  ratios=[1.0, 1.5, 0.7, 0.8, 0.6, 1.3, 1.8, 1.6]

Design:
  Variables: kernel (l2/diagonal/shrinkage_proxy), sigma_eff (0.08/0.15/0.22),
             noise_mode (uniform/heterogeneous), q_bar (0.60/0.80)
  Seeds: 10 per cell, 60 days, V=100 alerts/day, ρ=0 (independent factors)

Key question: does DiagonalKernel(weights=1/σ²) beat L2 under heterogeneous noise?

Usage:
    python experiments/factorial/run_heterogeneous_rerun.py --domain soc
    python experiments/factorial/run_heterogeneous_rerun.py --domain s2p
    python experiments/factorial/run_heterogeneous_rerun.py --domain both
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from gae.profile_scorer import ProfileScorer
from gae.kernels import L2Kernel, DiagonalKernel

# ── Constants ──────────────────────────────────────────────────────────────────
N_SEEDS      = 10
DAYS         = 60
ETA          = 0.05
ETA_NEG      = 0.05
ETA_OVERRIDE = 0.01
VERIFY_RATE  = 0.30
APD          = 100         # alerts per day (fixed for this re-run)

SIGMA_EFF_LEVELS = [0.08, 0.15, 0.22]
Q_BAR_LEVELS     = [0.60, 0.80]
KERNEL_TYPES     = ["l2", "diagonal", "shrinkage"]
NOISE_MODES      = ["uniform", "heterogeneous"]

# Heterogeneous noise ratios (mean ≈ 1.0 after rescaling)
SOC_HETERO_RATIOS = [0.7, 0.6, 0.5, 1.5, 1.0, 2.0]
S2P_HETERO_RATIOS = [1.0, 1.5, 0.7, 0.8, 0.6, 1.3, 1.8, 1.6]

FACTOR_NAMES_SOC = [
    "travel_match", "asset_criticality", "threat_intel_enrichment",
    "pattern_history", "time_anomaly", "device_trust",
]
FACTOR_NAMES_S2P = [
    "supplier_risk", "logistics_risk", "demand_risk", "inventory_risk",
    "regulatory_risk", "geopolitical_risk", "financial_risk", "environmental_risk",
]


# ── Noise array construction ───────────────────────────────────────────────────
def make_noise_array(sigma_eff: float, mode: str, hetero_ratios: list, d: int) -> np.ndarray:
    """
    mode='uniform'       → all factors get sigma_eff
    mode='heterogeneous' → factors scaled by hetero_ratios, rescaled so mean=sigma_eff
    """
    if mode == "uniform":
        return np.full(d, sigma_eff)
    else:
        ratios = np.array(hetero_ratios[:d], dtype=float)
        raw = sigma_eff * ratios
        # rescale so mean = sigma_eff
        raw = raw * (sigma_eff / raw.mean())
        return np.clip(raw, 0.03, 0.40)


# ── Kernel selection ───────────────────────────────────────────────────────────
def select_kernel(kernel_type: str, noise_array: np.ndarray):
    """
    Returns (kernel_object, actual_kernel_label).
    For l2: always L2Kernel regardless of noise shape.
    For diagonal/shrinkage: DiagonalKernel(weights=1/σ²) — with heterogeneous
      noise this differs meaningfully from L2.
    """
    if kernel_type == "l2":
        return L2Kernel(), "l2"

    weights = 1.0 / np.maximum(noise_array ** 2, 1e-4)
    weights /= weights.max()   # normalise to [0, 1]

    if kernel_type == "diagonal":
        return DiagonalKernel(weights), "diagonal"

    # shrinkage → proxy via DiagonalKernel (ShrinkageKernel ships at v6.5)
    return DiagonalKernel(weights), "shrinkage_proxy"


# ── Single cell simulation ─────────────────────────────────────────────────────
def run_one_cell(
    config: dict,
    noise_array: np.ndarray,
    kernel_type: str,
    q_bar: float,
    sigma_eff: float,
    noise_mode: str,
    cell_id: str,
) -> dict:
    mu_true    = config["mu"]            # (C, A, d)
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]

    C, A, d = mu_true.shape

    # GT distributions as array (C, A)
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        probs = np.array(gt_dists.get(cat, [1.0 / A] * A), dtype=float)
        probs = probs[:A]
        gt_arr[ci] = probs / probs.sum()

    # Category weights (uniform)
    cat_w = np.ones(C) / C

    # Analyst: one analyst per cell, q_bar quality, override_rate=0.25
    override_rate = 0.25
    q_analyst     = q_bar      # quality = fraction of overrides that are correct
    # Effective override rate with mild fatigue
    eff_override  = min(1.0, override_rate * 1.06)
    eff_quality   = max(0.4, q_bar * 0.96)

    kernel, actual_kernel = select_kernel(kernel_type, noise_array)

    all_day1  = []
    all_day30 = []
    all_day60 = []

    for si in range(N_SEEDS):
        rng = np.random.default_rng(42 + si)

        offset  = rng.uniform(-0.15, 0.15, mu_true.shape)
        mu_init = np.clip(mu_true + offset, 0, 1)

        scorer = ProfileScorer(
            mu_init.copy(), actions,
            scoring_kernel=kernel,
            eta_override=ETA_OVERRIDE,
        )
        scorer.eta     = ETA
        scorer.eta_neg = ETA_NEG

        daily_acc = np.zeros(DAYS)

        for day in range(DAYS):
            n_alerts = int(rng.poisson(APD))
            correct  = 0

            for _ in range(n_alerts):
                ci   = int(rng.choice(C, p=cat_w))
                a_gt = int(rng.choice(A, p=gt_arr[ci]))

                # Heterogeneous noise: each factor draws from its own σ
                z = rng.standard_normal(d)
                f = np.clip(mu_true[ci, a_gt] + noise_array * z, 0, 1)

                res    = scorer.score(f, ci)
                pred_a = res.action_index
                correct += int(pred_a == a_gt)

                # Analyst verification
                if rng.random() < VERIFY_RATE:
                    if rng.random() < eff_override:
                        gt_a = a_gt if rng.random() < eff_quality else int(
                            rng.choice([a for a in range(A) if a != a_gt])
                        )
                        scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                    else:
                        scorer.update(f, ci, pred_a, True)

            daily_acc[day] = correct / n_alerts if n_alerts > 0 else 0.0

        all_day1.append(daily_acc[0])
        all_day30.append(daily_acc[29] if DAYS >= 30 else daily_acc[-1])
        all_day60.append(daily_acc[-1])

    return {
        "cell_id":        cell_id,
        "kernel_type":    kernel_type,
        "actual_kernel":  actual_kernel,
        "noise_mode":     noise_mode,
        "sigma_eff":      round(float(sigma_eff), 4),
        "q_bar":          round(float(q_bar), 3),
        "noise_per_factor": [round(float(x), 4) for x in noise_array],
        "noise_min":      round(float(noise_array.min()), 4),
        "noise_max":      round(float(noise_array.max()), 4),
        "noise_range":    round(float(noise_array.max() - noise_array.min()), 4),
        "day1_accuracy":  round(float(np.mean(all_day1)),  4),
        "day30_accuracy": round(float(np.mean(all_day30)), 4),
        "day60_accuracy": round(float(np.mean(all_day60)), 4),
        "delta_d1_d60":   round(float(np.mean(all_day60)) - float(np.mean(all_day1)), 4),
    }


# ── Print helpers ──────────────────────────────────────────────────────────────
def print_summary(results: list, domain: str):
    print()
    print("=" * 80)
    print(f"HETEROGENEOUS NOISE RE-RUN: {domain.upper()} — {len(results)} cells")
    print("=" * 80)

    # Main table: kernel × noise_mode × sigma_eff
    print()
    print(f"  {'Cell ID':<35} {'Kernel':<18} {'Mode':<14} {'σ_eff':<7}"
          f" {'q̄':<6} {'Day1':>7} {'Day60':>7} {'Δ':>8}")
    print("  " + "-" * 110)
    for r in results:
        sign = "+" if r["delta_d1_d60"] >= 0 else ""
        print(f"  {r['cell_id']:<35} {r['actual_kernel']:<18} {r['noise_mode']:<14}"
              f" {r['sigma_eff']:<7.3f} {r['q_bar']:<6.2f}"
              f" {r['day1_accuracy']:>7.1%} {r['day60_accuracy']:>7.1%}"
              f" {sign}{r['delta_d1_d60']:>7.2%}")

    # Key comparison: kernel advantage under heterogeneous noise
    print()
    print("=" * 80)
    print("KERNEL ADVANTAGE: DiagonalKernel vs L2 (Δ Day60 accuracy)")
    print("=" * 80)
    print(f"  {'σ_eff':<8} {'q̄':<6} {'Mode':<14} {'L2 Day60':>10}"
          f" {'Diag Day60':>12} {'Diag-L2':>10} {'Shr-L2':>10}")
    print("  " + "-" * 80)

    for sig in SIGMA_EFF_LEVELS:
        for qb in Q_BAR_LEVELS:
            for mode in NOISE_MODES:
                def get_acc(kt):
                    cells = [r for r in results
                             if r["kernel_type"] == kt
                             and abs(r["sigma_eff"] - sig) < 0.01
                             and abs(r["q_bar"] - qb) < 0.05
                             and r["noise_mode"] == mode]
                    return float(np.mean([c["day60_accuracy"] for c in cells])) if cells else None

                l2_acc  = get_acc("l2")
                dia_acc = get_acc("diagonal")
                shr_acc = get_acc("shrinkage")

                if l2_acc is None:
                    continue

                dia_diff = (dia_acc - l2_acc) if dia_acc is not None else float("nan")
                shr_diff = (shr_acc - l2_acc) if shr_acc is not None else float("nan")

                dia_str = f"{dia_diff:+.2%}" if dia_acc is not None else "  N/A"
                shr_str = f"{shr_diff:+.2%}" if shr_acc is not None else "  N/A"
                dia_d60 = f"{dia_acc:.1%}" if dia_acc is not None else " N/A"

                print(f"  {sig:<8.3f} {qb:<6.2f} {mode:<14} {l2_acc:>9.1%}"
                      f" {dia_d60:>12} {dia_str:>10} {shr_str:>10}")

    # Finding
    print()
    print("=" * 80)
    print("FINDING")
    print("=" * 80)

    # Compare average kernel advantage in uniform vs heterogeneous mode
    for mode in NOISE_MODES:
        l2_cells   = [r for r in results if r["kernel_type"] == "l2"       and r["noise_mode"] == mode]
        dia_cells  = [r for r in results if r["kernel_type"] == "diagonal"  and r["noise_mode"] == mode]
        shr_cells  = [r for r in results if r["kernel_type"] == "shrinkage" and r["noise_mode"] == mode]
        if not l2_cells:
            continue
        l2_mean  = float(np.mean([r["day60_accuracy"] for r in l2_cells]))
        dia_mean = float(np.mean([r["day60_accuracy"] for r in dia_cells])) if dia_cells else None
        shr_mean = float(np.mean([r["day60_accuracy"] for r in shr_cells])) if shr_cells else None
        dia_lift = (dia_mean - l2_mean) if dia_mean is not None else None
        shr_lift = (shr_mean - l2_mean) if shr_mean is not None else None
        dia_str  = f"Diagonal lift = {dia_lift:+.3%}" if dia_lift is not None else ""
        shr_str  = f"  Shrinkage lift = {shr_lift:+.3%}" if shr_lift is not None else ""
        print(f"  {mode:<14}: L2={l2_mean:.1%}  {dia_str}{shr_str}")

    print()
    # Assess whether heterogeneous noise produces a real kernel advantage
    hetero_dia = [r for r in results if r["kernel_type"] == "diagonal" and r["noise_mode"] == "heterogeneous"]
    hetero_l2  = [r for r in results if r["kernel_type"] == "l2"       and r["noise_mode"] == "heterogeneous"]
    uniform_dia = [r for r in results if r["kernel_type"] == "diagonal" and r["noise_mode"] == "uniform"]
    uniform_l2  = [r for r in results if r["kernel_type"] == "l2"       and r["noise_mode"] == "uniform"]

    if hetero_dia and hetero_l2 and uniform_dia and uniform_l2:
        hetero_lift  = np.mean([r["day60_accuracy"] for r in hetero_dia]) - np.mean([r["day60_accuracy"] for r in hetero_l2])
        uniform_lift = np.mean([r["day60_accuracy"] for r in uniform_dia]) - np.mean([r["day60_accuracy"] for r in uniform_l2])
        incremental  = hetero_lift - uniform_lift

        print(f"  DiagonalKernel lift (uniform mode):      {uniform_lift:+.3%}")
        print(f"  DiagonalKernel lift (heterogeneous mode):{hetero_lift:+.3%}")
        print(f"  Incremental heterogeneous benefit:       {incremental:+.3%}")
        print()
        if incremental > 0.005:
            print("  VERDICT: DiagonalKernel shows MEANINGFUL benefit under heterogeneous noise.")
            print("  → Ship DiagonalKernel as default for mixed-scale factor domains (v6.5).")
        elif incremental > 0.001:
            print("  VERDICT: DiagonalKernel shows MARGINAL benefit under heterogeneous noise.")
            print("  → Useful for high-σ_range domains; L2 sufficient elsewhere.")
        else:
            print("  VERDICT: DiagonalKernel provides NO meaningful benefit even under heterogeneous noise.")
            print("  → L2 sufficient; DiagonalKernel deferred or dropped from v6.5.")
    print("=" * 80)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_domain(domain: str, output_dir: Path):
    if domain == "soc":
        config       = load_domain_config("soc_product_v50")
        factor_names = FACTOR_NAMES_SOC
        hetero_ratios = SOC_HETERO_RATIOS
    else:
        config       = load_domain_config("s2p_v03")
        factor_names = FACTOR_NAMES_S2P
        hetero_ratios = S2P_HETERO_RATIOS

    C, A, d = config["mu"].shape

    print()
    print("=" * 80)
    print(f"=== V-MV-KERNEL HETEROGENEOUS RE-RUN — {domain.upper()} ===")
    print("=" * 80)
    print(f"  Config:  {domain} — C={C} A={A} d={d}")
    print(f"  Seeds: {N_SEEDS}  |  Days: {DAYS}  |  V={APD}/day  |  ρ=0")
    print(f"  η_confirm={ETA}  η_override={ETA_OVERRIDE}")
    print(f"  Heterogeneous ratios: {hetero_ratios}")
    print(f"  NOTE: kernel_type='shrinkage' → DiagonalKernel proxy (ShrinkageKernel at v6.5)")

    # Build all cells
    cells = list(product(KERNEL_TYPES, SIGMA_EFF_LEVELS, Q_BAR_LEVELS, NOISE_MODES))
    print(f"  Cells: {len(cells)}")

    results  = []
    t_total  = time.time()

    for i, (kt, sig, qb, mode) in enumerate(cells):
        noise_array = make_noise_array(sig, mode, hetero_ratios, d)
        cell_id = f"{domain}-{kt[:4]}-s{int(sig*100):03d}-q{int(qb*100):02d}-{mode[:4]}"

        t0 = time.time()
        result = run_one_cell(
            config       = config,
            noise_array  = noise_array,
            kernel_type  = kt,
            q_bar        = qb,
            sigma_eff    = sig,
            noise_mode   = mode,
            cell_id      = cell_id,
        )
        elapsed = time.time() - t0

        sign = "+" if result["delta_d1_d60"] >= 0 else ""
        print(f"  [{i+1:>3}/{len(cells)}] {result['cell_id']:<35}"
              f"  kernel={result['actual_kernel']:<16}  Day60={result['day60_accuracy']:.1%}"
              f"  Δ={sign}{result['delta_d1_d60']:.2%}  ({elapsed:.1f}s)")
        results.append(result)

    total_time = time.time() - t_total
    print(f"\n  Completed {len(results)} cells in {total_time:.1f}s "
          f"({total_time/len(results):.1f}s/cell)")

    print_summary(results, domain)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"hetero_rerun_{domain}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="V-MV-KERNEL Heterogeneous Noise Re-run")
    parser.add_argument("--domain", required=True, choices=["soc", "s2p", "both"])
    parser.add_argument("--output", default="experiments/factorial/results",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    domains = ["soc", "s2p"] if args.domain == "both" else [args.domain]

    for domain in domains:
        run_domain(domain, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
