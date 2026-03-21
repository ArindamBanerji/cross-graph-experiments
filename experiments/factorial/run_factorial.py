"""
V-MV-KERNEL Factorial: 216 cells across 5 variables × 2 domains.

Variables: kernel (L2/diagonal/shrinkage), sigma_eff (0.08/0.15/0.22),
           q_bar (0.60/0.80), V (50/200), rho_max (0/0.3/0.6)

For each cell:
  1. Load persona from factorial JSON
  2. Select kernel based on kernel_type field
  3. Run PROD-5 (60-day convergence) with that kernel
  4. Record: accuracy (Day1/30/60), convergence, conservation signal

Output: one row per cell with all metrics.

NOTE on ShrinkageKernel proxy:
  ShrinkageKernel is not yet implemented in gae (ships at v6.5).
  For kernel_type='shrinkage', DiagonalKernel(weights=1/σ²) is used as proxy.
  This approximates the diagonal component of the shrinkage estimator without
  the off-diagonal regularisation. Cells with kernel='shrinkage' are tagged
  actual_kernel='shrinkage_proxy' in results.

Usage:
    python experiments/factorial/run_factorial.py \\
        --input  experiments/factorial/factorial_soc_streams.json \\
        --output experiments/factorial/results/ \\
        --domain soc

    python experiments/factorial/run_factorial.py \\
        --input  experiments/factorial/factorial_s2p_streams.json \\
        --output experiments/factorial/results/ \\
        --domain s2p
"""

import argparse
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
from gae.covariance import CovarianceEstimator

# ── Constants ─────────────────────────────────────────────────────────────────
N_SEEDS    = 10
DAYS       = 60
ETA        = 0.05
ETA_NEG    = 0.05
ETA_OVERRIDE = 0.01
TAU        = 0.10
CONV_EPS   = 0.10     # convergence threshold (max-action L2 error)
VERIFY_RATE = 0.30    # analyst verification probability

FACTOR_NAMES_SOC = [
    "travel_match", "asset_criticality", "threat_intel_enrichment",
    "pattern_history", "time_anomaly", "device_trust",
]
FACTOR_NAMES_S2P = [
    "supplier_risk", "logistics_risk", "demand_risk", "inventory_risk",
    "regulatory_risk", "geopolitical_risk", "financial_risk", "environmental_risk",
]


# ── Kernel selection ──────────────────────────────────────────────────────────
def select_kernel(kernel_type: str, sigma_per_factor: np.ndarray):
    """
    Dispatch to correct kernel. ShrinkageKernel → DiagonalKernel proxy.

    Returns (kernel_object, actual_kernel_label).
    """
    if kernel_type == "l2":
        return L2Kernel(), "l2"

    weights = 1.0 / np.maximum(sigma_per_factor ** 2, 1e-4)
    weights /= weights.max()   # normalise to [0, 1]

    if kernel_type == "diagonal":
        return DiagonalKernel(weights), "diagonal"

    if kernel_type == "shrinkage":
        # PROXY until ShrinkageKernel ships at v6.5
        # Approximates diagonal component of Ledoit-Wolf estimator.
        # Off-diagonal regularisation is absent — results are conservative
        # (shrinkage benefit will be larger with real kernel).
        return DiagonalKernel(weights), "shrinkage_proxy"

    # Unknown type — fall back to L2 and log
    return L2Kernel(), f"l2_fallback_{kernel_type}"


# ── Persona utilities ─────────────────────────────────────────────────────────
def get_factor_noise(persona: dict, factor_names: list) -> np.ndarray:
    """
    Extract per-factor noise from persona.
    Handles both flat dict {'factor': {'base_noise': x}} and nested
    {'per_factor': {'factor': {'base_noise': x}}} layouts.
    """
    profile = persona.get("factor_noise_profile", {})
    # Normalise to flat dict
    if "per_factor" in profile:
        profile = profile["per_factor"]

    noise = np.zeros(len(factor_names))
    for j, fname in enumerate(factor_names):
        entry = profile.get(fname, {})
        noise[j] = entry.get("base_noise", 0.15) if isinstance(entry, dict) else float(entry)
    return noise


def get_category_weights(persona: dict, categories: list) -> np.ndarray:
    cat_dist = persona.get("category_distribution", {})
    w = np.array([cat_dist.get(c, 0.0) for c in categories], dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(categories)) / len(categories)


def get_analyst_params(persona: dict):
    """
    Returns list of (eff_override_rate, eff_quality) per analyst.
    Mirrors run_harness.precompute_analyst_params fatigue scaling.
    """
    params = []
    for a in persona.get("analyst_team", []):
        ff  = a.get("fatigue_factor", 0.2)
        eo  = min(1.0, a["override_rate"] * (1 + ff * 0.3))
        eq  = max(0.4, a["override_quality"] * (1 - ff * 0.2))
        params.append((eo, eq))
    return params if params else [(0.25, 0.75)]


def build_noise_chol(noise: np.ndarray, rho_max: float,
                     persona: dict) -> np.ndarray:
    """
    Build Cholesky factor of the noise covariance matrix.

    rho_max=0  → diagonal (independent factors)
    rho_max>0  → use persona's 'correlation_matrix' if present,
                 else constant off-diagonal = rho_max.

    Returns L such that L @ z (z ~ N(0,I)) gives correlated noise.
    """
    d = len(noise)
    if rho_max == 0.0:
        return np.diag(noise)   # diagonal Cholesky = diag(σ)

    corr_raw = persona.get("correlation_matrix", None)
    if isinstance(corr_raw, dict):
        # S2P format: {"regime_a": [[...]], "domain_ordering": [...], ...}
        corr_raw = corr_raw.get("regime_a", corr_raw.get("matrix", None))
    if corr_raw is not None:
        corr = np.array(corr_raw, dtype=float)
    else:
        corr = np.full((d, d), rho_max)
        np.fill_diagonal(corr, 1.0)

    # Ensure positive-definite
    corr = (corr + corr.T) / 2.0
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-8:
        corr += (abs(eigvals.min()) + 1e-6) * np.eye(d)

    # Scale by σ: Σ = diag(σ) @ corr @ diag(σ)
    cov = np.outer(noise, noise) * corr
    return np.linalg.cholesky(cov)


# ── PROD-5 simulation for one cell ────────────────────────────────────────────
def run_one_cell(persona: dict, config: dict, factor_names: list) -> dict:
    mu_true    = config["mu"]                    # (C, A, d)
    categories = config["categories"]
    actions    = config["actions"]
    gt_dists   = config["gt_distributions"]      # dict: cat → [p_a0, ...]

    C, A, d = mu_true.shape
    fv  = persona["factorial_variables"]
    rho_max    = float(fv.get("rho_max", 0.0))
    noise      = get_factor_noise(persona, factor_names)
    cat_w      = get_category_weights(persona, categories)
    a_params   = get_analyst_params(persona)
    n_analysts = len(a_params)
    apd        = persona.get("alerts_per_day", 200)
    kernel_type = persona.get("kernel_type", "l2")

    # Build kernel
    kernel, actual_kernel = select_kernel(kernel_type, noise)

    # Noise Cholesky (reused across seeds)
    chol = build_noise_chol(noise, rho_max, persona)

    # GT distributions as array (C, A)
    gt_arr = np.zeros((C, A))
    for ci, cat in enumerate(categories):
        probs = np.array(gt_dists.get(cat, [1.0 / A] * A), dtype=float)
        probs = probs[:A]
        gt_arr[ci] = probs / probs.sum()

    # Covariance estimator (collects data per cell, not used for scoring)
    cov_est = CovarianceEstimator(d=d, half_life_decisions=300)

    all_day1  = []
    all_day30 = []
    all_day60 = []
    all_conv  = {cat: [] for cat in categories}
    all_daily_sig = []    # daily α·q·V for conservation

    for si in range(N_SEEDS):
        rng = np.random.default_rng(42 + si)

        # Cold-start offset ±0.15
        offset = rng.uniform(-0.15, 0.15, mu_true.shape)
        mu_init = np.clip(mu_true + offset, 0, 1)

        scorer = ProfileScorer(
            mu_init.copy(), actions,
            scoring_kernel=kernel,
            eta_override=ETA_OVERRIDE,
        )
        scorer.eta     = ETA
        scorer.eta_neg = ETA_NEG

        daily_acc  = np.zeros(DAYS)
        daily_sig  = np.zeros(DAYS)
        conv_day   = {cat: None for cat in categories}

        for day in range(DAYS):
            n_alerts  = int(rng.poisson(apd))
            correct   = 0
            day_ov = day_ovc = 0

            for _ in range(n_alerts):
                ci     = int(rng.choice(C, p=cat_w))
                cat    = categories[ci]
                a_gt   = int(rng.choice(A, p=gt_arr[ci]))

                # Generate factor vector with optional correlation
                if rho_max == 0.0:
                    z = rng.standard_normal(d)
                    f = np.clip(mu_true[ci, a_gt] + noise * z, 0, 1)
                else:
                    z = rng.standard_normal(d)
                    f = np.clip(mu_true[ci, a_gt] + chol @ z, 0, 1)

                cov_est.update(f)

                res = scorer.score(f, ci)
                pred_a = res.action_index
                correct += int(pred_a == a_gt)

                # Analyst verification
                if rng.random() < VERIFY_RATE:
                    ai_idx = rng.integers(n_analysts)
                    eff_o, eff_q = a_params[ai_idx]

                    if rng.random() < eff_o:
                        gt_a = a_gt if rng.random() < eff_q else int(
                            rng.choice([a for a in range(A) if a != a_gt])
                        )
                        scorer.update(f, ci, pred_a, False, gt_action_index=gt_a)
                        day_ov  += 1
                        day_ovc += int(gt_a == a_gt)
                    else:
                        scorer.update(f, ci, pred_a, True)

            daily_acc[day] = correct / n_alerts if n_alerts > 0 else 0.0

            # Conservation signal: α·q·V (correct overrides per day)
            alpha = day_ov / n_alerts if n_alerts > 0 else 0.0
            q_eff = day_ovc / day_ov if day_ov > 0 else 0.0
            daily_sig[day] = alpha * q_eff * n_alerts

            # Convergence check: max-action L2 error per category
            for ci2, cat in enumerate(categories):
                if conv_day[cat] is None:
                    err = max(
                        np.linalg.norm(scorer.centroids[ci2, a] - mu_true[ci2, a])
                        for a in range(A)
                    )
                    if err < CONV_EPS:
                        conv_day[cat] = day + 1  # 1-indexed

        all_day1.append(daily_acc[0])
        all_day30.append(daily_acc[29] if DAYS >= 30 else daily_acc[-1])
        all_day60.append(daily_acc[-1])
        all_daily_sig.append(daily_sig)
        for cat in categories:
            all_conv[cat].append(conv_day[cat] if conv_day[cat] is not None else DAYS + 1)

    snap = cov_est.get_snapshot()

    # Convergence summary
    conv_summary = {}
    n_converged  = 0
    for cat in categories:
        days_list = all_conv[cat]
        pct_conv  = float(np.mean([v <= DAYS for v in days_list]))
        mean_day  = float(np.mean([v for v in days_list if v <= DAYS])) if any(v <= DAYS for v in days_list) else None
        conv_summary[cat] = {
            "mean_conv_day":  round(mean_day, 1) if mean_day else None,
            "pct_converged":  round(pct_conv, 3),
        }
        if pct_conv >= 0.5:
            n_converged += 1

    # Conservation stats
    sig_arr  = np.stack(all_daily_sig)          # (N_SEEDS, DAYS)
    mean_sig = float(np.mean(sig_arr))
    min_sig  = float(np.min(sig_arr.mean(axis=1)))

    return {
        "cell_id":       persona.get("cell_id", persona.get("persona_id", "?")),
        "domain":        persona.get("domain", "?"),
        "kernel_type":   kernel_type,
        "actual_kernel": actual_kernel,
        "sigma_eff":     round(float(np.mean(noise)), 4),
        "q_bar":         round(float(np.mean([a.get("override_quality", 0.75) for a in persona.get("analyst_team", [])])), 3),
        "volume":        apd,
        "rho_max":       rho_max,
        "day1_accuracy": round(float(np.mean(all_day1)),  4),
        "day30_accuracy":round(float(np.mean(all_day30)), 4),
        "day60_accuracy":round(float(np.mean(all_day60)), 4),
        "delta_d1_d60":  round(float(np.mean(all_day60)) - float(np.mean(all_day1)), 4),
        "cats_converged":n_converged,
        "cats_total":    C,
        "convergence":   conv_summary,
        "conservation_mean": round(mean_sig, 3),
        "conservation_min":  round(min_sig,  3),
        "cov_n_samples":     snap.n_samples,
        "cov_shrinkage_lambda": round(float(snap.shrinkage_lambda), 4),
        "cov_condition_number": round(float(snap.condition_number),  2),
        "gate_stats":    scorer.update_gate_stats,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_summary(results: list, domain: str):
    cats_total = results[0]["cats_total"] if results else 0

    print()
    print("=" * 80)
    print(f"FACTORIAL SUMMARY: {domain.upper()} — {len(results)} cells")
    print("=" * 80)

    # Per-kernel block
    kernel_order = ["l2", "diagonal", "shrinkage"]
    for kt in kernel_order:
        cells = [r for r in results if r["kernel_type"] == kt]
        if not cells:
            continue
        proxy = cells[0]["actual_kernel"] if kt == "shrinkage" else kt
        accs   = [c["day60_accuracy"] for c in cells]
        deltas = [c["delta_d1_d60"] for c in cells]
        conv   = [c["cats_converged"] for c in cells]
        degrade = sum(1 for d in deltas if d < -0.01)
        proxy_note = "  (→ shrinkage_proxy)" if kt == "shrinkage" else ""
        print(f"\n  Kernel: {kt}{proxy_note}  ({len(cells)} cells)")
        print(f"    Day60 accuracy : {np.mean(accs):.1%} ± {np.std(accs):.2%}")
        print(f"    Mean Δ(D60-D1) : {np.mean(deltas):+.2%}")
        print(f"    Degrading cells: {degrade}/{len(cells)}")
        print(f"    Cats converged : {np.mean(conv):.1f}/{cats_total} mean")

    # Main effects
    print()
    print("=" * 80)
    print("MAIN EFFECTS")
    print("=" * 80)

    def _main_effect(var_key, label, rounding=None):
        values = sorted(set(
            round(r[var_key], rounding) if rounding is not None else r[var_key]
            for r in results
        ))
        print(f"\n  {label}:")
        for val in values:
            cells = [r for r in results if (
                round(r[var_key], rounding) == val if rounding is not None
                else r[var_key] == val
            )]
            mean_acc   = float(np.mean([c["day60_accuracy"] for c in cells]))
            mean_delta = float(np.mean([c["delta_d1_d60"] for c in cells]))
            n_degrade  = sum(1 for c in cells if c["delta_d1_d60"] < -0.01)
            print(f"    {str(val):<8}: Day60={mean_acc:.1%}  Δ={mean_delta:+.2%}"
                  f"  degrade={n_degrade}/{len(cells)}")

    _main_effect("kernel_type", "Kernel")
    _main_effect("sigma_eff",   "σ_eff",  rounding=3)
    _main_effect("q_bar",       "q̄",     rounding=2)
    _main_effect("volume",      "Volume")
    _main_effect("rho_max",     "ρ_max",  rounding=2)

    # Key interaction: kernel × ρ
    print()
    print("=" * 80)
    print("KEY INTERACTION: Kernel × ρ_max")
    print("=" * 80)
    print(f"  {'Kernel':<14} {'ρ=0.0':>10} {'ρ=0.3':>10} {'ρ=0.6':>10} {'Δ(ρ0→ρ0.6)':>13}")

    for kt in kernel_order:
        row = []
        for rho in [0.0, 0.3, 0.6]:
            cells = [r for r in results
                     if r["kernel_type"] == kt and abs(r["rho_max"] - rho) < 0.05]
            row.append(float(np.mean([c["day60_accuracy"] for c in cells])) if cells else None)
        if all(v is not None for v in row):
            drho = row[2] - row[0]
            print(f"  {kt:<14} {row[0]:>9.1%} {row[1]:>9.1%} {row[2]:>9.1%} {drho:>+12.2%}")

    print()
    l2_drho   = _rho_delta(results, "l2")
    dia_drho  = _rho_delta(results, "diagonal")
    shr_drho  = _rho_delta(results, "shrinkage")
    if l2_drho is not None and dia_drho is not None:
        diff = dia_drho - l2_drho
        if diff > 0.005:
            print("  FINDING: Diagonal/shrinkage degrade MORE at high ρ than L2.")
            print("  → Weight normalisation is penalised under correlated noise.")
        elif diff < -0.005:
            print("  FINDING: Diagonal/shrinkage handle correlated noise better than L2.")
            print("  → Weighted kernels provide ρ-robustness → ship at v6.5.")
        else:
            print("  FINDING: No meaningful difference across kernels at ρ=0.6.")
            print("  → L2 sufficient; DiagonalKernel deferred.")
    print("=" * 80)


def _rho_delta(results: list, kernel_type: str) -> float | None:
    lo = [r["day60_accuracy"] for r in results
          if r["kernel_type"] == kernel_type and abs(r["rho_max"] - 0.0) < 0.05]
    hi = [r["day60_accuracy"] for r in results
          if r["kernel_type"] == kernel_type and abs(r["rho_max"] - 0.6) < 0.05]
    if lo and hi:
        return float(np.mean(hi)) - float(np.mean(lo))
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V-MV-KERNEL Factorial Harness")
    parser.add_argument("--input",  required=True, help="Factorial JSON persona file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--domain", required=True, choices=["soc", "s2p"])
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.domain == "soc":
        config       = load_domain_config("soc_product_v50")
        factor_names = FACTOR_NAMES_SOC
    else:
        config       = load_domain_config("s2p_v03")
        factor_names = FACTOR_NAMES_S2P

    C = len(config["categories"])
    A = len(config["actions"])
    d = config["mu"].shape[2]

    with open(input_path, encoding="utf-8") as f:
        personas = json.load(f)

    print()
    print("=" * 80)
    print(f"=== V-MV-KERNEL FACTORIAL — {args.domain.upper()} ===")
    print("=" * 80)
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Config:  {args.domain} — C={C} A={A} d={d}")
    print(f"  Cells:   {len(personas)}  |  Seeds: {N_SEEDS}  |  Days: {DAYS}")
    print(f"  η_confirm={ETA}  η_override={ETA_OVERRIDE}  τ={TAU}")
    print(f"  NOTE: kernel_type='shrinkage' → DiagonalKernel proxy (ShrinkageKernel at v6.5)")

    results   = []
    t_total   = time.time()

    for i, persona in enumerate(personas):
        t0     = time.time()
        result = run_one_cell(persona, config, factor_names)
        elapsed = time.time() - t0

        kt     = result["actual_kernel"]
        d60    = result["day60_accuracy"]
        delta  = result["delta_d1_d60"]
        conv   = result["cats_converged"]
        sign   = "+" if delta >= 0 else ""
        print(f"  [{i+1:>3}/{len(personas)}] {result['cell_id']:<34}"
              f"  kernel={kt:<16}  Day60={d60:.1%}  Δ={sign}{delta:.2%}"
              f"  conv={conv}/{C}  ({elapsed:.1f}s)")
        results.append(result)

    total_time = time.time() - t_total

    print(f"\n  Completed {len(results)} cells in {total_time:.1f}s "
          f"({total_time/len(results):.1f}s/cell)")

    print_summary(results, args.domain)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"factorial_{args.domain}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
