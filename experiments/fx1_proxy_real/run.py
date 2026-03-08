"""
FX-1-PROXY-REAL: Realistic Factor Distribution Validation
experiments/fx1_proxy_real/run.py

QUESTION: How much does ProfileScorer accuracy and calibration degrade when
moving from centroidal synthetic data (tight Gaussians around GT centroids)
to realistic distributions (heavy tails, correlated factors, class overlap)?

Q1: Does Mahalanobis kernel reduce the degradation vs L2?
Q2: What fraction of decisions fall into each confidence band?

GATE-FX1:
  combined_acc (L2)   >= 0.80    (architecture still useful on realistic data)
  degradation (L2)    <= 0.20    (not catastrophic collapse)
  combined_ece (L2)   <= 0.10    (tau=0.1 still reasonably calibrated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.profile_scorer import ProfileScorer

from experiments.fx1_proxy_real.realistic_generator import (
    RealisticAlertGenerator,
    SOC_CATEGORIES,
    SOC_ACTIONS,
    SOC_FACTORS,
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

N_ALERTS  = 2000
N_SEEDS   = 10
TAU       = 0.1
SEEDS     = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

MODES   = ["centroidal", "heavy_tail", "correlated", "overlapping", "combined"]
KERNELS = ["l2", "mahalanobis"]

# Confidence band thresholds
BAND_AUTO_APPROVE  = 0.90   # >= 0.90  → auto-approve
BAND_HUMAN_REVIEW  = 0.60   # <  0.60  → human review
# 0.60 <= conf < 0.90 → agent_zone

EXP_DIR      = Path(__file__).parent
RESULTS_PATH = EXP_DIR / "results.json"
PAPER_DIR    = REPO_ROOT / "paper_figures"


# ---------------------------------------------------------------------------
# L2 scoring  (ProfileScorer)
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: list[float],
    correctness: list[int],
    n_bins:      int = 10,
) -> float:
    """Expected Calibration Error — standard equal-width 10-bin implementation."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(confidences)
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i < n_bins - 1:
            mask = [lo <= c < hi for c in confidences]
        else:
            mask = [lo <= c <= hi for c in confidences]
        count = sum(mask)
        if count == 0:
            continue
        bin_conf = float(np.mean([confidences[j] for j in range(n) if mask[j]]))
        bin_acc  = float(np.mean([correctness[j] for j in range(n) if mask[j]]))
        ece += (count / n) * abs(bin_conf - bin_acc)
    return float(ece)


def score_l2(alerts, gt_profiles: np.ndarray, tau: float):
    """
    L2 nearest-centroid scoring (ProfileScorer).
    Returns (accuracy, ece, confidences, correctness).
    """
    scorer      = ProfileScorer(gt_profiles.copy(), tau=tau)
    correct_cnt = 0
    confidences = []
    correctness = []

    for a in alerts:
        result     = scorer.score(a.factors, a.category_index)
        is_correct = int(result.action_index == a.gt_action_index)
        correct_cnt += is_correct
        confidences.append(float(result.probabilities[result.action_index]))
        correctness.append(is_correct)

    acc = correct_cnt / len(alerts)
    ece = compute_ece(confidences, correctness)
    return acc, ece, confidences, correctness


# ---------------------------------------------------------------------------
# Mahalanobis scoring
# ---------------------------------------------------------------------------

def build_mahalanobis_precision(
    alerts,
    gt_profiles: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    Estimate one global covariance matrix from all alert factor vectors.
    Returns precision matrix (inverse covariance), shape (n_factors, n_factors).

    Ridge regularization: Sigma += ridge * I  to ensure invertibility.
    """
    factors = np.array([a.factors for a in alerts], dtype=np.float64)  # (N, 6)
    mu      = factors.mean(axis=0)
    centered = factors - mu
    sigma   = (centered.T @ centered) / max(len(factors) - 1, 1)
    sigma  += ridge * np.eye(sigma.shape[0])
    return np.linalg.inv(sigma)


def score_mahalanobis(alerts, gt_profiles: np.ndarray, tau: float):
    """
    Mahalanobis nearest-centroid scoring.

    d²(f, mu_a) = (f - mu_a)^T Sigma^{-1} (f - mu_a)
    P(a | f, c) = softmax(-d²(f, mu[c,a,:]) / tau)

    Returns (accuracy, ece, confidences, correctness).
    """
    precision   = build_mahalanobis_precision(alerts, gt_profiles)
    correct_cnt = 0
    confidences = []
    correctness = []

    for a in alerts:
        f   = a.factors                          # shape (n_factors,)
        cat = a.category_index
        n_acts = gt_profiles.shape[1]

        dists = np.zeros(n_acts, dtype=np.float64)
        for act_idx in range(n_acts):
            diff       = f - gt_profiles[cat, act_idx, :]
            dists[act_idx] = float(diff @ precision @ diff)

        # softmax over negative scaled distances
        logits = -dists / tau
        logits -= logits.max()                   # numerical stability
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()

        pred_action = int(np.argmax(probs))
        is_correct  = int(pred_action == a.gt_action_index)
        correct_cnt += is_correct
        confidences.append(float(probs[pred_action]))
        correctness.append(is_correct)

    acc = correct_cnt / len(alerts)
    ece = compute_ece(confidences, correctness)
    return acc, ece, confidences, correctness


# ---------------------------------------------------------------------------
# Confidence band breakdown
# ---------------------------------------------------------------------------

def compute_confidence_bands(confidences: list[float], correctness: list[int]) -> dict:
    """
    Split decisions into three confidence bands and report:
      - fraction of decisions in band
      - accuracy within band
    Bands: auto_approve (>=0.90), agent_zone (0.60-0.90), human_review (<0.60)
    """
    n = len(confidences)
    bands: dict[str, dict] = {
        "auto_approve": {"lo": BAND_AUTO_APPROVE, "hi": 1.01},
        "agent_zone":   {"lo": BAND_HUMAN_REVIEW, "hi": BAND_AUTO_APPROVE},
        "human_review": {"lo": 0.0,               "hi": BAND_HUMAN_REVIEW},
    }
    result = {}
    for bname, brange in bands.items():
        lo, hi = brange["lo"], brange["hi"]
        idx    = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if bname == "auto_approve":
            idx = [i for i, c in enumerate(confidences) if c >= lo]
        count  = len(idx)
        frac   = count / n if n > 0 else 0.0
        acc    = float(np.mean([correctness[i] for i in idx])) if count > 0 else float("nan")
        result[bname] = {
            "count": count,
            "fraction": round(frac, 6),
            "accuracy": round(acc, 6) if not np.isnan(acc) else None,
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("\n=== FX-1-PROXY-REAL: Realistic Factor Distribution Validation ===")
    print(f"Settings: N={N_ALERTS}, seeds={N_SEEDS}, tau={TAU}")
    print(f"Modes: {MODES}")
    print(f"Kernels: {KERNELS}\n")

    # Sanity-check that taxonomy matches expectations
    assert SOC_CATEGORIES == [
        "travel_anomaly", "credential_access", "threat_intel_match",
        "insider_behavioral", "cloud_infrastructure",
    ], f"Category mismatch: {SOC_CATEGORIES}"
    assert SOC_ACTIONS == ["escalate", "investigate", "suppress", "monitor"], \
        f"Action mismatch: {SOC_ACTIONS}"

    # -----------------------------------------------------------------------
    # Main loop: 5 modes × 2 kernels × 10 seeds
    # -----------------------------------------------------------------------
    # results[mode][kernel] = {mean_acc, std_acc, mean_ece, std_ece, accs, eces}
    results: dict = {mode: {} for mode in MODES}

    for mode in MODES:
        for kernel in KERNELS:
            accs:       list[float] = []
            ece_scores: list[float] = []
            print(f"  [{mode}/{kernel}] running {N_SEEDS} seeds × {N_ALERTS} alerts ...")

            for seed in SEEDS:
                gen         = RealisticAlertGenerator(mode=mode, seed=seed)
                alerts      = gen.generate(N_ALERTS)
                gt_profiles = gen.get_profiles()

                if kernel == "l2":
                    acc, ece, _, _ = score_l2(alerts, gt_profiles, TAU)
                else:
                    acc, ece, _, _ = score_mahalanobis(alerts, gt_profiles, TAU)

                accs.append(acc)
                ece_scores.append(ece)

            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs))
            mean_ece = float(np.mean(ece_scores))
            std_ece  = float(np.std(ece_scores))

            results[mode][kernel] = {
                "mean_acc":  mean_acc,
                "std_acc":   std_acc,
                "mean_ece":  mean_ece,
                "std_ece":   std_ece,
                "accs":      [round(a, 6) for a in accs],
                "eces":      [round(e, 6) for e in ece_scores],
                "n_alerts":  N_ALERTS,
                "n_seeds":   N_SEEDS,
            }
            print(f"         acc={mean_acc*100:.2f}% ±{std_acc*100:.2f}pp  "
                  f"ECE={mean_ece:.4f} ±{std_ece:.4f}")

    # -----------------------------------------------------------------------
    # Confidence band breakdown — combined mode, L2, seed=42
    # -----------------------------------------------------------------------
    print("\n  Confidence band breakdown (combined mode, L2, seed=42) ...")
    gen_cb   = RealisticAlertGenerator(mode="combined", seed=42)
    alerts_cb = gen_cb.generate(N_ALERTS)
    gt_cb    = gen_cb.get_profiles()
    _, _, conf_l2, corr_l2 = score_l2(alerts_cb, gt_cb, TAU)
    bands_combined_l2 = compute_confidence_bands(conf_l2, corr_l2)

    print("  Confidence band breakdown (combined mode, Mahalanobis, seed=42) ...")
    _, _, conf_mh, corr_mh = score_mahalanobis(alerts_cb, gt_cb, TAU)
    bands_combined_mh = compute_confidence_bands(conf_mh, corr_mh)

    # -----------------------------------------------------------------------
    # Per-category breakdown — combined mode, L2, seed=42
    # -----------------------------------------------------------------------
    print("\n  Per-category breakdown (combined mode, L2, seed=42) ...")
    scorer_cb = ProfileScorer(gt_cb.copy(), tau=TAU)

    combined_per_cat_l2: dict[str, float] = {}
    for cat_idx, cat_name in enumerate(SOC_CATEGORIES):
        cat_alerts = [a for a in alerts_cb if a.category_index == cat_idx]
        if not cat_alerts:
            continue
        n_correct = sum(
            scorer_cb.score(a.factors, a.category_index).action_index == a.gt_action_index
            for a in cat_alerts
        )
        combined_per_cat_l2[cat_name] = n_correct / len(cat_alerts)

    # Centroidal per-category for comparison (using centroidal mode, seed=42)
    gen_cent   = RealisticAlertGenerator(mode="centroidal", seed=42)
    alerts_cent = gen_cent.generate(N_ALERTS)
    gt_cent    = gen_cent.get_profiles()
    scorer_ct  = ProfileScorer(gt_cent.copy(), tau=TAU)

    centroidal_per_cat: dict[str, float] = {}
    for cat_idx, cat_name in enumerate(SOC_CATEGORIES):
        cat_a = [a for a in alerts_cent if a.category_index == cat_idx]
        if not cat_a:
            continue
        n_correct = sum(
            scorer_ct.score(a.factors, a.category_index).action_index == a.gt_action_index
            for a in cat_a
        )
        centroidal_per_cat[cat_name] = n_correct / len(cat_a)

    # -----------------------------------------------------------------------
    # Print extended results table
    # -----------------------------------------------------------------------
    centroidal_l2_acc = results["centroidal"]["l2"]["mean_acc"]
    centroidal_l2_ece = results["centroidal"]["l2"]["mean_ece"]

    print(f"\n{'Mode':<15} {'Kernel':<12} {'Accuracy':>10} {'±std':>8} "
          f"{'ECE':>8} {'±std':>8} {'vs cent.L2':>12}")
    print("─" * 78)
    for mode in MODES:
        for kernel in KERNELS:
            r     = results[mode][kernel]
            delta = r["mean_acc"] - centroidal_l2_acc
            sign  = "+" if delta >= 0 else ""
            print(
                f"{mode:<15} {kernel:<12} {r['mean_acc']*100:>9.2f}%"
                f" {r['std_acc']*100:>7.2f}%"
                f" {r['mean_ece']:>8.4f}"
                f" {r['std_ece']:>7.4f}"
                f" {sign}{delta*100:>+10.2f}pp"
            )
        print()

    print(f"\nPer-category accuracy (combined mode, L2, seed=42):")
    for cat, acc in combined_per_cat_l2.items():
        centroidal_cat = centroidal_per_cat.get(cat, float("nan"))
        delta = (acc - centroidal_cat) * 100 if not np.isnan(centroidal_cat) else 0.0
        print(f"  {cat:<25} {acc*100:.2f}%  (centroidal: {centroidal_cat*100:.2f}%,"
              f" Δ={delta:+.2f}pp)")

    print(f"\nConfidence band breakdown (combined mode, seed=42):")
    print(f"  {'Band':<16} {'L2 frac':>9} {'L2 acc':>8}    {'Maha frac':>10} {'Maha acc':>9}")
    print("  " + "─" * 60)
    for bname in ("auto_approve", "agent_zone", "human_review"):
        bl  = bands_combined_l2[bname]
        bm  = bands_combined_mh[bname]
        l_acc = f"{bl['accuracy']*100:.2f}%" if bl['accuracy'] is not None else "  n/a  "
        m_acc = f"{bm['accuracy']*100:.2f}%" if bm['accuracy'] is not None else "  n/a  "
        print(f"  {bname:<16} {bl['fraction']*100:>8.1f}%  {l_acc:>7}    "
              f"{bm['fraction']*100:>9.1f}%  {m_acc:>8}")

    # -----------------------------------------------------------------------
    # Gate (L2, combined mode)
    # -----------------------------------------------------------------------
    combined_acc = results["combined"]["l2"]["mean_acc"]
    combined_ece = results["combined"]["l2"]["mean_ece"]
    degradation  = centroidal_l2_acc - combined_acc

    gate_acc  = combined_acc  >= 0.80
    gate_deg  = degradation   <= 0.20
    gate_ece  = combined_ece  <= 0.10
    gate_pass = gate_acc and gate_deg and gate_ece

    # Q1 answer: Mahalanobis delta
    maha_combined_acc  = results["combined"]["mahalanobis"]["mean_acc"]
    maha_cent_acc      = results["centroidal"]["mahalanobis"]["mean_acc"]
    maha_degradation   = maha_cent_acc - maha_combined_acc
    maha_delta_vs_l2   = maha_combined_acc - combined_acc

    print(f"\n=== GATE-FX1 ===")
    print(f"Centroidal baseline (L2):  {centroidal_l2_acc*100:.2f}% acc,  "
          f"ECE={centroidal_l2_ece:.4f}")
    print(f"Combined (L2):             {combined_acc*100:.2f}% acc,  "
          f"ECE={combined_ece:.4f}")
    print(f"Degradation (L2):          {degradation*100:+.2f}pp")
    print(f"\nQ1 — Mahalanobis vs L2 on combined mode:")
    print(f"  Maha combined acc:        {maha_combined_acc*100:.2f}%")
    print(f"  Maha degradation:         {maha_degradation*100:+.2f}pp")
    print(f"  Delta (Maha - L2):        {maha_delta_vs_l2*100:+.2f}pp")

    print(f"\nGate checks (L2 kernel):")
    print(f"  {'✓' if gate_acc else '✗'} combined accuracy >=80%:    {combined_acc*100:.2f}%")
    print(f"  {'✓' if gate_deg else '✗'} degradation <=20pp:         {degradation*100:.2f}pp")
    print(f"  {'✓' if gate_ece else '✗'} combined ECE <=0.10:        {combined_ece:.4f}")
    print(f"\nGATE-FX1: {'✅ PASS' if gate_pass else '❌ FAIL'}")

    if not gate_pass:
        print("\nInterpretation:")
        if not gate_acc:
            print("  Accuracy <80% — architecture may need kernel adaptation for real data.")
            print("  Consider Mahalanobis kernel (EXP-E1: wins on non-centroidal distributions).")
        if not gate_deg:
            print("  Degradation >20pp — centroidal synthetic data is too optimistic.")
            print("  Production claims need qualification: 'synthetic centroidal data'.")
        if not gate_ece:
            print("  ECE >0.10 — tau=0.1 recalibration needed for realistic distributions.")
            print("  TD-034: tau recalibration on real data (v5.5).")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    gate_checks = {
        "combined_acc_ge_80":  gate_acc,
        "degradation_le_20pp": gate_deg,
        "combined_ece_le_010": gate_ece,
    }

    # Flatten results for JSON: results_summary[mode][kernel] = metrics (no per-seed arrays)
    results_summary: dict = {}
    results_per_seed: dict = {}
    for mode in MODES:
        results_summary[mode]  = {}
        results_per_seed[mode] = {}
        for kernel in KERNELS:
            r = results[mode][kernel]
            results_summary[mode][kernel] = {
                k: v for k, v in r.items() if k not in ("accs", "eces")
            }
            results_per_seed[mode][kernel] = {
                "accs": r["accs"],
                "eces": r["eces"],
            }

    output = {
        "gate":                    "FX1",
        "pass":                    gate_pass,
        "n_alerts":                N_ALERTS,
        "n_seeds":                 N_SEEDS,
        "tau":                     TAU,
        "kernels":                 KERNELS,
        "results":                 results_summary,
        "results_per_seed":        results_per_seed,
        "combined_per_category_l2":   {k: round(v, 6) for k, v in combined_per_cat_l2.items()},
        "centroidal_per_category_l2": {k: round(v, 6) for k, v in centroidal_per_cat.items()},
        "degradation_pp_l2":       round(degradation * 100, 4),
        "mahalanobis_combined_acc": round(maha_combined_acc, 6),
        "mahalanobis_degradation_pp": round(maha_degradation * 100, 4),
        "mahalanobis_delta_vs_l2_pp": round(maha_delta_vs_l2 * 100, 4),
        "confidence_bands_combined_l2": bands_combined_l2,
        "confidence_bands_combined_mh": bands_combined_mh,
        "gate_checks":             gate_checks,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    try:
        from experiments.fx1_proxy_real.charts import generate_charts
        generate_charts(
            results=results,
            combined_per_cat=combined_per_cat_l2,
            centroidal_per_cat=centroidal_per_cat,
            bands_l2=bands_combined_l2,
            bands_mh=bands_combined_mh,
            paper_dir=str(PAPER_DIR),
        )
    except Exception as exc:
        import traceback
        print(f"\nWarning: chart generation failed: {exc}")
        traceback.print_exc()

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
