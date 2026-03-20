"""
TD-034: τ Recalibration — Synthetic Baseline
P30 gate: confirms τ=0.10 is optimal on A=4 synthetic alerts.
Sweep τ ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20}; ECE gate ≤ 0.05.
"""

import sys
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ── Config ────────────────────────────────────────────────────────────────────
TAU_VALUES  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
N_SEEDS     = 50
N_DECISIONS = 500
NOISE_RATE  = 0.10
ECE_GATE    = 0.05
SEEDS       = list(range(N_SEEDS))

EXP_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIGS  = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ── ECE ───────────────────────────────────────────────────────────────────────
def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — 10-bin uniform-width."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / len(confidences)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def compute_reliability_bins(confidences: np.ndarray, correctness: np.ndarray,
                              n_bins: int = 10) -> dict:
    """Return per-bin statistics for reliability diagram."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        n = int(mask.sum())
        bin_counts.append(n)
        if n == 0:
            bin_conf.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            bin_acc.append(float("nan"))
        else:
            bin_conf.append(float(confidences[mask].mean()))
            bin_acc.append(float(correctness[mask].mean()))
    return {"bin_confidence": bin_conf, "bin_accuracy": bin_acc, "bin_counts": bin_counts}


# ── Main sweep ────────────────────────────────────────────────────────────────
def run_sweep() -> dict:
    config = load_domain_config("soc_product_v50")

    tau_results   = {}
    # Collect raw (confidence, correct) at τ=0.10 for reliability diagram
    rel_confidences = []
    rel_correctness = []

    total_runs = len(TAU_VALUES) * N_SEEDS
    run_idx = 0

    for tau in TAU_VALUES:
        seed_ece  = []
        seed_acc  = []
        seed_conf = []

        for seed in SEEDS:
            run_idx += 1
            if run_idx % 50 == 1:
                print(f"  [{run_idx}/{total_runs}] τ={tau:.2f}, seed={seed}", flush=True)

            gen_kwargs = dict(config["generator_kwargs"])
            gen_kwargs["noise_rate"] = NOISE_RATE
            gen_kwargs["seed"]       = seed

            gen    = CategoryAlertGenerator(**gen_kwargs)
            scorer = ProfileScorer(config["mu"], tau=tau, eta=0.05, eta_neg=0.05, seed=seed)

            alerts = gen.generate(N_DECISIONS)

            confs  = np.empty(N_DECISIONS)
            corrects = np.empty(N_DECISIONS, dtype=float)

            for j, alert in enumerate(alerts):
                result      = scorer.score(alert.factors, alert.category_index)
                confs[j]    = result.confidence
                corrects[j] = float(result.action_index == alert.gt_action_index)

            seed_ece.append(compute_ece(confs, corrects))
            seed_acc.append(float(corrects.mean()))
            seed_conf.append(float(confs.mean()))

            if abs(tau - 0.10) < 1e-9:
                rel_confidences.append(confs)
                rel_correctness.append(corrects)

        tau_key = f"{tau:.2f}"
        gate    = bool(np.mean(seed_ece) <= ECE_GATE)
        tau_results[tau_key] = {
            "tau":       tau,
            "ece_mean":  float(np.mean(seed_ece)),
            "ece_std":   float(np.std(seed_ece)),
            "acc_mean":  float(np.mean(seed_acc)),
            "acc_std":   float(np.std(seed_acc)),
            "conf_mean": float(np.mean(seed_conf)),
            "conf_std":  float(np.std(seed_conf)),
            "gate_pass": gate,
        }

    # Optimal τ (lowest mean ECE)
    optimal_tau_str = min(tau_results, key=lambda k: tau_results[k]["ece_mean"])
    optimal_tau     = tau_results[optimal_tau_str]["tau"]
    optimal_ece     = tau_results[optimal_tau_str]["ece_mean"]
    gate_pass       = bool(optimal_ece <= ECE_GATE)

    if abs(optimal_tau - 0.10) < 1e-9:
        recommendation = f"Default confirmed. No recalibration needed."
    else:
        recommendation = f"Recalibration recommended: τ={optimal_tau:.2f} (ECE={optimal_ece:.4f})."

    # Reliability diagram data at τ=0.10
    all_conf  = np.concatenate(rel_confidences)
    all_corr  = np.concatenate(rel_correctness)
    rel_bins  = compute_reliability_bins(all_conf, all_corr)
    rel_bins["tau"]      = 0.10
    rel_bins["n_samples"] = int(len(all_conf))

    return {
        "config": {
            "tau_values":     TAU_VALUES,
            "n_seeds":        N_SEEDS,
            "n_decisions":    N_DECISIONS,
            "noise_rate":     NOISE_RATE,
            "domain":         "soc_product_v50",
            "gate_threshold": ECE_GATE,
        },
        "results":             tau_results,
        "optimal_tau":         optimal_tau,
        "optimal_ece":         optimal_ece,
        "gate_pass":           gate_pass,
        "recommendation":      recommendation,
        "reliability_diagram": rel_bins,
    }


# ── Print table ───────────────────────────────────────────────────────────────
def print_table(data: dict) -> None:
    print()
    print("=" * 60)
    print("=== TD-034: τ RECALIBRATION (A=4 synthetic) ===")
    print("=" * 60)
    print()
    header = (
        f"| {'τ':5s} | {'ECE mean':8s} | {'ECE std':7s} | "
        f"{'Accuracy':8s} | {'Confidence':10s} | {'Gate (ECE≤0.05)':15s} |"
    )
    sep = "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 17 + "|"
    print(header)
    print(sep)
    for tau_str, r in data["results"].items():
        gate_str = "PASS" if r["gate_pass"] else "FAIL"
        row = (
            f"| {r['tau']:5.2f} | {r['ece_mean']:8.4f} | {r['ece_std']:7.4f} | "
            f"{r['acc_mean']*100:7.2f}% | {r['conf_mean']:10.4f} | {gate_str:15s} |"
        )
        print(row)
    print()
    print(f"Optimal τ: {data['optimal_tau']:.2f} (lowest ECE = {data['optimal_ece']:.4f})")
    gate_verdict = "PASS" if data["gate_pass"] else "FAIL"
    print(f"Gate: {gate_verdict} (ECE ≤ {ECE_GATE})")
    print()
    print(data["recommendation"])
    print()
    print("REFERENCE VALUES (for real-alert pilot comparison):")
    print("  If real ECE at τ=0.10 exceeds synthetic ECE by >50%, recalibration needed.")
    tau_10 = data["results"]["0.10"]
    print(f"  Synthetic ECE at τ=0.10: {tau_10['ece_mean']:.4f} ±{tau_10['ece_std']:.4f}")
    print(f"  50%-above threshold: {tau_10['ece_mean'] * 1.5:.4f}")
    print()


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("TD-034: τ Recalibration sweep starting...")
    print(f"  τ values: {TAU_VALUES}")
    print(f"  Seeds: {N_SEEDS}  |  Decisions/seed: {N_DECISIONS}  |  Noise: {NOISE_RATE}")
    print()

    data = run_sweep()

    print_table(data)

    out_path = RESULTS_DIR / "td034_tau_synthetic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved → {out_path}")
    print()
    print("Run charts.py to generate paper figures.")
