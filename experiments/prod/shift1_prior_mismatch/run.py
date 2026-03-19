"""
SHIFT-1 — Learning Capacity Under Prior Mismatch
Breaks generator circularity: μ_true = μ₀ + δ

When δ=0 (circular baseline): μ_true = μ₀, so expected centroid update ≈ 0.
When δ>0: true profiles are shifted away from the expert prior μ₀.

LEARNING condition:  scorer starts at μ₀, centroids update during warmup.
FROZEN condition:    scorer stays at μ₀, no updates.

If learning_lift > 0 at δ > 0 but ~0 at δ=0:
  → Circularity was masking real learning capacity. Production will improve.
If learning_lift ~0 at ALL δ values:
  → Architecture is the ceiling. Learning cannot help even with signal.

Regime: centroidal synthetic. Ontology: SOC product v5.0+refer (C=6, A=5, d=6).
η_neg=0.05, τ=0.1, noise=0.10. Only δ and warmup length vary.
"""
from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

import numpy as np

_REPO_ROOT  = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DELTA_MAGNITUDES = [0.0, 0.05, 0.10, 0.15, 0.20]
WARMUP_LEVELS    = [200, 500, 1000]
N_SEEDS          = 30
N_EVAL           = 500
TAU              = 0.1
ETA              = 0.05
ETA_NEG          = 0.05        # η_neg=1.0 is FORBIDDEN
NOISE_RATE       = 0.10
ACCURACY_GATE    = 0.85
THRESHOLD_SWEEP  = list(np.arange(0.50, 1.00, 0.02))   # coarser sweep
MARGIN_THRESHOLDS = [0.3, 0.5, 0.7, 0.8]
RANDOM_SEED_BASE = 42
DOMAIN_CONFIG    = "soc_product_v50"

# ---------------------------------------------------------------------------
# Load domain config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]

print(f"Domain config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"Categories: {CATEGORIES}")
print(f"η_neg={ETA_NEG}, τ={TAU}, η={ETA}, noise={NOISE_RATE}")
print(f"Seeds: {N_SEEDS}, N_eval: {N_EVAL}")
print(f"Delta magnitudes: {DELTA_MAGNITUDES}")
print(f"Warmup levels: {WARMUP_LEVELS}")
print()

RESULTS_PATH = _SCRIPT_DIR / "shift1_results.json"

mu_zero = config["mu"].copy()    # shape (C, A, d) = (6, 5, 6) — expert prior / warm start

# ---------------------------------------------------------------------------
# Generate a FIXED random shift direction (same across all seeds and deltas).
# Normalised to unit L2 norm per (c, a) cell over the factor dimension.
# Each cell (c,a) has its own direction in R^d.  ||direction[c,a,:]||_2 = 1.
# → actual L2 displacement per cell = δ exactly (before clipping).
# ---------------------------------------------------------------------------

rng_direction = np.random.RandomState(12345)
raw_direction = rng_direction.randn(C, A, d)
norms = np.linalg.norm(raw_direction, axis=-1, keepdims=True)   # (C, A, 1)
norms[norms == 0] = 1.0
unit_direction = raw_direction / norms    # shape (C, A, d)

print(f"Shift direction: unit-normalised per cell, seed=12345")
print(f"  unit_direction shape: {unit_direction.shape}")
print(f"  sample norms (first cell): {np.linalg.norm(unit_direction[0, 0, :]):.6f}")
print()

# ---------------------------------------------------------------------------
# Helper: compute metrics from a flat list of records
# ---------------------------------------------------------------------------

def compute_metrics(records: list[dict]) -> dict:
    """Aggregate accuracy, threshold*, coverage, margin* from flat records."""
    if not records:
        return {
            "overall_accuracy": None,
            "threshold_star":    None,
            "coverage_at_star":  None,
            "margin_star":       None,
            "margin_coverage":   None,
        }
    overall_acc = float(np.mean([r["is_correct"] for r in records]))
    n_total     = len(records)

    # Confidence threshold*
    best_t   = None
    best_cov = None
    for t in THRESHOLD_SWEEP:
        above = [r for r in records if r["confidence"] >= t]
        if len(above) >= 10:
            acc = float(np.mean([r["is_correct"] for r in above]))
            cov = len(above) / n_total
            if acc >= ACCURACY_GATE and best_t is None:
                best_t   = float(t)
                best_cov = float(cov)

    # Margin threshold*
    best_m   = None
    best_m_cov = None
    for m in MARGIN_THRESHOLDS:
        above = [r for r in records if r["margin"] >= m]
        if len(above) >= 10:
            acc = float(np.mean([r["is_correct"] for r in above]))
            cov = len(above) / n_total
            if acc >= ACCURACY_GATE and best_m is None:
                best_m     = float(m)
                best_m_cov = float(cov)

    return {
        "overall_accuracy": round(overall_acc, 4),
        "threshold_star":   best_t,
        "coverage_at_star": round(best_cov, 4) if best_cov is not None else None,
        "margin_star":      best_m,
        "margin_coverage":  round(best_m_cov, 4) if best_m_cov is not None else None,
    }

# ---------------------------------------------------------------------------
# Main sweep: δ × warmup × seed
# ---------------------------------------------------------------------------

all_results: dict[str, dict] = {}

for delta in DELTA_MAGNITUDES:

    # Compute shifted truth: μ_true = clip(μ₀ + δ * unit_direction, 0, 1)
    mu_true     = np.clip(mu_zero + delta * unit_direction, 0.0, 1.0)
    actual_shift = float(np.linalg.norm(mu_true - mu_zero, axis=-1).mean())

    print(f"\n{'='*60}")
    print(f"=== DELTA = {delta:.2f} (mean L2 shift = {actual_shift:.4f}, "
          f"factor_sigma=0.15) ===")
    print(f"{'='*60}")

    # Build the shifted generator config (option b: inject via constructor kwarg)
    # Reconstruct action_conditional_profiles dict from mu_true tensor
    shifted_profiles: dict[str, dict[str, list[float]]] = {
        cat: {
            act: mu_true[c_idx, a_idx, :].tolist()
            for a_idx, act in enumerate(ACTIONS)
        }
        for c_idx, cat in enumerate(CATEGORIES)
    }
    gen_config = dict(config["generator_kwargs"])          # shallow copy of outer dict
    gen_config["action_conditional_profiles"] = shifted_profiles  # replace profiles

    # Verify on first delta only
    if delta == DELTA_MAGNITUDES[0]:
        test_gen = CategoryAlertGenerator(**gen_config, noise_rate=0.0, seed=0)
        sample_mu = np.array(test_gen.profiles["credential_access"]["escalate"])
        print(f"  [Verify] credential_access/escalate mu_zero: "
              f"{mu_zero[0,0,:].round(3).tolist()}")
        print(f"  [Verify] credential_access/escalate mu_true: "
              f"{mu_true[0,0,:].round(3).tolist()}")
        print(f"  [Verify] gen.profiles['credential_access']['escalate']: "
              f"{sample_mu.round(3).tolist()}")
        assert np.allclose(sample_mu, mu_true[0, 0, :], atol=1e-6), (
            "Generator not using mu_true!"
        )
        print(f"  [Verify] Profile injection OK\n")

    for n_warmup in WARMUP_LEVELS:
        print(f"\n--- delta={delta:.2f}, warmup={n_warmup} ---")

        # ============================================================
        # CONDITION A: LEARNING
        # ============================================================
        learning_records:  list[dict] = []
        learning_drifts:   list[float] = []

        for seed_idx in range(N_SEEDS):
            if seed_idx % 10 == 0:
                print(f"  Learning: seed {seed_idx+1}/{N_SEEDS}", flush=True)

            gen = CategoryAlertGenerator(
                **gen_config,
                noise_rate=NOISE_RATE,
                seed=RANDOM_SEED_BASE + seed_idx,
            )
            scorer = ProfileScorer(
                mu_zero.copy(),
                config["actions"],
                tau=TAU,
                eta=ETA,
                eta_neg=ETA_NEG,
            )

            # Warmup with learning (centroids update toward mu_true)
            warmup_alerts = gen.generate(n_warmup)
            for alert in warmup_alerts:
                result     = scorer.score(alert.factors, alert.category_index)
                is_correct = result.action_index == alert.gt_action_index
                scorer.update(
                    alert.factors,
                    alert.category_index,
                    result.action_index,
                    correct=is_correct,
                    gt_action_index=alert.gt_action_index,
                )

            # Measure how far centroids moved from μ₀
            drift = float(np.linalg.norm(scorer.mu - mu_zero, axis=-1).mean())
            learning_drifts.append(drift)

            # Evaluate (learning continues)
            eval_alerts = gen.generate(N_EVAL)
            for alert in eval_alerts:
                result     = scorer.score(alert.factors, alert.category_index)
                is_correct = result.action_index == alert.gt_action_index
                sorted_probs = np.sort(result.probabilities)[::-1]
                margin = float(sorted_probs[0] - sorted_probs[1])
                scorer.update(
                    alert.factors,
                    alert.category_index,
                    result.action_index,
                    correct=is_correct,
                    gt_action_index=alert.gt_action_index,
                )
                learning_records.append({
                    "category":   alert.category,
                    "confidence": float(result.confidence),
                    "margin":     margin,
                    "is_correct": bool(is_correct),
                })

        # ============================================================
        # CONDITION B: FROZEN
        # ============================================================
        frozen_records: list[dict] = []

        for seed_idx in range(N_SEEDS):
            if seed_idx % 10 == 0:
                print(f"  Frozen:   seed {seed_idx+1}/{N_SEEDS}", flush=True)

            gen = CategoryAlertGenerator(
                **gen_config,
                noise_rate=NOISE_RATE,
                seed=RANDOM_SEED_BASE + seed_idx,
            )
            scorer = ProfileScorer(
                mu_zero.copy(),
                config["actions"],
                tau=TAU,
                eta=ETA,
                eta_neg=ETA_NEG,
            )

            # Advance generator RNG by n_warmup to match LEARNING condition.
            # Discard outputs — scorer stays frozen at μ₀.
            _ = gen.generate(n_warmup)

            # Evaluate without any learning
            eval_alerts = gen.generate(N_EVAL)
            for alert in eval_alerts:
                result     = scorer.score(alert.factors, alert.category_index)
                is_correct = result.action_index == alert.gt_action_index
                sorted_probs = np.sort(result.probabilities)[::-1]
                margin = float(sorted_probs[0] - sorted_probs[1])
                frozen_records.append({
                    "category":   alert.category,
                    "confidence": float(result.confidence),
                    "margin":     margin,
                    "is_correct": bool(is_correct),
                })

        # ============================================================
        # Aggregate and store
        # ============================================================
        learning_metrics = compute_metrics(learning_records)
        frozen_metrics   = compute_metrics(frozen_records)
        mean_drift       = round(float(np.mean(learning_drifts)), 6)

        l_acc = learning_metrics["overall_accuracy"] or 0.0
        f_acc = frozen_metrics["overall_accuracy"]   or 0.0
        lift  = round(l_acc - f_acc, 4)

        result_key = f"d{delta:.2f}_w{n_warmup}"
        all_results[result_key] = {
            "delta":               delta,
            "warmup":              n_warmup,
            "mean_shift_from_mu0": round(actual_shift, 4),
            "mean_centroid_drift": mean_drift,
            "learning":            learning_metrics,
            "frozen":              frozen_metrics,
            "learning_lift":       lift,
        }

        print(f"  Frozen:   acc={f_acc:.1%}  "
              f"cov={frozen_metrics.get('coverage_at_star') or 'N/A'}")
        print(f"  Learning: acc={l_acc:.1%}  "
              f"cov={learning_metrics.get('coverage_at_star') or 'N/A'}  "
              f"drift={mean_drift:.4f}")
        print(f"  Lift:     {lift:+.1%}")

# ---------------------------------------------------------------------------
# Cross-condition summary
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("=== LEARNING LIFT: FROZEN vs LEARNED (overall accuracy) ===")
print(f"{'='*70}")
print(f"{'delta':>6} | {'warmup':>6} | {'frozen acc':>10} | "
      f"{'learned acc':>11} | {'lift':>7} | {'drift':>8}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*7}-+-{'-'*8}")
for key in sorted(all_results.keys()):
    r = all_results[key]
    f_acc = r["frozen"]["overall_accuracy"] or 0.0
    l_acc = r["learning"]["overall_accuracy"] or 0.0
    print(f"{r['delta']:6.2f} | {r['warmup']:6d} | "
          f"{f_acc:10.1%} | "
          f"{l_acc:11.1%} | "
          f"{r['learning_lift']:+6.1%} | "
          f"{r['mean_centroid_drift']:8.4f}")

print(f"\n{'='*70}")
print("=== COVERAGE GROWTH: LEARNED (confidence gate at 85%) ===")
print(f"{'='*70}")
print(f"{'delta':>6} | {'warmup':>6} | {'threshold*':>10} | {'coverage':>8}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}")
for key in sorted(all_results.keys()):
    r = all_results[key]
    t = r["learning"]["threshold_star"]
    c = r["learning"]["coverage_at_star"]
    t_str = f"{t:.3f}" if t is not None else "    NONE"
    c_str = f"{c:.1%}" if c is not None else "     ---"
    print(f"{r['delta']:6.2f} | {r['warmup']:6d} | {t_str:>10} | {c_str:>8}")

print(f"\n{'='*70}")
print("=== CENTROID DRIFT SUMMARY ===")
print(f"{'='*70}")
print(f"{'delta':>6} | {'warmup':>6} | {'mean drift':>10} | {'shift':>6}")
print(f"{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}")
for key in sorted(all_results.keys()):
    r = all_results[key]
    print(f"{r['delta']:6.2f} | {r['warmup']:6d} | "
          f"{r['mean_centroid_drift']:10.4f} | "
          f"{r['mean_shift_from_mu0']:6.4f}")

print()
print("=== INTERPRETATION GUIDE ===")
print("If learning_lift > 0 at delta > 0 but ~0 at delta = 0:")
print("  → Circularity was masking real learning capacity.")
print("  → Production (where delta > 0) will show genuine improvement.")
print("If learning_lift ~0 at ALL delta values:")
print("  → Architecture is the ceiling. Learning can't help even with signal.")
print("If coverage grows with warmup at delta > 0 but not delta = 0:")
print("  → Warmup plateau was a test artifact. Production will benefit from warmup.")
print("If drift is near 0 even at large delta:")
print("  → Centroid update rate (η=0.05) too small for the eval window; need more warmup.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

# Serialise float keys as strings (JSON limitation)
with open(RESULTS_PATH, "w") as fh:
    json.dump(all_results, fh, indent=2, default=str)

assert RESULTS_PATH.exists()
print(f"\nResults written to: {RESULTS_PATH.resolve()}")

# ---------------------------------------------------------------------------
# Launch charts
# ---------------------------------------------------------------------------

subprocess.run(
    [sys.executable, str(_SCRIPT_DIR / "charts.py")],
    check=True,
    cwd=str(_REPO_ROOT),
)
