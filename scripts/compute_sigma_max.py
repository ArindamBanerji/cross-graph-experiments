"""
P6 — Compute σ_max from empirical L2 margin distribution.

The design default σ_max=1.0 is a placeholder. The correct value is the
10th percentile of the L2 margin distribution across factor vectors drawn
from soc_product_v50 (A=4 geometry).

Margin definition (on squared L2 distances, matching ProfileScorer internals):
  d_a = ‖f − μ[c, a, :]‖²   for each action a
  a*  = argmin_a(d_a)          winning action (nearest centroid)
  margin = min_{a ≠ a*}(d_a) − d_a*   ≥ 0 always

σ_max = percentile(margins, 10)

Principle: At σ_max and λ=0.5, the maximum synthesis bias added to any
distance is σ_max × 0.5. This must be less than the median margin to
ensure that the centroid evidence (experience) overrides the operator
injection in the majority of decisions.
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SEEDS      = 50
N_ALERTS     = 500     # per seed → 25,000 total
NOISE_RATE   = 0.10
PERCENTILE   = 10      # σ_max = p10 of margin distribution
LAMBDA_REF   = 0.5     # canonical λ for influence calculation
DOMAIN_CONFIG = "soc_product_v50"
RANDOM_SEED_BASE = 42

OUT_DIR  = _REPO_ROOT / "results"
OUT_FILE = OUT_DIR / "sigma_max_derivation.json"

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

config     = load_domain_config(DOMAIN_CONFIG)
CATEGORIES = config["categories"]
ACTIONS    = config["actions"]
C, A, d    = config["C"], config["A"], config["d"]
mu         = config["mu"].copy()    # shape (C, A, d), float64

assert mu.shape == (6, 4, 6), f"Unexpected mu shape: {mu.shape}"
assert A == 4

print("=" * 60)
print("=== σ_max DERIVATION (A=4 geometry) ===")
print("=" * 60)
print(f"Config: {DOMAIN_CONFIG}  C={C}, A={A}, d={d}")
print(f"mu shape: {mu.shape}")
print(f"N_SEEDS={N_SEEDS}, N_ALERTS={N_ALERTS}  →  "
      f"{N_SEEDS * N_ALERTS:,} total factor vectors")
print(f"Margin percentile for σ_max: p{PERCENTILE}")
print()

# ---------------------------------------------------------------------------
# Generate factor vectors and compute margins
# ---------------------------------------------------------------------------

all_margins:     list[float] = []
per_cat_margins: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}

for seed_idx in range(N_SEEDS):
    gen = CategoryAlertGenerator(
        **config["generator_kwargs"],
        noise_rate=NOISE_RATE,
        seed=RANDOM_SEED_BASE + seed_idx,
    )
    alerts = gen.generate(N_ALERTS)

    for alert in alerts:
        f   = alert.factors.flatten()                     # (d,)
        c   = alert.category_index
        mu_c = mu[c]                                      # (A, d)

        # Squared L2 distances to all action centroids (matching ProfileScorer)
        diffs     = mu_c - f                              # (A, d)
        distances = np.sum(diffs ** 2, axis=1)            # (A,)  = d_a

        a_star = int(np.argmin(distances))
        d_star = distances[a_star]

        # Margin = min distance gap between winner and any other action
        others = [distances[a] for a in range(A) if a != a_star]
        margin = float(min(others) - d_star)              # always ≥ 0

        all_margins.append(margin)
        per_cat_margins[CATEGORIES[c]].append(margin)

n_samples = len(all_margins)
assert n_samples == N_SEEDS * N_ALERTS, \
    f"Expected {N_SEEDS * N_ALERTS} margins, got {n_samples}"

print(f"Generated {n_samples:,} margin values. Computing statistics...")

# ---------------------------------------------------------------------------
# Margin distribution statistics
# ---------------------------------------------------------------------------

margins_arr = np.array(all_margins, dtype=np.float64)

p5   = float(np.percentile(margins_arr, 5))
p10  = float(np.percentile(margins_arr, 10))   # σ_max
p25  = float(np.percentile(margins_arr, 25))
p50  = float(np.percentile(margins_arr, 50))
p75  = float(np.percentile(margins_arr, 75))
p90  = float(np.percentile(margins_arr, 90))
mean = float(margins_arr.mean())
std  = float(margins_arr.std())
mn   = float(margins_arr.min())
mx   = float(margins_arr.max())

sigma_max              = p10
max_synthesis_influence = sigma_max * LAMBDA_REF
median_margin          = p50
experience_dominates   = max_synthesis_influence < median_margin

dominance_factor = median_margin / max_synthesis_influence \
    if max_synthesis_influence > 1e-12 else float("inf")

# ---------------------------------------------------------------------------
# Per-category statistics
# ---------------------------------------------------------------------------

per_cat_stats: dict[str, dict] = {}
for cat in CATEGORIES:
    arr_c = np.array(per_cat_margins[cat], dtype=np.float64)
    per_cat_stats[cat] = {
        "p10":   float(np.percentile(arr_c, 10)),
        "p50":   float(np.percentile(arr_c, 50)),
        "mean":  float(arr_c.mean()),
        "std":   float(arr_c.std()),
        "n":     len(arr_c),
    }

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print()
print("Margin distribution (squared L2 distance units):")
print(f"  min:  {mn:.6f}")
print(f"  p5:   {p5:.6f}")
print(f"  p10:  {p10:.6f}  ← σ_max")
print(f"  p25:  {p25:.6f}")
print(f"  p50:  {p50:.6f}  ← median")
print(f"  p75:  {p75:.6f}")
print(f"  p90:  {p90:.6f}")
print(f"  mean: {mean:.6f}")
print(f"  std:  {std:.6f}")
print(f"  max:  {mx:.6f}")
print()
print(f"σ_max = p{PERCENTILE} = {sigma_max:.6f}")
print(f"At λ={LAMBDA_REF}: max synthesis influence = σ_max × λ = "
      f"{max_synthesis_influence:.6f}")
print(f"Median margin:                               {median_margin:.6f}")
print(f"Experience dominance ratio:                  {dominance_factor:.2f}×")
print(f"Experience dominates (influence < median):   "
      f"{'YES ✓' if experience_dominates else 'NO ✗'}")
print()
print(f"Design default was σ_max=1.0:")
if sigma_max < 0.5:
    print(f"  σ_max={sigma_max:.4f} < 0.5 — synthesis is TIGHTLY constrained.")
    print(f"  Default of 1.0 would allow {1.0 / sigma_max:.1f}× more influence "
          f"than the p10 margin.")
elif sigma_max <= 1.0:
    print(f"  σ_max={sigma_max:.4f} — within design default range.")
    print(f"  Default of 1.0 is borderline ({1.0 / sigma_max:.2f}× p10 margin).")
else:
    print(f"  σ_max={sigma_max:.4f} > 1.0 — design default was CONSERVATIVE (good).")

print()
print("Per-category p10 margin (tightest constraint per category):")
for cat in CATEGORIES:
    s = per_cat_stats[cat]
    print(f"  {cat:<28}: p10={s['p10']:.6f}  p50={s['p50']:.6f}  "
          f"mean={s['mean']:.6f}")

# ---------------------------------------------------------------------------
# Build interpretation string
# ---------------------------------------------------------------------------

interp = (
    f"At σ_max={sigma_max:.4f} and λ={LAMBDA_REF}, synthesis influence is "
    f"{max_synthesis_influence:.4f} distance units vs median margin of "
    f"{median_margin:.4f}. Experience dominates by factor {dominance_factor:.2f}×. "
    f"Only 10% of decisions have a margin below σ_max — for those, "
    f"a strong operator signal could influence the outcome. "
    f"Design default σ_max=1.0 {'was already safe' if sigma_max > 1.0 else f'should be reduced to {sigma_max:.4f}'}."
)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

output = {
    "sigma_max":          sigma_max,
    "percentile_used":    PERCENTILE,
    "margin_distribution": {
        "min":  mn,
        "p5":   p5,
        "p10":  p10,
        "p25":  p25,
        "p50":  p50,
        "p75":  p75,
        "p90":  p90,
        "mean": mean,
        "std":  std,
        "max":  mx,
    },
    "max_synthesis_influence_at_lambda_05": max_synthesis_influence,
    "median_margin":         median_margin,
    "experience_dominates":  experience_dominates,
    "dominance_factor":      dominance_factor,
    "n_samples":             n_samples,
    "n_seeds":               N_SEEDS,
    "n_alerts_per_seed":     N_ALERTS,
    "noise_rate":            NOISE_RATE,
    "lambda_ref":            LAMBDA_REF,
    "config":                f"{DOMAIN_CONFIG} A={A}",
    "per_category":          per_cat_stats,
    "interpretation":        interp,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as fh:
    json.dump(output, fh, indent=2)

print()
print(f"Results saved → {OUT_FILE}")
print()
print("=== SUMMARY ===")
print(f"  σ_max = {sigma_max:.6f}  (p{PERCENTILE} of {n_samples:,} margin values)")
print(f"  At λ={LAMBDA_REF}: max influence = {max_synthesis_influence:.6f}  "
      f"vs median margin {median_margin:.6f}")
print(f"  Experience dominance: {dominance_factor:.2f}×  "
      f"({'PASS ✓' if experience_dominates else 'FAIL ✗'})")
