"""
EXP-C1 SUPPLEMENT: Factor Magnitude Confounding Visualization

Demonstrates why dot product scoring fails on bounded [0,1] features:
high-magnitude factors dominate the dot product regardless of their
discriminative value, while L2 distance correctly isolates deviation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator, CATEGORIES, ACTIONS, FACTORS
from src.models.profile_scorer import ProfileScorer

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
C, A, d = 5, 4, 6
tau = 0.1
N_alerts = 2000
seed = 42
factor_names = FACTORS  # ["travel_match", "asset_criticality", "threat_intel",
                        #  "time_anomaly", "device_trust", "pattern_history"]

# ---------------------------------------------------------------------------
# Step 1 — Initialize warm scorer and extract centroid means per factor
# ---------------------------------------------------------------------------
gen = CategoryAlertGenerator(seed=seed)

scorer = ProfileScorer(C, A, d, tau=tau, seed=seed)
scorer.init_from_profiles(
    {cat: {act: list(gen.profiles[cat][act]) for act in gen.actions} for cat in gen.categories},
    gen.categories,
    gen.actions,
)

mu = scorer.mu  # shape (C, A, d)
assert mu.shape == (C, A, d), f"Expected ({C},{A},{d}), got {mu.shape}"

# Per factor: mean centroid value across all (category, action) combinations
factor_centroid_means = mu.reshape(-1, d).mean(axis=0)   # (d,)
factor_centroid_stds  = mu.reshape(-1, d).std(axis=0)    # (d,)

assert factor_centroid_means.shape == (d,), f"Shape mismatch: {factor_centroid_means.shape}"
assert np.all(factor_centroid_means >= 0.0) and np.all(factor_centroid_means <= 1.0), \
    f"Centroid means out of [0,1]: {factor_centroid_means}"

# ---------------------------------------------------------------------------
# Step 2 — Generate alerts and compute per-factor contributions
# ---------------------------------------------------------------------------
alerts = gen.generate(N_alerts)
factor_vectors = np.array([a.factors for a in alerts])   # (N_alerts, d)
factor_means_per_alert = factor_vectors.mean(axis=0)     # (d,)

# dot_contribution(i) = factor_mean(i) × centroid_mean(i)
# l2_contribution(i)  = (factor_mean(i) - centroid_mean(i))²
dot_contributions = factor_means_per_alert * factor_centroid_means  # (d,)
l2_contributions  = (factor_means_per_alert - factor_centroid_means) ** 2  # (d,)

# ---------------------------------------------------------------------------
# Step 3 — Compare discriminative power: action 0 vs action 1, category 0
# ---------------------------------------------------------------------------
mu_action0 = mu[0, 0, :]   # category 0, action 0 centroid
mu_action1 = mu[0, 1, :]   # category 0, action 1 centroid
f_example  = alerts[0].factors  # single example alert (d,)

# Dot product: per-factor component for each action
dot_comp_a0 = f_example * mu_action0   # (d,)
dot_comp_a1 = f_example * mu_action1   # (d,)
dot_diff    = np.abs(dot_comp_a0 - dot_comp_a1)  # separating power per factor

# L2: per-factor component for each action
l2_comp_a0 = (f_example - mu_action0) ** 2   # (d,)
l2_comp_a1 = (f_example - mu_action1) ** 2   # (d,)
l2_diff    = np.abs(l2_comp_a0 - l2_comp_a1)  # separating power per factor

assert np.all(dot_diff >= 0), "dot_diff has negative values"
assert np.all(l2_diff >= 0), "l2_diff has negative values"

# ---------------------------------------------------------------------------
# Print analysis
# ---------------------------------------------------------------------------
print("=== FACTOR MAGNITUDE ANALYSIS ===")
print()
print("Factor centroid means (sorted high to low):")
for i in np.argsort(factor_centroid_means)[::-1]:
    print(f"  {factor_names[i]:20s}  centroid_mean={factor_centroid_means[i]:.3f}  "
          f"dot_contrib={dot_contributions[i]:.4f}  l2_contrib={l2_contributions[i]:.6f}")
print()
print("Factors with centroid_mean > 0.70 are magnitude-confounding under dot product.")
high_mean = [factor_names[i] for i in range(d) if factor_centroid_means[i] > 0.70]
print(f"High-mean factors: {high_mean}")
print()
print(f"Example alert factors (alert[0]): {f_example}")
print(f"Category 0 action 0 centroid:    {mu_action0}")
print(f"Category 0 action 1 centroid:    {mu_action1}")
print()
print("Per-factor separating power (action 0 vs action 1, category 0):")
print(f"  {'Factor':20s}  {'dot_diff':>10s}  {'l2_diff':>10s}")
for i, fn in enumerate(factor_names):
    print(f"  {fn:20s}  {dot_diff[i]:10.6f}  {l2_diff[i]:10.6f}")
print()

# Validation checks
dot_norm = dot_diff / (dot_diff.sum() + 1e-9)
l2_norm  = l2_diff  / (l2_diff.sum()  + 1e-9)
print(f"dot_norm sum = {dot_norm.sum():.6f}  (expected ~1.0)")
print(f"l2_norm  sum = {l2_norm.sum():.6f}  (expected ~1.0)")
print()

if len(high_mean) == 0:
    print("WARNING: No high-mean factors found — check config profiles")
else:
    print(f"PASS: {len(high_mean)} magnitude-confounding factor(s) found")

# ---------------------------------------------------------------------------
# Generate charts
# ---------------------------------------------------------------------------
from experiments.expC1_magnitude.charts import make_charts

make_charts(
    factor_names=factor_names,
    factor_centroid_means=factor_centroid_means,
    factor_centroid_stds=factor_centroid_stds,
    dot_contributions=dot_contributions,
    l2_contributions=l2_contributions,
    dot_diff=dot_diff,
    l2_diff=l2_diff,
)

print()
print("=== EXP-C1 MAGNITUDE RUN COMPLETE ===")
