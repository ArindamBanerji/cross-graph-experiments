"""
PROFILE-CONFIG-CHECK: Profile Separation Experiment
Tests whether pushing refer_to_analyst centroids apart recovers accuracy.
4 conditions x 50 seeds x 500 decisions.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.data.domain_config import load_domain_config
from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.profile_scorer import ProfileScorer

config = load_domain_config('soc_product_v50')
mu_orig = config['mu'].copy()
cats = config['categories']
acts = config['actions']
C, A, d = len(cats), len(acts), mu_orig.shape[-1]

print(f"Config: C={C}, A={A}, d={d}")
print(f"Actions: {acts}")

ref_idx = acts.index('refer_to_analyst')
print(f"refer_to_analyst index: {ref_idx}")

# Current min distances
print("\n=== CURRENT MIN DISTANCES (refer_to_analyst vs others) ===")
for ci, cat in enumerate(cats):
    ref = mu_orig[ci, ref_idx, :]
    dists = []
    for ai, act in enumerate(acts):
        if ai == ref_idx:
            continue
        dists.append((np.linalg.norm(ref - mu_orig[ci, ai, :]), act))
    dists.sort()
    print(f"  {cat:>24}: {dists[0][1]:>18} dist={dists[0][0]:.3f}")

# ============================================================
N_SEEDS = 50
N_DECISIONS = 500
TAU = 0.1

def measure_accuracy(mu_test, acts_test, n_actions):
    accs = []
    for seed in range(N_SEEDS):
        # Build profiles for the actions we're testing
        profiles = {}
        for ci, cat in enumerate(cats):
            profiles[cat] = {}
            for ai in range(n_actions):
                profiles[cat][acts_test[ai]] = mu_test[ci, ai, :].tolist()

        # Copy generator kwargs and override profiles + actions
        gen_kwargs = dict(config['generator_kwargs'])
        gen_kwargs['action_conditional_profiles'] = profiles
        gen_kwargs['actions'] = acts_test[:n_actions]
        gen_kwargs['noise_rate'] = 0.0
        gen_kwargs['seed'] = 42 + seed

        gen = CategoryAlertGenerator(**gen_kwargs)
        scorer = ProfileScorer(
            mu_test[:, :n_actions, :].copy(),
            acts_test[:n_actions],
            tau=TAU, eta=0.05, eta_neg=0.05,
        )
        alerts = gen.generate(N_DECISIONS)
        correct = 0
        for a in alerts:
            result = scorer.score(a.factors, a.category_index)
            if result.action_index == a.gt_action_index:
                correct += 1
        accs.append(correct / N_DECISIONS)
    mean = np.mean(accs)
    ci = 1.96 * np.std(accs) / np.sqrt(N_SEEDS)
    return mean, ci

# ============================================================
# CONDITION 0: Original (A=5, as-is)
# ============================================================
print("\n--- Condition 0: Original (A=5, as-is) ---")
acc0, ci0 = measure_accuracy(mu_orig, acts, A)
print(f"  Accuracy: {acc0:.1%} +/- {ci0:.1%}")

# ============================================================
# CONDITION 1: Push refer_to_analyst +0.15
# ============================================================
mu_a1 = mu_orig.copy()
for ci in range(C):
    ref = mu_a1[ci, ref_idx, :].copy()
    min_d = 999
    nearest_idx = -1
    for ai in range(A):
        if ai == ref_idx:
            continue
        dv = np.linalg.norm(ref - mu_a1[ci, ai, :])
        if dv < min_d:
            min_d = dv
            nearest_idx = ai
    direction = ref - mu_a1[ci, nearest_idx, :]
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    mu_a1[ci, ref_idx, :] = np.clip(ref + 0.15 * direction, 0, 1)

print("\n--- Pushed distances (+0.15) ---")
for ci, cat in enumerate(cats):
    ref_new = mu_a1[ci, ref_idx, :]
    dists = []
    for ai, act in enumerate(acts):
        if ai == ref_idx:
            continue
        dists.append((np.linalg.norm(ref_new - mu_a1[ci, ai, :]), act))
    dists.sort()
    ref_old = mu_orig[ci, ref_idx, :]
    old_min = min(np.linalg.norm(ref_old - mu_orig[ci, ai, :])
                  for ai in range(A) if ai != ref_idx)
    print(f"  {cat:>24}: {dists[0][1]:>18} dist={dists[0][0]:.3f} (was {old_min:.3f})")

print("\n--- Condition 1: Pushed +0.15 (A=5) ---")
acc1, ci1 = measure_accuracy(mu_a1, acts, A)
print(f"  Accuracy: {acc1:.1%} +/- {ci1:.1%}")

# ============================================================
# CONDITION 2: Push refer_to_analyst +0.25
# ============================================================
mu_a2 = mu_orig.copy()
for ci in range(C):
    ref = mu_a2[ci, ref_idx, :].copy()
    min_d = 999
    nearest_idx = -1
    for ai in range(A):
        if ai == ref_idx:
            continue
        dv = np.linalg.norm(ref - mu_a2[ci, ai, :])
        if dv < min_d:
            min_d = dv
            nearest_idx = ai
    direction = ref - mu_a2[ci, nearest_idx, :]
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    mu_a2[ci, ref_idx, :] = np.clip(ref + 0.25 * direction, 0, 1)

print("\n--- Condition 2: Pushed +0.25 (A=5) ---")
acc2, ci2 = measure_accuracy(mu_a2, acts, A)
print(f"  Accuracy: {acc2:.1%} +/- {ci2:.1%}")

# ============================================================
# CONDITION 3: Drop refer_to_analyst (A=4)
# ============================================================
acts_4 = [a for a in acts if a != 'refer_to_analyst']
mu_4 = np.delete(mu_orig, ref_idx, axis=1)

print("\n--- Condition 3: Dropped refer_to_analyst (A=4) ---")
acc3, ci3 = measure_accuracy(mu_4, acts_4, 4)
print(f"  Accuracy: {acc3:.1%} +/- {ci3:.1%}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("=== PROFILE SEPARATION EXPERIMENT SUMMARY ===")
print("=" * 65)
print(f"  Original (A=5, as-is):   {acc0:.1%} +/- {ci0:.1%}")
print(f"  Pushed +0.15 (A=5):      {acc1:.1%} +/- {ci1:.1%}  ({(acc1-acc0)*100:+.1f}pp)")
print(f"  Pushed +0.25 (A=5):      {acc2:.1%} +/- {ci2:.1%}  ({(acc2-acc0)*100:+.1f}pp)")
print(f"  Dropped refer (A=4):     {acc3:.1%} +/- {ci3:.1%}  ({(acc3-acc0)*100:+.1f}pp)")
print()

if acc3 > acc0 + 0.05:
    print("  VERDICT: refer_to_analyst profiles compress Voronoi cells.")
    print(f"  Dropping refer recovers {(acc3-acc0)*100:+.1f}pp.")
    print("  Recommendation: Option B (policy-only refer) for v6.")

if acc2 > acc0 + 0.03:
    print("  VERDICT: Profile separation fixes the problem.")
    print(f"  Pushing +0.25 recovers {(acc2-acc0)*100:+.1f}pp.")
    print("  Recommendation: Option A (config fix) viable for v5.5.")

if acc2 <= acc0 + 0.03 and acc3 <= acc0 + 0.05:
    print("  VERDICT: Profile separation has minimal effect.")
    print("  The 80.6% ceiling is structural, not a config gap.")
