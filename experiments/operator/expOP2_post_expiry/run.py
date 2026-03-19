"""
EXP-OP2 SUPPLEMENT: Post-TTL Expiry Comparison (B-exp vs C-exp)

Tests whether TTL expiry is a sufficient safety mechanism.
Key finding: C-exp (harmful operator) causes lasting centroid damage
that persists after the operator expires at TTL=150.

5 conditions:
  A:     no operator (baseline)
  B:     correct operator (sigma[:, 0]=+0.4), active all N_post
  B-exp: correct operator, expires at decision TTL=150 → sigma=0
  C:     harmful operator (sigma[:, 0]=-0.4), active all N_post
  C-exp: harmful operator, expires at TTL=150 → sigma=0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.category_alert_generator import CategoryAlertGenerator, CATEGORIES, ACTIONS
from src.models.oracle import GTAlignedOracle
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
C_DIM, A_DIM, N_FACTORS = 5, len(ACTIONS), 6   # A_DIM=4 from actual config
AC_IDX  = ACTIONS.index("auto_close")           # 0
ESC_IDX = ACTIONS.index("escalate_incident")    # 3

tau        = 0.1
lambda_val = 0.5
TTL        = 150
N_seeds    = 20
N_pre      = 200
N_post     = 400
window_size = 50
eta        = 0.05
eta_neg    = 0.05    # CANONICAL. Never 1.0.
n_windows  = N_post // window_size   # 8

CONDITIONS = ["A", "B", "B-exp", "C", "C-exp"]

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== EXP-OP2 POST-EXPIRY SUPPLEMENT ===")
print(f"C={C_DIM} A={A_DIM} d={N_FACTORS}  tau={tau}  lambda={lambda_val}  TTL={TTL}")
print(f"N_seeds={N_seeds}  N_pre={N_pre}  N_post={N_post}  window={window_size}")
print()


def _build_mu_from_gen(gen: CategoryAlertGenerator) -> np.ndarray:
    """Extract warm-start centroids from generator profiles."""
    mu = np.zeros((C_DIM, A_DIM, N_FACTORS), dtype=np.float64)
    for c_idx, cat in enumerate(gen.categories):
        for a_idx, act in enumerate(gen.actions):
            mu[c_idx, a_idx, :] = gen.profiles[cat][act]
    return mu


def _build_sigma(cond: str) -> np.ndarray:
    """Build sigma tensor for the given condition.

    Uses dual-action design matching OP2:
      Correct (B): suppress auto_close (+), promote escalate_incident (-)
      Harmful (C): promote auto_close (-), suppress escalate_incident (+)
    σ[c,a] < 0 → action MORE likely (smaller effective distance)
    σ[c,a] > 0 → action LESS likely (larger effective distance)
    """
    sigma = np.zeros((C_DIM, A_DIM), dtype=np.float64)
    if cond in ("B", "B-exp"):
        sigma[:, AC_IDX]  = +0.4   # auto_close less likely
        sigma[:, ESC_IDX] = -0.4   # escalate_incident more likely
    elif cond in ("C", "C-exp"):
        sigma[:, AC_IDX]  = -0.4   # auto_close more likely (harmful)
        sigma[:, ESC_IDX] = +0.4   # escalate_incident less likely (harmful)
    # A: zero sigma
    return sigma


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# per_condition_deltas[cond] will be (N_seeds, n_windows)
per_condition_deltas: dict[str, list] = {cond: [] for cond in CONDITIONS}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for seed in range(N_seeds):
    # One gen_pre and gen_post per seed — shared across all conditions
    # so every condition sees exactly the same alert stream (fair comparison).
    gen_pre  = CategoryAlertGenerator(seed=seed)
    gen_post = CategoryAlertGenerator(seed=seed + 10000)
    oracle   = GTAlignedOracle(noise_rate=0.10, seed=seed + 20000)

    pre_alerts  = gen_pre.generate(N_pre)
    post_alerts = gen_post.generate(N_post)

    # Warm-start mu extracted once per seed; each condition gets its own copy
    mu_warm = _build_mu_from_gen(gen_pre)

    for cond in CONDITIONS:
        scorer = ProfileScorer(
            mu_warm.copy(), ACTIONS, tau=tau, eta=eta, eta_neg=eta_neg, seed=seed,
        )
        sigma_full  = _build_sigma(cond)   # (C, A)  — nonzero for B/B-exp/C/C-exp

        # --- Pre-shift warmup ---
        pre_correct: list[float] = []
        for alert in pre_alerts:
            result       = scorer.score(alert.factors, alert.category_index)
            is_correct   = result.action_index == alert.gt_action_index
            pre_correct.append(float(is_correct))
            oracle_result  = oracle.evaluate(ACTIONS[result.action_index], alert)
            outcome_correct = oracle_result.outcome > 0
            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, outcome_correct,
                gt_action_index=alert.gt_action_index,
            )
        auac_pre = float(np.mean(pre_correct))

        # --- Post-shift decisions ---
        post_correct: list[float] = []
        for decision, alert in enumerate(post_alerts):
            # TTL expiry for B-exp and C-exp
            if cond in ("B-exp", "C-exp") and decision >= TTL:
                synthesis = None   # operator expired → pure L2
            elif np.any(sigma_full != 0):
                synthesis = SynthesisBias(
                    sigma=sigma_full,
                    active_claims=1,
                    lambda_coupling=lambda_val,
                )
            else:
                synthesis = None   # condition A

            result       = scorer.score(alert.factors, alert.category_index,
                                        synthesis=synthesis)
            is_correct   = result.action_index == alert.gt_action_index
            post_correct.append(float(is_correct))

            oracle_result   = oracle.evaluate(ACTIONS[result.action_index], alert)
            outcome_correct = oracle_result.outcome > 0
            scorer.update(
                alert.factors, alert.category_index,
                result.action_index, outcome_correct,
                gt_action_index=alert.gt_action_index,
            )

        # --- Per-window AUAC deltas ---
        window_deltas: list[float] = []
        for w in range(n_windows):
            start = w * window_size
            end   = start + window_size
            window_acc = float(np.mean(post_correct[start:end]))
            window_deltas.append(window_acc - auac_pre)
        per_condition_deltas[cond].append(window_deltas)

    print(f"  seed={seed:2d} done  "
          f"pre_acc={auac_pre:.3f}  "
          f"C-exp_post_w7={per_condition_deltas['C-exp'][-1][-1]:+.4f}")

# ---------------------------------------------------------------------------
# Convert to arrays and compute stats
# ---------------------------------------------------------------------------
for cond in CONDITIONS:
    per_condition_deltas[cond] = np.array(per_condition_deltas[cond])  # (N_seeds, 8)
    assert per_condition_deltas[cond].shape == (N_seeds, n_windows), \
        f"Shape mismatch for {cond}: {per_condition_deltas[cond].shape}"

delta_mean = {cond: per_condition_deltas[cond].mean(axis=0) for cond in CONDITIONS}
delta_std  = {cond: per_condition_deltas[cond].std(axis=0)  for cond in CONDITIONS}

# Key summaries
post_expiry_idx = list(range(3, 8))   # windows 3-7: decisions 150-400
pre_expiry_idx  = [0, 1, 2]           # windows 0-2: decisions 0-150

b_exp_post     = float(delta_mean["B-exp"][post_expiry_idx].mean())
c_exp_post     = float(delta_mean["C-exp"][post_expiry_idx].mean())
b_exp_post_std = float(per_condition_deltas["B-exp"][:, post_expiry_idx].mean(axis=1).std())
c_exp_post_std = float(per_condition_deltas["C-exp"][:, post_expiry_idx].mean(axis=1).std())
b_exp_pre      = float(delta_mean["B-exp"][pre_expiry_idx].mean())
c_exp_pre      = float(delta_mean["C-exp"][pre_expiry_idx].mean())

print()
print("=== POST-EXPIRY RESULTS ===")
print(f"B-exp post-TTL (windows 3-7): {b_exp_post:+.4f} ± {b_exp_post_std:.4f}  "
      f"(published: +0.0128 ± 0.0190)")
print(f"C-exp post-TTL (windows 3-7): {c_exp_post:+.4f} ± {c_exp_post_std:.4f}  "
      f"(published: -0.0124 ± 0.0187)")
print(f"C-exp lasting damage confirmed: {c_exp_post < 0}")
print()
print("Per-window delta table:")
print(f"{'Window':>8}  {'decisions':>12}  " +
      "  ".join(f"{c:>8}" for c in CONDITIONS))
for w in range(n_windows):
    start = w * window_size
    end   = start + window_size
    ttl_marker = " <- TTL" if w == 3 else ""
    print(f"  w={w}     dec {start:3d}-{end:3d}  " +
          "  ".join(f"{delta_mean[c][w]:+.4f}" for c in CONDITIONS) + ttl_marker)

# Validation checks
assert per_condition_deltas["A"].shape == (N_seeds, n_windows), "A shape wrong"
print()
if c_exp_post < 0:
    print("PASS: C-exp post-expiry delta is negative (lasting damage confirmed)")
else:
    # At eta_neg=0.05 (vs OP2's 1.0), centroid drift is slower — damage may
    # heal within 250 post-expiry decisions. Report result without failing.
    print(f"NOTE: C-exp post-expiry delta={c_exp_post:+.4f} (positive — healed by dec 400).")
    print("      Lasting damage requires eta_neg >> 0.05 or longer TTL window to persist.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
np.save(
    str(OUT_DIR / "results.npy"),
    {cond: per_condition_deltas[cond] for cond in CONDITIONS},
    allow_pickle=True,
)
print(f"Results saved to {OUT_DIR / 'results.npy'}")
print("Calling charts.py ...")

# ---------------------------------------------------------------------------
# Generate charts
# ---------------------------------------------------------------------------
from experiments.operator.expOP2_post_expiry.charts import make_charts

make_charts()

print()
print("=== EXP-OP2 POST-EXPIRY RUN COMPLETE ===")
