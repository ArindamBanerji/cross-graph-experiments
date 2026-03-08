# Cross-Graph Experiments: Project Structure

## Overview

Experimental validation suite for the **Cross-Graph Attention** framework. The codebase has grown from the original 4 experiments into a comprehensive validation suite with 14 experiments across two phases:

**Phase 1 — Core Framework (Exp 1-4):** Validates the cross-graph attention equations at three levels.

| Level | Experiment | What it validates |
|-------|------------|-------------------|
| Level 1 | Exp 1: Scoring matrix convergence | Eq. 4 — asymmetric Hebbian learning |
| Level 2 | Exp 2: Cross-graph discovery | Eqs. 6, 8a, 8b — two-stage discovery |
| Level 3 | Exp 3: Multi-domain scaling | I(n,t) = n(n-1)/2 x richness(t)^gamma |
| Sensitivity | Exp 4: Parameter sweep | Phase transitions in 4 dimensions |

**Phase 2 — Bridge Layer (Exp 5, A-E):** Validates the bridge layer mechanism (L2 centroid scoring, oracle feedback, gating, transfer, kernels, scaling).

| Experiment | What it validates | Gate |
|------------|-------------------|------|
| Exp 5: Oracle Fix | GT-aligned oracle + ratio sweep | PASS |
| Exp A: Capacity Ceiling | Shared W ceiling + G-matrix lift | FAIL (honest) |
| Exp C1: Centroid Oracle | L2 nearest-centroid diagnostic | PASS |
| Exp B1: Profile Scoring | ProfileScorer warm/cold + learning | PASS |
| Exp D1: Cross-Category Transfer | Transfer vs cold vs config warm start | Honest: not competitive |
| Exp D2: Factor Interactions | Pairwise factor MI gain | — |
| Exp E1: Kernel Generalization | L2 vs Cosine vs Mahalanobis vs Dot | — |
| Exp E2: Scale Test | Architecture scaling with categories/factors | PASS |

**Phase 3 — Synthesis Layer (EXP-S series):** Validates Eq. 4-synthesis — structured intelligence claims bias action selection when profiles are stale (cold-start + pre-campaign design).

| Experiment | What it validates | Gate | Status |
|------------|-------------------|------|--------|
| EXP-S1: Bias Accuracy | Synthesis improves acc when profiles are stale | improvement≥3pp, p<0.05, ECE_delta≤0.02 | In progress |
| EXP-S2: Poisoning Resilience | 20%/40% poisoned claims degrade accuracy ≤2pp | degradation≤2pp, safety_eff≥0.50 | Awaiting S1 |
| EXP-S3: Loop Independence | Centroid drift under repeated sigma activation | drift≤0.01, freq_effect<2pp | Awaiting S1 |
| EXP-S4: Lambda Sensitivity | Stable plateau exists for lambda tuning | plateau_width≥0.05 | PASS (width=0.300) |

**Validation Experiments (addressing independent reviewer concerns):**

| Experiment | What it validates |
|------------|-------------------|
| V1A: Extended Scaling | Wider range [2-15] domains, revised b=2.11 |
| V1B: Norm Tracking | LayerNorm necessity (2.9Mx growth without) |
| V2: Push Stability | Centroid update clipping requirement |
| V3A: Baseline Comparison | L2 centroid vs XGBoost/RF/LR/KNN |
| V3B: Calibration | ECE analysis, tau=0.1 fix |

**Normalization Ablation (Exp 2 extension):**

| Pipeline | F1 | x random |
|----------|-----|----------|
| raw | 0.000 | 0x |
| zscore | 0.000 | 0x |
| l2 | 0.002 | 3x |
| zscore_l2 | 0.071 | 145x |

---

## Directory Tree

```
cross-graph-experiments/
├── CLAUDE.md                              # Project guidelines and rules
├── EXPERIMENTS.md                         # Detailed experiment specifications
├── experiments_project_structure.md       # This file
├── configs/
│   └── default.yaml                       # Central configuration (352 lines, no magic numbers)
│
├── src/
│   ├── __init__.py
│   ├── data/                              # Synthetic data generation with ground truth
│   │   ├── __init__.py
│   │   ├── alert_generator.py            # SOC alert generation (Exp 1-4)
│   │   ├── category_alert_generator.py   # 5-category action-conditional alerts (Exp 5+)
│   │   │                                 #   + generate_campaign()   — campaign-period GT shift
│   │   │                                 #   + generate_precampaign() — suppressed GT for cold-start gap
│   │   ├── claim_generator.py            # Synthesis claims: correct + poisoned (EXP-S series)
│   │   └── entity_generator.py           # Entity embeddings for Exp 2-4
│   ├── models/                            # Core mechanisms being tested
│   │   ├── __init__.py
│   │   ├── scoring_matrix.py             # Eq. 4: P(action|alert) = softmax(f*W^T / tau)
│   │   ├── cross_attention.py            # Eqs. 5-8: cross-graph attention & discovery
│   │   ├── oracle.py                     # BernoulliOracle + GTAlignedOracle (Exp 5+)
│   │   ├── profile_scorer.py             # L2 nearest-centroid scorer + synthesis bias support
│   │   ├── synthesis.py                  # SynthesisBias dataclass + SynthesisProjector protocol
│   │   ├── rule_projector.py             # RuleBasedProjector: claims -> SynthesisBias via rules
│   │   └── gating.py                     # Uniform/Hebbian/MI gating (Exp 7-9, A)
│   ├── eval/                              # Evaluation utilities
│   │   └── __init__.py
│   └── viz/                               # Publication-quality visualizations (300 DPI)
│       ├── bridge_common.py              # Shared COLORS, VIZ_DEFAULTS, setup_axes, save_figure
│       ├── synthesis_common.py           # save_results(), print_gate_result() for EXP-S
│       ├── exp1_charts.py                # Exp 1 — 4 figure types + LaTeX table
│       ├── exp1_blog_chart.py            # Exp 1 — simplified blog version
│       ├── exp2_charts.py                # Exp 2 — F1 bars + P-R curves
│       ├── exp3_charts.py                # Exp 3 — scaling with power-law overlay
│       ├── exp3_blog_chart.py            # Exp 3 — simplified blog version
│       ├── exp4_charts.py                # Exp 4 — 2x2 sensitivity panel
│       ├── exp5_charts.py                # Exp 5 — oracle accuracy, ratio sweep, warmup
│       ├── expA_charts.py                # Exp A — convergence, G heatmap, breakdown
│       ├── expB1_charts.py               # Exp B1 — warm/cold/centroid, LR heatmap
│       ├── expC1_charts.py               # Exp C1 — method comparison, confusion matrix
│       ├── expD1_charts.py               # Exp D1 — transfer matrix, convergence, speedup
│       ├── expD2_charts.py               # Exp D2 — MI single, interactions, augmentation
│       ├── expE1_charts.py               # Exp E1 — kernel ranking, per-category heatmap
│       └── expE2_charts.py               # Exp E2 — oracle scaling, cold/warm gap
│
├── experiments/
│   ├── exp1_scoring_convergence/
│   │   ├── run.py
│   │   └── results/
│   │       ├── convergence_data.csv      # 350 rows: 10 seeds x 5 methods x 7 checkpoints
│   │       └── weight_evolution.npz      # W matrix snapshots (compounding only)
│   │
│   ├── exp2_cross_graph_discovery/
│   │   ├── run.py
│   │   ├── run_normalization_ablation.py # Normalization pipeline ablation (raw/zscore/l2/zscore_l2)
│   │   ├── normalization_ablation_results.csv   # 240 rows
│   │   ├── normalization_summary.json
│   │   └── results/
│   │       ├── discovery_results.csv     # 1230 rows: full grid sweep
│   │       └── best_configs.json         # Optimal (theta, K) per method and domain pair
│   │
│   ├── exp3_multidomain_scaling/
│   │   ├── run.py                        # Original: 5 domain counts [2-6]
│   │   ├── run_extended.py               # Extended: 11 domain counts [2-15], b=2.11
│   │   ├── run_norm_tracking.py          # Norm explosion tracking without LayerNorm
│   │   ├── norm_tracking.csv             # 360 rows
│   │   ├── norm_tracking_summary.json
│   │   └── results/
│   │       ├── scaling_data.csv          # 50 rows: 10 seeds x 5 domain counts
│   │       ├── extended_scaling_data.csv # 110 rows: 10 seeds x 11 domain counts
│   │       └── extended_scaling_fit.json # b=2.11, CI [2.09, 2.14]
│   │
│   ├── exp4_sensitivity/
│   │   ├── run.py
│   │   └── results/
│   │       └── sensitivity_data.csv      # 105 rows across 4 parameter sweeps
│   │
│   ├── exp5_oracle_fix/
│   │   ├── run.py                        # Oracle fix + ratio sweep + warmup schedules
│   │   └── results/
│   │       ├── ratio_sweep.csv           # 1500 rows: 10 seeds x 5 oracles x 5 ratios x ...
│   │       ├── warmup_comparison.csv     # 180 rows: warmup schedule comparison
│   │       ├── oracle_comparison.csv
│   │       ├── fm1_analysis.csv
│   │       └── best_config.json
│   │
│   ├── expA_capacity_ceiling/
│   │   ├── run.py                        # W-only, W+G-static, W+G-learned, W-augmented, W-per-cat
│   │   └── results/
│   │       ├── accuracy_trajectories.csv # 600 rows
│   │       ├── g_matrices.csv
│   │       └── summary.json
│   │
│   ├── expB1_profile_scoring/
│   │   ├── run.py                        # ProfileScorer warm/cold + learning effect
│   │   └── results/
│   │       ├── accuracy_trajectories.csv # 3420 rows
│   │       └── summary.json
│   │
│   ├── expC1_centroid_oracle/
│   │   ├── run.py                        # L2/Cosine/Dot centroid classification
│   │   └── results/
│   │       ├── classification_results.csv # 100k rows
│   │       ├── confusion_matrices.json
│   │       └── summary.json
│   │
│   ├── expD1_cross_category_transfer/
│   │   ├── run.py                        # Cross-category centroid transfer
│   │   └── results/
│   │       ├── accuracy_trajectories.csv # 750 rows
│   │       ├── convergence_speed.csv
│   │       ├── transfer_matrix.csv
│   │       └── summary.json
│   │
│   ├── expD2_factor_interactions/
│   │   ├── run.py                        # Pairwise factor interaction MI
│   │   └── results/
│   │       ├── mi_single.csv
│   │       ├── mi_interaction.csv
│   │       ├── top_interactions.json
│   │       └── summary.json
│   │
│   ├── expE1_kernel_generalization/
│   │   ├── run.py                        # L2/Cosine/Mahalanobis/Dot x 3 distributions
│   │   └── results/
│   │       ├── phase1_oracle.csv         # 120 rows
│   │       ├── phase2_learning.csv       # 300 rows
│   │       ├── covariance_stats.csv      # 60 rows
│   │       └── summary.json
│   │
│   ├── expE2_scale_test/
│   │   ├── run.py                        # small/medium/large/xlarge scale configs
│   │   └── results/
│   │       ├── phase1_oracle.csv         # 40 rows
│   │       ├── phase2_learning.csv       # 480 rows
│   │       ├── phase3_separation.csv     # 50 rows
│   │       └── summary.json
│   │
│   ├── validation/
│   │   ├── run_push_stability.py         # V2: centroid update clipping analysis
│   │   ├── run_baseline_comparison.py    # V3A: L2 vs XGBoost/RF/LR/KNN
│   │   ├── run_calibration_analysis.py   # V3B: ECE + temperature calibration
│   │   ├── push_stability_results.csv    # 16k rows
│   │   ├── push_stability_summary.json
│   │   ├── baseline_static_results.csv   # 70 rows
│   │   ├── baseline_online_results.csv   # 240 rows
│   │   ├── baseline_summary.json
│   │   ├── calibration_results.csv       # 70 rows
│   │   └── calibration_summary.json
│   │
│   ├── synthesis/                         # Phase 3: Synthesis Layer experiments (EXP-S series)
│   │   ├── __init__.py
│   │   ├── expS1_bias_accuracy/
│   │   │   ├── run.py                    # Cold-start + pre-campaign; 3-condition design
│   │   │   ├── charts.py                 # accuracy_by_lambda, action_shift, category_heatmap, ECE
│   │   │   └── results.json             # Gate: improvement_pp, p_value, ece_degradation
│   │   ├── expS2_poisoning/
│   │   │   ├── run.py                    # 0%/20%/40% poisoned claims; campaign alerts
│   │   │   ├── charts.py
│   │   │   └── results.json
│   │   ├── expS3_loop_independence/
│   │   │   ├── run.py                    # Repeated sigma activation; centroid drift
│   │   │   ├── charts.py
│   │   │   └── results.json
│   │   └── expS4_lambda_sensitivity/
│   │       ├── run.py                    # Lambda sweep [0.0-0.5]; plateau detection
│   │       ├── charts.py
│   │       └── results.json             # GATE PASS: plateau_width=0.300
│   │
│   └── exp3_scaling/                     # (stale/empty — superseded by exp3_multidomain_scaling)
│       └── results/
│
├── paper_figures/                         # All publication outputs (PDF + PNG, 300 DPI)
│   ├── exp1_convergence.{pdf,png}
│   ├── exp1_window_accuracy.{pdf,png}
│   ├── exp1_per_action.{pdf,png}
│   ├── exp1_weight_evolution.{pdf,png}
│   ├── exp1_blog_convergence.{pdf,png}
│   ├── exp1_table.tex
│   ├── exp2_f1_comparison.{pdf,png}
│   ├── exp2_precision_recall.{pdf,png}
│   ├── exp2_table.tex
│   ├── exp3_scaling.{pdf,png}
│   ├── exp3_blog_scaling.{pdf,png}
│   ├── exp3_table.tex
│   ├── exp4_sensitivity.{pdf,png}
│   ├── exp4_table.tex
│   ├── exp5_oracle_accuracy.{pdf,png}
│   ├── exp5_oracle_accuracy_best_ratio.{pdf,png}
│   ├── exp5_ratio_sweep_accuracy.{pdf,png}
│   ├── exp5_warmup_comparison.{pdf,png}
│   ├── exp5_category_heatmap.{pdf,png}
│   ├── exp5_fm1_boundary.{pdf,png}
│   ├── exp5_w_entropy_trajectory.{pdf,png}
│   ├── exp5_w_stability.{pdf,png}
│   ├── expA_convergence.{pdf,png}
│   ├── expA_final_accuracy.{pdf,png}
│   ├── expA_category_breakdown.{pdf,png}
│   ├── expA_g_heatmap.{pdf,png}
│   ├── expB1_warm_vs_cold_vs_centroid.{pdf,png}
│   ├── expB1_lr_heatmap.{pdf,png}
│   ├── expB1_noise_robustness.{pdf,png}
│   ├── expB1_profile_drift.{pdf,png}
│   ├── expB1_comparison_waterfall.{pdf,png}
│   ├── expC1_method_comparison.{pdf,png}
│   ├── expC1_category_breakdown.{pdf,png}
│   ├── expC1_confusion_heatmap.{pdf,png}
│   ├── expC1_comparison_waterfall.{pdf,png}
│   ├── expD1_transfer_matrix.{pdf,png}
│   ├── expD1_convergence.{pdf,png}
│   ├── expD1_speedup.{pdf,png}
│   ├── expD1_delta_summary.{pdf,png}
│   ├── expD2_single_mi.{pdf,png}
│   ├── expD2_interaction_gain.{pdf,png}
│   ├── expD2_top_interactions.{pdf,png}
│   ├── expD2_augmentation.{pdf,png}
│   ├── expE1_kernel_ranking.{pdf,png}
│   ├── expE1_kernel_x_distribution.{pdf,png}
│   ├── expE1_dot_vs_l2.{pdf,png}
│   ├── expE1_mahalanobis_vs_l2.{pdf,png}
│   ├── expE1_mixed_scale_impact.{pdf,png}
│   ├── expE1_per_category_heatmap.{pdf,png}
│   ├── expE1_learning_curves.{pdf,png}
│   ├── expE1_gae_recommendation.{pdf,png}
│   ├── expE2_oracle_scaling.{pdf,png}
│   ├── expE2_learning_curves.{pdf,png}
│   ├── expE2_cold_vs_warm_gap.{pdf,png}
│   ├── expE2_decisions_per_centroid.{pdf,png}
│   ├── expE2_scaling_trend.{pdf,png}
│   ├── expE2_separation_vs_accuracy.{pdf,png}
│   ├── expS1_accuracy_by_lambda.png
│   ├── expS1_action_shift.png
│   ├── expS1_category_heatmap.png
│   ├── expS1_ece_by_lambda.png
│   ├── expS2_poisoning_accuracy.png
│   ├── expS2_safety_effectiveness.png
│   ├── expS2_seed_distribution.png
│   ├── expS3_centroid_trajectory.png
│   ├── expS3_centroids_alone_accuracy.png
│   ├── expS3_frobenius_divergence.png
│   ├── expS4_accuracy_vs_lambda.png
│   └── expS4_per_category_optimal.png
│
└── notebooks/                             # Placeholder
```

---

## Source Files

### `src/data/alert_generator.py`

Generates synthetic SOC alerts for Experiment 1.

**Key types:**
- `Alert` (dataclass): `alert_id`, `alert_type`, `factors[6]`, `ground_truth_action`, `is_noisy`
- `AlertGenerator`: `generate(n, seed)` -> reproducible list of alerts

**Alert model:**
- 6 alert types (`false_positive`, `routine_alert`, `suspicious_login`, `data_exfil`, `brute_force`, `insider_threat`), each with a Beta-distributed factor profile
- 4 ground-truth actions: `auto_close`, `enrich_and_watch`, `escalate_tier2`, `escalate_incident`
- Noise: 3-10% of alerts get wrong action labels

---

### `src/data/category_alert_generator.py`

Category-aware SOC alert generator for Bridge Layer Experiments (Exp 5+).

**Key types:**
- `CategoryAlert` (dataclass): category, factors[6], ground_truth_action, noise flag
- `CategoryAlertGenerator`: generates alerts conditioned on both category AND action

**Design — action-conditional profiles:**
- 5 categories: `credential_access`, `threat_intel_match`, `lateral_movement`, `data_exfiltration`, `insider_threat`
- 4 actions: `auto_close`, `escalate_tier2`, `enrich_and_watch`, `escalate_incident`
- 6 factors: `travel_match`, `asset_criticality`, `threat_intel`, `time_anomaly`, `device_trust`, `pattern_history`
- Factors sampled from N(mu[category][gt_action], factor_sigma) — each action has orthogonal primary factors for linear separability
- All profile data comes from `configs/default.yaml` (nothing hardcoded)
- Module-level constants: `CATEGORIES`, `ACTIONS`, `FACTORS`

---

### `src/data/entity_generator.py`

Generates unit-norm entity embeddings for Experiments 2-4.

**Key types:**
- `Entity` (dataclass): `entity_id`, `domain`, `embedding[d]`
- `EntityGenerator`: `generate_domain(name, n, seed)`, `generate_all(seed)`
- `inject_signals(entities_i, entities_j, n_signals, signal_strength, seed)` — plants ground-truth correlations via shared embedding dimensions

**Embedding layout (64-dim default):**

| Dims | Content |
|------|---------|
| 0-5 | Domain-specific semantics — N(domain_mean, sigma=0.30) |
| 6-9 | Geographic cluster signal (soft one-hot) |
| 10-13 | Temporal bucket signal (soft one-hot) |
| 14-63 | Background noise — N(0, sigma=0.05) |

**Domain profiles (entities per domain):**
- `security`: 200 entities
- `decision_history`: 300 entities
- `threat_intel`: 200 entities
- `network_flow`, `asset_inventory`, `user_behavior`: 200 each (Exp 3 extras)

---

### `src/models/scoring_matrix.py`

Implements **Eq. 4**: `P(action|alert) = softmax(f * W^T / tau)`

**Key type:** `ScoringMatrix`

| Parameter | Default | Role |
|-----------|---------|------|
| `n_actions` | 4 | Number of actions |
| `n_factors` | 6 | Alert factor dimensions |
| `temperature tau` | 0.25 | Softmax sharpness |
| `alpha_correct` | 0.002 | Hebbian reward step |
| `alpha_incorrect` | 0.04 | Hebbian penalty step (20x alpha_correct) |
| `weight_clamp` | 5.0 | Prevents unbounded growth |
| `decay_rate` | 0.001 | Inverse-time LR decay |

**Asymmetric Hebbian update rule:**
```
if correct:   W[action] += alpha_correct   * lr(t) * factors
if incorrect: W[action] -= alpha_incorrect * lr(t) * factors

lr(t) = 1 / (1 + decay_rate * t)
```

The 20:1 asymmetry drives rapid specialization; decay stabilizes learning over time.

---

### `src/models/cross_attention.py`

Implements **Eqs. 5-8**: cross-graph attention and entity pair discovery.

**Key type:** `CrossGraphAttention`

| Method | Equation | Description |
|--------|----------|-------------|
| `compute_logits(E_i, E_j)` | Eq. 5 | `S = E_i @ E_j.T / sqrt(d)` |
| `compute_attention(S)` | Eq. 6 | `A = softmax(S, axis=1)` (rows sum to 1) |
| `compute_output(A, V_j)` | Eq. 6 | `O = A @ V_j` |
| `discover_two_stage(E_i, E_j, theta, K)` | Eqs. 8a+8b | Stage 1 intersection Stage 2 |
| `discover_logit_only(E_i, E_j, theta)` | Eq. 8a | Pre-softmax threshold only |
| `discover_topk_only(E_i, E_j, K)` | Eq. 8b | Top-K softmax only |
| `cosine_baseline(E_i, E_j, threshold)` | — | Raw cosine (no sqrt(d) scaling) |

**Two-stage discovery logic:**
```
Stage 1 (Eq. 8a): keep (k, l) where S[k, l] > theta_logit
Stage 2 (Eq. 8b): keep (k, l) where l in top-K(softmax(S[k, :]))
Result:           intersection of Stage 1 and Stage 2
```

---

### `src/models/oracle.py`

Oracle implementations for Bridge Layer Experiments (Exp 5+).

**Key types:**

- `BernoulliOracle` — Legacy oracle (R1 problem). Outcome drawn from Bernoulli(category_rate) *independently* of action correctness. Converges to category-level bias, not ground truth.
- `GTAlignedOracle` — R1 fix. Outcome is +1 iff action matches ground truth, else -1. A `noise_rate` fraction of outcomes is randomly flipped to model analyst feedback noise.

---

### `src/models/profile_scorer.py`

L2 nearest-centroid scorer with online centroid update and synthesis bias support.

**Eq. 4-synthesis:** `P(a|f,c,σ) = softmax(-(||f-mu[c,a,:]||^2 + λ·σ[c,a]) / tau)`

When `synthesis=None` or `lambda_coupling=0`, reduces exactly to Eq. 4''.

**Key type:** `ProfileScorer`

| Parameter | Role |
|-----------|------|
| `tau` | Softmax temperature (lower = sharper). Never modified by synthesis. |
| `eta` | Learning rate for correct decisions (pull centroid toward f) |
| `eta_neg` | Learning rate for incorrect decisions (push centroid away) |

**Online update:** Asymmetric centroid pull/push with count-based decay. Warm start from configured profiles or cold start from uniform 0.5.

**Synthesis path:** `score(f, cat, synthesis)` — adds `lambda * sigma[c,:]` to distances before softmax. `tau` is unchanged (tau_modifier was removed; coupling temperature to claims degraded ECE). σ is NEVER passed to `update()` — centroids learn from experience only.

---

### `src/models/synthesis.py`

`SynthesisBias` frozen dataclass — immutable synthesis state passed to `score()`.

| Field | Description |
|-------|-------------|
| `sigma` | `(n_categories, n_actions)` — awareness bias tensor. σ[c,a]<0 → action more likely. |
| `active_claims` | Claims that passed extraction threshold. 0 = no effect. |
| `lambda_coupling` | Coupling strength λ. λ=0.0 → exact kill switch (guaranteed by softmax). |

**`SynthesisBias.neutral(n_cat, n_act)`** — zero sigma, λ=0, equivalent to no synthesis.
**`SynthesisBias.is_active`** — True iff λ>0 and active_claims>0.
**`SynthesisBias.effective_shift(cat_idx)`** — returns λ·σ[c,:] for auditing.

Also defines `SynthesisProjector` protocol for `RuleBasedProjector` and future `LearnedProjector`.

---

### `src/models/rule_projector.py`

`RuleBasedProjector` — maps a list of claims to a `SynthesisBias` via domain-configured rule templates.

**`project(claims, lambda_coupling)`** → `SynthesisBias`

Pipeline per claim:
1. Filter: `confidence * extraction_confidence < extraction_threshold (0.8)` → skip
2. Decay: `weight = confidence * exp(-rate * age_days)` where `rate` depends on `decay_class`
3. Accumulate: for each `(category, action)` pair in the claim's scope, `sigma[c,a] += direction * weight`
4. Clip: `sigma = clip(sigma, -sigma_max, +sigma_max)` (default sigma_max=1.0)

Also provides `project_with_trace()` returning `(bias, trace)` for Tab 5 audit display.

---

### `src/data/claim_generator.py`

Synthetic claims for EXP-S validation. Schema matches `ContextConnectors` (F13 at v6.5+).

**Key constants:**
- `SOC_SYNTHESIS_RULES` — 6 rule templates mapping `{action: direction}`. Negative direction = action more likely.
- `CATEGORIES` — 5 categories matching `CategoryAlertGenerator`
- `ACTIONS` — 4 actions: `["auto_close", "escalate_tier2", "enrich_and_watch", "escalate_incident"]`
- `CORRECT_TYPE_CATEGORY_MAP` — which categories each claim type semantically affects
- `POISON_TYPE_CATEGORY_MAP` — wrong-category assignments for adversarial claims

**Generators:**
- `generate_correct_claims(n, seed)` — n=20 claims, deterministic category coverage (`n_cats = len(possible_cats)`), confidence ∈ U(0.7,1.0), extraction ∈ U(0.85,1.0)
- `generate_poisoned_claims(n_correct, n_poison, seed)` — mixes correct + adversarial claims; poisoned claims apply rule types to wrong categories

**Known limitation:** Claims with `confidence * extraction_confidence < 0.8` are filtered by `RuleBasedProjector`. At ~2% of seeds this can produce zero-sigma for `threat_intel_match` (only reachable via `active_campaign`). Under investigation — do not raise extraction floor without running S1 first.

---

### `src/models/gating.py`

Gating mechanisms for Bridge Layer Experiments (Exp 5-9, A).

Three mechanisms control per-category, per-factor attention weights G[c, i]:

| Mechanism | Description |
|-----------|-------------|
| `UniformGating` | Baseline: G = ones, no selective attention, no learning |
| `HebbianGating` | Online learning from oracle outcomes (Eq. 4d). Gate values grow/shrink based on factor x weight-magnitude signals. Optional damping. |
| `MIGating` | Offline mutual-information estimation. Fitted once on a batch via `fit_from_data()`; gate values are then static. |

---

### `src/viz/bridge_common.py`

Shared visualization utilities for all bridge layer experiments.

**Exports:** `COLORS` dict, `VIZ_DEFAULTS` dict, `setup_axes()`, `save_figure()` (saves both PDF + PNG at 300 DPI).

Used by `exp5_charts.py` through `expE2_charts.py` for consistent styling.

---

## Configuration: `configs/default.yaml`

Single source of truth for all experiment parameters — no magic numbers in code. 352 lines covering:

### Experiments 1-4 (Core Framework)

**Experiment 1:** `n_alerts`: 5000, `noise_rate`: 0.03, checkpoints [50-5000], 5 baselines
**Experiment 2:** 3 domains, `embedding_dim`: 64, `signal_strength`: 8.0, theta/K grids
**Experiment 3:** `domain_counts`: [2-6], `entities_per_domain`: 200, fixed theta=0.02, K=3
**Experiment 4:** 4 parameter sweeps (asymmetry, temperature, noise, embedding_dim)

### Bridge Layer Common

- 5 categories, 4 actions, 6 factors (all named)
- `category_gt_distributions`: per-category action probability distributions
- `action_conditional_profiles`: 5x4 matrix of 6-factor mean vectors (orthogonal primary factors)
- Scoring defaults: tau=0.25, alpha_correct=0.02, alpha_incorrect=0.04, weight_clamp=5.0

### Profile Variants (for Exp A)

- `simplified_profiles`: Truly orthogonal, identical across all categories. Each action owns 1-2 factors (0.85), all others 0.10. Guarantees linear separability.
- `realistic_profiles`: Category-varying cross-conflicts. Same factor maps to different actions in different categories. Exceeds shared W capacity.

### Experiments 5-9

- **Exp 5:** 1000 decisions, 5 oracle configs (bernoulli + GT at 0/5/15/30% noise), ratio sweep
- **Exp 6:** MI between categories and factors (500 decisions)
- **Exp 7:** 4 gating configs (uniform, hebbian damped/undamped, mi_static)
- **Exp 8:** Alpha-g ratio sensitivity + damping
- **Exp 9:** Hidden factor detection (insider_threat)

### Exp A: Capacity Ceiling

- Scoring override: ratio=1 (symmetric), alpha=0.06 for viable cold-start
- Gating: hebbian (lr=0.005, damping, max_gate=2.0) + MI (threshold=0.1, fit_decisions=200)
- Gate thresholds: VA.1 > 80%, VA.3 < 55%, VA.6 gap >= 25pp

### Visualization Defaults

- DPI: 300, formats: PDF + PNG
- Colors: `main=#1E3A5F`, `baseline_fixed=#94A3B8`, `discovery=#D97706`
- Font sizes: title=13, label=11, tick=9, annotation=8.5

---

## Experiments

### Experiment 1: Scoring Matrix Convergence

**Runner:** `experiments/exp1_scoring_convergence/run.py`
**Outputs:** `convergence_data.csv` (350 rows), `weight_evolution.npz`

Validates Eq. 4 — asymmetric Hebbian learning specializes W to correct SOC actions.

**Setup:** 10 seeds x 5 methods x 7 checkpoints

**Key result:** `compounding` reaches ~69-71% cumulative accuracy at 5000 alerts vs. 25% random baseline.

---

### Experiment 2: Cross-Graph Discovery

**Runner:** `experiments/exp2_cross_graph_discovery/run.py`
**Outputs:** `discovery_results.csv` (1230 rows), `best_configs.json`

Validates Eqs. 6, 8a, 8b — two-stage entity pair discovery across domain graphs.

**Setup:** 10 seeds x 5 methods x 3 domain pairs x config grids

**Key result:** `two_stage` achieves ~116x F1 above the random baseline at optimal (theta, K).

**Extension — Normalization Ablation:**
**Runner:** `experiments/exp2_cross_graph_discovery/run_normalization_ablation.py`
**Outputs:** `normalization_ablation_results.csv` (240 rows), `normalization_summary.json`

Tests four normalization pipelines (raw, zscore, l2, zscore_l2). Finding: both z-score AND l2 normalization are required synergistically (F1=0.071 vs 0.000 without). Uses `gen._raw_matrix()` to bypass `generate_domain()` normalization for clean ablation.

---

### Experiment 3: Multi-Domain Scaling

**Runner:** `experiments/exp3_multidomain_scaling/run.py`
**Outputs:** `scaling_data.csv` (50 rows)

Validates the quadratic scaling law `I(n,t) = n(n-1)/2 x richness(t)^gamma`.

**Setup:** 10 seeds x 5 domain counts (2-6)

**Key result (original):** R^2 = 0.9995 fit to `discoveries ~ n^2.30`.

**Extension — Extended Scaling (V1A):**
**Runner:** `experiments/exp3_multidomain_scaling/run_extended.py`
**Outputs:** `extended_scaling_data.csv` (110 rows), `extended_scaling_fit.json`

Wider range [2-15] domains (11 points). Revised exponent: **b=2.11**, 95% CI [2.09, 2.14]. Both n^2.0 and n^2.3 are outside CI — superquadratic confirmed but original was overfit.

**Extension — Norm Tracking (V1B):**
**Runner:** `experiments/exp3_multidomain_scaling/run_norm_tracking.py`
**Outputs:** `norm_tracking.csv` (360 rows), `norm_tracking_summary.json`

Tracks embedding norm growth without LayerNorm across attention sweeps. Finding: catastrophic norm explosion (~40x per sweep after sweep 1, reaching 2.9Mx by sweep 5). Confirms reviewer concern: Eq. 13 without LayerNorm is unstable.

---

### Experiment 4: Parameter Sensitivity

**Runner:** `experiments/exp4_sensitivity/run.py`
**Outputs:** `sensitivity_data.csv` (105 rows)

Sweeps 4 parameters to locate optimal values and phase transitions.

| Sweep | Best value | Finding |
|-------|-----------|---------|
| A (asymmetry) | ratio = 20 | 0.657 accuracy |
| B (temperature) | tau = 0.25 | 0.657 accuracy |
| C (noise) | < 5% | Sharp degradation above this rate |
| D (embedding_dim) | d = 128 | F1 collapses at d = 256 |

---

### Experiment 5: Oracle Fix (Bridge Layer)

**Runner:** `experiments/exp5_oracle_fix/run.py`
**Outputs:** `ratio_sweep.csv` (1500 rows), `warmup_comparison.csv` (180 rows), `best_config.json`, `oracle_comparison.csv`, `fm1_analysis.csv`

Validates GT-aligned oracle fix and asymmetric learning ratio sweep.

**Setup:** Section A: 10 seeds x 5 oracle configs x 5 ratios [1,1.5,2,3,5] x 1000 decisions. Section B: 10 seeds x 3 warmup schedules.

**Key results:** V5.1: 79.65% (>75%) | V5.2: +26.09pp over Bernoulli | V5.3: 73.86% (>55%) | V5.4: ratio=5>3>2>1.5>1 monotonic. **GATE: PASS.**

---

### Experiment A: Capacity Ceiling

**Runner:** `experiments/expA_capacity_ceiling/run.py`
**Outputs:** `accuracy_trajectories.csv` (600 rows), `g_matrices.csv`, `summary.json`

Tests whether a gating matrix G can lift accuracy above the shared scoring matrix W ceiling.

**Scoring configs:** `w_only`, `w_g_static`, `w_g_learned`, `w_augmented` (one-hot category appended), `w_per_category` (5 separate W matrices)

**Key results:** Simplified W-only = 87.79% | Realistic W-only = 49.26% | G-lift = +2.35pp (threshold: 8pp). **GATE: FAIL (honest).** Root cause: data-per-category bottleneck (~200 decisions per W_c).

---

### Experiment C1: Centroid Oracle Diagnostic

**Runner:** `experiments/expC1_centroid_oracle/run.py`
**Outputs:** `classification_results.csv` (100k rows), `confusion_matrices.json`, `summary.json`

Diagnostic: can L2 nearest-centroid classification solve the task if given perfect centroids?

**Key results:** L2 = 97.89% +/- 0.14% | Cosine = 96.42% | Dot = 61.00%. All 5 categories >95% (lateral_movement hardest at 95.0%). Gap vs Exp A shared W: +48.6pp. **GATE: PASS.** Verdict: data has signal, learning is the bottleneck.

---

### Experiment B1: Profile Scoring + Learning

**Runner:** `experiments/expB1_profile_scoring/run.py`
**Outputs:** `accuracy_trajectories.csv` (3420 rows), `summary.json`

Tests ProfileScorer with warm/cold start and online centroid learning.

**Key results:** centroid_only = 98.0% | profile_warm (noise=0) = 98.2% | profile_cold = 90.7% | profile_warm (noise=0.30) = 98.1%. Learning effect: +0.2pp (neutral). Cold start recovers from 58.5% (t=100) to 90.7% (t=1000). **GATE: PASS.**

---

### Experiment D1: Cross-Category Transfer

**Runner:** `experiments/expD1_cross_category_transfer/run.py`
**Outputs:** `accuracy_trajectories.csv` (750 rows), `convergence_speed.csv`, `transfer_matrix.csv`, `summary.json`

Tests whether centroids learned in one category transfer to others.

**Key results:** Transfer competitive with config (>= config-2pp): 1/5 categories. Transfer adds 0-13pp over cold but is 2-14pp below config warm start. **Verdict: transfer NOT competitive.** Config warm start dominates. Transfer useful only when no expert profiles available.

---

### Experiment D2: Factor Interaction Discovery

**Runner:** `experiments/expD2_factor_interactions/run.py`
**Outputs:** `mi_single.csv`, `mi_interaction.csv`, `top_interactions.json`, `summary.json`

Tests whether pairwise factor interactions (f_i x f_j) contain additional discriminative information beyond individual factors.

**Setup:** Phase 1: single-factor and interaction MI across seeds (75 pairs x 5 categories). Phase 2: if interactions have gain > 1.5, test augmented ProfileScorer.

---

### Experiment E1: Kernel Generalization

**Runner:** `experiments/expE1_kernel_generalization/run.py`
**Outputs:** `phase1_oracle.csv` (120 rows), `phase2_learning.csv` (300 rows), `covariance_stats.csv` (60 rows), `summary.json`

Tests 4 distance kernels (L2, Cosine, Mahalanobis, Dot) across 3 factor distributions (original, normalized, mixed_scale).

**Key results:**
- Original: L2=97.9% > Mahalanobis=97.7% > Cosine=96.4% > Dot=61.0%
- Mixed scale: Mahalanobis=92.9% > L2=79.9% > Cosine=61.2% > Dot=41.9%
- Normalization fixes dot product: 61.0% -> 90.8%

**Recommendation:** Pluggable kernels with L2 default. Mahalanobis for scale-diverse data.

---

### Experiment E2: Scale Test

**Runner:** `experiments/expE2_scale_test/run.py`
**Outputs:** `phase1_oracle.csv` (40 rows), `phase2_learning.csv` (480 rows), `phase3_separation.csv` (50 rows), `summary.json`

Tests architecture scaling across 4 configs (small=5cat/6fac, medium=10/12, large=20/24, xlarge=30/30).

**Key results:** Oracle accuracy IMPROVES with scale (97.9% -> 99.9%). Cold start degrades (89.9% -> 72.7%). Warm start essential at scale. **GATE: PASS** (degradation < 5pp).

---

### Validation Experiments

**V2 — Push Update Stability:**
**Runner:** `experiments/validation/run_push_stability.py`
**Outputs:** `push_stability_results.csv` (16k rows), `push_stability_summary.json`

Tests centroid update bounds under normal, bad-streak, and worst-case conditions. **Fix:** `np.clip(mu, 0, 1)` after every update. Margin guard alone is insufficient.

**V3A — Baseline Comparison:**
**Runner:** `experiments/validation/run_baseline_comparison.py`
**Outputs:** `baseline_static_results.csv` (70 rows), `baseline_online_results.csv` (240 rows), `baseline_summary.json`

Static: L2=94.78% > RF=93.20% > LR=92.38% > GBT=92.24%, KNN=94.54%. L2 within 0.24pp of KNN, beats all tree/linear baselines. Online: L2 leads by 1.4-2.8pp throughout (more data-efficient).

**V3B — Calibration Analysis:**
**Runner:** `experiments/validation/run_calibration_analysis.py`
**Outputs:** `calibration_results.csv` (70 rows), `calibration_summary.json`

L2 centroid at tau=0.25: ECE=0.1897 (poor, underconfident). **Fix:** tau=0.1 -> ECE=0.0363 (well calibrated).

---

## Reproducibility

**Fixed seeds (all experiments):** `[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]`

**Data flow (Exp 2 example):**
```
configs/default.yaml
    -> EntityGenerator.generate_all(seed)
    -> inject_signals(sec, threat, n=20, strength=8.0)
    -> CrossGraphAttention.discover_two_stage(theta, K)
    -> compute P/R/F1 vs. ground truth
    -> one row in discovery_results.csv
```

**Data flow (Bridge Layer example):**
```
configs/default.yaml
    -> CategoryAlertGenerator(seed)
    -> ProfileScorer(profiles from config, tau, eta, eta_neg)
    -> GTAlignedOracle(noise_rate)
    -> decide(factors, category) -> action
    -> oracle.evaluate(action, gt_action) -> outcome
    -> scorer.update(category, action, factors, outcome)
    -> accuracy_trajectories.csv
```

**Each source module has a built-in `__main__` self-test** that validates data properties (distributions, norms, update directions, attention row sums, etc.).

---

## Performance Summary

| Experiment | Key metric | Value | Baseline |
|------------|-----------|-------|----------|
| Exp 1 | Cumulative accuracy at 5K alerts | ~69.4% | 25% (random) |
| Exp 2 | Best F1 | ~116x above random | ~0.025 F1 |
| Exp 2 (norm ablation) | zscore_l2 F1 | 0.071 (145x) | 0x (raw) |
| Exp 3 | Power-law exponent b | 2.11 (CI: 2.09-2.14) | — |
| Exp 4 | Critical noise threshold | ~5% | — |
| Exp 5 | GT oracle accuracy | 79.65% | 53.56% (Bernoulli) |
| Exp A | G-lift over W-only | +2.35pp | Threshold: 8pp (FAIL) |
| Exp C1 | L2 centroid accuracy | 97.89% | 61% (Dot) |
| Exp B1 | Warm start (noise=0.30) | 98.1% | 90.7% (cold) |
| Exp D1 | Transfer vs config gap | -2 to -14pp | Config dominates |
| Exp E1 | L2 accuracy (original) | 97.9% | 61% (Dot) |
| Exp E2 | Oracle degradation (small->xlarge) | -2pp | Threshold: 5pp (PASS) |
| V3A | L2 vs best ML baseline (KNN) | -0.24pp | Competitive |
| V3B | Calibrated ECE (tau=0.1) | 0.036 | 0.190 (tau=0.25) |
| EXP-S4 | Lambda plateau width | 0.300 (PASS) | Threshold: 0.05 |
| EXP-S1 | improvement_pp (best λ=0.2) | 2.30pp (in progress) | Threshold: 3.0pp |
| EXP-S1 | ECE_delta | -0.022 (PASS) | Threshold: ≤0.02 |
| EXP-S1 | Scalar ceiling (PERFECT_SIGMA, λ=0.3) | +9.27pp | Architecture sound |

---

## Equations Validated

| Equation | Description | Validator |
|----------|-------------|-----------|
| **Eq. 4** | `P(action\|alert) = softmax(f*W^T / tau)` | Convergence to >69% accuracy (Exp 1) |
| **Eq. 4''** | `P(a\|f,c) = softmax(-\|\|f - mu\|\|^2 / tau)` | L2 centroid 97.89% (Exp C1) |
| **Eq. 5** | `S_ij = E_i*E_j^T / sqrt(d)` | Logit shape and values (Exp 2) |
| **Eq. 6** | `A = softmax(S, axis=1)`, `O = A@V` | Row sums = 1, output shape (Exp 2) |
| **Eq. 8a** | `s_kl > theta_logit` | Stage 1 filtering (Exp 2) |
| **Eq. 8b** | `entity_l in top-K(softmax(S_k,:))` | Stage 2 filtering (Exp 2) |
| **Scaling** | `I(n,t) = n(n-1)/2 x richness(t)^gamma` | b=2.11, CI [2.09, 2.14] (Exp 3 ext) |

---

### EXP-S4: Lambda Sensitivity

**Runner:** `experiments/synthesis/expS4_lambda_sensitivity/run.py`
**Outputs:** `results.json`

Sweeps λ ∈ [0.0, 0.5] (17 values) across 10 seeds. Measures `improvement_pp = acc_with_sigma - acc_gap`. Finds plateau where improvement ≥ 2pp.

**Design:** Same cold-start + pre-campaign 3-condition design as S1.

**Key result:** plateau_width=0.300, plateau=[0.200, 0.500], peak_improvement=7.27pp at λ=0.500. **GATE: PASS** (plateau_width ≥ 0.05).

---

### EXP-S1: Synthesis Bias Accuracy

**Runner:** `experiments/synthesis/expS1_bias_accuracy/run.py`
**Outputs:** `results.json`

Tests whether Eq. 4-synthesis improves accuracy when profiles are stale.

**3-condition design per seed:**
- Phase 1: Cold-start ProfileScorer (`mu = 0.5` uniform), train on 400 pre-campaign alerts where the campaign action is suppressed to 5% GT probability
- Condition 2 (gap): Score 300 campaign alerts with stale profiles, no sigma
- Condition 3 (with sigma): Same alerts, sigma active from `generate_correct_claims(n=20)`

**Gate:** improvement_pp ≥ 3.0 AND p_value < 0.05 AND ECE_delta ≤ 0.02

**Current status (as of last run):**
- improvement_pp: 2.30pp (< 3pp — FAIL)
- p_value: 0.0356 (PASS)
- ECE_delta: -0.0215 (synthesis *improves* calibration — PASS, after tau_modifier removal)
- Scalar ceiling test (PERFECT_SIGMA): +9.27pp at λ=0.3 — architecture is sound
- Blocker: claim extraction threshold filtering causes ~6% zero-sigma rate across seeds

---

### EXP-S2: Poisoning Resilience

**Runner:** `experiments/synthesis/expS2_poisoning/run.py`
**Outputs:** `results.json`

Tests degradation under 20%/40% poisoned claims. Uses best λ from S1.

**Gate:** degradation_20pct ≤ 2pp AND safety_effectiveness ≥ 0.50. **Awaiting S1 gate.**

---

### EXP-S3: Loop Independence

**Runner:** `experiments/synthesis/expS3_loop_independence/run.py`
**Outputs:** `results.json`

Tests that sigma activation does not bleed into centroid updates (σ is NEVER passed to `update()`). Measures centroid drift and Frobenius divergence under repeated synthesis activation.

**Gate:** drift ≤ 0.01, freq_effect < 2pp. **Awaiting S1 gate.**

---

## Repo State

**Branch:** `main`

**Last committed:** `87b8e07` — publication-quality figures for 14 bridge + validation experiments

**Uncommitted modifications (M):**
| File | Change |
|------|--------|
| `configs/default.yaml` | Config updates |
| `docs.txt` | Documentation notes |
| `experiments_project_structure.md` | This file (in progress) |
| `src/data/category_alert_generator.py` | Added `generate_campaign()`, `generate_precampaign()`, `generate_alerts()` alias |
| `src/models/profile_scorer.py` | Added synthesis bias support; removed `tau_modifier` from temperature calculation |

**Untracked new files (??):**
| File/Directory | Description |
|----------------|-------------|
| `experiments/__init__.py` | Package init |
| `experiments/synthesis/` | Full EXP-S series (S1-S4 runners, charts, results) |
| `paper_figures/expS1_*.png` | S1 charts (4 files) |
| `paper_figures/expS2_*.png` | S2 charts (3 files) |
| `paper_figures/expS3_*.png` | S3 charts (3 files) |
| `paper_figures/expS4_*.png` | S4 charts (2 files) |
| `src/data/claim_generator.py` | Synthesis claim generation |
| `src/models/synthesis.py` | `SynthesisBias` dataclass + protocol |
| `src/models/rule_projector.py` | `RuleBasedProjector` |
| `src/models/profile_scorer_synthesis_patch.py` | Patch verification script (test harness only) |
| `src/viz/synthesis_common.py` | `save_results()`, `print_gate_result()` |

**Key architectural change — `tau_modifier` removed:**
`SynthesisBias` no longer has a `tau_modifier` field. Previously, `tau_eff = tau * tau_modifier` caused ECE_delta=+0.138 when synthesis was active (calibration degradation). Removal yielded ECE_delta=-0.022 (calibration improves). Temperature is now always `self.tau` regardless of synthesis state. Any code constructing `SynthesisBias(..., tau_modifier=...)` will raise `TypeError`.

---

## Pending Experiments

Configured in `configs/default.yaml` but not yet run:

| Experiment | Description |
|------------|-------------|
| Exp 6 | MI between categories and factors |
| Exp 7 | Gating mechanisms (uniform, hebbian damped/undamped, mi_static) |
| Exp 8 | Alpha-g ratio sensitivity + damping |
| Exp 9 | Hidden factor detection (insider_threat) |
