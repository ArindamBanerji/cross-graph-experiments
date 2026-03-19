# Cross-Graph Experiments: Project Structure

## Overview

Experimental validation suite for the **Cross-Graph Attention** framework. The codebase has grown from the original 4 experiments into a comprehensive validation suite across five phases.

**Phase 1 — Core Framework (Exp 1-4):** Validates the cross-graph attention equations at three levels.

| Level | Experiment | What it validates |
|-------|------------|-------------------|
| Level 1 | Exp 1: Scoring matrix convergence | Eq. 4 — asymmetric Hebbian learning |
| Level 2 | Exp 2: Cross-graph discovery | Eqs. 6, 8a, 8b — two-stage discovery |
| Level 3 | Exp 3: Multi-domain scaling | I(n,t) = n(n-1)/2 × richness(t)^γ |
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

**Phase 3 — Synthesis Layer (EXP-S series):** Validates Eq. 4-synthesis — structured intelligence claims bias action selection.

| Experiment | What it validates | Gate | Status |
|------------|-------------------|------|--------|
| EXP-S1: Bias Accuracy (warm) | Synthesis improves acc when profiles stale | improvement≥3pp, p<0.05 | GATE FAIL (honest) |
| EXP-S1b: Degraded Profiles | Sigma value under cold/noisy conditions | improvement≥1pp | GATE FAIL (honest) |
| EXP-S1c: Dynamic Recovery | Recovery curve under sigma fade | recovery≥70% | — |
| EXP-S1d: GateOP Replication | Replication of gateop conditions | — | — |
| EXP-S2: Poisoning Resilience | 20%/40% poisoned claims degrade ≤2pp | degradation≤2pp, safety_eff≥0.50 | PASS |
| EXP-S2-REPRO: Three-Arm Replication | Frozen/production/realistic poisoning | Arm0≤2pp; Arm A p90 T_rec<100 | Arm0 PASS; Arm A FAIL; Arm B DOMAIN EXPERT REVIEW |
| EXP-S3: Loop Independence | Centroid drift under repeated sigma | drift≤0.05 (Frobenius), acc_diff≤1pp | PASS |
| EXP-S4: Lambda Sensitivity | Stable plateau exists for λ tuning | plateau_width≥0.05 | PASS (width=0.300) |
| EXP-S5a: Real Threat Intel | KEV/NVD claims → sigma pipeline | — | complete |
| EXP-S5b: Work Artifact Extraction | LLM claim extraction F1 vs template | F1≥0.70 | PASS (F1=0.877) |

**OP series (operator synthesis dynamics):** Validates temporal operator claim behavior.

| Experiment | What it validates |
|------------|-------------------|
| expOP1_final | Baseline operator integration (AUAC curves, T70) |
| expOP1_revised | Revised integration with cold-start benefit |
| expOP1_imperfect | Imperfect extraction (epsilon sweep) |
| expOP1_scalar_loop2 | Scalar loop overshoot analysis |
| expOP2_harmful | Harmful claim recovery trajectories (N=20) |
| expOP2_n100 | OP2 replicated at N=100 seeds — tight CI on never-recover rate |
| expOP3_residual | Residual decay + early-warning ROC |
| expOP_margin | Margin distribution per category |

**FX series (production-realistic validation):** Tests the full pipeline under realistic SOC data distributions.

| Experiment | What it validates | Gate |
|------------|-------------------|------|
| FX-1-PROXY-REAL | Realistic vs centroidal accuracy gap | FAIL (honest): acc<80%, gap>20pp |
| FX-1-LEARNING | Learning lift under realistic data | FAIL (honest): acc@1000=77.65% < 82% |
| expFX1_proxy_real | Real IOC factor distribution characterization (CISA KEV, NVD, ATT&CK) | DISTRIBUTION GAP DETECTED (all KL > 0.5) |
| FX-2: Noise Distributions | 3 bias patterns: post-incident, fatigue, expertise | PASS (all gates), DESIGN WARNING |
| FX-T5-BREAKDOWN | Auto-approve band action analysis | RESULT C: marginal |
| FX-DI07 | Decision interval discount sweep | complete |

**Statistical Pass:** 50-seed validation of all published claims.

| Claim | Value | Gate |
|-------|-------|------|
| T1 Static realistic acc | 71.7% [71.4%, 71.9%] | FAIL vs 80% gate |
| T2 Learning dec 1000 | 78.9% [78.1%, 79.6%] | FAIL vs 82% gate |
| T3 Learning dec 1500 | 79.4% [78.7%, 80.1%] | FAIL vs 82% gate |
| T5 Auto-approve acc ≥0.90 | 90.7% [90.1%, 91.2%] | PASS vs 90% gate |
| DI-06 dec1500 vs dec1000 delta | +0.530pp, CI [-0.548, +1.608] | UNRESOLVED (need 233+ seeds) |

**Validation Experiments (addressing independent reviewer concerns):**

| Experiment | What it validates | Finding |
|------------|-------------------|---------|
| V1A: Extended Scaling | Wider range [2-15] domains | b=2.11, CI [2.09, 2.14] |
| V1B: Norm Tracking | LayerNorm necessity | 2.9M× growth without LayerNorm |
| V2: Push Stability | Centroid update clipping requirement | clip(mu,0,1) required; margin alone insufficient |
| V3A: Baseline Comparison | L2 centroid vs XGBoost/RF/LR/KNN | L2 within 0.24pp of KNN |
| V3B: Calibration | ECE analysis, temperature fix | tau=0.1 → ECE=0.036 (well calibrated) |

---

## Directory Tree

```
cross-graph-experiments/
├── CLAUDE.md                                    # Project guidelines and rules
├── EXPERIMENTS.md                               # Detailed experiment specifications
├── experiments_project_structure.md             # This file
├── configs/
│   └── default.yaml                             # Central config (no magic numbers)
│
├── src/
│   ├── data/
│   │   ├── alert_generator.py                   # SOC alert generation (Exp 1-4)
│   │   ├── category_alert_generator.py          # 5-category action-conditional alerts (Exp 5+)
│   │   │                                        #   + generate_campaign(), generate_precampaign()
│   │   ├── claim_generator.py                   # Synthesis claims (old API, frozen)
│   │   ├── entity_generator.py                  # Entity embeddings for Exp 2-4
│   │   └── generic_alert_generator.py           # Generic alert generation
│   ├── models/
│   │   ├── scoring_matrix.py                    # Eq. 4: P(action|alert) = softmax(f·W^T / τ)
│   │   ├── cross_attention.py                   # Eqs. 5-8: cross-graph attention & discovery
│   │   ├── oracle.py                            # BernoulliOracle + GTAlignedOracle
│   │   ├── profile_scorer.py                    # L2 nearest-centroid scorer + synthesis bias
│   │   ├── synthesis.py                         # SynthesisBias dataclass (old API, frozen)
│   │   ├── rule_projector.py                    # RuleBasedProjector (old API, frozen)
│   │   ├── gating.py                            # Uniform/Hebbian/MI gating
│   │   ├── operator_registry.py                 # Operator claim registry
│   │   ├── operator_spec.py                     # Operator specification dataclass
│   │   ├── residual_tracker.py                  # Residual norm tracking
│   │   └── profile_scorer_synthesis_patch.py    # Patch verification (test harness only)
│   ├── synthesis/                               # New-API synthesis module (mutable)
│   │   ├── synthesis.py                         # Mutable SynthesisBias: set/get/tensor()
│   │   ├── rule_projector.py                    # RuleBasedProjector() no-arg constructor
│   │   └── claim_generator.py                   # Claim dataclass, generate_correct/poisoned
│   ├── eval/
│   │   ├── auac.py                              # Area Under Accuracy Curve metric
│   │   └── op_harness.py                        # Operator experiment harness
│   └── viz/
│       ├── bridge_common.py                     # COLORS, VIZ_DEFAULTS, setup_axes, save_figure
│       ├── synthesis_common.py                  # save_results(), print_gate_result()
│       ├── exp1_charts.py                       # Exp 1 figures + LaTeX table
│       ├── exp1_blog_chart.py                   # Exp 1 blog version
│       ├── exp2_charts.py                       # Exp 2 F1 bars + P-R curves
│       ├── exp3_charts.py                       # Exp 3 scaling with power-law overlay
│       ├── exp3_blog_chart.py                   # Exp 3 blog version
│       ├── exp4_charts.py                       # Exp 4 2×2 sensitivity panel
│       ├── exp5_charts.py                       # Exp 5 oracle accuracy, ratio sweep, warmup
│       ├── expA_charts.py                       # Exp A convergence, G heatmap, breakdown
│       ├── expB1_charts.py                      # Exp B1 warm/cold/centroid, LR heatmap
│       ├── expC1_charts.py                      # Exp C1 method comparison, confusion matrix
│       ├── expD1_charts.py                      # Exp D1 transfer matrix, convergence, speedup
│       ├── expD2_charts.py                      # Exp D2 MI single, interactions, augmentation
│       ├── expE1_charts.py                      # Exp E1 kernel ranking, per-category heatmap
│       └── expE2_charts.py                      # Exp E2 oracle scaling, cold/warm gap
│
├── experiments/
│   ├── exp1_scoring_convergence/
│   │   ├── run.py
│   │   └── results/
│   │       ├── convergence_data.csv             # 350 rows: 10 seeds × 5 methods × 7 checkpoints
│   │       └── weight_evolution.npz             # W matrix snapshots (compounding only)
│   │
│   ├── exp2_cross_graph_discovery/
│   │   ├── run.py
│   │   ├── run_normalization_ablation.py
│   │   ├── normalization_ablation_results.csv   # 240 rows
│   │   ├── normalization_summary.json
│   │   └── results/
│   │       ├── discovery_results.csv            # 1230 rows: full grid sweep
│   │       └── best_configs.json
│   │
│   ├── exp3_multidomain_scaling/
│   │   ├── run.py                               # Original: 5 domain counts [2-6]
│   │   ├── run_extended.py                      # Extended: 11 domain counts [2-15], b=2.11
│   │   ├── run_norm_tracking.py                 # V1B: norm explosion tracking
│   │   ├── charts_val1b.py                      # V1B charts: norm growth, per-domain
│   │   ├── charts_val1b_explosion.py            # V1B blog figure: consolidated explosion panel
│   │   ├── charts_paper_scaling.py              # fig9_scaling_11point: log-log b=2.11 with CI band
│   │   ├── norm_tracking.csv                    # 360 rows
│   │   ├── norm_tracking_summary.json
│   │   └── results/
│   │       ├── scaling_data.csv                 # 50 rows: 10 seeds × 5 domain counts
│   │       ├── extended_scaling_data.csv        # 110 rows: 10 seeds × 11 domain counts
│   │       └── extended_scaling_fit.json        # b=2.11, CI [2.09, 2.14]
│   │
│   ├── exp4_sensitivity/
│   │   ├── run.py
│   │   └── results/
│   │       └── sensitivity_data.csv             # 105 rows across 4 parameter sweeps
│   │
│   ├── exp5_oracle_fix/
│   │   ├── run.py
│   │   └── results/
│   │       ├── ratio_sweep.csv
│   │       ├── warmup_comparison.csv
│   │       ├── oracle_comparison.csv
│   │       ├── fm1_analysis.csv
│   │       └── best_config.json
│   │
│   ├── expA_capacity_ceiling/
│   │   ├── run.py
│   │   └── results/
│   │       ├── accuracy_trajectories.csv        # 600 rows
│   │       ├── g_matrices.csv
│   │       └── summary.json
│   │
│   ├── expB1_profile_scoring/
│   │   ├── run.py
│   │   └── results/
│   │       ├── accuracy_trajectories.csv        # 3420 rows
│   │       └── summary.json
│   │
│   ├── expC1_centroid_oracle/
│   │   ├── run.py
│   │   └── results/
│   │       ├── classification_results.csv       # 100k rows
│   │       ├── confusion_matrices.json
│   │       └── summary.json
│   │
│   ├── expD1_cross_category_transfer/
│   │   ├── run.py
│   │   └── results/
│   │       ├── accuracy_trajectories.csv
│   │       ├── convergence_speed.csv
│   │       ├── transfer_matrix.csv
│   │       └── summary.json
│   │
│   ├── expD2_factor_interactions/
│   │   ├── run.py
│   │   └── results/
│   │       ├── mi_single.csv
│   │       ├── mi_interaction.csv
│   │       ├── top_interactions.json
│   │       └── summary.json
│   │
│   ├── expE1_kernel_generalization/
│   │   ├── run.py
│   │   └── results/
│   │       ├── phase1_oracle.csv                # 120 rows
│   │       ├── phase2_learning.csv              # 300 rows
│   │       ├── covariance_stats.csv             # 60 rows
│   │       └── summary.json
│   │
│   ├── expE2_scale_test/
│   │   ├── run.py
│   │   └── results/
│   │       ├── phase1_oracle.csv                # 40 rows
│   │       ├── phase2_learning.csv              # 480 rows
│   │       ├── phase3_separation.csv            # 50 rows
│   │       └── summary.json
│   │
│   ├── validation/
│   │   ├── run_push_stability.py                # V2: centroid update clipping analysis
│   │   ├── run_baseline_comparison.py           # V3A: L2 vs XGBoost/RF/LR/KNN
│   │   ├── run_calibration_analysis.py          # V3B: ECE + temperature calibration
│   │   ├── charts_val2.py                       # V2 charts: trajectories, fix comparison
│   │   ├── charts_val2_stability.py             # V2 blog figure: consolidated stability panel
│   │   ├── charts_val3b.py                      # V3B charts: ECE vs tau, reliability diagram
│   │   ├── charts_val3b_reliability.py          # V3B blog figure: two-panel reliability diagram
│   │   ├── charts_paper_reliability.py          # fig5_reliability_diagram: tau=0.25 vs tau=0.1 vs XGBoost
│   │   ├── push_stability_results.csv           # 16k rows
│   │   ├── push_stability_summary.json
│   │   ├── baseline_static_results.csv          # 70 rows
│   │   ├── baseline_online_results.csv          # 240 rows
│   │   ├── baseline_summary.json
│   │   ├── calibration_results.csv              # 70 rows
│   │   └── calibration_summary.json
│   │
│   ├── fx1_proxy_real/                          # FX-1-PROXY-REAL
│   │   ├── run.py
│   │   ├── realistic_generator.py
│   │   ├── charts.py
│   │   └── results.json
│   │
│   ├── fx1_learning/                            # FX-1-LEARNING
│   │   ├── run.py
│   │   ├── charts.py
│   │   └── results.json
│   │
│   ├── fx2_noise_distributions/                 # FX-2
│   │   ├── run.py
│   │   ├── bias_generator.py
│   │   ├── charts.py
│   │   └── results.json
│   │
│   ├── fx_t5_breakdown/                         # FX-T5-BREAKDOWN
│   │   ├── run.py
│   │   ├── charts.py
│   │   └── results.json
│   │
│   ├── fx_di07/                                 # FX-DI07: discount interval sweep
│   │   ├── run.py
│   │   └── results.json
│   │
│   ├── statistical_pass/                        # 50-seed published claims validation
│   │   ├── run.py
│   │   └── results.json
│   │
│   ├── expFX1_proxy_real/                       # Real IOC factor distribution (CISA KEV, NVD, ATT&CK)
│   │   ├── run.py                               # Orchestrator: pull → map → stats → charts
│   │   ├── data_pull.py                         # API fetchers (30s timeout, 24hr cache)
│   │   ├── factor_mapper.py                     # IOC → [threat_intel, asset_criticality, pattern_history]
│   │   ├── distribution_analysis.py             # fit_gaussian, compute_kl_divergence, distribution_stats
│   │   ├── charts.py                            # 3 charts: distributions, KL bars, stats table
│   │   ├── results.json                         # 2430 records, KL: TI=2.578, AC=1.880, PH=2.434
│   │   └── data/raw/                            # Cached API responses (cisa_kev.json, nvd_cves.json, mitre_attack.json)
│   │
│   ├── charts_paper_waterfall.py                # fig1_waterfall_progression (7-step equation progression)
│   ├── charts_paper_online_learning.py          # fig17_online_learning (L2 vs ML baselines)
│   ├── charts_paper_norm_ablation.py            # fig18_normalization_ablation (log-scale, 111× result)
│   ├── charts_paper_two_regimes.py              # fig6_two_regimes (arch validation vs deployment reality)
│   │
│   └── synthesis/                              # Phase 3 + OP series
│       ├── gate_m_decision.json                # Gate-M evaluation record
│       ├── gate_m_evaluation.py
│       ├── expS1_bias_accuracy/
│       │   ├── run.py
│       │   ├── charts.py
│       │   ├── results.csv                     # 10-seed detail
│       │   ├── results.json
│       │   └── gate_result.json               # GATE FAIL (honest)
│       ├── expS1b_degraded_profiles/
│       │   ├── run.py                          # 4 conditions × 6λ × 10 seeds × 500 alerts
│       │   ├── charts.py
│       │   ├── results.csv                     # 240 rows
│       │   ├── results.json
│       │   └── gate_result.json               # GATE FAIL (honest)
│       ├── expS1c_dynamic_recovery/
│       │   ├── run.py
│       │   ├── charts.py
│       │   ├── results.csv
│       │   ├── results.json
│       │   └── gate_result.json
│       ├── expS1d_gateop_replication/
│       │   ├── run.py
│       │   ├── charts.py
│       │   ├── results.csv
│       │   ├── results.json
│       │   └── gate_result.json
│       ├── expS2_poisoning/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json                   # GATE PASS
│       ├── expS2_repro/                        # Three-arm poisoning replication
│       │   ├── run.py                          # Arm0 frozen (10s×3pr), ArmA OPHarness (20s×4pr), ArmB SOC (10s×3pr)
│       │   ├── charts.py                       # 4 charts: arm0 bars, T_rec boxplot, AUAC line, A vs B
│       │   └── results.json                    # Arm0 PASS (-0.08pp); ArmA FAIL (NR=70%); ArmB DOMAIN EXPERT REVIEW
│       ├── expS3_loop_independence/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json                   # GATE PASS
│       ├── expS4_lambda_sensitivity/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json                   # GATE PASS: plateau_width=0.300
│       ├── expS5a_real_threat_intel/
│       │   ├── run.py
│       │   ├── fetch_kev.py
│       │   ├── fetch_nvd.py
│       │   ├── kev_raw.json / kev_claims.json
│       │   ├── nvd_raw.json / nvd_claims.json
│       │   ├── charts.py
│       │   └── results.json
│       ├── expS5b_work_artifacts/
│       │   ├── run.py
│       │   ├── extract_claims.py
│       │   ├── sample_artifacts.py
│       │   ├── charts.py
│       │   └── results.json                   # GATE PASS: LLM F1=0.877
│       ├── expOP1_final/                       # Baseline operator integration
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json
│       ├── expOP1_revised/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json
│       ├── expOP1_imperfect/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json
│       ├── expOP1_scalar_loop2/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json
│       ├── expOP2_harmful/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json                   # N=20 seeds; C NR=35% CI [15%, 59%] (too wide)
│       ├── expOP2_n100/                        # OP2 at N=100 seeds for tighter CI
│       │   ├── run.py                          # 100 seeds × 9 conditions = 900 runs, 0.40 min
│       │   ├── charts.py                       # 3 charts: NR CI bars, T_rec violin, B-exp bimodality
│       │   └── results.json                    # C NR=38% [29.1%, 47.8%]; B-exp BIMODAL (std/mean=3.64)
│       ├── expOP3_residual/
│       │   ├── run.py
│       │   ├── charts.py
│       │   └── results.json
│       └── expOP_margin/
│           ├── run.py
│           ├── charts.py
│           └── results.json
│
├── paper_figures/                              # All publication outputs (PDF + PNG, 300 DPI)
│   │
│   │  -- Core framework (Exp 1-4) --
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
│   │
│   │  -- Bridge layer (Exp 5, A-E) --
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
│   │
│   │  -- Synthesis layer (EXP-S series) --
│   ├── expS1_accuracy_by_lambda.{pdf,png}
│   ├── expS1_action_shift.{pdf,png}
│   ├── expS1_category_heatmap.{pdf,png}
│   ├── expS1_ece_by_lambda.{pdf,png}
│   ├── expS1b_accuracy_vs_lambda_by_condition.{pdf,png}
│   ├── expS1b_degradation_vs_baseline.{pdf,png}
│   ├── expS1b_improvement_heatmap.{pdf,png}
│   ├── expS1c_accuracy_curves.{pdf,png}
│   ├── expS1c_final_accuracy.{pdf,png}
│   ├── expS1c_recovery_distribution.{pdf,png}
│   ├── expS1d_category_heatmap.{pdf,png}
│   ├── expS1d_delta_by_lambda.{pdf,png}
│   ├── expS1d_warmup_vs_gateop.{pdf,png}
│   ├── expS2_poisoning_accuracy.png
│   ├── expS2_safety_effectiveness.png
│   ├── expS2_seed_distribution.png
│   ├── expS3_centroid_trajectory.png
│   ├── expS3_centroids_alone_accuracy.png
│   ├── expS3_frobenius_divergence.png
│   ├── expS4_accuracy_vs_lambda.png
│   ├── expS4_per_category_optimal.png
│   ├── expS5a_category_distribution.{pdf,png}
│   ├── expS5a_sigma_heatmap.{pdf,png}
│   ├── expS5a_source_comparison.{pdf,png}
│   ├── expS5a_urgency_distribution.{pdf,png}
│   ├── expS5b_claim_direction.{pdf,png}
│   ├── expS5b_extraction_f1.{pdf,png}
│   ├── expS5b_f1_by_type.{pdf,png}
│   ├── expS5b_sigma_comparison.{pdf,png}
│   │
│   │  -- OP series --
│   ├── expOP1_auac_curves.{pdf,png}
│   ├── expOP1_auac_delta.{pdf,png}
│   ├── expOP1_harmful_recovery.{pdf,png}
│   ├── expOP1_t70_comparison.{pdf,png}
│   ├── expOP1f_directionality.{pdf,png}
│   ├── expOP1f_loop2_overshoot.{pdf,png}
│   ├── expOP1f_stable_operation.{pdf,png}
│   ├── expOP1f_trajectories_lambda05.{pdf,png}
│   ├── expOP1f_window_sweep.{pdf,png}
│   ├── expOP1i_baseline_by_epsilon.{pdf,png}
│   ├── expOP1i_delta_by_epsilon.{pdf,png}
│   ├── expOP1i_directionality_check.{pdf,png}
│   ├── expOP1i_stable_operation.{pdf,png}
│   ├── expOP1i_t70_by_epsilon.{pdf,png}
│   ├── expOP1i_trajectories_epsilon010.{pdf,png}
│   ├── expOP1r_auac_curves.{pdf,png}
│   ├── expOP1r_auac_delta.{pdf,png}
│   ├── expOP1r_cold_start_benefit.{pdf,png}
│   ├── expOP1r_harmful_vs_correct.{pdf,png}
│   ├── expOP2_acute_phase.{pdf,png}
│   ├── expOP2_partial_accuracy_threshold.{pdf,png}
│   ├── expOP2_recovery_trajectories.{pdf,png}
│   ├── expOP2_t_recovery.{pdf,png}
│   ├── expOP3_decay_trajectories.{pdf,png}
│   ├── expOP3_diagnostic_scatter.{pdf,png}
│   ├── expOP3_early_warning_roc.{pdf,png}
│   ├── expOP3_per_category_norms.{pdf,png}
│   ├── expOPm_lambda_sweep.{pdf,png}
│   ├── expOPm_margin_distribution.{pdf,png}
│   ├── expOPm_per_category_margins.{pdf,png}
│   │
│   │  -- FX series --
│   ├── fx1_accuracy_by_mode.{pdf,png}
│   ├── fx1_accuracy_vs_ece_scatter.{pdf,png}
│   ├── fx1_confidence_bands.{pdf,png}
│   ├── fx1_ece_by_mode.{pdf,png}
│   ├── fx1_mahalanobis_vs_l2.{pdf,png}
│   ├── fx1_per_category_combined.{pdf,png}
│   ├── fx1_learning_per_category.{pdf,png}
│   ├── fx1_learning_trajectory.{pdf,png}
│   ├── fx2_accuracy_trajectories.{pdf,png}
│   ├── fx2_centroid_drift.{pdf,png}
│   ├── fx_t5_action_distribution.{pdf,png}
│   ├── fx_t5_cost_weighted.{pdf,png}
│   ├── fx_t5_error_direction.{pdf,png}
│   ├── fx_t5_per_action_accuracy.{pdf,png}
│   ├── fx_di07_discount_sweep.{pdf,png}
│   │
│   │  -- Statistical pass --
│   ├── statistical_pass_forest.{pdf,png}
│   │
│   │  -- Publication composite figures (paper-ready) --
│   ├── fig1_waterfall_progression.{pdf,png}    # 7-step equation progression 25%→98.2%
│   ├── fig5_reliability_diagram.{pdf,png}      # tau=0.25 vs tau=0.1 vs XGBoost calibration
│   ├── fig6_two_regimes.{pdf,png}              # Architecture validation vs deployment reality
│   ├── fig9_scaling_11point.{pdf,png}          # Log-log scaling b=2.11 with CI band
│   ├── fig17_online_learning.{pdf,png}         # L2 centroid online vs ML baselines (retrain/100)
│   ├── fig18_normalization_ablation.{pdf,png}  # Log-scale 111× result; raw/zscore-only = 0×
│   │
│   │  -- EXP-S2-REPRO charts --
│   ├── expS2r_arm0_replication.{pdf,png}       # Arm 0 frozen accuracy vs poison rate
│   ├── expS2r_t_recovery.{pdf,png}             # Arm A T_recovery boxplot by poison rate
│   ├── expS2r_auac_vs_poison.{pdf,png}         # Arm A mean AUAC ±SD line chart
│   ├── expS2r_realistic_auac_arm_b.{pdf,png}   # Side-by-side Arm A vs Arm B AUAC
│   │
│   │  -- EXP-OP2-N100 charts --
│   ├── expOP2n_never_recover_ci.{pdf,png}      # 9-bar NR rate + Wilson CI + N=20 diamonds
│   ├── expOP2n_t_recovery_violin.{pdf,png}     # 9 violin plots; bimodal conditions flagged salmon
│   ├── expOP2n_indirect_path_consistency.{pdf,png}  # B-exp bimodality check (histogram + std)
│   │
│   │  -- expFX1_proxy_real charts --
│   ├── fx1r_factor_distributions.{pdf,png}     # 3-panel real vs Gaussian fit vs synthetic ref
│   ├── fx1r_kl_divergence_from_synthetic.{pdf,png}  # KL bars (all > 0.5 — gap detected)
│   ├── fx1r_distribution_statistics.{pdf,png}  # Stats table: real vs synthetic colored grid
│   │
│   │  -- Validation (VAL-1B, VAL-2, VAL-3B) blog figures --
│   ├── val_1b_norm_explosion.{pdf,png}          # consolidated blog figure
│   ├── val_2_push_stability.{pdf,png}           # consolidated blog figure
│   ├── val_3b_reliability_diagram.{pdf,png}     # consolidated blog figure (tau=0.25 vs tau=0.1)
│   │
│   │  -- Validation component figures --
│   ├── val1b_norm_growth.{pdf,png}
│   ├── val1b_per_domain.{pdf,png}
│   ├── val2_norm_trajectories.{pdf,png}
│   ├── val2_fix_comparison.{pdf,png}
│   ├── val3b_temperature_ece.{pdf,png}
│   └── val3b_reliability_diagram.{pdf,png}
│
└── notebooks/                                  # Placeholder
```

---

## Source Files

### `src/data/alert_generator.py`

Generates synthetic SOC alerts for Experiment 1.

**Key types:**
- `Alert` (dataclass): `alert_id`, `alert_type`, `factors[6]`, `ground_truth_action`, `is_noisy`
- `AlertGenerator`: `generate(n, seed)` — reproducible list of alerts

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
- Factors sampled from N(mu[category][gt_action], factor_sigma) — orthogonal primary factors per action
- Module-level constants: `CATEGORIES`, `ACTIONS`, `FACTORS`
- Extensions: `generate_campaign(n, seed)` — campaign-period GT shift; `generate_precampaign(n, seed)` — suppressed GT for cold-start gap

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

---

### `src/models/scoring_matrix.py`

Implements **Eq. 4**: `P(action|alert) = softmax(f · W^T / τ)`

**Key type:** `ScoringMatrix`

| Parameter | Default | Role |
|-----------|---------|------|
| `n_actions` | 4 | Number of actions |
| `n_factors` | 6 | Alert factor dimensions |
| `temperature tau` | 0.25 | Softmax sharpness |
| `alpha_correct` | 0.002 | Hebbian reward step |
| `alpha_incorrect` | 0.04 | Hebbian penalty step (20× alpha_correct) |
| `weight_clamp` | 5.0 | Prevents unbounded growth |
| `decay_rate` | 0.001 | Inverse-time LR decay |

**Asymmetric Hebbian update rule:**
```
if correct:   W[action] += alpha_correct   * lr(t) * factors
if incorrect: W[action] -= alpha_incorrect * lr(t) * factors
lr(t) = 1 / (1 + decay_rate * t)
```

---

### `src/models/cross_attention.py`

Implements **Eqs. 5-8**: cross-graph attention and entity pair discovery.

**Key type:** `CrossGraphAttention`

| Method | Equation | Description |
|--------|----------|-------------|
| `compute_logits(E_i, E_j)` | Eq. 5 | `S = E_i @ E_j.T / sqrt(d)` |
| `compute_attention(S)` | Eq. 6 | `A = softmax(S, axis=1)` (rows sum to 1) |
| `compute_output(A, V_j)` | Eq. 6 | `O = A @ V_j` |
| `discover_two_stage(E_i, E_j, theta, K)` | Eqs. 8a+8b | Stage 1 ∩ Stage 2 |
| `discover_logit_only(E_i, E_j, theta)` | Eq. 8a | Pre-softmax threshold only |
| `discover_topk_only(E_i, E_j, K)` | Eq. 8b | Top-K softmax only |
| `cosine_baseline(E_i, E_j, threshold)` | — | Raw cosine (no sqrt(d) scaling) |

---

### `src/models/oracle.py`

Oracle implementations for Bridge Layer Experiments (Exp 5+).

- `BernoulliOracle` — Legacy oracle. Outcome drawn from Bernoulli(category_rate) independently of action correctness. Converges to category-level bias, not ground truth.
- `GTAlignedOracle` — R1 fix. Outcome is +1 iff action matches ground truth, else -1. A `noise_rate` fraction of outcomes is randomly flipped to model analyst feedback noise.

---

### `src/models/profile_scorer.py`

L2 nearest-centroid scorer with online centroid update and synthesis bias support.

**Eq. 4-synthesis:** `P(a|f,c,σ) = softmax(-(||f-mu[c,a,:]||^2 + λ·σ[c,a]) / τ)`

When `synthesis=None` or `lambda_coupling=0`, reduces exactly to Eq. 4''.

| Parameter | Role |
|-----------|------|
| `tau` | Softmax temperature. Never modified by synthesis. |
| `eta` | Learning rate for correct decisions (pull toward f) |
| `eta_neg` | Learning rate for incorrect decisions (push away) |

**Online update:** Asymmetric centroid pull/push with count-based decay. σ is NEVER passed to `update()` — centroids learn from experience only.

**Key interface fix (V2):** `np.clip(mu, 0, 1)` after every update prevents centroid escape under adversarial push conditions.

---

### `src/models/synthesis.py` (old API, frozen)

`SynthesisBias` frozen dataclass — immutable synthesis state passed to `score()`.

| Field | Description |
|-------|-------------|
| `sigma` | `(n_categories, n_actions)` — awareness bias tensor. σ[c,a]<0 → action more likely. |
| `active_claims` | Claims that passed extraction threshold. |
| `lambda_coupling` | Coupling strength λ. λ=0.0 → guaranteed kill switch via softmax. |

**`SynthesisBias.neutral(n_cat, n_act)`** — zero sigma, λ=0, no synthesis effect.

---

### `src/synthesis/` (new API, mutable)

New-API synthesis module used by all EXP-S and OP experiments.

- **`synthesis.py`**: Mutable `SynthesisBias` with `set()`, `get()`, `tensor()` methods; no `lambda_coupling` or `active_claims` fields.
- **`rule_projector.py`**: `RuleBasedProjector()` no-arg constructor; takes `Claim` objects. SIGN FIX: `value = -claim.direction * strength * tier_w * lambda` (negated direction — direction=+1 means "bias toward" → σ<0 → action more likely).
- **`claim_generator.py`**: `Claim` dataclass, `generate_correct_claims()`, `generate_poisoned_claims()`. Run with `PYTHONUTF8=1` on Windows.

---

### `src/models/rule_projector.py` (old API, frozen)

`RuleBasedProjector` — maps a list of claims to a `SynthesisBias` via domain-configured rule templates.

**`project(claims, lambda_coupling)`** → `SynthesisBias`

Pipeline per claim: filter (confidence × extraction_confidence < 0.8), decay, accumulate, clip to [-sigma_max, +sigma_max].

---

### `src/models/gating.py`

Three gating mechanisms for Exp 5-9, A:

| Mechanism | Description |
|-----------|-------------|
| `UniformGating` | Baseline: G = ones, no learning |
| `HebbianGating` | Online learning from oracle outcomes (Eq. 4d) |
| `MIGating` | Offline mutual-information estimation; static after `fit_from_data()` |

---

### `src/eval/auac.py`

Area Under Accuracy Curve metric for operator synthesis experiments. Computes AUAC from accuracy trajectory arrays, normalizes by episode length. Used by OP series experiments.

---

### `src/eval/op_harness.py`

Standardized operator experiment harness. Provides: episode runner, warm/cold start modes, oracle feedback integration, result aggregation across seeds.

---

### `src/viz/bridge_common.py`

Shared visualization utilities for all bridge layer experiments.

**Exports:** `COLORS` dict, `VIZ_DEFAULTS` dict, `setup_axes()`, `save_figure()` (saves PDF + PNG at 300 DPI).

```python
COLORS = {
    "bernoulli":        "#94A3B8",  # gray
    "gt_noise_0":       "#1E3A5F",  # dark blue
    "gt_noise_5":       "#2563EB",  # blue
    "gt_noise_15":      "#7C3AED",  # purple
    "gt_noise_30":      "#DC2626",  # red
    "uniform_gate":     "#94A3B8",
    "hebbian_damped":   "#1E3A5F",
    "hebbian_undamped": "#D97706",  # amber
    "mi_static":        "#059669",  # green
    "category_colors":  ["#1E3A5F","#D97706","#059669","#DC2626","#7C3AED"],
}
VIZ_DEFAULTS = {
    "dpi": 300, "title_fontsize": 13, "label_fontsize": 11,
    "tick_fontsize": 9, "annotation_fontsize": 8.5,
    "figsize_single": (8,5), "figsize_wide": (12,5), "figsize_heatmap": (10,6),
}
```

---

## Configuration: `configs/default.yaml`

Single source of truth for all experiment parameters — no magic numbers in code.

### Experiments 1-4 (Core Framework)

**Exp 1:** `n_alerts`: 5000, `noise_rate`: 0.03, checkpoints [50-5000], 5 baselines
**Exp 2:** 3 domains, `embedding_dim`: 64, `signal_strength`: 8.0, theta/K grids
**Exp 3:** `domain_counts`: [2-6], `entities_per_domain`: 200, fixed theta=0.02, K=3
**Exp 4:** 4 parameter sweeps (asymmetry, temperature, noise, embedding_dim)

### Bridge Layer Common

- 5 categories, 4 actions, 6 factors (all named)
- `category_gt_distributions`: per-category action probability distributions
- `action_conditional_profiles`: 5×4 matrix of 6-factor mean vectors
- Scoring defaults: tau=0.25, alpha_correct=0.02, alpha_incorrect=0.04, weight_clamp=5.0

---

## Experiments

### Experiment 1: Scoring Matrix Convergence

**Runner:** `experiments/exp1_scoring_convergence/run.py`
**Key result:** `compounding` reaches ~69-71% cumulative accuracy at 5000 alerts vs. 25% random baseline.

---

### Experiment 2: Cross-Graph Discovery

**Runner:** `experiments/exp2_cross_graph_discovery/run.py`
**Key result:** `two_stage` achieves ~116× F1 above random baseline at optimal (theta, K).

**Normalization Ablation:** zscore+l2 synergistic (F1=0.071, 145×); raw or zscore alone = 0.

---

### Experiment 3: Multi-Domain Scaling

**Runner:** `experiments/exp3_multidomain_scaling/run.py`
**Key result (original):** R² = 0.9995, b=2.30.
**Extended (V1A):** b=2.11, CI [2.09, 2.14]. Superquadratic confirmed; original was overfit to n≤6.

**V1B Norm Tracking:**
**Runner:** `experiments/exp3_multidomain_scaling/run_norm_tracking.py`
Catastrophic norm explosion without LayerNorm: ~40× per sweep after sweep 1, reaching **2.9M× by sweep 5** across 6 domains × 10 seeds. Confirms Eq. 13 reviewer concern.
**Blog figure:** `val_1b_norm_explosion.{pdf,png}` — consolidated single-panel log-scale chart with per-domain inset.

---

### Experiment 4: Parameter Sensitivity

**Key results:**

| Sweep | Best value | Finding |
|-------|-----------|---------|
| A (asymmetry) | ratio = 20 | 0.657 accuracy |
| B (temperature) | tau = 0.25 | 0.657 accuracy |
| C (noise) | < 5% | Sharp degradation above |
| D (embedding_dim) | d = 128 | F1 collapses at d = 256 |

---

### Experiment 5: Oracle Fix (Bridge Layer)

**Key results:** V5.1: 79.65% (>75%) | V5.2: +26.09pp over Bernoulli | ratio=5>3>2>1.5>1 monotonic. **GATE: PASS.**

---

### Experiment A: Capacity Ceiling

G-lift = +2.35pp (threshold: 8pp). **GATE: FAIL (honest).** Root cause: data-per-category bottleneck.

---

### Experiment C1: Centroid Oracle

L2 = 97.89% ±0.14% | Cosine = 96.42% | Dot = 61.00%. Gap vs Exp A shared W: +48.6pp. **GATE: PASS.**

---

### Experiment B1: Profile Scoring + Learning

centroid_only = 98.0% | profile_warm = 98.2% | profile_cold = 90.7%. Cold start recovers from 58.5% (t=100) to 90.7% (t=1000). **GATE: PASS.**

---

### Experiment D1: Cross-Category Transfer

Transfer competitive with config in 1/5 categories. Config warm start dominates. **Verdict: transfer NOT competitive.**

---

### Experiments D2, E1, E2

**E1 Key results:** L2=97.9% default; Mahalanobis=92.9% > L2=79.9% on mixed-scale data.
**E2:** Oracle accuracy improves with scale (97.9% → 99.9%). Cold start degrades at xlarge (89.9% → 72.7%). **GATE: PASS.**

---

### Validation Experiments

**V2 — Push Update Stability:**
5 conditions (normal 70/30, bad streak, worst-case 100% incorrect, clipped, margin guard). Condition C reaches max norm 4,607×. Condition D (clipped) stays bounded at 2.24×. **Fix:** `np.clip(mu, 0, 1)` after every update. Margin guard alone is insufficient (all_dims_in_bounds=False).
**Blog figure:** `val_2_push_stability.{pdf,png}` — 5-condition log-scale trajectories with escape window marker.

**V3A — Baseline Comparison:**
L2=94.78% > RF=93.20% > LR=92.38% > GBT=92.24%, KNN=94.54%. L2 within 0.24pp of KNN.

**V3B — Calibration Analysis:**
L2 at tau=0.25: ECE=0.190 (poor, underconfident). Fix: tau=0.1 → ECE=0.036 (well calibrated).
**Blog figure:** `val_3b_reliability_diagram.{pdf,png}` — two-panel reliability diagram, L2 tau=0.25 vs L2 tau=0.1. tau=0.1 bins recomputed from raw model (not stored in JSON).

---

### FX-1-PROXY-REAL

**Runner:** `experiments/fx1_proxy_real/run.py`

5 data modes × 2 kernels (L2, Mahalanobis) × 10 seeds × 2000 alerts.

| Mode | Accuracy | ECE |
|------|----------|-----|
| L2 centroidal | 91.81% | 0.117 |
| L2 combined (realistic) | 71.45% | 0.026 |

- **GATE FAIL (honest):** acc < 80%, degradation = 20.36pp (>20pp threshold by 0.36pp).
- Mahalanobis does NOT help on combined: delta = -0.97pp vs L2.
- Worst categories: travel_anomaly 69.45% (-23.63pp), credential_access 65.93% (-21.70pp).
- Auto-approve zone (conf≥0.90) accuracy = 91.47%, coverage = 10.5%.
- SOC categories taxonomy confirmed: `{travel_anomaly, credential_access, threat_intel_match, insider_behavioral, cloud_infrastructure}`.

---

### FX-1-LEARNING

**Runner:** `experiments/fx1_learning/run.py`

Oracle feedback learning trajectory under realistic combined distribution. 10 seeds × 2000 decisions, warm start.

| Checkpoint | Rolling-200 accuracy |
|------------|---------------------|
| Static baseline | 71.45% ±0.46pp |
| @500 | 76.15% |
| @1000 | 77.65% |
| @1500 | 80.00% |
| @2000 | 78.60% |

- **GATE FAIL (honest):** acc@1000 = 77.65% < 82% gate (by 4.35pp).
- Worst category: credential_access 66.40% @1000 (+3.17pp lift only).
- Best lift: travel_anomaly +7.78pp to 79.43%.
- **Design outcome:** 75-81% accuracy band → shadow mode is hard prerequisite; per-category config needed.

---

### FX-2: Production Noise Distributions

**Runner:** `experiments/fx2_noise_distributions/run.py`

3 bias patterns × 10 seeds × 1500 decisions on combined realistic data.

| Pattern | Centroid drift | Final accuracy | Gate |
|---------|---------------|----------------|------|
| POST_INCIDENT | 0.9054 | 78.12% | PASS (≥75%) — recovers @dec 251 |
| ALERT_FATIGUE | 0.7120 | 75.38% | PASS (≥65%) — no full recovery |
| EXPERTISE_GRADIENT | 0.7729 | 75.64% | PASS (≥70%) — no full recovery |

- **DESIGN WARNING:** alert_fatigue + expertise_gradient cause persistent centroid corruption (drift > 0.15). `ProfileScorer.update()` needs a `source` parameter to guard shadow feedback in v5.0.

---

### FX-T5-BREAKDOWN: Auto-Approve Band Action Analysis

**Runner:** `experiments/fx_t5_breakdown/run.py`

N=50 seeds, 2000 alerts/seed, τ=0.1, combined realistic data, static scorer.

| Metric | Value |
|--------|-------|
| Total auto-approve band (conf≥0.90) | 11,502 decisions (11.5% of all) |
| Monitor accuracy | 86.0% (fails 99% target) |
| Escalate accuracy | 100% |
| Investigate accuracy | 92.2% |
| Suppress accuracy | 99.9% |
| Total errors | 1,073 (9.33% of band) |
| Dangerous errors | 234 (2.03% of band) |
| Cost-weighted score | +0.427 (< 0.5 threshold) |

- **RESULT C:** Marginal. Monitor should be agent_zone not auto-approve.
- Dangerous errors driven by threat_intel_match (62%) + cloud_infrastructure (34%).

---

### expFX1_proxy_real: Real IOC Factor Distribution Characterization

**Runner:** `experiments/expFX1_proxy_real/run.py`

Pulls public threat intelligence, maps to SOC factor space, characterizes distribution gap from synthetic centroidal assumption. Runtime: 7 seconds (APIs cached after first pull).

**Data sources:**

| Source | Records pulled | Notes |
|--------|---------------|-------|
| CISA KEV | 1,542 | Full catalog; ransomware flag → threat_intel proxy |
| NVD CVE 2.0 | 200 | Single page, recent CVEs; CVSS v3.1 preferred |
| MITRE ATT&CK | 691 techniques | STIX bundle; detection text length → threat_intel |
| **Total mapped** | **2,430** | 3 skipped (missing CVSS) |

**Factor mapping:**
- `threat_intel` — CVSS baseScore / 10 (NVD/KEV); detection text length for ATT&CK
- `asset_criticality` — CWE type bucket (auth/priv→0.85, exec→0.75, disclosure→0.50, XSS→0.35, DoS→0.30)
- `pattern_history` — KEV recurrence count / 10 (min 0.10); ATT&CK platform breadth / 8

**Results:**

| Factor | N | Mean | Std | Skewness | Kurtosis | KL vs synthetic |
|--------|---|------|-----|----------|----------|----------------|
| Threat Intel Score | 2430 | 0.467 | 0.311 | −0.598 | −1.106 | **2.578** |
| Asset Criticality | 2430 | 0.646 | 0.225 | −0.600 | −1.391 | **1.880** |
| Pattern History | 2430 | 0.165 | 0.158 | +2.815 | +8.045 | **2.434** |

**Overall assessment: DISTRIBUTION GAP DETECTED** (all KL > 0.5 — centroidal assumption invalid for all three factors).

Key findings:
- **Threat intel**: Bimodal-ish (kurtosis −1.1), spread σ=0.311 vs synthetic σ=0.20. Real CVSS scores cluster at both extremes (0 and 1.0), not uniformly distributed.
- **Asset criticality**: Real data skews high (mean=0.646, not 0.50). KEV/NVD underrepresent low-criticality events — known selection bias (only known-exploited CVEs reach KEV).
- **Pattern history**: Extremely right-skewed (skew=2.8, kurtosis=8.0). Most vulnerabilities appear once (0.1); heavy tail of recurring issues. The synthetic uniform-ish assumption is entirely wrong for this factor.
- **Implication**: These results directly explain the 20pp accuracy degradation in FX-1-PROXY-REAL. Real threat data is not Gaussian around a centroid — it has heavy tails, bimodal structure, and extreme right-skew in recurrence.

---

### Statistical Pass

**Runner:** `experiments/statistical_pass/run.py`

Standing rule: ≥50 seeds + 95% CI + t-test for all published numbers.

| Claim | Mean [95% CI] | vs Gate |
|-------|---------------|---------|
| T1 Static realistic acc | 71.7% [71.4%, 71.9%] | FAIL (vs 80%) |
| T2 Learning @dec 1000 | 78.9% [78.1%, 79.6%] | FAIL (vs 82%) |
| T3 Learning @dec 1500 | 79.4% [78.7%, 80.1%] | FAIL (vs 82%) |
| T4 credential_access @dec 1000 | 68.0% [66.7%, 69.1%] | descriptive only |
| T5 Auto-approve acc ≥0.90 | 90.7% [90.1%, 91.2%] | PASS (p=0.024 vs 90%) |
| DI-06 dec1500 vs dec1000 | +0.530pp, CI [-0.548, +1.608] | UNRESOLVED |

Auto-approve coverage: 11.50% ±0.70% of all decisions.

---

### EXP-S Series

**EXP-S1 (warm-start):** baseline=96.88%, best improvement=+0.20pp (p=0.333). **GATE FAIL.** No gap for sigma when profiles are already warm.

**EXP-S1b (degraded profiles):** 4 conditions × 6λ × 10 seeds. cold_start (28.78% baseline): max +0.02pp. warm_start HARM at λ=0.5: -5.78pp (p=0.003). Root cause: random claim targets provide no systematic signal; high λ overrides correct profile predictions. **GATE FAIL.** Scientific conclusion: sigma value lies in non-accuracy dimensions (confidence shaping, threat briefing).

**EXP-S2 (poisoning):** degradation_20pct = -0.00pp (≤2pp), safety_effectiveness = 1.000 (≥0.50). **GATE PASS.**

**EXP-S2-REPRO (three-arm poisoning replication):** 140 total runs across three arms.

- **Arm 0 — Frozen synthesis** (10 seeds × 3 poison rates, bridge-common taxonomy): Centroids never update; SynthesisBias applied at scoring only. Degradation at 20% poison = −0.08pp. **GATE PASS** (≤2pp threshold). Frozen centroids are surprisingly resilient — sigma bias alone cannot meaningfully corrupt a warm-started scorer when it cannot write to centroids.
- **Arm A — Production condition** (20 seeds × 4 poison rates, OPHarness, N_pre=200, N_post=400, λ=0.5): never_recover_rate=70%, p90 T_recovery=3315 decisions at 20% poison. **GATE FAIL** (gates: p90<100, NR≤5%). Root cause: poisoned OperatorSpec at λ=0.5 persistently corrupts centroids during the 400-decision window; warm-start model cannot self-correct within this window. Consistent with FX-2 DESIGN WARNING.
- **Arm B — Realistic AUAC** (10 seeds × 3 poison rates, SOC combined mode, DOMAIN EXPERT REVIEW): AUAC stable at 0.583–0.598 across poison rates 0–40%. Gate: DOMAIN EXPERT REVIEW.

**EXP-S3 (loop independence):** mean_relative_frobenius = 0.0028 (≤0.05), mean_acc_diff = 0.050pp (≤1pp). σ does NOT contaminate μ; update() firewall sound. **GATE PASS.**

**EXP-S4 (lambda sensitivity):** plateau_width=0.300, plateau=[0.200, 0.500]. **GATE PASS.**

**EXP-S5b (claim extraction):** LLM F1=0.877 (≥0.70), template F1=0.621. **GATE PASS.**

---

### OP Series

**expOP1_final:** Baseline operator integration. AUAC curves and T70 (time to 70% accuracy). Establishes reference trajectory for synthesis-assisted operator scenarios.

**expOP1_revised:** Revised integration including cold-start benefit analysis (harmful vs correct claim separation).

**expOP1_imperfect:** Epsilon sweep (imperfect extraction accuracy). Validates that claim extraction above ~0.8 threshold maintains benefit; below threshold — harm.

**expOP1_scalar_loop2:** Scalar loop overshoot analysis. Identifies conditions where repeated synthesis activation causes oscillation.

**expOP2_harmful:** Harmful claim recovery trajectories (N=20 seeds). Acute phase characterization and partial accuracy threshold. Original never-recover CI: C=35% [15%, 59%] — too wide for a safety claim.

**expOP2_n100:** EXP-OP2 replicated at N=100 seeds for tighter CI on never-recover rate. 9 conditions × 100 seeds = 900 runs (0.40 min). Identical setup to OP2 — only N changes.

Key results (N=100):

| Condition | NR% | 95% Wilson CI | Safety policy |
|-----------|-----|---------------|---------------|
| A (baseline) | 24.0% | [16.7%, 33.2%] | REQUIRES CHECKPOINT+ROLLBACK |
| B (correct, full TTL) | 8.0% | [4.1%, 15.0%] | MONITORING REQUIRED |
| B-exp (correct, TTL=150) | 8.0% | [4.1%, 15.0%] | MONITORING REQUIRED |
| C (harmful, full TTL) | 38.0% | [29.1%, 47.8%] | REQUIRES CHECKPOINT+ROLLBACK |
| C-exp (harmful, TTL=150) | 38.0% | [29.1%, 47.8%] | REQUIRES CHECKPOINT+ROLLBACK |
| P-75 | 28.0% | [20.1%, 37.5%] | REQUIRES CHECKPOINT+ROLLBACK |
| P-50 | 20.0% | [13.3%, 28.9%] | REQUIRES CHECKPOINT+ROLLBACK |
| P-25 | 29.0% | [21.0%, 38.5%] | REQUIRES CHECKPOINT+ROLLBACK |
| P-0 | 38.0% | [29.1%, 47.8%] | REQUIRES CHECKPOINT+ROLLBACK |

- C CI collapsed from [15%, 59%] (N=20) to **[29.1%, 47.8%]** — 4× narrower. 35% estimate confirmed.
- Condition A (no operator) NR=24%: baseline without an operator also struggles to recover in campaign distribution.
- Only B approaches safe-deployment threshold (5%); CI [4.1%, 15.0%] straddles it — monitoring required.
- **B-exp bimodality: BIMODAL** — std/mean ratio = 3.640 (threshold 0.5). Distribution splits into fast-recoverers vs never-recoverers (sentineled at 401).

**expOP3_residual:** Residual decay tracking with per-category norm monitoring. Early-warning ROC curve for detecting claim staleness.

**expOP_margin:** Margin distribution per category. Characterizes decision confidence margins under synthesis bias.

---

## Reproducibility

**Fixed seeds (all experiments):** `[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]`

**Statistical standard (FX series + Statistical Pass):** ≥50 seeds + 95% CI + two-sided t-test for any published claim.

**Windows note:** Run with `PYTHONUTF8=1` for any script using lambda (λ) or sigma (σ) characters in print statements.

---

## Performance Summary

| Experiment | Key metric | Value | Baseline |
|------------|-----------|-------|----------|
| Exp 1 | Cumulative acc at 5K alerts | ~69.4% | 25% (random) |
| Exp 2 | Best F1 | ~116× above random | ~0.025 F1 |
| Exp 2 (norm ablation) | zscore_l2 F1 | 0.071 (145×) | 0× (raw) |
| Exp 3 | Power-law exponent b | 2.11 [2.09, 2.14] | — |
| Exp 4 | Critical noise threshold | ~5% | — |
| Exp 5 | GT oracle accuracy | 79.65% | 53.56% (Bernoulli) |
| Exp A | G-lift over W-only | +2.35pp | Threshold: 8pp (FAIL) |
| Exp C1 | L2 centroid accuracy | 97.89% | 61% (Dot) |
| Exp B1 | Warm start (noise=0.30) | 98.1% | 90.7% (cold) |
| Exp D1 | Transfer vs config gap | -2 to -14pp | Config dominates |
| Exp E1 | L2 accuracy (original) | 97.9% | 61% (Dot) |
| Exp E2 | Oracle degradation (small→xlarge) | -2pp | Threshold: 5pp (PASS) |
| V1B | Max norm growth (5 sweeps) | 2.94M× | 1× (with LayerNorm) |
| V2 | Max norm, worst case | 4,607× | 2.24× (clipped) |
| V3A | L2 vs best ML (KNN) | -0.24pp | Competitive |
| V3B | ECE tau=0.1 | 0.036 | 0.190 (tau=0.25) |
| FX-1-PROXY-REAL | Combined realistic acc | 71.45% | 91.81% (centroidal) |
| FX-1-LEARNING | Acc @dec 1000 | 77.65% [78.1%, 79.6%] | 82% gate (FAIL) |
| FX-T5 | Auto-approve band dangerous error | 2.03% | 0.5% threshold (RESULT C) |
| Stat pass T5 | Auto-approve acc ≥0.90 | 90.7% [90.1%, 91.2%] | 90% gate (PASS) |
| EXP-S2-REPRO Arm 0 | Frozen synthesis degradation @20% poison | −0.08pp | ≤2pp gate (PASS) |
| EXP-S2-REPRO Arm A | Never-recover rate @20% poison | 70.0% | ≤5% gate (FAIL) |
| EXP-OP2-N100 cond C | Never-recover rate | 38.0% [29.1%, 47.8%] | N=20 CI was [15%, 59%] — 4× tighter |
| EXP-OP2-N100 cond B | Never-recover rate | 8.0% [4.1%, 15.0%] | Straddles 5% threshold |
| expFX1_proxy_real | Max KL div (threat_intel) | 2.578 | >0.5 → GAP DETECTED |
| EXP-S4 | Lambda plateau width | 0.300 | Threshold: 0.05 (PASS) |
| EXP-S5b | LLM claim extraction F1 | 0.877 | Threshold: 0.70 (PASS) |

---

## Equations Validated

| Equation | Description | Validator |
|----------|-------------|-----------|
| **Eq. 4** | `P(action\|alert) = softmax(f·W^T / τ)` | Convergence to >69% accuracy (Exp 1) |
| **Eq. 4''** | `P(a\|f,c) = softmax(-\|\|f - mu\|\|^2 / τ)` | L2 centroid 97.89% (Exp C1) |
| **Eq. 4-synthesis** | `P(a\|f,c,σ) = softmax(-(d^2 + λ·σ) / τ)` | Loop independence confirmed (EXP-S3) |
| **Eq. 5** | `S_ij = E_i·E_j^T / sqrt(d)` | Logit shape and values (Exp 2) |
| **Eq. 6** | `A = softmax(S, axis=1)`, `O = A@V` | Row sums = 1, output shape (Exp 2) |
| **Eq. 8a** | `s_kl > theta_logit` | Stage 1 filtering (Exp 2) |
| **Eq. 8b** | `entity_l in top-K(softmax(S_k,:))` | Stage 2 filtering (Exp 2) |
| **Eq. 13** | Residual enrichment `E_i += Σ CrossAttn(G_i, G_j)` | Norm explosion 2.9M× (V1B) |
| **Scaling** | `I(n,t) = n(n-1)/2 × richness(t)^γ` | b=2.11, CI [2.09, 2.14] (Exp 3 ext) |

---

## Known Interface Fixes

| API | Correct call |
|-----|-------------|
| Alert generator | `gen.generate(N)` (not `generate_alerts`) |
| Profile extraction | `build_gt_array(gen)` — extract ndarray from `gen.profiles` dict |
| Alert factors | `alert.factors` (not `alert.factor_vector`) |
| ProfileScorer init | `ProfileScorer(mu_array, tau=0.1)` — shape extracted from array |
| `profile_scorer.score()` | `lambda_coupling=0.0` kwarg handles both old + new API |
| Windows Unicode | `PYTHONUTF8=1` for any script printing λ/σ/× characters |

---

## Pending / Not Yet Run

| Experiment | Description |
|------------|-------------|
| Exp 6 | MI between categories and factors |
| Exp 7 | Gating mechanisms (uniform, hebbian damped/undamped, mi_static) |
| Exp 8 | Alpha-g ratio sensitivity + damping |
| Exp 9 | Hidden factor detection (insider_threat) |
| DI-06 resolution | Need 233+ seeds to resolve dec1500 vs dec1000 delta |
