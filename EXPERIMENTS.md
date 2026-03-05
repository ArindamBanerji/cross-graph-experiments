# Experiments

Validation suite for the Cross-Graph Attention framework.
All experiments use fixed seeds, numpy-only computation, and output figures (PDF + PNG at 300 DPI), LaTeX tables, and raw CSV data.

---

## Experiment 1 — Scoring Matrix Convergence

**Runner:** `experiments/exp1_scoring_convergence/run.py`

Validates **Eq. 4**: `P(action | alert) = softmax(f · W^T / τ)`.

Tests how quickly an asymmetric compounding learning rule specialises the scoring matrix `W` to the correct action for each alert type, compared against four baselines.

**Baselines compared:**

| Method | Description |
|---|---|
| `compounding` | Asymmetric Hebbian update — large reward on correct, small penalty on incorrect |
| `fixed_weight` | No learning; evaluates the random initialisation |
| `random_policy` | Uniformly random action selection |
| `periodic_retrain` | Resets weights every 500 decisions |
| `symmetric` | Equal learning rates for correct and incorrect |

**Setup:** 10 seeds × 5 baselines × 5,000 alerts per trial. Checkpoints at [50, 100, 200, 500, 1000, 2000, 5000].

**Key result:** `compounding` reaches ~69% cumulative accuracy at 5,000 decisions vs. 25% for the random baseline.

**Outputs:**
- `experiments/exp1_scoring_convergence/results/convergence_data.csv` — per-checkpoint accuracy for all methods and seeds
- `experiments/exp1_scoring_convergence/results/weight_evolution.npz` — `W` snapshots at each checkpoint (compounding only)
- `paper_figures/exp1_blog_convergence.pdf/.png` — simplified convergence chart for blog

---

## Experiment 2 — Cross-Graph Discovery

**Runner:** `experiments/exp2_cross_graph_discovery/run.py`

Validates **Eq. 6** (`S_ij = E_i · E_j^T / √d`) and the two-stage discovery rules **Eq. 8a** (pre-softmax logit threshold `θ`) and **Eq. 8b** (top-K softmax filter).

Tests five methods for identifying ground-truth entity pairs across domain knowledge graphs.

**Domain pairs tested:**

| Pair | Ground-truth signals |
|---|---|
| security × threat_intel | 20 |
| decision_history × threat_intel | 15 |
| security × decision_history | 15 |

**Methods compared:**

| Method | Description |
|---|---|
| `two_stage` | Logit threshold (Eq. 8a) **and** top-K filter (Eq. 8b); sweeps `θ ∈ [0.01–0.06]`, `K ∈ [1, 2, 3, 5]` |
| `logit_only` | Logit threshold only; sweeps same `θ` range |
| `topk_only` | Top-K filter only; sweeps same `K` values |
| `cosine` | Raw cosine similarity threshold; sweeps `[0.10–0.35]` |
| `random` | Analytical expected P/R/F1 at the median `two_stage` discovery count |

**Setup:** 10 seeds × 3 domain pairs × all config grids. Entity embeddings: d=64, 200 entities per domain, signal strength 8.0. Shared dims (6–13) carry geo and time soft one-hot signals injected into GT pairs.

**Key result:** `two_stage` achieves the best F1, substantially above random (~116× at optimal config).

**Outputs:**
- `experiments/exp2_cross_graph_discovery/results/discovery_results.csv` — per-run P/R/F1 for all methods, configs, seeds, pairs
- `experiments/exp2_cross_graph_discovery/results/best_configs.json` — optimal config per (method, domain pair)
- `paper_figures/exp2_f1_bars.pdf/.png` — best F1 bar chart with error bars
- `paper_figures/exp2_precision_recall.pdf/.png` — precision-recall tradeoff chart
- `paper_figures/exp2_table.tex` — LaTeX results table

---

## Experiment 3 — Multi-Domain Scaling

**Runner:** `experiments/exp3_multidomain_scaling/run.py`

Validates the scaling law **I(n, t) = n(n−1)/2 × richness(t)^γ** — that cross-domain discovery count grows quadratically with the number of connected knowledge domains.

**Setup:** 10 seeds × 5 domain counts [2, 3, 4, 5, 6]. Six domains used: `security`, `decision_history`, `threat_intel`, `network_flow`, `asset_inventory`, `user_behavior`. 5 injected signals per domain pair, discovered via two-stage (θ=0.02, K=3). A power law `a · n^b` is fit to mean discoveries vs. domain count.

**Key results:**

| n domains | n(n−1)/2 pairs | Mean discoveries |
|---|---|---|
| 2 | 1 | 600 |
| 3 | 3 | 1,800 |
| 4 | 6 | 3,600 |
| 5 | 10 | 6,000 |
| 6 | 15 | 9,000 |

Fitted exponent **b = 2.30**, **R² = 0.9995** (expected ~2.0 for pure quadratic; slight super-quadratic from fitting `a·n^b` to the `n(n−1)` form over a short range).

**Outputs:**
- `experiments/exp3_multidomain_scaling/results/scaling_data.csv` — per-seed discovery counts for all domain counts
- `paper_figures/exp3_scaling.pdf/.png` — publication scaling chart with power-law overlay
- `paper_figures/exp3_table.tex` — LaTeX scaling table
- `paper_figures/exp3_blog_scaling.pdf/.png` — simplified chart for blog (annotated gap vs. linear growth)

---

## Experiment 4 — Parameter Sensitivity Analysis

**Runner:** `experiments/exp4_sensitivity/run.py`

Sensitivity analysis of the two key model components — the scoring matrix (Exp 1) and the cross-graph attention (Exp 2) — across four parameter sweeps.

**Sweeps:**

| Sweep | Parameter | Values tested | Metric |
|---|---|---|---|
| A | `asymmetry_ratio` (α_incorrect / α_correct) | 1, 5, 10, 20, 50 | Accuracy |
| B | `temperature` τ | 0.1, 0.25, 0.5, 1.0, 2.0 | Accuracy |
| C | `noise_rate` (alert label noise) | 0.0, 0.03, 0.05, 0.1, 0.2, 0.3 | Accuracy |
| D | `embedding_dim` d | 16, 32, 64, 128, 256 | F1 |

**Setup:** 5 seeds × 2,000 alerts per scoring-matrix trial. Embedding-dim trials: 50 entities per domain, 10 injected signals, two-stage discovery (θ=0.02, K=3). Dims below 14 skipped (shared slice requires d ≥ 14).

**Key results (best mean metric per sweep):**

| Sweep | Best value | Mean metric |
|---|---|---|
| A — asymmetry_ratio | 20 | 0.6569 accuracy |
| B — temperature | 0.25 | 0.6569 accuracy |
| C — noise_rate | 0 | 0.6858 accuracy |
| D — embedding_dim | 128 | 0.2444 F1 |

Notable findings: accuracy degrades sharply above noise_rate ≈ 0.05 (critical threshold); embedding_dim=256 collapses to F1=0 (over-parameterised for 50 entities with 10 signals).

**Outputs:**
- `experiments/exp4_sensitivity/results/sensitivity_data.csv` — 105 rows across all sweeps and seeds
- `paper_figures/exp4_sensitivity.pdf/.png` — 2×2 panel sensitivity grid
- `paper_figures/exp4_table.tex` — LaTeX summary table
