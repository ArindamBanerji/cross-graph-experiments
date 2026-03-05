# Cross-Graph Experiments: Project Structure

## Overview

Experimental validation suite for the **Cross-Graph Attention** framework. Four experiments validate the framework at three levels:

| Level | Experiment | What it validates |
|-------|------------|-------------------|
| Level 1 | Exp 1: Scoring matrix convergence | Eq. 4 — asymmetric Hebbian learning |
| Level 2 | Exp 2: Cross-graph discovery | Eqs. 6, 8a, 8b — two-stage discovery |
| Level 3 | Exp 3: Multi-domain scaling | I(n,t) = n(n−1)/2 × richness(t)^γ |
| Sensitivity | Exp 4: Parameter sweep | Phase transitions in 4 dimensions |

---

## Directory Tree

```
cross-graph-experiments/
├── CLAUDE.md                              # Project guidelines and rules
├── EXPERIMENTS.md                         # Detailed experiment specifications
├── configs/
│   └── default.yaml                       # Central configuration (no magic numbers in code)
├── src/
│   ├── __init__.py
│   ├── data/                              # Synthetic data generation with ground truth
│   │   ├── alert_generator.py            # SOC alert generation for Exp 1
│   │   └── entity_generator.py           # Entity embeddings for Exp 2–4
│   ├── models/                            # Core mechanisms being tested
│   │   ├── scoring_matrix.py             # Eq. 4: P(action|alert) = softmax(f·W^T / τ)
│   │   └── cross_attention.py            # Eqs. 5–8: cross-graph attention & discovery
│   ├── eval/                              # Evaluation utilities (metrics computed inline)
│   │   └── __init__.py
│   └── viz/                               # Publication-quality visualizations (300 DPI)
│       ├── exp1_charts.py                # Exp 1 — 4 figure types + LaTeX table
│       ├── exp1_blog_chart.py            # Exp 1 — simplified blog version
│       ├── exp2_charts.py                # Exp 2 — F1 bars + P-R curves
│       ├── exp3_charts.py                # Exp 3 — scaling with power-law overlay
│       ├── exp3_blog_chart.py            # Exp 3 — simplified blog version
│       └── exp4_charts.py                # Exp 4 — 2×2 sensitivity panel
├── experiments/
│   ├── exp1_scoring_convergence/
│   │   ├── run.py
│   │   └── results/
│   │       ├── convergence_data.csv      # 350 rows: 10 seeds × 5 methods × 7 checkpoints
│   │       └── weight_evolution.npz      # W matrix snapshots (compounding only)
│   ├── exp2_cross_graph_discovery/
│   │   ├── run.py
│   │   └── results/
│   │       ├── discovery_results.csv     # 1230 rows: full grid sweep
│   │       └── best_configs.json         # Optimal (θ, K) per method and domain pair
│   ├── exp3_multidomain_scaling/
│   │   ├── run.py
│   │   └── results/
│   │       └── scaling_data.csv          # 50 rows: 10 seeds × 5 domain counts
│   └── exp4_sensitivity/
│       ├── run.py
│       └── results/
│           └── sensitivity_data.csv      # 105 rows across 4 parameter sweeps
├── paper_figures/                         # All publication outputs (PDF + PNG)
│   ├── exp1_convergence.{pdf,png}
│   ├── exp1_window_accuracy.{pdf,png}
│   ├── exp1_per_action.{pdf,png}
│   ├── exp1_weight_evolution.{pdf,png}
│   ├── exp1_blog_convergence.{pdf,png}
│   ├── exp2_f1_comparison.{pdf,png}
│   ├── exp2_precision_recall.{pdf,png}
│   ├── exp3_scaling.{pdf,png}
│   ├── exp3_blog_scaling.{pdf,png}
│   └── exp4_sensitivity.{pdf,png}
└── notebooks/                             # Placeholder
```

---

## Source Files

### `src/data/alert_generator.py`

Generates synthetic SOC alerts for Experiment 1.

**Key types:**
- `Alert` (dataclass): `alert_id`, `alert_type`, `factors[6]`, `ground_truth_action`, `is_noisy`
- `AlertGenerator`: `generate(n, seed)` → reproducible list of alerts

**Alert model:**
- 6 alert types (`false_positive`, `routine_alert`, `suspicious_login`, `data_exfil`, `brute_force`, `insider_threat`), each with a Beta-distributed factor profile
- 4 ground-truth actions: `auto_close`, `enrich_and_watch`, `escalate_tier2`, `escalate_incident`
- Noise: 3–10% of alerts get wrong action labels

---

### `src/data/entity_generator.py`

Generates unit-norm entity embeddings for Experiments 2–4.

**Key types:**
- `Entity` (dataclass): `entity_id`, `domain`, `embedding[d]`
- `EntityGenerator`: `generate_domain(name, n, seed)`, `generate_all(seed)`
- `inject_signals(entities_i, entities_j, n_signals, signal_strength, seed)` — plants ground-truth correlations via shared embedding dimensions

**Embedding layout (64-dim default):**

| Dims | Content |
|------|---------|
| 0–5 | Domain-specific semantics — N(domain_mean, σ=0.30) |
| 6–9 | Geographic cluster signal (soft one-hot) |
| 10–13 | Temporal bucket signal (soft one-hot) |
| 14–63 | Background noise — N(0, σ=0.05) |

**Domain profiles (entities per domain):**
- `security`: 200 entities
- `decision_history`: 300 entities
- `threat_intel`: 200 entities
- `network_flow`, `asset_inventory`, `user_behavior`: 200 each (Exp 3 extras)

---

### `src/models/scoring_matrix.py`

Implements **Eq. 4**: `P(action|alert) = softmax(f · W^T / τ)`

**Key type:** `ScoringMatrix`

| Parameter | Default | Role |
|-----------|---------|------|
| `n_actions` | 4 | Number of actions |
| `n_factors` | 6 | Alert factor dimensions |
| `temperature τ` | 0.25 | Softmax sharpness |
| `α_correct` | 0.002 | Hebbian reward step |
| `α_incorrect` | 0.04 | Hebbian penalty step (20× α_correct) |
| `weight_clamp` | 5.0 | Prevents unbounded growth |
| `decay_rate` | 0.001 | Inverse-time LR decay |

**Asymmetric Hebbian update rule:**
```
if correct:   W[action] += α_correct   × lr(t) × factors
if incorrect: W[action] -= α_incorrect × lr(t) × factors

lr(t) = 1 / (1 + decay_rate × t)
```

The 20:1 asymmetry drives rapid specialization; decay stabilizes learning over time.

---

### `src/models/cross_attention.py`

Implements **Eqs. 5–8**: cross-graph attention and entity pair discovery.

**Key type:** `CrossGraphAttention`

| Method | Equation | Description |
|--------|----------|-------------|
| `compute_logits(E_i, E_j)` | Eq. 5 | `S = E_i @ E_j.T / √d` |
| `compute_attention(S)` | Eq. 6 | `A = softmax(S, axis=1)` (rows sum to 1) |
| `compute_output(A, V_j)` | Eq. 6 | `O = A @ V_j` |
| `discover_two_stage(E_i, E_j, θ, K)` | Eqs. 8a+8b | Stage 1 ∩ Stage 2 |
| `discover_logit_only(E_i, E_j, θ)` | Eq. 8a | Pre-softmax threshold only |
| `discover_topk_only(E_i, E_j, K)` | Eq. 8b | Top-K softmax only |
| `cosine_baseline(E_i, E_j, threshold)` | — | Raw cosine (no √d scaling) |

**Two-stage discovery logic:**
```
Stage 1 (Eq. 8a): keep (k, l) where S[k, l] > θ_logit
Stage 2 (Eq. 8b): keep (k, l) where l ∈ top-K(softmax(S[k, :]))
Result:           intersection of Stage 1 and Stage 2
```

---

## Configuration: `configs/default.yaml`

Single source of truth for all experiment parameters — no magic numbers in code.

### Experiment 1
- `n_alerts`: 5000, `noise_rate`: 0.03
- Checkpoints: `[50, 100, 200, 500, 1000, 2000, 5000]`
- Baselines: `compounding`, `symmetric`, `periodic_retrain`, `fixed_weight`, `random_policy`

### Experiment 2
- 3 domains, `embedding_dim`: 64, `signal_strength`: 8.0
- Ground-truth signals: security↔threat=20, decision↔threat=15, security↔decision=15
- θ_logit grid: `[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]`, top-K: `{1, 2, 3, 5}`

### Experiment 3
- `domain_counts`: `[2, 3, 4, 5, 6]`, `entities_per_domain`: 200
- `signals_per_pair`: 5, fixed `θ=0.02`, `K=3`

### Experiment 4 — parameter sweeps

| Sweep | Parameter | Values |
|-------|-----------|--------|
| A | `asymmetry_ratio` | 1, 5, 10, 20, 50 |
| B | `temperature τ` | 0.1, 0.25, 0.5, 1.0, 2.0 |
| C | `noise_rate` | 0.0, 0.03, 0.05, 0.10, 0.20, 0.30 |
| D | `embedding_dim` | 16, 32, 64, 128, 256 |

### Visualization defaults
- DPI: 300, formats: PDF + PNG
- Colors: `main=#1E3A5F`, `baseline_fixed=#94A3B8`, `discovery=#D97706`
- Font sizes: title=13, label=11, tick=9, annotation=8.5

---

## Experiments

### Experiment 1: Scoring Matrix Convergence

**Runner:** `experiments/exp1_scoring_convergence/run.py`
**Outputs:** `convergence_data.csv` (350 rows), `weight_evolution.npz`

Validates Eq. 4 — asymmetric Hebbian learning specializes W to correct SOC actions.

**Setup:** 10 seeds × 5 methods × 7 checkpoints

**Key result:** `compounding` reaches ~69–71% cumulative accuracy at 5000 alerts vs. 25% random baseline.

---

### Experiment 2: Cross-Graph Discovery

**Runner:** `experiments/exp2_cross_graph_discovery/run.py`
**Outputs:** `discovery_results.csv` (1230 rows), `best_configs.json`

Validates Eqs. 6, 8a, 8b — two-stage entity pair discovery across domain graphs.

**Setup:** 10 seeds × 5 methods × 3 domain pairs × config grids

**Key result:** `two_stage` achieves ~116× F1 above the random baseline at optimal (θ, K).

---

### Experiment 3: Multi-Domain Scaling

**Runner:** `experiments/exp3_multidomain_scaling/run.py`
**Outputs:** `scaling_data.csv` (50 rows)

Validates the quadratic scaling law `I(n,t) = n(n−1)/2 × richness(t)^γ`.

**Setup:** 10 seeds × 5 domain counts (2–6)

**Key result:** R² = 0.9995 fit to `discoveries ∝ n^2.30`.

---

### Experiment 4: Parameter Sensitivity

**Runner:** `experiments/exp4_sensitivity/run.py`
**Outputs:** `sensitivity_data.csv` (105 rows)

Sweeps 4 parameters to locate optimal values and phase transitions.

**Key results:**

| Sweep | Best value | Finding |
|-------|-----------|---------|
| A (asymmetry) | ratio = 20 | 0.657 accuracy |
| B (temperature) | τ = 0.25 | 0.657 accuracy |
| C (noise) | < 5% | Sharp degradation above this rate |
| D (embedding_dim) | d = 128 | F1 collapses at d = 256 |

---

## Reproducibility

**Fixed seeds (all experiments):** `[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]`

**Data flow (Exp 2 example):**
```
configs/default.yaml
    → EntityGenerator.generate_all(seed)
    → inject_signals(sec, threat, n=20, strength=8.0)
    → CrossGraphAttention.discover_two_stage(θ, K)
    → compute P/R/F1 vs. ground truth
    → one row in discovery_results.csv
```

**Each source module has a built-in `__main__` self-test** that validates data properties (distributions, norms, update directions, attention row sums, etc.).

---

## Performance Summary

| Experiment | Key metric | Value | Baseline |
|------------|-----------|-------|----------|
| Exp 1 | Cumulative accuracy at 5K alerts | ~69.4% | 25% (random) |
| Exp 2 | Best F1 | ~116× above random | ~0.025 F1 |
| Exp 3 | Power-law fit R² | 0.9995 | — |
| Exp 4 | Critical noise threshold | ~5% | — |

---

## Equations Validated

| Equation | Description | Validator |
|----------|-------------|-----------|
| **Eq. 4** | `P(action\|alert) = softmax(f·W^T / τ)` | Convergence to >69% accuracy |
| **Eq. 5** | `S_ij = E_i·E_j^T / √d` | Logit shape and values |
| **Eq. 6** | `A = softmax(S, axis=1)`, `O = A@V` | Row sums = 1, output shape |
| **Eq. 8a** | `s_kl > θ_logit` | Stage 1 filtering |
| **Eq. 8b** | `entity_l ∈ top-K(softmax(S_k,:))` | Stage 2 filtering |
| **Scaling** | `I(n,t) = n(n−1)/2 × richness(t)^γ` | R²=0.9995, γ≈2.30 |
