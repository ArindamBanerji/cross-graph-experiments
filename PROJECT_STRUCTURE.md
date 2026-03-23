# Cross-Graph Experiments — Project Structure
**Last Updated:** March 23, 2026
**Version:** main branch
**Tests:** ~104 experiments complete
**Purpose:** Experiment harness, synthetic data generation, validation suite for GAE and copilot architecture. NOT a product codebase — this is a research and validation repository. Results here inform design decisions; they do not ship to customers.

---

## Critical Values (canonical — used by all harnesses)

| Parameter | Value | Source |
|---|---|---|
| `θ_min` | **0.467** | T_max=21 days, η=0.05, N_half=14. All harnesses use this. |
| `τ` (default) | **0.1** | V3B validated ECE=0.036. Per-deployment τ from P28 TD-034 sweep (τ=0.08 optimal on realistic distributions). |
| `η_confirm` | **0.05** | Clean signal path |
| `η_override` | **0.01** | Noisy signal path — P0 fix |
| `η_neg` | **0.05** | η_neg=1.0 is FORBIDDEN |

---

## Directory Tree

```
cross-graph-experiments/
├── experiments/
│   ├── persona_sweeps/
│   │   └── run_harness.py          # Main sweep harness — θ_min=0.467 (fixed Mar 23)
│   ├── block5b_proxy/
│   │   └── run_all_harnesses.py    # BA_THETA_MIN=0.467 (fixed Mar 23)
│   ├── bridge_a_phase_b/
│   │   ├── run.py                  # THETA_MIN=0.467 (fixed Mar 23)
│   │   └── charts.py              # THETA_MIN=0.467 (fixed Mar 23)
│   ├── meta3_breach_window/
│   │   └── run.py                  # THETA_MIN=0.467 (fixed Mar 23)
│   ├── exp_refer_learn/            # EXP-REFER-LEARN ✅
│   ├── exp_refer_coverage/         # EXP-REFER-COVERAGE ✅
│   ├── exp_refer_layered/          # EXP-REFER-LAYERED ✅ (SHIPS Layer 2)
│   ├── exp_a4_diagonal/            # EXP-A4-DIAGONAL ✅ (A=4 confirmed)
│   ├── v_mv_kernel/                # V-MV-KERNEL ✅ 390 cells
│   ├── v_hc_config/                # V-HC-CONFIG-DIAGONAL ✅ 4 healthcare personas
│   └── [~100 other experiments]
├── configs/
│   └── s2p_domain_config.py        # S2PDomainConfig d=8 (5 categories, 5 actions, 8 factors)
├── paper_figures/                  # 143 figures — linked from arXiv. DO NOT DELETE.
└── [harness utilities]
```

---

## Experiment Series Summary

| Series | What | Status |
|---|---|---|
| Series 1-7 | Core validation (V1A-V3B, EXP-C1, EXP-E1, EXP-B1, realistic 50-seed) | ✅ Complete |
| Series 8 | arXiv paper figures | ✅ Complete — archive |
| Series 9 | OP series (synthesis layer, GATE-OP PASSED) | ✅ Complete |
| Series 10 | Kernel factorial (V-MV-KERNEL 390 cells, DiagonalKernel +13.2pp) | ✅ Complete |
| Series 11 | Healthcare personas (4 personas, corr=0.990) | ✅ Complete |
| Series 12 | Phase 1 sweeps (24 personas), Priority 1 validation | ✅ Complete |
| **Series 13** | **Referral architecture (4 experiments)** | **✅ Complete** |

### Series 13 Key Results (March 21, 2026)

| Experiment | Finding |
|---|---|
| EXP-A4-DIAGONAL | A=4 vs A=5: 13pp gap, kernel-independent. A=4 confirmed. |
| EXP-REFER-LEARN | Factor-only classifiers all fail. Signal is in context, not geometry. |
| EXP-REFER-COVERAGE | 65.5% rule-expressible, 13.8% context, 20.7% emergent. |
| EXP-REFER-LAYERED | **SHIPS:** Rules R1-R7 = 72.7% DR, 12% FPR. Confidence gate = 14% precision (harmful). |

---

## S2P Domain Config

`configs/s2p_domain_config.py` — S2P second domain configuration:
- C=5 categories, A=5 actions, d=8 factors
- Correlation prior: Regime A, 28 pairs from two-judge research
- penalty_ratio=5.0
- θ_min=0.35
- DiagonalKernel validated: +6.8pp on heterogeneous noise

---

## Pending Experiments (Phase 1 of MAP v3)

### Batch 1 — Highest risk (run first)
- **V-CGA-FROZEN**: Does graph enrichment lift frozen scorer? 50 seeds, 2 conditions. If FAILS → "second compounding pathway" claim needs design fix (GraphAttentionBridge), not language softening.
- **V-ENRICHMENT-NEGATIVE**: Can enrichment hurt? 2 personas, 20 seeds.

### Batch 2 — Architecture validation
- V-SIM (P28 pipeline on 9 LLM-judge streams)
- EXP-S2-REPRO at A=4 (poisoning resilience, was run at A=5)
- TD-034 LLM-judge streams (τ=0.08 on realistic distributions)

### Batch 3 — Free analysis on existing 390-cell factorial
- V-MV-RISK, V-MV-CONVERGENCE, V-MV-CONSERVATION, V-UCL-VALUE
- Gap G4 interaction mini-factorial

### Batch 4 — Safety
- P4-F (adversarial analyst, gradual quality degradation)
- P4-COMPLACENCY (quality degrades while override rate stays constant)

---

## Rules for This Repo

- **θ_min = 0.467 everywhere.** Any harness using 0.434 is stale — fix immediately.
- **Synthetic data is canonical.** LLM-judge personas generate all validation streams. No "waiting for real customer data."
- **Results are research outputs.** Historical JSON/CSV files with measured values (including values near 0.434) are results — do NOT modify them when fixing θ_min.
- **paper_figures/ is permanent.** Linked from arXiv paper and blog posts. Never delete.
- **Series 13 is settled.** A=4 confirmed. Rules R1-R7 ship. Confidence gate is action routing only. Do not reopen these decisions.

---

## Claim Policy

If an experiment forces a claim to weaken:
1. Identify the missing architectural capability
2. Add it to the MAP v3 Phase 3 feature queue with a concrete design answer
3. Only THEN update the claims registry

Do NOT soften language without a design fix. A weakened claim = a design gap.

---

*cross-graph-experiments · main · ~104 experiments · March 23, 2026*
