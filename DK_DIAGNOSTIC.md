# DK-DIAGNOSTIC

Read-only audit of DiagonalKernel evidence in `cross-graph-experiments` and `graph-attention-engine-v50`. No files in either target repository were modified during the audit.

## Scope result

The two requested repositories contain no H-KERNEL run or real-data DK artifact. The only de-circularization artifact found was outside both repositories, under `copilot-sdk/experiments/htrust/`. It is reported separately below and is not counted as evidence stored in either target repository.

## Inventory

| Experiment | Location | Data | Conditions / seeds | Key result | Artifact status |
|---|---|---|---:|---|---|
| V-MV-KERNEL-HET | `graph-attention-engine-v50/docs/math_synopsis_v14.md` | Synthetic | 390 cells; seed detail unavailable | +13.2pp SOC; +6.8pp S2P | Documentation only; raw run/results and manifest absent |
| UNI-DK-01 v5.3 | `graph-attention-engine-v50/docs/math_synopsis_v14.md` | Synthetic | 1,500 cells; 30 seeds/cell | +7.67pp at noise ratio 5.0 | Raw artifacts absent from both requested repositories |
| EXP-A4-DIAGONAL | `cross-graph-experiments/experiments/exp_a4_diagonal/` | Synthetic SOC personas | 20 conditions × 15 seeds = 300 runs | Mean A=4 DK−L2: +1.78pp; A=5: +2.08pp | Results present; no SHA manifest |
| V-HC-CONFIG / D2 | `cross-graph-experiments/experiments/factorial/` | Synthetic healthcare | 3 conditions × 15 seeds | Day-60 DK 73.90% vs L2 69.53%; +4.37pp | Results present; no SHA manifest |
| V-S2P-HETERO / D3 | `cross-graph-experiments/experiments/factorial/` | Synthetic S2P | 18 conditions × 10 seeds | Six direct gaps +2.11 to +11.43pp; mean +7.38pp | Results present; no SHA manifest |
| SVM003b diagonal | `cross-graph-experiments/experiments/svm003b_v_cga_frozen/` | Synthetic enrichment profiles | 3 profiles × 30 seeds | Mean additional DK lift +1.76pp | Results present; no SHA manifest |
| SVM003b corrected | `cross-graph-experiments/experiments/svm003b_corrected/` | Corrected synthetic apparatus | 2 reported profiles × 30 seeds | Mean additional DK lift −0.07pp; effectively null | Results present; no SHA manifest |
| V-KERNEL-W | `cross-graph-experiments/experiments/svm004_v_kernel_w/` | Synthetic analytical/Fisher test | 8 conditions; seed count unavailable | Pearson r=.9669 for σ and effective rate | Not an accuracy DK-vs-L2 test |
| V-KERNEL-W empirical | `cross-graph-experiments/experiments/svm004b_v_kernel_w/` | Synthetic | 8 conditions; seed count unavailable | r=.96692, p=.03308 | No DK/L2 accuracy comparison |
| D4 KernelSelector | `cross-graph-experiments/experiments/factorial/results/kernel_deliverables.json` | Synthetic operational simulation | 2 deployment streams | Healthcare DK−L2 −0.40pp; S2P +1.18pp | Agreement metric, not oracle accuracy |
| MR-COLAB-01 | Referenced in `graph-attention-engine-v50/docs/math_synopsis_v14.md` | Synthetic healthcare | 50 seeds; raw artifact absent | Reported +8.86pp peak | Documentation only |

The graph repository contains the implementation and tests. `gae/kernels.py` defines `DiagonalKernel` as `W=diag(1/sigma^2)`, but it contains no DK experiment raw directory or H-KERNEL result.

## Synthetic evidence summary

The canonical documented total is:

- V-MV-KERNEL-HET: 390 cells
- UNI-DK-01 v5.3: 1,500 cells
- Total: **1,890 synthetic cells**

V-MV used heterogeneous, known per-factor sigma profiles. The reported peak was **+13.2pp on SOC** and **+6.8pp on S2P**.

UNI-DK used fixed mean sigma 0.175, noise ratios 1, 1.5, 2, 3, and 5, 30 seeds per cell, 5,000 decisions, and 25% verification. Its reported asymptotic DK−L2 gaps were:

| Noise ratio | DK−L2 |
|---:|---:|
| 1.0 | 0.00pp |
| 1.5 | +0.72pp |
| 2.0 | +1.99pp |
| 3.0 | +4.43pp |
| 5.0 | +7.67pp |

The synthetic runs used the GAE `ProfileScorer`/kernel infrastructure. The corrected SVM003b result is a material caveat: its mean additional DK lift was −0.07pp, showing sensitivity to the apparatus/data-generation correction.

## Real-data evidence summary

No H-KERNEL experiment or real-data DK artifact exists in either requested repository.

The adjacent out-of-scope run is `copilot-sdk/experiments/htrust/`. It uses three real UCI datasets, three preprocessing conventions, and 20 seeds:

| Dataset | Records | Features | Domain |
|---|---:|---:|---|
| Credit Default | 30,000 | 23 | Credit risk |
| COIL 2000 | 5,822 | 85 | Insurance/customer targeting |
| Adult | 32,561 | 6 numeric/ordinal | Income classification |

The raw weighted-distance column is the DK comparison and raw uniform-distance is L2. Means from `copilot-sdk/experiments/htrust/v2_raw/c1_factorial_cells.csv`:

| Dataset | Convention | L2 | DK | DK−L2 |
|---|---|---:|---:|---:|
| Credit | Min-max | 63.960% | 64.846% | +0.886pp |
| Credit | Robust quantile | 64.841% | 64.133% | −0.708pp |
| Credit | Standardized | 65.090% | 64.846% | −0.243pp |
| COIL 2000 | Min-max | 64.842% | 65.847% | +1.005pp |
| COIL 2000 | Robust quantile | 66.467% | 65.840% | −0.627pp |
| COIL 2000 | Standardized | 65.531% | 65.849% | +0.319pp |
| Adult | Min-max | 72.391% | 73.466% | +1.075pp |
| Adult | Robust quantile | 74.245% | 74.550% | +0.305pp |
| Adult | Standardized | 73.598% | 73.466% | −0.132pp |

There are **180 real runs**: 3 datasets × 3 conventions × 20 seeds. The sign distribution is mixed: 5 positive and 4 negative dataset/convention means. No dataset is positive under every convention.

## H-KERNEL deep dive

The adjacent report is `copilot-sdk/docs/design/blogs/hypotheses/h-trust/results_v1.md`. It reports bootstrap confidence intervals over the 20 seeds. Reliability DK−L2 intervals were:

| Dataset | Min-max | Robust quantile | Standardized |
|---|---:|---:|---:|
| Credit | +0.886pp [+.617, +1.165] | −0.708pp [−.841, −.559] | −0.243pp [−.322, −.157] |
| COIL 2000 | +1.005pp [+.345, +1.695] | −0.627pp [−1.385, +.167] | +0.319pp [−.074, +.725] |
| Adult | +1.075pp [+.944, +1.191] | +0.305pp [+.177, +.426] | −0.132pp [−.274, +.015] |

The design strengths were independent UCI provenance, held-out test evaluation, group/duplicate handling, 20 seeds, and explicit preprocessing sensitivity.

The main caveats are that the run is outside both target repositories, does not use the production GAE scorer, is tabular rather than graph-mediated, and does not establish that DK can never help on real operational data. It does establish that the broad scale-robust DK advantage is not supported.

## Evidence balance and assessment

By raw run count:

- Canonical synthetic evidence: 1,890 cells
- Real de-circularization evidence: 180 runs
- Real/synthetic ratio: approximately 9.5%

A conservative canonical count is approximately 990 positive/claimed-support cells, 150 uniform-noise null cells, with V-MV per-cell raw outcomes unavailable. The real evidence challenges scale robustness in all three datasets but is not proportional to the synthetic factorial by cell count.

The defensible conclusion is:

> Synthetic DK gains are well characterized, but their real-world generality is not established; the scale-robust deployment claim is invalidated by the available de-circularization evidence.

This does **not** justify the stronger claim that DK cannot help on any real graph or production dataset.

## Missing expected artifacts

The following are referenced or implied but absent from the two requested repositories:

- H-KERNEL/H-TRUST run directory
- H-KERNEL run script
- H-KERNEL per-cell results
- H-KERNEL result manifest with result-file hashes
- Real-data copies or provenance package for H-KERNEL
- `cross-graph-experiments/experiments/v_mv_kernel/`
- V-MV-KERNEL-HET raw notebooks/results
- UNI-DK-01 v5.3 raw notebooks/results
- Per-cell V-MV DK/L2 outcomes
- Reproducible links from the synthetic factorial artifacts to the stated 390/1,500-cell claims
