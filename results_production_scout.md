# V-CGA-FROZEN production-trace scout

## Step 1 — schema inventory (scout completed before measurement)

The most detailed persisted V-CGA-FROZEN trace is `experiments/v_cga_frozen_two_stream_v2.json`.
Its top-level keys are `metadata`, `stream_a`, and `stream_b`; each stream has 2,700 rows
(30 seeds × 90 days), for 5,400 daily records total. The committed validator independently
declares the row schema as `seed`, `stream`, `day`, `centroid_frozen`, `enrichment_active`,
`graph_entity_count`, `graph_edge_count`, `accuracy_rolling_10`, and `reached_85_pct`
(`validate_vcga_v2.ps1:28-41`). The input content hash is recorded in
`experiments/production_scout/manifest.json`.

| Required trace | What exists | Consequence |
|---|---|---|
| Per-decision outcome/correctness over time | NOT FOUND in the persisted JSON; only daily `accuracy_rolling_10` aggregates exist | A per-decision learning-rate fit is not identifiable from this artifact |
| Enriched vs unenriched tag | Deployment/stream-level only: `stream_a` has `enrichment_active=false`; `stream_b` has `true` | Supports a paired treatment-vs-control day analysis, not within-deployment σ cohorts |
| Per-factor values or per-factor σ per decision | NOT FOUND in the persisted JSON | No decision-level high-σ/low-σ split; no update-direction check |
| Class separation Δf | NOT FOUND in the persisted JSON | No recovery of factor-specific mislabeling or enrichment-target dependence |
| Time sequence | Present at daily resolution (`day=1..90`) | Supports day-binned aggregate gap analysis, not decision-count bins |

The generator code defines a per-factor sigma schedule in `experiments/v_cga_frozen/run_v2.py`:
`get_sigma_vector(day, condition)` (`:102-128`), but those vectors are not serialized into the
two-stream JSON. Likewise, the runner creates per-decision `post_accs` in memory and returns
them (`run_v2.py:316-333`), while the persisted two-stream schema contains no `post_accs` field.
Therefore the requested measurements are: σ⊥μ coupling = **not possible**; mean-update
direction = **not possible**; P2 natural decay = **possible only as a deployment/day-level
aggregate diagnostic**, with decision-count bins unavailable.

## Step 2 — σ⊥μ coupling

**Not measurable from the persisted V-CGA-FROZEN traces.** The required split needs either
per-decision sigma/precision, or a persisted per-decision cohort label, plus decision-level
correctness and preferably centroid/update snapshots. None is present in the artifact above.
Consequently there is no defensible coupling percentage or confidence interval, and no mean-update
direction result. The metadata's `seed_enrichment_rates` is an enrichment-volume schedule, not a
per-decision sigma or precision measurement (`v_cga_frozen_two_stream_v2.json:metadata`).

The source runner does specify that condition A lowers selected factor noise over time and B holds
the base values (`run_v2.py:102-128`), but this is a model/configuration fact, not an observed
cohort label in the persisted trace. It cannot establish a measured coupling of learning rate to
sigma.

## Step 3 — P2 natural decay: supported day-level substitute

Because decisions are not persisted, the requested decision-count bins (`1-10`, `11-50`,
`51-200`, `200+`) cannot be computed. I used the closest supported, pre-existing time axis:
paired treatment-minus-control `accuracy_rolling_10` by day, then averaged within day bins
`1-10`, `11-50`, and `51-90` across the 30 paired seeds. CIs are 95% Student-t intervals over
the 30 seed-level bin means. Derived rows are in `experiments/production_scout/gap_by_day.csv`
and `gap_by_day_bin.csv`.

| Day bin | Treatment − control rolling accuracy | 95% CI | Seed signs (+/−/0) |
|---|---:|---:|---:|
| 1–10 | −3.3391 pp | [−3.3410, −3.3372] pp | 0 / 30 / 0 |
| 11–50 | −18.2008 pp | [−18.2191, −18.1825] pp | 0 / 30 / 0 |
| 51–90 | −2.8546 pp | [−2.9143, −2.7949] pp | 0 / 30 / 0 |

The curve is **not** a monotone shrinkage toward zero from day 1: the gap begins near zero,
becomes most negative at day 45 (−22.6540 pp), then recovers toward zero; selected daily values
are approximately day 10 −6.297 pp, day 45 −22.654 pp, day 60 −3.923 pp, and day 90 +0.034 pp.
The post-peak recovery slope over days 45–90 is +0.004777 per day (0.4777 pp/day), R²=0.720;
that is a recovery segment, not a single natural-decay fit for the whole series. The full-series
linear slope is +0.001535 per day, R²=0.212, so a single global decay model is not warranted.
The supported chart is `experiments/production_scout/gap_vs_day.png`.

This aggregate result should not be read as a σ effect: stream-level treatment also changes
entity/edge counts and freeze/unfreeze state. The trace has no decision-level counter to separate
those mechanisms.

## Step 4 — dependency findings

| Potential dependency | Finding | Evidence / limitation |
|---|---|---|
| σ→rate varies with decision count | NOT DETERMINED | No per-decision sigma, correctness, or update trace |
| Enrichment confounded with deployment/stream | YES, structurally | `stream_a` is always control/unenriched and `stream_b` always treatment/enriched; there is no within-stream randomized enrichment label |
| Enrichment confounded with entity/edge growth | YES | The persisted rows include graph counts; the treatment stream changes counts while control remains flat (`validate_vcga_v2.ps1:59-77`) |
| Freeze/unfreeze confounding | YES | `centroid_frozen` is a persisted phase flag, and the validator expects treatment seed 0 to change at day 46 (`validate_vcga_v2.ps1:79-89`) |
| Gap depends on which factors were enriched / Δf | NOT DETERMINED | Factor-level sigma and class-separation values are absent from the persisted artifact |
| Natural gap is monotone decay | NO in the supported day-level read | It deepens through day 45, then recovers; all three bins remain negative |

The negative treatment-minus-control gap is therefore a descriptive property of these two persisted
streams, not an attributable estimate of σ⊥μ or P2. The schema couples enrichment to stream,
entity/edge growth, and phase timing; those cells are not independently identifiable from this
artifact.

## Artifacts and reproducibility

Derived artifacts and their SHA-256 hashes are listed in
`experiments/production_scout/manifest.json`, including the source JSON hash, the day-level and
bin-level tables, `derived_summary.json`, and `gap_vs_day.png`. The analysis is scorer-free and
uses only the persisted stream arrays. No experiment was re-run and no scorer was imported.

