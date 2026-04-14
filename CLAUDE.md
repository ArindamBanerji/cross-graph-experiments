# Cross-Graph Attention Experiments

## Purpose
Experimental validation for the Cross-Graph Attention paper.
Four experiments validating the framework at three levels:
- Level 1: Scoring matrix convergence (Exp 1)
- Level 2: Cross-graph discovery precision/recall (Exp 2)
- Level 3: Multi-domain scaling (Exp 3)
- Sensitivity analysis (Exp 4)

## Key Rules
- Do not start debugger/servers
- Do not run git commands unless explicitly asked
- All experiments must be reproducible: fixed random seeds, logged parameters
- Every chart must be publication-quality (300 DPI, PDF + PNG)
- Every experiment outputs: figures (.pdf/.png), tables (.tex), raw data (.csv)
- Use numpy for all computation (no torch/tensorflow needed)
- All random seeds: [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

## Architecture
- src/data/: Synthetic data generation with ground truth
- src/models/: The actual mechanisms being tested
- src/eval/: Measurement and metrics
- src/viz/: Visualization (publication-quality)
- configs/: YAML configuration files (no magic numbers in code)
- experiments/: Experiment runners and results

## Mathematical Context
This validates the Cross-Graph Attention framework:
- Eq. 4: P(action|alert) = softmax(f · W^T / τ)  — scoring matrix
- Eq. 6: S_ij = E_i · E_j^T / √d  — cross-graph attention
- Eq. 8a: s_kl > θ_logit (pre-softmax threshold)
- Eq. 8b: entity_l ∈ top-K(softmax(S_k,:)) (top-K selection)
- I(n,t) = n(n-1)/2 × richness(t)^γ with γ > 1

## Conventions
- Config files: YAML (configs/*.yaml)
- Results: experiments/expN/results/
- Figures: paper_figures/
- All experiments print progress to stdout
- CSV output for all measured data

### No Silent Failure on Displayed Metrics
- If a try/except computes a NUMBER shown in the UI: the except block
  must set a flag (estimated=True, source="fallback") — never bare pass
- If a try/except computes OPTIONAL enrichment: bare pass is acceptable
- NEVER hardcode a number that looks like a computed metric (0.89, 23, 127)
  without a comment explaining why it's a constant and not computed
- The test: if the graph is empty, does the UI show zeros or plausible-looking
  fake numbers? If fake numbers: it's a mockup, not a fallback.

### AGE Is Not Neo4j — Three Critical Differences

1. **SET n = {props} WIPES all other properties**
   - NEVER: `SET d = {category: 'x'}` — destroys every other property
   - ALWAYS: `SET d.category = 'x'` — preserves all other properties
   - SAFE for bulk: `SET d += {a: 1, b: 2}` — merges, preserves existing
   - AGEClient rejects the destructive form with ValueError

2. **Concurrent writes to the same node fail**
   - "Entity failed to be updated: 3" = PostgreSQL row lock conflict
   - AGEClient retries with jitter (3 attempts, 100-250ms backoff)
   - Avoid concurrent writes to the same node when possible

3. **Decision nodes must be created atomically with their edge**
   - ALWAYS: `MATCH (a:Alert) CREATE (d:Decision {...})-[:DECIDED_ON]->(a)`
   - NEVER: CREATE Decision as one query, then edge as a second
   - If MATCH finds no Alert, no Decision is created (proven atomic)