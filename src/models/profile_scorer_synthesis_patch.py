"""
profile_scorer_synthesis_patch.py
==================================
SYNTH-EXP-0 deliverable.

THIS FILE IS NOT A STANDALONE MODULE.
It shows EXACTLY what to add to src/models/profile_scorer.py.
Follow the instructions below for each section.

CRITICAL RULE (enforced by the signature, not just convention):
  ProfileScorer.update() does NOT accept a synthesis parameter.
  Centroid learning NEVER uses σ.
  σ is awareness (days). μ is experience (months). They are different epistemic categories.

=============================================================
STEP 1: Add this import at the top of profile_scorer.py
=============================================================
"""

# Add at the top of profile_scorer.py, after existing imports:
_IMPORT_ADDITIONS = '''
from __future__ import annotations
from typing import Optional, Tuple

# Add this import (synthesis.py must exist in src/models/)
try:
    from src.models.synthesis import SynthesisBias
except ImportError:
    SynthesisBias = None  # Graceful degradation — experiments without synthesis installed
'''

"""
=============================================================
STEP 2: Add synthesis_active field to ScoringResult
=============================================================
Find your existing ScoringResult dataclass and add one field:
"""

# In ScoringResult (or equivalent return type), add:
_SCORING_RESULT_ADDITION = '''
@dataclass
class ScoringResult:
    # ... your existing fields (probabilities, action_index, confidence, etc.) ...
    synthesis_active: bool = False   # ← ADD THIS FIELD
'''

"""
=============================================================
STEP 3: Modify ProfileScorer.score() to accept synthesis
=============================================================
Find your existing score() method. Its current signature is roughly:
    def score(self, f: np.ndarray, category_index: int) -> ScoringResult:

Change to:
    def score(self, f: np.ndarray, category_index: int,
              synthesis: Optional["SynthesisBias"] = None) -> ScoringResult:

And modify the internal distance/tau computation as shown below.
"""

_SCORE_METHOD_MODIFICATION = '''
def score(
    self,
    f: np.ndarray,
    category_index: int,
    synthesis: Optional["SynthesisBias"] = None,
) -> ScoringResult:
    """
    Score factor vector f for category category_index.

    When synthesis is None OR synthesis.lambda_coupling == 0:
        Behavior is IDENTICAL to the original Eq. 4-final:
        P(a|f,c) = softmax(−‖f − μ[c,a,:]‖² / τ)

    When synthesis is active (lambda > 0 and sigma != 0):
        Implements Eq. 4-synthesis:
        P(a|f,c,σ) = softmax(−(‖f − μ[c,a,:]‖² + λ·σ[c,a]) / (τ · τ_mod))

    The experience term ‖f − μ‖² is UNCHANGED.
    σ adds an awareness bias on top.
    λ=0 is the kill switch — exact Eq. 4-final restored.
    """
    # --- Compute L2 distances (UNCHANGED from current implementation) ---
    mu_slice = self.mu[category_index]          # shape: (n_actions, n_factors)
    diff = f - mu_slice                          # shape: (n_actions, n_factors)
    distances = np.sum(diff ** 2, axis=-1)       # shape: (n_actions,)

    # --- Apply synthesis bias (NEW — only when active) ---
    synthesis_active = False
    tau_eff = self.tau                           # Default: unchanged temperature

    if (synthesis is not None
            and synthesis.lambda_coupling > 0.0
            and synthesis.active_claims > 0):
        sigma_slice = synthesis.sigma[category_index, :]   # shape: (n_actions,)
        distances = distances + synthesis.lambda_coupling * sigma_slice
        tau_eff = self.tau
        synthesis_active = True

    # --- Softmax over (negative) distances (UNCHANGED structure) ---
    logits = -distances / tau_eff
    logits -= logits.max()                       # Numerical stability
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum()
    action_index = int(np.argmax(probabilities))
    confidence = float(probabilities[action_index])

    return ScoringResult(
        probabilities=probabilities,
        action_index=action_index,
        confidence=confidence,
        synthesis_active=synthesis_active,
        # ... your other existing fields ...
    )
'''

"""
=============================================================
STEP 4: Add score_counterfactual() method to ProfileScorer
=============================================================
Add this new method AFTER score():
"""

_SCORE_COUNTERFACTUAL = '''
def score_counterfactual(
    self,
    f: np.ndarray,
    category_index: int,
    synthesis: "SynthesisBias",
) -> Tuple["ScoringResult", "ScoringResult"]:
    """
    Return (result_with_synthesis, result_without_synthesis).

    Enables counterfactual advisory logging:
        "Without the active campaign intelligence, the system
         would have suppressed this alert. With it, it escalates."

    Used by:
    - EXP-S3 (loop independence): track which actions change due to σ
    - Tab 5 (display): "synthesis changed this decision"
    - Future: counterfactual-aware update if S3 fails

    IMPORTANT: Both results use the SAME μ snapshot.
    This is a pure read operation — no centroid updates happen here.
    """
    result_with = self.score(f, category_index, synthesis)
    result_without = self.score(f, category_index, None)
    return result_with, result_without
'''

"""
=============================================================
STEP 5: Verify update() is NOT modified
=============================================================
Your existing update() method:
    def update(self, f, category_index, action_index, correct):

Must remain EXACTLY as-is. Do not add synthesis parameter. Do not pass σ.

The signature enforces the architectural constraint:
Loop 2 centroid learning physically cannot access σ through this API.
"""

_UPDATE_UNCHANGED_REMINDER = '''
# DO NOT MODIFY THIS METHOD — centroids NEVER use synthesis bias
def update(
    self,
    f: np.ndarray,
    category_index: int,
    action_index: int,
    correct: bool,
) -> None:
    """
    Centroid pull/push — Eq. 4b-final.
    Correct:   μ[c,a,:] ← clip(μ + η_eff · (f − μ), 0, 1)
    Incorrect: μ[c,a,:] ← clip(μ − η_neg_eff · (f − μ), 0, 1)
    NOTE: No σ parameter. This is intentional and architecturally required.
    """
    # ... your existing implementation, UNCHANGED ...
    pass
'''

"""
=============================================================
VERIFICATION TESTS
Run these after applying the patch to profile_scorer.py:
=============================================================
"""

VERIFICATION_TESTS = '''
# Test 1: Neutral synthesis == no synthesis
python -c "
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias
import numpy as np
np.random.seed(42)
mu = np.random.rand(6,4,6)*0.5+0.25
ps = ProfileScorer(mu, ['escalate','investigate','suppress','monitor'])
f = np.random.rand(6)
r1 = ps.score(f, 0)
r2 = ps.score(f, 0, SynthesisBias.neutral(6,4))
assert np.allclose(r1.probabilities, r2.probabilities), f'FAIL: {r1.probabilities} != {r2.probabilities}'
print('Test 1 PASS: neutral synthesis == no synthesis')
"

# Test 2: Positive σ reduces action probability
python -c "
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias
import numpy as np
np.random.seed(42)
mu = np.random.rand(6,4,6)*0.5+0.25
ps = ProfileScorer(mu, ['escalate','investigate','suppress','monitor'])
f = np.random.rand(6)
sigma = np.zeros((6,4))
sigma[0,2] = 0.5  # suppress for travel_anomaly: positive = less likely
bias = SynthesisBias(sigma=sigma, active_claims=1, lambda_coupling=0.1)
r1 = ps.score(f, 0)
r2 = ps.score(f, 0, bias)
assert r2.probabilities[2] < r1.probabilities[2], f'FAIL: positive sigma should decrease P(suppress)'
assert r2.synthesis_active == True
print('Test 2 PASS: positive sigma decreases action probability')
"

# Test 3: Lambda=0 kill switch
python -c "
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias
import numpy as np
np.random.seed(42)
mu = np.random.rand(6,4,6)*0.5+0.25
ps = ProfileScorer(mu, ['escalate','investigate','suppress','monitor'])
f = np.random.rand(6)
sigma = np.ones((6,4))  # Big bias
bias_zero = SynthesisBias(sigma=sigma, active_claims=5, lambda_coupling=0.0)
r1 = ps.score(f, 0)
r2 = ps.score(f, 0, bias_zero)
assert np.allclose(r1.probabilities, r2.probabilities), 'FAIL: lambda=0 must give exact Eq. 4-final'
print('Test 3 PASS: lambda=0 kill switch works')
"

# Test 4: Counterfactual returns two results
python -c "
from src.models.profile_scorer import ProfileScorer
from src.models.synthesis import SynthesisBias
import numpy as np
np.random.seed(42)
mu = np.random.rand(6,4,6)*0.5+0.25
ps = ProfileScorer(mu, ['escalate','investigate','suppress','monitor'])
f = np.random.rand(6)
sigma = np.zeros((6,4)); sigma[1,0] = -0.5  # credential_access, escalate: more likely
bias = SynthesisBias(sigma=sigma, active_claims=1, lambda_coupling=0.1)
r_with, r_without = ps.score_counterfactual(f, 1, bias)
assert r_with.synthesis_active == True
assert r_without.synthesis_active == False
print('Test 4 PASS: score_counterfactual returns (with, without) tuple')
"
'''

if __name__ == "__main__":
    print("This file is a PATCH GUIDE — not a runnable module.")
    print("Apply the changes described above to src/models/profile_scorer.py")
    print("Then run the VERIFICATION_TESTS using the commands in the VERIFICATION_TESTS string above.")
    print()
    print("Files to create/modify:")
    print("  MODIFY: src/models/profile_scorer.py  (add synthesis param to score(), add score_counterfactual)")
    print("  CREATE: src/models/synthesis.py        (SynthesisBias dataclass)")
    print("  CREATE: src/models/rule_projector.py   (RuleBasedProjector)")
    print("  CREATE: src/data/claim_generator.py    (SOC_SYNTHESIS_RULES, generate functions)")
    print("  CREATE: src/viz/synthesis_common.py    (shared visualization)")
