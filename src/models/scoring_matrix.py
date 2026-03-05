"""
Scoring matrix implementation for Experiment 1 (Scoring Matrix Convergence).

Implements Eq. 4 from the Cross-Graph Attention paper:

    P(action | alert) = softmax(f @ W.T / tau)

The asymmetric learning rule (20:1 ratio of alpha_incorrect : alpha_correct)
drives rapid convergence to correct action mappings, which is the central
claim validated by Experiment 1.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtracts max before exponentiation."""
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Scoring matrix
# ---------------------------------------------------------------------------

class ScoringMatrix:
    """
    Learns P(action | alert) = softmax(f @ W.T / tau) via asymmetric updates.

    Parameters
    ----------
    n_actions : int
        Number of possible actions (rows of W).
    n_factors : int
        Dimensionality of the alert factor vector (columns of W).
    temperature : float
        Softmax temperature tau.  Lower -> sharper distribution.
    alpha_correct : float
        Learning rate for reinforcing correct actions.
    alpha_incorrect : float
        Learning rate for penalising incorrect actions (default 20x larger).
    weight_clamp : float
        Symmetric hard clamp applied to all W entries after every update.
    init_method : str
        One of ``"uniform"`` | ``"random"`` | ``"zeros"``.
    decay_rate : float
        Inverse-time learning rate decay: effective_rate = 1 / (1 + decay_rate * t).
        decay_rate=0 disables decay (constant learning rates).
    """

    def __init__(
        self,
        n_actions: int = 4,
        n_factors: int = 6,
        temperature: float = 0.25,
        alpha_correct: float = 0.01,
        alpha_incorrect: float = 0.20,
        weight_clamp: float = 5.0,
        init_method: str = "uniform",
        decay_rate: float = 0.001,
    ) -> None:
        self.n_actions = n_actions
        self.n_factors = n_factors
        self.temperature = temperature
        self.alpha_correct = alpha_correct
        self.alpha_incorrect = alpha_incorrect
        self.weight_clamp = weight_clamp
        self.init_method = init_method
        self.decay_rate = decay_rate

        self.step_count: int = 0
        self.W: np.ndarray = self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> np.ndarray:
        """Return a freshly initialised weight matrix of shape (n_actions, n_factors)."""
        if self.init_method == "uniform":
            # All entries identical -> all logits equal for any factor vector
            # -> softmax yields a near-uniform prior over actions.
            return np.full(
                (self.n_actions, self.n_factors),
                fill_value=0.5 / self.n_factors,
                dtype=np.float64,
            )
        elif self.init_method == "random":
            return np.random.normal(0.0, 0.1, size=(self.n_actions, self.n_factors))
        elif self.init_method == "zeros":
            return np.zeros((self.n_actions, self.n_factors), dtype=np.float64)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method!r}")

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, factors: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Greedy decision: return argmax of the action probability distribution.

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors,)

        Returns
        -------
        action_index : int
        probs : np.ndarray, shape (n_actions,)
        """
        logits = factors @ self.W.T                  # (n_actions,)
        probs = softmax(logits / self.temperature)
        return int(np.argmax(probs)), probs

    def decide_stochastic(self, factors: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Stochastic decision: sample action from the probability distribution.
        Used during exploration phases of the experiment.

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors,)

        Returns
        -------
        action_index : int
        probs : np.ndarray, shape (n_actions,)
        """
        logits = factors @ self.W.T
        probs = softmax(logits / self.temperature)
        action = int(np.random.choice(self.n_actions, p=probs))
        return action, probs

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, factors: np.ndarray, action: int, correct: bool) -> None:
        """
        Asymmetric Hebbian update with inverse-time learning rate decay.

        Correct   -> W[action] += alpha_correct   * lr(t) * factors
        Incorrect -> W[action] -= alpha_incorrect * lr(t) * factors

        where lr(t) = 1 / (1 + decay_rate * t) and t is the global step count.
        decay_rate=0 recovers the constant-rate rule.

        All entries are clamped to [-weight_clamp, +weight_clamp] after update.

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors,)
        action : int
            Index of the action that was taken.
        correct : bool
            Whether the action matched the ground-truth label.
        """
        self.step_count += 1
        effective_rate = 1.0 / (1.0 + self.decay_rate * self.step_count)

        if correct:
            self.W[action] += self.alpha_correct * effective_rate * factors
        else:
            self.W[action] -= self.alpha_incorrect * effective_rate * factors
        np.clip(self.W, -self.weight_clamp, self.weight_clamp, out=self.W)

    def score_with_gate(
        self,
        factors: np.ndarray,
        gate_vector: np.ndarray,
    ) -> tuple[int, float]:
        """
        Greedy decision using a per-factor gate vector.

        Applies element-wise gating before computing logits so that
        factors with gate=0 contribute nothing to the action distribution,
        and factors with gate=1 contribute at full weight.

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors,)
            Raw alert factor values.
        gate_vector : np.ndarray, shape (n_factors,)
            Per-factor attention weights, typically in [0, 1].

        Returns
        -------
        action_index : int
            Argmax of the gated action probability distribution.
        confidence : float
            Probability assigned to the chosen action.
        """
        gated_factors = factors * gate_vector
        logits = gated_factors @ self.W.T
        probs = softmax(logits / self.temperature)
        action_idx = int(np.argmax(probs))
        return action_idx, float(probs[action_idx])

    def update_with_gated_factors(
        self,
        action_idx: int,
        outcome: int,
        factors: np.ndarray,
        gate_vector: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """
        Asymmetric Hebbian update using gated factors and an explicit time step.

        Unlike ``update()``, this method:
          - Uses ``outcome`` (+1 / -1) instead of a bool flag.
          - Accepts an explicit time step ``t`` for the LR decay calculation.
          - Does **not** increment ``self.step_count`` (bridge experiments
            manage their own step counter).

        Correct   (outcome > 0) -> W[action_idx] += alpha_correct   * lr(t) * gated_f
        Incorrect (outcome <= 0) -> W[action_idx] -= alpha_incorrect * lr(t) * gated_f

        All W entries are clamped to [-weight_clamp, +weight_clamp] after update.

        Parameters
        ----------
        action_idx : int
            Index of the action that was taken.
        outcome : int
            Oracle feedback: +1 (positive) or -1 (negative).
        factors : np.ndarray, shape (n_factors,)
            Raw alert factor values.
        gate_vector : np.ndarray, shape (n_factors,)
            Per-factor attention weights (from a gating mechanism).
        t : int
            Explicit time step for ``lr(t) = 1 / (1 + decay_rate * t)``.

        Returns
        -------
        delta : np.ndarray, shape (n_factors,)
            The update vector applied to W[action_idx] (before clipping).
        """
        gated_f = factors * gate_vector
        lr = 1.0 / (1.0 + self.decay_rate * t)

        if outcome > 0:
            delta = self.alpha_correct * lr * gated_f
        else:
            delta = -self.alpha_incorrect * lr * gated_f

        self.W[action_idx] += delta
        np.clip(self.W, -self.weight_clamp, self.weight_clamp, out=self.W)
        return delta

    def decide_augmented(
        self,
        factors: np.ndarray,
        category_index: int,
        n_categories: int,
    ) -> tuple[int, float]:
        """
        Score using category-augmented factor vector.

        Appends a one-hot category encoding to the factor vector,
        then computes softmax(f_aug @ W_aug.T / temperature).

        If W has not been expanded yet (n_factors == original factor size),
        expand it: add n_categories new columns initialized to small random
        values.  This is done ONCE on first call; W stays at the new size.

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors_original,)
        category_index : int, 0..n_categories-1
        n_categories : int

        Returns
        -------
        action_idx : int
        confidence : float (max probability)
        """
        f = factors.flatten()

        # Expand W on first call
        if self.W.shape[1] == len(f):
            rng = np.random.default_rng(42)
            W_cat = rng.uniform(-0.01, 0.01, (self.n_actions, n_categories))
            self.W = np.hstack([self.W, W_cat])
            self.n_factors = self.W.shape[1]

        # Build augmented factor vector
        cat_onehot = np.zeros(n_categories)
        cat_onehot[category_index] = 1.0
        f_aug = np.concatenate([f, cat_onehot])

        # Score
        logits = f_aug @ self.W.T / self.temperature
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        action_idx = int(np.argmax(probs))
        return action_idx, float(probs[action_idx])

    def update_augmented(
        self,
        factors: np.ndarray,
        category_index: int,
        n_categories: int,
        action_idx: int,
        outcome_positive: bool,
    ) -> None:
        """
        Update W using the augmented factor vector.
        Same asymmetric Hebbian rule as update(), but with f_aug instead of f.
        No learning-rate decay applied (matches cold-start augmented training).

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors_original,)
        category_index : int
        n_categories : int
        action_idx : int
        outcome_positive : bool
        """
        f = factors.flatten()
        cat_onehot = np.zeros(n_categories)
        cat_onehot[category_index] = 1.0
        f_aug = np.concatenate([f, cat_onehot])

        if outcome_positive:
            delta = self.alpha_correct * f_aug
        else:
            delta = -self.alpha_incorrect * f_aug

        self.W[action_idx, :] += delta
        self.W = np.clip(self.W, -self.weight_clamp, self.weight_clamp)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        """Return a copy of W with shape (n_actions, n_factors)."""
        return self.W.copy()

    def reset(self) -> None:
        """Reinitialise W and step_count using the original ``init_method``."""
        self.W = self._init_weights()
        self.step_count = 0

    def get_entropy(self, factors: np.ndarray) -> float:
        """
        Shannon entropy of the action distribution given *factors*.

        H = -sum(p_i * log(p_i))

        Higher entropy -> more uncertain / less specialised.

        Parameters
        ----------
        factors : np.ndarray, shape (n_factors,)

        Returns
        -------
        float
        """
        _, probs = self.decide(factors)
        probs_safe = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs_safe * np.log(probs_safe)))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    factors = rng.random(6)

    sm = ScoringMatrix()
    print("=== Initial weight matrix W (n_actions x n_factors) ===")
    print(sm.get_weights())

    # --- decide(): probs must sum to 1 ---
    action, probs = sm.decide(factors)
    print(f"\n=== decide() ===")
    print(f"  factors = {np.round(factors, 4)}")
    print(f"  probs   = {np.round(probs, 6)}")
    print(f"  sum     = {probs.sum():.12f}")
    print(f"  action  = {action}")
    assert abs(probs.sum() - 1.0) < 1e-9, f"probs sum = {probs.sum()}"

    # --- correct update: W[0] increases in the factor direction ---
    # effective_rate at step 1 = 1 / (1 + decay_rate * 1)
    sm_c = ScoringMatrix()
    W_before_c = sm_c.get_weights()
    sm_c.update(factors, action=0, correct=True)
    W_after_c = sm_c.get_weights()
    delta_c = W_after_c[0] - W_before_c[0]
    lr1 = 1.0 / (1.0 + sm_c.decay_rate * 1)   # effective rate at step 1
    expected_c = sm_c.alpha_correct * lr1 * factors
    print(f"\n=== update(correct=True, action=0) ===")
    print(f"  lr(t=1)  = {lr1:.6f}  (decay_rate={sm_c.decay_rate})")
    print(f"  dW[0]    = {np.round(delta_c, 6)}")
    print(f"  Expected = {np.round(expected_c, 6)}")
    print(f"  W after:\n{np.round(W_after_c, 6)}")
    assert np.allclose(delta_c, expected_c, atol=1e-10), "Correct update wrong direction"
    assert np.all(delta_c >= 0), "Correct update should increase all active weights"

    # --- incorrect update: W[0] decreases (and 20x larger magnitude) ---
    sm_i = ScoringMatrix()
    W_before_i = sm_i.get_weights()
    sm_i.update(factors, action=0, correct=False)
    W_after_i = sm_i.get_weights()
    delta_i = W_after_i[0] - W_before_i[0]
    lr1_i = 1.0 / (1.0 + sm_i.decay_rate * 1)
    expected_i = -sm_i.alpha_incorrect * lr1_i * factors
    print(f"\n=== update(correct=False, action=0) ===")
    print(f"  dW[0]    = {np.round(delta_i, 6)}")
    print(f"  Expected = {np.round(expected_i, 6)}")
    print(f"  W after:\n{np.round(W_after_i, 6)}")
    assert np.allclose(delta_i, expected_i, atol=1e-10), "Incorrect update wrong direction"
    assert np.all(delta_i <= 0), "Incorrect update should decrease all active weights"

    # --- 20:1 asymmetry ratio ---
    ratio = np.linalg.norm(delta_i) / np.linalg.norm(delta_c)
    print(f"\n=== Asymmetry ratio (should be 20.0) ===")
    print(f"  |d incorrect| / |d correct| = {ratio:.6f}")
    assert abs(ratio - 20.0) < 1e-6, f"Expected 20:1 ratio, got {ratio:.6f}"

    # --- weight clamp ---
    sm_clamp = ScoringMatrix(weight_clamp=5.0)
    sm_clamp.update(np.ones(6) * 100.0, action=0, correct=True)
    assert sm_clamp.W.max() <= 5.0, "Upper clamp failed"
    assert sm_clamp.W.min() >= -5.0, "Lower clamp failed"
    print(f"\n=== Weight clamp test (factors=100*1) ===")
    print(f"  max(W) = {sm_clamp.W.max():.4f}  min(W) = {sm_clamp.W.min():.4f}")

    # --- entropy: uniform init + any factors -> maximum entropy = log(n_actions) ---
    sm_ent = ScoringMatrix(init_method="uniform")
    h = sm_ent.get_entropy(factors)
    max_h = float(np.log(sm_ent.n_actions))
    print(f"\n=== Entropy test (uniform init -> max entropy) ===")
    print(f"  H = {h:.8f}   log({sm_ent.n_actions}) = {max_h:.8f}")
    assert abs(h - max_h) < 1e-6, f"Expected max entropy {max_h}, got {h}"

    # --- reset restores original weights ---
    sm_reset = ScoringMatrix()
    W_init = sm_reset.get_weights()
    sm_reset.update(factors, action=1, correct=False)
    sm_reset.reset()
    assert np.allclose(W_init, sm_reset.get_weights()), "Reset did not restore W"
    print(f"\n=== Reset test ===")
    print(f"  W matches initial: True")

    # --- score_with_gate: uniform gate (ones) must match decide() exactly ---
    factors_gv = np.random.default_rng(7).random(6)
    sm_gv = ScoringMatrix()
    act_decide, probs_decide = sm_gv.decide(factors_gv)
    act_gate, conf_gate = sm_gv.score_with_gate(factors_gv, np.ones(6))
    assert act_gate == act_decide, (
        f"score_with_gate(ones) action {act_gate} != decide() {act_decide}"
    )
    assert abs(conf_gate - probs_decide[act_decide]) < 1e-10, (
        f"score_with_gate(ones) confidence {conf_gate} != decide() {probs_decide[act_decide]}"
    )
    print(f"\n=== score_with_gate(uniform gate) ===")
    print(f"  action={act_gate}, confidence={conf_gate:.6f}  (matches decide())")

    # --- score_with_gate: zero gate -> all logits zero -> uniform probs -> conf=0.25 ---
    _, conf_zero = sm_gv.score_with_gate(factors_gv, np.zeros(6))
    assert abs(conf_zero - 0.25) < 1e-6, (
        f"score_with_gate(zeros) confidence {conf_zero:.6f} != 0.25"
    )
    print(f"\n=== score_with_gate(zero gate) ===")
    print(f"  confidence={conf_zero:.6f}  (expected 0.25 uniform)")

    # --- update_with_gated_factors: gated-off factors produce zero delta ---
    factors_gu = np.random.default_rng(11).random(6)   # all values in (0, 1)
    gate_mask = np.array([1.0, 1.0, 0.0, 0.0, 0.5, 0.5])

    # outcome = +1
    sm_gu = ScoringMatrix()
    delta_pos = sm_gu.update_with_gated_factors(0, +1, factors_gu, gate_mask, t=1)
    assert delta_pos[2] == 0.0, f"FAIL: delta[2]={delta_pos[2]} (gate=0 should give delta=0)"
    assert delta_pos[3] == 0.0, f"FAIL: delta[3]={delta_pos[3]} (gate=0 should give delta=0)"
    assert delta_pos[0] > 0.0,  f"FAIL: delta[0]={delta_pos[0]} should be > 0"
    assert delta_pos[1] > 0.0,  f"FAIL: delta[1]={delta_pos[1]} should be > 0"
    assert sm_gu.step_count == 0, (
        f"FAIL: step_count={sm_gu.step_count} (update_with_gated_factors must not increment)"
    )
    print(f"\n=== update_with_gated_factors(outcome=+1, gate=[1,1,0,0,0.5,0.5]) ===")
    print(f"  delta      = {np.round(delta_pos, 6)}")
    print(f"  delta[2]==0: {delta_pos[2]==0.0},  delta[3]==0: {delta_pos[3]==0.0}")
    print(f"  step_count = {sm_gu.step_count}  (not incremented)")

    # outcome = -1
    sm_gu2 = ScoringMatrix()
    delta_neg = sm_gu2.update_with_gated_factors(0, -1, factors_gu, gate_mask, t=1)
    assert delta_neg[0] < 0.0,  f"FAIL: delta[0]={delta_neg[0]} should be < 0"
    assert delta_neg[2] == 0.0, f"FAIL: delta[2]={delta_neg[2]} (gate=0 should give delta=0)"
    print(f"\n=== update_with_gated_factors(outcome=-1) ===")
    print(f"  delta      = {np.round(delta_neg, 6)}")
    print(f"  delta[0]<0: {delta_neg[0]<0.0},  delta[2]==0: {delta_neg[2]==0.0}")

    # --- augmented scoring: W expands to (4, 11), update keeps same shape ---
    print("\n--- augmented scoring test ---")
    sm_aug = ScoringMatrix(n_actions=4, n_factors=6, temperature=0.25)
    rng_aug = np.random.default_rng(99)
    act_aug, conf_aug = sm_aug.decide_augmented(
        rng_aug.random(6), category_index=0, n_categories=5
    )
    assert sm_aug.W.shape == (4, 11), (
        f"FAIL: W.shape={sm_aug.W.shape} after decide_augmented, expected (4, 11)"
    )
    sm_aug.update_augmented(
        rng_aug.random(6), category_index=0, n_categories=5,
        action_idx=act_aug, outcome_positive=True,
    )
    assert sm_aug.W.shape == (4, 11), (
        f"FAIL: W.shape={sm_aug.W.shape} after update_augmented, expected (4, 11)"
    )
    print(f"  W.shape after decide_augmented: {sm_aug.W.shape}  (4, 11) OK")
    print(f"  W.shape after update_augmented: {sm_aug.W.shape}  (4, 11) OK")
    print("augmented scoring check passed")

    print("\nAll checks passed")
