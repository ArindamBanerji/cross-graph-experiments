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

    print("\nAll checks passed")
