"""
Gating mechanisms for Bridge Layer Experiments (EXP 5-9).

Three mechanisms control per-category, per-factor attention weights G[c, i],
which modulate the factor vector before it enters the scoring matrix:

  UniformGating  — baseline: G = ones, no selective attention, no learning.
  HebbianGating  — online learning from oracle outcomes (Eq. 4d).  Gate values
                   grow/shrink based on factor × weight-magnitude signals.
  MIGating       — offline mutual-information estimation (Opus's proposal).
                   Fitted once on a batch; gate values are then static.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# UniformGating
# ---------------------------------------------------------------------------

class UniformGating:
    """
    Baseline gating: all gate values are 1.0 — factors are unmodulated.

    No learning occurs.  Use as a reference point when comparing against
    Hebbian or MI gating in bridge layer experiments.

    Parameters
    ----------
    n_categories : int
        Number of alert categories.
    n_factors : int
        Number of alert factors.
    """

    def __init__(self, n_categories: int, n_factors: int) -> None:
        self._n_categories = n_categories
        self._n_factors = n_factors
        self.G: np.ndarray = np.ones((n_categories, n_factors), dtype=np.float64)

    def get_gate(self, category_index: int) -> np.ndarray:
        """Return a ones vector of length n_factors (independent of category)."""
        return np.ones(self._n_factors, dtype=np.float64)

    def update(self) -> None:
        """No-op: uniform gating does not learn."""


# ---------------------------------------------------------------------------
# HebbianGating
# ---------------------------------------------------------------------------

class HebbianGating:
    """
    Online Hebbian gating from oracle outcomes (Eq. 4d, bridge_layer_design_v1).

    G[c, i] accumulates evidence that factor *i* is informative for category *c*
    based on a Hebbian-style signal: factor_value × weight_magnitude, scaled by
    the oracle outcome (+1 to increase importance, -1 to decrease).

    With ``damping=True`` the signal is normalised by the L1 norm of
    ``w_mag``, keeping update magnitudes bounded even when W grows large.
    Without damping, large weights produce proportionally larger gate updates.

    Parameters
    ----------
    n_categories : int
        Number of alert categories.
    n_factors : int
        Number of alert factors.
    learning_rate : float
        Step size for each update.  Default: 0.005.
    min_gate : float
        Hard lower bound on all gate values after clipping.  Default: 0.05.
    max_gate : float
        Hard upper bound on all gate values after clipping.  Default: 1.0.
    damping : bool
        If True, normalise the update signal by ||w_mag||_1 + 1e-8.
    """

    def __init__(
        self,
        n_categories: int,
        n_factors: int,
        learning_rate: float = 0.005,
        min_gate: float = 0.05,
        max_gate: float = 1.0,
        damping: bool = True,
    ) -> None:
        self._n_categories = n_categories
        self._n_factors = n_factors
        self.learning_rate = learning_rate
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.damping = damping
        self.G: np.ndarray = np.ones((n_categories, n_factors), dtype=np.float64)

    def get_gate(self, category_index: int) -> np.ndarray:
        """Return a copy of the gate vector for the given category."""
        return self.G[category_index].copy()

    def update(
        self,
        category_index: int,
        factor_vector: np.ndarray,
        action_index: int,
        outcome: int,
        W: np.ndarray,
    ) -> None:
        """
        Update the gate row for ``category_index`` based on the oracle outcome.

        Eq. 4d:
            w_mag  = |W[action_index, :]|
            signal = f * (w_mag / (||w_mag||_1 + 1e-8))   if damping=True
                   = f * w_mag                             if damping=False

            G[c,:] += learning_rate * signal   if outcome > 0
            G[c,:] -= learning_rate * signal   if outcome <= 0
            G[c,:] clamped to [min_gate, max_gate]

        Parameters
        ----------
        category_index : int
            Row of G to update.
        factor_vector : np.ndarray, shape (n_factors,)
            Alert factor values for the current decision.
        action_index : int
            Index of the action taken by the scoring matrix.
        outcome : int
            Oracle feedback: +1 (positive) or -1 (negative).
        W : np.ndarray, shape (n_actions, n_factors)
            Current scoring matrix weights (read-only).
        """
        c = category_index
        f = np.asarray(factor_vector, dtype=np.float64)
        w_mag = np.abs(W[action_index])

        if self.damping:
            signal = f * (w_mag / (np.sum(w_mag) + 1e-8))
        else:
            signal = f * w_mag

        if outcome > 0:
            self.G[c] += self.learning_rate * signal
        else:
            self.G[c] -= self.learning_rate * signal

        np.clip(self.G[c], self.min_gate, self.max_gate, out=self.G[c])


# ---------------------------------------------------------------------------
# MIGating
# ---------------------------------------------------------------------------

class MIGating:
    """
    Offline mutual-information gating (Opus's proposal).

    After a batch of decisions is collected, ``fit()`` estimates the mutual
    information between each binned factor and decision correctness for every
    category.  Gate values are set via a sigmoid centred at ``threshold``:

        G[c, i] = sigmoid((MI[c,i] − threshold) / threshold)

    so that MI = threshold → gate = 0.5, MI >> threshold → gate → 1.0,
    MI << threshold → gate → 0.0.

    Once fitted the gate is static: ``update()`` is a no-op.  Categories
    with zero samples in the fitting batch retain their initial gate of 1.0.

    Parameters
    ----------
    n_categories : int
        Number of alert categories.
    n_factors : int
        Number of alert factors.
    threshold : float
        Pivot MI value for the sigmoid.  Default: 0.1.
    """

    _N_BINS: int = 10

    def __init__(
        self,
        n_categories: int,
        n_factors: int,
        threshold: float = 0.1,
    ) -> None:
        self._n_categories = n_categories
        self._n_factors = n_factors
        self._threshold = threshold
        self.G: np.ndarray = np.ones((n_categories, n_factors), dtype=np.float64)
        self._fitted: bool = False

    def fit(
        self,
        alerts: list,
        system_actions: list[int],
        gt_actions: list[int],
    ) -> None:
        """
        Estimate per-(category, factor) mutual information and update G.

        Algorithm per (category c, factor i):
          1. Collect all samples where ``alert.category_index == c``.
          2. Bin factor *i* values into ``_N_BINS`` quantile bins using
             ``np.quantile`` + ``searchsorted`` (handles ties gracefully).
          3. Build a (n_bins, 2) contingency table {incorrect, correct};
             add 1e-10 epsilon everywhere to avoid log(0).
          4. Compute MI = Σ p(b,k) log(p(b,k) / (p(b) * p(k))).
          5. G[c,i] = sigmoid((MI − threshold) / threshold).

        Categories with zero samples are skipped; their G row stays at 1.0.

        Parameters
        ----------
        alerts : list
            Objects with ``.category_index`` (int) and ``.factors``
            (array-like, length n_factors).
        system_actions : list[int]
            Action index chosen by the system for each alert.
        gt_actions : list[int]
            Ground-truth action index for each alert.
        """
        n = len(alerts)
        for c in range(self._n_categories):
            indices = [i for i in range(n) if alerts[i].category_index == c]
            if not indices:
                continue  # edge case: no samples → G[c,:] unchanged (stays 1.0)

            f_mat = np.array(
                [alerts[i].factors for i in indices], dtype=np.float64
            )  # (n_c, n_factors)
            correct = np.array(
                [system_actions[i] == gt_actions[i] for i in indices], dtype=bool
            )  # (n_c,)

            for fi in range(self._n_factors):
                f_vals = f_mat[:, fi]

                # Bin into _N_BINS quantile bins; searchsorted handles ties
                edges = np.quantile(f_vals, np.linspace(0.0, 1.0, self._N_BINS + 1))
                internal_edges = edges[1:-1]  # 9 interior edges → 10 bins
                bin_idx = np.searchsorted(internal_edges, f_vals, side="right")
                # bin_idx values are in [0, _N_BINS - 1]

                # Contingency table: rows=bins, cols={incorrect(0), correct(1)}
                contingency = np.zeros((self._N_BINS, 2), dtype=np.float64)
                for b, k in zip(bin_idx, correct):
                    contingency[b, int(k)] += 1.0
                contingency += 1e-10  # epsilon everywhere to avoid log(0)

                # Compute mutual information
                total = contingency.sum()
                p_joint = contingency / total
                p_bin     = p_joint.sum(axis=1, keepdims=True)   # (n_bins, 1)
                p_correct = p_joint.sum(axis=0, keepdims=True)   # (1, 2)
                mi = float(np.sum(p_joint * np.log(p_joint / (p_bin * p_correct))))

                # Sigmoid gate: MI = threshold → 0.5; MI >> threshold → 1.0
                self.G[c, fi] = 1.0 / (
                    1.0 + np.exp(-((mi - self._threshold) / self._threshold))
                )

        self._fitted = True

    def fit_from_data(self, alerts: list, n_actions: int) -> None:
        """
        Compute MI(factor_i, gt_action | category) directly from alert data.
        No oracle or scoring matrix needed.

        For each category c, bins each factor into 10 equal-width bins [0, 1],
        builds a (n_bins × n_actions) contingency table, and computes MI.
        Gate values are normalized so the most informative factor = 1.0,
        with a minimum floor of 0.05.

        Parameters
        ----------
        alerts : list
            Objects with ``.category_index`` (int), ``.factors``
            (array-like, length n_factors), and ``.gt_action_index`` (int).
        n_actions : int
            Number of possible actions (columns in the contingency table).
        """
        n = len(alerts)
        bin_edges      = np.linspace(0.0, 1.0, self._N_BINS + 1)
        internal_edges = bin_edges[1:-1]   # 9 interior edges → 10 bins

        for c in range(self._n_categories):
            indices = [i for i in range(n) if alerts[i].category_index == c]
            if not indices:
                continue

            f_mat   = np.array(
                [alerts[i].factors for i in indices], dtype=np.float64
            )   # (n_c, n_factors)
            gt_acts = np.array(
                [alerts[i].gt_action_index for i in indices], dtype=np.int64
            )   # (n_c,)

            mi_vals = np.zeros(self._n_factors, dtype=np.float64)

            for fi in range(self._n_factors):
                f_vals  = f_mat[:, fi].clip(0.0, 1.0)
                bin_idx = np.searchsorted(internal_edges, f_vals, side="right")

                # Contingency table: rows=bins, cols=gt_action
                contingency = np.zeros((self._N_BINS, n_actions), dtype=np.float64)
                for b, a in zip(bin_idx, gt_acts):
                    contingency[b, int(a)] += 1.0

                total = float(contingency.sum())
                if total < 1:
                    continue

                p_joint = contingency / total
                p_bin   = p_joint.sum(axis=1)   # (n_bins,)
                p_act   = p_joint.sum(axis=0)   # (n_actions,)

                mi = 0.0
                for b in range(self._N_BINS):
                    for a in range(n_actions):
                        if contingency[b, a] == 0:
                            continue
                        pba = p_joint[b, a]
                        mi += pba * np.log(pba / (p_bin[b] * p_act[a]) + 1e-10)
                mi_vals[fi] = max(0.0, mi)

            # Normalize: most informative factor → 1.0, floor at 0.05
            max_mi = float(np.max(mi_vals))
            self.G[c, :] = mi_vals / (max_mi + 1e-10)
            np.clip(self.G[c, :], 0.05, None, out=self.G[c, :])

        self._fitted = True

    def get_gate(self, category_index: int) -> np.ndarray:
        """
        Return a copy of the gate vector for the given category.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "MIGating.get_gate() called before fit(). Call fit() first."
            )
        return self.G[category_index].copy()

    def update(self) -> None:
        """No-op: MI gating is static once fitted."""


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from types import SimpleNamespace

    # -------------------------------------------------------------------
    # a. UniformGating: all gates must be exactly ones
    # -------------------------------------------------------------------
    ug = UniformGating(5, 6)
    for c in range(5):
        gate = ug.get_gate(c)
        assert np.all(gate == 1.0), f"FAIL: UniformGating category {c} not all ones"
    print("=== UniformGating: all get_gate() == ones(6) for c in 0..4 OK ===")

    # Shared factor vector for Hebbian tests b, c, d
    fv = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])

    # -------------------------------------------------------------------
    # b. HebbianGating(damping=True): factor 0 (high f) should grow faster
    #    than factor 5 (low f) after 10 positive updates.
    #    G is reset to 0.5 so there is room below max_gate.
    # -------------------------------------------------------------------
    W_b = np.random.default_rng(42).random((4, 6))
    heb_b = HebbianGating(5, 6, damping=True)
    heb_b.G[:] = 0.5  # start below max_gate to allow measurable differentiation
    for _ in range(10):
        heb_b.update(0, fv, 0, +1, W_b)
    assert heb_b.G[0, 0] > heb_b.G[0, 5], (
        f"FAIL: G[0,0]={heb_b.G[0,0]:.4f} not > G[0,5]={heb_b.G[0,5]:.4f}"
    )
    print(f"\n=== HebbianGating(damping=True): 10 positive updates ===")
    print(f"  G[0,:] = {np.round(heb_b.G[0,:], 4)}")
    print(f"  G[0,0] > G[0,5]: {heb_b.G[0,0]:.4f} > {heb_b.G[0,5]:.4f} OK")

    # -------------------------------------------------------------------
    # c. HebbianGating(damping=True): large W → update magnitude bounded
    #    Normalisation keeps |dG| well below 0.1 per step.
    # -------------------------------------------------------------------
    W_large = np.full((4, 6), 3.0)
    heb_c = HebbianGating(5, 6, damping=True)
    heb_c.G[:] = 0.5
    max_changes_damped: list[float] = []
    for _ in range(100):
        G_before = heb_c.G[0].copy()
        heb_c.update(0, fv, 0, +1, W_large)
        max_changes_damped.append(float(np.max(np.abs(heb_c.G[0] - G_before))))
    max_damped = max(max_changes_damped)
    assert max_damped < 0.1, (
        f"FAIL: damped max |dG| per step {max_damped:.6f} >= 0.1"
    )
    print(f"\n=== HebbianGating(damping=True): W=3.0, 100 positive updates ===")
    print(f"  Max |dG| per step = {max_damped:.6f} (< 0.1) OK")

    # -------------------------------------------------------------------
    # d. HebbianGating(damping=False): same large-W setup → bigger updates
    #    Undamped max change must exceed damped max change.
    # -------------------------------------------------------------------
    heb_d = HebbianGating(5, 6, damping=False)
    heb_d.G[:] = 0.5
    max_changes_undamped: list[float] = []
    for _ in range(100):
        G_before = heb_d.G[0].copy()
        heb_d.update(0, fv, 0, +1, W_large)
        max_changes_undamped.append(float(np.max(np.abs(heb_d.G[0] - G_before))))
    max_undamped = max(max_changes_undamped)
    assert max_undamped > max_damped, (
        f"FAIL: undamped {max_undamped:.6f} not > damped {max_damped:.6f}"
    )
    print(f"\n=== HebbianGating(damping=False): W=3.0, 100 positive updates ===")
    print(f"  Max |dG| per step = {max_undamped:.6f}  (> damped {max_damped:.6f}) OK")

    # -------------------------------------------------------------------
    # e. MIGating: factor 0 perfectly predicts correctness → highest gate
    #    For 200 category-0 alerts: correct iff factor_0 > 0.5.
    #    All other factors are random noise.
    # -------------------------------------------------------------------
    rng_mi = np.random.default_rng(7)
    n_mi = 200
    test_alerts:  list = []
    test_sys:     list[int] = []
    test_gt:      list[int] = []

    for j in range(n_mi):
        f = rng_mi.random(6)
        # factor 0 > 0.5 → system chose correctly (sys_act == gt_act == 0)
        # factor 0 ≤ 0.5 → system chose incorrectly (sys_act=1, gt_act=0)
        is_correct = f[0] > 0.5
        gt_act  = 0
        sys_act = 0 if is_correct else 1
        test_alerts.append(SimpleNamespace(category_index=0, factors=f))
        test_sys.append(sys_act)
        test_gt.append(gt_act)
    # Categories 1-4 have zero samples → G[1:,:] stays at 1.0 (edge case)

    mi_gate = MIGating(5, 6)
    mi_gate.fit(test_alerts, test_sys, test_gt)

    assert mi_gate.G[0, 0] > 0.7, (
        f"FAIL: G[0,0] = {mi_gate.G[0,0]:.4f} not > 0.7"
    )
    assert mi_gate.G[0, 0] > mi_gate.G[0, 3], (
        f"FAIL: G[0,0]={mi_gate.G[0,0]:.4f} not > G[0,3]={mi_gate.G[0,3]:.4f}"
    )
    print(f"\n=== MIGating: factor 0 perfectly predicts correctness ===")
    print(f"  G[0,:] = {np.round(mi_gate.G[0,:], 4)}")
    print(f"  G[0,0] = {mi_gate.G[0,0]:.4f} > 0.7 OK")
    print(f"  G[0,0] > G[0,3]: {mi_gate.G[0,0]:.4f} > {mi_gate.G[0,3]:.4f} OK")

    # -------------------------------------------------------------------
    # f. fit_from_data: realistic_profiles → non-uniform G per category
    # -------------------------------------------------------------------
    print("\n--- fit_from_data test ---")
    import sys as _sys
    import yaml as _yaml
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parent.parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    from src.data.category_alert_generator import CategoryAlertGenerator as _CAG

    _cfg_path = _root / "configs" / "default.yaml"
    with open(_cfg_path) as _fh:
        _raw = _yaml.safe_load(_fh)
    _bc = _raw["bridge_common"]
    _rp = _raw["realistic_profiles"]

    _gen_fd = _CAG(
        categories=_bc["categories"],
        actions=_bc["actions"],
        factors=_bc["factors"],
        action_conditional_profiles=_rp["action_conditional_profiles"],
        gt_distributions=_rp["category_gt_distributions"],
        factor_sigma=float(_bc["factor_sigma"]),
        noise_rate=0.0,
        seed=42,
    )
    _alerts_fd = _gen_fd.generate_batch(400)
    assert len(_alerts_fd) == 2000, f"FAIL: expected 2000 alerts, got {len(_alerts_fd)}"

    _mi_fd = MIGating(5, 6)
    _mi_fd.fit_from_data(_alerts_fd, n_actions=4)

    print("  G matrix (row=category, col=factor):")
    for _ci, _cat in enumerate(_bc["categories"]):
        _vals = " ".join(f"{v:.2f}" for v in _mi_fd.G[_ci])
        print(f"    {_cat}: [{_vals}]")

    _spread_count = sum(
        1 for _ci in range(5)
        if float(np.max(_mi_fd.G[_ci]) - np.min(_mi_fd.G[_ci])) > 0.3
    )
    assert _spread_count >= 3, (
        f"FAIL: only {_spread_count}/5 categories have G-spread > 0.3 (need >= 3)"
    )
    _g_std = float(np.std(_mi_fd.G))
    assert _g_std > 0.1, f"FAIL: G std = {_g_std:.4f}, expected > 0.1 (non-uniform)"
    print("  fit_from_data check passed")

    print("\nAll checks passed")
