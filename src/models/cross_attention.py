"""
Cross-graph attention mechanism implementing Equations 5-8.

  Eq. 5  : S     = E_i @ E_j.T / sqrt(d)       (similarity logits)
  Eq. 6  : A     = softmax(S, axis=1)           (row-wise attention weights)
           O     = A @ V_j                      (attended representation)
  Eq. 8a : S[k,l] > theta_logit                 (pre-softmax threshold)
  Eq. 8b : l in top-K(softmax(S[k,:]))          (post-softmax top-K filter)

All embedding matrices are assumed to have unit-norm rows (L2-normalised),
consistent with the output of EntityGenerator.generate_domain().
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _row_softmax(X: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (subtracts row max before exp)."""
    shifted = X - X.max(axis=1, keepdims=True)
    exp_X   = np.exp(shifted)
    return exp_X / exp_X.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# CrossGraphAttention
# ---------------------------------------------------------------------------

class CrossGraphAttention:
    """
    Cross-graph attention mechanism (Equations 5-8).

    Parameters
    ----------
    d : int or None
        Embedding dimension used as the scaling factor sqrt(d) in Eq. 5.
        If None (default), inferred from E_i on each call.
    """

    def __init__(self, d: int | None = None) -> None:
        self.d = d

    def _scale(self, E: np.ndarray) -> float:
        return float(np.sqrt(self.d if self.d is not None else E.shape[1]))

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_logits(self, E_i: np.ndarray, E_j: np.ndarray) -> np.ndarray:
        """
        Similarity logit matrix (Eq. 5).

        S = E_i @ E_j.T / sqrt(d)

        Parameters
        ----------
        E_i : np.ndarray, shape (m_i, d)
        E_j : np.ndarray, shape (m_j, d)

        Returns
        -------
        S : np.ndarray, shape (m_i, m_j)
        """
        assert E_i.ndim == 2, (
            f"E_i must be 2-D, got shape {E_i.shape}"
        )
        assert E_j.ndim == 2, (
            f"E_j must be 2-D, got shape {E_j.shape}"
        )
        assert E_i.shape[1] == E_j.shape[1], (
            f"Embedding dims must match: E_i d={E_i.shape[1]}, E_j d={E_j.shape[1]}"
        )
        return E_i @ E_j.T / self._scale(E_i)

    def compute_attention(self, S: np.ndarray) -> np.ndarray:
        """
        Row-wise softmax to produce attention weights (Eq. 6).

        A = softmax(S, axis=1)

        Each row sums to 1.

        Parameters
        ----------
        S : np.ndarray, shape (m_i, m_j)

        Returns
        -------
        A : np.ndarray, shape (m_i, m_j)
        """
        assert S.ndim == 2, f"S must be 2-D, got shape {S.shape}"
        return _row_softmax(S)

    def compute_output(self, A: np.ndarray, V_j: np.ndarray) -> np.ndarray:
        """
        Attended cross-graph representation (Eq. 6).

        O = A @ V_j

        Parameters
        ----------
        A   : np.ndarray, shape (m_i, m_j)  — attention weights
        V_j : np.ndarray, shape (m_j, d)    — entity embeddings (value matrix)

        Returns
        -------
        O : np.ndarray, shape (m_i, d)
        """
        assert A.ndim == 2, f"A must be 2-D, got shape {A.shape}"
        assert V_j.ndim == 2, f"V_j must be 2-D, got shape {V_j.shape}"
        assert A.shape[1] == V_j.shape[0], (
            f"A.shape[1]={A.shape[1]} must equal V_j.shape[0]={V_j.shape[0]}"
        )
        return A @ V_j

    # ------------------------------------------------------------------
    # Discovery methods
    # ------------------------------------------------------------------

    def discover_two_stage(
        self,
        E_i: np.ndarray,
        E_j: np.ndarray,
        theta_logit: float,
        top_k: int,
    ) -> list[tuple[int, int, float, float]]:
        """
        Two-stage cross-graph entity discovery (Eq. 8a + 8b).

        Stage 1 (Eq. 8a): find all (k, l) where S[k, l] > theta_logit.
        Stage 2 (Eq. 8b): keep only those where l is in the top-K of
                          softmax(S[k, :]) (i.e. top-K by attention weight).

        Parameters
        ----------
        E_i         : np.ndarray, shape (m_i, d)
        E_j         : np.ndarray, shape (m_j, d)
        theta_logit : float  — pre-softmax threshold (Eq. 8a)
        top_k       : int   — candidates per source entity kept after Eq. 8b

        Returns
        -------
        list[tuple[int, int, float, float]]
            (i_idx, j_idx, logit_score, attention_weight),
            sorted by logit_score descending.
        """
        assert E_i.ndim == 2, f"E_i must be 2-D, got shape {E_i.shape}"
        assert E_j.ndim == 2, f"E_j must be 2-D, got shape {E_j.shape}"
        assert top_k >= 1, f"top_k must be >= 1, got {top_k}"

        S = self.compute_logits(E_i, E_j)    # (m_i, m_j)
        A = self.compute_attention(S)         # (m_i, m_j)

        m_j   = E_j.shape[0]
        k_eff = min(top_k, m_j)

        # Pre-compute top-K column index sets per row (from attention matrix)
        topk_cols_per_row = np.argsort(A, axis=1)[:, -k_eff:]   # (m_i, k_eff)
        topk_sets = [set(row.tolist()) for row in topk_cols_per_row]

        # Stage 1: threshold on raw logits
        rows_s1, cols_s1 = np.where(S > theta_logit)

        results: list[tuple[int, int, float, float]] = []
        for k, l in zip(rows_s1.tolist(), cols_s1.tolist()):
            if l in topk_sets[k]:                    # Stage 2 filter
                results.append((k, l, float(S[k, l]), float(A[k, l])))

        results.sort(key=lambda t: t[2], reverse=True)
        return results

    def discover_logit_only(
        self,
        E_i: np.ndarray,
        E_j: np.ndarray,
        theta_logit: float,
    ) -> list[tuple[int, int, float]]:
        """
        Stage-1 only discovery (Eq. 8a): pre-softmax threshold.

        Parameters
        ----------
        E_i         : np.ndarray, shape (m_i, d)
        E_j         : np.ndarray, shape (m_j, d)
        theta_logit : float

        Returns
        -------
        list[tuple[int, int, float]]
            (i_idx, j_idx, logit_score) for all pairs where S[k,l] > theta_logit,
            sorted by logit_score descending.
        """
        assert E_i.ndim == 2, f"E_i must be 2-D, got shape {E_i.shape}"
        assert E_j.ndim == 2, f"E_j must be 2-D, got shape {E_j.shape}"

        S = self.compute_logits(E_i, E_j)
        rows, cols = np.where(S > theta_logit)
        results = [
            (int(k), int(l), float(S[k, l]))
            for k, l in zip(rows.tolist(), cols.tolist())
        ]
        results.sort(key=lambda t: t[2], reverse=True)
        return results

    def discover_topk_only(
        self,
        E_i: np.ndarray,
        E_j: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, int, float]]:
        """
        Stage-2 only discovery (Eq. 8b): top-K attention weight per row.

        For each source entity k, returns the top_k target entities ranked
        by attention weight (regardless of whether the logit exceeds any
        threshold).

        Parameters
        ----------
        E_i   : np.ndarray, shape (m_i, d)
        E_j   : np.ndarray, shape (m_j, d)
        top_k : int  — number of targets returned per source entity

        Returns
        -------
        list[tuple[int, int, float]]
            (i_idx, j_idx, attention_weight), sorted by attention_weight
            descending.  Total entries = m_i * min(top_k, m_j).
        """
        assert E_i.ndim == 2, f"E_i must be 2-D, got shape {E_i.shape}"
        assert E_j.ndim == 2, f"E_j must be 2-D, got shape {E_j.shape}"
        assert top_k >= 1, f"top_k must be >= 1, got {top_k}"

        S     = self.compute_logits(E_i, E_j)
        A     = self.compute_attention(S)
        m_i   = E_i.shape[0]
        m_j   = E_j.shape[0]
        k_eff = min(top_k, m_j)

        # argsort ascending; take the last k_eff indices per row (highest values)
        topk_cols = np.argsort(A, axis=1)[:, -k_eff:]    # (m_i, k_eff)

        results: list[tuple[int, int, float]] = []
        for row_k in range(m_i):
            for col_l in topk_cols[row_k].tolist():
                results.append((row_k, col_l, float(A[row_k, col_l])))
        results.sort(key=lambda t: t[2], reverse=True)
        return results

    def cosine_baseline(
        self,
        E_i: np.ndarray,
        E_j: np.ndarray,
        threshold: float,
    ) -> list[tuple[int, int, float]]:
        """
        Cosine similarity baseline discovery.

        Because rows are unit-normed, cosine similarity reduces to the plain
        dot product E_i @ E_j.T (no sqrt(d) scaling — that is specific to
        the attention logits in Eq. 5).

        Parameters
        ----------
        E_i       : np.ndarray, shape (m_i, d)  — unit-normed rows
        E_j       : np.ndarray, shape (m_j, d)  — unit-normed rows
        threshold : float  — minimum cosine similarity to retain

        Returns
        -------
        list[tuple[int, int, float]]
            (i_idx, j_idx, cosine_sim) for all pairs where sim > threshold,
            sorted by cosine_sim descending.
        """
        assert E_i.ndim == 2, f"E_i must be 2-D, got shape {E_i.shape}"
        assert E_j.ndim == 2, f"E_j must be 2-D, got shape {E_j.shape}"
        assert E_i.shape[1] == E_j.shape[1], (
            f"Embedding dims must match: {E_i.shape[1]} vs {E_j.shape[1]}"
        )

        C = E_i @ E_j.T    # unit-normed: this is the cosine similarity matrix
        rows, cols = np.where(C > threshold)
        results = [
            (int(k), int(l), float(C[k, l]))
            for k, l in zip(rows.tolist(), cols.tolist())
        ]
        results.sort(key=lambda t: t[2], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d   = 16

    # --- build test matrices ---
    raw_i = rng.normal(size=(5, d))
    E_i   = raw_i / np.linalg.norm(raw_i, axis=1, keepdims=True)   # (5, 16)

    raw_j = rng.normal(size=(8, d))
    E_j   = raw_j / np.linalg.norm(raw_j, axis=1, keepdims=True)   # (8, 16)

    # Plant: E_j[3] nearly identical to E_i[0] (tiny noise, then re-normalise)
    E_j[3] = E_i[0] + rng.normal(scale=0.01, size=d)
    E_j[3] /= np.linalg.norm(E_j[3])

    cga = CrossGraphAttention()

    # --- compute_logits: shape (5, 8) ---
    S = cga.compute_logits(E_i, E_j)
    assert S.shape == (5, 8), f"Logit shape: expected (5,8), got {S.shape}"
    print(f"  compute_logits: shape {S.shape}  OK")

    # Planted pair should dominate row 0
    planted_logit = S[0, 3]
    assert planted_logit == S[0].max(), (
        f"S[0,3]={planted_logit:.4f} should be max of row 0 (max={S[0].max():.4f})"
    )
    print(f"  S[0,3] (planted) = {planted_logit:.4f}  (row-0 max)  OK")

    # --- compute_attention: rows sum to 1 ---
    A = cga.compute_attention(S)
    assert A.shape == (5, 8), f"Attention shape: expected (5,8), got {A.shape}"
    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), (
        f"Attention rows do not sum to 1: min={row_sums.min():.12f}, max={row_sums.max():.12f}"
    )
    print(
        f"  compute_attention: row sums in "
        f"[{row_sums.min():.10f}, {row_sums.max():.10f}]  OK"
    )

    # --- compute_output: shape (5, 16) ---
    O = cga.compute_output(A, E_j)
    assert O.shape == (5, d), f"Output shape: expected (5,{d}), got {O.shape}"
    print(f"  compute_output: shape {O.shape}  OK")

    # --- discover_two_stage: must find (0, 3) ---
    THETA = 0.15
    K     = 3
    hits_2s = cga.discover_two_stage(E_i, E_j, theta_logit=THETA, top_k=K)
    print(f"  discover_two_stage (theta={THETA}, K={K}): {len(hits_2s)} hit(s)")
    for h in hits_2s:
        print(f"    ({h[0]}, {h[1]})  logit={h[2]:.4f}  attn={h[3]:.4f}")
    found_2s = [(h[0], h[1]) for h in hits_2s]
    assert (0, 3) in found_2s, (
        f"Planted pair (0,3) not found in two_stage results: {found_2s}"
    )
    # Verify sorted descending by logit
    logits_2s = [h[2] for h in hits_2s]
    assert logits_2s == sorted(logits_2s, reverse=True), (
        "discover_two_stage results not sorted by logit descending"
    )
    print(f"  discover_two_stage: found (0,3), sorted descending  OK")

    # --- discover_logit_only ---
    hits_lo = cga.discover_logit_only(E_i, E_j, theta_logit=THETA)
    found_lo = [(h[0], h[1]) for h in hits_lo]
    assert (0, 3) in found_lo, (
        f"Planted pair (0,3) not in logit_only results: {found_lo}"
    )
    # All returned logits must exceed threshold
    assert all(h[2] > THETA for h in hits_lo), (
        "logit_only returned entry with logit <= theta"
    )
    logits_lo = [h[2] for h in hits_lo]
    assert logits_lo == sorted(logits_lo, reverse=True), (
        "discover_logit_only results not sorted descending"
    )
    print(f"  discover_logit_only: {len(hits_lo)} hit(s), found (0,3), sorted  OK")

    # --- discover_topk_only ---
    hits_tk = cga.discover_topk_only(E_i, E_j, top_k=K)
    expected_count = E_i.shape[0] * K   # 5 * 3 = 15
    assert len(hits_tk) == expected_count, (
        f"discover_topk_only: expected {expected_count} entries, got {len(hits_tk)}"
    )
    found_tk = [(h[0], h[1]) for h in hits_tk]
    assert (0, 3) in found_tk, (
        f"Planted pair (0,3) not in topk_only results: {found_tk}"
    )
    attn_tk = [h[2] for h in hits_tk]
    assert attn_tk == sorted(attn_tk, reverse=True), (
        "discover_topk_only results not sorted by attention descending"
    )
    print(
        f"  discover_topk_only: {len(hits_tk)} hit(s) (5 rows x {K} top-K), "
        f"found (0,3)  OK"
    )

    # --- cosine_baseline ---
    hits_cos = cga.cosine_baseline(E_i, E_j, threshold=0.9)
    found_cos = [(h[0], h[1]) for h in hits_cos]
    assert (0, 3) in found_cos, (
        f"Planted pair (0,3) not found by cosine_baseline at threshold=0.9: {found_cos}"
    )
    assert all(h[2] > 0.9 for h in hits_cos), (
        "cosine_baseline returned entry with similarity <= threshold"
    )
    cos_sims = [h[2] for h in hits_cos]
    assert cos_sims == sorted(cos_sims, reverse=True), (
        "cosine_baseline results not sorted descending"
    )
    print(
        f"  cosine_baseline (threshold=0.9): {len(hits_cos)} hit(s), "
        f"found (0,3)  OK"
    )

    print("\nAll checks passed")
