"""
distribution_analysis.py — Statistical characterization of real IOC factor distributions.
"""
from __future__ import annotations

import numpy as np


def fit_gaussian(values: np.ndarray) -> tuple[float, float]:
    """Fit Gaussian to values. Returns (mean, std)."""
    if len(values) == 0:
        return 0.5, 0.2
    return float(np.mean(values)), float(np.std(values))


def compute_kl_divergence(
    real_values:     np.ndarray,
    synthetic_mean:  float,
    synthetic_std:   float,
    n_bins:          int = 30,
    epsilon:         float = 1e-10,
) -> float:
    """
    KL divergence KL(real || synthetic) using histogram binning over [0, 1].
    Both distributions discretized into n_bins bins; epsilon-smoothed to avoid log(0).
    """
    if len(real_values) == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Real distribution (empirical histogram)
    hist_real, _ = np.histogram(real_values, bins=bins, density=False)
    p = hist_real.astype(float) + epsilon
    p /= p.sum()

    # Synthetic (Gaussian, clipped to [0,1])
    from scipy.stats import norm as _norm  # type: ignore
    try:
        q_raw = _norm.pdf(bin_centers, loc=synthetic_mean, scale=max(synthetic_std, 1e-6))
    except Exception:
        q_raw = np.exp(-0.5 * ((bin_centers - synthetic_mean) / max(synthetic_std, 1e-6))**2)
        q_raw /= (max(synthetic_std, 1e-6) * np.sqrt(2 * np.pi))

    q = q_raw.astype(float) + epsilon
    q /= q.sum()

    kl = float(np.sum(p * np.log(p / q)))
    return max(0.0, kl)   # numerical floor at 0


def distribution_stats(values: np.ndarray) -> dict:
    """
    Compute mean, std, skewness, kurtosis.
    Uses scipy.stats if available, else manual computation.
    """
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "skewness": float("nan"), "kurtosis": float("nan"), "n": 0}

    mean = float(np.mean(values))
    std  = float(np.std(values))

    try:
        from scipy import stats as _sp
        skew = float(_sp.skew(values))
        kurt = float(_sp.kurtosis(values))   # excess kurtosis (normal = 0)
    except Exception:
        # Manual: Fisher's moment coefficient
        n = len(values)
        if std > 0:
            z  = (values - mean) / std
            skew = float(np.mean(z**3))
            kurt = float(np.mean(z**4) - 3.0)
        else:
            skew = 0.0
            kurt = 0.0

    return {
        "mean":     mean,
        "std":      std,
        "skewness": skew,
        "kurtosis": kurt,
        "n":        len(values),
    }
