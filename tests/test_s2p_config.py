"""Tests for s2p_v03 domain config."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.domain_config import load_domain_config


def test_s2p_config_loads():
    config = load_domain_config("s2p_v03")
    assert config["mu"].shape == (5, 5, 8)
    assert len(config["categories"]) == 5
    assert len(config["actions"]) == 5


def test_s2p_correlation_prior():
    config = load_domain_config("s2p_v03")
    corr = config.get("correlation_prior")
    assert corr is not None
    assert corr.shape == (8, 8)
    assert np.allclose(corr, corr.T)          # symmetric
    assert np.allclose(np.diag(corr), 1.0)   # unit diagonal


def test_s2p_factors():
    config = load_domain_config("s2p_v03")
    assert len(config["factors"]) == 8
    assert "supplier_risk" in config["factors"]
    assert "device_trust" not in config["factors"]    # SOC factor, not S2P


def test_s2p_mu_in_range():
    config = load_domain_config("s2p_v03")
    mu = config["mu"]
    assert mu.min() >= 0.20 - 1e-9
    assert mu.max() <= 0.80 + 1e-9


def test_s2p_gt_distributions_sum_to_one():
    config = load_domain_config("s2p_v03")
    for cat, probs in config["gt_distributions"].items():
        assert abs(sum(probs) - 1.0) < 1e-6, f"{cat} gt_distribution does not sum to 1"


def test_s2p_hyperparams_present():
    config = load_domain_config("s2p_v03")
    assert config.get("tau") == 0.10
    assert config.get("eta") == 0.05
    assert config.get("eta_override") == 0.01
    assert config.get("penalty_ratio") == 5.0


def test_soc_config_unchanged():
    """Regression: existing SOC config still loads correctly after loader changes."""
    config = load_domain_config("soc_product_v50")
    assert config["mu"].shape == (6, 4, 6)
    assert len(config["categories"]) == 6
    assert len(config["actions"]) == 4
    assert len(config["factors"]) == 6
    # No correlation_prior in SOC config
    assert "correlation_prior" not in config
