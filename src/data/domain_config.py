"""Domain configuration loader for multi-domain experiments.

The GAE engine is domain-agnostic. Each domain (SOC, S2P, etc.) has its own
config file under configs/. This loader provides a uniform interface.

Usage:
    from src.data.domain_config import load_domain_config
    config = load_domain_config("soc_product_v50")
    gen = CategoryAlertGenerator(**config["generator_kwargs"])
    scorer = ProfileScorer(config["mu"], config["actions"], tau=0.1)
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Any

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_domain_config(name: str) -> dict[str, Any]:
    """Load a domain config by name (without .yaml extension).

    Returns dict with:
      - categories: list[str]
      - actions: list[str]
      - factors: list[str]
      - profiles: dict[str, dict[str, list[float]]]  (category -> action -> factor means)
      - gt_distributions: dict[str, list[float]]  (category -> action probabilities)
      - mu: np.ndarray of shape (C, A, d)  (built from profiles)
      - generator_kwargs: dict ready to pass to CategoryAlertGenerator
      - metadata: dict with domain, version, notes
    """
    config_path = _CONFIGS_DIR / f"{name}.yaml"
    if not config_path.exists():
        available = [p.stem for p in _CONFIGS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Domain config '{name}' not found at {config_path}. "
            f"Available: {available}"
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    categories = raw["categories"]
    actions = raw["actions"]
    factors = raw["factors"]
    profiles = raw["action_conditional_profiles"]
    gt_distributions = raw["category_gt_distributions"]

    # Build mu tensor from profiles dict
    C, A, d = len(categories), len(actions), len(factors)
    mu = np.zeros((C, A, d), dtype=np.float64)
    for c_idx, cat in enumerate(categories):
        for a_idx, act in enumerate(actions):
            mu[c_idx, a_idx, :] = profiles[cat][act]

    result = {
        "categories": categories,
        "actions": actions,
        "factors": factors,
        "profiles": profiles,
        "gt_distributions": gt_distributions,
        "mu": mu,
        "C": C,
        "A": A,
        "d": d,
        "generator_kwargs": {
            "categories": categories,
            "actions": actions,
            "factors": factors,
            "action_conditional_profiles": profiles,
            "gt_distributions": gt_distributions,
        },
        "metadata": {
            "domain": raw.get("domain", "unknown"),
            "version": raw.get("version", "unknown"),
            "notes": raw.get("notes", ""),
        },
    }

    # Optional fields — pass through if present
    if "correlation_prior" in raw:
        result["correlation_prior"] = np.array(raw["correlation_prior"], dtype=np.float64)
    for key in ("tau", "eta", "eta_neg", "eta_override", "penalty_ratio"):
        if key in raw:
            result[key] = raw[key]

    return result


def list_domain_configs() -> list[str]:
    """List available domain config names."""
    return [p.stem for p in _CONFIGS_DIR.glob("*.yaml")]
