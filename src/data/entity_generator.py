"""
Entity embedding generator for Experiment 2 (Cross-Graph Discovery).

Generates synthetic entity representations for three graph domains:
  security         (200 entities)
  decision_history (300 entities)
  threat_intel     (200 entities)

Embedding layout (d=64):
  dims  0-5 : domain-specific semantic features  N(domain_mean, 0.30)
  dims  6-13: SHARED cross-graph signals (soft one-hot, 8 dims total)
                dims  6-9  = geographic cluster dims  (4 clusters, centers [0.25,0.50,0.75,1.00])
                dims 10-13 = temporal bucket dims     (4 buckets,  centers [0.20,0.40,0.60,0.80])
                Each entity is assigned to one cluster/bucket; that dim gets the center
                value + noise, the other three get 0 + noise.
  dims 14-63: background noise              N(0, 0.05)

Processing pipeline per domain:
  1. Sample raw (n_entities, 64) matrix as described above.
  2. Z-score normalise each column (feature) across all entities in the domain.
  3. L2 normalise each row (entity) -> unit-norm output.

inject_signals() creates detectable cross-graph correlations by replacing
the shared dimensions of selected target entities with a scaled copy of the
corresponding source entity's shared dimensions, then re-normalising.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 64
SEMANTIC_SLICE = slice(0, 6)    # domain-specific dims
SHARED_SLICE   = slice(6, 14)   # cross-graph signal dims (4 geo + 4 time)
NOISE_SLICE    = slice(14, 64)  # background noise dims

_GEO_CLUSTERS  = np.array([0.25, 0.50, 0.75, 1.00])  # dims 6-9  (1 dim per cluster)
_TIME_BUCKETS  = np.array([0.2,  0.4,  0.6,  0.8 ])  # dims 10-13 (1 dim per bucket)

# Domain-specific semantic means (dims 0-5) — clearly separated per domain
_DOMAIN_PROFILES: dict[str, dict] = {
    "security": {
        "semantic_means": [0.8, 0.1, 0.6, 0.2, 0.7, 0.3],
        "n_entities": 200,
    },
    "decision_history": {
        "semantic_means": [0.2, 0.7, 0.1, 0.8, 0.3, 0.6],
        "n_entities": 300,
    },
    "threat_intel": {
        "semantic_means": [0.5, 0.4, 0.7, 0.1, 0.5, 0.8],
        "n_entities": 200,
    },
}


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    entity_id : str
    domain    : str
    embedding : np.ndarray   # shape (EMBEDDING_DIM,), unit L2 norm


# ---------------------------------------------------------------------------
# EntityGenerator
# ---------------------------------------------------------------------------

class EntityGenerator:
    """
    Generates synthetic entity embeddings for each graph domain.

    Parameters
    ----------
    config : dict or path-like
        Configuration dict or path to a YAML file.  Reads the
        ``experiment_2`` sub-section if present; falls back to top-level
        keys or built-in defaults otherwise.
    """

    def __init__(self, config: dict | str | Path) -> None:
        if isinstance(config, (str, Path)):
            with open(config, "r") as fh:
                raw = yaml.safe_load(fh)
            cfg = raw.get("experiment_2", raw)
        else:
            cfg = config.get("experiment_2", config)

        self.embedding_dim    : int   = int(cfg.get("embedding_dim",    EMBEDDING_DIM))
        self.semantic_std     : float = float(cfg.get("semantic_std",    0.30))
        self.shared_noise_std : float = float(cfg.get("shared_noise_std", 0.05))
        self.background_std   : float = float(cfg.get("background_std",  0.05))

        # Resolve domain profiles: prefer config, fall back to built-in table
        raw_domains = cfg.get("domain_profiles", None)
        if raw_domains is not None:
            self.domain_profiles = raw_domains
        else:
            # Build from configs/default.yaml "domains" block if present,
            # merging with built-in semantic means.
            domains_block = cfg.get("domains", {})
            self.domain_profiles: dict[str, dict] = {}
            for name, built in _DOMAIN_PROFILES.items():
                entry = dict(built)
                if name in domains_block and "n_entities" in domains_block[name]:
                    entry["n_entities"] = int(domains_block[name]["n_entities"])
                self.domain_profiles[name] = entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_matrix(
        self,
        domain_name: str,
        n_entities: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return un-normalised embedding matrix, shape (n_entities, d)."""
        profile = self.domain_profiles[domain_name]
        means   = profile["semantic_means"]   # length-6 list
        d       = self.embedding_dim

        mat = np.zeros((n_entities, d), dtype=np.float64)

        # dims 0-5: domain-specific semantics
        for k, mu in enumerate(means):
            mat[:, k] = rng.normal(mu, self.semantic_std, size=n_entities)

        # dims 6-9: geographic cluster dims (soft one-hot, 4 clusters)
        # Entity assigned to cluster c gets center value in dim 6+c, ~0 in others.
        geo_idx = rng.integers(0, len(_GEO_CLUSTERS), size=n_entities)
        for c, center in enumerate(_GEO_CLUSTERS):
            signal = np.where(geo_idx == c, center, 0.0)
            mat[:, 6 + c] = signal + rng.normal(0.0, self.shared_noise_std, size=n_entities)

        # dims 10-13: temporal bucket dims (soft one-hot, 4 buckets)
        # Entity assigned to bucket b gets center value in dim 10+b, ~0 in others.
        time_idx = rng.integers(0, len(_TIME_BUCKETS), size=n_entities)
        for b, center in enumerate(_TIME_BUCKETS):
            signal = np.where(time_idx == b, center, 0.0)
            mat[:, 10 + b] = signal + rng.normal(0.0, self.shared_noise_std, size=n_entities)

        # dims 14-63: background noise
        mat[:, 14:] = rng.normal(0.0, self.background_std, size=(n_entities, d - 14))

        return mat

    @staticmethod
    def _zscore(mat: np.ndarray) -> np.ndarray:
        """Z-score normalise each column independently (in-place-safe copy)."""
        mu  = mat.mean(axis=0)
        std = mat.std(axis=0)
        std[std < 1e-12] = 1.0   # avoid divide-by-zero for constant features
        return (mat - mu) / std

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        """L2-normalise each row to unit norm (in-place-safe copy)."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return mat / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_domain(
        self,
        domain_name: str,
        n_entities: int,
        seed: int,
    ) -> list[Entity]:
        """
        Generate *n_entities* unit-norm embeddings for *domain_name*.

        Pipeline: raw sample -> z-score per dim -> L2 per entity.

        Parameters
        ----------
        domain_name : str
            One of "security", "decision_history", "threat_intel".
        n_entities : int
        seed : int

        Returns
        -------
        list[Entity]
        """
        rng = np.random.default_rng(seed)
        mat = self._raw_matrix(domain_name, n_entities, rng)
        mat = self._zscore(mat)
        mat = self._l2_normalize(mat)
        return [
            Entity(
                entity_id = f"{domain_name}_{i:04d}",
                domain    = domain_name,
                embedding = mat[i],
            )
            for i in range(n_entities)
        ]

    def generate_all(self, seed: int) -> dict[str, list[Entity]]:
        """
        Generate all three domains.

        Each domain uses an independent sub-seed derived from *seed* so
        results are stable regardless of domain ordering.

        Parameters
        ----------
        seed : int

        Returns
        -------
        dict[str, list[Entity]]  keyed by domain name
        """
        rng      = np.random.default_rng(seed)
        n_doms   = len(self.domain_profiles)
        sub_seeds = rng.integers(0, 2**31, size=n_doms)

        result: dict[str, list[Entity]] = {}
        for (name, profile), sub_seed in zip(
            self.domain_profiles.items(), sub_seeds
        ):
            n = profile["n_entities"]
            result[name] = self.generate_domain(name, n, int(sub_seed))
        return result


# ---------------------------------------------------------------------------
# inject_signals
# ---------------------------------------------------------------------------

def inject_signals(
    entities_i: list[Entity],
    entities_j: list[Entity],
    n_signals: int,
    signal_strength: float,
    seed: int,
) -> list[tuple[str, str]]:
    """
    Inject cross-graph correlations by aligning shared dims (6-13) of
    selected entity pairs.

    For each selected pair (entity_i, entity_j):
      1. entity_j.embedding[6:14] <- signal_strength * entity_i.embedding[6:14]
      2. entity_j is re-L2-normalised (embedding modified in place on the
         Entity object; the list items themselves are not replaced).

    Parameters
    ----------
    entities_i : list[Entity]
        Source domain (read-only).
    entities_j : list[Entity]
        Target domain (shared dims modified in place).
    n_signals : int
        Number of signal pairs to inject.
    signal_strength : float
        Scale factor applied to entity_i's shared dims before writing to
        entity_j.  Larger values -> stronger, more detectable correlation.
    seed : int

    Returns
    -------
    list[tuple[str, str]]
        (entity_i.entity_id, entity_j.entity_id) for each injected pair.
    """
    rng   = np.random.default_rng(seed)
    idx_i = rng.choice(len(entities_i), size=n_signals, replace=False)
    idx_j = rng.choice(len(entities_j), size=n_signals, replace=False)

    pairs: list[tuple[str, str]] = []
    for ii, jj in zip(idx_i, idx_j):
        ei = entities_i[ii]
        ej = entities_j[jj]

        new_emb      = ej.embedding.copy()
        new_emb[SHARED_SLICE] = signal_strength * ei.embedding[SHARED_SLICE]

        norm = np.linalg.norm(new_emb)
        if norm > 1e-12:
            new_emb /= norm
        ej.embedding = new_emb

        pairs.append((ei.entity_id, ej.entity_id))
    return pairs


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {
        "experiment_2": {
            "embedding_dim":    64,
            "semantic_std":     0.30,
            "shared_noise_std": 0.05,
            "background_std":   0.05,
            "domain_profiles":  _DOMAIN_PROFILES,
        }
    }

    gen     = EntityGenerator(cfg)
    domains = gen.generate_all(seed=42)

    # --- entity counts ---
    expected_counts = {"security": 200, "decision_history": 300, "threat_intel": 200}
    for name, entities in domains.items():
        n = len(entities)
        assert n == expected_counts[name], (
            f"{name}: expected {expected_counts[name]}, got {n}"
        )
        print(f"  {name}: {n} entities  OK")

    # --- unit norm ---
    for name, entities in domains.items():
        norms = np.array([np.linalg.norm(e.embedding) for e in entities])
        max_dev = float(np.abs(norms - 1.0).max())
        assert max_dev < 1e-9, (
            f"{name}: embeddings not unit-norm (max deviation {max_dev:.2e})"
        )
    print("  Unit-norm check: all domains OK")

    # --- embedding dimension ---
    for name, entities in domains.items():
        assert all(e.embedding.shape == (64,) for e in entities), (
            f"{name}: bad embedding shape"
        )
    print("  Shape check: all embeddings (64,)  OK")

    # --- entity_id format ---
    for name, entities in domains.items():
        assert entities[0].entity_id  == f"{name}_0000"
        assert entities[-1].entity_id == f"{name}_{len(entities)-1:04d}"
    print("  Entity ID format check: OK")

    # --- inject_signals: mean dot-product of injected pairs > random pairs ---
    sec    = domains["security"]
    threat = domains["threat_intel"]

    # Deep-copy threat so the random baseline is unmodified
    threat_clean = copy.deepcopy(threat)

    N_SIG   = 10
    SIG_STR = 5.0

    pairs = inject_signals(
        sec, threat,
        n_signals       = N_SIG,
        signal_strength = SIG_STR,
        seed            = 99,
    )

    # Dot products for injected pairs (lookup by id after in-place modification)
    id_to_ej = {e.entity_id: e for e in threat}
    id_to_ei = {e.entity_id: e for e in sec}
    injected_dots = [
        float(id_to_ei[id_i].embedding @ id_to_ej[id_j].embedding)
        for id_i, id_j in pairs
    ]
    mean_injected = float(np.mean(injected_dots))

    # Dot products for random (non-injected) pairs using clean copies
    injected_ids_i = {p[0] for p in pairs}
    injected_ids_j = {p[1] for p in pairs}
    pool_i = [e for e in sec          if e.entity_id not in injected_ids_i]
    pool_j = [e for e in threat_clean if e.entity_id not in injected_ids_j]

    rng_base = np.random.default_rng(0)
    ri = rng_base.choice(len(pool_i), size=N_SIG, replace=False)
    rj = rng_base.choice(len(pool_j), size=N_SIG, replace=False)
    random_dots = [
        float(pool_i[ii].embedding @ pool_j[jj].embedding)
        for ii, jj in zip(ri, rj)
    ]
    mean_random = float(np.mean(random_dots))

    print(f"  Signal injection (strength={SIG_STR}, n={N_SIG}):")
    print(f"    mean dot (injected) = {mean_injected:.4f}")
    print(f"    mean dot (random)   = {mean_random:.4f}")

    assert mean_injected > mean_random, (
        f"Injected ({mean_injected:.4f}) should exceed random ({mean_random:.4f})"
    )
    print("  Signal detection: injected > random  OK")

    print("\nAll checks passed")
