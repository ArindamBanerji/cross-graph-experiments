"""
VALIDATION-1B: Norm tracking through enrichment sweeps.

Measures embedding norm growth under repeated application of Eq. 13:
  E_i_enriched = E_i + Σ_{j≠i} CrossAttention(G_i, G_j)

Configuration: n=6 domains, 200 entities/domain, 5 sweeps, 10 seeds.
Bidirectional enrichment per pair — for each (i<j):
  E_i += softmax(E_i_pre @ E_j_pre.T / √d) @ E_j_pre
  E_j += softmax(E_j_pre @ E_i_pre.T / √d) @ E_i_pre
  (using pre-update snapshots so neither direction affects the other)
Updates from earlier pairs WITHIN a sweep propagate to later pairs.

Outputs
-------
experiments/exp3_multidomain_scaling/norm_tracking.csv
experiments/exp3_multidomain_scaling/norm_tracking_summary.json
"""
from __future__ import annotations

import copy, csv, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.entity_generator import EntityGenerator, inject_signals, _DOMAIN_PROFILES
from src.models.cross_attention import CrossGraphAttention

EXP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared constants (identical to run.py for n=6)
# ---------------------------------------------------------------------------
N_DOMAINS = 6
N_SWEEPS  = 5
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

ALL_DOMAIN_NAMES = [
    "security", "decision_history", "threat_intel",
    "network_flow", "asset_inventory", "user_behavior",
]

_EXTRA_PROFILES = {
    "network_flow":    {"semantic_means": [0.3, 0.6, 0.2, 0.7, 0.4, 0.5], "n_entities": 200},
    "asset_inventory": {"semantic_means": [0.6, 0.3, 0.8, 0.4, 0.2, 0.7], "n_entities": 200},
    "user_behavior":   {"semantic_means": [0.1, 0.8, 0.4, 0.6, 0.9, 0.2], "n_entities": 200},
}

ALL_PROFILES = {**_DOMAIN_PROFILES, **_EXTRA_PROFILES}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config() -> tuple[dict, float]:
    with open(ROOT / "configs" / "default.yaml") as fh:
        raw = yaml.safe_load(fh)
    return raw["experiment_3"], float(raw["experiment_2"]["signal_strength"])


# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------
def _norm_stats(E: np.ndarray) -> dict:
    norms = np.linalg.norm(E, axis=1)
    return dict(
        mean_norm=float(norms.mean()),
        std_norm=float(norms.std()),
        max_norm=float(norms.max()),
        min_norm=float(norms.min()),
    )


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def _softmax_rows(X: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = X - X.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _enrich_pair(E: dict[str, np.ndarray], di: str, dj: str, d: int) -> None:
    """
    One bidirectional enrichment for a single pair (di, dj).
    Uses pre-update snapshots for both directions.
    """
    Ei_pre = E[di].copy()
    Ej_pre = E[dj].copy()
    scale  = float(np.sqrt(d))

    # i attends to j
    S_ij = Ei_pre @ Ej_pre.T / scale
    A_ij = _softmax_rows(S_ij)
    E[di] += A_ij @ Ej_pre

    # j attends to i (pre-update E_i)
    S_ji = Ej_pre @ Ei_pre.T / scale
    A_ji = _softmax_rows(S_ji)
    E[dj] += A_ji @ Ei_pre


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_norm_tracking() -> None:
    t0 = time.time()
    cfg, sig_str = _load_config()
    n_entities = cfg["entities_per_domain"]   # 200
    n_sig      = cfg["signals_per_pair"]       # 5

    gen = EntityGenerator({"experiment_2": {"domain_profiles": ALL_PROFILES}})

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    domain_pairs = list(combinations(ALL_DOMAIN_NAMES[:N_DOMAINS], 2))  # 15 pairs

    for seed in SEEDS:
        print(f"Seed {seed} ...", flush=True)
        rng       = np.random.default_rng(seed)
        dom_seeds = rng.integers(0, 2**31, size=N_DOMAINS)

        # 1. Generate all 6 domains
        base = {
            name: gen.generate_domain(name, n_entities, int(dom_seeds[k]))
            for k, name in enumerate(ALL_DOMAIN_NAMES[:N_DOMAINS])
        }

        # 2. Inject signals (same seed formula as run.py)
        for pi, (di, dj) in enumerate(domain_pairs):
            inject_signals(base[di], base[dj], n_sig, sig_str, seed * 100 + pi)

        # 3. Convert to mutable numpy matrices
        E: dict[str, np.ndarray] = {
            name: np.array([e.embedding for e in base[name]], dtype=np.float64)
            for name in ALL_DOMAIN_NAMES[:N_DOMAINS]
        }

        d = next(iter(E.values())).shape[1]  # 64

        # 4. Record sweep 0 (initial norms after generation + injection)
        for name in ALL_DOMAIN_NAMES[:N_DOMAINS]:
            rows.append({
                "seed": seed, "sweep": 0, "domain": name,
                "n_entities": E[name].shape[0],
                **_norm_stats(E[name]),
            })

        # 5. Enrichment sweeps 1..N_SWEEPS
        for sweep in range(1, N_SWEEPS + 1):
            for di, dj in domain_pairs:
                _enrich_pair(E, di, dj, d)

            for name in ALL_DOMAIN_NAMES[:N_DOMAINS]:
                rows.append({
                    "seed": seed, "sweep": sweep, "domain": name,
                    "n_entities": E[name].shape[0],
                    **_norm_stats(E[name]),
                })

        # Quick progress printout
        initial_max = max(
            r["max_norm"] for r in rows
            if r["seed"] == seed and r["sweep"] == 0
        )
        final_max = max(
            r["max_norm"] for r in rows
            if r["seed"] == seed and r["sweep"] == N_SWEEPS
        )
        print(f"  norm max: sweep0={initial_max:.3f}  "
              f"sweep{N_SWEEPS}={final_max:.3f}  "
              f"growth={final_max/initial_max:.2f}x", flush=True)

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path   = EXP_DIR / "norm_tracking.csv"
    fieldnames = ["seed", "sweep", "domain", "mean_norm", "std_norm",
                  "max_norm", "min_norm", "n_entities"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")

    # -----------------------------------------------------------------------
    # Build summary: aggregate across all seeds and domains per sweep
    # -----------------------------------------------------------------------
    sweep_vals: dict[int, dict[str, list[float]]] = {
        s: {"mean_norm": [], "max_norm": []}
        for s in range(N_SWEEPS + 1)
    }
    for r in rows:
        sweep_vals[r["sweep"]]["mean_norm"].append(r["mean_norm"])
        sweep_vals[r["sweep"]]["max_norm"].append(r["max_norm"])

    sweeps        = list(range(N_SWEEPS + 1))
    overall_mean  = [float(np.mean(sweep_vals[s]["mean_norm"])) for s in sweeps]
    overall_max   = [float(np.max(sweep_vals[s]["max_norm"]))   for s in sweeps]

    base_mean = overall_mean[0]
    base_max  = overall_max[0]

    growth_rate_mean = [None] + [round(overall_mean[s] / base_mean, 4) for s in sweeps[1:]]
    growth_rate_max  = [None] + [round(overall_max[s]  / base_max,  4) for s in sweeps[1:]]

    final_growth_max  = overall_max[N_SWEEPS]  / base_max
    final_growth_mean = overall_mean[N_SWEEPS] / base_mean

    if final_growth_max > 2.0:
        needs_norm: str | bool = True
    elif final_growth_max <= 1.5:
        needs_norm = False
    else:
        needs_norm = "recommended"

    # Per-domain breakdown at final sweep
    per_domain_final: dict[str, dict] = {}
    for name in ALL_DOMAIN_NAMES[:N_DOMAINS]:
        s0_rows = [r for r in rows if r["domain"] == name and r["sweep"] == 0]
        sN_rows = [r for r in rows if r["domain"] == name and r["sweep"] == N_SWEEPS]
        init_max  = float(np.max([r["max_norm"]  for r in s0_rows]))
        final_max_d = float(np.max([r["max_norm"] for r in sN_rows]))
        init_mean   = float(np.mean([r["mean_norm"] for r in s0_rows]))
        final_mean_d = float(np.mean([r["mean_norm"] for r in sN_rows]))
        per_domain_final[name] = {
            "initial_mean_norm":  round(init_mean,    4),
            "final_mean_norm":    round(final_mean_d, 4),
            "initial_max_norm":   round(init_max,     4),
            "final_max_norm":     round(final_max_d,  4),
            "growth_max":         round(final_max_d / init_max, 4),
        }

    summary = {
        "config": {
            "n_domains":          N_DOMAINS,
            "entities_per_domain": n_entities,
            "n_sweeps":           N_SWEEPS,
            "n_seeds":            len(SEEDS),
            "n_pairs_per_sweep":  len(domain_pairs),
        },
        "sweeps":              sweeps,
        "overall_mean_norm":   [round(v, 4) for v in overall_mean],
        "overall_max_norm":    [round(v, 4) for v in overall_max],
        "growth_rate_per_sweep_mean": growth_rate_mean,
        "growth_rate_per_sweep_max":  growth_rate_max,
        "final_growth_factor_mean":   round(final_growth_mean, 4),
        "final_growth_factor_max":    round(final_growth_max,  4),
        "needs_normalization":        needs_norm,
        "per_domain_final_sweep":     per_domain_final,
    }

    json_path = EXP_DIR / "norm_tracking_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("VALIDATION-1B: Norm Tracking  (Eq. 13 residual enrichment)")
    print("=" * 68)
    print(f"  Config: {N_DOMAINS} domains × {n_entities} entities × "
          f"{N_SWEEPS} sweeps × {len(SEEDS)} seeds")
    print(f"  {len(domain_pairs)} pairs per sweep, bidirectional")

    print(f"\n  {'Sweep':>6}  {'mean_norm':>10}  {'max_norm':>10}  "
          f"{'growth(mean)':>13}  {'growth(max)':>12}")
    print(f"  {'-'*57}")
    for s in sweeps:
        gmean = f"{growth_rate_mean[s]:.3f}x" if growth_rate_mean[s] is not None else "  —"
        gmax  = f"{growth_rate_max[s]:.3f}x"  if growth_rate_max[s]  is not None else "  —"
        print(f"  {s:>6}  {overall_mean[s]:>10.4f}  {overall_max[s]:>10.4f}  "
              f"{gmean:>13}  {gmax:>12}")

    print(f"\n  Growth after {N_SWEEPS} sweeps:")
    print(f"    Mean norm: {base_mean:.4f} -> {overall_mean[N_SWEEPS]:.4f}  "
          f"({final_growth_mean:.2f}x)")
    print(f"    Max  norm: {base_max:.4f} -> {overall_max[N_SWEEPS]:.4f}  "
          f"({final_growth_max:.2f}x)")

    print(f"\n  Per-domain growth (sweep 0 -> sweep {N_SWEEPS}):")
    print(f"  {'Domain':<20}  {'init_mean':>9}  {'final_mean':>10}  "
          f"{'init_max':>8}  {'final_max':>9}  {'growth_max':>10}")
    print(f"  {'-'*74}")
    for name, d_stats in per_domain_final.items():
        print(f"  {name:<20}  "
              f"{d_stats['initial_mean_norm']:>9.4f}  "
              f"{d_stats['final_mean_norm']:>10.4f}  "
              f"{d_stats['initial_max_norm']:>8.4f}  "
              f"{d_stats['final_max_norm']:>9.4f}  "
              f"{d_stats['growth_max']:>9.3f}x")

    # Verdict
    print(f"\n  VERDICT: needs_normalization = {needs_norm}")
    if needs_norm is True:
        print(f"    Max norm grew {final_growth_max:.1f}x (> 2.0x threshold)")
        print(f"    CONFIRMS reviewer concern: LayerNorm is required for Eq. 13")
    elif needs_norm == "recommended":
        print(f"    Max norm grew {final_growth_max:.1f}x (1.5x–2.0x range)")
        print(f"    Normalization RECOMMENDED but not strictly required")
    else:
        print(f"    Max norm grew only {final_growth_max:.2f}x (<= 1.5x)")
        print(f"    Norms stable — reviewer concern not supported by data")

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    run_norm_tracking()
