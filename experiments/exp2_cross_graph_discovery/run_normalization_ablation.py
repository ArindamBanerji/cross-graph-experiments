"""
EXP-2 Normalization Ablation.

Compares 4 embedding normalization pipelines for cross-graph discovery.
All other experiment parameters (seeds, signal_strength, theta, K, domain
pairs) are identical to run.py.

The ONLY variable is what normalization is applied to the embedding matrices
AFTER raw generation and BEFORE CrossGraphAttention.

Pipelines
---------
  raw       -- No normalization.  Raw embeddings as sampled.
  zscore    -- Z-score per dimension only (no L2).
  l2        -- L2 unit-norm per entity only (no z-score).
  zscore_l2 -- Z-score per dim + L2 per entity.  Current pipeline in run.py.

Implementation note
-------------------
entity_generator.generate_domain() always applies zscore+L2 internally.
To get raw matrices we replicate generate_all()'s sub-seed derivation and
call gen._raw_matrix() directly, bypassing the normalisation steps.
Signal injection then happens after applying the chosen pipeline, matching
the original run.py order (inject AFTER normalise).

Outputs
-------
experiments/exp2_cross_graph_discovery/normalization_ablation_results.csv
experiments/exp2_cross_graph_discovery/normalization_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.data.entity_generator import EntityGenerator
from src.models.cross_attention import CrossGraphAttention


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINES = ["raw", "zscore", "l2", "zscore_l2"]

# Best config from best_configs.json (theta=0.05 wins for all pairs)
THETA = 0.05
KS    = [1, 2]

# Domain pair spec: (source_domain, target_domain, label, n_signals_key)
DOMAIN_PAIRS_SPEC = [
    ("security",         "threat_intel",      "secxthr", "security_threat"),
    ("decision_history", "threat_intel",       "decxthr", "decision_threat"),
    ("security",         "decision_history",   "secxdec", "security_decision"),
]

# Shared dims slice (dims 6-13 in the 64-dim embeddings)
_SHARED = slice(6, 14)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["experiment_2"]


# ---------------------------------------------------------------------------
# Normalization pipelines
# ---------------------------------------------------------------------------

def apply_normalization(E: np.ndarray, pipeline: str) -> np.ndarray:
    """Apply normalization pipeline to embedding matrix E (m x d)."""
    if pipeline == "raw":
        return E.copy()
    elif pipeline == "zscore":
        means = E.mean(axis=0)
        stds  = E.std(axis=0)
        stds[stds == 0] = 1.0
        return (E - means) / stds
    elif pipeline == "l2":
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return E / norms
    elif pipeline == "zscore_l2":
        means = E.mean(axis=0)
        stds  = E.std(axis=0)
        stds[stds == 0] = 1.0
        E_z   = (E - means) / stds
        norms = np.linalg.norm(E_z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return E_z / norms
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}")


# ---------------------------------------------------------------------------
# Raw matrix extraction (bypasses generate_domain normalisation)
# ---------------------------------------------------------------------------

def _get_raw_matrices(gen: EntityGenerator, seed: int) -> dict[str, np.ndarray]:
    """
    Replicate generate_all()'s sub-seed derivation but call _raw_matrix()
    instead of generate_domain(), returning pre-normalisation matrices.

    This guarantees the SAME random state as the original experiment so that
    'zscore_l2' in this ablation reproduces run.py results exactly.
    """
    rng       = np.random.default_rng(seed)
    n_doms    = len(gen.domain_profiles)
    sub_seeds = rng.integers(0, 2**31, size=n_doms)

    raw_mats: dict[str, np.ndarray] = {}
    for (name, profile), sub_seed in zip(gen.domain_profiles.items(), sub_seeds):
        n     = profile["n_entities"]
        rng_d = np.random.default_rng(int(sub_seed))
        raw_mats[name] = gen._raw_matrix(name, n, rng_d)

    return raw_mats


# ---------------------------------------------------------------------------
# Matrix-level signal injection
# ---------------------------------------------------------------------------

def _inject_signals(
    mat_i: np.ndarray,
    mat_j: np.ndarray,
    n_signals: int,
    signal_strength: float,
    seed: int,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """
    Matrix-level equivalent of entity_generator.inject_signals().

    Uses the same seed -> same idx_i, idx_j as the original, so that the
    'zscore_l2' pipeline produces identical GT pairs and signal content.

    After injecting shared dims, always re-L2-normalises the modified row
    (matching original inject_signals behaviour).

    Returns
    -------
    mat_j_mod  : modified copy of mat_j
    gt_idx_set : set of (i_idx, j_idx) ground-truth pair indices
    """
    rng   = np.random.default_rng(seed)
    idx_i = rng.choice(mat_i.shape[0], size=n_signals, replace=False)
    idx_j = rng.choice(mat_j.shape[0], size=n_signals, replace=False)

    mat_j_mod = mat_j.copy()
    for ii, jj in zip(idx_i, idx_j):
        mat_j_mod[jj, _SHARED] = signal_strength * mat_i[ii, _SHARED]
        norm = np.linalg.norm(mat_j_mod[jj])
        if norm > 1e-12:
            mat_j_mod[jj] /= norm

    return mat_j_mod, set(zip(idx_i.tolist(), idx_j.tolist()))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(disc: set, gt: set, n_gt: int) -> dict:
    tp   = len(disc & gt)
    fp   = len(disc) - tp
    fn   = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / n_gt      if n_gt      > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return dict(n_discovered=len(disc),
                true_pos=tp, false_pos=fp, false_neg=fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4))


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_ablation() -> None:
    cfg      = _load_config()
    seeds    = cfg["seeds"]
    sig_str  = float(cfg["signal_strength"])
    n_sig_map = cfg["n_signals"]

    gen = EntityGenerator(cfg)
    cga = CrossGraphAttention()

    exp_dir = ROOT / "experiments" / "exp2_cross_graph_discovery"
    exp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # random_f1_data[pipeline][pair] -> list of per-seed random F1 values
    random_f1_data: dict[str, dict[str, list[float]]] = {
        p: {lbl: [] for _, _, lbl, _ in DOMAIN_PAIRS_SPEC}
        for p in PIPELINES
    }

    for pipeline in PIPELINES:
        print(f"\nPipeline: {pipeline}")

        for seed in seeds:
            print(f"  seed={seed} ...", end="", flush=True)

            raw_mats = _get_raw_matrices(gen, seed)

            # Apply this pipeline's normalisation to all domain matrices
            norm_mats = {
                name: apply_normalization(raw, pipeline)
                for name, raw in raw_mats.items()
            }

            for dom_i, dom_j, pair_lbl, n_sig_key in DOMAIN_PAIRS_SPEC:
                n_sig = n_sig_map[n_sig_key]
                m_i   = norm_mats[dom_i].shape[0]
                m_j   = norm_mats[dom_j].shape[0]

                # Inject signals AFTER normalisation (same order as run.py)
                mat_j_inj, gt_set = _inject_signals(
                    mat_i=norm_mats[dom_i],
                    mat_j=norm_mats[dom_j],
                    n_signals=n_sig,
                    signal_strength=sig_str,
                    seed=seed,       # identical seed -> identical idx_i, idx_j
                )
                mat_i = norm_mats[dom_i]
                n_gt  = len(gt_set)

                disc_counts: list[int] = []
                for k in KS:
                    hits = cga.discover_two_stage(mat_i, mat_j_inj, THETA, k)
                    disc = {(h[0], h[1]) for h in hits}
                    disc_counts.append(len(disc))
                    m = _metrics(disc, gt_set, n_gt)
                    rows.append({
                        "pipeline":    pipeline,
                        "seed":        seed,
                        "domain_pair": pair_lbl,
                        "top_k":       k,
                        **m,
                    })

                # Random baseline (analytical, same formula as run.py)
                n_disc  = int(np.median(disc_counts)) if disc_counts else 1
                exp_tp  = n_disc * n_gt / (m_i * m_j)
                prec_r  = exp_tp / n_disc if n_disc > 0 else 0.0
                rec_r   = exp_tp / n_gt   if n_gt   > 0 else 0.0
                f1_r    = (2 * prec_r * rec_r / (prec_r + rec_r)
                           if (prec_r + rec_r) > 0 else 0.0)
                random_f1_data[pipeline][pair_lbl].append(f1_r)

            print(" done")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path   = exp_dir / "normalization_ablation_results.csv"
    fieldnames = ["pipeline", "seed", "domain_pair", "top_k", "n_discovered",
                  "true_pos", "false_pos", "false_neg", "precision", "recall", "f1"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path} ({len(rows)} rows)")

    # -----------------------------------------------------------------------
    # Build summary
    # -----------------------------------------------------------------------
    pairs = ["secxthr", "decxthr", "secxdec"]
    summary: list[dict] = []

    for pipeline in PIPELINES:
        p_rows = [r for r in rows if r["pipeline"] == pipeline]

        # Overall mean F1 and random F1 across all (seed, pair, K)
        mean_f1 = float(np.mean([r["f1"] for r in p_rows])) if p_rows else 0.0

        all_rand_f1s = [f for lbl in pairs for f in random_f1_data[pipeline][lbl]]
        mean_rand_f1 = float(np.mean(all_rand_f1s)) if all_rand_f1s else 0.0
        ratio = mean_f1 / mean_rand_f1 if mean_rand_f1 > 0 else 0.0

        per_pair: dict[str, dict] = {}
        for lbl in pairs:
            pp_rows = [r for r in p_rows if r["domain_pair"] == lbl]
            pp_f1   = float(np.mean([r["f1"] for r in pp_rows])) if pp_rows else 0.0
            rand_f1s_pair = random_f1_data[pipeline][lbl]
            mean_rand_pair = float(np.mean(rand_f1s_pair)) if rand_f1s_pair else 0.0
            pair_ratio = pp_f1 / mean_rand_pair if mean_rand_pair > 0 else 0.0
            per_pair[lbl] = {
                "mean_f1": round(pp_f1, 4),
                "ratio":   round(pair_ratio, 1),
            }

        summary.append({
            "pipeline":          pipeline,
            "mean_f1":           round(mean_f1,      4),
            "mean_random_f1":    round(mean_rand_f1, 6),
            "ratio_above_random": round(ratio,        1),
            "per_pair":          per_pair,
        })

    json_path = exp_dir / "normalization_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXP-2: NORMALIZATION ABLATION  (two_stage, theta=0.05, K=1 and K=2)")
    print("=" * 72)

    print(f"\n  {'Pipeline':<12s} {'Mean F1':>8s} {'x random':>9s} | "
          f"{'secxthr':>8s} {'decxthr':>8s} {'secxdec':>8s}")
    print(f"  {'-'*60}")
    for entry in summary:
        pp_f1s = [entry["per_pair"][p]["mean_f1"] for p in pairs]
        print(f"  {entry['pipeline']:<12s} "
              f"{entry['mean_f1']:>8.4f} "
              f"{entry['ratio_above_random']:>8.1f}x | "
              f"{'  '.join(f'{v:.4f}' for v in pp_f1s)}")

    # Key comparisons
    raw_e = next(e for e in summary if e["pipeline"] == "raw")
    zl2_e = next(e for e in summary if e["pipeline"] == "zscore_l2")
    z_e   = next(e for e in summary if e["pipeline"] == "zscore")
    l2_e  = next(e for e in summary if e["pipeline"] == "l2")

    delta_zl2_raw = (zl2_e["mean_f1"] - raw_e["mean_f1"]) * 100
    delta_l2_raw  = (l2_e["mean_f1"]  - raw_e["mean_f1"]) * 100
    delta_z_raw   = (z_e["mean_f1"]   - raw_e["mean_f1"]) * 100

    print(f"\n  Key deltas vs raw:")
    print(f"    zscore_l2 - raw  : {delta_zl2_raw:+.1f}pp  "
          f"({zl2_e['mean_f1']:.4f} vs {raw_e['mean_f1']:.4f})")
    print(f"    l2 - raw         : {delta_l2_raw:+.1f}pp  "
          f"({l2_e['mean_f1']:.4f} vs {raw_e['mean_f1']:.4f})")
    print(f"    zscore - raw     : {delta_z_raw:+.1f}pp  "
          f"({z_e['mean_f1']:.4f} vs {raw_e['mean_f1']:.4f})")

    # Which step matters more: zscore or L2?
    # Isolate L2 contribution: l2 vs raw
    # Isolate zscore contribution: zscore vs raw
    print(f"\n  Component attribution:")
    print(f"    L2 alone       : {delta_l2_raw:+.1f}pp above raw")
    print(f"    zscore alone   : {delta_z_raw:+.1f}pp above raw")
    print(f"    zscore + L2    : {delta_zl2_raw:+.1f}pp above raw")
    interaction = delta_zl2_raw - delta_l2_raw - delta_z_raw
    print(f"    Interaction    : {interaction:+.1f}pp  "
          f"({'synergistic' if interaction > 0 else 'redundant'})")

    # Paper correction block
    print(f"\n  PAPER CORRECTION (Exp 2 normalization comparison table):")
    print(f"    Paper claimed: F1=0.293 with zscore+L2 normalisation")
    print(f"    Paper claimed: 23x above random WITHOUT normalisation")
    print(f"    Actual data (discovery_results.csv): max F1 = 0.1724")
    print(f"    Unnormalised comparison: NEVER computed before this ablation")
    print()
    print(f"    Correct values (this ablation, theta=0.05, K=1+2 average):")
    print(f"    {'Pipeline':<12s}  F1      x random")
    print(f"    {'-'*36}")
    for entry in summary:
        tag = "  <-- current pipeline" if entry["pipeline"] == "zscore_l2" else ""
        print(f"    {entry['pipeline']:<12s}  "
              f"{entry['mean_f1']:.4f}  "
              f"{entry['ratio_above_random']:.0f}x{tag}")

    print("=" * 72)


def run_single_pipeline(pipeline: str) -> None:
    """Run ablation for a single pipeline and append results to CSV."""
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}. Choose from {PIPELINES}")

    cfg      = _load_config()
    seeds    = cfg["seeds"]
    sig_str  = float(cfg["signal_strength"])
    n_sig_map = cfg["n_signals"]

    gen = EntityGenerator(cfg)
    cga = CrossGraphAttention()

    exp_dir = ROOT / "experiments" / "exp2_cross_graph_discovery"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "normalization_ablation_results.csv"

    fieldnames = ["pipeline", "seed", "domain_pair", "top_k", "n_discovered",
                  "true_pos", "false_pos", "false_neg", "precision", "recall", "f1"]

    rows: list[dict] = []
    random_f1_data: dict[str, list[float]] = {
        lbl: [] for _, _, lbl, _ in DOMAIN_PAIRS_SPEC
    }

    print(f"\nPipeline: {pipeline}")
    for seed in seeds:
        print(f"  seed={seed} ...", end="", flush=True)

        raw_mats  = _get_raw_matrices(gen, seed)
        norm_mats = {
            name: apply_normalization(raw, pipeline)
            for name, raw in raw_mats.items()
        }

        for dom_i, dom_j, pair_lbl, n_sig_key in DOMAIN_PAIRS_SPEC:
            n_sig = n_sig_map[n_sig_key]
            m_i   = norm_mats[dom_i].shape[0]
            m_j   = norm_mats[dom_j].shape[0]

            mat_j_inj, gt_set = _inject_signals(
                mat_i=norm_mats[dom_i],
                mat_j=norm_mats[dom_j],
                n_signals=n_sig,
                signal_strength=sig_str,
                seed=seed,
            )
            n_gt = len(gt_set)

            disc_counts: list[int] = []
            for k in KS:
                hits = cga.discover_two_stage(norm_mats[dom_i], mat_j_inj, THETA, k)
                disc = {(h[0], h[1]) for h in hits}
                disc_counts.append(len(disc))
                m = _metrics(disc, gt_set, n_gt)
                rows.append({
                    "pipeline":    pipeline,
                    "seed":        seed,
                    "domain_pair": pair_lbl,
                    "top_k":       k,
                    **m,
                })

            n_disc  = int(np.median(disc_counts)) if disc_counts else 1
            exp_tp  = n_disc * n_gt / (m_i * m_j)
            prec_r  = exp_tp / n_disc if n_disc > 0 else 0.0
            rec_r   = exp_tp / n_gt   if n_gt   > 0 else 0.0
            f1_r    = (2 * prec_r * rec_r / (prec_r + rec_r)
                       if (prec_r + rec_r) > 0 else 0.0)
            random_f1_data[pair_lbl].append(f1_r)

        print(" done")

    # Append or create CSV
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {csv_path}")

    # Per-pipeline summary
    mean_f1 = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
    all_rand = [f for lbl_list in random_f1_data.values() for f in lbl_list]
    mean_rand = float(np.mean(all_rand)) if all_rand else 0.0
    ratio = mean_f1 / mean_rand if mean_rand > 0 else 0.0
    print(f"\n  {pipeline}: mean_f1={mean_f1:.4f}  x_random={ratio:.1f}x")


def _print_summary_from_csv() -> None:
    """Load full CSV and print summary table (used after all pipelines done)."""
    exp_dir  = ROOT / "experiments" / "exp2_cross_graph_discovery"
    csv_path = exp_dir / "normalization_ablation_results.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run all pipelines first.")
        return

    import csv as csv_mod
    all_rows: list[dict] = []
    with open(csv_path, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            row["f1"]  = float(row["f1"])
            row["top_k"] = int(row["top_k"])
            all_rows.append(row)

    found_pipelines = list(dict.fromkeys(r["pipeline"] for r in all_rows))
    pairs = ["secxthr", "decxthr", "secxdec"]

    summary: list[dict] = []
    for pipeline in PIPELINES:
        p_rows = [r for r in all_rows if r["pipeline"] == pipeline]
        if not p_rows:
            continue
        mean_f1 = float(np.mean([r["f1"] for r in p_rows]))

        # Recompute random baselines from CSV metadata is unavailable — use
        # per-pair entity counts from config to compute analytically.
        cfg = _load_config()
        n_sig_map = cfg["n_signals"]
        gen = EntityGenerator(cfg)
        pairs_spec_map = {lbl: (dom_i, dom_j, n_k)
                          for dom_i, dom_j, lbl, n_k in DOMAIN_PAIRS_SPEC}
        rand_f1s: list[float] = []
        for lbl in pairs:
            dom_i, dom_j, n_k = pairs_spec_map[lbl]
            m_i = gen.domain_profiles[dom_i]["n_entities"]
            m_j = gen.domain_profiles[dom_j]["n_entities"]
            n_gt = n_sig_map[n_k]
            # median n_disc ≈ n_gt for two_stage at theta=0.05 (rough)
            # Use mean n_discovered from actual results
            pp = [r for r in p_rows if r["domain_pair"] == lbl]
            n_disc = int(np.median([int(r.get("n_discovered", n_gt)) for r in pp])) if pp else n_gt
            exp_tp = n_disc * n_gt / (m_i * m_j)
            pr = exp_tp / n_disc if n_disc > 0 else 0.0
            re = exp_tp / n_gt   if n_gt   > 0 else 0.0
            f1r = 2*pr*re/(pr+re) if (pr+re)>0 else 0.0
            rand_f1s.append(f1r)

        mean_rand = float(np.mean(rand_f1s)) if rand_f1s else 0.0
        ratio = mean_f1 / mean_rand if mean_rand > 0 else 0.0

        per_pair: dict[str, dict] = {}
        for lbl in pairs:
            pp_rows = [r for r in p_rows if r["domain_pair"] == lbl]
            pp_f1   = float(np.mean([r["f1"] for r in pp_rows])) if pp_rows else 0.0
            per_pair[lbl] = {"mean_f1": round(pp_f1, 4)}

        summary.append({
            "pipeline": pipeline,
            "mean_f1": round(mean_f1, 4),
            "mean_random_f1": round(mean_rand, 6),
            "ratio_above_random": round(ratio, 1),
            "per_pair": per_pair,
        })

    json_path = exp_dir / "normalization_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    print("\n" + "=" * 72)
    print("EXP-2: NORMALIZATION ABLATION  (two_stage, theta=0.05, K=1 and K=2)")
    print("=" * 72)
    print(f"\n  {'Pipeline':<12s} {'Mean F1':>8s} {'x random':>9s} | "
          f"{'secxthr':>8s} {'decxthr':>8s} {'secxdec':>8s}")
    print(f"  {'-'*60}")
    for entry in summary:
        pp_f1s = [entry["per_pair"][p]["mean_f1"] for p in pairs]
        print(f"  {entry['pipeline']:<12s} "
              f"{entry['mean_f1']:>8.4f} "
              f"{entry['ratio_above_random']:>8.1f}x | "
              f"{'  '.join(f'{v:.4f}' for v in pp_f1s)}")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EXP-2 normalization ablation")
    parser.add_argument(
        "--pipeline",
        choices=PIPELINES + ["summary"],
        default=None,
        help="Run one pipeline only (appends to CSV). Use 'summary' to print table.",
    )
    args = parser.parse_args()

    if args.pipeline == "summary":
        _print_summary_from_csv()
    elif args.pipeline is not None:
        run_single_pipeline(args.pipeline)
    else:
        run_ablation()
