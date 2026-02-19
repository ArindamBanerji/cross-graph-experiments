"""Experiment 2: Cross-Graph Discovery — runner."""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.entity_generator import EntityGenerator, inject_signals
from src.models.cross_attention import CrossGraphAttention

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _build_domain_pairs(cfg: dict) -> list:
    n = cfg["n_signals"]
    return [
        ("security",         "threat_intel",      n["security_threat"]),
        ("decision_history", "threat_intel",       n["decision_threat"]),
        ("security",         "decision_history",   n["security_decision"]),
    ]

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return raw["experiment_2"]


def build_embedding_matrix(entities) -> np.ndarray:
    """Stack entity embeddings into (n, d) array."""
    return np.array([e.embedding for e in entities], dtype=np.float64)


def build_gt_index_set(gt_pairs, entities_i, entities_j) -> set:
    """Convert (entity_id_i, entity_id_j) pairs to (int, int) index pairs."""
    lookup_i = {e.entity_id: idx for idx, e in enumerate(entities_i)}
    lookup_j = {e.entity_id: idx for idx, e in enumerate(entities_j)}
    return {(lookup_i[a], lookup_j[b]) for a, b in gt_pairs}


def compute_metrics(discovered_set: set, gt_set: set, n_gt: int) -> dict:
    """Return TP/FP/FN/precision/recall/F1 for one (method, config) run."""
    tp = len(discovered_set & gt_set)
    fp = len(discovered_set) - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / n_gt      if n_gt > 0         else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    t0  = time.time()
    cfg = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    domain_pairs = _build_domain_pairs(cfg)
    thetas  = cfg["theta_logit_range"]
    ks      = cfg["top_k_range"]
    cos_thr = cfg["cosine_thresholds"]
    sig_str = float(cfg["signal_strength"])
    seeds   = cfg["seeds"]

    rows = []

    def _row(method, seed, pair, theta, k, thr, disc_set, gt_set, n_gt):
        m = compute_metrics(disc_set, gt_set, n_gt)
        return dict(method=method, seed=seed, domain_pair=pair,
                    theta_logit=theta, top_k=k, threshold=thr,
                    n_discovered=len(disc_set),
                    true_pos=m["tp"], false_pos=m["fp"], false_neg=m["fn"],
                    precision=round(m["precision"], 4),
                    recall=round(m["recall"], 4),
                    f1=round(m["f1"], 4))

    for seed in seeds:
        print(f"Seed {seed}...")
        gen     = EntityGenerator(cfg)
        domains = gen.generate_all(seed)

        for dom_i, dom_j, n_sig in domain_pairs:
            gt_pairs = inject_signals(
                domains[dom_i], domains[dom_j], n_sig, sig_str, seed
            )
            E_i    = build_embedding_matrix(domains[dom_i])
            E_j    = build_embedding_matrix(domains[dom_j])
            gt_set = build_gt_index_set(gt_pairs, domains[dom_i], domains[dom_j])
            n_gt   = len(gt_set)
            cga    = CrossGraphAttention()
            pair   = f"{dom_i[:3]}x{dom_j[:3]}"
            m_i, m_j = E_i.shape[0], E_j.shape[0]

            # two_stage: full theta × K grid
            ts_counts = []
            for theta in thetas:
                for k in ks:
                    hits = cga.discover_two_stage(E_i, E_j, theta, k)
                    disc = {(h[0], h[1]) for h in hits}
                    ts_counts.append(len(disc))
                    rows.append(_row("two_stage", seed, pair, theta, k, "", disc, gt_set, n_gt))

            # logit_only: theta sweep
            for theta in thetas:
                hits = cga.discover_logit_only(E_i, E_j, theta)
                disc = {(h[0], h[1]) for h in hits}
                rows.append(_row("logit_only", seed, pair, theta, "", "", disc, gt_set, n_gt))

            # topk_only: K sweep
            for k in ks:
                hits = cga.discover_topk_only(E_i, E_j, k)
                disc = {(h[0], h[1]) for h in hits}
                rows.append(_row("topk_only", seed, pair, "", k, "", disc, gt_set, n_gt))

            # cosine: threshold sweep
            for thr in cos_thr:
                hits = cga.cosine_baseline(E_i, E_j, thr)
                disc = {(h[0], h[1]) for h in hits}
                rows.append(_row("cosine", seed, pair, "", "", thr, disc, gt_set, n_gt))

            # random: analytical expected P/R/F1 at median two_stage discovery count
            n_disc = int(np.median(ts_counts)) if ts_counts else 1
            exp_tp = n_disc * n_gt / (m_i * m_j)
            prec   = exp_tp / n_disc if n_disc > 0 else 0.0
            rec    = exp_tp / n_gt   if n_gt   > 0 else 0.0
            f1r    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rows.append(dict(method="random", seed=seed, domain_pair=pair,
                             theta_logit="", top_k="", threshold="",
                             n_discovered=n_disc,
                             true_pos=round(exp_tp, 4),
                             false_pos=round(n_disc - exp_tp, 4),
                             false_neg=round(n_gt - exp_tp, 4),
                             precision=round(prec, 4),
                             recall=round(rec, 4),
                             f1=round(f1r, 4)))

    # --- write CSV ---
    csv_path   = RESULTS_DIR / "discovery_results.csv"
    fieldnames = ["method", "seed", "domain_pair", "theta_logit", "top_k",
                  "threshold", "n_discovered", "true_pos", "false_pos",
                  "false_neg", "precision", "recall", "f1"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {csv_path}")

    # --- best_configs.json: best mean-F1 config per (method, domain_pair) ---
    agg = defaultdict(list)
    for r in rows:
        key = (r["method"], r["domain_pair"],
               r["theta_logit"], r["top_k"], r["threshold"])
        agg[key].append(float(r["f1"]))

    best = {}
    for (method, pair, theta, k, thr), f1s in agg.items():
        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best.get((method, pair), {}).get("mean_f1", -1.0):
            best[(method, pair)] = dict(method=method, domain_pair=pair,
                                        theta_logit=theta, top_k=k,
                                        threshold=thr,
                                        mean_f1=round(mean_f1, 4))

    json_path = RESULTS_DIR / "best_configs.json"
    with open(json_path, "w") as fh:
        json.dump(list(best.values()), fh, indent=2)
    print(f"Wrote best_configs -> {json_path}")

    # --- summary: best F1 per method ---
    method_f1 = defaultdict(list)
    for r in rows:
        method_f1[r["method"]].append(float(r["f1"]))
    print("\nBest F1 per method (over all configs / pairs / seeds):")
    for method in sorted(method_f1):
        f1s = method_f1[method]
        print(f"  {method:12s}  best={max(f1s):.4f}  mean={np.mean(f1s):.4f}")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_experiment()
