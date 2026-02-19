"""Experiment 3: Multi-domain scaling — validates discovery count scales as n(n-1)/2."""
import copy, csv, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.entity_generator import EntityGenerator, inject_signals, _DOMAIN_PROFILES
from src.models.cross_attention import CrossGraphAttention

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ALL_DOMAIN_NAMES = [
    "security", "decision_history", "threat_intel",
    "network_flow", "asset_inventory", "user_behavior",
]

# Profiles for the three extra domains — semantic means clearly separated from base three
_EXTRA_PROFILES = {
    "network_flow":    {"semantic_means": [0.3, 0.6, 0.2, 0.7, 0.4, 0.5], "n_entities": 200},
    "asset_inventory": {"semantic_means": [0.6, 0.3, 0.8, 0.4, 0.2, 0.7], "n_entities": 200},
    "user_behavior":   {"semantic_means": [0.1, 0.8, 0.4, 0.6, 0.9, 0.2], "n_entities": 200},
}


def load_config() -> tuple[dict, float]:
    with open(ROOT / "configs" / "default.yaml") as fh:
        raw = yaml.safe_load(fh)
    return raw["experiment_3"], float(raw["experiment_2"]["signal_strength"])


def run_experiment() -> None:
    t0 = time.time()
    cfg, sig_str = load_config()

    domain_counts = cfg["domain_counts"]        # [2, 3, 4, 5, 6]
    n_entities    = cfg["entities_per_domain"]  # 200
    n_sig         = cfg["signals_per_pair"]     # 5
    seeds         = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
    THETA, K      = 0.02, 3
    cga           = CrossGraphAttention()

    all_profiles = {**_DOMAIN_PROFILES, **_EXTRA_PROFILES}
    gen = EntityGenerator({"experiment_2": {"domain_profiles": all_profiles}})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in seeds:
        print(f"Seed {seed}...")
        rng       = np.random.default_rng(seed)
        dom_seeds = rng.integers(0, 2**31, size=len(ALL_DOMAIN_NAMES))

        # Generate all 6 domains once per seed; deep-copy per pair to isolate inject_signals
        base = {
            name: gen.generate_domain(name, n_entities, int(dom_seeds[k]))
            for k, name in enumerate(ALL_DOMAIN_NAMES)
        }

        for n_dom in domain_counts:
            names_n    = ALL_DOMAIN_NAMES[:n_dom]
            n_pairs    = n_dom * (n_dom - 1) // 2
            total_disc = total_gt = total_tp = 0

            for pi, (di, dj) in enumerate(combinations(names_n, 2)):
                ei = copy.deepcopy(base[di])
                ej = copy.deepcopy(base[dj])
                # Stable, unique seed per (seed, pair_index): keeps per-pair signals
                # consistent across n_domains so scaling is purely structural
                gt_pairs = inject_signals(ei, ej, n_sig, sig_str, seed * 100 + pi)

                E_i  = np.array([e.embedding for e in ei])
                E_j  = np.array([e.embedding for e in ej])
                hits = cga.discover_two_stage(E_i, E_j, THETA, K)

                disc_set = {(h[0], h[1]) for h in hits}
                id2i     = {e.entity_id: k for k, e in enumerate(ei)}
                id2j     = {e.entity_id: k for k, e in enumerate(ej)}
                gt_set   = {(id2i[a], id2j[b]) for a, b in gt_pairs}

                total_disc += len(disc_set)
                total_gt   += len(gt_set)
                total_tp   += len(disc_set & gt_set)

            rows.append(dict(seed=seed, n_domains=n_dom, n_pairs=n_pairs,
                             total_discoveries=total_disc,
                             total_ground_truth=total_gt, total_tp=total_tp))

    # --- write CSV ---
    csv_path   = RESULTS_DIR / "scaling_data.csv"
    fieldnames = ["seed", "n_domains", "n_pairs", "total_discoveries",
                  "total_ground_truth", "total_tp"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")

    # --- power-law fit: discoveries = a * n^b ---
    disc_by_n = {}
    for r in rows:
        disc_by_n.setdefault(r["n_domains"], []).append(r["total_discoveries"])
    ns   = np.array(sorted(disc_by_n), dtype=float)
    ys   = np.array([np.mean(disc_by_n[int(n)]) for n in ns])

    def power_law(x, a, b):
        return a * x ** b

    popt, _  = curve_fit(power_law, ns, ys, p0=[1.0, 2.0])
    y_pred   = power_law(ns, *popt)
    ss_res   = float(np.sum((ys - y_pred) ** 2))
    ss_tot   = float(np.sum((ys - ys.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    print(f"\nFitted exponent b = {popt[1]:.2f} (expected ~2.0 for quadratic)")
    print(f"R-squared          = {r2:.4f}")

    # --- summary table ---
    print(f"\n{'n_domains':>10}  {'mean_discoveries':>17}  {'n*(n-1)/2':>11}")
    for n, y in zip(ns.astype(int), ys):
        print(f"{n:>10}  {y:>17.1f}  {n*(n-1)//2:>11}")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_experiment()
