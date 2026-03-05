"""
Experiment 3 Extended: Multi-domain scaling n=2..15.

Extends run.py from domain_counts=[2,3,4,5,6] to
domain_counts=[2,3,4,5,6,7,8,9,10,12,15] (11 data points, range 2-15).

For domains 7-15, additional domain profiles are generated with
semantic_means drawn from N(0.5, 0.3), clipped to [0.05, 0.95], using
a fixed profile seed (1999) for reproducibility.

Fits discoveries = a*n^b on:
  - original range [2-6]  (reproduces ~2.30 from run.py)
  - full extended range [2-15]

Alternative model comparison (full range):
  - pure quadratic: a*n^2
  - quadratic+log:  a*n^2*log(n)

AIC/BIC computed for all models (Gaussian log-likelihood form).
95% CI on exponent b from curve_fit covariance matrix via t-distribution.

Outputs
-------
experiments/exp3_multidomain_scaling/results/extended_scaling_data.csv
experiments/exp3_multidomain_scaling/results/extended_scaling_fit.json
"""
from __future__ import annotations

import copy, csv, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.entity_generator import EntityGenerator, inject_signals, _DOMAIN_PROFILES
from src.models.cross_attention import CrossGraphAttention

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Domain name tables
# ---------------------------------------------------------------------------
ALL_DOMAIN_NAMES_BASE = [
    "security", "decision_history", "threat_intel",
    "network_flow", "asset_inventory", "user_behavior",
]
_EXTRA_NAMES_EXT = [f"domain_{i:02d}" for i in range(7, 16)]
ALL_DOMAIN_NAMES_EXT = ALL_DOMAIN_NAMES_BASE + _EXTRA_NAMES_EXT  # 15 total

EXTENDED_DOMAIN_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
ORIGINAL_DOMAIN_COUNTS = [2, 3, 4, 5, 6]
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# ---------------------------------------------------------------------------
# Domain profiles
# ---------------------------------------------------------------------------
# Same extra profiles as run.py for domains 4-6 (must be identical for
# reproducibility when n<=6)
_EXTRA_PROFILES_BASE = {
    "network_flow":    {"semantic_means": [0.3, 0.6, 0.2, 0.7, 0.4, 0.5], "n_entities": 200},
    "asset_inventory": {"semantic_means": [0.6, 0.3, 0.8, 0.4, 0.2, 0.7], "n_entities": 200},
    "user_behavior":   {"semantic_means": [0.1, 0.8, 0.4, 0.6, 0.9, 0.2], "n_entities": 200},
}

# Profiles for domains 7-15: fixed seed for reproducibility across runs
_rng_profile = np.random.default_rng(1999)
_EXTRA_PROFILES_EXT: dict[str, dict] = {}
for _dn in _EXTRA_NAMES_EXT:
    _raw = _rng_profile.normal(0.5, 0.3, size=6)
    _raw = np.clip(_raw, 0.05, 0.95)
    _EXTRA_PROFILES_EXT[_dn] = {"semantic_means": _raw.tolist(), "n_entities": 200}

ALL_PROFILES = {**_DOMAIN_PROFILES, **_EXTRA_PROFILES_BASE, **_EXTRA_PROFILES_EXT}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> tuple[dict, float]:
    with open(ROOT / "configs" / "default.yaml") as fh:
        raw = yaml.safe_load(fh)
    return raw["experiment_3"], float(raw["experiment_2"]["signal_strength"])


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------
def _power_law(x, a, b):
    return a * x ** b

def _pure_quadratic(x, a):
    return a * x ** 2

def _quad_log(x, a):
    return a * x ** 2 * np.log(x)

def _r2(ys: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def _aic(n_obs: int, k_params: int, sse: float) -> float:
    """AIC (Gaussian likelihood): n*log(SSE/n) + 2k."""
    if sse <= 0 or n_obs <= 0:
        return float("nan")
    return n_obs * float(np.log(sse / n_obs)) + 2 * k_params

def _bic(n_obs: int, k_params: int, sse: float) -> float:
    """BIC: n*log(SSE/n) + k*log(n)."""
    if sse <= 0 or n_obs <= 0:
        return float("nan")
    return n_obs * float(np.log(sse / n_obs)) + k_params * float(np.log(n_obs))

def fit_power_law(ns: np.ndarray, ys: np.ndarray) -> dict:
    """Fit discoveries = a*n^b. Returns full stats dict."""
    popt, pcov = curve_fit(_power_law, ns, ys, p0=[1.0, 2.0])
    a, b = float(popt[0]), float(popt[1])
    y_pred = _power_law(ns, a, b)
    sse    = float(np.sum((ys - y_pred) ** 2))
    r2     = _r2(ys, y_pred)
    n_obs  = len(ns)
    k      = 2
    # 95% CI on b via t-distribution
    se_b   = float(np.sqrt(max(pcov[1, 1], 0.0)))
    df     = max(n_obs - k, 1)
    t_crit = float(t_dist.ppf(0.975, df=df))
    return dict(
        a=round(a, 4), b=round(b, 4),
        r_squared=round(r2, 6),
        sse=round(sse, 4),
        aic=round(_aic(n_obs, k, sse), 4),
        bic=round(_bic(n_obs, k, sse), 4),
        ci_95_lower=round(b - t_crit * se_b, 4),
        ci_95_upper=round(b + t_crit * se_b, 4),
    )

def fit_1param(model_fn, ns: np.ndarray, ys: np.ndarray, model_str: str) -> dict:
    """Fit 1-parameter model. Returns stats dict."""
    popt, _ = curve_fit(model_fn, ns, ys, p0=[1.0])
    a       = float(popt[0])
    y_pred  = model_fn(ns, a)
    sse     = float(np.sum((ys - y_pred) ** 2))
    r2      = _r2(ys, y_pred)
    n_obs   = len(ns)
    k       = 1
    return dict(
        model=model_str, a=round(a, 4),
        r_squared=round(r2, 6),
        sse=round(sse, 4),
        aic=round(_aic(n_obs, k, sse), 4),
        bic=round(_bic(n_obs, k, sse), 4),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_extended() -> None:
    t0 = time.time()
    cfg, sig_str = load_config()

    n_entities = cfg["entities_per_domain"]  # 200
    n_sig      = cfg["signals_per_pair"]     # 5
    THETA, K   = 0.02, 3
    cga        = CrossGraphAttention()

    gen = EntityGenerator({"experiment_2": {"domain_profiles": ALL_PROFILES}})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in SEEDS:
        print(f"Seed {seed} ...", flush=True)
        rng       = np.random.default_rng(seed)
        # Draw sub-seeds for all 15 domains at once.
        # First 6 values are identical to run.py (same seed, same call order).
        dom_seeds = rng.integers(0, 2**31, size=len(ALL_DOMAIN_NAMES_EXT))

        # Generate all domains once per seed; deep-copy per pair to isolate inject_signals
        base = {
            name: gen.generate_domain(name, n_entities, int(dom_seeds[k]))
            for k, name in enumerate(ALL_DOMAIN_NAMES_EXT)
        }

        for n_dom in EXTENDED_DOMAIN_COUNTS:
            names_n = ALL_DOMAIN_NAMES_EXT[:n_dom]
            n_pairs = n_dom * (n_dom - 1) // 2

            total_disc = total_gt = total_tp = 0
            prec_sum = rec_sum = f1_sum = 0.0

            for pi, (di, dj) in enumerate(combinations(names_n, 2)):
                ei = copy.deepcopy(base[di])
                ej = copy.deepcopy(base[dj])
                # Identical seed formula to run.py: seed*100 + pair_index
                gt_pairs = inject_signals(ei, ej, n_sig, sig_str, seed * 100 + pi)

                E_i  = np.array([e.embedding for e in ei])
                E_j  = np.array([e.embedding for e in ej])
                hits = cga.discover_two_stage(E_i, E_j, THETA, K)

                disc_set = {(h[0], h[1]) for h in hits}
                id2i     = {e.entity_id: k for k, e in enumerate(ei)}
                id2j     = {e.entity_id: k for k, e in enumerate(ej)}
                gt_set   = {(id2i[a], id2j[b]) for a, b in gt_pairs}

                tp = len(disc_set & gt_set)
                nd = len(disc_set)
                ng = len(gt_set)

                prec = tp / nd if nd > 0 else 0.0
                rec  = tp / ng if ng > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                total_disc += nd
                total_gt   += ng
                total_tp   += tp
                prec_sum   += prec
                rec_sum    += rec
                f1_sum     += f1

            mean_prec = prec_sum / n_pairs if n_pairs > 0 else 0.0
            mean_rec  = rec_sum  / n_pairs if n_pairs > 0 else 0.0
            mean_f1   = f1_sum   / n_pairs if n_pairs > 0 else 0.0

            rows.append(dict(
                seed=seed, n_domains=n_dom, n_pairs=n_pairs,
                total_discoveries=total_disc,
                mean_f1=round(mean_f1, 4),
                mean_precision=round(mean_prec, 4),
                mean_recall=round(mean_rec, 4),
            ))
            print(f"  n={n_dom:2d}: {n_pairs:3d} pairs, "
                  f"{total_disc:5d} disc, F1={mean_f1:.3f}", flush=True)

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path   = RESULTS_DIR / "extended_scaling_data.csv"
    fieldnames = ["seed", "n_domains", "n_pairs", "total_discoveries",
                  "mean_f1", "mean_precision", "mean_recall"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")

    # -----------------------------------------------------------------------
    # Aggregate by n_domains (mean across seeds)
    # -----------------------------------------------------------------------
    disc_by_n: dict[int, list[float]] = {}
    f1_by_n:   dict[int, list[float]] = {}
    for r in rows:
        disc_by_n.setdefault(r["n_domains"], []).append(float(r["total_discoveries"]))
        f1_by_n.setdefault(r["n_domains"], []).append(float(r["mean_f1"]))

    all_ns = np.array(sorted(disc_by_n), dtype=float)
    all_ys = np.array([np.mean(disc_by_n[int(n)]) for n in all_ns])

    orig_mask = np.isin(all_ns, ORIGINAL_DOMAIN_COUNTS)
    orig_ns   = all_ns[orig_mask]
    orig_ys   = all_ys[orig_mask]

    # -----------------------------------------------------------------------
    # Power-law fits
    # -----------------------------------------------------------------------
    fit_orig = fit_power_law(orig_ns, orig_ys)
    fit_ext  = fit_power_law(all_ns,  all_ys)

    # Alternative models on full range
    fit_q  = fit_1param(_pure_quadratic, all_ns, all_ys, "a * n^2")
    fit_ql = fit_1param(_quad_log,       all_ns, all_ys, "a * n^2 * log(n)")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    fit_data = {
        "original_range": {
            "domain_counts":  [int(n) for n in orig_ns],
            "fitted_exponent": fit_orig["b"],
            "r_squared":       fit_orig["r_squared"],
            "a":               fit_orig["a"],
            "sse":             fit_orig["sse"],
            "aic":             fit_orig["aic"],
            "bic":             fit_orig["bic"],
            "ci_95_lower":     fit_orig["ci_95_lower"],
            "ci_95_upper":     fit_orig["ci_95_upper"],
        },
        "extended_range": {
            "domain_counts":  [int(n) for n in all_ns],
            "fitted_exponent": fit_ext["b"],
            "r_squared":       fit_ext["r_squared"],
            "a":               fit_ext["a"],
            "sse":             fit_ext["sse"],
            "aic":             fit_ext["aic"],
            "bic":             fit_ext["bic"],
            "ci_95_lower":     fit_ext["ci_95_lower"],
            "ci_95_upper":     fit_ext["ci_95_upper"],
        },
        "comparison": {
            "exponent_shift": round(fit_ext["b"] - fit_orig["b"], 4),
            "ci_95_lower":    fit_ext["ci_95_lower"],
            "ci_95_upper":    fit_ext["ci_95_upper"],
        },
        "alternative_models_full_range": {
            "pure_quadratic": fit_q,
            "quadratic_log":  fit_ql,
        },
    }
    json_path = RESULTS_DIR / "extended_scaling_fit.json"
    with open(json_path, "w") as fh:
        json.dump(fit_data, fh, indent=2)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    b_o = fit_orig["b"];  r2_o = fit_orig["r_squared"]
    b_e = fit_ext["b"];   r2_e = fit_ext["r_squared"]
    ci_lo_o = fit_orig["ci_95_lower"]; ci_hi_o = fit_orig["ci_95_upper"]
    ci_lo_e = fit_ext["ci_95_lower"];  ci_hi_e = fit_ext["ci_95_upper"]

    print("\n" + "=" * 68)
    print("EXP-3 EXTENDED: Power-law scaling  discoveries = a * n^b")
    print("=" * 68)
    print(f"\n  {'n':>4}  {'pairs':>6}  {'mean_disc':>10}  {'predicted':>10}  "
          f"{'mean_F1':>8}")
    print(f"  {'-'*52}")
    for n, y in zip(all_ns.astype(int), all_ys):
        pred = _power_law(float(n), fit_ext["a"], b_e)
        mf1  = float(np.mean(f1_by_n[n]))
        marker = " *" if n in ORIGINAL_DOMAIN_COUNTS else "  "
        print(f"{marker} {n:>4}  {n*(n-1)//2:>6}  {y:>10.1f}  {pred:>10.1f}  {mf1:>8.3f}")
    print(f"  (* = original range)")

    print(f"\n  Power-law fit  (discoveries = a * n^b):")
    print(f"    Original [2-6]:  b = {b_o:.4f}  R² = {r2_o:.4f}  "
          f"CI [{ci_lo_o:.4f}, {ci_hi_o:.4f}]")
    print(f"    Extended [2-15]: b = {b_e:.4f}  R² = {r2_e:.4f}  "
          f"CI [{ci_lo_e:.4f}, {ci_hi_e:.4f}]")
    print(f"    Exponent shift:  {fit_ext['b'] - fit_orig['b']:+.4f}")

    print(f"\n  Alternative models (full range, n=11 points):")
    print(f"  {'Model':<28s}  {'R²':>6}  {'AIC':>8}  {'BIC':>8}")
    print(f"  {'-'*54}")
    print(f"  {'power_law (b free)':<28s}  {r2_e:>6.4f}  "
          f"{fit_ext['aic']:>8.2f}  {fit_ext['bic']:>8.2f}")
    print(f"  {'pure_quadratic (b=2)':<28s}  {fit_q['r_squared']:>6.4f}  "
          f"{fit_q['aic']:>8.2f}  {fit_q['bic']:>8.2f}")
    print(f"  {'quadratic_log (b=2+log)':<28s}  {fit_ql['r_squared']:>6.4f}  "
          f"{fit_ql['aic']:>8.2f}  {fit_ql['bic']:>8.2f}")

    print(f"\n  Reviewer concern addressed:")
    verdict = "CONFIRMED" if abs(b_e - 2.0) < 0.15 else "REVISED"
    print(f"    Extended exponent b={b_e:.4f} (original b={b_o:.4f})")
    print(f"    95% CI [{ci_lo_e:.4f}, {ci_hi_e:.4f}]  -> {verdict}")
    if 2.0 >= ci_lo_e and 2.0 <= ci_hi_e:
        print(f"    n^2.0 is within the 95% CI — pure quadratic is plausible")
    else:
        print(f"    n^2.0 is OUTSIDE the 95% CI — superquadratic growth confirmed")

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    run_extended()
