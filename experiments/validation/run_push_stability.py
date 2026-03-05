"""
VALIDATION-2: Push Update Stability Test.

Tests Eq. 4b (incorrect case) of the profile update rule:

  mu_new = mu - eta_eff * (f - mu) = (1 + eta_eff)*mu - eta_eff*f

This is expansive: it pushes mu AWAY from f, which can drive centroid
dimensions outside [0, 1] under sustained incorrect outcomes.

Five conditions tested:
  A: Normal     -- 70% correct, 30% incorrect, 500 decisions
  B: Bad streak -- first 100 incorrect, then 70/30, 500 decisions
  C: Worst case -- 100% incorrect, same (c,a) pair, 200 decisions
  D: Clipped    -- same as C, clip mu to [0,1] after each update
  E: Margin     -- same as C, push only if ||f-mu|| < 1.0

Parameters match paper: eta=0.05, decay=0.001, n_factors=6.
Initial mu ~ Uniform(0.2, 0.8): safely inside [0, 1].

Outputs
-------
experiments/validation/push_stability_results.csv
experiments/validation/push_stability_summary.json
"""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_FACTORS    = 6
N_ACTIONS    = 4
N_CATEGORIES = 5
ETA          = 0.05
DECAY        = 0.001
PUSH_MARGIN  = 1.0     # Condition E threshold on ||f - mu||

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# Target centroid for all conditions: category 0, action 0
TARGET_CAT = 0
TARGET_ACT = 0


# ---------------------------------------------------------------------------
# Update rules  (Eq. 4b and variants)
# ---------------------------------------------------------------------------

def _eta_eff(eta: float, count: int, decay: float) -> float:
    return eta / (1.0 + count * decay)


def update_centroid(
    mu: np.ndarray, f: np.ndarray,
    is_correct: bool, eta: float, count: int, decay: float,
) -> np.ndarray:
    """Eq. 4b: pull toward f if correct, push away if incorrect."""
    ee = _eta_eff(eta, count, decay)
    if is_correct:
        return mu + ee * (f - mu)
    else:
        return mu - ee * (f - mu)


def update_centroid_clipped(
    mu: np.ndarray, f: np.ndarray,
    is_correct: bool, eta: float, count: int, decay: float,
) -> np.ndarray:
    """Eq. 4b + [0,1] clip after every step."""
    mu_new = update_centroid(mu, f, is_correct, eta, count, decay)
    return np.clip(mu_new, 0.0, 1.0)


def update_centroid_margin(
    mu: np.ndarray, f: np.ndarray,
    is_correct: bool, eta: float, count: int, decay: float,
    margin: float = PUSH_MARGIN,
) -> tuple[np.ndarray, bool]:
    """
    Eq. 4b with margin guard on push.
    Returns (mu_new, push_blocked).
    push_blocked is True when an incorrect update was suppressed.
    """
    ee = _eta_eff(eta, count, decay)
    if is_correct:
        return mu + ee * (f - mu), False
    else:
        dist = float(np.linalg.norm(f - mu))
        if dist < margin:
            return mu - ee * (f - mu), False
        else:
            return mu.copy(), True          # push suppressed


# ---------------------------------------------------------------------------
# Per-step statistics
# ---------------------------------------------------------------------------

def _stats(mu: np.ndarray) -> dict:
    return dict(
        mu_norm=float(np.linalg.norm(mu)),
        mu_min=float(mu.min()),
        mu_max=float(mu.max()),
        n_dims_outside=int(np.sum((mu < 0.0) | (mu > 1.0))),
    )


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_condition_A(seed: int) -> list[dict]:
    """Normal: 70% correct / 30% incorrect, 500 decisions."""
    rng = np.random.default_rng(seed)
    mu  = rng.uniform(0.2, 0.8, size=N_FACTORS)
    rows = []
    for dec in range(500):
        f          = rng.uniform(0.0, 1.0, size=N_FACTORS)
        is_correct = bool(rng.random() < 0.70)
        ee         = _eta_eff(ETA, dec, DECAY)
        mu         = update_centroid(mu, f, is_correct, ETA, dec, DECAY)
        rows.append(dict(
            condition="A_normal", seed=seed, decision=dec,
            is_correct=int(is_correct), eta_eff=round(ee, 6),
            **_stats(mu),
        ))
    return rows


def run_condition_B(seed: int) -> list[dict]:
    """Bad streak: first 100 incorrect, then 70/30, 500 decisions."""
    rng = np.random.default_rng(seed)
    mu  = rng.uniform(0.2, 0.8, size=N_FACTORS)
    rows = []
    for dec in range(500):
        f  = rng.uniform(0.0, 1.0, size=N_FACTORS)
        is_correct = False if dec < 100 else bool(rng.random() < 0.70)
        ee = _eta_eff(ETA, dec, DECAY)
        mu = update_centroid(mu, f, is_correct, ETA, dec, DECAY)
        rows.append(dict(
            condition="B_bad_streak", seed=seed, decision=dec,
            is_correct=int(is_correct), eta_eff=round(ee, 6),
            **_stats(mu),
        ))
    return rows


def run_condition_C(seed: int) -> list[dict]:
    """Worst case: 100% incorrect, 200 decisions on same (c,a)."""
    rng = np.random.default_rng(seed)
    mu  = rng.uniform(0.2, 0.8, size=N_FACTORS)
    rows = []
    for dec in range(200):
        f  = rng.uniform(0.0, 1.0, size=N_FACTORS)
        ee = _eta_eff(ETA, dec, DECAY)
        mu = update_centroid(mu, f, is_correct=False, eta=ETA, count=dec, decay=DECAY)
        rows.append(dict(
            condition="C_worst_case", seed=seed, decision=dec,
            is_correct=0, eta_eff=round(ee, 6),
            **_stats(mu),
        ))
    return rows


def run_condition_D(seed: int) -> list[dict]:
    """Clipped: same as C but clip mu to [0,1] after each update."""
    rng = np.random.default_rng(seed)
    mu  = rng.uniform(0.2, 0.8, size=N_FACTORS)
    rows = []
    n_clipped = 0
    for dec in range(200):
        f       = rng.uniform(0.0, 1.0, size=N_FACTORS)
        ee      = _eta_eff(ETA, dec, DECAY)
        mu_raw  = update_centroid(mu, f, is_correct=False, eta=ETA, count=dec, decay=DECAY)
        mu_clip = np.clip(mu_raw, 0.0, 1.0)
        if not np.array_equal(mu_raw, mu_clip):
            n_clipped += 1
        mu = mu_clip
        rows.append(dict(
            condition="D_clipped", seed=seed, decision=dec,
            is_correct=0, eta_eff=round(ee, 6),
            **_stats(mu),
            n_updates_clipped=n_clipped,   # cumulative
        ))
    return rows, n_clipped


def run_condition_E(seed: int) -> list[dict]:
    """Margin: same as C but push only if ||f-mu|| < margin=1.0."""
    rng = np.random.default_rng(seed)
    mu  = rng.uniform(0.2, 0.8, size=N_FACTORS)
    rows = []
    n_blocked = 0
    for dec in range(200):
        f  = rng.uniform(0.0, 1.0, size=N_FACTORS)
        ee = _eta_eff(ETA, dec, DECAY)
        mu, blocked = update_centroid_margin(
            mu, f, is_correct=False, eta=ETA, count=dec, decay=DECAY,
        )
        if blocked:
            n_blocked += 1
        rows.append(dict(
            condition="E_margin", seed=seed, decision=dec,
            is_correct=0, eta_eff=round(ee, 6),
            **_stats(mu),
            n_pushes_blocked=n_blocked,    # cumulative
        ))
    return rows, n_blocked


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def _summarize_AC(rows: list[dict], label: str) -> dict:
    """Summary for conditions A and B (track escape and final state)."""
    n_total    = len(rows)
    max_norm   = max(r["mu_norm"] for r in rows)
    final_norm = float(np.mean([r["mu_norm"] for r in rows if r["decision"] == rows[-1]["decision"]]))
    n_outside  = sum(1 for r in rows if r["n_dims_outside"] > 0)
    max_dims   = max(r["n_dims_outside"] for r in rows)

    first_escape = None
    for r in sorted(rows, key=lambda x: x["decision"]):
        if r["n_dims_outside"] > 0:
            first_escape = r["decision"]
            break

    return {
        "final_mean_norm":            round(float(np.mean([r["mu_norm"] for r in rows
                                                            if r["decision"] == rows[-1]["decision"]])), 4),
        "max_norm_seen":              round(max_norm, 4),
        "pct_decisions_outside_bounds": round(100 * n_outside / n_total, 2),
        "max_dims_outside":           max_dims,
        "decisions_until_first_escape": first_escape,
    }


def _summarize_D(rows: list[dict]) -> dict:
    all_outside = [r["n_dims_outside"] for r in rows]
    all_clip    = [r["n_updates_clipped"] for r in rows]
    n_dec       = max(r["decision"] for r in rows) + 1
    return {
        "final_mean_norm":    round(float(np.mean([r["mu_norm"] for r in rows
                                                    if r["decision"] == n_dec - 1])), 4),
        "max_norm_seen":      round(max(r["mu_norm"] for r in rows), 4),
        "pct_decisions_clipped": round(100 * all_clip[-1] / n_dec, 2),
        "all_dims_in_bounds": max(all_outside) == 0,
    }


def _summarize_E(rows: list[dict]) -> dict:
    all_outside = [r["n_dims_outside"] for r in rows]
    all_blocked = [r["n_pushes_blocked"] for r in rows]
    n_dec       = max(r["decision"] for r in rows) + 1
    return {
        "final_mean_norm":    round(float(np.mean([r["mu_norm"] for r in rows
                                                    if r["decision"] == n_dec - 1])), 4),
        "max_norm_seen":      round(max(r["mu_norm"] for r in rows), 4),
        "pct_pushes_blocked_by_margin": round(100 * all_blocked[-1] / n_dec, 2),
        "all_dims_in_bounds": max(all_outside) == 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_push_stability() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    # Per-condition data indexed by seed
    cond_D_data: dict[int, tuple[list, int]] = {}
    cond_E_data: dict[int, tuple[list, int]] = {}

    print("Condition A (normal 70/30) ...")
    for seed in SEEDS:
        all_rows.extend(run_condition_A(seed))

    print("Condition B (bad streak) ...")
    for seed in SEEDS:
        all_rows.extend(run_condition_B(seed))

    print("Condition C (worst case 100% incorrect) ...")
    for seed in SEEDS:
        all_rows.extend(run_condition_C(seed))

    print("Condition D (clipped) ...")
    for seed in SEEDS:
        rows_d, n_clip = run_condition_D(seed)
        cond_D_data[seed] = (rows_d, n_clip)
        all_rows.extend(rows_d)

    print("Condition E (margin) ...")
    for seed in SEEDS:
        rows_e, n_blk = run_condition_E(seed)
        cond_E_data[seed] = (rows_e, n_blk)
        all_rows.extend(rows_e)

    # -----------------------------------------------------------------------
    # Write CSV  (flatten extra condition-specific columns to common schema)
    # -----------------------------------------------------------------------
    csv_path   = OUT_DIR / "push_stability_results.csv"
    fieldnames = ["condition", "seed", "decision", "is_correct",
                  "eta_eff", "mu_norm", "mu_min", "mu_max", "n_dims_outside"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {csv_path}")

    # -----------------------------------------------------------------------
    # Build summaries per condition (aggregate across seeds)
    # -----------------------------------------------------------------------
    def _rows_for(cond_prefix: str) -> list[dict]:
        return [r for r in all_rows if r["condition"].startswith(cond_prefix)]

    rows_A = _rows_for("A_")
    rows_B = _rows_for("B_")
    rows_C = _rows_for("C_")
    rows_D = [r for rs, _ in cond_D_data.values() for r in rs]
    rows_E = [r for rs, _ in cond_E_data.values() for r in rs]

    sum_A = _summarize_AC(rows_A, "A")
    sum_B = _summarize_AC(rows_B, "B")
    sum_C = _summarize_AC(rows_C, "C")
    sum_C["final_max_dimension"] = round(max(r["mu_max"] for r in rows_C
                                            if r["decision"] == 199), 4)
    sum_D = _summarize_D(rows_D)
    sum_E = _summarize_E(rows_E)

    # Recommendation
    c_escaped  = sum_C["decisions_until_first_escape"] is not None
    d_in_bounds = sum_D["all_dims_in_bounds"]
    e_in_bounds = sum_E["all_dims_in_bounds"]

    if not c_escaped:
        recommendation = "push_is_safe"
    elif d_in_bounds and e_in_bounds:
        recommendation = "both"
    elif d_in_bounds:
        recommendation = "clip"
    elif e_in_bounds:
        recommendation = "margin"
    else:
        recommendation = "clip"   # clip is simplest; margin alone insufficient

    summary = {
        "condition_A_normal":     sum_A,
        "condition_B_bad_streak": sum_B,
        "condition_C_worst_case": sum_C,
        "condition_D_clipped":    sum_D,
        "condition_E_margin":     sum_E,
        "recommendation":         recommendation,
    }

    json_path = OUT_DIR / "push_stability_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("VALIDATION-2: Push Update Stability  (Eq. 4b incorrect case)")
    print("=" * 72)
    print(f"  eta={ETA}, decay={DECAY}, n_factors={N_FACTORS}, "
          f"mu_init=U(0.2,0.8), {len(SEEDS)} seeds")

    def _norm_trajectory(rows: list[dict], n_dec: int) -> str:
        """Mean norm at decision 0, 25%, 50%, 75%, 100%."""
        checkpoints = [0, n_dec // 4, n_dec // 2, 3 * n_dec // 4, n_dec - 1]
        vals = []
        for cp in checkpoints:
            cp_rows = [r for r in rows if r["decision"] == cp]
            vals.append(f"{np.mean([r['mu_norm'] for r in cp_rows]):.3f}")
        return "  ->  ".join(vals)

    print()
    print("  Condition A (normal 70/30, 500 decisions):")
    print(f"    Norm trajectory (dec 0,125,250,375,499):")
    print(f"      {_norm_trajectory(rows_A, 500)}")
    print(f"    max_norm={sum_A['max_norm_seen']:.4f}  "
          f"pct_outside={sum_A['pct_decisions_outside_bounds']:.1f}%  "
          f"max_dims_outside={sum_A['max_dims_outside']}")
    esc_A = sum_A["decisions_until_first_escape"]
    print(f"    first_escape={'never' if esc_A is None else f'dec {esc_A}'}")

    print()
    print("  Condition B (bad streak: 100 incorrect then 70/30, 500 decisions):")
    print(f"    Norm trajectory (dec 0,125,250,375,499):")
    print(f"      {_norm_trajectory(rows_B, 500)}")
    print(f"    max_norm={sum_B['max_norm_seen']:.4f}  "
          f"pct_outside={sum_B['pct_decisions_outside_bounds']:.1f}%  "
          f"max_dims_outside={sum_B['max_dims_outside']}")
    esc_B = sum_B["decisions_until_first_escape"]
    print(f"    first_escape={'never' if esc_B is None else f'dec {esc_B}'}")

    print()
    print("  Condition C (100% incorrect, 200 decisions) — WORST CASE:")
    print(f"    Norm trajectory (dec 0,50,100,150,199):")
    print(f"      {_norm_trajectory(rows_C, 200)}")
    print(f"    max_norm={sum_C['max_norm_seen']:.4f}  "
          f"max_dims_outside={sum_C['max_dims_outside']}  "
          f"final_max_dim={sum_C['final_max_dimension']:.4f}")
    esc_C = sum_C["decisions_until_first_escape"]
    print(f"    first_escape={'never' if esc_C is None else f'dec {esc_C}'}")

    print()
    print("  Condition D (clipped to [0,1]):")
    print(f"    Norm trajectory (dec 0,50,100,150,199):")
    print(f"      {_norm_trajectory(rows_D, 200)}")
    print(f"    max_norm={sum_D['max_norm_seen']:.4f}  "
          f"pct_clipped={sum_D['pct_decisions_clipped']:.1f}%  "
          f"all_in_bounds={sum_D['all_dims_in_bounds']}")

    print()
    print(f"  Condition E (margin={PUSH_MARGIN}, push blocked if ||f-mu||>=margin):")
    print(f"    Norm trajectory (dec 0,50,100,150,199):")
    print(f"      {_norm_trajectory(rows_E, 200)}")
    print(f"    max_norm={sum_E['max_norm_seen']:.4f}  "
          f"pct_blocked={sum_E['pct_pushes_blocked_by_margin']:.1f}%  "
          f"all_in_bounds={sum_E['all_dims_in_bounds']}")

    # Per-seed breakdown for condition C
    print()
    print("  Condition C — per-seed first escape decision:")
    for seed in SEEDS:
        seed_rows = sorted(
            [r for r in rows_C if r["seed"] == seed],
            key=lambda x: x["decision"],
        )
        first = next((r["decision"] for r in seed_rows if r["n_dims_outside"] > 0), None)
        final_norm = seed_rows[-1]["mu_norm"]
        print(f"    seed={seed:<5d}  first_escape={'never' if first is None else f'dec {first}':<10s}  "
              f"final_norm={final_norm:.3f}")

    print()
    print(f"  VERDICT: recommendation = '{recommendation}'")
    if recommendation == "push_is_safe":
        print(f"    Push update stays bounded under adversarial conditions")
    elif recommendation == "clip":
        print(f"    Clipping to [0,1] prevents escape; margin alone insufficient")
        print(f"    CONFIRMS reviewer: add np.clip(mu, 0, 1) after each update")
    elif recommendation == "margin":
        print(f"    Margin guard prevents escape; clipping also works")
    elif recommendation == "both":
        print(f"    Both clipping AND margin guard prevent escape")
        print(f"    CONFIRMS reviewer: push rule needs explicit boundary enforcement")

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    run_push_stability()
