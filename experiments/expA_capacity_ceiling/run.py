"""
EXP-A: Capacity Ceiling — Shared-W Limitations and Gating Recovery.

Tests whether per-category gating (G matrix) overcomes the scoring-matrix
capacity ceiling when alert profiles are category-varying (non-linearly
separable across categories).

Experiment design
-----------------
Profile sets × Scoring configs × Seeds × Decisions
  2           ×  3              × 10    × 1000

Profile sets
  simplified  — orthogonal factor signatures (bridge_common, linearly separable)
  realistic   — category-varying signatures (realistic_profiles, non-separable)

Scoring configs
  w_only       — UniformGating, pure Eq. 4 baseline (G = ones everywhere)
  w_g_static   — MIGating fitted once on a 200-alert oracle batch, then frozen
  w_g_learned  — HebbianGating, online Eq. 4d updates after each decision

Key design constraints
  - W initialised uniformly (no profile-based warm-start) — tests whether G
    eliminates the need for warm-start when profiles are orthogonal.
  - Stochastic policy: actions sampled from softmax(gated_f @ W.T / tau).
  - MIGating fit: gt_actions passed as both system_actions and gt_actions
    (all-correct scenario -> MI=0 -> G~0.27 everywhere, weaker than w_only).

Profile sets
  simplified  — from simplified_profiles section: truly orthogonal factor signatures
                (same across all categories). W can converge to ~85% from cold start.
  realistic   — from realistic_profiles section: category-varying signatures, structural
                conflicts prevent W alone from exceeding ~50%.

Outputs
-------
  experiments/expA_capacity_ceiling/results/accuracy_trajectories.csv  (360 rows)
  experiments/expA_capacity_ceiling/results/g_matrices.csv              (40 rows)
  experiments/expA_capacity_ceiling/results/summary.json

Validation gates
----------------
  VA.1: Simplified  W-only  final acc  > 80%   (orthogonal profiles are learnable cold-start)
  VA.3: Realistic   W-only  final acc  < 55%   (structural capacity ceiling)
  VA.6: capacity_gap = simplified_w_only - realistic_w_only >= 25pp -> PASS
        (G-config results are reported for context; Hebbian G insufficient alone)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.category_alert_generator import CategoryAlertGenerator
from src.models.scoring_matrix import ScoringMatrix, softmax
from src.models.gating import UniformGating, HebbianGating, MIGating

RESULTS_DIR = ROOT / "experiments" / "expA_capacity_ceiling" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(ROOT / "configs" / "default.yaml") as fh:
        return yaml.safe_load(fh)


def _build_sm(bc: dict, cfg: dict, w_init: str) -> ScoringMatrix:
    """
    Build a ScoringMatrix.

    Uses expA's ``scoring`` override if present; falls back to bridge_common scoring.
    This allows EXP-A to use a symmetric ratio (alpha_c == alpha_i, break-even at 50%)
    without touching EXP-5's bridge_common parameters.
    """
    sc = cfg.get("scoring", bc["scoring"])
    return ScoringMatrix(
        n_actions=int(bc["n_actions"]),
        n_factors=int(bc["n_factors"]),
        temperature=float(sc["temperature"]),
        alpha_correct=float(sc["alpha_correct"]),
        alpha_incorrect=float(sc["alpha_incorrect"]),
        weight_clamp=float(sc["weight_clamp"]),
        decay_rate=float(sc["decay_rate"]),
        init_method=w_init,
    )


def _decide_stochastic(
    sm: ScoringMatrix,
    factors: np.ndarray,
    gate: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Stochastic gated decision.

    Applies per-factor gating then samples action from softmax(gated_f @ W.T / tau).
    Does NOT use sm.decide_stochastic() because that method is not seeded with our
    dedicated policy RNG.

    Returns
    -------
    action_idx : int
    """
    gated_f = factors * gate
    logits = gated_f @ sm.W.T / sm.temperature
    probs = softmax(logits)
    return int(rng.choice(sm.n_actions, p=probs))


def _decide_augmented(
    sm: ScoringMatrix,
    factors: np.ndarray,
    category_index: int,
    n_categories: int,
    rng: np.random.Generator,
) -> int:
    """
    Stochastic decision using category-augmented factor vector.

    Appends a one-hot category encoding to f, expanding W from
    (n_actions, n_factors) to (n_actions, n_factors + n_categories) on
    the first call.  Subsequent calls reuse the already-expanded W.

    Returns
    -------
    action_idx : int
    """
    f = factors.flatten()
    if sm.W.shape[1] == len(f):
        rng_init = np.random.default_rng(42)
        W_cat = rng_init.uniform(-0.01, 0.01, (sm.n_actions, n_categories))
        sm.W = np.hstack([sm.W, W_cat])
        sm.n_factors = sm.W.shape[1]
    cat_onehot = np.zeros(n_categories)
    cat_onehot[category_index] = 1.0
    f_aug = np.concatenate([f, cat_onehot])
    logits = f_aug @ sm.W.T / sm.temperature
    probs = softmax(logits)
    return int(rng.choice(sm.n_actions, p=probs))


def _decide_per_category(
    sm_c: ScoringMatrix,
    factors: np.ndarray,
    rng: np.random.Generator,
    n_actions: int,
    temperature: float,
) -> int:
    """
    Score using a category-specific ScoringMatrix.
    Computes softmax(f @ W_c.T / tau) and samples stochastically.

    Returns
    -------
    action_idx : int
    """
    f = factors.flatten()
    logits = f @ sm_c.W.T / temperature
    probs = softmax(logits)
    return int(rng.choice(n_actions, p=probs))


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------

def _run_one(
    profile_set_name: str,
    profiles: dict,
    gt_distributions: dict,
    scoring_config_name: str,
    seed: int,
    cfg: dict,
    bc: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Run one experiment condition (profile_set × scoring_config × seed).

    Parameters
    ----------
    profile_set_name : str   "simplified" or "realistic"
    profiles         : dict  action_conditional_profiles for this profile set
    gt_distributions : dict  category_gt_distributions for this profile set
    scoring_config_name : str  "w_only" | "w_g_static" | "w_g_learned"
    seed             : int
    cfg              : dict  expA section from default.yaml
    bc               : dict  bridge_common section from default.yaml

    Returns
    -------
    trajectory_rows : list[dict]   checkpoint records for accuracy_trajectories.csv
    g_matrix_rows   : list[dict]   end-of-run G snapshot for g_matrices.csv
                                   (empty for w_only, one row otherwise)
    """
    n_decisions   = int(cfg["n_decisions"])
    checkpoints   = set(cfg["checkpoints"])
    n_factors     = int(bc["n_factors"])
    n_categories  = int(bc["n_categories"])
    n_actions     = int(bc["n_actions"])
    categories    = bc["categories"]
    factor_names  = bc["factors"]
    gating_cfg    = cfg["gating"]
    w_init        = cfg.get("w_init", "uniform")

    # --- Alert generator for decisions ---
    gen_dec = CategoryAlertGenerator(
        categories=bc["categories"],
        actions=bc["actions"],
        factors=bc["factors"],
        action_conditional_profiles=profiles,
        gt_distributions=gt_distributions,
        factor_sigma=float(bc["factor_sigma"]),
        noise_rate=0.0,
        seed=seed,
    )
    decision_alerts = gen_dec.generate(n_decisions)

    # --- Scoring matrix (uniform init — no warm-start) ---
    sm = _build_sm(bc, cfg, w_init)

    # --- Per-category matrices (populated only for w_per_category) ---
    category_matrices: dict = {}

    # --- Gating ---
    if scoring_config_name == "w_only":
        gating = UniformGating(n_categories, n_factors)

    elif scoring_config_name == "w_g_static":
        fit_n     = int(gating_cfg["mi"]["fit_decisions"])
        threshold = float(gating_cfg["mi"]["threshold"])
        gating    = MIGating(n_categories, n_factors, threshold=threshold)
        gen_fit = CategoryAlertGenerator(
            categories=bc["categories"],
            actions=bc["actions"],
            factors=bc["factors"],
            action_conditional_profiles=profiles,
            gt_distributions=gt_distributions,
            factor_sigma=float(bc["factor_sigma"]),
            noise_rate=0.0,
            seed=seed + 1000,
        )
        fit_alerts = gen_fit.generate(fit_n)
        gating.fit_from_data(fit_alerts, n_actions=int(bc["n_actions"]))
        if seed == 42:
            print(f"    MI-static G (seed={seed}, profile={profile_set_name}):")
            for c_idx in range(n_categories):
                cat_name = categories[c_idx]
                gate_c   = gating.get_gate(c_idx)
                vals     = " ".join(f"{v:.2f}" for v in gate_c)
                print(f"      {cat_name}: [{vals}]")

    elif scoring_config_name == "w_g_learned":
        lr      = float(gating_cfg["hebbian"]["learning_rate"])
        dmp     = bool(gating_cfg["hebbian"]["damping"])
        init_g  = float(gating_cfg["hebbian"].get("initial_g", 1.0))
        max_g   = float(gating_cfg["hebbian"].get("max_gate", 1.0))
        gating  = HebbianGating(n_categories, n_factors, learning_rate=lr,
                                damping=dmp, max_gate=max_g)
        if init_g != 1.0:
            gating.G[:] = init_g   # start below max_gate so both +/- outcomes move G

    elif scoring_config_name == "w_augmented":
        gating = UniformGating(n_categories, n_factors)  # interface placeholder; not used

    elif scoring_config_name == "w_per_category":
        gating = UniformGating(n_categories, n_factors)  # interface placeholder; not used
        sc_override = cfg.get("scoring", bc["scoring"])
        for c_idx in range(n_categories):
            category_matrices[c_idx] = ScoringMatrix(
                n_actions=n_actions,
                n_factors=n_factors,
                temperature=float(sc_override["temperature"]),
                alpha_correct=float(sc_override["alpha_correct"]),
                alpha_incorrect=float(sc_override["alpha_incorrect"]),
                weight_clamp=float(sc_override["weight_clamp"]),
                decay_rate=float(sc_override["decay_rate"]),
                init_method=w_init,
            )

    else:
        raise ValueError(f"Unknown scoring_config_name: {scoring_config_name!r}")

    # --- Policy RNG (dedicated, seeded independently of alert RNG) ---
    rng_policy = np.random.default_rng(seed + 3000)

    # --- Decision loop ---
    n_correct   = 0
    cat_correct = [0] * n_categories
    cat_total   = [0] * n_categories
    trajectory_rows: list[dict] = []

    for t, alert in enumerate(decision_alerts):
        gate = gating.get_gate(alert.category_index)
        if scoring_config_name == "w_per_category":
            sm_c = category_matrices[alert.category_index]
            action_idx = _decide_per_category(
                sm_c, alert.factors, rng_policy, n_actions,
                float(cfg.get("scoring", bc["scoring"])["temperature"]),
            )
        elif scoring_config_name == "w_augmented":
            action_idx = _decide_augmented(
                sm, alert.factors, alert.category_index, n_categories, rng_policy
            )
        else:
            action_idx = _decide_stochastic(sm, alert.factors, gate, rng_policy)

        correct = (action_idx == alert.gt_action_index)
        outcome = 1 if correct else -1
        if correct:
            n_correct += 1
        cat_correct[alert.category_index] += int(correct)
        cat_total[alert.category_index]   += 1

        # Update W
        if scoring_config_name == "w_per_category":
            sm_c = category_matrices[alert.category_index]
            f = alert.factors.flatten()
            if outcome > 0:
                delta = sm_c.alpha_correct * f
            else:
                delta = -sm_c.alpha_incorrect * f
            sm_c.W[action_idx, :] += delta
            sm_c.W = np.clip(sm_c.W, -sm_c.weight_clamp, sm_c.weight_clamp)
        elif scoring_config_name == "w_augmented":
            sm.update_augmented(
                alert.factors, alert.category_index, n_categories,
                action_idx, outcome_positive=(outcome > 0),
            )
        else:
            sm.update_with_gated_factors(action_idx, outcome, alert.factors, gate, t + 1)

        # Update gating
        if scoring_config_name == "w_g_learned":
            gating.update(alert.category_index, alert.factors, action_idx, outcome, sm.W)
        elif scoring_config_name in ("w_only", "w_g_static", "w_augmented"):
            gating.update()     # no-op for UniformGating and MIGating
        # w_per_category: no gating update needed

        # Record at checkpoints
        step = t + 1
        if step in checkpoints:
            cat_accs = [
                cat_correct[c] / max(cat_total[c], 1)
                for c in range(n_categories)
            ]
            trajectory_rows.append({
                "profile_set":       profile_set_name,
                "scoring_config":    scoring_config_name,
                "seed":              seed,
                "checkpoint":        step,
                "cumulative_gt_acc": n_correct / step,
                "gt_acc_credential": cat_accs[0],
                "gt_acc_threat":     cat_accs[1],
                "gt_acc_lateral":    cat_accs[2],
                "gt_acc_exfil":      cat_accs[3],
                "gt_acc_insider":    cat_accs[4],
            })

    # --- G matrix snapshot at end-of-run (one row per condition) ---
    g_matrix_rows: list[dict] = []
    if scoring_config_name in ("w_g_static", "w_g_learned"):
        row: dict = {
            "profile_set":    profile_set_name,
            "scoring_config": scoring_config_name,
            "seed":           seed,
        }
        for c_idx, cat in enumerate(categories):
            gate_c = gating.get_gate(c_idx)
            for f_idx, fac in enumerate(factor_names):
                col = f"g_{cat}_{fac}"
                row[col] = float(gate_c[f_idx])
        g_matrix_rows.append(row)

    return trajectory_rows, g_matrix_rows


# ---------------------------------------------------------------------------
# Validation reporting
# ---------------------------------------------------------------------------

def _print_validation(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("EXP-A: VALIDATION SUMMARY")
    print("=" * 60)
    for key, gate_spec in summary["gates"].items():
        val, threshold, direction, desc = gate_spec
        if direction == "gt":
            passed  = val > threshold
            cmp_str = f"{val:.4f} > {threshold}"
        else:
            passed  = val < threshold
            cmp_str = f"{val:.4f} < {threshold}"
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {desc} [{cmp_str}] -> {status}")
    overall = "PASS" if summary["overall_pass"] else "FAIL"
    print(f"\n  OVERALL GATE: {overall}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_cfg = _load_config()
    bc  = raw_cfg["bridge_common"]
    cfg = raw_cfg["expA"]
    rp  = raw_cfg["realistic_profiles"]

    seeds     = bc["seeds"][: int(cfg["n_seeds"])]
    checkpoints = cfg["checkpoints"]

    sp = raw_cfg["simplified_profiles"]   # truly orthogonal, uniform across categories

    PROFILE_SETS = {
        "simplified": {
            "profiles":         sp["action_conditional_profiles"],
            "gt_distributions": sp["category_gt_distributions"],
        },
        "realistic": {
            "profiles":         rp["action_conditional_profiles"],
            "gt_distributions": rp["category_gt_distributions"],
        },
    }
    SCORING_CONFIGS = ["w_only", "w_g_static", "w_g_learned", "w_augmented", "w_per_category"]

    all_trajectory: list[dict] = []
    all_g_matrices: list[dict] = []

    total   = len(PROFILE_SETS) * len(SCORING_CONFIGS) * len(seeds)
    run_idx = 0

    for ps_name, ps_data in PROFILE_SETS.items():
        for sc_name in SCORING_CONFIGS:
            for seed in seeds:
                run_idx += 1
                print(
                    f"  [{run_idx:3d}/{total}]  "
                    f"profile={ps_name:<12}  config={sc_name:<14}  seed={seed}"
                )
                traj, gmats = _run_one(
                    ps_name,
                    ps_data["profiles"],
                    ps_data["gt_distributions"],
                    sc_name,
                    seed,
                    cfg,
                    bc,
                )
                all_trajectory.extend(traj)
                all_g_matrices.extend(gmats)

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    traj_df = pd.DataFrame(all_trajectory)
    traj_path = RESULTS_DIR / "accuracy_trajectories.csv"
    traj_df.to_csv(traj_path, index=False)
    print(f"\n  Saved: {traj_path}  ({len(traj_df)} rows)")

    if all_g_matrices:
        g_df   = pd.DataFrame(all_g_matrices)
        g_path = RESULTS_DIR / "g_matrices.csv"
        g_df.to_csv(g_path, index=False)
        print(f"  Saved: {g_path}  ({len(g_df)} rows)")

    # --- Validation gates ---
    final_cp = max(checkpoints)
    final_df = traj_df[traj_df["checkpoint"] == final_cp]

    def _mean_acc(ps: str, sc: str) -> float:
        sub = final_df[
            (final_df["profile_set"] == ps) & (final_df["scoring_config"] == sc)
        ]
        return float(sub["cumulative_gt_acc"].mean()) if not sub.empty else 0.0

    va1            = _mean_acc("simplified",  "w_only")
    va3            = _mean_acc("realistic",   "w_only")
    va4            = _mean_acc("realistic",   "w_g_static")
    va5            = _mean_acc("realistic",   "w_g_learned")
    va9            = _mean_acc("realistic",   "w_augmented")
    va10           = _mean_acc("realistic",   "w_per_category")
    va10_simplified = _mean_acc("simplified", "w_per_category")
    va6_delta = max(va4, va5, va9, va10) - va3

    # Per-category accuracy for the best non-baseline config on realistic
    best_g_config = max(
        [("w_g_static", va4), ("w_g_learned", va5),
         ("w_augmented", va9), ("w_per_category", va10)],
        key=lambda x: x[1],
    )[0]
    best_g_sub = final_df[
        (final_df["profile_set"] == "realistic") &
        (final_df["scoring_config"] == best_g_config)
    ]
    cat_cols = ["gt_acc_credential", "gt_acc_threat", "gt_acc_lateral",
                "gt_acc_exfil", "gt_acc_insider"]
    cat_means = {col: float(best_g_sub[col].mean()) for col in cat_cols}

    # Learned G top-2 factors per category (from g_matrices data)
    g_top2: dict = {}
    if all_g_matrices:
        g_df = pd.DataFrame(all_g_matrices)
        g_learned_realistic = g_df[
            (g_df["profile_set"] == "realistic") &
            (g_df["scoring_config"] == "w_g_learned")
        ]
        if not g_learned_realistic.empty:
            for cat in bc["categories"]:
                factor_means_cat: dict = {}
                for fac in bc["factors"]:
                    col = f"g_{cat}_{fac}"
                    if col in g_learned_realistic.columns:
                        factor_means_cat[fac] = float(g_learned_realistic[col].mean())
                if factor_means_cat:
                    sorted_facs = sorted(
                        factor_means_cat.items(), key=lambda x: x[1], reverse=True
                    )
                    g_top2[cat] = sorted_facs[:2]

    # Gate decision (three-tier)
    if va6_delta >= 0.15:
        gate_decision = "PASS"
    elif va6_delta >= 0.08:
        gate_decision = "MARGINAL"
    else:
        gate_decision = "FAIL"

    overall_pass = (va6_delta >= 0.15)

    # Print validation report
    print("\n" + "=" * 60)
    print("EXP-A: VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  VA.1: Simplified W-only acc     = {va1:.4f}  (threshold > 0.80)  {'PASS' if va1 > 0.80 else 'FAIL'}")
    print(f"  VA.3: Realistic W-only acc      = {va3:.4f}  (threshold < 0.55)  {'PASS' if va3 < 0.55 else 'FAIL'}")
    print(f"  VA.4: Realistic W+G-static acc  = {va4:.4f}")
    print(f"  VA.5: Realistic W+G-learned acc = {va5:.4f}")
    print(f"  VA.9: Realistic W-augmented acc = {va9:.4f}")
    print(f"  VA.10: Realistic W-per-category  = {va10:.4f}")
    print(f"  VA.10s: Simplified W-per-category = {va10_simplified:.4f}")
    print(f"  VA.6: G-lift delta              = {va6_delta:.4f}  (>= 0.15 PASS, >= 0.08 MARGINAL)")
    print(f"        Gate decision: {gate_decision}")
    print(f"  VA.7: Per-category (best G-config = {best_g_config}):")
    for col in cat_cols:
        cat_name = col.replace("gt_acc_", "")
        print(f"        {cat_name}: {cat_means[col]:.4f}")
    print(f"  VA.8: Learned G top-2 factors per category:")
    for cat, top2 in g_top2.items():
        factors_str = ", ".join(f"{f}({v:.2f})" for f, v in top2)
        print(f"        {cat}: {factors_str}")

    # VA.10 per-category breakdown
    pc_sub = final_df[
        (final_df["profile_set"] == "realistic") &
        (final_df["scoring_config"] == "w_per_category")
    ]
    if not pc_sub.empty:
        print(f"  VA.10 per-category breakdown:")
        for col in cat_cols:
            cat_name = col.replace("gt_acc_", "")
            print(f"        {cat_name}: {float(pc_sub[col].mean()):.4f}")

    if va10 > 0.70:
        print(f"  >>> Per-category W BREAKS the ceiling! Lift = +{(va10 - va3) * 100:.1f}pp")
        print(f"  >>> Multi-head scoring validated. Shared W is the bottleneck.")
    elif va10 > 0.55:
        print(f"  >>> Per-category W helps (+{(va10 - va3) * 100:.1f}pp) but ceiling not fully broken.")
        print(f"  >>> May need more decisions per category or different learning rule.")
    else:
        print(f"  >>> Per-category W also fails. Problem is deeper than W capacity.")
        print(f"  >>> Run EXP-C1 (centroid oracle) to diagnose.")

    print(f"\n  OVERALL GATE: {gate_decision}")
    print("=" * 60)

    # Save summary
    summary = {
        "overall_pass":            overall_pass,
        "gate_decision":           gate_decision,
        "final_checkpoint":        final_cp,
        "va1_simplified_w_only":   va1,
        "va3_realistic_w_only":    va3,
        "va4_realistic_g_static":  va4,
        "va5_realistic_g_learned": va5,
        "va9_augmented":              va9,
        "va10_per_category":          va10,
        "va10_simplified_per_category": va10_simplified,
        "va6_delta":               va6_delta,
        "best_g_config":           best_g_config,
        "cat_means":               cat_means,
        "g_top2":                  {cat: [[f, v] for f, v in top2]
                                    for cat, top2 in g_top2.items()},
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
