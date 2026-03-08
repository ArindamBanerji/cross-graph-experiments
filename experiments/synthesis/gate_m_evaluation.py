"""
GATE-M Evaluation Script
experiments/synthesis/gate_m_evaluation.py

Reads results from EXP-S1 through S5b and produces the GATE-M go/no-go decision.
Run AFTER all of S1, S2, S3, S4 complete (plus optionally S5a/S5b for GATE-D-early).

GATE-M: Mathematical Validation (REQUIRED to proceed with Eq. 4-synthesis)
  S1: improvement >= 3pp AND p < 0.05 AND ECE degradation <= 0.02
  S2: poison_20pct_degradation <= 2pp AND safety_effectiveness >= 0.50
  S3: relative_frobenius <= 0.05 AND accuracy_diff <= 1pp
  S4: plateau_width >= 0.05

GATE-D-EARLY: Data Feasibility (INFORMATIONAL — does not block GATE-M)
  S5a: claims_extracted >= 5 AND sigma_nonzero >= 3 AND no_wrong_values
  S5b: llm_f1 >= 0.7 (or skipped) AND template_f1 >= 0.4 AND sigma_nonzero >= 4

GATE-M PASS → v5.0 proceeds with synthesis foundation (~135 lines in Phase 6)
GATE-M FAIL → diagnose cause, apply remedy, rerun — or accept display-only synthesis
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent.parent
SYNTHESIS_DIR = Path(__file__).parent

# Result file paths
RESULT_PATHS = {
    "S1": SYNTHESIS_DIR / "expS1_bias_accuracy"    / "results.json",
    "S2": SYNTHESIS_DIR / "expS2_poisoning"        / "results.json",
    "S3": SYNTHESIS_DIR / "expS3_loop_independence"/ "results.json",
    "S4": SYNTHESIS_DIR / "expS4_lambda_sensitivity"/ "results.json",
    "S5a": SYNTHESIS_DIR / "expS5a_real_threat_intel"/ "results.json",
    "S5b": SYNTHESIS_DIR / "expS5b_work_artifacts"  / "results.json",
}

GATE_PATH = SYNTHESIS_DIR / "gate_m_decision.json"

# Gate thresholds
THRESHOLDS = {
    "S1_improvement_pp":          3.0,
    "S1_p_value":                 0.05,
    "S1_ece_degradation":         0.02,
    "S2_poison_20pct_degradation_pp": 2.0,
    "S2_safety_effectiveness":    0.50,
    "S3_frobenius_relative":      0.05,
    "S3_accuracy_diff_pp":        1.0,
    "S4_plateau_width":           0.05,
    "S5a_claims_extracted":       5,
    "S5a_sigma_nonzero":          3,
    "S5b_llm_f1":                 0.7,
    "S5b_template_f1":            0.4,
    "S5b_sigma_nonzero":          4,
}


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_s1(data: dict) -> Tuple[bool, dict]:
    stat = data.get("statistical_test", {})
    gate = data.get("gate", {})
    improvement = gate.get("improvement_pp", 0.0)
    p_value      = gate.get("p_value", 1.0)
    ece_delta    = gate.get("ece_degradation", 999.0)
    best_lambda  = stat.get("best_lambda", None)
    passed = (
        improvement >= THRESHOLDS["S1_improvement_pp"] and
        p_value < THRESHOLDS["S1_p_value"] and
        ece_delta <= THRESHOLDS["S1_ece_degradation"]
    )
    return passed, {
        "improvement_pp": improvement,
        "p_value": p_value,
        "ece_degradation": ece_delta,
        "best_lambda": best_lambda,
        "passed": passed,
    }


def check_s2(data: dict) -> Tuple[bool, dict]:
    g = data.get("gate", {})
    deg = g.get("poison_20pct_degradation", 999.0)
    safety = g.get("safety_effectiveness", 0.0)
    passed = (
        deg <= THRESHOLDS["S2_poison_20pct_degradation_pp"] and
        safety >= THRESHOLDS["S2_safety_effectiveness"]
    )
    return passed, {
        "poison_20pct_degradation_pp": deg,
        "safety_effectiveness": safety,
        "passed": passed,
    }


def check_s3(data: dict) -> Tuple[bool, dict]:
    agg = data.get("aggregated", {})
    rel = agg.get("mean_relative_frobenius", 999.0)
    acc = agg.get("mean_accuracy_diff_pp", 999.0)
    passed = (
        rel <= THRESHOLDS["S3_frobenius_relative"] and
        acc <= THRESHOLDS["S3_accuracy_diff_pp"]
    )
    return passed, {
        "mean_relative_frobenius": rel,
        "mean_accuracy_diff_pp": acc,
        "passed": passed,
        "implication": (
            "No centroid contamination through indirect path — update() API unchanged"
            if passed
            else "FAIL: counterfactual-aware update mode needed — ProfileScorer.update() API may change"
        ),
    }


def check_s4(data: dict) -> Tuple[bool, dict]:
    g = data.get("gate", {})
    width = g.get("plateau_width", 0.0)
    lp = g.get("lambda_peak", None)
    peak_acc = g.get("peak_accuracy", 0.0)
    passed = width >= THRESHOLDS["S4_plateau_width"]
    return passed, {
        "plateau_width": width,
        "lambda_peak": lp,
        "peak_accuracy": peak_acc,
        "passed": passed,
        "implication": (
            f"Stable plateau exists — global λ={lp:.3f} is tunable"
            if passed
            else "FAIL: narrow spike — per-category λ[c] or adaptive coupling needed (changes SynthesisBias)"
        ),
    }


def check_s5a(data: dict) -> Tuple[Optional[bool], dict]:
    if data is None:
        return None, {"status": "not_run"}
    g = data.get("gate", {})
    claims = g.get("claims_extracted", 0)
    nonzero = g.get("sigma_nonzero", 0)
    no_wrong = g.get("no_wrong_values", False)
    passed = (
        claims >= THRESHOLDS["S5a_claims_extracted"] and
        nonzero >= THRESHOLDS["S5a_sigma_nonzero"] and
        no_wrong
    )
    return passed, {
        "claims_extracted": claims,
        "sigma_nonzero": nonzero,
        "no_wrong_values": no_wrong,
        "passed": passed,
        "status": "run",
    }


def check_s5b(data: dict) -> Tuple[Optional[bool], dict]:
    if data is None:
        return None, {"status": "not_run"}
    g = data.get("gate", {})
    llm_f1 = g.get("llm_f1", None)
    tmpl_f1 = g.get("template_f1", 0.0)
    nonzero = g.get("sigma_nonzero", 0)
    llm_ok = llm_f1 is None or llm_f1 >= THRESHOLDS["S5b_llm_f1"]  # None = skipped
    passed = (
        llm_ok and
        tmpl_f1 >= THRESHOLDS["S5b_template_f1"] and
        nonzero >= THRESHOLDS["S5b_sigma_nonzero"]
    )
    return passed, {
        "llm_f1": llm_f1,
        "template_f1": tmpl_f1,
        "sigma_nonzero": nonzero,
        "passed": passed,
        "llm_skipped": llm_f1 is None,
        "status": "run",
    }


def derive_recommended_lambda(s1_data: dict, s4_data: dict) -> Tuple[float, list]:
    """Derive recommended λ and safe range from S1 + S4 results."""
    best_from_s1 = None
    if s1_data:
        best_from_s1 = s1_data.get("statistical_test", {}).get("best_lambda")

    safe_range = [0.0, 0.0]
    if s4_data:
        plateau = s4_data.get("plateau", {})
        pl = plateau.get("plateau_lambdas", [])
        if pl:
            safe_range = [float(min(pl)), float(max(pl))]

    # Prefer the λ from S4 plateau center if available, else S1 best
    if safe_range[1] > safe_range[0]:
        recommended = round((safe_range[0] + safe_range[1]) / 2, 3)
    elif best_from_s1 is not None:
        recommended = float(best_from_s1)
    else:
        recommended = 0.1  # Default

    return recommended, safe_range


def main():
    print("Loading experiment results...")
    s1_data  = _load(RESULT_PATHS["S1"])
    s2_data  = _load(RESULT_PATHS["S2"])
    s3_data  = _load(RESULT_PATHS["S3"])
    s4_data  = _load(RESULT_PATHS["S4"])
    s5a_data = _load(RESULT_PATHS["S5a"])
    s5b_data = _load(RESULT_PATHS["S5b"])

    # Check required experiments
    missing = []
    for exp_id, data in [("S1", s1_data), ("S2", s2_data), ("S3", s3_data), ("S4", s4_data)]:
        if data is None:
            missing.append(exp_id)

    if missing:
        print(f"\n❌ CANNOT EVALUATE GATE-M: Missing results for: {missing}")
        print(f"Run: {', '.join(f'experiments/synthesis/exp{m}*/run.py' for m in missing)}")
        sys.exit(1)

    # Run checks
    s1_pass, s1_info = check_s1(s1_data)
    s2_pass, s2_info = check_s2(s2_data)
    s3_pass, s3_info = check_s3(s3_data)
    s4_pass, s4_info = check_s4(s4_data)
    s5a_pass, s5a_info = check_s5a(s5a_data)
    s5b_pass, s5b_info = check_s5b(s5b_data)

    gate_m_pass = s1_pass and s2_pass and s3_pass and s4_pass

    # Data feasibility
    if s5a_pass is None and s5b_pass is None:
        data_feasibility = "SKIPPED"
    elif s5a_pass is not None and s5b_pass is not None:
        data_feasibility = "PASS" if (s5a_pass and s5b_pass) else "FAIL"
    elif s5a_pass is not None:
        data_feasibility = "PASS" if s5a_pass else "FAIL"
    else:
        data_feasibility = "PASS" if s5b_pass else "FAIL"

    # Recommended lambda
    recommended_lambda, safe_range = derive_recommended_lambda(s1_data, s4_data)

    # ----------------------------------------------------------------
    # Print results table
    # ----------------------------------------------------------------
    W = 60
    def row(label, status, detail=""):
        status_str = "✅ PASS" if status else "❌ FAIL"
        line = f"║ {label:<28s} {status_str:<8s}  {detail:<17s} ║"
        print(line)

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "     GATE-M: MATHEMATICAL VALIDATION".center(W) + "║")
    print("╠" + "═" * W + "╣")

    row("EXP-S1 (accuracy)",
        s1_pass,
        f"+{s1_info['improvement_pp']:.2f}pp p={s1_info['p_value']:.4f}")
    row("EXP-S2 (poisoning)",
        s2_pass,
        f"-{s2_info['poison_20pct_degradation_pp']:.2f}pp")
    row("EXP-S3 (independence)",
        s3_pass,
        f"F={s3_info['mean_relative_frobenius']:.4f}")
    row("EXP-S4 (sensitivity)",
        s4_pass,
        f"width={s4_info['plateau_width']:.3f}")

    print("╠" + "═" * W + "╣")
    verdict_str = "✅ PASS" if gate_m_pass else "❌ FAIL"
    print(f"║ {'GATE-M DECISION:':<28s} {verdict_str:<10s}{'':>20s}║")
    print(f"║ {'Recommended λ:':<28s} {recommended_lambda:.3f}{'':>27s}║")
    print(f"║ {'Safe λ range:':<28s} [{safe_range[0]:.3f}, {safe_range[1]:.3f}]{'':>22s}║")

    print("╠" + "═" * W + "╣")
    print("║" + "  GATE-D-EARLY: DATA FEASIBILITY (informational)".center(W) + "║")
    print("╠" + "═" * W + "╣")

    if s5a_info["status"] == "not_run":
        print(f"║ {'EXP-S5a (real intel):':<28s} {'⏭ SKIPPED':<8s}{'':>22s}║")
    else:
        row("EXP-S5a (real intel)",
            s5a_pass,
            f"claims={s5a_info['claims_extracted']}")

    if s5b_info["status"] == "not_run":
        print(f"║ {'EXP-S5b (artifacts):':<28s} {'⏭ SKIPPED':<8s}{'':>22s}║")
    else:
        llm_str = "skipped" if s5b_info.get("llm_skipped") else f"F1={s5b_info.get('llm_f1', 0):.2f}"
        row("EXP-S5b (artifacts)",
            s5b_pass,
            llm_str)

    df_str = "✅" if data_feasibility == "PASS" else "❌" if data_feasibility == "FAIL" else "⏭"
    print(f"║ {'DATA FEASIBILITY:':<28s} {df_str} {data_feasibility:<35s}║")
    print("╚" + "═" * W + "╝")

    # ----------------------------------------------------------------
    # Remediation advice if FAIL
    # ----------------------------------------------------------------
    if not gate_m_pass:
        print("\n⚠  GATE-M FAILED. Diagnosis:")
        if not s1_pass:
            print(f"  S1 FAIL (improvement={s1_info['improvement_pp']:.2f}pp < 3pp or "
                  f"p={s1_info['p_value']:.4f} ≥ 0.05 or ECE={s1_info['ece_degradation']:.4f} > 0.02)")
            print("    → Options: (1) Redesign rule templates, rerun. "
                  "(2) Accept σ as display-only — Tab 5 briefing still ships.")
        if not s2_pass:
            print(f"  S2 FAIL (poison_deg={s2_info['poison_20pct_degradation_pp']:.2f}pp > 2pp or "
                  f"safety={s2_info['safety_effectiveness']:.3f} < 0.50)")
            print("    → Tighten sigma_max (try 0.5), raise extraction_threshold (try 0.9). Rerun.")
        if not s3_pass:
            print(f"  S3 FAIL (Frobenius={s3_info['mean_relative_frobenius']:.4f} > 0.05)")
            print("    → ARCHITECTURAL: implement counterfactual-aware update in ProfileScorer.update()")
            print("      This MUST be resolved before v5.0 Phase 1 (GAE-PROF-1)")
        if not s4_pass:
            print(f"  S4 FAIL (plateau_width={s4_info['plateau_width']:.3f} < 0.05)")
            print("    → ARCHITECTURAL: consider per-category λ[c] (changes SynthesisBias dataclass)")
            print("      This MUST be resolved before v5.0 Phase 6 (GAE-ENG-1)")
    else:
        print("\n✅  GATE-M PASSED.")
        print(f"  Recommended λ: {recommended_lambda:.3f}  Safe range: {safe_range}")
        print()
        print("  Next steps:")
        print("  1. Start v5.0 code sprint — Phase 1 (GAE-PROF-1)")
        print("  2. Synthesis foundation (~135 lines) lands in Phase 6 (GAE-ENG-1)")
        print("  3. Eq. 4-synthesis validated for v5.5+ implementation")

    # ----------------------------------------------------------------
    # Save decision JSON
    # ----------------------------------------------------------------
    decision = {
        "gate_m_decision": "PASS" if gate_m_pass else "FAIL",
        "recommended_lambda": recommended_lambda,
        "safe_range": safe_range,
        "per_experiment": {
            "S1": s1_info,
            "S2": s2_info,
            "S3": s3_info,
            "S4": s4_info,
        },
        "gate_d_early": {
            "S5a": s5a_info,
            "S5b": s5b_info,
            "feasibility": data_feasibility,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_PATH, "w") as f:
        json.dump(decision, f, indent=2, default=float)
    print(f"\nDecision saved to {GATE_PATH}")

    return gate_m_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
