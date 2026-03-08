"""
EXP-S5a: Real Threat Intel Pull — Display Coherence Validation
experiments/synthesis/expS5a_real_threat_intel/run.py

QUESTION: Do real CISA KEV + NVD claims flow through the synthesis pipeline
and produce operationally sensible sigma tensors and INTSUM briefing output?

Option B resolution (DI-02): sigma scoring does NOT ship. Lambda=0 always.
This experiment validates Tab 5 Panel A (INTSUM briefing display) only.

GATE-S5a (informational — does not block GATE-M):
  claims_extracted >= 5 AND sigma_nonzero_categories >= 3
  AND traces_present AND briefing_nonempty >= 1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.rule_projector import RuleBasedProjector
from src.viz.synthesis_common import save_results, print_gate_result

EXP_DIR       = Path(__file__).parent
RESULTS_PATH  = EXP_DIR / "results.json"
PAPER_FIG_DIR = REPO_ROOT / "paper_figures"

# Canonical SOC taxonomy — must match fetch_kev/fetch_nvd exactly
CATEGORIES: List[str] = [
    "travel_anomaly", "credential_access", "threat_intel_match",
    "insider_behavioral", "cloud_infrastructure",
]
ACTIONS: List[str] = ["escalate", "investigate", "suppress", "monitor"]

# Synthesis rules mapped to canonical action names
# Sign convention: negative → action MORE likely (closer in Eq.4-synthesis)
SOC_SYNTHESIS_RULES: Dict[str, Dict[str, float]] = {
    "cve_actively_exploited": {
        "escalate":    -0.3,
        "investigate": -0.2,
        "suppress":    +0.5,
        "monitor":      0.0,
    },
    "cve_published": {
        "escalate":    -0.1,
        "investigate": -0.2,
        "suppress":    +0.2,
        "monitor":      0.0,
    },
    "active_campaign": {
        "escalate":    -0.4,
        "investigate": -0.1,
        "suppress":    +0.6,
        "monitor":     +0.1,
    },
    "ciso_risk_directive": {
        "escalate":    -0.3,
        "investigate": -0.1,
        "suppress":    +0.4,
        "monitor":      0.0,
    },
    "vulnerability_patched": {
        "escalate":    +0.2,
        "investigate":  0.0,
        "suppress":    -0.1,
        "monitor":     -0.1,
    },
    "known_fp_pattern": {
        "escalate":    +0.3,
        "investigate": +0.1,
        "suppress":    -0.4,
        "monitor":      0.0,
    },
}

# Tab 5 Panel A — action names are already canonical, display as-is
ACTION_DISPLAY: Dict[str, str] = {a: a for a in ACTIONS}


# ---------------------------------------------------------------------------
# Briefing assembly
# ---------------------------------------------------------------------------

def generate_briefing_sentence(
    cat: str, action: str, sigma_val: float, n_claims: int
) -> str:
    """Produce one Tab 5 Panel A style INTSUM line."""
    display_action = ACTION_DISPLAY.get(action, action)
    if sigma_val < -0.1:
        verb = "strongly promote"
    elif sigma_val < -0.01:
        verb = "promote"
    elif sigma_val > 0.1:
        verb = "suppress"
    elif sigma_val > 0.01:
        verb = "weakly suppress"
    else:
        return ""
    cat_display = cat.replace("_", " ")
    return (
        f"[{cat_display}] {n_claims} claim(s) {verb} '{display_action}' "
        f"(sigma={sigma_val:+.3f})"
    )


def assemble_briefing(
    sigma: np.ndarray,
    trace: Dict,
    categories: List[str],
    actions: List[str],
) -> List[str]:
    """Collect all non-trivial briefing sentences, sorted by |sigma|."""
    lines = []
    for i, cat in enumerate(categories):
        for j, act in enumerate(actions):
            val = float(sigma[i, j])
            n_claims = len(trace.get(cat, {}).get(act, []))
            if abs(val) > 0.01 and n_claims > 0:
                line = generate_briefing_sentence(cat, act, val, n_claims)
                if line:
                    lines.append((abs(val), line))
    lines.sort(reverse=True, key=lambda x: x[0])
    return [l for _, l in lines]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    print("EXP-S5a: Real Threat Intel — Display Coherence Validation")
    print("Option B (DI-02): sigma does not ship. Lambda=0 in product. Display only.")
    print()

    # --- Section 1: Claim ingestion ---
    print("=== Section 1: Claim Ingestion ===")
    import experiments.synthesis.expS5a_real_threat_intel.fetch_kev as fetch_kev
    import experiments.synthesis.expS5a_real_threat_intel.fetch_nvd as fetch_nvd

    kev_claims = fetch_kev.run()
    nvd_claims = fetch_nvd.run()

    all_claims = kev_claims + nvd_claims
    print(f"\nTotal claims: {len(all_claims)}  ({len(kev_claims)} KEV + {len(nvd_claims)} NVD)")

    # Category distribution (bridge_common schema only)
    cat_dist = {cat: 0 for cat in CATEGORIES}
    for c in all_claims:
        for cat in c.get("categories_affected", []):
            if cat in cat_dist:
                cat_dist[cat] += 1

    # Claim type breakdown
    type_counts: Dict[str, int] = {}
    for c in all_claims:
        t = c.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nClaim type breakdown:")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:30s}: {cnt}")

    print(f"\nCategory distribution (bridge_common 5-category schema):")
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s}: {cnt}")

    urgencies = [float(c.get("urgency", 0.0)) for c in all_claims]
    if urgencies:
        print(
            f"\nUrgency stats: mean={np.mean(urgencies):.3f}  "
            f"min={np.min(urgencies):.3f}  max={np.max(urgencies):.3f}"
        )

    by_age = sorted(all_claims, key=lambda c: c.get("age_days", 99999))[:5]
    print(f"\nTop 5 most recent claims:")
    for c in by_age:
        print(
            f"  [{c.get('source','?'):10s}] {c.get('cve_id','?'):20s}  "
            f"age={c.get('age_days','?'):>4}d  "
            f"cats={c.get('categories_affected',[])}"
        )

    # --- Section 2: Sigma tensor ---
    print("\n=== Section 2: Sigma Tensor (lambda_coupling=1.0 for inspection) ===")
    projector = RuleBasedProjector(
        rules=SOC_SYNTHESIS_RULES,
        categories=CATEGORIES,
        actions=ACTIONS,
        extraction_threshold=0.8,
    )
    bias, trace = projector.project_with_trace(all_claims, lambda_coupling=1.0)

    print(f"\n  Active claims (passed extraction threshold): {bias.active_claims}")
    print(f"  sigma shape: {bias.sigma.shape}")

    print(f"\n  sigma tensor:")
    print(f"  {'':30s}" + "".join(f"{a:>18s}" for a in ACTIONS))
    nonzero_count = 0
    for i, cat in enumerate(CATEGORIES):
        row_vals = [float(bias.sigma[i, j]) for j in range(len(ACTIONS))]
        has_nonzero = any(abs(v) > 1e-6 for v in row_vals)
        if has_nonzero:
            nonzero_count += 1
        marker = "  ***" if has_nonzero else ""
        print(
            f"  {cat:30s}"
            + "".join(f"{v:>+18.4f}" for v in row_vals)
            + marker
        )

    # --- Section 3: Coherence traces ---
    print("\n=== Section 3: Coherence Traces ===")
    cells = []
    for i, cat in enumerate(CATEGORIES):
        for j, act in enumerate(ACTIONS):
            val = float(bias.sigma[i, j])
            if abs(val) > 1e-6:
                n_contrib = len(trace.get(cat, {}).get(act, []))
                cells.append((cat, act, val, n_contrib))
    cells.sort(key=lambda x: -abs(x[2]))

    traces_shown = min(3, len(cells))
    print(f"  (Showing top {traces_shown} cell(s) by |sigma|)")
    for cat, act, val, n_contrib in cells[:3]:
        display_act = ACTION_DISPLAY.get(act, act)
        print(
            f"\n  [{cat}] '{display_act}'  sigma={val:+.4f}  "
            f"({n_contrib} contributing claims):"
        )
        for entry in trace.get(cat, {}).get(act, [])[:3]:
            print(
                f"    type={entry['claim_type']:25s}  "
                f"w={entry['weight']:.4f}  "
                f"dir={entry['direction']:+.2f}  "
                f"contrib={entry['contribution']:+.4f}"
            )

    # --- Section 4: INTSUM briefing ---
    print("\n=== Section 4: Tab 5 Panel A — INTSUM Briefing ===")
    briefing_lines = assemble_briefing(bias.sigma, trace, CATEGORIES, ACTIONS)

    if briefing_lines:
        print(f"\n  {len(briefing_lines)} briefing line(s):")
        for line in briefing_lines:
            print(f"  {line}")
    else:
        print("  (no briefing lines generated)")

    # --- Gate checks ---
    gate_claims = len(all_claims) >= 5
    gate_sigma  = nonzero_count >= 3
    gate_traces = traces_shown >= 1
    gate_brief  = len(briefing_lines) >= 1

    # Sensibility: cve_actively_exploited claims should drive "escalate" NEGATIVE
    sensible = True
    escalate_idx = ACTIONS.index("escalate")
    for i, cat in enumerate(CATEGORIES):
        val = float(bias.sigma[i, escalate_idx])
        contribs = trace.get(cat, {}).get("escalate", [])
        has_kev = any(e["claim_type"] == "cve_actively_exploited" for e in contribs)
        if has_kev and val > 0.01:
            sensible = False
            print(
                f"  WARNING: sigma[{cat}, escalate]={val:+.4f} POSITIVE "
                f"despite KEV claims (wrong direction)"
            )

    gate_pass = gate_claims and gate_sigma and gate_traces and gate_brief

    # Saturation check: most cells at ±sigma_max?
    flat = bias.sigma.flatten()
    n_saturated = int(np.sum(np.abs(flat) >= 0.999))
    sigma_saturated = n_saturated >= (flat.size // 2)

    # --- Save results ---
    results = {
        "experiment": "EXP-S5a",
        "option_b_confirmed": True,
        "lambda_in_product": 0,
        "data_sources": {
            "kev_claims":             len(kev_claims),
            "nvd_claims":             len(nvd_claims),
            "total_claims":           len(all_claims),
            "active_after_threshold": bias.active_claims,
        },
        "claim_type_counts":       type_counts,
        "sigma_tensor":            bias.sigma.tolist(),
        "sigma_nonzero_categories": nonzero_count,
        "category_distribution":   cat_dist,
        "urgency_stats": {
            "mean":   float(np.mean(urgencies))   if urgencies else 0.0,
            "min":    float(np.min(urgencies))    if urgencies else 0.0,
            "max":    float(np.max(urgencies))    if urgencies else 0.0,
            "values": urgencies,
        },
        "briefing_lines":      briefing_lines,
        "traces_shown":        traces_shown,
        "sensible_direction":  sensible,
        "sigma_saturated":     sigma_saturated,
        "sigma_saturation_note": (
            "sigma_saturated=true, cause=447_cumulative_kev_claims, "
            "display_impact=none (saturation=strong_signal), "
            "scoring_impact=normalization_needed (future_research)"
        ) if sigma_saturated else "not_saturated",
        "gate": {
            "S5a_passed":           gate_pass,
            "claims_ge_5":          gate_claims,
            "sigma_nonzero_ge_3":   gate_sigma,
            "traces_present":       gate_traces,
            "briefing_nonempty":    gate_brief,
            "sensible_direction":   sensible,
            "note": "Informational gate — does not block GATE-M",
        },
    }

    save_results(results, str(RESULTS_PATH))
    print(f"\nResults saved to {RESULTS_PATH}")

    details = (
        f"claims={len(all_claims)} (need >=5) | "
        f"sigma_nonzero={nonzero_count} cats (need >=3) | "
        f"traces={traces_shown} (need >=1) | "
        f"briefing={len(briefing_lines)} lines (need >=1)"
    )
    print_gate_result("S5a (INFORMATIONAL)", gate_pass, details)

    # Generate charts
    try:
        import experiments.synthesis.expS5a_real_threat_intel.charts as charts
        charts.make_all_charts(results, str(EXP_DIR), str(PAPER_FIG_DIR))
        print("Charts saved.")
    except Exception as e:
        print(f"Warning: chart generation failed: {e}")

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
