"""
V-SWITCHING-COST -- Centroid moat quantification. Pure analytical derivation.

Model:
  Per-cell convergence under exponential decay:
    ||mu(n) - mu_0|| = D_MAX * (1 - exp(-eta_eff * n))
    IKS(n) = 100 * (1 - exp(-eta_eff * n))

  Inverse (decisions to reach IKS target):
    n = -ln(1 - IKS/100) / eta_eff

  Days to IKS=67 given deployment profile (V, alpha):
    decisions_per_cell_per_day = V * alpha / N_CELLS
    days = n_67 / (V * alpha / N_CELLS)
         = n_67 * N_CELLS / (V * alpha)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Canonical parameters (do not change)
# ---------------------------------------------------------------------------
N_HALF      = 14               # decisions for IKS 0 -> 50
ETA_EFF     = math.log(2) / N_HALF   # = 0.04951 per decision
N_CELLS     = 24               # 6 categories x 4 actions
D_MAX       = 0.20
IKS_TARGETS = [25, 50, 67, 75, 90]

# ---------------------------------------------------------------------------
# Step 1 -- decisions per cell to reach each IKS target
# n = -ln(1 - IKS/100) / eta_eff
# ---------------------------------------------------------------------------
def decisions_to_iks(iks_target: float) -> float:
    """Decisions on a single cell required to reach IKS = iks_target."""
    return -math.log(1.0 - iks_target / 100.0) / ETA_EFF

n_by_iks = {iks: decisions_to_iks(iks) for iks in IKS_TARGETS}

# Board-level target
n_67 = n_by_iks[67]
total_decisions_67 = n_67 * N_CELLS   # across all tensor cells

# ---------------------------------------------------------------------------
# Step 2 -- days to IKS=67 by deployment profile
# days = n_67 * N_CELLS / (V * alpha)
# ---------------------------------------------------------------------------
def days_to_iks67(V: int, alpha: float) -> float:
    """Calendar days to reach IKS=67 given daily alert volume V and
    analyst override rate alpha (uniform category distribution)."""
    decisions_per_cell_per_day = V * alpha / N_CELLS
    return n_67 / decisions_per_cell_per_day

profiles = {
    "small_v50_a025":    (50,   0.25),
    "medium_v200_a025":  (200,  0.25),   # primary claim
    "large_v500_a025":   (500,  0.25),
    "quality_v200_a040": (200,  0.40),
}

days_by_profile = {name: days_to_iks67(V, a) for name, (V, a) in profiles.items()}

# ---------------------------------------------------------------------------
# Step 3 -- sensitivity
# ---------------------------------------------------------------------------
sensitivity = {
    "low_alpha_015_v200":    days_to_iks67(200,  0.15),
    "high_volume_v1000_a025": days_to_iks67(1000, 0.25),
}

# ---------------------------------------------------------------------------
# Step 4 -- claims (draft language for roadmap session)
# ---------------------------------------------------------------------------
primary_days   = days_by_profile["medium_v200_a025"]
primary_round  = round(primary_days, 1)
total_dec_int  = round(total_decisions_67)

headline_claim = (
    f"A medium SOC (200 alerts/day, 25% override rate) builds IKS=67 "
    f"in {primary_round} days — {total_dec_int} verified decisions across "
    f"{N_CELLS} tensor cells. Every verified decision deepens the moat: "
    f"a customer switching to a competitor restarts from IKS=0."
)

competitive_statement = (
    "Microsoft Security Copilot: population-level reasoning, no per-customer "
    "centroid tensor. IKS=0 at go-live, IKS=0 after one year of use. "
    "CrowdStrike Charlotte AI: organization-wide threat graph, not per-SOC "
    "decision geometry. IKS=0. "
    f"ACCP at IKS=67 after {primary_round} days: the organization's verified "
    "escalation geometry is embedded in 24 action-category centroids that no "
    "competitor can transfer. Switching cost = full {total_dec_int} decisions "
    "to rebuild — irreversible organizational IP.".replace(
        "{total_dec_int}", str(total_dec_int)
    )
)

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
out = {
    "model": {
        "n_half":       N_HALF,
        "eta_eff":      round(ETA_EFF, 6),
        "eta_eff_exact": "ln(2)/14",
        "tensor_cells": N_CELLS,
        "d_max":        D_MAX,
        "iks_formula":  "IKS(n) = 100 * (1 - exp(-eta_eff * n))",
        "inverse":      "n = -ln(1 - IKS/100) / eta_eff",
    },
    "decisions_to_iks": {
        f"iks_{iks}": round(n_by_iks[iks], 2)
        for iks in IKS_TARGETS
    },
    "total_decisions_to_iks_67_all_cells": round(total_decisions_67, 1),
    "days_to_iks_67_by_profile":  {k: round(v, 2) for k, v in days_by_profile.items()},
    "sensitivity":                 {k: round(v, 2) for k, v in sensitivity.items()},
    "headline_claim":              headline_claim,
    "competitive_statement":       competitive_statement,
}

out_path = REPO_ROOT / "experiments" / "v_switching_cost" / "switching_cost_analysis.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print("V-SWITCHING-COST -- Centroid Moat Quantification:")
print()
print("Decisions per cell to reach IKS target:")
for iks in IKS_TARGETS:
    marker = "  <- board-level target" if iks == 67 else ""
    print(f"  IKS={iks:2d}: {n_by_iks[iks]:6.2f} decisions{marker}")
print()
print("Days to reach IKS=67 by deployment profile:")
labels = {
    "small_v50_a025":    "Small  (V=50,  a=0.25)",
    "medium_v200_a025":  "Medium (V=200, a=0.25)",
    "large_v500_a025":   "Large  (V=500, a=0.25)",
    "quality_v200_a040": "Qual   (V=200, a=0.40)",
}
for key, label in labels.items():
    marker = "  <- primary claim" if key == "medium_v200_a025" else ""
    print(f"  {label}: {days_by_profile[key]:6.2f} days{marker}")
print()
print("Sensitivity:")
print(f"  Low override (a=0.15, V=200):  {sensitivity['low_alpha_015_v200']:6.2f} days")
print(f"  High volume  (V=1000, a=0.25): {sensitivity['high_volume_v1000_a025']:6.2f} days")
print()
print(f"Headline claim: {headline_claim}")
print()
print(f"Competitive statement: {competitive_statement}")
print()
print(f"Saved: {out_path}")
print("Raw numbers for roadmap session review.")
