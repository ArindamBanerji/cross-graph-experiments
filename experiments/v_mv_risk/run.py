"""
V-MV-RISK: Fit continuous R score on factorial data.
Gate: AUC > 0.85.

Data source: experiments/factorial/results/factorial_soc_results.json +
             experiments/factorial/results/factorial_s2p_results.json
Features: sigma_eff, volume, q_bar, kernel (encoded), trajectory_slope
Target: success = day60_accuracy >= 0.80
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
FACT = REPO / "experiments" / "factorial" / "results"


def load_data():
    cells = []
    for fname in ("factorial_soc_results.json", "factorial_s2p_results.json"):
        with open(FACT / fname) as f:
            cells.extend(json.load(f))
    return cells


def encode_kernel(kt: str) -> float:
    return {"l2": 0.0, "diagonal": 1.0, "shrinkage": 0.5}[kt]


def make_features(cells):
    rows = []
    labels = []
    for c in cells:
        d1    = c["day1_accuracy"]
        d30   = c["day30_accuracy"]
        d60   = c["day60_accuracy"]
        slope = (d30 - d1) / 29.0      # improvement rate days 1→30
        pct_conv = (c["cats_converged"] / c["cats_total"]
                    if c["cats_total"] > 0 else 0.0)
        rows.append([
            c["sigma_eff"],
            c["volume"],
            c["q_bar"],
            encode_kernel(c["kernel_type"]),
            slope,
            c["rho_max"],
            pct_conv,
        ])
        labels.append(1 if d60 >= 0.80 else 0)
    return np.array(rows, dtype=float), np.array(labels, dtype=int)


def discrete_gate_score(cells):
    scores = []
    for c in cells:
        a = c["day60_accuracy"]
        scores.append(1.0 if a >= 0.85 else (0.5 if a >= 0.70 else 0.0))
    return np.array(scores)


def main():
    cells = load_data()
    X, y = make_features(cells)
    feature_names = ["sigma_eff", "volume", "q_bar", "kernel", "slope", "rho_max", "pct_converged"]

    # Standardise for comparable coefficients
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xs, y)
    proba = lr.predict_proba(Xs)[:, 1]

    auc_r    = float(roc_auc_score(y, proba))
    auc_gate = float(roc_auc_score(y, discrete_gate_score(cells)))

    coeff_raw = lr.coef_[0]
    coeff_dict = {n: round(float(v), 4) for n, v in zip(feature_names, coeff_raw)}

    # Top features by |coefficient|
    order     = sorted(feature_names, key=lambda n: abs(coeff_dict[n]), reverse=True)
    gate_pass = auc_r > 0.85

    out = {
        "experiment": "V-MV-RISK",
        "n_cells": len(cells),
        "success_rate": round(float(y.mean()), 4),
        "auc_r_score": round(auc_r, 4),
        "auc_discrete_gate": round(auc_gate, 4),
        "coefficients": coeff_dict,
        "top_features": order,
        "gate_pass": gate_pass,
        "data_source": "factorial_soc_results + factorial_s2p_results",
        "data_note": "trajectory_slope = (day30-day1)/29; final_accuracy=day60; no per-decision trajectories in factorial data",
    }
    out_path = REPO / "experiments" / "v_mv_risk" / "results" / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("V-MV-RISK:")
    print(f"  AUC R score:       {auc_r:.3f} [gate: >0.85] -> {'PASS' if gate_pass else 'FAIL'}")
    print(f"  AUC discrete gate: {auc_gate:.3f}")
    print(f"  Top features: {order}")
    print(f"  Coefficients: {coeff_dict}")


if __name__ == "__main__":
    main()
