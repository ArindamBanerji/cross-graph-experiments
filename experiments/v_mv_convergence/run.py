"""
V-MV-CONVERGENCE: Fit N_half = f(sigma, V, q_bar, kernel). R² > 0.80 required.

N_half estimation via linear interpolation between day1/day30/day60 snapshots.
N_half = day at which accuracy first reaches (day1 + day60) / 2.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
FACT = REPO / "experiments" / "factorial" / "results"

ALPHA = 0.80


def load_data():
    cells = []
    for fname in ("factorial_soc_results.json", "factorial_s2p_results.json"):
        with open(FACT / fname) as f:
            cells.extend(json.load(f))
    return cells


def encode_kernel(kt: str) -> float:
    return {"l2": 0.0, "diagonal": 1.0, "shrinkage": 0.5}[kt]


def compute_n_half_days(c: dict) -> float:
    """Interpolate N_half in days from day1/day30/day60 snapshots."""
    d1, d30, d60 = c["day1_accuracy"], c["day30_accuracy"], c["day60_accuracy"]
    target = (d1 + d60) / 2.0
    # Already past midpoint at day 1?
    if d1 >= target:
        return 1.0
    # Midpoint reached between day1 and day30?
    if d30 >= target and d30 > d1:
        t = 1.0 + 29.0 * (target - d1) / (d30 - d1)
        return float(t)
    # Midpoint between day30 and day60?
    if d60 >= target and d60 > d30:
        t = 30.0 + 30.0 * (target - d30) / (d60 - d30)
        return float(t)
    # Never reached (censored)
    return 60.0


def main():
    cells = load_data()

    n_half_days = np.array([compute_n_half_days(c) for c in cells])

    # Features
    sigma  = np.array([c["sigma_eff"]   for c in cells])
    volume = np.array([c["volume"]       for c in cells])
    qbar   = np.array([c["q_bar"]        for c in cells])
    kernel = np.array([encode_kernel(c["kernel_type"]) for c in cells])
    rho    = np.array([c["rho_max"]      for c in cells])

    # Build design matrix with interactions
    X = np.column_stack([
        sigma,
        volume,
        qbar,
        kernel,
        rho,
        sigma * volume,        # sigma × V
        sigma * kernel,        # sigma × kernel
        qbar  * volume,        # q_bar × V
    ])
    feature_names = ["sigma", "V", "q_bar", "kernel", "rho",
                     "sigma_x_V", "sigma_x_kernel", "q_bar_x_V"]

    lr = LinearRegression()
    lr.fit(X, n_half_days)
    pred = lr.predict(X)
    r2   = float(r2_score(n_half_days, pred))
    resid_std = float(np.std(n_half_days - pred))

    coeff_dict = {n: round(float(v), 4) for n, v in zip(feature_names, lr.coef_)}
    coeff_dict["intercept"] = round(float(lr.intercept_), 4)

    # Sort by |coeff|
    top_coeff = sorted(feature_names, key=lambda n: abs(coeff_dict[n]), reverse=True)

    gate_pass = r2 > 0.80

    # Calendar predictions (extrapolate to unseen parameter values)
    def predict_profile(sig, vol, qb, kern_enc):
        rho_val = 0.0   # assume 0 for calendar predictions
        xp = np.array([[sig, vol, qb, kern_enc, rho_val,
                         sig * vol, sig * kern_enc, qb * vol]])
        n_half_d = float(lr.predict(xp)[0])
        n_half_d = max(1.0, n_half_d)
        n_half_decisions = round(n_half_d * vol)
        days_calendar = round(n_half_d, 1)
        return n_half_decisions, days_calendar

    cal = {}
    if gate_pass:
        a_dec, a_days = predict_profile(0.15, 200, 0.75, 1.0)   # Diagonal
        b_dec, b_days = predict_profile(0.10, 100, 0.85, 0.0)   # L2
        c_dec, c_days = predict_profile(0.20,  50, 0.70, 1.0)   # Diagonal
        cal = {
            "profile_A": {"sigma": 0.15, "V": 200, "q_bar": 0.75, "kernel": "diagonal",
                          "n_half_decisions": a_dec, "n_half_days": a_days},
            "profile_B": {"sigma": 0.10, "V": 100, "q_bar": 0.85, "kernel": "l2",
                          "n_half_decisions": b_dec, "n_half_days": b_days},
            "profile_C": {"sigma": 0.20, "V":  50, "q_bar": 0.70, "kernel": "diagonal",
                          "n_half_decisions": c_dec, "n_half_days": c_days},
        }

    out = {
        "experiment": "V-MV-CONVERGENCE",
        "n_cells": len(cells),
        "r_squared": round(r2, 4),
        "residual_std_days": round(resid_std, 3),
        "coefficients": coeff_dict,
        "top_coefficients": top_coeff,
        "gate_pass": gate_pass,
        "calendar_predictions": cal,
        "data_note": "N_half interpolated from day1/day30/day60 snapshots; V not in factorial → extrapolation for profile B(V=100)",
    }
    out_path = REPO / "experiments" / "v_mv_convergence" / "results" / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("V-MV-CONVERGENCE:")
    print(f"  R²: {r2:.3f} [gate: >0.80] -> {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Residual std: {resid_std:.2f} days")
    print(f"  Top coefficients: {top_coeff}")
    if gate_pass:
        a = cal["profile_A"]; b = cal["profile_B"]; c = cal["profile_C"]
        print(f"  Calendar predictions:")
        print(f"    Profile A (sigma=0.15,V=200,q_bar=0.75,Diagonal): N_half={a['n_half_decisions']}d, {a['n_half_days']}days")
        print(f"    Profile B (sigma=0.10,V=100,q_bar=0.85,L2):       N_half={b['n_half_decisions']}d, {b['n_half_days']}days")
        print(f"    Profile C (sigma=0.20,V=50, q_bar=0.70,Diagonal): N_half={c['n_half_decisions']}d, {c['n_half_days']}days")


if __name__ == "__main__":
    main()
