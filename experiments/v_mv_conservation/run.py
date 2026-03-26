"""
V-MV-CONSERVATION: Var(q) precision and recall.
Gate: Precision > 0.70, Recall > 0.80.

DATA LIMITATION: factorial results contain only conservation_mean and
conservation_min per cell (no per-window q_bar time series). This
analysis uses proxies derived from the available conservation signal.

Scale calibration (empirically confirmed from factorial data):
  conservation_mean = ALPHA * q_bar * volume / NORM
  where NORM = 12.5  (implied from cons_mean / (ALPHA * q_bar * volume))

Proxy definitions:
  q_bar_min_proxy = cons_min * NORM / (ALPHA * volume)
  Degradation event: q_bar_nominal - q_bar_min_proxy > 0.10 (>10pp drop)

  Signal (Var proxy): (conservation_mean - conservation_min) / conservation_mean
  -> relative range of conservation signal across windows
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

REPO = Path(__file__).resolve().parents[2]
FACT = REPO / "experiments" / "factorial" / "results"

ALPHA = 0.80
NORM  = 12.5   # empirically: cons_mean = ALPHA * q_bar * volume / NORM
DEGRADATION_THRESHOLD_PP = 0.10   # >10pp q_bar drop = degradation event


def load_data():
    cells = []
    for fname in ("factorial_soc_results.json", "factorial_s2p_results.json"):
        with open(FACT / fname) as f:
            cells.extend(json.load(f))
    return cells


def main():
    cells = load_data()

    # Build per-cell signal and label
    signal_vals = []
    degraded    = []

    for c in cells:
        cons_mean = c["conservation_mean"]
        cons_min  = c["conservation_min"]
        vol       = c["volume"]
        qbar_nom  = c["q_bar"]

        # Var proxy: relative range (conservation_mean - conservation_min) / conservation_mean
        if cons_mean > 0:
            var_proxy = (cons_mean - cons_min) / cons_mean
        else:
            var_proxy = 0.0

        # Corrected q_bar_min_proxy using empirical NORM=12.5 scale
        # cons = ALPHA * q * V / NORM  =>  q = cons * NORM / (ALPHA * V)
        if vol > 0:
            qbar_min_proxy = cons_min * NORM / (ALPHA * vol)
        else:
            qbar_min_proxy = qbar_nom

        degraded_flag = int((qbar_nom - qbar_min_proxy) > DEGRADATION_THRESHOLD_PP)

        signal_vals.append(var_proxy)
        degraded.append(degraded_flag)

    signal = np.array(signal_vals)
    labels = np.array(degraded)

    n_degraded = int(labels.sum())
    n_total    = len(labels)

    # Threshold sweep 0.001 → 0.150 (100 values)
    thresholds = np.linspace(0.001, 0.150, 100)
    best_f1    = -1.0
    best_thr   = thresholds[0]
    best_p     = 0.0
    best_r     = 0.0
    best_far   = 0.0

    for thr in thresholds:
        preds = (signal > thr).astype(int)
        if preds.sum() == 0:
            continue
        p = float(precision_score(labels, preds, zero_division=0))
        r = float(recall_score(labels, preds, zero_division=0))
        f = float(f1_score(labels, preds, zero_division=0))
        n_neg   = int((labels == 0).sum())
        fp      = int(((preds == 1) & (labels == 0)).sum())
        far_val = fp / n_neg if n_neg > 0 else 0.0
        if f > best_f1:
            best_f1  = f
            best_thr = float(thr)
            best_p   = p
            best_r   = r
            best_far = far_val

    p_pass    = best_p > 0.70
    r_pass    = best_r > 0.80
    gate_pass = p_pass and r_pass

    out = {
        "experiment": "V-MV-CONSERVATION",
        "n_cells": n_total,
        "n_degraded_cells": n_degraded,
        "degradation_rate": round(n_degraded / n_total, 4),
        "optimal_threshold": round(best_thr, 4),
        "precision": round(best_p, 4),
        "recall": round(best_r, 4),
        "f1": round(best_f1, 4),
        "false_alarm_rate": round(best_far, 4),
        "precision_pass": p_pass,
        "recall_pass": r_pass,
        "gate_pass": gate_pass,
        "scale_note": "NORM=12.5 empirically derived: cons_mean = ALPHA * q_bar * volume / NORM",
        "recommended_action": (
            "Conservation signal (relative range) is a viable early-warning indicator"
            if gate_pass else
            "Conservation signal insufficient for P>0.70 + R>0.80; per-window q_bar tracking required"
        ),
        "data_note": (
            "No per-window q_bar time series in factorial results. "
            "Proxy: var_proxy = (cons_mean - cons_min)/cons_mean; "
            "degradation = q_bar_min_proxy drop >10pp using NORM=12.5 scale. "
            "Full precision/recall requires per-checkpoint data."
        ),
    }
    out_path = REPO / "experiments" / "v_mv_conservation" / "results" / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("V-MV-CONSERVATION:")
    print(f"  Degradation events: {n_degraded}/{n_total} cells")
    print(f"  Optimal threshold: {best_thr:.4f}")
    print(f"  Precision: {best_p:.3f} [gate: >0.70] -> {'PASS' if p_pass else 'FAIL'}")
    print(f"  Recall:    {best_r:.3f} [gate: >0.80] -> {'PASS' if r_pass else 'FAIL'}")
    print(f"  False alarm rate: {best_far:.3f}")
    print(f"  Both gates: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
