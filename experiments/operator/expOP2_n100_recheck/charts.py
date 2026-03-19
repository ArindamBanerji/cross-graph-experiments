"""
EXP-OP2-N100-RECHECK Charts — identical layout to expOP2_recheck/charts.py.
Imports make_charts from sibling directory to avoid code duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np

# Re-use chart code from expOP2_recheck
from experiments.operator.expOP2_recheck.charts import make_charts

CONDITIONS_ALL = ["A", "B", "B-exp", "C", "C-exp", "P-75", "P-50", "P-25", "P-0"]

if __name__ == "__main__":
    _out = Path(__file__).parent
    with open(_out / "results.json") as f:
        _results = json.load(f)
    _curves = np.load(str(_out / "acc_curves.npy"), allow_pickle=True).item()
    _t_recs = np.load(str(_out / "t_recovery.npy"), allow_pickle=True).item()
    _n      = len(list(_t_recs.values())[0])
    make_charts(_results, _curves, _t_recs, _n, tag="expOP2n_recheck")
