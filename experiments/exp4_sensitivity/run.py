"""Experiment 4: Parameter sensitivity analysis."""
import csv, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.alert_generator import AlertGenerator, ACTION_NAMES
from src.data.entity_generator import EntityGenerator, inject_signals, _DOMAIN_PROFILES
from src.models.scoring_matrix import ScoringMatrix
from src.models.cross_attention import CrossGraphAttention

RESULTS_DIR   = Path(__file__).resolve().parent / "results"
ACTION_TO_IDX = {n: i for i, n in enumerate(ACTION_NAMES)}
SEEDS5        = [42, 123, 456, 789, 1024]
N_ALERTS      = 2000

def _score_trial(alerts, sm, seed):
    """Run sm on alerts; return final cumulative accuracy. seed fixes np.random for stochastic."""
    np.random.seed(seed)
    correct = 0
    for a in alerts:
        true_idx = ACTION_TO_IDX[a.ground_truth_action]
        greedy, _ = sm.decide(a.factors)
        correct  += greedy == true_idx
        stoch, _  = sm.decide_stochastic(a.factors)
        sm.update(a.factors, stoch, stoch == true_idx)
    return correct / len(alerts)

def _disc_f1(dim, n_entities, n_sig, sig_str, seed):
    """Generate security + threat_intel at *dim*, inject signals, run two_stage, return F1."""
    gen    = EntityGenerator({"experiment_2": {
        "embedding_dim":  dim,
        "domain_profiles": {k: _DOMAIN_PROFILES[k] for k in ("security", "threat_intel")},
    }})
    sec    = gen.generate_domain("security",     n_entities, seed)
    threat = gen.generate_domain("threat_intel", n_entities, seed + 1)
    gt     = inject_signals(sec, threat, n_sig, sig_str, seed + 2)
    E_i    = np.array([e.embedding for e in sec])
    E_j    = np.array([e.embedding for e in threat])
    hits   = CrossGraphAttention().discover_two_stage(E_i, E_j, 0.02, 3)
    disc   = {(h[0], h[1]) for h in hits}
    id2i   = {e.entity_id: k for k, e in enumerate(sec)}
    id2j   = {e.entity_id: k for k, e in enumerate(threat)}
    gt_set = {(id2i[a], id2j[b]) for a, b in gt}
    tp     = len(disc & gt_set)
    prec   = tp / len(disc)   if disc   else 0.0
    rec    = tp / len(gt_set) if gt_set else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def run_experiment():
    t0 = time.time()
    with open(ROOT / "configs" / "default.yaml") as fh:
        raw = yaml.safe_load(fh)
    cfg1, cfg4 = raw["experiment_1"], raw["experiment_4"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(sweep, param, val, seed, metric, v):
        rows.append(dict(sweep=sweep, param_name=param, param_value=val,
                         seed=seed, metric_name=metric, metric_value=round(v, 6)))

    base_sm = dict(n_actions=cfg1["n_actions"], n_factors=cfg1["n_factors"],
                   weight_clamp=cfg1["weight_clamp"], init_method=cfg1["init_method"],
                   decay_rate=float(cfg1["decay_rate"]))

    # Sweep A: asymmetry_ratio — alpha_correct=0.002, alpha_incorrect=ratio*0.002
    gen_noise = AlertGenerator({"noise_rate": cfg1["noise_rate"]})
    for ratio in cfg4["asymmetry_ratios"]:
        sm_kw = {**base_sm, "temperature": cfg1["temperature"],
                 "alpha_correct": 0.002, "alpha_incorrect": ratio * 0.002}
        for seed in SEEDS5:
            add("A_asymmetry", "asymmetry_ratio", ratio, seed, "accuracy",
                _score_trial(gen_noise.generate(N_ALERTS, seed), ScoringMatrix(**sm_kw), seed))

    # Sweep B: temperature
    for temp in cfg4["temperatures"]:
        sm_kw = {**base_sm, "temperature": temp,
                 "alpha_correct": cfg1["alpha_correct"],
                 "alpha_incorrect": cfg1["alpha_incorrect"]}
        for seed in SEEDS5:
            add("B_temperature", "temperature", temp, seed, "accuracy",
                _score_trial(gen_noise.generate(N_ALERTS, seed), ScoringMatrix(**sm_kw), seed))

    # Sweep C: noise_rate — regenerate alerts per noise level
    for noise in cfg4["noise_rates"]:
        sm_kw = {**base_sm, "temperature": cfg1["temperature"],
                 "alpha_correct": cfg1["alpha_correct"],
                 "alpha_incorrect": cfg1["alpha_incorrect"]}
        gen_c = AlertGenerator({"noise_rate": noise})
        for seed in SEEDS5:
            add("C_noise", "noise_rate", noise, seed, "accuracy",
                _score_trial(gen_c.generate(N_ALERTS, seed), ScoringMatrix(**sm_kw), seed))

    # Sweep D: embedding_dim — 50 entities each domain, 10 signals
    sig_str = float(raw["experiment_2"]["signal_strength"])
    for dim in cfg4["embedding_dims"]:
        if dim < 14: continue        # shared slice (dims 6-13) requires d >= 14
        for seed in SEEDS5:
            add("D_embedding_dim", "embedding_dim", dim, seed, "f1",
                _disc_f1(dim, 50, 10, sig_str, seed))

    # --- write CSV ---
    csv_path   = RESULTS_DIR / "sensitivity_data.csv"
    fieldnames = ["sweep", "param_name", "param_value", "seed", "metric_name", "metric_value"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {csv_path}")
    # --- summary ---
    by_param = defaultdict(list)
    for r in rows:
        by_param[(r["sweep"], r["param_value"])].append(r["metric_value"])
    print(f"\n{'sweep':>22}  {'param_value':>12}  {'mean_metric':>12}")
    last = None
    for (sweep, val), vals in sorted(by_param.items()):
        if sweep != last:
            print(f"  --- {sweep} ---")
            last = sweep
        print(f"  {'':>20}  {str(val):>12}  {np.mean(vals):>12.4f}")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_experiment()
