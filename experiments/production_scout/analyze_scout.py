import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments" / "v_cga_frozen_two_stream_v2.json"
OUT = ROOT / "experiments" / "production_scout"
OUT.mkdir(parents=True, exist_ok=True)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    se = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * se)
    return mean, mean - half, mean + half


def linear_slope(x, y):
    slope, intercept, r, p, se = stats.linregress(x, y)
    return {"slope_per_day": float(slope), "intercept": float(intercept),
            "r2": float(r * r), "p_value": float(p), "se": float(se)}


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    metadata = data["metadata"]
    a = {(int(r["seed"]), int(r["day"])): r for r in data["stream_a"]}
    b = {(int(r["seed"]), int(r["day"])): r for r in data["stream_b"]}
    keys = sorted(set(a) & set(b))
    rows = []
    for seed, day in keys:
        ar, br = a[(seed, day)], b[(seed, day)]
        rows.append({
            "seed": seed,
            "day": day,
            "control_accuracy_rolling_10": ar["accuracy_rolling_10"],
            "treatment_accuracy_rolling_10": br["accuracy_rolling_10"],
            "gap_treatment_minus_control": br["accuracy_rolling_10"] - ar["accuracy_rolling_10"],
            "control_enrichment_active": ar["enrichment_active"],
            "treatment_enrichment_active": br["enrichment_active"],
            "control_entities": ar["graph_entity_count"],
            "treatment_entities": br["graph_entity_count"],
        })

    bins = [("1-10", 1, 10), ("11-50", 11, 50), ("51-90", 51, 90)]
    bin_rows = []
    for label, lo, hi in bins:
        for seed in sorted({r["seed"] for r in rows}):
            vals = [r["gap_treatment_minus_control"] for r in rows
                    if r["seed"] == seed and lo <= r["day"] <= hi]
            if vals:
                bin_rows.append({"bin": label, "seed": seed,
                                 "gap_mean": float(np.mean(vals)),
                                 "n_days": len(vals)})

    summary = []
    for label, lo, hi in bins:
        vals = np.array([r["gap_mean"] for r in bin_rows if r["bin"] == label])
        m, l, u = mean_ci(vals)
        summary.append({"bin": label, "day_start": lo, "day_end": hi,
                        "n_seeds": int(len(vals)), "gap_mean": m,
                        "ci95_low": l, "ci95_high": u,
                        "positive_seed_count": int(np.sum(vals > 0)),
                        "negative_seed_count": int(np.sum(vals < 0)),
                        "zero_seed_count": int(np.sum(vals == 0))})

    daily = []
    for day in sorted({r["day"] for r in rows}):
        vals = np.array([r["gap_treatment_minus_control"] for r in rows if r["day"] == day])
        m, l, u = mean_ci(vals)
        daily.append({"day": day, "n_seeds": int(len(vals)), "gap_mean": m,
                      "ci95_low": l, "ci95_high": u})

    # The raw file has aggregate rolling accuracy, so this is a day-rate
    # descriptive slope, not the requested per-decision k fit.
    x = np.array([r["day"] for r in daily], dtype=float)
    y = np.array([r["gap_mean"] for r in daily], dtype=float)
    slope = linear_slope(x, y)
    min_gap = min(daily, key=lambda r: r["gap_mean"])
    max_abs_gap = max(daily, key=lambda r: abs(r["gap_mean"]))
    recovery = [r for r in daily if r["day"] >= 45]
    recovery_slope = linear_slope(
        np.array([r["day"] for r in recovery], dtype=float),
        np.array([r["gap_mean"] for r in recovery], dtype=float),
    )

    # Deployment-level checks, not factor-level checks.
    enrichment_state = {
        "control_unique": sorted({bool(r["control_enrichment_active"]) for r in rows}),
        "treatment_unique": sorted({bool(r["treatment_enrichment_active"]) for r in rows}),
        "control_varies_by_day": len({(r["seed"], r["day"], r["control_enrichment_active"]) for r in rows})
        != len(rows),
        "treatment_varies_by_day": len({(r["seed"], r["day"], r["treatment_enrichment_active"]) for r in rows})
        != len(rows),
    }

    table_path = OUT / "gap_by_day.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=daily[0].keys())
        writer.writeheader()
        writer.writerows(daily)
    bin_path = OUT / "gap_by_day_bin.csv"
    with bin_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.plot(x, y * 100, color="#0072B2", linewidth=1.8, label="Treatment − control")
    lo = np.array([r["ci95_low"] for r in daily]) * 100
    hi = np.array([r["ci95_high"] for r in daily]) * 100
    ax.fill_between(x, lo, hi, color="#0072B2", alpha=0.18, label="95% t CI across seeds")
    ax.set_xlabel("Day")
    ax.set_ylabel("Rolling-10 accuracy gap (percentage points)")
    ax.set_title("V-CGA-FROZEN: enriched treatment minus control")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "gap_vs_day.png", dpi=300)
    plt.close(fig)

    result = {
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "n_rows": len(rows), "n_seeds": len({r["seed"] for r in rows}),
        "days": len({r["day"] for r in rows}),
        "supported_measurement": "deployment-day natural-decay descriptive gap",
        "not_supported": [
            "per-decision sigma-coupling rate",
            "mean-update direction check",
            "factor-specific delta_f or enriched-factor dependence",
            "decision-count bins (only day bins are persisted)",
        ],
        "bin_summary": summary,
        "daily_slope": slope,
        "minimum_gap": {"day": int(min_gap["day"]), "gap": min_gap["gap_mean"]},
        "maximum_absolute_gap": {"day": int(max_abs_gap["day"]), "gap": max_abs_gap["gap_mean"]},
        "post_peak_recovery_slope": recovery_slope,
        "enrichment_state": enrichment_state,
        "metadata_seed_enrichment_rates": metadata.get("seed_enrichment_rates"),
    }
    result_path = OUT / "derived_summary.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    files = [table_path, bin_path, OUT / "gap_vs_day.png", result_path]
    report_path = ROOT / "results_production_scout.md"
    if report_path.exists():
        files.append(report_path)
    manifest = {"schema": "production-scout-artifacts-v1", "files": []}
    manifest["source_input"] = {
        "path": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "bytes": INPUT.stat().st_size,
        "sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    }
    for p in files:
        manifest["files"].append({"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                                  "bytes": p.stat().st_size,
                                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"summary": summary, "daily_slope": slope,
                      "enrichment_state": enrichment_state}, indent=2))


if __name__ == "__main__":
    main()
