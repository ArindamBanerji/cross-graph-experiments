"""Publication charts for Experiment 4: Parameter sensitivity analysis."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent.parent
CSV_PATH    = ROOT / "experiments" / "exp4_sensitivity" / "results" / "sensitivity_data.csv"
FIGURES_DIR = ROOT / "paper_figures"
FONT        = dict(title=13, label=11, tick=9, anno=8.5)
DPI, COL    = 300, "#1E3A5F"

def _style():
    for n in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try: plt.style.use(n); break
        except OSError: pass
    plt.rcParams.update({"axes.titlesize": FONT["label"], "axes.labelsize": FONT["label"],
                          "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
                          "figure.dpi": DPI, "savefig.dpi": DPI, "pdf.fonttype": 42})

def _save(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES_DIR / f"{stem}.{e}" for e in ("pdf", "png")]
    for p in paths: fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig); return tuple(paths)

def _agg(df, sweep):
    s = df[df["sweep"] == sweep].copy()
    s["param_value"] = pd.to_numeric(s["param_value"])
    return s.groupby("param_value")["metric_value"].mean().reset_index().sort_values("param_value")

def _panel(ax, g, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=FONT["tick"]); ax.set_ylabel(ylabel, fontsize=FONT["tick"])
    ax.set_title(title, fontsize=FONT["label"]); ax.set_xticks(g["param_value"].tolist())
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def chart_sensitivity(df):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    kw = dict(color=COL, marker="o", markersize=5, linewidth=1.8)

    # Panel A: asymmetry ratio — mark peak at 20
    ax, g = axes[0, 0], _agg(df, "A_asymmetry")
    ax.plot(g["param_value"], g["metric_value"], **kw)
    ax.axvline(20, color="#EA580C", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(21.5, g["metric_value"].min() + 0.01, "peak (20:1)",
            fontsize=FONT["anno"], color="#EA580C")
    _panel(ax, g, "Asymmetry ratio", "Accuracy", "A: Asymmetry Ratio")
    # Panel B: temperature — mark default 0.25
    ax, g = axes[0, 1], _agg(df, "B_temperature")
    ax.plot(g["param_value"], g["metric_value"], **kw)
    ax.axvline(0.25, color="#EA580C", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.27, g["metric_value"].min() + 0.005, "default (0.25)",
            fontsize=FONT["anno"], color="#EA580C")
    _panel(ax, g, r"Temperature $\tau$", "Accuracy", "B: Temperature")
    # Panel C: noise rate — critical threshold at 0.048
    ax, g = axes[1, 0], _agg(df, "C_noise")
    ax.plot(g["param_value"], g["metric_value"], **kw)
    ax.axvline(0.048, color="#7C3AED", linestyle="--", linewidth=1.2)
    ax.text(0.052, g["metric_value"].max() * 0.92, "critical\nthreshold",
            fontsize=FONT["anno"], color="#7C3AED")
    _panel(ax, g, "Noise rate", "Accuracy", "C: Noise Rate")
    # Panel D: embedding dim
    ax, g = axes[1, 1], _agg(df, "D_embedding_dim")
    ax.plot(g["param_value"], g["metric_value"], **kw)
    _panel(ax, g, "Embedding dim", "F1", "D: Embedding Dimension")
    fig.suptitle("Parameter Sensitivity Analysis", fontsize=FONT["title"])
    fig.tight_layout()
    return _save(fig, "exp4_sensitivity")

def latex_table(df) -> Path:
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Sensitivity analysis: best mean metric per sweep (5 seeds).}",
             r"\label{tab:exp4}", r"\begin{tabular}{llcc}", r"\toprule",
             r"Sweep & Parameter & Best value & Mean metric \\", r"\midrule"]
    for sw, param in [("A_asymmetry", "asymmetry\_ratio"), ("B_temperature", "temperature"),
                      ("C_noise", "noise\_rate"),          ("D_embedding_dim", "embedding\_dim")]:
        s  = df[df["sweep"] == sw].copy()
        s["param_value"] = pd.to_numeric(s["param_value"])
        g  = s.groupby("param_value")["metric_value"].mean()
        bv = g.idxmax()
        sw_tex = sw.replace("_", r"\_"); lines.append(f"{sw_tex} & {param} & {bv:g} & {g[bv]:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / "exp4_table.tex"
    p.write_text("\n".join(lines) + "\n"); return p

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Run exp4 runner first: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype={"param_value": str})
    print(f"  exp4: {len(df)} rows, {df['sweep'].nunique()} sweeps")
    files = list(chart_sensitivity(df)) + [latex_table(df)]
    print("Generated files:")
    for p in files: print(f"  {p.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
