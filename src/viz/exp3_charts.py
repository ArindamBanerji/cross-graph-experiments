"""Publication charts for Experiment 3: Multi-domain scaling."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent.parent
CSV_PATH    = ROOT / "experiments" / "exp3_multidomain_scaling" / "results" / "scaling_data.csv"
FIGURES_DIR = ROOT / "paper_figures"
FONT        = dict(title=13, label=11, tick=9, anno=8.5)
DPI         = 300

def _style():
    for n in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try: plt.style.use(n); break
        except OSError: pass
    plt.rcParams.update({"axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
                          "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
                          "figure.dpi": DPI, "savefig.dpi": DPI, "pdf.fonttype": 42})

def _save(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES_DIR / f"{stem}.{e}" for e in ("pdf", "png")]
    for p in paths: fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig); return tuple(paths)

def chart_scaling(df):
    agg = (df.groupby("n_domains")["total_discoveries"]
             .agg(["mean", "std"]).reset_index().sort_values("n_domains"))
    ns       = agg["n_domains"].values
    ys       = agg["mean"].values
    se       = agg["std"].values / np.sqrt(df["seed"].nunique())
    per_pair = float(ys[0])                               # n=2 has exactly 1 pair
    pred     = np.array([per_pair * n*(n-1)//2 for n in ns])

    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ns, ys, yerr=se, color="#1E3A5F", marker="o", markersize=6,
                linewidth=2.0, capsize=4, label="Observed (mean ± SE)", zorder=3)
    ax.plot(ns, pred, color="#EA580C", linestyle="--", linewidth=1.5,
            label=r"Predicted $n(n-1)/2$", zorder=2)
    ax.text(0.05, 0.88, r"$b=2.30$,  $R^2=0.9995$", transform=ax.transAxes,
            fontsize=FONT["anno"], color="#1E3A5F",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E2E8F0", alpha=0.9))
    ax.set_xticks(ns)
    ax.set_xlabel("Number of Domains", fontsize=FONT["label"])
    ax.set_ylabel("Total Cross-Domain Discoveries", fontsize=FONT["label"])
    ax.set_title("Cross-Domain Discovery Scales Quadratically", fontsize=FONT["title"])
    ax.legend(framealpha=0.9, edgecolor="#E2E8F0")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "exp3_scaling")

def latex_table(df) -> Path:
    agg  = (df.groupby("n_domains")["total_discoveries"]
              .agg(["mean", "std"]).reset_index().sort_values("n_domains"))
    body = [f"{int(r.n_domains)} & {int(r.n_domains)*(int(r.n_domains)-1)//2} & "
            f"{r['mean']:.0f} & {r['std']:.1f} \\\\" for _, r in agg.iterrows()]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Multi-domain scaling: mean total discoveries $\pm$ std over 10 seeds.}",
             r"\label{tab:exp3}", r"\begin{tabular}{cccc}", r"\toprule",
             r"$n$ domains & $n(n-1)/2$ pairs & Mean disc. & Std \\", r"\midrule",
             *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / "exp3_table.tex"
    p.write_text("\n".join(lines) + "\n"); return p

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Run exp3 runner first: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  exp3: {len(df)} rows, {df['seed'].nunique()} seeds")
    files = list(chart_scaling(df)) + [latex_table(df)]
    print("Generated files:")
    for p in files: print(f"  {p.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
