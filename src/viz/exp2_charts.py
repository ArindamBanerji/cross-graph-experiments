"""
Publication-quality charts for Experiment 2: Cross-Graph Discovery.

Generates
---------
paper_figures/exp2_f1_comparison.{pdf,png}     Chart 1 -- best F1 by method (bar)
paper_figures/exp2_precision_recall.{pdf,png}  Chart 2 -- P-R tradeoff (two_stage vs cosine)
paper_figures/exp2_table.tex                   LaTeX summary table
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # file-only backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "exp2_cross_graph_discovery" / "results"
FIGURES_DIR = ROOT / "paper_figures"
CSV_PATH    = RESULTS_DIR / "discovery_results.csv"

# ---------------------------------------------------------------------------
# Visual constants  (mirrors configs/default.yaml visualization block)
# ---------------------------------------------------------------------------

FONT = dict(title=13, label=11, tick=9, anno=8.5)
DPI  = 300

METHOD_ORDER = ["two_stage", "logit_only", "topk_only", "cosine", "random"]
METHOD_COLORS = {
    "two_stage":  "#1E3A5F",
    "logit_only": "#7C3AED",
    "topk_only":  "#EA580C",
    "cosine":     "#94A3B8",
    "random":     "#CBD5E1",
}
METHOD_LABELS = {
    "two_stage":  "Two-stage",
    "logit_only": "Logit only",
    "topk_only":  "Top-K only",
    "cosine":     "Cosine",
    "random":     "Random",
}
LATEX_NAMES = {
    "two_stage":  "Two-stage (ours)",
    "logit_only": "Logit only",
    "topk_only":  "Top-K only",
    "cosine":     "Cosine",
    "random":     "Random",
}

# ---------------------------------------------------------------------------
# Style / IO helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    plt.rcParams.update({
        "font.size":       FONT["label"],
        "axes.titlesize":  FONT["title"],
        "axes.labelsize":  FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["tick"],
        "figure.dpi":      DPI,
        "savefig.dpi":     DPI,
        "pdf.fonttype":    42,   # embed TrueType fonts (required for journals)
        "ps.fonttype":     42,
    })


def _clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.7)


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return pdf, png

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Normalise config columns: NaN → "" so groupby works consistently
    for col in ("theta_logit", "top_k", "threshold"):
        df[col] = df[col].fillna("").astype(str)
    return df


def _best_config_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method, find the config (theta/k/threshold combo) that maximises
    mean F1 across seeds and domain pairs.  Returns a DataFrame indexed by
    method with columns: best_mean_f1, std_f1, best_prec, best_rec.
    """
    records = []
    for method, mdf in df.groupby("method"):
        # Mean F1 per config combo (averaged over seeds × domain pairs)
        mean_by_cfg = (
            mdf.groupby(["theta_logit", "top_k", "threshold"])["f1"].mean()
        )
        best_cfg = mean_by_cfg.idxmax()

        mask = (
            (mdf["theta_logit"] == best_cfg[0]) &
            (mdf["top_k"]       == best_cfg[1]) &
            (mdf["threshold"]   == best_cfg[2])
        )
        best_rows = mdf[mask]

        # Std: per-seed mean over domain_pairs, then std across seeds
        seed_means = best_rows.groupby("seed")["f1"].mean()
        records.append(dict(
            method       = method,
            best_mean_f1 = float(mean_by_cfg[best_cfg]),
            std_f1       = float(seed_means.std(ddof=1)),
            best_prec    = float(best_rows["precision"].mean()),
            best_rec     = float(best_rows["recall"].mean()),
        ))
    return pd.DataFrame(records).set_index("method")


# ---------------------------------------------------------------------------
# Chart 1 — Best F1 per method (bar chart)
# ---------------------------------------------------------------------------

def chart1_f1_bars(df: pd.DataFrame) -> tuple[Path, Path]:
    _apply_style()
    stats = _best_config_stats(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    xs      = np.arange(len(METHOD_ORDER))
    heights = [stats.loc[m, "best_mean_f1"] for m in METHOD_ORDER]
    errs    = [stats.loc[m, "std_f1"]       for m in METHOD_ORDER]
    colors  = [METHOD_COLORS[m]             for m in METHOD_ORDER]

    ax.bar(
        xs, heights,
        yerr=errs,
        color=colors,
        width=0.6,
        error_kw=dict(elinewidth=1.2, capsize=4, ecolor="#555555"),
        zorder=3,
    )

    # "Xx above random" annotation on two_stage bar
    ts_f1  = stats.loc["two_stage", "best_mean_f1"]
    rnd_f1 = stats.loc["random",    "best_mean_f1"]
    ratio  = ts_f1 / rnd_f1 if rnd_f1 > 0 else float("inf")
    ts_idx = METHOD_ORDER.index("two_stage")
    top    = heights[ts_idx] + errs[ts_idx]
    ax.text(
        ts_idx, top + 0.008,
        f"{ratio:.0f}× above random",
        ha="center", va="bottom",
        fontsize=FONT["anno"], color=METHOD_COLORS["two_stage"], fontweight="bold",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=FONT["tick"])
    ax.set_ylabel("Best F1 (mean ± std across seeds & domain pairs)", fontsize=FONT["label"])
    ax.set_title("Cross-Graph Discovery: Best F1 by Method", fontsize=FONT["title"])
    ax.set_ylim(0, max(heights) * 1.45)
    _clean_ax(ax)
    fig.tight_layout()
    return _save(fig, "exp2_f1_comparison")


# ---------------------------------------------------------------------------
# Chart 2 — Precision-Recall tradeoff (two_stage vs cosine)
# ---------------------------------------------------------------------------

def chart2_precision_recall(df: pd.DataFrame) -> tuple[Path, Path]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # two_stage: grid of (theta, k) — scatter
    ts = df[df["method"] == "two_stage"]
    ts_pts = (
        ts.groupby(["theta_logit", "top_k", "threshold"])[["precision", "recall"]]
        .mean()
        .reset_index()
    )
    ax.scatter(
        ts_pts["recall"], ts_pts["precision"],
        color=METHOD_COLORS["two_stage"], marker="o",
        s=40, alpha=0.80, zorder=4, label="Two-stage",
    )

    # cosine: 1D threshold sweep — line + markers (sorted by threshold → recall)
    cos = df[df["method"] == "cosine"]
    cos_pts = (
        cos.groupby(["theta_logit", "top_k", "threshold"])[["precision", "recall"]]
        .mean()
        .reset_index()
    )
    cos_pts["_thr"] = cos_pts["threshold"].astype(float)
    cos_pts = cos_pts.sort_values("_thr", ascending=False)   # high thr = high prec, low rec
    ax.plot(
        cos_pts["recall"], cos_pts["precision"],
        color=METHOD_COLORS["cosine"], marker="s",
        markersize=6, linewidth=1.4, alpha=0.85, zorder=3, label="Cosine",
    )

    ax.set_xlabel("Recall",    fontsize=FONT["label"])
    ax.set_ylabel("Precision", fontsize=FONT["label"])
    ax.set_title(
        "Precision-Recall Tradeoff: Two-Stage vs Cosine",
        fontsize=FONT["title"],
    )
    ax.legend(framealpha=0.9, edgecolor="#E2E8F0", fontsize=FONT["tick"])
    ax.grid(True, alpha=0.15, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "exp2_precision_recall")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(df: pd.DataFrame) -> Path:
    stats = _best_config_stats(df)

    rows: list[str] = []
    for method in METHOD_ORDER:
        row  = stats.loc[method]
        f1   = f"{row['best_mean_f1']:.4f} $\\pm$ {row['std_f1']:.4f}"
        prec = f"{row['best_prec']:.4f}"
        rec  = f"{row['best_rec']:.4f}"
        bold_open  = r"\textbf{" if method == "two_stage" else ""
        bold_close = r"}"         if method == "two_stage" else ""
        rows.append(
            f"{bold_open}{LATEX_NAMES[method]}{bold_close} & {f1} & {prec} & {rec}"
            + r" \\"
        )
        if method == "cosine":          # separator before random baseline
            rows.append(r"\midrule")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Cross-graph discovery: best F1 (mean $\pm$ std over 10 seeds), "
         r"precision, and recall per method. Best threshold configuration selected "
         r"per method. Evaluated on 50 ground-truth signal pairs across 40\,000 "
         r"candidate entity pairs.}"),
        r"\label{tab:exp2}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & F1 & Precision & Recall \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = FIGURES_DIR / "exp2_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {CSV_PATH}\n"
            "Run experiments/exp2_cross_graph_discovery/run.py first."
        )

    print("Loading results ...")
    df = _load_data()
    print(f"  {len(df)} rows, {df['seed'].nunique()} seeds, "
          f"{df['method'].nunique()} methods, {df['domain_pair'].nunique()} domain pairs")

    generated: list[Path] = []

    print("Chart 1: Best F1 bar chart ...")
    pdf, png = chart1_f1_bars(df)
    generated += [pdf, png]

    print("Chart 2: Precision-Recall tradeoff ...")
    pdf, png = chart2_precision_recall(df)
    generated += [pdf, png]

    print("LaTeX table ...")
    tex = generate_latex_table(df)
    generated.append(tex)

    print("\nGenerated files:")
    for p in generated:
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            rel = p
        print(f"  {rel}")


if __name__ == "__main__":
    main()
