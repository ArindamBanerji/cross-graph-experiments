"""Simplified scaling chart for CI blog (executive audience)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT        = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ROOT / "paper_figures"
DPI         = 300

def make_chart() -> None:
    ns       = np.array([2, 3, 4, 5, 6])
    observed = np.array([600, 1800, 3600, 6000, 9000])
    linear   = 600 * ns

    for n in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try: plt.style.use(n); break
        except OSError: pass
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                          "figure.dpi": DPI, "savefig.dpi": DPI})

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(ns, linear, observed, color="#BFDBFE", alpha=0.15)

    ax.plot(ns, observed, color="#2563EB", linewidth=3, marker="o",
            markersize=10, markeredgecolor="white", markeredgewidth=1.5,
            label="Actual: Super-Linear", zorder=3)
    ax.plot(ns, linear, color="#6B7280", linewidth=2, linestyle="--",
            marker="s", markersize=7, alpha=0.7, label="If Growth Were Linear",
            zorder=2)

    # Point labels for observed
    labels = ["600", "1,800", "3,600", "6,000", "9,000"]
    offsets = [(0, 280)] * 4 + [(0, 280)]
    for x, y, lbl, (dx, dy) in zip(ns, observed, labels, offsets):
        ax.text(x + dx, y + dy, lbl, ha="center", va="bottom",
                fontsize=9, color="#2563EB", fontweight="bold")

    # Double-headed arrow + annotation at n=6
    ax.annotate("", xy=(6.45, 9000), xytext=(6.45, 3600),
                arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=1.8))
    ax.text(6.55, 6300, "2.5× more\nthan linear", fontsize=10,
            color="#DC2626", fontweight="bold", va="center")

    ax.set_xlim(1.5, 7)
    ax.set_ylim(0, 11000)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n}\ndomains" for n in ns], fontsize=10)
    ax.set_xlabel("Number of Connected Knowledge Domains", fontsize=11, labelpad=6)
    ax.set_ylabel("Cross-Domain Discoveries", fontsize=11)
    ax.set_title("Each New Domain Is Worth More Than the Last",
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#E2E8F0", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.6)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"exp3_blog_scaling.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {path.relative_to(ROOT)}")
    plt.close(fig)

if __name__ == "__main__":
    make_chart()
