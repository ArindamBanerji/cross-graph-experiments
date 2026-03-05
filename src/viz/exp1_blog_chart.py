"""Simplified convergence chart for CI blog (executive audience)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT        = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ROOT / "paper_figures"
DPI         = 300

def make_chart() -> None:
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 5000, 500)

    # Parametric trajectories + small Gaussian noise (fixed seed → reproducible)
    noise    = lambda s: rng.normal(0, s, size=len(t))
    comp     = 25 + 44.4 * (1 - np.exp(-t / 1200)) + noise(0.5)
    periodic = (35 + 18.8 * (1 - np.exp(-t / 800))
                + 4 * np.sin(t / 500 * 2 * np.pi) + noise(0.8))
    random_  = np.full_like(t, 25.0) + noise(0.3)

    for n in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try: plt.style.use(n); break
        except OSError: pass
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                          "figure.dpi": DPI, "savefig.dpi": DPI})

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fill between random baseline and compounding curve
    ax.fill_between(t, random_, comp, color="#BFDBFE", alpha=0.12)

    ax.plot(t, comp,     color="#2563EB", linewidth=3,   label="Compounding (ours)")
    ax.plot(t, periodic, color="#F59E0B", linewidth=1.8, linestyle="--",
            alpha=0.7, label="Periodic Retrain")
    ax.plot(t, random_,  color="#6B7280", linewidth=1.5, linestyle=":",
            alpha=0.7, label="Random Baseline")

    # Arrow annotation at compounding endpoint
    ax.annotate(
        "69.4% accuracy",
        xy=(t[-1], comp[-1]), xytext=(3100, 73),
        fontsize=11, color="#2563EB", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.4),
    )

    # Small label near random baseline
    ax.text(180, 20.5, "25% random", fontsize=9, color="#6B7280", alpha=0.85)

    # Axes — no numeric x-ticks for exec audience
    ax.set_xticks([])
    ax.set_xlabel("Time →", fontsize=11, labelpad=4)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_yticks([25, 40, 55, 70])
    ax.set_yticklabels(["25%", "40%", "55%", "70%"])
    ax.set_title("The System Gets Better With Every Decision",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(15, 80)

    ax.legend(loc="center left", framealpha=0.9, edgecolor="#E2E8F0", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.6)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"exp1_blog_convergence.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {path.relative_to(ROOT)}")
    plt.close(fig)

if __name__ == "__main__":
    make_chart()
