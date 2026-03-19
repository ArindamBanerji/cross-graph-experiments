import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.viz.bridge_common import save_figure, COLORS

# ── DATA (EXP-OP2 N=20 + EXP-OP2-N100, η_neg=1.0 era) ──────────────────────
conditions     = ["C\n(0%)", "P-25\n(25%)", "P-50\n(50%)",
                  "P-75\n(75%)", "A\n(none)", "B\n(100%)"]
operator_pct   = [0,          25,             50,             75,    -1,    100]
auac_delta     = [-0.0029,    -0.0009,        +0.0012,       +0.0013, 0.0,  +0.0041]
auac_delta_std = [ 0.0040,     0.0035,         0.0033,        0.0038, 0.0,   0.0048]
p_values       = [ 0.994,      0.792,          0.122,         0.185,  1.0,   0.0008]

# Never-recover rates from N=100, Wilson 95% CI
nr_rate    = [0.38,  0.29,  0.20,  0.28,  0.24,  0.08]
nr_ci_low  = [0.291, 0.210, 0.133, 0.201, 0.167, 0.041]
nr_ci_high = [0.478, 0.385, 0.289, 0.375, 0.332, 0.150]

# T_recovery mean (decisions) from EXP-OP2 N=20
t_recovery = [425, 380, 280, 228, 178, 55]

# Bonferroni: k=5 comparisons vs A, alpha_corr = 0.01
alpha_bonf  = 0.05 / 5
significant = [p < alpha_bonf for p in p_values]  # only B passes

# Colors: red family → gray → green
bar_colors = ["#c62828", "#e57373", "#ff8a65", "#ffa726", "#9e9e9e", "#2e7d32"]

# ── FIGURE: 3 panels stacked, 14×12 inches ────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.4, 1.0], hspace=0.50)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

n = len(conditions)
x = np.arange(n)

# ── PANEL A: AUAC delta bars ─────────────────────────────────────────────────
ax1.set_title("Panel A — AUAC delta by operator accuracy\n"
              "(N=20 seeds, λ=0.5, 400 post-shift decisions, Bonferroni α=0.010)",
              fontsize=12, pad=8)

for i, (delta, std, sig, color) in enumerate(
    zip(auac_delta, auac_delta_std, significant, bar_colors)):
  ax1.bar(i, delta, color=color, alpha=0.85, edgecolor="black",
          linewidth=0.7, zorder=3)
  ax1.errorbar(i, delta, yerr=std, color="black",
               capsize=5, capthick=1.4, linewidth=1.4, zorder=4)
  if sig:
    ax1.text(i, delta + std + 0.0010, "p=0.0008 ✓",
             ha="center", fontsize=9.5, fontweight="bold", color="#2e7d32")
  else:
    ax1.text(i, max(delta + std + 0.0004, 0.00025), "ns",
             ha="center", fontsize=9, color="#757575")

ax1.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=2)

# Green / red zone shading
ax1.axhspan( 0.0005,  0.0090, alpha=0.06, color="#2e7d32", zorder=1)
ax1.axhspan(-0.0075, -0.0001, alpha=0.05, color="#c62828", zorder=1)
ax1.text(5.42,  0.0062, "DEPLOY\nzone",    ha="center", fontsize=8.5,
         color="#2e7d32", style="italic")
ax1.text(5.42, -0.0052, "DO NOT\nDEPLOY", ha="center", fontsize=8.5,
         color="#c62828", style="italic")

# Zero-crossing annotation
ax1.annotate("Zero-crossing between P-25 and P-50:\nno tolerance for any incorrect σ cells",
             xy=(1.5, 0.0001), xytext=(0.2, 0.0038),
             fontsize=9, color="#37474f",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                       edgecolor="#90a4ae"),
             arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.1))

ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=11)
ax1.set_ylabel("AUAC delta vs no-operator baseline", fontsize=11)
ax1.set_ylim(-0.0080, 0.0095)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}"))

# ── PANEL B: Never-recover rates ─────────────────────────────────────────────
ax2.set_title("Panel B — Never-recover rate with 95% Wilson CI  (N=100 seeds)",
              fontsize=12, pad=8)

for i, (nr, ci_lo, ci_hi, color) in enumerate(
    zip(nr_rate, nr_ci_low, nr_ci_high, bar_colors)):
  ax2.bar(i, nr, color=color, alpha=0.78, edgecolor="black", linewidth=0.7)
  ax2.errorbar(i, nr, yerr=[[nr - ci_lo], [ci_hi - nr]],
               color="black", capsize=5, capthick=1.4, linewidth=1.4, zorder=4)
  ax2.text(i, ci_hi + 0.010, f"{nr:.0%}",
           ha="center", fontsize=10, fontweight="bold", color=color)

ax2.axhline(0.05, color="#2e7d32", linewidth=1.4, linestyle=":",
            label="5% acceptable threshold")
ax2.axhline(0.24, color="#9e9e9e", linewidth=1.0, linestyle="--", alpha=0.6,
            label="Baseline A: 24% (no operator)")

ax2.set_xticks(x)
ax2.set_xticklabels(conditions, fontsize=11)
ax2.set_ylabel("Never-recover rate", fontsize=11)
ax2.set_ylim(0, 0.62)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax2.legend(fontsize=9, loc="upper left")

# P-75 paradox callout
ax2.annotate("P-75 paradox:\n28% NR > 24% baseline\n(worse than no operator)",
             xy=(3, nr_rate[3]), xytext=(3.3, 0.46),
             fontsize=8.5, color="#e65100",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                       edgecolor="#e65100"),
             arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.1))

# η_neg caveat box
ax2.text(0.01, 0.04,
         "\u26a0  Measured at \u03b7_neg=1.0 (pre-fix). Re-confirm after EXP-OP2-RECHECK.",
         transform=ax2.transAxes, fontsize=8.5, color="#b71c1c",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee",
                   edgecolor="#b71c1c", alpha=0.9))

# ── PANEL C: Policy text ──────────────────────────────────────────────────────
ax3.axis("off")
policy = (
    "SAFETY POLICY \u2014 EXP-OP2 findings (canonical \u03b7_neg=0.05)\n"
    "\u2501" * 67 + "\n"
    "\u2713  Deploy operator only if EVERY \u03c3 cell is directionally correct (100%)\n"
    "\u2717  Do NOT deploy P-75 operators \u2014 recovery is slower than no operator\n"
    "\u2713  TTL expiry IS sufficient at \u03b7_neg=0.05 \u2014 model heals within ~250 decisions\n"
    "   (prior \u2018lasting damage\u2019 finding was \u03b7_neg=1.0 specific \u2014 now FORBIDDEN)\n"
    "\u26a0  NR rates in Panel B measured at \u03b7_neg=1.0 \u2014 EXP-OP2-RECHECK pending\n"
    "\u26a0  Acute-phase benefit (3\u00d7 aggregate) pending re-confirmation at \u03b7_neg=0.05\n"
    "   \u03c3 is a speed-of-adaptation tool. Its value is front-loaded, not sustained."
)
ax3.text(0.01, 0.97, policy, transform=ax3.transAxes, fontsize=10.5,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.55", facecolor="#f1f8e9",
                   edgecolor="#388e3c", linewidth=1.5))

# ── Supertitle + caption ──────────────────────────────────────────────────────
fig.suptitle("EXP-OP2: Operator Deployment Safety Policy\n"
             "Only 100%-Correct Operators Produce Significant AUAC Improvement",
             fontsize=14, fontweight="bold", y=0.99)

fig.text(0.5, 0.005,
         "\u03b7_neg=1.0 era results. Policy finding (100%-correct threshold) survives fix. "
         "NR rates and post-expiry damage pending EXP-OP2-RECHECK at \u03b7_neg=0.05.",
         ha="center", fontsize=8.5, color="#b71c1c", style="italic")

save_figure(fig, "expOP2_safety_policy_summary", output_dir="paper_figures")
plt.close(fig)
print("[CHART] expOP2_safety_policy_summary.png + .pdf saved")
