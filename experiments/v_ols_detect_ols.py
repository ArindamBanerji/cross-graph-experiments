"""
V-OLS-DETECT-OLS: OLSMonitor Re-Validation Experiment (v3)
==========================================================

Purpose: Close the CLAIM-OLS-01 credibility gap.

Production OLSMonitor parameters (from gae/convergence.py):
    Signal:  EWMA of q̄ (accuracy), λ=0.1. NOT raw OLS ratio.
    h:       5.0 (FIXED). ARL₀ ≈ 500. Calibrated for OLS ∈ [1.0, 2.5].
    k:       q_baseline − margin (dynamic, set once after warmup).
             Production margin = 0.05 (5pp below baseline).
    Warmup:  50 decisions (CALIBRATION_PERIOD).

Design:
    - 3 conditions: adversarial (gradual), silent_quality (sudden), control
    - 3 volumes: V=50 (500 dec), V=100 (1000 dec), V=200 (2000 dec)
    - 3 analyst quality: q̄=0.60, 0.75, 0.85
    - 5 margins: 0.02, 0.03, 0.05 (production), 0.08, 0.10
    - 30 seeds per cell

Run:
    python v_ols_detect_ols.py          # full (~3 min)
    python v_ols_detect_ols.py --quick  # smoke test (~20s)

Author: Dakshineshwari LLC · May 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import csv
import sys
import time


# ============================================================
# Production-Faithful OLSMonitor
# ============================================================

class OLSMonitor:
    """
    Matches gae/convergence.py OLSMonitor exactly.

    Signal: EWMA of per-decision accuracy (q̄).
        q_ewma(t) = λ·q(t) + (1-λ)·q_ewma(t-1)
        where q(t) = 1 if correct, 0 if incorrect.

    CUSUM: one-sided, detects sustained drops below baseline.
        S(n) = max(0, S(n-1) + (k - q_ewma(n)))
        Alarm when S(n) > h.

    Parameters:
        h = 5.0 (fixed, empirically calibrated)
        k = q_baseline - margin (set once after warmup)
        λ = 0.1 (EWMA smoothing factor)
        warmup = 50 decisions
    """

    def __init__(self, h: float = 5.0, margin: float = 0.05,
                 ewma_lambda: float = 0.1, warmup: int = 50):
        self.h = h
        self.margin = margin
        self.ewma_lambda = ewma_lambda
        self.warmup = warmup

        self._n: int = 0
        self._q_ewma: float = 0.5       # initial EWMA estimate
        self._baseline: Optional[float] = None
        self._k: Optional[float] = None  # set after warmup
        self._cusum: float = 0.0
        self._alarm_fired: bool = False
        self._alarm_index: Optional[int] = None

    def observe(self, correct: bool, index: int) -> bool:
        """
        Feed one decision outcome. Returns True if alarm fires.

        Args:
            correct: True if the decision was correct
            index: decision index (for alarm timestamping)
        """
        self._n += 1
        q_t = 1.0 if correct else 0.0

        # Update EWMA
        self._q_ewma = (self.ewma_lambda * q_t +
                        (1 - self.ewma_lambda) * self._q_ewma)

        # Warmup phase: accumulate baseline
        if self._n <= self.warmup:
            return False

        # Freeze baseline and k after warmup (once)
        if self._baseline is None:
            self._baseline = self._q_ewma
            self._k = self._baseline - self.margin
            return False

        if self._alarm_fired:
            return True

        # CUSUM: accumulate when EWMA drops below k
        self._cusum = max(0.0, self._cusum + (self._k - self._q_ewma))

        if self._cusum > self.h:
            self._alarm_fired = True
            self._alarm_index = index
            return True

        return False

    @property
    def alarm_fired(self) -> bool:
        return self._alarm_fired

    @property
    def alarm_index(self) -> Optional[int]:
        return self._alarm_index

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def k_value(self) -> Optional[float]:
        return self._k


# ============================================================
# Decision Stream Generation
# ============================================================

def generate_stream(
    n: int, q_bar: float, override_rate: float,
    override_quality: float, rng: np.random.Generator,
) -> list[dict]:
    decisions = []
    for i in range(n):
        is_override = rng.random() < override_rate
        q = override_quality if is_override else q_bar
        decisions.append({
            "index": i,
            "is_override": is_override,
            "correct": rng.random() < q,
        })
    return decisions


def inject_adversarial(
    stream: list[dict], onset: int, override_quality: float,
    rate_per_100: float = 0.04, floor: float = 0.30,
    rng: np.random.Generator = None,
) -> list[dict]:
    """Override quality degrades gradually after onset."""
    rng = rng or np.random.default_rng(42)
    result = []
    for d in stream:
        nd = dict(d)
        if d["index"] >= onset and d["is_override"]:
            elapsed = d["index"] - onset
            q = max(override_quality - rate_per_100 * elapsed / 100, floor)
            nd["correct"] = rng.random() < q
        result.append(nd)
    return result


def inject_silent_quality(
    stream: list[dict], onset: int, override_quality: float,
    drop_to: float = 0.50, rng: np.random.Generator = None,
) -> list[dict]:
    """Override quality drops suddenly at onset."""
    rng = rng or np.random.default_rng(42)
    result = []
    for d in stream:
        nd = dict(d)
        if d["index"] >= onset and d["is_override"]:
            nd["correct"] = rng.random() < drop_to
        result.append(nd)
    return result


# ============================================================
# T_damage: when does degradation become harmful?
# ============================================================

def compute_t_damage(
    stream: list[dict], onset: int,
    ewma_lambda: float = 0.1,
    drop_pp: float = 0.05,
    consecutive: int = 20,
) -> Optional[int]:
    """
    T_damage = first index where EWMA q̄ has stayed ≥ drop_pp below
    pre-onset baseline for ≥ consecutive decisions.
    Uses the same EWMA signal the monitor uses (apples-to-apples).
    Returns None if damage threshold never reached.
    """
    q_ewma = 0.5
    pre_onset_ewma = []

    for d in stream:
        q_t = 1.0 if d["correct"] else 0.0
        q_ewma = ewma_lambda * q_t + (1 - ewma_lambda) * q_ewma
        if d["index"] < onset:
            pre_onset_ewma.append(q_ewma)

    if len(pre_onset_ewma) < 20:
        return None

    baseline = float(np.mean(pre_onset_ewma[-30:]))

    # Re-scan for sustained drop
    q_ewma = 0.5
    run = 0
    for d in stream:
        q_t = 1.0 if d["correct"] else 0.0
        q_ewma = ewma_lambda * q_t + (1 - ewma_lambda) * q_ewma
        if d["index"] < onset:
            continue
        if q_ewma < baseline - drop_pp:
            run += 1
            if run >= consecutive:
                return d["index"]
        else:
            run = 0

    return None


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass
class Config:
    conditions: list[str] = field(default_factory=lambda: [
        "adversarial", "silent_quality", "control"])
    volumes: list[int] = field(default_factory=lambda: [50, 100, 200])
    q_bars: list[float] = field(default_factory=lambda: [0.60, 0.75, 0.85])
    margins: list[float] = field(default_factory=lambda: [
        0.02, 0.03, 0.05, 0.08, 0.10])
    n_seeds: int = 30
    dec_per_v: int = 10
    onset_frac: float = 0.25
    override_rate: float = 0.25
    override_quality: float = 0.80
    adv_rate: float = 0.04
    adv_floor: float = 0.30
    silent_drop: float = 0.50
    # Production OLSMonitor constants
    h: float = 5.0
    ewma_lambda: float = 0.1
    warmup: int = 50

    def n_dec(self, v): return v * self.dec_per_v
    def onset(self, v): return int(self.n_dec(v) * self.onset_frac)


@dataclass
class Result:
    condition: str
    volume: int
    q_bar: float
    margin: float
    seed: int
    alarm_fired: bool
    alarm_index: Optional[int]
    t_damage: Optional[int]
    lead_time: Optional[int]
    miss: Optional[bool]       # None = degradation_insufficient or control
    baseline_q: Optional[float]
    k_used: Optional[float]


# ============================================================
# Cell Runner
# ============================================================

def run_cell(cfg: Config, cond: str, vol: int, qb: float,
             margin: float, seed: int) -> Result:
    n = cfg.n_dec(vol)
    onset = cfg.onset(vol)

    rng_s = np.random.default_rng(seed)
    stream = generate_stream(n, qb, cfg.override_rate,
                             cfg.override_quality, rng_s)

    rng_i = np.random.default_rng(seed + 1_000_000)
    if cond == "adversarial":
        degraded = inject_adversarial(stream, onset, cfg.override_quality,
                                      cfg.adv_rate, cfg.adv_floor, rng_i)
    elif cond == "silent_quality":
        degraded = inject_silent_quality(stream, onset, cfg.override_quality,
                                         cfg.silent_drop, rng_i)
    else:
        degraded = stream

    # T_damage (oracle, same EWMA signal)
    t_damage = None if cond == "control" else compute_t_damage(
        degraded, onset, cfg.ewma_lambda)

    # Run monitor
    mon = OLSMonitor(h=cfg.h, margin=margin,
                     ewma_lambda=cfg.ewma_lambda, warmup=cfg.warmup)
    for d in degraded:
        mon.observe(d["correct"], d["index"])

    # Evaluate
    if cond == "control" or t_damage is None:
        return Result(cond, vol, qb, margin, seed,
                      mon.alarm_fired, mon.alarm_index,
                      t_damage, None, None, mon.baseline, mon.k_value)

    if mon.alarm_fired and mon.alarm_index is not None:
        lead = t_damage - mon.alarm_index
        miss = lead <= 0
    else:
        lead = None
        miss = True

    return Result(cond, vol, qb, margin, seed,
                  mon.alarm_fired, mon.alarm_index,
                  t_damage, lead, miss, mon.baseline, mon.k_value)


# ============================================================
# Runner
# ============================================================

def run(cfg: Config) -> list[Result]:
    total = (len(cfg.conditions) * len(cfg.volumes) *
             len(cfg.q_bars) * len(cfg.margins) * cfg.n_seeds)
    print(f"V-OLS-DETECT-OLS v3 (production-faithful)")
    print(f"  {total} cells = {len(cfg.conditions)} cond × "
          f"{len(cfg.volumes)} vol × {len(cfg.q_bars)} q̄ × "
          f"{len(cfg.margins)} margins × {cfg.n_seeds} seeds")
    print(f"  Monitor: h={cfg.h} (fixed), k=baseline−margin, "
          f"EWMA λ={cfg.ewma_lambda}, warmup={cfg.warmup}")
    print(f"  Stream: V×{cfg.dec_per_v}, onset at {cfg.onset_frac:.0%}")
    print()

    results = []
    done = 0
    t0 = time.time()

    for cond in cfg.conditions:
        for vol in cfg.volumes:
            for qb in cfg.q_bars:
                for margin in cfg.margins:
                    for s in range(cfg.n_seeds):
                        r = run_cell(cfg, cond, vol, qb, margin,
                                     seed=s + vol*1000 + int(qb*100)*100_000)
                        results.append(r)
                        done += 1
                    pct = done / total * 100
                    elapsed = time.time() - t0
                    print(f"\r  [{pct:5.1f}%] {cond:16s} V={vol:3d} "
                          f"q̄={qb:.2f} margin={margin:.2f} "
                          f"({elapsed:.1f}s)", end="")

    print(f"\n\n  {len(results)} results in {time.time()-t0:.1f}s\n")
    return results


# ============================================================
# Self-Interpreting Analysis
# ============================================================

def analyze(results: list[Result]):
    groups = defaultdict(list)
    for r in results:
        groups[(r.condition, r.margin)].append(r)

    margins = sorted(set(r.margin for r in results))
    conditions = [c for c in ["adversarial", "silent_quality"]
                  if any(r.condition == c for r in results)]
    volumes = sorted(set(r.volume for r in results))

    def stats(cond, m):
        cells = groups.get((cond, m), [])
        ev = [c for c in cells if c.miss is not None]
        ins = len([c for c in cells if c.miss is None and c.condition != "control"])
        miss = np.mean([c.miss for c in ev]) * 100 if ev else None
        leads = [c.lead_time for c in ev if c.lead_time is not None and not c.miss]
        p90 = np.percentile(leads, 90) if leads else None
        med = np.median(leads) if leads else None
        fa_cells = groups.get(("control", m), [])
        fa = np.mean([c.alarm_fired for c in fa_cells]) * 100 if fa_cells else None
        return dict(miss=miss, p90=p90, med=med, fa=fa,
                    n_ev=len(ev), n_ins=ins, n_caught=len(leads))

    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 72)
    print("  V-OLS-DETECT-OLS — FINDINGS REPORT")
    print("  Monitor: EWMA q̄ (λ=0.1) + CUSUM (h=5.0, k=baseline−margin)")
    print("=" * 72)

    # ── 1. TRADEOFF MATRIX ──
    print()
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  1. DETECTION vs FALSE ALARMS                           │")
    print("│                                                          │")
    print("│  margin = how far EWMA q̄ must drop below baseline       │")
    print("│  before CUSUM starts accumulating. Production = 0.05.    │")
    print("│  Lower margin = more sensitive. Higher = more specific.  │")
    print("└──────────────────────────────────────────────────────────┘")
    print()

    hdr = "".join(f"{'m='+str(m):>10s}" for m in margins)
    print(f"  {'':20s}{hdr}")
    prod_marks = "".join(
        f"{'◄ PROD':>10s}" if m == 0.05 else f"{'':>10s}" for m in margins)
    print(f"  {'':20s}{prod_marks}")
    print(f"  {'─' * (20 + 10 * len(margins))}")

    for cond in conditions:
        row = f"  {cond+' miss%':20s}"
        for m in margins:
            s = stats(cond, m)
            if s["miss"] is not None:
                mark = " ✓" if s["miss"] <= 5 else " ✗" if s["miss"] > 20 else "  "
                row += f"{s['miss']:7.1f}%{mark} "
            else:
                row += f"{'N/A':>10s}"
        print(row)

    row = f"  {'false alarm%':20s}"
    for m in margins:
        s = stats(conditions[0], m)
        if s["fa"] is not None:
            mark = " ✓" if s["fa"] <= 5 else " ✗" if s["fa"] > 20 else "  "
            row += f"{s['fa']:7.1f}%{mark} "
        else:
            row += f"{'N/A':>10s}"
    print(row)

    print()
    print("  ✓ = good (miss ≤5%, FA ≤5%)  ✗ = bad (>20%)")

    sweet = None
    for m in margins:
        ok = all(
            (stats(c, m)["miss"] or 100) <= 10 for c in conditions)
        s0 = stats(conditions[0], m)
        if ok and s0["fa"] is not None and s0["fa"] <= 10:
            sweet = m
            break

    if sweet is not None:
        print(f"\n  ► Sweet spot: margin={sweet:.2f} (miss ≤10% AND FA ≤10%)")
    else:
        print(f"\n  ► No sweet spot found — every margin trades miss for FA.")

    # ── 2. EARLY WARNING ──
    print()
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  2. EARLY WARNING: When caught, how many decisions lead? │")
    print("│     Target: p90 ≥ 50 decisions before damage.            │")
    print("└──────────────────────────────────────────────────────────┘")
    print()
    print(f"  {'Condition':16s} {'margin':>6s} {'Caught':>7s} "
          f"{'Median':>7s} {'p90':>7s} {'≥50?':>5s}")
    print(f"  {'─' * 52}")

    for cond in conditions:
        for m in margins:
            s = stats(cond, m)
            if s["n_caught"] == 0:
                continue
            leads = [c.lead_time for c in groups[(cond, m)]
                     if c.lead_time is not None and not c.miss]
            p10 = np.percentile(leads, 10)
            p90_v = s["p90"]
            ok = "  ✓" if p90_v and p90_v >= 50 else "  ✗"
            tag = " ◄" if m == 0.05 else "  "
            print(f"  {cond:16s} {m:6.2f} {s['n_caught']:6d}  "
                  f"{s['med']:6.0f}  {p90_v:6.0f} {ok}{tag}")
        print()

    # ── 3. VOLUME SENSITIVITY ──
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  3. VOLUME SENSITIVITY (production margin=0.05 only)     │")
    print("│     Small SOCs (V=50) have less signal.                  │")
    print("└──────────────────────────────────────────────────────────┘")
    print()
    print(f"  {'Condition':16s} {'V':>4s} {'Stream':>7s} "
          f"{'Miss%':>7s} {'p90_lead':>8s}")
    print(f"  {'─' * 48}")

    vg = defaultdict(list)
    for r in results:
        if r.margin == 0.05:
            vg[(r.condition, r.volume)].append(r)

    for cond in conditions:
        for vol in volumes:
            cells = vg.get((cond, vol), [])
            ev = [c for c in cells if c.miss is not None]
            if not ev:
                continue
            miss = np.mean([c.miss for c in ev]) * 100
            leads = [c.lead_time for c in ev
                     if c.lead_time is not None and not c.miss]
            p90s = f"{np.percentile(leads, 90):7.0f}" if leads else "    N/A"
            print(f"  {cond:16s} {vol:4d} {vol*10:6d}  "
                  f"{miss:6.1f}% {p90s:>8s}")
        print()

    # ── 4. BASELINE DIAGNOSTIC ──
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  4. DIAGNOSTIC: Frozen baseline + computed k              │")
    print("│     Baseline = EWMA q̄ at end of warmup (50 decisions).   │")
    print("│     k = baseline − margin.                                │")
    print("└──────────────────────────────────────────────────────────┘")
    print()

    baselines = [r.baseline_q for r in results
                 if r.baseline_q is not None and r.margin == 0.05]
    if baselines:
        print(f"  Baseline q̄ across all cells (margin=0.05):")
        print(f"    mean={np.mean(baselines):.3f}, "
              f"std={np.std(baselines):.3f}, "
              f"range=[{np.min(baselines):.3f}, {np.max(baselines):.3f}]")
        cv = np.std(baselines) / np.mean(baselines) if np.mean(baselines) > 0 else 0
        if cv > 0.15:
            print(f"    ► Moderate baseline variance (CV={cv:.2f}). "
                  f"k varies across deployments.")
        else:
            print(f"    ► Stable baselines (CV={cv:.2f}). Good.")

    # ══════════════════════════════════════════════════════════
    # DIAGNOSIS + RECOMMENDATION
    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 72)
    print("  DIAGNOSIS")
    print("=" * 72)

    ps = {}
    for cond in conditions:
        ps[cond] = stats(cond, 0.05)
    prod_fa = ps[conditions[0]]["fa"]

    all_miss_ok = all(ps[c]["miss"] is not None and ps[c]["miss"] <= 5
                      for c in conditions)
    all_lead_ok = all(ps[c]["p90"] is not None and ps[c]["p90"] >= 50
                      for c in conditions)
    fa_ok = prod_fa is not None and prod_fa <= 5

    if all_miss_ok and all_lead_ok and fa_ok:
        outcome = "A"
    elif sweet is not None:
        outcome = "B"
    else:
        outcome = "C"

    issues = []
    for cond in conditions:
        if ps[cond]["miss"] and ps[cond]["miss"] > 5:
            issues.append(
                f"  {cond} miss rate: {ps[cond]['miss']:.1f}% "
                f"(target ≤5%, N={ps[cond]['n_ev']})")
    if prod_fa and prod_fa > 5:
        issues.append(
            f"  False alarm rate: {prod_fa:.1f}% (target ≤5%)")
    if not issues:
        issues.append("  All criteria met at production margin=0.05. ✓")

    for line in issues:
        print(line)

    if issues and issues[0] != "  All criteria met at production margin=0.05. ✓":
        # Explain likely causes
        any_miss_high = any(ps[c]["miss"] and ps[c]["miss"] > 30 for c in conditions)
        if any_miss_high:
            print()
            print("  Likely cause (miss): EWMA λ=0.1 is heavy smoothing.")
            print("  Gradual degradation is absorbed into the EWMA and the")
            print("  CUSUM accumulates slowly. Sudden shifts are easier to detect.")
        if prod_fa and prod_fa > 20:
            print()
            print("  Likely cause (FA): margin=0.05 may be too tight for")
            print("  the natural EWMA fluctuation at this q̄ and volume.")

    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)
    print()

    if outcome == "A":
        print("  ✅ OUTCOME A: CLOSE THE GAP")
        print()
        print("  Production margin=0.05 meets all three criteria:")
        for cond in conditions:
            print(f"    • {cond}: miss={ps[cond]['miss']:.1f}%, "
                  f"p90 lead={ps[cond]['p90']:.0f}")
        print(f"    • False alarm: {prod_fa:.1f}%")
        print()
        print("  Action:")
        print("    1. Promote CLAIM-OLS-01 to UNCONDITIONAL")
        print("    2. Remove 'pending re-validation' note")
        print("    3. Evidence: this experiment (v3, production-faithful)")

    elif outcome == "B":
        print(f"  🔧 OUTCOME B: TUNE AND RETEST")
        print()
        print(f"  Production margin=0.05 fails, but margin={sweet:.2f}")
        print(f"  achieves miss ≤10% AND FA ≤10%.")
        print()
        print(f"  Action:")
        print(f"    1. Update gae/convergence.py: margin 0.05 → {sweet:.2f}")
        print(f"    2. Re-run experiment with tuned margin")
        print(f"    3. If passes: close CLAIM-OLS-01 with tuned params")

    else:
        print("  ⚠️  OUTCOME C: REFRAME THE CLAIM")
        print()
        print("  No margin achieves both low miss AND low false alarm.")
        print()

        best_m = None
        best_score = float("inf")
        for m in margins:
            s = stats(conditions[0], m)
            if s["miss"] is not None and s["fa"] is not None:
                score = s["miss"] + s["fa"]
                if score < best_score:
                    best_score = score
                    best_m = m

        if best_m:
            bs = stats(conditions[0], best_m)
            print(f"  Best tradeoff: margin={best_m:.2f} "
                  f"(miss={bs['miss']:.0f}%, FA={bs['fa']:.0f}%)")

        print()
        print("  Options:")
        print("    A. Reframe: 'OLSMonitor detects degradation in X% of")
        print("       cases with Y decision lead time.' Remove '0% miss'.")
        print()
        print("    B. Investigate alternatives:")
        print("       • Lower EWMA λ (0.05 → faster response, noisier)")
        print("       • Two-stage: warning at margin=0.03, alarm at 0.08")
        print("       • Combined: EWMA q̄ × override_rate product")
        print()
        print("    C. Defer to pilot: run on real production data where")
        print("       signal-to-noise ratio is known.")

    print()


# ============================================================
# CSV Export
# ============================================================

def save_csv(results: list[Result],
             fn: str = "v_ols_detect_ols_results.csv"):
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "volume", "q_bar", "margin", "seed",
                     "alarm_fired", "alarm_index", "t_damage",
                     "lead_time", "miss", "baseline_q", "k_used"])
        for r in results:
            w.writerow([
                r.condition, r.volume, r.q_bar, r.margin, r.seed,
                r.alarm_fired, r.alarm_index, r.t_damage, r.lead_time,
                r.miss,
                f"{r.baseline_q:.4f}" if r.baseline_q else "",
                f"{r.k_used:.4f}" if r.k_used else "",
            ])
    print(f"Results saved to {fn}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = Config()

    if "--quick" in sys.argv:
        cfg.n_seeds = 5
        cfg.margins = [0.03, 0.05, 0.10]
        print("── QUICK MODE: 5 seeds, 3 margins ──\n")

    results = run(cfg)
    analyze(results)
    save_csv(results)
