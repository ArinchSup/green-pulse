#!/usr/bin/env python3
"""
analyze_events_v45.py - re-analyse a saved V45 event file.

WHY THIS EXISTS

1. The t-stat in the first version of class_pillar2_event_gen_v45.py was wrong. It
   pooled the Q5 values with the NEGATED Q1 values and averaged them within each
   month. Q5 and Q1 counts differ within a month about 80% of the time, so that
   average is a count-weighted blend rather than the spread: it shrinks the estimate
   and can flip its sign when the effect is weak. That is why the first run printed a
   +0.34% spread with t = -0.36. Here the spread is computed WITHIN each month and
   then t-tested across months.

2. The first run showed EVERY quintile drifting up - Q1 (the worst reactions) gained
   +2.52% at 60 bars. Benchmark adjustment should remove the market, so a uniformly
   positive CAR points at the universe itself: these are companies still listed in
   2026, so they outperformed on average. A long-only trade inherits that drift for
   free, which means the question is not "is Q5's CAR positive" but "does Q5 beat the
   average event". Both are reported.

3. You can only go long, so Q5 minus Q1 is not a strategy you can run. Q5 minus the
   all-event average is.

4. Only ~29% of detected events had quarterly spacing, so the sample is a mix of
   earnings and everything else. The likely-earnings subset is reported separately,
   because that is where drift is documented.

USAGE
  python analyze_events_v45.py events_v45_all_20050101.csv.gz
  python analyze_events_v45.py events_v45_deploy_20050101.csv.gz --quantiles 10
  python analyze_events_v45.py events_v45_all_20050101.csv.gz --horizon 20
"""
import argparse
import sys

import numpy as np
import pandas as pd

FWD_BARS = (1, 5, 10, 20, 60)


# =============================================================================
# STATISTICS
# =============================================================================
def month_spread_t(top_v, top_d, bot_v, bot_d):
    """
    Mean difference and its t-stat, clustering by calendar month. The difference is
    formed INSIDE each month (mean of one group minus mean of the other, using only
    months containing both), then t-tested across months, so unequal group sizes
    within a month cannot distort it.
    """
    a = pd.DataFrame({"v": np.asarray(top_v, float),
                      "m": pd.to_datetime(pd.Series(np.asarray(top_d))).dt.to_period("M")}).dropna()
    b = pd.DataFrame({"v": np.asarray(bot_v, float),
                      "m": pd.to_datetime(pd.Series(np.asarray(bot_d))).dt.to_period("M")}).dropna()
    if a.empty or b.empty:
        return np.nan, np.nan, 0
    s = (a.groupby("m")["v"].mean() - b.groupby("m")["v"].mean()).dropna()
    if len(s) < 5:
        return float(s.mean()) if len(s) else np.nan, np.nan, len(s)
    sd = s.std(ddof=1)
    t = float(s.mean() / (sd / np.sqrt(len(s)))) if sd else np.nan
    return float(s.mean()), t, len(s)


def fmt_t(t):
    return f"{t:>8.2f}" if np.isfinite(t) else f"{'n/a':>8}"


def verdict(spread, t):
    if not np.isfinite(t) or abs(t) <= 2:
        return "flat"
    return "drift" if spread > 0 else "reversal"


# =============================================================================
# SECTIONS
# =============================================================================
def quintile_table(d, q, label):
    print(f"\n  {label}: {len(d):,} events, "
          f"{d['_date'].dt.year.min()}-{d['_date'].dt.year.max()}")
    if len(d) < q * 25:
        print("    too few events to bucket")
        return None
    d = d.copy()
    d["bucket"] = pd.qcut(d["event_car"].rank(method="first"), q, labels=False)

    hdr = f"    {'bucket':<8}{'n':>7}{'event CAR':>12}"
    for h in FWD_BARS:
        hdr += f"{'CAR+' + str(h):>11}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for b in range(q):
        g = d[d["bucket"] == b]
        line = f"    Q{b + 1:<7}{len(g):>7}{g['event_car'].mean():>+11.2f}%"
        for h in FWD_BARS:
            line += f"{g[f'car_{h}'].mean():>+10.2f}%"
        print(line)
    line = f"    {'ALL':<8}{len(d):>7}{d['event_car'].mean():>+11.2f}%"
    for h in FWD_BARS:
        line += f"{d[f'car_{h}'].mean():>+10.2f}%"
    print(line)
    return d


def spread_tests(d, q, name):
    top, bot = d[d["bucket"] == q - 1], d[d["bucket"] == 0]
    print(f"\n    Q{q} MINUS Q1  (survivorship-free, but not long-only tradeable)")
    print(f"      {'horizon':>8}{'spread':>10}{'t':>8}{'months':>8}   reading")
    for h in FWD_BARS:
        sp, t, nm = month_spread_t(top[f"car_{h}"], top["_date"],
                                   bot[f"car_{h}"], bot["_date"])
        print(f"      {h:>8}{sp:>+9.2f}%{fmt_t(t)}{nm:>8}   {verdict(sp, t)}")

    print(f"\n    Q{q} MINUS ALL EVENTS  (what a long-only trade actually adds)")
    print(f"      {'horizon':>8}{'all':>9}{'Q' + str(q):>9}{'lift':>9}{'t':>8}   reading")
    for h in FWD_BARS:
        sp, t, _ = month_spread_t(top[f"car_{h}"], top["_date"],
                                  d[f"car_{h}"], d["_date"])
        print(f"      {h:>8}{d[f'car_{h}'].mean():>+8.2f}%{top[f'car_{h}'].mean():>+8.2f}%"
              f"{sp:>+8.2f}%{fmt_t(t)}   {verdict(sp, t)}")


def by_year(d, q, horizon):
    print(f"\n    BY YEAR at {horizon} bars")
    top = d[d["bucket"] == q - 1]
    for label, fn in [(f"Q{q} - Q1 ",
                       lambda g: (g[g["bucket"] == q - 1][f"car_{horizon}"].mean()
                                  - g[g["bucket"] == 0][f"car_{horizon}"].mean())),
                      (f"Q{q} - all",
                       lambda g: (g[g["bucket"] == q - 1][f"car_{horizon}"].mean()
                                  - g[f"car_{horizon}"].mean()))]:
        parts, pos, n = [], 0, 0
        for y, g in d.groupby(d["_date"].dt.year):
            v = fn(g)
            if not np.isfinite(v):
                continue
            parts.append(f"{int(y)}: {v:+.1f}%")
            pos += int(v > 0)
            n += 1
        if not parts:
            continue
        print(f"      [{label}]")
        for i in range(0, len(parts), 6):
            print("        " + "  ".join(parts[i:i + 6]))
        print(f"        positive in {pos}/{n} years ({pos / max(n, 1):.0%})"
              + ("   <- a coin flip" if n and 0.4 <= pos / n <= 0.6 else ""))


def direction_split(d, horizon):
    """
    Long-only reality check. Going long after a POSITIVE reaction is the trade you can
    actually place; the negative side would need a short.
    """
    print(f"\n    LONG-ONLY SLICES at {horizon} bars (lift over the all-event mean)")
    base_v, base_d = d[f"car_{horizon}"], d["_date"]
    print(f"      {'slice':<26}{'n':>7}{'CAR':>9}{'lift':>9}{'t':>8}")
    slices = [("all positive reactions", d[d["direction"] == "positive"]),
              ("all negative reactions", d[d["direction"] == "negative"])]
    if "event_sigma_mult" in d.columns:
        cut = d["event_sigma_mult"].quantile(0.75)
        slices.append((f"biggest surprises (>{cut:.1f}s)", d[d["event_sigma_mult"] >= cut]))
    if "event_vol_ratio" in d.columns:
        cut = d["event_vol_ratio"].quantile(0.75)
        slices.append((f"heaviest volume (>{cut:.1f}x)", d[d["event_vol_ratio"] >= cut]))
    if "pre_drift_60" in d.columns:
        slices.append(("gap up, was not extended",
                       d[(d["event_gap"] > 0) & (d["pre_drift_60"] < 0)]))
    for name, g in slices:
        if len(g) < 50:
            print(f"      {name:<26}{len(g):>7}   too few")
            continue
        sp, t, _ = month_spread_t(g[f"car_{horizon}"], g["_date"], base_v, base_d)
        print(f"      {name:<26}{len(g):>7}{g[f'car_{horizon}'].mean():>+8.2f}%"
              f"{sp:>+8.2f}%{fmt_t(t)}")


def drift_or_volatility(d, q, horizon):
    """
    THE DECISIVE TEST for a U-shaped quintile table.

    Sorting by SIGNED event CAR asks "do winners keep winning" - that is drift.
    Sorting by ABSOLUTE event CAR asks "do big movers keep moving" - that is a
    volatility premium, because a stock that just moved 15% is a volatile stock, and
    volatile names drifted up across this sample.

    If the absolute sort gives a cleaner, stronger ordering than the signed sort, the
    apparent drift is volatility wearing drift's clothes. Real PEAD is monotone in the
    SIGNED reaction, with the bottom bucket going DOWN.
    """
    print(f"\n    IS IT DRIFT, OR VOLATILITY?  (top bucket minus all, at {horizon} bars)")
    print("      signed sort = drift hypothesis | absolute sort = volatility hypothesis")
    print(f"      {'sorted by':<22}{'Q1':>9}{'Q' + str(q):>9}{'lift':>9}{'t':>8}   monotone")
    for name, key in [("signed event CAR", d["event_car"]),
                      ("|event CAR|", d["event_car"].abs())]:
        g = d.copy()
        g["b"] = pd.qcut(key.rank(method="first"), q, labels=False)
        top, bot = g[g["b"] == q - 1], g[g["b"] == 0]
        means = [g[g["b"] == i][f"car_{horizon}"].mean() for i in range(q)]
        mono = "yes" if all(np.diff(means) >= -0.05) else "no (U-shape)"
        sp, t, _ = month_spread_t(top[f"car_{horizon}"], top["_date"],
                                  g[f"car_{horizon}"], g["_date"])
        print(f"      {name:<22}{means[0]:>+8.2f}%{means[-1]:>+8.2f}%"
              f"{sp:>+8.2f}%{fmt_t(t)}   {mono}")

    # Show the volatility of each signed bucket. If Q1 and Q5 are BOTH the most
    # volatile, the U-shape in returns is just the U-shape in volatility.
    vol_col = next((c for c in ("atr_pct", "vol_20", "vol_60") if c in d.columns), None)
    if vol_col:
        g = d.copy()
        g["b"] = pd.qcut(g["event_car"].rank(method="first"), q, labels=False)
        vals = [g[g["b"] == i][vol_col].mean() for i in range(q)]
        print(f"\n      mean {vol_col} by SIGNED bucket: "
              + "  ".join(f"Q{i+1} {v:.3f}" for i, v in enumerate(vals)))
        print("      If Q1 and Q5 are the most volatile, the U-shape in returns is the")
        print("      U-shape in volatility, not a reaction to the news.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="events_v45_*.csv.gz from the generator")
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=20,
                    help="horizon used for the by-year and long-only slices")
    a = ap.parse_args()

    try:
        d = pd.read_csv(a.path)
    except Exception as e:
        sys.exit(f"Could not read {a.path}: {type(e).__name__}: {e}")
    d["_date"] = pd.to_datetime(d["event_date"])
    d = d.dropna(subset=["event_car"]).copy()
    if "direction" not in d.columns:
        d["direction"] = np.where(d["event_car"] >= 0, "positive", "negative")

    print("=" * 88)
    print(f"V45 EVENT RE-ANALYSIS   {a.path}")
    print("=" * 88)
    print(f"  {len(d):,} events | {d['ticker'].nunique()} tickers | "
          f"{d['_date'].dt.year.nunique()} years | "
          f"{d['_date'].min():%Y-%m-%d} -> {d['_date'].max():%Y-%m-%d}")
    print("\n  The t-stat here differs from the generator's first version, which was")
    print("  wrong: it blended Q5 with negated Q1 inside each month instead of taking")
    print("  the spread, so a positive spread could print a negative t.")

    q = a.quantiles
    cuts = [("ALL EVENTS", d)]

    # Likely-earnings subset: drift is documented for scheduled reports, and only ~29%
    # of detected events were spaced like quarterly reports, so the rest dilute it.
    if "bars_since_prev_event" in d.columns:
        quarterly = d[d["bars_since_prev_event"].between(50, 76)]
        if len(quarterly) >= q * 25:
            cuts.append(("LIKELY EARNINGS (50-76 bars since the last event)", quarterly))

    for label, sub in cuts:
        print("\n" + "=" * 88)
        print(label)
        print("=" * 88)
        b = quintile_table(sub, q, "events")
        if b is None:
            continue
        spread_tests(b, q, label)
        by_year(b, q, a.horizon)

    print("\n" + "=" * 88)
    print("LONG-ONLY SLICES  (all events)")
    print("=" * 88)
    direction_split(d, a.horizon)

    print("\n" + "=" * 88)
    print("DRIFT OR VOLATILITY?  (all events)")
    print("=" * 88)
    for h in (a.horizon, 60):
        drift_or_volatility(d, q, h)

    print("\n" + "=" * 88)
    print("  Read the 'Q5 minus all events' block, not 'Q5 minus Q1'. The spread is")
    print("  survivorship-free but needs a short leg you will not place; the lift over")
    print("  the all-event average is the number a long-only trade actually earns.")
    print("  A tradeable effect needs |t| > 2 AND most years positive.")


if __name__ == "__main__":
    main()
