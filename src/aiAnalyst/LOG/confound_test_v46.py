#!/usr/bin/env python3
"""
confound_test_v46.py - is the dilution result about DILUTION, or just about smallness
and volatility?

THE PROBLEM

Share-count growth separated blowups beautifully: 14.0% in the bucket that was buying
back stock, 41.6% in the bucket that grew its share count 44% in a year, monotone
across all five, t = -5.81.

But heavy diluters are small, speculative, volatile companies - and small, speculative,
volatile companies draw down more whatever their financing looks like. So the ranking
might be nothing more than a volatility ranking, exactly like the U-shape in the V45
event study turned out to be.

THE TEST

A symmetric double sort. Bucket the observations by volatility (or size), and inside
each bucket ask whether dilution still separates blowups. Then reverse it: bucket by
dilution, and ask whether volatility still separates.

  dilution survives inside volatility buckets, volatility dies inside dilution buckets
      -> dilution is the real signal

  volatility survives, dilution dies
      -> you rediscovered beta; the dilution story is decoration

  both survive
      -> two partly independent signals, and a model could use both

  neither survives
      -> the unconditional result was an artefact of how the buckets line up

The comparison that matters is each within-bucket gap against the UNCONDITIONAL gap
printed at the top. A control that shrinks the gap to nothing has explained it away.

REQUIRES  dilution_observations.csv (from dilution_test_v46.py) and price_cache_v43/.
Runs offline.

USAGE
  python confound_test_v46.py
  python confound_test_v46.py --control vol --quantiles 5
  python confound_test_v46.py --feature n_offerings_2y
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PRICE_CACHE = "price_cache_v43"


# =============================================================================
# CONTROLS COMPUTED FROM PRICE HISTORY
# =============================================================================
def add_controls(d, price_dir):
    """
    Trailing volatility and dollar volume as of the entry date - both known before the
    observation, so neither introduces look-ahead.
    """
    vols, advs, mom = [], [], []
    cache = {}
    for t, filed in zip(d["ticker"], d["filed"]):
        if t not in cache:
            p = os.path.join(price_dir, f"{t}.pkl")
            if os.path.exists(p):
                try:
                    df = pd.read_pickle(p)
                    if getattr(df.index, "tz", None) is not None:
                        df.index = df.index.tz_convert(None)
                    cache[t] = df
                except Exception:
                    cache[t] = None
            else:
                cache[t] = None
        df = cache[t]
        if df is None:
            vols.append(np.nan); advs.append(np.nan); mom.append(np.nan); continue
        past = df[df.index <= filed]
        if len(past) < 70:
            vols.append(np.nan); advs.append(np.nan); mom.append(np.nan); continue
        r = past["Close"].pct_change().tail(60)
        vols.append(float(r.std() * np.sqrt(252) * 100))
        dv = (past["Close"] * past["Volume"]).tail(20).mean()
        advs.append(float(np.log10(dv)) if dv > 0 else np.nan)
        c = past["Close"]
        mom.append(float(c.iloc[-1] / c.iloc[-min(len(c) - 1, 126)] - 1) * 100)
    d["ann_vol_60d"] = vols
    d["log_dollar_vol"] = advs
    d["mom_6m"] = mom
    return d


# =============================================================================
# STATISTICS
# =============================================================================
def quarter_t(a_vals, a_dates, b_vals, b_dates):
    """Mean difference with the t-stat clustered by calendar quarter."""
    A = pd.DataFrame({"v": np.asarray(a_vals, float),
                      "q": pd.to_datetime(pd.Series(np.asarray(a_dates))).dt.to_period("Q")}).dropna()
    B = pd.DataFrame({"v": np.asarray(b_vals, float),
                      "q": pd.to_datetime(pd.Series(np.asarray(b_dates))).dt.to_period("Q")}).dropna()
    if A.empty or B.empty:
        return np.nan, np.nan, 0
    s = (A.groupby("q")["v"].mean() - B.groupby("q")["v"].mean()).dropna()
    if len(s) < 5 or s.std(ddof=1) == 0:
        return (float(s.mean()) if len(s) else np.nan), np.nan, len(s)
    return float(s.mean()), float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s)


def fmt_t(t):
    return f"{t:>7.2f}" if np.isfinite(t) else f"{'n/a':>7}"


def top_minus_bottom(sub, feature, outcome, q=5):
    """Top-quintile minus bottom-quintile of `feature`, measured on `outcome`."""
    s = sub[sub[feature].notna() & sub[outcome].notna()]
    if len(s) < q * 12:
        return np.nan, np.nan, 0, len(s)
    b = pd.qcut(s[feature].rank(method="first"), q, labels=False, duplicates="drop")
    lo, hi = s[b == b.min()], s[b == b.max()]
    if len(lo) < 10 or len(hi) < 10:
        return np.nan, np.nan, 0, len(s)
    diff, t, nq = quarter_t(hi[outcome].astype(float) * 100, hi["filed"],
                            lo[outcome].astype(float) * 100, lo["filed"])
    return diff, t, nq, len(s)


# =============================================================================
# SECTIONS
# =============================================================================
def single_sorts(d, outcome, q):
    print("\n" + "=" * 94)
    print(f"1) EACH FEATURE ON ITS OWN  (top-quintile minus bottom-quintile "
          f"{outcome} rate)")
    print("=" * 94)
    print("   If volatility alone separates blowups as well as dilution does, dilution")
    print("   may be adding nothing.\n")
    print(f"   {'feature':<22}{'n':>8}{'spread':>10}{'t':>8}   reading")
    feats = [("dilution_1y_pct", "share growth 1y"),
             ("n_offerings_2y", "prior offerings 2y"),
             ("runway_q", "runway (quarters)"),
             ("ann_vol_60d", "annualised vol 60d"),
             ("log_dollar_vol", "log dollar volume"),
             ("mktcap", "market cap"),
             ("mom_6m", "prior 6m return")]
    for col, label in feats:
        if col not in d.columns:
            continue
        diff, t, nq, n = top_minus_bottom(d, col, outcome, q)
        if not np.isfinite(diff):
            print(f"   {label:<22}{n:>8}       too few")
            continue
        strong = np.isfinite(t) and abs(t) > 2
        print(f"   {label:<22}{n:>8}{diff:>+9.1f}pp{fmt_t(t)}   "
              f"{'separates' if strong else 'flat'}")


def double_sort(d, feature, control, outcome, q, label_f, label_c):
    """Inside each bucket of `control`, does `feature` still separate `outcome`?"""
    s = d[d[feature].notna() & d[control].notna() & d[outcome].notna()].copy()
    if len(s) < q * 40:
        print(f"\n   {label_f} inside {label_c}: only {len(s)} rows - too few")
        return
    uncond, ut, _, _ = top_minus_bottom(s, feature, outcome, q)
    s["cb"] = pd.qcut(s[control].rank(method="first"), q, labels=False, duplicates="drop")

    print(f"\n   {label_f.upper()} INSIDE {label_c.upper()} BUCKETS")
    print(f"     unconditional spread: {uncond:+.1f}pp (t {ut:.2f})")
    print(f"     {'bucket':<9}{'n':>7}{'control median':>16}{'spread':>10}{'t':>8}")
    kept = []
    for b in sorted(s["cb"].dropna().unique()):
        g = s[s["cb"] == b]
        diff, t, nq, n = top_minus_bottom(g, feature, outcome, q=3)
        med = g[control].median()
        if not np.isfinite(diff):
            print(f"     B{int(b)+1:<8}{len(g):>7}{med:>16.2f}       too few")
            continue
        kept.append(diff)
        print(f"     B{int(b)+1:<8}{n:>7}{med:>16.2f}{diff:>+9.1f}pp{fmt_t(t)}")
    if kept:
        avg = float(np.mean(kept))
        share = avg / uncond if uncond not in (0, np.nan) and np.isfinite(uncond) else np.nan
        print(f"     average within-bucket spread: {avg:+.1f}pp "
              f"({share:.0%} of the unconditional spread)"
              if np.isfinite(share) else
              f"     average within-bucket spread: {avg:+.1f}pp")
        if np.isfinite(share):
            if share > 0.6:
                print(f"     -> survives the control: most of the effect is NOT {label_c}")
            elif share > 0.3:
                print(f"     -> partly explained by {label_c}")
            else:
                print(f"     -> explained away by {label_c}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="dilution_observations.csv")
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--feature", default="dilution_1y_pct")
    ap.add_argument("--outcome", default="blowup", choices=["blowup", "next_offering"])
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--out", default="dilution_observations_controls.csv")
    a = ap.parse_args()

    if not os.path.exists(a.obs):
        sys.exit(f"{a.obs} not found. Run dilution_test_v46.py first.")
    d = pd.read_csv(a.obs, parse_dates=["filed", "period_end"])
    print("=" * 94)
    print("CONFOUND TEST - is it dilution, or is it smallness and volatility?")
    print("=" * 94)
    print(f"  {len(d):,} observations | {d['ticker'].nunique()} tickers | "
          f"{d['filed'].min():%Y-%m} -> {d['filed'].max():%Y-%m}")
    print(f"  outcome: {a.outcome}   feature under test: {a.feature}")

    print("\n  computing trailing volatility, liquidity and momentum from prices ...")
    d = add_controls(d, a.price_cache)
    d.to_csv(a.out, index=False)
    print(f"  controls present on {d['ann_vol_60d'].notna().mean():.0%} of rows")

    single_sorts(d, a.outcome, a.quantiles)

    print("\n" + "=" * 94)
    print("2) DOUBLE SORT  -  does the feature survive each control?")
    print("=" * 94)
    for ctrl, lab in [("ann_vol_60d", "volatility"), ("mktcap", "market cap"),
                      ("log_dollar_vol", "liquidity"), ("mom_6m", "prior 6m return")]:
        if ctrl in d.columns:
            double_sort(d, a.feature, ctrl, a.outcome, a.quantiles,
                        a.feature.replace("_", " "), lab)

    print("\n" + "=" * 94)
    print("3) THE REVERSE  -  does volatility survive inside dilution buckets?")
    print("=" * 94)
    print("   If dilution survives inside volatility buckets but volatility dies inside")
    print("   dilution buckets, dilution is the real driver rather than a proxy.")
    double_sort(d, "ann_vol_60d", a.feature, a.outcome, a.quantiles,
                "volatility", a.feature.replace("_", " "))

    print("\n" + "=" * 94)
    print("HOW TO READ THIS")
    print("=" * 94)
    print("  Compare section 2 with section 3.")
    print("   dilution survives, volatility dies    -> dilution is the signal")
    print("   volatility survives, dilution dies    -> you rediscovered beta")
    print("   both survive                          -> two usable, partly independent signals")
    print("   neither survives                      -> the unconditional result was an artefact")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
