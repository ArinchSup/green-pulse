#!/usr/bin/env python3
"""
voladj_test_v46.py - does anything predict drawdowns BEYOND what volatility
mechanically implies?

THE PROBLEM WITH THE PREVIOUS TEST

"Blowup" was defined as a 30% fall. A stock with 100% annualised volatility touches
-30% within six months far more often than one at 20%, with no difference in drift at
all - purely from diffusion. So "volatility predicts blowups" is close to a tautology,
and the confound test that used it was stacked against dilution from the start. It
reported that volatility survives inside dilution buckets (64%) while dilution mostly
dies inside volatility buckets (25%), which may say more about the definition than
about the world.

THE FIX

Normalise the drawdown by the stock's OWN expected move over the horizon.

    sigma_H   = annualised vol x sqrt(horizon / 365)
    mdd_z     = forward max drawdown / sigma_H
    mdd_ratio = forward max drawdown / (0.798 x sigma_H)

0.798 = sqrt(2/pi), the expected minimum of a driftless random walk over [0, T] in
units of sigma_H. So mdd_ratio = 1.0 means "fell exactly as far as a coin-flip stock
of this volatility would be expected to", and 2.0 means "fell twice as far as its own
volatility can explain".

Now the question is well posed: after removing what volatility already accounts for,
does share-count growth - or anything else - still predict an abnormal fall?

THE VALIDATION THAT MAKES THIS TRUSTWORTHY

Section 1 checks that the normalisation worked, by showing the vol-adjusted blowup
rate ACROSS volatility quintiles. It has to come out roughly flat. If it is still
steeply sloped, the adjustment failed and nothing below it means anything. The raw
rate is printed alongside so the before/after is visible.

REQUIRES  dilution_observations_controls.csv (written by confound_test_v46.py).
Runs offline.

USAGE
  python voladj_test_v46.py
  python voladj_test_v46.py --k 1.5 --quantiles 5
  python voladj_test_v46.py --feature n_offerings_2y
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

EXP_MIN = np.sqrt(2.0 / np.pi)     # 0.798: E[min] of a driftless walk, in sigma_H


# =============================================================================
# STATISTICS
# =============================================================================
def quarter_t(a_vals, a_dates, b_vals, b_dates):
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


def spread(sub, feature, outcome, scale, q=5):
    s = sub[sub[feature].notna() & sub[outcome].notna()]
    if len(s) < q * 12:
        return np.nan, np.nan, len(s)
    b = pd.qcut(s[feature].rank(method="first"), q, labels=False, duplicates="drop")
    lo, hi = s[b == b.min()], s[b == b.max()]
    if len(lo) < 10 or len(hi) < 10:
        return np.nan, np.nan, len(s)
    diff, t, _ = quarter_t(hi[outcome].astype(float) * scale, hi["filed"],
                           lo[outcome].astype(float) * scale, lo["filed"])
    return diff, t, len(s)


# =============================================================================
# SECTIONS
# =============================================================================
def validate(d, q):
    print("\n" + "=" * 94)
    print("1) DID THE ADJUSTMENT WORK?  (blowup rate across VOLATILITY quintiles)")
    print("=" * 94)
    print("   The raw rate should slope steeply - that is the tautology. The")
    print("   vol-adjusted rate has to come out roughly FLAT, or nothing below counts.\n")
    s = d[d["ann_vol_60d"].notna()].copy()
    s["vb"] = pd.qcut(s["ann_vol_60d"].rank(method="first"), q, labels=False)
    print(f"   {'vol bucket':<12}{'n':>7}{'ann vol':>10}{'raw blowup':>13}"
          f"{'vol-adj blowup':>17}{'mean mdd_ratio':>17}")
    for b in range(q):
        g = s[s["vb"] == b]
        print(f"   B{b+1:<11}{len(g):>7}{g['ann_vol_60d'].median():>9.0f}%"
              f"{g['blowup'].mean()*100:>12.1f}%{g['blowup_vs'].mean()*100:>16.1f}%"
              f"{g['mdd_ratio'].mean():>17.2f}")
    raw = s.groupby("vb")["blowup"].mean() * 100
    adj = s.groupby("vb")["blowup_vs"].mean() * 100
    print(f"\n   raw spread B{q} - B1:        {raw.iloc[-1] - raw.iloc[0]:+.1f}pp")
    print(f"   vol-adjusted spread B{q} - B1: {adj.iloc[-1] - adj.iloc[0]:+.1f}pp")
    ok = abs(adj.iloc[-1] - adj.iloc[0]) < 12
    print(f"   -> {'adjustment worked, the tautology is removed' if ok else 'STILL SLOPED: the adjustment did not fully work; read on with caution'}")
    return ok


def feature_table(d, outcome, scale, unit, q, title):
    print(f"\n   {title}")
    print(f"     {'feature':<22}{'n':>8}{'spread':>11}{'t':>8}   reading")
    for col, label in [("dilution_1y_pct", "share growth 1y"),
                       ("n_offerings_2y", "prior offerings 2y"),
                       ("runway_q", "runway (quarters)"),
                       ("cash_to_mktcap", "cash / market cap"),
                       ("has_shelf", "shelf on file"),
                       ("ann_vol_60d", "annualised vol 60d"),
                       ("mktcap", "market cap"),
                       ("log_dollar_vol", "log dollar volume")]:
        if col not in d.columns:
            continue
        sub = d.copy()
        if col == "has_shelf":            # boolean: compare the two groups directly
            a, b_ = sub[sub[col] == True], sub[sub[col] == False]   # noqa: E712
            if len(a) < 30 or len(b_) < 30:
                continue
            diff, t, _ = quarter_t(a[outcome].astype(float) * scale, a["filed"],
                                   b_[outcome].astype(float) * scale, b_["filed"])
            n = len(a) + len(b_)
        else:
            diff, t, n = spread(sub, col, outcome, scale, q)
        if not np.isfinite(diff):
            continue
        strong = np.isfinite(t) and abs(t) > 2
        print(f"     {label:<22}{n:>8}{diff:>+10.2f}{unit}{fmt_t(t)}   "
              f"{'SEPARATES' if strong else 'flat'}")


def double_sort(d, feature, control, outcome, scale, q, lf, lc):
    s = d[d[feature].notna() & d[control].notna() & d[outcome].notna()].copy()
    if len(s) < q * 40:
        print(f"\n   {lf} inside {lc}: too few rows ({len(s)})")
        return
    uncond, ut, _ = spread(s, feature, outcome, scale, q)
    s["cb"] = pd.qcut(s[control].rank(method="first"), q, labels=False, duplicates="drop")
    print(f"\n   {lf.upper()} INSIDE {lc.upper()} BUCKETS")
    print(f"     unconditional spread: {uncond:+.2f} (t {ut:.2f})")
    print(f"     {'bucket':<9}{'n':>7}{'control median':>16}{'spread':>11}{'t':>8}")
    kept = []
    for b in sorted(s["cb"].dropna().unique()):
        g = s[s["cb"] == b]
        diff, t, n = spread(g, feature, outcome, scale, q=3)
        if not np.isfinite(diff):
            print(f"     B{int(b)+1:<8}{len(g):>7}{g[control].median():>16.2f}    too few")
            continue
        kept.append(diff)
        print(f"     B{int(b)+1:<8}{n:>7}{g[control].median():>16.2f}"
              f"{diff:>+10.2f}{fmt_t(t)}")
    if kept and np.isfinite(uncond) and uncond != 0:
        avg = float(np.mean(kept))
        share = avg / uncond
        print(f"     average within-bucket: {avg:+.2f} ({share:.0%} of unconditional)")
        print(f"     -> {'survives' if share > 0.6 else 'partly explained' if share > 0.3 else 'explained away'} by {lc}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="dilution_observations_controls.csv")
    ap.add_argument("--horizon-days", type=int, default=180)
    ap.add_argument("--k", type=float, default=1.5,
                    help="vol-adjusted blowup = fell more than k x sigma_H")
    ap.add_argument("--feature", default="dilution_1y_pct")
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--out", default="voladj_observations.csv")
    a = ap.parse_args()

    if not os.path.exists(a.obs):
        sys.exit(f"{a.obs} not found. Run confound_test_v46.py first - it adds the "
                 f"volatility column this needs.")
    d = pd.read_csv(a.obs, parse_dates=["filed", "period_end"])
    if "ann_vol_60d" not in d.columns:
        sys.exit("No ann_vol_60d column. Re-run confound_test_v46.py.")

    # sigma over the holding horizon, as a fraction
    sigma_h = (d["ann_vol_60d"] / 100.0) * np.sqrt(a.horizon_days / 365.0)
    d["sigma_h_pct"] = sigma_h * 100
    d = d[d["sigma_h_pct"] > 1e-6].copy()
    d["mdd_z"] = (d["fwd_mdd"] / 100.0) / (d["sigma_h_pct"] / 100.0)
    d["ret_z"] = (d["fwd_ret"] / 100.0) / (d["sigma_h_pct"] / 100.0)
    d["mdd_ratio"] = d["mdd_z"].abs() / EXP_MIN
    d["blowup_vs"] = d["mdd_z"] <= -a.k

    print("=" * 94)
    print("VOL-ADJUSTED DRAWDOWN TEST")
    print("=" * 94)
    print(f"  {len(d):,} observations | {d['ticker'].nunique()} tickers | "
          f"{d['filed'].min():%Y-%m} -> {d['filed'].max():%Y-%m}")
    print(f"  horizon {a.horizon_days}d | sigma_H median {d['sigma_h_pct'].median():.1f}%")
    print(f"  raw blowup (30% fall):        {d['blowup'].mean():.1%}")
    print(f"  vol-adjusted blowup (>{a.k}x sigma_H): {d['blowup_vs'].mean():.1%}")
    print(f"  mean mdd_ratio: {d['mdd_ratio'].mean():.2f}  "
          f"(1.00 = exactly what a coin-flip stock of that volatility would do)")
    print("\n  threshold sweep, so you can see how the base rate moves:")
    for k in (1.0, 1.25, 1.5, 1.75, 2.0):
        print(f"    >{k:.2f}x sigma_H -> {(d['mdd_z'] <= -k).mean():.1%}")

    validate(d, a.quantiles)

    print("\n" + "=" * 94)
    print("2) WHAT PREDICTS AN ABNORMAL FALL?")
    print("=" * 94)
    print("   Volatility should now be near flat - it has been divided out. Anything")
    print("   else that still separates is predicting beyond what volatility explains.")
    feature_table(d, "blowup_vs", 100, "pp", a.quantiles,
                  "VOL-ADJUSTED BLOWUP RATE (top quintile minus bottom)")
    feature_table(d, "mdd_z", 1, "  ", a.quantiles,
                  "DRAWDOWN IN SIGMAS (mdd_z; more negative = worse)")
    feature_table(d, "ret_z", 1, "  ", a.quantiles,
                  "FORWARD RETURN IN SIGMAS (ret_z; the alpha question)")

    print("\n" + "=" * 94)
    print("3) DOES THE FEATURE SURVIVE VOLATILITY NOW?")
    print("=" * 94)
    for out, sc, lab in [("blowup_vs", 100, "vol-adjusted blowup"),
                         ("mdd_z", 1, "drawdown in sigmas")]:
        print(f"\n   --- outcome: {lab} ---")
        double_sort(d, a.feature, "ann_vol_60d", out, sc, a.quantiles,
                    a.feature.replace("_", " "), "volatility")

    d.to_csv(a.out, index=False)
    print("\n" + "=" * 94)
    print("HOW TO READ THIS")
    print("=" * 94)
    print("  Section 1 flat -> the tautology is gone and section 2 is meaningful.")
    print("  In section 2, volatility itself should now be flat. If share growth still")
    print("  SEPARATES on blowup_vs or mdd_z, it predicts falls that volatility alone")
    print("  does not explain - a real finding, and the one thing left worth modelling.")
    print("  If it goes flat too, the dilution hypothesis is closed.")
    print("  ret_z is the separate, harder question: does anything predict DIRECTION")
    print("  once magnitude is divided out? Expect flat, on this project's record.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
