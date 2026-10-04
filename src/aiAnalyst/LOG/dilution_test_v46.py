#!/usr/bin/env python3
"""
dilution_test_v46.py - the hypothesis test, before any model.

THE CLAIM BEING TESTED

Small high-growth companies that are burning cash and have a shelf registration in
place go on to dilute, and diluting hurts the share price. If that is true, balance
sheet data visible in a 10-Q should separate the names that later draw down hard from
the ones that do not - and it should do so at a horizon long enough to act on.

This tests it directly. No model, no training. Sort observations by runway, count how
often each bucket blows up, and see whether the buckets differ.

TWO OUTCOMES, DELIBERATELY SEPARATED

  next_offering   did a 424B* price in the next 6 months?
  blowup          did the stock fall more than THRESHOLD from entry at any point in
                  the next 6 months?

Keeping them apart answers the question that decides the project:

  predicts offering, predicts blowup     -> a usable risk veto
  predicts offering, NOT blowup          -> dilution is already priced; knowing it is
                                            coming buys you nothing
  predicts neither                       -> the premise is wrong, stop here

POINT-IN-TIME DISCIPLINE

Every feature is read as of the date the figure was FILED, never the period it
covers. companyfacts repeats a period each time a later filing shows it as a
comparative, so only the FIRST filing of each period is kept - using a later one
would date a year-old number as if it were fresh. Entry is the first trading day
after that filing date. Flow items (operating cash flow) are filtered to roughly
quarterly durations, because XBRL also carries year-to-date and annual versions of
the same tag, and mixing them makes a burn rate meaningless.

REQUIRES  edgar_cache/ (from edgar_audit_v46.py) and price_cache_v43/ (from the V43
work). Runs entirely offline.

USAGE
  python dilution_test_v46.py
  python dilution_test_v46.py --horizon-days 180 --drawdown 0.30
  python dilution_test_v46.py --universe all --quantiles 5
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

EDGAR_CACHE = "edgar_cache"
PRICE_CACHE = "price_cache_v43"

OFFERING_FORMS = ("424B1", "424B2", "424B3", "424B4", "424B5", "424B7")
DILUTIVE_FORMS = ("424B5", "424B3", "424B4", "424B1")   # equity takedowns, not debt
SHELF_FORMS = ("S-3", "S-3ASR", "S-3/A", "S-1", "S-1/A")

TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding",
               "CommonStockSharesIssued"],
}


# =============================================================================
# CACHE READERS  (offline only - the audit already populated these)
# =============================================================================
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ticker_to_cik(cache_dir):
    data = load_json(os.path.join(cache_dir, "company_tickers.json"))
    if not data:
        sys.exit(f"{cache_dir}/company_tickers.json missing. Run edgar_audit_v46.py first.")
    out = {}
    for row in (data.values() if isinstance(data, dict) else data):
        t = str(row.get("ticker", "")).upper()
        if t:
            out[t] = f"{int(row['cik_str']):010d}"
    return out


def filings_for(cache_dir, cik):
    data = load_json(os.path.join(cache_dir, f"sub_{cik}.json"))
    if not data:
        return None
    frames = []

    def rows(block):
        if not block or "form" not in block:
            return None
        return pd.DataFrame({"form": block["form"],
                             "filingDate": block.get("filingDate", [])})

    r = rows((data.get("filings") or {}).get("recent"))
    if r is not None:
        frames.append(r)
    for extra in (data.get("filings") or {}).get("files", []) or []:
        more = load_json(os.path.join(cache_dir, f"sub_{extra.get('name')}"))
        r = rows((more or {}).get("filings", {}).get("recent") or more)
        if r is not None:
            frames.append(r)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    return df.dropna(subset=["filingDate"]).sort_values("filingDate")


def facts_series(facts, tag_names, flow=False):
    """
    First matching tag, as first-publication observations.

    flow=True keeps only ~quarterly durations. XBRL carries the same OCF tag as a
    quarter, a year-to-date figure and a full year; averaging across them would make
    the burn rate meaningless.
    """
    if not facts:
        return None
    for space in ("us-gaap", "dei", "ifrs-full"):
        block = (facts.get("facts") or {}).get(space) or {}
        for tag in tag_names:
            if tag not in block:
                continue
            rows = []
            for unit, entries in (block[tag].get("units") or {}).items():
                for e in entries:
                    rows.append({"start": e.get("start"), "end": e.get("end"),
                                 "val": e.get("val"), "filed": e.get("filed")})
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
            df = df.dropna(subset=["end", "filed", "val"])
            if flow:
                dur = (df["end"] - df["start"]).dt.days
                df = df[dur.between(60, 120)]        # quarterly only
            if df.empty:
                continue
            # first publication of each period, nothing later
            df = df.sort_values("filed").drop_duplicates("end", keep="first")
            return df[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)
    return None


def prices_for(ticker, price_dir):
    p = os.path.join(price_dir, f"{ticker}.pkl")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_pickle(p)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


# =============================================================================
# OBSERVATION BUILDER
# =============================================================================
def build_observations(ticker, cik, cfg):
    facts = load_json(os.path.join(cfg.edgar_cache, f"facts_{cik}.json"))
    fil = filings_for(cfg.edgar_cache, cik)
    px = prices_for(ticker, cfg.price_cache)
    if facts is None or fil is None or px is None:
        return []

    cash = facts_series(facts, TAGS["cash"])
    ocf = facts_series(facts, TAGS["ocf"], flow=True)
    shares = facts_series(facts, TAGS["shares"])
    if cash is None or cash.empty:
        return []

    offerings = fil[fil["form"].isin(DILUTIVE_FORMS)]["filingDate"].sort_values()
    shelves = fil[fil["form"].isin(SHELF_FORMS)]["filingDate"].sort_values()
    off_arr = offerings.to_numpy()
    shelf_arr = shelves.to_numpy()

    hz = cfg.horizon_days
    rows = []
    for _, row in cash.iterrows():
        filed = row["filed"]
        # entry on the first trading day AFTER the filing became public
        fwd = px[px.index > filed]
        if len(fwd) < 20:
            continue
        entry = float(fwd["Close"].iloc[0])
        if entry <= 0:
            continue
        window = fwd[fwd.index <= filed + pd.Timedelta(days=hz)]
        if len(window) < 40:                      # need most of the horizon to exist
            continue

        # ---- features, all known at `filed` ----
        cash_val = float(row["val"])
        burn = np.nan
        if ocf is not None and not ocf.empty:
            past = ocf[ocf["filed"] <= filed]
            if len(past):
                recent = past.tail(4)["val"].astype(float)
                q = float(recent.mean())
                burn = -q if q < 0 else 0.0       # 0 = generating cash, not burning
        runway = np.nan
        if np.isfinite(burn):
            runway = 99.0 if burn <= 0 else cash_val / burn   # quarters of cash left

        sh_now = sh_prev = np.nan
        if shares is not None and not shares.empty:
            past = shares[shares["filed"] <= filed]
            if len(past):
                sh_now = float(past["val"].iloc[-1])
                old = past[past["end"] <= past["end"].iloc[-1] - pd.Timedelta(days=330)]
                if len(old):
                    sh_prev = float(old["val"].iloc[-1])
        dilution_1y = (sh_now / sh_prev - 1.0) if (np.isfinite(sh_now)
                                                   and np.isfinite(sh_prev)
                                                   and sh_prev > 0) else np.nan

        n_off_2y = int(((off_arr > np.datetime64(filed - pd.Timedelta(days=730)))
                        & (off_arr <= np.datetime64(filed))).sum())
        has_shelf = bool(((shelf_arr > np.datetime64(filed - pd.Timedelta(days=1095)))
                          & (shelf_arr <= np.datetime64(filed))).sum())
        mktcap = sh_now * entry if np.isfinite(sh_now) else np.nan

        # ---- outcomes ----
        closes = window["Close"].to_numpy(dtype=float)
        lows = window["Low"].to_numpy(dtype=float)
        fwd_ret = (closes[-1] - entry) / entry * 100.0
        fwd_mdd = (float(np.min(lows)) - entry) / entry * 100.0
        next_off = bool(((off_arr > np.datetime64(filed))
                         & (off_arr <= np.datetime64(filed + pd.Timedelta(days=hz)))).sum())

        rows.append({
            "ticker": ticker, "filed": filed, "period_end": row["end"],
            "entry": entry, "cash": cash_val, "quarterly_burn": burn,
            "runway_q": runway, "dilution_1y_pct": dilution_1y * 100 if np.isfinite(dilution_1y) else np.nan,
            "n_offerings_2y": n_off_2y, "has_shelf": has_shelf,
            "mktcap": mktcap,
            "cash_to_mktcap": cash_val / mktcap if np.isfinite(mktcap) and mktcap > 0 else np.nan,
            "fwd_ret": fwd_ret, "fwd_mdd": fwd_mdd,
            "blowup": fwd_mdd <= -cfg.drawdown * 100,
            "next_offering": next_off,
        })
    return rows


# =============================================================================
# STATISTICS
# =============================================================================
def quarter_cluster_t(a_vals, a_dates, b_vals, b_dates):
    """Difference in means with the t-stat clustered by calendar quarter."""
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
    return f"{t:>8.2f}" if np.isfinite(t) else f"{'n/a':>8}"


def bucket_report(d, feature, q, label, outcomes):
    sub = d[d[feature].notna()].copy()
    if len(sub) < q * 25:
        print(f"\n  {label}: only {len(sub)} observations with {feature} - too few")
        return
    sub["b"] = pd.qcut(sub[feature].rank(method="first"), q, labels=False)
    print(f"\n  SORTED BY {label}   ({len(sub):,} observations)")
    hdr = f"    {'bucket':<8}{'n':>7}{feature[:14]:>16}"
    for o, _ in outcomes:
        hdr += f"{o[:13]:>15}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for b in range(q):
        g = sub[sub["b"] == b]
        line = f"    Q{b+1:<7}{len(g):>7}{g[feature].median():>16.2f}"
        for o, kind in outcomes:
            v = g[o].mean() * (100 if kind == "rate" else 1)
            line += f"{v:>14.1f}{'%' if kind == 'rate' else ''}" if kind == "rate" else f"{v:>+14.2f}%"
        print(line)
    line = f"    {'ALL':<8}{len(sub):>7}{sub[feature].median():>16.2f}"
    for o, kind in outcomes:
        v = sub[o].mean() * (100 if kind == "rate" else 1)
        line += f"{v:>14.1f}{'%' if kind == 'rate' else ''}" if kind == "rate" else f"{v:>+14.2f}%"
    print(line)

    lo, hi = sub[sub["b"] == 0], sub[sub["b"] == q - 1]
    print(f"\n    Q1 minus Q{q}, t clustered by calendar quarter:")
    for o, kind in outcomes:
        a = lo[o].astype(float) * (100 if kind == "rate" else 1)
        b_ = hi[o].astype(float) * (100 if kind == "rate" else 1)
        diff, t, nq = quarter_cluster_t(a, lo["filed"], b_, hi["filed"])
        verdict = "SEPARATES" if np.isfinite(t) and abs(t) > 2 else "flat"
        print(f"      {o:<16}{diff:>+9.2f}{'pp' if kind == 'rate' else '%'}"
              f"{fmt_t(t)}   {nq} quarters   {verdict}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="deploy", choices=["deploy", "all"])
    ap.add_argument("--edgar-cache", default=EDGAR_CACHE)
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--horizon-days", type=int, default=180)
    ap.add_argument("--drawdown", type=float, default=0.30,
                    help="fall from entry that counts as a blowup")
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--out", default="dilution_observations.csv")
    cfg = ap.parse_args()

    try:
        from pillar2_v43_universe import DEPLOY_UNIVERSE, load_training_universe
        tickers = (DEPLOY_UNIVERSE if cfg.universe == "deploy"
                   else load_training_universe(include_deploy=True))
    except ImportError:
        sys.exit("Run next to pillar2_v43_universe.py.")

    print("=" * 94)
    print("DILUTION / BLOWUP HYPOTHESIS TEST")
    print("=" * 94)
    print(f"  universe: {cfg.universe} ({len(tickers)} tickers)")
    print(f"  horizon:  {cfg.horizon_days} calendar days from the filing date")
    print(f"  blowup:   a fall of {cfg.drawdown:.0%} or more from entry at any point")
    print(f"  entry:    first trading day AFTER the 10-Q was filed\n")

    cik_map = ticker_to_cik(cfg.edgar_cache)
    rows, skipped = [], Counter()
    for i, t in enumerate(tickers, 1):
        cik = cik_map.get(t.upper())
        if not cik:
            skipped["no CIK"] += 1
            continue
        got = build_observations(t, cik, cfg)
        if not got:
            skipped["no usable observations"] += 1
        rows.extend(got)
        if i % 20 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {len(rows):,} observations")

    if not rows:
        sys.exit("No observations built. Check that both caches are populated.")
    d = pd.DataFrame(rows)
    d.to_csv(cfg.out, index=False)

    print("\n" + "=" * 94)
    print("SAMPLE")
    print("=" * 94)
    print(f"  {len(d):,} observations | {d['ticker'].nunique()} tickers | "
          f"{d['filed'].min():%Y-%m} -> {d['filed'].max():%Y-%m}")
    print(f"  blowup rate ({cfg.drawdown:.0%} drawdown in {cfg.horizon_days}d): "
          f"{d['blowup'].mean():.1%}")
    print(f"  offering in the next {cfg.horizon_days}d: {d['next_offering'].mean():.1%}")
    print(f"  mean forward return: {d['fwd_ret'].mean():+.2f}%  |  "
          f"mean max drawdown: {d['fwd_mdd'].mean():+.2f}%")
    for c in ("runway_q", "dilution_1y_pct", "cash_to_mktcap"):
        print(f"  {c:<18} present on {d[c].notna().mean():.0%} of observations")
    if skipped:
        print(f"  skipped: {dict(skipped)}")

    outcomes = [("next_offering", "rate"), ("blowup", "rate"),
                ("fwd_ret", "pct"), ("fwd_mdd", "pct")]

    print("\n" + "=" * 94)
    print("DOES THE BALANCE SHEET SEPARATE THEM?")
    print("=" * 94)
    print("  Q1 = lowest value of the sort feature. For runway, Q1 is the most")
    print("  distressed, so a real effect shows MORE offerings and MORE blowups there.")
    for feat, label in [("runway_q", "RUNWAY (quarters of cash at current burn)"),
                        ("cash_to_mktcap", "CASH / MARKET CAP"),
                        ("dilution_1y_pct", "SHARE COUNT GROWTH over the prior year"),
                        ("n_offerings_2y", "PRIOR OFFERINGS in the last 2 years")]:
        if feat in d.columns:
            bucket_report(d, feat, cfg.quantiles, label, outcomes)

    print("\n" + "=" * 94)
    print("SHELF REGISTRATION  (a filed S-3 is stated intent to sell stock)")
    print("=" * 94)
    for flag, name in [("has_shelf", "shelf on file (last 3y)")]:
        a, b_ = d[d[flag]], d[~d[flag]]
        if len(a) < 30 or len(b_) < 30:
            print(f"  {name}: too few on one side ({len(a)} vs {len(b_)})")
            continue
        print(f"  {name}: {len(a):,} with, {len(b_):,} without")
        for o, kind in outcomes:
            av = a[o].astype(float) * (100 if kind == "rate" else 1)
            bv = b_[o].astype(float) * (100 if kind == "rate" else 1)
            diff, t, nq = quarter_cluster_t(av, a["filed"], bv, b_["filed"])
            print(f"    {o:<16}{av.mean():>8.2f} vs {bv.mean():>8.2f}"
                  f"{diff:>+9.2f}{fmt_t(t)}   "
                  f"{'SEPARATES' if np.isfinite(t) and abs(t) > 2 else 'flat'}")

    print("\n" + "=" * 94)
    print("HOW TO READ THIS")
    print("=" * 94)
    print("  predicts next_offering AND blowup -> a usable risk veto, worth modelling")
    print("  predicts next_offering but NOT blowup -> dilution is already in the price;")
    print("     knowing it is coming earns nothing, and the project stops here")
    print("  predicts neither -> the premise is wrong, stop here")
    print(f"\n  wrote {cfg.out}")


if __name__ == "__main__":
    main()
