#!/usr/bin/env python3
"""
survivorship_universe_v49.py - reconstruct who was actually listed on each date, and
measure how much your ticker list distorts the picture.

THE PROBLEM THIS ADDRESSES

Every result in this project rests on 327 tickers chosen in 2026. A company in the
2011 sample is one that survived fifteen more years. Features that predict good
forward returns - positive momentum, moderate volatility, size - also predict
SURVIVAL, so a model trained on that sample can partly be learning "this company will
still exist in 2026", which is information nobody had in 2011.

That matters right now because of the walk-forward result:

    test window    IC
    2011-14     +0.0547   <- strongest, and needs 15 years of forward survival
    2014-17     +0.0261
    2017-20     +0.0101
    2020-23     +0.0200
    2023-26     -0.0129   <- weakest, needs 3 years of forward survival

Genuine alpha decay produces that shape. So does survivorship bias that fades as the
test window approaches the date the ticker list was built. The two have completely
different implications and this data cannot separate them.

WHAT THIS SCRIPT DOES

EDGAR keeps the filings of companies that went bankrupt or were acquired - they
stopped filing, the record remains. The XBRL "frames" API returns one fact per
REPORTING ENTITY for a given quarter, so one request per quarter gives the full
cross-section of companies that were filing then, dead ones included.

From that it measures:

  1. how many companies were actually filing each quarter
  2. what share of them were still filing N years later - the true attrition rate
  3. how attrition varies with company size
  4. what fraction of the point-in-time population your 327-name universe covers
  5. an illustrative bound on how much survivor-only sampling can inflate a result

WHAT IT CANNOT DO, AND WHY

Rebuilding the universe is only half the job. The other half is PRICE history for
companies that no longer trade, and yfinance drops delisted tickers. That is exactly
what CRSP sells. So this quantifies the bias and produces the point-in-time entity
list; it cannot by itself produce a fully corrected backtest. Read the report as
"how wrong could the existing numbers be", not as a fixed dataset.

COST  ~85 requests (one per quarter), cached to disk. The Assets frame is a few MB per
quarter, so budget a few hundred MB for a full 2005-2026 run.

USAGE
  python survivorship_universe_v49.py --email you@example.com
  python survivorship_universe_v49.py --email you@example.com --start 2010 --end 2026
  python survivorship_universe_v49.py --offline            # use the cache only
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

FRAMES = "https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/CY{y}Q{q}I.json"


# =============================================================================
# FETCH
# =============================================================================
class Frames:
    def __init__(self, email, cache_dir, offline=False, pause=0.15):
        if not offline and ("@" not in email or "example.com" in email):
            sys.exit("SEC asks for a real contact address. Pass --email you@yourdomain.")
        self.headers = {"User-Agent": f"academic-research {email}",
                        "Accept-Encoding": "gzip, deflate"}
        self.dir, self.offline, self.pause = cache_dir, offline, pause
        os.makedirs(cache_dir, exist_ok=True)
        self.fetched = 0
        self.failed = []

    def quarter(self, concept, y, q):
        key = os.path.join(self.dir, f"frame_{concept}_{y}Q{q}.json")
        if os.path.exists(key):
            try:
                with open(key, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        if self.offline:
            return None
        try:
            import requests
        except ImportError:
            sys.exit("pip install requests")
        url = FRAMES.format(concept=concept, y=y, q=q)
        for attempt in range(3):
            try:
                time.sleep(self.pause)
                r = requests.get(url, headers=self.headers, timeout=60)
                if r.status_code == 404:
                    self.failed.append(f"{y}Q{q} (404)")
                    return None
                if r.status_code == 429:
                    time.sleep(3 + 3 * attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                with open(key, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                self.fetched += 1
                return data
            except Exception as e:
                if attempt == 2:
                    self.failed.append(f"{y}Q{q} ({type(e).__name__})")
                    return None
                time.sleep(2 + attempt)
        return None


# =============================================================================
# PANEL
# =============================================================================
def build_panel(fr, concept, start, end):
    """One row per (quarter, filing entity). Includes companies that later died."""
    rows = []
    quarters = [(y, q) for y in range(start, end + 1) for q in (1, 2, 3, 4)]
    for i, (y, q) in enumerate(quarters, 1):
        data = fr.quarter(concept, y, q)
        if not data:
            continue
        for e in data.get("data", []):
            cik = e.get("cik")
            if cik is None:
                continue
            rows.append({"y": y, "q": q, "cik": int(cik),
                         "name": e.get("entityName", ""), "val": e.get("val")})
        if i % 16 == 0 or i == len(quarters):
            print(f"    [{i}/{len(quarters)}] {y}Q{q}: {len(rows):,} entity-quarters "
                  f"({fr.fetched} requests)")
    if not rows:
        sys.exit("No frames data. Run once online, or check --start/--end.")
    d = pd.DataFrame(rows)
    d["period"] = d["y"] * 4 + (d["q"] - 1)
    return d


# =============================================================================
# ANALYSIS
# =============================================================================
def filer_counts(d):
    print("\n" + "=" * 92)
    print("1) HOW MANY COMPANIES WERE ACTUALLY FILING?")
    print("=" * 92)
    g = d.groupby("y")["cik"].nunique()
    print(f"   {'year':>6}{'filers':>10}")
    for y, n in g.items():
        print(f"   {y:>6}{n:>10,}")
    print("\n   This is the population a point-in-time universe would draw from.")
    return g


def survival(d):
    """Of the companies filing in year Y, what share were still filing later?"""
    print("\n" + "=" * 92)
    print("2) ATTRITION - THE THING A 2026 TICKER LIST HIDES")
    print("=" * 92)
    last = d.groupby("cik")["period"].max()
    first_year = d.groupby("cik")["y"].min()
    end_period = int(d["period"].max())
    horizons = [3, 5, 10, 15]
    print(f"   Of the companies filing in year Y, the share still filing N years on:\n")
    print(f"   {'cohort':>7}{'n':>8}" + "".join(f"{'+' + str(h) + 'y':>9}" for h in horizons))
    out = {}
    for y in sorted(d["y"].unique()):
        ciks = d.loc[d["y"] == y, "cik"].unique()
        if len(ciks) < 50:
            continue
        line = f"   {y:>7}{len(ciks):>8,}"
        row = {}
        for h in horizons:
            target = (y + h) * 4
            if target > end_period:
                line += f"{'-':>9}"
                continue
            alive = (last.reindex(ciks).fillna(-1) >= target).mean()
            row[h] = float(alive)
            line += f"{alive:>8.0%} "
        out[y] = row
        print(line)
    print("\n   Every company missing from that survival column is one your ticker list")
    print("   silently excludes. The longer the forward window, the more are missing.")
    return out


def attrition_by_size(d, horizon=10):
    print("\n" + "=" * 92)
    print(f"3) DOES ATTRITION DEPEND ON SIZE?  ({horizon}-year survival by {'assets'} decile)")
    print("=" * 92)
    last = d.groupby("cik")["period"].max()
    end_period = int(d["period"].max())
    years = [y for y in sorted(d["y"].unique()) if (y + horizon) * 4 <= end_period]
    if not years:
        print("   span too short for this horizon")
        return
    sub = d[d["y"].isin(years)].copy()
    sub = sub.groupby(["cik", "y"], as_index=False)["val"].median()
    sub = sub[sub["val"] > 0]
    sub["dec"] = sub.groupby("y")["val"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    sub["alive"] = (last.reindex(sub["cik"]).fillna(-1).to_numpy()
                    >= (sub["y"].to_numpy() + horizon) * 4)
    g = sub.groupby("dec").agg(n=("cik", "size"), alive=("alive", "mean"),
                               med=("val", "median"))
    print(f"   {'decile':>8}{'n':>10}{'median assets':>18}{'survived ' + str(horizon) + 'y':>18}")
    for dec, r in g.iterrows():
        print(f"   {int(dec)+1:>8}{int(r['n']):>10,}${r['med']/1e6:>16,.0f}M"
              f"{r['alive']:>17.0%}")
    lo, hi = g["alive"].iloc[0], g["alive"].iloc[-1]
    print(f"\n   smallest decile survives {lo:.0%}, largest {hi:.0%} "
          f"- a {hi - lo:+.0%} gap")
    print("   A survivor-only sample therefore over-represents large companies far more")
    print("   than it over-represents small ones, which distorts any size-related feature.")


def coverage(d, universe_ciks, label):
    print("\n" + "=" * 92)
    print(f"4) WHAT SHARE OF THE POPULATION IS YOUR UNIVERSE?")
    print("=" * 92)
    if not universe_ciks:
        print("   No CIKs supplied for the current universe - skipping.")
        return
    print(f"   {label}: {len(universe_ciks)} CIKs\n")
    print(f"   {'year':>6}{'filers':>10}{'yours':>9}{'share':>9}")
    for y in sorted(d["y"].unique()):
        ciks = set(d.loc[d["y"] == y, "cik"].unique())
        mine = len(ciks & universe_ciks)
        print(f"   {y:>6}{len(ciks):>10,}{mine:>9}{mine / max(len(ciks), 1):>8.2%}")
    print("\n   A small share is not itself a problem - a study can use a subset. The")
    print("   problem is that the subset was chosen by looking at 2026.")


def bias_illustration(surv):
    print("\n" + "=" * 92)
    print("5) HOW MUCH CAN SURVIVOR-ONLY SAMPLING INFLATE A RESULT?")
    print("=" * 92)
    if not surv:
        print("   not enough cohorts")
        return
    print("   An illustration, not a correction. If a fraction p of the population")
    print("   disappears over the forward window and those companies returned about")
    print("   -80% on the way out, a survivor-only sample overstates the mean forward")
    print("   return by roughly p x 80 percentage points.\n")
    print(f"   {'cohort':>7}{'died within 10y':>18}{'overstatement':>16}")
    for y, row in sorted(surv.items()):
        if 10 not in row:
            continue
        p = 1 - row[10]
        print(f"   {y:>7}{p:>17.0%}{p * 80:>15.0f}pp")
    print("\n   The same mechanism inflates a cross-sectional IC, though not by a")
    print("   formula this simple: the model is rewarded for features that happen to")
    print("   correlate with survival, and only survivors are ever scored.")


# =============================================================================
# MAIN
# =============================================================================
def load_universe_ciks(cache_dir, universe):
    """Map the current universe's tickers to CIKs using the audit's cached file."""
    p = os.path.join(cache_dir, "company_tickers.json")
    if not os.path.exists(p):
        return set(), "no company_tickers.json in the cache"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    lut = {}
    for row in (data.values() if isinstance(data, dict) else data):
        t = str(row.get("ticker", "")).upper()
        if t:
            lut[t] = int(row["cik_str"])
    try:
        from pillar2_v43_universe import DEPLOY_UNIVERSE, load_training_universe
        names = (DEPLOY_UNIVERSE if universe == "deploy"
                 else load_training_universe(include_deploy=True))
    except ImportError:
        return set(), "pillar2_v43_universe.py not importable"
    return {lut[t.upper()] for t in names if t.upper() in lut}, f"{universe} universe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="offline@local")
    ap.add_argument("--concept", default="Assets",
                    help="XBRL tag used to enumerate filers; Assets has the widest coverage")
    ap.add_argument("--start", type=int, default=2009)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--cache-dir", default="edgar_cache")
    ap.add_argument("--universe", default="all", choices=["all", "deploy"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="pit_universe.csv.gz")
    a = ap.parse_args()

    print("=" * 92)
    print("V49 - POINT-IN-TIME UNIVERSE AND SURVIVORSHIP AUDIT")
    print("=" * 92)
    print(f"  concept {a.concept} | {a.start}-{a.end} | cache {a.cache_dir}/")
    print("  XBRL coverage begins around 2009-2011; earlier quarters will be thin.\n")

    fr = Frames(a.email, a.cache_dir, offline=a.offline)
    d = build_panel(fr, a.concept, a.start, a.end)
    if fr.failed:
        print(f"\n  {len(fr.failed)} quarters unavailable: {', '.join(fr.failed[:8])}"
              f"{' ...' if len(fr.failed) > 8 else ''}")

    print(f"\n  panel: {len(d):,} entity-quarters | {d['cik'].nunique():,} distinct companies")

    filer_counts(d)
    surv = survival(d)
    attrition_by_size(d)
    ciks, label = load_universe_ciks(a.cache_dir, a.universe)
    coverage(d, ciks, label)
    bias_illustration(surv)

    d.to_csv(a.out, index=False, compression="gzip")
    print(f"\n  wrote {a.out}  (quarter, cik, name, {a.concept})")
    print("\n" + "=" * 92)
    print("  WHAT THIS DOES AND DOES NOT GIVE YOU")
    print("=" * 92)
    print("  It gives: the real filer population per quarter, the true attrition rate,")
    print("     and therefore a defensible statement of how biased the existing sample is.")
    print("  It does not give: price history for companies that stopped trading. Without")
    print("     that, the walk-forward cannot be re-run survivorship-free on free data.")
    print("     Quantifying the bias is the honest alternative to removing it.")


if __name__ == "__main__":
    main()
