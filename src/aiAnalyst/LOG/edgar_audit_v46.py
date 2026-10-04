#!/usr/bin/env python3
"""
edgar_audit_v46.py - can SEC EDGAR actually support a dilution/blowup-risk model
for YOUR universe? Answer that before writing a line of modelling code.

WHY EDGAR AND NOT A VENDOR

Two things every previous version needed and could not get for free:

  point-in-time fundamentals - every XBRL fact EDGAR returns carries the date it was
    FILED, not just the period it covers. Filter to facts filed on or before your
    signal date and you have exactly what Compustat's point-in-time product sells.
    Using today's balance sheet to explain a 2023 event is the look-ahead trap that
    would invalidate the whole study; the `filed` field is what prevents it.

  a survivorship-free universe - EDGAR keeps the filings of companies that went
    bankrupt or were acquired. They stopped filing; the record remains. A universe
    built from "who was filing on date D" has no survivorship bias, unlike a ticker
    list written in 2026.

WHAT THIS SCRIPT DOES

Nothing but measure availability. It does not test the hypothesis and it does not
train anything. It answers:

  1. how many of your tickers resolve to a CIK at all
  2. how far back each one's filing history goes
  3. how many dilution-related filings exist (S-3 shelf, 424B5 pricing, S-1)
  4. which XBRL tags are actually populated - cash, operating cash flow, shares
     outstanding - and how many quarters of each
  5. THE POINT-IN-TIME CHECK: do the facts carry `filed`, and how long after the
     period end does the filing appear? That lag is the delay you must apply to
     every feature, and if the field were missing the project would be dead.

If coverage is thin, you have lost a day instead of a term.

RATE LIMITS AND MANNERS
  SEC asks for a descriptive User-Agent with contact details and no more than about
  10 requests a second. This sleeps between calls and caches every response to disk,
  so a second run costs nothing.

USAGE
  python edgar_audit_v46.py --email you@example.com
  python edgar_audit_v46.py --email you@example.com --universe deploy
  python edgar_audit_v46.py --email you@example.com --offline     # cache only
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Filings that signal dilution. Form type and date alone carry the event - no
# document parsing needed for a first pass.
DILUTION_FORMS = {
    "S-3": "shelf registration (intent to sell)",
    "S-3ASR": "automatic shelf (large filers)",
    "S-3/A": "shelf amendment",
    "424B5": "prospectus supplement (offering priced)",
    "424B3": "prospectus supplement",
    "424B4": "prospectus supplement",
    "S-1": "registration statement",
    "S-1/A": "registration amendment",
}
PERIODIC_FORMS = {"10-Q", "10-K", "10-K/A", "10-Q/A", "8-K"}

# The XBRL tags a runway / dilution model needs. Several aliases per concept,
# because small filers tag inconsistently.
WANTED_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "CashAndCashEquivalentsFairValueDisclosure"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "shares_out": ["EntityCommonStockSharesOutstanding",      # dei
                   "CommonStockSharesOutstanding",
                   "CommonStockSharesIssued",
                   "WeightedAverageNumberOfSharesOutstandingBasic"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "revenue": ["Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet"],
}


# =============================================================================
# FETCHING
# =============================================================================
class Edgar:
    def __init__(self, email, cache_dir="edgar_cache", offline=False, pause=0.12):
        if not offline and ("@" not in email or "example.com" in email):
            sys.exit("SEC asks for a real contact address in the User-Agent. "
                     "Pass --email you@yourdomain.")
        self.headers = {"User-Agent": f"academic-research {email}",
                        "Accept-Encoding": "gzip, deflate"}
        self.dir = cache_dir
        self.offline = offline
        self.pause = pause
        os.makedirs(cache_dir, exist_ok=True)
        self.n_fetched = 0
        self.errors = Counter()

    def get(self, url, key):
        """Cache-first. A second run of this script costs no requests at all."""
        path = os.path.join(self.dir, key)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        if self.offline:
            return None
        try:
            import requests
        except ImportError:
            sys.exit("pip install requests")
        for attempt in range(3):
            try:
                time.sleep(self.pause)
                r = requests.get(url, headers=self.headers, timeout=30)
                if r.status_code == 404:
                    self.errors["404 not found"] += 1
                    return None
                if r.status_code == 429:
                    time.sleep(2 + 2 * attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                self.n_fetched += 1
                return data
            except Exception as e:
                if attempt == 2:
                    self.errors[f"{type(e).__name__}"] += 1
                    return None
                time.sleep(1 + attempt)
        return None


def resolve_ciks(ed, tickers):
    """SEC publishes the full ticker -> CIK map as one small file."""
    data = ed.get(SEC_TICKERS, "company_tickers.json")
    if not data:
        sys.exit("Could not load the ticker->CIK map. Check the network, or run once "
                 "online to populate the cache.")
    lookup = {}
    rows = data.values() if isinstance(data, dict) else data
    for row in rows:
        t = str(row.get("ticker", "")).upper()
        if t:
            lookup[t] = (f"{int(row['cik_str']):010d}", row.get("title", ""))
    out, missing = {}, []
    for t in tickers:
        hit = lookup.get(t.upper())
        if hit:
            out[t] = hit
        else:
            missing.append(t)
    return out, missing


def all_filings(ed, cik):
    """
    Filing history as a DataFrame of form + filingDate.

    submissions.json holds the recent filings inline and older ones in extra files
    listed under filings.files - a long history needs both.
    """
    data = ed.get(SUBMISSIONS.format(cik=cik), f"sub_{cik}.json")
    if not data:
        return None
    frames = []

    def _rows(block):
        if not block or "form" not in block:
            return None
        n = len(block["form"])
        return pd.DataFrame({
            "form": block["form"],
            "filingDate": block.get("filingDate", [None] * n),
            "accession": block.get("accessionNumber", [None] * n),
        })

    recent = _rows((data.get("filings") or {}).get("recent"))
    if recent is not None:
        frames.append(recent)
    for extra in (data.get("filings") or {}).get("files", []) or []:
        name = extra.get("name")
        if not name:
            continue
        more = ed.get(f"https://data.sec.gov/submissions/{name}", f"sub_{name}")
        r = _rows(more)
        if r is not None:
            frames.append(r)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    return df.dropna(subset=["filingDate"]).sort_values("filingDate")


def fact_series(facts, tag_names):
    """
    Pull the first matching tag out of a companyfacts payload.

    Returns a DataFrame with end (period), val, filed (the date the number became
    public) and form. `filed` is the whole point: it is what makes this point-in-time.
    """
    if not facts:
        return None, None
    for space in ("us-gaap", "dei", "ifrs-full", "srt"):
        block = (facts.get("facts") or {}).get(space) or {}
        for tag in tag_names:
            if tag not in block:
                continue
            rows = []
            for unit, entries in (block[tag].get("units") or {}).items():
                for e in entries:
                    rows.append({"end": e.get("end"), "val": e.get("val"),
                                 "filed": e.get("filed"), "form": e.get("form"),
                                 "fy": e.get("fy"), "fp": e.get("fp"), "unit": unit})
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
            return df.dropna(subset=["end"]).sort_values("end"), f"{space}:{tag}"
    return None, None


# =============================================================================
# AUDIT
# =============================================================================
def audit(ed, tickers, ciks):
    rows, lag_samples = [], []
    for i, t in enumerate(tickers, 1):
        if t not in ciks:
            continue
        cik, name = ciks[t]
        rec = {"ticker": t, "cik": cik, "company": name[:38]}

        fil = all_filings(ed, cik)
        if fil is None or fil.empty:
            rec["filings"] = 0
            rows.append(rec)
            continue
        rec["filings"] = len(fil)
        rec["first_filing"] = fil["filingDate"].min().date()
        rec["last_filing"] = fil["filingDate"].max().date()
        rec["years"] = round((fil["filingDate"].max() - fil["filingDate"].min()).days / 365.25, 1)

        forms = Counter(fil["form"])
        rec["n_shelf"] = sum(v for k, v in forms.items() if k.startswith(("S-3", "S-1")))
        rec["n_424b"] = sum(v for k, v in forms.items() if k.startswith("424B"))
        rec["n_10q"] = forms.get("10-Q", 0)
        rec["n_8k"] = forms.get("8-K", 0)

        facts = ed.get(COMPANYFACTS.format(cik=cik), f"facts_{cik}.json")
        for concept, tags in WANTED_TAGS.items():
            s, used = fact_series(facts, tags)
            if s is None or s.empty:
                rec[f"{concept}_n"] = 0
                continue
            rec[f"{concept}_n"] = int(s["end"].nunique())
            rec[f"{concept}_from"] = s["end"].min().date()
            rec[f"{concept}_tag"] = used
            if concept == "cash":
                ok = s.dropna(subset=["filed"])
                rec["has_filed_date"] = len(ok) > 0
                if len(ok):
                    lag = (ok["filed"] - ok["end"]).dt.days
                    lag = lag[(lag >= 0) & (lag < 400)]
                    rec["median_filing_lag_days"] = float(lag.median()) if len(lag) else np.nan
                    lag_samples.extend(lag.tolist())
        rows.append(rec)
        if i % 10 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] audited  ({ed.n_fetched} requests so far)")
    return pd.DataFrame(rows), lag_samples


def report(df, missing, lag_samples, min_years, ed):
    print("\n" + "=" * 92)
    print("EDGAR COVERAGE AUDIT")
    print("=" * 92)
    n = len(df)
    print(f"  tickers requested: {n + len(missing)} | resolved to a CIK: {n} | "
          f"unresolved: {len(missing)}")
    if missing:
        print(f"    unresolved: {', '.join(missing[:20])}"
              f"{' ...' if len(missing) > 20 else ''}")
        print("    (usually a ticker change, a foreign issuer filing 20-F, or an ADR)")
    if n == 0:
        print("\n  Nothing resolved - stop here.")
        return

    have = df[df["filings"] > 0]
    print(f"\n  1) FILING HISTORY")
    print(f"     tickers with any filings: {len(have)}/{n}")
    if len(have):
        print(f"     history length (years): median {have['years'].median():.1f}, "
              f"min {have['years'].min():.1f}, max {have['years'].max():.1f}")
        deep = have[have["years"] >= min_years]
        print(f"     with >= {min_years} years: {len(deep)}/{n} "
              f"({len(deep) / n:.0%})")
        print(f"     earliest filing across the universe: "
              f"{pd.to_datetime(have['first_filing'], errors='coerce').min().date()}")

    print(f"\n  2) DILUTION-RELATED FILINGS  (form type and date alone - no parsing)")
    if len(have):
        print(f"     tickers with >=1 shelf (S-1/S-3):    "
              f"{(have['n_shelf'] > 0).sum()}/{len(have)}"
              f"   total {int(have['n_shelf'].sum())}")
        print(f"     tickers with >=1 pricing (424B*):    "
              f"{(have['n_424b'] > 0).sum()}/{len(have)}"
              f"   total {int(have['n_424b'].sum())}")
        print(f"     median 424B* per ticker: {have['n_424b'].median():.0f}"
              f"  |  median 10-Q: {have['n_10q'].median():.0f}")

    print(f"\n  3) XBRL TAG COVERAGE  (quarters of data per concept)")
    print(f"     {'concept':<15}{'tickers with data':>20}{'median quarters':>18}"
          f"{'earliest':>12}")
    for concept in WANTED_TAGS:
        col = f"{concept}_n"
        if col not in df.columns:
            print(f"     {concept:<15}{'0':>20}")
            continue
        got = df[df[col] > 0]
        # the column mixes date objects with NaN where a ticker lacks the concept,
        # so coerce before reducing
        earliest = "n/a"
        if f"{concept}_from" in df.columns:
            e = pd.to_datetime(df[f"{concept}_from"], errors="coerce").min()
            if pd.notna(e):
                earliest = e.date()
        print(f"     {concept:<15}{f'{len(got)}/{n}':>20}"
              f"{got[col].median() if len(got) else 0:>18.0f}{str(earliest):>12}")

    print(f"\n  4) POINT-IN-TIME CHECK  (the field the whole study depends on)")
    if "has_filed_date" in df.columns and df["has_filed_date"].any():
        pct = df["has_filed_date"].mean()
        print(f"     facts carrying a `filed` date: {pct:.0%} of tickers")
        if lag_samples:
            s = pd.Series(lag_samples)
            print(f"     lag from period end to filing: median {s.median():.0f} days, "
                  f"p90 {s.quantile(0.9):.0f}, max {s.max():.0f}")
            print(f"     -> a feature for a quarter ending on D is only usable from")
            print(f"        about D+{s.quantile(0.9):.0f}. Apply that lag, or you are")
            print(f"        trading on numbers nobody had yet.")
    else:
        print("     !! No `filed` dates found. Without them these data are NOT")
        print("        point-in-time and the study cannot be run honestly.")

    # ---- verdict ----
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    def _col(name):
        return df[name] if name in df.columns else pd.Series(0, index=df.index)
    usable = df[(df["filings"] > 0) & (_col("cash_n") > 4) & (_col("shares_out_n") > 4)]
    if len(have):
        usable = usable[usable["years"] >= min_years]
    print(f"  tickers with >= {min_years}y history AND cash AND shares-outstanding: "
          f"{len(usable)}/{n}")
    events = int(have["n_424b"].sum()) if len(have) else 0
    print(f"  total 424B* offerings across the universe: {events}")
    if len(usable) >= 30 and events >= 100:
        print("\n  ENOUGH TO PROCEED. Next step is the hypothesis test: do runway,")
        print("  shelf status and dilution history separate the names that go on to")
        print("  draw down hard from the ones that do not? If that table is flat,")
        print("  stop there.")
    elif len(usable) >= 15:
        print("\n  THIN. Workable only if the universe is widened - build it from")
        print("  everyone filing 10-Qs in the relevant size band rather than from a")
        print("  fixed ticker list. That also removes survivorship bias.")
    else:
        print("\n  NOT ENOUGH. Coverage cannot support the study on this universe.")

    if ed.errors:
        print(f"\n  fetch errors: {dict(ed.errors)}")
    return usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="offline@local",
                    help="contact address for the SEC User-Agent (they ask for one)")
    ap.add_argument("--universe", default="deploy", choices=["deploy", "all"])
    ap.add_argument("--cache-dir", default="edgar_cache")
    ap.add_argument("--min-years", type=float, default=8.0)
    ap.add_argument("--offline", action="store_true",
                    help="use only what is already cached")
    ap.add_argument("--out", default="edgar_audit.csv")
    a = ap.parse_args()

    try:
        from pillar2_v43_universe import DEPLOY_UNIVERSE, load_training_universe
        tickers = (DEPLOY_UNIVERSE if a.universe == "deploy"
                   else load_training_universe(include_deploy=True))
    except ImportError:
        sys.exit("Run this next to pillar2_v43_universe.py.")

    print("=" * 92)
    print(f"EDGAR AUDIT  -  {a.universe} universe, {len(tickers)} tickers")
    print("=" * 92)
    print("  Measuring data availability only. No modelling, no hypothesis test.")
    print(f"  cache: {a.cache_dir}/  (a second run makes no requests)\n")

    ed = Edgar(a.email, a.cache_dir, offline=a.offline)
    ciks, missing = resolve_ciks(ed, tickers)
    print(f"  resolved {len(ciks)}/{len(tickers)} tickers to a CIK\n")

    df, lags = audit(ed, tickers, ciks)
    usable = report(df, missing, lags, a.min_years, ed)
    df.to_csv(a.out, index=False)
    print(f"\n  wrote {a.out}")
    if usable is not None and len(usable):
        print(f"  usable tickers: {', '.join(sorted(usable['ticker'])[:25])}"
              f"{' ...' if len(usable) > 25 else ''}")


if __name__ == "__main__":
    main()
