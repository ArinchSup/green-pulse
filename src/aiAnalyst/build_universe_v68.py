#!/usr/bin/env python3
"""
build_universe_v68.py - expand the price cache from 337 names to ~1,200.

WHY MORE NAMES AND NOT MORE ROWS

V66/V67 put the binding constraint in plain sight: the 10% cut fired 12,738 times
but was worth about 269 independent observations, because overlapping 90-day
windows across correlated names repeat the same information. Every FILTER tried
in V67 spent that budget. Only `spaced` raised it, by discarding rows that were
duplicates of each other.

More NAMES is the other way to raise it, and the only one that adds information
rather than removing redundancy. But be honest about the size of the gain: 337 to
1,200 tickers is 3.6x the rows and nowhere near 3.6x the evidence, because large
US equities share a market factor and move together. A realistic expectation is
1.5-2x on n_eff, and most of that comes from the names that are LEAST like the
ones already in the cache - smaller, less liquid, different sectors.

THE TRAP THIS FILE EXISTS TO AVOID

The easy way to get 1,200 tickers is to take today's index membership. That is
survivorship bias, and at 1,200 names it is a BIGGER absolute distortion than at
337, not a smaller one: every name on the list is a company that still exists and
still trades, which is precisely the selection V58 measured. A model trained on
survivors learns that dips recover, because in that sample they always did.

So the universe is assembled POINT IN TIME:

    1. SEC company_tickers.json gives CIK -> ticker for every registrant. Free,
       no key, no rate limit worth worrying about.
    2. The V49 point-in-time frame (pit_universe.csv.gz: CIK, year, quarter,
       total assets) says which companies were ALREADY BIG at each date. A name
       enters the universe in the first year it clears the size floor, not in the
       year you happened to download it.
    3. Names that cannot be downloaded are RECORDED, not silently dropped. That
       failure list is the delisting estimate: it is the closest thing to a
       survivorship correction available without a paid database, and V58 turns
       it into a bias number.

WHAT THE QUALITY GATES ARE FOR

This container's own caches turned out to be synthetic - Open equalled Close on
100% of bars, one ticker was priced at 0.00, and 'market' columns took 120
distinct values on a single date. None of that was caught for weeks. So the gates
below run on every downloaded frame and refuse it with a reason:

    flat_bars      Open == Close on more than 5% of bars. Real OHLC almost never
                   does this; generated data does it constantly.
    zero_price     any non-positive close.
    short_history  fewer than MIN_BARS usable bars.
    thin           median dollar volume below the floor - untradeable, and its
                   barriers would be noise.
    stale          more than 10% of bars repeating the previous close exactly.
    split_jump     an unadjusted split: a one-day move beyond +/-60% that
                   reverses next day.

USAGE
  python build_universe_v68.py            # dry run: build the list, no download
  python build_universe_v68.py --download # fetch prices into the cache
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
OUT_CACHE = "price_cache_v68"
UNIVERSE_CSV = "universe_v68.csv"
MANIFEST = "universe_v68_manifest.json"

PIT_FILE = "pit_universe.csv.gz"     # CIK, y, q, val (total assets)
PIT_CACHE_DIR = "frames_cache"       # one JSON per quarter, so a rerun is free
PIT_CONCEPT = "Assets"               # us-gaap tag, instantaneous
PIT_START_YEAR = 2009                # XBRL is not usable much before this
PIT_END_YEAR = None                  # None -> current year
TICKER_MAP_CACHE = "edgar_cache/company_tickers.json"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

TARGET_NAMES = 1200
MIN_ASSETS_USD = 3e8          # size floor for entering the universe
START = "2005-01-01"
END = None                    # None -> today

# quality gates
MIN_BARS = 500
MIN_MEDIAN_DOLLAR_VOL = 2e6
MAX_FLAT_BAR_SHARE = 0.05     # Open == Close
MAX_STALE_SHARE = 0.10        # Close == previous Close
SPLIT_JUMP = 0.60

BATCH = 40                    # tickers per download call
SLEEP = 1.0                   # seconds between batches
EMAIL = "you@example.com"     # SEC wants a contact in the User-Agent


# =============================================================================
# THE LIST
# =============================================================================
def sec_ticker_map(cache=TICKER_MAP_CACHE, email=EMAIL, offline=False):
    """CIK -> ticker. Cached, because it changes slowly and SEC throttles."""
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
        if len(raw) > 100:
            return {int(v["cik_str"]): v["ticker"] for v in raw.values()}
        print(f"  {cache} holds {len(raw)} entries - too few to be the real "
              f"file, refetching")
    if offline:
        raise RuntimeError(f"{cache} is missing or a stub and offline=True")
    import urllib.request
    req = urllib.request.Request(
        SEC_TICKER_URL, headers={"User-Agent": f"universe-builder {email}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.loads(r.read().decode())
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return {int(v["cik_str"]): v["ticker"] for v in raw.values()}


def _sec_headers(email):
    """
    What SEC actually wants. The email is not validated by them - a gmail is
    fine - but the User-Agent has to identify something, and the other two
    headers are in their fair-access notes. urllib's default UA gets a 403.
    """
    return {"User-Agent": f"stock-research-project {email}",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
            "Host": "data.sec.gov"}


def check_sec(email=EMAIL, verbose=True):
    """
    One request, full diagnosis. Run this before blaming the email address.
    """
    import urllib.error
    import urllib.request
    url = ("https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/"
           "CY2020Q1I.json")
    if verbose:
        print(f"  probing {url}")
        print(f"  User-Agent: stock-research-project {email}")
    try:
        req = urllib.request.Request(url, headers=_sec_headers(email))
        with urllib.request.urlopen(req, timeout=60) as r:
            n = len(json.loads(r.read().decode()).get("data", []))
        if verbose:
            print(f"  OK - {n:,} filers returned. SEC access works.")
        return True, f"ok ({n} filers)"
    except urllib.error.HTTPError as e:
        msg = {403: "SEC refused it. Try a different network (some ISPs and "
                    "university proxies are blocked wholesale), or skip SEC "
                    "entirely with universe_from_listing().",
               404: "endpoint moved or the quarter is unpublished.",
               429: "rate limited - wait a minute and retry."}.get(
                   e.code, "unexpected status.")
        if verbose:
            print(f"  HTTP {e.code}: {msg}")
        return False, f"HTTP {e.code}"
    except Exception as e:
        if verbose:
            print(f"  network error: {e}")
            print(f"  SEC may be unreachable from here. "
                  f"universe_from_listing() needs no SEC access.")
        return False, str(e)


def build_pit_universe(out=PIT_FILE, cache_dir=PIT_CACHE_DIR,
                       concept=PIT_CONCEPT, start_year=PIT_START_YEAR,
                       end_year=PIT_END_YEAR, email=EMAIL, verbose=True):
    """
    Build the point-in-time frame from SEC XBRL frames. No key, no paid data.

    For each quarter, https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/
    CY<year>Q<q>I.json returns EVERY filer that reported total assets for that
    period. Stack the quarters and you have, for each date, which companies
    existed and how big they were - the only thing needed to stop today's
    survivors being retro-fitted into 2010's universe.

    About 70 requests for 2009-2026, each cached to disk, so a rerun is free.
    """
    import urllib.error
    import urllib.request
    end_year = end_year or pd.Timestamp.today().year
    os.makedirs(cache_dir, exist_ok=True)
    rows = []
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            tag = f"CY{y}Q{q}I"
            path = os.path.join(cache_dir, f"{concept}_{tag}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            else:
                url = (f"https://data.sec.gov/api/xbrl/frames/us-gaap/"
                       f"{concept}/USD/{tag}.json")
                req = urllib.request.Request(url, headers=_sec_headers(email))
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        payload = json.loads(r.read().decode())
                except urllib.error.HTTPError as e:
                    if verbose:
                        # 403 is SEC refusing the request, usually over the
                        # User-Agent. 404 means the quarter genuinely is not
                        # published. Calling both "not published yet" sent
                        # someone hunting for the wrong problem.
                        why = {403: "FORBIDDEN - SEC refused the request, not a "
                                    "missing quarter. Run check_sec().",
                               404: "not published",
                               429: "rate limited - slow down"}.get(
                                   e.code, "see status code")
                        print(f"    {tag}: HTTP {e.code} - {why}")
                    if e.code == 403:
                        raise RuntimeError(
                            "SEC returned 403. This is almost never the email "
                            "address itself - it is the User-Agent or the "
                            "network. Run build_universe_v68.check_sec() for a "
                            "one-request diagnosis, or use "
                            "universe_from_listing() which needs no SEC access "
                            "at all.") from None
                    continue
                except Exception as e:
                    if verbose:
                        print(f"    {tag}: {e}")
                    continue
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                time.sleep(0.15)              # SEC asks for under 10 req/sec
            for rec in payload.get("data", []):
                rows.append({"y": y, "q": q, "cik": int(rec["cik"]),
                             "name": rec.get("entityName", ""),
                             "val": float(rec.get("val", 0.0))})
            if verbose:
                print(f"    {tag}: {len(payload.get('data', [])):,} filers "
                      f"(running total {len(rows):,})")
    if not rows:
        raise RuntimeError("SEC frames returned nothing - check the network and "
                           "that a real contact address is in EMAIL")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, compression="gzip")
    if verbose:
        print(f"  wrote {out}: {len(df):,} rows, "
              f"{df['cik'].nunique():,} distinct companies, "
              f"{df['y'].min()}-{df['y'].max()}")
    return df


NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


def _parse_listing(text, sym_col, etf_col, test_col):
    """Pipe-delimited Nasdaq Trader file. The last line is a timestamp, not data."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    head = lines[0].split("|")
    idx = {name: head.index(name) for name in (sym_col, etf_col, test_col)
           if name in head}
    if sym_col not in idx:
        raise RuntimeError(f"unexpected listing header: {head}")
    out = []
    for l in lines[1:]:
        if l.startswith("File Creation Time"):
            continue
        parts = l.split("|")
        if len(parts) <= max(idx.values()):
            continue
        sym = parts[idx[sym_col]].strip()
        if not sym or not sym.isalpha():
            # units, warrants, preferreds and rights carry $ . or digits. They
            # are not the common stock this project trades.
            continue
        if etf_col in idx and parts[idx[etf_col]].strip().upper() == "Y":
            continue
        if test_col in idx and parts[idx[test_col]].strip().upper() == "Y":
            continue
        out.append(sym)
    return sorted(set(out))


def universe_from_listing(cache="listed_symbols.txt", verbose=True):
    """
    Every common stock listed on a US exchange, from Nasdaq Trader's free
    symbol-directory files. No key, no User-Agent rules, no SEC.

    WHAT THIS FIXES AND WHAT IT DOES NOT

    It fixes nothing about survivorship on its own - it is a CURRENT listing, so
    delisted companies are absent, exactly as they are absent from any free price
    source. What it gives you is a much wider candidate pool (~6,000 names rather
    than 337) and no dependence on a size ranking measured today.

    An honest word about the SEC step this replaces: it buys less than it looks
    like. SEC frames tell you WHICH companies existed in 2011 - but you still
    cannot download prices for the ones that are gone, because no free source
    keeps them. So the SEC list's real use is feeding V58's bias estimate, not
    curing the bias. Skipping it costs you that estimate, not the expansion.
    """
    import urllib.request
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            syms = [l.strip() for l in f if l.strip()]
        if verbose:
            print(f"  {cache}: {len(syms):,} symbols (cached)")
        return syms
    syms = []
    for url, sym, etf, test in (
            (NASDAQ_LISTED, "Symbol", "ETF", "Test Issue"),
            (OTHER_LISTED, "ACT Symbol", "ETF", "Test Issue")):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 stock-research"})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", "replace")
            got = _parse_listing(text, sym, etf, test)
            syms += got
            if verbose:
                print(f"  {url.rsplit('/', 1)[-1]}: {len(got):,} common stocks")
        except Exception as e:
            if verbose:
                print(f"  {url.rsplit('/', 1)[-1]}: {e}")
    syms = sorted(set(syms))
    if not syms:
        raise RuntimeError("both listing files failed - check the network")
    with open(cache, "w", encoding="utf-8") as f:
        f.write("\n".join(syms))
    if verbose:
        print(f"  {len(syms):,} unique common stocks -> {cache}")
    return syms


def universe_from_prices(cache_dir, target=TARGET_NAMES, verbose=True):
    """
    FALLBACK when there is no point-in-time frame: rank the names already cached
    by median dollar volume.

    This is weaker, and the weakness has a name. Ranking a cache you already
    downloaded selects on companies that still trade today, which is exactly the
    survivorship bias V58 measured. Use it to get moving, not to publish.
    """
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pkl"))
    rows = []
    for f in files:
        try:
            d = pd.read_pickle(os.path.join(cache_dir, f))
            dv = float(np.median(d["Close"] * d["Volume"]))
        except Exception:
            continue
        rows.append({"ticker": os.path.splitext(f)[0],
                     "median_dollar_vol": dv,
                     "first_year": int(pd.DatetimeIndex(d.index).year.min())})
    u = (pd.DataFrame(rows).sort_values("median_dollar_vol", ascending=False)
         .head(target))
    if verbose:
        print(f"  FALLBACK universe from {cache_dir}: {len(u):,} names ranked "
              f"by liquidity")
        print(f"    WARNING this selects on names that still trade, which is "
              f"survivorship bias.")
        print(f"    Fine for a first pass; build the point-in-time frame before "
              f"quoting any result.")
    return u


def point_in_time_universe(pit_file=PIT_FILE, min_assets=MIN_ASSETS_USD,
                           target=TARGET_NAMES, verbose=True, build=True):
    """
    For each CIK, the FIRST year it cleared the size floor. That year is when the
    name becomes eligible - which is the whole point. Selecting on size measured
    today would put 2024's winners into the 2009 universe.
    """
    if not os.path.exists(pit_file):
        if not build:
            raise RuntimeError(f"{pit_file} not found and build=False")
        if verbose:
            print(f"  {pit_file} not found - building it from SEC XBRL frames "
                  f"(~70 cached requests)")
        build_pit_universe(out=pit_file, verbose=verbose)
    pit = pd.read_csv(pit_file)
    need = {"cik", "y", "val"}
    if not need.issubset(pit.columns):
        raise RuntimeError(f"{pit_file} needs columns {sorted(need)}")
    big = pit[pit["val"] >= min_assets]
    first = big.groupby("cik")["y"].min().rename("first_year")
    size = big.groupby("cik")["val"].median().rename("median_assets")
    u = pd.concat([first, size], axis=1).reset_index()
    u = u.sort_values("median_assets", ascending=False).head(target)
    if verbose:
        print(f"  point-in-time universe: {len(u):,} CIKs clear "
              f"${min_assets/1e6:.0f}M assets")
        print(f"    entry years span {int(u['first_year'].min())} to "
              f"{int(u['first_year'].max())} - a name is only eligible from its "
              f"own entry year")
    return u


def build_list(pit_file=PIT_FILE, target=TARGET_NAMES, min_assets=MIN_ASSETS_USD,
               offline=False, out=UNIVERSE_CSV, fallback_cache=None,
               verbose=True):
    try:
        u = point_in_time_universe(pit_file, min_assets, target, verbose,
                                   build=not offline)
    except RuntimeError as e:
        if not fallback_cache:
            raise
        print(f"  point-in-time build failed ({e})")
        u = universe_from_prices(fallback_cache, target, verbose)
        u.to_csv(out, index=False)
        return u
    cmap = sec_ticker_map(offline=offline)
    u["ticker"] = u["cik"].map(cmap)
    missing = int(u["ticker"].isna().sum())
    u = u.dropna(subset=["ticker"]).drop_duplicates("ticker")
    if verbose:
        print(f"  mapped {len(u):,} CIKs to tickers "
              f"({missing:,} had no current ticker - most of those are the "
              f"DELISTED names, and that is signal, not noise)")
    u.to_csv(out, index=False)
    return u


# =============================================================================
# QUALITY GATES
# =============================================================================
def check_frame(df, ticker):
    """Returns (ok, reason). Runs on every frame before it is cached."""
    if df is None or len(df) == 0:
        return False, "empty"
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return False, "missing columns"
    df = df.dropna(subset=["Close"])
    if len(df) < MIN_BARS:
        return False, f"short_history ({len(df)} bars)"
    c = df["Close"].to_numpy(float)
    if not np.all(np.isfinite(c)) or (c <= 0).any():
        return False, "zero_price"
    flat = float((df["Open"].to_numpy(float) == c).mean())
    if flat > MAX_FLAT_BAR_SHARE:
        return False, f"flat_bars (Open==Close on {flat:.0%} of bars)"
    stale = float((np.diff(c) == 0).mean())
    if stale > MAX_STALE_SHARE:
        return False, f"stale ({stale:.0%} unchanged closes)"
    dv = float(np.median(c * df["Volume"].to_numpy(float)))
    if not np.isfinite(dv) or dv < MIN_MEDIAN_DOLLAR_VOL:
        return False, f"thin (median ${dv/1e6:.1f}M)"
    r = np.diff(np.log(c))
    jump = np.flatnonzero(np.abs(r) > np.log(1 + SPLIT_JUMP))
    for i in jump:
        if i + 1 < len(r) and np.sign(r[i]) != np.sign(r[i + 1]) \
                and abs(r[i + 1]) > np.log(1 + SPLIT_JUMP) * 0.7:
            return False, f"split_jump at bar {i}"
    return True, "ok"


def audit_cache(cache_dir, verbose=True):
    """Run the gates over an EXISTING cache. Use it before trusting one."""
    import pickle
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pkl"))
    bad = []
    for f in files:
        t = os.path.splitext(f)[0]
        try:
            df = pd.read_pickle(os.path.join(cache_dir, f))
        except Exception as e:
            bad.append((t, f"unreadable: {e}"))
            continue
        ok, why = check_frame(df, t)
        if not ok:
            bad.append((t, why))
    if verbose:
        print(f"\n  AUDIT of {cache_dir}: {len(files)} frames, "
              f"{len(bad)} would be REFUSED")
        from collections import Counter
        for reason, k in Counter(w.split(" (")[0] for _, w in bad).most_common():
            print(f"    {reason:20s} {k:5d}")
        for t, w in bad[:6]:
            print(f"      e.g. {t}: {w}")
    return bad


# =============================================================================
# DOWNLOAD
# =============================================================================
def download(tickers, out_cache=OUT_CACHE, start=START, end=END, batch=BATCH,
             sleep=SLEEP, verbose=True):
    """
    Batched daily OHLCV. Failures are recorded with a reason - a name that cannot
    be fetched is usually a name that stopped trading, and that list is the input
    V58 needs to bound survivorship bias.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("pip install yfinance")
    os.makedirs(out_cache, exist_ok=True)
    have = {os.path.splitext(f)[0] for f in os.listdir(out_cache)
            if f.endswith(".pkl")}
    todo = [t for t in tickers if t not in have]
    kept, failed = list(have), []
    if verbose:
        print(f"\n  downloading {len(todo):,} tickers "
              f"({len(have):,} already cached) in batches of {batch}")

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            raw = yf.download(chunk, start=start, end=end, interval="1d",
                              auto_adjust=True, group_by="ticker",
                              progress=False, threads=True)
        except Exception as e:
            failed += [(t, f"download error: {e}") for t in chunk]
            continue
        for t in chunk:
            try:
                df = raw[t] if len(chunk) > 1 else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            except Exception as e:
                failed.append((t, f"parse: {e}"))
                continue
            ok, why = check_frame(df, t)
            if not ok:
                failed.append((t, why))
                continue
            df.to_pickle(os.path.join(out_cache, f"{t}.pkl"))
            kept.append(t)
        if verbose:
            print(f"    [{min(i + batch, len(todo)):,}/{len(todo):,}] "
                  f"kept {len(kept):,} | refused {len(failed):,}")
        time.sleep(sleep)
    return kept, failed


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_build_v68(pit_file=PIT_FILE, target=TARGET_NAMES,
                  min_assets=MIN_ASSETS_USD, out_cache=OUT_CACHE,
                  do_download=False, audit_existing=None, offline=False,
                  fallback_cache=None, email=EMAIL, out=MANIFEST,
                  verbose=True):
    global EMAIL
    EMAIL = email
    print("=" * 88)
    print("V68 - BUILD A BIGGER, POINT-IN-TIME PRICE CACHE")
    print("=" * 88)

    if audit_existing:
        audit_cache(audit_existing, verbose)

    u = build_list(pit_file, target, min_assets, offline,
                   fallback_cache=fallback_cache, verbose=verbose)
    tickers = sorted(u["ticker"].astype(str))
    print(f"\n  universe: {len(tickers):,} tickers -> {UNIVERSE_CSV}")
    print(f"  first 12: {', '.join(tickers[:12])}")

    kept, failed = [], []
    if do_download:
        kept, failed = download(tickers, out_cache, verbose=verbose)
        print(f"\n  cached {len(kept):,} | refused {len(failed):,}")
        from collections import Counter
        for reason, k in Counter(w.split(" (")[0] for _, w in failed).most_common():
            print(f"    {reason:24s} {k:5d}")
        print(f"\n  the refused list is the survivorship input: feed it to "
              f"evaluate_survivorbias_v58.py")
    else:
        print(f"\n  dry run - pass --download to fetch prices")

    print(f"\n  WHAT TO EXPECT FROM THE EXTRA NAMES")
    print(f"    n_eff at the V66 10% cut was 269 on 337 tickers. Large US "
          f"equities share a market")
    print(f"    factor, so {len(tickers):,} names will NOT give "
          f"{len(tickers)/337:.1f}x the evidence. Expect 1.5-2x, and")
    print(f"    expect most of it from the names LEAST like the 337 you have - "
          f"smaller, other sectors.")
    print(f"    If the clustered interval on the `spaced` rule tightens from "
          f"[42.8%, 51.2%] to")
    print(f"    something like [44%, 49%], the sniper is established. If it "
          f"stays as wide, it is not.")

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_universe": len(tickers), "tickers": tickers,
                   "kept": kept, "failed": failed,
                   "min_assets_usd": min_assets}, f, indent=2, default=str)
    print(f"\n  wrote {out}")
    return tickers, kept, failed


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pit-file", default=PIT_FILE)
    ap.add_argument("--target", type=int, default=TARGET_NAMES)
    ap.add_argument("--min-assets", type=float, default=MIN_ASSETS_USD)
    ap.add_argument("--out-cache", default=OUT_CACHE)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--audit", default=None,
                    help="run the quality gates over an existing cache dir")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fallback-cache", default=None,
                    help="rank an existing cache by liquidity if the "
                         "point-in-time build is unavailable")
    ap.add_argument("--email", default=EMAIL,
                    help="SEC wants a real contact in the User-Agent")
    ap.add_argument("--out", default=MANIFEST)
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_build_v68(c.pit_file, c.target, c.min_assets, c.out_cache,
                  c.download, c.audit, c.offline, c.fallback_cache, c.email,
                  c.out)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_PIT_FILE   = PIT_FILE          # V49's point-in-time frame
    RUN_TARGET     = TARGET_NAMES      # how many names to aim for
    RUN_MIN_ASSETS = MIN_ASSETS_USD    # size floor for entering the universe
    RUN_OUT_CACHE  = OUT_CACHE
    RUN_DOWNLOAD   = False             # True actually fetches prices
    RUN_AUDIT      = "price_cache_v43" # run the gates over this cache first
    RUN_OFFLINE    = False             # True refuses to touch the network

    # SEC asks for a real contact address in the User-Agent and throttles
    # requests without one. Put yours here.
    RUN_EMAIL      = "you@example.com"

    # If the point-in-time build cannot run, rank this existing cache by
    # liquidity instead. Gets you moving; carries survivorship bias, so it is
    # not what you publish from.
    RUN_FALLBACK   = "price_cache_v43"

    RUN_OUT        = MANIFEST
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_build_v68(pit_file=RUN_PIT_FILE, target=RUN_TARGET,
                      min_assets=RUN_MIN_ASSETS, out_cache=RUN_OUT_CACHE,
                      do_download=RUN_DOWNLOAD, audit_existing=RUN_AUDIT,
                      offline=RUN_OFFLINE, fallback_cache=RUN_FALLBACK,
                      email=RUN_EMAIL, out=RUN_OUT)
