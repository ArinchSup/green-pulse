"""
V107 - BUILD OR REFRESH THE PRICE FOLDERS (instead of keeping them in git).

The price folders are thousands of files, so they stay out of GitHub. What
goes into GitHub is one small text file per folder - the list of tickers in
it - and this script turns the list back into the folder on any machine.

    1. On your PC, once:   RUN_MODE = "list"
       writes tickers_price_cache_v43.txt and tickers_price_cache_v68.txt
       from the folders you have now. Commit those two files.

    2. On the server:      RUN_MODE = "download"
       downloads every ticker in each list (daily prices from 2005, free,
       yfinance) into its folder. A fresh server: builds the folders. Run it
       again later: refreshes them. Files written in the last
       RUN_SKIP_NEWER_THAN_DAYS are skipped, so an interrupted run continues
       where it stopped.

THE FOLDERS
-----------
  price_cache_v43   the reference stocks. Each month's ENTER line is the 99th
                    percentile of the models' scores on these stocks over the
                    year before the month began. NEEDED. Refresh it at the
                    start of every month; otherwise answers still come back,
                    with a warning that the reference stocks end N days before
                    the month.
  price_cache_v68   the wide universe, where a ticker's prices are looked up.
                    Not needed for today's answers if the app calls
                    get_entry(..., refresh=True) (that downloads the one ticker
                    into price_cache_live/, once a day). Needed for the
                    model-building scripts and as a fallback.

entry_cache_v90/ and price_cache_live/ are made by the API itself - never
upload them. After refreshing price_cache_v43, restart the app (or call
entry_api_v105.clear_cache()) so it re-scores the reference stocks; that takes
a few minutes per horizon, once.

A fresh download carries today's dividend adjustments, so scores can differ
from your PC's in the last decimals. Tickers that no longer trade cannot be
downloaded and are listed at the end.
"""

import datetime as dt
import os
import time

import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]
GATED = {"price_cache_v68"}       # V68 built this one with its quality gates
MIN_BARS = 260                    # the models need a year of history


def list_file(folder):
    return f"tickers_{folder}.txt"


def write_lists(folders):
    for f in folders:
        if not os.path.isdir(f):
            print(f"  {f}: folder not found - skipped")
            continue
        names = sorted(os.path.splitext(x)[0] for x in os.listdir(f)
                       if x.endswith(".pkl"))
        with open(list_file(f), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        print(f"  {list_file(f)}: {len(names):,} tickers - commit this file")


def read_list(folder):
    with open(list_file(folder), encoding="utf-8") as fh:
        return [x.strip() for x in fh if x.strip()]


def fresh(path, days):
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return days > 0 and age < days * 86400


def one_frame(raw, t, n_chunk):
    """The ticker's OHLCV from a yfinance download (any column layout)."""
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0)
        df = raw[t] if t in lv0 else raw.xs(t, axis=1, level=1)
    elif n_chunk == 1:
        df = raw
    else:
        raise KeyError(t)
    df = df[COLS].dropna()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def basic_check(df):
    if not len(df):
        return False, "no data (delisted or renamed?)"
    if len(df) < MIN_BARS:
        return False, f"only {len(df)} days of prices"
    if (df["Close"] <= 0).any():
        return False, "zero or negative prices"
    return True, "ok"


def download(folder, start, batch, skip_days, pause):
    import yfinance as yf
    gate = None
    if folder in GATED:
        try:
            from build_universe_v68 import check_frame as gate
        except Exception:
            gate = None
    names = read_list(folder)
    os.makedirs(folder, exist_ok=True)
    todo = [t for t in names
            if not fresh(os.path.join(folder, f"{t}.pkl"), skip_days)]
    print(f"\n  {folder}: {len(names):,} tickers, {len(todo):,} to download "
          f"({len(names) - len(todo):,} already fresh)")
    done, failed = 0, []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            raw = yf.download(chunk, start=start, interval="1d",
                              auto_adjust=True, group_by="ticker",
                              progress=False, threads=True)
        except Exception as e:
            failed += [(t, f"download error: {e}") for t in chunk]
            continue
        for t in chunk:
            path = os.path.join(folder, f"{t}.pkl")
            try:
                df = one_frame(raw, t, len(chunk))
            except Exception:
                failed.append((t, "no data (delisted or renamed?)"))
                continue
            ok, why = basic_check(df)
            if ok and gate is not None:
                ok, why = gate(df, t)
            if not ok:
                failed.append((t, why + (" - old file kept"
                                         if os.path.exists(path) else "")))
                continue
            tmp = path + ".tmp"
            df.to_pickle(tmp)
            os.replace(tmp, path)          # never a half-written file
            done += 1
        print(f"    [{min(i + batch, len(todo)):,}/{len(todo):,}] "
              f"saved {done:,} | failed {len(failed):,}")
        time.sleep(pause)
    have = sum(x.endswith(".pkl") for x in os.listdir(folder))
    print(f"  {folder}: {have:,} of {len(names):,} tickers on disk")
    for t, why in failed[:30]:
        print(f"    {t:<8} {why}")
    if len(failed) > 30:
        print(f"    ... and {len(failed) - 30} more")
    return done, failed


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODE = "download"           # "list" on your PC, "download" on the server
    RUN_FOLDERS = ["price_cache_v43", "price_cache_v68"]
    RUN_START = "2005-01-01"    # as the folders were built
    RUN_BATCH = 40              # tickers per download call
    RUN_SKIP_NEWER_THAN_DAYS = 1    # files this fresh are not downloaded again
    RUN_PAUSE_SECONDS = 1.0     # between calls, to stay polite to the source
    # -------------------------------------------------------------------------

    print(f"  {dt.datetime.now():%Y-%m-%d %H:%M}  mode {RUN_MODE}")
    if RUN_MODE == "list":
        write_lists(RUN_FOLDERS)
    elif RUN_MODE == "download":
        for f in RUN_FOLDERS:
            if not os.path.exists(list_file(f)):
                print(f"  {list_file(f)} not found - run RUN_MODE = 'list' on "
                      f"the PC that has {f}/, and commit the file")
                continue
            download(f, RUN_START, RUN_BATCH, RUN_SKIP_NEWER_THAN_DAYS,
                     RUN_PAUSE_SECONDS)
        print("\n  done. If the app is running, restart it (or call "
              "entry_api_v105.clear_cache()) so it uses the new prices.")
    else:
        print("  RUN_MODE must be 'list' or 'download'")
