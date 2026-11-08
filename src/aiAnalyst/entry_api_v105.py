"""
V105 - ENTRY API: ONE FUNCTION FOR THE WHOLE PROJECT.

    from entry_api_v105 import get_entry

    r = get_entry("AMD", "MID", "2026-09-25")   # ticker, horizon, date
    r = get_entry("AMD", "LONG")                # no date = the latest bar

    if r["enter"]:
        print(r["entry"], r["stop"], r["target"], r["holding_bars"])

    # all three horizons for one stock
    three = {h: get_entry("AMD", h) for h in ("SHORT", "MID", "LONG")}

It answers the same question as entry_recommender_v103, with V103's own code,
so the answers are identical:

    "I have already decided to buy THIS stock around THIS date.
     Is this a good day to enter, or should I wait?"

It never says which stock to buy.

HORIZONS
--------
  "SHORT"  V101 model, 10-trading-day trade
  "MID"    V88 model, 60-trading-day trade
  "LONG"   V103 model, 250-trading-day (one-year) trade
  (upper or lower case both work)

THE DATE
--------
  None, "2026-09-25", a datetime/date or a pandas Timestamp. The model scores
  the stock's last trading day ON OR BEFORE that date (a weekend gives the
  Friday) and assumes you enter at that day's close.

WHAT YOU GET BACK (a dict - the same keys every time)
----------------------------------------------------
  signal        "ENTER" or "WAIT". When there is no answer:
                "NO DATA" (no price file for the ticker), "NOT SCORABLE"
                (under 260 bars of history, missing data, no valid
                stop/target) or "NO CUT" (the reference universe is too short
                before that month to set the ENTER line)
  enter         True only for ENTER
  ok            True when the model gave an answer (ENTER or WAIT)
  ticker, horizon
  requested     the date asked for ("YYYY-MM-DD") or "latest"
  bar_date      the trading day actually scored ("YYYY-MM-DD")
  score         the model's score for that day
  cut           that month's ENTER line: the 99th percentile of the scores of
                the trailing year. ENTER only when score is ABOVE it
  top_pct_of_window  where the score ranks in that year (0.004 = top 0.4%)
  score_is_probability  False for all three models: the score ranks days, it
                is not a calibrated chance - do not show it as a percentage
  entry         that day's close (the assumed entry price)
  stop, target  price levels;  stop_pct, target_pct  distance in percent
  rr            reward-to-risk (about 1.67);  breakeven_win_rate  0.375
  holding_bars  the most trading days the trade is held (10, 60 or 250)
  in_sample     True when that day is inside the model's training period:
                it shows what the model says, not an out-of-sample result
  stale         True when the prices end more than 7 days before the date
                asked for (or before today, for the latest bar)
  meaning       one plain sentence on what the signal means
  model, trained_through, price_cache
  warnings      a list of plain-text warnings (empty when there are none)

Values that do not exist for an answer are None (for example score and
entry under "NO DATA"). Everything is a plain Python type (str, float, int,
bool, list), so a result can go straight into JSON, a DataFrame or a prompt.
A wrong horizon raises ValueError; a missing model file raises
FileNotFoundError. A ticker without data does NOT raise - it comes back as
"NO DATA".

ALSO HERE
---------
  get_entries(tickers, horizons, date)  many at once -> a pandas DataFrame
  model_info(horizon)                   which model, trained through when,
                                        and what its signal has meant
  clear_cache()                         forget loaded models and answers
                                        (after retraining or new price files)

  refresh=True on get_entry / get_entries downloads the latest daily prices
  for that ticker first (free, yfinance; at most once per ticker per day)
  into price_cache_live/, which is checked before the other price caches.

SPEED
-----
The first call for a horizon loads its model (about a second). If that
model's reference universe has not been scored before, it is scored once
(a few minutes) and cached in entry_cache_v90/ - shared with V90/V101/V103,
so after running V103 there is nothing to wait for. After that a call takes
about 0.1-0.3 s, and the same request again is answered from memory.

The model files are looked up in class_model/ and the price folders next to
this file, so it works from any folder (for example the GreenPulse root). Calls from several threads take turns (the SHORT and LONG models
briefly change trade_config's target limits while they run).
"""

import copy
import datetime as dt
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

HOME = os.path.dirname(os.path.abspath(__file__))   # models + caches are here
if HOME not in sys.path:
    sys.path.insert(0, HOME)

import entry_recommender_v103 as V103        # noqa: E402  (the tested code)

# =============================================================================
# SETTINGS - only change these if your folders are different
# =============================================================================
PRICE_CACHES = ["price_cache_v68"]  # where to find a ticker's prices (V103's)
LIVE_CACHE = "price_cache_live"     # refreshed prices; checked first
UNIVERSE_CACHE = None               # None = each model's own training cache
CACHE_DIR = V103.CACHE_DIR          # reference-universe scores (shared)
VERBOSE = True                      # one line if a universe must be scored
MEMO_SIZE = 20000                   # answers kept in memory

HORIZONS = ("SHORT", "MID", "LONG")
FIELDS = ["ticker", "horizon", "requested", "bar_date", "signal", "enter",
          "ok", "score", "cut", "top_pct_of_window", "score_is_probability",
          "entry", "stop", "target", "stop_pct", "target_pct", "rr",
          "breakeven_win_rate", "holding_bars", "in_sample", "stale",
          "meaning", "model", "trained_through", "price_cache", "warnings"]
MEANING = {
    "ENTER": "a top-1% entry day: if you are buying this stock for this "
             "horizon, this is a good day to do it",
    "WAIT": "not a top-1% entry day: if you are buying this stock for this "
            "horizon, waiting for a signal day has been better",
    "NO DATA": "no answer: there is no price data for this ticker",
    "NOT SCORABLE": "no answer: this day cannot be scored (under 260 bars of "
                    "history, missing data, or no valid stop/target)",
    "NO CUT": "no answer: the reference universe has too little history "
              "before this month to set the ENTER line",
}

_LOCK = threading.RLock()
_RECS = {}                 # horizon -> (stamp, V103.EntryRecommender)
_MEMO = OrderedDict()      # request -> answer
_REFRESHED = {}            # ticker -> (day, downloaded?)


# =============================================================================
# HELPERS
# =============================================================================
def _abs(path):
    """A path next to this file, unless it is already absolute."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(HOME, path)


def _horizon(horizon):
    h = str(horizon).strip().upper()
    if h not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}: use 'SHORT' (10-day), "
                         f"'MID' (60-day) or 'LONG' (one-year)")
    return h


def _date(date):
    if date is None:
        return None
    d = pd.Timestamp(date)
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    return d.normalize()


def _plain(v):
    """numpy / pandas / datetime values -> plain Python (NaN -> None)."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date, np.datetime64)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if np.isfinite(v) else None
    return v


def _label(rec):
    v = rec.prov.get("version")
    if not v:                                   # older files: from the name
        m = re.search(r"v(\d+)", os.path.basename(rec.model_file), re.I)
        v = m.group(0) if m else os.path.basename(rec.model_file)
    return str(v).upper()


def _recommender(h):
    """One V103 recommender per horizon, kept for the whole run (rebuilt only
    if the model file or a setting changes)."""
    reg = V103.MODEL_REGISTRY[h]
    path = _abs(reg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - train it first ({reg['module']}.py) or copy "
            f"the model file there")
    stamp = (path, os.path.getmtime(path), _abs(UNIVERSE_CACHE),
             _abs(CACHE_DIR))
    got = _RECS.get(h)
    if got is not None and got[0] == stamp:
        return got[1]
    rec = V103.EntryRecommender(h, model_file=path,
                                universe_cache=_abs(UNIVERSE_CACHE),
                                price_caches=[], live_cache=None,
                                cache_dir=_abs(CACHE_DIR), verbose=VERBOSE)
    # a model's own training cache may be stored relative to the project
    rec.universe_cache = _abs(rec.universe_cache)
    _RECS[h] = (stamp, rec)
    return rec


def _price_caches(rec):
    """V103's search order (live cache, PRICE_CACHES, the model's universe),
    as absolute paths, checked on every call so a refresh is seen at once."""
    c = [_abs(LIVE_CACHE)] + [_abs(p) for p in PRICE_CACHES] + \
        [rec.universe_cache]
    return [p for i, p in enumerate(c)
            if p and os.path.isdir(p) and p not in c[:i]]


def _price_stamp(caches, ticker):
    """Which price file answers for this ticker, and its version."""
    for c in caches:
        p = os.path.join(c, f"{ticker}.pkl")
        if os.path.exists(p):
            st = os.stat(p)
            return p, st.st_mtime_ns, st.st_size
    return None


def _refresh(ticker):
    """Download the latest prices once per ticker per day. True if the live
    file is from today."""
    today = dt.date.today()
    if ticker in _REFRESHED and _REFRESHED[ticker][0] == today:
        return _REFRESHED[ticker][1]
    live = _abs(LIVE_CACHE)
    p = os.path.join(live, f"{ticker}.pkl")
    fresh = os.path.exists(p) and \
        dt.date.fromtimestamp(os.path.getmtime(p)) == today
    if not fresh:
        V103.refresh_prices([ticker], live, verbose=VERBOSE)
        fresh = os.path.exists(p) and \
            dt.date.fromtimestamp(os.path.getmtime(p)) == today
    _REFRESHED[ticker] = (today, fresh)
    return fresh


def _answer(raw, rec, ticker, req, today):
    """V103's dict -> the API's dict: fixed keys, plain types, flags."""
    out = dict.fromkeys(FIELDS)
    for k, v in raw.items():
        if k in out and k != "warnings":
            out[k] = _plain(v)
    sig = raw.get("signal")
    out.update(ticker=ticker, horizon=rec.horizon,
               requested="latest" if req is None else _plain(req),
               enter=sig == "ENTER", ok=sig in ("ENTER", "WAIT"),
               meaning=MEANING.get(sig, ""), model=_label(rec),
               trained_through=_plain(rec.trained_through),
               holding_bars=int(rec.hold),
               score_is_probability=bool(
                   rec.model.get("display_probability")))
    warn = [str(w) for w in raw.get("warnings", [])]
    if sig == "NO DATA":                 # V103's text names its runner's flags
        warn = [f"no price file for {ticker} in "
                f"{', '.join(rec.price_caches) or 'any price folder'} - call "
                f"with refresh=True to download it, or add its folder to "
                f"PRICE_CACHES in {os.path.basename(__file__)}"]
    bar = raw.get("bar_date")
    if bar is not None:
        bar = pd.Timestamp(bar)
        gap = ((today if req is None else req) - bar).days
        out["in_sample"] = bool(bar <= rec.trained_through)
        out["stale"] = bool(gap > V103.STALE_DAYS)
        if req is None and out["stale"]:
            warn.append(f"the latest price bar for {ticker} is "
                        f"{bar.date()}, {gap} days ago - prices are out of "
                        f"date (refresh=True downloads new ones)")
    out["warnings"] = warn
    return out


# =============================================================================
# THE API
# =============================================================================
def get_entry(ticker, horizon="MID", date=None, refresh=False):
    """
    ENTER or WAIT for buying `ticker` on `date` (None = the latest bar) for
    `horizon` ("SHORT", "MID" or "LONG"), with entry, stop and target.
    Returns a dict with the same keys every time (see the top of this file).
    """
    h = _horizon(horizon)
    t = str(ticker).upper().strip()
    req = _date(date)
    with _LOCK:
        fresh = _refresh(t) if refresh else True
        rec = _recommender(h)
        rec.price_caches = _price_caches(rec)
        today = pd.Timestamp(dt.date.today())
        stamp = _price_stamp(rec.price_caches, t)
        key = (h, t, req if req is not None else ("latest", today), stamp,
               _RECS[h][0])
        if stamp is not None and key in _MEMO:
            _MEMO.move_to_end(key)
            out = copy.deepcopy(_MEMO[key])
        else:
            out = _answer(rec.recommend(t, req), rec, t, req, today)
            if stamp is not None:
                _MEMO[key] = copy.deepcopy(out)
                while len(_MEMO) > MEMO_SIZE:
                    _MEMO.popitem(last=False)
    if not fresh:
        out["warnings"].append(f"could not download new prices for {t}; "
                               f"the answer uses the cached prices")
    return out


def get_entries(tickers, horizons=HORIZONS, date=None, refresh=False):
    """Many answers at once: one row per (ticker, horizon), in that order,
    with the same columns as get_entry's keys (warnings joined by ' | ')."""
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(horizons, str):
        horizons = [horizons]
    rows = []
    for t in tickers:
        for h in horizons:
            r = get_entry(t, h, date, refresh)
            r["warnings"] = " | ".join(r["warnings"])
            rows.append(r)
    return pd.DataFrame(rows, columns=FIELDS)


def model_info(horizon="MID"):
    """Which model answers for a horizon, and what its signal has meant."""
    h = _horizon(horizon)
    with _LOCK:
        rec = _recommender(h)
        return {"horizon": h, "model": _label(rec),
                "model_file": rec.model_file,
                "trained_through": _plain(rec.trained_through),
                "holding_bars": int(rec.hold), "note": rec.note,
                "evidence": rec.evidence(),
                "score_is_probability": bool(
                    rec.model.get("display_probability")),
                "reference_universe": rec.universe_cache}


def clear_cache():
    """Forget loaded models, answers and refresh marks (for example after
    retraining a model or rebuilding a price cache)."""
    with _LOCK:
        _RECS.clear()
        _MEMO.clear()
        _REFRESHED.clear()


# =============================================================================
# RUN ME - a quick look at what the project will get back
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_TICKERS = ["AMD", "NVDA", "AAPL"]
    RUN_HORIZONS = ["SHORT", "MID", "LONG"]
    RUN_DATE = None             # None = latest bar, or "2026-09-25"
    RUN_REFRESH = False         # True = download the latest prices first
    RUN_SHOW_DICT = True        # print one answer exactly as code receives it
    # -------------------------------------------------------------------------

    W = 100
    print("=" * W)
    print("  V105 ENTRY API - get_entry(ticker, horizon, date)")
    print("=" * W)
    for hz in RUN_HORIZONS:
        try:
            info = model_info(hz)
        except FileNotFoundError as e:
            print(f"  {hz:<5}  skipped - {e}")
            continue
        print(f"  {hz:<5}  {info['model']}, holds up to "
              f"{info['holding_bars']} trading days, trained through "
              f"{info['trained_through']}")
    have = [hz for hz in RUN_HORIZONS
            if os.path.exists(_abs(V103.MODEL_REGISTRY[_horizon(hz)]["file"]))]

    t0 = time.time()
    table = get_entries(RUN_TICKERS, have, RUN_DATE, refresh=RUN_REFRESH)
    t1 = time.time()
    get_entries(RUN_TICKERS, have, RUN_DATE)          # the same, from memory
    t2 = time.time()

    show = table[["ticker", "horizon", "bar_date", "signal", "score", "cut",
                  "entry", "stop", "target", "holding_bars", "in_sample",
                  "stale"]]
    print("\n" + show.to_string(index=False, float_format=lambda v:
                                f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"))
    print(f"\n  {len(table)} answers in {t1 - t0:.1f}s; asked again: "
          f"{t2 - t1:.2f}s (from memory)")
    w = table[table["warnings"] != ""]
    if len(w):
        print("\n  WARNINGS")
        for _, r in w.iterrows():
            for line in r["warnings"].split(" | "):
                print(f"  {r['ticker']:<6} {r['horizon']:<5}  {line}")

    if RUN_SHOW_DICT and RUN_TICKERS and have:
        r = get_entry(RUN_TICKERS[0], have[0], RUN_DATE)
        print(f"\n  WHAT YOUR CODE GETS: get_entry({RUN_TICKERS[0]!r}, "
              f"{have[0]!r}, {RUN_DATE!r})")
        print("  " + json.dumps(r, indent=2).replace("\n", "\n  "))
