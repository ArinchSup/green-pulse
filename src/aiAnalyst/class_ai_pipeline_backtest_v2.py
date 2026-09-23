"""
class_ai_pipeline_backtest_v2.py  -  Pillar 2 backtest you can trust

What changed from class_ai_pipeline_backtest_fast.py
  1. Fixed exit simulator
       - the breakeven stop is armed AFTER the bar that reaches the halfway point,
         so it only protects from the next bar on (the old code could book a
         day-1 +20% runner as a 0% breakeven)
       - a gap through the stop fills at the open, not at the stop price
       - a breakeven stop that gaps more than 0.5% below entry is booked as a Loss,
         not a Breakeven (the return was always right; now the label is too)
  2. Nothing hidden, everything repeatable
       - fixed END_DATE instead of date.today(), and a seeded draw of real trading days
       - every error is counted and printed (pillar 2, prices, fundamentals)
       - daily prices are downloaded once per ticker and cached on disk
       - every sample is saved to a CSV and the printed summary to a .txt file
  3. Comparison rows on the SAME samples with the SAME exit rules
       Random entry (no model) | All bullish calls | Model >= MIN_CONFIDENCE | + fundamentals
       plus a 95% range for each row's average and for the model-minus-random gap,
       and a split of that gap into timing (which weeks) vs stock picking (which stocks)
  4. Training overlap check: reads the training dataset and warns if the backtest
     window overlaps the data the model learned from
  5. Scope split: software/internet, chips/hardware, grey zone, out of scope
  6. Ranking check across ALL samples (does a higher score mean a better outcome?), the
     concentration test (does the lead survive without its best few stock-months?), and
     excess return vs QQQ over the holding window
  7. Pluggable pillar 2: --pillar2-module picks the module (V40: class_ai_pillar2,
     V41: class_ai_pillar2_v41) and --pillar2-model the model file for modules with load_model()

Usage
  python class_ai_pipeline_backtest_v2.py
  python class_ai_pipeline_backtest_v2.py --seed 88 --universe TECH_ONLY
  python class_ai_pipeline_backtest_v2.py --model-label V40-cut --training-dataset dataset_pillar2_mid_v40_cut20250627.json
  python class_ai_pipeline_backtest_v2.py --pillar2-module class_ai_pillar2_v41 --pillar2-model pillar2_v41_cut20250627_model.json --model-label V41 --min-confidence 0 --training-dataset dataset_pillar2_mid_v41_qqq_cut20250627.json
"""
import argparse
import datetime
import importlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from class_ai_pillar1_funda_helper import give_me_foundation
    HAVE_FUNDAMENTALS = True
except Exception:  # the fundamentals row is skipped (and reported) if the helper can't load
    HAVE_FUNDAMENTALS = False


# =============================================================================
# SETTINGS   (a * means a command-line flag can override it)
# =============================================================================
SEED             = 56                            # * --seed
SEARCH_ATTEMPTS  = 5000                          # * --attempts
UNIVERSE         = "SHAY"                        # * --universe  SHAY | TECH_ONLY | SP500
END_DATE         = datetime.date(2026, 9, 22)    # * --end-date  fixed "today": same seed = same test
MIN_DAYS_AGO     = 100                           # newest signal = END_DATE - 100 days
MAX_DAYS_AGO     = 360                           # oldest signal = END_DATE - 360 days

HOLD_BARS        = 60      # trading days per trade, same as the MID label's lookahead_bars
TARGET_PCT       = 0.20    # V39 label geometry: +20% target ...
STOP_PCT         = 0.12    # ... and -12% stop, measured from the real entry (next day's open)
MAX_LOSS_PCT     = 0.125   # hard cap on how far below entry any stop may sit
USE_BREAKEVEN    = True    # * --no-breakeven  move stop to entry once price is halfway to target
BE_TOLERANCE_PCT = 0.5     # a breakeven exit filled worse than -0.5% (a gap through it) counts as a Loss
LEVELS_MODE      = "FIXED" # * --levels  FIXED: every row uses +20/-12 from the real entry, so rows
                           #   differ ONLY in which trades they take (the fair comparison).
                           #   PILLAR2: model rows use pillar 2's actionable_levels (old behaviour).

MODEL_LABEL      = "V40"   # * --model-label  name shown in the report (e.g. "V40-cut")
PILLAR2_MODULE   = "class_ai_pillar2"   # * --pillar2-module  V41: class_ai_pillar2_v41
PILLAR2_MODEL    = None    # * --pillar2-model  model file, for modules that have load_model()
BENCHMARK        = "QQQ"   # * --benchmark  for the excess-return and "beat the benchmark" checks
MIN_CONFIDENCE   = 0.65    # * --min-confidence
FUNDA_MIN_SCORE  = 4.5
FUNDA_ON_ERROR   = "pass"  # "pass" or "veto" when the fundamentals helper fails (always counted)

TRAINING_DATASET = "dataset_pillar2_mid_v40_cut20250627.json"  # * --training-dataset  file the model was trained on
LABEL_GAP_DAYS   = 92      # a training label looks 60 trading days (~90 calendar days) ahead

PRICE_CACHE_DIR      = "price_cache"
PRICE_CACHE_START    = datetime.date(2015, 1, 1)
CACHE_PILLAR2_PRICES = True   # serve pillar 2's plain start/end yfinance calls from the same cache
OUTPUT_DIR           = "backtest_runs"
BOOTSTRAP_ROUNDS     = 2000
SAME_WEEK_DAYS       = 3      # timing vs picking: compare each signal with random entries within +/- 3 days
RANK_BOOT_ROUNDS     = 1000   # resamples for the ranking-check ranges (done in chunks to save memory)


# =============================================================================
# TICKER UNIVERSE AND SCOPE GROUPS
# =============================================================================
SHAY_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI", "CRDO",
]

# Edit these to change what counts as "high-growth tech"
TICKER_GROUPS = {
    "software_internet": ["MSFT", "GOOGL", "META", "AMZN", "ZS", "NET", "PANW", "CRWD", "PATH",
                          "MDB", "SNOW", "DOCN", "PLTR", "ORCL", "RDDT", "DUOL", "NBIS"],
    "chips_hardware":    ["NVDA", "AMD", "AVGO", "MU", "MRVL", "ALAB", "CRDO", "AEHR", "COHR",
                          "LITE", "AAOI", "OSS", "INTC", "ASML", "TSM", "AAPL"],
    "grey_zone":         ["TSLA", "SYM", "SOFI", "MSTR", "WULF", "CIFR", "IREN", "APLD",
                          "RKLB", "ASTS", "RDW", "BKSY", "UMAC", "ONDS"],
    "out_of_scope":      ["CEG", "GEV", "OKLO", "LEU", "BE", "FCX", "MP", "UUUU", "UAMY",
                          "IDR", "CRML", "LLY", "OSCR", "ISRG", "PAYX"],
}
TICKER_TO_GROUP = {t: g for g, members in TICKER_GROUPS.items() for t in members}
TECH_GROUPS = ("software_internet", "chips_hardware")


def group_of(ticker):
    return TICKER_TO_GROUP.get(ticker, "other")


def load_universe(name):
    if name == "SHAY":
        return list(SHAY_TICKERS)
    if name == "TECH_ONLY":
        return [t for g in TECH_GROUPS for t in TICKER_GROUPS[g]]
    if name == "SP500":
        # No silent fallback: if this fails the run stops, so an "S&P" run can never
        # quietly turn into a SHAY run. Note: today's members only (survivorship bias).
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))[0]
        return [str(t).replace(".", "-") for t in table["Symbol"].tolist()]
    raise ValueError(f"Unknown universe: {name}")


# =============================================================================
# ERROR LOG
# =============================================================================
class ErrorLog:
    """Counts problems per stage and keeps a few example messages instead of hiding them."""

    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, stage, exc=None):
        self.counts[stage] += 1
        if exc is not None:
            msg = f"{type(exc).__name__}: {exc}"[:150]
            if msg not in self.examples[stage] and len(self.examples[stage]) < 3:
                self.examples[stage].append(msg)


# =============================================================================
# PRICE CACHE
# =============================================================================
_ORIGINAL_HISTORY = yf.Ticker.history


def _as_index_ts(value, index):
    """Turn a date/string into a Timestamp comparable with a (possibly tz-aware) index."""
    ts = pd.Timestamp(value)
    tz = getattr(index, "tz", None)
    if tz is not None:
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


class PriceStore:
    """Downloads each ticker's daily history once, keeps it on disk, serves slices of it."""

    def __init__(self, cache_dir, start, end, errors):
        self.start, self.end, self.errors = start, end, errors
        self.dir = os.path.join(cache_dir, f"{start:%Y%m%d}_{end:%Y%m%d}")
        os.makedirs(self.dir, exist_ok=True)
        self.mem = {}

    def full(self, ticker):
        if ticker in self.mem:
            return self.mem[ticker]
        path = os.path.join(self.dir, f"{ticker}.pkl")
        df = None
        if os.path.exists(path):
            try:
                df = pd.read_pickle(path)
            except Exception as e:
                self.errors.add("price cache file unreadable (re-downloading)", e)
        if df is None:
            try:
                df = _ORIGINAL_HISTORY(yf.Ticker(ticker), start=self.start.isoformat(),
                                       end=(self.end + datetime.timedelta(days=1)).isoformat())
                if df is None or df.empty:
                    raise ValueError("empty price history")
                df.to_pickle(path)
            except Exception as e:
                self.errors.add("price download failed", e)
                df = None
        self.mem[ticker] = df
        return df

    def trading_days(self, ticker, first, last):
        df = self.full(ticker)
        if df is None:
            return []
        lo, hi = _as_index_ts(first, df.index), _as_index_ts(last, df.index)
        return [ts.date() for ts in df.index[(df.index >= lo) & (df.index <= hi)]]

    def forward_bars(self, ticker, signal_day, n_bars):
        """The n trading days strictly AFTER the signal day. Entry = first bar's open."""
        df = self.full(ticker)
        if df is None:
            return None
        return df[df.index > _as_index_ts(signal_day, df.index)].iloc[:n_bars]


def install_pillar2_price_cache(store):
    """
    Serve plain daily history(start=..., end=...) calls from the cache, so pillar 2 and the
    simulator read identical prices and Yahoo isn't hit 1,000 times. Any call with other
    options (period=, intraday, adjusted differently, ...) goes to Yahoo exactly as before.
    """
    defaults = {"interval": "1d", "auto_adjust": True, "actions": True, "prepost": False,
                "back_adjust": False, "repair": False, "keepna": False, "rounding": False}
    passthrough = {"start", "end", "timeout", "raise_errors"}

    def cached_history(self, *args, **kwargs):
        plain = (not args and "start" in kwargs and kwargs.get("end") is not None
                 and all(k in passthrough or (k in defaults and v == defaults[k])
                         for k, v in kwargs.items()))
        if plain:
            df = store.full(self.ticker)
            if df is not None and len(df):
                start = _as_index_ts(kwargs["start"], df.index)
                end = _as_index_ts(kwargs["end"], df.index)
                cache_lo = _as_index_ts(store.start, df.index)
                cache_hi = _as_index_ts(store.end + datetime.timedelta(days=1), df.index)
                if cache_lo <= start and end <= cache_hi:
                    return df[(df.index >= start) & (df.index < end)].copy()
        return _ORIGINAL_HISTORY(self, *args, **kwargs)

    yf.Ticker.history = cached_history


# =============================================================================
# TRAINING OVERLAP CHECK
# =============================================================================
class TrainingGuard:
    """Checks whether the backtest window overlaps the data the model was trained on."""

    def __init__(self, path, first_day, last_day):
        self.keys, self.tickers, self.lines = set(), set(), []
        self.status = "unchecked"
        if not path or not os.path.exists(path):
            self.lines.append(f"Training dataset not found at '{path}', so overlap can't be checked.")
            self.lines.append("Pass --training-dataset with the JSON the loaded model was trained on.")
            return
        try:
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            self.lines.append(f"Could not read '{path}': {type(e).__name__}: {e}")
            return
        dates = []
        for r in rows:
            d, t = str(r.get("signal_date", ""))[:10], r.get("ticker")
            if d and t:
                dates.append(d)
                self.keys.add((t, d))
                self.tickers.add(t)
        if not dates:
            self.lines.append(f"No signal dates found in '{path}'.")
            return
        first_s, last_s = first_day.isoformat(), last_day.isoformat()
        safe_last = (first_day - datetime.timedelta(days=LABEL_GAP_DAYS)).isoformat()
        inside = sum(first_s <= d <= last_s for d in dates)
        self.lines.append(f"Training rows: {len(dates)} | signal dates {min(dates)} -> {max(dates)} "
                          f"| {len(self.tickers)} tickers")
        self.lines.append(f"Backtest signals: {first_s} -> {last_s} | latest safe training date: {safe_last}")
        if max(dates) > safe_last:
            self.status = "IN-SAMPLE"
            self.lines.append(f"!! IN-SAMPLE: {inside} training rows ({inside / len(dates):.0%}) sit inside the "
                              f"backtest window and training labels reach into it.")
            self.lines.append("   The model has already seen these months, so this run can't measure real skill.")
            self.lines.append(f"   Retrain on signals dated on or before {safe_last}, then rerun.")
        else:
            self.status = "clean"
            self.lines.append("OK: every training label ends before the first backtest signal.")


# =============================================================================
# EXIT SIMULATOR (fixed)
# =============================================================================
def simulate_trade(bars, target, stop, use_breakeven=True, max_loss_pct=MAX_LOSS_PCT):
    """
    bars: daily OHLC starting the trading day AFTER the signal. Entry = first bar's open.
    Where a daily bar can't tell us the order of events, assume the worse case:
      - the stop is checked before the target (if a bar hits both, it counts as the stop)
      - a gap through the stop fills at the open, not at the stop price
      - if the breakeven stop was armed but the fill is worse than -BE_TOLERANCE_PCT (it gapped
        through), the exit is booked as a Loss: the return is what counts, the label follows it
      - a gap through the target fills at the open (what a resting sell order would get)
      - the breakeven stop is armed AFTER the bar that reaches halfway, active from the next bar
    Returns (outcome, return_pct, exit_day), or None if the levels make no sense.
    """
    opens = bars["Open"].to_numpy(dtype=float)
    highs = bars["High"].to_numpy(dtype=float)
    lows = bars["Low"].to_numpy(dtype=float)
    closes = bars["Close"].to_numpy(dtype=float)
    if len(opens) == 0:
        return None
    entry = opens[0]
    if not np.isfinite(entry) or entry <= 0 or not target > entry:
        return None
    stop = max(stop, entry * (1.0 - max_loss_pct))
    if stop >= entry:
        return None
    halfway = entry + (target - entry) * 0.5
    armed = False
    for day in range(len(opens)):
        o, h, low = opens[day], highs[day], lows[day]
        if low <= stop:
            fill = min(o, stop)
            ret = (fill - entry) / entry * 100.0
            # a breakeven stop that gaps well below entry is a loss, not a breakeven
            outcome = "Breakeven" if armed and ret > -BE_TOLERANCE_PCT else "Loss"
            return outcome, ret, day + 1
        if h >= target:
            fill = max(o, target)
            return "Win", (fill - entry) / entry * 100.0, day + 1
        if use_breakeven and not armed and h >= halfway:
            armed, stop = True, max(stop, entry)
    return "Expired", (closes[-1] - entry) / entry * 100.0, len(opens)


# =============================================================================
# FUNDAMENTALS (pillar 1)
# =============================================================================
_funda_cache = {}


def composite_fundamental(ticker, errors):
    """
    Average of profitability / growth / financial_health (0-10), or None if unavailable.
    WARNING: the helper returns TODAY's fundamentals, which is look-ahead for past trades.
    """
    if ticker in _funda_cache:
        return _funda_cache[ticker]
    score = None
    try:
        data = give_me_foundation(ticker) or {}
        vals = [data.get(k) for k in ("profitability", "growth", "financial_health")]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            score = round(sum(vals) / len(vals), 1)
        else:
            errors.add("fundamentals returned no scores (counted once per ticker)")
    except Exception as e:
        errors.add("fundamentals raised an error (counted once per ticker)", e)
    _funda_cache[ticker] = score
    return score


# =============================================================================
# MAIN LOOP
# =============================================================================
def run(cfg, errors, skips, guard, tickers, first_day, last_day, pillar2_fn):
    store = PriceStore(PRICE_CACHE_DIR, PRICE_CACHE_START, cfg.end_date, errors)
    if CACHE_PILLAR2_PRICES:
        install_pillar2_price_cache(store)
    rng = random.Random(cfg.seed)
    samples, seen = [], set()

    for attempt in range(1, cfg.attempts + 1):
        ticker = rng.choice(tickers)
        days = store.trading_days(ticker, first_day, last_day)
        if not days:
            skips["ticker has no price history in the window"] += 1
            continue
        day = rng.choice(days)
        if (ticker, day) in seen:
            skips["same ticker and day drawn twice"] += 1
            continue
        seen.add((ticker, day))
        date_str = day.isoformat()

        try:
            tech = pillar2_fn(ticker, target_date_str=date_str)
        except Exception as e:
            errors.add("pillar 2 raised an error", e)
            continue
        if not tech:
            errors.add("pillar 2 returned nothing")
            continue

        bars = store.forward_bars(ticker, day, HOLD_BARS)
        if bars is None or len(bars) < int(HOLD_BARS * 0.8):
            errors.add("not enough future price bars")
            continue

        entry = float(bars["Open"].iloc[0])
        fixed_target, fixed_stop = entry * (1 + TARGET_PCT), entry * (1 - STOP_PCT)
        rand_main = simulate_trade(bars, fixed_target, fixed_stop, cfg.use_breakeven)
        rand_alt = simulate_trade(bars, fixed_target, fixed_stop, not cfg.use_breakeven)
        if rand_main is None:
            errors.add("unusable entry price")
            continue

        # The trade under the TRAINING label's rules (+20% before -12%, no breakeven stop)
        label_trade = rand_alt if cfg.use_breakeven else rand_main

        # Buy and hold over the same bars vs the benchmark (the V41 target; exits ignored)
        hold_ret = (float(bars["Close"].iloc[-1]) - entry) / entry * 100.0
        bench_ret = excess_ret = np.nan
        bench_df = store.full(cfg.benchmark)
        if bench_df is not None:
            try:
                b_open = float(bench_df.loc[bars.index[0], "Open"])
                b_close = float(bench_df.loc[bars.index[-1], "Close"])
                bench_ret = (b_close - b_open) / b_open * 100.0
                excess_ret = hold_ret - bench_ret
            except KeyError:
                errors.add(f"{cfg.benchmark} has no bar on the entry or exit date")

        sentiment = str(tech.get("technical_sentiment", ""))
        try:
            conf = float(tech.get("confidence") or 0.0)
        except (TypeError, ValueError) as e:
            errors.add("confidence is not a number", e)
            conf = 0.0
        bullish = sentiment == "Bullish"

        model_main = model_alt = None
        if bullish:
            if cfg.levels == "FIXED":
                model_main, model_alt = rand_main, rand_alt
            else:
                try:
                    lv = tech["actionable_levels"]
                    tgt, stp = float(lv["target_price"]), float(lv["stop_loss"])
                    model_main = simulate_trade(bars, tgt, stp, cfg.use_breakeven)
                    model_alt = simulate_trade(bars, tgt, stp, not cfg.use_breakeven)
                except Exception as e:
                    errors.add("pillar 2 levels unreadable", e)
                if model_main is None:
                    errors.add("pillar 2 levels unusable (bullish call dropped)")
                    bullish = False
        is_model = bullish and conf >= cfg.min_confidence

        funda_score, funda_pass = None, False
        if is_model and HAVE_FUNDAMENTALS:
            funda_score = composite_fundamental(ticker, errors)
            if funda_score is None:
                skips[f"model trades with no fundamentals score (treated as '{FUNDA_ON_ERROR}')"] += 1
            funda_pass = (funda_score >= FUNDA_MIN_SCORE if funda_score is not None
                          else FUNDA_ON_ERROR == "pass")

        samples.append({
            "attempt": attempt,
            "ticker": ticker,
            "group": group_of(ticker),
            "signal_date": date_str,
            "cluster": f"{ticker}|{date_str[:7]}",
            "in_training_set": (ticker, date_str) in guard.keys,
            "sentiment": sentiment,
            "confidence": round(conf, 4),
            "entry_date": str(bars.index[0].date()),
            "entry_price": round(entry, 4),
            "rand_outcome": rand_main[0],
            "rand_ret": round(rand_main[1], 4),
            "rand_exit_day": rand_main[2],
            "rand_ret_alt": round(rand_alt[1], 4),
            "label_outcome": label_trade[0],
            "hold_ret": round(hold_ret, 4),
            "bench_ret": round(bench_ret, 4) if np.isfinite(bench_ret) else np.nan,
            "excess_ret": round(excess_ret, 4) if np.isfinite(excess_ret) else np.nan,
            "is_bullish": bullish,
            "is_model": is_model,
            "model_outcome": model_main[0] if bullish else "",
            "model_ret": round(model_main[1], 4) if bullish else np.nan,
            "model_exit_day": model_main[2] if bullish else np.nan,
            "model_ret_alt": round(model_alt[1], 4) if bullish and model_alt else np.nan,
            "funda_score": funda_score,
            "is_model_funda": bool(is_model and funda_pass),
        })

        if is_model:
            print(f"[{attempt}/{cfg.attempts}] {ticker:<5} {date_str}  conf {conf:.2f}  "
                  f"{model_main[0]:<9} {model_main[1]:+7.2f}%  (day {model_main[2]})")
        elif attempt % 100 == 0:
            print(f"[{attempt}/{cfg.attempts}] ... {len(samples)} samples so far")

    return pd.DataFrame(samples)


# =============================================================================
# STATISTICS
# =============================================================================
def outcome_stats(outcomes, rets):
    c = Counter(outcomes)
    n = len(outcomes)
    w, l = c["Win"], c["Loss"]
    return {
        "n": n, "W": w, "L": l, "BE": c["Breakeven"], "Exp": c["Expired"],
        "resolved_wr": w / (w + l) * 100 if (w + l) else float("nan"),
        "avg": float(np.mean(rets)) if n else float("nan"),
        "sum": float(np.sum(rets)) if n else 0.0,
    }


def _cluster_sums(df, mask, col):
    """Per-cluster sum and count of df[col] over masked rows. Clusters = ticker-months."""
    codes, uniques = pd.factorize(df["cluster"])
    vals = df[col].to_numpy(dtype=float)
    use = np.asarray(mask, dtype=bool) & np.isfinite(vals)
    sums = np.bincount(codes, weights=np.where(use, vals, 0.0), minlength=len(uniques))
    counts = np.bincount(codes, weights=use.astype(float), minlength=len(uniques))
    return sums, counts


def _resampled_means(sums, counts, idx):
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def _range95(values):
    values = values[np.isfinite(values)]
    return tuple(np.percentile(values, [2.5, 97.5])) if len(values) else (float("nan"), float("nan"))


def bootstrap_index(df, rounds=BOOTSTRAP_ROUNDS, seed=0):
    n_clusters = df["cluster"].nunique()
    return np.random.default_rng(seed).integers(0, n_clusters, size=(rounds, n_clusters))


def mean_range(df, mask, col, idx):
    """95% range for the average of df[col] over masked rows, resampling whole ticker-months."""
    return _range95(_resampled_means(*_cluster_sums(df, mask, col), idx))


def gap_range(df, mask, col, base_col, idx):
    """Model-row average minus random-entry average, with a 95% range from the same resamples."""
    everyone = np.ones(len(df), dtype=bool)
    gaps = (_resampled_means(*_cluster_sums(df, mask, col), idx)
            - _resampled_means(*_cluster_sums(df, everyone, base_col), idx))
    point = df.loc[np.asarray(mask, dtype=bool), col].mean() - df[base_col].mean()
    return (point, *_range95(gaps))


def plain_auc(score, pos, neg):
    """Chance that a random 'pos' sample scores above a random 'neg' one (0.5 = coin flip)."""
    m = pos | neg
    y, s = pos[m], score[m]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())) if ok.sum() > 2 else float("nan")


class RankBoot:
    """Ranges for the ranking check: resample whole ticker-months, 100 resamples at a time."""

    def __init__(self, df, rounds=RANK_BOOT_ROUNDS, seed=1, chunk=100):
        self.codes, uniq = pd.factorize(df["cluster"])
        self.n_cl, self.rounds, self.seed, self.chunk = len(uniq), rounds, seed, chunk

    def _weights(self):
        rng = np.random.default_rng(self.seed)
        p = np.full(self.n_cl, 1.0 / self.n_cl)
        for done in range(0, self.rounds, self.chunk):
            k = min(self.chunk, self.rounds - done)
            yield rng.multinomial(self.n_cl, p, size=k)[:, self.codes].astype(np.float32)

    def auc(self, score, pos, neg):
        order = np.argsort(score, kind="mergesort")
        pos_o, neg_o = pos[order], neg[order]
        vals = []
        for W in self._weights():
            Wo = W[:, order]
            wp, wn = Wo * pos_o, Wo * neg_o
            below = np.cumsum(wn, axis=1) - wn
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append((wp * below).sum(axis=1) / (wp.sum(axis=1) * wn.sum(axis=1)))
        return _range95(np.concatenate(vals))

    def corr(self, x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        rx = pd.Series(x[ok]).rank().to_numpy()
        ry = pd.Series(y[ok]).rank().to_numpy()
        vals = []
        for W in self._weights():
            W = W[:, ok]
            sw = W.sum(axis=1)
            mx, my = (W @ rx) / sw, (W @ ry) / sw
            dx, dy = rx - mx[:, None], ry - my[:, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append((W * dx * dy).sum(axis=1) / np.sqrt((W * dx * dx).sum(axis=1) * (W * dy * dy).sum(axis=1)))
        return _range95(np.concatenate(vals))

    def spread(self, conf, val, q=0.2):
        lo_c, hi_c = np.nanquantile(conf, [q, 1 - q])
        ok = np.isfinite(val)
        top, bot = (conf >= hi_c) & ok, (conf <= lo_c) & ok
        v = np.where(ok, val, 0.0)
        vals = []
        for W in self._weights():
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append((W[:, top] @ v[top]) / W[:, top].sum(axis=1)
                            - (W[:, bot] @ v[bot]) / W[:, bot].sum(axis=1))
        point = val[top].mean() - val[bot].mean()
        return point, _range95(np.concatenate(vals))


def add_timing_columns(df):
    """
    For each model trade, the average random-entry result of the OTHER samples drawn within
    +/- SAME_WEEK_DAYS calendar days (any ticker). That splits the model's lead into timing
    (it fires in weeks when most stocks rose) and stock picking (its stock beat the others that week).
    """
    week = np.full(len(df), np.nan)
    if len(df):
        dates = pd.to_datetime(df["signal_date"]).to_numpy()
        rand = df["rand_ret"].to_numpy(dtype=float)
        window = np.timedelta64(SAME_WEEK_DAYS, "D")
        for i in np.flatnonzero(df["is_model"].to_numpy(dtype=bool)):
            near = np.abs(dates - dates[i]) <= window
            near[i] = False
            if near.any():
                week[i] = rand[near].mean()
    df["week_random_avg"] = np.round(week, 4)
    df["picking_edge"] = np.round(df["model_ret"].to_numpy(dtype=float) - week, 4) if len(df) else week
    return df


# =============================================================================
# REPORT
# =============================================================================
class Report:
    def __init__(self):
        self.lines = []

    def line(self, text=""):
        print(text)
        self.lines.append(text)

    def rule(self, ch="-"):
        self.line(ch * 100)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def model_row_name(cfg):
    if cfg.min_confidence > 0:
        return f"{cfg.model_label} conf >= {cfg.min_confidence:.2f}"
    return f"{cfg.model_label} Bullish calls"


def fmt(x, spec="+.2f", suffix=""):
    return "   n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}{suffix}"


def summarize(df, cfg, guard, errors, skips, tickers, first_day, last_day, csv_path):
    R = Report()
    alt_label = "breakeven OFF" if cfg.use_breakeven else "breakeven ON"
    R.rule("=")
    R.line(f"PILLAR 2 BACKTEST | model {cfg.model_label} | universe {cfg.universe} | seed {cfg.seed} "
           f"| {cfg.attempts} attempts")
    R.line(f"Signals {first_day} -> {last_day} (END_DATE {cfg.end_date}) | hold up to {HOLD_BARS} trading days")
    R.line(f"Exits: +{TARGET_PCT:.0%} / -{STOP_PCT:.0%} from the next day's open | breakeven stop "
           f"{'ON' if cfg.use_breakeven else 'OFF'} | levels {cfg.levels}")
    R.line(f"Pillar 2: {cfg.pillar2_module}" + (f" | model {cfg.pillar2_model}" if cfg.pillar2_model else "")
           + f" | benchmark {cfg.benchmark}")
    if guard.status == "IN-SAMPLE":
        R.line("!! RESULTS ARE IN-SAMPLE: the model trained on these same months (see section 1).")
    R.rule("=")

    R.line("1) TRAINING OVERLAP CHECK")
    for text in guard.lines:
        R.line("   " + text)
    if guard.keys and len(df):
        exact = int(df["in_training_set"].sum())
        R.line(f"   Backtest samples that are exact training rows (same ticker, same day): "
               f"{exact} of {len(df)} ({exact / len(df):.1%})")
        R.line(f"   Universe tickers the model trained on: {len(set(tickers) & guard.tickers)} of {len(set(tickers))}")
    R.line()

    if df.empty:
        R.line("No usable samples. See the errors below.")
        report_errors(R, errors, skips)
        return R

    idx = bootstrap_index(df)
    rows = [("Random entry (no model)", np.ones(len(df), dtype=bool), "rand"),
            ("All bullish calls", df["is_bullish"].to_numpy(dtype=bool), "model"),
            (model_row_name(cfg), df["is_model"].to_numpy(dtype=bool), "model")]
    if HAVE_FUNDAMENTALS:
        rows.append((f"{cfg.model_label} + fundamentals*", df["is_model_funda"].to_numpy(dtype=bool), "model"))

    R.line("2) SAME SAMPLES, SAME EXITS - EACH ROW ONLY DIFFERS IN WHICH TRADES IT TAKES")
    R.line(f"   {'Row':<28}{'Trades':>7}{'W':>5}{'L':>5}{'BE':>5}{'Exp':>5}{'Resolved WR':>13}"
           f"{'Avg %':>9}   {'95% range of avg':<19}{'Sum %':>9}")
    for name, mask, prefix in rows:
        sub = df[mask]
        st = outcome_stats(sub[f"{prefix}_outcome"].tolist(), sub[f"{prefix}_ret"].to_numpy(dtype=float))
        lo, hi = mean_range(df, mask, f"{prefix}_ret", idx) if st["n"] else (float("nan"),) * 2
        R.line(f"   {name:<28}{st['n']:>7}{st['W']:>5}{st['L']:>5}{st['BE']:>5}{st['Exp']:>5}"
               f"{fmt(st['resolved_wr'], '.1f', '%'):>13}{fmt(st['avg']):>9}   "
               f"{fmt(lo) + ' to ' + fmt(hi):<19}{st['sum']:>+9.1f}")
    if HAVE_FUNDAMENTALS:
        R.line("   * the fundamentals helper uses TODAY's data, which is look-ahead for past trades.")
    else:
        R.line("   (fundamentals row skipped: class_ai_pillar1_funda_helper could not be imported)")
    bull = df["is_bullish"].to_numpy(dtype=bool)
    if bull.any() and (bull == df["is_model"].to_numpy(dtype=bool)).all():
        R.line(f"   Note: 'All bullish calls' equals the model row: pillar 2 only calls Bullish at confidence "
               f">= {df.loc[bull, 'confidence'].min():.2f} (its own cutoff).")
        R.line("   Section 5b shows how every score level did, including the ones below that cutoff.")
    R.line()

    R.line("3) WHAT THE MODEL ADDS  (row average minus random-entry average, in percentage points)")
    for name, mask, prefix in rows[1:]:
        if not mask.any():
            R.line(f"   {name:<28} no trades")
            continue
        gap, lo, hi = gap_range(df, mask, f"{prefix}_ret", "rand_ret", idx)
        if np.isfinite(lo) and lo > 0:
            verdict = "above zero: better than random in this window"
        elif np.isfinite(hi) and hi < 0:
            verdict = "below zero: worse than random in this window"
        else:
            verdict = "includes zero: can't tell apart from random"
        R.line(f"   {name:<28}{fmt(gap):>8} pts   95% range {fmt(lo)} to {fmt(hi)}   ({verdict})")
    mm = df["is_model"].to_numpy(dtype=bool)
    if mm.any() and df["excess_ret"].notna().any():
        g, lo, hi = gap_range(df, mm, "excess_ret", "excess_ret", idx)
        R.line(f"   Excess return vs {cfg.benchmark}, buy and hold {HOLD_BARS} days (exits ignored; the V41 target):")
        R.line(f"   random {fmt(df['excess_ret'].mean())}% | {cfg.model_label} {fmt(df.loc[mm, 'excess_ret'].mean())}% "
               f"| gap {fmt(g)} pts, 95% range {fmt(lo)} to {fmt(hi)}")
    R.line("   Ranges resample whole ticker-months, because nearby days on one stock are near-copies.")
    R.line()

    R.line(f"3b) TIMING VS STOCK PICKING  ({model_row_name(cfg)} vs random entries "
           f"within +/-{SAME_WEEK_DAYS} days of each signal)")
    usable = df["is_model"].to_numpy(dtype=bool) & np.isfinite(df["week_random_avg"].to_numpy(dtype=float))
    if not usable.any():
        R.line("   no model trades to split")
    else:
        all_rand = df["rand_ret"].mean()
        total = df.loc[usable, "model_ret"].mean() - all_rand
        timing = df.loc[usable, "week_random_avg"].mean() - all_rand
        picking = df.loc[usable, "picking_edge"].mean()
        lo, hi = mean_range(df, usable, "picking_edge", idx)
        R.line(f"   Lead over random entry, total:       {fmt(total):>7} pts   ({int(usable.sum())} trades)")
        R.line(f"     from timing (which weeks):         {fmt(timing):>7} pts   random entries in those weeks vs all weeks")
        R.line(f"     from stock picking (which stocks): {fmt(picking):>7} pts   95% range {fmt(lo)} to {fmt(hi)}")
        R.line("   Timing: it fires in weeks when most stocks rose anyway (it reads the market's direction).")
        R.line("   Picking: its stock beat the other stocks sampled that same week (it chooses well).")
    R.line()

    R.line("3c) CONCENTRATION  (does the lead survive without the model's best few stock-months?)")
    mt = df[df["is_model"].to_numpy(dtype=bool)]
    if len(mt) < 5:
        R.line("   too few model trades")
    else:
        base = df["rand_ret"].mean()
        by_cluster = mt.groupby("cluster")["model_ret"].sum().sort_values(ascending=False)
        R.line(f"   All {len(mt)} trades ({len(by_cluster)} stock-months):{'':<18}lead {fmt(mt['model_ret'].mean() - base)} pts")
        for k in (1, 3, 5, 10):
            if k >= len(by_cluster):
                break
            keep = mt[~mt["cluster"].isin(by_cluster.index[:k])]
            R.line(f"   without the best {k:>2} stock-month{'s' if k > 1 else ' '} ({len(mt) - len(keep):>3} trades removed):"
                   f"  lead {fmt(keep['model_ret'].mean() - base)} pts")
        by_ticker = mt.groupby("ticker")["model_ret"].sum().sort_values(ascending=False)
        total = mt["model_ret"].sum()
        share = by_ticker.head(5).sum() / total * 100 if total > 0 else float("nan")
        R.line(f"   Best 5 tickers: {', '.join(by_ticker.head(5).index)} ({fmt(share, '.0f', '%')} of the summed return)")
        R.line("   A lead that vanishes after removing a few stock-months rests on a handful of setups.")
    R.line()

    R.line("4) BY SCOPE GROUP  (model row vs random entry inside the same group)")
    R.line(f"   {'Group':<22}{'Model trades':>13}{'Model avg':>11}{'Samples':>9}{'Random avg':>12}{'Gap':>9}")
    model_mask = df["is_model"].to_numpy(dtype=bool)
    for g in ("software_internet", "chips_hardware", "tech (both)", "grey_zone", "out_of_scope", "other"):
        gmask = (df["group"].isin(TECH_GROUPS) if g == "tech (both)" else df["group"] == g).to_numpy()
        if not gmask.any():
            continue
        m_avg = df.loc[gmask & model_mask, "model_ret"].mean() if (gmask & model_mask).any() else float("nan")
        r_avg = df.loc[gmask, "rand_ret"].mean()
        R.line(f"   {g:<22}{int((gmask & model_mask).sum()):>13}{fmt(m_avg):>11}{int(gmask.sum()):>9}"
               f"{fmt(r_avg):>12}{fmt(m_avg - r_avg):>9}")
    R.line()

    R.line("5) BULLISH CALLS BY CONFIDENCE  (a real signal should improve as confidence rises)")
    R.line(f"   {'Confidence':<14}{'Trades':>7}{'Resolved WR':>13}{'Avg %':>9}")
    edges = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    bull = df[df["is_bullish"].to_numpy(dtype=bool)]
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        sub = bull[(bull["confidence"] >= lo_e) & (bull["confidence"] < hi_e)]
        if sub.empty:
            continue
        label = f"< {hi_e:.2f}" if lo_e == 0 else (f">= {lo_e:.2f}" if hi_e > 1 else f"{lo_e:.2f}-{hi_e:.2f}")
        st = outcome_stats(sub["model_outcome"].tolist(), sub["model_ret"].to_numpy(dtype=float))
        R.line(f"   {label:<14}{st['n']:>7}{fmt(st['resolved_wr'], '.1f', '%'):>13}{fmt(st['avg']):>9}")
    R.line(f"   Random entry, all samples: {fmt(df['rand_ret'].mean())}% per trade (the bar to beat)")
    R.line()

    report_ranking(R, df, cfg)

    R.line(f"6) BREAKEVEN RULE CHECK  (average % per trade with the rule as run vs {alt_label})")
    for name, mask, prefix in rows:
        sub = df[mask]
        if sub.empty:
            continue
        R.line(f"   {name:<28}{fmt(sub[f'{prefix}_ret'].mean()):>9}  vs {fmt(sub[f'{prefix}_ret_alt'].mean())}")
    R.line()

    report_errors(R, errors, skips)
    R.line(f"Per-sample log: {csv_path}")
    R.rule("=")
    return R


def report_ranking(R, df, cfg):
    """Section 5b: does a higher score mean a better outcome, across ALL samples?"""
    R.line("5b) RANKING ACROSS ALL SAMPLES  (does a higher score mean a better outcome? AUC 0.50 = coin flip)")
    conf = df["confidence"].to_numpy(dtype=float)
    if len(df) < 200 or not np.nanstd(conf) > 0:
        R.line("   needs at least 200 samples with a spread of scores")
        R.line()
        return
    q = 10 if len(df) >= 1000 else 5
    grp = pd.qcut(df["confidence"], q, labels=False, duplicates="drop").to_numpy()
    excess = df["excess_ret"].to_numpy(dtype=float)
    lab = df["label_outcome"].to_numpy()
    R.line(f"   {'Group':<7}{'Scores':<13}{'Samples':>8}{'Hit +20% first':>16}{'Beat ' + cfg.benchmark:>10}"
           f"{'Avg excess':>12}{'Avg trade':>11}")
    for g in np.unique(grp[~np.isnan(grp)]):
        m = grp == g
        w, l = int((lab[m] == "Win").sum()), int((lab[m] == "Loss").sum())
        hit = w / (w + l) * 100 if (w + l) else float("nan")
        ex = excess[m][np.isfinite(excess[m])]
        beat = (ex > 0).mean() * 100 if len(ex) else float("nan")
        R.line(f"   {int(g) + 1:<7}{conf[m].min():.2f}-{conf[m].max():.2f}{'':<4}{int(m.sum()):>8}"
               f"{fmt(hit, '.1f', '%'):>16}{fmt(beat, '.1f', '%'):>10}{fmt(ex.mean() if len(ex) else float('nan')):>11}%"
               f"{fmt(df['rand_ret'].to_numpy(dtype=float)[m].mean()):>10}%")

    boot = RankBoot(df)
    win, loss = lab == "Win", lab == "Loss"
    ok = np.isfinite(excess)
    beat_pos, beat_neg = ok & (excess > 0), ok & (excess <= 0)
    week = pd.to_datetime(df["signal_date"]).dt.strftime("%G-W%V")
    ex_vs_week = (df["excess_ret"] - df.groupby(week)["excess_ret"].transform("mean")).to_numpy(dtype=float)

    def show(label, point, rng):
        R.line(f"   {label:<48}{fmt(point, '+.3f'):>7}   95% range {fmt(rng[0], '+.3f')} to {fmt(rng[1], '+.3f')}")

    rng = boot.auc(conf, win, loss)
    R.line(f"   {'AUC, hit +20% before -12% (training-label rules)':<48}{plain_auc(conf, win, loss):>7.3f}   "
           f"95% range {rng[0]:.3f} to {rng[1]:.3f}")
    if ok.any():
        rng = boot.auc(conf, beat_pos, beat_neg)
        R.line(f"   {'AUC, beat ' + cfg.benchmark + ' over the holding window':<48}"
               f"{plain_auc(conf, beat_pos, beat_neg):>7.3f}   95% range {rng[0]:.3f} to {rng[1]:.3f}")
        show("Rank correlation, score vs excess return", rank_corr(conf, excess), boot.corr(conf, excess))
        show("Same, excess vs its own week (stock picking)", rank_corr(conf, ex_vs_week), boot.corr(conf, ex_vs_week))
        point, rng = boot.spread(conf, excess)
        R.line(f"   {'Top 20% minus bottom 20% of scores, avg excess':<48}{fmt(point):>7} pts  "
               f"95% range {fmt(rng[0])} to {fmt(rng[1])}")
    R.line("   A useful score has its AUC range above 0.50 for the target it was trained on")
    R.line(f"   (V40: hit +20% before -12%; V41: beat {cfg.benchmark}).")
    R.line()


def report_errors(R, errors, skips):
    R.line("7) ERRORS AND SKIPS  (counted, not hidden)")
    if not errors.counts and not skips:
        R.line("   none")
    for stage, n in errors.counts.most_common():
        R.line(f"   ERROR {stage}: {n}")
        for ex in errors.examples.get(stage, []):
            R.line(f"         e.g. {ex}")
    for what, n in skips.most_common():
        R.line(f"   skip  {what}: {n}")
    R.line()


# =============================================================================
# ENTRY POINT
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pillar 2 backtest with baselines and a training-overlap check")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--attempts", type=int, default=SEARCH_ATTEMPTS)
    p.add_argument("--universe", choices=["SHAY", "TECH_ONLY", "SP500"], default=UNIVERSE)
    p.add_argument("--levels", choices=["FIXED", "PILLAR2"], default=LEVELS_MODE)
    p.add_argument("--no-breakeven", action="store_true", help="turn the breakeven stop off")
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    p.add_argument("--model-label", default=MODEL_LABEL)
    p.add_argument("--training-dataset", default=TRAINING_DATASET)
    p.add_argument("--end-date", default=END_DATE.isoformat(), help="fixed 'today' as YYYY-MM-DD")
    p.add_argument("--pillar2-module", default=PILLAR2_MODULE)
    p.add_argument("--pillar2-model", default=PILLAR2_MODEL)
    p.add_argument("--benchmark", default=BENCHMARK)
    a = p.parse_args()
    return SimpleNamespace(seed=a.seed, attempts=a.attempts, universe=a.universe, levels=a.levels,
                           use_breakeven=not a.no_breakeven, min_confidence=a.min_confidence,
                           model_label=a.model_label, training_dataset=a.training_dataset,
                           end_date=datetime.date.fromisoformat(a.end_date),
                           pillar2_module=a.pillar2_module, pillar2_model=a.pillar2_model,
                           benchmark=a.benchmark)


def main():
    cfg = parse_args()
    grouped = sorted(TICKER_TO_GROUP)
    if grouped != sorted(SHAY_TICKERS):
        print("Note: TICKER_GROUPS and SHAY_TICKERS don't list the same tickers; "
              "unlisted tickers show up as 'other'.")

    first_day = cfg.end_date - datetime.timedelta(days=MAX_DAYS_AGO)
    last_day = cfg.end_date - datetime.timedelta(days=MIN_DAYS_AGO)
    errors, skips = ErrorLog(), Counter()
    try:
        tickers = load_universe(cfg.universe)
    except Exception as e:
        sys.exit(f"Could not load the {cfg.universe} ticker list ({type(e).__name__}: {e}). Stopping "
                 f"rather than silently switching to another universe.")
    guard = TrainingGuard(cfg.training_dataset, first_day, last_day)
    for text in guard.lines:
        print(text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = (f"{cfg.model_label}_{cfg.universe}_seed{cfg.seed}_{cfg.end_date:%Y%m%d}_{cfg.levels}"
              f"{'' if cfg.use_breakeven else '_nobe'}")
    csv_path = os.path.join(OUTPUT_DIR, f"{run_id}_samples.csv")

    try:
        p2 = importlib.import_module(cfg.pillar2_module)
    except Exception as e:
        sys.exit(f"Could not import pillar 2 module '{cfg.pillar2_module}' ({type(e).__name__}: {e}).")
    if cfg.pillar2_model:
        if not hasattr(p2, "load_model"):
            sys.exit(f"{cfg.pillar2_module} has no load_model(), so --pillar2-model can't be used with it.")
        p2.load_model(cfg.pillar2_model)

    df = run(cfg, errors, skips, guard, tickers, first_day, last_day, p2.run_pillar2_technical_quant)
    df = add_timing_columns(df)
    df.to_csv(csv_path, index=False)
    report = summarize(df, cfg, guard, errors, skips, tickers, first_day, last_day, csv_path)
    report.save(os.path.join(OUTPUT_DIR, f"{run_id}_summary.txt"))


if __name__ == "__main__":
    main()
