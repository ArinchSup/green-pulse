"""
V103 - ENTRY RECOMMENDER, THREE HORIZONS.

    ticker + date + horizon  ->  ENTER or WAIT, with entry, stop and target

V101's recommender with the LONG horizon added. It loads the final entry
model for the requested horizon and answers one question:

    "I have already decided to buy THIS stock around THIS date.
     Is this a good day to enter, or should I wait?"

It never says which stock to buy (V89, V100, V102: the signals time entries
into stocks chosen elsewhere).

HORIZONS (all confirmed on unseen stocks, pre-registered)
---------------------------------------------------------
  SHORT  V101 - a 10-trading-day trade (most end within about 4 days).
         Target 0.5 x ATR% x sqrt(10) within 3.27%-16.33%. Timing +11.7 pp
         over other days of the same stock and month (V100); a simple
         low-RSI-14 rule does as well.
  MID    V88 - a 60-trading-day trade. Target 0.5 x ATR% x sqrt(60) within
         8%-40% (trade_config). Beats the rule fixed in advance (V86/V87).
  LONG   V103 - a 250-trading-day (one-year) trade, about 105 days on
         average. Target 0.5 x ATR% x sqrt(250) within 16.33%-81.65%.
         Timing +9.5 pp (V102, 12-month blocks); beats the rule fixed in
         advance (low 5-day return) by +2.4 pp.

The stop is always 0.6 x the target (break-even 37.5% of trades that end at
the target or the stop). SHORT and LONG carry their trade geometry in the
model file - trade_config is not changed.

HOW A RECOMMENDATION IS MADE (each horizon separately)
------------------------------------------------------
  1. The stock's features are built for its last bar on or before the date,
     with the exact code the model was trained on (V88/V101/V103 parity).
  2. The model scores that bar.
  3. The score is compared with that month's cut: the 99th percentile of the
     scores the same model gave the reference universe over the trailing 252
     trading days before the month began. ENTER only if STRICTLY above it.
  4. The stop and target come from the model's own trade geometry. Entry is
     assumed at that bar's close.

Reference-universe scores take a few minutes per horizon the first time and
are cached in entry_cache_v90/ (shared with V90/V101 - the keys include the
model file, so nothing is mixed up).

USE FROM OTHER CODE
-------------------
    from entry_recommender_v103 import EntryRecommender
    long = EntryRecommender("LONG")
    r = long.recommend("AMD")                  # latest bar
    r = long.recommend("AMD", "2026-09-25")    # a specific date
    h = long.history("AMD", "2026-01-01")      # every day in a range
    r["signal"], r["entry"], r["stop"], r["target"], r["holding_bars"]

WARNINGS IT GIVES
-----------------
  stale prices   the latest bar is well before the requested date
  in-sample      the date is inside the model's training period
  stale universe the reference universe ends well before the month began
  plateau        an unusual share of the window sits at or above the cut
"""

import os
import importlib
import warnings

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import measure_v82 as V82
import measure_v83 as V83
from trade_config import HORIZON_CONFIGS, compute_levels

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="divide by zero")

MODEL_DIR = "class_model"           # the three model files live here
MODEL_REGISTRY = {
    "MID": {"file": os.path.join(MODEL_DIR, "entry_model_v88.joblib"),
            "module": "class_ai_entry_model_v88",
            "note": "V88, 60-day trade: confirmed on unseen stocks (V86/V87)"},
    "SHORT": {"file": os.path.join(MODEL_DIR, "entry_model_v101.joblib"),
              "module": "class_ai_entry_model_v101",
              "note": "V101, 10-day trade: timing confirmed on unseen stocks "
                      "(V100)"},
    "LONG": {"file": os.path.join(MODEL_DIR, "entry_model_v103.joblib"),
             "module": "class_ai_entry_model_v103",
             "note": "V103, one-year trade: timing confirmed on unseen stocks "
                     "(V102); beats the rule fixed in advance"},
}
CACHE_DIR = "entry_cache_v90"
STALE_DAYS = 7          # price data older than this vs the request -> warning
PLATEAU_WARN = 0.03


def _cache_names(path):
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(path) if f.endswith(".pkl"))


def _dir_stamp(path):
    files = [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith(".pkl")]
    return len(files), max((os.path.getmtime(f) for f in files), default=0)


def _day(ts):
    return pd.Timestamp(ts).normalize()


def _hash(obj):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


# =============================================================================
# OPTIONAL: refresh prices for the requested tickers (as V90)
# =============================================================================
def refresh_prices(tickers, live_cache="price_cache_live", start="2005-01-01",
                   verbose=True):
    """
    Downloads full daily history for these tickers into a SEPARATE cache, so
    the caches the models were built on are never overwritten. Needs yfinance
    and internet. The reference universe does not need daily refreshing: a
    month's cut only uses data from before that month began.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  refresh skipped: yfinance is not installed (pip install "
              "yfinance)")
        return []
    try:
        from build_universe_v68 import check_frame
    except Exception:
        check_frame = None
    os.makedirs(live_cache, exist_ok=True)
    try:
        raw = yf.download(list(tickers), start=start, interval="1d",
                          auto_adjust=True, group_by="ticker",
                          progress=False, threads=True)
    except Exception as e:
        print(f"  refresh failed ({e}); using cached prices")
        return []
    done = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                lv0 = raw.columns.get_level_values(0)
                df = raw[t] if t in lv0 else raw.xs(t, axis=1, level=1)
            else:
                df = raw
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        except Exception as e:
            print(f"  refresh: {t} could not be parsed ({e})")
            continue
        if check_frame is not None:
            ok, why = check_frame(df, t)
            if not ok:
                print(f"  refresh: {t} refused by the quality gates ({why})")
                continue
        df.to_pickle(os.path.join(live_cache, f"{t}.pkl"))
        done.append(t)
        if verbose:
            print(f"  refreshed {t}: {len(df):,} bars to "
                  f"{df.index.max().date()}")
    return done


# =============================================================================
# THE RECOMMENDER
# =============================================================================
class EntryRecommender:

    def __init__(self, horizon="MID", model_file=None, universe_cache=None,
                 price_caches=None, live_cache="price_cache_live",
                 cache_dir=CACHE_DIR, verbose=True):
        h = str(horizon).upper()
        if h not in MODEL_REGISTRY:
            raise ValueError(f"unknown horizon {horizon!r}; choose from "
                             f"{sorted(MODEL_REGISTRY)}")
        reg = MODEL_REGISTRY[h]
        if reg is None:
            have = [k for k, v in MODEL_REGISTRY.items() if v]
            raise NotImplementedError(
                f"no entry model is trained for the {h} horizon yet. "
                f"Available: {', '.join(have)}.")
        self.horizon, self.verbose, self.note = h, verbose, reg["note"]
        self.mod = importlib.import_module(reg["module"])
        self.model_file = model_file or reg["file"]
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"{self.model_file} not found - train it "
                                    f"first ({reg['module']}.py)")
        self.model = self.mod.load_model(self.model_file)
        p = self.model["provenance"]
        if p.get("horizon", h) != h:
            raise ValueError(f"{self.model_file} was trained for "
                             f"{p['horizon']}, not {h}")
        self.prov, self.rule = p, self.model["rule"]
        geo = self.model.get("geometry") or {}
        self.hold = int(geo.get("hold")
                        or HORIZON_CONFIGS[h]["lookahead_bars"])
        self.step = p.get("step", E64.STEP)
        self.trained_through = _day(p["trained_through"])
        self.universe_cache = universe_cache or p["price_cache"]
        caches = [live_cache] + list(price_caches or []) + \
            [self.universe_cache]
        self.price_caches = [c for i, c in enumerate(caches)
                             if c and os.path.isdir(c) and c not in caches[:i]]
        self.cache_dir = cache_dir
        self._U = None
        self._cuts = {}

    # ---- reference universe ---------------------------------------------------
    def _universe_key(self):
        n, mt = _dir_stamp(self.universe_cache)
        return _hash([os.path.abspath(self.model_file),
                      os.path.getmtime(self.model_file),
                      os.path.abspath(self.universe_cache), n, mt,
                      self.step, self.horizon])

    def universe_scores(self, refresh=False):
        """Model scores for every candidate row of the reference universe,
        sorted by date. Cached on disk."""
        if self._U is not None and not refresh:
            return self._U
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir,
                            f"universe_{self.horizon}_{self._universe_key()}"
                            f".pkl")
        if os.path.exists(path) and not refresh:
            self._U = pd.read_pickle(path)
            return self._U
        names = _cache_names(self.universe_cache)
        if len(names) < 20:
            raise RuntimeError(f"{self.universe_cache} holds {len(names)} "
                               f"stocks - the cut needs a reference universe")
        if self.verbose:
            print(f"  [{self.horizon}] scoring the reference universe "
                  f"({len(names)} stocks in {self.universe_cache}) - once, "
                  f"then cached ...")
        F = self.mod.serving_frame(names, self.universe_cache, self.horizon,
                                   self.step)
        p = self.mod.predict(self.model["boosters"], F)
        d = pd.DatetimeIndex(F["date"]).values
        o = np.argsort(d, kind="mergesort")
        self._U = {"dates": d[o], "p": np.asarray(p, float)[o],
                   "last": _day(F["date"].max()), "n_names": len(names)}
        pd.to_pickle(self._U, path)
        return self._U

    def cut_for(self, when):
        """The cut that applies to the month containing `when`."""
        m = pd.Period(_day(when), freq="M").to_timestamp()
        if m in self._cuts:
            return self._cuts[m]
        U = self.universe_scores()
        lo = np.searchsorted(U["dates"], (m - V82._win()).to_datetime64(),
                             "left")
        hi = np.searchsorted(U["dates"], m.to_datetime64(), "left")
        w = U["p"][lo:hi]
        w = np.sort(w[np.isfinite(w)])
        if w.size < self.rule["min_window_rows"]:
            out = {"cut": np.nan, "window": w, "share_at_or_above": np.nan}
        else:
            c = float(np.quantile(w, 1.0 - self.rule["quantile"]))
            out = {"cut": c, "window": w,
                   "share_at_or_above": float((w >= c).mean())}
        out["month"] = m
        out["universe_gap_days"] = int((m - U["last"]).days)
        self._cuts[m] = out
        return out

    # ---- one stock ---------------------------------------------------------------
    def _find(self, ticker):
        for c in self.price_caches:
            if os.path.exists(os.path.join(c, f"{ticker}.pkl")):
                return c
        raise FileNotFoundError(
            f"no price file for {ticker} in {', '.join(self.price_caches)}. "
            f"Set RUN_REFRESH_PRICES = True to download it, or add its cache "
            f"to RUN_PRICE_CACHES.")

    def _frame(self, ticker, as_of, step):
        cache = self._find(ticker)
        end = None if as_of is None else \
            _day(as_of) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        F = self.mod.serving_frame([ticker], cache, self.horizon, step,
                                   as_of=end)
        return F, cache

    def _levels(self, ticker, cache, bar_date):
        df = E64.P2.load_prices(ticker, cache)
        df = df[pd.DatetimeIndex(df.index).normalize() <= _day(bar_date)]
        pan = E64.feature_panel(df)
        atr = float(pan["_atr_abs"].iloc[-1])
        entry = df["Close"].to_numpy(float)[-1]     # numpy, as in training
        if hasattr(self.mod, "levels"):            # the model's own geometry
            lv = self.mod.levels(entry, atr, self.model)
        else:                                      # MID: trade_config's
            lv = compute_levels(entry, atr, self.horizon)
        return float(entry), lv

    def recommend(self, ticker, date=None):
        warn = []
        ticker = str(ticker).upper().strip()
        req = _day(date) if date is not None else None
        out = {"ticker": ticker, "horizon": self.horizon,
               "requested": req.date() if req is not None else "latest"}
        try:
            F, cache = self._frame(ticker, req, self.step)
        except FileNotFoundError as e:
            return {**out, "signal": "NO DATA", "warnings": [str(e)]}
        df_last = _day(E64.P2.load_prices(ticker, cache).index.max())
        if req is not None and (req - df_last).days > STALE_DAYS:
            warn.append(f"price data for {ticker} ends {df_last.date()}, "
                        f"{(req - df_last).days} days before the requested "
                        f"date - refresh prices")
        row = F[F["is_latest"]] if len(F) else F
        if not len(row):
            return {**out, "signal": "NOT SCORABLE", "warnings": warn + [
                "the bar on that date is not eligible: under 260 bars of "
                "history, missing data, or no valid stop/target"]}
        row = row.iloc[[-1]]
        bar = _day(row["date"].iloc[0])
        score = float(self.mod.predict(self.model["boosters"], row)[0])
        c = self.cut_for(bar)
        cut = c["cut"]
        if not np.isfinite(cut):
            return {**out, "bar_date": bar.date(), "signal": "NO CUT",
                    "score": score, "warnings": warn + [
                        "the reference universe has too little history "
                        "before this month to set a cut"]}
        fires = score > cut if self.rule.get("strict", True) else score >= cut
        w = c["window"]
        top = 1.0 - np.searchsorted(w, score, side="right") / w.size
        entry, lv = self._levels(ticker, cache, bar)
        if bar <= self.trained_through:
            warn.append(f"{bar.date()} is inside the model's training period "
                        f"(through {self.trained_through.date()}): this is what "
                        f"the model says, not an out-of-sample result")
        if c["universe_gap_days"] > STALE_DAYS:
            warn.append(f"the reference universe ends "
                        f"{c['universe_gap_days']} days before this month "
                        f"began - refresh {self.universe_cache}")
        if np.isfinite(c["share_at_or_above"]) and \
                c["share_at_or_above"] > PLATEAU_WARN:
            warn.append(f"plateau: {c['share_at_or_above']:.1%} of the window "
                        f"is at or above the cut - retrain before trusting "
                        f"signals")
        show_p = bool(self.model.get("display_probability"))
        return {**out, "bar_date": bar.date(),
                "signal": "ENTER" if fires else "WAIT",
                "score": score, "cut": cut, "top_pct_of_window": top,
                "score_is_probability": show_p,
                "entry": entry,
                "stop": lv.get("stop"), "target": lv.get("target"),
                "stop_pct": lv.get("stop_pct"),          # percent
                "target_pct": lv.get("target_pct"),      # percent
                "rr": lv.get("rr"), "breakeven_win_rate": lv.get("breakeven_wr"),
                "holding_bars": self.hold,
                "price_cache": cache, "warnings": warn}

    def recommend_many(self, requests):
        """requests: list of tickers, or of (ticker, date) pairs."""
        rows = []
        for r in requests:
            t, d = (r, None) if isinstance(r, str) else (r[0], r[1])
            res = self.recommend(t, d)
            res["warnings"] = " | ".join(res.get("warnings", []))
            rows.append(res)
        return pd.DataFrame(rows)

    def history(self, ticker, start, end=None):
        """The signal on EVERY trading day in [start, end] (not just every
        5th), each against its own month's cut."""
        ticker = str(ticker).upper().strip()
        F, cache = self._frame(ticker, end, 1)
        if not len(F):
            return pd.DataFrame()
        di = pd.DatetimeIndex(F["date"]).normalize()
        F = F[(di >= _day(start)) &
              ((di <= _day(end)) if end is not None else True)].copy()
        if not len(F):
            return pd.DataFrame()
        F["score"] = self.mod.predict(self.model["boosters"], F)
        cuts, tops = [], []
        for d, s in zip(F["date"], F["score"]):
            c = self.cut_for(d)
            cuts.append(c["cut"])
            w = c["window"]
            tops.append(1.0 - np.searchsorted(w, s, side="right") / w.size
                        if w.size else np.nan)
        F["cut"] = cuts
        F["top_pct_of_window"] = tops
        F["signal"] = np.where(V83.apply_rule(
            F["score"].to_numpy(float), F["cut"].to_numpy(float),
            strict=self.rule.get("strict", True)), "ENTER", "WAIT")
        F["date"] = pd.DatetimeIndex(F["date"]).normalize().date
        return F[["date", "entry", "score", "cut", "top_pct_of_window",
                  "signal"]].rename(columns={"entry": "close"}) \
            .reset_index(drop=True)

    # ---- context -------------------------------------------------------------
    def evidence(self):
        """One line on what a signal has meant, from the model's own record."""
        if hasattr(self.mod, "claim_line"):        # the model describes itself
            return self.mod.claim_line(self.model)
        me = self.model.get("measured", {})
        arms = pd.DataFrame(me.get("v86_arms", []))
        if len(arms):
            r = arms[arms["Universe"].astype(str).str.startswith(
                "fresh, liquid") & arms["Arm"].astype(str).str.startswith(
                "combo top-3")]
            if len(r):
                r = r.iloc[0]
                return (f"On {int(r['Names'])} stocks it never saw "
                        f"(2017-2026), ENTER days beat other days in the "
                        f"same stock and month by {r['Lift pp']:+.1f} pp "
                        f"win rate ({r['ExpR']:+.2f}R per trade). It does "
                        f"not say whether to own the stock.")
        return ("ENTER days are top-1% entry days within the stock and month; "
                "the signal does not say whether to own the stock.")


# =============================================================================
# PRINTING
# =============================================================================
def print_card(r):
    print("-" * 96)
    head = f"  {r['ticker']:<8} {r['horizon']} horizon"
    if r.get("bar_date"):
        head += f"   bar {r['bar_date']}"
        if str(r.get("requested")) not in ("latest", str(r["bar_date"])):
            head += f" (requested {r['requested']})"
    print(head)
    sig = r["signal"]
    if sig in ("ENTER", "WAIT"):
        lab = "probability" if r.get("score_is_probability") else "score"
        print(f"  SIGNAL   {sig:<6}  {lab} {r['score']:.4f}  |  cut "
              f"{r['cut']:.4f}  |  ranks in the top "
              f"{r['top_pct_of_window']:.1%} of the trailing year")
        if r.get("stop") is not None:
            print(f"  LEVELS   entry {r['entry']:.2f} (that bar's close)   "
                  f"stop {r['stop']:.2f} ({-r['stop_pct']:+.1f}%)   "
                  f"target {r['target']:.2f} ({r['target_pct']:+.1f}%)   "
                  f"R:R {r['rr']:.2f}   holding up to {r['holding_bars']} "
                  f"trading days")
        if sig == "ENTER":
            print("  MEANING  a top-1% entry day: if you are buying this stock "
                  "for this horizon, this is a good day to do it")
        else:
            print("  MEANING  not a top-1% entry day: if you are buying this "
                  "stock for this horizon, waiting for a signal day has been "
                  "better")
    else:
        print(f"  SIGNAL   {sig}")
    for w in r.get("warnings", []) or []:
        print(f"  WARNING  {w}")


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_HORIZONS       = ["SHORT", "MID", "LONG"]  # any of these
    RUN_MODE           = "recommend"     # "recommend" or "history"
    RUN_TICKERS        = ["AMD", "NVDA", "AAPL"]
    RUN_DATE           = None            # None = latest bar, or "2026-09-25"
    RUN_HISTORY_START  = "2026-01-01"    # history mode: every day in range
    RUN_HISTORY_END    = None            # None = latest bar
    RUN_UNIVERSE_CACHE = None            # None = each model's training cache
    RUN_PRICE_CACHES   = ["price_cache_v68"]  # more places to find tickers
    RUN_REFRESH_PRICES = False           # download latest prices (yfinance)
    RUN_LIVE_CACHE     = "price_cache_live"   # where refreshed prices go
    RUN_SAVE_CSV       = "entry_recommendations_v103.csv"  # or None
    # -------------------------------------------------------------------------

    if RUN_REFRESH_PRICES:
        refresh_prices(RUN_TICKERS, RUN_LIVE_CACHE)

    results, frames = [], []
    for HZ in RUN_HORIZONS:
        try:
            rec = EntryRecommender(HZ, universe_cache=RUN_UNIVERSE_CACHE,
                                   price_caches=RUN_PRICE_CACHES,
                                   live_cache=RUN_LIVE_CACHE)
        except (NotImplementedError, FileNotFoundError, ValueError) as e:
            print(f"\n  {HZ}: skipped - {e}")
            continue
        print("\n" + "=" * 96)
        print(f"  {HZ} HORIZON - model {rec.model_file} (trained through "
              f"{rec.trained_through.date()}; holding up to {rec.hold} "
              f"trading days)")
        print(f"  {rec.note}")
        print(f"  {rec.evidence()}")
        print("=" * 96)

        if RUN_MODE == "history":
            print(f"  days up to {rec.trained_through.date()} are inside the "
                  f"model's training period: they show what the model says, "
                  f"not how it would have performed then")
            for t in RUN_TICKERS:
                h = rec.history(t, RUN_HISTORY_START, RUN_HISTORY_END)
                if not len(h):
                    print(f"  {t}: no scorable days in that range")
                    continue
                h.insert(0, "ticker", t)
                h.insert(1, "horizon", HZ)
                frames.append(h)
                on = h[h["signal"] == "ENTER"]
                print(f"\n  {t}: {len(h)} trading days, {len(on)} ENTER days")
                if len(on):
                    print(on.to_string(index=False, float_format=lambda v:
                                       f"{v:.4f}"))
        else:
            for t in RUN_TICKERS:
                r = rec.recommend(t, RUN_DATE)
                print_card(r)
                results.append(r)
            print("-" * 96)

    if RUN_SAVE_CSV and (results or frames):
        if frames:
            pd.concat(frames).to_csv(RUN_SAVE_CSV, index=False)
        else:
            pd.DataFrame([{**r, "warnings": " | ".join(r.get("warnings", []))}
                          for r in results]).to_csv(RUN_SAVE_CSV, index=False)
        print(f"\n  saved {RUN_SAVE_CSV}")
