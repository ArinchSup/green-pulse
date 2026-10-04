#!/usr/bin/env python3
"""
class_ai_pillar2_v43.py - Pillar 2 V43 inference. A COHORT SCANNER, not a per-ticker call.

WHY THIS CANNOT WORK LIKE class_ai_pillar2.py

V43 scores a stock relative to the other stocks trading that day. Two of its
ingredients simply do not exist for a single ticker in isolation:

  - xs_* features are percentile ranks WITHIN the day's cohort. With one ticker,
    every rank is 0.5 and the model gets no information at all.
  - a ranker's output is a SCORE, not a probability. 0.63 means nothing on its own;
    it only means something next to the other scores from the same day.

So the unit of inference is a DAY, not a ticker. Score the whole universe, rank it,
take the top N. That is also how you would actually use the tool: not "is NVDA good
today?" but "of the 62 names I follow, which 3 look best today?".

TWO INTERFACES

  scan(date)                          -> the ranked table for that day. Use this.
  run_pillar2_technical_quant(ticker, target_date_str=...)
                                      -> the old per-ticker signature, so
                                         class_ai_pipeline_backtest_v2.py works
                                         unchanged. Internally it scans the whole
                                         cohort for that date (cached) and looks the
                                         ticker up. "confidence" is the ticker's
                                         PERCENTILE within that day's cohort, so
                                         section 5b of the backtest measures exactly
                                         what the model was trained to do.

SELECTION
  TOP_N     how many names a day may produce. This replaces CONFIDENCE_THRESHOLD; a
            global cutoff is meaningless for a ranker.
  MIN_PCTL  a floor on cohort percentile, so a bad day does not force out N picks
            just because something has to be ranked first.

USAGE
  python class_ai_pillar2_v43.py                      # today's ranked list
  python class_ai_pillar2_v43.py --date 2024-03-15 --top 5
  python class_ai_pillar2_v43.py --model class_model/xgboost_mid_v43b.joblib
"""
import argparse
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd

import pillar2_v43_features as F
from pillar2_v43_universe import BENCHMARK, DEPLOY_UNIVERSE
from trade_config import compute_levels, describe_geometry

MODEL_PATH = os.environ.get("PILLAR2_V43_MODEL",
                            os.path.join("class_model", "xgboost_mid_v43b.joblib"))
CACHE_DIR = "price_cache_v43"
HORIZON = "MID"
TOP_N = 3            # names per day
MIN_PCTL = 0.80      # must also be in the top 20% of YOUR tradeable names
# Which names the model may PICK FROM. The reference cohort (what xs_* ranks are
# computed against) is always the training universe; this is a separate question.
#   "deploy"   the 62 SHAY names (default)
#   "training" the 347-name training universe - use this to test whether the skill
#              measured in cross-validation exists on the kind of universe the model
#              was actually trained for
SELECT_UNIVERSE = os.environ.get("PILLAR2_V43_SELECT", "deploy").lower()
USE_REFERENCE_COHORT = True   # compute xs_* ranks against the training universe,
                              # not just the deploy names — see scan() for why
MAX_BAR_STALENESS = 10   # calendar days: if a ticker has no bar within this of the
                         # target date it is treated as not trading then (acquired,
                         # delisted, halted) and dropped from that day's cohort
MIN_BARS      = 260  # need ~1 year of bars for the 200-EMA and 250-bar momentum
FETCH_START   = "2004-01-01"  # one wide download per ticker, reused for every date

# Panels are kept in memory for every ticker in the cohort. With ~347 tickers and
# 20 years of history that is about 1 GB in float64, which on most machines means
# swapping - and swapping looks exactly like a freeze. Set PANEL_KEEP_FROM to a date
# shortly before the earliest one you will ever score (an env var or edit here) and
# only that slice is retained; the features are still COMPUTED from full history, so
# the values are identical. 2023-01-01 with a 2023-10 backtest start cuts memory ~6x.
PANEL_KEEP_FROM = os.environ.get("PILLAR2_V43_PANEL_FROM")   # e.g. "2023-01-01"

_state = {}
_cohort_cache = {}


# =============================================================================
# MODEL
# =============================================================================
def load_model(path=None):
    path = path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Train one with class_xgboost_v43b.py, or set "
            f"PILLAR2_V43_MODEL.")
    b = joblib.load(path)
    if b.get("feature_version") != F.FEATURE_VERSION:
        print(f"  WARNING: model built with features {b.get('feature_version')}, "
              f"this code is {F.FEATURE_VERSION}. Retrain, or use the matching "
              f"features file — mismatched columns silently feed wrong values.")
    _state.clear()
    _state.update(bundle=b, path=path)
    _cohort_cache.clear()
    return b


def _bundle():
    if "bundle" not in _state:
        load_model()
    return _state["bundle"]


# =============================================================================
# PRICES
# =============================================================================
class PriceCache:
    """
    One download per ticker PER SESSION, over a wide fixed window.

    The previous version asked for target-700d..target+1 on every scan and re-fetched
    whenever a later date came up. Under the backtest that was catastrophic: the
    backtest patches yf.Ticker.history to serve from its own store, but only when the
    requested range fits INSIDE that store. A 700-day lookback does not, so every
    call fell through to live Yahoo — 62 tickers x ~180 dates instead of 62.
    """

    DEAD_FILE = "_unavailable.json"
    DEAD_STRIKES = 3      # failures before a ticker is written off

    def __init__(self, cache_dir=CACHE_DIR, allow_fetch=True, fetch_start=FETCH_START):
        self.dir = cache_dir
        self.allow_fetch = allow_fetch
        self.fetch_start = fetch_start
        os.makedirs(cache_dir, exist_ok=True)
        self.mem = {}
        self._fetched = set()          # tickers already downloaded this session
        # Several training-universe names were acquired or delisted after the training
        # window (ANSS, JNPR, HES, DFS, IPG, K ...). They have history for 2006-2023 but
        # do not exist in a 2025-26 backtest. Without memoising that, every run retries
        # them and every retry costs a 404 round-trip.
        self._dead = self._load_dead()
        self._strikes = {}
        try:
            import logging
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        except Exception:
            pass

    def _load_dead(self):
        path = os.path.join(self.dir, self.DEAD_FILE)
        if os.path.exists(path):
            try:
                import json
                with open(path, encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _save_dead(self):
        try:
            import json
            with open(os.path.join(self.dir, self.DEAD_FILE), "w", encoding="utf-8") as f:
                json.dump(sorted(self._dead), f)
        except Exception:
            pass

    @staticmethod
    def _covers(df, need_end):
        if df is None or not len(df):
            return False
        if need_end is None:
            return True
        return df.index[-1] >= pd.Timestamp(need_end) - pd.Timedelta(days=6)

    def _load_pkl(self, ticker):
        path = os.path.join(self.dir, f"{ticker}.pkl")
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_pickle(path)
            if df is not None and getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert(None)
            return df
        except Exception:
            return None

    def _download(self, ticker):
        try:
            import yfinance as yf
            end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.Ticker(ticker).history(start=self.fetch_start, end=end,
                                           interval="1d", auto_adjust=True)
            if df is None or df.empty:
                return None
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df.to_pickle(os.path.join(self.dir, f"{ticker}.pkl"))
            return df
        except Exception:
            return None

    def get(self, ticker, need_end=None):
        df = self.mem[ticker] if ticker in self.mem else self._load_pkl(ticker)
        if (self._covers(df, need_end) or not self.allow_fetch
                or ticker in self._fetched or ticker in self._dead):
            self.mem[ticker] = df
            return df
        fresh = self._download(ticker)
        self._fetched.add(ticker)
        if fresh is None:
            # A single failure does NOT mean the ticker is gone - Yahoo throttles under
            # load, and blacklisting on one miss poisons the cache permanently. Only
            # give up after DEAD_STRIKES failures, and only when there is no usable
            # cached history to fall back on.
            self._strikes[ticker] = self._strikes.get(ticker, 0) + 1
            if self._strikes[ticker] >= self.DEAD_STRIKES and (df is None or not len(df)):
                self._dead.add(ticker)
                self._save_dead()
            else:
                self._fetched.discard(ticker)      # allow a retry later in the session
        else:
            df = fresh
            self._dead.discard(ticker)
        self.mem[ticker] = df
        return df

    def prime(self, tickers, need_end=None, verbose=True):
        """Download everything up front, so a long backtest never stalls mid-run."""
        names = sorted(set(tickers))
        ok, missing = 0, []
        for i, t in enumerate(names, 1):
            df = self.get(t, need_end)
            if df is None or not len(df):
                missing.append(t)
            else:
                ok += 1
            if verbose and (i % 25 == 0 or i == len(names)):
                print(f"    primed {i}/{len(names)}  ({ok} ok, {len(missing)} unavailable)")
        if verbose and missing:
            print(f"\n    {len(missing)} ticker(s) have no data and will be skipped: "
                  f"{', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
            print(f"    Recorded in {os.path.join(self.dir, self.DEAD_FILE)} so later "
                  f"runs do not retry them.")
        return self


# Feature panels are computed ONCE per ticker over its whole history and then sliced
# by date. Every window in compute_panel is backward-looking (rolling / ewm with
# adjust=False), so the value at date D is identical whether the frame ends at D or
# years later — slicing is safe, and it turns ~11,000 panel builds into ~62.
_panels = {}


def _panel_for(ticker, df, bench_close):
    """
    Returns (panel, offset). `offset` is how many leading rows were trimmed, so a
    position in the FULL frame maps to the panel as pos - offset. Trimming only
    discards rows; every value was still computed from the whole history.
    """
    key = (ticker, len(df), df.index[-1])
    hit = _panels.get(ticker)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    panel = F.compute_panel(df, bench_close=bench_close)
    offset = 0
    if PANEL_KEEP_FROM:
        keep = panel.index >= pd.Timestamp(PANEL_KEEP_FROM)
        offset = int((~keep).sum())
        panel = panel[keep]
    panel = panel.astype("float32")          # half the memory, no meaningful precision loss
    _panels[ticker] = (key, panel, offset)
    return panel, offset


# =============================================================================
# SCAN
# =============================================================================
def scan(date=None, tickers=None, cache=None, top_n=TOP_N, min_pctl=MIN_PCTL,
         reference=USE_REFERENCE_COHORT):
    """
    Score a cohort for `date` and return the ranked table for the names in `tickers`.

    THE REFERENCE COHORT, AND WHY IT MATTERS

    xs_* features are percentile ranks WITHIN the cohort scored that day, so their
    meaning depends entirely on who else is in the cohort. Training cohorts were
    ~290 diversified names - utilities, REITs, staples, banks, megacap tech - where
    xs_atr_pct = 0.9 marks a genuinely extreme stock.

    Scoring only the 62 deploy names breaks that. They are all high-beta speculative
    tech, so 0.9 there means "slightly more volatile than other very volatile
    things". Same number, different meaning, and the model has never seen that
    distribution. The first V43 backtest showed exactly this: within-date AUC 0.521
    in cross-validation collapsed to a rank correlation of -0.000 against a 57-name
    cohort, with top-minus-bottom excess at -5.18 pts.

    So by default the cohort is built from the TRAINING universe plus the deploy
    names (~347), cross-sectional ranks and breadth are computed across all of them,
    and only then is the table filtered to `tickers` for ranking and selection. The
    features keep the meaning they were trained with; you still only trade your own
    list. Pass reference=False to reproduce the old deploy-only behaviour.
    """
    b = _bundle()
    tickers = list(tickers or DEPLOY_UNIVERSE)
    if reference:
        from pillar2_v43_universe import load_training_universe
        cohort_names = sorted(set(load_training_universe()) | set(tickers))
    else:
        cohort_names = sorted(set(tickers))

    target = pd.Timestamp(date) if date else pd.Timestamp.today().normalize()
    key = (target.strftime("%Y-%m-%d"), tuple(sorted(tickers)), top_n, min_pctl,
           bool(reference))
    if key in _cohort_cache:
        return _cohort_cache[key]

    cache = cache or _state.setdefault("cache", PriceCache())

    bench_full = cache.get(b.get("benchmark") or BENCHMARK, target)
    if bench_full is None or bench_full.empty:
        _cohort_cache[key] = None
        return None
    regime_full = _panels.get("__regime__")
    rkey = (len(bench_full), bench_full.index[-1])
    if regime_full is None or regime_full[0] != rkey:
        regime_full = (rkey, F.compute_regime_panel(bench_full))
        _panels["__regime__"] = regime_full
    rpos = regime_full[1].index.searchsorted(target, side="right") - 1
    regime = regime_full[1].iloc[:rpos + 1] if rpos >= 0 else regime_full[1].iloc[:0]

    rows = []
    for t in cohort_names:
        df_full = cache.get(t, target)
        if df_full is None or df_full.empty:
            continue
        panel, offset = _panel_for(t, df_full, bench_full["Close"])
        # position in the FULL frame, so the history check is unaffected by trimming
        pos = df_full.index.searchsorted(target, side="right") - 1
        if pos < MIN_BARS - 1:
            continue
        ppos = pos - offset
        if ppos < 0 or ppos >= len(panel):
            continue
        # A name acquired or delisted before `target` still has an old cached frame.
        # Using its last 2024 bar as if it were today's would be plain wrong, so the
        # ticker simply leaves the cohort — which is what a point-in-time universe
        # does anyway.
        if (target - df_full.index[pos]).days > MAX_BAR_STALENESS:
            continue
        feat = panel.iloc[ppos].to_dict()
        entry = float(df_full["Close"].iloc[pos])
        atr_v = (feat.get("atr_pct") or np.nan) * entry
        if not np.isfinite(atr_v) or atr_v <= 0 or entry <= 0:
            continue
        feat.update(ticker=t, entry=entry, atr_value=atr_v,
                    bar_date=str(df_full.index[pos].date()))
        rows.append(feat)

    if len(rows) < 5:
        _cohort_cache[key] = None
        return None

    cohort = pd.DataFrame(rows)
    # ranks and breadth across the FULL reference cohort - this is the fix
    cohort = F.add_cross_sectional(cohort)
    cohort["breadth_above_ema200"] = F.breadth(cohort)
    if len(regime):
        last = regime.iloc[-1]
        for c in regime.columns:
            cohort[c] = last[c]
    else:
        for c in regime.columns:
            cohort[c] = np.nan

    feats = b["feature_names"]
    for c in feats:
        if c not in cohort.columns:
            cohort[c] = np.nan

    model = b["model"]
    if b.get("is_ranker"):
        cohort["score"] = model.predict(cohort[feats])
    else:
        cohort["score"] = model.predict_proba(cohort[feats])[:, 1]
    cohort["pctl_ref"] = cohort["score"].rank(pct=True)     # place in the whole cohort
    cohort["cohort_size_ref"] = len(cohort)

    # now narrow to the names actually tradeable
    sel = cohort[cohort["ticker"].isin(tickers)].copy()
    if len(sel) < 3:
        _cohort_cache[key] = None
        return None
    sel["pctl"] = sel["score"].rank(pct=True)               # place among YOUR names
    sel = sel.sort_values("score", ascending=False).reset_index(drop=True)
    sel["rank"] = np.arange(1, len(sel) + 1)
    sel["is_pick"] = (sel["rank"] <= top_n) & (sel["pctl"] >= min_pctl)

    lv = [compute_levels(float(r.entry), float(r.atr_value), HORIZON)
          for r in sel.itertuples()]
    for col in ("target", "stop", "target_pct", "stop_pct", "rr"):
        sel[col] = [x.get(col, np.nan) for x in lv]
    sel["levels_ok"] = [bool(x.get("entry")) for x in lv]
    sel.loc[~sel["levels_ok"], "is_pick"] = False

    _cohort_cache[key] = sel
    return sel

# =============================================================================
# BACKWARDS-COMPATIBLE PER-TICKER INTERFACE
# =============================================================================
def selection_names(extra=None):
    """The names eligible to be picked, per SELECT_UNIVERSE."""
    if SELECT_UNIVERSE.startswith("train"):
        from pillar2_v43_universe import load_training_universe
        names = set(load_training_universe())
    else:
        names = set(DEPLOY_UNIVERSE)
    if extra:
        names |= set(extra)          # always score the ticker actually being asked about
    return sorted(names)


def run_pillar2_technical_quant(ticker, target_date_str=None):
    """
    Same signature the backtest and pipeline already call. Internally this scans the
    whole cohort for that date (cached per date, so a 5000-attempt backtest pays for
    each date once) and reports where `ticker` landed.

    "confidence" is the ticker's PERCENTILE in that day's cohort, in [0, 1] — not a
    probability. Section 5b of the backtest then measures the model on precisely the
    axis it was trained on.
    """
    # include the requested ticker even if it is outside the selection list, so a
    # backtest over a different universe (e.g. --universe SP500) still gets a ranking
    cohort = scan(target_date_str, tickers=selection_names([ticker]))
    if cohort is None:
        return None
    hit = cohort[cohort["ticker"] == ticker]
    if hit.empty:
        return None
    r = hit.iloc[0]

    pick = bool(r["is_pick"])
    sentiment = "Bullish" if pick else "Neutral"
    out = {
        "technical_sentiment": sentiment,
        "confidence": round(float(r["pctl"]), 4),
        "raw_score": float(r["score"]),
        "cohort_rank": int(r["rank"]),
        "cohort_size": int(len(cohort)),
        "cohort_pctl_ref": round(float(r["pctl_ref"]), 4),
        "reference_cohort_size": int(r["cohort_size_ref"]),
        "should_trade": pick,
        "trading_setup_type": (f"Cohort rank {int(r['rank'])}/{len(cohort)}"
                               if pick else
                               f"Not a top-{TOP_N} name today "
                               f"(rank {int(r['rank'])}/{len(cohort)})"),
        "model_version": _bundle().get("variant", "v43"),
        "signal_date": str(r["bar_date"]),
    }
    if pick and bool(r["levels_ok"]):
        out["actionable_levels"] = {"entry_price": float(r["entry"]),
                                    "target_price": float(r["target"]),
                                    "stop_loss": float(r["stop"])}
        out["risk_reward_ratio"] = f"1:{r['rr']}"
        out["level_detail"] = {"stop_pct": float(r["stop_pct"]),
                               "target_pct": float(r["target_pct"])}
    else:
        out["actionable_levels"] = {"entry_price": 0.0, "target_price": 0.0,
                                    "stop_loss": 0.0}
    return out


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="signal date (default: today)")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--min-pctl", type=float, default=MIN_PCTL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--no-reference", action="store_true",
                    help="score only the deploy names (the old, broken behaviour) — "
                         "for A/B comparison")
    ap.add_argument("--prime", action="store_true",
                    help="download the whole universe up front, then exit — "
                         "do this once before a long backtest")
    a = ap.parse_args()

    b = load_model(a.model)
    print("=" * 92)
    print(f"PILLAR 2 V43 COHORT SCAN")
    print("=" * 92)
    print(f"  model:    {_state['path']}  ({b.get('variant')}, "
          f"{'ranker' if b.get('is_ranker') else 'classifier'})")
    print(f"  trained:  through {b.get('train_end')}   "
          f"CV within-date AUC {b.get('cv_within_date_auc', float('nan')):.3f}")
    print(f"  geometry: {describe_geometry(HORIZON)}")
    print(f"  select:   top {a.top} of the '{SELECT_UNIVERSE}' universe "
          f"({len(selection_names())} names), min percentile {a.min_pctl:.0%}")

    if a.prime:
        print("\n  priming the price cache (one download per ticker) ...")
        from pillar2_v43_universe import load_training_universe
        names = list(DEPLOY_UNIVERSE) + [b.get("benchmark") or BENCHMARK]
        if not a.no_reference:
            names += load_training_universe()
        PriceCache().prime(names)
        print("  done — the backtest will now run from cache.")
        return

    cohort = scan(a.date, tickers=selection_names(), top_n=a.top,
                  min_pctl=a.min_pctl, reference=not a.no_reference)
    if cohort is None:
        sys.exit("Could not build a cohort. Check the price cache or the date.")

    print(f"\n  cohort: {len(cohort)} tradeable names on {cohort['bar_date'].iloc[0]}"
          + (f", ranked inside a {int(cohort['cohort_size_ref'].iloc[0])}-name "
             f"reference cohort" if not a.no_reference else
             "  (NO reference cohort — xs_* ranks will not match training)"))
    print(f"\n  {'#':>3}  {'ticker':<7}{'score':>9}{'pctl':>7}  "
          f"{'entry':>9}{'target':>9}{'stop':>9}{'tgt%':>7}{'stop%':>7}  pick")
    print("  " + "-" * 84)
    for r in cohort.head(a.show).itertuples():
        print(f"  {r.rank:>3}  {r.ticker:<7}{r.score:>9.4f}{r.pctl:>7.2f}  "
              f"{r.entry:>9.2f}{r.target:>9.2f}{r.stop:>9.2f}"
              f"{r.target_pct:>7.1f}{r.stop_pct:>7.1f}  "
              f"{'YES' if r.is_pick else ''}")

    picks = cohort[cohort["is_pick"]]
    print(f"\n  {len(picks)} pick(s) today"
          + (f": {', '.join(picks['ticker'])}" if len(picks) else
             " — nothing cleared the percentile floor."))
    print("\n  Scores are cohort-relative. A score is only meaningful next to the other")
    print("  scores from the same day, which is why this scans the universe as a whole.")


if __name__ == "__main__":
    main()
