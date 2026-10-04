#!/usr/bin/env python3
"""
class_pillar2_data_gen_v43.py - cohort dataset builder for Pillar 2 V43.

WHAT IS STRUCTURALLY DIFFERENT FROM V40/V42

  Old: draw a random (ticker, date), call yfinance, build one row. 5000 rows took a
       long time because every row was a network round-trip, and no row knew what any
       OTHER stock was doing that day — so cross-sectional features were impossible.

  New: download each ticker's full history ONCE into a local cache, compute its whole
       feature panel in one vectorised pass, then walk forward date by date. On each
       sample date, EVERY eligible ticker is scored together as a cohort. That cohort
       is what makes possible:
         - cross-sectional ranks (is this the 5th weakest of 300 today?)
         - breadth (what share of the universe is above its 200-day EMA?)
       and it makes 150k rows a pandas job instead of 150k HTTP calls.

WHAT IS KEPT FROM V42 (both earned it — AUC went from straddling 0.50 to clearing it
on two independent seeds, and the volatility screen fell from rho -0.465 to -0.209):
  - GEOMETRY_MODE = "VOL_SCALED" in trade_config
  - EXPIRED_AS = "Bearish": an expired trade did not reach target and tied up capital.
    Discarding expiries trains P(target | resolved) while you deploy P(target).

OUTPUT
  <name>.parquet  the dataset (falls back to .csv.gz if pyarrow is missing)
  <name>_index.json   [{ticker, signal_date}] - what the backtest's training-overlap
                      check reads, without loading the whole panel
  <name>_meta.json    feature list, config, date range, label counts

USAGE
  python class_pillar2_data_gen_v43.py --start 2005-01-01 --train-end 2023-06-30
  python class_pillar2_data_gen_v43.py --every 5 --max-rows 200000
  python class_pillar2_data_gen_v43.py --no-fetch          # cache only
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

import pillar2_v43_features as F
from pillar2_v43_universe import BENCHMARK, load_training_universe, universe_summary
from trade_config import (GEOMETRY_MODE, HORIZON_CONFIGS, compute_levels,
                          describe_geometry, geometry_tag, walk_trade)

# =============================================================================
# SETTINGS  (* = CLI flag)
# =============================================================================
HORIZON         = "MID"          # trade_config horizon -> 60 bars, VOL_SCALED levels
START_DATE      = "2005-01-01"   # * --start   how far back to sample signals
TRAIN_END_DATE  = "2023-06-30"   # * --train-end  last signal date; everything after
                                 #   this is left free for out-of-sample testing
SAMPLE_EVERY    = 10             # * --every   trading days between cohort dates
MIN_PAST_BARS   = 260            # need ~1 year for the 200-EMA and 250-bar momentum
MIN_DOLLAR_VOL  = 2_000_000      # * --min-dollar-vol  skip untradeable rows
EXPIRED_AS      = "Bearish"      # keep as Bearish; see the module docstring
MAX_ROWS        = 250_000        # * --max-rows  hard cap so the file stays sane

CACHE_DIR       = "price_cache_v43"
OUT_DIR         = "."
DATASET_VERSION = "v43"


# =============================================================================
# PRICE CACHE
# =============================================================================
class PriceCache:
    """One download per ticker, kept on disk, reused across every run."""

    def __init__(self, cache_dir, start, end, allow_fetch=True):
        self.dir = cache_dir
        self.start, self.end = start, end
        self.allow_fetch = allow_fetch
        os.makedirs(cache_dir, exist_ok=True)
        self.mem = {}

    def get(self, ticker):
        if ticker in self.mem:
            return self.mem[ticker]
        path = os.path.join(self.dir, f"{ticker}.pkl")
        df = None
        if os.path.exists(path):
            try:
                df = pd.read_pickle(path)
            except Exception:
                df = None
        if df is None and self.allow_fetch:
            try:
                import yfinance as yf
                df = yf.Ticker(ticker).history(start=self.start, end=self.end,
                                               interval="1d", auto_adjust=True)
                if df is None or df.empty:
                    df = None
                else:
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert(None)
                    df.to_pickle(path)
            except Exception as e:
                print(f"    {ticker}: download failed ({type(e).__name__}: {e})")
                df = None
        if df is not None and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        self.mem[ticker] = df
        return df


# =============================================================================
# LABELS
# =============================================================================
def label_one(panel_row, forward_bars, entry_price, atr_value):
    """
    Build the label for one (ticker, date) using the SAME functions the backtest
    grades with, so labels and grading cannot drift apart.

    Returns (label, outcome, levels) or None if the row is unusable.
    """
    lv = compute_levels(entry_price, atr_value, HORIZON)
    if not lv.get("entry") or lv.get("reason") not in (None, "ok"):
        return None
    res = walk_trade(forward_bars, lv["entry"], lv["stop"], lv["target"])
    outcome = res["outcome"]
    if outcome == "target_hit":
        label = 1
    elif outcome == "stop_hit":
        label = 0
    else:                                    # expired
        if EXPIRED_AS == "SKIP":
            return None
        label = 0
    return label, outcome, lv, res


def bars_from_frame(df):
    return [{"low": float(l), "high": float(h), "close": float(c)}
            for l, h, c in zip(df["Low"].to_numpy(), df["High"].to_numpy(),
                               df["Close"].to_numpy())]


# =============================================================================
# BUILD
# =============================================================================
def build(cfg):
    cfg_lookahead = HORIZON_CONFIGS[HORIZON]["lookahead_bars"]
    tickers = load_training_universe(include_deploy=not cfg.no_deploy)

    print("=" * 78)
    print(f"PILLAR 2 V43 DATASET BUILDER")
    print("=" * 78)
    print(f"  universe:   {universe_summary()}")
    print(f"  signals:    {cfg.start} -> {cfg.train_end}   (every {cfg.every} trading days)")
    print(f"  horizon:    {HORIZON} | {cfg_lookahead} bars forward")
    print(f"  geometry:   {describe_geometry(HORIZON)}")
    print(f"  expired as: {EXPIRED_AS}")
    print()

    fetch_end = (pd.Timestamp(cfg.train_end) + pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    cache = PriceCache(cfg.cache_dir, cfg.start, fetch_end, allow_fetch=not cfg.no_fetch)

    # ── benchmark first: needed for relative strength and regime ──────────
    print(f"Loading benchmark {BENCHMARK} ...")
    bench = cache.get(BENCHMARK)
    if bench is None or bench.empty:
        sys.exit(f"Could not load benchmark {BENCHMARK}. Check the network or the cache.")
    regime = F.compute_regime_panel(bench)
    bench_close = bench["Close"]

    # ── every ticker's panel, once ────────────────────────────────────────
    print(f"\nBuilding feature panels for {len(tickers)} tickers ...")
    panels, prices, dropped = {}, {}, []
    for i, t in enumerate(tickers, 1):
        df = cache.get(t)
        if df is None or len(df) < MIN_PAST_BARS + cfg_lookahead:
            dropped.append(t)
            continue
        panels[t] = F.compute_panel(df, bench_close=bench_close)
        prices[t] = df
        if i % 25 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {len(panels)} usable, {len(dropped)} dropped")
    if not panels:
        sys.exit("No usable tickers. Run once with network access to fill the cache.")
    print(f"\n  {len(panels)} tickers usable; dropped {len(dropped)} for short history"
          + (f": {', '.join(dropped[:12])}{' ...' if len(dropped) > 12 else ''}"
             if dropped else ""))

    # ── cohort dates: the benchmark's trading days, thinned ───────────────
    lo, hi = pd.Timestamp(cfg.start), pd.Timestamp(cfg.train_end)
    all_days = bench.index[(bench.index >= lo) & (bench.index <= hi)]
    dates = all_days[::cfg.every]
    print(f"\n  {len(dates)} cohort dates from {len(all_days)} trading days")

    rows, n_skipped = [], 0
    for di, day in enumerate(dates, 1):
        cohort = []
        for t, panel in panels.items():
            if day not in panel.index:
                continue
            pos = panel.index.get_loc(day)
            if pos < MIN_PAST_BARS:
                continue
            df = prices[t]
            fwd = df.iloc[pos + 1: pos + 1 + cfg_lookahead]
            if len(fwd) < cfg_lookahead:
                continue                      # not enough future to resolve the label

            feat = panel.iloc[pos]
            if not np.isfinite(feat.get("log_dollar_vol", np.nan)):
                continue
            if 10 ** feat["log_dollar_vol"] < cfg.min_dollar_vol:
                continue

            entry = float(df["Close"].iloc[pos])
            atr_v = float(feat["atr_pct"] * entry) if np.isfinite(feat.get("atr_pct", np.nan)) else 0.0
            if entry <= 0 or atr_v <= 0:
                continue

            lab = label_one(feat, bars_from_frame(fwd), entry, atr_v)
            if lab is None:
                n_skipped += 1
                continue
            label, outcome, lv, res = lab

            r = feat.to_dict()
            r.update(ticker=t, signal_date=day.strftime("%Y-%m-%d"), label=label,
                     outcome=outcome, entry=lv["entry"], target=lv["target"],
                     stop=lv["stop"], target_pct=lv["target_pct"],
                     stop_pct=lv["stop_pct"], exit_return_pct=res["exit_return_pct"],
                     bars_held=res["bars_held"])
            cohort.append(r)

        if len(cohort) < cfg.min_cohort:
            continue

        c = pd.DataFrame(cohort)
        c = F.add_cross_sectional(c)
        c["breadth_above_ema200"] = F.breadth(c)
        if day in regime.index:
            for col in regime.columns:
                c[col] = regime.loc[day, col]
        else:
            for col in regime.columns:
                c[col] = np.nan
        rows.append(c)

        if di % 25 == 0 or di == len(dates):
            total = sum(len(x) for x in rows)
            print(f"  [{di}/{len(dates)}] {day:%Y-%m-%d}  cohort {len(cohort):>3}  "
                  f"total rows {total:,}")
        if sum(len(x) for x in rows) >= cfg.max_rows:
            print(f"  hit --max-rows ({cfg.max_rows:,}), stopping at {day:%Y-%m-%d}")
            break

    if not rows:
        sys.exit("No rows produced. Widen the date range or lower --min-cohort.")
    data = pd.concat(rows, ignore_index=True)

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  rows:        {len(data):,}")
    print(f"  tickers:     {data['ticker'].nunique()}")
    print(f"  dates:       {data['signal_date'].nunique()} "
          f"({data['signal_date'].min()} -> {data['signal_date'].max()})")
    print(f"  features:    {len(F.FEATURE_NAMES)}")
    oc = data["outcome"].value_counts().to_dict()
    print(f"  outcomes:    {oc}")
    print(f"  base rate:   {data['label'].mean():.1%}  (target hit before stop)")
    print(f"  skipped:     {n_skipped:,} rows unusable at labelling")

    miss = data[F.FEATURE_NAMES].isna().mean().sort_values(ascending=False)
    bad = miss[miss > 0.10]
    if len(bad):
        print(f"\n  features >10% missing (XGBoost handles NaN, but check these):")
        for k, v in bad.head(8).items():
            print(f"    {k:<24} {v:.1%}")

    print(f"\n  base rate by year (a healthy set varies with the market):")
    yr = data.assign(year=data["signal_date"].str[:4]).groupby("year")["label"]
    for y, g in yr:
        print(f"    {y}  n={len(g):>7,}  base rate {g.mean():.1%}")

    # ── save ──────────────────────────────────────────────────────────────
    base = (f"dataset_pillar2_{HORIZON.lower()}_{DATASET_VERSION}"
            f"_{geometry_tag()}_cut{pd.Timestamp(cfg.train_end):%Y%m%d}")
    out = os.path.join(cfg.out_dir, base)
    try:
        data.to_parquet(out + ".parquet", index=False)
        data_path = out + ".parquet"
    except Exception as e:
        print(f"\n  parquet unavailable ({type(e).__name__}), writing csv.gz instead")
        data.to_csv(out + ".csv.gz", index=False, compression="gzip")
        data_path = out + ".csv.gz"

    # index the backtest's overlap check can read without loading the panel
    with open(out + "_index.json", "w", encoding="utf-8") as f:
        json.dump(data[["ticker", "signal_date"]].to_dict("records"), f)

    meta = {
        "version": DATASET_VERSION,
        "feature_version": F.FEATURE_VERSION,
        "feature_names": F.FEATURE_NAMES,
        "n_rows": int(len(data)),
        "geometry": GEOMETRY_MODE,
        "geometry_tag": geometry_tag(),
        "horizon": HORIZON,
        "lookahead_bars": cfg_lookahead,
        "expired_as": EXPIRED_AS,
        "benchmark": BENCHMARK,
        "sample_every": cfg.every,
        "start": cfg.start,
        "train_end": cfg.train_end,
        "min_dollar_vol": cfg.min_dollar_vol,
        "base_rate": float(data["label"].mean()),
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  wrote {data_path}")
    print(f"  wrote {out}_index.json")
    print(f"  wrote {out}_meta.json")
    print(f"\n  Signals stop at {cfg.train_end}. Anything after that date is free for")
    print(f"  out-of-sample testing — leave at least {cfg_lookahead} trading days of gap.")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--train-end", default=TRAIN_END_DATE)
    p.add_argument("--every", type=int, default=SAMPLE_EVERY,
                   help="trading days between cohort dates; lower = more rows, "
                        "more overlap between labels")
    p.add_argument("--min-cohort", type=int, default=30,
                   help="skip dates with fewer usable tickers than this — "
                        "cross-sectional ranks are meaningless on a tiny cohort")
    p.add_argument("--min-dollar-vol", type=float, default=MIN_DOLLAR_VOL)
    p.add_argument("--max-rows", type=int, default=MAX_ROWS)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-deploy", action="store_true",
                   help="train only on the long-history names, excluding the "
                        "deploy universe entirely")
    build(p.parse_args())


if __name__ == "__main__":
    main()
