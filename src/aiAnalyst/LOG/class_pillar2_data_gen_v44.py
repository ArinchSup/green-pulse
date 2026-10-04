#!/usr/bin/env python3
"""
class_pillar2_data_gen_v44.py - cohort dataset with FORWARD RETURN labels.

WHAT CHANGED FROM V43, AND WHY

V39 through V43 all used a barrier label: "did it hit +X% before -Y% within 60 bars?"
That label is path-dependent and volatility-contaminated - whether a barrier is
touched depends on direction AND on how much the stock moves at all. The two kept
getting tangled:

    V40  learned to pick low-volatility names, because under FIXED geometry their
         trades expired instead of resolving (48.2% "success" in training vs 27.9%
         in the backtest)
    V42  VOL_SCALED fixed that, and the clamps flipped the bias the other way
    V43  atr_pct and xs_atr_pct stayed in the top 12 by gain throughout

Meanwhile every documented cross-sectional equity effect - short-term reversal,
12-1 momentum, low-volatility, illiquidity - predicts RETURNS, not barrier hits. We
have been asking a noisier question than the literature asks.

V44 labels each row with plain forward returns at several horizons, and with those
returns DEMEANED WITHIN THE DATE. Demeaning removes the market move entirely, so what
is left is the only thing a cross-sectional ranker can actually exploit: did this
stock beat the others scored the same day.

    fwd_ret_N      % return, entered at the NEXT bar's open, exited at the close
                   N bars later. Next-open entry is what the backtest simulates, so
                   labels and grading agree.
    fwd_exc_N      fwd_ret_N minus the cohort mean for that date (market removed)
    fwd_rank_N     percentile rank of fwd_ret_N within the date, in [0, 1]

Horizons 5, 10, 20 and 60 bars are all emitted from ONE build, so testing which
horizon carries signal costs four training runs, not four rebuilds. 60 is kept purely
as the control that reproduces V43's horizon.

The V43 barrier label is also kept (label / outcome columns) so the old and new
formulations can be compared on identical rows.

FEATURES ARE UNCHANGED - pillar2_v43_features.py, same 67 columns. This isolates the
label change: if V44 works and V43 did not, it is the target that mattered.

USAGE
  python class_pillar2_data_gen_v44.py --start 2005-01-01 --train-end 2023-06-30
  python class_pillar2_data_gen_v44.py --every 5 --max-rows 300000
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

HORIZONS = (5, 10, 20, 60)        # forward bars to label
MAX_H = max(HORIZONS)

HORIZON = "MID"
START_DATE = "2005-01-01"
TRAIN_END_DATE = "2023-06-30"
SAMPLE_EVERY = 10
MIN_PAST_BARS = 260
MIN_DOLLAR_VOL = 2_000_000
MAX_ROWS = 300_000
CACHE_DIR = "price_cache_v43"      # shared with V43; no need to re-download
DATASET_VERSION = "v44"


class PriceCache:
    """One download per ticker, kept on disk, reused across runs."""

    def __init__(self, cache_dir, start, end, allow_fetch=True):
        self.dir, self.start, self.end = cache_dir, start, end
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
                print(f"    {ticker}: {type(e).__name__}")
                df = None
        if df is not None and getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        self.mem[ticker] = df
        return df


def forward_returns(df, pos):
    """
    Returns at each horizon, entered at the NEXT bar's open (pos+1) and exited at the
    close N bars later. Entering at the signal bar's own close would be look-ahead:
    you cannot trade a close you are still using to generate the signal.
    """
    if pos + 1 >= len(df):
        return None
    entry = float(df["Open"].iloc[pos + 1])
    if entry <= 0:
        return None
    out = {}
    for h in HORIZONS:
        end = pos + h
        if end >= len(df):
            return None
        out[f"fwd_ret_{h}"] = (float(df["Close"].iloc[end]) - entry) / entry * 100.0
    out["_entry_open"] = entry
    return out


def barrier_label(df, pos, entry_close, atr_value):
    """The V43 label, kept so both formulations can be compared on identical rows."""
    lv = compute_levels(entry_close, atr_value, HORIZON)
    if not lv.get("entry") or lv.get("reason") not in (None, "ok"):
        return None
    bars = df.iloc[pos + 1: pos + 1 + HORIZON_CONFIGS[HORIZON]["lookahead_bars"]]
    if len(bars) < HORIZON_CONFIGS[HORIZON]["lookahead_bars"]:
        return None
    rows = [{"low": float(l), "high": float(h), "close": float(c)}
            for l, h, c in zip(bars["Low"], bars["High"], bars["Close"])]
    res = walk_trade(rows, lv["entry"], lv["stop"], lv["target"])
    return (1 if res["outcome"] == "target_hit" else 0), res["outcome"], lv, res


def build(cfg):
    lookahead = HORIZON_CONFIGS[HORIZON]["lookahead_bars"]
    need_forward = max(MAX_H, lookahead) + 2
    tickers = load_training_universe(include_deploy=not cfg.no_deploy)

    print("=" * 80)
    print("PILLAR 2 V44 DATASET BUILDER  (forward-return labels)")
    print("=" * 80)
    print(f"  universe:   {universe_summary()}")
    print(f"  signals:    {cfg.start} -> {cfg.train_end}  (every {cfg.every} trading days)")
    print(f"  horizons:   {HORIZONS} bars forward, entered at the next bar's open")
    print(f"  geometry:   {describe_geometry(HORIZON)}  (barrier label kept for comparison)")
    print()

    fetch_end = (pd.Timestamp(cfg.train_end) + pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    cache = PriceCache(cfg.cache_dir, cfg.start, fetch_end, allow_fetch=not cfg.no_fetch)

    print(f"Loading benchmark {BENCHMARK} ...")
    bench = cache.get(BENCHMARK)
    if bench is None or bench.empty:
        sys.exit(f"Could not load {BENCHMARK}.")
    regime = F.compute_regime_panel(bench)
    bench_close = bench["Close"]

    print(f"\nBuilding panels for {len(tickers)} tickers ...")
    panels, prices, dropped = {}, {}, []
    for i, t in enumerate(tickers, 1):
        df = cache.get(t)
        if df is None or len(df) < MIN_PAST_BARS + need_forward:
            dropped.append(t)
            continue
        panels[t] = F.compute_panel(df, bench_close=bench_close)
        prices[t] = df
        if i % 50 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {len(panels)} usable, {len(dropped)} dropped")
    if not panels:
        sys.exit("No usable tickers.")

    lo, hi = pd.Timestamp(cfg.start), pd.Timestamp(cfg.train_end)
    days = bench.index[(bench.index >= lo) & (bench.index <= hi)][::cfg.every]
    print(f"\n  {len(days)} cohort dates")

    rows, skipped = [], 0
    for di, day in enumerate(days, 1):
        cohort = []
        for t, panel in panels.items():
            if day not in panel.index:
                continue
            pos = panel.index.get_loc(day)
            if pos < MIN_PAST_BARS:
                continue
            df = prices[t]
            feat = panel.iloc[pos]
            if not np.isfinite(feat.get("log_dollar_vol", np.nan)):
                continue
            if 10 ** feat["log_dollar_vol"] < cfg.min_dollar_vol:
                continue

            fwd = forward_returns(df, pos)
            if fwd is None:
                skipped += 1
                continue

            entry_close = float(df["Close"].iloc[pos])
            atr_v = float(feat["atr_pct"] * entry_close) if np.isfinite(
                feat.get("atr_pct", np.nan)) else 0.0
            if entry_close <= 0 or atr_v <= 0:
                continue

            r = feat.to_dict()
            r.update(ticker=t, signal_date=day.strftime("%Y-%m-%d"),
                     entry_open=fwd.pop("_entry_open"), **fwd)

            bl = barrier_label(df, pos, entry_close, atr_v)
            if bl is not None:
                lab, outcome, lv, res = bl
                r.update(label=lab, outcome=outcome,
                         exit_return_pct=res["exit_return_pct"],
                         target_pct=lv["target_pct"], stop_pct=lv["stop_pct"])
            else:
                r.update(label=np.nan, outcome="", exit_return_pct=np.nan,
                         target_pct=np.nan, stop_pct=np.nan)
            cohort.append(r)

        if len(cohort) < cfg.min_cohort:
            continue
        c = pd.DataFrame(cohort)

        # ── the V44 targets: market removed, within this date ────────────
        for h in HORIZONS:
            col = f"fwd_ret_{h}"
            c[f"fwd_exc_{h}"] = c[col] - c[col].mean()     # cohort-demeaned
            c[f"fwd_rank_{h}"] = c[col].rank(pct=True)     # percentile within the day

        c = F.add_cross_sectional(c)
        c["breadth_above_ema200"] = F.breadth(c)
        if day in regime.index:
            for col in regime.columns:
                c[col] = regime.loc[day, col]
        else:
            for col in regime.columns:
                c[col] = np.nan
        rows.append(c)

        if di % 50 == 0 or di == len(days):
            print(f"  [{di}/{len(days)}] {day:%Y-%m-%d}  cohort {len(cohort):>3}  "
                  f"rows {sum(len(x) for x in rows):,}")
        if sum(len(x) for x in rows) >= cfg.max_rows:
            print(f"  hit --max-rows at {day:%Y-%m-%d}")
            break

    if not rows:
        sys.exit("No rows produced.")
    data = pd.concat(rows, ignore_index=True)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"  rows {len(data):,} | tickers {data['ticker'].nunique()} | "
          f"dates {data['signal_date'].nunique()}")
    print(f"  span {data['signal_date'].min()} -> {data['signal_date'].max()}")
    print(f"  skipped (no forward window): {skipped:,}")

    print(f"\n  {'horizon':>8}{'mean ret':>11}{'sd':>9}{'mean exc':>11}{'sd exc':>9}"
          f"{'% positive':>12}")
    for h in HORIZONS:
        r_, e_ = data[f"fwd_ret_{h}"], data[f"fwd_exc_{h}"]
        print(f"  {h:>8}{r_.mean():>+11.2f}{r_.std():>9.2f}{e_.mean():>+11.3f}"
              f"{e_.std():>9.2f}{(r_ > 0).mean()*100:>11.1f}%")
    print("\n  mean excess must be ~0 by construction — it is the market, removed.")
    print("  The sd of the excess column is the dispersion a ranker can exploit.")

    if data["label"].notna().any():
        print(f"\n  V43 barrier label kept: base rate {data['label'].mean():.1%} "
              f"on {int(data['label'].notna().sum()):,} rows")

    base = (f"dataset_pillar2_{HORIZON.lower()}_{DATASET_VERSION}"
            f"_{geometry_tag()}_cut{pd.Timestamp(cfg.train_end):%Y%m%d}")
    out = os.path.join(cfg.out_dir, base)
    try:
        data.to_parquet(out + ".parquet", index=False)
        path = out + ".parquet"
    except Exception:
        data.to_csv(out + ".csv.gz", index=False, compression="gzip")
        path = out + ".csv.gz"
    with open(out + "_index.json", "w", encoding="utf-8") as f:
        json.dump(data[["ticker", "signal_date"]].to_dict("records"), f)
    with open(out + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"version": DATASET_VERSION, "feature_version": F.FEATURE_VERSION,
                   "feature_names": F.FEATURE_NAMES, "horizons": list(HORIZONS),
                   "targets": [f"fwd_exc_{h}" for h in HORIZONS],
                   "n_rows": int(len(data)), "geometry": GEOMETRY_MODE,
                   "benchmark": BENCHMARK, "sample_every": cfg.every,
                   "start": cfg.start, "train_end": cfg.train_end,
                   "built": datetime.datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"\n  wrote {path}")
    print(f"  wrote {out}_index.json / _meta.json")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--train-end", default=TRAIN_END_DATE)
    p.add_argument("--every", type=int, default=SAMPLE_EVERY)
    p.add_argument("--min-cohort", type=int, default=30)
    p.add_argument("--min-dollar-vol", type=float, default=MIN_DOLLAR_VOL)
    p.add_argument("--max-rows", type=int, default=MAX_ROWS)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-deploy", action="store_true")
    build(p.parse_args())


if __name__ == "__main__":
    main()
