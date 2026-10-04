#!/usr/bin/env python3
"""
check_volatility_screen.py - is V40's confidence a direction signal, or a volatility screen?

The pattern that prompted this: across score buckets the WIN rate is flat (38.4 / 37.1 / 38.5 /
38.2 / 39.5 / 42.3) while the LOSS rate falls steadily (57.4 -> 38.5) and the EXPIRY rate rises
(4.2 -> 19.2). Losses turning into expiries, not into wins, is what you'd see if the model were
picking low-volatility names: a quiet stock reaches neither +20% nor -12% inside 60 bars, so it
expires near a mildly positive close in a rising market. That would make the mean-return trend an
artifact of volatility selection rather than evidence of directional skill.

This script tests that three ways:
  1. What does confidence actually correlate with? (Spearman, stock-month clustered ranges)
  2. Do outcomes move with volatility the way the hypothesis predicts?
  3. THE DECISIVE TEST - inside a single volatility bucket, does confidence still predict
     anything? If the edge disappears once volatility is held constant, it was never direction.

Usage
  python check_volatility_screen.py --model V40
  python check_volatility_screen.py --model V40 --cache atr_cache.csv
  python check_volatility_screen.py --model V40 --no-fetch        # cache only, no downloads
  python check_volatility_screen.py --model V40 --conf-split 0.65

Price data is fetched ONCE PER TICKER over the whole date span and cached, not once per row.
"""
import argparse
import datetime
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

RUN_RE = re.compile(
    r"^(?P<model>.+)_(?P<universe>SHAY|TECH_ONLY|SP500)_seed(?P<seed>\d+)"
    # [A-Z_]+ rather than a fixed list, so a new levels mode never breaks the parser
    r"_(?P<end>\d{8})_(?P<levels>[A-Z0-9_]+?)(?P<nobe>_nobe)?_samples\.csv$"
)

WIN, LOSS, BE, EXP = "Win", "Loss", "Breakeven", "Expired"

# features computed at each signal date, all point-in-time (bars up to and including that day)
FEATURES = {
    "atr_pct":     "ATR(14) / close - the volatility measure your geometry is exposed to",
    "vol_20d":     "stdev of daily returns over 20 bars (%)",
    "ret_60d":     "return over the prior 60 bars (%) - how extended the name already is",
    "gap_ema200":  "close / EMA(200) - 1 (%)",
    "log_dollar_vol": "log10 of 20-bar average dollar volume",
}
LOOKBACK_DAYS = 420      # enough calendar days for a 200-bar EMA to settle


# =============================================================================
# SAMPLES
# =============================================================================
def load_samples(directory, model_filter):
    paths = sorted(glob.glob(os.path.join(directory, "*_samples.csv")))
    if not paths:
        sys.exit(f"No *_samples.csv in '{directory}'.")
    frames = []
    for p in paths:
        m = RUN_RE.match(os.path.basename(p))
        if not m or (model_filter and m.group("model") != model_filter):
            continue
        df = pd.read_csv(p)
        df["model_label"] = m.group("model")
        df["seed"] = int(m.group("seed"))
        frames.append(df)
    if not frames:
        sys.exit(f"No runs matched model '{model_filter}'.")
    df = pd.concat(frames, ignore_index=True)
    # same de-duplication as the pooling script: a (ticker, day) both seeds drew is one trade
    before = len(df)
    df = df.sort_values(["seed", "attempt"]).drop_duplicates(
        subset=["ticker", "signal_date"], keep="first").reset_index(drop=True)
    print(f"  loaded {before} rows from {len(frames)} file(s), "
          f"{len(df)} after de-duplication")
    return df


# =============================================================================
# POINT-IN-TIME FEATURES
# =============================================================================
def compute_frame_features(hist):
    """Add the feature columns to one ticker's daily history. Every value uses bars up to
    and including that row - nothing from the future."""
    h = hist.copy()
    close, high, low = h["Close"], h["High"], h["Low"]

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    h["atr_pct"] = (tr.rolling(14).mean() / close) * 100.0

    ret = close.pct_change()
    h["vol_20d"] = ret.rolling(20).std() * 100.0
    h["ret_60d"] = (close / close.shift(60) - 1.0) * 100.0
    h["gap_ema200"] = (close / close.ewm(span=200, adjust=False).mean() - 1.0) * 100.0
    dv = (close * h["Volume"]).rolling(20).mean()
    h["log_dollar_vol"] = np.log10(dv.where(dv > 0))
    return h


def fetch_features(tickers, first_date, last_date):
    """One download per ticker over the whole span, then read off each signal date."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. pip install yfinance, or use --no-fetch with a cache.")

    start = (pd.Timestamp(first_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(last_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    out = {}
    for i, t in enumerate(sorted(tickers), 1):
        try:
            h = yf.Ticker(t).history(start=start, end=end, interval="1d")
            if h is None or h.empty:
                print(f"  [{i}/{len(tickers)}] {t:<6} no data")
                continue
            if h.index.tz is not None:
                h.index = h.index.tz_convert(None)
            out[t] = compute_frame_features(h)
            print(f"  [{i}/{len(tickers)}] {t:<6} {len(h)} bars")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {t:<6} failed: {type(e).__name__}: {e}")
    return out


def attach_features(df, frames):
    """Look up each (ticker, signal_date) in its ticker's frame, taking the last bar at or
    before the signal date."""
    cols = list(FEATURES)
    vals = {c: [] for c in cols}
    missing = 0
    for t, d in zip(df["ticker"], df["signal_date"]):
        h = frames.get(t)
        if h is None:
            missing += 1
            for c in cols:
                vals[c].append(np.nan)
            continue
        ts = pd.Timestamp(str(d)[:10])
        sub = h.loc[h.index <= ts]
        if sub.empty:
            missing += 1
            for c in cols:
                vals[c].append(np.nan)
            continue
        row = sub.iloc[-1]
        for c in cols:
            vals[c].append(float(row.get(c, np.nan)))
    for c in cols:
        df[c] = vals[c]
    if missing:
        print(f"  note: {missing} rows had no price data at their signal date")
    return df


# =============================================================================
# STATS
# =============================================================================
def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb)) if sa and sb else np.nan


def strict_wr(out):
    return float((out == WIN).sum()) / len(out) * 100.0 if len(out) else np.nan


def rate(out, which):
    return float((out == which).sum()) / len(out) * 100.0 if len(out) else np.nan


class ClusterBoot:
    def __init__(self, df, rounds, seed):
        self.rounds, self.rng = rounds, np.random.default_rng(seed)
        g = df.groupby("cluster").indices
        self.idx = [np.asarray(v) for v in g.values()]

    def __iter__(self):
        k = len(self.idx)
        for _ in range(self.rounds):
            yield np.concatenate([self.idx[i] for i in self.rng.integers(0, k, size=k)])


def ranged(vals, lo=2.5, hi=97.5):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return (np.nan, np.nan) if v.size < 20 else (float(np.percentile(v, lo)),
                                                 float(np.percentile(v, hi)))


def f(x, spec=".2f", suf=""):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}{suf}"


# =============================================================================
# SECTIONS
# =============================================================================
def section_correlations(df, rounds, seed):
    print("\n" + "=" * 96)
    print("1) WHAT DOES CONFIDENCE CORRELATE WITH?  (Spearman, stock-month clustered 95% range)")
    print("=" * 96)
    conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype=float)
    boot = list(ClusterBoot(df, rounds, seed))

    print(f"{'feature':<16}{'rho':>8}{'95% range':>20}   meaning")
    print("-" * 96)
    for c, desc in FEATURES.items():
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        r = spearman(conf, x)
        lo, hi = ranged([spearman(conf[rows], x[rows]) for rows in boot])
        verdict = ""
        if np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0:
            verdict = "  <-- clear of zero"
        print(f"{c:<16}{f(r, '+.3f'):>8}{f(lo, '+.3f') + ' to ' + f(hi, '+.3f'):>20}"
              f"   {desc}{verdict}")
    print("\n  A strongly NEGATIVE rho on atr_pct or vol_20d means high confidence goes to quiet")
    print("  stocks - the model is screening volatility, not predicting direction.")


def section_outcomes_by_vol(df, vol_col, q, min_n):
    print("\n" + "=" * 96)
    print(f"2) OUTCOMES BY {vol_col.upper()} QUINTILE  (all samples, training-label rules)")
    print("=" * 96)
    print("   The hypothesis predicts: quiet names -> fewer losses AND fewer wins, more expiries.")
    print()
    buckets = pd.qcut(pd.to_numeric(df[vol_col], errors="coerce"), q,
                      labels=[f"Q{i+1}" for i in range(q)], duplicates="drop")
    out = df["label_outcome"].to_numpy()
    ret = pd.to_numeric(df["rand_ret"], errors="coerce").to_numpy(dtype=float)

    header = (f"{'Bucket':<8}{vol_col + ' range':>22}{'n':>7}{'WinRate':>9}{'LossRate':>10}"
              f"{'ExpRate':>9}{'MeanRet':>10}{'MeanConf':>10}{'ModelFires':>12}")
    print(header)
    print("-" * len(header))
    for b in [x for x in buckets.cat.categories if (buckets == x).any()]:
        m = (buckets == b).to_numpy()
        v = pd.to_numeric(df.loc[m, vol_col], errors="coerce")
        fires = int((df.loc[m, "is_model"] == True).sum())  # noqa: E712
        flag = "  <- small" if m.sum() < min_n else ""
        print(f"{b:<8}{f(v.min(), '.2f') + ' to ' + f(v.max(), '.2f'):>22}{m.sum():>7}"
              f"{f(strict_wr(out[m]), '.1f', '%'):>9}{f(rate(out[m], LOSS), '.1f', '%'):>10}"
              f"{f(rate(out[m], EXP), '.1f', '%'):>9}{f(np.nanmean(ret[m]), '+.2f'):>10}"
              f"{f(pd.to_numeric(df.loc[m, 'confidence'], errors='coerce').mean(), '.3f'):>10}"
              f"{fires:>12}{flag}")
    return buckets


def section_conditional(df, buckets, split, rounds, seed, min_n):
    """The decisive test: inside one volatility bucket, does confidence still separate?"""
    print("\n" + "=" * 96)
    print(f"3) DOES CONFIDENCE STILL PREDICT *INSIDE* A VOLATILITY BUCKET?  (split at {split:.2f})")
    print("=" * 96)
    print("   If the gaps below cluster around zero, confidence was carrying volatility, not")
    print("   direction. If they stay positive within buckets, there is real signal underneath.")
    print()
    conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(dtype=float)
    out = df["label_outcome"].to_numpy()
    hi_mask = conf >= split
    boot = list(ClusterBoot(df, rounds, seed))
    bvals = buckets.to_numpy()

    header = (f"{'Bucket':<8}{'n low':>8}{'WR low':>9}{'n high':>8}{'WR high':>9}"
              f"{'gap':>9}{'95% range':>20}")
    print(header)
    print("-" * len(header))
    for b in [x for x in buckets.cat.categories if (buckets == x).any()]:
        m = bvals == b
        lo_m, hi_m = m & ~hi_mask, m & hi_mask
        if hi_m.sum() == 0 or lo_m.sum() == 0:
            print(f"{b:<8}{lo_m.sum():>8}{'':>9}{hi_m.sum():>8}"
                  f"{'':>9}{'n/a':>9}{'one side empty':>20}")
            continue
        gap = strict_wr(out[hi_m]) - strict_wr(out[lo_m])
        gaps = []
        for rows in boot:
            bb, cc, oo = bvals[rows], conf[rows], out[rows]
            mm = bb == b
            l2, h2 = mm & (cc < split), mm & (cc >= split)
            if l2.sum() and h2.sum():
                gaps.append(strict_wr(oo[h2]) - strict_wr(oo[l2]))
        glo, ghi = ranged(gaps)
        flag = "  <- small" if hi_m.sum() < min_n else ""
        print(f"{b:<8}{lo_m.sum():>8}{f(strict_wr(out[lo_m]), '.1f', '%'):>9}"
              f"{hi_m.sum():>8}{f(strict_wr(out[hi_m]), '.1f', '%'):>9}"
              f"{f(gap, '+.1f', 'pp'):>9}"
              f"{f(glo, '+.1f') + ' to ' + f(ghi, '+.1f'):>20}{flag}")

    pooled = strict_wr(out[hi_mask]) - strict_wr(out[~hi_mask])
    print(f"\n  Unconditional gap (ignoring volatility): {f(pooled, '+.1f', 'pp')}")
    print("  Compare that against the within-bucket gaps above. If the unconditional gap is")
    print("  clearly bigger, the difference is volatility selection.")


def section_where_it_fires(df, buckets):
    print("\n" + "=" * 96)
    print("4) WHERE THE MODEL FIRES vs WHERE THE SAMPLES ARE")
    print("=" * 96)
    fired = (df["is_model"] == True).to_numpy()  # noqa: E712
    total_f = fired.sum()
    if not total_f:
        print("   No model-fired trades in these files.")
        return
    print(f"{'Bucket':<8}{'% of samples':>15}{'% of model trades':>20}{'over/under':>14}")
    print("-" * 57)
    bvals = buckets.to_numpy()
    for b in [x for x in buckets.cat.categories if (buckets == x).any()]:
        m = bvals == b
        s_share = m.sum() / len(df) * 100
        f_share = (m & fired).sum() / total_f * 100
        print(f"{b:<8}{s_share:>14.1f}%{f_share:>19.1f}%{f_share - s_share:>13.1f}pp")
    print("\n  A model that is really a volatility screen fires far more than its share in the")
    print("  low-volatility buckets (Q1/Q2) and far less in the high ones.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="backtest_runs")
    p.add_argument("--model", default="V40")
    p.add_argument("--cache", default="atr_cache.csv",
                   help="where the per-sample features are stored between runs")
    p.add_argument("--no-fetch", action="store_true", help="use the cache only, download nothing")
    p.add_argument("--vol-col", default="atr_pct", choices=list(FEATURES))
    p.add_argument("--quintiles", type=int, default=5)
    p.add_argument("--conf-split", type=float, default=0.65)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--out", default=None, help="write the per-sample table with features here")
    a = p.parse_args()

    print("=" * 96)
    print(f"VOLATILITY SCREEN CHECK   model={a.model}   dir={a.dir}")
    print("=" * 96)
    df = load_samples(a.dir, a.model)
    df["signal_date"] = df["signal_date"].astype(str).str[:10]

    # ---- features, from cache where possible ----
    need = True
    if os.path.exists(a.cache):
        cache = pd.read_csv(a.cache)
        cache["signal_date"] = cache["signal_date"].astype(str).str[:10]
        keep = ["ticker", "signal_date"] + [c for c in FEATURES if c in cache.columns]
        df = df.merge(cache[keep].drop_duplicates(["ticker", "signal_date"]),
                      on=["ticker", "signal_date"], how="left")
        have = df[list(FEATURES)].notna().all(axis=1).mean() if all(
            c in df.columns for c in FEATURES) else 0.0
        print(f"  cache '{a.cache}': {have:.0%} of rows covered")
        need = have < 0.98
    if need and a.no_fetch:
        print("  --no-fetch set; continuing with whatever the cache had")
    elif need:
        print(f"\nFetching daily bars for {df['ticker'].nunique()} tickers "
              f"({df['signal_date'].min()} -> {df['signal_date'].max()})")
        frames = fetch_features(set(df["ticker"]), df["signal_date"].min(),
                                df["signal_date"].max())
        df = df.drop(columns=[c for c in FEATURES if c in df.columns], errors="ignore")
        df = attach_features(df, frames)
        df[["ticker", "signal_date"] + list(FEATURES)].to_csv(a.cache, index=False)
        print(f"  cached features to '{a.cache}'")

    df = df[df[a.vol_col].notna()].reset_index(drop=True)
    if df.empty:
        sys.exit("No rows have volatility features. Check the cache or the fetch.")
    print(f"\n  analysing {len(df)} samples with features, "
          f"{int((df['is_model'] == True).sum())} of them model-fired")  # noqa: E712

    section_correlations(df, a.rounds, a.boot_seed)
    buckets = section_outcomes_by_vol(df, a.vol_col, a.quintiles, a.min_n)
    section_conditional(df, buckets, a.conf_split, a.rounds, a.boot_seed, a.min_n)
    section_where_it_fires(df, buckets)

    if a.out:
        df.to_csv(a.out, index=False)
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
