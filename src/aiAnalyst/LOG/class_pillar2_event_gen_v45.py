#!/usr/bin/env python3
"""
class_pillar2_event_gen_v45.py - catalyst-event dataset, plus the raw effect measured
BEFORE any model is trained on it.

WHY EVENTS, AND WHY THIS WAY

V39-V44 scored every stock every few days and found nothing that survived out of
sample. That is the panel approach: mostly noise, because on any given day most stocks
have no particular reason to move. Event studies invert it - you only look at stocks
that just had a reason to move - so signal-to-noise is far higher even though the
sample is 1-2% the size.

Post-earnings announcement drift is among the most durable documented anomalies, and
it is strongest exactly where you trade: smaller, high-growth, retail-heavy names.

THE DATA PROBLEM, AND THE WAY AROUND IT

Real earnings dates with EPS surprises need a data vendor. yfinance carries only a few
quarters per ticker, and free APIs cap at 25-250 calls/day against 347 tickers - weeks
of work just to assemble the calendar.

The PEAD literature uses two signals: standardized unexpected earnings (needs EPS) and
the ABNORMAL ANNOUNCEMENT RETURN - the market's own reaction to the news. Chan,
Jegadeesh & Lakonishok document drift on both. The second needs only price and volume,
which is already cached for 20 years. So the event is detected from the reaction
itself: an outsized move on outsized volume.

That catches more than earnings - guidance, M&A, FDA decisions, analyst days. For a
trading tool that is arguably better, since all of them are catalysts. It does mean
this measures POST-CATALYST DRIFT rather than PEAD strictly, and the script reports the
spacing between consecutive events so you can see what share look like scheduled
quarterly reports.

NO LOOK-AHEAD. The event is detected from day D's own close and volume, both known at
D's close. The volatility and volume baselines are computed THROUGH D-1 (shifted by one
bar). Entry is at D+1's open. Forward returns run from that open.

THE RAW EFFECT COMES FIRST

Before any model: sort events by their announcement-day abnormal return into quintiles
and look at mean forward abnormal return. If drift exists, the top quintile keeps
rising and the bottom keeps falling. If the quintiles are flat, there is nothing to
model and you have saved yourself another five versions.

USAGE
  python class_pillar2_event_gen_v45.py
  python class_pillar2_event_gen_v45.py --sigma 2.5 --min-move 0.05 --start 2005-01-01
  python class_pillar2_event_gen_v45.py --universe deploy      # your 62 names only
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

import pillar2_v43_features as F
from pillar2_v43_universe import BENCHMARK, DEPLOY_UNIVERSE, load_training_universe

FWD_BARS = (1, 5, 10, 20, 60)
MAX_FWD = max(FWD_BARS)
MIN_PAST_BARS = 260          # ~1 year: enough for the 200-EMA and 250-bar momentum
CACHE_DIR = "price_cache_v43"


# =============================================================================
# PRICES  (shares the V43/V44 cache - no new downloads)
# =============================================================================
class PriceCache:
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
            except Exception:
                df = None
        if df is not None and getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        self.mem[ticker] = df
        return df


# =============================================================================
# EVENT DETECTION
# =============================================================================
def detect_events(df, sigma_mult, vol_mult, min_move, min_gap_bars):
    """
    An event is a day whose move is large relative to the stock's OWN recent volatility
    and which trades on unusual volume. Both baselines are shifted one bar, so they use
    only what was known before day D - nothing about day D leaks into its own detection.

    Returns integer positions of event days.
    """
    close = df["Close"]
    ret = close.pct_change()
    sigma = ret.rolling(20).std().shift(1)
    avgvol = df["Volume"].rolling(20).mean().shift(1)

    big = ret.abs() >= (sigma_mult * sigma)
    heavy = df["Volume"] >= (vol_mult * avgvol)
    material = ret.abs() >= min_move
    # copy=True: pandas can hand back a read-only view, and the burn-in below writes to it
    flags = np.asarray((big & heavy & material).fillna(False).to_numpy(), dtype=bool).copy()
    if len(flags) > MIN_PAST_BARS:
        flags[:MIN_PAST_BARS] = False
    else:
        return []

    # A catalyst often moves price for two or three days. Keep the first, then enforce
    # a gap so one piece of news is not counted as several events.
    positions, last = [], -10_000
    for i in np.flatnonzero(flags):
        if i - last >= min_gap_bars:
            positions.append(int(i))
            last = i
    return positions


def forward_abnormal(df, bench, pos, bars=FWD_BARS):
    """
    Market-adjusted forward returns, entered at D+1's OPEN.

    Cohort demeaning (V44's approach) does not work here: events are sparse, so a given
    day has too few of them to form a cross-section. Subtracting the benchmark over the
    same window is the standard event-study adjustment.
    """
    if pos + 1 >= len(df):
        return None
    entry = float(df["Open"].iloc[pos + 1])
    if entry <= 0:
        return None
    try:
        b_entry = float(bench["Open"].asof(df.index[pos + 1]))
    except Exception:
        return None
    if not np.isfinite(b_entry) or b_entry <= 0:
        return None

    out = {}
    for h in bars:
        end = pos + h
        if end >= len(df):
            return None
        r = (float(df["Close"].iloc[end]) - entry) / entry * 100.0
        try:
            b_end = float(bench["Close"].asof(df.index[end]))
            br = (b_end - b_entry) / b_entry * 100.0
        except Exception:
            br = np.nan
        out[f"fwd_{h}"] = r
        out[f"car_{h}"] = r - br if np.isfinite(br) else np.nan
    out["_entry_open"] = entry
    return out


# =============================================================================
# BUILD
# =============================================================================
def build(cfg):
    tickers = (DEPLOY_UNIVERSE if cfg.universe == "deploy"
               else load_training_universe(include_deploy=True))
    print("=" * 84)
    print("PILLAR 2 V45 - CATALYST EVENT DATASET")
    print("=" * 84)
    print(f"  universe:  {cfg.universe} ({len(tickers)} tickers)")
    print(f"  window:    {cfg.start} -> {cfg.end}")
    print(f"  event:     |return| >= {cfg.sigma}x its own 20d sigma")
    print(f"             AND volume >= {cfg.vol_mult}x its 20d average")
    print(f"             AND |return| >= {cfg.min_move:.0%}, min {cfg.min_gap} bars apart")
    print(f"  entry:     next bar's open | forward bars {FWD_BARS}")
    print()

    cache = PriceCache(cfg.cache_dir, cfg.start, cfg.end, allow_fetch=not cfg.no_fetch)
    bench = cache.get(BENCHMARK)
    if bench is None or bench.empty:
        sys.exit(f"Could not load benchmark {BENCHMARK} from '{cfg.cache_dir}'. "
                 f"Run without --no-fetch once, or point --cache-dir at the V43 cache.")
    bench_ret = bench["Close"].pct_change()

    rows, n_usable, spacing_all, skipped = [], 0, [], 0
    for i, t in enumerate(tickers, 1):
        df = cache.get(t)
        if df is None or len(df) < MIN_PAST_BARS + MAX_FWD + 5:
            continue
        df = df[(df.index >= pd.Timestamp(cfg.start)) & (df.index <= pd.Timestamp(cfg.end))]
        if len(df) < MIN_PAST_BARS + MAX_FWD + 5:
            continue

        panel = F.compute_panel(df, bench_close=bench["Close"])
        evs = detect_events(df, cfg.sigma, cfg.vol_mult, cfg.min_move, cfg.min_gap)
        if len(evs) > 1:
            spacing_all.extend(np.diff(evs).tolist())
        n_usable += 1

        close = df["Close"]
        ret = close.pct_change()
        sigma_base = ret.rolling(20).std().shift(1)
        vol_base = df["Volume"].rolling(20).mean().shift(1)

        for k, pos in enumerate(evs):
            fwd = forward_abnormal(df, bench, pos)
            if fwd is None:
                skipped += 1
                continue

            day_ret = float(ret.iloc[pos]) * 100.0
            try:
                b_day = float(bench_ret.asof(df.index[pos])) * 100.0
            except Exception:
                b_day = np.nan
            prev_close = float(close.iloc[pos - 1])
            open_px = float(df["Open"].iloc[pos])
            sig = float(sigma_base.iloc[pos])
            vb = float(vol_base.iloc[pos])

            r = panel.iloc[pos].to_dict()
            r.update(
                ticker=t,
                event_date=df.index[pos].strftime("%Y-%m-%d"),
                # --- the event itself: everything known at day D's close ---
                event_ret=day_ret,
                event_car=day_ret - b_day if np.isfinite(b_day) else np.nan,
                event_gap=(open_px - prev_close) / prev_close * 100.0 if prev_close else np.nan,
                event_intraday=(float(close.iloc[pos]) - open_px) / open_px * 100.0 if open_px else np.nan,
                event_vol_ratio=float(df["Volume"].iloc[pos]) / vb if vb else np.nan,
                event_sigma_mult=abs(day_ret / 100.0) / sig if sig else np.nan,
                # --- context leading into the event ---
                pre_drift_20=float(close.iloc[pos] / close.iloc[pos - 20] - 1) * 100.0,
                pre_drift_60=float(close.iloc[pos] / close.iloc[pos - 60] - 1) * 100.0,
                bars_since_prev_event=float(pos - evs[k - 1]) if k > 0 else np.nan,
                entry_open=fwd.pop("_entry_open"),
                **fwd)
            rows.append(r)

        if i % 50 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {n_usable} tickers usable, {len(rows):,} events")

    if not rows:
        sys.exit("No events detected. Loosen --sigma / --vol-mult / --min-move.")
    data = pd.DataFrame(rows)
    data["_date"] = pd.to_datetime(data["event_date"])
    data["direction"] = np.where(data["event_car"] >= 0, "positive", "negative")

    print("\n" + "=" * 84)
    print("EVENTS FOUND")
    print("=" * 84)
    n_tick = data["ticker"].nunique()
    n_year = data["_date"].dt.year.nunique()
    print(f"  {len(data):,} events | {n_tick} tickers | {n_year} years")
    print(f"  events per ticker per year: {len(data) / max(n_tick, 1) / max(n_year, 1):.1f}")
    print(f"  positive reactions: {(data['direction'] == 'positive').mean():.1%}")
    print(f"  skipped (no forward window): {skipped:,}")
    if spacing_all:
        sp = pd.Series(spacing_all)
        near_q = float(((sp >= 50) & (sp <= 76)).mean())
        print(f"\n  spacing between consecutive events (trading days): "
              f"median {sp.median():.0f}, mean {sp.mean():.0f}")
        print(f"  {near_q:.0%} fall in the 50-76 bar band, i.e. roughly quarterly - a rough")
        print(f"  read on how many detected events are scheduled earnings reports.")
    return data


# =============================================================================
# THE RAW EFFECT  (before any model)
# =============================================================================
def month_cluster_t(top_vals, top_dates, bot_vals, bot_dates):
    """
    t-stat on the Q5-minus-Q1 spread, clustering by calendar month. Events in the same
    month share the market's move, so treating them as independent overstates the
    evidence.

    The spread is computed WITHIN each month (mean of top minus mean of bottom, using
    only months that contain both), then t-tested across months. Pooling the two groups
    and averaging instead would weight by how many of each fell in that month - and
    those counts differ ~80% of the time, which shrinks the estimate and can flip its
    sign when the effect is weak.
    """
    a = pd.DataFrame({"v": np.asarray(top_vals, dtype=float),
                      "m": pd.to_datetime(pd.Series(np.asarray(top_dates))).dt.to_period("M")}).dropna()
    b = pd.DataFrame({"v": np.asarray(bot_vals, dtype=float),
                      "m": pd.to_datetime(pd.Series(np.asarray(bot_dates))).dt.to_period("M")}).dropna()
    if a.empty or b.empty:
        return np.nan, 0
    s = (a.groupby("m")["v"].mean() - b.groupby("m")["v"].mean()).dropna()
    if len(s) < 5 or not np.isfinite(s.std(ddof=1)) or s.std(ddof=1) == 0:
        return np.nan, len(s)
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s)


def measure_drift(data, q=5):
    print("\n" + "=" * 84)
    print("THE RAW EFFECT - quintiles by announcement-day abnormal return")
    print("=" * 84)
    print("  If drift exists, Q5 keeps rising and Q1 keeps falling AFTER the event.")
    print("  Entry is the next open, so none of this includes the event-day move itself.\n")

    d = data.dropna(subset=["event_car"]).copy()
    if len(d) < q * 20:
        print(f"  Only {len(d)} events - too few to quintile. Loosen the thresholds.")
        return
    d["bucket"] = pd.qcut(d["event_car"].rank(method="first"), q, labels=False)

    hdr = f"  {'quintile':<10}{'n':>7}{'event CAR':>12}"
    for h in FWD_BARS:
        hdr += f"{'CAR+' + str(h):>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in range(q):
        g = d[d["bucket"] == b]
        line = f"  Q{b + 1:<9}{len(g):>7}{g['event_car'].mean():>+11.2f}%"
        for h in FWD_BARS:
            line += f"{g[f'car_{h}'].mean():>+10.2f}%"
        print(line)

    print("\n  Q5 minus Q1, with t-stats clustered by month:")
    print(f"    {'horizon':>8}{'spread':>10}{'t':>8}{'months':>8}   reading")
    top, bot = d[d["bucket"] == q - 1], d[d["bucket"] == 0]
    for h in FWD_BARS:
        spread = top[f"car_{h}"].mean() - bot[f"car_{h}"].mean()
        tt, nm = month_cluster_t(top[f"car_{h}"], top["_date"],
                                 bot[f"car_{h}"], bot["_date"])
        strong = np.isfinite(tt) and abs(tt) > 2
        reading = ("drift - winners keep winning" if spread > 0 and strong
                   else "reversal - winners give it back" if spread < 0 and strong
                   else "flat")
        t_txt = f"{tt:>8.2f}" if np.isfinite(tt) else f"{'n/a':>8}"
        print(f"    {h:>8}{spread:>+9.2f}%{t_txt}{nm:>8}   {reading}")

    # A long-only reader cannot trade the spread, and the spread also cancels the
    # universe's own upward drift (these are companies still listed in 2026, so they
    # outperformed on average). Both numbers are needed to read the table honestly.
    print("\n  BASELINE: mean CAR across ALL events - the drift a long-only trade gets")
    print("  for free from this universe, before any selection. A survivorship-biased")
    print("  ticker list makes this positive, so Q5 must beat THIS, not zero.")
    print(f"    {'horizon':>8}{'all events':>13}{'Q5':>10}{'Q5 - all':>11}{'t':>8}")
    for h in FWD_BARS:
        base = d[f"car_{h}"].mean()
        tq = top[f"car_{h}"].mean()
        tt, _ = month_cluster_t(top[f"car_{h}"], top["_date"], d[f"car_{h}"], d["_date"])
        t_txt = f"{tt:>8.2f}" if np.isfinite(tt) else f"{'n/a':>8}"
        print(f"    {h:>8}{base:>+12.2f}%{tq:>+9.2f}%{tq - base:>+10.2f}%{t_txt}")

    print("\n  BY DIRECTION (positive and negative reactions drift differently)")
    print(f"    {'group':<12}{'n':>7}" + "".join(f"{'CAR+' + str(h):>11}" for h in FWD_BARS))
    for name, g in d.groupby("direction"):
        line = f"    {name:<12}{len(g):>7}"
        for h in FWD_BARS:
            line += f"{g[f'car_{h}'].mean():>+10.2f}%"
        print(line)

    print("\n  BY YEAR at 20 bars (a real effect is not one good year)")
    for label, fn in [("Q5 - Q1 ", lambda g: (g[g["bucket"] == q - 1]["car_20"].mean()
                                              - g[g["bucket"] == 0]["car_20"].mean())),
                      ("Q5 - all", lambda g: (g[g["bucket"] == q - 1]["car_20"].mean()
                                              - g["car_20"].mean()))]:
        parts, pos_years, n_years = [], 0, 0
        for y, g in d.groupby(d["_date"].dt.year):
            v = fn(g)
            if not np.isfinite(v):
                continue
            parts.append(f"{int(y)}: {v:+.1f}%")
            pos_years += int(v > 0)
            n_years += 1
        if not parts:
            continue
        print(f"    [{label}]")
        for i in range(0, len(parts), 6):
            print("      " + "  ".join(parts[i:i + 6]))
        print(f"      positive in {pos_years}/{n_years} years "
              f"({pos_years / max(n_years, 1):.0%})")


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=datetime.date.today().isoformat())
    p.add_argument("--universe", default="all", choices=["all", "deploy"])
    p.add_argument("--sigma", type=float, default=3.0,
                   help="move as a multiple of the stock's own 20d sigma")
    p.add_argument("--vol-mult", type=float, default=2.0,
                   help="volume as a multiple of its 20d average")
    p.add_argument("--min-move", type=float, default=0.04,
                   help="minimum absolute move, so a quiet stock's 3-sigma day still counts as news")
    p.add_argument("--min-gap", type=int, default=25,
                   help="trading days between events on one ticker")
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    data = build(a)
    measure_drift(data, a.quantiles)

    out = a.out or f"events_v45_{a.universe}_{pd.Timestamp(a.start):%Y%m%d}.csv.gz"
    data.to_csv(out, index=False, compression="gzip")
    with open(out.replace(".csv.gz", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n_events": int(len(data)), "universe": a.universe,
                   "sigma": a.sigma, "vol_mult": a.vol_mult, "min_move": a.min_move,
                   "min_gap": a.min_gap, "fwd_bars": list(FWD_BARS),
                   "benchmark": BENCHMARK, "feature_version": F.FEATURE_VERSION,
                   "start": a.start, "end": a.end,
                   "built": datetime.datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"\n  wrote {out}")
    print(f"  wrote {out.replace('.csv.gz', '_meta.json')}")
    print("\n  Read the quintile table before training anything. If Q5 minus Q1 is flat")
    print("  at every horizon, the effect is not there and no model will find it.")


if __name__ == "__main__":
    main()
