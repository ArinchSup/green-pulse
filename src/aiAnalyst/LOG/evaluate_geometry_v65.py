#!/usr/bin/env python3
"""
evaluate_geometry_v65.py - the win rate is a DIAL, not a discovery.

THE POINT OF THIS FILE

"A 55% win rate, just a little better than a coin flip" sounds like a modest
ask. It is not an ask at all: the win rate of a triple-barrier trade is set by
where you put the barriers, and you can have almost any number you like without
a model, without a feature, without machine learning of any kind.

    stop = SL_TP_RATIO x target

At SL_TP_RATIO 0.60 - the deployed setting - the stop is closer than the target,
so most trades lose: R:R 1.67, break-even 37.5%, realised about 42%. Widen the
stop past the target and the win rate climbs through 55%, 65%, 75%. Nothing has
improved. The trades that used to be small losses are now small wins, and the
ones that used to be wins are now bigger losses.

This file sweeps that dial with NO MODEL AT ALL - every candidate entry, bought
blind - and prints win rate and expectancy side by side. Wherever the win rate
crosses 55%, that is the number a coin flip gets you at that geometry. A model
has to beat THAT, not 50%.

WHAT TO TAKE FROM IT

The honest target is not a win rate. It is expectancy above the same geometry's
blind baseline, net of costs. Read the `edge needed` column: that is how much
expectancy a model must add for its picks to be worth making, once round-trip
friction is paid at that geometry's stop distance.

USAGE
  python evaluate_geometry_v65.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2
import class_ai_entry_v64 as E64
from trade_config import HORIZON_CONFIGS

# =============================================================================
# CONFIG
# =============================================================================
PRICE_CACHE = P2.PRICE_CACHE
HORIZON = "MID"
STEP = 5
MIN_HISTORY = 260

# stop = RATIO x target. Below 1.0 the stop is tighter than the target and most
# trades lose; above 1.0 most trades win. 0.60 is deployed.
RATIOS = (0.60, 1.00, 1.60, 2.50)

# How far the target sits, as a fraction of the stock's own expected move over
# the holding window. Moving the stop alone is not enough to lift the win rate:
# the target stays where it was, so the trades that stop losing simply TIME OUT
# instead - on the first run the win rate plateaued at 48.7% with 48.4% of
# trades flat. A CLOSER target is what converts those into wins.
VOL_KS = (0.20, 0.35, 0.50)
VOL_K = 0.50            # deployed
VOL_MIN_TP_PCT = 0.08
VOL_MAX_TP_PCT = 0.40

FRICTION_BPS = 10.0        # round-trip cost as basis points of position value


# =============================================================================
# ONE PASS, EVERY GEOMETRY
# =============================================================================
def sweep_geometry(tickers, cache_dir=None, horizon=HORIZON, step=STEP,
                   ratios=RATIOS, vol_ks=VOL_KS, friction_bps=FRICTION_BPS,
                   verbose=True):
    """
    Walk the prices once and grade every candidate entry under every ratio.

    Doing it in one pass matters: the ENTRIES are identical across ratios, so
    the comparison is not confounded by a different sample. Only the barriers
    move.
    """
    cache_dir = cache_dir or PRICE_CACHE
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    ratios, vol_ks = list(ratios), list(vol_ks)
    grid = [(k, r) for k in vol_ks for r in ratios]
    acc = {g: {"wins": 0, "losses": 0, "flats": 0, "R": []} for g in grid}
    n_entries = 0

    for i, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache_dir)
        if df is None or len(df) < MIN_HISTORY + n_bars + 2:
            continue
        c = df["Close"].to_numpy(float)
        hi = df["High"].to_numpy(float)
        lo = df["Low"].to_numpy(float)
        atr = E64._atr(df, 14).to_numpy(float)

        for pos in range(MIN_HISTORY, len(df) - n_bars - 1, step):
            entry, a = c[pos], atr[pos]
            if not (np.isfinite(entry) and np.isfinite(a)) or entry <= 0 or a <= 0:
                continue
            # the deployed target rule: a fraction of the stock's own expected
            # move over the holding window, clamped at both ends
            sigma_h = (a / entry) * np.sqrt(n_bars)
            n_entries += 1
            H = hi[pos + 1:pos + 1 + n_bars]
            L = lo[pos + 1:pos + 1 + n_bars]
            C = c[pos + 1:pos + 1 + n_bars]
            if len(C) < 3:
                continue
            for k, r in grid:
                tp = float(np.clip(k * sigma_h, VOL_MIN_TP_PCT,
                                   VOL_MAX_TP_PCT))
                target = entry * (1 + tp)
                stop = entry * (1 - tp * r)
                risk = entry - stop
                if risk <= 0:
                    continue
                g = (k, r)
                ht = np.flatnonzero(H >= target)
                hs = np.flatnonzero(L <= stop)
                ti = ht[0] if ht.size else np.inf
                si = hs[0] if hs.size else np.inf
                if ti == np.inf and si == np.inf:
                    acc[g]["flats"] += 1
                    acc[g]["R"].append((C[-1] - entry) / risk)
                elif si <= ti:                       # ties to the stop
                    acc[g]["losses"] += 1
                    acc[g]["R"].append(-1.0)
                else:
                    acc[g]["wins"] += 1
                    acc[g]["R"].append((target - entry) / risk)
        if verbose and (i % 20 == 0 or i == len(tickers)):
            print(f"    [{i}/{len(tickers)}] {n_entries:,} entries graded "
                  f"under {len(grid)} geometries")

    rows = []
    for (k, r) in grid:
        a = acc[(k, r)]
        n = a["wins"] + a["losses"] + a["flats"]
        if n == 0:
            continue
        R = np.asarray(a["R"], float)
        rr = 1.0 / r
        rows.append({"vol_k": k, "ratio": r, "rr": rr,
                     "breakeven_wr": 1.0 / (1.0 + rr),
                     "n": int(n),
                     "win_rate": a["wins"] / n,
                     "loss_rate": a["losses"] / n,
                     "flat_rate": a["flats"] / n,
                     "expectancy_R": float(R.mean()),
                     "sd_R": float(R.std(ddof=1))})
    return pd.DataFrame(rows).sort_values("win_rate"), n_entries


def report(df, friction_bps=FRICTION_BPS, target_wr=0.55, out=sys.stdout):
    def w(*a):
        print(*a, file=out)

    w("\n" + "=" * 96)
    w("  WIN RATE WITH NO MODEL AT ALL - every candidate bought blind")
    w("=" * 96)
    w(f"  {'target k':>9s} {'stop/tgt':>9s} {'R:R':>6s} {'breakeven':>10s} "
      f"{'WIN RATE':>9s} {'loss':>7s} {'flat':>7s} "
      f"{'expectancy':>11s} {'sd':>6s}")
    for _, q in df.iterrows():
        dep = (abs(q["ratio"] - 0.60) < 1e-9 and abs(q["vol_k"] - 0.50) < 1e-9)
        w(f"  {q['vol_k']:9.2f} {q['ratio']:9.2f} {q['rr']:6.2f} "
          f"{q['breakeven_wr']:10.1%} "
          f"{q['win_rate']:9.1%} {q['loss_rate']:7.1%} {q['flat_rate']:7.1%} "
          f"{q['expectancy_R']:+11.3f} {q['sd_R']:6.2f}"
          + ("   <-- deployed" if dep else ""))

    hit = df[df["win_rate"] >= target_wr]
    w("")
    if hit.empty:
        w(f"  No geometry in this sweep reaches a {target_wr:.0%} blind win "
          f"rate; widen the ratios.")
    else:
        q = hit.iloc[0]
        w(f"  A {target_wr:.0%} WIN RATE COSTS NOTHING. At target k "
          f"{q['vol_k']:.2f}, stop/target {q['ratio']:.2f}, every candidate")
        w(f"  bought blind wins {q['win_rate']:.1%} of the time - no model, no "
          f"features, no training.")
        w(f"  Break-even at that geometry is {q['breakeven_wr']:.1%}, so "
          f"{q['win_rate']:.1%} is "
          + ("ABOVE" if q["win_rate"] > q["breakeven_wr"] else "BELOW")
          + " it by "
          f"{abs(q['win_rate'] - q['breakeven_wr']) * 100:.1f}pp.")

    best = df.loc[df["expectancy_R"].idxmax()]
    w(f"\n  And the win rate is not where the money is. Expectancy peaks at "
      f"k {best['vol_k']:.2f} / ratio {best['ratio']:.2f}")
    w(f"  ({best['expectancy_R']:+.3f}R at a {best['win_rate']:.1%} win rate), "
      f"not at the highest win rate")
    w(f"  ({df.loc[df['win_rate'].idxmax(), 'win_rate']:.1%} at "
      f"{df.loc[df['win_rate'].idxmax(), 'expectancy_R']:+.3f}R).")

    w(f"\n  WHAT A MODEL WOULD HAVE TO ADD, at {friction_bps:.0f}bp round-trip")
    w(f"  {'target k':>9s} {'stop/tgt':>9s} {'blind wr':>9s} {'blind R':>8s} "
      f"{'friction R':>11s}")
    for _, q in df.iterrows():
        # 1R is the stop distance as a share of price, so a wider stop makes a
        # fixed basis-point cost a smaller share of R
        stop_pct = np.clip(q["vol_k"] * 0.20, VOL_MIN_TP_PCT,
                           VOL_MAX_TP_PCT) * q["ratio"]
        fr = (friction_bps / 10000.0) / max(stop_pct, 1e-6)
        w(f"  {q['vol_k']:9.2f} {q['ratio']:9.2f} {q['win_rate']:9.1%} "
          f"{q['expectancy_R']:+8.3f} {fr:11.4f}")
    w(f"\n  'edge needed' is expectancy ABOVE the blind baseline, just to cover "
      f"costs.")
    w(f"  Measured entry-timing edge in V64, after removing each held-out "
      f"half's two best")
    w(f"  years: -0.002R median across splits. That is the number to compare "
      f"against this column.")


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_geometry_v65(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
                     ratios=RATIOS, vol_ks=VOL_KS, friction_bps=FRICTION_BPS,
                     target_wr=0.55, tickers=None, out="geometry_v65.json",
                     verbose=True):
    print("=" * 96)
    print("V65 - IS A 55% WIN RATE A DISCOVERY OR A SETTING?")
    print("=" * 96)
    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    names = [t for t in (tickers or available) if t in available]
    cfg = HORIZON_CONFIGS[horizon]
    print(f"  {len(names)} tickers | horizon {horizon} "
          f"({cfg['eval_days']}d, {cfg['lookahead_bars']} bars) | step {step}")
    print(f"  grading every entry under {len(ratios) * len(vol_ks)} "
          f"geometries in one pass")

    df, n = sweep_geometry(names, price_cache, horizon, step, ratios, vol_ks,
                           friction_bps, verbose)
    if df.empty:
        raise RuntimeError("no entries graded")
    report(df, friction_bps, target_wr)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(df.to_dict("records"), f, indent=2, default=float)
    print(f"\n  wrote {out}")
    return df


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--horizon", default=HORIZON, choices=list(HORIZON_CONFIGS))
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--friction-bps", type=float, default=FRICTION_BPS)
    ap.add_argument("--target-wr", type=float, default=0.55)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--out", default="geometry_v65.json")
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_geometry_v65(c.price_cache, c.horizon, c.step, RATIOS, VOL_KS,
                     c.friction_bps, c.target_wr, c.tickers, c.out)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE  = PRICE_CACHE
    RUN_HORIZON      = HORIZON       # "SHORT" | "MID" | "LONG"
    RUN_STEP         = STEP
    RUN_RATIOS       = RATIOS        # stop = ratio x target
    RUN_VOL_KS       = VOL_KS        # target = k x the stock's expected move
    RUN_FRICTION_BPS = FRICTION_BPS  # round-trip cost, bp of position
    RUN_TARGET_WR    = 0.55          # the win rate you want to reach
    RUN_TICKERS      = None
    RUN_OUT          = "geometry_v65.json"
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_geometry_v65(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
                         step=RUN_STEP, ratios=RUN_RATIOS, vol_ks=RUN_VOL_KS,
                         friction_bps=RUN_FRICTION_BPS,
                         target_wr=RUN_TARGET_WR, tickers=RUN_TICKERS,
                         out=RUN_OUT)
