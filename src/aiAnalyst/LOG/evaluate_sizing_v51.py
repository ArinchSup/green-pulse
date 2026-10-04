#!/usr/bin/env python3
"""
evaluate_sizing_v51.py - does the risk layer's position sizing earn its place?

THE EXPERIMENT

Entries are RANDOM and held fixed. Only the sizing rule varies. Because the
entry rule, the exit levels and the exit dates are all identical across rules,
every difference in outcome is attributable to sizing and to nothing else.

That design is the reason this is the first test worth running. The pipeline's
direction comes from the news pillar, which has an unavoidable look-ahead
problem: an LLM's training data contains the outcomes it is being asked to
predict. Any evaluation that depends on entry quality inherits that
contamination. This one does not depend on entry quality at all - random entries
are the point. It isolates the one thing Pillar 2 can defensibly claim to do.

WHAT IS COMPARED

  equal_notional     constant dollars per position, the naive baseline
  equal_weight       constant share of CURRENT equity, so it compounds
  flat_risk_capped   2% risk budget / stop distance, capped at 20%
                     - the pre-fix behaviour, kept as the thing to beat
  tier_ceiling       2% budget, ceiling scaled by the calibrated drawdown tier
                     - what the pipeline does now
  inverse_vol        position proportional to 1 / forecast volatility
                     - a one-line heuristic that needs no model at all
  random_size        position drawn uniformly from the same range
                     - the control. If tier_ceiling cannot beat a random number
                       from the same range, the calibration is decoration.

inverse_vol is the honest benchmark and random_size is the null. Beating
flat_risk_capped alone proves little; beating a heuristic that needs no
calibration is the claim worth making.

HOW THE COMPARISON IS PAIRED

Each path is one random entry set drawn from one random contiguous window, so a
path is a plausible portfolio history rather than trades scattered across twenty
years. Every rule runs over that SAME set, and results are compared as paired
differences within a path. Path-to-path variance here is far larger than the
effect being measured, so pairing is not a refinement - comparing marginal
distributions would need thousands of paths to see what pairing sees in a couple
of hundred.

WHAT IS NOT MEASURED, AND WHY

  Kelly sizing is absent: Kelly needs an expected edge, and this project
  established there is none to plug in.

  The EDGAR flags (HIGH_DILUTION, SHORT_RUNWAY) are excluded from tier_ceiling.
  Reconstructing them point-in-time for every historical date is not implemented,
  and applying today's filings to a 2015 date would be look-ahead. The two flags
  computable from price alone (LOW_LIQUIDITY, VOL_EXPANDING) are applied, so what
  is tested is the price-derived part of the ceiling logic.

SURVIVORSHIP

The price cache holds currently-listed tickers. V49 found only 41% of the 2011
cohort still filing ten years later, and the missing names are the ones that went
to zero. Every absolute number below is therefore optimistic, and for a RISK
model the bias runs the unsafe way: drawdowns are under-represented, so sizing
comes out too large. The saving grace of this design is that the bias reaches all
six rules through the same trades, so the RANKING survives it even though the
levels do not. Report the ranking; caveat the levels.

USAGE
  python evaluate_sizing_v51.py --paths 200
  python evaluate_sizing_v51.py --paths 400 --trades-per-path 120 --equity 25000
  python evaluate_sizing_v51.py --window-years 5 --max-concurrent 6
  python evaluate_sizing_v51.py --quick          # 40 paths, subsampled
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk as P2
from trade_config import HORIZON_CONFIGS, compute_levels

HORIZON = "MID"
RUIN_LEVEL = 0.50           # equity below half the start counts as ruin
MAX_POSITION_PCT = 20.0
BASE_RISK = 0.02
MIN_POSITION_PCT = 2.0      # floor for the heuristic and control rules

# Ceiling multipliers keyed on the calibrated drawdown probability - the same
# table class_ai_pipeline_v2.py uses.
RISK_TIERS = [(0.10, 1.00, "low"), (0.18, 0.75, "moderate"),
              (0.25, 0.50, "elevated")]
IVOL_REF_VOL = 30.0         # inverse_vol is normalised so 30% vol -> 12%
IVOL_REF_PCT = 12.0


EXCESSIVE_FLOOR = 0.25      # tier_size keeps trading the worst names, tiny


def tier_mult(p, veto=True):
    """
    Missing probability must not mean full size - that is how a risk model
    becomes decoration.

    veto=True is what the pipeline does: above MAX_DRAWDOWN_PROB the trade is
    refused. veto=False keeps the trade at EXCESSIVE_FLOOR of the ceiling, so the
    two effects can be measured apart. Without that split the comparison confuses
    "scaling by the tier helps" with "refusing the worst names helps", and those
    are different claims with different implications - the second one needs no
    calibrated probability, only a threshold.
    """
    if p is None or not np.isfinite(p):
        return 0.50
    for bound, mult, _ in RISK_TIERS:
        if p < bound:
            return mult
    return 0.0 if veto else EXCESSIVE_FLOOR


# =============================================================================
# TRADE TABLE
# =============================================================================
def build_trades(tickers, cache, calib, every, start, end, verbose=True):
    """
    Every candidate trade, with its exit resolved.

    Exits do not depend on the sizing rule - levels come from price and ATR only
    - so they are computed once and reused by all six rules. That is what makes
    the comparison exactly paired rather than approximately paired.

    Entry is the NEXT bar's open after the signal, never the signal bar's close.
    When one bar's range covers both stop and target, the STOP is taken: the bar
    does not say which came first, and assuming the good one is how backtests
    flatter themselves.
    """
    bars = HORIZON_CONFIGS[HORIZON]["lookahead_bars"]
    rows, closes = [], {}
    for n, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache)
        if df is None or len(df) < P2.MIN_BARS + bars + 2:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        closes[t] = df["Close"].astype("float32")

        o = df["Open"].to_numpy(float)
        h = df["High"].to_numpy(float)
        lw = df["Low"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        v = (df["Volume"].to_numpy(float) if "Volume" in df
             else np.full(len(df), np.nan))
        idx = df.index

        cs = pd.Series(c, index=idx)
        prev = np.r_[np.nan, c[:-1]]
        tr = np.maximum.reduce([h - lw, np.abs(h - prev), np.abs(lw - prev)])
        atr = pd.Series(tr, index=idx).rolling(14).mean().to_numpy()
        ret = cs.pct_change()
        vol60 = (ret.rolling(60).std() * np.sqrt(252) * 100).to_numpy()
        vol250 = (ret.rolling(250).std() * np.sqrt(252) * 100).to_numpy()
        dv20 = (cs * pd.Series(v, index=idx)).rolling(20).mean().to_numpy()

        for i in range(P2.MIN_BARS, len(df) - bars - 1, every):
            if start is not None and idx[i] < start:
                continue
            if end is not None and idx[i] > end:
                break
            if not (np.isfinite(atr[i]) and np.isfinite(vol60[i]) and vol60[i] > 0):
                continue
            entry = o[i + 1]
            if not np.isfinite(entry) or entry <= 0:
                continue
            lv = compute_levels(entry, atr[i], HORIZON)
            stop, target = lv.get("stop"), lv.get("target")
            if not stop or not target or stop <= 0 or stop >= entry:
                continue

            a, b = i + 1, min(i + 1 + bars, len(df))
            hs = np.flatnonzero(lw[a:b] <= stop)
            ht = np.flatnonzero(h[a:b] >= target)
            fs = hs[0] if hs.size else np.inf
            ft = ht[0] if ht.size else np.inf
            if np.isfinite(fs) and fs <= ft:              # stop-first on ties
                k, px, why = int(fs), stop, "stop"
            elif np.isfinite(ft):
                k, px, why = int(ft), target, "target"
            else:
                k, px, why = b - a - 1, c[b - 1], "expired"

            vr = (vol60[i] / vol250[i] if np.isfinite(vol250[i]) and vol250[i] > 0
                  else np.nan)
            rows.append((t, idx[i], idx[a], entry, stop, target, idx[a + k], px,
                         why, px / entry - 1.0, vol60[i],
                         P2.forecast_vol(vol60[i], calib),
                         P2.prob_drawdown(vol60[i], calib), lv["stop_pct"],
                         dv20[i], vr))
        if verbose and n % 50 == 0:
            print(f"    [{n}/{len(tickers)}] {len(rows):,} candidate trades")

    trades = pd.DataFrame(rows, columns=[
        "ticker", "signal_date", "entry_date", "entry", "stop", "target",
        "exit_date", "exit", "reason", "ret", "vol60", "fvol", "p_dd",
        "stop_pct", "dv20", "vol_ratio"])
    # Forward-filled wide close matrix for mark-to-market. ffill covers calendar
    # gaps between tickers; it only ever applies inside a position's own life,
    # since exits use the ticker's own bars.
    close = pd.DataFrame(closes).sort_index().ffill()
    return trades, close


# =============================================================================
# SIZING RULES   f(row, equity, equity0, rng) -> target % of equity
# =============================================================================
def _soft_flag_mult(dv20, vol_ratio):
    """The two pipeline flags reconstructible from price alone."""
    m = 1.0
    if np.isfinite(dv20) and dv20 < 2_000_000:
        m *= 0.80                                    # LOW_LIQUIDITY
    if np.isfinite(vol_ratio) and vol_ratio > 1.5:
        m *= 0.80                                    # VOL_EXPANDING
    return max(m, 0.50)


def s_equal_notional(r, eq, eq0, rng):
    return MAX_POSITION_PCT * eq0 / eq if eq > 0 else 0.0


def s_equal_weight(r, eq, eq0, rng):
    return MAX_POSITION_PCT


def s_flat_risk_capped(r, eq, eq0, rng):
    return min(BASE_RISK / (r["stop_pct"] / 100.0) * 100.0, MAX_POSITION_PCT)


def _tier(r, veto):
    m = tier_mult(r["p_dd"], veto) * _soft_flag_mult(r["dv20"], r["vol_ratio"])
    if m <= 0:
        return 0.0
    return min(BASE_RISK / (r["stop_pct"] / 100.0) * 100.0,
               MAX_POSITION_PCT * m)


def s_tier_size(r, eq, eq0, rng):
    """Tier scales the ceiling; nothing is refused."""
    return _tier(r, veto=False)


def s_tier_veto(r, eq, eq0, rng):
    """What the pipeline does: scale, and refuse above the ceiling."""
    return _tier(r, veto=True)


def s_inverse_vol(r, eq, eq0, rng):
    f = r["fvol"]
    if not np.isfinite(f) or f <= 0:
        return 0.0
    return float(np.clip(IVOL_REF_PCT * IVOL_REF_VOL / f,
                         MIN_POSITION_PCT, MAX_POSITION_PCT))


def s_random(r, eq, eq0, rng):
    return float(rng.uniform(MIN_POSITION_PCT, MAX_POSITION_PCT))


SIZERS = {
    "equal_notional": s_equal_notional,
    "equal_weight": s_equal_weight,
    "flat_risk_capped": s_flat_risk_capped,
    "tier_size": s_tier_size,
    "tier_veto": s_tier_veto,
    "inverse_vol": s_inverse_vol,
    "random_size": s_random,
}
BASELINE = "flat_risk_capped"     # the pre-fix behaviour, the thing to beat
PIPELINE = "tier_veto"            # what the pipeline actually does now


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================
class Book:
    """Positional numpy view of the close matrix - pandas lookups inside the day
    loop were the whole cost of this simulation."""

    def __init__(self, close):
        self.arr = close.to_numpy(dtype=np.float32)
        self.dates = close.index
        self.row = {d: i for i, d in enumerate(close.index)}
        self.col = {t: i for i, t in enumerate(close.columns)}

    def px(self, bi, ci):
        v = self.arr[bi, ci]
        return float(v) if np.isfinite(v) else np.nan


def simulate(samp, book, sizer, equity0, max_concurrent, rng):
    """
    Walk a daily calendar. Exits settle before entries on the same bar.

    Cash constraint: a position is SCALED DOWN to available cash rather than
    skipped. Skipping would let different rules take different trade sets, which
    breaks the one property this design rests on - that every rule sees the same
    trades. Scaling preserves it, and how often the constraint binds is reported
    instead, because a rule whose nominal size is routinely unaffordable is
    telling you something real.
    """
    if samp.empty:
        return None
    lo = book.row[samp["entry_date"].min()]
    hi = book.row[samp["exit_date"].max()]
    ent = {}
    for rec in samp.to_dict("records"):
        ent.setdefault(book.row[rec["entry_date"]], []).append(rec)

    cash = float(equity0)
    open_pos = []                    # [exit_bar, col, shares, exit_px]
    curve = np.empty(hi - lo + 1, dtype=float)
    binds = taken = 0

    for bi in range(lo, hi + 1):
        if open_pos:
            keep = []
            for p in open_pos:
                if p[0] == bi:
                    cash += p[2] * p[3]
                else:
                    keep.append(p)
            open_pos = keep

        todays = ent.get(bi)
        if todays:
            mtm = 0.0
            for p in open_pos:
                q = book.px(bi, p[1])
                if np.isfinite(q):
                    mtm += p[2] * q
            eq = cash + mtm
            for r in todays:
                if len(open_pos) >= max_concurrent:
                    break
                pct = sizer(r, eq, equity0, rng)
                if not np.isfinite(pct) or pct <= 0:
                    continue
                want = eq * pct / 100.0
                if want > cash:
                    binds += 1
                    want = cash
                sh = int(want / r["entry"])
                if sh < 1:
                    continue
                cash -= sh * r["entry"]
                open_pos.append([book.row[r["exit_date"]], book.col[r["ticker"]],
                                 sh, r["exit"]])
                taken += 1

        if open_pos:
            mtm = 0.0
            for p in open_pos:
                q = book.px(bi, p[1])
                if np.isfinite(q):
                    mtm += p[2] * q
            curve[bi - lo] = cash + mtm
        else:
            curve[bi - lo] = cash

    eq = pd.Series(curve, index=book.dates[lo:hi + 1])
    return stats(eq, equity0, taken, binds)


def stats(eq, equity0, taken, binds):
    dd = eq / eq.cummax() - 1.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / equity0) ** (1 / years) - 1
    m = eq.resample("ME").last().pct_change().dropna()
    sharpe = (m.mean() / m.std() * np.sqrt(12)) if len(m) > 2 and m.std() > 0 \
        else np.nan
    mdd = float(dd.min())
    return {
        "terminal": float(eq.iloc[-1]),
        "cagr": float(cagr),
        "max_dd": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < -1e-9 else np.nan,
        "sharpe": float(sharpe),
        "worst_month": float(m.min()) if len(m) else np.nan,
        "ruin": bool((eq / equity0).min() < RUIN_LEVEL),
        "n_trades": int(taken),
        "cash_binds": int(binds),
    }


# =============================================================================
# PATHS
# =============================================================================
def run_paths(trades, book, cfg):
    """
    One path = one random contiguous window + a random sample of the trades
    entering inside it. Windows differ across paths, so the spread of outcomes
    spans regimes (some paths contain 2008 or 2020) rather than reflecting
    sampling noise alone.
    """
    out = {k: [] for k in SIZERS}
    first = trades["entry_date"].min()
    last = trades["entry_date"].max()
    span = pd.Timedelta(days=int(cfg.window_years * 365.25))
    latest_start = last - span
    if latest_start <= first:
        latest_start = first
    total_days = max((latest_start - first).days, 1)
    used = 0

    for pi in range(cfg.paths):
        rng = np.random.default_rng(cfg.seed + pi)
        w0 = first + pd.Timedelta(days=int(rng.integers(0, total_days + 1)))
        w1 = w0 + span
        pool = trades[(trades["entry_date"] >= w0) & (trades["entry_date"] < w1)]
        if len(pool) < 20:
            continue
        k = min(cfg.trades_per_path, len(pool))
        samp = (pool.sample(n=k, random_state=cfg.seed + pi)
                .sort_values("entry_date").reset_index(drop=True))
        for r, fn in SIZERS.items():
            # a fresh rng per rule keeps random_size reproducible and stops draw
            # order leaking between rules
            s = simulate(samp, book, fn, cfg.equity, cfg.max_concurrent,
                         np.random.default_rng(cfg.seed + pi))
            if s:
                s["window"] = f"{w0:%Y-%m}"
                out[r].append(s)
        used += 1
        if used % 25 == 0:
            print(f"    {used}/{cfg.paths} paths")
    return out, used


def med(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
    return float(np.median(v)) if v else np.nan


def paired(out, rule, base, key, n_boot=5000, seed=0):
    """Paired difference (rule - base) on the same paths, bootstrap CI."""
    d = np.array([x[key] - y[key] for x, y in zip(out[rule], out[base])
                  if np.isfinite(x.get(key, np.nan))
                  and np.isfinite(y.get(key, np.nan))])
    if len(d) < 5:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n_boot, len(d)), replace=True).mean(axis=1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)), float((bs <= 0).mean()))


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_sizing_v51() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--calib", default=P2.CALIB_FILE)
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--trades-per-path", type=int, default=100)
    ap.add_argument("--window-years", type=float, default=3.0)
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--every", type=int, default=10, help="bars between candidates")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="sizing_eval_v51.json")
    return ap


def run_sizing_v51(cache=None, calib=None, paths=None, trades_per_path=None, window_years=None, equity=None, max_concurrent=None, every=None, start=None, end=None, max_tickers=None, seed=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_sizing_v51()
        run_sizing_v51(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "calib": calib, "paths": paths, "trades_per_path": trades_per_path, "window_years": window_years, "equity": equity, "max_concurrent": max_concurrent, "every": every, "start": start, "end": end, "max_tickers": max_tickers, "seed": seed, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.paths, cfg.max_tickers, cfg.every = 40, 60, 20

    if not os.path.exists(cfg.calib):
        sys.exit(f"{cfg.calib} not found. Run "
                 f"class_ai_pillar2_risk.py --calibrate first.")
    calib = P2.load_calibration(cfg.calib)

    tickers = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(cfg.cache, "*.pkl")))
    if not tickers:
        sys.exit(f"No .pkl files in {cfg.cache}.")
    if cfg.max_tickers:
        tickers = tickers[:cfg.max_tickers]

    print("=" * 92)
    print("V51 - DOES THE SIZING EARN ITS PLACE?   (entries random, held fixed)")
    print("=" * 92)
    print(f"  {len(tickers)} tickers | candidate every {cfg.every} bars | "
          f"equity ${cfg.equity:,.0f} | max {cfg.max_concurrent} concurrent")
    print("\n  building the trade table (exits are rule-independent, computed once)")
    trades, close = build_trades(
        tickers, cfg.cache, calib, cfg.every,
        pd.Timestamp(cfg.start) if cfg.start else None,
        pd.Timestamp(cfg.end) if cfg.end else None)
    if trades.empty:
        sys.exit("No candidate trades. Check --cache and the date range.")

    print(f"\n  {len(trades):,} candidate trades | {trades['ticker'].nunique()} "
          f"tickers | {trades['signal_date'].min():%Y-%m} -> "
          f"{trades['signal_date'].max():%Y-%m}")
    rc = trades["reason"].value_counts()
    print("  outcomes: " + "  ".join(f"{k} {v:,} ({v/len(trades):.1%})"
                                     for k, v in rc.items()))
    print(f"  mean trade return {trades['ret'].mean():+.2%} | "
          f"median {trades['ret'].median():+.2%}")
    print("  NOTE these are RANDOM entries. A positive mean is the market's drift")
    print("       plus survivorship, not a signal, and it accrues to all six rules")
    print("       equally - which is exactly why the paired difference is the result.")
    tm = trades["p_dd"].apply(tier_mult)
    print(f"  tier mix: full {(tm == 1.0).mean():.0%} | 0.75 {(tm == 0.75).mean():.0%}"
          f" | 0.50 {(tm == 0.50).mean():.0%} | vetoed {(tm == 0.0).mean():.0%}")

    book = Book(close)
    print(f"\n  running {cfg.paths} paths x {len(SIZERS)} rules, "
          f"{cfg.trades_per_path} trades per {cfg.window_years:g}y window")
    out, used = run_paths(trades, book, cfg)
    if used < 5:
        sys.exit("Too few usable paths - widen --window-years or lower --every.")

    print("\n" + "=" * 92)
    print(f"MEDIAN OUTCOME ACROSS {used} PATHS")
    print("=" * 92)
    print(f"  {'rule':<19}{'terminal':>10}{'CAGR':>8}{'max DD':>9}{'Calmar':>8}"
          f"{'Sharpe':>8}{'worst mo':>10}{'ruin':>7}{'trades':>8}")
    for r in SIZERS:
        if not out[r]:
            continue
        print(f"  {r:<19}{med(out[r], 'terminal'):>10,.0f}"
              f"{med(out[r], 'cagr'):>8.1%}{med(out[r], 'max_dd'):>9.1%}"
              f"{med(out[r], 'calmar'):>8.2f}{med(out[r], 'sharpe'):>8.2f}"
              f"{med(out[r], 'worst_month'):>10.1%}"
              f"{np.mean([x['ruin'] for x in out[r]]):>7.0%}"
              f"{med(out[r], 'n_trades'):>8.0f}")

    print("\n" + "=" * 92)
    print(f"PAIRED AGAINST {BASELINE}   (identical entries, so this is sizing alone)")
    print("=" * 92)
    print("  A positive difference is better for every metric shown: higher CAGR,")
    print("  shallower drawdown (less negative), higher Calmar.")
    print(f"\n  {'rule':<19}{'metric':<11}{'mean diff':>12}{'95% CI':>28}"
          f"{'P(diff<=0)':>12}")
    for r in SIZERS:
        if r == BASELINE or not out[r]:
            continue
        for key in ("cagr", "max_dd", "calmar"):
            d, lo_, hi_, p = paired(out, r, BASELINE, key, seed=cfg.seed)
            if not np.isfinite(d):
                continue
            f = (lambda x: f"{x:+.3f}") if key == "calmar" else (lambda x: f"{x:+.2%}")
            print(f"  {r:<19}{key:<11}{f(d):>12}"
                  f"{f(lo_) + ' to ' + f(hi_):>28}{p:>11.1%}")

    print("\n" + "=" * 92)
    print(f"HEAD TO HEAD AGAINST {PIPELINE}   (the pipeline's actual rule)")
    print("=" * 92)
    print("  This is the block that settles it. Beating flat_risk_capped is easy -")
    print("  almost any volatility awareness does. The question is whether the")
    print("  CALIBRATED probability beats a one-line heuristic (inverse_vol) and a")
    print("  random number from the same range (random_size), and whether the tier")
    print("  scaling adds anything once the veto is already in place (tier_size).")
    print("  Here a positive difference means the OTHER rule beat the pipeline.")
    print(f"\n  {'vs':<19}{'metric':<11}{'mean diff':>12}{'95% CI':>28}"
          f"{'P(diff<=0)':>12}")
    for r in SIZERS:
        if r == PIPELINE or not out[r]:
            continue
        for key in ("cagr", "max_dd", "calmar"):
            d, lo_, hi_, pv = paired(out, r, PIPELINE, key, seed=cfg.seed)
            if not np.isfinite(d):
                continue
            f = (lambda x: f"{x:+.3f}") if key == "calmar" else (lambda x: f"{x:+.2%}")
            print(f"  {r:<19}{key:<11}{f(d):>12}"
                  f"{f(lo_) + ' to ' + f(hi_):>28}{pv:>11.1%}")

    print("\n" + "=" * 92)
    print("HOW TO READ IT")
    print("=" * 92)
    print("  Work down the head-to-head block. Each line is a different claim.")
    print("\n  random_size WORSE than tier_veto on Calmar, CI excluding 0")
    print("      -> position size carries information at all. If this fails, stop:")
    print("         nothing downstream matters, because a random number in the same")
    print("         range did as well as the model.")
    print("  inverse_vol WORSE than tier_veto on Calmar, CI excluding 0")
    print("      -> the calibrated probability beats a one-line heuristic. This is")
    print("         the strong claim and the one worth defending in the paper.")
    print("  inverse_vol INDISTINGUISHABLE from tier_veto")
    print("      -> the honest result is that volatility awareness is what helps and")
    print("         the calibration adds nothing beyond it. Say so plainly. It is")
    print("         still a finding, and 1/vol sizing being enough is a clean,")
    print("         defensible conclusion - just a smaller one than hoped.")
    print("  tier_size INDISTINGUISHABLE from tier_veto")
    print("      -> the veto contributes nothing and all the value is in scaling.")
    print("         Drop MAX_DRAWDOWN_PROB from the pipeline; it costs trades for")
    print("         no measurable benefit.")
    print("  tier_size WORSE than tier_veto")
    print("      -> the opposite: refusing the worst names is where the benefit is,")
    print("         and a plain threshold would do. That needs no calibration curve,")
    print("         only a volatility cutoff - report it that way rather than")
    print("         crediting the calibration for it.")
    print("\n  On CAGR: expect tier_veto to LOSE return to the wider rules. That is")
    print("  the correct outcome, not a failure. Volatility cuts both ways, so")
    print("  shrinking the volatile names must cost upside. Quote Calmar. The claim")
    print("  is a better risk-adjusted outcome, never more money.")
    print("\n  Absolute levels are inflated by survivorship (V49: 41% ten-year")
    print("  survival, and the missing names are the ones that collapsed). The")
    print("  paired ranking survives that bias. The levels do not.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg), "n_candidate_trades": len(trades),
                   "n_paths_used": used, "results": out},
                  f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_sizing_v51(),
# and each value shown is that argument's own default, except QUICK which starts
# True so that a plain run finishes fast. Set it False for the real experiment.
#
# Passing any command-line flag still works and takes over, so the old CLI is
# not lost: it just is not the default way in any more.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    # <<< small, fast sanity run. False = the real thing.
    RUN_QUICK           = True
    RUN_CACHE           = 'price_cache_v43'
    RUN_CALIB           = 'pillar2_risk_calibration.json'
    RUN_PATHS           = 200
    RUN_TRADES_PER_PATH = 100
    RUN_WINDOW_YEARS    = 3.0
    RUN_EQUITY          = 10_000.0
    RUN_MAX_CONCURRENT  = 8
    RUN_EVERY           = 10                # bars between candidates
    RUN_START           = None
    RUN_END             = None
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_OUT             = 'sizing_eval_v51.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_sizing_v51.py --quick
    else:
        run_sizing_v51(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            calib=RUN_CALIB,
            paths=RUN_PATHS,
            trades_per_path=RUN_TRADES_PER_PATH,
            window_years=RUN_WINDOW_YEARS,
            equity=RUN_EQUITY,
            max_concurrent=RUN_MAX_CONCURRENT,
            every=RUN_EVERY,
            start=RUN_START,
            end=RUN_END,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            out=RUN_OUT,
        )
