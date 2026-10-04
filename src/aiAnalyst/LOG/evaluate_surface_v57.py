#!/usr/bin/env python3
"""
evaluate_surface_v57.py - where does volatility-based risk forecasting work?

WHY A SURFACE

Everything in this project has asked one question: will this name fall 30% or
more within 180 days. That is a single point on a two-dimensional surface, and
the point was chosen before anyone knew whether it was a good place to stand.

Thresholds 10-50% by horizons 30-360 days is the same machinery answering
twenty-five questions. It is worth doing for three reasons, in ascending order of
interest:

  DEPLOYMENT      a user wants "how bad, how soon", not one number. A term
                  structure of drawdown risk is a usable output; a single
                  probability is a curiosity.
  THESIS CONTENT  twenty-five calibrated forecasts out of data already on disk.
  THE FINDING     where the method WORKS, and where it stops working. That is
                  new information and it is the point of the script.

WHAT TO EXPECT, AND WHY BOTH ENDS SHOULD FAIL

A 50% fall inside 30 days is very rare, so a cell like that has almost no events
and nothing can be established in it - the interval will swallow any result. A 10%
fall inside 360 days happens to most names, so there is little left to
discriminate and skill collapses toward zero from the other direction. Somewhere
between those is a band where the event is frequent enough to estimate and rare
enough to be informative. Mapping that band is the contribution; a surface that
worked uniformly would mean the metrics were not measuring anything.

EFFICIENCY, AND WHY IT MATTERS FOR HONESTY

The volatility forecast depends on the HORIZON but not on the THRESHOLD, so it is
fitted once per horizon-year and reused across all five thresholds. Only the
empirical curve is refitted per threshold. That is not just cheaper - it keeps
every threshold within a horizon resting on an identical volatility forecast, so
differences down a column are differences in the question, not in the model.

EFFECTIVE EVENT COUNTS

V52 measured a design effect of 438: drawdowns arrive together, so 160,000 rows
carry roughly 368 independent observations. A cell with a 1% base rate therefore
holds about four effective events, and no amount of rows changes that. Each cell
reports its effective event count, and cells below a floor are left blank rather
than filled with a number that cannot mean anything.

USAGE
  python evaluate_surface_v57.py                       # builds panel_v57.pkl
  python evaluate_surface_v57.py --quick
  python evaluate_surface_v57.py --thresholds 20 30 40 --horizons 90 180 360
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk as P2
import evaluate_calibration_v52 as V52
from evaluate_volforecast_v53 import curve_from, p_from_curve, yang_zhang

HORIZONS = [30, 60, 90, 180, 360]       # calendar days
THRESHOLDS = [10, 20, 30, 40, 50]       # percent fall from entry
VOL = ["yz5", "yz22", "yz66"]
MIN_EFF_EVENTS = 8.0                    # below this a cell says nothing
N_CURVE_BINS = 12


# =============================================================================
# PANEL
# =============================================================================
def build_panel(tickers, cache, step, horizons, verbose=True):
    """
    One row per (ticker, date) with the worst drawdown and the realised forward
    volatility at EVERY horizon.

    Each horizon needs its own minimum bar count in the forward window - the
    fixed floor of 80 bars that the 180-day work used would reject every 30-day
    observation outright.
    """
    need = {h: max(10, int(0.6 * h * 252 / 365)) for h in horizons}
    rows = []
    for n, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache)
        if df is None or len(df) < P2.MIN_BARS + 60:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        o, h_, l_, c = df["Open"], df["High"], df["Low"], df["Close"]
        f = pd.DataFrame(index=df.index)
        for w in (5, 22, 66):
            f[f"yz{w}"] = yang_zhang(o, h_, l_, c, w)
        arr = f.to_numpy(float)
        lows, closes = l_.to_numpy(float), c.to_numpy(float)
        idx = df.index
        idx_np = idx.to_numpy()

        for p in range(P2.MIN_BARS, len(df) - need[min(horizons)] - 1, step):
            row = arr[p]
            if not np.isfinite(row).all():
                continue
            entry = closes[p]
            if not np.isfinite(entry) or entry <= 0:
                continue
            rec = {"ticker": t, "date": idx[p], "yz5": row[0], "yz22": row[1],
                   "yz66": row[2]}
            any_h = False
            for hz in horizons:
                end = idx[p] + pd.Timedelta(days=hz)
                j = np.searchsorted(idx_np, np.datetime64(end), side="right")
                a, b = p + 1, min(j, len(df))
                if b - a < need[hz]:
                    rec[f"mdd{hz}"] = np.nan
                    rec[f"fv{hz}"] = np.nan
                    continue
                any_h = True
                rec[f"mdd{hz}"] = (float(np.nanmin(lows[a:b])) - entry) / entry * 100
                rec[f"fv{hz}"] = float(pd.Series(closes[a:b]).pct_change().std()
                                       * np.sqrt(252) * 100)
            if any_h:
                rows.append(rec)
        if verbose and n % 40 == 0:
            print(f"    [{n}/{len(tickers)}] {len(rows):,} rows")
    d = pd.DataFrame(rows)
    if not d.empty:
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").reset_index(drop=True)
    return d


# =============================================================================
# ONE HORIZON, ALL THRESHOLDS
# =============================================================================
def walk_horizon(panel, hz, thresholds, cfg):
    """
    Refit per year. The volatility model is horizon-specific but threshold-blind,
    so it is fitted once here and every threshold's curve is built on the same
    forecast - differences between thresholds are then differences in the
    question being asked, not in the model answering it.
    """
    d = panel.dropna(subset=[f"mdd{hz}", f"fv{hz}"])
    if len(d) < 10000:
        return None
    yrs = sorted(d["date"].dt.year.unique())
    out = {th: [] for th in thresholds}
    for yr in [y for y in yrs if y >= yrs[0] + int(cfg.min_train_years)]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        # the embargo is the horizon itself: a training label must have resolved
        # before the window it is used to predict
        tr = d[d["date"] <= opens - pd.Timedelta(days=hz)]
        te = d[(d["date"] >= opens) & (d["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        Xtr, Xte = tr[VOL].to_numpy(float), te[VOL].to_numpy(float)
        y = tr[f"fv{hz}"].to_numpy(float)
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y) & (y > 0)
        if ok.sum() < 2000:
            continue
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(ok.sum()), Xtr[ok]], y[ok],
                                   rcond=None)
        ftr = beta[0] + Xtr @ beta[1:]
        fte = beta[0] + Xte @ beta[1:]
        for th in thresholds:
            ev_tr = (tr[f"mdd{hz}"].to_numpy(float) <= -th).astype(float)
            cv = curve_from(ftr, ev_tr, N_CURVE_BINS)
            if cv is None:
                continue
            p = p_from_curve(fte, cv)
            out[th].append(pd.DataFrame({
                "date": te["date"].to_numpy(),
                "p": np.where(np.isfinite(fte), p, np.nan),
                "y": (te[f"mdd{hz}"].to_numpy(float) <= -th).astype(float)}))
    return {th: (pd.concat(v, ignore_index=True) if v else None)
            for th, v in out.items()}


def cell_stats(rec, cfg):
    """
    Statistics for one (horizon, threshold) cell, with the design effect MEASURED
    rather than assumed.

    V52's figure of 438 was measured at one horizon. Clustering has two sources -
    cross-sectional correlation, roughly constant across horizons because
    everything falls together in a crisis, and temporal overlap, which scales with
    the horizon: at a 10-bar sampling step a 360-day window overlaps its
    neighbours about 36 times over and a 30-day window barely twice. Carrying one
    number across the whole surface would therefore penalise the short horizons
    for clustering they do not have. So each cell block-bootstraps its own.

    The blocks are half-yearly, which stays conservative at short horizons: a
    30-day forecast is treated as though six months of them moved together. Better
    conservative than flattering.
    """
    if rec is None or len(rec) < 1000:
        return None
    p, y, dts = rec["p"].to_numpy(), rec["y"].to_numpy(), rec["date"].to_numpy()
    ok = np.isfinite(p) & np.isfinite(y)
    p, y, dts = p[ok], y[ok], dts[ok]
    if len(y) < 1000 or y.sum() < 20:
        return None
    base = float(y.mean())
    en = V52.effective_n(dts, y, seed=cfg.seed)
    eff = float(en["n_effective"] * base) if np.isfinite(en["n_effective"]) \
        else np.nan
    common = {"base": base, "n": len(y), "events": int(y.sum()),
              "eff": eff, "design_effect": en["design_effect"]}
    if not np.isfinite(eff) or eff < MIN_EFF_EVENTS:
        return {**common, "thin": True}
    return {**common, "thin": False, "mae": V52.mae(p, y), "auc": V52.auc(p, y),
            "skill": V52.skill(p, y, base)}


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_surface_v57() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--panel-cache", default="panel_v57.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=6)
    ap.add_argument("--horizons", type=int, nargs="*", default=HORIZONS)
    ap.add_argument("--thresholds", type=int, nargs="*", default=THRESHOLDS)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="surface_eval_v57.json")
    return ap


def run_surface_v57(cache=None, panel_cache=None, rebuild=None, step=None, min_train_years=None, horizons=None, thresholds=None, max_tickers=None, seed=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_surface_v57()
        run_surface_v57(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "panel_cache": panel_cache, "rebuild": rebuild, "step": step, "min_train_years": min_train_years, "horizons": horizons, "thresholds": thresholds, "max_tickers": max_tickers, "seed": seed, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.max_tickers, cfg.step = 60, 20
    V52.N_BOOT = 200

    print("=" * 96)
    print("V57 - THE RISK SURFACE: WHERE DOES THIS WORK?")
    print("=" * 96)

    if os.path.exists(cfg.panel_cache) and not cfg.rebuild:
        panel = pd.read_pickle(cfg.panel_cache)
        print(f"  loaded {len(panel):,} rows from {cfg.panel_cache}")
    else:
        tickers = sorted(os.path.splitext(os.path.basename(p))[0]
                         for p in glob.glob(os.path.join(cfg.cache, "*.pkl")))
        if cfg.max_tickers:
            tickers = tickers[:cfg.max_tickers]
        if not tickers:
            sys.exit(f"No .pkl files in {cfg.cache}.")
        print(f"  building the panel from {len(tickers)} tickers, "
              f"horizons {cfg.horizons}")
        panel = build_panel(tickers, cfg.cache, cfg.step, cfg.horizons)
        panel.to_pickle(cfg.panel_cache)
        print(f"  cached to {cfg.panel_cache}")
    if panel.empty:
        sys.exit("Empty panel.")
    miss = [f"mdd{h}" for h in cfg.horizons if f"mdd{h}" not in panel.columns]
    if miss:
        sys.exit(f"Panel lacks {miss} - rebuild with --rebuild.")

    print(f"  {len(panel):,} rows | {panel['ticker'].nunique()} tickers | "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"\n  BASE RATES  (share of observations reaching each fall)")
    print(f"    {'fall':>6}" + "".join(f"{str(h) + 'd':>10}" for h in cfg.horizons))
    for th in cfg.thresholds:
        cells = []
        for hz in cfg.horizons:
            m = panel[f"mdd{hz}"].dropna()
            cells.append(f"{(m <= -th).mean():>9.1%}" if len(m) else "        -")
        print(f"    {str(th) + '%':>6}" + "".join(cells))

    print("\n" + "=" * 96)
    print("WALK-FORWARD, ONE VOLATILITY MODEL PER HORIZON")
    print("=" * 96)
    surf = {}
    for hz in cfg.horizons:
        res = walk_horizon(panel, hz, cfg.thresholds, cfg)
        if res is None:
            print(f"  {hz}d: too few usable rows")
            continue
        got = sum(1 for v in res.values() if v is not None)
        n = max((len(v) for v in res.values() if v is not None), default=0)
        print(f"  {hz:>4}d: {got}/{len(cfg.thresholds)} thresholds, "
              f"{n:,} predictions each")
        surf[hz] = {th: cell_stats(v, cfg) for th, v in res.items()}

    def grid(key, fmt, title, note):
        print(f"\n  {title}")
        print(f"    {note}")
        print(f"    {'fall':>6}" + "".join(f"{str(h) + 'd':>10}"
                                          for h in cfg.horizons))
        for th in cfg.thresholds:
            cells = []
            for hz in cfg.horizons:
                c = (surf.get(hz) or {}).get(th)
                if c is None:
                    cells.append("        -")
                elif c.get("thin"):
                    cells.append("     thin")
                else:
                    cells.append(fmt.format(c[key]).rjust(10))
            print(f"    {str(th) + '%':>6}" + "".join(cells))

    print("\n" + "=" * 96)
    print("THE SURFACE")
    print("=" * 96)
    grid("design_effect", "{:.0f}", "DESIGN EFFECT (measured per cell)",
         "how many correlated rows one independent observation is worth")
    grid("eff", "{:.0f}", "EFFECTIVE EVENTS",
         f"independent events behind each cell. 'thin' = under "
         f"{MIN_EFF_EVENTS:.0f}, nothing reportable.")
    grid("mae", "{:.2f}", "CALIBRATION ERROR, pp  (lower is better)",
         "out-of-sample mean absolute reliability error")
    grid("skill", "{:+.4f}", "BRIER SKILL vs the base rate  (higher is better)",
         "at or below zero means the model is the base rate with extra steps")
    grid("auc", "{:.4f}", "AUC  (0.50 = no ability to rank)",
         "discrimination, independent of calibration")

    print("\n" + "=" * 96)
    print("HOW TO READ IT")
    print("=" * 96)
    print("  Read the SKILL grid first and find the band where it is clearly")
    print("  positive. That band is the answer to 'where does this work', and it")
    print("  is the result of the script.")
    print("\n  Expect both ends to fail, for opposite reasons:")
    print("    deep falls at short horizons  -> too few events, marked thin")
    print("    shallow falls at long horizons -> happens to nearly everything, so")
    print("       there is nothing left to discriminate and skill goes to zero")
    print("       even while calibration error stays small")
    print("  A LOW CALIBRATION ERROR WITH ZERO SKILL IS NOT A WORKING CELL. It")
    print("  means the model learned the base rate and nothing else, which the")
    print("  calibration metric cannot see. Both grids have to agree.")
    print("\n  For the deployed model, the useful output is a ROW: for one name,")
    print("  the probability of a given fall at each horizon. That is a term")
    print("  structure of risk and it is far more informative than the single")
    print("  30%/180d number Pillar 2 currently emits - provided the cells it is")
    print("  read from are inside the working band.")
    print("\n  The 30%/180d cell should reproduce V52/V53 (calibration error near")
    print("  2.2pp). If it does not, this panel and that one disagree and the")
    print("  difference has to be found before anything here is trusted.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg), "surface":
                   {str(h): {str(t): v for t, v in d.items()}
                    for h, d in surf.items()}}, f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_surface_v57(),
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
    RUN_PANEL_CACHE     = 'panel_v57.pkl'
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_MIN_TRAIN_YEARS = 6
    RUN_HORIZONS        = [30, 60, 90, 180, 360]
    RUN_THRESHOLDS      = [10, 20, 30, 40, 50]
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_OUT             = 'surface_eval_v57.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_surface_v57.py --quick
    else:
        run_surface_v57(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            panel_cache=RUN_PANEL_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            horizons=RUN_HORIZONS,
            thresholds=RUN_THRESHOLDS,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            out=RUN_OUT,
        )
