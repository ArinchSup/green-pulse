#!/usr/bin/env python3
"""
evaluate_operating_point_v59.py - rebuild the model where it actually works.

WHAT V57 AND V56 HANDED OVER

V57 mapped the risk surface and found the deployed setting is in the worse half of
the usable band. For a 30% fall:

    horizon      30d     60d     90d    180d (deployed)
    error       0.37    0.89    1.34    2.20 pp
    AUC        0.786   0.746   0.723   0.692

V56 found the curve is too optimistic about the worst cases. The reason is
visible in the drawdown curve itself: with twelve equal-count bins, the top bin
spans everything above roughly 50% forecast volatility and predicts that group's
AVERAGE. The wildest names inside it are far worse than the average, so the bin
under-predicts - by 1.3pp in the deployed calibration's top bin, and much more in
V56's quantile version.

THREE CHANGES, TESTED TOGETHER

  HORIZON     30 / 60 / 90 / 180 days, each with the full metric suite rather
              than read off a surface grid.
  TAIL BINS   equal-count bins against bins that get finer at the top, where the
              consequential decisions are. Isotonic afterwards, because finer
              bins hold fewer observations and can invert - V55 learned that the
              hard way, losing 0.076 of AUC to a curve that was not monotone.
  MARKET      market stress added to the VOLATILITY FORECAST.

WHY MARKET STATE MIGHT WORK HERE WHEN IT FAILED TWICE BEFORE

V54 fed market state to a drawdown classifier and V55 used it as a level
multiplier on the finished probability. Both failed. Neither put it where it
economically belongs: market-wide stress is a predictor of VOLATILITY, and
volatility is the first stage. If it improves that forecast, the drawdown
probability improves through the curve for free.

The horizon matters for the same reason. Volatility regimes persist for weeks to
a couple of months, so a stress reading today says a lot about the next 30 days
and little about days 120-180. Both earlier tests used 180 days, where most of the
window sits beyond the regime's memory - the signal was being asked a question it
could not answer. At 30-90 days it is being asked one it can.

So this is not a third attempt at the same thing. It is the first attempt at the
right thing, and if it fails at every horizon then market state is finished as an
idea rather than merely untested.

EVERY ARM ON THE SAME ROWS

Horizons need different amounts of forward data, so their natural row sets differ.
Comparing a 30-day arm scored on more observations against a 180-day arm scored on
fewer would make part of every difference a difference of sample. All arms are
intersected onto the rows every one of them scored, and what that costs is
printed.

USAGE
  python evaluate_operating_point_v59.py                  # builds panel_v59.pkl
  python evaluate_operating_point_v59.py --quick
  python evaluate_operating_point_v59.py --thresholds 20 30 --horizons 60 90 180
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk as P2
import evaluate_calibration_v52 as V52
from evaluate_direct_v54 import paired_stat
from evaluate_volforecast_v53 import r2, yang_zhang

HORIZONS = [30, 60, 90, 180]
THRESHOLDS = [20, 30]
VOL = ["yz5", "yz22", "yz66"]
MARKET = ["mkt_vol", "mkt_breadth"]
DEPLOYED = ("base", 180, 30)        # what is live today, the reference for everything
N_BINS = 12
# Denser at the top: the last five bins cover the worst 20% of forecast
# volatility, where a position-size decision actually turns.
TAIL_EDGES = [0.0, .10, .20, .30, .40, .50, .60, .70, .80, .86, .91, .95, .98, 1.0]


# =============================================================================
# PANEL
# =============================================================================
def build_panel(tickers, cache, step, horizons, verbose=True):
    need = {h: max(10, int(0.6 * h * 252 / 365)) for h in horizons}
    rows, daily = [], {}
    for n, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache)
        if df is None or len(df) < P2.MIN_BARS + 60:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        o, h_, l_, c = df["Open"], df["High"], df["Low"], df["Close"]
        f = pd.DataFrame(index=df.index)
        for w in (5, 22, 66):
            f[f"yz{w}"] = yang_zhang(o, h_, l_, c, w)
        ema200 = c.ewm(span=200, adjust=False).mean()
        daily[t] = pd.DataFrame({"yz22": f["yz22"],
                                 "above200": (c > ema200).astype(float)})
        arr = f[VOL].to_numpy(float)
        lows, closes, idx = l_.to_numpy(float), c.to_numpy(float), df.index
        idx_np = idx.to_numpy()
        for p in range(P2.MIN_BARS, len(df) - need[min(horizons)] - 1, step):
            row = arr[p]
            if not np.isfinite(row).all():
                continue
            entry = closes[p]
            if not np.isfinite(entry) or entry <= 0:
                continue
            rec = {"ticker": t, "date": idx[p]}
            rec.update(dict(zip(VOL, row)))
            keep = False
            for hz in horizons:
                end = idx[p] + pd.Timedelta(days=hz)
                j = np.searchsorted(idx_np, np.datetime64(end), side="right")
                a, b = p + 1, min(j, len(df))
                if b - a < need[hz]:
                    rec[f"mdd{hz}"] = rec[f"fv{hz}"] = np.nan
                    continue
                keep = True
                rec[f"mdd{hz}"] = (float(np.nanmin(lows[a:b])) - entry) / entry * 100
                rec[f"fv{hz}"] = float(pd.Series(closes[a:b]).pct_change().std()
                                       * np.sqrt(252) * 100)
            if keep:
                rows.append(rec)
        if verbose and n % 40 == 0:
            print(f"    [{n}/{len(tickers)}] {len(rows):,} rows")
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    vol = pd.DataFrame({t: v["yz22"] for t, v in daily.items()})
    ab = pd.DataFrame({t: v["above200"] for t, v in daily.items()})
    mkt = pd.DataFrame({"mkt_vol": vol.median(axis=1, skipna=True),
                        "mkt_breadth": ab.mean(axis=1, skipna=True),
                        "_n": vol.notna().sum(axis=1)})
    thin = int(min(20, max(5, 0.1 * len(daily))))
    mkt.loc[mkt["_n"] < thin, MARKET] = np.nan
    med = int(mkt["_n"].median())
    if verbose:
        print(f"    market state: median {med} tickers per trading date")
    if med < 20:
        print(f"    WARNING cross-section of {med} is too thin to be a market "
              f"state - the mkt arms will be uninformative")
    d = d.merge(mkt[MARKET], left_on="date", right_index=True, how="left")
    d.attrs["mkt_median_n"] = med
    return d.sort_values("date").reset_index(drop=True)


# =============================================================================
# CURVE
# =============================================================================
def make_curve(fvol, event, tail, n_bins=N_BINS):
    """
    Forecast volatility -> probability, isotonically.

    Equal-count bins put the top bin across everything above roughly 50%
    volatility and report its average, so the worst names inside it are
    under-predicted. `tail` uses edges that get finer at the top instead.

    Isotonic afterwards is not decoration. Finer bins hold fewer observations and
    can invert; a non-monotone curve reorders the names, and since AUC is
    invariant under monotone maps, any AUC it loses is pure damage. V55 lost 0.076
    that way before this was understood.
    """
    d = pd.DataFrame({"f": fvol, "e": np.asarray(event, float)}).dropna()
    d = d[np.isfinite(d["f"]) & (d["f"] > 0)]
    if len(d) < 1000:
        return None
    if tail:
        qs = np.quantile(d["f"], TAIL_EDGES[1:-1])
        d["b"] = np.digitize(d["f"], np.unique(qs))
    else:
        d["b"] = pd.qcut(d["f"].rank(method="first"), n_bins, labels=False,
                         duplicates="drop")
    g = (d.groupby("b").agg(f=("f", "median"), p=("e", "mean"), n=("e", "size"))
         .sort_values("f"))
    g = g[g["n"] >= 30]
    if len(g) < 4:
        return None
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip",
                            y_min=0.0, y_max=1.0)
    ir.fit(g["f"].to_numpy(), g["p"].to_numpy(), sample_weight=g["n"].to_numpy())
    return {"x": g["f"].to_numpy(), "y": ir.predict(g["f"].to_numpy())}


def apply_curve(fvol, cv):
    p = np.interp(np.asarray(fvol, float), cv["x"], cv["y"])
    return np.where(np.isfinite(fvol), p, np.nan)


# =============================================================================
# WALK FORWARD
# =============================================================================
def ticker_groups(tickers, share, seed):
    if share <= 0:
        return set()
    return {t for t in tickers
            if int(hashlib.sha256(f"{seed}:{t}".encode()).hexdigest()[:8], 16)
            / 0xFFFFFFFF < share}


def walk(panel, hz, arms, thresholds, held, cfg):
    """
    One volatility model per (arm, horizon, year); the curves for both thresholds
    hang off it. The embargo is the horizon, so a training label has resolved
    before the window it predicts.
    """
    d = panel.dropna(subset=[f"mdd{hz}", f"fv{hz}"])
    if len(d) < 10000:
        return None, None
    yrs = sorted(d["date"].dt.year.unique())
    out = {(a, th): [] for a in arms for th in thresholds}
    volfit = {a: [] for a in arms}
    for yr in [y for y in yrs if y >= yrs[0] + int(cfg.min_train_years)]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = d[(d["date"] <= opens - pd.Timedelta(days=hz))
               & (~d["ticker"].isin(held))]
        te = d[(d["date"] >= opens) & (d["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        ytr = tr[f"fv{hz}"].to_numpy(float)
        for arm, (feats, tail) in arms.items():
            Xtr, Xte = tr[feats].to_numpy(float), te[feats].to_numpy(float)
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr) & (ytr > 0)
            if ok.sum() < 2000:
                continue
            beta, *_ = np.linalg.lstsq(np.c_[np.ones(ok.sum()), Xtr[ok]], ytr[ok],
                                       rcond=None)
            ftr = beta[0] + Xtr @ beta[1:]
            fte = beta[0] + Xte @ beta[1:]
            fin = np.isfinite(Xte).all(axis=1)
            fte = np.where(fin, fte, np.nan)
            volfit[arm].append((fte, te[f"fv{hz}"].to_numpy(float)))
            for th in thresholds:
                ev = (tr[f"mdd{hz}"].to_numpy(float) <= -th).astype(float)
                cv = make_curve(ftr, ev, tail)
                if cv is None:
                    continue
                out[(arm, th)].append(pd.DataFrame({
                    "rid": te.index.to_numpy(), "date": te["date"].to_numpy(),
                    "p": apply_curve(fte, cv),
                    "y": (te[f"mdd{hz}"].to_numpy(float) <= -th).astype(float),
                    "unseen": te["ticker"].isin(held).to_numpy()}))
    recs = {k: (pd.concat(v, ignore_index=True) if v else None)
            for k, v in out.items()}
    vr = {}
    for a, lst in volfit.items():
        if lst:
            pp = np.concatenate([x[0] for x in lst])
            aa = np.concatenate([x[1] for x in lst])
            vr[a] = r2(pp, aa)
    return recs, vr


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_operating_point_v59() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--panel-cache", default="panel_v59.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=6)
    ap.add_argument("--horizons", type=int, nargs="*", default=HORIZONS)
    ap.add_argument("--thresholds", type=int, nargs="*", default=THRESHOLDS)
    ap.add_argument("--ticker-holdout", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="operating_point_v59.json")
    return ap


def run_operating_point_v59(cache=None, panel_cache=None, rebuild=None, step=None, min_train_years=None, horizons=None, thresholds=None, ticker_holdout=None, n_boot=None, max_tickers=None, seed=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_operating_point_v59()
        run_operating_point_v59(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "panel_cache": panel_cache, "rebuild": rebuild, "step": step, "min_train_years": min_train_years, "horizons": horizons, "thresholds": thresholds, "ticker_holdout": ticker_holdout, "n_boot": n_boot, "max_tickers": max_tickers, "seed": seed, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.max_tickers, cfg.step, cfg.n_boot = 60, 20, 150
    V52.N_BOOT = cfg.n_boot

    ARMS = {"base": (VOL, False), "tail": (VOL, True),
            "mkt": (VOL + MARKET, True)}

    print("=" * 100)
    print("V59 - REBUILD AT THE RIGHT OPERATING POINT")
    print("=" * 100)

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
    miss = [c for c in VOL + MARKET if c not in panel.columns]
    miss += [f"mdd{h}" for h in cfg.horizons if f"mdd{h}" not in panel.columns]
    if miss:
        sys.exit(f"Panel missing {miss} - rerun with --rebuild.")

    all_t = sorted(panel["ticker"].unique())
    held = ticker_groups(all_t, cfg.ticker_holdout, cfg.seed)
    print(f"  {len(panel):,} rows | {len(all_t)} tickers | "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"  market cross-section: median "
          f"{panel.attrs.get('mkt_median_n', 'not recorded')} tickers/date")
    print(f"  ticker holdout: {len(held)} names never trained on")
    print(f"  arms: base = {N_BINS} equal bins | tail = "
          f"{len(TAIL_EDGES) - 1} bins, finer at the top | mkt = tail + market "
          f"state in the volatility forecast")

    print("\n" + "=" * 100)
    print("WALK-FORWARD")
    print("=" * 100)
    recs, volr2 = {}, {}
    for hz in cfg.horizons:
        r, vr = walk(panel, hz, ARMS, cfg.thresholds, held, cfg)
        if r is None:
            print(f"  {hz:>4}d: too few rows")
            continue
        volr2[hz] = vr
        got = 0
        for k, v in r.items():
            if v is not None:
                recs[(k[0], hz, k[1])] = v
                got += 1
        n = max((len(v) for v in r.values() if v is not None), default=0)
        print(f"  {hz:>4}d: {got} cells, {n:,} predictions each")
    if not recs:
        sys.exit("Nothing ran.")

    # every arm on the same rows
    common = None
    for v in recs.values():
        s = set(v["rid"])
        common = s if common is None else (common & s)
    print(f"\n  aligning all {len(recs)} cells on the {len(common):,} rows every "
          f"one scored")
    recs = {k: v[v["rid"].isin(common)].reset_index(drop=True)
            for k, v in recs.items()}

    print("\n" + "=" * 100)
    print("VOLATILITY FORECAST  (the first stage - does market state help it?)")
    print("=" * 100)
    print("  out-of-sample R2 on realised forward volatility at each horizon")
    print(f"\n  {'arm':<8}" + "".join(f"{str(h) + 'd':>10}" for h in cfg.horizons))
    for arm in ARMS:
        cells = [f"{volr2.get(h, {}).get(arm, float('nan')):>10.4f}"
                 for h in cfg.horizons]
        print(f"  {arm:<8}" + "".join(cells))
    print("\n  base and tail share one volatility model, so their rows match by")
    print("  construction - only the curve differs. mkt is the test: if market")
    print("  state predicts volatility at all, it shows up here first.")

    ref = recs.get(DEPLOYED)
    refbase = float(np.nanmean(ref["y"].to_numpy())) if ref is not None else None

    def stat(rec, sub=None):
        d = rec if sub is None else rec[sub]
        p, y, dt = d["p"].to_numpy(), d["y"].to_numpy(), d["date"].to_numpy()
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 1000:
            return None
        p, y, dt = p[ok], y[ok], dt[ok]
        b = float(y.mean())
        g = V52.by_year(dt, p, y)
        return {"mae": V52.mae(p, y), "auc": V52.auc(p, y),
                "skill": V52.skill(p, y, b), "base": b,
                "within5": (f"{int((g['error'].abs() <= 5).sum())}/{len(g)}"
                            if len(g) else "-"),
                "worst": float(g["error"].abs().max()) if len(g) else np.nan}

    for th in cfg.thresholds:
        print("\n" + "=" * 100)
        print(f"DRAWDOWN OF {th}%  - calibration, discrimination, year coverage")
        print("=" * 100)
        print(f"  {'arm':<8}{'horizon':>9}{'base':>8}{'MAE pp':>9}{'skill':>10}"
              f"{'AUC':>9}{'within 5pp':>12}{'worst yr':>10}{'unseen MAE':>12}")
        for arm in ARMS:
            for hz in cfg.horizons:
                rec = recs.get((arm, hz, th))
                if rec is None:
                    continue
                s = stat(rec)
                if s is None:
                    continue
                u = stat(rec, rec["unseen"]) if cfg.ticker_holdout > 0 else None
                mark = "  <- live" if (arm, hz, th) == DEPLOYED else ""
                ustr = f"{u['mae']:.2f}" if u else "-"
                print(f"  {arm:<8}{str(hz) + 'd':>9}{s['base']:>8.1%}"
                      f"{s['mae']:>9.2f}{s['skill']:>+10.4f}{s['auc']:>9.4f}"
                      f"{s['within5']:>12}{s['worst']:>10.1f}"
                      f"{ustr:>12}{mark}")

    # ------------------------------------------------------------------ pairing
    # ONLY within a cell. A 30-day 30% fall and a 180-day 30% fall are DIFFERENT
    # OUTCOMES, so there is no shared y to score two horizons against - a first
    # version graded the 180-day model's probabilities on the 30-day outcome
    # vector and produced a Brier skill difference of +1.28, which is impossible
    # since skill is bounded above by 1. That is the tell.
    #
    # So horizons are compared on the descriptive table, each cell scored against
    # its own outcome, with bootstrap intervals below. Arms ARE paired, because
    # they answer the identical question on identical rows.
    print("\n" + "=" * 100)
    print("ARMS PAIRED WITHIN EACH CELL  (same question, same rows)")
    print("=" * 100)
    print("  tail vs base isolates the bin structure; mkt vs tail isolates the")
    print("  market-state features. Horizons cannot be paired - different horizons")
    print("  are different events, with no common outcome to score both against.")
    print(f"\n  {'cell':<16}{'contrast':<14}{'metric':<9}{'diff':>9}"
          f"{'95% CI':>24}{'verdict':>14}")
    gains = {}
    for th in cfg.thresholds:
        for hz in cfg.horizons:
            for a, b in (("tail", "base"), ("mkt", "tail")):
                ra, rb = recs.get((a, hz, th)), recs.get((b, hz, th))
                if ra is None or rb is None:
                    continue
                m = ra[["rid", "p", "y", "date"]].merge(
                    rb[["rid", "p"]], on="rid", suffixes=("_a", "_b"))
                if len(m) < 1000:
                    continue
                pa, pb = m["p_a"].to_numpy(), m["p_b"].to_numpy()
                y, dt = m["y"].to_numpy(), m["date"].to_numpy()
                bs = float(np.nanmean(y))
                cell = f"{hz}d {th}%"
                key = f"{cell} {a}-{b}"
                gains[key] = {}
                for lbl, fn, low in (
                        ("MAE pp", lambda x, z: V52.mae(x, z), True),
                        ("skill", lambda x, z: V52.skill(x, z, bs), False),
                        ("AUC", V52.auc, False)):
                    dd, lo, hi = paired_stat(pa, pb, y, dt, fn, cfg.n_boot,
                                             cfg.seed)
                    if not np.isfinite(dd):
                        continue
                    gains[key][lbl] = (dd, lo, hi)
                    good = (hi < 0) if low else (lo > 0)
                    bad = (lo > 0) if low else (hi < 0)
                    v = "better" if good else "WORSE" if bad else "not estab."
                    f = "{:+.2f}" if lbl == "MAE pp" else "{:+.4f}"
                    print(f"  {cell:<16}{a + ' vs ' + b:<14}{lbl:<9}"
                          f"{f.format(dd):>9}"
                          f"{f.format(lo) + ' to ' + f.format(hi):>24}{v:>14}")

    # ------------------------------------------------- per-cell intervals
    print("\n" + "=" * 100)
    print("PER-CELL INTERVALS  (how horizons should be compared)")
    print("=" * 100)
    print("  Each cell bootstrapped on its own outcome, half-year blocks. Two")
    print("  horizons differ meaningfully only if their AUC intervals separate.")
    cells = {}
    for th in cfg.thresholds:
        print(f"\n  drawdown of {th}%")
        print(f"    {'arm':<7}{'horizon':>9}{'AUC':>9}{'95% CI':>22}"
              f"{'skill':>10}{'95% CI':>24}")
        for arm in ARMS:
            for hz in cfg.horizons:
                rec = recs.get((arm, hz, th))
                if rec is None:
                    continue
                pp = rec["p"].to_numpy()
                yy = rec["y"].to_numpy()
                dd = rec["date"].to_numpy()
                ok = np.isfinite(pp) & np.isfinite(yy)
                if ok.sum() < 1000:
                    continue
                pp, yy, dd = pp[ok], yy[ok], dd[ok]
                bs = float(yy.mean())
                _, alo, ahi = V52.block_boot(dd, pp, yy, V52.auc, seed=cfg.seed)
                _, slo, shi = V52.block_boot(
                    dd, pp, yy, lambda x, z: V52.skill(x, z, bs), seed=cfg.seed)
                cells[f"{arm}_{hz}d_{th}pct"] = {
                    "auc": V52.auc(pp, yy), "auc_lo": alo, "auc_hi": ahi,
                    "skill": V52.skill(pp, yy, bs), "skill_lo": slo,
                    "skill_hi": shi}
                mark = "  <- live" if (arm, hz, th) == DEPLOYED else ""
                print(f"    {arm:<7}{str(hz) + 'd':>9}{V52.auc(pp, yy):>9.4f}"
                      f"{f'{alo:.4f} to {ahi:.4f}':>22}"
                      f"{V52.skill(pp, yy, bs):>+10.4f}"
                      f"{f'{slo:+.4f} to {shi:+.4f}':>24}{mark}")

    print("\n" + "=" * 100)
    print("HOW TO READ IT")
    print("=" * 100)
    print("  Compare LIKE WITH LIKE. Across horizons at one threshold the base")
    print("  rate changes, so a smaller MAE partly reflects a rarer event. AUC is")
    print("  the horizon-fair measure: it asks whether the model can still tell")
    print("  the risky names from the safe ones, whatever the base rate.")
    print("\n  tail BEATS base AT THE SAME HORIZON")
    print("      -> the twelve equal bins were the problem, not the features. One")
    print("         function changes and the worst cases stop being understated.")
    print("  A SHORTER HORIZON WINS ON AUC AND HOLDS ON SKILL")
    print("      -> move the deployed model there. 180 days was chosen before")
    print("         anyone knew the surface, and V58 adds a second reason: the")
    print("         missing-companies exposure roughly halves at 90 days.")
    print("  mkt BEATS tail ON THE VOLATILITY R2, AT SHORT HORIZONS ONLY")
    print("      -> the regime-persistence argument holds. Market stress predicts")
    print("         weeks, not half-years, which is why V54 and V55 found nothing")
    print("         at 180 days. Keep it at the short end and say why.")
    print("  mkt ADDS NOTHING AT ANY HORIZON")
    print("      -> market state is finished. Three structurally different uses,")
    print("         four horizons, no effect. That closes it properly rather than")
    print("         leaving it as an untested maybe.")
    print("\n  Then check the unseen-ticker column against the main MAE. A gap")
    print("  there means the gain is fitted to the universe rather than real, and")
    print("  it matters more for the finer tail bins - they hold fewer names each.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg), "vol_r2": volr2,
                   "cells": {f"{k[0]}_{k[1]}d_{k[2]}pct": stat(v)
                             for k, v in recs.items()},
                   "intervals": cells, "paired_within_cell": gains},
                  f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_operating_point_v59(),
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
    RUN_PANEL_CACHE     = 'panel_v59.pkl'
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_MIN_TRAIN_YEARS = 6
    RUN_HORIZONS        = [30, 60, 90, 180]
    RUN_THRESHOLDS      = [20, 30]
    RUN_TICKER_HOLDOUT  = 0.25
    RUN_N_BOOT          = 400
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_OUT             = 'operating_point_v59.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_operating_point_v59.py --quick
    else:
        run_operating_point_v59(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            panel_cache=RUN_PANEL_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            horizons=RUN_HORIZONS,
            thresholds=RUN_THRESHOLDS,
            ticker_holdout=RUN_TICKER_HOLDOUT,
            n_boot=RUN_N_BOOT,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            out=RUN_OUT,
        )
