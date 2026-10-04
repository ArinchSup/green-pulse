#!/usr/bin/env python3
"""
evaluate_direct_v54.py - is the two-stage architecture throwing information away?

THE ARGUMENT

The deployed model goes features -> volatility -> drawdown curve. That is only
optimal if volatility is a SUFFICIENT STATISTIC for drawdown risk, and there are
two concrete reasons to doubt it.

    VOLATILITY IS SYMMETRIC, A 30% FALL IS NOT.  Realised variance counts up
    moves and down moves alike. Splitting it by sign added nothing to the
    VOLATILITY forecast in V53 - the leverage effect washes out over 180 days -
    but that is a different question from whether it helps forecast a one-sided
    event. Nobody has asked the second question.

    HOW FAR THERE IS TO FALL IS NOT A VOLATILITY FACT.  A stock at the top of
    its 60-day range has further to drop before it is down 30% than one already
    down 40% from its high. Distance from trend and position in range are
    mechanically relevant to the threshold being asked about, and they cannot
    reach the answer through a volatility channel at all.

A direct model of P(drawdown | features) can use both. The two-stage model
cannot, by construction: whatever the features say, they are compressed into one
number before the curve sees them.

THE LADDER

Each rung adds one theory-motivated block. Features are never searched - V52 put
the effective sample size at ~368 independent observations, so a search would fit
noise and look convincing doing it.

    two_stage      the deployed architecture, reproduced as the baseline
    vol_direct     the same three Yang-Zhang features, straight to the event
                   - isolates what the twelve-bin curve costs or buys
    + one_sided    downside/upside semivariance ratio, trailing 60-bar drawdown
    + position     position in the 60-day range, gap to the 200-day EMA
    + market       cross-sectional median volatility, breadth above the 200 EMA

The market block is the one aimed at V52's temporal flatness. Everything else in
this project is a per-name feature, and per-name features cannot move a whole
year's forecast: on any given date they disagree with each other and average out.
A market-state variable is the same for every name that day, so it is the only
kind of feature that can raise or lower the forecast for an entire period. Watch
the tracking and spread rows on that rung specifically.

MODEL CHOICE

Logistic regression, deliberately. It emits a probability directly, so
calibration - the metric that matters here - is not an afterthought, and with
nine features it is hard to overfit. --model gbm runs gradient boosting as a
check on whether anything nonlinear is being left behind; if it wins by a wide
margin, suspect the margin before believing it.

EVERYTHING IS WALK-FORWARD, WITH V52'S METRICS

Refit each year on data whose forward windows closed before the year opened. The
metrics are imported from evaluate_calibration_v52 rather than reimplemented, so
every number below is directly comparable with the figures in that run.

USAGE
  python evaluate_direct_v54.py
  python evaluate_direct_v54.py --model gbm
  python evaluate_direct_v54.py --panel-cache panel_v54.pkl
  python evaluate_direct_v54.py --quick
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
from evaluate_volforecast_v53 import (close_to_close, curve_from, p_from_curve,
                                      r2, yang_zhang)

MIN_TRAIN_YEARS = 6
N_CURVE_BINS = 12

VOL = ["yz5", "yz22", "yz66"]
ONE_SIDED = ["semivar_ratio22", "mdd60"]
POSITION = ["range_pos60", "ema200_gap"]
MARKET = ["mkt_vol", "mkt_breadth"]

LADDER = {
    "two_stage": None,                                  # special: vol -> curve
    "vol_direct": VOL,
    "one_sided": VOL + ONE_SIDED,
    "position": VOL + ONE_SIDED + POSITION,
    "market": VOL + ONE_SIDED + POSITION + MARKET,
}
BASELINE = "two_stage"


# =============================================================================
# PANEL
# =============================================================================
def build_panel(tickers, cache, step, verbose=True):
    """
    Per-name features plus the same forward target V52 and V53 used, then the
    market-state columns merged on afterwards.

    The target is copied from P2.build_observations on purpose - 180 calendar
    days forward, at least 80 bars in the window, event at -30% on the low. Any
    drift in that definition would break comparability with the earlier runs,
    which is the whole point of this script.
    """
    rows = []
    daily = {}          # ticker -> full daily series used for the market state
    for n, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache)
        if df is None or len(df) < P2.MIN_BARS + 140:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        ret = c.pct_change()

        f = pd.DataFrame(index=df.index)
        for w in (5, 22, 66):
            f[f"yz{w}"] = yang_zhang(o, h, l, c, w)
        f["rv60"] = close_to_close(ret, 60)

        # one-sided: realised semivariance ratio, and the fall already suffered
        neg = (ret.where(ret < 0, 0.0) ** 2).rolling(22).mean()
        pos = (ret.where(ret > 0, 0.0) ** 2).rolling(22).mean()
        f["semivar_ratio22"] = np.sqrt(neg / (pos + 1e-12))
        f["mdd60"] = (c / c.rolling(60).max() - 1.0) * 100

        # position: how much room there is above, and distance from trend
        hi60, lo60 = h.rolling(60).max(), l.rolling(60).min()
        f["range_pos60"] = (c - lo60) / (hi60 - lo60).replace(0, np.nan)
        ema200 = c.ewm(span=200, adjust=False).mean()
        f["ema200_gap"] = (c / ema200 - 1.0) * 100
        f["_above200"] = (c > ema200).astype(float)

        # Keep the FULL daily series for the cross-section. Aggregating market
        # state from the sampled observation rows was a bug: each ticker's grid
        # starts at its own history offset and steps by its own bars, so the
        # grids barely overlap and a typical date held about 3 names. The market
        # state has to come from every ticker on every date.
        daily[t] = pd.DataFrame({"yz22": f["yz22"], "above200": f["_above200"]})
        cols = list(f.columns)
        arr = f.to_numpy(float)
        lows, closes, idx = l.to_numpy(float), c.to_numpy(float), df.index

        for p in range(P2.MIN_BARS, len(df) - 130, step):
            row = arr[p]
            if not np.isfinite(row).all():
                continue
            entry = closes[p]
            if not np.isfinite(entry) or entry <= 0:
                continue
            end = idx[p] + pd.Timedelta(days=P2.HORIZON_DAYS)
            j = np.searchsorted(idx.to_numpy(), np.datetime64(end), side="right")
            a, b = p + 1, min(j, len(df))
            if b - a < 80:
                continue
            fwd = closes[a:b]
            mdd = (float(np.nanmin(lows[a:b])) - entry) / entry * 100
            rec = {"ticker": t, "date": idx[p],
                   "fwd_vol": float(pd.Series(fwd).pct_change().std()
                                    * np.sqrt(252) * 100),
                   "mdd": mdd, "event": float(mdd <= -P2.DRAWDOWN * 100)}
            rec.update({k: row[i] for i, k in enumerate(cols)})
            rows.append(rec)
        if verbose and n % 40 == 0:
            print(f"    [{n}/{len(tickers)}] {len(rows):,} rows")

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])

    # Market state: the cross-section on each TRADING date, over every ticker
    # with data that day, contemporaneous only. Every name on a date gets the
    # same value, which is what makes this block able to move a whole period's
    # forecast when per-name features cannot.
    vol = pd.DataFrame({t: v["yz22"] for t, v in daily.items()})
    ab = pd.DataFrame({t: v["above200"] for t, v in daily.items()})
    mkt = pd.DataFrame({"mkt_vol": vol.median(axis=1, skipna=True),
                        "mkt_breadth": ab.mean(axis=1, skipna=True),
                        "_n": vol.notna().sum(axis=1)})
    thin = int(min(20, max(5, 0.1 * len(daily))))
    mkt.loc[mkt["_n"] < thin, ["mkt_vol", "mkt_breadth"]] = np.nan
    med = int(mkt["_n"].median())
    if verbose:
        print(f"    market state: {len(mkt):,} trading dates, median "
              f"{med} tickers in the cross-section "
              f"(dates under {thin} dropped)")
    if med < 20:
        print(f"    WARNING cross-section of {med} names is too thin to be a "
              f"market state - treat the market rung as uninformative")
    d = d.merge(mkt[["mkt_vol", "mkt_breadth"]], left_on="date",
                right_index=True, how="left")
    d.attrs["mkt_median_n"] = med
    return d.drop(columns=["_above200"]).sort_values("date").reset_index(drop=True)


# =============================================================================
# MODELS
# =============================================================================
def fit_logit(Xtr, ytr, Xte, kind="logit"):
    """
    Standardise on the TRAINING rows only, then fit. Scaling on the pooled panel
    would leak the test period's distribution into the transform - a small leak,
    and the kind that quietly flatters every fold.
    """
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    if ok.sum() < 500 or len(np.unique(ytr[ok])) < 2:
        return None
    Xtr, ytr = Xtr[ok], ytr[ok]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    finite = np.isfinite(Zte).all(axis=1)
    Zte = np.nan_to_num(Zte, nan=0.0, posinf=0.0, neginf=0.0)

    if kind == "gbm":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=2.0, n_jobs=4, verbosity=0,
                              eval_metric="logloss")
        m.fit(Ztr, ytr)
        p = m.predict_proba(Zte)[:, 1]
    else:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Ztr, ytr)
        p = m.predict_proba(Zte)[:, 1]
    return np.where(finite, p, np.nan)


def two_stage(tr, te):
    """
    The deployed architecture: OLS from the Yang-Zhang features to forward
    volatility, then a twelve-bin empirical curve from that forecast to the
    event. The curve is fitted on the training set's OWN fitted values, which is
    what class_ai_pillar2_risk_v2 does.
    """
    Xtr, Xte = tr[VOL].to_numpy(float), te[VOL].to_numpy(float)
    y = tr["fwd_vol"].to_numpy(float)
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(y) & (y > 0)
    if ok.sum() < 500:
        return None
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(ok.sum()), Xtr[ok]], y[ok],
                               rcond=None)
    ftr = beta[0] + Xtr @ beta[1:]
    cv = curve_from(ftr, tr["event"].to_numpy(float), N_CURVE_BINS)
    if cv is None:
        return None
    fte = beta[0] + Xte @ beta[1:]
    p = p_from_curve(fte, cv)
    return np.where(np.isfinite(fte), p, np.nan)


# =============================================================================
# WALK FORWARD
# =============================================================================
def walk(panel, name, feats, cfg):
    yrs = sorted(panel["date"].dt.year.unique())
    out = []
    for yr in [y for y in yrs if y >= yrs[0] + int(cfg.min_train_years)]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = panel[panel["date"] <= opens - pd.Timedelta(days=cfg.embargo_days)]
        te = panel[(panel["date"] >= opens) &
                   (panel["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        if name == BASELINE:
            p = two_stage(tr, te)
        else:
            p = fit_logit(tr[feats].to_numpy(float),
                          tr["event"].to_numpy(float),
                          te[feats].to_numpy(float), cfg.model)
        if p is None:
            continue
        out.append(pd.DataFrame({"rid": te.index.to_numpy(),
                                 "date": te["date"].to_numpy(), "p": p,
                                 "y": te["event"].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else None


def summarise(rec, ref):
    p, y, dts = rec["p"].to_numpy(), rec["y"].to_numpy(), rec["date"].to_numpy()
    bd = V52.brier_decomp(p, y)
    g = V52.by_year(dts, p, y)
    tk = V52.tracking(g)
    return {
        "mae": V52.mae(p, y), "auc": V52.auc(p, y),
        "skill": V52.skill(p, y, ref),
        "resolution": bd["resolution"] if bd else np.nan,
        "reliability": bd["reliability"] if bd else np.nan,
        "spearman": tk["spearman"] if tk else np.nan,
        "pearson": tk["pearson"] if tk else np.nan,
        "pred_spread": tk["pred_spread_pp"] if tk else np.nan,
        "real_spread": tk["real_spread_pp"] if tk else np.nan,
        "within5": (f"{int((g['error'].abs() <= 5).sum())}/{len(g)}"
                    if len(g) else "n/a"),
        "worst_year": float(g["error"].abs().max()) if len(g) else np.nan,
        "by_year": g.to_dict("records"),
    }


def paired_stat(pa, pb, y, dates, fn, n_boot, seed):
    """Block-bootstrap the difference in a metric between two models on the
    same rows. Half-year blocks, because drawdowns arrive together."""
    ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y)
    pa, pb, y, d = pa[ok], pb[ok], y[ok], np.asarray(dates)[ok]
    blk = V52.blocks_of(d)
    uniq = np.unique(blk)
    idx = {k: np.flatnonzero(blk == k) for k in uniq}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in
                              rng.choice(uniq, len(uniq), replace=True)])
        va, vb = fn(pa[sel], y[sel]), fn(pb[sel], y[sel])
        if np.isfinite(va) and np.isfinite(vb):
            reps.append(va - vb)
    if len(reps) < 50:
        return np.nan, np.nan, np.nan
    reps = np.array(reps)
    return (float(fn(pa, y) - fn(pb, y)), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)))


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_direct_v54() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--panel-cache", default="panel_v54.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=MIN_TRAIN_YEARS)
    ap.add_argument("--embargo-days", type=int, default=P2.HORIZON_DAYS)
    ap.add_argument("--model", default="logit", choices=["logit", "gbm"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="direct_eval_v54.json")
    return ap


def run_direct_v54(cache=None, panel_cache=None, rebuild=None, step=None, min_train_years=None, embargo_days=None, model=None, n_boot=None, max_tickers=None, seed=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_direct_v54()
        run_direct_v54(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "panel_cache": panel_cache, "rebuild": rebuild, "step": step, "min_train_years": min_train_years, "embargo_days": embargo_days, "model": model, "n_boot": n_boot, "max_tickers": max_tickers, "seed": seed, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.max_tickers, cfg.step, cfg.n_boot = 60, 20, 200
    V52.N_BOOT = cfg.n_boot

    print("=" * 96)
    print("V54 - DOES A DIRECT DRAWDOWN MODEL BEAT THE TWO-STAGE ONE?")
    print("=" * 96)

    if os.path.exists(cfg.panel_cache) and not cfg.rebuild:
        panel = pd.read_pickle(cfg.panel_cache)
        print(f"  loaded {len(panel):,} rows from {cfg.panel_cache} "
              f"(--rebuild to redo)")
    else:
        tickers = sorted(os.path.splitext(os.path.basename(p))[0]
                         for p in glob.glob(os.path.join(cfg.cache, "*.pkl")))
        if cfg.max_tickers:
            tickers = tickers[:cfg.max_tickers]
        if not tickers:
            sys.exit(f"No .pkl files in {cfg.cache}.")
        print(f"  building the panel from {len(tickers)} tickers "
              f"(step {cfg.step})")
        panel = build_panel(tickers, cfg.cache, cfg.step)
        panel.to_pickle(cfg.panel_cache)
        print(f"  cached to {cfg.panel_cache}")
    if panel.empty:
        sys.exit("Empty panel.")

    print(f"  {len(panel):,} rows | {panel['ticker'].nunique()} tickers | "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"  base rate {panel['event'].mean():.1%} | event = a fall of "
          f"{P2.DRAWDOWN:.0%} within {P2.HORIZON_DAYS} days")
    # A market rung whose cross-section cannot be verified is worse than no
    # market rung: it reports a conclusion about a feature that is not there.
    # Skip it rather than print a number somebody will read as a finding.
    mkn = panel.attrs.get("mkt_median_n")
    skip_market = False
    if {"mkt_vol", "mkt_breadth"} <= set(panel.columns):
        if mkn is None:
            print("  market state: cross-section size NOT RECORDED in this cached")
            print("    panel. It may have been built before the fix, in which case")
            print("    the cross-section held about 3 names and the market rung is")
            print("    meaningless. SKIPPING that rung - rerun with --rebuild to")
            print("    test it.")
            skip_market = True
        else:
            print(f"  market state cross-section: median {mkn} tickers per "
                  f"trading date")
            if mkn < 20:
                print(f"  {mkn} names is too thin to be a market state - SKIPPING "
                      f"that rung")
                skip_market = True
    if skip_market:
        LADDER.pop("market", None)

    print("\n" + "=" * 96)
    print(f"WALK-FORWARD  ({cfg.model}, refit yearly, "
          f"{cfg.embargo_days}d embargo)")
    print("=" * 96)
    runs, stats = {}, {}
    for name, feats in LADDER.items():
        miss = [f for f in (feats or []) if f not in panel.columns]
        if miss:
            print(f"  {name}: missing {miss}, skipped")
            continue
        rec = walk(panel, name, feats, cfg)
        if rec is None:
            print(f"  {name}: no usable windows")
            continue
        runs[name] = rec
        nf = "curve" if feats is None else f"{len(feats)}f"
        print(f"  {name:<12} {len(rec):>8,} predictions   {nf}")
    if BASELINE not in runs:
        sys.exit("Baseline did not run - nothing to compare against.")

    base = runs[BASELINE]
    ref = float(np.nanmean(base["y"].to_numpy()))
    for name, rec in runs.items():
        stats[name] = summarise(rec, ref)

    print("\n" + "=" * 96)
    print("CALIBRATION AND DISCRIMINATION")
    print("=" * 96)
    print(f"  {'model':<12}{'MAE pp':>8}{'within 5pp':>12}{'worst yr':>10}"
          f"{'skill':>9}{'resolution':>12}{'AUC':>8}")
    for name in LADDER:
        if name not in stats:
            continue
        s = stats[name]
        print(f"  {name:<12}{s['mae']:>8.2f}{s['within5']:>12}"
              f"{s['worst_year']:>10.1f}{s['skill']:>+9.4f}"
              f"{s['resolution']:>12.5f}{s['auc']:>8.4f}")

    print("\n" + "=" * 96)
    print("TRACKING ACROSS YEARS  - the temporal flatness V52 flagged")
    print("=" * 96)
    print("  A forecast that only knows the base rate is flat across years. The")
    print("  spread columns say how much it moves against how much reality did.")
    print("  Spread ratio is forecast spread over reality's: 1.00 is ideal, below")
    print("  1.00 too flat, ABOVE 1.00 too volatile. Overshooting is its own")
    print("  failure, not progress past the finish line.")
    print(f"\n  {'model':<12}{'Spearman':>10}{'Pearson':>10}"
          f"{'forecast spread':>18}{'reality':>10}{'ratio':>8}{'':>4}")
    for name in LADDER:
        if name not in stats:
            continue
        st = stats[name]
        ratio = (st["pred_spread"] / st["real_spread"]
                 if np.isfinite(st["real_spread"]) and st["real_spread"] > 0
                 else np.nan)
        tag = ""
        if np.isfinite(ratio):
            tag = "flat" if ratio < 0.85 else "over" if ratio > 1.15 else "ok"
        print(f"  {name:<12}{st['spearman']:>+10.3f}{st['pearson']:>+10.3f}"
              f"{st['pred_spread']:>17.1f}pp{st['real_spread']:>9.1f}pp"
              f"{ratio:>8.2f}{tag:>6}")

    # Two comparisons, because they answer different questions. Against the
    # baseline is the ARCHITECTURE question (curve versus direct model). Against
    # the previous rung is the FEATURE question (what did this block add), and
    # it is the only one that isolates a feature block from the change of link
    # function - without it a rung's loss to the baseline cannot be told apart
    # from the logit link fitting worse than an empirical curve.
    order = [k for k in LADDER if k in runs]
    pairs = [(nm, BASELINE, "vs baseline") for nm in order if nm != BASELINE]
    pairs += [(nm, order[i - 1], "vs previous")
              for i, nm in list(enumerate(order))[2:]]
    print("\n" + "=" * 96)
    print("PAIRED COMPARISONS  (same rows, half-year block bootstrap)")
    print("=" * 96)
    print("  Calibration error: NEGATIVE is better (less error). Skill and AUC:")
    print("  positive is better. An interval spanning zero means not established.")
    print("  'vs baseline' tests the architecture; 'vs previous' tests the block")
    print("  this rung added, holding the model family fixed.")
    print(f"\n  {'model':<12}{'against':<14}{'metric':<9}{'diff':>9}"
          f"{'95% CI':>24}{'verdict':>18}")
    gains = {}
    for name, against, kind in pairs:
        other = runs[against]
        m = runs[name][["rid", "p"]].merge(other[["rid", "p", "y", "date"]],
                                           on="rid", suffixes=("_m", "_b"))
        if len(m) < 1000:
            print(f"  {name:<12}{kind:<14}{'too few shared rows':>50}")
            continue
        pa, pb = m["p_m"].to_numpy(), m["p_b"].to_numpy()
        y, dts = m["y"].to_numpy(), m["date"].to_numpy()
        key = f"{name} {kind}"
        gains[key] = {}
        for label, fn, better_low in (
                ("MAE pp", lambda a, b: V52.mae(a, b), True),
                ("skill", lambda a, b: V52.skill(a, b, ref), False),
                ("AUC", V52.auc, False)):
            dd, lo, hi = paired_stat(pa, pb, y, dts, fn, cfg.n_boot, cfg.seed)
            if not np.isfinite(dd):
                continue
            gains[key][label] = (dd, lo, hi)
            good = (hi < 0) if better_low else (lo > 0)
            bad = (lo > 0) if better_low else (hi < 0)
            v = "better" if good else "WORSE" if bad else "not estab."
            fmt = "{:+.3f}" if label != "MAE pp" else "{:+.2f}"
            print(f"  {name:<12}{against:<14}{label:<9}{fmt.format(dd):>9}"
                  f"{fmt.format(lo) + ' to ' + fmt.format(hi):>24}{v:>18}")

    print("\n" + "=" * 96)
    print("HOW TO READ IT")
    print("=" * 96)
    print("  vol_direct BEATS two_stage")
    print("      -> the twelve-bin curve was costing accuracy. Replace the curve")
    print("         with the direct model and keep the same features.")
    print("  vol_direct MATCHES two_stage")
    print("      -> the curve is fine and loses nothing. Any gain further down")
    print("         the ladder is then attributable to the new features, not to")
    print("         the change of architecture, which is the cleaner story.")
    print("  one_sided or position BEATS vol_direct")
    print("      -> volatility was NOT a sufficient statistic for drawdown, and")
    print("         the two-stage design was discarding usable information. This")
    print("         is the result the script was built to find.")
    print("  market RAISES THE SPREAD RATIO TOWARD 1.00 AND TRACKING HOLDS")
    print("      -> the specific fix for V52's temporal flatness. Per-name")
    print("         features cannot do this; a market-state variable can. Judge")
    print("         this rung on tracking and the ratio, not on pooled MAE.")
    print("  market WIDENS THE SPREAD BUT TRACKING FALLS")
    print("      -> the worst outcome to misread. A bigger spread with a worse")
    print("         ordering is noise that happens to vary over time: the forecast")
    print("         now moves a lot and moves wrongly, which is more dangerous to")
    print("         a user than a forecast that is merely flat. Reject the rung.")
    print("  NOTHING BEATS two_stage")
    print("      -> volatility really is sufficient for this question at this")
    print("         horizon, and the deployed architecture is already the right")
    print("         one. A clean negative, and it settles the design question.")
    # The link function is a confound in the architecture comparison, so say so
    # when the numbers show its signature rather than leaving it to be misread.
    if cfg.model == "logit" and "vol_direct" in stats:
        if stats["vol_direct"]["auc"] < stats[BASELINE]["auc"] - 0.01:
            print("\n  READ THE ARCHITECTURE ROWS WITH CARE. two_stage is ahead on AUC")
            print("  using the SAME features, so the difference is the link, not the")
            print("  information. The twelve-bin curve is a nonparametric monotone")
            print("  link fitted from data; logistic regression forces a logit of a")
            print("  LINEAR combination. The volatility-to-drawdown relation is")
            print("  strongly convex, so the curve is the more flexible of the two,")
            print("  not the more constrained. Re-run with --model gbm for a")
            print("  like-for-like comparison: gradient boosting can bend the same")
            print("  way the curve does, so it isolates the architecture question")
            print("  from the shape of the link. The 'vs previous' rows are already")
            print("  clean - they hold the family fixed and test only the features.")

    print("\n  Two cautions. A logistic model with nine features on ~368 effective")
    print("  observations can still overfit, so an unestablished interval here is")
    print("  the expected outcome rather than a disappointment. And survivorship")
    print("  still biases every absolute number toward under-predicting, because")
    print("  the names missing from the cache are the ones that collapsed.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg), "n_rows": len(panel),
                   "stats": {k: {a: b for a, b in v.items() if a != "by_year"}
                             for k, v in stats.items()},
                   "by_year": {k: v["by_year"] for k, v in stats.items()},
                   "paired": gains}, f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_direct_v54(),
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
    RUN_PANEL_CACHE     = 'panel_v54.pkl'
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_MIN_TRAIN_YEARS = 6
    RUN_EMBARGO_DAYS    = 180
    RUN_MODEL           = 'logit'
    RUN_N_BOOT          = 500
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_OUT             = 'direct_eval_v54.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_direct_v54.py --quick
    else:
        run_direct_v54(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            panel_cache=RUN_PANEL_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            embargo_days=RUN_EMBARGO_DAYS,
            model=RUN_MODEL,
            n_boot=RUN_N_BOOT,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            out=RUN_OUT,
        )
