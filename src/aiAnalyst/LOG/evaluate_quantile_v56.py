#!/usr/bin/env python3
"""
evaluate_quantile_v56.py - stop throwing away the depth of the drawdown.

THE ARGUMENT, FROM THIS PROJECT'S OWN RESULTS

Every drawdown model so far has used a binary target: did the name fall 30% or
more. At a 10.7% base rate that indicator discards almost everything it is handed
- a name that fell 29% and one that fell 5% are recorded identically, as "nothing
happened".

Compare what the SAME features, universe and walk-forward produced against a
continuous target versus a binary one:

    forward volatility (continuous)   R2 = 0.53, comfortably established
    P(30% drawdown)    (binary)       Brier skill CI touching zero

That is not volatility being special. It is the target. A binary outcome at
p = 0.107 carries about half a bit per observation; the actual depth of the worst
drawdown carries several. With the ~368 effective observations V52 measured, that
is not an affordable loss.

So model the DISTRIBUTION of the worst drawdown directly, by quantile regression
on mdd - which is already in the panel. P(fall >= 30%) then falls out as one point
on the predicted distribution, which makes it directly comparable with the
deployed model, and the rest of the distribution is new information: a median
expected drawdown, a 90th percentile, an expected shortfall.

WHAT IS MEASURED

  COVERAGE       does the predicted t-quantile actually have t of outcomes below
                 it? This is the reliability table's analogue for a quantile
                 model, and it is the metric that matters.
  PINBALL LOSS   against an unconditional quantile baseline, so a skill score
                 exists for the continuous target rather than only for the
                 binary one.
  THE SHARED     P(mdd <= -30%) read off the predicted distribution, scored with
  QUESTION       V52's metrics against the deployed two-stage model on identical
                 rows.

TICKER HOLDOUT - THE VALIDATION NOBODY HAS RUN

Every test in this project has held out TIME. None has held out NAMES. So it is
not known whether the calibration is a general property of equity volatility or
partly fitted to these 335 specific tickers' quirks.

--ticker-holdout reserves a share of tickers that no model ever trains on, and
scores them separately. Predictions then come in two flavours: seen names in
future years (temporal holdout, what every earlier script measured) and unseen
names in future years (temporal AND cross-sectional). If calibration transfers to
names the model has never met, the claim is materially stronger, and this is the
objection an examiner is most likely to raise.

USAGE
  python evaluate_quantile_v56.py
  python evaluate_quantile_v56.py --ticker-holdout 0.25
  python evaluate_quantile_v56.py --model linear --quick
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
from evaluate_direct_v54 import (MARKET, ONE_SIDED, POSITION, VOL, build_panel,
                                 paired_stat, summarise, two_stage)

ALL_FEATURES = VOL + ONE_SIDED + POSITION + MARKET
# Denser in the tail, because the tail is the question. A 30% fall sits near the
# 10th percentile of the mdd distribution, so resolution there is what decides
# whether the shared comparison is fair.
TAUS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
THRESHOLD = -P2.DRAWDOWN * 100      # -30, in percent, on the mdd scale


# =============================================================================
# QUANTILE MODEL
# =============================================================================
def fit_quantiles(Xtr, ytr, Xte, taus, kind):
    """
    One model per quantile, then sorted across quantiles.

    Separate fits can cross - the fitted 20th percentile can land above the
    fitted 30th - which would make the implied distribution non-monotone and the
    CDF read-off meaningless. Sorting each row's predictions across taus is the
    standard repair and it cannot hurt: a crossing is always an estimation error,
    never a real feature of a distribution.
    """
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    if ok.sum() < 500:
        return None
    Xtr, ytr = Xtr[ok], ytr[ok]
    mu = Xtr.mean(axis=0)
    sd = np.where(Xtr.std(axis=0) > 1e-9, Xtr.std(axis=0), 1.0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    finite = np.isfinite(Zte).all(axis=1)
    Zte = np.nan_to_num(Zte, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.empty((len(Zte), len(taus)), dtype=float)
    for i, t in enumerate(taus):
        if kind == "gbm":
            import xgboost as xgb
            m = xgb.XGBRegressor(objective="reg:quantileerror",
                                 quantile_alpha=t, n_estimators=250,
                                 max_depth=3, learning_rate=0.06,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=2.0, n_jobs=4, verbosity=0)
        else:
            from sklearn.linear_model import QuantileRegressor
            m = QuantileRegressor(quantile=t, alpha=1e-4, solver="highs")
        m.fit(Ztr, ytr)
        out[:, i] = m.predict(Zte)
    out = np.sort(out, axis=1)
    out[~finite] = np.nan
    return out


def cdf_at(qpred, taus, threshold):
    """
    P(mdd <= threshold) read off the predicted quantiles by interpolating the
    inverse CDF. Clamped to the fitted tau range: outside it the model has no
    information, and extrapolating a tail it never estimated is how a quantile
    model manufactures false confidence.
    """
    t = np.asarray(taus, float)
    out = np.full(len(qpred), np.nan)
    good = np.isfinite(qpred).all(axis=1)
    q = qpred[good]
    # each row's quantile curve is increasing in tau, so interpolate tau on value
    res = np.empty(len(q))
    for i in range(len(q)):
        res[i] = np.interp(threshold, q[i], t, left=t[0] / 2.0, right=t[-1])
    out[good] = res
    return out


def pinball(y, qpred, taus):
    """Mean quantile (pinball) loss over all taus - the proper scoring rule for a
    quantile forecast."""
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(qpred).all(axis=1)
    y, q = y[ok], qpred[ok]
    tot = 0.0
    for i, t in enumerate(taus):
        d = y - q[:, i]
        tot += np.mean(np.maximum(t * d, (t - 1) * d))
    return float(tot / len(taus))


def coverage(y, qpred, taus):
    """Realised share of outcomes at or below each predicted quantile. A
    well-specified model puts tau of them there."""
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(qpred).all(axis=1)
    y, q = y[ok], qpred[ok]
    return pd.DataFrame({"tau": taus,
                         "realised": [(y <= q[:, i]).mean() for i in range(len(taus))],
                         "n": len(y)})


# =============================================================================
# SPLITS
# =============================================================================
def ticker_groups(tickers, share, seed):
    """Deterministic hash split, so the holdout set is identical between runs and
    between the arms compared within a run."""
    if share <= 0:
        return set(), set(tickers)
    hold = set()
    for t in tickers:
        h = hashlib.sha256(f"{seed}:{t}".encode()).hexdigest()
        if int(h[:8], 16) / 0xFFFFFFFF < share:
            hold.add(t)
    return hold, set(tickers) - hold


def walk(panel, cfg, held):
    """
    Refit per year on data whose forward windows closed before the year opened,
    and never on a held-out ticker. Each test row is tagged seen/unseen so the
    two kinds of generalisation are reported apart.
    """
    yrs = sorted(panel["date"].dt.year.unique())
    rows = []
    for yr in [y for y in yrs if y >= yrs[0] + int(cfg.min_train_years)]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = panel[(panel["date"] <= opens - pd.Timedelta(days=cfg.embargo_days))
                   & (~panel["ticker"].isin(held))]
        te = panel[(panel["date"] >= opens) &
                   (panel["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        q = fit_quantiles(tr[ALL_FEATURES].to_numpy(float),
                          tr["mdd"].to_numpy(float),
                          te[ALL_FEATURES].to_numpy(float), TAUS, cfg.model)
        if q is None:
            continue
        p_bin = two_stage(tr, te)
        rec = pd.DataFrame({
            "rid": te.index.to_numpy(), "date": te["date"].to_numpy(),
            "ticker": te["ticker"].to_numpy(), "mdd": te["mdd"].to_numpy(float),
            "event": te["event"].to_numpy(float),
            "p_q": cdf_at(q, TAUS, THRESHOLD),
            "p_bin": p_bin if p_bin is not None else np.nan,
            "unseen": te["ticker"].isin(held).to_numpy()})
        for i, t in enumerate(TAUS):
            rec[f"q{t}"] = q[:, i]
        # The reference forecast has to come from the TRAINING period. A first
        # version computed the unconditional quantiles from the test outcomes
        # themselves, which handed the baseline the answer and made it the
        # in-sample optimum for the very data being scored - the model was then
        # judged against something it could not beat by construction.
        tq = np.nanquantile(tr["mdd"].to_numpy(float), TAUS)
        for i, t in enumerate(TAUS):
            rec[f"b{t}"] = tq[i]
        rows.append(rec)
        if cfg.verbose:
            print(f"    {yr}: train {len(tr):,} -> test {len(te):,} "
                  f"({int(rec['unseen'].sum()):,} unseen)")
    return pd.concat(rows, ignore_index=True) if rows else None


# =============================================================================
# REPORTING
# =============================================================================
def report_block(d, label, cfg):
    if len(d) < 1000:
        print(f"\n  {label}: only {len(d):,} rows, skipped")
        return None
    q = d[[f"q{t}" for t in TAUS]].to_numpy(float)
    y = d["mdd"].to_numpy(float)
    print(f"\n  {label}  ({len(d):,} rows, {d['ticker'].nunique()} tickers)")

    cov = coverage(y, q, TAUS)
    print(f"    {'tau':>6}{'predicted':>11}{'realised':>10}{'error':>9}"
          f"{'median q':>11}")
    for _, r in cov.iterrows():
        i = TAUS.index(r["tau"])
        print(f"    {r['tau']:>6.2f}{r['tau']:>11.1%}{r['realised']:>10.1%}"
              f"{r['realised'] - r['tau']:>+9.1%}"
              f"{np.nanmedian(q[:, i]):>10.1f}%")
    mae_cov = float((cov["realised"] - cov["tau"]).abs().mean() * 100)
    print(f"    mean absolute coverage error: {mae_cov:.2f}pp")

    # the reference is the unconditional quantile from each row's own TRAINING
    # period, walked forward exactly like the model
    pin = pinball(y, q, TAUS)
    b = d[[f"b{t}" for t in TAUS]].to_numpy(float)
    pin_base = pinball(y, b, TAUS)
    print(f"    pinball loss {pin:.3f} vs unconditional {pin_base:.3f}"
          f"   skill {1 - pin / pin_base:+.4f}")
    # A median forecast minimises ABSOLUTE error, so scoring it with R2 - a
    # squared-error measure against the mean - penalises it for doing its job and
    # can read negative on a perfectly good model. Compare absolute errors
    # against the unconditional median instead.
    med = q[:, TAUS.index(0.50)]
    b_med = b[:, TAUS.index(0.50)]
    ok = np.isfinite(med) & np.isfinite(y) & np.isfinite(b_med)
    mae_med = float(np.mean(np.abs(y[ok] - med[ok])))
    mae_null = float(np.mean(np.abs(y[ok] - b_med[ok])))
    print(f"    median forecast: mean |error| {mae_med:.2f}pp vs "
          f"{mae_null:.2f}pp for a constant median"
          f"   skill {1 - mae_med / mae_null:+.4f}")
    return {"coverage_mae_pp": mae_cov, "pinball": pin,
            "pinball_base": pin_base, "pinball_skill": 1 - pin / pin_base,
            "median_mae": mae_med, "median_mae_null": mae_null,
            "median_mae_skill": 1 - mae_med / mae_null, "n": len(d)}


def shared_question(d, label, cfg):
    """The binary comparison, on identical rows."""
    m = d.dropna(subset=["p_q", "p_bin"])
    if len(m) < 1000:
        print(f"\n  {label}: too few rows for the shared comparison")
        return None
    pq, pb = m["p_q"].to_numpy(), m["p_bin"].to_numpy()
    y, dts = m["event"].to_numpy(), m["date"].to_numpy()
    ref = float(np.nanmean(y))
    print(f"\n  {label} - P(fall >= {abs(THRESHOLD):.0f}%), {len(m):,} rows")
    print(f"    {'model':<14}{'MAE pp':>9}{'skill':>10}{'AUC':>9}")
    out = {}
    for nm, p in (("two_stage", pb), ("quantile", pq)):
        out[nm] = {"mae": V52.mae(p, y), "skill": V52.skill(p, y, ref),
                   "auc": V52.auc(p, y)}
        print(f"    {nm:<14}{out[nm]['mae']:>9.2f}{out[nm]['skill']:>+10.4f}"
              f"{out[nm]['auc']:>9.4f}")
    print(f"\n    paired (quantile - two_stage, half-year blocks)")
    for lbl, fn, low_better in (("MAE pp", lambda a, b: V52.mae(a, b), True),
                                ("skill", lambda a, b: V52.skill(a, b, ref), False),
                                ("AUC", V52.auc, False)):
        dd, lo, hi = paired_stat(pq, pb, y, dts, fn, cfg.n_boot, cfg.seed)
        if not np.isfinite(dd):
            continue
        good = (hi < 0) if low_better else (lo > 0)
        bad = (lo > 0) if low_better else (hi < 0)
        v = "better" if good else "WORSE" if bad else "not estab."
        f = "{:+.2f}" if lbl == "MAE pp" else "{:+.4f}"
        print(f"      {lbl:<9}{f.format(dd):>9}"
              f"{f.format(lo) + ' to ' + f.format(hi):>26}{v:>14}")
        out.setdefault("paired", {})[lbl] = (dd, lo, hi)
    return out


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_quantile_v56() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--panel-cache", default="panel_v54.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=6)
    ap.add_argument("--embargo-days", type=int, default=P2.HORIZON_DAYS)
    ap.add_argument("--model", default="gbm", choices=["gbm", "linear"])
    ap.add_argument("--ticker-holdout", type=float, default=0.25,
                    help="share of tickers no model ever trains on (0 disables)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="quantile_eval_v56.json")
    return ap


def run_quantile_v56(cache=None, panel_cache=None, rebuild=None, step=None, min_train_years=None, embargo_days=None, model=None, ticker_holdout=None, n_boot=None, max_tickers=None, seed=None, verbose=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_quantile_v56()
        run_quantile_v56(verbose=False)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "panel_cache": panel_cache, "rebuild": rebuild, "step": step, "min_train_years": min_train_years, "embargo_days": embargo_days, "model": model, "ticker_holdout": ticker_holdout, "n_boot": n_boot, "max_tickers": max_tickers, "seed": seed, "verbose": verbose, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.max_tickers, cfg.step, cfg.n_boot = 60, 20, 200
    V52.N_BOOT = cfg.n_boot

    print("=" * 96)
    print("V56 - THE DRAWDOWN DISTRIBUTION, NOT A YES/NO")
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
        print(f"  building the panel from {len(tickers)} tickers")
        panel = build_panel(tickers, cfg.cache, cfg.step)
        panel.to_pickle(cfg.panel_cache)
    if panel.empty:
        sys.exit("Empty panel.")
    miss = [f for f in ALL_FEATURES + ["mdd"] if f not in panel.columns]
    if miss:
        sys.exit(f"Panel missing {miss} - rebuild with V54.")

    all_t = sorted(panel["ticker"].unique())
    held, keep = ticker_groups(all_t, cfg.ticker_holdout, cfg.seed)
    print(f"  {len(panel):,} rows | {len(all_t)} tickers | "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"  mdd: median {panel['mdd'].median():.1f}%, "
          f"10th percentile {panel['mdd'].quantile(0.10):.1f}%, "
          f"{panel['event'].mean():.1%} at or past {THRESHOLD:.0f}%")
    print(f"  ticker holdout: {len(held)} of {len(all_t)} names never trained on")
    print(f"  {cfg.model} quantile regression at {len(TAUS)} taus: {TAUS}")

    print("\n" + "=" * 96)
    print("WALK-FORWARD")
    print("=" * 96)
    d = walk(panel, cfg, held)
    if d is None:
        sys.exit("No usable windows.")
    print(f"\n  {len(d):,} predictions | "
          f"{int((~d['unseen']).sum()):,} on seen names, "
          f"{int(d['unseen'].sum()):,} on unseen")

    print("\n" + "=" * 96)
    print("THE DRAWDOWN DISTRIBUTION  (coverage is the metric)")
    print("=" * 96)
    print("  A well-specified quantile model puts tau of the outcomes at or below")
    print("  its predicted tau-quantile. Realised BELOW tau means the model is too")
    print("  pessimistic there; ABOVE means too optimistic - and being optimistic")
    print("  in the lower tail is the dangerous direction for a risk model.")
    res = {"all": report_block(d, "ALL PREDICTIONS", cfg)}
    if cfg.ticker_holdout > 0:
        res["seen"] = report_block(d[~d["unseen"]], "SEEN TICKERS (time held out "
                                                    "only)", cfg)
        res["unseen"] = report_block(d[d["unseen"]], "UNSEEN TICKERS (time AND "
                                                     "name held out)", cfg)

    print("\n" + "=" * 96)
    print(f"THE SHARED QUESTION  - does the continuous target beat the binary one?")
    print("=" * 96)
    res["shared_all"] = shared_question(d, "ALL PREDICTIONS", cfg)
    if cfg.ticker_holdout > 0:
        res["shared_unseen"] = shared_question(d[d["unseen"]], "UNSEEN TICKERS",
                                               cfg)

    print("\n" + "=" * 96)
    print("HOW TO READ IT")
    print("=" * 96)
    print("  COVERAGE ERROR SMALL AND THE PINBALL SKILL CLEARLY POSITIVE")
    print("      -> the drawdown DISTRIBUTION is forecastable, which is a stronger")
    print("         and more useful claim than one threshold probability. Report")
    print("         the median and the 10th percentile as the headline outputs.")
    print("  QUANTILE BEATS two_stage ON THE SHARED QUESTION")
    print("      -> the information argument holds: the binary target was wasting")
    print("         the depth of the drawdown, and using it improves even the")
    print("         narrow question the deployed model was built for.")
    print("  QUANTILE TIES two_stage ON THE SHARED QUESTION")
    print("      -> the distribution is still worth having for its own sake, but")
    print("         the deployed model stays as it is for P(30%). Say both.")
    print("  UNSEEN TICKERS MATCH SEEN TICKERS")
    print("      -> the calibration is a property of equity volatility, not of")
    print("         these 335 names. This is the answer to the first objection an")
    print("         examiner will raise, and nothing before this run tested it.")
    print("  UNSEEN CLEARLY WORSE THAN SEEN")
    print("      -> part of what looked like calibration was fitted to the")
    print("         universe. Every figure in V52 and V53 then needs the unseen")
    print("         number beside it, and that is the honest headline.")
    print("\n  Coverage in the far tail (tau 0.02, 0.05) rests on few events and")
    print("  will be the noisiest row. Read the 0.10-0.30 band as the load-bearing")
    print("  part: it is where the 30% threshold lives.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items()},
                   "taus": TAUS, "results": res}, f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_quantile_v56(),
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
    RUN_MODEL           = 'gbm'
    RUN_TICKER_HOLDOUT  = 0.25              # share of tickers no model ever trains on (0 disables)
    RUN_N_BOOT          = 500
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_VERBOSE         = True
    RUN_OUT             = 'quantile_eval_v56.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_quantile_v56.py --quick
    else:
        run_quantile_v56(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            panel_cache=RUN_PANEL_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            embargo_days=RUN_EMBARGO_DAYS,
            model=RUN_MODEL,
            ticker_holdout=RUN_TICKER_HOLDOUT,
            n_boot=RUN_N_BOOT,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            verbose=RUN_VERBOSE,
            out=RUN_OUT,
        )
