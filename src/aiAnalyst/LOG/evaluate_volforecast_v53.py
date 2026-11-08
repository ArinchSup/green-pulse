#!/usr/bin/env python3
"""
evaluate_volforecast_v53.py - can the volatility forecast be improved, and does
that improve the drawdown probability?

WHY THIS AND NOT ANOTHER DIRECTION MODEL

V52 established the one claim that survived: calibrated out-of-sample drawdown
probabilities, MAE 2.29pp, tracking realised risk across sixteen years at
Spearman +0.63. The entire thing runs on this:

    forward_vol = 10.53 + 0.693 x trailing_60d          R2 0.517 (in sample)

One feature. A single linear regression on 60-day close-to-close volatility.
Nobody has tried to improve it, and four standard things are missing from it.

    THE HIGH AND LOW ARE DISCARDED.  Close-to-close volatility uses one number
    per day and throws away the bar's range. Range estimators - Parkinson,
    Garman-Klass, Rogers-Satchell, Yang-Zhang - use the whole bar and are several
    times more statistically efficient for the same sample.

    ONE LOOKBACK.  Volatility is persistent at several scales at once. HAR-RV
    (Corsi 2009) regresses on short, medium and long windows together and beats
    single-window regressions almost universally.

    NO ASYMMETRY.  Volatility rises more after falls than after rises. A model
    fed only the magnitude of past moves cannot see that. Realised semivariance
    splits it.

    NO SCALE HANDLING.  Volatility is positive and right-skewed. The HAR
    literature works in logs, which fits better and cannot forecast a negative
    volatility.

THE LADDER

Each model adds exactly one idea to the one before, so the table reads as
"what did this idea buy". Features are chosen from theory, never searched - with
the effective sample size V52 measured, a feature search would fit noise and
look wonderful doing it.

    baseline       rv60                      the current model, reproduced
    har            rv5 + rv22 + rv66         multi-scale, close-to-close
    har_yz         yz5 + yz22 + yz66         same scales, efficient estimator
    har_yz_asym    + semivar ratio, down share
    har_yz_log     the same features in logs

THE TEST THAT DECIDES IT

A better R2 on volatility is not the point. The point is whether it improves the
DRAWDOWN forecast, which is the thing being claimed. So the script does not stop
at R2: it takes the winning volatility model, rebuilds the drawdown curve on top
of it, and re-runs V52's metrics - calibration error, Brier skill, AUC, and the
year-by-year tracking - so the comparison is like for like against V52's numbers.

An improvement in volatility R2 with no improvement in calibration or tracking
means the extra accuracy landed somewhere the drawdown question does not care
about. That is a real possible outcome and the script reports it rather than
quoting the R2 and stopping.

EVERYTHING IS WALK-FORWARD

V52's in-sample 0.517 is a floor, exactly as its 0.23pp calibration error was.
Every number here comes from refitting each year on data whose forward windows
had closed before that year opened. The baseline's OOS R2 is reported too - it
will be below 0.517, and that gap is the honest starting point.

USAGE
  python evaluate_volforecast_v53.py
  python evaluate_volforecast_v53.py --panel-cache panel_v53.pkl
  python evaluate_volforecast_v53.py --model gbm        # check for nonlinearity
  python evaluate_volforecast_v53.py --quick
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

MIN_TRAIN_YEARS = 6
N_CURVE_BINS = 12

# --- the ladder ---------------------------------------------------------------
HAR = ["rv5", "rv22", "rv66"]
HAR_YZ = ["yz5", "yz22", "yz66"]
ASYM = ["semivar_ratio22", "down_share22"]

MODELS = {
    "baseline":    {"feats": ["rv60"], "log": False},
    "har":         {"feats": HAR, "log": False},
    "har_yz":      {"feats": HAR_YZ, "log": False},
    "har_yz_asym": {"feats": HAR_YZ + ASYM, "log": False},
    "har_yz_log":  {"feats": HAR_YZ + ASYM, "log": True},
}
BASELINE = "baseline"


# =============================================================================
# ESTIMATORS
# =============================================================================
def _ann(var):
    """Variance per bar -> annualised volatility in percent."""
    return np.sqrt(np.maximum(var, 0) * 252.0) * 100.0


def close_to_close(ret, w):
    """
    Matches P2.realised_vol exactly: SIMPLE returns, population-style rolling
    std, annualised. Keeping the convention identical is what makes the baseline
    row a faithful reproduction of the shipped model rather than an approximation
    of it.
    """
    return ret.rolling(w).std() * np.sqrt(252) * 100


def yang_zhang(o, h, l, c, w):
    """
    Yang-Zhang (2000): the only common range estimator that handles both drift
    and overnight gaps. Built from overnight, open-to-close and Rogers-Satchell
    components, weighted to minimise variance.
    """
    co = np.log(o / c.shift())          # overnight
    oc = np.log(c / o)                  # open to close
    u = np.log(h / o)
    d = np.log(l / o)
    rs = u * (u - oc) + d * (d - oc)    # Rogers-Satchell, drift-free
    v_o = co.rolling(w).var(ddof=1)
    v_c = oc.rolling(w).var(ddof=1)
    v_rs = rs.rolling(w).mean()
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    return _ann(v_o + k * v_c + (1 - k) * v_rs)


def semivariance(ret, w):
    """
    Realised semivariance (Barndorff-Nielsen et al.): split the sum of squared
    returns by sign. The ratio is a direct read on asymmetry - above 1 means
    recent variance came mostly from falls.
    """
    neg = (ret.where(ret < 0, 0.0) ** 2).rolling(w).mean()
    pos = (ret.where(ret > 0, 0.0) ** 2).rolling(w).mean()
    return np.sqrt(neg / (pos + 1e-12)), (ret < 0).rolling(w).mean()


# =============================================================================
# PANEL
# =============================================================================
def build_panel(tickers, cache, step, verbose=True):
    """
    One row per (ticker, date) with every feature and the same forward target
    V52 used, so the two scripts' numbers are comparable.

    The target definition is copied from P2.build_observations deliberately -
    forward window of HORIZON_DAYS calendar days, at least 80 bars in it,
    forward vol from simple returns, event at -30% on the low. Any drift in that
    definition would make the comparison against V52 meaningless.
    """
    rows = []
    for n, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache)
        if df is None or len(df) < P2.MIN_BARS + 140:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        ret = c.pct_change()

        f = pd.DataFrame(index=df.index)
        f["rv60"] = close_to_close(ret, 60)
        for w in (5, 22, 66):
            f[f"rv{w}"] = close_to_close(ret, w)
            f[f"yz{w}"] = yang_zhang(o, h, l, c, w)
        f["semivar_ratio22"], f["down_share22"] = semivariance(ret, 22)
        # vol of vol: how unstable the volatility itself has been
        f["volvol60"] = f["rv22"].rolling(60).std()

        lows = l.to_numpy(float)
        closes = c.to_numpy(float)
        idx = df.index
        feat_arr = f.to_numpy(float)
        cols = list(f.columns)

        for pos in range(P2.MIN_BARS, len(df) - 130, step):
            row = feat_arr[pos]
            if not np.isfinite(row[cols.index("rv60")]):
                continue
            entry = closes[pos]
            if not np.isfinite(entry) or entry <= 0:
                continue
            end = idx[pos] + pd.Timedelta(days=P2.HORIZON_DAYS)
            j = np.searchsorted(idx.to_numpy(), np.datetime64(end), side="right")
            a, b = pos + 1, min(j, len(df))
            if b - a < 80:
                continue
            fwd_c = closes[a:b]
            fwd_vol = float(pd.Series(fwd_c).pct_change().std() * np.sqrt(252) * 100)
            mdd = (float(np.nanmin(lows[a:b])) - entry) / entry * 100
            rec = {"ticker": t, "date": idx[pos], "fwd_vol": fwd_vol,
                   "mdd": mdd, "event": mdd <= -P2.DRAWDOWN * 100}
            rec.update({k: row[i] for i, k in enumerate(cols)})
            rows.append(rec)
        if verbose and n % 40 == 0:
            print(f"    [{n}/{len(tickers)}] {len(rows):,} rows")
    return pd.DataFrame(rows)


# =============================================================================
# FITTING
# =============================================================================
def fit_predict(tr, te, feats, use_log, kind="ols"):
    """
    OLS by default, and that is a choice not laziness. With the effective sample
    size V52 measured, a flexible learner on eight features will find structure
    that is not there. --model gbm is available to check whether anything
    nonlinear is being left behind; if it beats OLS by a wide margin, suspect the
    margin before believing it.
    """
    Xtr = tr[feats].to_numpy(float)
    Xte = te[feats].to_numpy(float)
    ytr = tr["fwd_vol"].to_numpy(float)
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr) & (ytr > 0)
    if ok.sum() < 500:
        return None
    Xtr, ytr = Xtr[ok], ytr[ok]
    if use_log:
        Xtr = np.log(np.maximum(Xtr, 1e-6))
        Xte = np.log(np.maximum(Xte, 1e-6))
        ytr = np.log(ytr)

    if kind == "gbm":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=2.0, n_jobs=4, verbosity=0)
        m.fit(Xtr, ytr)
        p = m.predict(np.nan_to_num(Xte, nan=np.nanmean(Xtr)))
    else:
        A = np.c_[np.ones(len(Xtr)), Xtr]
        beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        p = np.c_[np.ones(len(Xte)), np.nan_to_num(Xte, nan=0.0)] @ beta
        p = np.where(np.isfinite(Xte).all(axis=1), p, np.nan)

    return np.exp(p) if use_log else p


def curve_from(fvol, event, n_bins=N_CURVE_BINS):
    """Empirical P(drawdown | forecast vol). A lookup table, as in the shipped
    model: twelve numbers cannot overfit and are trivial to audit."""
    d = pd.DataFrame({"f": fvol, "e": np.asarray(event, float)}).dropna()
    d = d[d["f"] > 0]
    if len(d) < 500:
        return None
    d["b"] = pd.qcut(d["f"].rank(method="first"), n_bins, labels=False,
                     duplicates="drop")
    g = d.groupby("b").agg(f=("f", "median"), p=("e", "mean")).sort_values("f")
    return {"x": g["f"].to_numpy(), "y": g["p"].to_numpy()}


def p_from_curve(fvol, cv):
    return np.interp(np.asarray(fvol, float), cv["x"], cv["y"])


# =============================================================================
# WALK FORWARD
# =============================================================================
def walk(panel, feats, use_log, min_train_years, embargo, kind, want_curve=False,
         verbose=False):
    """Refit each year on data whose forward windows had closed before it."""
    yrs = sorted(panel["date"].dt.year.unique())
    out = []
    for yr in [y for y in yrs if y >= yrs[0] + int(min_train_years)]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = panel[panel["date"] <= opens - pd.Timedelta(days=embargo)]
        te = panel[(panel["date"] >= opens) &
                   (panel["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        p = fit_predict(tr, te, feats, use_log, kind)
        if p is None:
            continue
        rec = pd.DataFrame({"rid": te.index.to_numpy(),
                            "date": te["date"].to_numpy(), "pred": p,
                            "actual": te["fwd_vol"].to_numpy(float),
                            "event": te["event"].to_numpy(float)})
        if want_curve:
            # the curve is fitted on the TRAINING set's own fitted values, which
            # is what the shipped model does - never on test predictions
            ptr = fit_predict(tr, tr, feats, use_log, kind)
            cv = curve_from(ptr, tr["event"].to_numpy(float))
            rec["p_dd"] = p_from_curve(p, cv) if cv else np.nan
        out.append(rec)
        if verbose:
            print(f"    {yr}: train {len(tr):,} -> test {len(te):,}")
    return pd.concat(out, ignore_index=True) if out else None


# =============================================================================
# METRICS
# =============================================================================
def r2(pred, actual):
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(p) & np.isfinite(a)
    if ok.sum() < 100:
        return np.nan
    p, a = p[ok], a[ok]
    return float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2))


def rmse(pred, actual):
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(p) & np.isfinite(a)
    return float(np.sqrt(np.mean((a[ok] - p[ok]) ** 2))) if ok.sum() else np.nan


def paired_r2(a_pred, b_pred, actual, dates, n_boot, seed):
    """
    Block-bootstrap the DIFFERENCE in R2 between two models on the same rows.
    Pairing matters: both models see identical observations, so the difference
    has far less variance than either level.
    """
    a, b, y = (np.asarray(x, float) for x in (a_pred, b_pred, actual))
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y, d = a[ok], b[ok], y[ok], np.asarray(dates)[ok]
    blk = V52.blocks_of(d)
    uniq = np.unique(blk)
    idx = {k: np.flatnonzero(blk == k) for k in uniq}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in
                              rng.choice(uniq, len(uniq), replace=True)])
        reps.append(r2(a[sel], y[sel]) - r2(b[sel], y[sel]))
    reps = np.array([x for x in reps if np.isfinite(x)])
    if len(reps) < 50:
        return np.nan, np.nan, np.nan
    return (float(r2(a, y) - r2(b, y)), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)))


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_volforecast_v53() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--panel-cache", default="panel_v53.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=MIN_TRAIN_YEARS)
    ap.add_argument("--embargo-days", type=int, default=P2.HORIZON_DAYS)
    ap.add_argument("--model", default="ols", choices=["ols", "gbm"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="volforecast_eval_v53.json")
    return ap


def run_volforecast_v53(cache=None, panel_cache=None, rebuild=None, step=None, min_train_years=None, embargo_days=None, model=None, n_boot=None, max_tickers=None, seed=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_volforecast_v53()
        run_volforecast_v53(cache=...)

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

    print("=" * 92)
    print("V53 - CAN THE VOLATILITY FORECAST BE IMPROVED, AND DOES IT MATTER?")
    print("=" * 92)

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
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    print(f"  {len(panel):,} rows | {panel['ticker'].nunique()} tickers | "
          f"{panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    print(f"  drawdown base rate {panel['event'].mean():.1%} | "
          f"mean forward vol {panel['fwd_vol'].mean():.1f}%")

    # sanity: the baseline must reproduce the shipped in-sample fit
    _x = panel["rv60"].to_numpy(float)
    _y = panel["fwd_vol"].to_numpy(float)
    _ok = np.isfinite(_x) & np.isfinite(_y)
    ins = np.polyfit(_x[_ok], _y[_ok], 1)
    pins = ins[1] + ins[0] * _x[_ok]
    print(f"\n  CHECK  in-sample single-feature fit: forward_vol = "
          f"{ins[1]:.2f} + {ins[0]:.3f} x rv60, R2 {r2(pins, _y[_ok]):.3f}")
    print("         V52's shipped figure was 10.53 + 0.693x, R2 0.517 - these")
    print("         should match closely, confirming the panel rebuilds the same")
    print("         target. Everything below is walk-forward instead.")

    # ---------------------------------------------------------------- the ladder
    print("\n" + "=" * 92)
    print(f"WALK-FORWARD VOLATILITY FORECAST  ({cfg.model.upper()}, "
          f"refit yearly, {cfg.embargo_days}d embargo)")
    print("=" * 92)
    runs = {}
    for name, spec in MODELS.items():
        miss = [f for f in spec["feats"] if f not in panel.columns]
        if miss:
            print(f"  {name}: missing {miss}, skipped")
            continue
        rec = walk(panel, spec["feats"], spec["log"], cfg.min_train_years,
                   cfg.embargo_days, cfg.model)
        if rec is None:
            print(f"  {name}: no usable windows")
            continue
        runs[name] = rec
        print(f"  {name:<14} OOS R2 {r2(rec['pred'], rec['actual']):>7.4f}   "
              f"RMSE {rmse(rec['pred'], rec['actual']):>5.2f} vol pts   "
              f"n {len(rec):,}   {len(spec['feats'])} features")

    if BASELINE not in runs:
        sys.exit("Baseline did not run - cannot compare.")
    base = runs[BASELINE]
    print(f"\n  The baseline's OOS R2 is the honest starting point. V52's 0.517")
    print(f"  was fitted and graded on one sample; expect this to be lower.")

    print("\n" + "=" * 92)
    print(f"PAIRED AGAINST {BASELINE}  (same rows, half-year block bootstrap)")
    print("=" * 92)
    print(f"  {'model':<14}{'R2 gain':>10}{'95% CI':>26}{'verdict':>26}")
    gains = {}
    for name, rec in runs.items():
        if name == BASELINE:
            continue
        # align on the panel row id: a model that skipped a year would
        # otherwise be compared row-for-row against the wrong observations
        m = rec[["rid", "pred"]].merge(
            base[["rid", "pred", "actual", "date"]], on="rid",
            suffixes=("_m", "_b"))
        if len(m) < 1000:
            print(f"  {name:<14}{'too few shared rows':>62}")
            continue
        d, lo, hi = paired_r2(m["pred_m"].to_numpy(), m["pred_b"].to_numpy(),
                              m["actual"].to_numpy(), m["date"].to_numpy(),
                              cfg.n_boot, cfg.seed)
        gains[name] = (d, lo, hi)
        v = ("real" if np.isfinite(lo) and lo > 0 else
             "not distinguishable" if np.isfinite(lo) else "n/a")
        print(f"  {name:<14}{d:>+10.4f}{f'{lo:+.4f} to {hi:+.4f}':>26}{v:>26}")

    # ------------------------------------------------------- close the loop
    ranked = sorted(runs, key=lambda k: r2(runs[k]["pred"], runs[k]["actual"]),
                    reverse=True)
    best = ranked[0]
    print("\n" + "=" * 92)
    print(f"DOES IT IMPROVE THE DRAWDOWN FORECAST?  {BASELINE} vs {best}")
    print("=" * 92)
    print("  A better volatility R2 only counts if it moves these numbers. If it")
    print("  does not, the extra accuracy landed somewhere the drawdown question")
    print("  does not care about.")

    comp = {}
    for name in dict.fromkeys([BASELINE, best]):
        spec = MODELS[name]
        rec = walk(panel, spec["feats"], spec["log"], cfg.min_train_years,
                   cfg.embargo_days, cfg.model, want_curve=True)
        if rec is None or rec["p_dd"].isna().all():
            continue
        p, y, dts = (rec["p_dd"].to_numpy(), rec["event"].to_numpy(),
                     rec["date"].to_numpy())
        g = V52.by_year(dts, p, y)
        tr = V52.tracking(g)
        bd = V52.brier_decomp(p, y)
        ref = float(np.nanmean(y))
        comp[name] = {
            "vol_r2": r2(rec["pred"], rec["actual"]),
            "mae_pp": V52.mae(p, y), "auc": V52.auc(p, y),
            "skill": V52.skill(p, y, ref),
            "resolution": bd["resolution"] if bd else np.nan,
            "tracking_spearman": tr["spearman"] if tr else np.nan,
            "tracking_pearson": tr["pearson"] if tr else np.nan,
            "pred_spread_pp": tr["pred_spread_pp"] if tr else np.nan,
            "real_spread_pp": tr["real_spread_pp"] if tr else np.nan,
            "worst_year_pp": float(g["error"].abs().max()) if len(g) else np.nan,
            "years_within_5pp": f"{int((g['error'].abs() <= 5).sum())}/{len(g)}",
            "by_year": g.to_dict("records"),
        }

    if len(comp) >= 2:
        a, b = BASELINE, best
        rowdefs = [("volatility R2", "vol_r2", "{:.4f}"),
                   ("calibration MAE (pp)", "mae_pp", "{:.2f}"),
                   ("worst year error (pp)", "worst_year_pp", "{:.1f}"),
                   ("years within 5pp", "years_within_5pp", "{}"),
                   ("Brier skill", "skill", "{:+.4f}"),
                   ("resolution", "resolution", "{:.5f}"),
                   ("AUC", "auc", "{:.4f}"),
                   ("tracking Spearman", "tracking_spearman", "{:+.3f}"),
                   ("tracking Pearson", "tracking_pearson", "{:+.3f}"),
                   ("forecast spread (pp)", "pred_spread_pp", "{:.1f}"),
                   ("reality spread (pp)", "real_spread_pp", "{:.1f}")]
        print(f"\n  {'metric':<24}{a:>16}{b:>16}")
        for label, k, fmt in rowdefs:
            va, vb = comp[a].get(k), comp[b].get(k)
            fa = fmt.format(va) if va is not None and (
                isinstance(va, str) or np.isfinite(va)) else "n/a"
            fb = fmt.format(vb) if vb is not None and (
                isinstance(vb, str) or np.isfinite(vb)) else "n/a"
            print(f"  {label:<24}{fa:>16}{fb:>16}")
        print("\n  BY YEAR")
        ga = {r["year"]: r for r in comp[a]["by_year"]}
        gb = {r["year"]: r for r in comp[b]["by_year"]}
        print(f"    {'year':<7}{'realised':>10}{a + ' err':>18}{b + ' err':>18}")
        for yr in sorted(set(ga) & set(gb)):
            print(f"    {int(yr):<7}{gb[yr]['realised']:>10.1%}"
                  f"{ga[yr]['error']:>+18.1f}{gb[yr]['error']:>+18.1f}")

    print("\n" + "=" * 92)
    print("HOW TO READ IT")
    print("=" * 92)
    print("  R2 GAIN CI EXCLUDING ZERO, AND THE DRAWDOWN TABLE IMPROVES")
    print("      -> a real improvement to the one claim that survived. Replace")
    print("         fit_vol_shrinkage in class_ai_pillar2_risk.py and recalibrate.")
    print("  R2 GAIN REAL, DRAWDOWN TABLE UNCHANGED")
    print("      -> the better forecast is more accurate about volatility in a")
    print("         range where the drawdown curve is flat. Worth one paragraph,")
    print("         not a rewrite: report the volatility result on its own and")
    print("         leave the risk model as it is.")
    print("  FORECAST SPREAD RISES TOWARD REALITY'S SPREAD")
    print("      -> this is the specific fix for V52's temporal flatness, which")
    print("         was the weakest part of that result. Even with no change in")
    print("         pooled MAE, narrowing that gap is a genuine gain.")
    print("  NOTHING MOVES")
    print("      -> 60-day close-to-close volatility already carries what daily")
    print("         OHLC has to say about the next six months. That is a clean,")
    print("         citable negative and it closes the question properly rather")
    print("         than leaving it open.")
    print("\n  2019 will not be fixed by any of this. February 2020 was exogenous")
    print("  and late-2019 volatility was low on every estimator here. Watch the")
    print("  ordinary years instead - that is where this can actually help.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(cfg), "n_rows": len(panel),
                   "oos_r2": {k: r2(v["pred"], v["actual"])
                              for k, v in runs.items()},
                   "paired_gains": gains, "drawdown_comparison": comp},
                  f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_volforecast_v53(),
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
    RUN_PANEL_CACHE     = 'panel_v53.pkl'
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_MIN_TRAIN_YEARS = 6
    RUN_EMBARGO_DAYS    = 180
    RUN_MODEL           = 'ols'
    RUN_N_BOOT          = 500
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_OUT             = 'volforecast_eval_v53.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_volforecast_v53.py --quick
    else:
        run_volforecast_v53(
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
