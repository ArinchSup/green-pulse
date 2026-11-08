#!/usr/bin/env python3
"""
class_ai_pillar2_surface_v63.py - one learned model for the whole risk surface.

WHAT THIS REPLACES

The deployed model answers exactly one question: P(fall of 20% within 90 days).
The horizon and the threshold are module constants, the calibration file is
fitted at one setting, and assess() has no horizon argument. A user who wants a
30-day view or a 1-year view cannot have one.

This module answers P(fall of d% within h days) for any (h, d) on a grid, from a
single fitted model, and emits the whole term structure in one call.

THE ARCHITECTURE - AND WHY IT IS NOT JUST "A BIGGER MODEL"

Naively you would stack every (h, d) cell into a long table and let a gradient
booster loose on it. That fails here for a specific reason: V52 put the effective
sample size of this problem at roughly 368 independent observations. Stacking 36
cells per date multiplies the ROWS by 36 and the INFORMATION by nothing, so a
free-form model spends its capacity relearning the square-root-of-time scaling
that is already known exactly, and overfits whatever is left.

So the model is built in three stages, each doing the job it is suited to.

    STAGE 1  volatility forecast, PER HORIZON.  Trailing Yang-Zhang volatility at
             three scales -> forward volatility over h days. Fitted separately
             for each horizon rather than annualised and scaled by sqrt(T),
             because volatility mean-reverts: a long-horizon forecast has to
             shrink further toward the long-run level than a short one. The
             coefficients are the term structure of that mean reversion, and they
             are learned, not assumed.

    STAGE 2  closed-form barrier probability, ZERO PARAMETERS.

                 P = 2 * Phi( ln(1 - d) / sigma_h )

             The probability that driftless geometric Brownian motion touches a
             barrier d below its start within the window (reflection principle).
             This is the physics of the problem. It is not a free parameter and
             it cannot overfit.

    STAGE 3  a learned correction, MONOTONE-CONSTRAINED.  Stage 2 is wrong in
             known directions - real returns have fat tails, equities drift up,
             volatility clusters - so a gradient booster learns the correction on
             the log-odds scale, with stage 2 supplied as base_margin:

                 p_model = sigmoid( logit(p_theory) + g(features, h, d) )

             With g identically zero the model IS the closed form. The learned
             part only has to capture what the closed form gets wrong, which is
             where the ~368 effective observations are actually worth spending.

    Monotonicity is enforced rather than hoped for. A term structure where
    P(60d) < P(30d) is not a small blemish, it is nonsense a user would see
    immediately. logit(p_theory) is already increasing in h and decreasing in d,
    so constraining g the same way (+1 on the horizon, -1 on the threshold) makes
    the sum monotone by construction.

CENSORING - WHY THE LAST YEAR OF DATA IS NOT THROWN AWAY

A 365-day horizon needs 252 forward bars. Requiring that of every observation
would discard the most recent year for every ticker, at every horizon. Instead
the labels are built from FIRST PASSAGE TIME: tau_d is the bar at which the fall
first reaches d, or infinity. Then

    label(h, d) = 1  if tau_d <= bars(h)

and a 1 is known as soon as the event happens, even if the window is incomplete.
Only a 0 needs the full window. An observation with 200 forward bars therefore
still contributes every (h, d) cell whose event already fired, plus every cell
whose horizon fits. That is standard right-censoring, and it recovers most of
what a naive filter would drop.

THE EVENT DEFINITION IS THE ONE THE DEPLOYED MODEL USES

Fall from the entry CLOSE, measured against subsequent intraday LOWS, over a
window counted in CALENDAR days. V60 was built on a different definition by
accident - peak-to-trough on closes - and produced a 6.5pp error that looked
like a finding. verify_against_p2() below re-derives P2's own label on the same
rows and refuses to run if the two disagree.

WHAT THIS MODULE DOES NOT DO

It does not forecast direction. Nothing here estimates whether a price goes up.

USAGE
  python class_ai_pillar2_surface_v63.py            # fit, then show a term structure
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2

# =============================================================================
# CONFIG
# =============================================================================
# The grid the model is fitted and served on. Calendar days, matching
# HORIZON_DAYS in the deployed module so 90 is directly comparable.
HORIZON_GRID = [14, 30, 60, 90, 180, 365]
THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# SHORT in trade_config is 14 days on HOURLY bars. This grid is daily
# throughout, so 14 here is "14 calendar days of daily bars" - about 10 bars.
# With ten bars the forward window is thin and the event is rare; treat the
# shortest horizon as indicative until hourly data exists.
MIN_BARS_FOR_HORIZON = 6

PRICE_CACHE = P2.PRICE_CACHE
MODEL_FILE = "pillar2_surface_v63.json"
STEP = 10                   # trading days between observations
MIN_CELL_OBS = 200          # refuse to fit a cell thinner than this

# Stage 3. Deliberately small: the correction is a correction, not the model.
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.9,
                  reg_lambda=5.0, min_child_weight=100,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")

# Feature order for stage 3, and the monotonicity each one carries.
#   h_bars    longer window -> event at least as likely      -> +1
#   drawdown  deeper fall   -> event no more likely          -> -1
#   the volatility features are left free: sigma_h already carries the
#   volatility effect through the base margin, and the correction is allowed to
#   go either way in them.
G_FEATURES = ["yz5", "yz22", "yz66", "sigma_h", "h_bars", "drawdown"]
G_MONOTONE = (0, 0, 0, 1, 1, -1)

PROB_FLOOR = 0.005          # never report a probability below this
PROB_CEIL = 0.995


# =============================================================================
# CLOSED FORM  (stage 2)
# =============================================================================
def _phi(x):
    """Standard normal CDF, vectorised, no scipy dependency."""
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def barrier_prob(drawdown, sigma_h):
    """
    P(driftless GBM touches a barrier `drawdown` below its start within a window
    whose total return volatility is `sigma_h`), by the reflection principle.

        drawdown  0.20 for a 20% fall, as a fraction
        sigma_h   volatility OVER THE WINDOW, as a fraction (not annualised)

    Returns 0 where sigma_h is not positive: no volatility, no barrier touch.
    """
    d = np.asarray(drawdown, float)
    s = np.asarray(sigma_h, float)
    out = np.zeros(np.broadcast(d, s).shape, float)
    ok = (s > 0) & (d > 0) & (d < 1)
    if not np.any(ok):
        return out if out.shape else float(out)
    z = np.log(1.0 - np.broadcast_to(d, out.shape)[ok]) / np.broadcast_to(s, out.shape)[ok]
    out[ok] = np.clip(2.0 * _phi(z), 1e-9, 1 - 1e-9)
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# =============================================================================
# LABELS  (first passage)
# =============================================================================
def first_passage(low, entry, thresholds):
    """
    For each threshold, the number of bars until the fall from `entry` first
    reaches it. `low` is the forward series of intraday lows, already sliced to
    start at the bar AFTER the observation.

    Returns (tau, n_fwd) where tau[k] is a bar count (1-based) or np.inf, and
    n_fwd is how many forward bars were available.

    Using the running minimum means one pass gives every threshold, and every
    horizon, at once.
    """
    low = np.asarray(low, float)
    n_fwd = len(low)
    tau = np.full(len(thresholds), np.inf)
    if n_fwd == 0:
        return tau, 0
    run_min = np.minimum.accumulate(low)
    fall = (run_min - entry) / entry          # most negative so far, per bar
    for k, d in enumerate(thresholds):
        hit = np.flatnonzero(fall <= -d)
        if hit.size:
            tau[k] = float(hit[0] + 1)       # 1-based: bar 1 is the next bar
    return tau, n_fwd


def _bars(days):
    return P2._fwd_bars(days)


def build_surface_observations(tickers, cache_dir=None, step=STEP,
                               horizons=None, thresholds=None, verbose=True):
    """
    One row per (ticker, date, horizon, threshold) with the features, the
    closed-form input, and the label - censored cells dropped, resolved cells
    kept even when the window is short.

    The forward LOOKBACK for features and the forward WINDOW for labels never
    overlap: features come from bars up to and including `pos`, labels from
    `pos + 1` onward.
    """
    cache_dir = cache_dir or PRICE_CACHE
    horizons = list(horizons or HORIZON_GRID)
    thresholds = list(thresholds or THRESHOLD_GRID)
    h_bars = {h: _bars(h) for h in horizons}
    max_bars = max(h_bars.values())

    rows = []
    for i, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache_dir)
        if df is None or len(df) < P2.MIN_BARS + MIN_BARS_FOR_HORIZON + 2:
            continue
        close = df["Close"].to_numpy(float)
        low = df["Low"].to_numpy(float)
        dates = df.index
        dnp = dates.to_numpy()          # hoisted: searchsorted runs per bar
        vpanel = P2.vol_feature_panel(df)
        varr = vpanel[P2.VOL_FEATURES].to_numpy(float)

        for pos in range(P2.MIN_BARS, len(df) - MIN_BARS_FOR_HORIZON - 1, step):
            vrow = varr[pos]
            if not np.isfinite(vrow).all() or (vrow <= 0).any():
                continue
            entry = close[pos]
            if not np.isfinite(entry) or entry <= 0:
                continue

            # The window is defined in CALENDAR days and holds every trading bar
            # inside it - which is not the same as a fixed bar count, because 90
            # calendar days can contain 61 or 64 trading days depending on where
            # the holidays fall. P2 slices by date alone; capping at the
            # estimated bar count as well disagreed with it on 0.6% of rows,
            # every one an event that landed on a bar the cap had cut off.
            end = dates[pos] + pd.Timedelta(days=max(horizons))
            j = int(np.searchsorted(dnp, np.datetime64(end), "right"))
            tau, n_fwd = first_passage(low[pos + 1:j], entry, thresholds)
            if n_fwd < MIN_BARS_FOR_HORIZON:
                continue

            fwd_close = close[pos + 1:j]
            base = {"ticker": t, "date": dates[pos]}
            base.update(dict(zip(P2.VOL_FEATURES, vrow)))

            for h in horizons:
                end_h = dates[pos] + pd.Timedelta(days=h)
                jh = int(np.searchsorted(dnp, np.datetime64(end_h), "right"))
                nb_avail = min(jh - (pos + 1), n_fwd)
                if nb_avail < MIN_BARS_FOR_HORIZON:
                    continue
                # the window elapsed only if the series reaches past its end;
                # otherwise a non-event is unknown, not a zero
                complete = bool(dates[-1] >= end_h)
                seg = fwd_close[:nb_avail]
                if len(seg) < 3:
                    continue
                fv = float(pd.Series(seg).pct_change().std()
                           * np.sqrt(252) * 100)

                for k, d in enumerate(thresholds):
                    hit = bool(tau[k] <= nb_avail)
                    if not hit and not complete:
                        continue          # censored: we cannot call this a zero
                    # h_bars is the NOMINAL bar count for the horizon, not the
                    # realised one. It has to be a deterministic function of h so
                    # that sigma_h is scaled identically at fit and predict time.
                    rows.append({**base, "horizon_days": h,
                                 "h_bars": h_bars[h],
                                 "bars_in_window": int(nb_avail),
                                 "drawdown": d, "fwd_vol": fv,
                                 "event": hit,
                                 "tau": float(tau[k]),
                                 "complete": complete})
        if verbose and (i % 20 == 0 or i == len(tickers)):
            print(f"    [{i}/{len(tickers)}] {len(rows):,} cells")
    return pd.DataFrame(rows)


# =============================================================================
# STAGE 1  volatility forecast, per horizon
# =============================================================================
def fit_vol_term_structure(obs, features=None, verbose=True):
    """
    One OLS per horizon: trailing Yang-Zhang at three scales -> realised forward
    volatility over that horizon. Returns {horizon_days: {intercept, coef, r2, n}}.

    The point of fitting separately is the coefficient SUM. Volatility mean-
    reverts, so the sum should fall as the horizon lengthens - a one-year
    forecast leans on today's reading less than a one-month forecast does. If it
    does not fall, the mean reversion is not there and sqrt(T) scaling would have
    been fine.
    """
    features = list(features or P2.VOL_FEATURES)
    out = {}
    # one row per (ticker, date, horizon): the vol target does not vary by
    # threshold, so collapsing first avoids weighting a date by its cell count
    uniq = obs.drop_duplicates(subset=["ticker", "date", "horizon_days"])
    for h, g in uniq.groupby("horizon_days"):
        g = g[np.isfinite(g["fwd_vol"])]
        if len(g) < 100:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float)
                                                for f in features])
        y = g["fwd_vol"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss = ((y - y.mean()) ** 2).sum()
        out[int(h)] = {"intercept": float(beta[0]),
                       "coef": [float(v) for v in beta[1:]],
                       "features": features,
                       "r2": float(1 - ((y - pred) ** 2).sum() / ss) if ss else 0.0,
                       "n": int(len(g))}
    if verbose:
        print("\n  STAGE 1  volatility forecast per horizon")
        print(f"    {'horizon':>8s} {'intercept':>10s} "
              + " ".join(f"{f:>8s}" for f in features)
              + f" {'sum':>7s} {'R2':>6s} {'n':>8s}")
        for h in sorted(out):
            s = out[h]
            print(f"    {h:6d}d {s['intercept']:10.2f} "
                  + " ".join(f"{c:8.3f}" for c in s["coef"])
                  + f" {sum(s['coef']):7.3f} {s['r2']:6.3f} {s['n']:8,d}")
        sums = [sum(out[h]["coef"]) for h in sorted(out)]
        if len(sums) > 1:
            trend = "falls" if sums[-1] < sums[0] else "does NOT fall"
            print(f"    coefficient sum {trend} with the horizon "
                  f"({sums[0]:.3f} at {min(out)}d -> {sums[-1]:.3f} "
                  f"at {max(out)}d)")
            if sums[-1] >= sums[0]:
                print("    -> mean reversion is not showing here; sqrt(T) "
                      "scaling from one annual forecast would do as well")
    return out


def forecast_vol_h(feats, vol_ts, horizon_days):
    """Annualised forward volatility (percent) for one horizon."""
    s = vol_ts.get(int(horizon_days)) or vol_ts.get(str(horizon_days))
    if s is None:
        return np.nan
    v = s["intercept"]
    for f, c in zip(s["features"], s["coef"]):
        v += c * float(feats[f])
    return float(v)


def _sigma_h(fvol_annual_pct, h_bars):
    """Annualised percent -> window volatility as a fraction."""
    return np.asarray(fvol_annual_pct, float) / 100.0 * np.sqrt(
        np.asarray(h_bars, float) / 252.0)


# =============================================================================
# STAGE 3  the learned correction
# =============================================================================
def forecast_vol_h_vec(obs, vol_ts, horizon_col="horizon_days"):
    """
    Vectorised stage 1. One dot product per horizon instead of one Python call
    per row: at 780k cells the row-by-row version took minutes, and the real
    universe is five times larger.
    """
    fv = np.full(len(obs), np.nan)
    h_arr = obs[horizon_col].to_numpy()
    for h in np.unique(h_arr):
        s = vol_ts.get(int(h)) or vol_ts.get(str(int(h)))
        if s is None:
            continue
        m = h_arr == h
        X = obs.loc[m, s["features"]].to_numpy(float)
        fv[m] = s["intercept"] + X @ np.asarray(s["coef"], float)
    return fv


def _design(obs, vol_ts):
    """Feature matrix, base margin and label for stage 3."""
    fv = forecast_vol_h_vec(obs, vol_ts)
    sig = _sigma_h(fv, obs["h_bars"].to_numpy(float))
    p_th = barrier_prob(obs["drawdown"].to_numpy(float), sig)
    X = pd.DataFrame({"yz5": obs["yz5"].to_numpy(float),
                      "yz22": obs["yz22"].to_numpy(float),
                      "yz66": obs["yz66"].to_numpy(float),
                      "sigma_h": sig,
                      "h_bars": obs["h_bars"].to_numpy(float),
                      "drawdown": obs["drawdown"].to_numpy(float)})[G_FEATURES]
    y = obs["event"].to_numpy(int)
    return X, _logit(p_th), y, p_th, sig, fv


def _cell_weights(obs):
    """
    Each (ticker, date) contributes many cells but only one date's worth of
    information, so its cells share a total weight of 1. Without this the loss
    is dominated by whichever dates happened to resolve the most cells, and the
    model looks far more certain than the data supports.
    """
    n = obs.groupby(["ticker", "date"])["event"].transform("size")
    return (1.0 / n).to_numpy(float)


def fit_surface(obs, vol_ts=None, params=None, verbose=True):
    """Fit stage 1 (unless given) and stage 3. Returns the model dict."""
    from xgboost import XGBClassifier

    vol_ts = vol_ts or fit_vol_term_structure(obs, verbose=verbose)
    X, margin, y, p_th, _, _ = _design(obs, vol_ts)
    w = _cell_weights(obs)

    ok = np.isfinite(X.to_numpy(float)).all(axis=1) & np.isfinite(margin)
    X, margin, y, w, p_th = X[ok], margin[ok], y[ok], w[ok], p_th[ok]
    if len(X) < MIN_CELL_OBS:
        raise RuntimeError(f"only {len(X)} usable cells; need {MIN_CELL_OBS}")

    p = dict(XGB_PARAMS)
    p.update(params or {})
    p["monotone_constraints"] = "(" + ",".join(str(c) for c in G_MONOTONE) + ")"
    booster = XGBClassifier(**p)
    booster.fit(X, y, sample_weight=w, base_margin=margin, verbose=False)

    if verbose:
        print("\n  STAGE 3  learned correction")
        print(f"    {len(X):,} cells, {obs.groupby(['ticker','date']).ngroups:,} "
              f"distinct (ticker, date)")
        print(f"    monotone: h_bars +1, drawdown -1 "
              f"(so the term structure cannot invert)")
        imp = sorted(zip(G_FEATURES, booster.feature_importances_),
                     key=lambda t: -t[1])
        print("    gain share: " + "  ".join(f"{k} {v:.2f}" for k, v in imp))

    return {"vol_term_structure": vol_ts,
            "booster": booster,
            "g_features": G_FEATURES,
            "horizons": sorted(set(int(h) for h in obs["horizon_days"])),
            "thresholds": sorted(set(float(d) for d in obs["drawdown"])),
            "n_cells": int(len(X)),
            "n_dates": int(obs.groupby(["ticker", "date"]).ngroups),
            "base_rate_by_cell": {
                f"{int(h)}d_{int(d*100)}pct": float(gg["event"].mean())
                for (h, d), gg in obs.groupby(["horizon_days", "drawdown"])},
            }


# =============================================================================
# PREDICTION
# =============================================================================
def predict_cells(feats, model, horizons=None, thresholds=None):
    """
    The surface for one set of features. Returns a DataFrame with the closed
    form and the model side by side, so the correction is always visible.
    """
    horizons = list(horizons or model["horizons"])
    thresholds = list(thresholds or model["thresholds"])
    grid = [(h, d) for h in horizons for d in thresholds]
    rec = []
    for h, d in grid:
        fv = forecast_vol_h(feats, model["vol_term_structure"], h)
        nb = _bars(h)
        sig = float(_sigma_h(fv, nb))
        rec.append({"horizon_days": h, "h_bars": nb, "drawdown": d,
                    "forecast_vol_annual_pct": fv, "sigma_h": sig,
                    "yz5": float(feats["yz5"]), "yz22": float(feats["yz22"]),
                    "yz66": float(feats["yz66"])})
    df = pd.DataFrame(rec)
    df["p_theory"] = barrier_prob(df["drawdown"].to_numpy(float),
                                 df["sigma_h"].to_numpy(float))
    X = df[model["g_features"]]
    margin = _logit(df["p_theory"].to_numpy(float))
    raw = model["booster"].predict_proba(X, base_margin=margin)[:, 1]
    df["p_model"] = np.clip(raw, PROB_FLOOR, PROB_CEIL)
    df["correction_pp"] = (df["p_model"] - df["p_theory"]) * 100
    return df


def term_structure(ticker, model, target_date=None, drawdown=0.20,
                   price_cache=None, horizons=None):
    """
    The product-facing call: one ticker, one threshold, every horizon.

        ts = term_structure("NVDA", model, drawdown=0.20)
        ts[["horizon_days", "p_model"]]
    """
    df = P2.load_prices(ticker, price_cache or PRICE_CACHE)
    if df is None or len(df) < P2.MIN_BARS:
        return None
    if target_date is not None:
        df = df[df.index <= pd.Timestamp(target_date)]
        if len(df) < P2.MIN_BARS:
            return None
    feats = P2.vol_features_at(df, -1)
    if feats is None:
        return None
    out = predict_cells(feats, model, horizons=horizons, thresholds=[drawdown])
    out.insert(0, "ticker", ticker)
    out.insert(1, "as_of", str(df.index[-1].date()))
    out.insert(2, "price", round(float(df["Close"].iloc[-1]), 4))
    return out


# =============================================================================
# INVARIANTS
# =============================================================================
def monotone_violations(df, key=("ticker", "drawdown"), value="p_model"):
    """Cells where the probability falls as the horizon lengthens."""
    bad = 0
    for _, g in df.groupby(list(key)):
        v = g.sort_values("horizon_days")[value].to_numpy(float)
        bad += int((np.diff(v) < -1e-9).sum())
    return bad


def threshold_violations(df, value="p_model"):
    """Cells where the probability RISES as the required fall gets deeper."""
    bad = 0
    for _, g in df.groupby("horizon_days"):
        v = g.sort_values("drawdown")[value].to_numpy(float)
        bad += int((np.diff(v) > 1e-9).sum())
    return bad


def verify_against_p2(tickers, cache_dir=None, horizon_days=None,
                      drawdown=None, step=40, verbose=True):
    """
    Re-derive P2's own label on the same rows and refuse to continue if this
    module's first-passage label disagrees. V60 shipped a 6.5pp "finding" that
    was nothing but a different event definition; this is the guard against a
    repeat.
    """
    horizon_days = horizon_days or P2.HORIZON_DAYS
    drawdown = drawdown or P2.DRAWDOWN
    cache_dir = cache_dir or PRICE_CACHE

    p2 = P2.build_observations(tickers, cache_dir, step=step,
                              horizon_days=horizon_days, drawdown=drawdown,
                              verbose=False)
    mine = build_surface_observations(tickers, cache_dir, step=step,
                                      horizons=[horizon_days],
                                      thresholds=[drawdown], verbose=False)
    if p2.empty or mine.empty:
        raise RuntimeError("event-definition check produced no rows")
    j = p2[["ticker", "date", "event"]].merge(
        mine[["ticker", "date", "event"]], on=["ticker", "date"],
        suffixes=("_p2", "_v63"))
    if j.empty:
        raise RuntimeError("event-definition check found no shared rows")
    agree = (j["event_p2"] == j["event_v63"]).mean()
    if verbose:
        print(f"  event definition vs P2: {len(j):,} shared rows, "
              f"{agree*100:.2f}% agreement "
              f"(P2 rate {j['event_p2'].mean():.4f}, "
              f"v63 rate {j['event_v63'].mean():.4f})")
    if agree < 0.999:
        raise RuntimeError(
            f"first-passage labels disagree with P2 on "
            f"{(1-agree)*100:.2f}% of rows - fix the definition before fitting")
    return agree


# =============================================================================
# I/O
# =============================================================================
def save_model(model, path=MODEL_FILE):
    booster_path = os.path.splitext(path)[0] + "_booster.json"
    model["booster"].save_model(booster_path)
    meta = {k: v for k, v in model.items() if k != "booster"}
    meta["booster_file"] = booster_path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    return path


def load_model(path=MODEL_FILE):
    from xgboost import XGBClassifier
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found. Fit the model first.")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    b = XGBClassifier()
    b.load_model(meta["booster_file"])
    meta["booster"] = b
    meta["vol_term_structure"] = {int(k): v for k, v in
                                  meta["vol_term_structure"].items()}
    return meta


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_fit(universe="all", price_cache=None, step=STEP, horizons=None,
            thresholds=None, out=MODEL_FILE, check_definition=True,
            verbose=True):
    """Build observations, fit all three stages, save. Returns the model."""
    price_cache = price_cache or PRICE_CACHE
    tickers = P2.universe_tickers(universe) if hasattr(P2, "universe_tickers") \
        else _tickers_from_cache(price_cache)

    if verbose:
        print("=" * 88)
        print("V63 - MULTI-HORIZON RISK SURFACE")
        print("=" * 88)
        print(f"  {len(tickers)} tickers | step {step} bars")
        print(f"  horizons   {horizons or HORIZON_GRID}")
        print(f"  thresholds {[f'{d:.0%}' for d in (thresholds or THRESHOLD_GRID)]}")

    if check_definition:
        verify_against_p2(tickers[:25], price_cache, verbose=verbose)

    obs = build_surface_observations(tickers, price_cache, step=step,
                                     horizons=horizons, thresholds=thresholds,
                                     verbose=verbose)
    if obs.empty:
        raise RuntimeError("no observations built")
    if verbose:
        print(f"\n  {len(obs):,} cells from "
              f"{obs.groupby(['ticker','date']).ngroups:,} (ticker, date) pairs")
        print(f"  censoring kept "
              f"{100*(~obs['complete']).mean():.1f}% of cells that a "
              f"complete-window filter would have dropped")
        print("\n  BASE RATE BY CELL (what the model has to beat by knowing nothing)")
        piv = obs.pivot_table(index="horizon_days", columns="drawdown",
                              values="event", aggfunc="mean")
        print("    " + piv.round(3).to_string().replace("\n", "\n    "))

    model = fit_surface(obs, verbose=verbose)
    save_model(model, out)
    if verbose:
        print(f"\n  wrote {out}")
    return model, obs


def _tickers_from_cache(cache_dir):
    if not os.path.isdir(cache_dir):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(cache_dir)
                  if f.endswith(".pkl"))


def describe_term_structure(ts, out=sys.stdout):
    def w(*a):
        print(*a, file=out)
    if ts is None or ts.empty:
        w("  no term structure (not enough history)")
        return
    r0 = ts.iloc[0]
    w("")
    w("=" * 68)
    w(f"  {r0['ticker']}  as of {r0['as_of']}  ${r0['price']}")
    w(f"  probability of a fall of {r0['drawdown']:.0%} or more")
    w("=" * 68)
    w(f"  {'horizon':>9s} {'bars':>5s} {'fcast vol':>10s} "
      f"{'window vol':>11s} {'closed form':>12s} {'MODEL':>8s} {'correction':>11s}")
    for _, r in ts.sort_values("horizon_days").iterrows():
        w(f"  {int(r['horizon_days']):8d}d {int(r['h_bars']):5d} "
          f"{r['forecast_vol_annual_pct']:9.1f}% {r['sigma_h']*100:10.1f}% "
          f"{r['p_theory']*100:11.1f}% {r['p_model']*100:7.1f}% "
          f"{r['correction_pp']:+10.1f}pp")
    bad = monotone_violations(ts, key=("ticker", "drawdown"))
    w(f"\n  monotone in the horizon: {'yes' if bad == 0 else f'NO ({bad})'}")


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--ticker")
    ap.add_argument("--drawdown", type=float, default=0.20)
    ap.add_argument("--universe", default="all")
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--out", default=MODEL_FILE)
    ap.add_argument("--no-check", action="store_true")
    return ap


def main(argv=None):
    cfg = _parser().parse_args(argv)
    if cfg.fit:
        run_fit(universe=cfg.universe, price_cache=cfg.price_cache,
                step=cfg.step, out=cfg.out, check_definition=not cfg.no_check)
        return 0
    if not cfg.ticker:
        sys.exit("Pass --ticker, or --fit to train first.")
    model = load_model(cfg.out)
    describe_term_structure(term_structure(cfg.ticker, model,
                                          drawdown=cfg.drawdown,
                                          price_cache=cfg.price_cache))
    return 0


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. RUN_MODE "fit" trains and saves,
# "show" loads the saved model and prints one ticker's term structure, "both"
# does the two in order (what you want the first time).
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODE        = "both"            # "both" | "fit" | "show"

    RUN_UNIVERSE    = "all"
    RUN_PRICE_CACHE = PRICE_CACHE
    RUN_STEP        = STEP              # trading days between observations
    RUN_HORIZONS    = HORIZON_GRID      # calendar days
    RUN_THRESHOLDS  = THRESHOLD_GRID    # as fractions
    RUN_CHECK_DEF   = True              # cross-check the event label against P2
    RUN_OUT         = MODEL_FILE

    RUN_TICKER      = "NVDA"            # for "show"
    RUN_DRAWDOWN    = 0.20              # which threshold's term structure
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())                # e.g. python class_ai_pillar2_surface_v63.py --fit

    if RUN_MODE in ("fit", "both"):
        run_fit(universe=RUN_UNIVERSE, price_cache=RUN_PRICE_CACHE,
                step=RUN_STEP, horizons=RUN_HORIZONS,
                thresholds=RUN_THRESHOLDS, out=RUN_OUT,
                check_definition=RUN_CHECK_DEF)

    if RUN_MODE in ("show", "both"):
        _model = load_model(RUN_OUT)
        describe_term_structure(
            term_structure(RUN_TICKER, _model, drawdown=RUN_DRAWDOWN,
                           price_cache=RUN_PRICE_CACHE))
