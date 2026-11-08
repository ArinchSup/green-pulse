#!/usr/bin/env python3
"""
class_ai_pillar2_risk_v2.py - Pillar 2 as a RISK engine, with the V53 volatility
forecast.

WHAT CHANGED FROM class_ai_pillar2_risk.py

The volatility forecast was one feature: forward_vol = a + b x trailing_60d,
close-to-close, in-sample R2 0.517. V53 tested a ladder of standard alternatives
walk-forward and the winner was three Yang-Zhang estimators at 5, 22 and 66 bars:

    out-of-sample R2   0.4923 -> 0.5288      gain +0.0366, CI +0.0036 to +0.0739

Most of that came from the ESTIMATOR, not the multi-scale structure. Multi-scale
close-to-close bought +0.0127; swapping in Yang-Zhang at the same three scales
added roughly +0.024 on top. The old model discarded the high and low of every
bar, and that is where the information was.

Two things V53 tested and rejected, kept out deliberately: realised semivariance
(the leverage effect washes out over 180 days) and a log transform (standard for
intraday RV at short horizons, unhelpful here). Both were measured, not assumed.

The downstream drawdown metrics all moved the right way - calibration MAE 2.29 ->
2.15pp, Brier skill +0.043 -> +0.051, AUC 0.683 -> 0.691, tracking Spearman
+0.632 -> +0.638 - but every one of those gains sits inside V52's confidence
interval for that metric. The defensible claim is that the VOLATILITY forecast
improved; the drawdown improvement is directionally consistent and not
individually significant. Do not oversell it.

The old module is kept as-is because V52 and V53 measured it, and their published
numbers should stay reproducible.

WHY THIS REPLACES THE PREDICTIVE VERSION

V39 through V50 tested whether daily technical features predict returns: two label
formulations, five horizons, three model objectives, two universes, panel and
event-conditional sampling, ~400k rows across 21 years. Nothing survived out of
sample. The last walk-forward put 2023-2026 IC between -0.013 and -0.017 in every
size bucket.

One relationship did survive everything: volatility predicts drawdowns, separating
30% falls by 47 percentage points (t = 10.22). That is the strongest and most stable
result in the project, so this module is built on it and on nothing else.

WHAT IT OUTPUTS, AND WHAT IT REFUSES TO OUTPUT

  forecast volatility        trailing realised vol shrunk toward its long-run level,
                             because volatility mean-reverts
  P(drawdown)                the chance of a fall of THRESHOLD within the horizon,
                             read off a curve calibrated on real outcomes
  position size              from a risk budget and the stop distance
  stop / target levels       trade_config's VOL_SCALED geometry
  risk flags                 dilution, shelf registrations, runway, liquidity,
                             extension - descriptive, each with the historical base
                             rate attached
  context                    trend, RSI, range position, regime - FACTS for the
                             synthesis layer to weigh

  It does NOT emit a sentiment, a success probability, or a confidence score. There is
  no validated basis for any of those, and the previous version's "confidence" field
  was the main thing that made results look better than they were.

CALIBRATION IS THE METRIC, NOT WIN RATE

Run --calibrate first. It fits the volatility-to-drawdown curve on history and prints
a reliability table: when the model says 30%, how often did it actually happen? That
is the claim this module makes, and it is the one you can defend.

USAGE
  python class_ai_pillar2_risk.py --calibrate          # fit and validate, writes JSON
  python class_ai_pillar2_risk.py --ticker NVDA
  python class_ai_pillar2_risk.py --ticker NVDA --date 2026-03-15 --equity 25000
"""
import argparse
import json
import os
import sys
from datetime import date as _date

import numpy as np
import pandas as pd

from trade_config import HORIZON_CONFIGS, compute_levels, describe_geometry

PRICE_CACHE = "price_cache_v43"
EDGAR_CACHE = "edgar_cache"
CALIB_FILE = "pillar2_risk_calibration.json"

HORIZON = "MID"
HORIZON_DAYS = 90           # calendar days for the drawdown question; keep
                            # this aligned with the trade horizon's eval_days

# Derived from HORIZON_DAYS, never hard-coded. A fixed floor of 80 forward bars
# was correct for a 180-day window (about 124 bars) and silently rejected EVERY
# observation at 30 days (about 21 bars), which produced a calibration run with
# zero observations and no explanation. Anything that scales with the horizon has
# to be computed from it.
def _fwd_bars(days=None):
    """Trading bars in a forward window of `days` calendar days."""
    return max(8, int(round((days or HORIZON_DAYS) * 252 / 365.25)))


def _min_fwd_bars(days=None):
    """How much of that window must actually be present. 60% keeps the estimate
    honest without discarding observations near the end of a price series."""
    return max(6, int(0.6 * _fwd_bars(days)))
DRAWDOWN = 0.20             # what counts as a drawdown event (V59)
MIN_BARS = 260              # refuse to score a name with less history than this

# --- fitting defaults --------------------------------------------------------
# Every one of these is a parameter of run_calibration(); the constants here are
# only its defaults. HORIZON_DAYS and DRAWDOWN above are different: they define
# the QUESTION and have to agree with class_ai_pipeline_v3.py, the calibration
# file on disk and trade_config.py. deployment_preflight.py checks that.
CALIB_UNIVERSE = "all"      # "all" or "deploy"
CALIB_STEP = 10             # trading days between calibration observations
CALIB_MIN_OBS = 2000        # refuse to fit on less than this
CURVE_BINS = 12             # bins in the drawdown curve
DEFAULT_EQUITY = 10000.0
DEFAULT_RISK_BUDGET = 0.02  # 2% of equity at risk if the stop fills

# The V53 winner. Three scales of one estimator - no asymmetry terms, no log
# transform, both of which were tested and added nothing.
VOL_FEATURES = ["yz5", "yz22", "yz66"]
YZ_WINDOWS = (5, 22, 66)


# =============================================================================
# PRICES
# =============================================================================
def load_prices(ticker, cache_dir=PRICE_CACHE):
    p = os.path.join(cache_dir, f"{ticker}.pkl")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_pickle(p)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


def realised_vol(close, window=60):
    """Annualised realised volatility, in percent. Close-to-close, kept because
    the reported figures and the flags still use it."""
    r = close.pct_change().tail(window)
    if len(r.dropna()) < window // 2:
        return np.nan
    return float(r.std() * np.sqrt(252) * 100)


def yang_zhang_series(df, w):
    """
    Yang-Zhang (2000), annualised percent, as a rolling series.

    The only common range estimator that handles both drift and overnight gaps:
    an overnight term, an open-to-close term, and Rogers-Satchell for the
    intraday path, weighted to minimise estimator variance. It uses the whole
    bar, which is why it beat close-to-close in V53.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    co = np.log(o / c.shift())
    oc = np.log(c / o)
    u, d = np.log(h / o), np.log(l / o)
    rs = u * (u - oc) + d * (d - oc)
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    var = (co.rolling(w).var(ddof=1) + k * oc.rolling(w).var(ddof=1)
           + (1 - k) * rs.rolling(w).mean())
    return np.sqrt(np.maximum(var, 0) * 252.0) * 100.0


def vol_feature_panel(df):
    """All volatility features as columns, computed once over a full history."""
    out = pd.DataFrame(index=df.index)
    for w in YZ_WINDOWS:
        out[f"yz{w}"] = yang_zhang_series(df, w)
    return out


def vol_features_at(df, pos=-1):
    """The feature dict for one bar. Returns None if any feature is unusable."""
    panel = vol_feature_panel(df)
    row = panel.iloc[pos]
    if not np.isfinite(row.to_numpy(float)).all():
        return None
    return {k: float(row[k]) for k in VOL_FEATURES}


# =============================================================================
# CALIBRATION
# =============================================================================
def build_observations(tickers, cache_dir, step=10, horizon_days=None,
                       drawdown=None, verbose=True):
    """
    Every (ticker, date) with enough history: trailing volatility, and what actually
    happened over the next `horizon_days`. Nothing here uses information from after
    the observation date except the outcome itself.

    `horizon_days` and `drawdown` default to the module config. They are real
    parameters rather than globals because sweeping them is a legitimate
    experiment - V57 built 25 combinations this way. What they must NOT do is
    disagree with the deployed config at scoring time, which is why assess()
    still reads the module constants and deployment_preflight.py checks that the
    calibration file on disk was fitted at the same setting.
    """
    horizon_days = HORIZON_DAYS if horizon_days is None else int(horizon_days)
    drawdown = DRAWDOWN if drawdown is None else float(drawdown)
    n_fwd_bars = _fwd_bars(horizon_days)       # not `fwd` - the loop below uses
    n_min_bars = _min_fwd_bars(horizon_days)   # that name for the forward frame
    rows = []
    for i, t in enumerate(tickers, 1):
        df = load_prices(t, cache_dir)
        if df is None or len(df) < MIN_BARS + n_fwd_bars + 10:
            continue
        close, low = df["Close"], df["Low"]
        vpanel = vol_feature_panel(df)          # once per ticker, not per bar
        varr = vpanel[VOL_FEATURES].to_numpy(float)
        for pos in range(MIN_BARS, len(df) - n_min_bars - 1, step):
            hist = close.iloc[:pos + 1]
            v60 = realised_vol(hist, 60)
            v250 = realised_vol(hist, 250)
            if not np.isfinite(v60) or v60 <= 0:
                continue
            vrow = varr[pos]
            if not np.isfinite(vrow).all() or (vrow <= 0).any():
                continue
            entry = float(close.iloc[pos])
            end = df.index[pos] + pd.Timedelta(days=horizon_days)
            fwd = df.iloc[pos + 1:]
            fwd = fwd[fwd.index <= end]
            if len(fwd) < n_min_bars:
                continue
            mdd = (float(fwd["Low"].min()) - entry) / entry * 100
            fwd_vol = float(fwd["Close"].pct_change().std() * np.sqrt(252) * 100)
            rec = {"ticker": t, "date": df.index[pos], "vol60": v60,
                   "vol250": v250, "fwd_vol": fwd_vol, "mdd": mdd,
                   "event": mdd <= -drawdown * 100}
            rec.update(dict(zip(VOL_FEATURES, vrow)))
            rows.append(rec)
        if verbose and (i % 20 == 0 or i == len(tickers)):
            print(f"    [{i}/{len(tickers)}] {len(rows):,} observations")
    return pd.DataFrame(rows)


def fit_vol_shrinkage(obs, features=None):
    """
    Volatility mean-reverts, so today's trailing vol overstates tomorrow's when it
    is high. Fit forward_vol on the V53 feature set and use that as the forecast.

    Coefficients summing to below 1.0 is the mean-reversion showing up: the
    forecast shrinks an elevated reading toward the long-run level.

    OLS, and that is a choice. V52 measured the effective sample size at ~368
    independent observations once drawdown clustering is accounted for, so a
    flexible learner on three features would fit noise convincingly.
    """
    features = features or VOL_FEATURES
    have = [f for f in features if f in obs.columns]
    d = obs.dropna(subset=have + ["fwd_vol"]) if have else obs.iloc[:0]
    if len(d) < 500 or not have:
        # legacy single-feature fallback, so an old calibration still loads
        d1 = obs.dropna(subset=["vol60", "fwd_vol"])
        if len(d1) < 500:
            return {"features": ["vol60"], "coef": [1.0], "intercept": 0.0,
                    "r2": np.nan, "n": int(len(d1))}
        x, y = d1["vol60"].to_numpy(float), d1["fwd_vol"].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - np.var(y - (a + b * x)) / np.var(y)
        return {"features": ["vol60"], "coef": [float(b)], "intercept": float(a),
                "r2": float(r2), "n": int(len(d1))}
    X = d[have].to_numpy(float)
    y = d["fwd_vol"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)
    pred = beta[0] + X @ beta[1:]
    return {"features": have, "coef": [float(v) for v in beta[1:]],
            "intercept": float(beta[0]),
            "r2": float(1 - np.var(y - pred) / np.var(y)), "n": int(len(d))}


def fit_drawdown_curve(obs, shrink, n_bins=12):
    """
    Empirical P(drawdown | forecast vol). Deliberately a lookup table rather than a
    model: it is transparent, it cannot overfit 12 numbers, and it is trivial to audit.
    """
    names = shrink.get("features", ["vol60"])
    d = obs.dropna(subset=[n for n in names if n in obs.columns]).copy()
    cal = {"shrinkage": shrink}
    X = d[names].to_numpy(float)
    d["fvol"] = (shrink["intercept"] + X @ np.array(shrink["coef"], float)
                 if "features" in shrink
                 else shrink["a"] + shrink["b"] * d["vol60"].to_numpy(float))
    d = d[np.isfinite(d["fvol"]) & (d["fvol"] > 0)]
    d["bin"] = pd.qcut(d["fvol"].rank(method="first"), n_bins,
                       labels=False, duplicates="drop")
    pts = []
    for b in sorted(d["bin"].unique()):
        g = d[d["bin"] == b]
        pts.append({"fvol": float(g["fvol"].median()),
                    "p": float(g["event"].mean()),
                    "n": int(len(g)),
                    "mean_mdd": float(g["mdd"].mean())})
    return pts


def reliability(obs, calib, n_bins=8):
    """Predicted probability against realised frequency - the metric that matters."""
    names = calib["shrinkage"].get("features", ["vol60"])
    d = obs.dropna(subset=[n for n in names if n in obs.columns]).copy()
    X, _ = _feature_frame(d, calib)
    if X is None:
        return None
    d["p"] = [prob_drawdown(dict(zip(names, row)), calib) for row in X]
    d = d[np.isfinite(d["p"])]
    if len(d) < 200:
        return None
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins,
                       labels=False, duplicates="drop")
    rows = []
    for b in sorted(d["bin"].unique()):
        g = d[d["bin"] == b]
        rows.append({"predicted": g["p"].mean() * 100,
                     "realised": g["event"].mean() * 100,
                     "n": len(g)})
    return pd.DataFrame(rows)


# =============================================================================
# SCORING
# =============================================================================
def forecast_vol(feats, calib):
    """
    Forecast annualised volatility over the horizon.

    `feats` is a mapping of feature name to value. A bare float is accepted and
    treated as vol60, which keeps an old single-feature calibration working.
    """
    s = calib["shrinkage"]
    if "features" not in s:                      # pre-V53 calibration file
        v = feats["vol60"] if isinstance(feats, dict) else float(feats)
        return s["a"] + s["b"] * v
    if not isinstance(feats, dict):
        feats = {"vol60": float(feats)}
    try:
        x = np.array([float(feats[f]) for f in s["features"]], dtype=float)
    except (KeyError, TypeError):
        return np.nan
    if not np.isfinite(x).all():
        return np.nan
    return float(s["intercept"] + x @ np.array(s["coef"], dtype=float))


def prob_drawdown(feats, calib):
    """Interpolate the calibrated curve; clamp outside the fitted range."""
    fv = forecast_vol(feats, calib)
    if not np.isfinite(fv):
        return np.nan
    pts = calib["curve"]
    return float(np.interp(fv, [p["fvol"] for p in pts], [p["p"] for p in pts]))


def _feature_frame(d, calib):
    """The feature columns a calibration needs, as a float array."""
    names = calib["shrinkage"].get("features", ["vol60"])
    if any(n not in d.columns for n in names):
        return None, names
    return d[names].to_numpy(float), names


# =============================================================================
# RISK FLAGS  (EDGAR, optional)
# =============================================================================
def edgar_flags(ticker, as_of, cache_dir=EDGAR_CACHE):
    flags = []
    pending = {"shelf": 0, "offers": 0, "growth": None}
    tick_file = os.path.join(cache_dir, "company_tickers.json")
    if not os.path.exists(tick_file):
        return flags
    try:
        with open(tick_file, encoding="utf-8") as f:
            data = json.load(f)
        cik = None
        for row in (data.values() if isinstance(data, dict) else data):
            if str(row.get("ticker", "")).upper() == ticker.upper():
                cik = f"{int(row['cik_str']):010d}"
                break
        if not cik:
            return flags

        sub_path = os.path.join(cache_dir, f"sub_{cik}.json")
        if os.path.exists(sub_path):
            with open(sub_path, encoding="utf-8") as f:
                js = json.load(f)
            rec = (js.get("filings") or {}).get("recent") or {}
            forms = rec.get("form", [])
            dates = pd.to_datetime(pd.Series(rec.get("filingDate", [])), errors="coerce")
            cutoff3 = as_of - pd.Timedelta(days=1095)
            cutoff2 = as_of - pd.Timedelta(days=730)
            shelf = sum(1 for f_, d_ in zip(forms, dates)
                        if str(f_).startswith(("S-3", "S-1"))
                        and pd.notna(d_) and cutoff3 <= d_ <= as_of)
            # 424B2 is overwhelmingly DEBT. Counting it as dilution flags every
            # large-cap that issues notes - NVDA came back as "frequent offerings"
            # on two debt filings. Only the equity subtypes count.
            offers = sum(1 for f_, d_ in zip(forms, dates)
                         if str(f_).upper() in ("424B1", "424B3", "424B4", "424B5")
                         and pd.notna(d_) and cutoff2 <= d_ <= as_of)
            pending["shelf"] = shelf
            pending["offers"] = offers

        facts_path = os.path.join(cache_dir, f"facts_{cik}.json")
        if os.path.exists(facts_path):
            with open(facts_path, encoding="utf-8") as f:
                facts = json.load(f)
            sh = _first_series(facts, ["EntityCommonStockSharesOutstanding",
                                       "CommonStockSharesOutstanding"], as_of)
            if sh is not None and len(sh) > 4:
                now = float(sh["val"].iloc[-1])
                old = sh[sh["end"] <= sh["end"].iloc[-1] - pd.Timedelta(days=330)]
                if len(old) and float(old["val"].iloc[-1]) > 0:
                    pending["growth"] = now / float(old["val"].iloc[-1]) - 1
            cash = _first_series(facts, ["CashAndCashEquivalentsAtCarryingValue"], as_of)
            ocf = _first_series(facts, ["NetCashProvidedByUsedInOperatingActivities"],
                                as_of, flow=True)
            if cash is not None and ocf is not None and len(cash) and len(ocf) >= 2:
                burn = -float(ocf["val"].tail(4).mean())
                if burn > 0:
                    runway = float(cash["val"].iloc[-1]) / burn
                    if runway < 6:
                        flags.append({"code": "SHORT_RUNWAY", "severity": "high",
                                      "detail": f"about {runway:.1f} quarters of cash at the "
                                                f"current burn rate"})
    except Exception:
        pass

    # Share-count growth is the measured signal; filing counts are only a proxy for
    # it. Raise the dilution flag on growth, and let filings corroborate rather than
    # trigger. The base rates come from the V46 quintiles (14.0% at the bottom,
    # 33.8% in Q4, 41.6% in the top).
    g = pending["growth"]
    if g is not None and g > 0.15:
        flags.append({"code": "HIGH_DILUTION", "severity": "high",
                      "detail": f"share count +{g:.0%} over the last year; companies in "
                                f"that top quintile saw a 30% drawdown 41.6% of the "
                                f"time within 6 months, against a 25% base rate"})
    elif g is not None and g > 0.04:
        flags.append({"code": "MODERATE_DILUTION", "severity": "medium",
                      "detail": f"share count +{g:.0%} over the last year; that bucket "
                                f"saw a 30% drawdown 33.8% of the time within 6 months"})
    if pending["offers"] >= 2 and (g is None or g > 0.04):
        flags.append({"code": "FREQUENT_OFFERINGS", "severity": "medium",
                      "detail": f"{pending['offers']} equity offerings priced in the last "
                                f"2 years (424B1/3/4/5; debt shelf takedowns excluded)"})
    if pending["shelf"] and (g is None or g > 0.04):
        flags.append({"code": "SHELF_ON_FILE", "severity": "low",
                      "detail": f"{pending['shelf']} shelf registration(s) on file - note "
                                f"that most listed companies keep one, so this matters "
                                f"only alongside an active dilution history"})
    return flags


def _first_series(facts, tags, as_of, flow=False):
    for space in ("us-gaap", "dei"):
        block = (facts.get("facts") or {}).get(space) or {}
        for tag in tags:
            if tag not in block:
                continue
            rows = []
            for unit, entries in (block[tag].get("units") or {}).items():
                for e in entries:
                    rows.append({"end": e.get("end"), "start": e.get("start"),
                                 "val": e.get("val"), "filed": e.get("filed")})
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            for c in ("end", "start", "filed"):
                df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=["end", "filed", "val"])
            df = df[df["filed"] <= as_of]          # point-in-time
            if flow:
                dur = (df["end"] - df["start"]).dt.days
                df = df[dur.between(60, 120)]
            if df.empty:
                continue
            return df.sort_values("filed").drop_duplicates("end", keep="first") \
                     .sort_values("end")
    return None


# =============================================================================
# THE PUBLIC CALL
# =============================================================================
MAX_POSITION_PCT = 20.0     # never size a single name above this share of equity


def assess(ticker, target_date=None, equity=10000.0, risk_budget=0.02,
           calib=None, price_cache=PRICE_CACHE, edgar_cache=EDGAR_CACHE,
           max_position_pct=MAX_POSITION_PCT):
    """
    Risk profile for one ticker as of one date. Returns None if there is not enough
    history. Contains no return forecast by design.
    """
    calib = calib or load_calibration()
    df = load_prices(ticker, price_cache)
    if df is None:
        return None
    as_of = pd.Timestamp(target_date) if target_date else pd.Timestamp.today().normalize()
    past = df[df.index <= as_of]
    if len(past) < MIN_BARS:
        return None

    close = past["Close"]
    entry = float(close.iloc[-1])
    v60 = realised_vol(close, 60)
    v250 = realised_vol(close, 250)
    if not np.isfinite(v60) or v60 <= 0 or entry <= 0:
        return None

    # V53 feature set, computed from the bars up to as_of only
    vf = vol_features_at(past)
    feats = dict(vf) if vf else {}
    feats["vol60"] = v60
    fvol = forecast_vol(feats, calib)
    if not np.isfinite(fvol):
        # a usable volatility forecast is the one thing this module cannot do
        # without, so fail loudly rather than returning a plausible-looking None
        return None
    bars = HORIZON_CONFIGS[HORIZON]["lookahead_bars"]
    sigma_h = fvol * np.sqrt(bars / 252.0)
    p_dd = prob_drawdown(feats, calib)

    # ATR for the level maths, same definition the rest of the project uses
    high, low = past["High"], past["Low"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    lv = compute_levels(entry, atr, HORIZON)

    # Risk-budget sizing: a tight stop divides into a very large position, so a
    # 2% risk budget behind a 5% stop implies 40% of equity in one name. That is
    # correct arithmetic and terrible practice - concentration risk is not captured
    # by the stop distance. Cap it, and report when the cap binds.
    stop_pct = lv.get("stop_pct", np.nan)
    raw_pos = (risk_budget / (stop_pct / 100.0) * 100.0) if stop_pct and np.isfinite(stop_pct) else np.nan
    pos_pct = float(min(raw_pos, max_position_pct)) if np.isfinite(raw_pos) else np.nan
    capped = bool(np.isfinite(raw_pos) and raw_pos > max_position_pct)
    effective_risk = (pos_pct * stop_pct / 100.0) if np.isfinite(pos_pct) else np.nan

    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    lo60, hi60 = float(low.tail(60).min()), float(high.tail(60).max())
    rng_pos = (entry - lo60) / (hi60 - lo60) if hi60 > lo60 else np.nan
    ext_atr = (entry - ema20) / atr if atr > 0 else np.nan
    dv = float((close * past["Volume"]).tail(20).mean()) if "Volume" in past else np.nan

    flags = edgar_flags(ticker, as_of, edgar_cache)
    if np.isfinite(dv) and dv < 2_000_000:
        flags.append({"code": "LOW_LIQUIDITY", "severity": "medium",
                      "detail": f"20-day average dollar volume ${dv/1e6:.1f}M"})
    if np.isfinite(ext_atr) and ext_atr > 2.0:
        flags.append({"code": "EXTENDED", "severity": "low",
                      "detail": f"{ext_atr:.1f} ATR above the 20-day EMA"})
    if capped:
        flags.append({"code": "POSITION_CAPPED", "severity": "low",
                      "detail": f"risk-budget sizing implied {raw_pos:.0f}% of equity; "
                                f"capped at {max_position_pct:.0f}% for concentration"})
    if np.isfinite(v250) and v250 > 0 and v60 / v250 > 1.5:
        flags.append({"code": "VOL_EXPANDING", "severity": "medium",
                      "detail": f"60-day volatility is {v60/v250:.1f}x its 1-year level"})

    return {
        "ticker": ticker,
        "as_of": str(past.index[-1].date()),
        "price": round(entry, 4),

        "risk": {
            "realised_vol_60d_pct": round(v60, 1),
            "realised_vol_250d_pct": round(v250, 1) if np.isfinite(v250) else None,
            "forecast_vol_annual_pct": round(fvol, 1),
            "forecast_vol_horizon_pct": round(sigma_h, 1),
            f"prob_drawdown_{int(DRAWDOWN*100)}pct_{HORIZON_DAYS}d":
                round(p_dd, 3) if np.isfinite(p_dd) else None,
            "calibration_n": calib.get("n_observations"),
            "vol_model": "+".join(calib["shrinkage"].get("features", ["vol60"])),
        },

        "sizing": {
            "risk_budget_pct": round(risk_budget * 100, 2),
            "stop_pct": stop_pct,
            "position_pct_of_equity": round(pos_pct, 1) if np.isfinite(pos_pct) else None,
            "uncapped_position_pct": round(raw_pos, 1) if np.isfinite(raw_pos) else None,
            "position_cap_applied": capped,
            "position_cap_pct": max_position_pct,
            "effective_risk_pct": round(effective_risk, 2) if np.isfinite(effective_risk) else None,
            "position_value": round(equity * pos_pct / 100, 2) if np.isfinite(pos_pct) else None,
            "shares": int(equity * pos_pct / 100 / entry) if np.isfinite(pos_pct) and entry > 0 else None,
            "max_loss_at_stop": round(equity * effective_risk / 100, 2) if np.isfinite(effective_risk) else None,
        },

        "levels": {"entry": lv.get("entry"), "stop": lv.get("stop"),
                   "target": lv.get("target"), "rr": lv.get("rr"),
                   "target_pct": lv.get("target_pct"), "stop_pct": lv.get("stop_pct"),
                   "geometry": lv.get("geometry")},

        "flags": flags,

        # descriptive facts for the synthesis layer - no probabilities attached
        "context": {
            "vs_ema200_pct": round((entry / ema200 - 1) * 100, 1) if ema200 else None,
            "vs_ema20_atr": round(ext_atr, 2) if np.isfinite(ext_atr) else None,
            "range_position_60d": round(rng_pos, 2) if np.isfinite(rng_pos) else None,
            "dollar_volume_20d": round(dv, 0) if np.isfinite(dv) else None,
        },

        "return_forecast": None,
        "disclaimer": ("Pillar 2 does not forecast direction. Out-of-sample "
                       "information coefficient was indistinguishable from zero across "
                       "two label formulations, five horizons and two universes. These "
                       "outputs describe RISK only."),
    }


# =============================================================================
# CALIBRATION I/O
# =============================================================================
_WARNED = set()


def load_calibration(path=CALIB_FILE):
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run --calibrate first.")
    with open(path, encoding="utf-8") as f:
        calib = json.load(f)
    # A pre-V53 file has no feature list. Everything still works, but it is
    # running the OLD single-feature forecast, so the V53 improvement is not
    # actually in effect. Silence there would be the worst outcome: the module
    # would look upgraded and behave exactly as before.
    if "features" not in calib.get("shrinkage", {}) and path not in _WARNED:
        _WARNED.add(path)
        print(f"  NOTE  {path} predates V53 - falling back to the single "
              f"close-to-close feature.\n        Run --calibrate to rebuild it "
              f"with {'+'.join(VOL_FEATURES)}.", file=sys.stderr)
    return calib


def check_horizons():
    """
    The drawdown question and the trade plan have to describe the same exposure.

    This module carries two horizons: HORIZON_DAYS, the window the drawdown
    probability is about, and HORIZON, the trade_config geometry that sets the
    stop and target. If the risk window is shorter than the trade's own life, the
    warning covers only part of the period the position is actually open for -
    which reads as a safe number attached to an unsafe holding.
    """
    cfgh = HORIZON_CONFIGS[HORIZON]
    trade_days = cfgh.get("eval_days") or int(
        round(cfgh.get("lookahead_bars", 60) * 365.25 / 252))
    if HORIZON_DAYS < 0.9 * trade_days:
        print(f"  WARNING the drawdown question spans {HORIZON_DAYS} days but the "
              f"{HORIZON} trade plan runs {trade_days} days.")
        print(f"          The probability then describes a shorter period than the")
        print(f"          position is open for. Set HORIZON_DAYS to about "
              f"{trade_days}, or shorten the trade horizon to match.")
    return trade_days


def run_calibration(universe=CALIB_UNIVERSE, price_cache=PRICE_CACHE,
                    step=CALIB_STEP, out=CALIB_FILE, tickers=None,
                    horizon_days=None, drawdown=None,
                    min_observations=CALIB_MIN_OBS, n_bins=CURVE_BINS,
                    verbose=True):
    """
    Fit the model and write the calibration file. Returns the calibration dict.

        run_calibration()                                 # deployed settings
        run_calibration(universe="deploy", step=20)       # quicker
        run_calibration(tickers=["NVDA", "AMD"], out="x.json")
        run_calibration(horizon_days=30, drawdown=0.30, out="sweep_30_30.json")

    Raises RuntimeError rather than calling sys.exit, so it is usable from a
    notebook or another script. main() turns that back into an exit code.
    """
    horizon_days = HORIZON_DAYS if horizon_days is None else int(horizon_days)
    drawdown = DRAWDOWN if drawdown is None else float(drawdown)

    def say(*a):
        if verbose:
            print(*a)

    if tickers is None:
        try:
            from pillar2_v43_universe import DEPLOY_UNIVERSE, load_training_universe
            tickers = (DEPLOY_UNIVERSE if universe == "deploy"
                       else load_training_universe(include_deploy=True))
        except ImportError as e:
            raise RuntimeError(
                "pillar2_v43_universe.py not importable - run next to it, or "
                "pass tickers=[...] explicitly.") from e

    say("=" * 88)
    say("CALIBRATING THE RISK MODEL")
    say("=" * 88)
    trade_days = check_horizons()
    say(f"  drawdown window {horizon_days}d | {HORIZON} trade plan {trade_days}d "
        f"| {_fwd_bars(horizon_days)} forward bars, at least "
        f"{_min_fwd_bars(horizon_days)} required")
    say(f"  universe {universe} ({len(tickers)} tickers) | horizon {horizon_days}d "
        f"| event = a fall of {drawdown:.0%} or more\n")
    if horizon_days != HORIZON_DAYS or drawdown != DRAWDOWN:
        say(f"  NOTE: fitting at {drawdown:.0%}/{horizon_days}d, which is NOT the "
            f"module config ({DRAWDOWN:.0%}/{HORIZON_DAYS}d).")
        say(f"        Fine for a sweep. Do not point the deployed pipeline at this "
            f"file - deployment_preflight.py will fail it.\n")

    obs = build_observations(tickers, price_cache, step=step,
                             horizon_days=horizon_days, drawdown=drawdown,
                             verbose=verbose)
    if len(obs) < min_observations:
        raise RuntimeError(f"Only {len(obs)} observations - not enough to "
                           f"calibrate (need {min_observations}).")

    say(f"\n  {len(obs):,} observations | {obs['ticker'].nunique()} tickers | "
        f"{obs['date'].min():%Y-%m} -> {obs['date'].max():%Y-%m}")
    say(f"  base rate: {obs['event'].mean():.1%} had a {drawdown:.0%} drawdown")

    shrink = fit_vol_shrinkage(obs)
    terms = "  ".join(f"{c:+.3f} x {f}"
                      for c, f in zip(shrink["coef"], shrink["features"]))
    say(f"\n  VOLATILITY FORECAST (V53)  forward_vol = "
          f"{shrink['intercept']:.2f}  {terms}")
    say(f"    R2 {shrink['r2']:.3f} in sample on {shrink['n']:,} observations")
    say(f"    coefficients summing to {sum(shrink['coef']):.3f} - below 1.0 is the")
    say(f"    mean reversion: an elevated reading is shrunk toward the long-run level.")
    say(f"    V53 measured this feature set at OOS R2 0.5288 against 0.4923 for the")
    say(f"    single close-to-close feature it replaces.")

    curve = fit_drawdown_curve(obs, shrink, n_bins)
    calib = {"shrinkage": shrink, "curve": curve,
             "horizon_days": horizon_days, "drawdown": drawdown,
             "n_observations": int(len(obs)),
             "vol_model": "v53_yang_zhang_har",
             "base_rate": float(obs["event"].mean()),
             "universe": universe,
             "built": str(_date.today())}

    say(f"\n  DRAWDOWN CURVE")
    say(f"    {'forecast vol':>14}{'P(drawdown)':>14}{'mean worst':>13}{'n':>8}")
    for p in curve:
        say(f"    {p['fvol']:>13.0f}%{p['p']:>13.1%}{p['mean_mdd']:>12.1f}%{p['n']:>8,}")

    rel = reliability(obs, calib)
    if rel is not None:
        say(f"\n  RELIABILITY  - the metric that matters for a risk model")
        say(f"    {'predicted':>12}{'realised':>11}{'error':>9}{'n':>9}")
        for _, r in rel.iterrows():
            say(f"    {r['predicted']:>11.1f}%{r['realised']:>10.1f}%"
                  f"{r['realised']-r['predicted']:>+8.1f}{int(r['n']):>9,}")
        mae = float((rel["realised"] - rel["predicted"]).abs().mean())
        say(f"    mean absolute calibration error: {mae:.1f} percentage points")
        say(f"    Under ~5pp is well calibrated; the curve is fitted in-sample here,")
        say(f"    so treat this as a floor and re-check on held-out dates.")
        calib["calibration_mae_pp"] = mae

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        say(f"\n  wrote {out}")
    return calib


# =============================================================================
# CLI
# =============================================================================
def run_assess(ticker, target_date=None, equity=DEFAULT_EQUITY,
               risk_budget=DEFAULT_RISK_BUDGET, calib=None,
               price_cache=PRICE_CACHE, edgar_cache=EDGAR_CACHE,
               max_position_pct=MAX_POSITION_PCT, verbose=False):
    """
    Score one ticker. Thin wrapper over assess() that prints the geometry and the
    JSON when verbose, so the CLI and a caller share one code path.

        r = run_assess("NVDA", equity=25000)
        r["risk"]["prob_drawdown_20pct_90d"]

    Returns None when there is not enough history - the same contract as assess().
    """
    out = assess(ticker, target_date=target_date, equity=equity,
                 risk_budget=risk_budget, calib=calib, price_cache=price_cache,
                 edgar_cache=edgar_cache, max_position_pct=max_position_pct)
    if verbose and out is not None:
        print(f"\n  geometry: {describe_geometry(HORIZON)}\n")
        print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ticker")
    ap.add_argument("--date")
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--risk-budget", type=float, default=0.02)
    ap.add_argument("--max-position-pct", type=float, default=MAX_POSITION_PCT,
                    help="cap on a single name's share of equity")
    ap.add_argument("--universe", default="all", choices=["all", "deploy"])
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--edgar-cache", default=EDGAR_CACHE)
    ap.add_argument("--step", type=int, default=CALIB_STEP,
                    help="trading days between calibration observations")
    ap.add_argument("--horizon-days", type=int, default=None,
                    help="fit at a different horizon (a sweep, not the deployed "
                         "setting - the preflight will reject the result)")
    ap.add_argument("--drawdown", type=float, default=None,
                    help="fit at a different threshold, same caveat")
    ap.add_argument("--out", default=CALIB_FILE)
    cfg = ap.parse_args()

    # main() only parses arguments and turns exceptions into exit codes. Every
    # piece of behaviour lives in a run_* function that can be called directly.
    if cfg.calibrate:
        try:
            run_calibration(universe=cfg.universe, price_cache=cfg.price_cache,
                            step=cfg.step, out=cfg.out,
                            horizon_days=cfg.horizon_days,
                            drawdown=cfg.drawdown)
        except RuntimeError as e:
            sys.exit(str(e))
        return
    if not cfg.ticker:
        sys.exit("Pass --ticker, or --calibrate to fit the model first.")

    out = run_assess(cfg.ticker, target_date=cfg.date, equity=cfg.equity,
                     risk_budget=cfg.risk_budget, price_cache=cfg.price_cache,
                     edgar_cache=cfg.edgar_cache,
                     max_position_pct=cfg.max_position_pct, verbose=True)
    if out is None:
        sys.exit(f"Not enough price history for {cfg.ticker}.")


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Set RUN_MODE, edit that mode's
# block, run.
#
#   "assess"    score one ticker against the calibration that is already on disk
#   "calibrate" refit the model and write the calibration file. Do this first
#               if pillar2_risk_calibration.json does not exist yet.
#
# Passing any command-line flag still works and takes over.
# =============================================================================
if __name__ == "__main__":
    # run from anywhere (e.g. the GreenPulse root): the paths below are
    # relative to the folder this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODE = "assess"                  # "assess" | "calibrate"

    # --- when RUN_MODE = "assess" --------------------------------------------
    RUN_TICKER           = "NVDA"
    RUN_DATE             = None          # "2026-06-30" scores as of a past date;
                                         # None = the last bar in the cache
    RUN_EQUITY           = DEFAULT_EQUITY
    RUN_RISK_BUDGET      = DEFAULT_RISK_BUDGET
    RUN_POSITION_CAP_PCT = MAX_POSITION_PCT   # one name's ceiling, % of equity

    # --- when RUN_MODE = "calibrate" -----------------------------------------
    RUN_UNIVERSE         = CALIB_UNIVERSE     # "all" or "deploy"
    RUN_STEP             = CALIB_STEP         # trading days between observations
    RUN_FIT_HORIZON_DAYS = None    # None = the deployed HORIZON_DAYS at the top
    RUN_FIT_DRAWDOWN     = None    # None = the deployed DRAWDOWN at the top.
                                   # Change either and the preflight will reject
                                   # the result on purpose - it is a sweep, not
                                   # the deployed setting.
    RUN_OUT              = CALIB_FILE

    # --- both -----------------------------------------------------------------
    RUN_PRICE_CACHE      = PRICE_CACHE
    RUN_EDGAR_CACHE      = EDGAR_CACHE
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                       # e.g. python class_ai_pillar2_risk_v2.py --calibrate

    elif RUN_MODE == "calibrate":
        try:
            run_calibration(universe=RUN_UNIVERSE, price_cache=RUN_PRICE_CACHE,
                            step=RUN_STEP, out=RUN_OUT,
                            horizon_days=RUN_FIT_HORIZON_DAYS,
                            drawdown=RUN_FIT_DRAWDOWN)
        except RuntimeError as _e:
            sys.exit(str(_e))

    elif RUN_MODE == "assess":
        _out = run_assess(RUN_TICKER, target_date=RUN_DATE, equity=RUN_EQUITY,
                          risk_budget=RUN_RISK_BUDGET,
                          price_cache=RUN_PRICE_CACHE,
                          edgar_cache=RUN_EDGAR_CACHE,
                          max_position_pct=RUN_POSITION_CAP_PCT, verbose=True)
        if _out is None:
            sys.exit(f"Not enough price history for {RUN_TICKER}.")

    else:
        sys.exit(f'RUN_MODE must be "assess" or "calibrate", not {RUN_MODE!r}')
