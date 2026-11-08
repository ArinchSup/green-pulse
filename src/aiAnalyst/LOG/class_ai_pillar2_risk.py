#!/usr/bin/env python3
"""
class_ai_pillar2_risk.py - Pillar 2, rebuilt as a RISK engine.

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
HORIZON_DAYS = 180          # calendar days for the drawdown question
DRAWDOWN = 0.30             # what counts as a drawdown event
MIN_BARS = 260


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
    """Annualised realised volatility, in percent."""
    r = close.pct_change().tail(window)
    if len(r.dropna()) < window // 2:
        return np.nan
    return float(r.std() * np.sqrt(252) * 100)


# =============================================================================
# CALIBRATION
# =============================================================================
def build_observations(tickers, cache_dir, step=10):
    """
    Every (ticker, date) with enough history: trailing volatility, and what actually
    happened over the next HORIZON_DAYS. Nothing here uses information from after the
    observation date except the outcome itself.
    """
    rows = []
    for i, t in enumerate(tickers, 1):
        df = load_prices(t, cache_dir)
        if df is None or len(df) < MIN_BARS + 140:
            continue
        close, low = df["Close"], df["Low"]
        for pos in range(MIN_BARS, len(df) - 130, step):
            hist = close.iloc[:pos + 1]
            v60 = realised_vol(hist, 60)
            v250 = realised_vol(hist, 250)
            if not np.isfinite(v60) or v60 <= 0:
                continue
            entry = float(close.iloc[pos])
            end = df.index[pos] + pd.Timedelta(days=HORIZON_DAYS)
            fwd = df.iloc[pos + 1:]
            fwd = fwd[fwd.index <= end]
            if len(fwd) < 80:
                continue
            mdd = (float(fwd["Low"].min()) - entry) / entry * 100
            fwd_vol = float(fwd["Close"].pct_change().std() * np.sqrt(252) * 100)
            rows.append({"ticker": t, "date": df.index[pos], "vol60": v60,
                         "vol250": v250, "fwd_vol": fwd_vol, "mdd": mdd,
                         "event": mdd <= -DRAWDOWN * 100})
        if i % 20 == 0 or i == len(tickers):
            print(f"    [{i}/{len(tickers)}] {len(rows):,} observations")
    return pd.DataFrame(rows)


def fit_vol_shrinkage(obs):
    """
    Volatility mean-reverts, so today's trailing vol overstates tomorrow's when it is
    high. Fit forward_vol = a + b * trailing_vol and use that as the forecast.
    """
    d = obs.dropna(subset=["vol60", "fwd_vol"])
    if len(d) < 500:
        return {"a": 0.0, "b": 1.0, "r2": np.nan, "n": len(d)}
    x, y = d["vol60"].to_numpy(), d["fwd_vol"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - np.var(y - pred) / np.var(y)
    return {"a": float(a), "b": float(b), "r2": float(r2), "n": int(len(d))}


def fit_drawdown_curve(obs, shrink, n_bins=12):
    """
    Empirical P(drawdown | forecast vol). Deliberately a lookup table rather than a
    model: it is transparent, it cannot overfit 12 numbers, and it is trivial to audit.
    """
    d = obs.dropna(subset=["vol60"]).copy()
    d["fvol"] = shrink["a"] + shrink["b"] * d["vol60"]
    d = d[d["fvol"] > 0]
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
    d = obs.dropna(subset=["vol60"]).copy()
    d["p"] = [prob_drawdown(v, calib) for v in d["vol60"]]
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
def forecast_vol(vol60, calib):
    s = calib["shrinkage"]
    return s["a"] + s["b"] * vol60


def prob_drawdown(vol60, calib):
    """Interpolate the calibrated curve; clamp outside the fitted range."""
    if not np.isfinite(vol60):
        return np.nan
    fv = forecast_vol(vol60, calib)
    pts = calib["curve"]
    xs = [p["fvol"] for p in pts]
    ys = [p["p"] for p in pts]
    return float(np.interp(fv, xs, ys))


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

    fvol = forecast_vol(v60, calib)
    bars = HORIZON_CONFIGS[HORIZON]["lookahead_bars"]
    sigma_h = fvol * np.sqrt(bars / 252.0)
    p_dd = prob_drawdown(v60, calib)

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
def load_calibration(path=CALIB_FILE):
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run --calibrate first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_calibration(cfg):
    try:
        from pillar2_v43_universe import DEPLOY_UNIVERSE, load_training_universe
        tickers = (DEPLOY_UNIVERSE if cfg.universe == "deploy"
                   else load_training_universe(include_deploy=True))
    except ImportError:
        sys.exit("Run next to pillar2_v43_universe.py.")

    print("=" * 88)
    print("CALIBRATING THE RISK MODEL")
    print("=" * 88)
    print(f"  universe {cfg.universe} ({len(tickers)} tickers) | horizon {HORIZON_DAYS}d "
          f"| event = a fall of {DRAWDOWN:.0%} or more\n")
    obs = build_observations(tickers, cfg.price_cache, step=cfg.step)
    if len(obs) < 2000:
        sys.exit(f"Only {len(obs)} observations - not enough to calibrate.")

    print(f"\n  {len(obs):,} observations | {obs['ticker'].nunique()} tickers | "
          f"{obs['date'].min():%Y-%m} -> {obs['date'].max():%Y-%m}")
    print(f"  base rate: {obs['event'].mean():.1%} had a {DRAWDOWN:.0%} drawdown")

    shrink = fit_vol_shrinkage(obs)
    print(f"\n  VOLATILITY FORECAST  forward_vol = {shrink['a']:.2f} + "
          f"{shrink['b']:.3f} x trailing_60d")
    print(f"    R2 {shrink['r2']:.3f} on {shrink['n']:,} observations")
    print(f"    slope below 1.0 means volatility mean-reverts: today's high vol")
    print(f"    overstates tomorrow's, and the forecast shrinks it accordingly.")

    curve = fit_drawdown_curve(obs, shrink)
    calib = {"shrinkage": shrink, "curve": curve,
             "horizon_days": HORIZON_DAYS, "drawdown": DRAWDOWN,
             "n_observations": int(len(obs)),
             "base_rate": float(obs["event"].mean()),
             "universe": cfg.universe,
             "built": str(_date.today())}

    print(f"\n  DRAWDOWN CURVE")
    print(f"    {'forecast vol':>14}{'P(drawdown)':>14}{'mean worst':>13}{'n':>8}")
    for p in curve:
        print(f"    {p['fvol']:>13.0f}%{p['p']:>13.1%}{p['mean_mdd']:>12.1f}%{p['n']:>8,}")

    rel = reliability(obs, calib)
    if rel is not None:
        print(f"\n  RELIABILITY  - the metric that matters for a risk model")
        print(f"    {'predicted':>12}{'realised':>11}{'error':>9}{'n':>9}")
        for _, r in rel.iterrows():
            print(f"    {r['predicted']:>11.1f}%{r['realised']:>10.1f}%"
                  f"{r['realised']-r['predicted']:>+8.1f}{int(r['n']):>9,}")
        mae = float((rel["realised"] - rel["predicted"]).abs().mean())
        print(f"    mean absolute calibration error: {mae:.1f} percentage points")
        print(f"    Under ~5pp is well calibrated; the curve is fitted in-sample here,")
        print(f"    so treat this as a floor and re-check on held-out dates.")
        calib["calibration_mae_pp"] = mae

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print(f"\n  wrote {cfg.out}")


# =============================================================================
# CLI
# =============================================================================
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
    ap.add_argument("--step", type=int, default=10,
                    help="trading days between calibration observations")
    ap.add_argument("--out", default=CALIB_FILE)
    cfg = ap.parse_args()

    if cfg.calibrate:
        run_calibration(cfg)
        return
    if not cfg.ticker:
        sys.exit("Pass --ticker, or --calibrate to fit the model first.")

    out = assess(cfg.ticker, cfg.date, cfg.equity, cfg.risk_budget,
                 price_cache=cfg.price_cache, edgar_cache=cfg.edgar_cache,
                 max_position_pct=cfg.max_position_pct)
    if out is None:
        sys.exit(f"Not enough price history for {cfg.ticker}.")
    print(f"\n  geometry: {describe_geometry(HORIZON)}\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
