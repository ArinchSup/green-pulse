"""
pillar2_v41_features.py - the ONE place V41 turns a technical snapshot into model features.

The trainer (class_xgboost_v41.py) and inference (class_ai_pillar2_v41.py) both import from
here, so training and serving can never compute features differently.

Every feature is scale-free: ratios, positions inside a range, states and flags. There are no
raw dollar levels. A raw price like $187.35 lets a tree model recognise WHICH stock and WHICH
month it is looking at, which fits the past well and fails on new periods.
"""
import math

FEATURE_VERSION = "v41.1"
RS_WINDOWS = (20, 60, 120)   # trading days for the relative-strength features

_MACD = {"Strong Bearish": 0.0, "Bearish Crossover (Pullback)": 1.0,
         "Bullish Crossover (Recovery)": 2.0, "Strong Bullish": 3.0}
_TREND = {"Downtrend": -1.0, "Sideways / Consolidation": 0.0, "Uptrend": 1.0}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _ratio(a, b):
    """a / b - 1, or NaN when it can't be computed (XGBoost treats NaN as missing)."""
    a, b = _num(a), _num(b)
    return a / b - 1.0 if math.isfinite(a) and math.isfinite(b) and b != 0 else math.nan


def _position(x, lo, hi):
    """Where x sits between lo (0.0) and hi (1.0)."""
    x, lo, hi = _num(x), _num(lo), _num(hi)
    return (x - lo) / (hi - lo) if math.isfinite(x) and hi > lo else math.nan


def relative_strength(stock_close, bench_close, windows=RS_WINDOWS):
    """
    Point-in-time returns up to the signal bar (the LAST element of each series):
      ret_N = the stock's return over the last N bars
      rs_N  = ret_N minus the benchmark's return over the same bars
    Both inputs are pandas Series of closes that END at the signal bar.
    """
    out = {}
    for n in windows:
        r_s = _ratio(stock_close.iloc[-1], stock_close.iloc[-1 - n]) if len(stock_close) > n else math.nan
        r_b = _ratio(bench_close.iloc[-1], bench_close.iloc[-1 - n]) if len(bench_close) > n else math.nan
        out[f"ret_{n}"] = r_s
        out[f"rs_{n}"] = r_s - r_b if math.isfinite(r_s) and math.isfinite(r_b) else math.nan
    return out


def snapshot_to_features(snap):
    """Technical snapshot (as built by build_snapshot + relative_strength) -> dict of features."""
    price = snap.get("current_price")
    bb = snap.get("bollinger_bands") or {}
    kl = snap.get("key_levels") or {}
    cp = snap.get("candlestick_patterns") or {}
    rs = snap.get("relative_strength") or {}
    profile = str(snap.get("volume_profile_trend", ""))
    vol_pct = _num(str(snap.get("volume_vs_avg_20", "")).replace("%", ""))

    f = {
        "price_vs_ema20":   _ratio(price, snap.get("ema_20")),
        "price_vs_ema200":  _ratio(price, snap.get("ema_200")),
        "ema20_vs_ema200":  _ratio(snap.get("ema_20"), snap.get("ema_200")),
        "price_vs_vwap20":  _ratio(price, snap.get("vwap_20")),
        "bb_position":      _position(price, bb.get("lower"), bb.get("upper")),
        "bb_width":         _ratio(bb.get("upper"), bb.get("lower")),
        "rsi_14":           _num(snap.get("rsi_14")),
        "macd_state":       _MACD.get(snap.get("macd_status"), math.nan),
        "trend_state":      _TREND.get(snap.get("graph_trend"), math.nan),
        "volume_ratio":     vol_pct / 100.0 if math.isfinite(vol_pct) else math.nan,
        "volume_profile":   -1.0 if "Distribution" in profile else (1.0 if "Accumulation" in profile else 0.0),
        "atr_pct":          (_num(snap.get("atr_14")) / _num(price)) if _num(price) else math.nan,
        "support1_gap":     _ratio(price, kl.get("support_1")),
        "resist1_gap":      _ratio(kl.get("resistance_1"), price),
        "range30_position": _position(price, kl.get("support_2_swing_low"), kl.get("resistance_2_swing_high")),
        "range30_width":    _ratio(kl.get("resistance_2_swing_high"), kl.get("support_2_swing_low")),
    }
    for i in range(1, 6):
        for k in ("body_ratio", "upper_wick_ratio", "lower_wick_ratio", "is_bullish"):
            f[f"bar_{i}_{k}"] = _num(cp.get(f"bar_{i}_{k}"))
    for k in ("is_shooting_star", "is_hammer", "is_bearish_engulfing", "is_bullish_engulfing",
              "is_rejecting_resistance", "5bar_close_slope", "volume_trend_slope"):
        f[k] = _num(cp.get(k))
    for n in RS_WINDOWS:
        f[f"ret_{n}"] = _num(rs.get(f"ret_{n}"))
        f[f"rs_{n}"] = _num(rs.get(f"rs_{n}"))
    return f


FEATURE_NAMES = list(snapshot_to_features({}).keys())


def to_vector(snap):
    """Features in the fixed FEATURE_NAMES order, ready for a model."""
    f = snapshot_to_features(snap)
    return [f[name] for name in FEATURE_NAMES]
