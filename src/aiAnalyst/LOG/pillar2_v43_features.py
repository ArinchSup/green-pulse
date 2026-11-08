"""
pillar2_v43_features.py - the ONE place V43 turns price history into model features.

The generator and inference both import from here, so training and serving cannot
compute features differently. Same discipline as pillar2_v41_features.py, extended.

WHAT CHANGED FROM V40/V42
  + Market regime. get_macro_trend() existed in class_news_fetcher.py and was never
    called, so V40/V42 could not see whether the market was rising or falling. They
    made dip-buying calls blind to context. Now every row carries the benchmark's
    own state and the breadth of the universe on that date.
  + Multi-horizon momentum and relative strength vs the benchmark, ported from V41.
    "Down 15%" and "down 15% while the market rose 5%" are different setups.
  + Cross-sectional ranks. Each of the headline features is ALSO expressed as its
    percentile among every ticker scored that same day, which is how most working
    equity models frame the question: not "is RSI 45 low" but "is this the 5th
    weakest of 300 names today".
  ~ RSI is now real Wilder RSI. The old calculate_rsi() in class_news_fetcher.py
    diffed the ENTIRE price series and divided by 14 — no rolling window at all — so
    "rsi_14" was a year-long up/down ratio whose scale depended on history length.
  ~ MACD is a continuous normalised histogram instead of a 4-level ordinal.
  - Fibonacci levels are gone. The V39 ablation put their contribution at exactly
    0.0000, and they were still ranked #2-3 by SHAP in V40/V42, meaning the model was
    spending capacity fitting noise.

EVERY FEATURE IS SCALE-FREE — ratios, positions in a range, z-scores, ranks. No raw
dollar levels: a price like $187.35 lets a tree identify WHICH stock and WHICH month
it is looking at, which fits the past and fails on new periods.

Computation is VECTORISED over a ticker's whole history (compute_panel), so building
100k rows is pandas work on cached frames, not 100k network calls.
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = "v43.0"

RS_WINDOWS = (20, 60, 120)
MOM_WINDOWS = (5, 20, 60, 120, 250)

# Features that also get a cross-sectional (same-day, across-universe) rank.
# Keep this list short: each entry adds a column, and ranking noise is still noise.
XS_RANK_FEATURES = [
    "ret_20", "ret_60", "rs_60", "atr_pct", "rsi_14",
    "volume_ratio_20", "gap_ema200", "bb_position",
]


# =============================================================================
# SMALL HELPERS
# =============================================================================
def _safe_ratio(a, b):
    """a / b - 1, NaN where b is 0 or missing. XGBoost treats NaN as missing."""
    return pd.Series(np.where((b != 0) & b.notna() & a.notna(), a / b - 1.0, np.nan),
                     index=a.index)


def _position(x, lo, hi):
    """Where x sits between lo (0.0) and hi (1.0); NaN on a degenerate range."""
    span = hi - lo
    return pd.Series(np.where(span > 0, (x - lo) / span, np.nan), index=x.index)


def wilder_rsi(close, period=14):
    """
    Real Wilder RSI. The old implementation summed every up-move in the series and
    divided by 14, which is not RSI and drifts with how much history you passed it.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(roll_down != 0, 100.0)


def atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =============================================================================
# PER-STOCK PANEL
# =============================================================================
def compute_panel(df, bench_close=None):
    """
    All per-stock features for a ticker's ENTIRE history at once.

    df          : daily OHLCV, DatetimeIndex ascending, tz-naive.
    bench_close : benchmark close Series on the same index (reindexed inside).

    Returns a DataFrame indexed by date. Every row uses only bars up to and
    including that date — the rolling/ewm windows are all backward-looking, and
    nothing here shifts data forward.
    """
    out = pd.DataFrame(index=df.index)
    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # ── trend / location ────────────────────────────────────────────────
    out["price_vs_ema20"] = _safe_ratio(close, ema20)
    out["price_vs_ema50"] = _safe_ratio(close, ema50)
    out["gap_ema200"] = _safe_ratio(close, ema200)
    out["ema20_vs_ema50"] = _safe_ratio(ema20, ema50)
    out["ema50_vs_ema200"] = _safe_ratio(ema50, ema200)

    # ── momentum, multi-horizon ─────────────────────────────────────────
    for n in MOM_WINDOWS:
        out[f"ret_{n}"] = _safe_ratio(close, close.shift(n))
    # classic 12-1 momentum: last year excluding the most recent month
    out["ret_250_ex_20"] = _safe_ratio(close.shift(20), close.shift(250))

    # ── volatility ──────────────────────────────────────────────────────
    a14 = atr(high, low, close, 14)
    out["atr_pct"] = a14 / close.replace(0.0, np.nan)
    ret1 = close.pct_change()
    out["vol_20"] = ret1.rolling(20).std()
    out["vol_60"] = ret1.rolling(60).std()
    # is volatility expanding or contracting right now?
    out["vol_ratio_20_60"] = out["vol_20"] / out["vol_60"].replace(0.0, np.nan)

    # ── bands / range position ──────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    upper, lower = sma20 + 2 * sd20, sma20 - 2 * sd20
    out["bb_position"] = _position(close, lower, upper)
    out["bb_width"] = (upper - lower) / sma20.replace(0.0, np.nan)
    for n in (30, 60):
        out[f"range{n}_position"] = _position(close, low.rolling(n).min(),
                                              high.rolling(n).max())

    # ── oscillators ─────────────────────────────────────────────────────
    out["rsi_14"] = wilder_rsi(close, 14)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist_norm"] = (macd - signal) / close.replace(0.0, np.nan)
    out["macd_norm"] = macd / close.replace(0.0, np.nan)

    # ── volume ──────────────────────────────────────────────────────────
    vol20 = volume.rolling(20).mean()
    out["volume_ratio_20"] = volume / vol20.replace(0.0, np.nan)
    out["volume_trend_20"] = _safe_ratio(vol20, volume.rolling(60).mean())
    dollar_vol = (close * volume).rolling(20).mean()
    out["log_dollar_vol"] = np.log10(dollar_vol.where(dollar_vol > 0))

    # ── recent bar shapes (last 5 bars, scale-free) ─────────────────────
    bar_range = (high - low).replace(0.0, np.nan)
    body_ratio = (close - df["Open"]).abs() / bar_range
    upper_wick = (high - close.combine(df["Open"], max)) / bar_range
    lower_wick = (close.combine(df["Open"], min) - low) / bar_range
    is_bull = (close > df["Open"]).astype(float)
    for i in range(1, 6):
        out[f"bar_{i}_body"] = body_ratio.shift(i - 1)
        out[f"bar_{i}_upper_wick"] = upper_wick.shift(i - 1)
        out[f"bar_{i}_lower_wick"] = lower_wick.shift(i - 1)
        out[f"bar_{i}_bullish"] = is_bull.shift(i - 1)
    out["close_slope_5"] = _safe_ratio(close, close.shift(5))

    # ── relative strength vs benchmark ──────────────────────────────────
    if bench_close is not None:
        b = bench_close.reindex(df.index).ffill()
        for n in RS_WINDOWS:
            out[f"rs_{n}"] = out[f"ret_{n}"] - _safe_ratio(b, b.shift(n))
        out["beta_60"] = (ret1.rolling(60).cov(b.pct_change())
                          / b.pct_change().rolling(60).var().replace(0.0, np.nan))
    else:
        for n in RS_WINDOWS:
            out[f"rs_{n}"] = np.nan
        out["beta_60"] = np.nan

    return out


# =============================================================================
# MARKET REGIME  (identical for every ticker on a given date)
# =============================================================================
def compute_regime_panel(bench_df):
    """
    What the market itself was doing. This is the piece V40/V42 were missing:
    without it the model cannot tell a dip in an uptrend from a dip in a crash.
    """
    out = pd.DataFrame(index=bench_df.index)
    close = bench_df["Close"]
    out["mkt_vs_ema200"] = _safe_ratio(close, close.ewm(span=200, adjust=False).mean())
    out["mkt_vs_ema50"] = _safe_ratio(close, close.ewm(span=50, adjust=False).mean())
    for n in (20, 60, 120):
        out[f"mkt_ret_{n}"] = _safe_ratio(close, close.shift(n))
    r = close.pct_change()
    out["mkt_vol_20"] = r.rolling(20).std()
    out["mkt_vol_ratio"] = out["mkt_vol_20"] / r.rolling(60).std().replace(0.0, np.nan)
    # drawdown from the running 1-year high: 0 at the highs, negative in a selloff
    out["mkt_drawdown_250"] = close / close.rolling(250, min_periods=60).max() - 1.0
    return out


REGIME_FEATURES = ["mkt_vs_ema200", "mkt_vs_ema50", "mkt_ret_20", "mkt_ret_60",
                   "mkt_ret_120", "mkt_vol_20", "mkt_vol_ratio", "mkt_drawdown_250",
                   "breadth_above_ema200"]


# =============================================================================
# CROSS-SECTIONAL RANKS  (within one date's cohort)
# =============================================================================
def add_cross_sectional(cohort, features=None):
    """
    cohort: one row per ticker, all on the SAME date.
    Adds xs_<f> = percentile rank of that feature among the tickers scored that day,
    in [0, 1]. This is what turns an absolute reading into a comparative one and is
    the main structural upgrade in V43.
    """
    features = features or XS_RANK_FEATURES
    for f in features:
        if f in cohort.columns:
            cohort[f"xs_{f}"] = cohort[f].rank(pct=True)
        else:
            cohort[f"xs_{f}"] = np.nan
    return cohort


def breadth(cohort):
    """Share of the cohort trading above its own 200-day EMA — a market-health read
    that does not depend on which index you picked."""
    g = cohort.get("gap_ema200")
    return float((g > 0).mean()) if g is not None and g.notna().any() else np.nan


# =============================================================================
# CANONICAL FEATURE ORDER
# =============================================================================
def _template_columns():
    idx = pd.date_range("2020-01-01", periods=3)
    dummy = pd.DataFrame({c: [1.0, 1.0, 1.0] for c in
                          ["Open", "High", "Low", "Close", "Volume"]}, index=idx)
    panel = compute_panel(dummy, bench_close=dummy["Close"])
    cols = list(panel.columns)
    cols += [f"xs_{f}" for f in XS_RANK_FEATURES]
    cols += REGIME_FEATURES
    return cols


FEATURE_NAMES = _template_columns()


def to_vector(row):
    """Features in the fixed FEATURE_NAMES order, ready for a model."""
    return [float(row.get(name, np.nan)) if row.get(name) is not None else np.nan
            for name in FEATURE_NAMES]


if __name__ == "__main__":
    print(f"{FEATURE_VERSION}: {len(FEATURE_NAMES)} features")
    groups = {
        "trend/location": [c for c in FEATURE_NAMES if "ema" in c and not c.startswith(("xs_", "mkt_"))],
        "momentum": [c for c in FEATURE_NAMES if c.startswith("ret_")],
        "volatility": [c for c in FEATURE_NAMES if "vol" in c and not c.startswith(("xs_", "mkt_"))] + ["atr_pct"],
        "bands/range": [c for c in FEATURE_NAMES if c.startswith(("bb_", "range"))],
        "oscillators": [c for c in FEATURE_NAMES if c.startswith(("rsi", "macd"))],
        "bars": [c for c in FEATURE_NAMES if c.startswith("bar_")],
        "relative strength": [c for c in FEATURE_NAMES if c.startswith("rs_")] + ["beta_60"],
        "cross-sectional": [c for c in FEATURE_NAMES if c.startswith("xs_")],
        "market regime": REGIME_FEATURES,
    }
    for name, cols in groups.items():
        print(f"  {name:<20} {len(cols):>3}  {', '.join(cols[:5])}"
              f"{' ...' if len(cols) > 5 else ''}")
