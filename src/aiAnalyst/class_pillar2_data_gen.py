import os
import json
import random
import datetime
import yfinance as yf
import pandas as pd
import numpy as np

from trade_config import (compute_levels, walk_trade, df_to_bars,
                          describe_geometry, GEOMETRY_MODE)

# ==========================================
# MASTER CONFIGURATION
# ==========================================
HORIZON = "MID"                  # "SHORT" | "MID" | "LONG"

TARGET_DATASET_SIZE = 5000      
# ── Class balance ────────────────────────────────────────────────
# True  = accept whatever the market produces. The model then sees the
#         same base rate it will meet in production, so its probabilities
#         mean something. Quota sampling made the dataset 40% Bullish
#         while the real rate is ~53%, which is a large part of why the
#         v9 confidence curve came out inverted.
# False = fill separate quotas at TARGET_BULLISH_PCT (legacy behaviour).
USE_NATURAL_RATE   = True
TARGET_BULLISH_PCT = 0.40        # only used when USE_NATURAL_RATE = False

# ── Label definition ─────────────────────────────────────────────
# "TP_SL"          — did +TP hit before -SL within the window?
#                    Mostly measures market direction, which single-stock
#                    technicals cannot predict. Random entry already earns
#                    +5.1% per trade on this universe, so the model has to
#                    beat beta before it shows any skill.
# "BEAT_BENCHMARK" — did the stock outperform SPY over the same window?
#                    Strips market beta out. Base rate lands near 50% by
#                    construction, so any lift is unambiguous skill.
LABEL_MODE       = "TP_SL"       # "TP_SL" | "BEAT_BENCHMARK"
BENCHMARK_TICKER = "SPY"

# Constrain sampling to the same window the backtest uses. Training on 5
# years while grading on 3.3 means the two measure different regimes.
MAX_LOOKBACK_DAYS = 1200         # None = use all available history

RANDOM_SEED = 42                 # reproducible generation

# ── The critical toggle ───────────────────────────────────────────
# False (RECOMMENDED): label depends ONLY on future price action.
#   The model must learn which setups work — that's the whole point.
# True: only consider candidates that pass the technical filter.
#   Use ONLY if you want the model to specialise within already-good
#   setups. Never re-add the filter to the label itself.
USE_STRUCTURAL_PREFILTER = False

# Expired trades (neither target nor stop hit in the window)
EXPIRED_AS = "SKIP"           # "Bearish" (capital tied up) or "SKIP" (discard)

# Filename encodes the config. Without this, running a second variant
# resumes from the first one's file, silently mixing label definitions —
# and since the old file already holds 2000 rows, the new run would hit
# its target instantly and save the OLD data under the NEW name.
def _config_tag() -> str:
    if LABEL_MODE == "BEAT_BENCHMARK":
        return f"bench{BENCHMARK_TICKER.lower()}"
    return "tpsl_resolved" if EXPIRED_AS == "SKIP" else "tpsl_all"

OUTPUT_FILE = f"dataset_pillar2_{HORIZON.lower()}_v39.json"
GROWTH_WEIGHT = 0.75   # 60% growth, 40% defensive

# ==========================================
# HORIZON CONFIGS
# ==========================================
HORIZON_CONFIGS = {
    "SHORT": {
        "interval":        "1h",
        "lookahead_bars":  70,
        "atr_stop_mult":   3.0,
        "atr_target_mult": 3.5,
        "max_stop_pct":    0.06,
        "min_profit_pct":  0.04,
        "min_rr":          1.2,
        "history_period":  "730d",
        "min_past_bars":   250,
        "prompt_context":  "Short-Term (1-2 weeks, using an Hourly 1H Chart)",
    },
    "MID": {
        "interval":        "1d",
        "lookahead_bars":  60,
        "atr_stop_mult":   2.5,
        "atr_target_mult": 4.0,
        "max_stop_pct":    0.15,
        "min_profit_pct":  0.08,
        "min_rr":          1.5,
        "history_period":  "5y",     
        "min_past_bars":   250,
        "prompt_context":  "Mid-Term (1-3 months, using a Daily 1D Chart)",
    },
    "LONG": {
        "interval":        "1d",
        "lookahead_bars":  250,
        "atr_stop_mult":   4.5,
        "atr_target_mult": 10.0,
        "max_stop_pct":    0.20,
        "min_profit_pct":  0.20,
        "min_rr":          2.0,
        "history_period":  "10y",
        "min_past_bars":   250,
        "prompt_context":  "Long-Term (3-12 months, using a Daily 1D Chart)",
    },
}

# ==========================================
# TICKER UNIVERSE
# ==========================================
GROWTH_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI", "CRDO"
]

DEFENSIVE_TICKERS = [
    "JPM", "BAC", "V", "MA", "BRK-B", "GS",
    "UNH", "JNJ", "ABBV", "MRK", "PFE",
    "PG", "KO", "PEP", "COST",
    "XOM", "CVX", "CAT", "GE", "LMT", "BA",
]



# ==========================================
# INDICATOR HELPERS
# ==========================================
def calculate_rsi(prices, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = pd.Series(prices).diff()
    gain   = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
    loss   = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
    rs     = gain / loss
    rsi    = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_atr(hist: pd.DataFrame, period: int = 14) -> float:
    high_low   = hist["High"] - hist["Low"]
    high_close = (hist["High"] - hist["Close"].shift()).abs()
    low_close  = (hist["Low"]  - hist["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr        = true_range.rolling(window=period).mean()
    return round(float(atr.iloc[-1]), 2) if not pd.isna(atr.iloc[-1]) else 0.0


# ==========================================
# CANDLESTICK PATTERNS
# ==========================================
def compute_candlestick_patterns(bars: list, resistance_1: float) -> dict:
    """Computed inline so no separate conversion step is ever needed."""
    p = {}

    for i, b in enumerate(bars):
        o, h, l, c = b["open"], b["high"], b["low"], b["close"]
        rng  = (h - l) if h != l else 1e-9
        body = abs(c - o)
        uw   = h - max(o, c)
        lw   = min(o, c) - l
        tag  = f"bar_{i+1}"
        p[f"{tag}_body_ratio"]       = round(body / rng, 4)
        p[f"{tag}_upper_wick_ratio"] = round(uw   / rng, 4)
        p[f"{tag}_lower_wick_ratio"] = round(lw   / rng, 4)
        p[f"{tag}_is_bearish"]       = int(c < o)
        p[f"{tag}_is_bullish"]       = int(c >= o)

    last = bars[-1]
    o5, h5, l5, c5 = last["open"], last["high"], last["low"], last["close"]
    rng5  = (h5 - l5) if h5 != l5 else 1e-9
    uw5   = h5 - max(o5, c5)
    lw5   = min(o5, c5) - l5
    body5 = abs(c5 - o5)

    p["is_shooting_star"] = int(uw5 > 2 * body5 and (c5 - l5) / rng5 < 0.35 and c5 < o5)
    p["is_hammer"]        = int(lw5 > 2 * body5 and (h5 - c5) / rng5 < 0.35 and c5 > o5)

    if len(bars) >= 2:
        prev   = bars[-2]
        o4, c4 = prev["open"], prev["close"]
        p["is_bearish_engulfing"] = int(c5 < o5 and c4 > o4 and o5 >= c4 and c5 <= o4)
        p["is_bullish_engulfing"] = int(c5 > o5 and c4 < o4 and o5 <= c4 and c5 >= o4)
    else:
        p["is_bearish_engulfing"] = 0
        p["is_bullish_engulfing"] = 0

    closes  = [b["close"]  for b in bars]
    volumes = [b["volume"] for b in bars]
    p["5bar_close_slope"]   = round((closes[-1]  - closes[0])  / (closes[0]  + 1e-9), 4)
    p["volume_trend_slope"] = round((volumes[-1] - volumes[0]) / (volumes[0] + 1e-9), 4)

    touched   = abs(h5 - resistance_1) / (resistance_1 + 1e-9) < 0.01
    strong_uw = (uw5 / rng5) > 0.40
    p["is_rejecting_resistance"] = int(touched and strong_uw)

    return p


# ==========================================
# BUILD TECHNICAL SNAPSHOT (point-in-time)
# ==========================================
def build_snapshot(df_past: pd.DataFrame, interval: str) -> dict:
    """
    Everything here uses ONLY df_past — no future data touches the snapshot.
    """
    current_close = float(round(df_past["Close"].iloc[-1], 2))

    # Last 5 bars OHLCV
    bars = []
    for date, row in df_past.tail(5).iterrows():
        bars.append({
            "date":   date.strftime("%Y-%m-%d %H:%M") if interval == "1h" else date.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    # Volume
    avg_vol_20 = df_past["Volume"].tail(20).mean()
    cur_vol    = df_past["Volume"].iloc[-1]
    vol_pct    = round((cur_vol / avg_vol_20) * 100) if avg_vol_20 > 0 else 100

    recent_10  = df_past.tail(10)
    up_vols    = recent_10[recent_10["Close"] > recent_10["Open"]]["Volume"]
    down_vols  = recent_10[recent_10["Close"] < recent_10["Open"]]["Volume"]
    avg_up     = up_vols.mean()   if not up_vols.empty   else 0
    avg_down   = down_vols.mean() if not down_vols.empty else 0

    if avg_down > avg_up * 1.15:
        vol_trend = "Distribution (Higher volume on down periods)"
    elif avg_up > avg_down * 1.15:
        vol_trend = "Accumulation (Higher volume on up periods)"
    else:
        vol_trend = "Neutral (Balanced volume flow)"

    # VWAP
    typical = (df_past["High"] + df_past["Low"] + df_past["Close"]) / 3
    vwap_s  = (typical * df_past["Volume"]).rolling(20).sum() / df_past["Volume"].rolling(20).sum()
    vwap    = float(round(vwap_s.iloc[-1], 2)) if not pd.isna(vwap_s.iloc[-1]) else current_close

    # Bollinger
    sma20 = df_past["Close"].rolling(20).mean()
    std20 = df_past["Close"].rolling(20).std()
    if pd.isna(sma20.iloc[-1]):
        bb_up = bb_lo = bb_mid = current_close
    else:
        bb_up  = float(round((sma20 + std20 * 2).iloc[-1], 2))
        bb_lo  = float(round((sma20 - std20 * 2).iloc[-1], 2))
        bb_mid = float(round(sma20.iloc[-1], 2))

    # Levels & Fibonacci
    swing_low  = float(round(df_past["Low"].tail(30).min(),  2))
    swing_high = float(round(df_past["High"].tail(30).max(), 2))
    support_1  = float(round(df_past["Low"].tail(10).min(),  2))
    resist_1   = float(round(df_past["High"].tail(10).max(), 2))

    diff      = swing_high - swing_low
    fib_0786  = float(round(swing_high - diff * 0.786, 2))
    fib_0618  = float(round(swing_high - diff * 0.618, 2))
    fib_0382  = float(round(swing_high - diff * 0.382, 2))

    # Trend indicators
    ema_200 = float(round(df_past["Close"].ewm(span=200, adjust=False).mean().iloc[-1], 2))
    ema_20  = float(round(df_past["Close"].ewm(span=20, adjust=False).mean().iloc[-1], 2)) # 🌟 NEW FEATURE
    atr_14  = calculate_atr(df_past)
    rsi_14  = calculate_rsi(df_past["Close"].values)

    exp1   = df_past["Close"].ewm(span=12, adjust=False).mean()
    exp2   = df_past["Close"].ewm(span=26, adjust=False).mean()
    macd   = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    m, s   = float(macd.iloc[-1]), float(signal.iloc[-1])

    if   m > s and m > 0:  macd_status = "Strong Bullish"
    elif m > s and m <= 0: macd_status = "Bullish Crossover (Recovery)"
    elif m < s and m > 0:  macd_status = "Bearish Crossover (Pullback)"
    else:                  macd_status = "Strong Bearish"

    if   current_close > ema_200 and rsi_14 > 50: trend = "Uptrend"
    elif current_close < ema_200 and rsi_14 < 50: trend = "Downtrend"
    else:                                         trend = "Sideways / Consolidation"

    snapshot = {
        "timeframe":            "Hourly (1H)" if interval == "1h" else "Daily (1D)",
        "current_price":        current_close,
        "last_5_bars_ohlcv":    bars,
        "graph_trend":          trend,
        "ema_20":               ema_20,
        "ema_200":              ema_200,
        "vwap_20":              vwap,
        "bollinger_bands":      {"upper": bb_up, "mid_sma20": bb_mid, "lower": bb_lo},
        "rsi_14":               rsi_14,
        "macd_status":          macd_status,
        "volume_vs_avg_20":     f"{vol_pct}%",
        "volume_profile_trend": vol_trend,
        "atr_14":               atr_14,
        "key_levels": {
            "support_1":               support_1,
            "support_2_swing_low":     swing_low,
            "resistance_1":            resist_1,
            "resistance_2_swing_high": swing_high,
        },
        "fibonacci": {
            "fib_0.382": fib_0382,
            "fib_0.618": fib_0618,
            "fib_0.786": fib_0786,
        },
    }

    snapshot["candlestick_patterns"] = compute_candlestick_patterns(bars, resist_1)
    return snapshot


# ==========================================
# OUTCOME LABEL — pure future price action
# ==========================================
def compute_outcome_label(snapshot: dict, df_future: pd.DataFrame, config: dict) -> tuple:
    """
    Simulates the trade and returns (label, trade_info).

    NO technical filter here — the label is purely what the PRICE DID.

    Levels and the forward walk both come from trade_config, so labels are
    built with exactly the rules the backtest grades with. Under
    GEOMETRY_MODE = "FIXED" every trade carries the same R:R, which is what
    makes win rate a measure of direction prediction rather than a measure
    of which trades happened to have nearer targets.
    """
    entry = float(snapshot["current_price"])
    atr   = float(snapshot["atr_14"])

    lv = compute_levels(entry, atr, HORIZON)

    if not lv.get("valid"):
        if lv.get("reason") in ("bad_entry", "bad_atr", "zero_risk"):
            return "SKIP", {}
        # Only ATR mode can fail the R:R bar; FIXED never does
        return "Bearish", {"entry": entry, "rr": lv.get("rr", 0),
                           "outcome": "invalid_rr"}

    res = walk_trade(df_to_bars(df_future), lv["entry"], lv["stop"], lv["target"])

    info = {
        "entry":           lv["entry"],
        "stop":            lv["stop"],
        "target":          lv["target"],
        "rr":              lv["rr"],
        "stop_pct":        lv["stop_pct"],
        "target_pct":      lv["target_pct"],
        "breakeven_wr":    lv["breakeven_wr"],
        "geometry":        lv["geometry"],
        "outcome":         res["outcome"],
        "bars_held":       res["bars_held"],
        "max_gain_pct":    res["max_gain_pct"],
        "exit_return_pct": res["exit_return_pct"],
        "label_mode":      LABEL_MODE,
        "expired_as":      EXPIRED_AS,
    }

    if res["outcome"] == "target_hit":
        return "Bullish", info
    if res["outcome"] == "stop_hit":
        return "Bearish", info

    if EXPIRED_AS == "SKIP":
        return "SKIP", info
    return "Bearish", info

# ==========================================
# OPTIONAL STRUCTURAL PREFILTER 
# ==========================================
def passes_structural_filter(snapshot: dict) -> bool:
    price    = float(snapshot["current_price"])
    ema200   = float(snapshot["ema_200"])
    vwap     = float(snapshot["vwap_20"])
    bb_upper = float(snapshot["bollinger_bands"]["upper"])
    macd     = snapshot["macd_status"]
    vol_prof = snapshot["volume_profile_trend"]
    rsi      = float(snapshot["rsi_14"])
    res_2    = float(snapshot["key_levels"]["resistance_2_swing_high"])
    vol_pct  = float(snapshot["volume_vs_avg_20"].replace("%", ""))

    macro_room     = price < (res_2 * 0.98)
    macro_breakout = price >= res_2

    return (
        price > ema200 and
        price > vwap and
        (price < bb_upper * 0.985 or macro_breakout) and
        "Distribution" not in vol_prof and
        "Bearish" not in macd and
        40.0 <= rsi <= 72.0 and
        (macro_room or macro_breakout) and
        vol_pct >= 105.0
    )


# ==========================================
# TEMPLATE RATIONALE (replaces the LLM)
# ==========================================
def build_rationale(snapshot: dict, label: str, info: dict) -> str:
    price   = snapshot["current_price"]
    ema200  = snapshot["ema_200"]
    rsi     = snapshot["rsi_14"]
    macd    = snapshot["macd_status"]
    vol     = snapshot["volume_vs_avg_20"]
    trend   = snapshot["graph_trend"]
    vprof   = snapshot["volume_profile_trend"].split(" (")[0]
    ema_rel = "above" if price > ema200 else "below"
    ema_pct = abs((price - ema200) / ema200) * 100

    base = (
        f"Price at ${price:.2f} sits {ema_pct:.1f}% {ema_rel} the 200 EMA "
        f"(${ema200:.2f}) in a {trend.lower()}. MACD reads {macd} with RSI-14 "
        f"at {rsi:.1f} and relative volume at {vol} ({vprof.lower()} flow)."
    )

    if label == "Bullish":
        return (
            f"{base} The setup offers a {info.get('rr', 0):.1f}:1 reward-to-risk "
            f"with a {info.get('stop_pct', 0):.1f}% stop and "
            f"{info.get('target_pct', 0):.1f}% target."
        )

    reason = {
        "stop_hit":   "the structure broke down and the protective stop was violated",
        "expired":    "price failed to reach the profit objective within the horizon",
        "invalid_rr": "the reward-to-risk ratio does not meet the minimum threshold",
    }.get(info.get("outcome", ""), "the setup does not meet entry criteria")

    return f"{base} No trade is warranted here — {reason}."


# ==========================================
# SAMPLE ONE CANDIDATE
# ==========================================
_price_cache: dict = {}


def get_history(ticker: str, config: dict):
    """Cached per-ticker history so each ticker is downloaded only once."""
    key = (ticker, config["interval"], config["history_period"])
    if key in _price_cache:
        return _price_cache[key]

    try:
        df = yf.Ticker(ticker).history(
            period   = config["history_period"],
            interval = config["interval"],
        )
        if df.empty or len(df) < config["min_past_bars"] + config["lookahead_bars"] + 10:
            _price_cache[key] = None
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        _price_cache[key] = df
        return df
    except Exception:
        _price_cache[key] = None
        return None


_bench_cache = {}


def get_benchmark_series(config: dict):
    """SPY closes, fetched once and reused for every benchmark comparison."""
    key = (BENCHMARK_TICKER, config["interval"], config["history_period"])
    if key in _bench_cache:
        return _bench_cache[key]
    try:
        df = yf.Ticker(BENCHMARK_TICKER).history(
            period=config["history_period"], interval=config["interval"])
        if df.empty:
            _bench_cache[key] = None
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        _bench_cache[key] = df["Close"]
        return _bench_cache[key]
    except Exception:
        _bench_cache[key] = None
        return None


def benchmark_return(config: dict, start_ts, end_ts):
    """SPY return between two timestamps, or None if unavailable."""
    s = get_benchmark_series(config)
    if s is None:
        return None
    window = s[(s.index >= start_ts) & (s.index <= end_ts)]
    if len(window) < 2:
        return None
    return (float(window.iloc[-1]) - float(window.iloc[0])) / float(window.iloc[0]) * 100.0


def compute_benchmark_relative_label(snapshot, df_future, config, start_ts):
    """
    Label = did this stock beat SPY over the same window?

    This asks a question technicals have a real chance at, unlike absolute
    direction which is dominated by market beta.
    """
    entry = float(snapshot["current_price"])
    if entry <= 0 or df_future.empty:
        return "SKIP", {}

    exit_px    = float(df_future["Close"].iloc[-1])
    stock_ret  = (exit_px - entry) / entry * 100.0
    bench_ret  = benchmark_return(config, start_ts, df_future.index[-1])

    if bench_ret is None:
        return "SKIP", {}

    excess = stock_ret - bench_ret
    return ("Bullish" if excess > 0 else "Bearish"), {
        "entry":            round(entry, 2),
        "exit_price":       round(exit_px, 2),
        "stock_return_pct": round(stock_ret, 2),
        "bench_return_pct": round(bench_ret, 2),
        "excess_return_pct": round(excess, 2),
        "exit_return_pct":  round(excess, 2),   
        "outcome":          "target_hit" if excess > 0 else "stop_hit",
        "bars_held":        len(df_future),
        "geometry":         "BENCHMARK_RELATIVE",
        "label_mode":       LABEL_MODE,
    }


def sample_candidate(ticker: str, config: dict, used_keys: set):
    """Returns (label, row_dict) or (None, None)."""
    df_full = get_history(ticker, config)
    if df_full is None:
        return None, None

    lookahead = config["lookahead_bars"]
    min_past  = config["min_past_bars"]

    valid_range = range(min_past, len(df_full) - lookahead)

    # Only sample dates the backtest could also sample, so the training
    # distribution and the evaluation distribution cover the same regimes
    if MAX_LOOKBACK_DAYS is not None:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=MAX_LOOKBACK_DAYS)
        valid_range = [i for i in valid_range if df_full.index[i] >= cutoff]

    valid_range = list(valid_range)
    if len(valid_range) < 1:
        return None, None

    split_idx = random.choice(valid_range)
    df_past   = df_full.iloc[:split_idx]
    df_future = df_full.iloc[split_idx : split_idx + lookahead]

    if len(df_future) < lookahead * 0.8:
        return None, None

    signal_date = df_past.index[-1]
    date_str    = (signal_date.strftime("%Y-%m-%d %H:%M")
                   if config["interval"] == "1h"
                   else signal_date.strftime("%Y-%m-%d"))

    # Dedup
    key = (ticker, date_str)
    if key in used_keys:
        return None, None

    try:
        snapshot = build_snapshot(df_past, config["interval"])
    except Exception:
        return None, None

    # Optional sampling prefilter — NOT part of the label
    if USE_STRUCTURAL_PREFILTER and not passes_structural_filter(snapshot):
        return None, None

    if LABEL_MODE == "BEAT_BENCHMARK":
        label, info = compute_benchmark_relative_label(
            snapshot, df_future, config, df_future.index[0])
    else:
        label, info = compute_outcome_label(snapshot, df_future, config)
    if label == "SKIP":
        return None, None

    row = {
        "instruction": (
            f"Analyze the following technical snapshot for {ticker}. "
            f"Provide a {config['prompt_context']} trading strategy with "
            f"technical sentiment, setup type, and rationale."
        ),
        "ticker":      ticker,
        "signal_date": date_str,
        "input":       snapshot,
        "output": {
            "rationale":          build_rationale(snapshot, label, info),
            "technical_sentiment": label,
            "trading_setup_type": ("Trend Following Pullback" if label == "Bullish"
                                   else "Avoid / No Trade"),
            "label_source":       "pure_outcome",
            "trade_simulation":   info,
        },
    }
    used_keys.add(key)
    return label, row


# ==========================================
# MAIN GENERATION LOOP
# ==========================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    config = HORIZON_CONFIGS[HORIZON]

    if USE_NATURAL_RATE:
        # No quotas — every sampled candidate is kept, so the class balance
        # is whatever the market actually produced
        bull_quota = bear_quota = TARGET_DATASET_SIZE
    else:
        bull_quota = int(TARGET_DATASET_SIZE * TARGET_BULLISH_PCT)
        bear_quota = TARGET_DATASET_SIZE - bull_quota

    print("=" * 60)
    print(f"Dataset Generator v6 — {HORIZON} Horizon")
    print("=" * 60)
    print(f"Target size:        {TARGET_DATASET_SIZE}")
    if USE_NATURAL_RATE:
        print(f"Class balance:      NATURAL (no quota)")
    else:
        print(f"Bullish quota:      {bull_quota} ({TARGET_BULLISH_PCT:.0%})")
        print(f"Bearish quota:      {bear_quota} ({1-TARGET_BULLISH_PCT:.0%})")
    print(f"Label mode:         {LABEL_MODE}")
    print(f"Sample window:      last {MAX_LOOKBACK_DAYS} days"
          if MAX_LOOKBACK_DAYS else "Sample window:      all history")
    print(f"Structural filter:  {'ON (sampling only)' if USE_STRUCTURAL_PREFILTER else 'OFF ← recommended'}")
    print(f"Expired handling:   {EXPIRED_AS}")
    print(f"Window:             {config['lookahead_bars']} bars")
    print(f"Geometry:           {describe_geometry(HORIZON)}")
    print(f"Stop:  ATR x{config['atr_stop_mult']} (max {config['max_stop_pct']:.0%})")
    print(f"Target: ATR x{config['atr_target_mult']} (min {config['min_profit_pct']:.0%})")
    print("=" * 60 + "\n")

    bull_rows, bear_rows = [], []
    used_keys = set()
    attempts  = 0
    max_attempts = TARGET_DATASET_SIZE * 60   # generous ceiling

    # Resume support
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            # Refuse to resume from rows built under a different label
            # definition — mixing them produces a dataset that means nothing
            bad = 0
            for r in existing:
                sim = r.get("output", {}).get("trade_simulation", {})
                row_mode = sim.get("label_mode", "TP_SL")
                if row_mode != LABEL_MODE:
                    bad += 1
                    continue
                lbl = r["output"]["technical_sentiment"]
                used_keys.add((r.get("ticker"), r.get("signal_date")))
                (bull_rows if lbl == "Bullish" else bear_rows).append(r)

            if bad:
                print(f"Discarded {bad} rows built under a different LABEL_MODE")
            print(f"Resumed: {len(bull_rows)} Bullish, {len(bear_rows)} Bearish\n")
        except Exception:
            print("Could not load existing file — starting fresh\n")

    def _done():
        if USE_NATURAL_RATE:
            return (len(bull_rows) + len(bear_rows)) >= TARGET_DATASET_SIZE
        return len(bull_rows) >= bull_quota and len(bear_rows) >= bear_quota

    while not _done() and attempts < max_attempts:
        attempts += 1

        group  = random.choices(["GROWTH", "DEFENSIVE"],
                                weights=[GROWTH_WEIGHT, 1 - GROWTH_WEIGHT])[0]
        ticker = random.choice(GROWTH_TICKERS if group == "GROWTH" else DEFENSIVE_TICKERS)

        label, row = sample_candidate(ticker, config, used_keys)
        if label is None:
            continue

        # Respect quotas — this is honest balancing, and the true base
        # rate is reported at the end so you know what you're working with
        if label == "Bullish":
            if not USE_NATURAL_RATE and len(bull_rows) >= bull_quota:
                continue
            bull_rows.append(row)
        else:
            if not USE_NATURAL_RATE and len(bear_rows) >= bear_quota:
                continue
            bear_rows.append(row)

        total = len(bull_rows) + len(bear_rows)
        if total % 25 == 0:
            pct = total / TARGET_DATASET_SIZE * 100
            rate = len(bull_rows) / total if total else 0
            print(f"  [{total}/{TARGET_DATASET_SIZE}] {pct:5.1f}% | "
                  f"Bull {len(bull_rows)} ({rate:.0%}) | Bear {len(bear_rows)} | "
                  f"attempts {attempts}")

        if total % 100 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(bull_rows + bear_rows, f, indent=4, ensure_ascii=False)

    # ── Sort chronologically ────────────────────────────────────────
    # CRITICAL: TimeSeriesSplit assumes rows are in time order.
    # Unsorted data makes CV folds meaningless.
    dataset = bull_rows + bear_rows
    dataset.sort(key=lambda r: r.get("signal_date", ""))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    # ── Summary ─────────────────────────────────────────────────────
    n_bull  = len(bull_rows)
    n_bear  = len(bear_rows)
    n_total = len(dataset)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total rows:    {n_total}")
    print(f"Bullish:       {n_bull:5d}  ({n_bull/n_total:.1%})")
    print(f"Bearish:       {n_bear:5d}  ({n_bear/n_total:.1%})")
    print(f"Attempts:      {attempts}")
    print(f"Hit rate:      {n_total/attempts:.1%}  (rows kept per candidate sampled)")

    if n_total > 0:
        dates = [r["signal_date"][:10] for r in dataset if r.get("signal_date")]
        if dates:
            print(f"Date range:    {min(dates)} → {max(dates)}")

        tickers = {}
        for r in dataset:
            tickers[r["ticker"]] = tickers.get(r["ticker"], 0) + 1
        print(f"Unique tickers: {len(tickers)}")
        top = sorted(tickers.items(), key=lambda x: -x[1])[:5]
        print(f"Most sampled:  {', '.join(f'{t}({c})' for t, c in top)}")

        # Outcome breakdown on the Bearish side
        outcomes = {}
        for r in bear_rows:
            o = r["output"]["trade_simulation"].get("outcome", "?")
            outcomes[o] = outcomes.get(o, 0) + 1
        if outcomes:
            print(f"\nBearish outcome breakdown:")
            for o, c in sorted(outcomes.items(), key=lambda x: -x[1]):
                print(f"  {o:12s}: {c:4d}  ({c/n_bear:.1%})")

    if attempts >= max_attempts:
        print(f"\nHit attempt ceiling before filling quotas.")
        print(f"   Lower TARGET_DATASET_SIZE or widen MAX_LOOKBACK_DAYS.")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"  1. In class_xgboost_v5.py set:")
    print(f"       DATASET_FILE = '{OUTPUT_FILE}'")
    print(f"       label_map    = {{'Bearish': 0, 'Avoid / No Trade': 0, 'Bullish': 1}}")
    print(f"  2. Rows are already sorted by date — TimeSeriesSplit will be valid.")
    print(f"  3. candlestick_patterns are already included — no conversion needed.")


if __name__ == "__main__":
    main()