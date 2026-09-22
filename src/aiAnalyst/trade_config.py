"""
trade_config.py — Single Source of Truth for Trade Level Math
==============================================================
Every module that touches stop/target levels imports from here:
  class_pillar2_data_gen.py  (builds labels)
  class_ai_pillar2.py        (live inference)
  class_ai_pillar2_backtest.py (grading)

GEOMETRY_MODE controls how levels are set:

  "FIXED"  (recommended)
     stop   = entry x (1 - FIXED_SL_PCT)
     target = entry x (1 + FIXED_TP_PCT)
     Every trade has identical R:R, so win rate is comparable across
     trades and across the model/baseline arms. Critically, this stops
     atr_pct from acting as a difficulty proxy: under ATR geometry a
     low-ATR stock got a nearer target and therefore an easier win, so
     the model learned "prefer low volatility" instead of "predict
     direction". That is what produced a +8pp win-rate lift alongside a
     +0.3pp EV lift.

  "ATR"
     stop   = max(entry - ATR x atr_stop_mult, entry x (1 - max_stop_pct))
     target = max(entry + ATR x atr_target_mult, entry x (1 + min_profit_pct))
     Volatility-adaptive, but win rates are not comparable between trades.

!!  Changing GEOMETRY_MODE means the dataset must be REGENERATED and the
    model RETRAINED. Labels and grading must use the same geometry.
"""

GEOMETRY_MODE = "FIXED"   # "FIXED" | "ATR"

# Used when GEOMETRY_MODE == "FIXED"
FIXED_TP_PCT = 0.20   # +20% target
FIXED_SL_PCT = 0.12   # -12% stop  → R:R = 1.67, breakeven = 37.5%

HORIZON_CONFIGS = {
    "SHORT": {
        "interval":        "1h",
        "lookahead_bars":  70,
        "eval_days":       14,
        "atr_stop_mult":   3.0,
        "atr_target_mult": 3.5,
        "max_stop_pct":    0.06,
        "min_profit_pct":  0.04,
        "min_rr":          1.2,
    },
    "MID": {
        "interval":        "1d",
        "lookahead_bars":  60,
        "eval_days":       90,
        "atr_stop_mult":   2.5,
        "atr_target_mult": 4.0,
        "max_stop_pct":    0.15,
        "min_profit_pct":  0.08,
        "min_rr":          1.5,
    },
    "LONG": {
        "interval":        "1d",
        "lookahead_bars":  250,
        "eval_days":       365,
        "atr_stop_mult":   4.5,
        "atr_target_mult": 10.0,
        "max_stop_pct":    0.20,
        "min_profit_pct":  0.20,
        "min_rr":          2.0,
    },
}


def compute_levels(entry: float, atr: float, horizon: str = "MID") -> dict:
    """
    THE canonical stop/target calculation. Behaviour depends on
    GEOMETRY_MODE above.

    Returns: valid, reason, entry, stop, target, risk, reward, rr,
             stop_pct, target_pct, breakeven_wr
    """
    cfg = HORIZON_CONFIGS[horizon]

    if entry <= 0:
        return {"valid": False, "reason": "bad_entry"}

    if GEOMETRY_MODE == "FIXED":
        # ATR is not needed for the levels themselves, but a missing ATR
        # signals bad upstream data, so reject it either way.
        if atr <= 0:
            return {"valid": False, "reason": "bad_atr"}
        stop     = round(entry * (1 - FIXED_SL_PCT), 2)
        target   = round(entry * (1 + FIXED_TP_PCT), 2)
        min_rr   = 0.0   # R:R is constant by construction, nothing to filter

    elif GEOMETRY_MODE == "ATR":
        if atr <= 0:
            return {"valid": False, "reason": "bad_atr"}
        raw_stop   = entry - (atr * cfg["atr_stop_mult"])
        min_stop   = entry * (1 - cfg["max_stop_pct"])
        stop       = round(max(raw_stop, min_stop), 2)

        raw_target = entry + (atr * cfg["atr_target_mult"])
        min_target = entry * (1 + cfg["min_profit_pct"])
        target     = round(max(raw_target, min_target), 2)
        min_rr     = cfg["min_rr"]

    else:
        raise ValueError(f"Unknown GEOMETRY_MODE: {GEOMETRY_MODE}")

    risk   = entry - stop
    reward = target - entry

    if risk <= 0:
        return {"valid": False, "reason": "zero_risk"}

    rr = reward / risk

    return {
        "valid":        rr >= min_rr,
        "reason":       "ok" if rr >= min_rr else "rr_below_min",
        "entry":        round(entry, 2),
        "stop":         stop,
        "target":       target,
        "risk":         round(risk, 4),
        "reward":       round(reward, 4),
        "rr":           round(rr, 2),
        "stop_pct":     round((risk / entry) * 100, 2),
        "target_pct":   round((reward / entry) * 100, 2),
        "breakeven_wr": round(risk / (risk + reward), 4),
        "geometry":     GEOMETRY_MODE,
    }


def walk_trade(bars, entry: float, stop: float, target: float) -> dict:
    """
    THE canonical forward walk. Used by the generator to build labels and
    by the backtest to grade them, so the two cannot disagree.

    Two rules that matter:
      1. `bars` must start the bar AFTER the signal bar. The signal came
         from the signal bar's close; letting that bar resolve the trade
         is look-ahead.
      2. If one bar's low hits the stop and its high hits the target,
         it counts as a LOSS. Intrabar order is unknowable from OHLC, so
         the pessimistic reading keeps the backtest honest.

    Returns: outcome, bars_held, max_gain_pct, exit_return_pct
    """
    max_favour = entry
    bars_held  = 0
    last_close = entry

    for i, bar in enumerate(bars):
        low  = float(bar["low"])
        high = float(bar["high"])
        bars_held  = i + 1
        last_close = float(bar.get("close", last_close))

        # STOP CHECKED FIRST — matches the label generator
        if low <= stop:
            return {
                "outcome":          "stop_hit",
                "bars_held":        bars_held,
                "max_gain_pct":     round(((max_favour - entry) / entry) * 100, 2),
                "exit_return_pct":  round(((stop - entry) / entry) * 100, 2),
            }

        if high > max_favour:
            max_favour = high

        if high >= target:
            return {
                "outcome":          "target_hit",
                "bars_held":        bars_held,
                "max_gain_pct":     round(((max_favour - entry) / entry) * 100, 2),
                "exit_return_pct":  round(((target - entry) / entry) * 100, 2),
            }

    # Expired — mark to the last close, since that is what you would
    # actually realise if you closed the position at the horizon.
    return {
        "outcome":         "expired",
        "bars_held":       bars_held,
        "max_gain_pct":    round(((max_favour - entry) / entry) * 100, 2),
        "exit_return_pct": round(((last_close - entry) / entry) * 100, 2),
    }


def df_to_bars(df) -> list:
    """Convert a yfinance DataFrame into the list-of-dicts walk_trade wants."""
    return [
        {"low": float(r["Low"]), "high": float(r["High"]), "close": float(r["Close"])}
        for _, r in df.iterrows()
    ]


def describe_geometry(horizon: str = "MID") -> str:
    """One-line summary for logging at the top of any run."""
    if GEOMETRY_MODE == "FIXED":
        rr = FIXED_TP_PCT / FIXED_SL_PCT
        be = FIXED_SL_PCT / (FIXED_SL_PCT + FIXED_TP_PCT)
        return (f"FIXED  +{FIXED_TP_PCT:.0%} target / -{FIXED_SL_PCT:.0%} stop  "
                f"| R:R 1:{rr:.2f} | breakeven {be:.1%}")
    c = HORIZON_CONFIGS[horizon]
    return (f"ATR  x{c['atr_target_mult']} target (min {c['min_profit_pct']:.0%}) / "
            f"x{c['atr_stop_mult']} stop (max {c['max_stop_pct']:.0%})")