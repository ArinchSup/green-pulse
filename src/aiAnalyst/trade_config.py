"""
trade_config.py — Single Source of Truth for Trade Level Math
==============================================================
Every module that touches stop/target levels imports from here:
  class_pillar2_data_gen*.py   (builds labels)
  class_ai_pillar2.py          (live inference)
  class_ai_pillar2_backtest.py (grading)
  class_ai_pipeline_backtest_v2.py

GEOMETRY_MODE controls how levels are set:

  "VOL_SCALED"  (recommended)
     target = entry x (1 + tp),  tp = clamp(VOL_K x sigma_H, VOL_MIN_TP, VOL_MAX_TP)
     stop   = entry x (1 - tp x SL_TP_RATIO)
     where sigma_H = (ATR / entry) x sqrt(lookahead_bars) — the stock's own
     expected move over the holding window.

     WHY THIS EXISTS. FIXED geometry gives every trade the same R:R, but NOT
     the same chance of reaching its target. Over 60 bars a +20% move is about
     1.9 sigma for a 1.4%-ATR stock and about 0.25 sigma for a 10%-ATR stock.
     The quiet stock therefore almost never resolves — it expires. Measured on
     V40's backtest, expiry rate ran 22.7% in the lowest ATR quintile and 0.0%
     in the highest, and the lowest quintile won 27.9% of the time against
     ~41% everywhere else. So atr_pct never stopped being a difficulty proxy
     under FIXED; it just moved from "whose target is nearer" to "who can
     reach a target at all", and the model learned to prefer names whose
     target was out of reach.

     Because the stop is always SL_TP_RATIO x the target, R:R and the
     breakeven win rate are IDENTICAL for every trade, exactly as under FIXED.
     Win rates stay comparable across trades and across model/baseline arms.

  "FIXED"
     stop   = entry x (1 - FIXED_SL_PCT)
     target = entry x (1 + FIXED_TP_PCT)
     Same R:R on every trade, but reachability varies with volatility — see
     above. Kept for reproducing older V39/V40/V41 datasets.

  "ATR"
     stop   = max(entry - ATR x atr_stop_mult, entry x (1 - max_stop_pct))
     target = max(entry + ATR x atr_target_mult, entry x (1 + min_profit_pct))
     Volatility-adaptive, but the min/max caps let R:R drift per trade, so win
     rates are NOT comparable between trades. This is the mode the FIXED
     docstring was originally written against. Do not confuse it with
     VOL_SCALED, which holds R:R constant by construction.

!!  Changing GEOMETRY_MODE means the dataset must be REGENERATED and the
    model RETRAINED. Labels and grading must use the same geometry.
    geometry_tag() below is meant to go in dataset filenames so a run can
    never silently resume from a file built under different rules.
"""
import math

GEOMETRY_MODE = "VOL_SCALED"   # "VOL_SCALED" | "FIXED" | "ATR"

# ── Shared risk shape ────────────────────────────────────────────────
# stop = SL_TP_RATIO x target, so R:R = 1/SL_TP_RATIO for EVERY trade.
# 0.6 reproduces the old +20%/-12% shape: R:R 1.67, breakeven 37.5%.
SL_TP_RATIO = 0.60

# Used when GEOMETRY_MODE == "VOL_SCALED"
VOL_K          = 0.50   # target = K x sigma over the holding window
VOL_MIN_TP_PCT = 0.08   # floor: below this, costs and noise dominate
VOL_MAX_TP_PCT = 0.40   # ceiling: keeps a 30%-ATR name from getting a +200% target

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


def _round_px(p: float) -> float:
    """
    Round a price without destroying sub-dollar tickers. Rounding UMAC at
    $0.28 to 2 decimals turned an 8% target into a 7.1% one; the universe has
    several names under $1, so the precision has to follow the price.
    """
    if p >= 10:
        return round(p, 2)
    if p >= 1:
        return round(p, 3)
    return round(p, 4)


def expected_move_pct(atr: float, entry: float, horizon: str = "MID") -> float:
    """
    The stock's own 1-sigma move over the holding window, as a fraction.
    ATR is a daily range proxy, so sigma_H ~ atr_pct x sqrt(bars).
    """
    if entry <= 0 or atr <= 0:
        return float("nan")
    bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    return (atr / entry) * math.sqrt(bars)


def compute_levels(entry: float, atr: float, horizon: str = "MID") -> dict:
    """
    THE canonical stop/target calculation. Behaviour depends on
    GEOMETRY_MODE above.

    Returns: valid, reason, entry, stop, target, risk, reward, rr,
             stop_pct, target_pct, breakeven_wr, geometry
             (+ sigma_pct and tp_clamped under VOL_SCALED)
    """
    cfg = HORIZON_CONFIGS[horizon]
    extra = {}

    if entry <= 0:
        return {"valid": False, "reason": "bad_entry"}

    if GEOMETRY_MODE == "VOL_SCALED":
        if atr <= 0:
            return {"valid": False, "reason": "bad_atr"}
        sigma = expected_move_pct(atr, entry, horizon)
        if not math.isfinite(sigma) or sigma <= 0:
            return {"valid": False, "reason": "bad_atr"}

        raw_tp = VOL_K * sigma
        tp_pct = min(max(raw_tp, VOL_MIN_TP_PCT), VOL_MAX_TP_PCT)
        sl_pct = tp_pct * SL_TP_RATIO          # R:R constant by construction

        stop   = _round_px(entry * (1 - sl_pct))
        target = _round_px(entry * (1 + tp_pct))
        min_rr = 0.0
        extra = {
            "sigma_pct":  round(sigma * 100, 2),
            "raw_tp_pct": round(raw_tp * 100, 2),
            # flags trades where the clamp bound, i.e. the target is no longer
            # a pure multiple of the stock's own volatility
            "tp_clamped": ("floor" if raw_tp < VOL_MIN_TP_PCT
                           else "ceiling" if raw_tp > VOL_MAX_TP_PCT else ""),
        }

    elif GEOMETRY_MODE == "FIXED":
        # ATR is not needed for the levels themselves, but a missing ATR
        # signals bad upstream data, so reject it either way.
        if atr <= 0:
            return {"valid": False, "reason": "bad_atr"}
        stop     = _round_px(entry * (1 - FIXED_SL_PCT))
        target   = _round_px(entry * (1 + FIXED_TP_PCT))
        min_rr   = 0.0   # R:R is constant by construction, nothing to filter

    elif GEOMETRY_MODE == "ATR":
        if atr <= 0:
            return {"valid": False, "reason": "bad_atr"}
        raw_stop   = entry - (atr * cfg["atr_stop_mult"])
        min_stop   = entry * (1 - cfg["max_stop_pct"])
        stop       = _round_px(max(raw_stop, min_stop))

        raw_target = entry + (atr * cfg["atr_target_mult"])
        min_target = entry * (1 + cfg["min_profit_pct"])
        target     = _round_px(max(raw_target, min_target))
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
        "entry":        _round_px(entry),
        "stop":         stop,
        "target":       target,
        "risk":         round(risk, 4),
        "reward":       round(reward, 4),
        "rr":           round(rr, 2),
        "stop_pct":     round((risk / entry) * 100, 2),
        "target_pct":   round((reward / entry) * 100, 2),
        "breakeven_wr": round(risk / (risk + reward), 4),
        "geometry":     GEOMETRY_MODE,
        **extra,
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


def geometry_tag() -> str:
    """
    Short, filename-safe description of the CURRENT geometry. Put this in
    dataset filenames: a run must never resume from a file built under
    different level rules.
    """
    if GEOMETRY_MODE == "VOL_SCALED":
        return (f"volk{VOL_K:g}".replace(".", "") +
                f"r{SL_TP_RATIO:g}".replace(".", "") +
                f"_{VOL_MIN_TP_PCT:.0%}-{VOL_MAX_TP_PCT:.0%}".replace("%", ""))
    if GEOMETRY_MODE == "FIXED":
        return f"fixed{FIXED_TP_PCT:.0%}-{FIXED_SL_PCT:.0%}".replace("%", "")
    return "atr"


def describe_geometry(horizon: str = "MID") -> str:
    """One-line summary for logging at the top of any run."""
    if GEOMETRY_MODE == "VOL_SCALED":
        rr = 1.0 / SL_TP_RATIO
        be = SL_TP_RATIO / (1.0 + SL_TP_RATIO)
        bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
        return (f"VOL_SCALED  target {VOL_K:g}x sigma over {bars} bars "
                f"(clamped {VOL_MIN_TP_PCT:.0%}-{VOL_MAX_TP_PCT:.0%}), "
                f"stop {SL_TP_RATIO:g}x target | R:R 1:{rr:.2f} | breakeven {be:.1%}")
    if GEOMETRY_MODE == "FIXED":
        rr = FIXED_TP_PCT / FIXED_SL_PCT
        be = FIXED_SL_PCT / (FIXED_SL_PCT + FIXED_TP_PCT)
        return (f"FIXED  +{FIXED_TP_PCT:.0%} target / -{FIXED_SL_PCT:.0%} stop  "
                f"| R:R 1:{rr:.2f} | breakeven {be:.1%}")
    c = HORIZON_CONFIGS[horizon]
    return (f"ATR  x{c['atr_target_mult']} target (min {c['min_profit_pct']:.0%}) / "
            f"x{c['atr_stop_mult']} stop (max {c['max_stop_pct']:.0%})")


if __name__ == "__main__":
    # Sanity table: what the current geometry does across the ATR range seen
    # in the SHAY universe, and how reachable each target is in sigma terms.
    print(describe_geometry("MID"))
    print(f"\n{'price':>8}{'atr%':>7}{'sigma60%':>10}{'target%':>9}{'stop%':>8}"
          f"{'R:R':>7}{'BE%':>7}{'tp in sigma':>13}{'clamp':>9}")
    print("-" * 78)
    for price, atr_pct in [(100, 1.4), (100, 2.5), (100, 5.0), (100, 8.0),
                           (100, 12.0), (0.28, 6.0), (0.28, 15.0)]:
        lv = compute_levels(price, price * atr_pct / 100, "MID")
        if not lv.get("entry"):
            print(f"{price:>8}{atr_pct:>7}  {lv.get('reason')}")
            continue
        sigma = expected_move_pct(price * atr_pct / 100, price, "MID") * 100
        print(f"{price:>8}{atr_pct:>7.1f}{sigma:>10.1f}{lv['target_pct']:>9.1f}"
              f"{lv['stop_pct']:>8.1f}{lv['rr']:>7.2f}{lv['breakeven_wr']*100:>7.1f}"
              f"{lv['target_pct']/sigma:>13.2f}{lv.get('tp_clamped', ''):>9}")
