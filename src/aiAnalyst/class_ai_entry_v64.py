"""
CLEANED IN V106: removed 27 functions, 13 constants, the old run section and 4
imports that nothing in the current project uses. The full original is in
log/originals_v106/class_ai_entry_v64.py.

"""
#!/usr/bin/env python3
"""
class_ai_entry_v64.py - an XGBoost model for WHEN TO BUY.

WHAT V40 WAS, AND WHAT IS DIFFERENT HERE

V40 reported a 56% win rate and a -0.465 correlation with ATR. The second number
was the warning: most of what it had learned was "low volatility names win more
often", which is a volatility fact dressed as a timing signal. Later versions
added features and lost the thread, and V44's dataset turned out to contain a
column that correlated 0.22 with the FUTURE and 0.00 with the past - a leak that
survived several versions because nothing was checking for one.

So this file is built around three commitments.

    THE LABEL IS A TRADE, NOT A RETURN.  "Should I buy" is not "will the price
    be higher in 20 days". It is "will this trade, with the stop and target this
    system would actually place, reach its target before its stop". That is the
    triple-barrier label, and the barriers come from compute_levels() - the same
    geometry the deployed pipeline uses. A model trained on raw forward returns
    optimises something nobody trades.

    EVERY FEATURE IS BUILT HERE, FROM OHLCV.  No inherited dataset. If a feature
    is wrong, it is wrong in code that can be read, not in a CSV whose provenance
    is gone.

    THE RESULT IS NOT REPORTED UNTIL IT SURVIVES A LEAK AUDIT.  Three checks run
    before any performance number is printed, and a failure is fatal:

        structural   features that must correlate by construction actually do
                     (RSI with trailing return, vol_20 with vol_60). The V44
                     dataset failed this at 0.002 and would have been caught
                     on day one.
        univariate   no single feature may beat the whole model out of sample,
                     and no backward-looking feature may correlate more with
                     the future than with the past.
        shuffle      labels permuted WITHIN each date, model refitted. If that
                     scores above chance, the pipeline leaks and every other
                     number in the file is meaningless.

WHAT IS BEING PREDICTED, PRECISELY

For each (ticker, date) with enough history: place a long entry at the close,
a stop and target from compute_levels() at that day's ATR, and hold for at most
the horizon's eval_days.

    win   the target is touched before the stop          label 1
    loss  the stop is touched before the target          label 0
    flat  neither, the clock runs out                    label 0

Flat counts as a loss because the question is whether the trade PAID, not
whether it avoided disaster. The R multiple of every outcome is recorded
separately, so expectancy is measured rather than inferred from the win rate.

THE NUMBER THAT MATTERS IS NOT THE WIN RATE

With this geometry the reward-to-risk ratio is roughly 1.66, so break-even sits
near 37.6%. A 56% win rate means nothing on its own - a model that only fires on
easy setups can post a high win rate and lose money after costs. Expectancy in
R, measured out of sample on the decile the model actually likes, is the number
to read.

NOTHING HERE OVERRIDES PILLAR 2

This is a direction/timing model. The risk model sizes whatever it selects.
A high score is not permission to take a bigger position.

USAGE
  python class_ai_entry_v64.py
"""

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2
from trade_config import HORIZON_CONFIGS, compute_levels

# =============================================================================
# CONFIG
# =============================================================================
PRICE_CACHE = P2.PRICE_CACHE
HORIZON = "MID"
STEP = 5                        # trading days between candidate entries
MIN_HISTORY = 260


LABEL_MODE = "barrier"
                           # mattered counts more than one that did not


XGB_PARAMS = dict(n_estimators=350, max_depth=4, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=3.0, min_child_weight=50,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")


# =============================================================================
# FEATURES  (all from OHLCV, all backward-looking)
# =============================================================================
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def feature_panel(df):
    """
    Every feature for one ticker, as columns aligned to the price index.

    Each one uses bars up to and including the current bar and never beyond.
    That is the property the leak audit exists to verify rather than assume.
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    out["px_vs_ema20"] = c / e20 - 1
    out["px_vs_ema50"] = c / e50 - 1
    out["px_vs_ema200"] = c / e200 - 1
    out["ema20_vs_ema50"] = e20 / e50 - 1
    out["ema50_vs_ema200"] = e50 / e200 - 1

    for n in (5, 20, 60, 120):
        out[f"ret_{n}"] = c / c.shift(n) - 1
    # 12-month momentum skipping the last month, the classic construction
    out["ret_250_ex_20"] = c.shift(20) / c.shift(250) - 1

    out["rsi_14"] = _rsi(c, 14)
    rmax, rmin = h.rolling(60).max(), l.rolling(60).min()
    out["range60_position"] = (c - rmin) / (rmax - rmin).replace(0, np.nan)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_position"] = (c - ma20) / (2 * sd20).replace(0, np.nan)
    out["bb_width"] = (4 * sd20) / ma20.replace(0, np.nan)

    yz = P2.vol_feature_panel(df)
    for k in P2.VOL_FEATURES:
        out[k] = yz[k]
    atr = _atr(df, 14)
    out["atr_pct"] = atr / c
    out["vol_ratio_20_60"] = (c.pct_change().rolling(20).std()
                              / c.pct_change().rolling(60).std().replace(0, np.nan))

    dv = c * v
    out["log_dollar_vol"] = np.log1p(dv.rolling(20).mean())
    out["volume_ratio_20"] = v / v.rolling(20).mean().replace(0, np.nan)

    # ---- PATTERN BLOCK -------------------------------------------------------
    # Candles, VWAP, MACD and distance to the nearest level. Everything here is
    # scaled by ATR or by price, never left as a raw level: a $400 stock and a
    # $4 stock have to land on the same axis or the model learns the ticker.
    for i in (1, 2, 3):
        o, hh, ll, cc = (df["Open"].shift(i - 1), h.shift(i - 1),
                         l.shift(i - 1), c.shift(i - 1))
        rng = (hh - ll).replace(0, np.nan)
        out[f"bar{i}_body"] = (cc - o) / atr.replace(0, np.nan)
        out[f"bar{i}_upper_wick"] = (hh - np.maximum(o, cc)) / rng
        out[f"bar{i}_lower_wick"] = (np.minimum(o, cc) - ll) / rng

    tp = (h + l + c) / 3.0
    vwap20 = ((tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan))
    out["px_vs_vwap20"] = c / vwap20 - 1

    e12, e26 = _ema(c, 12), _ema(c, 26)
    macd = e12 - e26
    signal = _ema(macd, 9)
    out["macd_norm"] = macd / c
    out["macd_hist_norm"] = (macd - signal) / c

    # Distance to the nearest level, in ATR. This is the one genuinely new IDEA
    # in the block - the others are re-encodings of trend and volatility the
    # model already sees. Fibonacci levels are deterministic functions of the
    # same swing high and low, so they carry no information these two do not.
    swing_hi = h.rolling(60).max()
    swing_lo = l.rolling(60).min()
    out["dist_resistance_atr"] = (swing_hi - c) / atr.replace(0, np.nan)
    out["dist_support_atr"] = (c - swing_lo) / atr.replace(0, np.nan)

    out["volume_trend_20"] = (v.rolling(20).mean()
                              / v.rolling(60).mean().replace(0, np.nan) - 1)

    out["_atr_abs"] = atr           # for the barriers, not a feature
    return out


FEATURES = ["px_vs_ema20", "px_vs_ema50", "px_vs_ema200", "ema20_vs_ema50",
            "ema50_vs_ema200", "ret_5", "ret_20", "ret_60", "ret_120",
            "ret_250_ex_20", "rsi_14", "range60_position", "bb_position",
            "bb_width", "yz5", "yz22", "yz66", "atr_pct", "vol_ratio_20_60",
            "log_dollar_vol", "volume_ratio_20"]
# From the technical_data list: candles, VWAP, MACD, distance to levels.
# NOT included, on purpose: raw price and raw ATR (not scale-free, the model
# would learn the ticker), timeframe (constant), and the Fibonacci levels
# (deterministic functions of the same 60-day swing high and low that
# dist_support_atr and dist_resistance_atr already carry).
PATTERN_FEATURES = ["bar1_body", "bar1_upper_wick", "bar1_lower_wick",
                    "bar2_body", "bar2_upper_wick", "bar2_lower_wick",
                    "bar3_body", "bar3_upper_wick", "bar3_lower_wick",
                    "px_vs_vwap20", "macd_norm", "macd_hist_norm",
                    "dist_resistance_atr", "dist_support_atr",
                    "volume_trend_20"]

MKT_LEVEL = ["mkt_breadth_ema200", "mkt_ret_20", "mkt_vol_20"]
# The two CHANGE columns were added to fix the 2020 failure. On the real sweep
# they moved the worst year from -10.2% to -8.2% and cost 0.082R of edge - two
# extra market columns is a lot of capacity against ~20-30 independent regime
# episodes. "no_confirm" below exists so that trade can be measured rather than
# argued about.
MKT_CONFIRM = ["mkt_breadth_chg_20", "mkt_vol_chg_20"]
MKT_FEATURES = MKT_LEVEL + MKT_CONFIRM


# =============================================================================
# LABEL  (triple barrier, this system's own geometry)
# =============================================================================
def triple_barrier(high, low, close, entry, stop, target, n_bars):
    """
    Which barrier is touched first over the next `n_bars`.

    Returns (label, r_multiple, bars_held, outcome).

    Within a single bar both barriers can be inside the range. That is
    genuinely ambiguous at daily resolution, so it is resolved AGAINST the
    trade - stop first. Resolving it in favour would inflate the win rate by
    exactly the amount that is hardest to argue for.
    """
    risk = entry - stop
    if risk <= 0 or n_bars <= 0:
        return None
    hi = np.asarray(high[:n_bars], float)
    lo = np.asarray(low[:n_bars], float)
    cl = np.asarray(close[:n_bars], float)
    hit_t = np.flatnonzero(hi >= target)
    hit_s = np.flatnonzero(lo <= stop)
    t_i = hit_t[0] if hit_t.size else np.inf
    s_i = hit_s[0] if hit_s.size else np.inf

    if t_i == np.inf and s_i == np.inf:
        exit_px = cl[-1]
        return (0, (exit_px - entry) / risk, len(cl), "flat")
    if s_i <= t_i:                      # ties go to the stop, on purpose
        return (0, -1.0, int(s_i) + 1, "loss")
    return (1, (target - entry) / risk, int(t_i) + 1, "win")


# =============================================================================
# DATASET
# =============================================================================
def apply_label_mode(d, mode=LABEL_MODE):
    """Swap in the chosen label. Both are kept on the frame either way."""
    if mode not in ("barrier", "profit"):
        raise RuntimeError(f'LABEL_MODE must be "barrier" or "profit", not {mode!r}')
    d = d.copy()
    d["label"] = d["label_barrier" if mode == "barrier" else "label_profit"]
    return d


def build_dataset(tickers, cache_dir=None, horizon=HORIZON, step=STEP,
                  verbose=True):
    """Per-ticker features and labels, then cross-sectional and market columns."""
    cache_dir = cache_dir or PRICE_CACHE
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]

    frames = []
    for i, t in enumerate(tickers, 1):
        df = P2.load_prices(t, cache_dir)
        if df is None or len(df) < MIN_HISTORY + n_bars + 2:
            continue
        pan = feature_panel(df)
        hi = df["High"].to_numpy(float)
        lo = df["Low"].to_numpy(float)
        cl = df["Close"].to_numpy(float)
        rows = []
        for pos in range(MIN_HISTORY, len(df) - n_bars - 1, step):
            frow = pan.iloc[pos]
            if not np.isfinite(
                    frow[FEATURES + PATTERN_FEATURES].to_numpy(float)).all():
                continue
            entry = cl[pos]
            atr = float(frow["_atr_abs"])
            if not np.isfinite(atr) or atr <= 0 or entry <= 0:
                continue
            lv = compute_levels(entry, atr, horizon)
            if not lv.get("valid"):
                continue
            res = triple_barrier(hi[pos + 1:], lo[pos + 1:], cl[pos + 1:],
                                 entry, lv["stop"], lv["target"], n_bars)
            if res is None:
                continue
            label, r, held, outcome = res
            rec = {"ticker": t, "date": df.index[pos], "entry": entry,
                   "stop_pct": lv["stop_pct"], "target_pct": lv["target_pct"],
                   "rr": lv["rr"], "label": label, "r_multiple": r,
                   "bars_held": held, "outcome": outcome}
            rec.update({k: float(frow[k])
                        for k in FEATURES + PATTERN_FEATURES})
            rows.append(rec)
        if rows:
            frames.append(pd.DataFrame(rows))
        if verbose and (i % 20 == 0 or i == len(tickers)):
            print(f"    [{i}/{len(tickers)}] "
                  f"{sum(len(f) for f in frames):,} candidate entries")
    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, ignore_index=True).sort_values("date")
    d["label_barrier"] = d["label"]
    d["label_profit"] = (d["r_multiple"] > 0).astype(int)

    # cross-sectional: rank within the date, so the model sees relative standing
    for src, dst in (("ret_20", "xs_ret_20"), ("rsi_14", "xs_rsi_14"),
                     ("atr_pct", "xs_atr_pct"),
                     ("px_vs_ema200", "xs_px_vs_ema200")):
        d[dst] = d.groupby("date")[src].rank(pct=True)

    # market: ONE value per date, identical for every name. The V44 dataset had
    # 120 distinct "market" values on a single date; assert_market_columns()
    # below refuses to continue if that happens again.
    g = d.groupby("date")
    d["mkt_breadth_ema200"] = g["px_vs_ema200"].transform(lambda s: (s > 0).mean())
    d["mkt_ret_20"] = g["ret_20"].transform("median")
    d["mkt_vol_20"] = g["yz22"].transform("median")

    # CONFIRMATION, not just level. mkt_breadth_ema200 scores AUC 0.4737 on its
    # own - low breadth goes with wins - which is a contrarian market signal. Its
    # failure mode is exactly 2020: breadth collapsed in March, the level said
    # "buy hard", and the market kept falling for another three weeks. A level
    # cannot tell a bottom from the way down. Its 20-day CHANGE can: low AND
    # turning up is a different state from low AND still falling.
    per_date = (d.groupby("date")[["mkt_breadth_ema200", "mkt_vol_20"]]
                .first().sort_index())
    chg = pd.DataFrame({
        "mkt_breadth_chg_20": per_date["mkt_breadth_ema200"].diff(20),
        "mkt_vol_chg_20": per_date["mkt_vol_20"].diff(20)}).fillna(0.0)
    d = d.merge(chg, left_on="date", right_index=True, how="left")
    d[["mkt_breadth_chg_20", "mkt_vol_chg_20"]] = \
        d[["mkt_breadth_chg_20", "mkt_vol_chg_20"]].fillna(0.0)
    return d.reset_index(drop=True)


def assert_market_columns(d):
    for c in MKT_FEATURES:
        n = d.groupby("date")[c].nunique().max()
        if n > 1:
            raise RuntimeError(
                f"{c} takes {n} values on a single date - it is not a market "
                f"variable. This is the exact defect found in the V44 dataset.")
    return True
