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
import argparse
import json
import os
import sys

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
MODEL_FILE = "entry_v64.json"

# What counts as a win.
#   "barrier"  the target is touched before the stop. What the system trades,
#              but it scores a trade that drifts up 0.8R and times out as a
#              loss - and the first real run showed the model's WORST decile
#              earning more than its best because of exactly that.
#   "profit"   the trade ends above water in R terms. Closer to the thing you
#              actually want ranked.
# What the model is asked to predict.
#   "classify"    P(win). Ranking by it produced a U-shaped decile table on the
#                 real run: deciles 1 and 10 both earned ~0.31-0.40R while the
#                 middle sagged to ~0.20R, and AUC sat at 0.50 because a rank
#                 statistic cancels on a U.
#   "expectancy"  regress on the R multiple and rank by PREDICTED R. The target
#                 is then the thing being ranked, so a U in any feature is
#                 representable instead of being flattened into a score that
#                 means two different things at its two ends.
PREDICT_MODE = "expectancy"

LABEL_MODE = "barrier"
WEIGHT_BY_R = False        # weight each training row by |R|, so a trade that
                           # mattered counts more than one that did not

MIN_TRAIN_YEARS = 5
N_SEEDS = 3
SEED = 64
TOP_DECILE = 0.10

XGB_PARAMS = dict(n_estimators=350, max_depth=4, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=3.0, min_child_weight=50,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")

# Leak-audit thresholds.
LEAK_UNIVARIATE_AUC = 0.58      # no lone feature should reach this
LEAK_SHUFFLE_AUC = 0.53         # a shuffled model must land near 0.50
STRUCTURAL_MIN_CORR = 0.30      # relationships that must exist, do


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

XS_FEATURES = ["xs_ret_20", "xs_rsi_14", "xs_atr_pct", "xs_px_vs_ema200"]
MKT_LEVEL = ["mkt_breadth_ema200", "mkt_ret_20", "mkt_vol_20"]
# The two CHANGE columns were added to fix the 2020 failure. On the real sweep
# they moved the worst year from -10.2% to -8.2% and cost 0.082R of edge - two
# extra market columns is a lot of capacity against ~20-30 independent regime
# episodes. "no_confirm" below exists so that trade can be measured rather than
# argued about.
MKT_CONFIRM = ["mkt_breadth_chg_20", "mkt_vol_chg_20"]
MKT_FEATURES = MKT_LEVEL + MKT_CONFIRM
ALL_FEATURES = FEATURES + PATTERN_FEATURES + XS_FEATURES + MKT_FEATURES

# Named feature sets. "before" and "after" are the two the technical-data
# question turns on: identical in every other respect, differing only by the
# fifteen columns built from candles, VWAP, MACD and distance to levels.
FEATURE_SETS = {
    "per_name":        FEATURES,
    "+pattern":        FEATURES + PATTERN_FEATURES,
    "+cross_section":  FEATURES + PATTERN_FEATURES + XS_FEATURES,
    "+market":         ALL_FEATURES,
    # the two the pattern question needs, with nothing else moving
    "before_pattern":  FEATURES + XS_FEATURES + MKT_FEATURES,
    "after_pattern":   ALL_FEATURES,
    # market LEVELS only, no change columns - what the +0.089R run actually had
    "no_confirm":      FEATURES + XS_FEATURES + MKT_LEVEL,
    # a meaningless ranking, as a yardstick for every row above
    "random":          FEATURES,
}


def resolve_features(name):
    if name not in FEATURE_SETS:
        raise RuntimeError(f"unknown feature set {name!r}; "
                           f"choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])


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


# =============================================================================
# LEAK AUDIT
# =============================================================================
def _safe_corr(a, b):
    """corr() on a constant column divides by zero and numpy shouts. A constant
    feature carries no information, so nan is the honest answer."""
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan
    return float(a.corr(b))


def auc(y, p):
    y = np.asarray(y, float)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def structural_check(d, verbose=True):
    """
    Relationships that MUST hold if the features are what their names say.
    The V44 dataset put corr(rsi_14, ret_20) at 0.002; it should be strong.
    """
    pairs = [("rsi_14", "ret_20"), ("yz22", "yz66"), ("ret_20", "ret_60"),
             ("bb_position", "range60_position"), ("px_vs_ema20", "px_vs_ema50")]
    bad = []
    if verbose:
        print("\n  LEAK AUDIT 1/3 - structural")
    for a, b in pairs:
        r = _safe_corr(d[a], d[b])
        flag = abs(r) < STRUCTURAL_MIN_CORR
        bad.append((a, b, r)) if flag else None
        if verbose:
            print(f"    corr({a:16s}, {b:16s}) = {r:+.4f}"
                  f"{'   <-- TOO WEAK' if flag else ''}")
    if bad:
        raise RuntimeError(
            "features do not relate the way their names imply "
            + ", ".join(f"{a}/{b}={r:+.3f}" for a, b, r in bad)
            + ". The data is not what it claims to be; fix it before training.")
    return True


def univariate_check(d, features, label="label", verbose=True):
    """
    The leak signature is NOT a high AUC. It is a backward-looking feature that
    knows more about the FUTURE than about the PAST.

    The first version of this check fired on yz66, yz22 and atr_pct at AUC 0.61
    and called them leaks. They are not. The barriers come from compute_levels(),
    which sets them from ATR and then CLAMPS the target between a floor and a
    ceiling - so in volatility units a high-vol name's barriers sit closer than a
    low-vol name's, and volatility genuinely decides how the trade resolves. A
    volatility feature scoring 0.61 on this label is mechanics, not leakage. It
    is also V40's -0.465 ATR correlation, arriving again by a different route.

    What separates the two cases is the pair of correlations below. A leaked
    column looks like V44's rsi_14: 0.22 with the future, 0.00 with the past. A
    real feature looks like yz66: 0.04 with the future, 0.15 with the past.
    """
    y = d[label].to_numpy(int)
    rows = []
    for c in features:
        a = auc(y, d[c].to_numpy(float))
        fwd = abs(_safe_corr(d[c], d["r_multiple"]))
        back = (abs(_safe_corr(d[c], d["ret_20"]))
                if c != "ret_20" else np.nan)
        rows.append((c, a, fwd, back))
    rows.sort(key=lambda t: -abs(t[1] - 0.5))
    if verbose:
        print("\n  LEAK AUDIT 2/3 - univariate  (top 6 by |AUC - 0.50|)")
        print(f"    {'feature':22s} {'AUC':>7s} {'|corr fwd|':>11s} "
              f"{'|corr past|':>12s}  verdict")
        for c, a, f, b in rows[:6]:
            bs = f"{b:12.4f}" if np.isfinite(b) else f"{'-':>12s}"
            v = ("knows the future" if _leaky(f, b) else
                 "reads the past, as it should")
            print(f"    {c:22s} {a:7.4f} {f:11.4f} {bs}  {v}")
    hot = [(c, f, b) for c, _, f, b in rows if _leaky(f, b)]
    if hot:
        raise RuntimeError(
            "features that correlate with the FUTURE far more than with the "
            "PAST: "
            + ", ".join(f"{c} (fwd {f:.3f} vs past {b:.3f})" for c, f, b in hot)
            + ". That is the V44 signature; the column is not what it claims.")
    return rows


def _leaky(fwd, past, floor=0.10, ratio=3.0):
    """Strong link to the outcome AND a much weaker one to the recent past."""
    if not np.isfinite(fwd) or fwd < floor:
        return False
    if not np.isfinite(past):
        return True
    return fwd > ratio * max(past, 1e-6)


def beats_model_check(te, features, verbose=True):
    """
    Run AFTER the walk-forward: no single raw feature may out-rank the fitted
    model out of sample. On the V44 dataset rsi_14 scored 0.6080 against the
    67-feature model's 0.5999, which was the whole result sitting in one column.
    """
    y = te["label"].to_numpy(int)
    model_auc = auc(y, te["p"].to_numpy(float))
    best, best_auc = None, 0.5
    for c in features:
        a = auc(y, te[c].to_numpy(float))
        if np.isfinite(a) and abs(a - 0.5) > abs(best_auc - 0.5):
            best, best_auc = c, a
    if verbose:
        print(f"\n  POST-CHECK - can one raw feature out-rank the model?")
        print(f"    model {model_auc:.4f}   best single feature "
              f"{best} {best_auc:.4f}")
    if abs(best_auc - 0.5) > abs(model_auc - 0.5) + 1e-9:
        direction = ("the same way as" if best_auc > 0.5
                     else "INVERTED relative to")
        print(f"    WARNING: {best} alone carries more ordering information "
              f"than the fitted model, {direction} the label.")
        if best_auc < 0.5:
            print(f"      An AUC below 0.50 means LOW values of {best} go with "
                  f"WINS. That is information,")
            print(f"      not noise - but the model is not using it as well as "
                  f"the raw column does.")
        if best in MKT_FEATURES:
            print(f"      {best} is a MARKET variable, identical for every name "
                  f"on a date. An edge that")
            print(f"      leans on one has as many independent observations as "
                  f"there are market regimes,")
            print(f"      not as there are rows. Check LIFT BY TEST YEAR before "
                  f"believing it.")
    else:
        print(f"    the model is at least as good as its best single input")
    return {"model_auc": model_auc, "best_feature": best,
            "best_feature_auc": best_auc}


def shuffle_check(train, test, features, seed=SEED, verbose=True):
    """
    Permute labels WITHIN each date and refit. Within-date is the right
    permutation: it destroys the feature-label link while preserving the
    date structure and the per-date base rate, so a model that still scores is
    reading something it should not be able to see.
    """
    from xgboost import XGBClassifier
    rng = np.random.default_rng(seed)
    tr = train.copy()
    tr["label"] = tr.groupby("date")["label"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    m = XGBClassifier(random_state=seed, **XGB_PARAMS)
    m.fit(tr[features], tr["label"], verbose=False)
    a = auc(test["label"].to_numpy(int), m.predict_proba(test[features])[:, 1])
    if verbose:
        print(f"\n  LEAK AUDIT 3/3 - shuffle control")
        print(f"    labels permuted within date, model refitted -> "
              f"OOS AUC {a:.4f} (chance is 0.500)")
    if np.isfinite(a) and abs(a - 0.5) > LEAK_SHUFFLE_AUC - 0.5:
        raise RuntimeError(
            f"a model trained on shuffled labels scores {a:.4f} out of sample. "
            f"The pipeline leaks; no result from it is usable.")
    return a


# =============================================================================
# WALK-FORWARD
# =============================================================================
def expectancy(r):
    return float(np.mean(r)) if len(r) else np.nan


def set_predict_mode(mode):
    """PREDICT_MODE is read inside the fit, so switching it needs a rebind."""
    global PREDICT_MODE
    if mode not in ("classify", "expectancy"):
        raise RuntimeError(f'PREDICT_MODE must be "classify" or "expectancy", '
                           f'not {mode!r}')
    PREDICT_MODE = mode
    return mode


def _fit_predict(tr, te, features, seed, mode=None, random_scores=False):
    """
    One fit, one set of scores. The scores are only ever used to RANK, so a
    regressor's R prediction and a classifier's probability are interchangeable
    downstream - which is what lets PREDICT_MODE switch cleanly.
    """
    from xgboost import XGBClassifier, XGBRegressor
    if random_scores:
        # The yardstick. Sixteen configurations means sixteen chances for one to
        # look good, so every table needs a row produced by a ranking that
        # cannot know anything.
        return np.random.default_rng(seed).random(len(te))
    mode = mode or PREDICT_MODE
    params = dict(XGB_PARAMS)
    if mode == "expectancy":
        params.pop("objective", None)
        params.pop("eval_metric", None)
        m = XGBRegressor(random_state=seed, objective="reg:squarederror",
                         **params)
        m.fit(tr[features], tr["r_multiple"], verbose=False)
        return m.predict(te[features])
    if mode != "classify":
        raise RuntimeError(f'PREDICT_MODE must be "classify" or "expectancy", '
                           f'not {mode!r}')
    w = tr["r_multiple"].abs().to_numpy(float) if WEIGHT_BY_R else None
    m = XGBClassifier(random_state=seed, **XGB_PARAMS)
    m.fit(tr[features], tr["label"], sample_weight=w, verbose=False)
    return m.predict_proba(te[features])[:, 1]


def walk_forward(d, features=None, min_train_years=MIN_TRAIN_YEARS,
                 n_seeds=N_SEEDS, seed=SEED, horizon=HORIZON, verbose=True,
                 random_scores=False):
    from xgboost import XGBClassifier
    features = features or ALL_FEATURES
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)

    d = d.sort_values("date").reset_index(drop=True)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    years = sorted(d["year"].unique())

    keep, preds = [], []
    if verbose:
        print("\n" + "=" * 88)
        print("  PURGED WALK-FORWARD")
        print("=" * 88)
        print(f"  embargo {embargo.days}d - the holding window, so no training "
              f"trade is still open when the test year opens")
    for y in years[min_train_years:]:
        start = pd.Timestamp(f"{y}-01-01")
        tr = d[pd.DatetimeIndex(d["date"]) < start - embargo]
        te = d[d["year"] == y]
        if len(tr) < 2000 or len(te) < 200:
            continue
        ps = []
        for s in range(n_seeds):
            ps.append(_fit_predict(tr, te, features, seed + s,
                                   random_scores=random_scores))
        preds.append(np.mean(ps, axis=0))
        keep.append(te)
        if verbose:
            print(f"    {y}: train {len(tr):,} -> test {len(te):,}")
    if not keep:
        raise RuntimeError("no usable walk-forward folds")

    te = pd.concat(keep, ignore_index=True)
    te["p"] = np.concatenate(preds)
    return te


def report(te, verbose=True):
    y = te["label"].to_numpy(int)
    r = te["r_multiple"].to_numpy(float)
    p = te["p"].to_numpy(float)
    base_wr = float(y.mean())
    base_exp = expectancy(r)
    rr = float(te["rr"].median())
    be = 1.0 / (1.0 + rr)

    k = max(1, int(len(p) * TOP_DECILE))
    top = np.argsort(-p)[:k]
    bot = np.argsort(p)[:k]

    if verbose:
        print(f"\n  {len(te):,} out-of-sample trades")
        print(f"  median reward:risk {rr:.2f}  ->  break-even win rate "
              f"{be:.1%}")
        print(f"\n  {'slice':22s} {'n':>8s} {'win rate':>9s} "
              f"{'expectancy R':>13s} {'AUC':>7s}")
        print(f"  {'every candidate':22s} {len(y):8,d} {base_wr:9.1%} "
              f"{base_exp:13.3f} {auc(y, p):7.4f}")
        print(f"  {'model top decile':22s} {k:8,d} {y[top].mean():9.1%} "
              f"{expectancy(r[top]):13.3f} {'':>7s}")
        print(f"  {'model bottom decile':22s} {k:8,d} {y[bot].mean():9.1%} "
              f"{expectancy(r[bot]):13.3f} {'':>7s}")
        print(f"\n  outcome mix: "
              + ", ".join(f"{o} {(te['outcome'] == o).mean():.1%}"
                          for o in ("win", "loss", "flat")))

        lift = y[top].mean() - base_wr
        print(f"\n  READ IT LIKE THIS")
        print(f"    the model's top decile wins {y[top].mean():.1%} against "
              f"{base_wr:.1%} for buying everything ({lift:+.1%}).")
        if expectancy(r[top]) <= 0:
            print(f"    expectancy there is {expectancy(r[top]):+.3f}R, so the "
                  f"selection LOSES money before costs. A win rate above "
                  f"break-even is not")
            print(f"    the same as a profitable rule when the winners are "
                  f"smaller than the losers.")
        elif expectancy(r[top]) <= base_exp:
            print(f"    expectancy {expectancy(r[top]):+.3f}R is no better than "
                  f"buying everything ({base_exp:+.3f}R), so the ranking is "
                  f"not adding value.")
        else:
            print(f"    expectancy {expectancy(r[top]):+.3f}R against "
                  f"{base_exp:+.3f}R for buying everything. Costs and slippage "
                  f"come out of that difference.")
    return {"n": int(len(te)), "base_win_rate": base_wr,
            "base_expectancy_R": base_exp, "rr": rr, "breakeven_wr": be,
            "auc": auc(y, p), "top_decile_win_rate": float(y[top].mean()),
            "top_decile_expectancy_R": expectancy(r[top]),
            "bottom_decile_win_rate": float(y[bot].mean()),
            "bottom_decile_expectancy_R": expectancy(r[bot])}


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_entry_v64(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
                  min_train_years=MIN_TRAIN_YEARS, n_seeds=N_SEEDS, seed=SEED,
                  skip_audit=False, label_mode=LABEL_MODE, ticker_splits=5,
                  tickers=None, run_ablation=True, sector_mode="slice",
                  predict_mode=None, feature_set="+market",
                  out="entry_v64_result.json", verbose=True):
    print("=" * 88)
    print("V64 - ENTRY TIMING: SHOULD THIS BE BOUGHT NOW?")
    print("=" * 88)
    mode = set_predict_mode(predict_mode or PREDICT_MODE)

    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    if tickers:
        missing = [t for t in tickers if t not in available]
        if missing:
            print(f"  not in the cache, skipped: {', '.join(missing[:10])}"
                  + (" ..." if len(missing) > 10 else ""))
        tickers = [t for t in tickers if t in available]
        if not tickers:
            raise RuntimeError("none of the requested tickers are cached")
    else:
        tickers = available
    cfg = HORIZON_CONFIGS[horizon]
    print(f"  {len(tickers)} tickers | horizon {horizon} "
          f"({cfg['eval_days']}d, {cfg['lookahead_bars']} bars) | step {step}")

    d = build_dataset(tickers, price_cache, horizon, step, verbose=verbose)
    if d.empty:
        raise RuntimeError("no candidate entries built")
    assert_market_columns(d)
    d = apply_label_mode(d, label_mode)
    print(f"  label mode: {label_mode}   |   ranking by: "
          + ("predicted expectancy (regression on R)" if mode == "expectancy"
             else "P(win) (classification)")
          + ("  weighted by |R|" if WEIGHT_BY_R and mode == "classify" else ""))
    print(f"\n  {len(d):,} candidate entries, "
          f"{d['date'].nunique():,} dates, base win rate {d['label'].mean():.1%}")

    if not skip_audit:
        structural_check(d, verbose=verbose)
        univariate_check(d, ALL_FEATURES, verbose=verbose)
        cut = d["date"].quantile(0.7)
        shuffle_check(d[d["date"] < cut], d[d["date"] >= cut],
                      ALL_FEATURES, seed=seed, verbose=verbose)
        print("\n  audit passed - performance numbers below are worth reading")

    feats = resolve_features(feature_set)
    print(f"  feature set: {feature_set} ({len(feats)} columns)")
    te = walk_forward(d, feats, min_train_years, n_seeds, seed,
                      horizon, verbose, random_scores=(feature_set == "random"))
    res = report(te, verbose=verbose)
    res["deciles"] = decile_table(te, verbose=verbose)
    res["post_check"] = beats_model_check(te, feats, verbose=verbose)
    res["confound_vol"] = within_bucket_check(te, by="yz66", verbose=verbose)
    res["confound_market"] = within_bucket_check(
        te, by="mkt_breadth_ema200", verbose=verbose)
    res["per_year"] = per_year_lift(te, verbose=verbose)
    if ticker_splits:
        res["ticker_holdout"] = ticker_holdout(d, n_splits=ticker_splits,
                                               seed=seed, verbose=verbose)
    if sector_mode:
        try:
            res["by_sector"] = by_sector(te, d, mode=sector_mode, seed=seed,
                                         min_train_years=min_train_years,
                                         horizon=horizon, verbose=verbose)
        except RuntimeError as e:
            print(f"\n  sector breakdown skipped: {e}")
    if run_ablation:
        res["ablation"] = ablation(d, min_train_years, 1, seed, horizon,
                                   verbose=verbose)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\n  wrote {out}")
    return res, te, d


def decile_table(te, verbose=True):
    """
    All ten deciles, not just the ends, with the FLAT share alongside - because
    the first real run showed the bottom decile earning MORE than the top
    (+0.403R against +0.339R) while winning less often, and the reason is
    invisible without that column.

    The barrier label calls a trade a loss whenever the target is not touched,
    including a trade that drifts up 0.8R and times out. A model trained on it
    learns "will this touch the target", which is not the same as "will this
    make money". Ranking by such a model can be worse than ranking by its
    inverse, and that is a property of the LABEL, not of the features.
    """
    te = te.copy()
    te["_d"] = pd.qcut(te["p"].rank(method="first"), 10, labels=False,
                       duplicates="drop")
    rows = []
    for d, g in te.groupby("_d"):
        rows.append({"decile": int(d) + 1, "n": int(len(g)),
                     "win_rate": float(g["label"].mean()),
                     "expectancy_R": float(g["r_multiple"].mean()),
                     "flat_share": float((g["outcome"] == "flat").mean()),
                     "mean_p": float(g["p"].mean())})
    r = pd.DataFrame(rows)
    if verbose:
        print(f"\n  DECILE TABLE  (1 = the model likes it least)")
        print(f"    {'decile':>7s} {'n':>8s} {'win rate':>9s} "
              f"{'expectancy R':>13s} {'flat share':>11s}")
        for _, q in r.iterrows():
            print(f"    {int(q['decile']):7d} {int(q['n']):8,d} "
                  f"{q['win_rate']:9.1%} {q['expectancy_R']:13.3f} "
                  f"{q['flat_share']:11.1%}")
        top, bot = r.iloc[-1], r.iloc[0]
        rho = float(r["decile"].corr(r["expectancy_R"], method="spearman"))
        print(f"\n    rank correlation between decile and expectancy: "
              f"{rho:+.3f}")

        # A U is not an inversion. Both ends high with a sagging middle means
        # the score tracks something whose relationship to the outcome is
        # U-shaped - volatility does exactly that here - and a rank statistic
        # like AUC cancels to 0.50 on it. Saying "inverted" would send you
        # looking for the wrong bug.
        ends = 0.5 * (float(top["expectancy_R"]) + float(bot["expectancy_R"]))
        middle = float(r["expectancy_R"].iloc[3:7].mean())
        # A real U needs BOTH ends above the middle and no strong overall trend.
        # Judging on the endpoint average alone called a -0.88 monotone decline a
        # U because decile 10 ticked up 0.04 off the floor.
        trend = float(r["decile"].iloc[:8].corr(r["expectancy_R"].iloc[:8],
                                                method="spearman"))
        u_shaped = (float(top["expectancy_R"]) > middle + 0.02
                    and float(bot["expectancy_R"]) > middle + 0.02
                    and abs(trend) < 0.6)

        if u_shaped:
            print(f"    -> U-SHAPED, not inverted: both ends earn more "
                  f"({bot['expectancy_R']:+.3f}R and {top['expectancy_R']:+.3f}R) "
                  f"than the middle ({middle:+.3f}R).")
            print(f"       The score tracks something whose payoff is U-shaped "
                  f"in it. Check the base R column of the volatility confound "
                  f"table: if that")
            print(f"       is U-shaped too, the model has found volatility "
                  f"again and a rank metric cannot see it, which is why AUC "
                  f"sits at 0.50.")
            print(f"       |score - median| may rank better than score. A "
                  f"monotone label will not fix a U.")
        elif top["expectancy_R"] <= bot["expectancy_R"]:
            print(f"    -> INVERTED: the decile the model likes LEAST earns "
                  f"{bot['expectancy_R']:+.3f}R, the one it likes most "
                  f"{top['expectancy_R']:+.3f}R.")
            print(f"       Flat share runs {bot['flat_share']:.0%} at the "
                  f"bottom against {top['flat_share']:.0%} at the top. If the "
                  f"gap is large the model is")
            print(f"       avoiding trades that end profitably without "
                  f"touching the target, which is a LABEL problem: try "
                  f"RUN_LABEL_MODE = 'profit'.")
        elif rho < 0.5:
            print(f"    -> the ordering is weak: expectancy does not rise "
                  f"cleanly with the score.")
        else:
            print(f"    -> expectancy rises with the score, which is what a "
                  f"usable ranking looks like.")
    return rows


def per_year_lift(te, verbose=True):
    """
    Is the edge spread across years, or a couple of good ones? A market-level
    feature can only move with the market, so an edge that leans on one has far
    fewer independent observations than the row count suggests.
    """
    rows = []
    for y, g in te.groupby("year"):
        k = max(1, int(len(g) * TOP_DECILE))
        top = g.nlargest(k, "p")
        rows.append({"year": int(y), "n": int(len(g)),
                     "base_wr": float(g["label"].mean()),
                     "top_wr": float(top["label"].mean()),
                     "lift": float(top["label"].mean() - g["label"].mean()),
                     "base_R": float(g["r_multiple"].mean()),
                     "top_R": float(top["r_multiple"].mean()),
                     # the statistic the headline actually claims
                     "edge": float(top["r_multiple"].mean()
                                   - g["r_multiple"].mean()),
                     "flat_share_top": float((top["outcome"] == "flat").mean()),
                     "flat_share_base": float((g["outcome"] == "flat").mean()),
                     "edge_ex_flat": (
                         float(top[top["outcome"] != "flat"]["r_multiple"].mean()
                               - g[g["outcome"] != "flat"]["r_multiple"].mean())
                         if (top["outcome"] != "flat").sum() > 20 else np.nan)})
    r = pd.DataFrame(rows)
    if verbose:
        print(f"\n  LIFT BY TEST YEAR")
        print(f"    {'year':>6s} {'n':>8s} {'base wr':>8s} {'top wr':>7s} "
              f"{'lift':>7s} {'base R':>8s} {'top R':>7s}")
        for _, q in r.iterrows():
            print(f"    {int(q['year']):6d} {int(q['n']):8,d} "
                  f"{q['base_wr']:8.1%} {q['top_wr']:7.1%} {q['lift']:+7.1%} "
                  f"{q['base_R']:8.3f} {q['top_R']:7.3f}")
        pos = int((r["lift"] > 0).sum())
        print(f"\n    positive in {pos} of {len(r)} years   "
              f"median lift {r['lift'].median():+.1%}   "
              f"worst {r['lift'].min():+.1%}")
        # How much does the whole picture rest on the single worst year? Print
        # it rather than making anyone delete a year to find out.
        worst_year = int(r.loc[r["lift"].idxmin(), "year"])
        ex = r[r["year"] != worst_year]
        print(f"    without {worst_year}: median {ex['lift'].median():+.1%}, "
              f"mean {ex['lift'].mean():+.1%}, sd {ex['lift'].std(ddof=1):.1%} "
              f"(with it: mean {r['lift'].mean():+.1%}, "
              f"sd {r['lift'].std(ddof=1):.1%})")
        print(f"    this line is a SENSITIVITY, not a result. A model is "
              f"allowed to be judged on the")
        print(f"    years it would have traded, and {worst_year} is one of "
              f"them.")
        if pos <= len(r) * 0.6:
            print(f"    -> the edge is not consistent across years; it is "
                  f"concentrated in particular regimes.")
    return rows


def ticker_holdout(d, features=None, n_splits=5, holdout_frac=0.3,
                   seed=SEED, drop_market=True, date_split=True, verbose=True):
    """
    Train on one set of TICKERS, test on names the model has never seen.

    The walk-forward answers "does this work on future dates". It does not
    answer "does this work on other companies", and splitting by ticker asks
    that second question.

    BUT SPLITTING BY TICKER ALONE LEAKS, AND BADLY.

    The first version of this function shared dates between train and test and
    reported +24.5% lift on unseen names against the walk-forward's +2.3%, with
    a standard deviation of 0.8% across splits. That gap is not generalisation,
    it is the market block. mkt_breadth_ema200, mkt_ret_20 and mkt_vol_20 are
    IDENTICAL for every name on a date, so a model trained on 235 names on date
    D learns what happened to trades opened on date D - and is then asked about
    100 other names on that same date D. It can read the answer off the date.

    So both defences are on by default:

        drop_market  train without the market block, so no column can identify
                     a date
        date_split   hold out LATER dates as well as other names, which is the
                     combinatorial split rather than the ticker-only one

    Turning either off is a diagnostic, not a result.
    """
    from xgboost import XGBClassifier
    features = list(features or ALL_FEATURES)
    if drop_market:
        features = [f for f in features if f not in MKT_FEATURES]
    rng = np.random.default_rng(seed)
    tickers = np.array(sorted(d["ticker"].unique()))
    cut = d["date"].quantile(0.6) if date_split else None
    rows = []
    if verbose:
        print(f"\n  TICKER HOLDOUT - {n_splits} splits, "
              f"{holdout_frac:.0%} of names held out")
        print(f"    market block {'DROPPED' if drop_market else 'KEPT'}"
              f" | dates {'also split at ' + str(pd.Timestamp(cut).date()) if date_split else 'SHARED (leaks - diagnostic only)'}")
        print(f"    {'split':>6s} {'train names':>12s} {'test names':>11s} "
              f"{'base wr':>8s} {'top wr':>7s} {'lift':>7s} {'top R':>7s}")
    for s in range(n_splits):
        held = set(rng.choice(tickers, int(len(tickers) * holdout_frac),
                              replace=False))
        tr = d[~d["ticker"].isin(held)]
        te = d[d["ticker"].isin(held)]
        if date_split:
            embargo = pd.Timedelta(days=120)
            tr = tr[pd.DatetimeIndex(tr["date"]) < cut - embargo]
            te = te[pd.DatetimeIndex(te["date"]) >= cut]
        if len(tr) < 2000 or len(te) < 500:
            continue
        te = te.copy()
        te["p"] = _fit_predict(tr, te, features, seed + s)
        k = max(1, int(len(te) * TOP_DECILE))
        top = te.nlargest(k, "p")
        rows.append({"split": s, "n_train_names": len(tickers) - len(held),
                     "n_test_names": len(held), "n_test": int(len(te)),
                     "base_wr": float(te["label"].mean()),
                     "top_wr": float(top["label"].mean()),
                     "lift": float(top["label"].mean() - te["label"].mean()),
                     "base_R": float(te["r_multiple"].mean()),
                     "top_R": float(top["r_multiple"].mean()),
                     "auc": auc(te["label"].to_numpy(int),
                                te["p"].to_numpy(float))})
        if verbose:
            q = rows[-1]
            print(f"    {s:6d} {q['n_train_names']:12d} {q['n_test_names']:11d} "
                  f"{q['base_wr']:8.1%} {q['top_wr']:7.1%} {q['lift']:+7.1%} "
                  f"{q['top_R']:7.3f}")
    if rows and verbose:
        r = pd.DataFrame(rows)
        print(f"\n    mean lift on unseen names {r['lift'].mean():+.1%} "
              f"(sd {r['lift'].std(ddof=1):.1%})   "
              f"mean AUC {r['auc'].mean():.4f}")
        if not drop_market or not date_split:
            print(f"    WARNING this configuration can leak. Only the default "
                  f"(market dropped, dates split)")
            print(f"            is a result; anything else is a diagnostic.")
    return rows


def within_bucket_check(te, by="yz66", n_buckets=5, verbose=True):
    """
    THE decisive test for a timing model.

    If the model's whole edge is "prefer low-volatility names", then sorting
    trades into volatility buckets first and asking for lift INSIDE each bucket
    should leave nothing. That is what V46 found for the dilution flag - lift
    that looked real pooled and vanished once volatility was held fixed - and
    the same confound is the obvious suspect here, because the barriers are
    ATR-scaled.

    Pooled lift with no within-bucket lift means the model has rediscovered
    volatility. Within-bucket lift that survives is timing skill.
    """
    te = te.copy()
    te["_b"] = pd.qcut(te[by].rank(method="first"), n_buckets,
                       labels=False, duplicates="drop")
    if verbose:
        what = {"yz66": "volatility",
                "mkt_breadth_ema200": "the market regime"}.get(by, by)
        print(f"\n  CONFOUND TEST - is the edge just {what}? "
              f"(buckets by {by})")
        print(f"    {'bucket':>8s} {'n':>7s} {'base wr':>8s} {'top-dec wr':>11s} "
              f"{'lift':>7s} {'base R':>8s} {'top-dec R':>10s}")
    rows = []
    for b, g in te.groupby("_b"):
        k = max(1, int(len(g) * TOP_DECILE))
        top = g.nlargest(k, "p")
        base_wr, top_wr = float(g["label"].mean()), float(top["label"].mean())
        rows.append({"bucket": int(b), "n": int(len(g)), "base_wr": base_wr,
                     "top_wr": top_wr, "lift": top_wr - base_wr,
                     "base_R": float(g["r_multiple"].mean()),
                     "top_R": float(top["r_multiple"].mean())})
        if verbose:
            print(f"    {int(b):8d} {len(g):7,d} {base_wr:8.1%} {top_wr:11.1%} "
                  f"{top_wr - base_wr:+7.1%} {g['r_multiple'].mean():8.3f} "
                  f"{top['r_multiple'].mean():10.3f}")
    r = pd.DataFrame(rows)
    mean_lift = float(r["lift"].mean())
    pooled_k = max(1, int(len(te) * TOP_DECILE))
    pooled_lift = float(te.nlargest(pooled_k, "p")["label"].mean()
                        - te["label"].mean())
    # A mean across buckets hides the case where one bucket carries everything,
    # which is what happened the first time this ran: +13.5% in the lowest
    # volatility bucket and -2.5% to +1.0% in the other four. Report how many
    # buckets actually show lift, not just the average of five numbers.
    n_pos = int((r["lift"] > 0.01).sum())
    carried = bool(n_pos <= 1 and mean_lift > 0)
    if verbose:
        print(f"\n    pooled lift {pooled_lift:+.1%}   "
              f"mean within-bucket lift {mean_lift:+.1%}   "
              f"buckets with lift above +1%: {n_pos}/{len(r)}")
        if carried:
            print(f"    -> WATCH OUT: the mean is carried by ONE bucket. "
                  f"In {len(r) - n_pos} of {len(r)} {what} buckets the "
                  f"model adds nothing or hurts.")
            print(f"       Treat this as no reliable within-bucket edge rather "
                  f"than as {mean_lift/pooled_lift:.0%} of the pooled number "
                  f"surviving.")
        elif mean_lift <= 0.2 * pooled_lift:
            print(f"    -> the edge IS {what}. Hold it fixed and the model "
                  f"stops adding anything.")
            print(f"       That is not a timing signal; Pillar 2 already "
                  f"measures volatility, better and with calibration.")
        elif mean_lift < 0.6 * pooled_lift:
            print(f"    -> most of the pooled edge is {what}, but "
                  f"{mean_lift/pooled_lift:.0%} of it survives with volatility "
                  f"held fixed.")
        else:
            print(f"    -> the edge SURVIVES with {what} held fixed. It is "
                  f"not a {what.replace('the ', '')} sort.")
    return {"pooled_lift": pooled_lift, "mean_within_bucket_lift": mean_lift,
            "buckets_with_lift": n_pos, "n_buckets": int(len(r)),
            "carried_by_one_bucket": carried, "buckets": rows}


def direction_holdout(d, features=None, split_frac=0.6, seed=SEED,
                      min_train_years=MIN_TRAIN_YEARS, horizon=HORIZON,
                      verbose=True):
    """
    THE test for the inversion hypothesis.

    Reading the bottom decile as the pick was chosen after seeing that every
    decile rank correlation in the first sweep was negative. That makes it a
    hypothesis drawn from the same data it was then measured on, and twelve
    configurations agreeing about it is not twelve pieces of evidence - it is one
    hypothesis viewed twelve ways.

    So: decide the DIRECTION using the early years only, then apply that decision
    to the later years and never revisit it.

        early   walk-forward over the first `split_frac` of years. Its decile
                rank correlation picks the sign. Nothing else is taken from here.
        late    walk-forward over the remaining years, scored with that sign.

    If inversion is a real property of this data it was already visible early and
    the late half pays. If it was an artefact of looking at everything at once,
    the late half will not care which sign it was handed.
    """
    features = features or resolve_features("no_confirm")
    d = d.sort_values("date").reset_index(drop=True)
    years = sorted(pd.DatetimeIndex(d["date"]).year.unique())
    n_early = max(min_train_years + 3, int(len(years) * split_frac))
    early_years, late_years = years[:n_early], years[n_early:]
    if len(late_years) < 3:
        raise RuntimeError("not enough later years to test a direction on")

    cut = pd.Timestamp(f"{late_years[0]}-01-01")
    d_early = d[pd.DatetimeIndex(d["date"]) < cut]

    if verbose:
        print("\n" + "=" * 88)
        print("  DIRECTION HOLDOUT - was the inversion visible BEFORE we looked?")
        print("=" * 88)
        print(f"  direction chosen on {early_years[0]}-{early_years[-1]}, "
              f"tested on {late_years[0]}-{late_years[-1]}")

    te_e = walk_forward(d_early, features, min_train_years, 1, seed, horizon,
                        verbose=False)
    dec_e = decile_table(te_e, verbose=False)
    rho_e = float(pd.Series([q["decile"] for q in dec_e]).corr(
        pd.Series([q["expectancy_R"] for q in dec_e]), method="spearman"))
    sign = -1.0 if rho_e < 0 else 1.0

    te_l = walk_forward(d, features, n_early, 1, seed, horizon, verbose=False)
    te_l = te_l[te_l["year"].isin(late_years)].copy()
    if len(te_l) < 500:
        raise RuntimeError("late half is too thin to score")
    te_l["p"] = sign * te_l["p"]

    k = max(1, int(len(te_l) * TOP_DECILE))
    top = te_l.nlargest(k, "p")
    edge = float(top["r_multiple"].mean() - te_l["r_multiple"].mean())
    py = pd.DataFrame(per_year_lift(te_l, verbose=False))
    dec_l = decile_table(te_l, verbose=False)
    rho_l = float(pd.Series([q["decile"] for q in dec_l]).corr(
        pd.Series([q["expectancy_R"] for q in dec_l]), method="spearman"))

    if verbose:
        print(f"\n  EARLY half: decile rho {rho_e:+.3f}  ->  direction "
              f"{'INVERTED' if sign < 0 else 'as fitted'} "
              f"({len(te_e):,} trades)")
        print(f"  LATE half:  decile rho {rho_l:+.3f}   edge {edge:+.3f}R "
              f"({len(te_l):,} trades, base {te_l['r_multiple'].mean():.3f}R)")
        print(f"\n  {'year':>6s} {'n':>8s} {'edge':>8s} {'ex-flat':>8s} "
              f"{'flat% picked':>13s}")
        for _, q in py.iterrows():
            print(f"  {int(q['year']):6d} {int(q['n']):8,d} {q['edge']:+8.3f} "
                  f"{q['edge_ex_flat']:+8.3f} {q['flat_share_top']:13.1%}")
        pos = int((py["edge"] > 0).sum())
        print(f"\n  positive in {pos} of {len(py)} held-out years, "
              f"mean {py['edge'].mean():+.3f}R, worst {py['edge'].min():+.3f}R")

        # HOW CONCENTRATED IS IT? A mean carried by two years is a mean that
        # will not be there next year. This is the same sensitivity the yearly
        # table prints for a NEGATIVE result, applied to a positive one - a
        # crisis year is not allowed to make a result look good either.
        e = py["edge"].sort_values()
        tot = float(e.sum())
        top2 = float(e.iloc[-2:].sum())
        rest = e.iloc[:-2]
        if tot > 0 and len(rest):
            print(f"  best two years contribute {top2:+.3f} of {tot:+.3f} "
                  f"= {100 * top2 / tot:.0f}% of the total")
            print(f"  without them: {float(rest.mean()):+.3f}R over "
                  f"{len(rest)} years")
            if top2 > 0.6 * tot:
                print(f"  -> CONCENTRATED. The mean is two years, not a rate. "
                      f"Size on {float(rest.mean()):+.3f}R,")
                print(f"     the figure the other {len(rest)} years actually "
                      f"produced.")
        what = "INVERTED" if sign < 0 else "as-fitted"
        if edge <= 0:
            print(f"  -> the {what} direction did NOT survive. It was an "
                  f"artefact of choosing the sign")
            print(f"     after seeing the whole sample. Drop it.")
        elif pos >= len(py) * 0.7:
            print(f"  -> the {what} direction was visible early and held up: "
                  f"{edge:+.3f}R on held-out years,")
            print(f"     positive in {pos}/{len(py)}. This is a genuine "
                  f"out-of-sample result, and the number to")
            print(f"     size on is the WORST year ({py['edge'].min():+.3f}R), "
                  f"not the mean.")
        else:
            print(f"  -> mixed: the {what} sign pays {edge:+.3f}R overall but "
                  f"only {pos}/{len(py)} years are")
            print(f"     positive. One good year can carry a pooled number; "
                  f"wait for more held-out data.")
    return {"rho_early": rho_e, "sign": sign, "rho_late": rho_l,
            "edge_late": edge, "years": py.to_dict("records")}


def rolling_direction_holdout(d, features=None, first_split=None,
                              seed=SEED, min_train_years=MIN_TRAIN_YEARS,
                              horizon=HORIZON, verbose=True):
    """
    THE robustness test for the inversion.

    One holdout at one split year can pass by luck - and the single split that
    passed had 88% of its mean in two years, one of them 2020. So run the whole
    holdout again at every plausible split point. The direction is re-chosen from
    scratch on each early half, and each late half is scored with whatever that
    early half said.

    If inversion is a property of this data, it is visible from most starting
    points and most late halves pay. If it is a property of one split, only one
    column will be green and that is the answer.
    """
    features = features or resolve_features("no_confirm")
    years = sorted(pd.DatetimeIndex(d["date"]).year.unique())
    lo = first_split or (years[0] + min_train_years + 4)
    splits = [y for y in years if lo <= y <= years[-1] - 3]
    if not splits:
        raise RuntimeError("no usable split points")

    if verbose:
        print("\n" + "=" * 88)
        print("  ROLLING DIRECTION HOLDOUT - does it pass at every split, "
              "or only one?")
        print("=" * 88)
        print(f"  {'split':>6s} {'early rho':>10s} {'sign':>9s} "
              f"{'late edge':>10s} {'late mean/yr':>13s} {'pos':>7s} "
              f"{'worst':>8s} {'ex-top2':>9s}")

    rows = []
    for sp in splits:
        try:
            r = direction_holdout(
                d, features=features,
                split_frac=(years.index(sp) / max(len(years), 1)),
                seed=seed, min_train_years=min_train_years, horizon=horizon,
                verbose=False)
        except RuntimeError:
            continue
        py = pd.DataFrame(r["years"])
        e = py["edge"].sort_values()
        rest = e.iloc[:-2]
        rows.append({"split": int(sp), "rho_early": r["rho_early"],
                     "sign": r["sign"], "edge_late": r["edge_late"],
                     "yr_mean": float(py["edge"].mean()),
                     "yr_pos": int((py["edge"] > 0).sum()),
                     "n_years": int(len(py)),
                     "yr_worst": float(py["edge"].min()),
                     "ex_top2": float(rest.mean()) if len(rest) else np.nan})
        if verbose:
            q = rows[-1]
            print(f"  {q['split']:6d} {q['rho_early']:+10.3f} "
                  f"{'INVERTED' if q['sign'] < 0 else 'as-fitted':>9s} "
                  f"{q['edge_late']:+10.3f} {q['yr_mean']:+13.3f} "
                  f"{q['yr_pos']:4d}/{q['n_years']:<2d} {q['yr_worst']:+8.3f} "
                  f"{q['ex_top2']:+9.3f}")

    if not rows:
        raise RuntimeError("no split produced a scorable late half")
    r = pd.DataFrame(rows)
    if verbose:
        n_inv = int((r["sign"] < 0).sum())
        n_pay = int((r["edge_late"] > 0).sum())
        n_pay_ex = int((r["ex_top2"] > 0).sum())
        print(f"\n  inversion chosen at {n_inv}/{len(r)} splits")
        print(f"  late half pays at {n_pay}/{len(r)} splits "
              f"(median {float(r['edge_late'].median()):+.3f}R)")
        print(f"  still pays with each late half's best TWO years removed: "
              f"{n_pay_ex}/{len(r)} "
              f"(median {float(r['ex_top2'].median()):+.3f}R)")
        if n_pay_ex >= 0.7 * len(r) and float(r["ex_top2"].median()) > 0.02:
            print(f"  -> ROBUST. It is not one split and it is not two good "
                  f"years. This is the strongest")
            print(f"     evidence in the project for anything directional.")
        elif n_pay >= 0.7 * len(r):
            print(f"  -> the pooled edge is consistent across splits, but it "
                  f"thins out to "
                  f"{float(r['ex_top2'].median()):+.3f}R once each half's two "
                  f"best years come out.")
            print(f"     Real but small. Size on the ex-top2 column, and "
                  f"expect years that pay nothing.")
        else:
            print(f"  -> NOT robust: it pays at only {n_pay} of {len(r)} "
                  f"splits. The single passing holdout was luck.")
    return rows


def _sweep_row(te, lm, pm, fs, feats, draw=0):
    k = max(1, int(len(te) * TOP_DECILE))
    top = te.nlargest(k, "p")
    py = pd.DataFrame(per_year_lift(te, verbose=False))
    dec = decile_table(te, verbose=False)
    return {
        "label": lm, "rank_by": pm, "features": fs, "draw": draw,
        "n_feats": 0 if fs == "random" else len(feats),
        "lift": float(top["label"].mean() - te["label"].mean()),
        "top_R": float(top["r_multiple"].mean()),
        "base_R": float(te["r_multiple"].mean()),
        "edge": float(top["r_multiple"].mean() - te["r_multiple"].mean()),
        # The same edge among trades that actually resolved at a barrier. If the
        # headline edge survives here it is about selection; if it collapses, the
        # model is being paid for picking trades that drift without resolving,
        # which in a 16-year bull market is closer to a beta bet than to timing.
        "edge_ex_flat": (
            float(top[top["outcome"] != "flat"]["r_multiple"].mean()
                  - te[te["outcome"] != "flat"]["r_multiple"].mean())
            if (top["outcome"] != "flat").sum() > 50 else np.nan),
        "auc": auc(te["label"].to_numpy(int), te["p"].to_numpy(float)),
        # Win-rate lift AND expectancy edge, per year. The headline is an
        # expectancy number, so judging its stability on win-rate lift is
        # judging the wrong statistic - which is exactly what the first
        # inverted sweep did: edge +0.160 with a NEGATIVE mean yearly lift.
        "yr_mean": float(py["lift"].mean()),
        "yr_sd": float(py["lift"].std(ddof=1)),
        "yr_worst": float(py["lift"].min()),
        "yr_pos": int((py["lift"] > 0).sum()), "n_years": int(len(py)),
        "yr_edge_mean": float(py["edge"].mean()),
        "yr_edge_sd": float(py["edge"].std(ddof=1)),
        "yr_edge_worst": float(py["edge"].min()),
        "yr_edge_pos": int((py["edge"] > 0).sum()),
        "flat_top": float(py["flat_share_top"].mean()),
        "flat_base": float(py["flat_share_base"].mean()),
        "decile_rho": float(pd.Series([q["decile"] for q in dec]).corr(
            pd.Series([q["expectancy_R"] for q in dec]), method="spearman")),
    }


def sweep(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
          label_modes=("barrier", "profit"),
          predict_modes=("classify", "expectancy"),
          feature_sets=("per_name", "no_confirm", "before_pattern",
                        "after_pattern", "random"),
          n_random_draws=6, invert=False,
          min_train_years=MIN_TRAIN_YEARS, seed=SEED, tickers=None,
          skip_audit=False, out="entry_v64_sweep.json", verbose=True):
    """
    Every combination in one run, on ONE dataset build.

    The dataset is the expensive part - four hundred thousand candidate entries
    with their barriers - and it does not depend on the label or the ranking
    target, so it is built once and every configuration is scored against it.
    That also makes the comparison exact: no combination gets a different sample.

    WHAT IS SWEPT

        label_mode     barrier | profit        what counts as a win
        predict_mode   classify | expectancy   what is ranked
        feature_set    per_name | before_pattern | after_pattern | +market
                       | random

    before_pattern and after_pattern differ ONLY by the fifteen technical-data
    columns, so the difference between those two rows is the answer to "was the
    extra technical data worth adding".

    THE RANDOM ROW IS NOT DECORATION

    Twenty combinations means twenty chances for one to look good. The `random`
    feature set ranks by noise, so whatever edge it posts is the size of edge
    this test produces from nothing. A real configuration has to beat THAT, not
    zero.
    """
    # Two kinds of duplicate get collapsed, because running them wastes time and
    # then makes the table look like corroboration.
    #   1. feature sets that resolve to the SAME columns ("after_pattern" is
    #      literally "+market")
    #   2. in expectancy mode the ranking regresses on r_multiple, which does
    #      not depend on the label at all - so label_mode changes only the
    #      reported win rate, never the ranking. One label is enough there.
    print("=" * 96)
    print("V64 SWEEP")
    print("=" * 96)
    seen_cols, keep_fs = {}, []
    for fs in feature_sets:
        key = tuple(resolve_features(fs)) + (fs == "random",)
        if key in seen_cols:
            if verbose:
                print(f"  note: feature set {fs!r} is the same columns as "
                      f"{seen_cols[key]!r} - running it once")
            continue
        seen_cols[key] = fs
        keep_fs.append(fs)

    combos = []
    for fs in keep_fs:
        for pm in predict_modes:
            lms = list(label_modes)
            if pm == "expectancy" and len(lms) > 1:
                lms = [lms[0]]
                if verbose and fs == keep_fs[0]:
                    print(f"  note: expectancy ranking is label-independent "
                          f"(it regresses on R), so only "
                          f"{lms[0]!r} is run for it")
            for lm in lms:
                combos.append((lm, pm, fs))
    print(f"  {len(combos)} distinct configurations on one dataset build")

    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    names = [t for t in (tickers or available) if t in available]
    d0 = build_dataset(names, price_cache, horizon, step, verbose=verbose)
    if d0.empty:
        raise RuntimeError("no candidate entries built")
    assert_market_columns(d0)
    print(f"\n  {len(d0):,} candidate entries from {len(names)} tickers, "
          f"{d0['date'].nunique():,} dates")

    if not skip_audit:
        structural_check(d0, verbose=verbose)
        cut = d0["date"].quantile(0.7)
        dd = apply_label_mode(d0, "barrier")
        univariate_check(dd, ALL_FEATURES, verbose=verbose)
        shuffle_check(dd[dd["date"] < cut], dd[dd["date"] >= cut],
                      ALL_FEATURES, seed=seed, verbose=verbose)
        print("\n  audit passed")

    rows = []
    for i, (lm, pm, fs) in enumerate(combos, 1):
        d = apply_label_mode(d0, lm)
        feats = resolve_features(fs)
        set_predict_mode(pm)
        draws = n_random_draws if fs == "random" else 1
        for draw in range(draws):
            try:
                te = walk_forward(d, feats, min_train_years, 1, seed + 1000 * draw,
                                  horizon, verbose=False,
                                  random_scores=(fs == "random"))
            except RuntimeError:
                continue
            if invert:
                te["p"] = -te["p"]
            rows.append(_sweep_row(te, lm, pm, fs, feats, draw))
            if verbose and draw == 0:
                q = rows[-1]
                print(f"  [{i}/{len(combos)}] {lm:8s} {pm:11s} {fs:15s} "
                      f"edge {q['edge']:+.3f}  rho {q['decile_rho']:+.2f}")
        continue
    r = pd.DataFrame(rows)
    # collapse the random draws to one displayed row, keeping their spread
    rnd_all = r[r["features"] == "random"]["edge"].to_numpy(float)
    r = (pd.concat([r[r["features"] != "random"],
                    r[r["features"] == "random"].head(1)])
         .sort_values("edge", ascending=False))
    # The bar is the BEST of several random draws, not one draw. One draw tells
    # you nothing about the spread, and the first version of this table used one.
    bar = float(np.max(rnd_all)) if len(rnd_all) else 0.0

    print("\n" + "=" * 96)
    print("  SWEEP RESULTS  (sorted by edge = top-decile R minus buy-everything R)")
    print("=" * 96)
    print(f"  {'label':8s} {'rank by':11s} {'features':14s} {'n':>3s} "
          f"{'edge':>7s} {'rho':>5s} | PER YEAR, on EXPECTANCY: "
          f"{'mean':>7s} {'sd':>6s} {'worst':>7s} {'pos':>6s} | "
          f"{'flat%':>6s}")
    for _, q in r.iterrows():
        mark = "  <-- noise" if q["features"] == "random" else ""
        print(f"  {q['label']:8s} {q['rank_by']:11s} {q['features']:14s} "
              f"{int(q['n_feats']):3d} {q['edge']:+7.3f} "
              f"{q['decile_rho']:+5.2f} | {'':26s}"
              f"{q['yr_edge_mean']:+7.3f} {q['yr_edge_sd']:6.3f} "
              f"{q['yr_edge_worst']:+7.3f} "
              f"{int(q['yr_edge_pos'])}/{int(q['n_years']):<3d} | "
              f"{q['flat_top']:6.1%}{mark}")
    print(f"\n  flat% is the share of the picked decile that timed out rather "
          f"than touching a barrier")
    print(f"  (the whole sample runs about "
          f"{float(r['flat_base'].mean()):.1%}). A large gap means the edge is "
          f"coming from trades")
    print(f"  that drift, not from trades that hit their target.")

    if len(rnd_all):
        print(f"\n  THE BAR: {len(rnd_all)} random rankings produced edges "
              f"{np.min(rnd_all):+.3f} to {np.max(rnd_all):+.3f} "
              f"(mean {np.mean(rnd_all):+.3f}, sd {np.std(rnd_all, ddof=1):.3f}).")
        print(f"  Anything at or below {bar:+.3f} is inside the range noise "
              f"alone produces here.")
    if invert:
        print(f"\n  NOTE scores are INVERTED in this sweep: the bottom decile "
              f"is being read as the pick.")
        print(f"       This direction was chosen AFTER seeing that every "
              f"decile_rho was negative, so it is")
        print(f"       a post-hoc hypothesis. It needs its own forward test "
              f"before it counts as a result.")
    real = r[(r["features"] != "random") & (r["edge"] > bar)]
    if real.empty:
        print(f"  -> NO configuration beats the noise row. The signal is not "
              f"there in any of them.")
    else:
        b = real.iloc[0]
        print(f"  -> {int(len(real))} of {int((r['features'] != 'random').sum())} "
              f"configurations clear it. Best: {b['label']} / {b['rank_by']} / "
              f"{b['features']}")
        print(f"     edge {b['edge']:+.3f}, worst year {b['yr_worst']:+.1%}, "
              f"positive in {int(b['yr_pos'])}/{int(b['n_years'])} years")
        if b["decile_rho"] < 0.5:
            print(f"     but its decile rho is {b['decile_rho']:+.2f} - the "
                  f"ranking is not monotone, so only the top decile means "
                  f"anything.")

    # before vs after the technical data, holding everything else fixed
    piv = r[r["features"].isin(("before_pattern", "after_pattern"))]
    if not piv.empty:
        print(f"\n  WAS THE EXTRA TECHNICAL DATA WORTH IT?")
        print(f"  {'label':8s} {'rank by':11s} {'before':>8s} {'after':>8s} "
              f"{'delta':>8s}")
        for (lm, pm), g in piv.groupby(["label", "rank_by"]):
            b = g[g["features"] == "before_pattern"]["edge"]
            a = g[g["features"] == "after_pattern"]["edge"]
            if len(b) and len(a):
                print(f"  {lm:8s} {pm:11s} {float(b.iloc[0]):+8.3f} "
                      f"{float(a.iloc[0]):+8.3f} "
                      f"{float(a.iloc[0]) - float(b.iloc[0]):+8.3f}")
        print(f"  fifteen columns: candles, VWAP, MACD, distance to levels. "
              f"Positive delta means they paid.")

    print(f"\n  edge with FLAT trades removed (same rows, barrier-resolved "
          f"only):")
    for _, q in r[r["features"] != "random"].head(4).iterrows():
        ex = q.get("edge_ex_flat", np.nan)
        print(f"    {q['label']:8s} {q['rank_by']:11s} {q['features']:14s} "
              f"{q['edge']:+.3f} -> {ex:+.3f}"
              + ("   collapses" if np.isfinite(ex) and ex < 0.3 * q["edge"]
                 else "   holds"))

    try:
        payload_dh = direction_holdout(
            apply_label_mode(d0, label_modes[0]),
            features=resolve_features("no_confirm"), seed=seed,
            min_train_years=min_train_years, horizon=horizon, verbose=verbose)
    except RuntimeError as e:
        print(f"\n  direction holdout skipped: {e}")
        payload_dh = None

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "direction_holdout": payload_dh}, f, indent=2,
                  default=float)
    print(f"\n  wrote {out}")
    return rows


def sector_map(module="pillar2_v43_universe"):
    """
    ticker -> sector, read from the universe module's _SECTOR lists. Anything
    not listed comes back as "OTHER" rather than being dropped, so the counts
    always add up to the universe you actually ran.
    """
    import importlib
    try:
        U = importlib.import_module(module)
    except Exception as e:
        raise RuntimeError(f"cannot import {module}: {e}")
    out = {}
    for name in dir(U):
        if not name.startswith("_") or name.startswith("__"):
            continue
        v = getattr(U, name)
        if isinstance(v, (list, tuple, set)) and v and all(
                isinstance(x, str) for x in v):
            for t in v:
                out[t] = name.lstrip("_")
    return out


def by_sector(te, d=None, mode="slice", min_names=8, n_random=None, seed=SEED,
              min_train_years=MIN_TRAIN_YEARS, horizon=HORIZON, verbose=True):
    """
    Does the model work better in some sectors than others?

        mode "slice"  take the out-of-sample predictions the walk-forward
                      already produced and cut them by sector. Cheap, and it
                      asks the question you usually mean: does the ONE model
                      serve some sectors better?
        mode "refit"  train AND test inside one sector - a separate model per
                      sector, seeing only that sector's names. Needs `d`.
                      Two things are changed for it, and both matter: the market
                      block is DROPPED (it is computed across the whole
                      universe, so a sector-only model must not see it) and the
                      cross-sectional ranks are RECOMPUTED against that
                      sector's own peers instead of all 337 names. Without
                      those two the "sector-only" model is not sector-only.

    WHY THERE IS A RANDOM CONTROL

    Twelve sectors means twelve chances for one to look good. Run enough splits
    of anything and the best one always looks convincing. So the same slicing is
    repeated on RANDOM groupings with the same size profile, and what gets
    reported is whether the real sectors separate more than random groups of the
    same sizes do. If they do not, sector differences here are noise and the
    best-looking sector is just the luckiest one.
    """
    # refitting every random grouping as well would be ten times the work, so
    # the control gets fewer shuffles in refit mode and says so
    if n_random is None:
        # 8 shuffles can only bound a p-value at about 1/9, so a "0 of 8"
        # result is not significance. 24 is the minimum that can say < 0.05.
        n_random = 24
    smap = sector_map()
    te = te.copy()
    te["sector"] = te["ticker"].map(smap).fillna("OTHER")

    def slice_stats(frame, label):
        k = max(1, int(len(frame) * TOP_DECILE))
        top = frame.nlargest(k, "p")
        return {"group": label, "n_names": int(frame["ticker"].nunique()),
                "n": int(len(frame)),
                "base_wr": float(frame["label"].mean()),
                "top_wr": float(top["label"].mean()),
                "lift": float(top["label"].mean() - frame["label"].mean()),
                "base_R": float(frame["r_multiple"].mean()),
                "top_R": float(top["r_multiple"].mean()),
                "edge_R": float(top["r_multiple"].mean()
                                - frame["r_multiple"].mean())}

    rows = []
    for sec, g in te.groupby("sector"):
        if g["ticker"].nunique() < min_names:
            continue
        if mode == "refit":
            if d is None:
                raise RuntimeError('mode="refit" needs the full dataset `d`')
            names = set(g["ticker"].unique())
            sub = d[d["ticker"].isin(names)]
            # The market block is computed across the WHOLE universe, so leaving
            # it in would feed a sector-only model information from every other
            # sector. Inside a sector the cross-sectional ranks are also
            # recomputed against that sector's own peers, not the universe.
            sub = sub.copy()
            for src, dst in (("ret_20", "xs_ret_20"), ("rsi_14", "xs_rsi_14"),
                             ("atr_pct", "xs_atr_pct"),
                             ("px_vs_ema200", "xs_px_vs_ema200")):
                sub[dst] = sub.groupby("date")[src].rank(pct=True)
            feats = [f for f in ALL_FEATURES if f not in MKT_FEATURES]
            try:
                g = walk_forward(sub, feats, min_train_years, 1, seed,
                                 horizon, verbose=False)
            except RuntimeError:
                if verbose:
                    print(f"    {sec}: too few usable folds, skipped")
                continue
        rows.append(slice_stats(g, sec))
    r = pd.DataFrame(rows).sort_values("edge_R", ascending=False)

    if verbose:
        print("\n" + "=" * 88)
        print(f"  BY SECTOR  (mode={mode}, sectors with at least "
              f"{min_names} names)")
        print("=" * 88)
        print(f"  {'sector':16s} {'names':>6s} {'n':>8s} {'base wr':>8s} "
              f"{'top wr':>7s} {'lift':>7s} {'base R':>8s} {'top R':>7s} "
              f"{'edge R':>8s}")
        for _, q in r.iterrows():
            print(f"  {q['group']:16s} {int(q['n_names']):6d} {int(q['n']):8,d} "
                  f"{q['base_wr']:8.1%} {q['top_wr']:7.1%} {q['lift']:+7.1%} "
                  f"{q['base_R']:8.3f} {q['top_R']:7.3f} {q['edge_R']:+8.3f}")

    # --- the control: random groups of the same sizes -----------------------
    rng = np.random.default_rng(seed)
    tickers = te["ticker"].unique()
    sizes = r["n_names"].tolist()
    spreads = []
    for _ in range(n_random):
        perm = rng.permutation(tickers)
        i, fake = 0, {}
        for gi, sz in enumerate(sizes):
            for t in perm[i:i + sz]:
                fake[t] = f"R{gi}"
            i += sz
        tf = te[te["ticker"].isin(fake)].copy()
        tf["g"] = tf["ticker"].map(fake)
        e = [slice_stats(g, k)["edge_R"] for k, g in tf.groupby("g")
             if len(g) > 200]
        if len(e) > 1:
            spreads.append(max(e) - min(e))
    real_spread = float(r["edge_R"].max() - r["edge_R"].min())

    if verbose and spreads:
        med = float(np.median(spreads))
        pct = float((np.array(spreads) >= real_spread).mean())
        print(f"\n  RANDOM-GROUP CONTROL ({n_random} shuffles, same sizes"
              + (", sliced not refitted - approximate" if mode == "refit"
                 else "") + ")")
        print(f"    real sector spread in edge R: {real_spread:.3f}")
        print(f"    random groups of the same sizes: median {med:.3f}, "
              f"{pct:.0%} of shuffles reach the real spread")
        if pct > 0.10:
            print(f"    -> sectors do NOT separate more than random groups "
                  f"do. The best-looking sector")
            print(f"       here is the luckiest one, not the most suitable "
                  f"one. Do not select on it.")
        else:
            print(f"    -> sectors separate more than chance. Worth "
                  f"conditioning on, but check that the")
            print(f"       leaders are also the ones with the most names - a "
                  f"thin sector can clear this")
            print(f"       bar on a handful of trades.")
    return {"sectors": rows, "real_spread": real_spread,
            "random_spreads": spreads}


def ablation(d, min_train_years=MIN_TRAIN_YEARS, n_seeds=1, seed=SEED,
             horizon=HORIZON, verbose=True):
    """
    Which feature block is actually earning its place?

    The ladder is V54's, applied to timing. Each rung adds one block, and the
    metrics that matter are the STABILITY ones, not the mean. A block that
    lifts the average while doubling the year-to-year spread has not helped -
    it has made the model chase whatever regime it was fitted on, and the
    per-year table on the real run (+18.9% in 2022, -25.1% in 2020) is what
    that looks like from the outside.

        per_name         one company's own history, nothing else
        + cross_section  where it stands against its peers THAT DAY
        + market         the state of the whole market that day
    """
    ladder = {"per_name": FEATURES,
              "+ pattern": FEATURES + PATTERN_FEATURES,
              "+ cross_section": FEATURES + PATTERN_FEATURES + XS_FEATURES,
              "+ market": ALL_FEATURES}
    rows = []
    if verbose:
        print("\n" + "=" * 88)
        print("  FEATURE-BLOCK ABLATION")
        print("=" * 88)
    for name, feats in ladder.items():
        te = walk_forward(d, feats, min_train_years, n_seeds, seed, horizon,
                          verbose=False)
        k = max(1, int(len(te) * TOP_DECILE))
        top = te.nlargest(k, "p")
        per_year = pd.DataFrame(per_year_lift(te, verbose=False))
        rows.append({
            "block": name, "n_features": len(feats),
            "lift": float(top["label"].mean() - te["label"].mean()),
            "top_R": float(top["r_multiple"].mean()),
            "base_R": float(te["r_multiple"].mean()),
            "auc": auc(te["label"].to_numpy(int), te["p"].to_numpy(float)),
            "year_lift_mean": float(per_year["lift"].mean()),
            "year_lift_sd": float(per_year["lift"].std(ddof=1)),
            "year_lift_worst": float(per_year["lift"].min()),
            "years_positive": int((per_year["lift"] > 0).sum()),
            "n_years": int(len(per_year))})
    r = pd.DataFrame(rows)
    if verbose:
        print(f"  {'block':17s} {'feats':>6s} {'lift':>7s} {'top R':>7s} "
              f"{'base R':>7s} {'edge':>7s} | {'yr mean':>8s} {'yr sd':>7s} "
              f"{'IR':>5s} {'worst yr':>9s} {'pos':>6s}")
        for _, q in r.iterrows():
            edge = q["top_R"] - q["base_R"]
            ir = (q["year_lift_mean"] / q["year_lift_sd"]
                  if q["year_lift_sd"] > 0 else np.nan)
            print(f"  {q['block']:17s} {int(q['n_features']):6d} "
                  f"{q['lift']:+7.1%} {q['top_R']:7.3f} {q['base_R']:7.3f} "
                  f"{edge:+7.3f} | {q['year_lift_mean']:+8.1%} "
                  f"{q['year_lift_sd']:7.1%} {ir:5.2f} "
                  f"{q['year_lift_worst']:+9.1%} "
                  f"{int(q['years_positive'])}/{int(q['n_years']):<4d}")

        # An arm whose top decile does not beat BUYING EVERYTHING has no
        # economic edge, however steady its win-rate lift looks. An earlier
        # version of this verdict compared only lift and spread and recommended
        # the steadiest arm - which on the real run was the one whose top decile
        # earned 0.243R against a 0.249R base. Steady and worthless.
        r = r.assign(edge=r["top_R"] - r["base_R"])
        paying = r[r["edge"] > 0]
        print()
        if paying.empty:
            print(f"    -> NO ARM PAYS. Every block's top decile earns less "
                  f"than buying everything, so the")
            print(f"       ranking is not worth acting on whatever the win-rate "
                  f"lift says.")
        else:
            best_edge = paying.loc[paying["edge"].idxmax()]
            best_ir = paying.loc[(paying["year_lift_mean"]
                                  / paying["year_lift_sd"]).idxmax()]
            dead = r[r["edge"] <= 0]["block"].tolist()
            if dead:
                print(f"    no economic edge at all: {', '.join(dead)} "
                      f"(top decile below the base rate)")
            print(f"    best expectancy: {best_edge['block']} "
                  f"({best_edge['edge']:+.3f}R over buying everything)")
            print(f"    best information ratio among arms that pay: "
                  f"{best_ir['block']}")
            if best_edge["year_lift_worst"] < -0.05:
                print(f"    -> but its worst year is "
                      f"{best_edge['year_lift_worst']:+.1%}. The edge is real "
                      f"and the tail is real. Sizing, not")
                print(f"       selection, is what makes that survivable - and "
                      f"Pillar 2 already does sizing.")
    return rows


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--horizon", default=HORIZON, choices=list(HORIZON_CONFIGS))
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--label-mode", default=LABEL_MODE,
                    choices=["barrier", "profit"])
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--sector-mode", default="slice",
                    choices=["slice", "refit", "off"])
    ap.add_argument("--predict-mode", default=PREDICT_MODE,
                    choices=["classify", "expectancy"])
    ap.add_argument("--feature-set", default="+market")
    ap.add_argument("--sweep", action="store_true",
                    help="run every combination instead of one")
    ap.add_argument("--invert", action="store_true",
                    help="read the BOTTOM decile as the pick (post-hoc)")
    ap.add_argument("--n-random-draws", type=int, default=6)
    ap.add_argument("--rolling-holdout", action="store_true",
                    help="re-run the direction holdout at every split year")
    ap.add_argument("--direction-holdout", action="store_true",
                    help="choose the ranking direction on early years only, "
                         "then test it on later ones")
    ap.add_argument("--ticker-splits", type=int, default=5,
                    help="ticker-holdout repeats; 0 turns it off")
    ap.add_argument("--tickers", nargs="*",
                    help="run on these names only, instead of the whole cache")
    ap.add_argument("--out", default="entry_v64_result.json")
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    if c.rolling_holdout:
        av = sorted(os.path.splitext(f)[0]
                    for f in os.listdir(c.price_cache) if f.endswith(".pkl"))
        d = build_dataset([t for t in (c.tickers or av) if t in av],
                          c.price_cache, c.horizon, c.step, verbose=True)
        rolling_direction_holdout(apply_label_mode(d, c.label_mode),
                                  seed=c.seed,
                                  min_train_years=c.min_train_years,
                                  horizon=c.horizon)
        return 0
    if c.sweep:
        sweep(price_cache=c.price_cache, horizon=c.horizon, step=c.step,
              min_train_years=c.min_train_years, seed=c.seed,
              tickers=c.tickers, skip_audit=c.skip_audit,
              n_random_draws=c.n_random_draws, invert=c.invert,
              out="entry_v64_sweep.json")
        return 0
    run_entry_v64(c.price_cache, c.horizon, c.step, c.min_train_years,
                  c.n_seeds, c.seed, c.skip_audit, c.label_mode,
                  c.ticker_splits, c.tickers, not c.no_ablation,
                  None if c.sector_mode == "off" else c.sector_mode,
                  c.predict_mode, c.feature_set, c.out)
    return 0


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    # "sweep"  = every label x ranking x feature-set combination, one table
    # "single" = one configuration, with all the diagnostics
    RUN_MODE            = "sweep"

    RUN_PRICE_CACHE     = PRICE_CACHE
    RUN_HORIZON         = HORIZON       # "SHORT" | "MID" | "LONG"
    RUN_STEP            = STEP          # trading days between candidate entries
    RUN_MIN_TRAIN_YEARS = MIN_TRAIN_YEARS
    RUN_N_SEEDS         = N_SEEDS       # predictions averaged over this many fits
    RUN_SEED            = SEED
    RUN_SKIP_AUDIT      = False         # leave False. The audit is the point.

    # WHAT THE MODEL PREDICTS, and therefore what the ranking means.
    #   "expectancy" regress on the R multiple, rank by predicted R
    #   "classify"   predict P(win), rank by probability  (what V40 did)
    # Run both and compare the DECILE TABLE: if "classify" gives a U and
    # "expectancy" gives a rising table, the ranking target was the problem.
    RUN_PREDICT_MODE    = PREDICT_MODE

    # "barrier" = target before stop (what the system trades)
    # "profit"  = the trade ends above water. Run both and compare the decile
    #             tables - if barrier gives an INVERTED table and profit does
    #             not, the label was the problem, not the features.
    RUN_LABEL_MODE      = LABEL_MODE

    # Test on a slice instead of the whole cache: a sector, a size band, or
    # names the model has not seen. None = every ticker in the cache.
    RUN_TICKERS         = None          # e.g. ["NVDA", "AMD", "AVGO", "MU"]
    RUN_TICKER_SPLITS   = 5             # ticker-holdout repeats; 0 turns it off
    RUN_ABLATION        = True          # per_name -> +cross_section -> +market

    # Sector breakdown. "slice" cuts the walk-forward predictions by sector
    # (cheap, the usual question). "refit" trains a model per sector (slow, and
    # the small sectors hold too few names to trust). None turns it off.
    #   "slice" = one model on all 337 names, results cut by sector
    #   "refit" = a separate model trained AND tested inside each sector, with
    #             the market block dropped and peer ranks recomputed per sector
    RUN_SECTOR_MODE     = "refit"

    # Which columns the model may see. "before_pattern" and "after_pattern"
    # differ ONLY by the fifteen technical-data columns, so comparing those two
    # is the before/after test. "random" ranks by noise and is the yardstick.
    #   per_name | +pattern | +cross_section | +market
    #   before_pattern | after_pattern | random
    RUN_FEATURE_SET     = "+market"

    # What the sweep covers. Trim any list to make it faster.
    SWEEP_LABELS        = ("barrier", "profit")
    SWEEP_RANK_BY       = ("classify", "expectancy")
    SWEEP_FEATURES      = ("per_name", "no_confirm", "before_pattern",
                           "after_pattern", "random")

    # How many random rankings to draw for the noise bar. One draw cannot show
    # a spread, and the first sweep used one.
    SWEEP_RANDOM_DRAWS  = 6

    # Read the BOTTOM decile as the pick instead of the top. Every configuration
    # in the first real sweep had a NEGATIVE decile rank correlation, which means
    # the ranking is upside down. Setting this tests that directly - but the
    # direction was chosen after seeing the result, so it is a hypothesis to be
    # tested forward, not a finding.
    SWEEP_INVERT        = False

    RUN_OUT             = "entry_v64_result.json"
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())                # e.g. python class_ai_entry_v64.py --sweep
    elif RUN_MODE == "sweep":
        sweep(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON, step=RUN_STEP,
              label_modes=SWEEP_LABELS, predict_modes=SWEEP_RANK_BY,
              feature_sets=SWEEP_FEATURES,
              min_train_years=RUN_MIN_TRAIN_YEARS, seed=RUN_SEED,
              tickers=RUN_TICKERS, skip_audit=RUN_SKIP_AUDIT,
              n_random_draws=SWEEP_RANDOM_DRAWS, invert=SWEEP_INVERT,
              out="entry_v64_sweep.json")
    else:
        run_entry_v64(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
                      step=RUN_STEP, min_train_years=RUN_MIN_TRAIN_YEARS,
                      n_seeds=RUN_N_SEEDS, seed=RUN_SEED,
                      skip_audit=RUN_SKIP_AUDIT, label_mode=RUN_LABEL_MODE,
                      ticker_splits=RUN_TICKER_SPLITS, tickers=RUN_TICKERS,
                      run_ablation=RUN_ABLATION,
                      sector_mode=RUN_SECTOR_MODE,
                      predict_mode=RUN_PREDICT_MODE,
                      feature_set=RUN_FEATURE_SET, out=RUN_OUT)
