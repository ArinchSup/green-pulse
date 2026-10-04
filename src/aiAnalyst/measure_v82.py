"""
CLEANED IN V106: removed 6 functions, 5 constants, the old run section and 6
imports that nothing in the current project uses. The full original is in
log/originals_v106/measure_v82.py. MIN_ROWS now holds its value (2000) instead
of reading it from class_ai_entry_model_v76.

V82 - MEASURE THE RULE THE WAY PRODUCTION RUNS IT.

WHAT V81's INSPECTOR FOUND
--------------------------
V81 fired on wildly different shares of candidates from year to year:

    2008   178    2009      0    2010  7,735    2011  4,057    2012    68
    2013 4,335    2014  3,914    2015  1,952    2018    469    2019    43

That is fold-to-fold score drift. The walk-forward fits a NEW model each year,
and an expectancy regressor's output level tracks the average R of its own
training period, so each January the scale jumps. The rolling threshold was
taken from the trailing window - scored by LAST year's model - and applied to
THIS year's model. When the new model sat higher it flooded; when lower it
starved (2009: zero). The composite, whose scores are ranks, fired 93-186 every
year; V76 had the same disease (2 in 2009, 966 in 2022).

WHY THIS IS A MEASUREMENT BUG, NOT A PRODUCT BUG
------------------------------------------------
In production, score() rates the WHOLE trailing window with the single serving
model and takes the threshold from those scores. One model, one scale - the
fold mixing never happens. So the served V81 was fine; its measurement was not
measuring it. This file rescores the trailing window with EACH FOLD'S OWN model,
which is exactly what the served rule does. The served boosters are unchanged.

The trailing rows are mostly that fold's training data, so their scores are
in-sample. Production has the same property - the serving model scores its own
history to set today's cut - so the measurement now matches what is deployed.

ALSO FIXED HERE
---------------
MARKET BREADTH. V64 computes breadth over each DATE'S CANDIDATES - a median of
6 of ~300 stocks - which is why it read exactly 0.000 on every crash low. Here
breadth is rebuilt from EVERY ticker's daily close against its own 200-day EMA,
over all names alive that day. It is used for the regime split only; V81 does
not take it as a feature. The regime answer to "how does it do in normal and
bull markets" is only trustworthy on this series.

THE COMPOSITE. V80 ranked each feature within each date - among ~6 names - so
the "simple baseline" was handicapped. Composite v2 maps each feature to its
percentile in the trailing 252-day pool of all candidates, recomputed monthly:
causal, scale-free, and independent of how many names share a date. The old
version is kept beside it so the difference is visible.

INTEGRITY CHECK
---------------
The V81 arm is also scored under the OLD rule. The walk-forward here uses the
same builder, seeds and folds as V81, so that row must reproduce V81's 24,900
signals exactly. If it does not, something is nondeterministic and nothing
below can be compared with V81's output.
"""


import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v81 as M81

QUANTILE = M81.QUANTILE
WINDOW_DAYS = M81.WINDOW_DAYS
MIN_ROWS = 2000      # = class_ai_entry_model_v76.MIN_WINDOW_ROWS
BREADTH_CUTS = (0.40, 0.60)


def _win():
    return pd.Timedelta(days=int(WINDOW_DAYS * 365.25 / 252))


# =============================================================================
# 1. FULL-UNIVERSE MARKET BREADTH
# =============================================================================
def full_breadth(price_cache, names, min_names=50):
    """Share of ALL live tickers closing above their own 200-day EMA, daily."""
    cols = {}
    for t in names:
        df = E64.P2.load_prices(t, price_cache)
        if df is None or len(df) < 260:
            continue
        c = df["Close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")]
        e = c.ewm(span=200, adjust=False).mean()
        a = (c > e).astype(float)
        a.iloc[:200] = np.nan                  # EMA not yet meaningful
        cols[t] = a
    W = pd.DataFrame(cols)
    n = W.notna().sum(axis=1)
    b = W.mean(axis=1, skipna=True)
    b[n < min_names] = np.nan
    return b, n


# =============================================================================
# 3. COMPOSITE v2 - trailing pooled percentiles, not per-date ranks
# =============================================================================
def trailing_pct(frame, col, min_rows=MIN_ROWS):
    win = _win()
    di = pd.DatetimeIndex(frame["date"])
    v = frame[col].to_numpy(float)
    o = np.argsort(di.values, kind="mergesort")
    ds, vs = di.values[o], v[o]
    per = pd.PeriodIndex(di, freq="M")
    codes, uniq = pd.factorize(per)
    out = np.full(len(frame), np.nan)
    for k, m in enumerate(uniq):
        t = m.to_timestamp()
        lo = np.searchsorted(ds, (t - win).to_datetime64(), "left")
        hi = np.searchsorted(ds, t.to_datetime64(), "left")
        w = vs[lo:hi]
        w = np.sort(w[np.isfinite(w)])
        if w.size < min_rows:
            continue
        idx = np.flatnonzero(codes == k)
        out[idx] = np.searchsorted(w, v[idx], side="right") / w.size
    return out


def composite_v2(frame, feats_dirs):
    parts = []
    for f, sgn in feats_dirs:
        pc = trailing_pct(frame, f)
        parts.append(pc if sgn > 0 else 1.0 - pc)
    return np.nanmean(np.vstack(parts), axis=0)
