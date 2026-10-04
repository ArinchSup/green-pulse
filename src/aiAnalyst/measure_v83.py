"""
CLEANED IN V106: removed 1 function, 3 constants, the old run section and 6
imports that nothing in the current project uses. The full original is in
log/originals_v106/measure_v83.py.

V83 - TIES, NOT DRIFT: WHY V81 STILL FLOODS, AND THE FIX.

WHAT V82 SHOWED
---------------
V82 rescored each fold's trailing window with that fold's own model, which
removes fold-to-fold scale drift completely. V81 still fired 7,637 times in
2010 - 53% of that year's candidates - against 7,735 under the old cut. So the
drift diagnosis was wrong.

The arithmetic points one way. Over a year, the monthly cut is the 99th
percentile of a window that increasingly contains that year's own scores. More
than half of those rows can sit at or above it only if they are TIED AT THE CUT.
Under `score >= cut`, a tied block is admitted whole. Some fold models must be
close to flat: a few distinct values, a large share of candidates on the top
one.

WHY A FOLD WOULD GO FLAT (a hypothesis this file tests, not a finding)
---------------------------------------------------------------------
V81 forces every feature to be "low is good". Folds trained on years dominated
by 2008 - when oversold stocks kept falling - see data that CONTRADICTS that
direction. A monotone model that cannot fit the allowed direction has little
choice but to stay flat, so its predictions collapse onto a handful of values.
If that is right, the flat folds are the early ones, and the problem fades as
2008 becomes a smaller share of each fold's training data.

WHAT THIS FILE DOES
-------------------
  [3] DEGENERACY per year: distinct score values, the share sitting on the
      year's maximum, and the modal share - for V81's fold models, V76's, and
      the SERVED model scoring the whole history (which is what production
      faces). This confirms or kills the tie explanation directly.

  [4]-[5] every arm under BOTH rules:
        >=   the current rule; a tied block at the cut is admitted whole
        >    strict; a block tied AT the cut is never admitted. A model that
             cannot tell candidates apart does not fire on them - the right
             behaviour for a model meant to fire only when very confident.
      Both are reported for every arm, because strict also changes features
      with natural mass points (range60_position is exactly 0 for every stock
      at its 60-day low). The cost to those arms is shown, not hidden.

  WARM START FOR EVERY ARM. V82 computed non-model arms on the test rows only,
  so their trailing windows started cold in 2008, and composite v2 lost the
  whole of 2008 because NaN scores in a window turned its cut to NaN. Here every
  non-model arm is scored on the full dataset from 2005 with NaN-aware cuts,
  then read off on the test rows - the same warm start the model arms already
  had through their rescored trailing windows.
"""


import numpy as np
import pandas as pd

import measure_v82 as V82

QUANTILE = V82.QUANTILE
MIN_ROWS = V82.MIN_ROWS


# =============================================================================
# RULES
# =============================================================================
def rolling_cut(dates, scores, quantile=QUANTILE, min_rows=MIN_ROWS):
    """Per-row monthly cut from the trailing window. NaN-aware."""
    win = V82._win()
    di = pd.DatetimeIndex(dates)
    s = np.asarray(scores, float)
    o = np.argsort(di.values, kind="mergesort")
    ds, ss = di.values[o], s[o]
    codes, uniq = pd.factorize(pd.PeriodIndex(di, freq="M"))
    cut = np.full(len(s), np.nan)
    for k, m in enumerate(uniq):
        t = m.to_timestamp()
        lo = np.searchsorted(ds, (t - win).to_datetime64(), "left")
        hi = np.searchsorted(ds, t.to_datetime64(), "left")
        w = ss[lo:hi]
        w = w[np.isfinite(w)]
        if w.size < min_rows:
            continue
        cut[codes == k] = float(np.quantile(w, 1.0 - quantile))
    return cut


def apply_rule(score, cut, strict):
    s = np.asarray(score, float)
    ok = np.isfinite(cut) & np.isfinite(s)
    return ok & ((s > cut) if strict else (s >= cut))


def fold_cut(te, refs, quantile=QUANTILE, min_rows=MIN_ROWS):
    """V82's fold-consistent cut, NaN-aware, returned per row."""
    win = V82._win()
    cut = np.full(len(te), np.nan)
    per = pd.PeriodIndex(pd.DatetimeIndex(te["date"]), freq="M")
    yrs = te["year"].to_numpy()
    for y, ref in refs.items():
        rd = pd.DatetimeIndex(ref["date"]).values
        rp = ref["p"].to_numpy(float)
        o = np.argsort(rd, kind="mergesort")
        rd, rp = rd[o], rp[o]
        rows = np.flatnonzero(yrs == y)
        if not len(rows):
            continue
        pr = per[rows]
        for m in pd.unique(pr):
            t = m.to_timestamp()
            lo = np.searchsorted(rd, (t - win).to_datetime64(), "left")
            hi = np.searchsorted(rd, t.to_datetime64(), "left")
            w = rp[lo:hi]
            w = w[np.isfinite(w)]
            if w.size < min_rows:
                continue
            cut[rows[pr == m]] = float(np.quantile(w, 1.0 - quantile))
    return cut


# =============================================================================
# DEGENERACY
# =============================================================================
def degeneracy(frame, score):
    """
    'Share >= own p99' is the number that decides it. For a continuous score
    it is ~1%. If a plateau sits on or just under the top, the 99th percentile
    lands ON the plateau and this share jumps - which is exactly what makes
    `>=` admit half a year. 'Share at max' alone misses a plateau one notch
    below a handful of higher values; the development data showed that case.
    """
    s = np.round(np.asarray(score, float), 9)
    rows = []
    for y in sorted(pd.unique(frame["year"])):
        v = s[(frame["year"] == y).to_numpy()]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        vals, cnt = np.unique(v, return_counts=True)
        p99 = float(np.quantile(v, 0.99))
        rows.append({"Year": int(y), "Rows": int(v.size),
                     "Distinct": int(vals.size),
                     "Share >= own p99": float((v >= p99).mean()),
                     "Share at max": float(cnt[-1] / v.size),
                     "Modal share": float(cnt.max() / v.size)})
    return pd.DataFrame(rows)
