"""
class_ai_portfolio.py  -  PORTFOLIO RISK LAYER

WHAT THIS FIXES
---------------
Pillar 2 sizes one name at a time. Nothing in it, and nothing in the pipeline,
looks at the book as a whole. Six names at the 20% concentration ceiling is 120%
of equity; eight names each risking 2% at their stops is 16% of equity gone if
they stop out together; and in a technology-heavy universe they largely DO stop
out together. A per-name risk model that ignores correlation is not conservative,
it is blind in the one direction that matters.

This module takes the per-name positions the pipeline produced and shrinks them
until the BOOK satisfies four constraints. It never vetoes a name - V51 tested
refusing against shrinking and refusing lost (Calmar worse by 0.020, 95% CI
+0.004 to +0.035). Shrinking is what the evidence supports, so shrinking is all
this does.

    1. gross exposure        sum of positions <= MAX_GROSS_PCT
    2. risk at stops         sum of effective risk <= MAX_PORTFOLIO_RISK_PCT
    3. cluster concentration no correlation cluster above MAX_CLUSTER_PCT
    4. book drawdown         P(book falls DRAWDOWN in HORIZON_DAYS) <= ceiling

Only #4 uses the correlation matrix for anything other than grouping, and it is
the only one of the four that makes a new empirical claim. That claim is stated
and bounded in HONEST LIMITS below, and evaluate_portfolio_v60.py tests it.

WHAT IS MEASURED AND WHAT IS CONVENTION
---------------------------------------
Measured from the cached daily bars:
    - the correlation matrix (log returns, CORR_WINDOW bars up to as_of,
      intersected across names so every pair uses the same sample)
    - the book's forecast volatility, w' (D R D) w, with Pillar 2's forecast
      volatilities on the diagonal
    - the diversification ratio, and each name's marginal contribution

Convention, not a finding:
    - MAX_PORTFOLIO_RISK_PCT = 6.0. This is the widely used "six percent rule",
      adopted because it has a provenance, NOT because this project measured it.
      Tuning it needs a portfolio-level version of the V51 path simulation; that
      test does not exist yet. Do not present 6% as a result.
    - MAX_GROSS_PCT = 100.0 simply refuses implicit leverage.
    - CLUSTER_CORR = 0.70 and MAX_CLUSTER_PCT = 40.0 are judgment.
    - MAX_PORTFOLIO_DD_PROB is pinned to the calibration's own base rate, so it
      means something concrete: the book must be no likelier to fall DRAWDOWN
      than the average single name in the universe.

HONEST LIMITS
-------------
* The drawdown curve was fitted on SINGLE NAMES, and feeding it a book's forecast
  volatility assumes the volatility-to-drawdown mapping depends on the volatility
  LEVEL and not on what is producing it. V60 tested that directly: 2,496
  historical books over 337 names, with the event definition cross-checked against
  P2's own arithmetic on 381 observations and a single-name control that
  reproduces the fitted curve (predicted 13.2%, realised 14.7% on P2's own
  measure, calibration error 3.86pp). The instrument is sound. What it found:

      USABLE, NOT EXACT. Inside the fitted range the prediction averaged 10.6%
      against realised outcomes bracketed by 6.9% and 8.7% - the bracket being the
      measurement convention, since a book has no single intraday low the way one
      stock does.

      ACCURACY IS COMPARABLE AT BEST AND HALF AS GOOD AT WORST. Calibration error
      is 1.90pp measured on daily lows and 3.67pp on closes, against 1.55pp for the
      single-name model out of sample. Which end holds depends on how far a book's
      true intraday low sits from its close, which daily bars cannot resolve. Do
      not read the bootstrap interval's lower end here: for a mean-ABSOLUTE error
      the percentile interval drifts upward, far enough that it can sit above the
      point estimate, and treating that as evidence would be quoting the
      bootstrap's own bias.

      IT LEANS HIGH, BY ROUGHLY TWO TO FOUR POINTS. The true fall lies between the
      close-based and daily-lows measures, so the true bias lies between the two
      biases computed from them: -3.67pp and -1.85pp. Both negative, so the bracket
      does not span zero and the direction does not depend on the convention - the
      curve OVER-predicts. It is not significant at the least favourable edge of
      that bracket (daily lows, 95% CI -4.57 to +1.42), so the magnitude is a
      range, not a number.

      BREADTH MATTERS BEYOND VOLATILITY, AND IT MATTERS THE SAFE WAY. Pooled over
      the fitted range, at matched book volatility, a single name fell 20% or more
      5.2pp more often than a book of three or more did (95% CI +1.1 to +9.0,
      curve-adjusted so the residual volatility difference between the groups is
      absorbed). The mechanism is aggregation: averaging several names thins the
      tails for a given standard deviation, and a 20% fall is a tail event. A book
      and a single name at the same volatility are not the same distribution.

  That breadth result and the bias bracket are two independent measurements
  pointing the same way: the curve was fitted on single names, so it over-states a
  diversified book. The breadth figure is a lower bound besides, because the
  close-based measure under-counts a single name more than it under-counts a book
  and so understates the gap between them.

  It is deliberately left uncorrected. A breadth adjustment fitted on roughly 400
  effective observations would be a new model with nothing validating it, traded
  against an error already on the safe side.

  Practical reading: treat the book probability as indicative and slightly
  cautious, with a couple of points of slack, and do not present it as calibrated
  the way the per-name probability is.

* Below the curve's fitted volatility range the number is a clamped bound rather
  than a reading, which is the ordinary case for a diversified book. V60 measured
  that bound as safe with room: 1,537 books, a 3.2% floor reported against 0.3%
  realised (95% CI 0.0 to 0.8), still holding at 0.7% on a synchronised-lows
  over-count. The cluster ceiling and the risk-at-stops budget are what actually
  bind on a normal book; this constraint is a backstop for concentrated ones.

* Summing effective risk assumes every stop fills - the perfectly correlated
  case. That is deliberately the worst case. A correlation-adjusted expected
  simultaneous loss would need a joint model of stop-hitting times, which is
  more machinery than the evidence supports, so this file does not invent one.

* Correlation is estimated on a trailing window and correlations rise in
  exactly the stress the constraints exist for. The estimate is therefore
  optimistic when it matters most. Shrinkage toward the mean pairwise
  correlation pushes the other way a little; it does not fix it.

* Scaling is pro-rata. It preserves the relative sizes the per-name model chose
  rather than re-optimising them, which would be a second model layered on the
  first with nothing validating it.

CLI
    python class_ai_portfolio.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2


# =============================================================================
# CONFIGURATION
# =============================================================================
CORR_WINDOW = 252           # trading bars of history for the correlation matrix
MIN_CORR_OBS = 120          # below this overlap a pair is not estimated
CORR_FALLBACK = 0.60        # correlation assumed for a name with no usable
                            # overlap, used only if nothing better is measurable

CLUSTER_CORR = 0.70         # single-linkage threshold: at or above this, two
                            # names are treated as one bet

# A cluster is one bet, so its ceiling is stated as a MULTIPLE of the per-name
# concentration ceiling rather than typed in as a number. At 2x, a group of
# names that move together may carry twice what any single name may - generous
# relative to the per-name limit, and it stays coherent if that limit changes.
CLUSTER_CAP_MULT = 2.0
MAX_CLUSTER_PCT = CLUSTER_CAP_MULT * getattr(P2, "MAX_POSITION_PCT", 20.0)

MAX_GROSS_PCT = 100.0       # no implicit leverage
MAX_PORTFOLIO_RISK_PCT = 6.0    # convention - see the header
DEFAULT_EQUITY = getattr(P2, "DEFAULT_EQUITY", 10000.0)
DEFAULT_RISK_BUDGET = getattr(P2, "DEFAULT_RISK_BUDGET", 0.02)
MIN_POSITION_PCT = 1.0      # below this a scaled position is dropped, not
                            # traded as a rounding artefact

# The book's drawdown ceiling is set from the calibration's base rate rather
# than typed in, so it follows the question when the question changes. None
# means "use the base rate"; a float overrides it.
MAX_PORTFOLIO_DD_PROB = None
PORTFOLIO_DD_MULT = 1.00    # multiple of the base rate to use when the above
                            # is None

SCALE_FLOOR = 0.05          # never scale below this; a smaller book than this
                            # is a configuration problem, not a trade


# =============================================================================
# CORRELATION
# =============================================================================
def _returns_panel(tickers, as_of=None, price_cache=P2.PRICE_CACHE,
                   window=CORR_WINDOW):
    """
    Aligned log returns for `tickers`, ending at or before as_of.

    Names are intersected on dates so every pair is estimated on the SAME
    sample. Pairwise-complete correlation on different samples is the usual way
    to end up with a matrix that is not positive semi-definite, and then a
    negative portfolio variance and a square root of a negative number.
    """
    cols, missing = {}, []
    for t in tickers:
        df = P2.load_prices(t, price_cache)
        if df is None or "Close" not in df:
            missing.append(t)
            continue
        s = df["Close"]
        if as_of is not None:
            s = s[s.index <= pd.Timestamp(as_of)]
        s = s.dropna()
        if len(s) < MIN_CORR_OBS + 1:
            missing.append(t)
            continue
        cols[t] = np.log(s).diff()
    if not cols:
        return pd.DataFrame(), missing
    panel = pd.DataFrame(cols).dropna(how="any")
    if len(panel) > window:
        panel = panel.iloc[-window:]
    if len(panel) < MIN_CORR_OBS:
        # The intersection collapsed - usually one short history dragging the
        # rest down. Drop the shortest name and say so rather than returning a
        # correlation matrix built on 30 days.
        lens = {t: int(cols[t].notna().sum()) for t in cols}
        shortest = min(lens, key=lens.get)
        remaining = [t for t in cols if t != shortest]
        if remaining:
            sub, sub_missing = _returns_panel(remaining, as_of, price_cache,
                                              window)
            return sub, missing + [shortest] + sub_missing
        return pd.DataFrame(), missing + [shortest]
    return panel, missing


def _shrink_to_constant(R, n_obs):
    """
    Shrink a sample correlation matrix toward its own mean off-diagonal.

    Intensity k/n is a heuristic, not the Ledoit-Wolf optimum: with k names and
    n observations the sample matrix has k(k-1)/2 free parameters and a fixed
    n, so the estimate degrades as k grows. This is cheap insurance that also
    keeps the matrix well conditioned. It is not claimed to be optimal.
    """
    k = R.shape[0]
    if k < 2 or n_obs <= 0:
        return R, 0.0, float("nan")
    off = R[~np.eye(k, dtype=bool)]
    rbar = float(np.mean(off))
    lam = float(min(0.5, k / float(n_obs)))
    T = np.full_like(R, rbar)
    np.fill_diagonal(T, 1.0)
    return (1.0 - lam) * R + lam * T, lam, rbar


def _nearest_psd(R, floor=1e-8):
    """Clip eigenvalues at a positive floor and restore a unit diagonal."""
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    if w.min() >= floor:
        np.fill_diagonal(R, 1.0)
        return R, False
    w = np.clip(w, floor, None)
    R2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.clip(np.diag(R2), floor, None))
    R2 = R2 / np.outer(d, d)
    np.fill_diagonal(R2, 1.0)
    return 0.5 * (R2 + R2.T), True


def correlation_matrix(tickers, as_of=None, price_cache=P2.PRICE_CACHE,
                       window=CORR_WINDOW):
    """
    Correlation matrix for `tickers`, in the order given.

    Names with no usable overlap get the mean measured pairwise correlation of
    the names that do - not zero. Assuming independence for a name you could
    not measure is the single most dangerous default available here, because it
    makes the book look diversified precisely where you know least.
    """
    n = len(tickers)
    if n == 0:
        return np.zeros((0, 0)), {"n_obs": 0, "measured": [], "assumed": [],
                                  "mean_corr": float("nan"), "shrinkage": 0.0,
                                  "psd_repaired": False, "start": None,
                                  "end": None}
    if n == 1:
        return np.ones((1, 1)), {"n_obs": 0, "measured": list(tickers),
                                 "assumed": [], "mean_corr": float("nan"),
                                 "shrinkage": 0.0, "psd_repaired": False,
                                 "start": None, "end": None}

    panel, missing = _returns_panel(tickers, as_of, price_cache, window)
    measured = [t for t in tickers if t in panel.columns]
    assumed = [t for t in tickers if t not in panel.columns]

    R = np.eye(n)
    info = {"n_obs": int(len(panel)), "measured": measured, "assumed": assumed,
            "shrinkage": 0.0, "psd_repaired": False,
            "start": str(panel.index[0].date()) if len(panel) else None,
            "end": str(panel.index[-1].date()) if len(panel) else None}

    if len(measured) >= 2:
        Rm = panel[measured].corr().to_numpy(float)
        Rm = np.nan_to_num(Rm, nan=0.0)
        np.fill_diagonal(Rm, 1.0)
        Rm, lam, rbar = _shrink_to_constant(Rm, len(panel))
        info["shrinkage"] = round(lam, 4)
        fallback = rbar
        idx = {t: i for i, t in enumerate(measured)}
        for a in measured:
            for b in measured:
                R[tickers.index(a), tickers.index(b)] = Rm[idx[a], idx[b]]
    else:
        fallback = CORR_FALLBACK

    if assumed:
        for t in assumed:
            i = tickers.index(t)
            for j in range(n):
                if j != i:
                    R[i, j] = R[j, i] = fallback

    R, repaired = _nearest_psd(R)
    info["psd_repaired"] = bool(repaired)
    off = R[~np.eye(n, dtype=bool)]
    info["mean_corr"] = round(float(np.mean(off)), 4) if off.size else float("nan")
    info["fallback_used"] = round(float(fallback), 4) if assumed else None
    return R, info


def clusters(R, tickers, threshold=CLUSTER_CORR):
    """
    Single-linkage grouping at `threshold`. Two names in the same group are one
    bet for exposure purposes.

    Negative correlation diversifies, so the test is rho >= threshold, not
    |rho| >= threshold.
    """
    n = len(tickers)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if R[i, j] >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(tickers[i])
    # largest cluster first, then alphabetically, so the output is stable
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# =============================================================================
# PORTFOLIO RISK
# =============================================================================
def portfolio_vol(w_pct, vols_pct, R):
    """
    Annualised forecast volatility of the book, in percent.

    w_pct are positions as PERCENT OF EQUITY, so a book that is not fully
    invested has the rest in cash and a correspondingly lower volatility. That
    is the intended meaning: this is the volatility of the equity, not of the
    invested sleeve.
    """
    w = np.asarray(w_pct, float) / 100.0
    s = np.asarray(vols_pct, float)
    if w.size == 0 or not np.isfinite(w).all() or not np.isfinite(s).all():
        return float("nan")
    cov = R * np.outer(s, s)
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)))


def risk_contributions(w_pct, vols_pct, R):
    """
    Each name's share of book volatility. Sums to 1.

    This is the number that surprises people: the largest position is often not
    the largest contributor, because contribution is weight times correlation
    with the rest of the book, not weight alone.
    """
    w = np.asarray(w_pct, float) / 100.0
    s = np.asarray(vols_pct, float)
    cov = R * np.outer(s, s)
    sp = np.sqrt(max(float(w @ cov @ w), 0.0))
    if sp <= 0:
        return np.full(w.shape, np.nan)
    return (w * (cov @ w)) / (sp ** 2)


def diversification_ratio(w_pct, vols_pct, R):
    """
    Book volatility divided by the fully-correlated sum. 1.0 means correlation
    bought nothing; lower is better. For a single-sector book expect something
    in the high 0.8s, which is the point of reporting it.
    """
    w = np.asarray(w_pct, float) / 100.0
    s = np.asarray(vols_pct, float)
    denom = float(np.sum(w * s))
    if denom <= 0:
        return float("nan")
    return portfolio_vol(w_pct, vols_pct, R) / denom


# =============================================================================
# THE BOOK DRAWDOWN PROBABILITY
# =============================================================================
def curve_bounds(calib):
    """(min p, max p, min fvol, max fvol) of the fitted curve."""
    pts = calib["curve"]
    ps = [float(p["p"]) for p in pts]
    fs = [float(p["fvol"]) for p in pts]
    return min(ps), max(ps), min(fs), max(fs)


def book_drawdown_prob(book_vol_pct, calib):
    """
    Read the book's forecast volatility off the SINGLE-NAME curve.

    Returns (probability, kind) where kind is "point", "upper" or "lower".

    The kind matters more than it looks. A scaled, diversified book routinely
    has a lower forecast volatility than the calmest SINGLE NAME the curve was
    fitted on, because diversification and the uninvested remainder both push it
    down. np.interp then clamps, and the clamped value is not an estimate: it is
    the curve's floor, and the book's true probability is somewhere BELOW it. A
    floor reported as a point estimate would make this constraint look like it
    was measuring something when it had simply run out of range - so it is
    labelled, and the report prints "<=" rather than "=".

    The practical consequence: this constraint is a backstop that bites only on
    a concentrated or very volatile book. The cluster ceiling and the risk-at-
    stops budget are what do the work on an ordinary one.

    Flagged as an extrapolation in composition either way. See HONEST LIMITS.
    """
    if not np.isfinite(book_vol_pct):
        return float("nan"), "none"
    pts = calib["curve"]
    xs = [float(p["fvol"]) for p in pts]
    ys = [float(p["p"]) for p in pts]
    p = float(np.interp(book_vol_pct, xs, ys))
    if book_vol_pct < min(xs):
        return p, "upper"
    if book_vol_pct > max(xs):
        return p, "lower"
    return p, "point"


def base_rate(calib):
    """The calibration's unconditional event rate, however the file spells it."""
    for k in ("base_rate", "event_rate", "unconditional_rate"):
        if k in calib and calib[k] is not None:
            try:
                v = float(calib[k])
                return v / 100.0 if v > 1.0 else v
            except (TypeError, ValueError):
                pass
    # Fall back to the population-weighted mean of the curve, which is the base
    # rate by construction when the bins are equal-count.
    pts = calib.get("curve") or []
    ns = [float(p.get("n", 1) or 1) for p in pts]
    ps = [float(p["p"]) for p in pts]
    if ps and sum(ns) > 0:
        return float(np.average(ps, weights=ns))
    return float("nan")


def dd_ceiling(calib):
    """The book's drawdown ceiling, and where it came from."""
    if MAX_PORTFOLIO_DD_PROB is not None:
        return float(MAX_PORTFOLIO_DD_PROB), "configured"
    br = base_rate(calib)
    if not np.isfinite(br):
        return float("nan"), "unavailable"
    return br * PORTFOLIO_DD_MULT, f"{PORTFOLIO_DD_MULT:.2f}x base rate {br:.1%}"


def _scale_for_prob(w_new, w_held, vols_pct, R, calib, target, n_grid=201):
    """
    Largest s in [0, 1] with P(book falls) <= target, where the book is
    w_held + s * w_new. Only the new positions scale; holdings are already on.

    A GRID, not bisection. With no holdings and non-negative correlations the
    probability is monotone in s and bisection would be safe, but neither holds
    in general: a negatively correlated addition can lower book volatility, so
    P(s) need not be monotone and bisection could converge on the wrong side of
    a bracket. 201 quadratic forms on a matrix this small cost nothing, and the
    grid is correct whatever the shape.

    Returns (s, reachable). reachable is False when the curve's own minimum
    probability already exceeds the target - a configuration error, not a market
    condition, and one that must NOT be answered by scaling the book to nothing.
    """
    if not np.isfinite(target):
        return 1.0, True
    full = portfolio_vol(w_held + w_new, vols_pct, R)
    if not np.isfinite(full) or full <= 0:
        return 1.0, True
    p1, _kind = book_drawdown_prob(full, calib)
    if np.isfinite(p1) and p1 <= target:
        return 1.0, True
    p_floor, _, _, _ = curve_bounds(calib)
    if p_floor > target:
        return 1.0, False            # unreachable; caller warns, no scaling
    best = 0.0
    for s in np.linspace(0.0, 1.0, n_grid):
        v = portfolio_vol(w_held + s * w_new, vols_pct, R)
        p, _ = book_drawdown_prob(v, calib)
        if np.isfinite(p) and p <= target:
            best = float(s)
    return best, True


def assess_candidates(tickers, calib=None, equity=10000.0, risk_budget=0.02,
                      as_of=None, price_cache=P2.PRICE_CACHE):
    """
    Turn tickers into candidate dicts through Pillar 2.

    Used for holdings, whose volatility forecast and stop distance the portfolio
    layer needs but which never went through the pipeline in this run.
    """
    calib = calib or P2.load_calibration()
    out, failed = [], []
    for t in tickers:
        r = P2.assess(t, target_date=as_of, equity=equity,
                      risk_budget=risk_budget, calib=calib,
                      price_cache=price_cache)
        if r is None:
            failed.append(t)
            continue
        out.append({"ticker": t,
                    "position_pct_of_equity": r["sizing"]["position_pct_of_equity"],
                    "stop_pct": r["levels"]["stop_pct"],
                    "entry": r["levels"]["entry"],
                    "forecast_vol_annual_pct": r["risk"]["forecast_vol_annual_pct"]})
    return out, failed


def parse_holdings(specs, calib=None, equity=10000.0, as_of=None,
                   price_cache=P2.PRICE_CACHE):
    """
    Parse TICKER:PCT_OF_EQUITY strings into held-position dicts.

    The stop distance and volatility forecast come from Pillar 2; only the
    weight comes from the caller, because that is the one thing Pillar 2 cannot
    know. A holding Pillar 2 cannot assess is returned in `failed` rather than
    assumed away - leaving it out would understate every limit it consumes.
    """
    held, failed = [], []
    for spec in specs or []:
        if ":" not in spec:
            failed.append({"spec": spec, "reason": "expected TICKER:PCT"})
            continue
        t, _, v = spec.partition(":")
        t = t.strip().upper()
        try:
            pct = float(v)
        except ValueError:
            failed.append({"spec": spec, "reason": "percentage not a number"})
            continue
        if pct <= 0:
            failed.append({"spec": spec, "reason": "percentage not positive"})
            continue
        cands, bad = assess_candidates([t], calib=calib, equity=equity,
                                       as_of=as_of, price_cache=price_cache)
        if bad or not cands:
            failed.append({"spec": spec,
                           "reason": "Pillar 2 could not assess this holding, so "
                                     "the limits it consumes cannot be measured"})
            continue
        c = cands[0]
        c["position_pct_of_equity"] = pct
        held.append(c)
    return held, failed


# =============================================================================
# CONSTRUCTION
# =============================================================================
def _need(c, key):
    v = c.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _validate(rows, label):
    """Split candidate dicts into usable ones and named exclusions."""
    usable, excluded = [], []
    for c in rows or []:
        t = c.get("ticker")
        pos = _need(c, "position_pct_of_equity")
        stop = _need(c, "stop_pct")
        entry = _need(c, "entry")
        vol = _need(c, "forecast_vol_annual_pct")
        if not t:
            excluded.append({"ticker": t, "reason": f"{label}: no ticker"})
        elif pos is None or pos <= 0:
            excluded.append({"ticker": t, "reason": f"{label}: no position size"})
        elif stop is None or stop <= 0:
            excluded.append({"ticker": t, "reason": f"{label}: no stop distance"})
        elif entry is None or entry <= 0:
            excluded.append({"ticker": t, "reason": f"{label}: no entry price"})
        elif vol is None or vol <= 0:
            excluded.append({"ticker": t,
                             "reason": f"{label}: no volatility forecast"})
        else:
            usable.append({"ticker": t, "pos0": pos, "stop_pct": stop,
                           "entry": entry, "vol": vol})
    return usable, excluded


def build_book(candidates, calib=None, equity=10000.0, as_of=None,
               price_cache=P2.PRICE_CACHE, held=None,
               max_gross_pct=MAX_GROSS_PCT,
               max_risk_pct=MAX_PORTFOLIO_RISK_PCT,
               max_cluster_pct=MAX_CLUSTER_PCT,
               max_name_pct=None,
               cluster_corr=CLUSTER_CORR,
               min_position_pct=MIN_POSITION_PCT):
    """
    Turn per-name positions into a book that satisfies the four constraints.

    `candidates` is a list of dicts, each needing:
        ticker, position_pct_of_equity, stop_pct, entry, forecast_vol_annual_pct

    Nothing else is read, so this does not depend on the pipeline's result
    shape. Names missing any of those fields are returned untouched in
    `excluded` with a reason - they are NOT silently assigned a weight of zero
    and they are NOT silently included with a guessed volatility.

    `held` is existing positions in the same shape. They CONSUME the limits but
    are never resized, because this module cannot sell what it did not buy. A
    screen run without them measures an empty book and will happily authorise a
    second full allocation on top of one already on - which is the more dangerous
    of the two errors, so holdings are supported rather than assumed away. When
    holdings already fill a limit the answer is zero new exposure, and that is
    reported as such instead of being floored at something tradeable.
    """
    calib = calib or P2.load_calibration()
    if max_name_pct is None:
        max_name_pct = getattr(P2, "MAX_POSITION_PCT", 20.0)

    new_u, excluded = _validate(candidates, "candidate")
    held_u, held_bad = _validate(held, "holding")
    excluded += held_bad

    out = {"equity": equity, "n_candidates": len(candidates or []),
           "n_held_existing": len(held_u), "excluded": excluded,
           "positions": [], "warnings": []}
    if held_bad:
        out["warnings"].append(
            f"{len(held_bad)} existing holding(s) could not be measured and are "
            f"NOT counted against the limits, which therefore understate the "
            f"book: {', '.join(str(e['ticker']) for e in held_bad)}")

    if not new_u:
        out["portfolio"] = None
        out["warnings"].append("no usable candidates")
        return out

    # ONE ROW PER TICKER. Holding a name that is also a candidate is the common
    # case - adding to a winner - and the first version gave it two rows. The
    # correlation matrix was then built from a ticker list containing duplicates:
    # the returns frame deduplicated the column, tickers.index() resolved both
    # slots to the first one, and the second row stayed at the identity. Measured
    # mean correlation fell from 0.57 to 0.31 and the book looked twice as
    # diversified as it was, purely because one name appeared twice. Worse, the
    # two sleeves were sized independently, so a 15% holding plus a 20% buy
    # passed every check as 35% of equity in one name.
    order, slot = [], {}
    for u in held_u + new_u:
        if u["ticker"] not in slot:
            slot[u["ticker"]] = len(order)
            order.append(u["ticker"])
    tickers = order
    n = len(tickers)

    vols = np.zeros(n)
    stops = np.zeros(n)
    entries = np.zeros(n)
    wn0 = np.zeros(n)                 # proposed new exposure
    wh = np.zeros(n)                  # already on, immovable
    # holdings first, then candidates, so a candidate's fresher stop and
    # volatility win for a ticker that appears as both
    for u in held_u:
        i = slot[u["ticker"]]
        wh[i] += u["pos0"]
        vols[i], stops[i], entries[i] = u["vol"], u["stop_pct"], u["entry"]
    dup_cands = set()
    for u in new_u:
        i = slot[u["ticker"]]
        if wn0[i] > 0:
            dup_cands.add(u["ticker"])
        wn0[i] += u["pos0"]
        vols[i], stops[i], entries[i] = u["vol"], u["stop_pct"], u["entry"]
    if dup_cands:
        out["warnings"].append(
            f"the same candidate appeared more than once and its sizes were "
            f"summed, not traded twice: {', '.join(sorted(dup_cands))}")
    adding_to = [t for t in tickers if wh[slot[t]] > 0 and wn0[slot[t]] > 0]
    if adding_to:
        out["warnings"].append(
            f"adding to an existing position in {', '.join(adding_to)}; the "
            f"per-name ceiling applies to the combined size")

    R, cinfo = correlation_matrix(tickers, as_of=as_of, price_cache=price_cache,
                                 window=CORR_WINDOW)
    if cinfo.get("assumed"):
        out["warnings"].append(
            f"correlation assumed for {', '.join(cinfo['assumed'])} at "
            f"{cinfo.get('fallback_used')}: too little overlapping history")
    if cinfo.get("psd_repaired"):
        out["warnings"].append("correlation matrix required eigenvalue repair")

    groups = clusters(R, tickers, cluster_corr)
    idx = slot

    # --- 0. per-name ceiling on the COMBINED size -----------------------------
    # The pipeline caps each new position on its own. Only here is the existing
    # holding visible, so only here can the ceiling be applied to the total.
    wn = wn0.copy()
    name_scale = 1.0
    name_capped = []
    if max_name_pct and max_name_pct > 0:
        for i, t in enumerate(tickers):
            if wh[i] + wn[i] > max_name_pct and wn[i] > 0:
                room = max(0.0, max_name_pct - wh[i])
                s_i = room / wn[i]
                wn[i] *= s_i
                name_scale = min(name_scale, s_i)
                name_capped.append(t)
        if name_capped:
            out["warnings"].append(
                f"combined size cut to the {max_name_pct:.0f}% per-name ceiling "
                f"for {', '.join(name_capped)}")
    after_name = float(wn.sum())

    # --- 1. cluster ceiling ---------------------------------------------------
    # A cluster's ceiling applies to the WHOLE cluster, holdings included, but
    # only the new names in it can be scaled. A cluster already at its ceiling
    # from holdings alone admits no new exposure at all.
    cluster_report = []
    for g in groups:
        ii = [idx[t] for t in g]
        g_held = float(wh[ii].sum())
        g_new = float(wn[ii].sum())
        s = 1.0
        if max_cluster_pct > 0 and g_held + g_new > max_cluster_pct:
            room = max(0.0, max_cluster_pct - g_held)
            s = (room / g_new) if g_new > 0 else 1.0
            wn[ii] *= s
        cluster_report.append({
            "members": g, "n_members": len(g),
            "held_pct": round(g_held, 1),
            "gross_pct_before": round(g_held + g_new, 1),
            "scale": round(s, 4),
            "gross_pct_after": round(float(wh[ii].sum() + wn[ii].sum()), 1)})
    after_cluster = float(wn.sum())

    # --- 2/3/4. global constraints -------------------------------------------
    # Each scale is solved against the HEADROOM left by the holdings.
    gross_h, gross_n = float(wh.sum()), float(wn.sum())
    risk_h = float(np.sum(wh * stops / 100.0))
    risk_n = float(np.sum(wn * stops / 100.0))

    def headroom(limit, used, proposed):
        if limit is None or limit <= 0 or proposed <= 0:
            return 1.0
        return float(np.clip((limit - used) / proposed, 0.0, 1.0))

    s_gross = headroom(max_gross_pct, gross_h, gross_n)
    s_risk = headroom(max_risk_pct, risk_h, risk_n)

    target, target_src = dd_ceiling(calib)
    s_prob, reachable = _scale_for_prob(wn, wh, vols, R, calib, target)
    if not reachable:
        pf, _, _, _ = curve_bounds(calib)
        out["warnings"].append(
            f"book drawdown ceiling {target:.1%} is below the curve's own "
            f"minimum {pf:.1%}; the constraint cannot be met by scaling and was "
            f"skipped - raise the ceiling or recalibrate")

    limit_scales = {"gross": s_gross, "risk_at_stops": s_risk,
                    "book_drawdown": s_prob}
    s = float(min(limit_scales.values()))
    wn = wn * s

    # WHICH CONSTRAINT ACTUALLY SHRANK THE BOOK.
    #
    # The three limits above produce headroom fractions on the whole book. The
    # two earlier steps - the per-name ceiling and the cluster ceiling - cut
    # individual names, so their per-name fractions are not comparable with the
    # global ones. Comparing them directly produced a report that read "bound by
    # name cap 0.250" on a book where the name cap had touched two names and the
    # risk budget had cut the other four: the smallest number won the label
    # without being the reason.
    #
    # So each stage is expressed as the fraction of PROPOSED BOOK EXPOSURE it
    # left behind. Those are comparable, they multiply to the total, and the
    # largest reduction is the honest answer.
    prop = float(wn0.sum())
    e_name = (float(after_name) / prop) if prop > 0 else 1.0
    e_cluster = (float(after_cluster) / after_name) if after_name > 0 else 1.0
    stage_effect = {"name_cap": e_name, "cluster": e_cluster, "global": s}
    tightest_limit = min(limit_scales, key=limit_scales.get)
    worst_stage = min(stage_effect, key=stage_effect.get)
    if min(stage_effect.values()) >= 1.0 - 1e-9:
        binding = None
    elif worst_stage == "global":
        binding = tightest_limit
    else:
        binding = worst_stage

    total_scale = float(wn.sum() / prop) if prop > 0 else 1.0
    if total_scale < SCALE_FLOOR:
        out["warnings"].append(
            f"new exposure scaled to {total_scale:.1%} of what was proposed - "
            f"more candidates than the "
            f"{binding.replace('_', ' ') if binding else 'budget'} limit can "
            f"carry. Rank them and screen fewer, rather than trading every name "
            f"at a size that cannot matter.")

    # --- drop dust, then report. No re-inflation: raising the survivors to use
    # the freed budget would push the book back toward the constraint it just
    # cleared, one name at a time, with no rule saying where to stop.
    dropped = []
    for i in range(n):
        if 0 < wn[i] < min_position_pct:
            dropped.append({"ticker": tickers[i],
                            "scaled_pct": round(float(wn[i]), 2),
                            "reason": f"below the {min_position_pct:.1f}% "
                                      f"minimum after portfolio scaling"})
            wn[i] = 0.0

    shares = np.zeros(n)
    for i in range(n):
        if wn[i] > 0 and entries[i] > 0:
            shares[i] = float(int(equity * wn[i] / 100.0 / entries[i]))
    # Integerising can only round DOWN, so every constraint stays satisfied.
    # Report the realised weights, not the pre-rounding intent.
    wn_real = np.zeros(n)
    for i in range(n):
        wn_real[i] = (shares[i] * entries[i] / equity * 100.0
                      if equity > 0 else 0.0)
        if wn[i] > 0 and shares[i] < 1:
            dropped.append({"ticker": tickers[i],
                            "scaled_pct": round(float(wn[i]), 2),
                            "reason": "rounds to zero shares at this equity"})
            wn_real[i] = 0.0

    w_book = wh + wn_real
    rc = risk_contributions(w_book, vols, R)
    bvol = portfolio_vol(w_book, vols, R)
    dr = diversification_ratio(w_book, vols, R)
    p_book, p_kind = book_drawdown_prob(bvol, calib)
    if np.isfinite(bvol) and p_kind in ("upper", "lower"):
        _, _, fmin, fmax = curve_bounds(calib)
        side = "below" if p_kind == "upper" else "above"
        out["warnings"].append(
            f"book volatility {bvol:.1f}% is {side} the curve's fitted range "
            f"{fmin:.1f}-{fmax:.1f}%, so {p_book:.1%} is "
            f"{'an upper' if p_kind == 'upper' else 'a lower'} bound, not an "
            f"estimate; the cluster and risk-at-stops limits are what bind here")

    for i, t in enumerate(tickers):
        out["positions"].append({
            "ticker": t,
            "existing": bool(wh[i] > 0),        # some of this was already on
            "proposed": bool(wn0[i] > 0),       # the screen wanted to buy it
            "existing_pct": round(float(wh[i]), 2),
            "forecast_vol_annual_pct": round(float(vols[i]), 1),
            "stop_pct": round(float(stops[i]), 2),
            # what the book would have been without this layer, for this name
            "position_pct_before": round(float(wh[i] + wn0[i]), 1),
            "new_pct": round(float(wn_real[i]), 2),
            "position_pct": round(float(w_book[i]), 2),
            "scaled_by": (round(float(wn_real[i] / wn0[i]), 3)
                          if wn0[i] > 0 else None),
            "effective_risk_pct": round(float(w_book[i] * stops[i] / 100.0), 3),
            "shares_to_buy": int(shares[i]),
            "position_value": round(equity * w_book[i] / 100.0, 2),
            "risk_contribution_pct": (round(float(rc[i]) * 100, 1)
                                      if np.isfinite(rc[i]) else None),
            "held": bool(w_book[i] > 0),
        })
    out["positions"].sort(key=lambda q: (not q["proposed"], -q["position_pct"]))

    out["dropped"] = dropped
    out["clusters"] = cluster_report
    out["correlation"] = {
        "window_bars": cinfo["n_obs"], "from": cinfo["start"], "to": cinfo["end"],
        "mean_pairwise": cinfo["mean_corr"], "shrinkage": cinfo["shrinkage"],
        "assumed_for": cinfo["assumed"], "threshold": cluster_corr,
    }
    out["portfolio"] = {
        "n_held": int((w_book > 0).sum()),
        "n_new": int((wn_real > 0).sum()),
        "gross_pct_existing": round(gross_h, 1),
        "gross_pct_before": round(float(gross_h + wn0.sum()), 1),
        "gross_pct": round(float(w_book.sum()), 1),
        "risk_at_stops_pct_before":
            round(float(np.sum((wh + wn0) * stops / 100.0)), 2),
        "risk_at_stops_pct": round(float(np.sum(w_book * stops / 100.0)), 2),
        "forecast_vol_annual_pct": round(bvol, 1) if np.isfinite(bvol) else None,
        "sum_of_parts_vol_pct": round(float(np.sum(w_book / 100.0 * vols)), 1),
        "diversification_ratio": round(dr, 3) if np.isfinite(dr) else None,
        "prob_book_drawdown": round(p_book, 3) if np.isfinite(p_book) else None,
        "prob_bound": p_kind,          # point | upper | lower
        "prob_in_fitted_range": p_kind == "point",
        "prob_key": (f"prob_book_falls_{int(P2.DRAWDOWN * 100)}pct_"
                     f"{P2.HORIZON_DAYS}d"),
        "ceiling": round(target, 3) if np.isfinite(target) else None,
        "ceiling_source": target_src,
        "scale_applied": round(total_scale, 3),
        "global_scale": round(s, 3),
        "binding_constraint": binding,
        # what fraction of proposed book exposure each STAGE left behind
        "stage_effect": {k: round(float(v), 3) for k, v in stage_effect.items()},
        # headroom fraction each global LIMIT allowed, for diagnostics
        "limit_scales": {k: round(float(v), 3) for k, v in limit_scales.items()},
        "name_capped": name_capped,
        "limits": {"gross_pct": max_gross_pct, "risk_at_stops_pct": max_risk_pct,
                   "cluster_pct": max_cluster_pct, "name_pct": max_name_pct},
    }
    return out


# =============================================================================
# REPORT
# =============================================================================
def describe(book, out=sys.stdout):
    def w(*a):
        print(*a, file=out)

    p = book.get("portfolio")
    if not p:
        w("  no book: " + "; ".join(book.get("warnings") or ["no candidates"]))
        return

    c = book["correlation"]
    w(f"\n{'=' * 64}\n  PORTFOLIO\n{'=' * 64}")
    w(f"  correlation: {c['window_bars']} bars "
      f"({c['from']} to {c['to']}), mean pairwise {c['mean_pairwise']}, "
      f"shrinkage {c['shrinkage']}")

    multi = [cl for cl in book["clusters"] if cl["n_members"] > 1]
    if multi:
        w(f"  clusters at rho >= {c['threshold']}:")
        for cl in multi:
            tail = (f" -> {cl['gross_pct_after']}%" if cl["scale"] < 1 else "")
            hp = f", {cl['held_pct']}% already on" if cl["held_pct"] > 0 else ""
            w(f"    {' + '.join(cl['members'])}: "
              f"{cl['gross_pct_before']}% gross{hp}{tail}")
    else:
        w(f"  no clusters at rho >= {c['threshold']}")

    shown = [q for q in book["positions"] if q["held"]]
    w(f"\n  {'ticker':<8}{'vol%':>7}{'on':>7}{'wanted':>8}{'final':>8}"
      f"{'risk%':>8}{'of book':>9}{'buy':>7}")
    for q in shown:
        rc = q["risk_contribution_pct"]
        on = f"{q['existing_pct']:.1f}" if q["existing"] else "-"
        want_v = q["position_pct_before"] - q["existing_pct"]
        want = f"{want_v:.1f}" if q["proposed"] else "-"
        share = str(q["shares_to_buy"]) if q["proposed"] else "-"
        contrib = f"{rc:.1f}" if rc is not None else "n/a"
        w(f"  {q['ticker']:<8}{q['forecast_vol_annual_pct']:>7.1f}{on:>7}"
          f"{want:>8}{q['position_pct']:>8.2f}{q['effective_risk_pct']:>8.2f}"
          f"{contrib:>9}{share:>7}")
    for q in book.get("dropped", []):
        w(f"  {q['ticker']:<8}dropped: {q['reason']}")
    for q in book.get("excluded", []):
        w(f"  {str(q['ticker'] or '?'):<8}excluded: {q['reason']}")

    if p["gross_pct_existing"] > 0:
        w(f"\n  {p['gross_pct_existing']:.1f}% of equity was already on and was "
          f"not resized; the limits below are what is left over")
    w(f"\n  gross      {p['gross_pct_before']:.1f}% -> {p['gross_pct']:.1f}% "
      f"(limit {p['limits']['gross_pct']:.0f}%)")
    w(f"  risk at stops  {p['risk_at_stops_pct_before']:.2f}% -> "
      f"{p['risk_at_stops_pct']:.2f}% (limit "
      f"{p['limits']['risk_at_stops_pct']:.1f}%)")
    w(f"  book vol   {p['forecast_vol_annual_pct']}% vs "
      f"{p['sum_of_parts_vol_pct']}% if perfectly correlated "
      f"(diversification {p['diversification_ratio']})")
    if p["prob_book_drawdown"] is not None:
        rel = {"point": "=", "upper": "<=", "lower": ">="}.get(p["prob_bound"], "=")
        w(f"  P({P2.DRAWDOWN:.0%} book fall in {P2.HORIZON_DAYS}d) {rel} "
          f"{p['prob_book_drawdown']:.1%}  ceiling {p['ceiling']:.1%} "
          f"({p['ceiling_source']})")
        if p["prob_bound"] == "upper":
            w("    book volatility is below the curve's fitted range, so this is"
              " a bound - a backstop, not the binding limit here")
        n_names = p.get("n_held") or 0
        if n_names > 1:
            w("    single-name curve read on a book: V60 put the prediction "
              "inside the bracket of realised outcomes, at roughly twice the "
              "per-name model's error - indicative, not calibrated")
        else:
            w("    one name, so the curve is being read on the object it was "
              "fitted on")
    if p["binding_constraint"]:
        w(f"  scaled {p['scale_applied']:.3f}x overall, bound by "
          f"{p['binding_constraint'].replace('_', ' ')}")
        w("    of what was proposed, each stage left: " + ", ".join(
            f"{k.replace('_', ' ')} {v:.3f}"
            for k, v in sorted(p["stage_effect"].items(), key=lambda x: x[1])))
        w("    global limit headroom: " + ", ".join(
            f"{k.replace('_', ' ')} {v:.3f}"
            for k, v in sorted(p["limit_scales"].items(), key=lambda x: x[1])))
        if p.get("name_capped"):
            w(f"    per-name ceiling cut {', '.join(p['name_capped'])}")
    else:
        w("  no constraint binding; positions unchanged")
    for m in book.get("warnings", []):
        w(f"  WARNING: {m}")


# =============================================================================
# SELF-TEST
# =============================================================================
def _fake_calib():
    """A plausible V59-shaped calibration, for testing arithmetic only."""
    fv = np.linspace(22.0, 95.0, 12)
    pp = np.linspace(0.032, 0.414, 12)
    return {"shrinkage": {"features": P2.VOL_FEATURES, "intercept": 7.35,
                          "coef": [0.162, 0.146, 0.448]},
            "curve": [{"fvol": float(a), "p": float(b), "n": 13640}
                      for a, b in zip(fv, pp)],
            "n_observations": 163702, "base_rate": 0.132,
            "horizon_days": P2.HORIZON_DAYS, "drawdown": P2.DRAWDOWN}


def _fake_cache(dirname, tickers, rho=0.75, vol=0.032, n=520, seed=7):
    """Correlated GBM paths, so the measured matrix has a known target."""
    os.makedirs(dirname, exist_ok=True)
    rs = np.random.RandomState(seed)
    k = len(tickers)
    common = rs.normal(size=n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    for j, t in enumerate(tickers):
        z = np.sqrt(rho) * common + np.sqrt(1 - rho) * rs.normal(size=n)
        r = vol * z
        close = 100.0 * np.exp(np.cumsum(r))
        hi = close * (1 + np.abs(rs.normal(scale=0.004, size=n)))
        lo = close * (1 - np.abs(rs.normal(scale=0.004, size=n)))
        op = np.concatenate([[100.0], close[:-1]]) * (
            1 + rs.normal(scale=0.003, size=n))
        pd.DataFrame({"Open": op, "High": np.maximum(hi, np.maximum(op, close)),
                      "Low": np.minimum(lo, np.minimum(op, close)),
                      "Close": close, "Volume": 5e6},
                     index=idx).to_pickle(os.path.join(dirname, f"{t}.pkl"))


def selftest():
    calib = _fake_calib()
    cache = "/tmp/_portfolio_selftest_cache"
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    _fake_cache(cache, names, rho=0.75)
    fails = []

    def check(ok, label, detail=""):
        print(f"  [{'  ok  ' if ok else ' FAIL ':^6}] {label}"
              + (f"\n            {detail}" if detail else ""))
        if not ok:
            fails.append(label)

    R, info = correlation_matrix(names, price_cache=cache)
    check(R.shape == (6, 6) and np.allclose(np.diag(R), 1.0),
          "correlation matrix is square with a unit diagonal")
    check(np.allclose(R, R.T), "correlation matrix is symmetric")
    check(np.linalg.eigvalsh(R).min() > -1e-10,
          "correlation matrix is positive semi-definite",
          f"smallest eigenvalue {np.linalg.eigvalsh(R).min():.2e}")
    check(0.60 <= info["mean_corr"] <= 0.88,
          "measured correlation recovers the simulated 0.75",
          f"mean pairwise {info['mean_corr']}")

    g = clusters(R, names, 0.70)
    check(len(g) == 1 and len(g[0]) == 6,
          "six names at rho 0.75 form one cluster", f"{g}")
    check(R.shape[0] == 6 and all(len(x) >= 1 for x in g), "clustering is total")
    Ri = np.eye(6)
    check(len(clusters(Ri, names, 0.70)) == 6,
          "independent names form no cluster")

    # an unconstrained book, to see the ordering of the three scales
    cands = [{"ticker": t, "position_pct_of_equity": 20.0, "stop_pct": 10.0,
              "entry": 100.0, "forecast_vol_annual_pct": 50.0} for t in names]
    bk = build_book(cands, calib=calib, equity=100000.0, price_cache=cache)
    p = bk["portfolio"]
    check(p["gross_pct"] <= MAX_GROSS_PCT + 1e-6,
          "gross exposure respected", f"{p['gross_pct']}%")
    check(p["risk_at_stops_pct"] <= MAX_PORTFOLIO_RISK_PCT + 1e-6,
          "risk at stops respected",
          f"{p['risk_at_stops_pct']}% vs limit {MAX_PORTFOLIO_RISK_PCT}%")
    for cl in bk["clusters"]:
        if len(cl["members"]) > 1:
            check(cl["gross_pct_after"] <= MAX_CLUSTER_PCT + 1e-6,
                  "cluster ceiling respected", f"{cl['gross_pct_after']}%")
    check(0 < p["diversification_ratio"] <= 1.0 + 1e-9,
          "diversification ratio in (0, 1]", f"{p['diversification_ratio']}")
    check(p["forecast_vol_annual_pct"] <= p["sum_of_parts_vol_pct"] + 1e-6,
          "book volatility never exceeds the fully correlated sum")
    check(p["prob_book_drawdown"] <= p["ceiling"] + 1e-6,
          "book drawdown ceiling respected",
          f"{p['prob_book_drawdown']:.1%} vs {p['ceiling']:.1%}")

    rcs = [q["risk_contribution_pct"] for q in bk["positions"]
           if q["held"] and q["risk_contribution_pct"] is not None]
    check(abs(sum(rcs) - 100.0) < 0.6,
          "risk contributions sum to 100%", f"{sum(rcs):.1f}%")

    # THE REPORT MUST AGREE WITH THE ARITHMETIC. The first version printed "no
    # constraint binding; positions unchanged" on this exact book, because the
    # cluster cap had already brought it inside the other three limits and only
    # those three were being reported. Every silent wrong number in this project
    # has come from a line of output disagreeing with the computation behind it,
    # so the agreement is now a test.
    moved = any(abs(q["position_pct"] - q["position_pct_before"]) > 0.05
                for q in bk["positions"])
    check(moved == (p["binding_constraint"] is not None),
          "a named binding constraint whenever positions actually moved",
          f"moved={moved}, binding={p['binding_constraint']}, "
          f"scale={p['scale_applied']}")
    check(p["scale_applied"] < 0.99 and abs(
        p["scale_applied"] - sum(q["position_pct"] for q in bk["positions"])
        / sum(q["position_pct_before"] for q in bk["positions"])) < 0.02,
          "the reported overall scale matches the positions actually produced",
          f"reported {p['scale_applied']}")
    check(p["prob_bound"] in ("point", "upper", "lower"),
          "the book probability says whether it is an estimate or a bound",
          f"{p['prob_bound']}")

    # a book that already fits must not be touched
    small = [{"ticker": "AAA", "position_pct_of_equity": 5.0, "stop_pct": 8.0,
              "entry": 100.0, "forecast_vol_annual_pct": 30.0}]
    bs = build_book(small, calib=calib, equity=100000.0, price_cache=cache)
    check(bs["portfolio"]["scale_applied"] == 1.0
          and bs["portfolio"]["binding_constraint"] is None,
          "a book inside every limit is left alone")

    # A realistic single-sector book: large-cap technology pairwise correlation
    # runs nearer 0.5 than 0.75, which is BELOW the clustering threshold. The
    # cluster cap then does nothing and the risk budget has to do the work. If
    # that is not what happens, the layer only functions in the extreme case.
    cache3 = "/tmp/_portfolio_selftest_cache_mid"
    _fake_cache(cache3, names, rho=0.50, seed=23)
    bm = build_book(cands, calib=calib, equity=100000.0, price_cache=cache3)
    pm = bm["portfolio"]
    check(not any(len(cl["members"]) > 1 for cl in bm["clusters"]),
          "at rho 0.50 nothing clusters, as intended",
          f"{[cl['members'] for cl in bm['clusters']]}")
    check(pm["binding_constraint"] == "risk_at_stops",
          "on an unclustered book the risk budget is what binds",
          f"bound by {pm['binding_constraint']}, "
          f"stage effects {pm['stage_effect']}")
    check(pm["risk_at_stops_pct"] <= MAX_PORTFOLIO_RISK_PCT + 1e-6
          and pm["gross_pct"] <= MAX_GROSS_PCT + 1e-6,
          "the unclustered book still satisfies every limit",
          f"gross {pm['gross_pct']}%, risk {pm['risk_at_stops_pct']}%")

    # independence must produce a lower book vol than correlation does
    cache2 = "/tmp/_portfolio_selftest_cache_indep"
    _fake_cache(cache2, names, rho=0.02, seed=11)
    bi = build_book(cands, calib=calib, equity=100000.0, price_cache=cache2)
    check(bi["portfolio"]["diversification_ratio"]
          < bk["portfolio"]["diversification_ratio"],
          "uncorrelated names diversify more than correlated ones",
          f"{bi['portfolio']['diversification_ratio']} vs "
          f"{bk['portfolio']['diversification_ratio']}")
    check(bi["portfolio"]["gross_pct"] >= bk["portfolio"]["gross_pct"] - 1e-6,
          "an uncorrelated book is allowed at least as much exposure",
          f"{bi['portfolio']['gross_pct']}% vs {bk['portfolio']['gross_pct']}%")

    # --- existing holdings ----------------------------------------------------
    # The layer must treat a position already on as spent budget. Without this
    # a screen run against a full book authorises a second full allocation.
    hold = [{"ticker": "EEE", "position_pct_of_equity": 15.0, "stop_pct": 10.0,
             "entry": 100.0, "forecast_vol_annual_pct": 50.0}]
    bh = build_book(cands[:4], calib=calib, equity=100000.0, price_cache=cache3,
                    held=hold)
    ph = bh["portfolio"]
    new_h = sum(q["new_pct"] for q in bh["positions"])
    b0 = build_book(cands[:4], calib=calib, equity=100000.0, price_cache=cache3)
    new_0 = sum(q["new_pct"] for q in b0["positions"])
    check(new_h < new_0 - 0.5,
          "an existing holding leaves less room for new positions",
          f"{new_h:.1f}% new with a holding on vs {new_0:.1f}% without")
    check(ph["risk_at_stops_pct"] <= MAX_PORTFOLIO_RISK_PCT + 1e-6,
          "the risk budget counts the holding too",
          f"{ph['risk_at_stops_pct']}% total including the 1.50% already on")
    check(any(q["existing"] and not q["proposed"] and q["shares_to_buy"] == 0
              and q["position_pct"] > 0 for q in bh["positions"]),
          "a holding is carried at its size with nothing to buy")

    # THE DUPLICATE-TICKER CASE. Holding a name the screen also wants is the
    # ordinary "add to a winner" situation, and the first version gave it two
    # rows: correlation collapsed from 0.57 to 0.31 because the second row stayed
    # at the identity, and the two sleeves were sized independently so a 15%
    # holding plus a 20% buy passed every check as 35% of one name.
    bd = build_book(cands[:4], calib=calib, equity=100000.0, price_cache=cache3,
                    held=[{"ticker": "AAA", "position_pct_of_equity": 15.0,
                           "stop_pct": 10.0, "entry": 100.0,
                           "forecast_vol_annual_pct": 50.0}])
    rows_aaa = [q for q in bd["positions"] if q["ticker"] == "AAA"]
    check(len(rows_aaa) == 1,
          "a ticker held and proposed occupies one row, not two",
          f"{len(rows_aaa)} row(s) for AAA")
    check(rows_aaa[0]["position_pct"]
          <= getattr(P2, "MAX_POSITION_PCT", 20.0) + 1e-6,
          "the per-name ceiling applies to holding plus addition",
          f"AAA lands at {rows_aaa[0]['position_pct']}% "
          f"({rows_aaa[0]['existing_pct']}% already on)")
    check(abs(bd["correlation"]["mean_pairwise"]
              - b0["correlation"]["mean_pairwise"]) < 0.08,
          "adding a duplicate holding does not move measured correlation",
          f"{bd['correlation']['mean_pairwise']} with the holding vs "
          f"{b0['correlation']['mean_pairwise']} without")
    check(any("per-name ceiling applies to the combined size" in m
              for m in bd["warnings"]),
          "and the pipeline is told it is adding to an existing position")

    # holdings that already fill the budget must yield zero new exposure, not a
    # floored-but-tradeable position
    full = [{"ticker": t, "position_pct_of_equity": 20.0, "stop_pct": 10.0,
             "entry": 100.0, "forecast_vol_annual_pct": 50.0}
            for t in ["DDD", "EEE", "FFF"]]
    bf = build_book(cands[:3], calib=calib, equity=100000.0, price_cache=cache3,
                    held=full)
    newf = sum(q["new_pct"] for q in bf["positions"])
    check(newf == 0.0 and bf["portfolio"]["scale_applied"] == 0.0,
          "a book already at its risk limit admits no new exposure",
          f"{newf:.2f}% new, scale {bf['portfolio']['scale_applied']}")
    check(bf["portfolio"]["risk_at_stops_pct"] == 6.0,
          "and the existing risk is still reported honestly",
          f"{bf['portfolio']['risk_at_stops_pct']}%")

    # a cluster filled by holdings admits no new members
    bc = build_book([cands[0]], calib=calib, equity=100000.0, price_cache=cache,
                    held=[{"ticker": "BBB", "position_pct_of_equity": 40.0,
                           "stop_pct": 10.0, "entry": 100.0,
                           "forecast_vol_annual_pct": 50.0}])
    newc = sum(q["new_pct"] for q in bc["positions"])
    check(newc == 0.0,
          "a correlation cluster already at its ceiling admits no new name",
          f"{newc:.2f}% new alongside a 40% holding in the same cluster")

    # an unmeasurable holding must not be quietly ignored
    bu = build_book(cands[:3], calib=calib, equity=100000.0, price_cache=cache3,
                    held=[{"ticker": "GHOST", "position_pct_of_equity": 30.0,
                           "stop_pct": 10.0, "entry": 100.0}])
    check(any("NOT counted against the limits" in m for m in bu["warnings"]),
          "an unmeasurable holding is flagged as understating the limits")

    # an unreachable ceiling must be reported, not answered by emptying the book
    tight = dict(calib)
    tight["base_rate"] = 0.001
    bt = build_book(cands, calib=tight, equity=100000.0, price_cache=cache)
    check(any("cannot be met by scaling" in m for m in bt["warnings"])
          and bt["portfolio"]["gross_pct"] > 0,
          "an unreachable drawdown ceiling warns instead of zeroing the book")

    # malformed candidates are named, not silently dropped
    bad = cands[:2] + [{"ticker": "ZZZ", "position_pct_of_equity": 10.0,
                        "stop_pct": 9.0, "entry": 50.0}]
    bb = build_book(bad, calib=calib, equity=100000.0, price_cache=cache)
    check([e["ticker"] for e in bb["excluded"]] == ["ZZZ"]
          and "volatility" in bb["excluded"][0]["reason"],
          "a candidate with no volatility forecast is excluded by name")

    # a name with no price history gets an assumed correlation, not zero
    bn = build_book(cands[:3] + [{"ticker": "NOPE",
                                  "position_pct_of_equity": 20.0,
                                  "stop_pct": 10.0, "entry": 100.0,
                                  "forecast_vol_annual_pct": 50.0}],
                    calib=calib, equity=100000.0, price_cache=cache)
    check("NOPE" in (bn["correlation"]["assumed_for"] or [])
          and any("correlation assumed" in m for m in bn["warnings"]),
          "an unmeasurable name is assumed correlated, and says so")

    describe(bk)
    print(f"\n  {len(fails)} failed")
    return 1 if fails else 0


def run_book(tickers, held=None, equity=DEFAULT_EQUITY,
             risk_budget=DEFAULT_RISK_BUDGET, as_of=None, calib=None,
             price_cache=P2.PRICE_CACHE,
             max_gross_pct=MAX_GROSS_PCT,
             max_risk_pct=MAX_PORTFOLIO_RISK_PCT,
             max_cluster_pct=MAX_CLUSTER_PCT,
             max_name_pct=None,
             cluster_corr=CLUSTER_CORR,
             min_position_pct=MIN_POSITION_PCT,
             verbose=True):
    """
    Tickers in, a constrained book out. One call for the whole layer.

        book = run_book(["NVDA", "AMD", "AVGO"], equity=25000)
        book = run_book(["NVDA", "AMD"], held=["AVGO:12", "MU:8"], equity=25000)
        book = run_book([...], max_risk_pct=4.0)      # tighter than the default

    `held` takes either "TICKER:PCT" strings or already-built candidate dicts.
    Every limit is a parameter, so a sweep over the 6% risk budget - the one
    number in this file that is convention rather than measurement - is a loop
    rather than an edit.

    Returns the same dict build_book() returns. Raises RuntimeError if the
    calibration is missing.
    """
    if calib is None:
        if not os.path.exists(P2.CALIB_FILE):
            raise RuntimeError(f"{P2.CALIB_FILE} not found. Run: "
                               f"python {P2.__name__}.py --calibrate")
        calib = P2.load_calibration()

    cands, failed = assess_candidates(tickers, calib=calib, equity=equity,
                                      risk_budget=risk_budget, as_of=as_of,
                                      price_cache=price_cache)
    if verbose:
        for t in failed:
            print(f"  {t}: no assessment", file=sys.stderr)

    held_rows, held_bad = [], []
    if held:
        specs = [h for h in held if isinstance(h, str)]
        held_rows = [h for h in held if not isinstance(h, str)]
        if specs:
            parsed, held_bad = parse_holdings(specs, calib=calib, equity=equity,
                                              as_of=as_of,
                                              price_cache=price_cache)
            held_rows += parsed
        if verbose:
            for b in held_bad:
                print(f"  holding {b['spec']}: {b['reason']}", file=sys.stderr)

    return build_book(cands, calib=calib, equity=equity, as_of=as_of,
                      price_cache=price_cache, held=held_rows,
                      max_gross_pct=max_gross_pct, max_risk_pct=max_risk_pct,
                      max_cluster_pct=max_cluster_pct, max_name_pct=max_name_pct,
                      cluster_corr=cluster_corr,
                      min_position_pct=min_position_pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--held", nargs="*", metavar="TICKER:PCT",
                    help="positions already on, as a percent of equity, e.g. "
                         "NVDA:12 AVGO:8. They consume the limits and are never "
                         "resized. Leaving them out measures an empty book.")
    ap.add_argument("--equity", type=float, default=10000.0)
    ap.add_argument("--date", default=None)
    ap.add_argument("--risk", type=float, default=0.02)
    ap.add_argument("--json", action="store_true")
    cfg = ap.parse_args()

    if cfg.selftest:
        sys.exit(selftest())

    if not cfg.tickers:
        sys.exit("Pass --tickers, or --selftest.")
    try:
        book = run_book(cfg.tickers, held=cfg.held, equity=cfg.equity,
                        risk_budget=cfg.risk, as_of=cfg.date)
    except RuntimeError as e:
        sys.exit(str(e))
    if cfg.json:
        print(json.dumps(book, indent=2))
    else:
        describe(book)


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Set RUN_MODE, edit, run.
#
#   "book"     build a constrained portfolio from candidates + what you hold
#   "selftest" 37 internal checks on the layer itself, no price data needed
#              beyond the cache. Run this after changing any limit.
#
# Passing any command-line flag still works and takes over.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODE = "book"                    # "book" | "selftest"

    RUN_TICKERS = ["NVDA", "AMD", "AVGO"]   # candidates you are considering
    RUN_HELD    = []          # already on, as % of equity: ["AVGO:12", "MU:8"].
                              # These consume the limits and are never resized.
    RUN_EQUITY  = DEFAULT_EQUITY
    RUN_RISK    = DEFAULT_RISK_BUDGET    # risk budget per new name
    RUN_DATE    = None        # "2026-06-30" builds the book as of a past date
    RUN_AS_JSON = False       # True prints the raw dict instead of the report

    # the four limits. Lower any of them to see what starts binding.
    RUN_MAX_GROSS_PCT   = MAX_GROSS_PCT           # total exposure
    RUN_MAX_RISK_PCT    = MAX_PORTFOLIO_RISK_PCT  # summed risk at stops
    RUN_MAX_CLUSTER_PCT = MAX_CLUSTER_PCT         # one correlated group
    RUN_CLUSTER_CORR    = CLUSTER_CORR            # correlation that makes a group
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                           # e.g. python class_ai_portfolio.py --selftest

    elif RUN_MODE == "selftest":
        sys.exit(selftest())

    elif RUN_MODE == "book":
        try:
            _book = run_book(RUN_TICKERS, held=RUN_HELD, equity=RUN_EQUITY,
                             risk_budget=RUN_RISK, as_of=RUN_DATE,
                             max_gross_pct=RUN_MAX_GROSS_PCT,
                             max_risk_pct=RUN_MAX_RISK_PCT,
                             max_cluster_pct=RUN_MAX_CLUSTER_PCT,
                             cluster_corr=RUN_CLUSTER_CORR)
        except RuntimeError as _e:
            sys.exit(str(_e))
        if RUN_AS_JSON:
            print(json.dumps(_book, indent=2))
        else:
            describe(_book)

    else:
        sys.exit(f'RUN_MODE must be "book" or "selftest", not {RUN_MODE!r}')
