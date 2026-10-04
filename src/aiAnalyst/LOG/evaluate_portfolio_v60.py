#!/usr/bin/env python3
"""
evaluate_portfolio_v60.py - does the single-name curve work on a BOOK?

THE ONE NEW CLAIM
-----------------
class_ai_portfolio.py computes a book's forecast volatility from Pillar 2's
per-name forecasts and a measured correlation matrix, then reads a drawdown
probability off the curve that was fitted on SINGLE NAMES. That step assumes the
volatility-to-drawdown mapping depends on the volatility LEVEL and not on what
is producing it. V54 and V55 found volatility effectively sufficient for the
single-name mapping, which makes the assumption plausible. Plausible is not
tested, and every other claim in this project has been tested, so:

    Build historical books. Compute what the curve predicts. Compare against
    what those books actually did.

WHY THE ANSWER MATTERS EVEN WHEN THE CONSTRAINT NEVER BINDS
-----------------------------------------------------------
A scaled, diversified book usually has a LOWER forecast volatility than the
calmest single name the curve was fitted on, so the curve clamps and reports its
floor. class_ai_portfolio labels that an upper bound rather than an estimate.
Whether the bound is SAFE is an empirical question:

    if realised well below the clamped floor  the backstop is conservative, and
                                              the cluster and risk-at-stops
                                              limits are doing the real work
    if realised at or above the floor         the clamp understates book risk and
                                              the constraint is worse than
                                              decoration - it is reassuring

So the table below deliberately includes buckets BELOW the fitted range. Those
rows are the point of the exercise, not an edge case.

DESIGN
------
  * books are formed on a monthly grid from the cached universe, k names drawn at
    random, equal weight, at a target gross exposure. The remainder is cash.
  * BUY AND HOLD, no rebalancing, because that is how the pipeline trades. A
    rebalanced book has different path properties and would be answering a
    question nobody asked.
  * the book's forecast volatility uses ONLY information up to the formation
    date: Pillar 2's own volatility features, its fitted shrinkage, and a
    trailing correlation window. No outcome touches it.
  * the event is a peak-to-trough fall of DRAWDOWN or more in total equity within
    HORIZON_DAYS of formation, measured on the book's own path.
  * intervals come from V52's half-year block bootstrap, and the design effect is
    measured rather than assumed. Overlapping windows and overlapping name sets
    make row-level resampling meaningless here for the same reason it was
    meaningless in V52.

WHAT WOULD FALSIFY THE ASSUMPTION
---------------------------------
Reliability error at the book level materially worse than the 1.55pp the single
name model shows out of sample, or a systematic sign to the error that survives
the bootstrap. Either would mean composition matters and the book probability
should be dropped from the report rather than labelled.

USAGE
  python evaluate_portfolio_v60.py --universe tech --n-books 4000
  python evaluate_portfolio_v60.py --k 1 3 5 8 --gross 50 100 --n-boot 400
"""
import argparse
import os
import sys
from datetime import date as _date

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2
import class_ai_portfolio as PF
import evaluate_calibration_v52 as V52

STEP_DAYS = 21              # formation dates, roughly monthly
MIN_NAMES = 12              # a universe below this cannot support random books
CORR_WINDOW = PF.CORR_WINDOW


# =============================================================================
# PANEL
# =============================================================================
def build_panel(tickers, price_cache=P2.PRICE_CACHE, verbose=True):
    """
    Closes, lows and per-name forecast volatility, aligned on one date index.

    Lows are carried because P2 measures its event on the forward LOW, not the
    close, and anything compared with the curve has to be measured the way the
    curve was.

    The volatility features are computed ONCE per name over the whole history and
    then read positionally, exactly as P2.build_observations does. Recomputing
    them per formation date would be the same numbers at 300x the cost.
    """
    closes, lows, fvols, skipped = {}, {}, {}, []
    calib = P2.load_calibration()
    s = calib["shrinkage"]
    if "features" not in s:
        return None, None, None, tickers
    for t in tickers:
        df = P2.load_prices(t, price_cache)
        if df is None or len(df) < P2.MIN_BARS + 40 or "Low" not in df:
            skipped.append(t)
            continue
        vp = P2.vol_feature_panel(df)
        X = vp[P2.VOL_FEATURES].to_numpy(float)
        fv = s["intercept"] + X @ np.array(s["coef"], float)
        closes[t] = df["Close"]
        lows[t] = df["Low"]
        fvols[t] = pd.Series(fv, index=df.index)
    if verbose and skipped:
        print(f"  skipped {len(skipped)} names with too little history")
    if not closes:
        return None, None, None, skipped
    C = pd.DataFrame(closes).sort_index()
    L = pd.DataFrame(lows).reindex(C.index)
    F = pd.DataFrame(fvols).reindex(C.index)
    return C, L, F, skipped


def book_falls(Cseg, Lseg, P0, w):
    """
    How far a buy-and-hold book fell below its FORMATION value, bracketed.

    `w` are weights as fractions of equity summing to at most 1; the remainder is
    cash and does not move, so a half-invested book needs a 40% fall in its names
    to lose 20% of equity.

    Returns (close_based, daily_lows) as positive fractions, or None if any price
    is missing - carrying a stale price through a drawdown measurement invents a
    calm patch that never happened.

        close_based   the book's worst CLOSING value. UNDER-counts against P2,
                      which uses intraday lows.
        daily_lows    each DAY's book value with every name at that day's low,
                      then the worst of those days. Still an over-count, because
                      the names' lows are not simultaneous within a day, but a
                      valid upper bound: within day t the book's true minimum is
                      at least cash + sum_i w_i Low_i,t / P_i0.

    WHY NOT THE OBVIOUS BOUND. The first version combined each name's
    PERIOD-minimum low - every name at its worst low of the whole window, all on
    the same day. That is also a valid upper bound but a badly loose one for a
    wide book, and it is loose in a way that scales with breadth: exact at k=1,
    hopeless at k=8. It made the upper edge of the bracket useless for exactly the
    books the pipeline trades, and the bias test then straddled zero because one
    edge was doing no work. Combining within a day first and minimising afterwards
    is strictly tighter -
        min_t sum_i w_i Low_i,t  >=  sum_i w_i min_t Low_i,t
    - and identical at k=1, so the control is unaffected.
    """
    if (not np.isfinite(Cseg).all() or not np.isfinite(Lseg).all()
            or not np.isfinite(P0).all() or (P0 <= 0).any()):
        return None
    cash = float(1.0 - np.sum(w))
    wv = np.asarray(w, float)
    path = cash + (Cseg / P0) @ wv
    low_path = cash + (Lseg / P0) @ wv
    return float(1.0 - np.min(path)), float(1.0 - np.min(low_path))


def verify_against_p2(C, L, horizon_days, price_cache=P2.PRICE_CACHE,
                      n_names=3, stride=37):
    """
    Prove this file measures the same event P2 calibrated on, before measuring
    anything else.

    For one name fully invested there is only one low, so the daily-lows
    fall must equal P2's own
        (min forward low - entry) / entry
    exactly, not approximately. The first version of this file did not: it used a
    running-peak drawdown on closes, which counts every dip below an interim high
    and therefore finds strictly more events. The single-name control came back at
    19.5% realised against 13.0% predicted and looked like a finding about books.
    It was a definition mismatch.

    Returns (n_checked, n_mismatch, worst_abs_diff_pp).
    """
    names = [t for t in C.columns][:n_names]
    checked = bad = 0
    worst = 0.0
    mn = P2._min_fwd_bars(horizon_days)
    idxv = C.index.to_numpy()
    for t in names:
        df = P2.load_prices(t, price_cache)
        if df is None or "Low" not in df:
            continue
        for pos in range(P2.MIN_BARS, len(C.index) - mn - 1, stride):
            t0 = C.index[pos]
            end = t0 + pd.Timedelta(days=horizon_days)
            f0 = pos + 1
            f1 = int(np.searchsorted(idxv, np.datetime64(end), "right"))
            if f1 - f0 < mn:
                continue
            entry = float(C[t].iloc[pos])
            if not np.isfinite(entry) or entry <= 0:
                continue
            # P2's arithmetic, on P2's own frame
            fwd = df.iloc[df.index.get_indexer([t0])[0] + 1:]
            fwd = fwd[fwd.index <= end]
            if len(fwd) < mn:
                continue
            p2_mdd = (float(fwd["Low"].min()) - entry) / entry * 100.0
            # this file's arithmetic, on the shared panel
            seg = C[[t]].iloc[f0:f1].to_numpy(float)
            lseg = L[[t]].iloc[f0:f1].to_numpy(float)
            f = book_falls(seg, lseg, np.array([entry]), np.array([1.0]))
            if f is None:
                continue
            checked += 1
            diff = abs((-f[1] * 100.0) - p2_mdd)
            worst = max(worst, diff)
            if diff > 1e-6:
                bad += 1
    return checked, bad, worst


# =============================================================================
# EXPERIMENT
# =============================================================================
def run(C, L, F, ks, grosses, n_books, seed, horizon_days, drawdown,
        verbose=True):
    calib = P2.load_calibration()
    fwd = P2._fwd_bars(horizon_days)
    min_fwd = P2._min_fwd_bars(horizon_days)
    rng = np.random.default_rng(seed)
    names_all = list(C.columns)
    idx = C.index
    start = max(CORR_WINDOW + 5, P2.MIN_BARS)
    positions = list(range(start, len(idx) - min_fwd - 1, STEP_DAYS))
    if not positions:
        return None
    if verbose:
        print(f"  {len(positions)} formation dates, {len(names_all)} names, "
              f"{fwd} forward bars ({min_fwd} minimum)")

    logret = np.log(C).diff()
    Carr, Larr, Farr = (C.to_numpy(float), L.to_numpy(float), F.to_numpy(float))
    col = {t: j for j, t in enumerate(C.columns)}
    rows = []
    per_date = max(1, n_books // (len(positions) * len(ks) * len(grosses)))

    for pos in positions:
        t0 = idx[pos]
        # The forward window is a CALENDAR cutoff excluding the formation bar,
        # exactly as P2.build_observations defines it. A fixed bar count is close
        # but not the same, and "close but not the same" is what the control is
        # there to catch.
        end = t0 + pd.Timedelta(days=horizon_days)
        f0 = pos + 1
        f1 = int(np.searchsorted(idx.to_numpy(), np.datetime64(end), "right"))
        if f1 - f0 < min_fwd:
            continue

        usable = [t for t in names_all
                  if np.isfinite(Farr[pos, col[t]]) and Farr[pos, col[t]] > 0
                  and np.isfinite(Carr[pos, col[t]]) and Carr[pos, col[t]] > 0]
        if len(usable) < MIN_NAMES:
            continue
        win = logret[usable].iloc[pos - CORR_WINDOW + 1:pos + 1].dropna(how="any")
        if len(win) < PF.MIN_CORR_OBS:
            continue
        Rfull = np.nan_to_num(win.corr().to_numpy(float), nan=0.0)
        np.fill_diagonal(Rfull, 1.0)
        slot = {t: i for i, t in enumerate(usable)}
        jj = np.array([col[t] for t in usable])
        seg_all = Carr[f0:f1, jj]
        low_all = Larr[f0:f1, jj]        # the whole segment, not its min
        p0_all = Carr[pos, jj]
        vol_all = Farr[pos, jj]

        for k in ks:
            if k > len(usable):
                continue
            for gross in grosses:
                for _ in range(per_date):
                    ii = rng.choice(len(usable), k, replace=False)
                    w = np.full(k, gross / 100.0 / k)
                    Rk = Rfull[np.ix_(ii, ii)]
                    Rk, _lam, _ = PF._shrink_to_constant(Rk, len(win))
                    Rk, _ = PF._nearest_psd(Rk)
                    sig = vol_all[ii]
                    bvol = PF.portfolio_vol(w * 100.0, sig, Rk)
                    if not np.isfinite(bvol) or bvol <= 0:
                        continue
                    f = book_falls(seg_all[:, ii], low_all[:, ii], p0_all[ii], w)
                    if f is None:
                        continue
                    fall_close, fall_low = f
                    p, kind = PF.book_drawdown_prob(bvol, calib)
                    rows.append({"date": t0, "k": k, "gross": gross,
                                 "book_vol": bvol, "p": p, "bound": kind,
                                 "fall_close": fall_close,
                                 "fall_low_daily": fall_low,
                                 "event": int(fall_close >= drawdown),
                                 "event_low": int(fall_low >= drawdown),
                                 "mean_name_vol": float(sig.mean()),
                                 "mean_corr": float(
                                     Rk[~np.eye(k, dtype=bool)].mean())
                                 if k > 1 else 1.0})
    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================
def vol_buckets(d, calib, n_below=2, n_in=5, n_above=2):
    """
    Buckets over book volatility, split at BOTH ends of the curve's fitted range.

    The boundaries are forced rather than left to quantiles, because outside the
    fitted range the curve clamps and its output changes meaning: below the range
    the prediction is an upper bound, above it a lower bound. A bucket straddling
    a boundary mixes an estimate with a bound and its error is uninterpretable.
    The first version split only at the bottom, so the top row spanned 35% to
    140% volatility and reported +15.4pp - part real top-end error, part the
    lower-bound clamp above 68%, with no way to tell which.

    Inside each region the edges are QUANTILES, not equal widths. Equal widths
    produced 1-percentage-point buckets holding 38 books each, because almost
    every diversified book sits below the fitted range and the handful above it
    spread thinly over a wide span. Forty books cannot measure a 13% event rate.
    """
    fmin = min(float(p["fvol"]) for p in calib["curve"])
    fmax = max(float(p["fvol"]) for p in calib["curve"])
    v = d["book_vol"].to_numpy(float)
    below, inside, above = v[v < fmin], v[(v >= fmin) & (v <= fmax)], v[v > fmax]
    edges = [float(v.min()) - 1e-9]
    if below.size:
        edges += [float(np.quantile(below, q))
                  for q in np.linspace(0, 1, n_below + 1)[1:-1]]
        edges.append(fmin)
    if inside.size:
        edges += [float(np.quantile(inside, q))
                  for q in np.linspace(0, 1, n_in + 1)[1:-1]]
        if above.size:
            edges.append(fmax)
    if above.size:
        edges += [float(np.quantile(above, q))
                  for q in np.linspace(0, 1, n_above + 1)[1:-1]]
    edges = sorted(set(round(e, 4) for e in edges))
    if edges[-1] <= float(v.max()):
        edges.append(float(v.max()) + 1e-6)
    return edges, fmin, fmax


def report(d, calib, drawdown, horizon_days, n_boot, seed,
           ref_mae_pp=1.55):
    print(f"\n{'=' * 84}")
    print(f"  BOOK DRAWDOWN: predicted from the single-name curve vs realised")
    print(f"  event = a {drawdown:.0%} fall in total equity within "
          f"{horizon_days} days of formation")
    print(f"{'=' * 84}")

    eff = V52.effective_n(d["date"].to_numpy(), d["event"].to_numpy(float),
                          n_boot=n_boot, seed=seed)
    # The bracket must not be inverted: the close-based fall is a lower bound on
    # the true fall and the daily-lows fall an upper bound, so close <= lows must
    # hold for every book. It is the one invariant that catches a mix-up between
    # the two paths, and it costs one comparison.
    inv = int((d["fall_close"] > d["fall_low_daily"] + 1e-12).sum())
    if inv:
        print(f"\n  BRACKET INVERTED on {inv:,} of {len(d):,} books - the "
              f"close-based fall exceeded the daily-lows fall, which is "
              f"impossible. The two paths are crossed; fix book_falls before "
              f"reading anything below.")

    print(f"\n  {len(d):,} books | base rate {d['event'].mean():.1%} | "
          f"{eff['n_blocks']} half-year blocks")
    print(f"  design effect {eff['design_effect']:.0f}x -> about "
          f"{eff['n_effective']:.0f} independent observations")
    print(f"  Overlapping windows and shared names, so the honest sample is a few")
    print(f"  hundred, not {len(d):,}. Read the intervals, not the point estimates.")

    edges, fmin, fmax = vol_buckets(d, calib)
    d = d.copy()
    d["bucket"] = pd.cut(d["book_vol"], bins=edges, include_lowest=True)
    print(f"\n  curve fitted over book-equivalent volatility {fmin:.0f}% to "
          f"{fmax:.0f}%; below {fmin:.0f}% the prediction is a CLAMPED UPPER BOUND")
    print(f"\n  {'book vol':>16}{'n':>8}{'predicted':>11}{'realised':>10}"
          f"{'error':>9}{'lows':>8}{'mean k':>8}{'status':>10}")
    for b, g in d.groupby("bucket", observed=True):
        if len(g) < 30:
            continue
        pred = g["p"].mean() * 100
        real = g["event"].mean() * 100
        low = g["event_low"].mean() * 100
        kinds = g["bound"].value_counts()
        kind = kinds.index[0]
        status = {"point": "fitted", "upper": "upper bd",
                  "lower": "lower bd"}.get(kind, kind)
        lab = f"{b.left:.0f}-{b.right:.0f}%"
        print(f"  {lab:>16}{len(g):>8,}{pred:>10.1f}%{real:>9.1f}%"
              f"{real - pred:>+9.1f}{low:>7.1f}%{g['k'].mean():>8.1f}"
              f"{status:>10}")
    print(f"  'realised' is close-based and under-counts; 'lows' takes each day's lows")
    print(f"  together and over-counts. The truth is between the two columns.")

    # --- CONTROL ------------------------------------------------------------
    # One name at 100% gross is not a portfolio: it is exactly the object the
    # curve was fitted on. That row therefore MUST come back calibrated. If it
    # does not, the fault is in this harness - the path construction, the
    # positional feature lookup, the forward window - and nothing below it can be
    # believed. Checking the assumption without checking the instrument first is
    # how V54 spent a run measuring a market state built from three tickers.
    # For k=1 the synchronised-lows measure IS P2's measure - one name has only
    # one low - so event_low is the exact reproduction and event is the
    # close-based version of the same thing. Reporting both sizes the convention
    # gap that the multi-name rows cannot avoid.
    ctl = d[(d["k"] == 1) & (np.isclose(d["gross"], 100.0))]
    print(f"\n  CONTROL: one name, fully invested  ({len(ctl):,} books)")
    if len(ctl) >= 200:
        cm = V52.mae(ctl["p"].to_numpy(float), ctl["event_low"].to_numpy(float))
        _, clo, chi = V52.block_boot(
            ctl["date"].to_numpy(), ctl["p"].to_numpy(float),
            ctl["event_low"].to_numpy(float), V52.mae, n_boot=n_boot, seed=seed)
        print(f"    predicted {ctl['p'].mean():.1%}, realised on lows "
              f"{ctl['event_low'].mean():.1%} (P2's own measure), on closes "
              f"{ctl['event'].mean():.1%}")
        print(f"    calibration error {cm:.2f}pp against P2's measure "
              f"(95% CI {clo:.2f} to {chi:.2f})")
        gap = (ctl["event_low"].mean() - ctl["event"].mean()) * 100
        print(f"    the lows-vs-closes convention is worth {gap:+.1f}pp here, which")
        print(f"    is the size of the under-count in every multi-name row above")
        if cm > 8.0:
            print(f"    -> HARNESS FAULT. This is the fitted object itself and it")
            print(f"       is not calibrated, so the rest of this report measures")
            print(f"       the harness, not the assumption. Check the price cache,")
            print(f"       the calibration's horizon, and the forward window before")
            print(f"       reading anything below.")
        else:
            print(f"    -> the instrument reproduces the curve it was fitted on, "
                  f"so the comparisons below are about composition")
    else:
        print(f"    too few single-name books to verify the harness - pass "
              f"--k 1 ... and --gross 100")

    fit = d[d["bound"] == "point"]
    clamp = d[d["bound"] == "upper"]

    print(f"\n  INSIDE THE FITTED RANGE  ({len(fit):,} books)")
    if len(fit) >= 200:
        # BOTH CONVENTIONS, OR NEITHER.
        #
        # The close-based event is a KNOWN one-sided under-count: the control
        # measures the gap directly and it is of the same order as any bias worth
        # reporting. A bootstrap interval covers sampling error only, so testing
        # the close-based measure alone and printing SYSTEMATIC when the interval
        # clears zero treats a biased estimator's precision as if it settled the
        # question. It does not. The first run of this file did exactly that and
        # announced a systematic over-prediction of 3.67pp that reversed sign the
        # moment the other convention was used.
        #
        # TWO SEPARATE QUESTIONS, AND THE BRACKET ANSWERS THEM DIFFERENTLY.
        #
        # The true fall lies between the close-based and daily-lows measures, so
        # the true bias lies between the two biases computed from them. If both
        # point estimates share a sign, the BRACKET does not contain zero and the
        # direction follows from the bracket alone, whatever either interval does.
        # Sampling uncertainty is then a separate question, and the edge that
        # matters is the one NEAREST zero - the least favourable end of the
        # bracket.
        #
        # An earlier version required both intervals to clear zero and otherwise
        # printed "the two conventions disagree on the sign". On a run where the
        # biases were -3.67pp and -1.85pp that message was simply false: they
        # agreed, and only the second interval failed to clear zero. Conflating
        # "not individually significant" with "disagree" hid a consistent result.
        def _bias(p, y):
            return (y.mean() - p.mean()) * 100

        conv = []
        for lab, col in (("closes (under-counts)", "event"),
                         ("daily lows (over-counts)", "event_low")):
            y = fit[col].to_numpy(float)
            pv = fit["p"].to_numpy(float)
            m = V52.mae(pv, y)
            _, lo, hi = V52.block_boot(fit["date"].to_numpy(), pv, y, V52.mae,
                                       n_boot=n_boot, seed=seed)
            b = _bias(pv, y)
            _, blo, bhi = V52.block_boot(fit["date"].to_numpy(), pv, y, _bias,
                                         n_boot=n_boot, seed=seed)
            conv.append({"lab": lab, "err": m, "elo": lo, "ehi": hi,
                         "bias": b, "blo": blo, "bhi": bhi})
            print(f"    on {lab:<28} error {m:>5.2f}pp [{lo:.2f}, {hi:.2f}]   "
                  f"bias {b:+.2f}pp [{blo:+.2f}, {bhi:+.2f}] "
                  f"({'UNDER-predicts' if b > 0 else 'over-predicts'})")
        print(f"    predicted mean {fit['p'].mean():.1%}; realised is bracketed by "
              f"{fit['event'].mean():.1%} and {fit['event_low'].mean():.1%}")

        bpts = [c["bias"] for c in conv]
        if (bpts[0] > 0) != (bpts[1] > 0):
            print(f"    -> DIRECTION UNKNOWN. The bracket spans zero: the two "
                  f"conventions put the bias on opposite sides, so any real bias "
                  f"is smaller than the measurement convention.")
        else:
            over = bpts[0] < 0
            near = min(conv, key=lambda c: abs(c["bias"]))   # least favourable
            clears = near["bhi"] < 0 if over else near["blo"] > 0
            word = "OVER-predicts" if over else "UNDER-predicts"
            print(f"    -> the curve {word} on BOTH conventions "
                  f"({bpts[0]:+.2f}pp and {bpts[1]:+.2f}pp), so the bracket does "
                  f"not span zero and the direction does not depend on how the "
                  f"fall is measured.")
            if clears:
                print(f"       ESTABLISHED: even at the least favourable edge of "
                      f"the bracket ({near['lab']}) the interval "
                      f"[{near['blo']:+.2f}, {near['bhi']:+.2f}] clears zero.")
            else:
                print(f"       Consistent but not significant at the least "
                      f"favourable edge ({near['lab']}, "
                      f"[{near['blo']:+.2f}, {near['bhi']:+.2f}] includes zero), "
                      f"so read it as a direction with a magnitude of roughly "
                      f"{min(abs(b) for b in bpts):.1f} to "
                      f"{max(abs(b) for b in bpts):.1f}pp.")
            if over:
                print(f"       Over-prediction is the CONSERVATIVE direction for a "
                      f"risk limit - the constraint binds earlier than it needs "
                      f"to. Leave it rather than 'correcting' it.")
            else:
                print(f"       Under-prediction is the dangerous direction. Do not "
                      f"report the book probability until it is understood.")

        # MAGNITUDE. Compared on POINT estimates, not interval ends.
        #
        # V52.mae is a mean ABSOLUTE error over reliability bins. Resampling adds
        # noise, the absolute value turns noise into positive bias, and the
        # percentile interval drifts upward - far enough that its lower end can sit
        # ABOVE the point estimate (one run produced 1.90pp with an interval of
        # [2.20, 5.48]). Reading "error exceeds 1.55pp because the interval's lower
        # end is 2.20" would then be reading the bootstrap's own bias as evidence.
        # So the comparison uses point estimates and the upward drift is reported
        # where it appears.
        drift = [c for c in conv if c["elo"] > c["err"] + 1e-9]
        if drift:
            print(f"    note: the bootstrap interval for a mean-absolute error "
                  f"drifts upward (resampling noise, made one-sided by the "
                  f"absolute value); on "
                  f"{' and '.join(c['lab'].split()[0] for c in drift)} its lower "
                  f"end sits above the point estimate, so use the point estimates "
                  f"for this comparison.")
        epts = [c["err"] for c in conv]
        lo_e, hi_e = min(epts), max(epts)
        if lo_e > ref_mae_pp * 1.25:
            print(f"    -> error {lo_e:.2f} to {hi_e:.2f}pp against the "
                  f"single-name model's {ref_mae_pp:.2f}pp: worse under both "
                  f"conventions. Aggregation costs accuracy.")
        elif hi_e <= ref_mae_pp * 1.25:
            print(f"    -> error {lo_e:.2f} to {hi_e:.2f}pp, in line with the "
                  f"single-name model's {ref_mae_pp:.2f}pp; the curve transfers "
                  f"without measurable cost.")
        else:
            print(f"    -> error {lo_e:.2f} to {hi_e:.2f}pp against the "
                  f"single-name model's {ref_mae_pp:.2f}pp: comparable at the "
                  f"favourable edge, roughly double at the other. Which one holds "
                  f"depends on how far a book's true low sits from its close, "
                  f"which daily bars cannot resolve.")
        print(f"       (reference: {ref_mae_pp:.2f}pp is the single-name model's "
              f"out-of-sample calibration error, V59; change it with "
              f"--ref-mae-pp)")
    else:
        print(f"    too few books inside the fitted range to say anything")

    print(f"\n  BELOW THE FITTED RANGE  ({len(clamp):,} books, prediction clamped "
          f"at the curve floor)")
    if len(clamp) >= 200:
        floor = clamp["p"].mean() * 100
        real = clamp["event"].mean() * 100
        _, lo, hi = V52.block_boot(
            clamp["date"].to_numpy(), clamp["p"].to_numpy(float),
            clamp["event"].to_numpy(float),
            lambda p, y: y.mean() * 100, n_boot=n_boot, seed=seed)
        print(f"    clamped prediction {floor:.1f}%, realised {real:.1f}% "
              f"(95% CI {lo:.1f} to {hi:.1f})")
        if hi <= floor:
            verdict = ("SAFE. The clamp is conservative: these books fell less "
                       "often than the floor it reports, so the bound holds and "
                       "the cluster and risk-at-stops limits are what bind.")
        elif lo > floor:
            verdict = ("UNSAFE. These books fell MORE often than the clamped "
                       "floor, so the reported bound is not a bound. Drop the "
                       "book probability from the output or extend the curve "
                       "downward before relying on it.")
        else:
            verdict = ("INCONCLUSIVE at this sample size. The interval spans the "
                       "floor. Keep the label as a bound and do not promote it "
                       "to an estimate.")
        print(f"    -> {verdict}")
        lowr = clamp["event_low"].mean() * 100
        print(f"    on daily lows (an over-count) {lowr:.1f}%, so the bound"
              f" {'still holds' if lowr <= floor else 'is not safe even as a bound'}")
    else:
        print(f"    too few clamped books to judge the bound")

    over = d[d["bound"] == "lower"]
    if len(over) >= 200:
        # Above the fitted range the curve clamps at its TOP, so the prediction is
        # a floor: realised should come in at or above it. This is the opposite
        # test to the one below the range and it is easy to read backwards.
        cap = over["p"].mean() * 100
        real = over["event"].mean() * 100
        _, lo, hi = V52.block_boot(
            over["date"].to_numpy(), over["p"].to_numpy(float),
            over["event"].to_numpy(float), lambda p, y: y.mean() * 100,
            n_boot=n_boot, seed=seed)
        print(f"\n  ABOVE THE FITTED RANGE  ({len(over):,} books, prediction "
              f"clamped at the curve ceiling)")
        print(f"    clamped prediction {cap:.1f}%, realised {real:.1f}% "
              f"(95% CI {lo:.1f} to {hi:.1f})")
        if lo >= cap:
            print(f"    -> the ceiling behaves as a lower bound, as intended. "
                  f"These books are riskier than the curve can express; the "
                  f"number understates them and should be read as 'at least'.")
        else:
            print(f"    -> the ceiling is NOT acting as a lower bound. Books this "
                  f"volatile are outside what the curve was fitted on in both "
                  f"directions; do not quote a probability for them at all.")

    # does composition matter once volatility is held fixed? This is the V54
    # question asked one level up: if k and correlation add nothing beyond book
    # volatility, the single-name curve transfers and the whole approach is sound.
    print(f"\n  DOES COMPOSITION MATTER BEYOND BOOK VOLATILITY?")
    print(f"  Within a volatility band, compare books of different width.")
    # Coarse bands on purpose. The reliability buckets above are deliberately
    # fine, which left roughly 190 books per bucket - enough to read one event
    # rate, not enough to split it four ways by k. Worse, the only buckets that
    # cleared a 200-book floor were the two below the fitted range, where the
    # event rate is under 1% and no comparison can say anything. The bands here
    # are wide enough that the k split lands where events actually happen.
    bands, fmin, fmax = [], min(float(p["fvol"]) for p in calib["curve"]), \
        max(float(p["fvol"]) for p in calib["curve"])
    v = d["book_vol"]
    if (v < fmin).sum() >= 300:
        bands.append(("below the fitted range", d[v < fmin]))
    ins = d[(v >= fmin) & (v <= fmax)]
    if len(ins) >= 600:
        cut = float(ins["book_vol"].median())
        bands.append((f"fitted, {fmin:.0f}-{cut:.0f}%",
                      ins[ins["book_vol"] <= cut]))
        bands.append((f"fitted, {cut:.0f}-{fmax:.0f}%",
                      ins[ins["book_vol"] > cut]))
    elif len(ins) >= 250:
        bands.append((f"fitted, {fmin:.0f}-{fmax:.0f}%", ins))
    if (v > fmax).sum() >= 250:
        bands.append(("above the fitted range", d[v > fmax]))

    print(f"  {'band':>24}{'k':>5}{'n':>8}{'book vol':>10}{'realised':>10}"
          f"{'vs band':>9}")
    for label, g in bands:
        base = g["event"].mean() * 100
        for k, gk in g.groupby("k", observed=True):
            if len(gk) < 60:
                continue
            print(f"  {label:>24}{k:>5}{len(gk):>8,}"
                  f"{gk['book_vol'].mean():>9.1f}%"
                  f"{gk['event'].mean() * 100:>9.1f}%"
                  f"{gk['event'].mean() * 100 - base:>+9.1f}")
        print(f"  {'':>24}{'all':>5}{len(g):>8,}{g['book_vol'].mean():>9.1f}%"
              f"{base:>9.1f}%{0.0:>+9.1f}")

    # ONE NAME AGAINST MANY, AT MATCHED VOLATILITY.
    #
    # This is the test the table is for, so it is computed rather than left to
    # the reader's eye. k=1 is the object the curve was fitted on; k>=3 is what
    # the pipeline actually trades. If they differ at the SAME book volatility
    # then volatility is not sufficient at the book level and composition is
    # doing something of its own.
    #
    # The 'book vol' column is the control on this comparison: if it drifts
    # across k within a band, the band is too wide and the difference is residual
    # volatility rather than breadth.
    # POOL THE FITTED RANGE.
    #
    # The bands above are split so the per-k event rates can be read against a
    # comparable volatility, but that split halves the sample for the breadth
    # test and leaves it underpowered: on this data the two fitted bands returned
    # +3.5pp [+0.2, +7.0] and +7.4pp [-0.0, +15.0] - same sign, same mechanism,
    # neither conclusive. Pooling is legitimate here precisely BECAUSE the
    # comparison is curve-adjusted: netting each group against its own prediction
    # is what removes the volatility difference that motivated the split.
    breadth = list(bands)
    if len(ins) >= 400 and sum(1 for b in bands if "fitted" in b[0]) > 1:
        breadth.append((f"fitted range, pooled", ins))

    print(f"\n  ONE NAME vs MANY, at matched book volatility")
    for label, g in breadth:
        one = g[g["k"] == 1]
        many = g[g["k"] >= 3]
        if len(one) < 60 or len(many) < 120:
            continue
        dv = one["book_vol"].mean() - many["book_vol"].mean()
        d1 = one["event"].mean() * 100
        dm = many["event"].mean() * 100

        # Compare each group against its OWN prediction and difference the
        # errors, rather than differencing the raw rates. Within a band k=1 sits
        # at a slightly higher book volatility than k>=8 does - on this data
        # 35.8% against 34.0% - and a raw rate difference would charge that gap to
        # breadth. Netting each group against the curve removes it to first
        # order, using the curve's own slope, which is the only slope available.
        #
        # The residual bias runs one way and is worth stating: a single name has a
        # wider intraday range relative to its closes than an average of several
        # does, so the close-based under-count is LARGER for k=1. That pushes this
        # estimator down, so a positive result here is a lower bound on the true
        # aggregation effect, not an overstatement of it.
        sub = g[(g["k"] == 1) | (g["k"] >= 3)].copy()
        err = sub["event"].to_numpy(float) - sub["p"].to_numpy(float)
        ind = (sub["k"] == 1).to_numpy(float)

        def _errdiff(i, e):
            a, b = i > 0.5, i <= 0.5
            if not a.any() or not b.any():
                return np.nan
            return (e[a].mean() - e[b].mean()) * 100

        point = _errdiff(ind, err)
        _, blo, bhi = V52.block_boot(sub["date"].to_numpy(), ind, err,
                                     _errdiff, n_boot=n_boot, seed=seed)
        verdict = ("one name is riskier" if blo > 0 else
                   "many names are riskier" if bhi < 0 else
                   "no difference established")
        print(f"    {label:<24} k=1 {d1:>5.1f}%  k>=3 {dm:>5.1f}%   "
              f"curve-adjusted diff {point:+5.1f}pp [{blo:+.1f}, {bhi:+.1f}]  "
              f"{verdict}")
        dp = (one["p"].mean() - many["p"].mean()) * 100
        if abs(dp) < 0.1:
            print(f"    {'':<24} (the curve is flat across both groups here, so "
                  f"the adjustment is inert and this is a raw rate difference)")
        else:
            print(f"    {'':<24} (book volatility differs by {dv:+.1f}pp between "
                  f"the groups, worth {dp:+.1f}pp of prediction; absorbed)")
    print(f"\n  If one name is consistently riskier at the same book volatility, the")
    print(f"  mechanism is aggregation: averaging several names thins the tails for a")
    print(f"  given standard deviation, and a 20% fall is a tail event. The curve was")
    print(f"  fitted on single names, so it then OVER-states a diversified book -")
    print(f"  which is the conservative direction for a limit, and a reason to label")
    print(f"  the number rather than recalibrate it on a few hundred effective")
    print(f"  observations.")


# =============================================================================
# CLI
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_portfolio_v60() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--universe", default=None,
                    help="a text file of tickers, one per line")
    ap.add_argument("--price-cache", default=P2.PRICE_CACHE)
    ap.add_argument("--k", nargs="*", type=int, default=[1, 3, 5, 8])
    ap.add_argument("--gross", nargs="*", type=float, default=[50.0, 100.0])
    ap.add_argument("--n-books", type=int, default=6000)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--horizon-days", type=int, default=P2.HORIZON_DAYS)
    ap.add_argument("--drawdown", type=float, default=P2.DRAWDOWN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-mae-pp", type=float, default=1.55,
                    help="single-name out-of-sample calibration error to compare "
                         "the book-level error against (V59 measured 1.55)")
    ap.add_argument("--out", default=None, help="write the book table to CSV")
    return ap


def run_portfolio_v60(tickers=None, universe=None, price_cache=None, k=None, gross=None, n_books=None, n_boot=None, horizon_days=None, drawdown=None, seed=None, ref_mae_pp=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_portfolio_v60()
        run_portfolio_v60(tickers=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"tickers": tickers, "universe": universe, "price_cache": price_cache, "k": k, "gross": gross, "n_books": n_books, "n_boot": n_boot, "horizon_days": horizon_days, "drawdown": drawdown, "seed": seed, "ref_mae_pp": ref_mae_pp, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if not os.path.exists(P2.CALIB_FILE):
        sys.exit(f"{P2.CALIB_FILE} not found. Run:\n"
                 f"  python {P2.__name__}.py --calibrate")
    calib = P2.load_calibration()
    if calib.get("horizon_days") not in (None, cfg.horizon_days):
        print(f"  NOTE: the calibration was fitted at "
              f"{calib['horizon_days']}d and this run tests {cfg.horizon_days}d. "
              f"Different horizons are different events - the comparison is only "
              f"meaningful when they match.", file=sys.stderr)

    if cfg.universe and os.path.exists(cfg.universe):
        tickers = [ln.strip().upper() for ln in open(cfg.universe)
                   if ln.strip() and not ln.startswith("#")]
    elif cfg.tickers:
        tickers = [t.upper() for t in cfg.tickers]
    else:
        tickers = sorted(f[:-4] for f in os.listdir(cfg.price_cache)
                         if f.endswith(".pkl"))
    print(f"  universe: {len(tickers)} tickers from {cfg.price_cache}")

    C, L, F, _ = build_panel(tickers, cfg.price_cache)
    if C is None or C.shape[1] < MIN_NAMES:
        sys.exit(f"need at least {MIN_NAMES} names with usable history, and a "
                 f"calibration carrying a multivariate volatility model")

    nchk, nbad, worst = verify_against_p2(C, L, cfg.horizon_days,
                                          cfg.price_cache)
    if nchk < 10:
        print(f"  WARNING: could only cross-check {nchk} observations against "
              f"P2's own event definition", file=sys.stderr)
    elif nbad:
        sys.exit(f"  HARNESS DISAGREES WITH P2 on {nbad} of {nchk} cross-checked "
                 f"observations (worst {worst:.4f}pp).\n"
                 f"  This file would be measuring a different event from the one "
                 f"the curve was fitted on,\n  so every number below would be "
                 f"meaningless. Fix book_falls or the forward window first.")
    else:
        print(f"  event definition matches P2 exactly on {nchk} cross-checked "
              f"observations")

    d = run(C, L, F, cfg.k, cfg.gross, cfg.n_books, cfg.seed,
            cfg.horizon_days, cfg.drawdown)
    if d is None or len(d) < 200:
        sys.exit(f"only {0 if d is None else len(d)} books formed - widen the "
                 f"universe or lower --k")
    report(d, calib, cfg.drawdown, cfg.horizon_days, cfg.n_boot, cfg.seed,
           ref_mae_pp=cfg.ref_mae_pp)
    if cfg.out:
        d.to_csv(cfg.out, index=False)
        print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_portfolio_v60(),
# and each value shown is that argument's own default. This one has no QUICK mode - a
# full run takes a while.
#
# Passing any command-line flag still works and takes over, so the old CLI is
# not lost: it just is not the default way in any more.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_TICKERS      = None
    RUN_UNIVERSE     = None                 # a text file of tickers, one per line
    RUN_PRICE_CACHE  = 'price_cache_v43'
    RUN_K            = [1, 3, 5, 8]
    RUN_GROSS        = [50.0, 100.0]
    RUN_N_BOOKS      = 6000
    RUN_N_BOOT       = 400
    RUN_HORIZON_DAYS = 90
    RUN_DRAWDOWN     = 0.2
    RUN_SEED         = 0
    # single-name out-of-sample calibration error to compare the book-level error against (V59 measured 1.55)
    RUN_REF_MAE_PP   = 1.55
    RUN_OUT          = None                 # write the book table to CSV
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_portfolio_v60.py --tickers ...
    else:
        run_portfolio_v60(
            tickers=RUN_TICKERS,
            universe=RUN_UNIVERSE,
            price_cache=RUN_PRICE_CACHE,
            k=RUN_K,
            gross=RUN_GROSS,
            n_books=RUN_N_BOOKS,
            n_boot=RUN_N_BOOT,
            horizon_days=RUN_HORIZON_DAYS,
            drawdown=RUN_DRAWDOWN,
            seed=RUN_SEED,
            ref_mae_pp=RUN_REF_MAE_PP,
            out=RUN_OUT,
        )
