#!/usr/bin/env python3
"""
evaluate_scorecard_v61.py - SCORE THE MODEL'S OUTPUT, THE WAY YOU'D SCORE ANY
                            CLASSIFIER, AT THE DECISION POINTS ACTUALLY USED.

WHY THIS FILE EXISTS
--------------------
Everything from V52 to V60 asked whether the probability is well formed. None of
it asked the plain question: when the model says a name is high risk, is it? The
answer to that is a confusion matrix, and there has never been one.

The reason it kept getting skipped is that the model does not output "yes" or
"no". It outputs 5.2%. A single 5.2% prediction cannot be right or wrong. But the
PIPELINE does make a decision - it maps the probability onto a tier and cuts the
position - and a decision can be scored. So this file evaluates the output at the
tier boundaries the deployed pipeline uses, and reports it in the form anyone can
check:

    of the N names the model put in its top tier, X% fell 20% or more
    of the N names in its lowest tier,             Y% did

Base rate is around 13%, so accuracy is a trap: predicting "no fall" for
everything scores about 87% and is worthless. Precision, recall and lift over the
base rate are the honest measures for an imbalanced event, and all three are
reported with clustered intervals.

WHAT ELSE IS IN HERE
--------------------
  DECILES        rank every prediction, cut into ten, show the realised rate and
                 the lift per decile. The simplest possible test of whether the
                 output separates anything.
  CONFUSION      at each deployed tier boundary, out of sample.
  DISCRIMINATION AUC and average precision, with clustered intervals.
  RESOLUTION     Murphy's decomposition. A model that always prints the base rate
                 is perfectly calibrated and carries no information; resolution is
                 the term that rules that out, and it has been sitting unreported
                 in V52 since the beginning.
  THE LADDER     the empirical curve against a closed-form barrier probability.
                 For driftless geometric Brownian motion,

                     P(fall of d within T) = 2 * Phi( ln(1-d) / (sigma * sqrt(T)) )

                 No fitting, no 160,000 observations, no twelve bins - just the
                 volatility forecast and a normal CDF. If that matches the fitted
                 curve, then the contribution of this project is the VOLATILITY
                 FORECAST and not the curve, and the write-up should say so. This
                 is a real falsification risk and it has not been taken until now.
  REGIMES        by year, because a risk model that only calibrates in calm
                 markets fails in the one condition it exists for.

AN ANALYTICAL POINT THAT SAVES A MEASUREMENT
--------------------------------------------
AUC is invariant under monotone transformations of the score. The empirical curve
and the closed-form barrier probability are both monotone in forecast volatility,
so they have IDENTICAL AUC by construction. Discrimination therefore belongs
entirely to the volatility forecast, and the curve's only possible contribution is
calibration. That is not measured below, it is asserted and then CHECKED - if the
two AUCs differ, something has become non-monotone and the run says so.

WHY NOT REUSE V52'S WALK-FORWARD
--------------------------------
V52 imports class_ai_pillar2_risk - the ORIGINAL module, 30% over 180 days with a
single close-to-close feature - and its predict() feeds obs["vol60"] in as a bare
float. Against the current calibration that path returns NaN for every row, and
its walkforward() reads shrinkage keys "a" and "b" that the V53 multivariate fit
does not have. V52's published numbers belong to the old configuration and the old
module is kept so they stay reproducible. So this file borrows only V52's METRIC
functions, which take p and y arrays and know nothing about any model, and does
its own fitting against class_ai_pillar2_risk_v2.

USAGE
  python evaluate_scorecard_v61.py
  python evaluate_scorecard_v61.py --quick
  python evaluate_scorecard_v61.py --n-boot 800 --min-train-years 6
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk_v2 as P2
import evaluate_calibration_v52 as V52

N_CURVE_BINS = 12
N_DECILES = 10
MIN_TRAIN_YEARS = 6
MIN_TRAIN_ROWS = 5000
MIN_TEST_ROWS = 200


# =============================================================================
# THE NORMAL CDF, WITHOUT A SCIPY DEPENDENCY
# =============================================================================
def _ndtr(z):
    """Standard normal CDF. scipy if present, math.erf otherwise."""
    z = np.asarray(z, float)
    try:
        from scipy.special import ndtr
        return ndtr(z)
    except Exception:
        return 0.5 * (1.0 + np.frompyfunc(math.erf, 1, 1)(
            z / math.sqrt(2.0)).astype(float))


def barrier_prob(vol_annual_pct, drawdown, horizon_days):
    """
    P(a fall of `drawdown` at some point within `horizon_days`) for driftless
    geometric Brownian motion, by the reflection principle:

        P = 2 * Phi( ln(1 - d) / (sigma * sqrt(T)) )

    Deliberately crude. It assumes no drift, lognormal returns with no fat tails,
    and continuous monitoring. Real names have positive drift (fewer barrier hits),
    fat tails (more), and are monitored on daily bars (fewer). The fitted curve
    absorbs all three; this absorbs none of them. That is exactly why it is the
    right baseline - whatever the curve is worth, it is worth it relative to this.
    """
    s = np.asarray(vol_annual_pct, float) / 100.0
    T = float(horizon_days) / 365.25
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(1.0 - drawdown) / (s * math.sqrt(T))
    p = 2.0 * _ndtr(z)
    return np.clip(np.where(np.isfinite(p), p, np.nan), 0.0, 1.0)


# =============================================================================
# OUT-OF-SAMPLE PREDICTIONS
# =============================================================================
def isotonic(vals, weights):
    """
    Weighted pool-adjacent-violators. Returns the closest non-decreasing sequence.

    Needed because P2.fit_drawdown_curve does NOT enforce monotonicity: it cuts
    forecast volatility into equal-count bins and takes each bin's mean event rate,
    and empirical rates on a subset can easily run backwards. The shipped
    calibration happens to come out monotone - deployment_preflight.py checks it -
    but a yearly refit on a training subset is a smaller sample and need not. A
    non-monotone curve destroys ranking information the volatility forecast had,
    which is exactly what V55 measured at 0.076 of AUC.
    """
    v = [float(x) for x in vals]
    w = [max(float(x), 1e-9) for x in weights]
    blocks = [[v[i] * w[i], w[i], 1] for i in range(len(v))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] <= blocks[i + 1][0] / blocks[i + 1][1] + 1e-15:
            i += 1
        else:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks[i][2] += blocks[i + 1][2]
            del blocks[i + 1]
            if i > 0:
                i -= 1
    out = []
    for b in blocks:
        out.extend([b[0] / b[1]] * b[2])
    return out


def monotone_violation(curve):
    """Largest backward step in a curve's probabilities, in percentage points."""
    ps = [float(q["p"]) for q in curve]
    worst = 0.0
    for a, b in zip(ps, ps[1:]):
        worst = max(worst, (a - b) * 100.0)
    return worst


def predict_v53(obs, calib):
    """Vectorised equivalent of P2.prob_drawdown over a whole frame."""
    sh = calib["shrinkage"]
    names = sh.get("features", ["vol60"])
    if any(n not in obs.columns for n in names):
        return None, None
    X = obs[names].to_numpy(float)
    if "features" in sh:
        fv = float(sh["intercept"]) + X @ np.asarray(sh["coef"], float)
    else:
        fv = float(sh["a"]) + float(sh["b"]) * X[:, 0]
    pts = calib["curve"]
    p = np.interp(fv, [q["fvol"] for q in pts], [q["p"] for q in pts])
    return p, fv


def walkforward(obs, min_train_years, embargo_days, drawdown, horizon_days,
                force_isotonic=False, verbose=True):
    """
    Refit each year on data whose outcomes had already resolved when the year
    opened, then predict that year. Returns pooled out-of-sample rows carrying
    every forecast the ladder needs, so all of them are scored on exactly the same
    observations - scoring competing forecasts on different row sets is how V55
    first produced a difference that was really a sample change.
    """
    yrs = sorted(obs["date"].dt.year.unique())
    first = yrs[0] + int(min_train_years)
    keep, fits = [], []
    for yr in [y for y in yrs if y >= first]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = obs[obs["date"] <= opens - pd.Timedelta(days=embargo_days)]
        te = obs[(obs["date"] >= opens) &
                 (obs["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < MIN_TRAIN_ROWS or len(te) < MIN_TEST_ROWS:
            continue
        sh = P2.fit_vol_shrinkage(tr)
        curve = P2.fit_drawdown_curve(tr, sh, N_CURVE_BINS)
        # Monotonicity is checked INSIDE each fit. V55 twice got this wrong by
        # pooling across years - years have different curves, so a pooled check
        # invents thousands of inversions that no single fit contains.
        viol = monotone_violation(curve)
        if force_isotonic and viol > 0:
            fixed = isotonic([q["p"] for q in curve],
                             [q.get("n", 1) for q in curve])
            curve = [dict(q, p=float(fx)) for q, fx in zip(curve, fixed)]
        cal = {"shrinkage": sh, "curve": curve}
        p_model, fvol = predict_v53(te, cal)
        if p_model is None:
            continue
        keep.append(pd.DataFrame({
            "date": te["date"].to_numpy(),
            "ticker": te["ticker"].to_numpy(),
            "y": te["event"].to_numpy(float),
            "p_model": p_model,
            "fvol": fvol,
            # the ladder, all from information available at prediction time
            "p_barrier_fcast": barrier_prob(fvol, drawdown, horizon_days),
            "p_barrier_trail": barrier_prob(te["vol60"].to_numpy(float),
                                            drawdown, horizon_days),
            "p_base": float(tr["event"].mean()),
        }))
        fits.append({"year": yr, "n_train": len(tr), "n_test": len(te),
                     "train_base_rate": float(tr["event"].mean()),
                     "r2": float(sh.get("r2", np.nan)),
                     "coef_sum": float(np.sum(sh.get("coef", [np.nan]))),
                     "monotone_violation_pp": float(viol),
                     "isotonic_applied": bool(force_isotonic and viol > 0)})
        if verbose:
            print(f"    {yr}: trained on {len(tr):,} to "
                  f"{opens - pd.Timedelta(days=embargo_days):%Y-%m}, "
                  f"predicted {len(te):,}")
    if not keep:
        return None, None
    return pd.concat(keep, ignore_index=True), pd.DataFrame(fits)


# =============================================================================
# CLASSIFIER METRICS
# =============================================================================
def confusion(p, y, thr):
    """Counts and rates for one threshold. Lift is precision over the base rate."""
    p, y = np.asarray(p, float), np.asarray(y, float) > 0.5
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    flag = p >= thr
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    tn = int((~flag & ~y).sum())
    base = y.mean() if len(y) else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "flagged": tp + fp, "flagged_pct": (tp + fp) / len(y) * 100
            if len(y) else np.nan,
            "precision": prec,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "lift": prec / base if base and np.isfinite(prec) else np.nan,
            "base_rate": base}


def average_precision(p, y):
    """
    Area under the precision-recall curve, by the step-wise definition.

    Reported alongside AUC because they answer different questions on a 13% event:
    AUC asks how well the ranking separates the two classes overall, average
    precision asks how clean the TOP of the ranking is, which is what a risk limit
    acts on.
    """
    p, y = np.asarray(p, float), np.asarray(y, float) > 0.5
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if y.sum() == 0 or len(y) == 0:
        return np.nan
    o = np.argsort(-p, kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float(np.sum(prec * yy) / yy.sum())


def decile_table(p, y, n=N_DECILES):
    d = pd.DataFrame({"p": np.asarray(p, float),
                      "y": np.asarray(y, float)}).dropna()
    if len(d) < n * 30:
        return None
    d["g"] = pd.qcut(d["p"].rank(method="first"), n, labels=False,
                     duplicates="drop")
    base = d["y"].mean()
    t = d.groupby("g").agg(predicted=("p", "mean"), realised=("y", "mean"),
                           n=("y", "size")).reset_index()
    t["lift"] = t["realised"] / base if base > 0 else np.nan
    return t, base


# =============================================================================
# REPORT
# =============================================================================
def scorecard(oos, drawdown, horizon_days, tiers, n_boot, seed):
    p = oos["p_model"].to_numpy(float)
    y = oos["y"].to_numpy(float)
    dates = oos["date"].to_numpy()

    print("\n" + "=" * 88)
    print(f"  SCORING THE OUTPUT:  P(a {drawdown:.0%} fall within "
          f"{horizon_days} days)")
    print("=" * 88)

    eff = V52.effective_n(dates, y, n_boot=n_boot, seed=seed)
    print(f"\n  {len(oos):,} out-of-sample predictions | "
          f"{oos['ticker'].nunique()} tickers | "
          f"{oos['date'].min():%Y-%m} to {oos['date'].max():%Y-%m}")
    print(f"  base rate {y.mean():.1%} | design effect "
          f"{eff['design_effect']:.0f}x -> about {eff['n_effective']:.0f} "
          f"independent observations")
    print(f"  Every interval below is a half-year block bootstrap. Row-level "
          f"resampling would")
    print(f"  treat one market episode as thousands of independent facts.")

    # ---------------------------------------------------------------- deciles
    dt = decile_table(p, y)
    if dt is not None:
        t, base = dt
        print(f"\n  DOES THE OUTPUT SEPARATE ANYTHING?  (predictions ranked, cut "
              f"into ten)")
        print(f"    {'decile':>8}{'n':>8}{'predicted':>11}{'realised':>10}"
              f"{'lift':>7}")
        for _, r in t.iterrows():
            print(f"    {int(r['g']) + 1:>8}{int(r['n']):>8,}"
                  f"{r['predicted'] * 100:>10.1f}%{r['realised'] * 100:>9.1f}%"
                  f"{r['lift']:>7.2f}")
        top, bot = t.iloc[-1], t.iloc[0]
        print(f"\n    In plain terms: of the {int(top['n']):,} names in the "
              f"riskiest tenth, {top['realised']:.1%} fell {drawdown:.0%} or more.")
        print(f"    Of the {int(bot['n']):,} in the calmest tenth, "
              f"{bot['realised']:.1%} did. Base rate {base:.1%}.")
        if top["realised"] > bot["realised"]:
            print(f"    That is a {top['realised'] / max(bot['realised'], 1e-9):.1f}x "
                  f"ratio top to bottom.")
        else:
            print(f"    THE RANKING IS INVERTED OR FLAT - the output does not "
                  f"separate. Nothing below matters.")

        # THE SHAPE OF THE RESIDUAL, AND WHICH END OF IT IS DANGEROUS.
        #
        # A small pooled calibration error can still hide a systematic tilt. If the
        # residual runs from positive at the calm end to negative at the risky end,
        # the curve is too STEEP: it overstates the names it flags and understates
        # the ones it clears. Overstating the flagged end is the safe direction -
        # those positions get cut anyway. Understating the cleared end is not: the
        # lowest tier carries a 1.00x multiplier, so those names are traded at full
        # size on the strength of a number that is too low.
        resid = (t["realised"] - t["predicted"]).to_numpy() * 100
        lo_r, hi_r = float(resid[:3].mean()), float(resid[-3:].mean())
        if lo_r > 0.5 and hi_r < -0.5:
            b1 = t.iloc[0]
            ratio = b1["realised"] / max(b1["predicted"], 1e-9)
            print(f"\n    RESIDUAL SHAPE: the curve is too STEEP - bottom three "
                  f"deciles under-predict by")
            print(f"    {lo_r:+.1f}pp on average, top three over-predict by "
                  f"{hi_r:+.1f}pp.")
            print(f"    The top end is the safe direction: those names get cut by "
                  f"the tiers regardless.")
            print(f"    The bottom is not. The calmest decile predicted "
                  f"{b1['predicted']:.1%} and realised "
                  f"{b1['realised']:.1%} -")
            print(f"    small in absolute terms, {ratio:.1f}x in relative terms, and "
                  f"those names are traded")
            print(f"    at FULL size because they sit in the lowest tier. Worth a "
                  f"line in the write-up,")
            print(f"    and worth considering a floor on the reported probability "
                  f"rather than a refit.")
        elif lo_r < -0.5 and hi_r > 0.5:
            print(f"\n    RESIDUAL SHAPE: the curve is too FLAT - it overstates the "
                  f"calm end ({lo_r:+.1f}pp)")
            print(f"    and understates the risky end ({hi_r:+.1f}pp). Understating "
                  f"the flagged end is the")
            print(f"    dangerous direction: those are the names the limits are "
                  f"supposed to catch.")
        else:
            print(f"\n    RESIDUAL SHAPE: no systematic tilt across the deciles "
                  f"(bottom {lo_r:+.1f}pp, top {hi_r:+.1f}pp).")

    # ------------------------------------------------------------- confusion
    print(f"\n  AT THE PIPELINE'S OWN DECISION POINTS")
    print(f"    Accuracy is omitted on purpose: with a {y.mean():.1%} base rate, "
          f"answering 'no fall'")
    print(f"    to everything scores {1 - y.mean():.1%} and is worthless. Lift is "
          f"precision over base rate.")
    print(f"\n    {'tier at':>9}{'flagged':>9}{'of all':>8}{'caught':>8}"
          f"{'precision':>11}{'recall':>8}{'lift':>7}{'95% CI on lift':>18}")
    rows = []
    for thr, label in tiers:
        c = confusion(p, y, thr)
        _, lo, hi = V52.block_boot(
            dates, p, y,
            lambda pp, yy, t=thr: confusion(pp, yy, t)["lift"],
            n_boot=n_boot, seed=seed)
        rows.append(dict(c, label=label, lift_lo=lo, lift_hi=hi))
        print(f"    {thr:>8.0%} {c['flagged']:>8,}{c['flagged_pct']:>7.1f}%"
              f"{c['tp']:>8,}{c['precision'] * 100:>10.1f}%"
              f"{c['recall'] * 100:>7.1f}%{c['lift']:>7.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>18}")
    for r in rows:
        if np.isfinite(r["lift_lo"]) and r["lift_lo"] <= 1.0 <= r["lift_hi"]:
            print(f"    NOTE: at the {r['thr']:.0%} boundary the lift interval "
                  f"includes 1.0 - flagging at that level is not "
                  f"distinguishable from flagging at random.")

    # --------------------------------------------------------- discrimination
    print(f"\n  DISCRIMINATION")
    a = V52.auc(p, y)
    _, alo, ahi = V52.block_boot(dates, p, y, V52.auc, n_boot=n_boot, seed=seed)
    ap = average_precision(p, y)
    _, plo, phi = V52.block_boot(dates, p, y, average_precision,
                                 n_boot=n_boot, seed=seed)
    print(f"    AUC                {a:.3f}  [{alo:.3f}, {ahi:.3f}]   "
          f"(0.500 = coin flip)")
    print(f"    average precision  {ap:.3f}  [{plo:.3f}, {phi:.3f}]   "
          f"(base rate {y.mean():.3f} = no ranking)")

    # ------------------------------------------------------------ resolution
    print(f"\n  IS IT MORE THAN A CONSTANT?  (Murphy decomposition)")
    bd = V52.brier_decomp(p, y)
    if bd:
        print(f"    Brier {bd['brier']:.5f} = reliability {bd['reliability']:.5f} "
              f"- resolution {bd['resolution']:.5f} + uncertainty "
              f"{bd['uncertainty']:.5f}")
        print(f"    A model that always prints {bd['base_rate']:.1%} has "
              f"reliability 0 and resolution 0.")
        _, rlo, rhi = V52.block_boot(
            dates, p, y,
            lambda pp, yy: (V52.brier_decomp(pp, yy) or {}).get("resolution",
                                                                np.nan),
            n_boot=n_boot, seed=seed)
        if np.isfinite(rlo) and rlo > 0:
            print(f"    resolution 95% CI [{rlo:.5f}, {rhi:.5f}] clears zero: the "
                  f"output carries information, not just the base rate.")
        else:
            print(f"    resolution 95% CI [{rlo:.5f}, {rhi:.5f}] includes zero - "
                  f"the output is not distinguishable from a constant forecast.")

    return rows


def ladder(oos, n_boot, seed, isotonic_applied=False):
    """
    The empirical curve against the closed form, on identical rows.

    The claim under test is narrow: the twelve-bin curve was fitted on 160,000
    observations, and a textbook first-passage formula needs none. If they score
    the same, the curve is not where the value is.
    """
    y = oos["y"].to_numpy(float)
    dates = oos["date"].to_numpy()
    ref = oos["p_base"].to_numpy(float)

    arms = [("fitted curve on forecast vol", "p_model"),
            ("closed form on forecast vol", "p_barrier_fcast"),
            ("closed form on trailing vol", "p_barrier_trail"),
            ("constant training base rate", "p_base")]

    print(f"\n  THE LADDER - what is the fitted curve actually worth?")
    print(f"    All four scored on the same {len(oos):,} rows.")
    print(f"\n    {'forecast':>30}{'mean':>7}{'Brier':>9}{'reliab':>9}"
          f"{'resol':>9}{'cal err':>9}{'skill':>8}")
    out = {}
    for label, col in arms:
        pv = oos[col].to_numpy(float)
        bd = V52.brier_decomp(pv, y) or {}
        m = V52.mae(pv, y)
        sk = V52.skill(pv, y, ref.mean())
        out[label] = {"brier": bd.get("brier"), "mae_pp": m, "skill": sk,
                      "reliability": bd.get("reliability"),
                      "resolution": bd.get("resolution"),
                      "auc": V52.auc(pv, y), "mean": float(np.nanmean(pv))}
        print(f"    {label:>30}{np.nanmean(pv) * 100:>6.1f}%"
              f"{bd.get('brier', np.nan):>9.5f}{bd.get('reliability', np.nan):>9.5f}"
              f"{bd.get('resolution', np.nan):>9.5f}{m:>8.2f}pp{sk:>8.3f}")

    # WHAT EACH RUNG BOUGHT. The ladder is ordered so the differences read as
    # contributions, and stating them here stops the table being eyeballed into
    # the wrong conclusion.
    order = ["constant training base rate", "closed form on trailing vol",
             "closed form on forecast vol", "fitted curve on forecast vol"]
    print(f"\n    what each step bought, in Brier skill:")
    prev = None
    for label in order:
        sk = out[label]["skill"]
        step = "" if prev is None else f"   {sk - prev:+.3f}"
        print(f"      {label:<30}{sk:>7.3f}{step}")
        prev = sk
    gain_vol = out["closed form on forecast vol"]["skill"] - \
        out["closed form on trailing vol"]["skill"]
    gain_curve = out["fitted curve on forecast vol"]["skill"] - \
        out["closed form on forecast vol"]["skill"]
    if np.isfinite(gain_vol) and np.isfinite(gain_curve):
        v_verb = "added" if gain_vol > 0 else "COST"
        c_verb = "added" if gain_curve > 0 else "COST"
        print(f"    The volatility forecast {v_verb} {abs(gain_vol):.3f}; "
              f"the fitted curve {c_verb} {abs(gain_curve):.3f}.")
        if gain_vol > 0 and gain_curve > 0:
            print(f"    Both are real, and the ratio is about "
                  f"{gain_vol / gain_curve:.1f} to 1 - the write-up should lead "
                  f"with the one carrying the weight.")
        elif gain_vol > 0 >= gain_curve:
            print(f"    Only the volatility forecast earns anything on this "
                  f"metric. The curve's case has")
            print(f"    to rest on the reliability column below, not on skill.")
        elif gain_curve > 0 >= gain_vol:
            print(f"    The curve earns its keep here and the volatility forecast "
                  f"does not, which is the")
            print(f"    reverse of what V53 measured - check that the forecast is "
                  f"being fed the right features.")
        else:
            print(f"    Neither step earns anything on this metric, so the whole "
                  f"chain reduces to the")
            print(f"    closed form on trailing volatility. That is a finding, and "
                  f"a blunt one.")

    # THE INVARIANCE CHECK MUST BE RUN PER YEAR, NOT POOLED.
    #
    # AUC is invariant under monotone transformation of the score, so within ONE
    # fit the curve and the closed form must rank identically. Across the pooled
    # walk-forward they need not: there are sixteen different curves, so forecast
    # volatility 40% maps to one probability in 2013 and another in 2020, and the
    # pooled score is not a monotone function of volatility at all. The closed form
    # is the same function every year, so it is.
    #
    # The first version compared the pooled AUCs and, on a run where every refit
    # curve was monotone, reported that the curve was "DESTROYING discrimination".
    # It was reading cross-year curve variation as within-year inversions. This is
    # the third time in this project that pooling across refits has broken a
    # within-fit invariant - V55 invented 4,900 monotonicity violations the same
    # way, and V59 paired probabilities across horizons that shared no outcome.
    ydf = oos.copy()
    ydf["year"] = pd.to_datetime(ydf["date"]).dt.year
    gaps = []
    for yr, g in ydf.groupby("year"):
        if len(g) < 200:
            continue
        gaps.append((yr,
                     V52.auc(g["p_model"].to_numpy(float),
                             g["y"].to_numpy(float)),
                     V52.auc(g["p_barrier_fcast"].to_numpy(float),
                             g["y"].to_numpy(float))))
    if gaps:
        worst_yr, wa1, wa2 = max(gaps, key=lambda r: abs(r[1] - r[2]))
        print(f"\n    within-year invariance check (the theorem only holds inside "
              f"one fit):")
        print(f"      largest per-year AUC gap {wa1 - wa2:+.4f} in {worst_yr} "
              f"({wa1:.4f} vs {wa2:.4f}) across {len(gaps)} years")
        if abs(wa1 - wa2) < 0.005:
            print(f"      Within every year the two rank the same, as they must - "
                  f"one curve per year is")
            print(f"      monotone in forecast volatility, and the residual is ties "
                  f"from flat bins. So")
            print(f"      ALL discrimination belongs to the volatility forecast; "
                  f"the curve's only")
            print(f"      possible contribution is calibration. Compare the "
                  f"reliability column.")
        else:
            print(f"      A within-year gap this large means that year's curve is "
                  f"non-monotone over the")
            print(f"      range actually observed. Check the per-fit violations "
                  f"reported above.")

    a1 = out["fitted curve on forecast vol"]["auc"]
    a2 = out["closed form on forecast vol"]["auc"]
    print(f"\n    pooled AUC {a1:.4f} vs {a2:.4f}. Any pooled gap is cross-year "
          f"curve variation,")
    print(f"    not a within-year defect - it reorders a 2013 name against a 2020 "
          f"name, which is")
    print(f"    a comparison nobody makes. The per-year check above is the one that "
          f"means anything.")
    # the verdict on the curve
    c_rel = out["fitted curve on forecast vol"]["reliability"]
    b_rel = out["closed form on forecast vol"]["reliability"]
    diff = (b_rel - c_rel) if (c_rel is not None and b_rel is not None) else np.nan
    _, lo, hi = V52.block_boot(
        dates, oos["fvol"].to_numpy(float), y,
        lambda fv, yy: ((V52.brier_decomp(barrier_prob(fv, P2.DRAWDOWN,
                                                       P2.HORIZON_DAYS), yy)
                         or {}).get("reliability", np.nan)),
        n_boot=n_boot, seed=seed)
    print(f"\n    the closed form's reliability {b_rel:.5f} "
          f"(95% CI [{lo:.5f}, {hi:.5f}])")
    print(f"    the fitted curve's reliability {c_rel:.5f}")
    if np.isfinite(diff) and np.isfinite(lo) and c_rel < lo:
        print(f"    -> the curve is better calibrated than the closed form, "
              f"beyond the interval.")
        print(f"       Fitting the curve earned something. Say so, and say it is "
              f"calibration only.")
    elif np.isfinite(diff) and diff <= 0:
        print(f"    -> the CLOSED FORM is at least as well calibrated. The curve "
              f"is not earning its")
        print(f"       keep: the contribution of this work is the volatility "
              f"forecast, and the honest")
        print(f"       claim shrinks to that. This is a real result, not a "
              f"failure - a textbook")
        print(f"       formula plus a better volatility estimate is a cleaner "
              f"thesis than twelve")
        print(f"       fitted bins.")
    else:
        print(f"    -> comparable. The curve is not clearly better than the "
              f"closed form; treat the")
        print(f"       volatility forecast as the contribution and the curve as "
              f"convenience.")
    return out


def regimes(oos, n_boot, seed, pooled_cal_err=None):
    print(f"\n  BY YEAR - does it hold when it matters?")
    print(f"    {'year':>6}{'n':>8}{'predicted':>11}{'realised':>10}{'error':>8}"
          f"{'AUC':>7}{'top-decile rate':>17}")
    oos = oos.copy()
    oos["year"] = pd.to_datetime(oos["date"]).dt.year
    for yr, g in oos.groupby("year"):
        if len(g) < 200:
            continue
        pv, yv = g["p_model"].to_numpy(float), g["y"].to_numpy(float)
        a = V52.auc(pv, yv)
        thr = np.nanquantile(pv, 0.9)
        top = yv[pv >= thr]
        print(f"    {yr:>6}{len(g):>8,}{np.nanmean(pv) * 100:>10.1f}%"
              f"{yv.mean() * 100:>9.1f}%{(yv.mean() - np.nanmean(pv)) * 100:>+8.1f}"
              f"{a:>7.3f}{top.mean() * 100:>16.1f}%")
    # YEAR-TO-YEAR DISPERSION IS THE NUMBER THAT MATTERS FOR DEPLOYMENT.
    #
    # A pooled calibration error averages a year that ran hot against a year that
    # ran cold and reports something far smaller than either. Nobody trades the
    # pooled sample - they trade one year at a time - so the honest figure is the
    # mean ABSOLUTE yearly error and its worst case.
    rec = []
    for yr, g in oos.groupby("year"):
        if len(g) < 200:
            continue
        pv, yv = g["p_model"].to_numpy(float), g["y"].to_numpy(float)
        rec.append({"year": int(yr), "err": (yv.mean() - np.nanmean(pv)) * 100,
                    "auc": V52.auc(pv, yv)})
    if rec:
        errs = np.array([r["err"] for r in rec])
        signed = float(np.nanmean(errs))
        mae_y = float(np.mean(np.abs(errs)))
        hot = max(rec, key=lambda r: r["err"])
        cold = min(rec, key=lambda r: r["err"])
        print(f"\n    DISPERSION: mean absolute yearly error {mae_y:.1f}pp, range "
              f"{cold['err']:+.1f} ({cold['year']}) to {hot['err']:+.1f} "
              f"({hot['year']}).")
        if abs(signed) < 0.3 * max(mae_y, 1e-9):
            print(f"    Signed bias across years {signed:+.1f}pp - small against "
                  f"the {mae_y:.1f}pp typical year, so")
            print(f"    the model is close to unbiased on average and the yearly "
                  f"errors are timing, not level.")
        else:
            print(f"    Signed bias across years {signed:+.1f}pp - NOT small "
                  f"against the {mae_y:.1f}pp typical")
            print(f"    year, so there is a level bias on top of the year-to-year "
                  f"scatter.")
        # COMPARE AGAINST THE NUMBER ACTUALLY QUOTED.
        # An earlier version divided the mean absolute yearly error by the mean
        # SIGNED error and called the ratio an understatement. With symmetric
        # errors the signed mean goes to zero, so that ratio is unbounded and says
        # nothing - and nobody was quoting the signed mean as an error anyway. The
        # figure that gets quoted is the pooled binned calibration error, so that
        # is what the yearly dispersion has to be held against.
        if pooled_cal_err is not None and np.isfinite(pooled_cal_err):
            print(f"    Pooled calibration error is {pooled_cal_err:.2f}pp; the "
                  f"typical YEAR is {mae_y:.1f}pp,")
            print(f"    about {mae_y / max(pooled_cal_err, 1e-9):.1f}x that, and "
                  f"the worst year "
                  f"{max(abs(hot['err']), abs(cold['err'])):.1f}pp, about "
                  f"{max(abs(hot['err']), abs(cold['err'])) / max(pooled_cal_err, 1e-9):.1f}x.")
            print(f"    Quote the pooled figure for how well calibrated the model "
                  f"is, and the yearly")
            print(f"    figure for how wrong it can be in the year you are actually "
                  f"trading.")

        # THE LAG EFFECT, MEASURED RATHER THAN ASSERTED.
        # A trailing volatility feature reads high AFTER a shock, when the fall has
        # already happened, and low before one. If that is what is going on, this
        # year's signed error should run NEGATIVE (over-prediction) following a bad
        # year - so the correlation between error and the previous year's realised
        # rate should be negative. It is one line to check and it was tempting to
        # assert instead.
        if len(rec) >= 6:
            rec_s = sorted(rec, key=lambda r: r["year"])
            realised = {r["year"]: None for r in rec_s}
            for yr, g in oos.groupby("year"):
                if int(yr) in realised:
                    realised[int(yr)] = float(g["y"].mean() * 100)
            xs, ys = [], []
            for a, b in zip(rec_s, rec_s[1:]):
                if b["year"] == a["year"] + 1 and realised[a["year"]] is not None:
                    xs.append(realised[a["year"]])
                    ys.append(b["err"])
            if len(xs) >= 5:
                r = float(np.corrcoef(xs, ys)[0, 1])
                # FIFTEEN YEAR PAIRS IS NOT MANY.
                # A correlation of -0.34 on 15 points does not clear the 5% level,
                # and the first version announced the lag mechanism as though it
                # had. Resample the PAIRS to get an interval, since there is no
                # block structure left to exploit at this level of aggregation.
                rng = np.random.default_rng(seed)
                xa, ya = np.asarray(xs, float), np.asarray(ys, float)
                reps = []
                for _ in range(2000):
                    k = rng.integers(0, len(xa), len(xa))
                    if np.std(xa[k]) > 0 and np.std(ya[k]) > 0:
                        reps.append(np.corrcoef(xa[k], ya[k])[0, 1])
                rlo, rhi = (float(np.percentile(reps, 2.5)),
                            float(np.percentile(reps, 97.5))) if len(reps) > 100 \
                    else (np.nan, np.nan)
                print(f"\n    LAG CHECK: correlation between a year's signed error "
                      f"and the PREVIOUS year's")
                print(f"    realised rate is {r:+.2f} over {len(xs)} year pairs, "
                      f"95% CI [{rlo:+.2f}, {rhi:+.2f}].")
                if np.isfinite(rlo) and rlo <= 0 <= rhi:
                    print(f"    The interval includes zero, so the lag story is "
                          f"DIRECTIONALLY CONSISTENT but not")
                    print(f"    established - sixteen years is sixteen data points "
                          f"however many observations")
                    print(f"    sit inside them. Report it as a mechanism that fits "
                          f"the evidence, not a result.")
                if r < -0.3:
                    print(f"    Negative, as a trailing volatility feature predicts: "
                          f"after a bad year the")
                    print(f"    volatility reading is high but the fall has already "
                          f"happened, so the model")
                    print(f"    over-predicts; before a shock it reads calm and "
                          f"under-predicts. That is a")
                    print(f"    structural property of the feature, not a tuning "
                          f"problem.")
                elif r > 0.3:
                    print(f"    Positive, which is NOT the lag story - errors "
                          f"persist in the same direction")
                    print(f"    year to year. That points at a level bias in the "
                          f"calibration rather than a")
                    print(f"    timing effect.")
                else:
                    print(f"    Close to zero: no lag effect detectable at this "
                          f"sample size. The yearly")
                    print(f"    dispersion above is real but this particular "
                          f"explanation is not supported.")

        inv = [r for r in rec if np.isfinite(r["auc"]) and r["auc"] < 0.5]
        if inv:
            worst = min(inv, key=lambda r: r["auc"])
            print(f"\n    RANKING INVERTED in "
                  f"{', '.join(str(r['year']) for r in inv)} "
                  f"(worst {worst['year']}, AUC {worst['auc']:.3f}).")
            print(f"    In those years the names it called riskiest fell LESS often "
                  f"than the ones it")
            print(f"    called safest. Calibration can still be excellent there - "
                  f"the model knew the")
            print(f"    market was dangerous - while the ranking carried no "
                  f"information about WHICH")
            print(f"    names. A systematic shock hits everything, so there is "
                  f"nothing to rank; and a")
            print(f"    post-crash reading is high volatility with the drawdown "
                  f"already behind it.")
            print(f"    This is the limitation to put in the write-up. Aggregate "
                  f"skill is not the")
            print(f"    same as skill in the year you needed it.")


# =============================================================================
# CLI
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_scorecard_v61() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--obs-cache", default=None,
                    help="defaults to a name carrying the question, so a table "
                         "built for another threshold or horizon can never be "
                         "silently reused")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-train-years", type=float, default=MIN_TRAIN_YEARS)
    ap.add_argument("--embargo-days", type=int, default=None)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--isotonic", action="store_true",
                    help="force each yearly refit's curve to be "
                         "non-decreasing, to price what monotonicity buys")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="scorecard_v61.json")
    return ap


def run_scorecard_v61(cache=None, obs_cache=None, rebuild=None, step=None, min_train_years=None, embargo_days=None, max_tickers=None, n_boot=None, seed=None, isotonic=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_scorecard_v61()
        run_scorecard_v61(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "obs_cache": obs_cache, "rebuild": rebuild, "step": step, "min_train_years": min_train_years, "embargo_days": embargo_days, "max_tickers": max_tickers, "n_boot": n_boot, "seed": seed, "isotonic": isotonic, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    if cfg.quick:
        cfg.max_tickers, cfg.step, cfg.n_boot = 60, 20, 150
    embargo = cfg.embargo_days or P2.HORIZON_DAYS
    obs_cache = cfg.obs_cache or (f"obs_{int(P2.DRAWDOWN * 100)}pct_"
                                  f"{P2.HORIZON_DAYS}d.pkl")

    print("=" * 88)
    print("V61 - SCORING THE MODEL'S OUTPUT AT ITS DECISION POINTS")
    print("=" * 88)
    print(f"  question: a fall of {P2.DRAWDOWN:.0%} or more within "
          f"{P2.HORIZON_DAYS} days")
    print(f"  features: {'+'.join(P2.VOL_FEATURES)}   embargo {embargo}d")

    # the deployed tier boundaries, read from the pipeline rather than typed here
    tiers, src = [], "defaults"
    for mod in ("class_ai_pipeline_v3", "class_ai_pipeline_v2"):
        try:
            PL = __import__(mod)
            tiers = [(float(t[0]), str(t[2])) for t in PL.RISK_TIERS]
            src = mod
            break
        except Exception:
            continue
    if not tiers:
        tiers = [(0.12, "low"), (0.22, "moderate"), (0.31, "elevated")]
    print(f"  decision points: "
          f"{', '.join(f'{t:.0%} ({l})' for t, l in tiers)}  [from {src}]")

    if os.path.exists(obs_cache) and not cfg.rebuild:
        obs = pd.read_pickle(obs_cache)
        print(f"  loaded {len(obs):,} observations from {obs_cache}")
    else:
        tickers = sorted(os.path.splitext(os.path.basename(q))[0]
                         for q in glob.glob(os.path.join(cfg.cache, "*.pkl")))
        if cfg.max_tickers:
            tickers = tickers[:cfg.max_tickers]
        if not tickers:
            sys.exit(f"No .pkl files in {cfg.cache}.")
        print(f"  building observations from {len(tickers)} tickers "
              f"(step {cfg.step}) - slow, cached afterwards")
        obs = P2.build_observations(tickers, cfg.cache, cfg.step)
        if obs.empty:
            sys.exit("No observations built.")
        obs.to_pickle(obs_cache)
        print(f"  cached to {obs_cache}")

    obs = obs.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.sort_values("date").reset_index(drop=True)
    obs["event"] = obs["event"].astype(float)
    missing = [c for c in P2.VOL_FEATURES + ["vol60", "fwd_vol", "mdd"]
               if c not in obs.columns]
    if missing:
        sys.exit(f"the observation table is missing {missing} - it was built by an "
                 f"older module. Delete {obs_cache} and rerun with --rebuild.")
    print(f"  {len(obs):,} observations | {obs['ticker'].nunique()} tickers | "
          f"{obs['date'].min():%Y-%m} to {obs['date'].max():%Y-%m} | "
          f"base rate {obs['event'].mean():.1%}")

    print(f"\n  WALK-FORWARD (refit yearly, {embargo}-day embargo)")
    oos, fits = walkforward(obs, cfg.min_train_years, embargo,
                            P2.DRAWDOWN, P2.HORIZON_DAYS,
                            force_isotonic=cfg.isotonic)
    if oos is None or len(oos) < 500:
        sys.exit("not enough out-of-sample rows - lower --min-train-years")

    bad = fits[fits["monotone_violation_pp"] > 1e-9]
    if len(bad):
        worst = float(bad["monotone_violation_pp"].max())
        tail = ("  [isotonic applied]" if cfg.isotonic
                else "  Run --isotonic to price the fix.")
        print(f"    {len(bad)} of {len(fits)} refit curves run BACKWARDS somewhere "
              f"(worst step {worst:.1f}pp). P2.fit_drawdown_curve does not enforce "
              f"monotonicity; the shipped curve happens to be monotone, but a "
              f"refit on a training subset need not be.{tail}")
    else:
        print(f"    all {len(fits)} refit curves are non-decreasing")

    rows = scorecard(oos, P2.DRAWDOWN, P2.HORIZON_DAYS, tiers,
                     cfg.n_boot, cfg.seed)
    lad = ladder(oos, cfg.n_boot, cfg.seed, isotonic_applied=cfg.isotonic)
    regimes(oos, cfg.n_boot, cfg.seed,
            pooled_cal_err=V52.mae(oos['p_model'].to_numpy(float),
                                   oos['y'].to_numpy(float)))

    payload = {"question": {"drawdown": P2.DRAWDOWN,
                            "horizon_days": P2.HORIZON_DAYS,
                            "features": P2.VOL_FEATURES},
               "n_oos": int(len(oos)),
               "base_rate": float(oos["y"].mean()),
               "tiers": [{"threshold": t, "label": l} for t, l in tiers],
               "confusion": [{k: (None if isinstance(v, float)
                                  and not np.isfinite(v) else v)
                              for k, v in r.items()} for r in rows],
               "ladder": lad,
               "fits": fits.to_dict("records") if fits is not None else []}
    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_scorecard_v61(),
# and each value shown is that argument's own default, except QUICK which starts
# True so that a plain run finishes fast. Set it False for the real experiment.
#
# Passing any command-line flag still works and takes over, so the old CLI is
# not lost: it just is not the default way in any more.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    # <<< small, fast sanity run. False = the real thing.
    RUN_QUICK           = True
    RUN_CACHE           = 'price_cache_v43'
    # defaults to a name carrying the question, so a table built for another threshold or horizon can never be silently reused
    RUN_OBS_CACHE       = None
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_MIN_TRAIN_YEARS = 6
    RUN_EMBARGO_DAYS    = None
    RUN_MAX_TICKERS     = None
    RUN_N_BOOT          = 400
    RUN_SEED            = 61
    # force each yearly refit's curve to be non-decreasing, to price what monotonicity buys
    RUN_ISOTONIC        = False
    RUN_OUT             = 'scorecard_v61.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_scorecard_v61.py --quick
    else:
        run_scorecard_v61(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            obs_cache=RUN_OBS_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            embargo_days=RUN_EMBARGO_DAYS,
            max_tickers=RUN_MAX_TICKERS,
            n_boot=RUN_N_BOOT,
            seed=RUN_SEED,
            isotonic=RUN_ISOTONIC,
            out=RUN_OUT,
        )
