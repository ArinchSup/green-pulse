#!/usr/bin/env python3
"""
evaluate_calibration_v52.py - is the drawdown probability actually correct?

WHAT THIS ASKS, AND HOW IT DIFFERS FROM V51

V51 asked whether sizing on the probability makes you better off. It does not:
random position sizes matched the calibrated ones on Calmar. That is a decision
problem, and it can fail while the forecast itself is fine - a forecaster who
says "30% chance of rain" and sees rain on 30% of those days is accurate whether
or not carrying an umbrella pays. Violent moves go both directions, so knowing
which names will move violently does not tell you how to size them.

This script tests the forecast on its own terms. No trading, no sizing, no
returns. Only: when the model says 12%, does it happen 12% of the time?

WHAT IS WRONG WITH THE 0.2pp NUMBER FROM THE FIRST CALIBRATION RUN

  FITTED AND GRADED ON THE SAME DATA.  The drawdown curve is twelve numbers
  learned from those 160,443 observations and then scored on them.

  POOLED ACROSS 2005-2026.  A model can be perfect on average and wrong every
  single year. If it says 10% always, and calm years deliver 2% while 2008 and
  2020 deliver 45%, the pooled error is zero and the forecast is useless - it
  never warned about anything. This is the failure mode that matters, and a
  pooled number cannot see it.

  CORRELATED EVENTS.  In March 2020 everything fell together. 160,443
  observations do not carry 160,443 observations' worth of information about
  drawdown risk; the effective count is closer to the number of distinct market
  episodes. The script estimates that directly.

WHAT IT RUNS

  WALKFORWARD (primary)  For each year, refit using only data whose outcomes had
                         fully resolved before that year began, then predict that
                         year. Yields ~15 genuinely out-of-sample years instead
                         of one test window, so the year-by-year table and the
                         tracking statistic are both fully OOS.

  SPLIT (secondary)      The simple version: fit through a cutoff, test after.
                         Easier to describe in a paper, far less powerful.

THE EMBARGO

A 180-day forward window means an observation dated near the cutoff has an
outcome that resolves after it. Training on those leaks the test period's
realizations into the fitted curve. Every fit here therefore ends HORIZON_DAYS
before the period it predicts. Without that purge the walk-forward is not out of
sample, it just looks like it.

THE STATISTIC THAT DECIDES IT

  TRACKING: across years, the correlation between mean predicted probability and
  the realised event rate. A model that anticipates elevated risk predicts more
  in the years that turn out badly, so tracking is high. A model that has learned
  only the long-run average says the same thing every year and tracks at zero -
  while still scoring well on pooled calibration. Tracking is what separates
  "knows the base rate" from "knows when risk is high", and it is the claim the
  thesis needs.

USAGE
  python evaluate_calibration_v52.py
  python evaluate_calibration_v52.py --cutoff 2023-12-31 --mode both
  python evaluate_calibration_v52.py --obs-cache obs_v52.pkl      # reuse the build
  python evaluate_calibration_v52.py --quick
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_pillar2_risk as P2

N_CURVE_BINS = 12       # matches the shipped calibration
N_REL_BINS = 8
N_BRIER_BINS = 10
MIN_TRAIN_YEARS = 6
BLOCK = "2QE"           # bootstrap block ~ the 180-day forward window


# =============================================================================
# METRICS
# =============================================================================
def predict(obs, calib):
    return np.array([P2.prob_drawdown(v, calib) for v in obs["vol60"].to_numpy()])


def auc(p, y):
    """Mann-Whitney U. Discrimination: can it rank which names draw down?"""
    p, y = np.asarray(p, float), np.asarray(y, bool)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier_decomp(p, y, n_bins=N_BRIER_BINS):
    """
    Murphy's decomposition:  BS = reliability - resolution + uncertainty

      reliability  how far predictions sit from the frequencies they imply. Lower
                   is better; zero means perfectly calibrated.
      resolution   how far the binned frequencies sit from the overall base rate.
                   HIGHER is better - it is the part that says the forecast
                   distinguishes situations at all.
      uncertainty  the base rate's own variance. A property of the data, not of
                   the model.

    A model that always outputs the base rate has reliability 0 AND resolution 0:
    flawless calibration, no information. That is the outcome to rule out, and it
    is why calibration alone is never enough.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 50:
        return None
    obar = y.mean()
    bs = float(np.mean((p - y) ** 2))
    b = pd.qcut(pd.Series(p).rank(method="first"), n_bins,
                labels=False, duplicates="drop").to_numpy()
    rel = res = 0.0
    for k in np.unique(b):
        m = b == k
        w = m.sum() / len(p)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - obar) ** 2
    return {"brier": bs, "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(obar * (1 - obar)), "base_rate": float(obar),
            "n": int(len(p))}


def skill(p, y, p_ref):
    """
    Brier skill against a constant forecast of p_ref. p_ref MUST come from the
    training period - using the test period's own base rate would let the
    reference forecast peek at the answer.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    bs = np.mean((p - y) ** 2)
    ref = np.mean((p_ref - y) ** 2)
    return float(1 - bs / ref) if ref > 0 else np.nan


def reliability_table(p, y, n_bins=N_REL_BINS):
    d = pd.DataFrame({"p": p, "y": np.asarray(y, float)}).dropna()
    if len(d) < n_bins * 20:
        return None
    d["b"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False,
                     duplicates="drop")
    g = d.groupby("b").agg(predicted=("p", "mean"), realised=("y", "mean"),
                           n=("y", "size")).reset_index(drop=True)
    g["error"] = (g["predicted"] - g["realised"]) * 100
    return g


def mae(p, y, n_bins=N_REL_BINS):
    t = reliability_table(p, y, n_bins)
    return float(t["error"].abs().mean()) if t is not None else np.nan


# =============================================================================
# CLUSTERED UNCERTAINTY
# =============================================================================
def blocks_of(dates):
    return pd.to_datetime(pd.Series(np.asarray(dates))).dt.to_period(
        BLOCK[:-1] if BLOCK.endswith("E") else BLOCK).astype(str).to_numpy()


N_BOOT = 1000               # set by --n-boot


def block_boot(dates, p, y, fn, n_boot=None, seed=0):
    """
    Resample whole half-year blocks, not rows. Drawdowns are correlated across
    names within a period, so row-level resampling treats one market episode as
    thousands of independent facts and produces intervals that are far too tight.
    """
    n_boot = n_boot or N_BOOT
    blk = blocks_of(dates)
    uniq = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[b] for b in pick])
        v = fn(p[sel], y[sel])
        if v is not None and np.isfinite(v):
            out.append(v)
    if len(out) < 50:
        return np.nan, np.nan, np.nan
    a = np.array(out)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def effective_n(dates, y, n_boot=None, seed=0):
    """
    How many independent observations do 160,000 correlated ones amount to?

    Compare the clustered standard error of the event rate with the binomial one
    that assumes independence. Their squared ratio is the design effect, and
    N / design effect is the honest sample size.
    """
    n_boot = n_boot or N_BOOT
    y = np.asarray(y, float)
    n = len(y)
    naive = np.sqrt(y.mean() * (1 - y.mean()) / n)
    blk = blocks_of(dates)
    uniq = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in uniq}
    rng = np.random.default_rng(seed)
    reps = [y[np.concatenate([idx[b] for b in
                              rng.choice(uniq, len(uniq), replace=True)])].mean()
            for _ in range(n_boot)]
    clustered = float(np.std(reps))
    deff = (clustered / naive) ** 2 if naive > 0 else np.nan
    return {"n": n, "n_blocks": len(uniq), "naive_se_pp": naive * 100,
            "clustered_se_pp": clustered * 100, "design_effect": float(deff),
            "n_effective": float(n / deff) if deff and np.isfinite(deff) else np.nan}


# =============================================================================
# YEAR VIEW
# =============================================================================
def by_year(dates, p, y):
    d = pd.DataFrame({"year": pd.to_datetime(pd.Series(np.asarray(dates))).dt.year,
                      "p": p, "y": np.asarray(y, float)}).dropna()
    g = d.groupby("year").agg(predicted=("p", "mean"), realised=("y", "mean"),
                              n=("y", "size")).reset_index()
    g["error"] = (g["predicted"] - g["realised"]) * 100
    return g[g["n"] >= 200].reset_index(drop=True)


def tracking(g):
    """
    Does predicted risk rise in the years that turn out badly?

    Pearson on the levels and Spearman on the ranks. Near zero means the model
    outputs roughly the same thing every year: it has learned the unconditional
    base rate, not when risk is elevated. A pooled calibration table cannot
    distinguish those two; this can.
    """
    if len(g) < 5:
        return None
    a, b = g["predicted"].to_numpy(), g["realised"].to_numpy()
    sp = (pd.Series(a).rank().corr(pd.Series(b).rank()))
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(sp),
            "n_years": int(len(g)),
            "pred_spread_pp": float((a.max() - a.min()) * 100),
            "real_spread_pp": float((b.max() - b.min()) * 100)}


# =============================================================================
# SPLITS
# =============================================================================
def fit_calib(obs, n_bins=N_CURVE_BINS):
    sh = P2.fit_vol_shrinkage(obs)
    cv = P2.fit_drawdown_curve(obs, sh, n_bins)
    return {"shrinkage": sh, "curve": cv, "n_observations": int(len(obs))}


def walkforward(obs, min_train_years, embargo_days, verbose=True):
    """
    Refit per year on data whose outcomes had resolved before the year opened.
    Returns the pooled OOS predictions plus a per-year record of the fit.
    """
    yrs = sorted(obs["date"].dt.year.unique())
    first = yrs[0] + int(min_train_years)
    keep, fits = [], []
    for yr in [y for y in yrs if y >= first]:
        opens = pd.Timestamp(year=yr, month=1, day=1)
        tr = obs[obs["date"] <= opens - pd.Timedelta(days=embargo_days)]
        te = obs[(obs["date"] >= opens) &
                 (obs["date"] < opens + pd.DateOffset(years=1))]
        if len(tr) < 5000 or len(te) < 200:
            continue
        cal = fit_calib(tr)
        p = predict(te, cal)
        keep.append(pd.DataFrame({"date": te["date"].to_numpy(), "p": p,
                                  "y": te["event"].to_numpy(float)}))
        fits.append({"year": yr, "n_train": len(tr), "n_test": len(te),
                     "train_base_rate": float(tr["event"].mean()),
                     "slope": cal["shrinkage"]["b"],
                     "intercept": cal["shrinkage"]["a"],
                     "r2": cal["shrinkage"]["r2"]})
        if verbose:
            print(f"    {yr}: trained on {len(tr):,} (to "
                  f"{opens - pd.Timedelta(days=embargo_days):%Y-%m}), "
                  f"predicted {len(te):,}")
    if not keep:
        return None, None
    return pd.concat(keep, ignore_index=True), pd.DataFrame(fits)


# =============================================================================
# REPORTING
# =============================================================================
def report(name, dates, p, y, p_ref, seed, insample_mae=None):
    print("\n" + "=" * 92)
    print(f"{name}")
    print("=" * 92)
    n = int(np.isfinite(p).sum())
    print(f"  {n:,} predictions | base rate {np.mean(y):.1%} | "
          f"reference forecast {p_ref:.1%} (from training data)")

    t = reliability_table(p, y)
    if t is not None:
        print(f"\n  RELIABILITY\n    {'predicted':>10}{'realised':>10}"
              f"{'error pp':>10}{'n':>9}")
        for _, r in t.iterrows():
            print(f"    {r['predicted']:>9.1%}{r['realised']:>10.1%}"
                  f"{r['error']:>+10.1f}{int(r['n']):>9,}")
        m = t["error"].abs().mean()
        mm, lo, hi = block_boot(dates, p, y, lambda a, b: mae(a, b), seed=seed)
        print(f"    mean absolute calibration error: {m:.2f}pp"
              f"   95% CI (half-year blocks) {lo:.2f} to {hi:.2f}pp")
        if insample_mae is not None:
            print(f"    in-sample was {insample_mae:.2f}pp - the gap is the "
                  f"optimism that fitting and grading on one sample buys")

    bd = brier_decomp(p, y)
    if bd:
        ss = skill(p, y, p_ref)
        _, slo, shi = block_boot(dates, p, y,
                                 lambda a, b: skill(a, b, p_ref), seed=seed)
        print(f"\n  BRIER  {bd['brier']:.5f}  = reliability {bd['reliability']:.5f}"
              f"  - resolution {bd['resolution']:.5f}"
              f"  + uncertainty {bd['uncertainty']:.5f}")
        print(f"    skill vs a constant {p_ref:.1%} forecast: {ss:+.4f}"
              f"   95% CI {slo:+.4f} to {shi:+.4f}")
        print("    skill at or below 0 means the model adds nothing over always")
        print("    predicting the training base rate.")
        a = auc(p, y)
        _, alo, ahi = block_boot(dates, p, y, auc, seed=seed)
        print(f"    AUC {a:.4f}   95% CI {alo:.4f} to {ahi:.4f}"
              f"    (0.50 = no ability to rank)")
    return t, bd


def year_report(g, label):
    print(f"\n  BY YEAR - {label}")
    print(f"    {'year':<7}{'predicted':>11}{'realised':>10}{'error pp':>10}{'n':>9}")
    for _, r in g.iterrows():
        print(f"    {int(r['year']):<7}{r['predicted']:>10.1%}{r['realised']:>10.1%}"
              f"{r['error']:>+10.1f}{int(r['n']):>9,}")
    print(f"    worst year error {g['error'].abs().max():.1f}pp | "
          f"mean absolute {g['error'].abs().mean():.1f}pp | "
          f"years within 5pp: {(g['error'].abs() <= 5).sum()}/{len(g)}")
    tr = tracking(g)
    if tr:
        print(f"\n  TRACKING  predicted vs realised across {tr['n_years']} years")
        print(f"    Pearson {tr['pearson']:+.3f} | Spearman {tr['spearman']:+.3f}")
        print(f"    the model's yearly forecast moves over a "
              f"{tr['pred_spread_pp']:.1f}pp range; reality moved over "
              f"{tr['real_spread_pp']:.1f}pp")
        if tr["pred_spread_pp"] < tr["real_spread_pp"] / 3:
            print("    the forecast is far flatter than reality - it is close to")
            print("    predicting a constant, whatever the pooled table shows")
        if tr["pearson"] - tr["spearman"] > 0.3:
            print("    Pearson well above Spearman: the correlation rests on a few")
            print("    extreme years. It flags the crises and cannot order the")
            print("    ordinary ones - worth saying, since a user of the deployed")
            print("    model sees mostly ordinary years.")
        elif tr["spearman"] - tr["pearson"] > 0.3:
            print("    Spearman above Pearson: the ordering is right but the")
            print("    magnitudes are off, which recalibration could fix.")
    return tr


# =============================================================================
# MAIN
# =============================================================================
def _parser():
    """Every CLI flag lives here so run_calibration_v52() and main() share one
    definition of the defaults."""
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=P2.PRICE_CACHE)
    ap.add_argument("--obs-cache", default="obs_v52.pkl",
                    help="observation table is slow to build; cached here")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--cutoff", default="2023-12-31")
    ap.add_argument("--mode", default="both",
                    choices=["walkforward", "split", "both"])
    ap.add_argument("--min-train-years", type=float, default=MIN_TRAIN_YEARS)
    ap.add_argument("--embargo-days", type=int, default=P2.HORIZON_DAYS)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="block-bootstrap replicates; lower it if the run drags")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="calibration_eval_v52.json")
    return ap


def run_calibration_v52(cache=None, obs_cache=None, rebuild=None, step=None, cutoff=None, mode=None, min_train_years=None, embargo_days=None, max_tickers=None, seed=None, n_boot=None, quick=None, out=None):
    """
    Run this experiment from code. Every CLI flag is a keyword argument; leave
    one as None to keep its default.

        run_calibration_v52()
        run_calibration_v52(cache=...)

    Returns whatever the experiment returns (usually None - the finding is in
    what it prints). Raises SystemExit on a fatal input problem, same as the CLI.
    """
    cfg = _parser().parse_args([])
    for _k, _v in {"cache": cache, "obs_cache": obs_cache, "rebuild": rebuild, "step": step, "cutoff": cutoff, "mode": mode, "min_train_years": min_train_years, "embargo_days": embargo_days, "max_tickers": max_tickers, "seed": seed, "n_boot": n_boot, "quick": quick, "out": out}.items():
        if _v is not None:
            setattr(cfg, _k, _v)
    return _run(cfg)


def _run(cfg):
    globals()["N_BOOT"] = cfg.n_boot
    if cfg.quick:
        cfg.max_tickers, cfg.step = 60, 20

    print("=" * 92)
    print("V52 - IS THE DRAWDOWN PROBABILITY CORRECT?   (no trading, no sizing)")
    print("=" * 92)

    if os.path.exists(cfg.obs_cache) and not cfg.rebuild:
        obs = pd.read_pickle(cfg.obs_cache)
        print(f"  loaded {len(obs):,} observations from {cfg.obs_cache} "
              f"(--rebuild to redo)")
    else:
        import glob
        tickers = sorted(os.path.splitext(os.path.basename(p))[0]
                         for p in glob.glob(os.path.join(cfg.cache, "*.pkl")))
        if cfg.max_tickers:
            tickers = tickers[:cfg.max_tickers]
        if not tickers:
            sys.exit(f"No .pkl files in {cfg.cache}.")
        print(f"  building observations from {len(tickers)} tickers "
              f"(step {cfg.step})")
        obs = P2.build_observations(tickers, cfg.cache, cfg.step)
        obs.to_pickle(cfg.obs_cache)
        print(f"  cached to {cfg.obs_cache}")

    if obs.empty:
        sys.exit("No observations.")
    obs = obs.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.sort_values("date").reset_index(drop=True)
    obs["event"] = obs["event"].astype(bool)
    print(f"  {len(obs):,} observations | {obs['ticker'].nunique()} tickers | "
          f"{obs['date'].min():%Y-%m} -> {obs['date'].max():%Y-%m}")
    print(f"  base rate {obs['event'].mean():.1%} | "
          f"event = a fall of {P2.DRAWDOWN:.0%} or more within "
          f"{P2.HORIZON_DAYS} days")

    en = effective_n(obs["date"].to_numpy(), obs["event"].to_numpy(float),
                     seed=cfg.seed)
    print(f"\n  HOW MUCH DATA IS REALLY HERE")
    print(f"    {en['n']:,} rows across {en['n_blocks']} half-year periods")
    print(f"    naive SE on the base rate {en['naive_se_pp']:.3f}pp | "
          f"clustered {en['clustered_se_pp']:.3f}pp")
    print(f"    design effect {en['design_effect']:.0f}x -> effectively about "
          f"{en['n_effective']:,.0f} independent observations")
    print("    Drawdowns arrive together, so row counts overstate precision by")
    print("    this factor. Every interval below uses the clustered version.")

    # the shipped, in-sample number, for the comparison
    full = fit_calib(obs)
    p_in = predict(obs, full)
    mae_in = mae(p_in, obs["event"].to_numpy(float))
    print(f"\n  in-sample (fit and graded on everything): MAE {mae_in:.2f}pp, "
          f"vol R2 {full['shrinkage']['r2']:.3f}")

    results = {"observations": len(obs), "effective_n": en,
               "in_sample_mae_pp": mae_in}

    if cfg.mode in ("walkforward", "both"):
        print("\n" + "=" * 92)
        print("WALK-FORWARD: refit each year on resolved data only")
        print("=" * 92)
        oos, fits = walkforward(obs, cfg.min_train_years, cfg.embargo_days)
        if oos is None:
            print("  not enough span for walk-forward")
        else:
            ref = float(fits["train_base_rate"].mean())
            t, bd = report("POOLED OUT-OF-SAMPLE (walk-forward)",
                           oos["date"].to_numpy(), oos["p"].to_numpy(),
                           oos["y"].to_numpy(), ref, cfg.seed, mae_in)
            g = by_year(oos["date"].to_numpy(), oos["p"].to_numpy(),
                        oos["y"].to_numpy())
            tr = year_report(g, "every year out of sample")
            results["walkforward"] = {
                "pooled_mae_pp": mae(oos["p"].to_numpy(), oos["y"].to_numpy()),
                "brier": bd, "tracking": tr,
                "by_year": g.to_dict("records"),
                "fits": fits.to_dict("records")}

    if cfg.mode in ("split", "both"):
        print("\n" + "=" * 92)
        print(f"SINGLE SPLIT: fit to {cfg.cutoff} minus {cfg.embargo_days}d "
              f"embargo, test after {cfg.cutoff}")
        print("=" * 92)
        cut = pd.Timestamp(cfg.cutoff)
        tr_ = obs[obs["date"] <= cut - pd.Timedelta(days=cfg.embargo_days)]
        te_ = obs[obs["date"] > cut]
        print(f"  train {len(tr_):,} (to {tr_['date'].max():%Y-%m}) | "
              f"test {len(te_):,} ({te_['date'].min():%Y-%m} -> "
              f"{te_['date'].max():%Y-%m})")
        if len(tr_) < 5000 or len(te_) < 500:
            print("  test window too small to say anything")
        else:
            cal = fit_calib(tr_)
            p = predict(te_, cal)
            yv = te_["event"].to_numpy(float)
            ref = float(tr_["event"].mean())
            print(f"  refitted: forward_vol = {cal['shrinkage']['a']:.2f} + "
                  f"{cal['shrinkage']['b']:.3f} x trailing_60d  "
                  f"(R2 {cal['shrinkage']['r2']:.3f})")
            t2, bd2 = report(f"OUT-OF-SAMPLE {te_['date'].min():%Y-%m} to "
                             f"{te_['date'].max():%Y-%m}",
                             te_["date"].to_numpy(), p, yv, ref, cfg.seed, mae_in)
            results["split"] = {"cutoff": cfg.cutoff, "n_train": len(tr_),
                                "n_test": len(te_), "brier": bd2,
                                "mae_pp": mae(p, yv)}
            if len(te_["date"].dt.year.unique()) >= 2:
                year_report(by_year(te_["date"].to_numpy(), p, yv),
                            "test window only")

    print("\n" + "=" * 92)
    print("HOW TO READ IT")
    print("=" * 92)
    print("  Three things have to hold together. Any one of them failing changes")
    print("  what you are allowed to claim.")
    print("\n  1. POOLED OOS CALIBRATION within a few pp, CI not far from it")
    print("       -> the forecast is honest out of sample. Necessary, not")
    print("          sufficient: a constant forecast passes this too.")
    print("  2. BRIER SKILL above 0, CI excluding 0")
    print("       -> it beats always predicting the base rate. If skill is at or")
    print("          below zero, the model IS the base rate with extra steps, and")
    print("          the resolution term will be near zero to confirm it.")
    print("  3. TRACKING positive across years, forecast spread comparable to")
    print("     reality's spread")
    print("       -> it knows WHEN risk is elevated, not just the long-run rate.")
    print("          This is the claim worth defending. Without it the honest")
    print("          sentence is: the model estimates the average drawdown rate")
    print("          but does not anticipate when risk is high.")
    print("\n  If the by-year table swings from large negative errors in calm")
    print("  years to large positive ones in 2008, 2020 and 2022, the pooled")
    print("  number is an average of two opposite failures, and it is the year")
    print("  table that should go in the paper.")
    print("\n  SURVIVORSHIP still applies, and here it biases toward UNDER-")
    print("  predicting: the names absent from the cache are the ones that")
    print("  collapsed. Realised rates below are lower than the population's, so")
    print("  a well-calibrated-looking model is calibrated to survivors.")

    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  wrote {cfg.out}")



def main(argv=None):
    """Thin: parse, then hand off to _run()."""
    return _run(_parser().parse_args(argv))


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed. Edit the values below to change
# what it does. Each RUN_X below is the keyword argument x of run_calibration_v52(),
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
    RUN_OBS_CACHE       = 'obs_v52.pkl'     # observation table is slow to build; cached here
    RUN_REBUILD         = False
    RUN_STEP            = 10
    RUN_CUTOFF          = '2023-12-31'
    RUN_MODE            = 'both'
    RUN_MIN_TRAIN_YEARS = 6
    RUN_EMBARGO_DAYS    = 180
    RUN_MAX_TICKERS     = None
    RUN_SEED            = 56
    RUN_N_BOOT          = 1000              # block-bootstrap replicates; lower it if the run drags
    RUN_OUT             = 'calibration_eval_v52.json'
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g.  python evaluate_calibration_v52.py --quick
    else:
        run_calibration_v52(
            quick=RUN_QUICK,
            cache=RUN_CACHE,
            obs_cache=RUN_OBS_CACHE,
            rebuild=RUN_REBUILD,
            step=RUN_STEP,
            cutoff=RUN_CUTOFF,
            mode=RUN_MODE,
            min_train_years=RUN_MIN_TRAIN_YEARS,
            embargo_days=RUN_EMBARGO_DAYS,
            max_tickers=RUN_MAX_TICKERS,
            seed=RUN_SEED,
            n_boot=RUN_N_BOOT,
            out=RUN_OUT,
        )
