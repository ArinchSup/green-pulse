#!/usr/bin/env python3
"""
evaluate_sniper_v66.py - does a HIGH-CONFIDENCE, LOW-COVERAGE rule exist?

WHY THIS IS A NEW QUESTION, NOT A RERUN

Every V64 diagnostic measured the top DECILE - always ten percent of whatever it
was shown. A sniper is the opposite shape: it abstains on almost everything and
fires only where it is confident. Those are different claims, and the statistics
that killed the first one cannot see the second.

    AUC is an average over the whole ranking. A model can be worthless on 99% of
    candidates and genuinely sharp on the extreme 1%, and the AUC will still read
    0.50. Pooled decile tables miss it for the same reason: the top decile is
    25,000 trades, and a sniper that fires 1,500 times is 6% of that decile,
    diluted by the fifteen-out-of-sixteen it never wanted.

So: precision at low coverage, measured directly.

HOW THE THRESHOLD IS CHOSEN - AND WHY THAT MATTERS MOST

A threshold picked by looking at the test set is not a threshold, it is a
description. So in every walk-forward fold the cut is taken from the TRAINING
scores - "fire on the most confident 1% of what I saw while learning" - and then
applied, unchanged, to the next year. That is how a sniper would actually run:
you must decide how confident is confident enough BEFORE you see the outcomes.

WHAT WOULD COUNT AS A FIND

At the deployed geometry, R:R is about 1.67, break-even is 37.5%, and buying
blind wins about 42%. So the bar is not 50% and it is not break-even - it is
BEATING 42% BY ENOUGH TO PAY FOR ITSELF, with:

    - a win-rate confidence interval whose LOWER bound clears the blind rate.
      At 0.5% coverage the sample is small and the interval is wide; that is a
      fact about the evidence, not a technicality to wave past.
    - trades in most years. A rule that fires 400 times in 2019 and never again
      is a description of 2019.
    - a random-score control run through the identical pipeline. Pick the best
      of six thresholds on noise and something always looks good.

USAGE
  python evaluate_sniper_v66.py
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
from trade_config import HORIZON_CONFIGS

# =============================================================================
# CONFIG
# =============================================================================
PRICE_CACHE = E64.PRICE_CACHE
HORIZON = E64.HORIZON
STEP = E64.STEP
MIN_TRAIN_YEARS = E64.MIN_TRAIN_YEARS
SEED = 66
N_SEEDS = 3

# "fire on the most confident X% of the TRAINING scores". Coverage on the test
# set will differ, and that difference is itself informative: a model whose
# confident region shrinks out of sample was fitting the training years.
TRAIN_QUANTILES = (0.10, 0.05, 0.02, 0.01, 0.005, 0.002)

FEATURE_SET = "no_confirm"
LABEL_MODE = "barrier"          # "did it reach the target before the stop"
PREDICT_MODE = "classify"       # a sniper needs a probability, not a score


# =============================================================================
# STATISTICS
# =============================================================================
def _n_eff(n, lo, hi, lo_w, hi_w):
    """Effective sample size implied by how much wider the clustered interval is."""
    if not all(np.isfinite(x) for x in (lo, hi, lo_w, hi_w)):
        return np.nan
    w_wil, w_blk = (hi_w - lo_w) / 2, (hi - lo) / 2
    if w_wil <= 0 or w_blk <= 0:
        return np.nan
    return float(n / max((w_blk / w_wil) ** 2, 1.0))


def wilson(k, n, z=1.96):
    """
    Wilson interval for a proportion. Used instead of the normal approximation
    because at 0.2% coverage n can be a few hundred and the normal interval
    misbehaves exactly where this analysis lives.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def block_bootstrap_mean(values, dates, n_boot=400, seed=SEED, min_blocks=4):
    """
    Half-year block resample, because overlapping 90-day trades are not
    independent draws.

    A bootstrap needs blocks to resample. When every trade sits inside one or two
    half-years there is nothing to vary, and the interval collapses to something
    absurdly tight - the first run returned [75.5%, 76.2%] for a set of 208
    trades that were ALL in a single year. That is not precision, it is a
    degenerate resample, so it is refused instead of printed.
    """
    if len(values) < 30:
        return (np.nan, np.nan)
    if pd.PeriodIndex(pd.DatetimeIndex(dates), freq="6M").nunique() < min_blocks:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    blocks = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="6M")
    uniq = blocks.unique()
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        if len(idx) < 20:
            continue
        out.append(float(np.mean(np.asarray(values)[idx])))
    if not out:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# =============================================================================
# THE CURVE
# =============================================================================
def sniper_curve(d, features=None, quantiles=TRAIN_QUANTILES,
                 min_train_years=MIN_TRAIN_YEARS, n_seeds=N_SEEDS, seed=SEED,
                 horizon=HORIZON, random_scores=False, verbose=True):
    """
    Walk forward; in each fold take the cut from TRAIN scores and apply it to the
    test year. Returns one row per quantile, pooled over folds.
    """
    features = features or E64.resolve_features(FEATURE_SET)
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)

    d = d.sort_values("date").reset_index(drop=True)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    years = sorted(d["year"].unique())

    fired = {q: [] for q in quantiles}
    base_rows = []
    for y in years[min_train_years:]:
        start = pd.Timestamp(f"{y}-01-01")
        tr = d[pd.DatetimeIndex(d["date"]) < start - embargo]
        te = d[d["year"] == y]
        if len(tr) < 2000 or len(te) < 200:
            continue
        ps_tr, ps_te = [], []
        for s in range(n_seeds):
            ps_tr.append(E64._fit_predict(tr, tr, features, seed + s,
                                          mode=PREDICT_MODE,
                                          random_scores=random_scores))
            ps_te.append(E64._fit_predict(tr, te, features, seed + s,
                                          mode=PREDICT_MODE,
                                          random_scores=random_scores))
        p_tr = np.mean(ps_tr, axis=0)
        te = te.copy()
        te["p"] = np.mean(ps_te, axis=0)
        base_rows.append(te)
        for q in quantiles:
            cut = float(np.quantile(p_tr, 1.0 - q))
            hit = te[te["p"] >= cut]
            if len(hit):
                fired[q].append(hit.assign(_cut=cut))

    if not base_rows:
        raise RuntimeError("no usable folds")
    allte = pd.concat(base_rows, ignore_index=True)
    base_wr = float(allte["label"].mean())
    base_R = float(allte["r_multiple"].mean())

    rows = []
    for q in quantiles:
        if not fired[q]:
            rows.append({"train_q": q, "n": 0})
            continue
        f = pd.concat(fired[q], ignore_index=True)
        k = int(f["label"].sum())
        n = int(len(f))
        # Wilson assumes n INDEPENDENT draws. These are not: overlapping 90-day
        # windows across 337 correlated names, and the fired trades cluster in
        # time far harder than the full sample does. So the honest interval is a
        # half-year BLOCK bootstrap, which resamples whole periods and therefore
        # keeps the clustering. Wilson is kept alongside only to show how much
        # narrower the wrong assumption makes it look.
        lo_w, hi_w = wilson(k, n)
        lo, hi = block_bootstrap_mean(f["label"].to_numpy(float),
                                      f["date"].to_numpy(), seed=seed)
        rlo, rhi = block_bootstrap_mean(f["r_multiple"].to_numpy(float),
                                        f["date"].to_numpy(), seed=seed)
        # how concentrated in time? one busy quarter can be most of the sample
        per_q = f.groupby(pd.PeriodIndex(pd.DatetimeIndex(f["date"]),
                                         freq="Q")).size()
        per_year = f.groupby("year")["label"].agg(["size", "mean"])
        rows.append({
            "train_q": q, "n": n,
            "coverage": n / len(allte),
            "mean_cut": float(f["_cut"].mean()),
            "win_rate": k / n,
            "wr_lo": lo, "wr_hi": hi,                 # block bootstrap
            "wr_lo_wilson": lo_w, "wr_hi_wilson": hi_w,
            "top_quarter_share": float(per_q.max() / n),
            "n_quarters": int(len(per_q)),
            # How many INDEPENDENT bets is this really? Back it out of the two
            # intervals: the block interval is wider than Wilson by the square
            # root of the design effect, so n_eff = n / (width ratio)^2. At the
            # 2% cut, 1,676 trades turned out to be worth about 60.
            "n_eff": _n_eff(n, lo, hi, lo_w, hi_w),
            "expectancy_R": float(f["r_multiple"].mean()),
            "R_lo": rlo, "R_hi": rhi,
            "years_fired": int(len(per_year)),
            "years_total": int(allte["year"].nunique()),
            "years_wr_above_base": int((per_year["mean"] > base_wr).sum()),
            "min_per_year": int(per_year["size"].min()),
            "flat_share": float((f["outcome"] == "flat").mean()),
            "per_year": [{"year": int(y), "n": int(r_["size"]),
                          "win_rate": float(r_["mean"])}
                         for y, r_ in per_year.iterrows()],
        })
    return pd.DataFrame(rows), base_wr, base_R, len(allte)


def report(df, base_wr, base_R, n_all, breakeven, label="MODEL", out=sys.stdout):
    def w(*a):
        print(*a, file=out)
    w("\n" + "=" * 100)
    w(f"  {label}: WIN RATE AS THE RULE GETS PICKIER")
    w("=" * 100)
    w(f"  {n_all:,} out-of-sample candidates | blind win rate {base_wr:.1%} | "
      f"blind expectancy {base_R:+.3f}R | break-even {breakeven:.1%}")
    w("")
    w(f"  {'fire on top':>11s} {'n':>7s} {'cover':>7s} {'WIN RATE':>9s} "
      f"{'block-boot 95%':>17s} {'n_eff':>7s} {'exp R':>7s} "
      f"{'years':>7s} {'>base':>6s} {'busiest Q':>10s}")
    for _, q in df.iterrows():
        if not q.get("n"):
            w(f"  {q['train_q']:10.1%}  never fired")
            continue
        flag = ""
        if int(q["years_fired"]) <= 4:
            flag = f"  <-- ONLY {int(q['years_fired'])} YEAR(S). not a rule."
        elif np.isfinite(q["wr_lo"]) and q["wr_lo"] > base_wr:
            flag = "  <-- clears the blind rate"
        w(f"  {q['train_q']:10.1%} {int(q['n']):7,d} {q['coverage']:7.2%} "
          f"{q['win_rate']:9.1%} "
          f"{f'[{q.wr_lo:.1%}, {q.wr_hi:.1%}]' if np.isfinite(q.wr_lo) else 'too few blocks':>17s} "
          f"{(f'{q.n_eff:,.0f}' if np.isfinite(q.n_eff) else '-'):>7s} "
          f"{q['expectancy_R']:+7.3f} "
          f"{int(q['years_fired']):3d}/{int(q['years_total']):<3d} "
          f"{int(q['years_wr_above_base']):6d} "
          f"{q['top_quarter_share']:10.1%}{flag}")
    w(f"\n  block-boot resamples half-year blocks, so it keeps the clustering "
      f"Wilson ignores. n_eff is how")
    w(f"  many INDEPENDENT bets the sample is really worth, backed out of the "
      f"two interval widths.")
    w(f"  'busiest Q' is the share of fired trades landing in ONE quarter - the "
      f"random control runs")
    w(f"  under 2%, so anything far above that means the rule fires in bursts, "
      f"not on setups.")
    return df


def verdict(model_df, rnd_df, base_wr, out=sys.stdout):
    def w(*a):
        print(*a, file=out)
    w("\n" + "=" * 100)
    w("  VERDICT")
    w("=" * 100)
    live = model_df[model_df["n"] > 0]
    if live.empty:
        w("  the rule never fired at any threshold.")
        return
    best_wr = float(rnd_df[rnd_df["n"] > 0]["win_rate"].max()) if (
        rnd_df is not None and (rnd_df["n"] > 0).any()) else np.nan
    if np.isfinite(best_wr):
        w(f"  best win rate a RANDOM score reached through the same pipeline: "
          f"{best_wr:.1%}")
        w(f"  (six thresholds on noise, and the best of six always looks good)")
    # A threshold that fires in a handful of years is a description of those
    # years. On the first real run, top 0.2% posted 75.5% - all 208 trades in ONE
    # year out of seventeen - and top 0.5% put 285 trades in four years with only
    # one of them above the blind rate. Both are excluded here by rule, not by
    # judgement after the fact.
    ok = live[(live["wr_lo"] > base_wr)
              & (live["years_fired"] >= 0.7 * live["years_total"])
              & (live["years_wr_above_base"] >= 0.6 * live["years_fired"])
              & (live["top_quarter_share"] <= 0.35)]
    if np.isfinite(best_wr):
        ok = ok[ok["win_rate"] > best_wr]
    if ok.empty:
        w(f"\n  -> NO threshold clears all three tests at once: a lower bound "
          f"above the blind {base_wr:.1%},")
        w(f"     firing in most years, and beating the best noise threshold.")
        near = live.loc[live["win_rate"].idxmax()]
        w(f"     closest: fire on top {near['train_q']:.1%} -> "
          f"{near['win_rate']:.1%} on {int(near['n']):,} trades, "
          f"interval [{near['wr_lo']:.1%}, {near['wr_hi']:.1%}], "
          f"fired in {int(near['years_fired'])}/{int(near['years_total'])} years.")
        if np.isfinite(near["wr_lo"]) and near["wr_lo"] <= base_wr:
            w(f"     its interval includes the blind rate, so it is not "
              f"distinguishable from buying anything.")
    else:
        # Select on the LOWER BOUND, not the point estimate. Picking the highest
        # win rate picks the tightest threshold, which is the one with the fewest
        # trades and the widest interval - on the first run that was 78 trades
        # with a 21-point interval, chosen precisely because it was noisiest.
        b = ok.loc[ok["wr_lo"].idxmax()]
        w(f"\n  -> A SNIPER EXISTS at the top {b['train_q']:.1%} of training "
          f"confidence:")
        w(f"     {b['win_rate']:.1%} win rate on {int(b['n']):,} trades "
          f"(interval [{b['wr_lo']:.1%}, {b['wr_hi']:.1%}]), against a blind "
          f"{base_wr:.1%}.")
        w(f"     expectancy {b['expectancy_R']:+.3f}R, fired in "
          f"{int(b['years_fired'])}/{int(b['years_total'])} years, at least "
          f"{int(b['min_per_year'])} trades in every one of them.")
        w(f"     It covers {b['coverage']:.2%} of candidates - that is the "
          f"price of the precision, and it is")
        w(f"     the right shape for the product: it says nothing most days.")
        w(f"     (chosen by the best LOWER bound, not the best point estimate)")

        tight = live.loc[live["win_rate"].idxmax()]
        if tight["train_q"] != b["train_q"]:
            w(f"\n     A tighter cut (top {tight['train_q']:.1%}) shows "
              f"{tight['win_rate']:.1%} on only {int(tight['n']):,} trades, "
              f"interval")
            w(f"     [{tight['wr_lo']:.1%}, {tight['wr_hi']:.1%}] - "
              f"{100*(tight['wr_hi']-tight['wr_lo']):.0f} points wide. That "
              f"number is the more exciting one and the")
            w(f"     less trustworthy one; it is what selecting on a maximum "
              f"looks like.")
        py = pd.DataFrame(b["per_year"])
        w(f"\n     year by year at that cut:")
        w(f"     {'year':>6s} {'n':>6s} {'win rate':>9s}")
        for _, r_ in py.iterrows():
            mark = "  <-- below the blind rate" if r_["win_rate"] < base_wr else ""
            w(f"     {int(r_['year']):6d} {int(r_['n']):6d} "
              f"{r_['win_rate']:9.1%}{mark}")
        w(f"     busiest quarter holds {b['top_quarter_share']:.1%} of all "
          f"fired trades, spread over {int(b['n_quarters'])} quarters")

        if b["min_per_year"] < 8:
            w(f"\n     USABILITY: it fires as few as {int(b['min_per_year'])} "
              f"times in a year. Statistically that may")
            w(f"     be a signal; as a product it is a page that says nothing "
              f"for months. Loosen the cut")
            w(f"     until it fires enough to be worth opening, and re-read "
              f"the interval at THAT row.")


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_sniper_v66(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
                   quantiles=TRAIN_QUANTILES, feature_set=FEATURE_SET,
                   label_mode=LABEL_MODE, min_train_years=MIN_TRAIN_YEARS,
                   n_seeds=N_SEEDS, seed=SEED, tickers=None,
                   out="sniper_v66.json", verbose=True):
    print("=" * 100)
    print("V66 - THE SNIPER: IS THERE A HIGH-CONFIDENCE, LOW-COVERAGE RULE?")
    print("=" * 100)
    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    names = [t for t in (tickers or available) if t in available]
    print(f"  {len(names)} tickers | horizon {horizon} | features {feature_set} "
          f"| label {label_mode}")
    print(f"  thresholds taken from TRAIN scores each fold, applied unchanged "
          f"to the next year")

    d = E64.build_dataset(names, price_cache, horizon, step, verbose=verbose)
    if d.empty:
        raise RuntimeError("no candidate entries built")
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, label_mode)
    feats = E64.resolve_features(feature_set)
    rr = float(d["rr"].median())
    breakeven = 1.0 / (1.0 + rr)

    m_df, base_wr, base_R, n_all = sniper_curve(
        d, feats, quantiles, min_train_years, n_seeds, seed, horizon,
        random_scores=False, verbose=verbose)
    report(m_df, base_wr, base_R, n_all, breakeven, "MODEL")

    r_df, *_ = sniper_curve(d, feats, quantiles, min_train_years, 1, seed,
                            horizon, random_scores=True, verbose=False)
    report(r_df, base_wr, base_R, n_all, breakeven, "RANDOM SCORES (control)")

    verdict(m_df, r_df, base_wr)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": m_df.to_dict("records"),
                   "random": r_df.to_dict("records"),
                   "base_win_rate": base_wr, "base_R": base_R,
                   "breakeven": breakeven, "n": n_all}, f, indent=2,
                  default=float)
    print(f"\n  wrote {out}")
    return m_df, r_df


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--horizon", default=HORIZON, choices=list(HORIZON_CONFIGS))
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--label-mode", default=LABEL_MODE,
                    choices=["barrier", "profit"])
    ap.add_argument("--min-train-years", type=int, default=MIN_TRAIN_YEARS)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--out", default="sniper_v66.json")
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_sniper_v66(c.price_cache, c.horizon, c.step, TRAIN_QUANTILES,
                   c.feature_set, c.label_mode, c.min_train_years, c.n_seeds,
                   c.seed, c.tickers, c.out)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE     = PRICE_CACHE
    RUN_HORIZON         = HORIZON      # "SHORT" | "MID" | "LONG"
    RUN_STEP            = STEP
    # "fire on the most confident X% of TRAINING scores"
    RUN_QUANTILES       = TRAIN_QUANTILES
    RUN_FEATURE_SET     = FEATURE_SET  # no_confirm | per_name | +market | ...
    RUN_LABEL_MODE      = LABEL_MODE   # "barrier" = reached the target first
    RUN_MIN_TRAIN_YEARS = MIN_TRAIN_YEARS
    RUN_N_SEEDS         = N_SEEDS
    RUN_SEED            = SEED
    RUN_TICKERS         = None
    RUN_OUT             = "sniper_v66.json"
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_sniper_v66(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
                       step=RUN_STEP, quantiles=RUN_QUANTILES,
                       feature_set=RUN_FEATURE_SET, label_mode=RUN_LABEL_MODE,
                       min_train_years=RUN_MIN_TRAIN_YEARS,
                       n_seeds=RUN_N_SEEDS, seed=RUN_SEED,
                       tickers=RUN_TICKERS, out=RUN_OUT)
