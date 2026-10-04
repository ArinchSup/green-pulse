#!/usr/bin/env python3
"""
evaluate_filters_v67.py - can a second filter straighten the sniper's results?

THE IDEA, AND THE TRAP

V66 found a usable cut: fire on the top 10% of training confidence, 12,806
out-of-sample trades, 47.0% against a blind 42.1%, spread across all 17 years.
Its weakness is not the win rate, it is the EVIDENCE: n_eff about 245, so the
honest interval is [40.5%, 52.8%] and contains the baseline.

The obvious next move is to filter further. It is also the most reliable way to
produce a backtest that does not work, for a reason that is pure arithmetic:

    a filter that keeps half the trades halves n_eff. Two filters leave a
    quarter. At n_eff 60 a 25-point interval becomes a 40-point one, and a
    3-point gain in win rate is invisible inside it.

And if the filter is chosen by trying several and keeping the one that helped,
the search itself manufactures the result. With ten candidate filters and this
much effective data, two or three will look good by chance.

SO THE BAR IS THE LOWER BOUND, NOT THE WIN RATE

A filter earns its place only if the CLUSTERED LOWER BOUND goes up. A filter that
lifts the point estimate while cutting n_eff usually lowers the bound, which means
you know less than you did before. That column is the verdict; everything else is
decoration.

WHAT IS TESTED, AND WHY EACH ONE IS HERE

Every filter below has a reason that was written before its effect was measured.
None was chosen because it helped.

    spaced        cap trades per calendar week, keeping the highest-scored.
                  Reason: the known defect is bursts - at tighter cuts 45-99% of
                  trades landed in ONE quarter. This attacks that directly
                  instead of chasing return.
    liquid        above-median dollar volume that day. Reason: friction and
                  capacity, not prediction.
    not_extended  drop the most stretched names vs their 20-day EMA. Reason:
                  Pillar 2 already flags EXTENDED; buying a vertical move is a
                  worse entry at the same forecast.
    mid_vol       drop the top and bottom volatility quintiles. Reason: the
                  barriers are CLAMPED at both ends, so the trade being graded
                  there is not the trade the geometry intended.
    calm_market   only when market volatility is below its median. Reason: this
                  is the CONFOUND, included on purpose. It should raise the win
                  rate and make the clustering worse, and it is here to show
                  what that looks like.

THE CONTROL THAT MATTERS

Every filter is matched against RANDOM filters that keep the same fraction of
trades, several draws each. A random filter of matched size is what "no
information, same sample cost" looks like. A real filter has to beat that, not
beat zero.

USAGE
  python evaluate_filters_v67.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import evaluate_sniper_v66 as S66
from trade_config import HORIZON_CONFIGS

# =============================================================================
# CONFIG
# =============================================================================
PRICE_CACHE = E64.PRICE_CACHE
HORIZON = E64.HORIZON
STEP = E64.STEP
MIN_TRAIN_YEARS = E64.MIN_TRAIN_YEARS
SEED = 67
N_SEEDS = 3

BASE_QUANTILE = 0.10        # the V66 cut this builds on
FEATURE_SET = "no_confirm"
LABEL_MODE = "barrier"

MAX_PER_WEEK = 3            # for the `spaced` filter
N_RANDOM_DRAWS = 8          # matched-size random filters per real filter


# =============================================================================
# FILTERS  - each returns a boolean mask over the fired trades
# =============================================================================
def f_spaced(f, max_per_week=MAX_PER_WEEK):
    """Keep at most `max_per_week` trades per calendar week, highest score first."""
    wk = pd.PeriodIndex(pd.DatetimeIndex(f["date"]), freq="W")
    keep = np.zeros(len(f), bool)
    order = f.assign(_wk=wk).sort_values("p", ascending=False)
    taken = {}
    for pos, (_, r) in zip(order.index, order.iterrows()):
        w = r["_wk"]
        if taken.get(w, 0) < max_per_week:
            taken[w] = taken.get(w, 0) + 1
            keep[f.index.get_loc(pos)] = True
    return keep


def _above_median_in_date(f, col):
    return (f[col] > f.groupby("date")[col].transform("median")).to_numpy(bool)


def f_liquid(f):
    return _above_median_in_date(f, "log_dollar_vol")


def f_not_extended(f):
    q = f.groupby("date")["px_vs_ema20"].transform(lambda s: s.quantile(0.80))
    return (f["px_vs_ema20"] <= q).to_numpy(bool)


def f_mid_vol(f):
    lo = f.groupby("date")["yz66"].transform(lambda s: s.quantile(0.20))
    hi = f.groupby("date")["yz66"].transform(lambda s: s.quantile(0.80))
    return ((f["yz66"] >= lo) & (f["yz66"] <= hi)).to_numpy(bool)


def f_calm_market(f):
    med = float(np.median(f["mkt_vol_20"]))
    return (f["mkt_vol_20"] <= med).to_numpy(bool)


FILTERS = {
    "spaced": f_spaced,
    "liquid": f_liquid,
    "not_extended": f_not_extended,
    "mid_vol": f_mid_vol,
    "calm_market": f_calm_market,
}

# combinations worth one look, named up front rather than searched
COMBOS = {
    "spaced+liquid": ("spaced", "liquid"),
    "spaced+mid_vol": ("spaced", "mid_vol"),
}


# =============================================================================
# SCORING
# =============================================================================
def score(f, base_wr, label, seed=SEED):
    """Everything that matters about a filtered set, in one row."""
    n = len(f)
    if n < 60:
        return {"filter": label, "n": n, "note": "too few trades to score"}
    k = int(f["label"].sum())
    lo_w, hi_w = S66.wilson(k, n)
    lo, hi = S66.block_bootstrap_mean(f["label"].to_numpy(float),
                                      f["date"].to_numpy(), seed=seed)
    per_q = f.groupby(pd.PeriodIndex(pd.DatetimeIndex(f["date"]),
                                     freq="Q")).size()
    per_year = f.groupby("year")["label"].agg(["size", "mean"])
    return {"filter": label, "n": n,
            "n_eff": S66._n_eff(n, lo, hi, lo_w, hi_w),
            "win_rate": k / n, "wr_lo": lo, "wr_hi": hi,
            "expectancy_R": float(f["r_multiple"].mean()),
            "years_fired": int(len(per_year)),
            "years_above_base": int((per_year["mean"] > base_wr).sum()),
            "top_quarter_share": float(per_q.max() / n),
            "min_per_year": int(per_year["size"].min())}


def fire_base(d, features, quantile=BASE_QUANTILE, min_train_years=MIN_TRAIN_YEARS,
              n_seeds=N_SEEDS, seed=SEED, horizon=HORIZON, verbose=True):
    """Reproduce the V66 cut and return the fired trades plus the blind baseline."""
    n_bars = HORIZON_CONFIGS[horizon]["lookahead_bars"]
    embargo = pd.Timedelta(days=int(n_bars * 365.25 / 252) + 5)
    d = d.sort_values("date").reset_index(drop=True)
    d["year"] = pd.DatetimeIndex(d["date"]).year
    years = sorted(d["year"].unique())

    fired, alltest = [], []
    for y in years[min_train_years:]:
        start = pd.Timestamp(f"{y}-01-01")
        tr = d[pd.DatetimeIndex(d["date"]) < start - embargo]
        te = d[d["year"] == y]
        if len(tr) < 2000 or len(te) < 200:
            continue
        ptr = np.mean([E64._fit_predict(tr, tr, features, seed + s,
                                       mode="classify") for s in range(n_seeds)],
                      axis=0)
        te = te.copy()
        te["p"] = np.mean([E64._fit_predict(tr, te, features, seed + s,
                                           mode="classify")
                           for s in range(n_seeds)], axis=0)
        alltest.append(te)
        cut = float(np.quantile(ptr, 1.0 - quantile))
        hit = te[te["p"] >= cut]
        if len(hit):
            fired.append(hit)
        if verbose:
            print(f"    {y}: {len(hit):,} fired of {len(te):,}")
    if not fired:
        raise RuntimeError("the base cut never fired")
    allte = pd.concat(alltest, ignore_index=True)
    return (pd.concat(fired, ignore_index=True),
            float(allte["label"].mean()), float(allte["r_multiple"].mean()))


# =============================================================================
# TOP LEVEL
# =============================================================================
def run_filters_v67(price_cache=PRICE_CACHE, horizon=HORIZON, step=STEP,
                    quantile=BASE_QUANTILE, feature_set=FEATURE_SET,
                    label_mode=LABEL_MODE, min_train_years=MIN_TRAIN_YEARS,
                    n_seeds=N_SEEDS, seed=SEED, n_random=N_RANDOM_DRAWS,
                    tickers=None, out="filters_v67.json", verbose=True):
    print("=" * 104)
    print("V67 - CAN A SECOND FILTER STRAIGHTEN THE SNIPER?")
    print("=" * 104)
    available = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(price_cache) if f.endswith(".pkl"))
    names = [t for t in (tickers or available) if t in available]
    print(f"  {len(names)} tickers | base cut = top {quantile:.0%} of train "
          f"confidence | features {feature_set}")

    d = E64.build_dataset(names, price_cache, horizon, step, verbose=verbose)
    if d.empty:
        raise RuntimeError("no candidate entries built")
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, label_mode)
    feats = E64.resolve_features(feature_set)

    print(f"\n  reproducing the base cut")
    f0, base_wr, base_R = fire_base(d, feats, quantile, min_train_years,
                                    n_seeds, seed, horizon, verbose)
    rows = [score(f0, base_wr, "NONE (the V66 cut)", seed)]
    print(f"\n  blind win rate {base_wr:.1%} | blind expectancy {base_R:+.3f}R")

    # ---- real filters -------------------------------------------------------
    specs = {k: (v,) for k, v in FILTERS.items()}
    specs.update({k: tuple(FILTERS[n] for n in v) for k, v in COMBOS.items()})
    for label, fns in specs.items():
        m = np.ones(len(f0), bool)
        for fn in fns:
            m &= fn(f0)
        rows.append(score(f0[m].reset_index(drop=True), base_wr, label, seed))

    # ---- matched-size random filters ---------------------------------------
    rng = np.random.default_rng(seed)
    rnd_rows = []
    for label in list(FILTERS) + list(COMBOS):
        keep = [r for r in rows if r["filter"] == label]
        if not keep or not keep[0].get("n"):
            continue
        frac = keep[0]["n"] / len(f0)
        for draw in range(n_random):
            m = rng.random(len(f0)) < frac
            r = score(f0[m].reset_index(drop=True), base_wr,
                      f"random@{frac:.2f}", seed + draw)
            r["matched_to"] = label
            rnd_rows.append(r)

    base = rows[0]
    print("\n" + "=" * 104)
    print("  DOES THE FILTER RAISE THE LOWER BOUND?  (that is the only question)")
    print("=" * 104)
    print(f"  {'filter':18s} {'n':>7s} {'keep':>6s} {'n_eff':>7s} "
          f"{'win rate':>9s} {'clustered 95%':>17s} {'bound vs base':>14s} "
          f"{'exp R':>7s} {'busyQ':>6s} {'yrs':>6s}")
    for r in rows:
        if not r.get("n_eff"):
            print(f"  {r['filter']:18s} {r.get('n', 0):7,d}  {r.get('note', '')}")
            continue
        d_bound = r["wr_lo"] - base["wr_lo"]
        ci = f"[{r['wr_lo']:.1%}, {r['wr_hi']:.1%}]"
        tag = ""
        if r["filter"] != base["filter"]:
            tag = ("  BETTER" if d_bound > 0.005
                   else "  worse" if d_bound < -0.005 else "  no change")
        print(f"  {r['filter']:18s} {int(r['n']):7,d} "
              f"{r['n'] / base['n']:6.0%} {r['n_eff']:7,.0f} "
              f"{r['win_rate']:9.1%} "
              f"{ci:>17s} "
              f"{d_bound:+14.1%} {r['expectancy_R']:+7.3f} "
              f"{r['top_quarter_share']:6.1%} "
              f"{int(r['years_fired']):3d}/17{tag}")

    if rnd_rows:
        rr = pd.DataFrame([r for r in rnd_rows if r.get("n_eff")])
        print(f"\n  MATCHED RANDOM FILTERS ({n_random} draws each): what keeping "
              f"the same FRACTION of trades")
        print(f"  does when the filter knows nothing.")
        print(f"  {'matched to':18s} {'keep':>6s} {'win rate range':>20s} "
              f"{'bound range':>20s}")
        for label, g in rr.groupby("matched_to"):
            print(f"  {label:18s} {g['n'].mean() / base['n']:6.0%} "
                  f"{f'{g.win_rate.min():.1%} to {g.win_rate.max():.1%}':>20s} "
                  f"{f'{g.wr_lo.min():.1%} to {g.wr_lo.max():.1%}':>20s}")

    real = [r for r in rows[1:] if r.get("n_eff")
            and r["wr_lo"] > base["wr_lo"] + 0.005]
    print("\n" + "=" * 104)
    print("  VERDICT")
    print("=" * 104)
    print(f"  base cut: {base['win_rate']:.1%} on {base['n']:,} trades, "
          f"n_eff {base['n_eff']:,.0f}, bound {base['wr_lo']:.1%}")
    print(f"  {len(specs)} filters were tested. With this much effective data, "
          f"expect one or two to look")
    print(f"  good by chance - which is what the matched-random ranges above "
          f"are for.")
    if not real:
        print(f"\n  -> NO filter raises the lower bound. Every one of them "
              f"costs more evidence than it")
        print(f"     adds information. The base cut is the best version of "
              f"this rule, and the way to")
        print(f"     improve it is MORE NAMES, not more conditions.")
    else:
        best = max(real, key=lambda r: r["wr_lo"])
        rr = pd.DataFrame([r for r in rnd_rows
                           if r.get("matched_to") == best["filter"]
                           and r.get("n_eff")])
        beat = (float(rr["wr_lo"].max()) if len(rr) else -np.inf)
        print(f"\n  -> {len(real)} filter(s) raise the bound. Best: "
              f"{best['filter']}")
        print(f"     {best['win_rate']:.1%} on {best['n']:,} trades "
              f"(n_eff {best['n_eff']:,.0f}), bound "
              f"{best['wr_lo']:.1%} vs {base['wr_lo']:.1%} unfiltered")
        if best["wr_lo"] <= beat:
            print(f"     BUT a random filter keeping the same fraction reached "
                  f"a bound of {beat:.1%}, so this")
            print(f"     is not distinguishable from thinning the sample at "
                  f"random. Do not keep it.")
        else:
            print(f"     and it beats the best matched random filter "
                  f"({beat:.1%}). Worth keeping - then re-test")
            print(f"     it under the rolling split before trusting it.")

    payload = {"base": base, "filters": rows[1:], "random": rnd_rows,
               "base_win_rate": base_wr, "base_R": base_R}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n  wrote {out}")
    return rows, rnd_rows


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-cache", default=PRICE_CACHE)
    ap.add_argument("--horizon", default=HORIZON, choices=list(HORIZON_CONFIGS))
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--quantile", type=float, default=BASE_QUANTILE)
    ap.add_argument("--feature-set", default=FEATURE_SET)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-random", type=int, default=N_RANDOM_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--out", default="filters_v67.json")
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_filters_v67(c.price_cache, c.horizon, c.step, c.quantile,
                    c.feature_set, LABEL_MODE, MIN_TRAIN_YEARS, c.n_seeds,
                    c.seed, c.n_random, c.tickers, c.out)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_PRICE_CACHE = PRICE_CACHE
    RUN_HORIZON     = HORIZON
    RUN_STEP        = STEP
    RUN_QUANTILE    = BASE_QUANTILE   # the V66 cut to filter on top of
    RUN_FEATURE_SET = FEATURE_SET
    RUN_N_SEEDS     = N_SEEDS
    RUN_N_RANDOM    = N_RANDOM_DRAWS  # matched random draws per filter
    RUN_SEED        = SEED
    RUN_TICKERS     = None
    RUN_OUT         = "filters_v67.json"
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_filters_v67(price_cache=RUN_PRICE_CACHE, horizon=RUN_HORIZON,
                        step=RUN_STEP, quantile=RUN_QUANTILE,
                        feature_set=RUN_FEATURE_SET, n_seeds=RUN_N_SEEDS,
                        n_random=RUN_N_RANDOM, seed=RUN_SEED,
                        tickers=RUN_TICKERS, out=RUN_OUT)
