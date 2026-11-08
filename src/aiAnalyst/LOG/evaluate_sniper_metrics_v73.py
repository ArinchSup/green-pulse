"""
V73 - METRICS THAT FIT A SNIPER MODEL.

THE PROBLEM WITH WHAT CAME BEFORE
---------------------------------
V71 judged the signal on compounded return against the best of 20 random draws.
Both halves were wrong. The max of N draws is a ~p<0.025 bar carried by a single
draw, and return is downstream of exposure and position sizing, which an entry
model does not claim to control.

But swapping return for precision alone is not enough either. Precision asks one
narrow question - did it touch the target before the stop - and a sniper makes
more claims than that. "Now is a good moment to buy this" implies the trade
works, works FAST, and does not need to be carried for months to get there.
None of that is visible in a win rate.

THE METRICS HERE
----------------
Every one is computed on the fired set and on a NAME-AND-MONTH-MATCHED control:
for each signal on ticker T in month M, a different candidate entry on T in M.
Same names, same regimes, same exposure, same position count - only the entry
day moves. Everything the two arms share cancels, so what is left is timing.

  precision        touched target before stop. The promise made to the user.
  expectancy_R     value per trade. Sizing-free.
  R_per_bar        R earned per bar the capital was tied up. A sniper that makes
                   0.5R in 20 bars is worth more than one that makes 0.5R in 90:
                   same R, a quarter of the exposure, and the capital is free to
                   work again. This is the metric entry timing should move and
                   the one no previous table measured.
  bars_to_target   median bars for a WINNER to reach the target. If the entry is
                   genuinely well-timed the thesis should play out sooner, not
                   merely more often.
  flat_share       share that touched neither barrier and exited at the horizon.
                   A dead trade is not a loss but it is a wasted slot, and
                   converting flats into wins is most of what good timing does.
  payoff_ratio     mean win R over mean |loss| R.
  loss_share       share stopped out. Worth reporting separately from precision
                   because precision moves when flats move.

Two further checks that are NOT comparisons against a control:

  per-year sign test   in how many years does the fired win rate beat that same
                       year's blind rate? A binomial sign test over years needs
                       no effective-sample estimate and no block bootstrap, so
                       it sidesteps the n_eff problem that makes the pooled
                       interval uselessly wide. It is low power, but it is valid,
                       and 13 of 17 years is a real result.
  calibration          does a score of 0.62 mean 62%? A sniper states a
                       confidence to a user who acts on it, so ranking is not
                       enough - the number has to mean something. Brier score
                       plus a reliability table by score decile.

NOT MEASURED, and worth saying so: maximum adverse excursion - how far underwater
a trade goes before it works. That is arguably the sharpest entry-timing metric
of all, and it cannot be computed here because `triple_barrier` records only the
bar a barrier was touched, not the path to it. Adding it means returning the
running minimum from that function and rebuilding the dataset.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

import class_ai_entry_v64 as E64
import class_ai_entry_model_v70 as M70

N_DRAWS = 200      # the empirical p floor is 1/(N+1), so 20 draws
                   # cannot report anything below p = 0.048
SEED = 73
OUT_DIR = "thesis_tables_v73"


# =============================================================================
# METRICS - each takes the selected trades, returns one number
# =============================================================================
def _precision(t):
    return float(t["label"].mean())


def _expectancy(t):
    return float(t["r_multiple"].mean())


def _r_per_bar(t):
    b = t["bars_held"].to_numpy(float)
    b = np.where(b > 0, b, np.nan)
    return float(np.nansum(t["r_multiple"].to_numpy(float)) / np.nansum(b))


def _bars_to_target(t):
    w = t[t["outcome"] == "win"]
    return float(w["bars_held"].median()) if len(w) else np.nan


def _flat_share(t):
    return float((t["outcome"] == "flat").mean())


def _loss_share(t):
    return float((t["outcome"] == "loss").mean())


def _payoff(t):
    r = t["r_multiple"].to_numpy(float)
    win, loss = r[r > 0], r[r < 0]
    if not win.size or not loss.size:
        return np.nan
    return float(win.mean() / abs(loss.mean()))


METRICS = {
    "precision":      (_precision,      True,  "{:.4f}"),
    "expectancy_R":   (_expectancy,     True,  "{:+.4f}"),
    "R_per_bar":      (_r_per_bar,      True,  "{:+.5f}"),
    "bars_to_target": (_bars_to_target, False, "{:.1f}"),
    "flat_share":     (_flat_share,     False, "{:.4f}"),
    "loss_share":     (_loss_share,     False, "{:.4f}"),
    "payoff_ratio":   (_payoff,         True,  "{:.3f}"),
}


# =============================================================================
# THE CONTROL: same name, same month, different day
# =============================================================================
def fire_mask(te, quantile):
    p = te["p"].to_numpy(float)
    return p >= float(np.quantile(p, 1.0 - quantile))


def name_matched_draws(te, quantile, seed=SEED, n_draws=N_DRAWS):
    """
    `n_draws` alternative trade sets, each the same size as the fired set, each
    holding name and month fixed and moving only the entry day. Returns frames
    rather than summary numbers so any metric can be run over them.
    """
    te = te.reset_index(drop=True)
    di = pd.DatetimeIndex(te["date"])
    keys = pd.DataFrame({"ticker": te["ticker"].to_numpy(),
                         "_per": pd.PeriodIndex(di, freq="M"),
                         "_qtr": pd.PeriodIndex(di, freq="Q"),
                         "_yr": pd.PeriodIndex(di, freq="Y")})
    fired_idx = np.flatnonzero(fire_mask(te, quantile))
    if not fired_idx.size:
        return None, None
    pools = {k: dict(keys.groupby(["ticker", k]).indices)
             for k in ("_per", "_qtr", "_yr")}
    f = keys.iloc[fired_idx]
    rng = np.random.default_rng(seed)

    draws, widened = [], 0
    for _ in range(n_draws):
        picks, w = [], 0
        for tk, per, qtr, yr in zip(f["ticker"], f["_per"], f["_qtr"],
                                    f["_yr"]):
            pool = pools["_per"].get((tk, per))
            if pool is None or len(pool) < 2:
                w += 1
                pool = pools["_qtr"].get((tk, qtr))
                if pool is None or len(pool) < 2:
                    pool = pools["_yr"].get((tk, yr))
            if pool is None or not len(pool):
                continue
            picks.append(int(rng.choice(pool)))
        if picks:
            draws.append(te.iloc[picks])
            widened = max(widened, w)
    return te.iloc[fired_idx], {"draws": draws, "widened": widened}


def scorecard(te, quantile, seed=SEED, n_draws=N_DRAWS):
    """Every metric, model vs the matched control, reported as a rank."""
    fired, ctl = name_matched_draws(te, quantile, seed, n_draws)
    if fired is None or not ctl["draws"]:
        return None
    rows = []
    for name, (fn, higher, fmt) in METRICS.items():
        mine = fn(fired)
        vals = np.array([fn(d) for d in ctl["draws"]], float)
        vals = vals[np.isfinite(vals)]
        if not np.isfinite(mine) or not vals.size:
            continue
        beat = int((vals < mine).sum() if higher else (vals > mine).sum())
        n = int(vals.size)
        rows.append({
            "Metric": name,
            "Model": fmt.format(mine),
            "Control median": fmt.format(float(np.median(vals))),
            "Control range": f"{fmt.format(vals.min())} .. "
                             f"{fmt.format(vals.max())}",
            "Better than": f"{beat}/{n}",
            "p (one-sided)": (n - beat + 1) / (n + 1),
            "Direction": "higher" if higher else "lower"})
    return pd.DataFrame(rows), fired, ctl


# =============================================================================
# CHECKS THAT NEED NO CONTROL
# =============================================================================
def per_year_sign_test(te, quantile):
    """
    Years in which the fired win rate beat that year's own blind rate.

    This is deliberately crude. It throws away magnitude and keeps only the
    sign, which costs power - but a sign test over years needs no effective
    sample size, no block bootstrap and no independence assumption beyond
    'different years are different draws'. When n_eff makes the pooled interval
    too wide to say anything, this can still say something.
    """
    fired = te[fire_mask(te, quantile)]
    rows = []
    for y, g in te.groupby("year"):
        f = fired[fired["year"] == y]
        if not len(f):
            continue
        rows.append({"Year": int(y), "Signals": len(f),
                     "Fired win rate": float(f["label"].mean()),
                     "Blind win rate": float(g["label"].mean()),
                     "Fired R": float(f["r_multiple"].mean()),
                     "Blind R": float(g["r_multiple"].mean())})
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df["Beat"] = df["Fired win rate"] > df["Blind win rate"]
    df["Beat R"] = df["Fired R"] > df["Blind R"]
    k, n = int(df["Beat"].sum()), len(df)
    kr = int(df["Beat R"].sum())
    from math import comb
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    pr = sum(comb(n, i) for i in range(kr, n + 1)) / 2 ** n
    return df, {"years": n, "beat_wr": k, "p_wr": p,
                "beat_R": kr, "p_R": pr}


def calibration(te, n_bins=10):
    """
    Does the score mean what it says, and does more score mean more outcome?

    The score column is NOT always a probability. With PREDICT_MODE =
    "expectancy" the model regresses the R multiple, so `p` is an R prediction
    and runs negative. Scoring that with Brier against a 0/1 label compares two
    different units and returns a meaningless number - an earlier version of
    this function reported a Brier skill of -0.41 that way, which reads as
    'catastrophically anti-calibrated' and actually meant 'wrong units'. So the
    mode is detected and the right target is used.

    MONOTONICITY is the part that transfers either way. A sniper only makes
    sense if a higher score really does mean a better outcome across the whole
    range - if the relationship is flat until the last decile, the model has one
    lucky bucket rather than a ranking.
    """
    p = te["p"].to_numpy(float)
    y = te["label"].to_numpy(float)
    r = te["r_multiple"].to_numpy(float)
    is_prob = bool(np.all((p >= 0) & (p <= 1)))

    q = pd.qcut(pd.Series(p), n_bins, labels=False, duplicates="drop")
    f = pd.DataFrame({"p": p, "y": y, "r": r, "b": q})
    rows = []
    for b, g in f.groupby("b"):
        row = {"Score decile": int(b) + 1, "n": len(g),
               "Mean score": float(g["p"].mean()),
               "Actual win rate": float(g["y"].mean()),
               "Actual mean R": float(g["r"].mean())}
        if is_prob:
            row["Gap (score - win rate)"] = row["Mean score"] - \
                row["Actual win rate"]
        rows.append(row)
    df = pd.DataFrame(rows)

    def _mono(col):
        d = np.diff(df[col].to_numpy(float))
        return int((d > 0).sum()), int(len(d))

    up_wr, n_wr = _mono("Actual win rate")
    up_r, n_r = _mono("Actual mean R")
    stat = {"is_prob": is_prob,
            "spearman_wr": float(pd.Series(df["Mean score"]).corr(
                df["Actual win rate"], method="spearman")),
            "spearman_r": float(pd.Series(df["Mean score"]).corr(
                df["Actual mean R"], method="spearman")),
            "mono_wr": (up_wr, n_wr), "mono_r": (up_r, n_r),
            "spread_wr": float(df["Actual win rate"].iloc[-1]
                               - df["Actual win rate"].iloc[0]),
            "spread_r": float(df["Actual mean R"].iloc[-1]
                              - df["Actual mean R"].iloc[0])}
    if is_prob:
        brier = float(np.mean((p - y) ** 2))
        base = float(np.mean((y.mean() - y) ** 2))
        stat.update({"brier": brier, "brier_baseline": base,
                     "skill": 1 - brier / base if base else np.nan})
    return df, stat


# =============================================================================
# RUNNER
# =============================================================================
def run_v73(model_file="entry_model_v70.joblib", price_cache=None,
            out_dir=OUT_DIR, quantile=None, seed=SEED, n_draws=N_DRAWS,
            verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    model = M70.load_model(model_file)
    pr = model["provenance"]
    price_cache = price_cache or pr["price_cache"]
    q = quantile if quantile is not None else model["quantile"]

    print("=" * 92)
    print("V73 - SNIPER-APPROPRIATE METRICS")
    print("=" * 92)
    print(f"  model {model_file} | cache {price_cache} | cut top {q:.0%}")

    names = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(price_cache) if f.endswith(".pkl"))
    d = E64.build_dataset(names, price_cache, pr["horizon"], E64.STEP,
                          verbose=verbose)
    E64.assert_market_columns(d)
    d = E64.apply_label_mode(d, pr["label_mode"])
    te = E64.walk_forward(d, model["features"], E64.MIN_TRAIN_YEARS, 1, seed,
                          pr["horizon"], verbose=False)
    print(f"  {len(te):,} out-of-sample trades, {te['year'].nunique()} years, "
          f"{te['ticker'].nunique()} names")

    sc = scorecard(te, q, seed, n_draws)
    if sc:
        board, fired, ctl = sc
        print("\n" + "=" * 92)
        print("  SCORECARD  model vs random entry in the SAME NAME and MONTH")
        print("=" * 92)
        print(f"  {len(fired):,} signals, {len(ctl['draws'])} control draws. "
              f"Only the entry DAY differs.")
        print(board.to_string(index=False,
                              float_format=lambda v: f"{v:.3f}"))
        print("\n  'Better than k/n' counts control draws the model beat in the"
              " direction that")
        print("  favours it. p is the one-sided empirical rank, NOT a "
              "comparison to the best draw.")
        print(f"  p cannot go below {1/(len(ctl['draws'])+1):.4f} with "
              f"{len(ctl['draws'])} draws - a clean sweep reports that floor.")
        if ctl["widened"]:
            print(f"  ({ctl['widened']} signals had no alternative entry in the"
                  f" same ticker-month; widened)")
        board.to_csv(f"{out_dir}/scorecard.csv", index=False)

    ydf, ystat = per_year_sign_test(te, q)
    if ydf is not None:
        print("\n" + "=" * 92)
        print("  PER-YEAR SIGN TEST  (no n_eff required)")
        print("=" * 92)
        print(ydf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n  win rate beat blind in {ystat['beat_wr']}/"
              f"{ystat['years']} years   sign-test p = {ystat['p_wr']:.4f}")
        print(f"  expectancy beat blind in {ystat['beat_R']}/"
              f"{ystat['years']} years   sign-test p = {ystat['p_R']:.4f}")
        ydf.to_csv(f"{out_dir}/per_year.csv", index=False)

    cdf, cstat = calibration(te)
    print("\n" + "=" * 92)
    print("  RANKING QUALITY  does more score mean more outcome")
    print("=" * 92)
    print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    mw, nw = cstat["mono_wr"]
    mr, nr = cstat["mono_r"]
    print(f"\n  win rate rises in {mw}/{nw} decile steps "
          f"(Spearman {cstat['spearman_wr']:+.3f}), "
          f"decile 1 -> 10 spread {cstat['spread_wr']:+.4f}")
    print(f"  mean R   rises in {mr}/{nr} decile steps "
          f"(Spearman {cstat['spearman_r']:+.3f}), "
          f"decile 1 -> 10 spread {cstat['spread_r']:+.4f}")
    print("  A monotone rise across ALL deciles is a ranking. A flat run with "
          "one high bucket")
    print("  at the end is a lucky bucket, and it would not survive a change of"
          " threshold.")
    if cstat["is_prob"]:
        print(f"\n  Brier {cstat['brier']:.4f} vs {cstat['brier_baseline']:.4f}"
              f" for always predicting the base rate -> skill "
              f"{cstat['skill']:+.4f}")
        print("  A large positive Gap means the model OVERSTATES its "
              "confidence to the user.")
    else:
        print("\n  Scores are EXPECTANCY predictions (R units, they run "
              "negative), not probabilities,")
        print("  so Brier and a confidence gap do not apply and are omitted. To"
              " show a user a")
        print("  percentage, refit with PREDICT_MODE = 'classify' or fit an "
              "isotonic map from")
        print("  score to win rate on training folds - do NOT relabel these "
              "scores as confidence.")
    cdf.to_csv(f"{out_dir}/calibration.csv", index=False)

    print(f"\n  wrote CSVs to {out_dir}/")
    return {"scorecard": sc[0] if sc else None, "per_year": ydf,
            "per_year_stat": ystat, "calibration": cdf,
            "calibration_stat": cstat}


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default="entry_model_v70.joblib")
    ap.add_argument("--price-cache", default=None)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(argv=None):
    c = _parser().parse_args(argv)
    run_v73(c.model_file, c.price_cache, c.out_dir, c.quantile, c.seed,
            c.n_draws)
    return 0


# =============================================================================
# RUN ME
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_MODEL_FILE  = "entry_model_v70.joblib"
    RUN_PRICE_CACHE = None      # None uses the cache the model was trained on
    RUN_OUT_DIR     = "thesis_tables_v73"
    RUN_QUANTILE    = None      # None uses the model's own deployed cut
    RUN_N_DRAWS     = N_DRAWS
    RUN_SEED        = SEED
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_v73(model_file=RUN_MODEL_FILE, price_cache=RUN_PRICE_CACHE,
                out_dir=RUN_OUT_DIR, quantile=RUN_QUANTILE,
                n_draws=RUN_N_DRAWS, seed=RUN_SEED)
