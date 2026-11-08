#!/usr/bin/env python3
"""
evaluate_oos_v43.py - does the cross-validated skill survive out of sample?

WHY NOT JUST USE THE BACKTEST

The backtest samples random (ticker, date) pairs and asks pillar 2 about one ticker at
a time. For a cohort ranker that is a poor fit: it measures a handful of top-N picks
per run, needs a live price fetch for every unfamiliar ticker, and the S&P 500 attempt
died with 4,692 of 5,000 samples returning nothing because Yahoo throttled the
downloads. None of that is a statement about the model.

This script measures the model the same way cross-validation did, on data it has never
seen. It reuses the DATASET GENERATOR - identical cohort construction, identical
features, identical VOL_SCALED labels - for dates AFTER the training cut, then scores
those rows and reports:

    within-date AUC   the number CV reported as 0.522 [0.512, 0.529]
    top-N lift        picks per date vs the base rate of THOSE SAME dates
    decile table      out-of-sample, on the universe the model was trained for

Apples to apples. If within-date AUC lands near 0.52 here, the skill is real and the
SHAY universe was simply the wrong deployment target. If it sits at 0.50, the CV
number did not survive and there is no exploitable signal - which is a finding too.

USAGE
  python evaluate_oos_v43.py
  python evaluate_oos_v43.py --start 2023-10-03 --end 2026-06-14 --every 5
  python evaluate_oos_v43.py --model class_model/xgboost_mid_v43b.joblib --reuse
"""
import argparse
import glob
import os
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pillar2_v43_features as F


# =============================================================================
# METRICS  (same definitions class_xgboost_v43b.py used, so numbers compare)
# =============================================================================
def safe_auc(y, p):
    y = np.asarray(y)
    return np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def within_date_auc(dates, y, p, min_rows=20):
    d = pd.DataFrame({"date": np.asarray(dates), "y": np.asarray(y), "p": np.asarray(p)})
    aucs, w = [], []
    for _, g in d.groupby("date"):
        if len(g) < min_rows or g["y"].nunique() < 2:
            continue
        a = safe_auc(g["y"], g["p"])
        if np.isfinite(a):
            aucs.append(a)
            w.append(len(g))
    return float(np.average(aucs, weights=w)) if aucs else np.nan


def date_bootstrap(dates, y, p, stat_fn, rounds=500, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(dates)
    groups = pd.Series(range(len(d))).groupby(d).apply(lambda s: s.to_numpy())
    idx = [groups[k] for k in groups.index]
    out = []
    for _ in range(rounds):
        rows = np.concatenate([idx[i] for i in rng.integers(0, len(idx), size=len(idx))])
        v = stat_fn(d[rows], np.asarray(y)[rows], np.asarray(p)[rows])
        if np.isfinite(v):
            out.append(v)
    return ((np.nan, np.nan) if len(out) < 20
            else (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))))


def topn_per_date(dates, y, p, rets, ns=(1, 3, 5, 10), min_cohort=20, rounds=500):
    d = pd.DataFrame({"date": np.asarray(dates), "y": np.asarray(y),
                      "p": np.asarray(p), "r": np.asarray(rets)})
    groups = [g for _, g in d.groupby("date") if len(g) >= min_cohort]
    if not groups:
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    rows = []
    for n in ns:
        picks = [g.nlargest(n, "p") for g in groups]
        prec = [g["y"].mean() for g in picks]
        bases = [g["y"].mean() for g in groups]
        lifts = []
        k = len(groups)
        for _ in range(rounds):
            sel = rng.integers(0, k, size=k)
            lifts.append((np.mean([prec[i] for i in sel])
                          - np.mean([bases[i] for i in sel])) * 100)
        rows.append({"n": n, "trades": sum(len(x) for x in picks), "dates": k,
                     "precision": float(np.mean(prec)) * 100,
                     "same_date_base": float(np.mean(bases)) * 100,
                     "lift": float(np.mean(prec) - np.mean(bases)) * 100,
                     "lo": float(np.percentile(lifts, 2.5)),
                     "hi": float(np.percentile(lifts, 97.5)),
                     "mean_ret": float(np.nanmean(pd.concat(picks)["r"]))})
    return pd.DataFrame(rows)


def fmt(x, spec=".3f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}"


# =============================================================================
# DATA
# =============================================================================
def build_or_load(a):
    """Build the out-of-sample cohort dataset with the generator, or reuse it."""
    tag = f"oos_{pd.Timestamp(a.start):%Y%m%d}_{pd.Timestamp(a.end):%Y%m%d}"
    existing = [p for p in glob.glob(f"*{tag}*")
                if p.endswith((".parquet", ".csv.gz", ".csv"))]
    if a.reuse and existing:
        path = sorted(existing)[-1]
        print(f"  reusing {path}")
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    import class_pillar2_data_gen_v43 as gen
    print(f"  building out-of-sample rows {a.start} -> {a.end} with the generator")
    print("  (same cohorts, same features, same VOL_SCALED labels as training)\n")
    gen.DATASET_VERSION = f"v43{tag}"
    cfg = SimpleNamespace(start=a.start, train_end=a.end, every=a.every,
                          min_cohort=a.min_cohort, min_dollar_vol=2_000_000,
                          max_rows=a.max_rows, cache_dir=a.cache_dir, out_dir=".",
                          no_fetch=a.no_fetch, no_deploy=False)
    return gen.build(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join("class_model",
                                                    "xgboost_mid_v43b.joblib"))
    ap.add_argument("--start", default="2023-10-03",
                    help="first signal date; must clear the training cut plus the "
                         "60-bar label horizon")
    ap.add_argument("--end", default="2026-06-14",
                    help="last signal date; needs 60 trading days of future bars")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-cohort", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--cache-dir", default="price_cache_v43")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--rounds", type=int, default=500)
    a = ap.parse_args()

    if not os.path.exists(a.model):
        sys.exit(f"{a.model} not found.")
    b = joblib.load(a.model)

    print("=" * 96)
    print("V43 OUT-OF-SAMPLE EVALUATION")
    print("=" * 96)
    print(f"  model:        {a.model}  ({b.get('variant')}, "
          f"{'ranker' if b.get('is_ranker') else 'classifier'})")
    print(f"  trained thru: {b.get('train_end')}")
    print(f"  CV reported:  within-date AUC "
          f"{b.get('cv_within_date_auc', float('nan')):.3f}   "
          f"overall {b.get('cv_auc', float('nan')):.3f}")
    print(f"  testing on:   {a.start} -> {a.end}   (never seen)\n")

    df = build_or_load(a)
    if df is None or not len(df):
        sys.exit("No out-of-sample rows built.")
    df["_date"] = pd.to_datetime(df["signal_date"])

    feats = b["feature_names"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing {len(missing)} model features, e.g. {missing[:5]}. "
                 f"The features file changed since training — regenerate or retrain.")

    X = df[feats]
    p = (b["model"].predict(X) if b.get("is_ranker")
         else b["model"].predict_proba(X)[:, 1])
    y = df["label"].to_numpy()
    d = df["_date"].to_numpy()
    r = df["exit_return_pct"].to_numpy() if "exit_return_pct" in df.columns else np.full(len(df), np.nan)

    print("\n" + "=" * 96)
    print(f"RESULTS   {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df['signal_date'].nunique()} dates | base rate {y.mean():.1%}")
    print("=" * 96)

    wd = within_date_auc(d, y, p)
    wlo, whi = date_bootstrap(d, y, p, within_date_auc, a.rounds)
    au = safe_auc(y, p)
    alo, ahi = date_bootstrap(d, y, p, lambda dd, yy, pp: safe_auc(yy, pp), a.rounds)
    cv = b.get("cv_within_date_auc", float("nan"))

    print(f"  within-date AUC  {fmt(wd)}   95% {fmt(wlo)} to {fmt(whi)}"
          f"{'   <-- clear of 0.50' if wlo > 0.5 else '   (touches 0.50)'}")
    print(f"  overall AUC      {fmt(au)}   95% {fmt(alo)} to {fmt(ahi)}")
    print(f"\n  cross-validation said {fmt(cv)}; out of sample {fmt(wd)} "
          f"({wd - cv:+.3f})")
    if np.isfinite(wd) and np.isfinite(cv):
        if wlo > 0.5:
            print("  The skill held. CV was measuring something real.")
        elif wd < 0.505:
            print("  The skill did NOT hold. Whatever CV measured did not survive "
                  "out of sample,")
            print("  so it was either period-specific or an artefact of the fold "
                  "construction.")

    print("\n  DECILES (out of sample)")
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    print(f"    {'decile':<8}{'n':>8}{'hit rate':>11}{'mean ret':>11}")
    for i in sorted(pd.unique(dec)):
        m = dec == i
        print(f"    {i+1:<8}{int(m.sum()):>8}{y[m].mean()*100:>10.1f}%"
              f"{np.nanmean(r[m]):>+11.2f}")

    tn = topn_per_date(d, y, p, r, rounds=a.rounds)
    if len(tn):
        print("\n  TOP-N PER DATE (lift vs the base rate of the same dates)")
        print(f"    {'N':>4}{'trades':>9}{'dates':>8}{'precision':>12}"
              f"{'same-day base':>15}{'lift':>9}{'95% range':>20}{'mean ret':>11}")
        for _, x in tn.iterrows():
            flag = "  <- clear of zero" if x["lo"] > 0 else ""
            print(f"    {int(x['n']):>4}{int(x['trades']):>9,}{int(x['dates']):>8}"
                  f"{x['precision']:>11.1f}%{x['same_date_base']:>14.1f}%"
                  f"{x['lift']:>+8.1f}p{f'{x.lo:+.1f} to {x.hi:+.1f}':>20}"
                  f"{x['mean_ret']:>+11.2f}{flag}")

    print("\n" + "=" * 96)
    print("  This is the same measurement cross-validation made, on data the model has")
    print("  never seen, over the universe it was trained for. It is the cleanest read")
    print("  available on whether the signal is real.")


if __name__ == "__main__":
    main()
