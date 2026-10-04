#!/usr/bin/env python3
"""
evaluate_oos_v44.py - does V44's cross-sectional signal survive out of sample?

Cross-validation reported IC +0.0280 at 5 bars and +0.0288 at 20 bars, both with
ranges clear of zero, both clean on permuted targets and stable across embargoes.
V43 also looked good in cross-validation (within-date AUC 0.521) and came back 0.507
out of sample, so the only thing that settles this is data the model has never seen.

WHAT THIS DOES
  1. fits the model on ALL training rows (no folds needed - the out-of-sample period
     is genuinely held out by the training cut)
  2. builds out-of-sample rows with the SAME generator, features and labels
  3. measures IC exactly as cross-validation measured it, plus quintile spread and
     top-N excess return
  4. breaks IC down by year, because a signal that only works in one period is not
     a signal

THE OVERLAP CORRECTION

Sampling every 5 trading days with a 20-bar forward window means neighbouring rows
share ~75% of their outcome window, but the bootstrap treats each date as an
independent draw. The effective number of independent observations is closer to
n_dates / (horizon / sample_every), so this reports an adjusted t-stat alongside the
raw one. At 5 bars with 5-day sampling the windows barely overlap and no correction is
needed; at 20 and 60 bars it matters a great deal.

USAGE
  python evaluate_oos_v44.py
  python evaluate_oos_v44.py --horizons 5,20 --start 2023-10-03 --end 2026-06-14
  python evaluate_oos_v44.py --reuse            # skip rebuilding the OOS dataset
"""
import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb

import pillar2_v43_features as F
from class_xgboost_v44 import (PARAMS, find_dataset, ic_per_date, ic_stats, load,
                               quantile_spread, topn_excess)


def build_or_load_oos(a):
    tag = f"oos{pd.Timestamp(a.start):%Y%m%d}_{pd.Timestamp(a.end):%Y%m%d}"
    hits = [p for p in glob.glob(f"*{tag}*")
            if p.endswith((".parquet", ".csv.gz", ".csv"))]
    if a.reuse and hits:
        path = sorted(hits)[-1]
        print(f"  reusing {path}")
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    else:
        import class_pillar2_data_gen_v44 as gen
        print(f"  building out-of-sample rows {a.start} -> {a.end}")
        print("  (same generator, same features, same labels as training)\n")
        gen.DATASET_VERSION = f"v44{tag}"
        df = gen.build(SimpleNamespace(
            start=a.start, train_end=a.end, every=a.every, min_cohort=a.min_cohort,
            min_dollar_vol=2_000_000, max_rows=a.max_rows, cache_dir=a.cache_dir,
            out_dir=".", no_fetch=a.no_fetch, no_deploy=False))
    df = df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    df["_date"] = pd.to_datetime(df["signal_date"])
    return df


def fmt(x, spec="+.4f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset", default=None)
    ap.add_argument("--horizons", default="5,10,20,60")
    ap.add_argument("--target", default="rank", choices=["rank", "exc"])
    ap.add_argument("--start", default="2023-10-03")
    ap.add_argument("--end", default="2026-06-14")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-cohort", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--cache-dir", default="price_cache_v43")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    tpath = a.train_dataset or find_dataset()
    train, meta = load(tpath)
    feats = [c for c in F.FEATURE_NAMES if c in train.columns]
    horizons = [int(h) for h in a.horizons.split(",") if h.strip()]

    print("=" * 96)
    print("V44 OUT-OF-SAMPLE EVALUATION")
    print("=" * 96)
    print(f"  train:   {tpath}")
    print(f"           {len(train):,} rows | {train['signal_date'].min()} -> "
          f"{train['signal_date'].max()}")
    print(f"  test:    {a.start} -> {a.end}   (never seen)")
    print(f"  target:  fwd_{a.target}_N, IC measured vs fwd_exc_N\n")

    oos = build_or_load_oos(a)
    if oos is None or not len(oos):
        sys.exit("No out-of-sample rows.")
    missing = [c for c in feats if c not in oos.columns]
    if missing:
        sys.exit(f"OOS dataset missing {len(missing)} features, e.g. {missing[:5]}")

    print("\n" + "=" * 96)
    print(f"RESULTS   {len(oos):,} rows | {oos['ticker'].nunique()} tickers | "
          f"{oos['signal_date'].nunique()} dates")
    print("=" * 96)

    rows = []
    preds = {}
    for h in horizons:
        tcol = f"fwd_{'rank' if a.target == 'rank' else 'exc'}_{h}"
        acol = f"fwd_exc_{h}"
        if tcol not in train.columns or acol not in oos.columns:
            print(f"  {h} bars: target columns missing, skipped")
            continue
        ok = np.isfinite(train[tcol].to_numpy())
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(train.loc[ok, feats], train.loc[ok, tcol], verbose=False)

        p = model.predict(oos[feats])
        preds[h] = p
        m = np.isfinite(p) & np.isfinite(oos[acol].to_numpy())
        d, pp, aa = oos.loc[m, "_date"].to_numpy(), p[m], oos.loc[m, acol].to_numpy()
        _, ics = ic_per_date(d, pp, aa)
        st = ic_stats(ics)

        # neighbouring windows overlap; the bootstrap assumes they do not
        overlap = max(1.0, h / a.every)
        st["t_adj"] = st["t"] / np.sqrt(overlap) if np.isfinite(st["t"]) else np.nan
        st["overlap"] = overlap
        st["h"] = h
        st["q"] = quantile_spread(d, pp, aa)
        st["topn"] = topn_excess(d, pp, aa)
        st["dates"] = d
        st["pred"] = pp
        st["act"] = aa
        rows.append(st)

    if not rows:
        sys.exit("Nothing evaluated.")

    print(f"\n  {'horizon':>8}{'IC':>10}{'95% range':>22}{'ICIR':>7}{'t raw':>8}"
          f"{'t adj':>8}{'Q5-Q1':>9}{'dates':>7}")
    for s in rows:
        rng = f"{s['lo']:+.4f} to {s['hi']:+.4f}"
        print(f"  {s['h']:>8}{s['ic']:>+10.4f}{rng:>22}{s['icir']:>+7.2f}"
              f"{s['t']:>+8.2f}{s['t_adj']:>+8.2f}"
              f"{s['q'].get('spread', float('nan')):>+8.2f}%{s['n']:>7}")
    print("\n  't adj' divides by sqrt(horizon / sampling interval): neighbouring rows")
    print("  share most of their forward window, so the raw t-stat overstates the")
    print("  evidence at longer horizons. At 5 bars with 5-day sampling, adj = raw.")

    # ── the comparison that matters ───────────────────────────────────
    cv = {5: 0.0280, 10: 0.0230, 20: 0.0288, 60: 0.0038}
    print("\n  CROSS-VALIDATION vs OUT OF SAMPLE")
    print(f"    {'horizon':>8}{'CV IC':>10}{'OOS IC':>10}{'change':>10}   verdict")
    for s in rows:
        c = cv.get(s["h"], float("nan"))
        if s["lo"] > 0:
            v = "HELD — range clear of zero"
        elif s["hi"] < 0:
            v = "REVERSED — reliably negative"
        else:
            v = "did not hold"
        print(f"    {s['h']:>8}{fmt(c):>10}{fmt(s['ic']):>10}"
              f"{fmt(s['ic'] - c):>10}   {v}")

    # ── stability ─────────────────────────────────────────────────────
    print("\n  IC BY YEAR (a signal that works in one year only is not a signal)")
    for s in rows:
        _, ics = ic_per_date(s["dates"], s["pred"], s["act"])
        dts, _ = ic_per_date(s["dates"], s["pred"], s["act"])
        byyear = pd.DataFrame({"y": pd.to_datetime(dts).year, "ic": ics})
        parts = "  ".join(f"{y}: {g['ic'].mean():+.3f}"
                          for y, g in byyear.groupby("y"))
        print(f"    {s['h']:>3} bars   {parts}")

    best = max(rows, key=lambda s: s["ic"] if np.isfinite(s["ic"]) else -9)
    tn = best["topn"]
    if len(tn):
        print(f"\n  TOP-N PER DATE at {best['h']} bars (mean excess return of picks)")
        print(f"    {'N':>4}{'dates':>8}{'mean excess':>14}{'t':>8}{'t adj':>8}")
        for _, r in tn.iterrows():
            print(f"    {int(r['n']):>4}{int(r['dates']):>8}{r['mean_excess']:>+13.3f}%"
                  f"{r['t']:>+8.2f}{r['t'] / np.sqrt(best['overlap']):>+8.2f}")
        q = best["q"]
        if q:
            print(f"\n    Quintile spread is the honest strategy number: "
                  f"top {q['top']:+.2f}% vs bottom {q['bottom']:+.2f}% "
                  f"= {q['spread']:+.2f}% per {best['h']}-bar period")

    print("\n" + "=" * 96)
    print("  V43's within-date AUC went 0.521 in CV to 0.507 out of sample. This is the")
    print("  same test for V44, on the same universe, with the same generator.")


if __name__ == "__main__":
    main()
