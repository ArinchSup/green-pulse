#!/usr/bin/env python3
"""
class_xgboost_v44.py - train and validate the V44 cross-sectional return model.

THE METRIC IS NOW IC, NOT AUC

The target is a forward return demeaned within its date, so the natural measure is the
INFORMATION COEFFICIENT: the rank correlation between prediction and realised excess
return, computed separately inside each date, then averaged across dates.

  IC     mean per-date rank correlation. A genuine cross-sectional equity signal lives
         around 0.02 to 0.05. That is small, it is what professionals actually work
         with, and at ~900 dates it is detectable. Anything above 0.10 on daily price
         features means a bug, not a discovery.
  ICIR   mean IC / sd of IC across dates. Consistency, not size. ~0.3+ is useful.
  t-stat mean IC / (sd / sqrt(n_dates)). Above 2 is the conventional bar.

TWO CONTROLS, BOTH MANDATORY

V43 reported within-date AUC 0.521 in cross-validation and 0.507 out of sample on
44,481 rows. The most likely cause was FEATURE-WINDOW overlap: ret_250 and the 200-day
EMA look back a year, so a training row 100 days before a fold boundary shares most of
its feature window with validation rows on the same tickers. Label purging does not fix
that. So:

  --embargo-sweep   reports IC at 100, 200 and 400 day embargoes. If IC falls as the
                    embargo grows, the leakage is real and measurable - which also
                    explains V43 retroactively.
  --shuffle-control trains on targets permuted WITHIN each date. The cross-sectional
                    structure survives, the stock-to-outcome link does not, so IC must
                    come out ~0. Anything else means the pipeline itself leaks, and it
                    is far cheaper to learn that here than after five model versions.

USAGE
  python class_xgboost_v44.py                          # sweep 5/10/20/60, rank target
  python class_xgboost_v44.py --horizons 20 --target exc
  python class_xgboost_v44.py --horizons 20 --embargo-sweep --shuffle-control
  python class_xgboost_v44.py --horizons 20 --final --out class_model/xgb_v44.joblib
"""
import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import pillar2_v43_features as F

PARAMS = dict(
    n_estimators=500, learning_rate=0.03, max_depth=5,
    min_child_weight=60,          # 270k rows of very weak signal: resist memorising
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=5.0,
    objective="reg:squarederror", tree_method="hist",
    n_jobs=-1, random_state=42,
)
MIN_COHORT = 20


# =============================================================================
# DATA
# =============================================================================
def find_dataset(pattern="dataset_pillar2_mid_v44_*"):
    hits = [p for p in glob.glob(pattern) if p.endswith((".parquet", ".csv.gz", ".csv"))]
    if not hits:
        sys.exit(f"No dataset matching '{pattern}'. Pass --dataset.")
    return sorted(hits)[-1]


def load(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    meta = {}
    mp = path.split(".parquet")[0].split(".csv")[0] + "_meta.json"
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    df = df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    df["_date"] = pd.to_datetime(df["signal_date"])
    return df, meta


# =============================================================================
# IC
# =============================================================================
def ic_per_date(dates, pred, actual, min_rows=MIN_COHORT):
    """Spearman inside each date. Returns (dates_kept, ic_values)."""
    d = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred),
                      "a": np.asarray(actual)})
    keys, vals = [], []
    for k, g in d.groupby("d"):
        g = g.dropna()
        if len(g) < min_rows:
            continue
        rp, ra = g["p"].rank(), g["a"].rank()
        if rp.std() == 0 or ra.std() == 0:
            continue
        vals.append(float(np.corrcoef(rp, ra)[0, 1]))
        keys.append(k)
    return np.asarray(keys), np.asarray(vals)


def ic_stats(ics, rounds=2000, seed=0):
    """Mean IC with a bootstrap over DATES — each date is one independent draw."""
    if len(ics) < 10:
        return dict(ic=np.nan, icir=np.nan, t=np.nan, lo=np.nan, hi=np.nan, n=len(ics))
    m, s = float(np.mean(ics)), float(np.std(ics, ddof=1))
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(ics, size=len(ics), replace=True)) for _ in range(rounds)]
    return dict(ic=m, icir=(m / s if s else np.nan),
                t=(m / (s / np.sqrt(len(ics))) if s else np.nan),
                lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)), n=len(ics))


def quantile_spread(dates, pred, actual, q=5, min_rows=MIN_COHORT):
    """Mean excess return of the top bucket minus the bottom, averaged over dates."""
    d = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred),
                      "a": np.asarray(actual)})
    tops, bots, spreads = [], [], []
    for _, g in d.groupby("d"):
        g = g.dropna()
        if len(g) < max(min_rows, q * 2):
            continue
        b = pd.qcut(g["p"].rank(method="first"), q, labels=False)
        t_, o_ = g["a"][b == q - 1].mean(), g["a"][b == 0].mean()
        tops.append(t_)
        bots.append(o_)
        spreads.append(t_ - o_)
    if not spreads:
        return {}
    s = np.asarray(spreads)
    return {"top": float(np.mean(tops)), "bottom": float(np.mean(bots)),
            "spread": float(np.mean(s)),
            "t": float(np.mean(s) / (np.std(s, ddof=1) / np.sqrt(len(s))))
            if np.std(s, ddof=1) else np.nan, "dates": len(s)}


def topn_excess(dates, pred, actual, ns=(1, 3, 5, 10), min_rows=MIN_COHORT):
    d = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred),
                      "a": np.asarray(actual)})
    groups = [g.dropna() for _, g in d.groupby("d")]
    groups = [g for g in groups if len(g) >= min_rows]
    rows = []
    for n in ns:
        per_date = [g.nlargest(n, "p")["a"].mean() for g in groups]
        v = np.asarray(per_date)
        sd = np.std(v, ddof=1)
        rows.append({"n": n, "dates": len(v), "mean_excess": float(np.mean(v)),
                     "t": float(np.mean(v) / (sd / np.sqrt(len(v)))) if sd else np.nan})
    return pd.DataFrame(rows)


# =============================================================================
# FOLDS
# =============================================================================
def folds(dates, n_folds, embargo_days):
    n = len(dates)
    edges = np.linspace(int(n * 0.40), n, n_folds + 1).astype(int)
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 2:
            continue
        val = dates[lo:hi]
        tr = dates[dates < val[0] - pd.Timedelta(days=embargo_days)]
        if len(tr) < 20:
            continue
        yield tr, val


# =============================================================================
# ONE CONFIGURATION
# =============================================================================
def run(df, feats, horizon, target, embargo, n_folds, shuffle=False, seed=0, quiet=False):
    tcol = f"fwd_{'rank' if target == 'rank' else 'exc'}_{horizon}"
    acol = f"fwd_exc_{horizon}"            # IC is ALWAYS measured against real excess
    if tcol not in df.columns or acol not in df.columns:
        sys.exit(f"{tcol} or {acol} missing — regenerate with class_pillar2_data_gen_v44.py")

    y = df[tcol].to_numpy(dtype=float)
    if shuffle:
        # permute WITHIN each date: cross-sectional structure survives, the link dies
        rng = np.random.default_rng(seed)
        y = y.copy()
        for _, idx in df.groupby("_date").indices.items():
            y[idx] = rng.permutation(y[idx])

    dates = np.array(sorted(df["_date"].unique()))
    X = df[feats]
    oof = np.full(len(df), np.nan)

    for tr, va in folds(dates, n_folds, embargo):
        trm = df["_date"].isin(tr).to_numpy()
        vam = df["_date"].isin(va).to_numpy()
        ok = np.isfinite(y) & trm
        if ok.sum() < 1000:
            continue
        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X[ok], y[ok], verbose=False)
        oof[vam] = m.predict(X[vam])

    mask = np.isfinite(oof) & np.isfinite(df[acol].to_numpy())
    d, p, a = (df.loc[mask, "_date"].to_numpy(), oof[mask],
               df.loc[mask, acol].to_numpy())
    _, ics = ic_per_date(d, p, a)
    st = ic_stats(ics)
    st.update(horizon=horizon, target=target, embargo=embargo, shuffled=shuffle,
              rows=int(mask.sum()))
    st["quintile"] = quantile_spread(d, p, a)
    st["topn"] = topn_excess(d, p, a)
    st["oof"] = oof
    if not quiet:
        print(f"    IC {st['ic']:+.4f}  95% [{st['lo']:+.4f}, {st['hi']:+.4f}]  "
              f"ICIR {st['icir']:+.2f}  t {st['t']:+.2f}  ({st['n']} dates)")
    return st


def verdict(st):
    if not np.isfinite(st["ic"]):
        return "no result"
    if st["lo"] > 0:
        return "POSITIVE — range excludes zero"
    if st["hi"] < 0:
        return "NEGATIVE — reliably backwards"
    return "flat — indistinguishable from zero"


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--horizons", default="5,10,20,60")
    ap.add_argument("--target", default="rank", choices=["rank", "exc"],
                    help="rank = percentile within the date (robust to a single +200% "
                         "move dominating the loss); exc = raw demeaned return")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=200)
    ap.add_argument("--embargo-sweep", action="store_true")
    ap.add_argument("--shuffle-control", action="store_true")
    ap.add_argument("--shuffle-seeds", type=int, default=3,
                    help="permutations per horizon; one marginal hit is "
                         "noise, a consistent one is a leak")
    ap.add_argument("--final", action="store_true", help="fit and save the best horizon")
    ap.add_argument("--out", default=os.path.join("class_model", "xgb_v44.joblib"))
    a = ap.parse_args()

    path = a.dataset or find_dataset()
    df, meta = load(path)
    feats = [c for c in F.FEATURE_NAMES if c in df.columns]
    horizons = [int(h) for h in a.horizons.split(",") if h.strip()]

    print("=" * 92)
    print("PILLAR 2 V44 — CROSS-SECTIONAL RETURN MODEL")
    print("=" * 92)
    print(f"  dataset:  {path}")
    print(f"  rows {len(df):,} | tickers {df['ticker'].nunique()} | "
          f"dates {df['signal_date'].nunique()} | features {len(feats)}")
    print(f"  span {df['signal_date'].min()} -> {df['signal_date'].max()}")
    print(f"  target:   fwd_{a.target}_N   (IC always measured vs fwd_exc_N)")
    print(f"  folds:    {a.folds}, embargo {a.embargo_days}d, split by date\n")

    print("=" * 92)
    print("HORIZON SWEEP")
    print("=" * 92)
    results = {}
    for h in horizons:
        print(f"  {h:>3} bars forward:")
        results[h] = run(df, feats, h, a.target, a.embargo_days, a.folds)

    print(f"\n  {'horizon':>8}{'IC':>10}{'95% range':>22}{'ICIR':>8}{'t':>8}"
          f"{'Q5-Q1 excess':>15}{'verdict':>34}")
    for h in horizons:
        s = results[h]
        q = s["quintile"]
        rng_txt = f"{s['lo']:+.4f} to {s['hi']:+.4f}"
        print(f"  {h:>8}{s['ic']:>+10.4f}{rng_txt:>22}"
              f"{s['icir']:>+8.2f}{s['t']:>+8.2f}"
              f"{q.get('spread', float('nan')):>+14.2f}%{verdict(s):>34}")
    print("\n  A real cross-sectional signal sits around IC 0.02-0.05 with t > 2.")
    print("  IC above 0.10 from daily price features is a bug, not a discovery.")

    best = max(horizons, key=lambda h: results[h]["ic"] if np.isfinite(results[h]["ic"]) else -9)
    print(f"\n  best horizon by IC: {best} bars")
    tn = results[best]["topn"]
    if len(tn):
        print(f"\n  TOP-N PER DATE at {best} bars (mean excess return of the picks)")
        print(f"    {'N':>4}{'dates':>8}{'mean excess':>14}{'t':>8}")
        for _, r in tn.iterrows():
            print(f"    {int(r['n']):>4}{int(r['dates']):>8}{r['mean_excess']:>+13.3f}%"
                  f"{r['t']:>+8.2f}")

    if a.shuffle_control:
        print("\n" + "=" * 92)
        print("SHUFFLE CONTROL  (targets permuted within each date — IC must be ~0)")
        print("=" * 92)
        print("  Each horizon is run with several permutations. One marginal result at")
        print("  ~2 sigma is expected by chance across a handful of tests; a leak shows up")
        print("  as a CONSISTENT positive across seeds.\n")
        for h in horizons:
            ics, ts = [], []
            for sd in range(a.shuffle_seeds):
                s = run(df, feats, h, a.target, a.embargo_days, a.folds,
                        shuffle=True, seed=sd, quiet=True)
                ics.append(s["ic"])
                ts.append(s["t"])
                print(f"    {h:>3} bars, seed {sd}:  IC {s['ic']:+.4f}  t {s['t']:+.2f}")
            mean_ic = float(np.nanmean(ics))
            strong = sum(1 for t_ in ts if np.isfinite(t_) and t_ > 3)
            print(f"    -> mean shuffled IC {mean_ic:+.4f} over "
                  f"{a.shuffle_seeds} permutations")
            if strong >= max(2, a.shuffle_seeds // 2) or mean_ic > 0.01:
                print("    !! Shuffled targets give a consistent positive IC. The "
                      "pipeline itself leaks;")
                print("       every number above is suspect. Check fold construction "
                      "before anything else.")
            else:
                print("    OK: no consistent signal on permuted targets.\n")

    if a.embargo_sweep:
        print("\n" + "=" * 92)
        print("EMBARGO SWEEP  (does IC shrink as train and validation are pushed apart?)")
        print("=" * 92)
        print("  Features look back up to 250 bars, so a short embargo lets training rows")
        print("  share feature windows with validation rows. Label purging misses this.")
        print(f"\n  {'horizon':>8}{'embargo':>10}{'IC':>10}{'95% range':>22}{'t':>8}")
        for h in horizons:
            for emb in (100, 200, 400):
                s = run(df, feats, h, a.target, emb, a.folds, quiet=True)
                rng_txt = f"{s['lo']:+.4f} to {s['hi']:+.4f}"
                print(f"  {h:>8}{emb:>10}{s['ic']:>+10.4f}{rng_txt:>22}{s['t']:>+8.2f}")
        print("\n  A sharp fall from 100 to 400 is the feature-window leak, measured.")

    if a.final:
        h = best
        tcol = f"fwd_{'rank' if a.target == 'rank' else 'exc'}_{h}"
        ok = np.isfinite(df[tcol].to_numpy())
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(df.loc[ok, feats], df.loc[ok, tcol], verbose=False)
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        print("\n" + "=" * 92)
        print(f"FINAL MODEL  horizon {h}, target {tcol}")
        print("=" * 92)
        for k, v in imp.head(12).items():
            print(f"    {k:<26}{v:.4f}")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        joblib.dump({"model": model, "feature_names": feats, "is_ranker": False,
                     "variant": f"v44_h{h}_{a.target}", "horizon_bars": h,
                     "target": tcol, "feature_version": F.FEATURE_VERSION,
                     "geometry": meta.get("geometry"), "benchmark": meta.get("benchmark"),
                     "train_end": meta.get("train_end"),
                     "cv_ic": results[h]["ic"], "cv_ic_t": results[h]["t"],
                     "params": PARAMS}, a.out)
        print(f"\n  saved {a.out}")
        print("  NOTE: output is a predicted cross-sectional rank, not a probability.")
        print("  Score the whole cohort and take the top N.")


if __name__ == "__main__":
    main()
