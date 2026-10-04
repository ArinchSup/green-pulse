#!/usr/bin/env python3
"""
class_xgboost_v43.py - train and HONESTLY validate the Pillar 2 V43 classifier.

WHY THIS IS NOT class_xgboost.py WITH A NEW PATH

The old script used TimeSeriesSplit on row order. Rows came from random (ticker, date)
draws, so row order was not date order: "time series CV" was a random split across
overlapping 60-bar labels. That is why V39 reported 60% precision at 0.65 and V42
reported 80%, and why neither survived contact with the backtest.

V43 rows are cohorts - roughly 290 tickers share every signal date - so a naive split
would put the SAME DAY on both sides of the fold boundary. Every stock in a cohort
shares that day's market move, so this leaks hard.

Three defences, all mandatory:

  1. SPLIT BY DATE, NEVER BY ROW. A date is wholly in train or wholly in validation.
  2. EMBARGO. A label opened on date D resolves 60 trading days later. Training dates
     within the embargo of the validation start are DROPPED, so no training label can
     see into the validation window.
  3. WALK FORWARD. Folds move forward in time; the model is never trained on the
     future of its own validation set.

WHAT IT REPORTS THAT THE OLD SCRIPT DID NOT

  - AUC, not just F1. F1 at a 0.5 cutoff hides ranking ability, which is the thing
    that actually decides whether a threshold can produce a sniper.
  - WITHIN-DATE AUC. Overall AUC rewards a model that only says "2008 is bad" - real
    but purely market timing. Within-date AUC asks the harder question: on one given
    day, can it rank THESE stocks against each other? That is stock picking, and it
    is the number that matters for a tool that must choose what to buy today.
  - A logistic-regression baseline on rank-transformed features. If the gradient
    boosting cannot beat a linear model, that is a finding, not a failure.
  - Date-clustered bootstrap ranges on every headline number.

USAGE
  python class_xgboost_v43.py
  python class_xgboost_v43.py --dataset dataset_..._cut20230630.csv.gz --folds 5
  python class_xgboost_v43.py --no-shap --out class_model/xgboost_mid_v43.joblib
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pillar2_v43_features as F

DEFAULT_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=5,
    min_child_weight=40,          # high: 128k rows of weak signal, resist memorising
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=3.0,
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", n_jobs=-1, random_state=42,
)


# =============================================================================
# LOAD
# =============================================================================
def find_dataset(pattern="dataset_pillar2_mid_v43_*"):
    hits = [p for p in glob.glob(pattern)
            if p.endswith((".parquet", ".csv.gz", ".csv"))]
    if not hits:
        sys.exit(f"No dataset matching '{pattern}'. Pass --dataset.")
    return sorted(hits)[-1]


def load_dataset(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    meta_path = path.split(".parquet")[0].split(".csv")[0] + "_meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    df = df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    return df, meta


# =============================================================================
# PURGED, EMBARGOED, DATE-GROUPED WALK-FORWARD FOLDS
# =============================================================================
def walk_forward_folds(dates, n_folds, embargo_days):
    """
    dates: sorted unique signal dates (as Timestamps).
    Yields (train_dates, val_dates). Validation slices move forward in time; every
    training date ends at least `embargo_days` before its validation window starts,
    so no training label can resolve inside that window.
    """
    n = len(dates)
    # leave the first 40% as the minimum training base, split the rest into folds
    start = int(n * 0.40)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        v_lo, v_hi = edges[i], edges[i + 1]
        if v_hi - v_lo < 2:
            continue
        val = dates[v_lo:v_hi]
        cutoff = val[0] - pd.Timedelta(days=embargo_days)
        train = dates[dates < cutoff]
        if len(train) < 20:
            continue
        yield train, val


# =============================================================================
# METRICS
# =============================================================================
def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def within_date_auc(dates, y, p, min_rows=20):
    """
    Average AUC computed SEPARATELY inside each date, weighted by cohort size.
    Strips out the market-timing component: a model that only knows "today is bad"
    scores 0.50 here, because every stock on that day gets the same nudge.
    """
    d = pd.DataFrame({"date": dates, "y": y, "p": p})
    aucs, weights = [], []
    for _, g in d.groupby("date"):
        if len(g) < min_rows or g["y"].nunique() < 2:
            continue
        a = safe_auc(g["y"], g["p"])
        if np.isfinite(a):
            aucs.append(a)
            weights.append(len(g))
    if not aucs:
        return np.nan
    return float(np.average(aucs, weights=weights))


def date_bootstrap(dates, y, p, stat_fn, rounds=400, seed=0):
    """Resample whole DATES with replacement - rows inside a date are not independent."""
    rng = np.random.default_rng(seed)
    d = pd.Series(dates).to_numpy()
    groups = pd.Series(range(len(d))).groupby(d).apply(lambda s: s.to_numpy())
    keys = list(groups.index)
    idx = [groups[k] for k in keys]
    out = []
    for _ in range(rounds):
        pick = rng.integers(0, len(idx), size=len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        v = stat_fn(d[rows], np.asarray(y)[rows], np.asarray(p)[rows])
        if np.isfinite(v):
            out.append(v)
    if len(out) < 20:
        return (np.nan, np.nan)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def fmt(x, spec=".3f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}"


# =============================================================================
# BASELINE
# =============================================================================
def logistic_baseline(Xtr, ytr, Xva):
    """
    Logistic regression on rank-transformed features. If XGBoost cannot beat this,
    the extra capacity is fitting noise.
    """
    tr = Xtr.rank(pct=True).fillna(0.5)
    va = Xva.rank(pct=True).fillna(0.5)
    m = LogisticRegression(max_iter=1000, C=0.1)
    m.fit(tr, ytr)
    return m.predict_proba(va)[:, 1]


# =============================================================================
# THRESHOLD SWEEP
# =============================================================================
def threshold_sweep(y, p, rets, thresholds):
    rows = []
    for t in thresholds:
        m = p >= t
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "threshold": t, "trades": n,
            "share": n / len(p) * 100,
            "precision": float(np.mean(np.asarray(y)[m])) * 100,
            "mean_ret": float(np.nanmean(np.asarray(rets)[m])) if rets is not None else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=100,
                    help="calendar days between the last training date and the "
                         "validation start; must exceed the label horizon")
    ap.add_argument("--out", default=os.path.join("class_model", "xgboost_mid_v43.joblib"))
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--quiet-folds", action="store_true")
    a = ap.parse_args()

    path = a.dataset or find_dataset()
    df, meta = load_dataset(path)
    feats = [c for c in F.FEATURE_NAMES if c in df.columns]
    missing = [c for c in F.FEATURE_NAMES if c not in df.columns]

    print("=" * 96)
    print("PILLAR 2 V43 TRAINING")
    print("=" * 96)
    print(f"  dataset:   {path}")
    print(f"  rows:      {len(df):,}   tickers: {df['ticker'].nunique()}   "
          f"dates: {df['signal_date'].nunique()}")
    print(f"  span:      {df['signal_date'].min()} -> {df['signal_date'].max()}")
    print(f"  features:  {len(feats)}" + (f"  ({len(missing)} MISSING: {missing[:5]})"
                                          if missing else ""))
    print(f"  geometry:  {meta.get('geometry', '?')}   expired_as: {meta.get('expired_as', '?')}")
    print(f"  base rate: {df['label'].mean():.1%}")
    if missing:
        print("\n  !! The dataset lacks features this code expects. Regenerate it with the "
              "matching pillar2_v43_features.py before trusting anything below.")

    df["_date"] = pd.to_datetime(df["signal_date"])
    dates = np.array(sorted(df["_date"].unique()))
    X_all, y_all = df[feats], df["label"].to_numpy()
    ret_all = (df["exit_return_pct"].to_numpy()
               if "exit_return_pct" in df.columns else None)

    # ── cross-validation ──────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"PURGED WALK-FORWARD CV   {a.folds} folds, {a.embargo_days}-day embargo, "
          f"split by DATE not by row")
    print("=" * 96)

    oof = np.full(len(df), np.nan)
    fold_rows = []
    for k, (tr_dates, va_dates) in enumerate(
            walk_forward_folds(dates, a.folds, a.embargo_days), 1):
        tr_m = df["_date"].isin(tr_dates).to_numpy()
        va_m = df["_date"].isin(va_dates).to_numpy()
        Xtr, ytr = X_all[tr_m], y_all[tr_m]
        Xva, yva = X_all[va_m], y_all[va_m]

        model = xgb.XGBClassifier(**DEFAULT_PARAMS)
        model.fit(Xtr, ytr, verbose=False)
        p = model.predict_proba(Xva)[:, 1]
        oof[va_m] = p

        try:
            p_lin = logistic_baseline(Xtr, ytr, Xva)
            lin_auc = safe_auc(yva, p_lin)
        except Exception:
            lin_auc = np.nan

        row = {
            "fold": k,
            "train_rows": int(tr_m.sum()), "val_rows": int(va_m.sum()),
            "val_from": pd.Timestamp(va_dates[0]).date(),
            "val_to": pd.Timestamp(va_dates[-1]).date(),
            "base_rate": float(np.mean(yva)) * 100,
            "auc": safe_auc(yva, p),
            "auc_within_date": within_date_auc(df.loc[va_m, "_date"], yva, p),
            "auc_linear": lin_auc,
        }
        fold_rows.append(row)
        if not a.quiet_folds:
            print(f"  fold {k}  val {row['val_from']} -> {row['val_to']}  "
                  f"n={row['val_rows']:>6,}  base {row['base_rate']:>5.1f}%  "
                  f"AUC {fmt(row['auc'])}  within-date {fmt(row['auc_within_date'])}  "
                  f"linear {fmt(row['auc_linear'])}")

    if not fold_rows:
        sys.exit("No usable folds. Lower --folds or --embargo-days.")
    folds = pd.DataFrame(fold_rows)

    print(f"\n  mean AUC              {folds['auc'].mean():.3f} ± {folds['auc'].std():.3f}")
    print(f"  mean within-date AUC  {folds['auc_within_date'].mean():.3f} "
          f"± {folds['auc_within_date'].std():.3f}")
    print(f"  mean linear AUC       {folds['auc_linear'].mean():.3f}")
    if folds["auc"].mean() <= folds["auc_linear"].mean() + 0.003:
        print("  !! XGBoost is not beating a linear model on ranks. The extra capacity "
              "is not buying anything.")

    # ── pooled out-of-fold ────────────────────────────────────────────
    m = np.isfinite(oof)
    y_oof, p_oof, d_oof = y_all[m], oof[m], df.loc[m, "_date"].to_numpy()
    r_oof = ret_all[m] if ret_all is not None else None

    print("\n" + "=" * 96)
    print(f"POOLED OUT-OF-FOLD  ({m.sum():,} rows, date-clustered 95% ranges)")
    print("=" * 96)
    auc = safe_auc(y_oof, p_oof)
    lo, hi = date_bootstrap(d_oof, y_oof, p_oof,
                            lambda d, y, p: safe_auc(y, p), a.rounds)
    wda = within_date_auc(d_oof, y_oof, p_oof)
    wlo, whi = date_bootstrap(d_oof, y_oof, p_oof, within_date_auc, a.rounds)

    print(f"  AUC              {fmt(auc)}   95% {fmt(lo)} to {fmt(hi)}"
          f"{'   <-- clear of 0.50' if lo > 0.5 else '   (touches 0.50)'}")
    print(f"  within-date AUC  {fmt(wda)}   95% {fmt(wlo)} to {fmt(whi)}"
          f"{'   <-- real stock picking' if wlo > 0.5 else '   (touches 0.50: timing only)'}")
    print("\n  Overall AUC rewards knowing WHICH WEEKS are good. Within-date AUC is the")
    print("  harder, more useful skill: ranking stocks against each other on one day.")

    # ── deciles ───────────────────────────────────────────────────────
    print("\n  DECILES (out-of-fold)")
    dec = pd.qcut(p_oof, 10, labels=False, duplicates="drop")
    print(f"    {'decile':<8}{'n':>8}{'score range':>18}{'hit rate':>11}"
          + (f"{'mean ret':>11}" if r_oof is not None else ""))
    for i in sorted(pd.unique(dec)):
        s = dec == i
        line = (f"    {i+1:<8}{int(s.sum()):>8}"
                f"{f'{p_oof[s].min():.3f}-{p_oof[s].max():.3f}':>18}"
                f"{np.mean(y_oof[s])*100:>10.1f}%")
        if r_oof is not None:
            line += f"{np.nanmean(r_oof[s]):>+11.2f}"
        print(line)

    # ── threshold sweep ───────────────────────────────────────────────
    print("\n  THRESHOLD SWEEP (out-of-fold — these are the numbers to trust)")
    sweep = threshold_sweep(y_oof, p_oof, r_oof,
                            [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    print(f"    {'thresh':>8}{'trades':>10}{'% of days':>11}{'precision':>12}{'mean ret':>11}")
    for _, r in sweep.iterrows():
        print(f"    {r['threshold']:>8.2f}{int(r['trades']):>10,}{r['share']:>10.1f}%"
              f"{r['precision']:>11.1f}%{r['mean_ret']:>+11.2f}")
    print(f"\n    Base rate to beat: {np.mean(y_oof)*100:.1f}%")
    print("    Pick the threshold where precision is clearly above base rate AND the")
    print("    trade count is still large enough to mean something. Do NOT pick the row")
    print("    with the best total return - that just takes more trades.")

    # ── final model ───────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("FINAL MODEL (trained on all rows)")
    print("=" * 96)
    final = xgb.XGBClassifier(**DEFAULT_PARAMS)
    final.fit(X_all, y_all, verbose=False)

    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)
    print("  top 15 features by gain:")
    for k, v in imp.head(15).items():
        print(f"    {k:<26}{v:.4f}")

    groups = {"cross-sectional": "xs_", "market regime": "mkt_", "bars": "bar_"}
    print("\n  share of total importance by group:")
    for name, pre in groups.items():
        s = imp[[i for i in imp.index if i.startswith(pre)]].sum()
        print(f"    {name:<20}{s:.1%}")
    print(f"    {'breadth':<20}{imp.get('breadth_above_ema200', 0):.1%}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    bundle = {
        "model": final,
        "feature_names": feats,
        "feature_version": F.FEATURE_VERSION,
        "geometry": meta.get("geometry"),
        "expired_as": meta.get("expired_as"),
        "benchmark": meta.get("benchmark"),
        "horizon": meta.get("horizon"),
        "train_end": meta.get("train_end"),
        "dataset": os.path.basename(path),
        "base_rate": float(np.mean(y_all)),
        "cv_auc": float(folds["auc"].mean()),
        "cv_auc_within_date": float(folds["auc_within_date"].mean()),
        "params": DEFAULT_PARAMS,
    }
    joblib.dump(bundle, a.out)
    print(f"\n  saved {a.out}")
    print(f"    {len(feats)} features | geometry {bundle['geometry']} | "
          f"trained through {bundle['train_end']}")

    if not a.no_shap:
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            sample = X_all.sample(min(4000, len(X_all)), random_state=0)
            sv = shap.TreeExplainer(final).shap_values(sample)
            for kind, fn in [("magnitude", dict(plot_type="bar")), ("direction", {})]:
                plt.figure()
                shap.summary_plot(sv, sample, show=False, max_display=20, **fn)
                plt.title(f"SHAP {kind} - V43 ({bundle['geometry']})")
                plt.tight_layout()
                plt.savefig(f"v43_shap_{kind}.png", dpi=110)
                plt.close()
            print("  saved v43_shap_magnitude.png / v43_shap_direction.png")
        except ImportError:
            print("  (shap or matplotlib not installed; skipping plots)")

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  The out-of-fold numbers are the honest ones. The old script's threshold")
    print("  sweep ran on a leaky fold, which is why V39 showed 60% and V42 showed 80%")
    print("  and neither survived the backtest. These folds are split by date with an")
    print(f"  {a.embargo_days}-day embargo, so a sweep number here should roughly hold up.")
    print("  Next: run the backtest on signals AFTER the training cut, then pool seeds.")


if __name__ == "__main__":
    main()
