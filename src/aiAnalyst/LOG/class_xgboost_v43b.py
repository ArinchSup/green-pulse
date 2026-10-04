#!/usr/bin/env python3
"""
class_xgboost_v43b.py - run the three V43 variants side by side and compare them on
the metric that actually generalised: WITHIN-DATE AUC.

WHY THIS EXISTS

V43's first run said two things at once:
  - within-date AUC 0.522, 95% 0.511 to 0.531, +/-0.010 across five folds. Stable,
    clear of 0.50, and it survived 2008, 2015, 2018 and 2022. Real picking skill.
  - overall AUC 0.511, 95% 0.490 to 0.530, +/-0.034 across folds, with fold 2
    (2015-02 to 2017-03, a calm market) coming in at 0.467 - BELOW chance.

The top nine features by gain were all mkt_* or breadth. Those are identical for
every ticker on a date, so they contribute EXACTLY ZERO to within-date ranking. Their
only job is ordering dates against each other, and that is the part that is unstable
and occasionally backwards. 29% of model capacity was going to a task it performs at
roughly a coin flip.

Checking where the high-confidence calls landed confirmed it: 65% of them fell in
2008-2013 and 2020, and 2015 produced none at all - while 2017, which had one of the
highest base rates in the dataset (49.1%), produced 45. The model is not firing when
trades work; it is firing after drawdowns and in high-volatility regimes. That is a
bet on "crashes recover", supported by about five episodes in the sample.

THE THREE VARIANTS

  base       every feature, binary:logistic. The V43 configuration, for reference.
  noregime   drops the nine mkt_* features and breadth_above_ema200. If within-date
             AUC holds at ~0.522, you keep the picking skill and lose the macro bet.
  rank       XGBRanker with objective rank:pairwise and qid = signal date. This asks
             the model to order stocks WITHIN each day directly, which is what
             within-date AUC measures after the fact. Also drops regime features,
             since constant-within-group columns cannot help a pairwise objective.

PERCENTILE THRESHOLDS, NOT ABSOLUTE SCORES

The CV sweep said 375 rows cleared 0.70 (0.5%). The final model - trained on all
128,760 rows including those - fires on 5,193 (4.0%) at the same cutoff. An absolute
threshold does NOT transfer from cross-validation to the deployed model, because the
deployed model is more confident on data it has seen. So every operating point here
is expressed as a PERCENTILE, and the saved bundle carries a percentile -> score map
for the final model so inference can reproduce the intended selectivity.

USAGE
  python class_xgboost_v43b.py
  python class_xgboost_v43b.py --variants noregime,rank --folds 5
  python class_xgboost_v43b.py --final noregime --out class_model/xgboost_mid_v43b.joblib
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
from sklearn.metrics import roc_auc_score

import pillar2_v43_features as F

REGIME_COLS = set(F.REGIME_FEATURES)          # mkt_* plus breadth_above_ema200

BASE_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=5,
    min_child_weight=40, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=3.0,
    tree_method="hist", n_jobs=-1, random_state=42,
)

PERCENTILES = [50, 25, 10, 5, 2, 1, 0.5]      # "top X% of scores"

VARIANTS = {
    "base":     dict(drop_regime=False, ranker=False),
    "noregime": dict(drop_regime=True,  ranker=False),
    "rank":     dict(drop_regime=True,  ranker=True),
}


# =============================================================================
# DATA
# =============================================================================
def find_dataset(pattern="dataset_pillar2_mid_v43_*"):
    hits = [p for p in glob.glob(pattern) if p.endswith((".parquet", ".csv.gz", ".csv"))]
    if not hits:
        sys.exit(f"No dataset matching '{pattern}'. Pass --dataset.")
    return sorted(hits)[-1]


def load_dataset(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    meta_path = path.split(".parquet")[0].split(".csv")[0] + "_meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    df = df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    df["_date"] = pd.to_datetime(df["signal_date"])
    return df, meta


def feature_list(df, drop_regime):
    feats = [c for c in F.FEATURE_NAMES if c in df.columns]
    if drop_regime:
        feats = [c for c in feats if c not in REGIME_COLS]
    return feats


# =============================================================================
# FOLDS
# =============================================================================
def walk_forward_folds(dates, n_folds, embargo_days):
    n = len(dates)
    start = int(n * 0.40)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        v_lo, v_hi = edges[i], edges[i + 1]
        if v_hi - v_lo < 2:
            continue
        val = dates[v_lo:v_hi]
        train = dates[dates < val[0] - pd.Timedelta(days=embargo_days)]
        if len(train) < 20:
            continue
        yield train, val


# =============================================================================
# METRICS
# =============================================================================
def safe_auc(y, p):
    y = np.asarray(y)
    return np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def within_date_auc(dates, y, p, min_rows=20):
    d = pd.DataFrame({"date": dates, "y": np.asarray(y), "p": np.asarray(p)})
    aucs, w = [], []
    for _, g in d.groupby("date"):
        if len(g) < min_rows or g["y"].nunique() < 2:
            continue
        a = safe_auc(g["y"], g["p"])
        if np.isfinite(a):
            aucs.append(a)
            w.append(len(g))
    return float(np.average(aucs, weights=w)) if aucs else np.nan


def date_bootstrap(dates, y, p, stat_fn, rounds=300, seed=0):
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


def fmt(x, spec=".3f"):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:{spec}}"


# =============================================================================
# PERCENTILE SWEEP
# =============================================================================
def percentile_sweep(fold_ids, y, p, rets, dates, pcts=PERCENTILES):
    """
    Select the top X% of scores WITHIN EACH FOLD, then pool. Per-fold selection is
    what makes this comparable across folds and across variants: ranker scores are
    not probabilities and different folds sit on different scales, so an absolute
    cutoff would silently mean different selectivity in each.
    """
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for q in pcts:
        sel = np.zeros(len(p), dtype=bool)
        for f in np.unique(fold_ids):
            m = fold_ids == f
            if m.sum() == 0:
                continue
            sel[m] = p[m] >= np.percentile(p[m], 100 - q)
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append({
            "top_pct": q, "trades": n,
            "precision": float(np.mean(y[sel])) * 100,
            "mean_ret": float(np.nanmean(np.asarray(rets)[sel])) if rets is not None else np.nan,
            "dates": int(pd.Series(np.asarray(dates)[sel]).nunique()),
            "years": int(pd.Series(np.asarray(dates)[sel]).astype("datetime64[ns]")
                         .dt.year.nunique()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# TOP-N PER DATE  -  how the model will ACTUALLY be used
# =============================================================================
def topn_per_date(dates, y, p, rets, ns=(1, 3, 5, 10), min_cohort=20):
    """
    A global threshold is the wrong question for a ranker. In deployment you score
    the whole universe each morning and take the best few names - you never ask
    "is this score above 0.70?".

    This measures exactly that: for every date, take the top N by score, pool the
    picks, and compare against the base rate OF THOSE SAME DATES. Comparing to the
    overall base rate would let a model look good purely by firing on good dates;
    comparing within the date removes that entirely, so what is left is stock
    picking and nothing else.
    """
    d = pd.DataFrame({"date": np.asarray(dates), "y": np.asarray(y),
                      "p": np.asarray(p),
                      "r": np.asarray(rets) if rets is not None else np.nan})
    groups = [g for _, g in d.groupby("date") if len(g) >= min_cohort]
    if not groups:
        return pd.DataFrame()

    rng = np.random.default_rng(0)
    rows = []
    for n in ns:
        picks = [g.nlargest(n, "p") for g in groups]
        bases = np.array([g["y"].mean() for g in groups])
        pooled = pd.concat(picks)
        prec = [g["y"].mean() for g in picks]

        # clustered bootstrap: resample whole DATES, recompute the lift
        lifts = []
        k = len(groups)
        for _ in range(300):
            sel = rng.integers(0, k, size=k)
            lifts.append(float(np.mean([prec[i] for i in sel])
                               - np.mean([bases[i] for i in sel])) * 100)
        lo, hi = np.percentile(lifts, 2.5), np.percentile(lifts, 97.5)

        rows.append({"n": n, "trades": len(pooled), "dates": len(groups),
                     "precision": float(np.mean(prec)) * 100,
                     "same_date_base": float(np.mean(bases)) * 100,
                     "lift": float(np.mean(prec) - np.mean(bases)) * 100,
                     "lift_lo": lo, "lift_hi": hi,
                     "mean_ret": float(np.nanmean(pooled["r"]))})
    return pd.DataFrame(rows)


# =============================================================================
# ONE VARIANT
# =============================================================================
def run_variant(name, df, meta, cfg):
    spec = VARIANTS[name]
    feats = feature_list(df, spec["drop_regime"])
    dates = np.array(sorted(df["_date"].unique()))
    X_all, y_all = df[feats], df["label"].to_numpy()
    rets = df["exit_return_pct"].to_numpy() if "exit_return_pct" in df.columns else None

    print("\n" + "=" * 100)
    print(f"VARIANT: {name}   ({len(feats)} features"
          f"{', regime dropped' if spec['drop_regime'] else ''}"
          f"{', rank:pairwise by date' if spec['ranker'] else ''})")
    print("=" * 100)

    oof = np.full(len(df), np.nan)
    fold_id = np.full(len(df), -1)
    rows = []
    for k, (tr_dates, va_dates) in enumerate(
            walk_forward_folds(dates, cfg.folds, cfg.embargo_days), 1):
        tr_m = df["_date"].isin(tr_dates).to_numpy()
        va_m = df["_date"].isin(va_dates).to_numpy()

        if spec["ranker"]:
            # XGBRanker needs rows grouped by qid and qid sorted; the frame is already
            # sorted by signal_date, so a factorised date is a valid qid.
            qid_tr = pd.factorize(df.loc[tr_m, "_date"])[0]
            model = xgb.XGBRanker(objective="rank:pairwise", **BASE_PARAMS)
            model.fit(X_all[tr_m], y_all[tr_m], qid=qid_tr, verbose=False)
            p = model.predict(X_all[va_m])
        else:
            model = xgb.XGBClassifier(objective="binary:logistic",
                                      eval_metric="auc", **BASE_PARAMS)
            model.fit(X_all[tr_m], y_all[tr_m], verbose=False)
            p = model.predict_proba(X_all[va_m])[:, 1]

        oof[va_m] = p
        fold_id[va_m] = k
        yva = y_all[va_m]
        r = {"fold": k,
             "val_from": pd.Timestamp(va_dates[0]).date(),
             "val_to": pd.Timestamp(va_dates[-1]).date(),
             "n": int(va_m.sum()), "base": float(np.mean(yva)) * 100,
             "auc": safe_auc(yva, p),
             "wd_auc": within_date_auc(df.loc[va_m, "_date"], yva, p)}
        rows.append(r)
        print(f"  fold {k}  {r['val_from']} -> {r['val_to']}  n={r['n']:>6,}  "
              f"base {r['base']:>5.1f}%  AUC {fmt(r['auc'])}  "
              f"within-date {fmt(r['wd_auc'])}")

    folds = pd.DataFrame(rows)
    m = np.isfinite(oof)
    y_o, p_o, d_o, f_o = y_all[m], oof[m], df.loc[m, "_date"].to_numpy(), fold_id[m]
    r_o = rets[m] if rets is not None else None

    wd = within_date_auc(d_o, y_o, p_o)
    wlo, whi = date_bootstrap(d_o, y_o, p_o, within_date_auc, cfg.rounds)
    au = safe_auc(y_o, p_o)
    alo, ahi = date_bootstrap(d_o, y_o, p_o, lambda d, y, p: safe_auc(y, p), cfg.rounds)

    print(f"\n  pooled overall AUC      {fmt(au)}  95% {fmt(alo)} to {fmt(ahi)}"
          f"{'  <-- clear of 0.50' if alo > 0.5 else ''}")
    print(f"  pooled within-date AUC  {fmt(wd)}  95% {fmt(wlo)} to {fmt(whi)}"
          f"{'  <-- clear of 0.50' if wlo > 0.5 else ''}")
    print(f"  fold spread: overall +/-{folds['auc'].std():.3f}, "
          f"within-date +/-{folds['wd_auc'].std():.3f}   "
          f"(worst fold overall {folds['auc'].min():.3f})")

    sweep = percentile_sweep(f_o, y_o, p_o, r_o, d_o)
    print(f"\n  SELECTIVITY (top X% of scores within each fold)")
    print(f"    {'top %':>7}{'trades':>9}{'precision':>12}{'mean ret':>11}"
          f"{'dates':>8}{'years':>7}")
    for _, r in sweep.iterrows():
        flag = "  <- few dates" if r["dates"] < 25 else ""
        print(f"    {r['top_pct']:>7}{int(r['trades']):>9,}{r['precision']:>11.1f}%"
              f"{r['mean_ret']:>+11.2f}{int(r['dates']):>8}{int(r['years']):>7}{flag}")
    print(f"    base rate {np.mean(y_o)*100:.1f}%   "
          f"'dates' and 'years' show whether a bucket is spread across time "
          f"or concentrated in a few episodes.")

    tn = topn_per_date(d_o, y_o, p_o, r_o)
    if len(tn):
        print(f"\n  TOP-N PER DATE (score the whole cohort, take the best N - "
              f"how the tool will actually run)")
        print(f"    {'N':>4}{'trades':>9}{'dates':>8}{'precision':>12}"
              f"{'same-day base':>15}{'lift':>9}{'95% range':>20}{'mean ret':>11}")
        for _, r in tn.iterrows():
            clear = "" if r["lift_lo"] <= 0 else "  <- clear of zero"
            print(f"    {int(r['n']):>4}{int(r['trades']):>9,}{int(r['dates']):>8}"
                  f"{r['precision']:>11.1f}%{r['same_date_base']:>14.1f}%"
                  f"{r['lift']:>+8.1f}p"
                  f"{f'{r.lift_lo:+.1f} to {r.lift_hi:+.1f}':>20}"
                  f"{r['mean_ret']:>+11.2f}{clear}")
        print("    Lift is measured against the base rate of THE SAME DATES, so it")
        print("    cannot be earned by firing on good days. This is pure picking.")

    return {"name": name, "feats": feats, "spec": spec, "folds": folds, "topn": tn,
            "auc": au, "auc_lo": alo, "auc_hi": ahi,
            "wd": wd, "wd_lo": wlo, "wd_hi": whi,
            "wd_std": float(folds["wd_auc"].std()),
            "auc_std": float(folds["auc"].std()),
            "auc_worst": float(folds["auc"].min()),
            "sweep": sweep}


# =============================================================================
# FINAL MODEL
# =============================================================================
def fit_final(df, meta, res, out_path):
    feats, spec = res["feats"], res["spec"]
    X, y = df[feats], df["label"].to_numpy()
    if spec["ranker"]:
        qid = pd.factorize(df["_date"])[0]
        model = xgb.XGBRanker(objective="rank:pairwise", **BASE_PARAMS)
        model.fit(X, y, qid=qid, verbose=False)
        scores = model.predict(X)
    else:
        model = xgb.XGBClassifier(objective="binary:logistic",
                                  eval_metric="auc", **BASE_PARAMS)
        model.fit(X, y, verbose=False)
        scores = model.predict_proba(X)[:, 1]

    # percentile -> score map, so a deployed operating point means the same
    # selectivity the cross-validation measured
    pmap = {str(q): float(np.percentile(scores, 100 - q)) for q in PERCENTILES}

    print("\n" + "=" * 100)
    print(f"FINAL MODEL: {res['name']}")
    print("=" * 100)
    print("  percentile -> score on the FINAL model (use these, not the CV cutoffs):")
    for q in PERCENTILES:
        print(f"    top {q:>4}%   score >= {pmap[str(q)]:.4f}")

    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\n  top 12 features by gain:")
    for k, v in imp.head(12).items():
        print(f"    {k:<26}{v:.4f}")
    for label, pre in [("cross-sectional", "xs_"), ("bars", "bar_"),
                       ("relative strength", "rs_")]:
        s = imp[[i for i in imp.index if i.startswith(pre)]].sum()
        print(f"  {label:<20}{s:>7.1%} of total importance")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({
        "model": model, "feature_names": feats,
        "variant": res["name"], "is_ranker": spec["ranker"],
        "feature_version": F.FEATURE_VERSION,
        "geometry": meta.get("geometry"), "expired_as": meta.get("expired_as"),
        "benchmark": meta.get("benchmark"), "horizon": meta.get("horizon"),
        "train_end": meta.get("train_end"),
        "percentile_scores": pmap,
        "cv_within_date_auc": res["wd"], "cv_auc": res["auc"],
        "base_rate": float(np.mean(y)), "params": BASE_PARAMS,
    }, out_path)
    print(f"\n  saved {out_path}")
    if spec["ranker"]:
        print("  NOTE: this is a RANKER. predict() returns a score, not a probability.")
        print("  Inference must score the whole cohort and take the top N by percentile.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--variants", default="base,noregime,rank")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--final", default=None,
                    help="which variant to fit and save; default = best within-date AUC")
    ap.add_argument("--out", default=os.path.join("class_model", "xgboost_mid_v43b.joblib"))
    ap.add_argument("--no-final", action="store_true")
    cfg = ap.parse_args()

    path = cfg.dataset or find_dataset()
    df, meta = load_dataset(path)

    print("=" * 100)
    print("PILLAR 2 V43b - VARIANT COMPARISON")
    print("=" * 100)
    print(f"  dataset: {path}")
    print(f"  rows {len(df):,} | tickers {df['ticker'].nunique()} | "
          f"dates {df['signal_date'].nunique()} | base rate {df['label'].mean():.1%}")
    print(f"  span {df['signal_date'].min()} -> {df['signal_date'].max()}")

    names = [v.strip() for v in cfg.variants.split(",") if v.strip()]
    bad = [v for v in names if v not in VARIANTS]
    if bad:
        sys.exit(f"Unknown variant(s) {bad}. Choose from {list(VARIANTS)}.")

    results = [run_variant(n, df, meta, cfg) for n in names]

    print("\n" + "=" * 100)
    print("COMPARISON  (within-date AUC is the one that generalised; regime features")
    print("             contribute exactly zero to it, by construction)")
    print("=" * 100)
    print(f"  {'variant':<12}{'feats':>7}{'within-date':>13}{'95% range':>18}"
          f"{'+/-folds':>10}{'overall':>9}{'worst fold':>12}")
    for r in results:
        print(f"  {r['name']:<12}{len(r['feats']):>7}{fmt(r['wd']):>13}"
              f"{fmt(r['wd_lo']) + ' to ' + fmt(r['wd_hi']):>18}"
              f"{r['wd_std']:>10.3f}{fmt(r['auc']):>9}{r['auc_worst']:>12.3f}")

    print("\n  What to look for:")
    print("   - noregime holding within-date AUC near base: the picking skill was never")
    print("     coming from the macro features, and dropping them removes a bet on")
    print("     'drawdowns recover' that rests on about five episodes.")
    print("   - rank beating both: optimising order within the day directly is the")
    print("     right objective for a tool that must choose among today's candidates.")
    print("   - a variant whose worst fold is below 0.50 is unreliable in some regime,")
    print("     however good its average looks.")

    if not cfg.no_final:
        pick = cfg.final
        if pick is None:
            # Do NOT pick on within-date AUC alone: the variants land within ~0.002
            # of each other, which is far inside the noise. Break the tie on the
            # worst fold, because a variant that goes below 0.50 in some regime is
            # unreliable however good its average looks.
            best_wd = max(r["wd"] for r in results if np.isfinite(r["wd"]))
            close = [r for r in results if np.isfinite(r["wd"])
                     and best_wd - r["wd"] <= 0.005]
            pick = max(close, key=lambda r: r["auc_worst"])["name"]
            print(f"\n  within-date AUC is tied within 0.005 across "
                  f"{len(close)} variant(s); breaking the tie on worst-fold "
                  f"stability -> {pick}")
        res = next((r for r in results if r["name"] == pick), None)
        if res is None:
            sys.exit(f"--final {pick} was not among the variants run.")
        fit_final(df, meta, res, cfg.out)


if __name__ == "__main__":
    main()
