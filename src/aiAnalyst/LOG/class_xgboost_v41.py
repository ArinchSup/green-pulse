"""
class_xgboost_v41.py - train and honestly validate the Pillar 2 V41 model

  python class_xgboost_v41.py --dataset dataset_pillar2_mid_v41_qqq_cut20250627.json

Steps
  1. Features come from pillar2_v41_features.py (the same code inference uses), all scale-free.
  2. Purged walk-forward cross-validation: the date range is cut into blocks and each fold
     trains only on rows dated at least GAP_DAYS before its test block, so no training label
     reaches into the test period.
  3. The report leads with ranking skill: AUC per fold and on all out-of-fold predictions
     (0.50 = coin flip), plus a table by score group. This is the check V40 failed (0.505).
  4. The final model is trained on all rows and saved next to a metadata file:
       <prefix>_model.json   the XGBoost model
       <prefix>_meta.json    features, training dates, label, cutoffs, CV results, dataset hash
     The Bullish / Neutral cutoffs are the 90th / 50th percentiles of the out-of-fold scores,
     so the model calls Bullish on roughly its top 10% of setups. They come from training data
     only; nothing is tuned on the backtest window.
"""
import argparse
import datetime
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import xgboost as xgb

from pillar2_v41_features import FEATURE_NAMES, FEATURE_VERSION, to_vector

N_FOLDS = 4
GAP_DAYS = 92          # the label looks 60 trading days (~90 calendar days) ahead
NUM_ROUNDS = 300
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 3,          # shallow trees: 5,000 noisy rows can't support deep interactions
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "lambda": 2.0,
    "seed": 42,
}
BULLISH_PCTL = 90
NEUTRAL_PCTL = 50
BOOT_ROUNDS = 1000


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------
def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("signal_date") and r.get("input")]
    rows.sort(key=lambda r: r["signal_date"])
    X = np.array([to_vector(r["input"]) for r in rows], dtype=float)
    y = np.array([r["output"]["technical_sentiment"] == "Bullish" for r in rows])
    sims = [r["output"].get("trade_simulation", {}) or {} for r in rows]
    excess = np.array([s.get("excess_return_pct", s.get("exit_return_pct", np.nan)) for s in sims],
                      dtype=float)
    dates = np.array([r["signal_date"][:10] for r in rows], dtype="datetime64[D]")
    clusters = np.array([f"{r['ticker']}|{r['signal_date'][:7]}" for r in rows])
    modes = sorted({str(s.get("label_mode", "unknown")) for s in sims})
    return X, y, excess, dates, clusters, modes


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------
def auc(y, s):
    """Chance that a random positive scores above a random negative (0.5 = coin flip)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_range(y, s, clusters, rounds=BOOT_ROUNDS, seed=0):
    """95% range of the AUC, resampling whole ticker-months (nearby days are near-copies)."""
    codes, uniq = pd.factorize(pd.Series(clusters))
    order = np.argsort(np.asarray(s, float), kind="mergesort")
    y_o, c_o = np.asarray(y, bool)[order], codes[order]
    rng = np.random.default_rng(seed)
    vals = []
    for done in range(0, rounds, 100):
        k = min(100, rounds - done)
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=k)[:, c_o].astype(np.float32)
        pos, neg = w * y_o, w * ~y_o
        below = np.cumsum(neg, axis=1) - neg
        with np.errstate(invalid="ignore", divide="ignore"):
            vals.append((pos * below).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1)))
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [float("nan")] * 2


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
def fit(X, y):
    return xgb.train(XGB_PARAMS, xgb.DMatrix(X, label=y.astype(int), feature_names=FEATURE_NAMES),
                     num_boost_round=NUM_ROUNDS)


def predict(booster, X):
    return booster.predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES))


def purged_folds(dates, n_folds=N_FOLDS, gap_days=GAP_DAYS):
    """Contiguous date blocks; block 0 only ever trains. Each fold trains strictly before its block."""
    blocks = np.array_split(np.unique(dates), n_folds + 1)
    for k in range(1, n_folds + 1):
        start, end = blocks[k][0], blocks[k][-1]
        train = np.flatnonzero(dates < start - np.timedelta64(gap_days, "D"))
        test = np.flatnonzero((dates >= start) & (dates <= end))
        yield k, train, test, start, end


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train and validate the Pillar 2 V41 model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-prefix", default=None, help="default: pillar2_v41_cut<cutoff from dataset name>")
    ap.add_argument("--benchmark", default=None, help="default: taken from the dataset name, else QQQ")
    a = ap.parse_args()

    X, y, excess, dates, clusters, modes = load_dataset(a.dataset)
    name = os.path.basename(a.dataset)
    m_cut = re.search(r"cut(\d{8})", name)
    m_bench = re.search(r"_v41_([a-z]+)_cut", name)
    benchmark = a.benchmark or (m_bench.group(1).upper() if m_bench else "QQQ")
    prefix = a.out_prefix or f"pillar2_v41_cut{m_cut.group(1) if m_cut else str(dates.max()).replace('-', '')}"

    line = "=" * 92
    print(line)
    print(f"PILLAR 2 V41 TRAINING | {name}")
    print(f"{len(y)} rows | signals {dates.min()} -> {dates.max()} | {len(FEATURE_NAMES)} scale-free features "
          f"({FEATURE_VERSION})")
    print(f"Label: {', '.join(modes)} vs {benchmark} | base rate (share that beat it): {y.mean():.1%}")
    print(line)
    if "BEAT_BENCHMARK" not in modes:
        print("!! These rows were not built with LABEL_MODE = BEAT_BENCHMARK. Use class_pillar2_data_gen_v41.py.")
    if np.isnan(X[:, FEATURE_NAMES.index("rs_60")]).mean() > 0.5:
        print("!! Most rows have no relative-strength inputs. Use class_pillar2_data_gen_v41.py.")

    # ---- 1) purged walk-forward CV --------------------------------------------------------
    print(f"\n1) PURGED WALK-FORWARD CV  ({N_FOLDS} folds, {GAP_DAYS}-day gap between training and test)")
    print(f"   {'Fold':<6}{'Test dates':<26}{'Train':>7}{'Test':>7}{'Base rate':>11}{'AUC':>8}"
          f"{'Top 20% excess':>16}{'Bottom 20%':>12}")
    oof = np.full(len(y), np.nan)
    folds = []
    for k, tr, te, start, end in purged_folds(dates):
        if len(tr) < 200 or len(te) < 50:
            print(f"   {k:<6}{str(start) + ' -> ' + str(end):<26}{len(tr):>7}{len(te):>7}   skipped (too few rows)")
            continue
        p = predict(fit(X[tr], y[tr]), X[te])
        oof[te] = p
        q20, q80 = np.quantile(p, [0.2, 0.8])
        top, bot = excess[te][p >= q80].mean(), excess[te][p <= q20].mean()
        fa = auc(y[te], p)
        folds.append({"fold": k, "test_start": str(start), "test_end": str(end), "train_rows": int(len(tr)),
                      "test_rows": int(len(te)), "base_rate": float(y[te].mean()), "auc": fa,
                      "top20_excess": float(top), "bottom20_excess": float(bot)})
        print(f"   {k:<6}{str(start) + ' -> ' + str(end):<26}{len(tr):>7}{len(te):>7}{y[te].mean():>11.1%}"
              f"{fa:>8.3f}{top:>+15.2f}%{bot:>+11.2f}%")

    valid = np.isfinite(oof)
    if valid.sum() < 100:
        raise SystemExit("Not enough out-of-fold predictions to evaluate. Use a bigger dataset.")
    oof_auc = auc(y[valid], oof[valid])
    lo, hi = auc_range(y[valid], oof[valid], clusters[valid])

    # ---- 2) ranking on all out-of-fold predictions ------------------------------------------
    print(f"\n2) RANKING ON ALL {int(valid.sum())} OUT-OF-FOLD PREDICTIONS  (does a higher score mean a better outcome?)")
    frame = pd.DataFrame({"score": oof[valid], "beat": y[valid], "excess": excess[valid]})
    frame["group"] = pd.qcut(frame["score"], 10, labels=False, duplicates="drop") + 1
    print(f"   {'Group':<7}{'Score range':<15}{'Rows':>6}{'Beat ' + benchmark:>10}{'Avg excess':>12}")
    for g, s in frame.groupby("group"):
        print(f"   {int(g):<7}{s.score.min():.3f}-{s.score.max():.3f}  {len(s):>6}{s.beat.mean():>10.1%}"
              f"{s.excess.mean():>+11.2f}%")
    q20, q80 = frame.score.quantile([0.2, 0.8])
    spread = frame.excess[frame.score >= q80].mean() - frame.excess[frame.score <= q20].mean()
    print(f"   AUC {oof_auc:.3f}  (95% range {lo:.3f} to {hi:.3f}) | top 20% minus bottom 20% excess: {spread:+.2f} pts")
    if lo > 0.5:
        verdict = "the range sits above 0.50: a real ranking signal on training-period data"
    elif hi < 0.5:
        verdict = "the range sits below 0.50: the scores point the wrong way"
    else:
        verdict = "the range includes 0.50: no reliable ranking signal yet"
    print(f"   Verdict: {verdict}.")
    print("   The backtest on the unseen window has the final say.")

    # ---- 3) final model ------------------------------------------------------------------------
    final = fit(X, y)
    bull_cut, neut_cut = (float(v) for v in np.percentile(oof[valid], [BULLISH_PCTL, NEUTRAL_PCTL]))
    gain = final.get_score(importance_type="gain")
    top_feats = sorted(gain.items(), key=lambda kv: -kv[1])[:12]
    print("\n3) FINAL MODEL (all rows) - top features by gain")
    for fname, g in top_feats:
        print(f"   {fname:<26}{g:>10.2f}")

    model_path, meta_path = f"{prefix}_model.json", f"{prefix}_meta.json"
    final.save_model(model_path)
    meta = {
        "version": "V41",
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "label_mode": modes,
        "benchmark": benchmark,
        "horizon_bars": 60,
        "train_rows": int(len(y)),
        "train_signal_dates": [str(dates.min()), str(dates.max())],
        "base_rate": float(y.mean()),
        "bullish_cutoff": bull_cut,
        "neutral_cutoff": neut_cut,
        "cv": {"n_folds": N_FOLDS, "gap_days": GAP_DAYS, "folds": folds,
               "oof_auc": oof_auc, "oof_auc_range": [lo, hi], "top20_minus_bottom20_excess": float(spread)},
        "xgb_params": XGB_PARAMS,
        "num_rounds": NUM_ROUNDS,
        "dataset_file": name,
        "dataset_sha256": file_sha256(a.dataset),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBullish when score >= {bull_cut:.4f} (top {100 - BULLISH_PCTL}%), "
          f"Neutral when >= {neut_cut:.4f}, Bearish below.")
    print(f"Saved: {model_path}\n       {meta_path}")
    print("\nNext step (out-of-sample backtest):")
    print(f"  python class_ai_pipeline_backtest_v2.py --pillar2-module class_ai_pillar2_v41 "
          f"--pillar2-model {model_path} --model-label V41 --min-confidence 0 --training-dataset {a.dataset}")
    print(line)


if __name__ == "__main__":
    main()
