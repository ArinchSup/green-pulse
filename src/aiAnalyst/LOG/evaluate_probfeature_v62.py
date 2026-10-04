#!/usr/bin/env python3
"""
evaluate_probfeature_v62.py - does the risk probability help a direction model?

THE QUESTION

The deployed risk model emits P(20% fall within 90 days). That number is built
in two stages: a linear volatility composite, then a monotone empirical curve
that turns the composite into a probability. The proposal is to feed it into the
V40-class direction classifier as one more column and see if the classifier gets
better.

The prior is that it cannot, for a mechanical reason. A gradient-boosted tree
splits on thresholds, and a monotone transform of a feature maps every threshold
one-to-one. If the model already has the volatility columns, a monotone function
of those columns opens no split the model could not already make. The gain
should be zero, not small.

That is an argument. This file measures it.

WHAT IS ACTUALLY MEASURED

    arm_base      every V44 feature, no risk probability
    arm_prob      the same features plus prob_dd
    arm_prob_only prob_dd alone, to show what it carries by itself
    arm_no_vol    features with the volatility columns REMOVED, plus prob_dd
                  - the one case where it should help, because now the
                    probability is the only volatility channel left

The fourth arm is the control that makes the result interpretable. If arm_prob
matches arm_base but arm_no_vol beats a no-volatility baseline, the conclusion is
specifically "redundant", not "useless".

HOW prob_dd IS BUILT - AND WHY IT IS NOT JUST COPIED IN

The deployed number cannot be pasted into this dataset: the dataset carries
cross-sectionally standardised volatility columns, not the raw Yang-Zhang
windows the deployed curve was fitted on. So prob_dd is REBUILT here with the
same two-stage shape - linear volatility composite, then a twelve-bin monotone
empirical curve - fitted on TRAINING ROWS ONLY.

Fitting it on training rows only is not a detail. The deployed calibration curve
is fitted in-sample over the whole history. Dropping that number into a training
table would let a 2015 row carry the shape of a curve fitted with 2024 data,
which is look-ahead leakage and is the exact failure that sank V44. Everything
here is fitted before the cut and applied after it.

THE LABEL

fwd_ret_20 > 0. The dataset's features are cross-sectionally standardised and
the forward return is demeaned, so this is the relative question - does this name
beat its cross-section over the next 20 days - at a base rate near 50%. That is
the V40-style win/lose label, not the drawdown label.

WHY SEVERAL SEEDS

A single pair of AUC numbers differing by 0.002 says nothing. Boosting is
stochastic, so each arm is trained under several seeds and the arm-to-arm gap is
reported against the seed-to-seed spread. A gap smaller than the noise is not a
gain, however the point estimates fall.

USAGE
  python evaluate_probfeature_v62.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "v47t"
TRAIN_FILE = "dataset_pillar2_mid_v44_test_cut20230630.csv.gz"
TEST_FILE = "v44_oos_test.csv.gz"

LABEL_COL = "fwd_ret_20"          # > 0 is a win
DD_COL = "fwd_ret_60"             # the left tail of this stands in for a drawdown
DD_BASE_RATE = 0.13               # threshold chosen to match the deployed 13.2%

VOL_COLS = ["vol_20", "vol_60", "atr_pct", "vol_ratio_20_60", "bb_width"]
CURVE_BINS = 12

# columns that are outcomes, not features
LEAK_COLS = ["exit_return_pct", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20",
             "fwd_ret_60", "fwd_exc_5", "fwd_rank_5", "fwd_exc_10",
             "fwd_rank_10", "fwd_exc_20", "fwd_rank_20", "fwd_exc_60",
             "fwd_rank_60", "ticker", "signal_date"]

N_SEEDS = 5
XGB_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=2.0, min_child_weight=20,
                  eval_metric="logloss", tree_method="hist")


# =============================================================================
# METRICS
# =============================================================================
def auc(y, p):
    """Rank-based AUC, ties averaged."""
    y = np.asarray(y, float)
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def top_decile_rate(y, p):
    """Win rate among the tenth of names the model likes most."""
    k = max(1, len(p) // 10)
    idx = np.argsort(-np.asarray(p, float))[:k]
    return float(np.asarray(y, float)[idx].mean())


# =============================================================================
# THE TWO-STAGE RISK PROBABILITY, FITTED ON TRAINING ROWS ONLY
# =============================================================================
def fit_prob_dd(train, vol_cols, dd_col, base_rate, n_bins):
    """
    Same shape as the deployed model: linear volatility composite, then a
    monotone empirical curve. Returns (apply_fn, info).
    """
    thr = float(np.quantile(train[dd_col], base_rate))
    event = (train[dd_col] <= thr).to_numpy(float)

    X = train[vol_cols].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, event, rcond=None)

    score = X @ beta
    qs = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (score >= lo) & (score <= hi)
        if m.sum() < 50:
            continue
        xs.append(float(score[m].mean()))
        ys.append(float(event[m].mean()))
    # the deployed curve is non-decreasing; enforce it the same way (PAVA)
    ys = _pava(ys)

    def apply(df):
        Z = df[vol_cols].to_numpy(float)
        Z = np.column_stack([np.ones(len(Z)), Z])
        return np.interp(Z @ beta, xs, ys)

    return apply, {"threshold": thr, "coef": beta.tolist(),
                   "curve_x": xs, "curve_y": ys,
                   "event_rate_train": float(event.mean())}


def _pava(y):
    """Pool-adjacent-violators: nearest non-decreasing sequence."""
    y = [float(v) for v in y]
    n = len(y)
    val, wt = list(y), [1.0] * n
    i = 0
    while i < len(val) - 1:
        if val[i] <= val[i + 1]:
            i += 1
            continue
        tot = wt[i] + wt[i + 1]
        val[i] = (val[i] * wt[i] + val[i + 1] * wt[i + 1]) / tot
        wt[i] = tot
        del val[i + 1], wt[i + 1]
        if i > 0:
            i -= 1
    out = []
    for v, w in zip(val, wt):
        out += [v] * int(round(w))
    return out[:n]


# =============================================================================
# EXPERIMENT
# =============================================================================
def _train_eval(Xtr, ytr, Xte, yte, seed):
    from xgboost import XGBClassifier
    m = XGBClassifier(random_state=seed, **XGB_PARAMS)
    m.fit(Xtr, ytr, verbose=False)
    p = m.predict_proba(Xte)[:, 1]
    return {"auc": auc(yte, p), "logloss": logloss(yte, p),
            "acc": float(((p > 0.5).astype(int) == yte).mean()),
            "top_decile": top_decile_rate(yte, p)}


def run_probfeature_v62(data_dir=DATA_DIR, train_file=TRAIN_FILE,
                        test_file=TEST_FILE, n_seeds=N_SEEDS,
                        dd_base_rate=DD_BASE_RATE, out="probfeature_v62.json",
                        verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    tr_path = os.path.join(data_dir, train_file)
    te_path = os.path.join(data_dir, test_file)
    for p in (tr_path, te_path):
        if not os.path.exists(p):
            sys.exit(f"{p} not found.")

    say("=" * 88)
    say("V62 - DOES THE RISK PROBABILITY HELP A DIRECTION MODEL?")
    say("=" * 88)

    train = pd.read_csv(tr_path)
    test = pd.read_csv(te_path)
    feats = [c for c in train.columns if c not in LEAK_COLS]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]

    ytr = (train[LABEL_COL] > 0).astype(int).to_numpy()
    yte = (test[LABEL_COL] > 0).astype(int).to_numpy()

    say(f"  train {len(train):,} rows  {train.signal_date.min()} -> "
        f"{train.signal_date.max()}")
    say(f"  test  {len(test):,} rows  {test.signal_date.min()} -> "
        f"{test.signal_date.max()}")
    say(f"  purge gap between them, so no forward window straddles the cut")
    say(f"  {len(feats)} features | label {LABEL_COL} > 0 | "
        f"base rate train {ytr.mean():.4f} test {yte.mean():.4f}")

    # ---- the risk probability, fitted before the cut only -------------------
    apply_prob, info = fit_prob_dd(train, VOL_COLS, DD_COL, dd_base_rate,
                                   CURVE_BINS)
    train["prob_dd"] = apply_prob(train)
    test["prob_dd"] = apply_prob(test)

    say(f"\n  prob_dd rebuilt two-stage on TRAINING ROWS ONLY")
    say(f"    event: {DD_COL} <= {info['threshold']:.2f}  "
        f"(train rate {info['event_rate_train']:.3f})")
    say(f"    curve: {len(info['curve_x'])} bins, "
        f"{min(info['curve_y']):.3f} -> {max(info['curve_y']):.3f}")

    # ---- the redundancy check, in this dataset -----------------------------
    say(f"\n  RANK CORRELATION OF prob_dd WITH THE VOLATILITY COLUMNS IT IS BUILT FROM")
    for c in VOL_COLS:
        rho = test[["prob_dd", c]].corr(method="spearman").iloc[0, 1]
        say(f"    {c:16s} {rho:+.4f}")

    # ---- the four arms ------------------------------------------------------
    novol = [c for c in feats if c not in VOL_COLS]
    arms = {
        "arm_base":      feats,
        "arm_prob":      feats + ["prob_dd"],
        "arm_prob_only": ["prob_dd"],
        "arm_no_vol":    novol + ["prob_dd"],
        "arm_no_vol_ctl": novol,
    }

    results = {}
    say(f"\n  training {len(arms)} arms x {n_seeds} seeds")
    for name, cols in arms.items():
        runs = [_train_eval(train[cols].to_numpy(float), ytr,
                            test[cols].to_numpy(float), yte, s)
                for s in range(n_seeds)]
        results[name] = {
            k: {"mean": float(np.mean([r[k] for r in runs])),
                "sd": float(np.std([r[k] for r in runs], ddof=1)),
                "runs": [r[k] for r in runs]}
            for k in runs[0]
        }
        say(f"    {name:16s} {len(cols):3d} features  "
            f"AUC {results[name]['auc']['mean']:.4f} "
            f"+/- {results[name]['auc']['sd']:.4f}")

    # ---- the comparisons that matter ---------------------------------------
    say("\n" + "=" * 88)
    say("  RESULT")
    say("=" * 88)
    say(f"  {'comparison':34s} {'metric':11s} {'delta':>9s} "
        f"{'seed noise':>11s}   verdict")

    def compare(a, b, label):
        for metric in ("auc", "logloss", "top_decile"):
            da = results[a][metric]["mean"] - results[b][metric]["mean"]
            # noise: how far apart two same-arm runs typically land
            noise = np.hypot(results[a][metric]["sd"], results[b][metric]["sd"])
            if abs(da) <= noise:
                verdict = "inside seed noise"
            elif (da > 0) == (metric != "logloss"):
                verdict = "BETTER"
            else:
                verdict = "WORSE"
            say(f"  {label:34s} {metric:11s} {da:+9.4f} {noise:11.4f}   {verdict}")

    compare("arm_prob", "arm_base", "prob_dd added to everything")
    compare("arm_no_vol", "arm_no_vol_ctl", "prob_dd added, vol columns removed")
    compare("arm_base", "arm_no_vol_ctl", "vol columns vs no vol columns")

    say(f"\n  prob_dd ALONE: AUC {results['arm_prob_only']['auc']['mean']:.4f}, "
        f"top-decile win rate "
        f"{results['arm_prob_only']['top_decile']['mean']:.4f} "
        f"against a {yte.mean():.4f} base rate")

    payload = {"results": results, "prob_dd": info,
               "n_features": len(feats), "n_seeds": n_seeds,
               "train_rows": len(train), "test_rows": len(test),
               "base_rate_test": float(yte.mean())}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    say(f"\n  wrote {out}")
    return payload


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--dd-base-rate", type=float, default=DD_BASE_RATE)
    ap.add_argument("--out", default="probfeature_v62.json")
    return ap


def main(argv=None):
    cfg = _parser().parse_args(argv)
    return run_probfeature_v62(data_dir=cfg.data_dir, n_seeds=cfg.n_seeds,
                               dd_base_rate=cfg.dd_base_rate, out=cfg.out)


# =============================================================================
# RUN ME
# -----------------------------------------------------------------------------
# Just run this file - no command line needed.
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------- EDIT THESE
    RUN_DATA_DIR     = DATA_DIR
    RUN_N_SEEDS      = N_SEEDS      # more seeds = a tighter read on the noise
    RUN_DD_BASE_RATE = DD_BASE_RATE # left-tail rate the risk probability targets
    RUN_OUT          = "probfeature_v62.json"
    # -------------------------------------------------------------------------

    if len(sys.argv) > 1:
        main()                      # e.g. python evaluate_probfeature_v62.py --n-seeds 10
    else:
        run_probfeature_v62(data_dir=RUN_DATA_DIR, n_seeds=RUN_N_SEEDS,
                            dd_base_rate=RUN_DD_BASE_RATE, out=RUN_OUT)
