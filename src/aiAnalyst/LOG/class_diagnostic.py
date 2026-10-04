"""
pillar2_diagnostics.py — Does the model actually know anything?
================================================================
Six tests, run in order of how damning they are. Run this BEFORE
spending more time on features or hyperparameters.

  1. PERMUTATION BASELINE
     Train on shuffled labels. Whatever F1 that produces is what "zero
     signal" looks like in your setup. If the real model scores inside
     the shuffled distribution, there is no signal — full stop.

  2. CONFIDENCE DECILES
     Precision and mean return by predicted-probability decile. A working
     model climbs monotonically. v9 fell from 51.9% at 0.50 to 37.5% at
     0.75, which is backwards.

  3. MONOTONICITY (Spearman)
     Puts a number on test 2. Positive = confidence tracks accuracy.
     Negative = the model is most wrong when most certain.

  4. CALIBRATION
     Predicted probability vs observed frequency. Quota sampling plus
     balanced class weights both distort this.

  5. FADE TEST
     If high confidence is anti-predictive, does inverting the signal
     beat random? Only meaningful if tests 2-3 show inversion.

  6. FEATURE ABLATION
     Drop each feature group, retrain, measure the damage. A group whose
     removal changes nothing was never contributing.

Run:
  python pillar2_diagnostics.py
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from class_xgboost import load_and_prep_data, DATASET_FILE

N_PERMUTATIONS = 5
N_SPLITS       = 3
SEED           = 67

PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
              subsample=0.8, colsample_bytree=0.8,
              random_state=SEED, eval_metric="logloss")


def fit_predict_folds(X, y, shuffle_y=False, rng=None, use_weights=True):
    """
    Walk-forward across folds, return pooled out-of-fold predictions.
    Pooling gives one large sample instead of three noisy small ones.
    """
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    probs  = np.full(len(y), np.nan)
    trues  = np.full(len(y), np.nan)

    for tr, va in tscv.split(X):
        y_tr = y[tr].copy()
        if shuffle_y:
            # Shuffle only the training labels — the validation labels stay
            # real, so we measure "can noise predict truth", which is the
            # right null hypothesis
            rng.shuffle(y_tr)

        w = compute_sample_weight("balanced", y_tr) if use_weights else None
        clf = xgb.XGBClassifier(**PARAMS)
        clf.fit(X.iloc[tr], y_tr, sample_weight=w, verbose=False)

        probs[va] = clf.predict_proba(X.iloc[va])[:, 1]
        trues[va] = y[va]

    mask = ~np.isnan(probs)
    return probs[mask], trues[mask].astype(int), mask


# ==========================================
# TEST 1 — PERMUTATION BASELINE
# ==========================================
def test_permutation(X, y):
    print("\n" + "=" * 68)
    print("TEST 1 — PERMUTATION BASELINE")
    print("=" * 68)
    print("Training on shuffled labels to measure what zero signal looks like.\n")

    probs, trues, _ = fit_predict_folds(X, y)
    real_auc = roc_auc_score(trues, probs)

    rng = np.random.default_rng(SEED)
    perm_aucs = []
    for i in range(N_PERMUTATIONS):
        p, t, _ = fit_predict_folds(X, y, shuffle_y=True, rng=rng)
        perm_aucs.append(roc_auc_score(t, p))
        print(f"  permutation {i+1:>2}/{N_PERMUTATIONS}: AUC {perm_aucs[-1]:.4f}")

    perm_aucs = np.array(perm_aucs)
    mu, sd    = perm_aucs.mean(), perm_aucs.std()
    z         = (real_auc - mu) / sd if sd > 0 else 0
    pct       = (perm_aucs < real_auc).mean()

    print(" " * 68)
    print(f"Real model AUC:        {real_auc:.4f}")
    print(f"Shuffled AUC:          {mu:.4f} ± {sd:.4f}")
    print(f"z-score:               {z:+.2f}")
    print(f"Percentile vs shuffle: {pct:.0%}")

    print()
    if real_auc < 0.5:
        print("🔴 AUC BELOW 0.50 — the model is worse than a coin flip.")
        print("   Predictions carry inverted information. See tests 2, 3, 5.")
    elif z < 2:
        print("🔴 NO SIGNAL — indistinguishable from training on random labels.")
        print("   More features or tuning will not help. The target is the problem.")
    elif z < 3:
        print("🟡 WEAK SIGNAL — above noise but thin.")
    else:
        print("🟢 SIGNAL PRESENT — clearly above the permutation floor.")

    return probs, trues, real_auc


# ==========================================
# TEST 2 + 3 — DECILES AND MONOTONICITY
# ==========================================
def test_deciles(probs, trues, returns):
    print("\n" + "=" * 68)
    print("TEST 2 — CONFIDENCE DECILES  (pooled out-of-fold)")
    print("=" * 68)
    print(f"{'Decile':>7} {'ProbRange':>16} {'n':>6} {'Precision':>10} {'MeanRet':>9}")
    print("-" * 68)

    edges = np.percentile(probs, np.arange(0, 101, 10))
    precisions, decile_ids = [], []

    for d in range(10):
        lo, hi = edges[d], edges[d + 1]
        sel = (probs >= lo) & (probs < hi) if d < 9 else (probs >= lo)
        n = int(sel.sum())
        if n < 5:
            continue
        prec = trues[sel].mean()
        ret  = returns[sel].mean() if returns is not None else float("nan")
        precisions.append(prec)
        decile_ids.append(d + 1)
        print(f"{d+1:>7} {lo:>7.3f}-{hi:<7.3f} {n:>6} {prec:>9.1%} {ret:>8.2f}%")

    # ── Test 3: monotonicity ────────────────────────────────────
    print("\n" + "=" * 68)
    print("TEST 3 — MONOTONICITY")
    print("=" * 68)

    if len(precisions) < 3:
        print("Not enough populated deciles.")
        return None

    # Spearman via ranks — no scipy dependency
    ranks_d = pd.Series(decile_ids).rank().values
    ranks_p = pd.Series(precisions).rank().values
    rho = np.corrcoef(ranks_d, ranks_p)[0, 1]

    base = trues.mean()
    top, bottom = precisions[-1], precisions[0]

    print(f"Base rate:              {base:.1%}")
    print(f"Bottom decile:          {bottom:.1%}  ({bottom-base:+.1f} pp vs base)")
    print(f"Top decile:             {top:.1%}  ({top-base:+.1f} pp vs base)")
    print(f"Spearman(decile, prec): {rho:+.3f}")

    print()
    if rho > 0.6:
        print("🟢 Confidence tracks accuracy. Thresholding is meaningful.")
    elif rho > 0.2:
        print("🟡 Weak ordering. Thresholds will behave unreliably.")
    elif rho > -0.2:
        print("🔴 No ordering. Confidence carries no information —")
        print("   any CONFIDENCE_THRESHOLD is arbitrary.")
    else:
        print("🔴 INVERTED. The model is least accurate when most confident.")
        print("   Run test 5 to see whether fading it is exploitable.")

    return rho


# ==========================================
# TEST 4 — CALIBRATION
# ==========================================
def test_calibration(probs, trues):
    print("\n" + "=" * 68)
    print("TEST 4 — CALIBRATION")
    print("=" * 68)
    print("Predicted probability vs observed frequency.")
    print("Systematic gaps mean quota sampling / class weights distorted the")
    print("probability scale, so thresholds do not mean what they appear to.\n")
    print(f"{'Bucket':>12} {'n':>6} {'Predicted':>11} {'Observed':>10} {'Gap':>8}")
    print("-" * 68)

    bins = np.arange(0.0, 1.01, 0.1)
    gaps = []
    for i in range(len(bins) - 1):
        sel = (probs >= bins[i]) & (probs < bins[i + 1])
        n = int(sel.sum())
        if n < 10:
            continue
        pred = probs[sel].mean()
        obs  = trues[sel].mean()
        gap  = obs - pred
        gaps.append(abs(gap) * n)
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}{'':>4} {n:>6} {pred:>10.1%} {obs:>9.1%} {gap:>+7.1%}")

    if gaps:
        ece = sum(gaps) / len(probs)
        print(f"\nExpected calibration error: {ece:.1%}")
        if ece > 0.10:
            print("🔴 Badly miscalibrated. Drop the quota and class weights,")
            print("   or wrap the model in CalibratedClassifierCV.")
        elif ece > 0.05:
            print("🟡 Moderately miscalibrated.")
        else:
            print("🟢 Reasonably calibrated.")


# ==========================================
# TEST 5 — FADE TEST
# ==========================================
def test_fade(probs, trues, returns):
    print("\n" + "=" * 68)
    print("TEST 5 — FADE TEST")
    print("=" * 68)
    print("If high confidence is anti-predictive, is the inverse tradeable?\n")

    if returns is None:
        print("No return data available.")
        return

    base_ret = returns.mean()
    print(f"{'Strategy':>28} {'n':>6} {'MeanRet':>10} {'vs All':>9}")
    print("-" * 68)
    print(f"{'Trade everything':>28} {len(returns):>6} {base_ret:>9.2f}% {0.0:>8.2f}")

    for label, sel in [
        ("Follow: top 30% conf",    probs >= np.percentile(probs, 70)),
        ("Follow: top 10% conf",    probs >= np.percentile(probs, 90)),
        ("Fade: bottom 30% conf",   probs <= np.percentile(probs, 30)),
        ("Fade: bottom 10% conf",   probs <= np.percentile(probs, 10)),
    ]:
        n = int(sel.sum())
        if n < 20:
            continue
        r = returns[sel].mean()
        print(f"{label:>28} {n:>6} {r:>9.2f}% {r-base_ret:>+8.2f}")

    print("\nA strategy only matters if it beats 'trade everything' by enough")
    print("to survive a t-test on a fresh sample. Small gaps here are noise.")


# ==========================================
# TEST 6 — FEATURE ABLATION
# ==========================================
FEATURE_GROUPS = {
    "MACD":        lambda c: c.startswith("macd"),
    "Volume":      lambda c: "volume" in c,
    "Candlestick": lambda c: c.startswith("bar_") or c.startswith("bar5"),
    "Fibonacci":   lambda c: "fib" in c,
    "Momentum":    lambda c: c in ("rsi_14", "rsi_zone", "bb_position", "bb_width_pct"),
    "Volatility":  lambda c: c == "atr_pct",
    "Trend":       lambda c: "ema200" in c or "vwap" in c or c == "5bar_close_slope",
    "S/R levels":  lambda c: "resistance" in c or "support" in c,
}


def test_ablation(X, y):
    print("\n" + "=" * 68)
    print("TEST 6 — FEATURE ABLATION")
    print("=" * 68)
    print("Retrain without each group. A group whose removal costs nothing")
    print("was contributing nothing.\n")

    probs, trues, _ = fit_predict_folds(X, y)
    full_auc = roc_auc_score(trues, probs)
    print(f"{'Group removed':>16} {'n feats':>8} {'AUC':>8} {'Delta':>9}")
    print("-" * 68)
    print(f"{'(none)':>16} {X.shape[1]:>8} {full_auc:>8.4f} {0.0:>9.4f}")

    for name, pred in FEATURE_GROUPS.items():
        drop = [c for c in X.columns if pred(c)]
        if not drop:
            continue
        Xr = X.drop(columns=drop)
        p, t, _ = fit_predict_folds(Xr, y)
        auc = roc_auc_score(t, p)
        print(f"{name:>16} {Xr.shape[1]:>8} {auc:>8.4f} {auc-full_auc:>+9.4f}")

    print("\nPositive delta = the model got BETTER without that group.")
    print("Those features are adding noise; drop them.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    import json, pandas as pd
    data = json.load(open(DATASET_FILE))
    df = pd.DataFrame([{
        "atr_pct": d["input"]["atr_14"] / d["input"]["current_price"],
        "outcome": d["output"]["trade_simulation"].get("outcome"),
    } for d in data])
    df["atr_quintile"] = pd.qcut(df.atr_pct, 5, labels=["lowest","low","mid","high","highest"])
    print(pd.crosstab(df.atr_quintile, df.outcome, normalize="index").round(3))
    print(f"📥 Loading {DATASET_FILE}")
    X, y, returns, raw = load_and_prep_data(DATASET_FILE)
    print(f"📊 {X.shape[0]} samples | {X.shape[1]} features | base rate {y.mean():.1%}")

    probs, trues, auc = test_permutation(X, y)

    # Align returns to the pooled out-of-fold rows
    _, _, mask = fit_predict_folds(X, y)
    ret_aligned = returns[mask] if returns is not None else None

    rho = test_deciles(probs, trues, ret_aligned)
    test_calibration(probs, trues)
    test_fade(probs, trues, ret_aligned)
    test_ablation(X, y)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"AUC {auc:.4f} | monotonicity {rho:+.3f}" if rho is not None else f"AUC {auc:.4f}")
    print()
    if auc < 0.52:
        print("The current feature set does not predict this target.")
        print("Next move is to change the TARGET, not the features:")
        print("  • benchmark-relative labels (beat SPY, not go up)")
        print("  • shorter horizon (20 bars, not 60)")
        print("  • cross-sectional ranking instead of per-stock binary")
    else:
        print("Some signal present. Worth pursuing calibration and thresholds.")