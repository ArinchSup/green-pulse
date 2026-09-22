import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

from trade_config import describe_geometry, GEOMETRY_MODE, FIXED_TP_PCT, FIXED_SL_PCT

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_FILE   = "dataset_pillar2_mid_v39.json"
MODEL_FILENAME = "xgboost_mid_v39.joblib"
BEST_N         = 300

# ── Class weighting ──────────────────────────────────────────────
# False (recommended when the dataset uses USE_NATURAL_RATE = True):
#   leaves probabilities on the real scale, so a 0.70 output means the
#   model genuinely saw ~70% of similar setups succeed.
# True: rebalances toward the minority class. Combined with quota
#   sampling this stacks two distortions on top of each other, which is
#   how v9 ended up predicting 85% on setups that succeeded 53% of the time.
USE_CLASS_WEIGHTS = False


# ==========================================
# FEATURE EXTRACTOR
# ==========================================
def extract_tabular_features(sample_input: dict) -> dict:
    price   = float(sample_input.get("current_price", 1.0))
    ema200  = float(sample_input.get("ema_200", price))
    ema20   = float(sample_input.get("ema_20", price))
    vwap    = float(sample_input.get("vwap_20d", sample_input.get("vwap_20", price)))
    bb      = sample_input.get("bollinger_bands", {})
    levels  = sample_input.get("key_levels", {})
    candles = sample_input.get("candlestick_patterns", {})
    fib     = sample_input.get("fibonacci", {})

    vol_str = str(sample_input.get("volume_vs_avg_20d",
                                   sample_input.get("volume_vs_avg_20", "100%")))
    vol_pct = float(vol_str.replace("%", "")) / 100.0
    rsi     = float(sample_input.get("rsi_14", 50.0))

    vol_trend_map = {
        "Distribution (Higher volume on down periods)": 0,
        "Distribution (Higher volume on down days)":   0,
        "Neutral (Balanced volume flow)":              1,
        "Accumulation (Higher volume on up periods)":  2,
        "Accumulation (Higher volume on up days)":     2,
    }

    bb_upper = float(bb.get("upper", price))
    bb_lower = float(bb.get("lower", price))
    bb_mid   = float(bb.get("mid_sma20", price))
    bb_range = (bb_upper - bb_lower) if bb_upper != bb_lower else 1e-9

    res_1 = float(levels.get("resistance_1", price))
    sup_1 = float(levels.get("support_1", price))

    macd_map = {
        "Strong Bearish":               0,
        "Bearish Crossover (Pullback)": 1,
        "Bullish Crossover (Recovery)": 2,
        "Strong Bullish":               3,
    }
    raw_macd     = sample_input.get("macd_status", "")
    macd_ordinal = macd_map.get(raw_macd, 1)

    feats = {
        "price_vs_ema20_pct":      (price - ema20) / (ema20 + 1e-9),
        "price_vs_ema200_pct":     (price - ema200) / (ema200 + 1e-9),
        "price_vs_vwap_pct":       (price - vwap)   / (vwap   + 1e-9),

        "macd_status":             macd_ordinal,
        "macd_is_bullish":         int("Bullish"   in raw_macd),
        "macd_is_strong":          int("Strong"    in raw_macd),
        "macd_is_crossover":       int("Crossover" in raw_macd),

        "rsi_14":                  rsi,
        "rsi_zone":                0 if rsi < 35 else (2 if rsi > 65 else 1),
        "bb_position":             (price - bb_lower) / bb_range,
        "bb_width_pct":            bb_range / (bb_mid + 1e-9),

        "atr_pct":                 float(sample_input.get("atr_14", 0.0)) / price,

        "volume_vs_avg20":         vol_pct,
        "volume_profile_trend":    vol_trend_map.get(sample_input.get("volume_profile_trend"), 1),
        "volume_trend_slope":      float(candles.get("volume_trend_slope", 0.0)),

        "dist_to_resistance1_pct": (res_1 - price) / price,
        "dist_to_support1_pct":    (price - sup_1)  / price,

        "dist_to_fib_382_pct":     (price - float(fib.get("fib_0.382", price))) / price,
        "dist_to_fib_618_pct":     (price - float(fib.get("fib_0.618", price))) / price,
        "dist_to_fib_786_pct":     (price - float(fib.get("fib_0.786", price))) / price,

        "is_rejecting_resistance": int(candles.get("is_rejecting_resistance", 0)),
        "5bar_close_slope":        float(candles.get("5bar_close_slope", 0.0)),

        "bar5_close_vs_range":     0.5,
        "bar5_vs_resistance":      0.0,
    }

    ohlcv    = sample_input.get("last_5_bars_ohlcv", [])
    last_bar = ohlcv[-1] if ohlcv else {}
    if last_bar and "close" in last_bar:
        h = last_bar.get("high",  price)
        l = last_bar.get("low",   price)
        c = last_bar.get("close", price)
        rng = (h - l) if h != l else 1e-9
        feats["bar5_close_vs_range"] = (c - l) / rng
        feats["bar5_vs_resistance"]  = (res_1 - c) / price

    for i in range(1, 6):
        tag = f"bar_{i}"
        feats[f"{tag}_body_ratio"]       = float(candles.get(f"{tag}_body_ratio",       0.5))
        feats[f"{tag}_upper_wick_ratio"] = float(candles.get(f"{tag}_upper_wick_ratio", 0.25))
        feats[f"{tag}_lower_wick_ratio"] = float(candles.get(f"{tag}_lower_wick_ratio", 0.25))
        feats[f"{tag}_is_bearish"]       = int(candles.get(f"{tag}_is_bearish",         0))

    return feats


# ==========================================
# DATASET LOADING
# ==========================================
def load_and_prep_data(filepath: str):
    """
    Returns X, y, returns, raw.

    `returns` is the realised per-trade return in percent, taken from
    trade_simulation.exit_return_pct. Keeping it lets the threshold sweep
    score candidate thresholds on money rather than on hit count.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {"Bearish": 0, "Avoid / No Trade": 0, "Neutral": 0, "Bullish": 1}
    X_list, y_list, ret_list = [], [], []
    geometries = set()

    for item in data:
        X_list.append(extract_tabular_features(item["input"]))

        sim     = item.get("output", {}).get("trade_simulation", {})
        outcome = sim.get("outcome")
        geometries.add(sim.get("geometry", "unknown"))

        if outcome:
            y_list.append(1 if outcome == "target_hit" else 0)
        else:
            raw = item.get("output", {}).get("technical_sentiment", "Bearish")
            y_list.append(label_map.get(raw, 0))

        # Fall back to the nominal target/stop if exit_return_pct is absent
        if "exit_return_pct" in sim:
            ret_list.append(float(sim["exit_return_pct"]))
        elif outcome == "target_hit":
            ret_list.append(float(sim.get("target_pct", FIXED_TP_PCT * 100)))
        else:
            ret_list.append(-float(sim.get("stop_pct", FIXED_SL_PCT * 100)))

    if geometries and geometries != {"unknown"}:
        print(f"Dataset geometry: {geometries}")
        # BENCHMARK_RELATIVE datasets have no stop/target geometry at all,
        # so trade_config's GEOMETRY_MODE simply does not apply to them
        if "BENCHMARK_RELATIVE" in geometries:
            print(f"   Benchmark-relative labels — GEOMETRY_MODE not applicable.")
        elif GEOMETRY_MODE not in geometries:
            print(f">>  MISMATCH — trade_config is '{GEOMETRY_MODE}' but the dataset "
                  f"was built with {geometries}. Regenerate the dataset.")

    return pd.DataFrame(X_list), np.array(y_list), np.array(ret_list), data


# ==========================================
# LABEL LEAKAGE DIAGNOSTIC
# ==========================================
def run_leakage_check(X: pd.DataFrame, y: np.ndarray):
    print("\n" + "=" * 54)
    print(">> LABEL LEAKAGE DIAGNOSTIC")
    print("=" * 54)

    df = X.copy()
    df["label"] = y
    bull = df[df["label"] == 1]

    if bull.empty:
        print(">>  No Bullish samples — check label loading.")
        return

    strong = (bull["macd_status"] == 3).mean()
    highv  = (bull["volume_vs_avg20"] > 1.5).mean()
    both   = ((bull["macd_status"] == 3) & (bull["volume_vs_avg20"] > 1.5)).mean()

    print(f"Bullish w/ Strong MACD:   {strong:.1%}")
    print(f"Bullish w/ Volume > 150%: {highv:.1%}")
    print(f"Bullish w/ BOTH:          {both:.1%}   (target < 60%)")

    mask = (df["macd_status"] == 3) & (df["volume_vs_avg20"] > 1.5)
    if mask.any():
        print(f"Strong MACD + High Vol → {(df[mask]['label'] == 1).mean():.1%} Bullish")

    # Under ATR geometry, atr_pct correlating strongly with the label is a
    # difficulty proxy, not a signal. Under FIXED it is legitimate.
    corr = np.corrcoef(X["atr_pct"], y)[0, 1]
    print(f"\ncorr(atr_pct, label):     {corr:+.3f}   [geometry: {GEOMETRY_MODE}]")
    if GEOMETRY_MODE == "ATR" and abs(corr) > 0.15:
        print("    Under ATR geometry this is target-distance leakage.")

    print(f"\n{'  HIGH LEAKAGE' if both > 0.60 else '✅ ACCEPTABLE'}\n")


# ==========================================
# THRESHOLD SWEEP — measure, don't guess
# ==========================================
def threshold_sweep(model, X_val, y_val, ret_val):
    """
    Reports precision AND expected value per trade at each threshold.

    Precision alone picks thresholds that fire four times at 90%. EV x
    trade count is what actually compounds, so both are printed and the
    recommendation is based on total return.
    """
    proba = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 66)
    print(">> THRESHOLD SWEEP (validation fold)")
    print("=" * 66)
    print(f"{'Thresh':>7} {'Trades':>7} {'Precision':>10} {'MeanRet':>9} {'TotalRet':>10}")
    print("-" * 66)

    best = None
    for t in np.arange(0.35, 0.85, 0.05):
        sel = proba >= t
        n   = int(sel.sum())
        if n < 15:
            continue
        prec      = float(y_val[sel].mean())
        mean_ret  = float(ret_val[sel].mean())
        total_ret = mean_ret * n

        print(f"{t:>7.2f} {n:>7d} {prec:>9.1%} {mean_ret:>8.2f}% {total_ret:>9.1f}%")

        if best is None or total_ret > best[3]:
            best = (t, n, prec, total_ret, mean_ret)

    if best:
        t, n, prec, total, mean_ret = best
        print("-" * 66)
        print(f"Best by total return: threshold {t:.2f} "
              f"({n} trades, {prec:.1%} precision, {mean_ret:+.2f}% per trade)")
        print(f"   Set CONFIDENCE_THRESHOLD = {t:.2f} in class_ai_pillar2.py")
    return best


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f">> Loading: {DATASET_FILE}")
    print(f">> Geometry: {describe_geometry('MID')}\n")

    X, y, returns, raw = load_and_prep_data(DATASET_FILE)

    print(f">> {X.shape[0]} samples | {X.shape[1]} features")
    print(f"   Target hit: {int(np.sum(y == 1))}  |  Failed: {int(np.sum(y == 0))}")
    print(f"   Base rate:  {y.mean():.1%}")
    print(f"   Mean return if trading everything: {returns.mean():+.2f}%")

    run_leakage_check(X, y)

    # ── Cross-validation ────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=3)
    print("⚡ TimeSeriesSplit Cross-Validation\n")

    last_clf = last_val = None
    f1s = []

    for fold, (tr, va) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        w = compute_sample_weight("balanced", y_tr) if USE_CLASS_WEIGHTS else None

        clf = xgb.XGBClassifier(
            n_estimators=BEST_N, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=67, eval_metric="logloss",
        )
        clf.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_va, y_va)], verbose=False)
        preds = clf.predict(X_va)

        uniq  = np.unique(y_va)
        names = ["Fail", "Success"]
        rep   = classification_report(y_va, preds, labels=uniq,
                                      target_names=[names[i] for i in uniq],
                                      zero_division=0, output_dict=True)
        f1 = rep.get("Success", {}).get("f1-score", 0.0)
        f1s.append(f1)

        print(f"--- Fold {fold+1} | base rate {y_va.mean():.1%} | Success F1 {f1:.3f} ---")
        print(classification_report(y_va, preds, labels=uniq,
                                    target_names=[names[i] for i in uniq],
                                    zero_division=0))

        last_clf, last_val = clf, (X_va, y_va, returns[va])

    print(f">> Mean Success F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # ── Threshold sweep on the most recent fold ─────────────────
    if last_clf is not None:
        threshold_sweep(last_clf, *last_val)

    # ── Final model ─────────────────────────────────────────────
    print("\n>>  Training final model on full dataset...")
    final_model = xgb.XGBClassifier(
        n_estimators=BEST_N, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=67, eval_metric="logloss",
    )
    final_model.fit(
        X, y,
        sample_weight=compute_sample_weight("balanced", y) if USE_CLASS_WEIGHTS else None,
    )

    # ── SHAP ────────────────────────────────────────────────────
    print("\nSHAP plots...")
    sv = shap.TreeExplainer(final_model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.title(f"SHAP magnitude — Success class ({GEOMETRY_MODE} geometry)")
    plt.tight_layout(); plt.savefig("shap_bar_v9.png", dpi=150, bbox_inches="tight"); plt.show()

    # Beeswarm shows DIRECTION: red = high feature value,
    # right of centre = pushes toward Success
    shap.summary_plot(sv, X, show=False)
    plt.title("SHAP direction — red = high value, right = more likely Success")
    plt.tight_layout(); plt.savefig("shap_beeswarm_v9.png", dpi=150, bbox_inches="tight"); plt.show()

    # ── Save BUNDLE, not bare model ─────────────────────────────
    # feature_names lets class_ai_pillar2.py enforce column order, which
    # catches a reordered or added feature instead of silently feeding
    # the wrong values into the wrong columns.
    bundle = {
        "model":         final_model,
        "feature_names": X.columns.tolist(),
        "label_map":     {0: "Bearish", 1: "Bullish"},
        "geometry":      GEOMETRY_MODE,
        "fixed_tp_pct":  FIXED_TP_PCT,
        "fixed_sl_pct":  FIXED_SL_PCT,
        "dataset":       DATASET_FILE,
        "n_samples":     int(X.shape[0]),
        "n_features":    int(X.shape[1]),
        "base_rate":     float(y.mean()),
        "class_weights": USE_CLASS_WEIGHTS,
    }
    joblib.dump(bundle, MODEL_FILENAME)

    print(f"\nSaved bundle: {MODEL_FILENAME}")
    print(f"   {X.shape[1]} features | geometry {GEOMETRY_MODE}")
    print(f"\nNext: set CHOSEN_MODEL_PATH to '{MODEL_FILENAME}' in class_ai_pillar2.py")
    print(f"      and CONFIDENCE_THRESHOLD to the sweep's recommendation.")