import os
import json
import joblib
import numpy as np
import pandas as pd

from class_news_fetcher import fetch_stock_profile
from class_xgboost import extract_tabular_features
from trade_config import compute_levels, HORIZON_CONFIGS, describe_geometry, GEOMETRY_MODE

# ==========================================
# CONFIGURATION
# ==========================================
ACTIVE_HORIZON       = "MID"
CONFIDENCE_THRESHOLD = 0.65   # set from class_xgboost.py threshold sweep
TURN_ON_PYTHON_FILTER = False   # True = reject setups whose R:R < min_rr

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ""))

CHOSEN_MODEL_PATH = os.path.join(PROJECT_ROOT, "class_model", "xgboost_mid_v40.joblib")

# ==========================================
# MODEL LOADING 
# ==========================================
print(f"Loading classifier: {CHOSEN_MODEL_PATH}")
_loaded = joblib.load(CHOSEN_MODEL_PATH)

if isinstance(_loaded, dict) and "model" in _loaded:
    PILLAR2_CLASSIFIER = _loaded["model"]
    FEATURE_NAMES      = _loaded.get("feature_names")
    print(f"   Bundle loaded — {len(FEATURE_NAMES) if FEATURE_NAMES else '?'} features")
    _bundle_geo = _loaded.get("geometry")
    if _bundle_geo and _bundle_geo != GEOMETRY_MODE:
        print(f"   GEOMETRY MISMATCH — model trained on '{_bundle_geo}', "
              f"trade_config is '{GEOMETRY_MODE}'. Retrain or switch mode.")
    print(f"   Geometry: {describe_geometry(ACTIVE_HORIZON)}")
else:
    PILLAR2_CLASSIFIER = _loaded
    FEATURE_NAMES      = getattr(_loaded, "feature_names_in_", None)
    if FEATURE_NAMES is not None:
        FEATURE_NAMES = list(FEATURE_NAMES)
    print("   Raw model loaded (no bundle)")


# ==========================================
# DATA FETCH
# ==========================================
def get_live_technical_snapshot(ticker, target_date_str=None):
    try:
        profile = fetch_stock_profile(ticker, target_date_str)
        if profile and "technical" in profile:
            return profile["technical"]
        return None
    except Exception as e:
        print(f"Error fetching technical snapshot: {e}")
        return None


# ==========================================
# INFERENCE
# ==========================================
def run_pillar2_technical_quant(ticker, target_date_str=None):
    tech = get_live_technical_snapshot(ticker, target_date_str)
    if not tech:
        return None

    # ── 1. Features ──────────────────────────────────────────────
    feats = extract_tabular_features(tech)
    df_single = pd.DataFrame([feats])

    # Enforce training column order. Without this a reordered or added
    # feature silently feeds the wrong values into the wrong columns.
    if FEATURE_NAMES is not None:
        missing = set(FEATURE_NAMES) - set(df_single.columns)
        if missing:
            print(f"  Missing features for {ticker}: {sorted(missing)[:5]}")
            return None
        df_single = df_single[FEATURE_NAMES]

    # ── 2. Predict ───────────────────────────────────────────────
    proba   = PILLAR2_CLASSIFIER.predict_proba(df_single)[0]
    classes = PILLAR2_CLASSIFIER.classes_

    prob_dict = {int(c): float(p) for c, p in zip(classes, proba)}
    bull_prob = prob_dict.get(1, 0.0)
    bear_prob = prob_dict.get(0, 0.0)

    predicted  = int(classes[int(np.argmax(proba))])
    label_map  = {0: "Bearish", 1: "Bullish"}
    sentiment  = label_map.get(predicted, "Bearish")
    confidence = round(bull_prob, 4)

    # ── 3. Confidence gate ───────────────────────────────────────
    # Threshold applies to the BULLISH probability specifically, not to
    # whichever class happened to win argmax.
    # ── 3. Confidence gate ───────────────────────────────────────
    should_trade = bull_prob >= CONFIDENCE_THRESHOLD

    if not should_trade:
        setup_type = (f"Avoid (Low Confidence: {bull_prob:.2f})"
                    if sentiment == "Bullish" else "Avoid / No Trade")
        sentiment  = "Neutral" if sentiment == "Bullish" else sentiment
    else:
        setup_type = f"Classifier Breakout (Conf: {bull_prob:.2f})"

    out = {
        "technical_sentiment": sentiment,
        "confidence":          confidence,
        "bullish_probability": round(bull_prob, 4),
        "class_probabilities": {
            "Bullish": round(bull_prob, 4),
            "Bearish": round(bear_prob, 4),
        },
        "trading_setup_type": setup_type,
        "should_trade":       should_trade,
    }

    # ── 4. Levels from the SHARED formula ────────────────────────
    if not should_trade:
        out["actionable_levels"] = {"entry_price": 0.0, "target_price": 0.0, "stop_loss": 0.0}
        return out

    entry = float(tech.get("current_price", 0))
    atr   = float(tech.get("atr_14", 0))
    lv    = compute_levels(entry, atr, ACTIVE_HORIZON)

    if not lv.get("valid") and TURN_ON_PYTHON_FILTER:
        out["technical_sentiment"] = "Bearish"
        out["trading_setup_type"]  = f"Avoid (Python Override: {lv.get('reason')})"
        out["should_trade"]        = False
        out["actionable_levels"]   = {"entry_price": 0.0, "target_price": 0.0, "stop_loss": 0.0}
        return out

    if "entry" not in lv:
        out["should_trade"] = False
        out["actionable_levels"] = {"entry_price": 0.0, "target_price": 0.0, "stop_loss": 0.0}
        return out

    out["actionable_levels"] = {
        "entry_price":  lv["entry"],
        "target_price": lv["target"],
        "stop_loss":    lv["stop"],
    }
    out["risk_reward_ratio"] = f"1:{lv['rr']}"
    out["level_detail"]      = {"stop_pct": lv["stop_pct"], "target_pct": lv["target_pct"]}

    return out


if __name__ == "__main__":
    for t in ["META", "NVDA"]:
        print(f"\n  Pillar 2 — {t}")
        r = run_pillar2_technical_quant(t)
        print(json.dumps(r, indent=4) if r else "  No data")