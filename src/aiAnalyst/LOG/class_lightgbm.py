import json
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

DATASET_FILE = "dataset_pillar2_mid_v55_class.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
RATIONALE_MODEL = "llama3.1"

# ==========================================
# 🛠️ FEATURE EXTRACTOR
# ==========================================
def extract_tabular_features(sample_input: dict) -> dict:
    price = float(sample_input.get("current_price", 1.0))
    ema200 = float(sample_input.get("ema_200", price))
    vwap = float(sample_input.get("vwap_20d", sample_input.get("vwap_20", price)))
    bb = sample_input.get("bollinger_bands", {})
    levels = sample_input.get("key_levels", {})
    candles = sample_input.get("candlestick_patterns", {})

    vol_str = str(sample_input.get("volume_vs_avg_20d", sample_input.get("volume_vs_avg_20", "100%")))
    vol_pct = float(vol_str.replace("%", "")) / 100.0

    rsi = float(sample_input.get("rsi_14", 50.0))
    
    # Categorical Mappings
    macd_map = {"Strong Bearish": 0, "Bearish Crossover (Pullback)": 1, "Bullish Crossover (Recovery)": 2, "Strong Bullish": 3}
    trend_map = {"Downtrend": 0, "Sideways / Consolidation": 1, "Uptrend": 2}
    vol_trend_map = {"Distribution (Higher volume on down periods)": 0, "Distribution (Higher volume on down days)": 0,
                     "Neutral (Balanced volume flow)": 1, 
                     "Accumulation (Higher volume on up periods)": 2, "Accumulation (Higher volume on up days)": 2}

    bb_upper = float(bb.get("upper", price))
    bb_lower = float(bb.get("lower", price))
    bb_mid = float(bb.get("mid_sma20", price))
    bb_range = (bb_upper - bb_lower) if bb_upper != bb_lower else 1e-9

    res_1 = float(levels.get("resistance_1", price))
    sup_1 = float(levels.get("support_1", price))

    feats = {
        # Moving Averages & Price Ratios
        "price_vs_ema200_pct": (price - ema200) / (ema200 + 1e-9),
        "price_vs_vwap_pct": (price - vwap) / (vwap + 1e-9),

        # Momentum & Volatility
        "rsi_14": rsi,
        "rsi_zone": 0 if rsi < 35 else (2 if rsi > 65 else 1),
        "bb_position": (price - bb_lower) / bb_range,
        "bb_width_pct": bb_range / (bb_mid + 1e-9),
        "atr_pct": float(sample_input.get("atr_14", 0.0)) / price,

        # Volume
        "volume_vs_avg20": vol_pct,
        "volume_profile_trend": vol_trend_map.get(sample_input.get("volume_profile_trend"), 1),

        # Support & Resistance Proximity
        "dist_to_resistance1_pct": (res_1 - price) / price,
        "dist_to_support1_pct": (price - sup_1) / price,

        # Encoded Categoricals
        "macd_status": macd_map.get(sample_input.get("macd_status"), 1),
        "graph_trend": trend_map.get(sample_input.get("graph_trend"), 1),

        # Candlestick Patterns
        "is_shooting_star": candles.get("is_shooting_star", 0),
        "is_hammer": candles.get("is_hammer", 0),
        "is_bearish_engulfing": candles.get("is_bearish_engulfing", 0),
        "is_bullish_engulfing": candles.get("is_bullish_engulfing", 0),
        "is_rejecting_resistance": candles.get("is_rejecting_resistance", 0),
        "5bar_close_slope": candles.get("5bar_close_slope", 0.0),
        "volume_trend_slope": candles.get("volume_trend_slope", 0.0),
    }
    return feats

# ==========================================
# 📊 DATASET BUILDING
# ==========================================
def load_and_prep_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_list, y_list = [], []
    label_map = {"Bearish": 0, "Avoid / No Trade": 0, "Neutral": 1, "Bullish": 2}

    for item in data:
        feats = extract_tabular_features(item["input"])
        raw_label = item["output"].get("technical_sentiment", "Bearish")
        
        y = label_map.get(raw_label, 0)
        X_list.append(feats)
        y_list.append(y)

    df_X = pd.DataFrame(X_list)
    array_y = np.array(y_list)
    return df_X, array_y, data

# ==========================================
# 🤖 CONDITIONED RATIONALE GENERATOR
# ==========================================
def generate_conditioned_rationale(sample_input: dict, sentiment: str, confidence: float) -> str:
    prompt = f"""You are a quantitative technical analyst explaining a model decision.
The LightGBM classification model has made its decision: {sentiment} (Confidence: {confidence:.0%}).
Do NOT question or change the decision. Write a clear, 2-sentence technical rationale explaining why the indicators support this conclusion.

FULL TECHNICAL DATA:
- Trend: {sample_input.get('graph_trend')}
- Price: ${sample_input.get('current_price')} (vs EMA200: ${sample_input.get('ema_200')})
- RSI-14: {sample_input.get('rsi_14')}
- MACD Status: {sample_input.get('macd_status')}
- Relative Volume: {sample_input.get('volume_vs_avg_20d', sample_input.get('volume_vs_avg_20'))}
- Rejection Flag: {sample_input.get('candlestick_patterns', {}).get('is_rejecting_resistance', 0)}

Output ONLY 2 sentences explaining the {sentiment} signal clearly.
"""
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": RATIONALE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }, timeout=30).json()
        return res["response"].strip()
    except Exception as e:
        return f"Price is aligned with {sentiment} trend parameters across key technical metrics."

# ==========================================
# 🎯 CLAUDE OUTPUT SPECIFICATION FUNCTION
# ==========================================
def pillar2_classifier_output(model, sample_input: dict, confidence_threshold=0.60) -> dict:
    """
    Implements Claude's exact output specification for Pillar 2
    """
    feats = extract_tabular_features(sample_input)
    df_single = pd.DataFrame([feats])
    
    proba = model.predict_proba(df_single)[0]
    
    # Handle classes if LightGBM dropped an empty class
    classes = model.classes_
    prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
    for idx, c in enumerate(classes):
        prob_dict[c] = float(proba[idx])

    predicted_class = int(classes[np.argmax(proba)])
    label_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    sentiment = label_map[predicted_class]
    confidence = round(prob_dict[predicted_class], 4)

    def get_signal_strength(conf):
        if conf >= 0.75: return "Strong"
        if conf >= 0.60: return "Moderate"
        return "Weak"

    output = {
        "technical_sentiment": sentiment,
        "confidence": confidence,
        "class_probabilities": {
            "Bullish": round(prob_dict[2], 4),
            "Neutral": round(prob_dict[1], 4),
            "Bearish": round(prob_dict[0], 4)
        },
        "signal_strength": get_signal_strength(confidence),
        "rationale": None
    }

    # Generate rationale only if confidence is above threshold
    if confidence >= confidence_threshold:
        output["rationale"] = generate_conditioned_rationale(sample_input, sentiment, confidence)

    return output

# ==========================================
# 🚀 MAIN TRAINING & EVALUATION PIPELINE
# ==========================================
if __name__ == "__main__":
    print(f"📥 Loading dataset for LightGBM training: {DATASET_FILE}...")
    X, y, raw_data = load_and_prep_data(DATASET_FILE)

    print(f"📊 Dataset Shape: Features={X.shape} | Bullish Class Count: {np.sum(y == 2)} | Bearish Class Count: {np.sum(y == 0)}")

    # TimeSeriesSplit Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)
    print("\n⚡ Running TimeSeriesSplit Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=20,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        print(f"\n--- Fold {fold+1} Classification Report ---")
        print(classification_report(y_val, preds, target_names=["Bearish", "Neutral", "Bullish"][:len(np.unique(y_val))], zero_division=0))

    # Final Model Fit on full dataset
    print("\n🏋️ Training Final LightGBM Production Model...")
    final_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=25,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)

    # Test Output Schema on a sample item
    sample_sample = raw_data[0]["input"]
    print("\n🔍 Testing Output Schema on Sample Input (AEHR):")
    final_output = pillar2_classifier_output(final_model, sample_sample)
    print(json.dumps(final_output, indent=4))
    
    import joblib

    # 💾 Save the model to your local directory
    model_filename = "lightgbm_mid_v1.joblib"
    joblib.dump(final_model, model_filename)

    print(f"\n✅ Model successfully saved to {model_filename}")