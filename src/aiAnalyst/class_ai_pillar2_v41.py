"""
class_ai_pillar2_v41.py - Pillar 2 V41 inference, same interface as class_ai_pillar2

    from class_ai_pillar2_v41 import load_model, run_pillar2_technical_quant
    load_model("pillar2_v41_cut20250627_model.json")   # optional: else $PILLAR2_V41_MODEL
    run_pillar2_technical_quant("NVDA", target_date_str="2026-01-15")

Returns {"technical_sentiment", "confidence", "actionable_levels", "model_version", "signal_date"}.
  confidence = the model's probability that the stock beats the benchmark (QQQ) over the next
               60 trading days. It is a ranking score; treat it as "stronger vs weaker setup",
               not as a promise of a percentage.
  sentiment  = Bullish / Neutral / Bearish from the cutoffs saved at training time.

Only bars up to and including target_date_str are used, and the features come from
pillar2_v41_features.py - the exact code the model was trained with - so training and serving
can't drift apart. load_model() refuses a model whose feature list doesn't match.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

import class_pillar2_data_gen_v41 as gen
from pillar2_v41_features import FEATURE_NAMES, FEATURE_VERSION, relative_strength, to_vector

DEFAULT_MODEL = os.environ.get("PILLAR2_V41_MODEL", "pillar2_v41_model.json")
LOOKBACK_DAYS = 1100          # ~750 trading days, enough for the 200-day EMA to settle
TARGET_PCT, STOP_PCT = 0.20, 0.12
_state = {}


def _meta_path(model_path):
    if model_path.endswith("_model.json"):
        return model_path[: -len("_model.json")] + "_meta.json"
    return model_path + ".meta.json"


def load_model(path=DEFAULT_MODEL):
    """Load a V41 model and its metadata; refuse it if its features don't match this code."""
    booster = xgb.Booster()
    booster.load_model(path)
    with open(_meta_path(path), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError(f"{path} was trained with a different feature list than "
                           f"pillar2_v41_features.py ({FEATURE_VERSION}) builds. Retrain, or use the "
                           f"features file that matches the model.")
    _state.clear()
    _state.update(booster=booster, meta=meta, path=path)
    return meta


def _daily(ticker, start, end):
    df = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)   # same convention as the generator
    return df


def run_pillar2_technical_quant(ticker, target_date_str=None):
    if "booster" not in _state:
        load_model()
    meta = _state["meta"]
    target = pd.Timestamp(target_date_str) if target_date_str else pd.Timestamp.today().normalize()
    start, end = target - pd.Timedelta(days=LOOKBACK_DAYS), target + pd.Timedelta(days=1)

    df_past = _daily(ticker, start, end)                 # bars up to and including the target date
    bench = _daily(meta.get("benchmark", "QQQ"), start, end)
    min_bars = gen.HORIZON_CONFIGS["MID"]["min_past_bars"]
    if df_past is None or bench is None or len(df_past) < min_bars:
        return None

    snapshot = gen.build_snapshot(df_past, "1d")
    snapshot["relative_strength"] = relative_strength(df_past["Close"],
                                                      bench["Close"][bench.index <= df_past.index[-1]])
    x = xgb.DMatrix(np.array([to_vector(snapshot)], dtype=float), feature_names=FEATURE_NAMES)
    score = float(_state["booster"].predict(x)[0])

    if score >= meta["bullish_cutoff"]:
        sentiment = "Bullish"
    elif score >= meta["neutral_cutoff"]:
        sentiment = "Neutral"
    else:
        sentiment = "Bearish"
    price = float(snapshot["current_price"])
    return {
        "technical_sentiment": sentiment,
        "confidence": round(score, 4),
        "actionable_levels": {"entry": price,
                              "target_price": round(price * (1 + TARGET_PCT), 2),
                              "stop_loss": round(price * (1 - STOP_PCT), 2)},
        "model_version": meta.get("version"),
        "signal_date": str(df_past.index[-1].date()),
    }
