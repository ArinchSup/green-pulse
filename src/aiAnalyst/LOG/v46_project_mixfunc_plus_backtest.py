# ==========================================
# Don't forget to crate model in Ollama by using: ollama create technical-mid-v46-project -f Modelfile.technicalmidv46project
# ==========================================

import json
import requests
import datetime
import random
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# CONFIG
# ==========================================
OLLAMA_URL = "http://localhost:11434/api/generate"

ACTIVE_HORIZON = "MID" 
MODEL_NAME = "technical-mid-v46-project"
CONTEXT = "Mid-Term (1-3 months, using a Daily 1D Chart)"  

TARGET_AMOUNT = 500
OUTPUT_FILE = f"backtest_results_for_only_ai_project.json"
BACKTEST_CONFIGS = {
    "MID": {
        "eval_days": 90,          
        "interval": "1d",          
        "max_lookback": 400      
    }
}

SHAY_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CRWV", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI"
]

# ==========================================
# Fetch Stock Profile
# ==========================================
def calculate_atr(df, period=14):
    if len(df) < period: return 0.0
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return float(round(atr.iloc[-1], 2))

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    if down == 0: return 100
    rs = up / down
    return float(round(100. - 100. / (1. + rs), 2))

def fetch_stock_profile(ticker, target_date_str=None):
    try:
        if target_date_str:
            target_dt = pd.to_datetime(target_date_str)
            start_date = target_dt - datetime.timedelta(days=365)
            end_date = target_dt + datetime.timedelta(days=1)
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
            
            if hist.index.tz is not None:
                target_dt_tz = target_dt.tz_localize(hist.index.tz)
                hist = hist.loc[hist.index <= target_dt_tz]
            else:
                hist = hist.loc[hist.index <= target_dt]
        else:
            hist = yf.Ticker(ticker).history(period="1y")
            target_dt = datetime.datetime.now()

        if hist.empty: raise ValueError("No historical data")

        current_close = float(round(hist['Close'].iloc[-1], 2))
        
        # 1. Last 5 Days OHLCV
        last_5_days_df = hist.tail(5)
        last_5_days_ohlcv = []
        for date, row in last_5_days_df.iterrows():
            last_5_days_ohlcv.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        # 2. Volume Metrics
        avg_vol_20d = hist['Volume'].tail(20).mean()
        current_vol = hist['Volume'].iloc[-1]
        vol_pct = round((current_vol / avg_vol_20d) * 100) if avg_vol_20d > 0 else 100
        
        # Volume Profile Trend
        recent_10 = hist.tail(10)
        up_vols = recent_10[recent_10['Close'] > recent_10['Open']]['Volume']
        down_vols = recent_10[recent_10['Close'] < recent_10['Open']]['Volume']
        avg_up_vol = up_vols.mean() if not up_vols.empty else 0
        avg_down_vol = down_vols.mean() if not down_vols.empty else 0
        
        if avg_down_vol > avg_up_vol * 1.15:
            vol_trend = "Distribution (Higher volume on down days)"
        elif avg_up_vol > avg_down_vol * 1.15:
            vol_trend = "Accumulation (Higher volume on up days)"
        else:
            vol_trend = "Neutral (Balanced volume flow)"

        # 3. VWAP (20-day rolling)
        typical_p = (hist['High'] + hist['Low'] + hist['Close']) / 3
        vwap_20 = (typical_p * hist['Volume']).rolling(20).sum() / hist['Volume'].rolling(20).sum()
        vwap_val = float(round(vwap_20.iloc[-1], 2)) if not pd.isna(vwap_20.iloc[-1]) else current_close

        # 4. Bollinger Bands
        sma_20 = hist['Close'].rolling(20).mean()
        std_20 = hist['Close'].rolling(20).std()
        bb_upper = float(round((sma_20 + (std_20 * 2)).iloc[-1], 2)) if not pd.isna(sma_20.iloc[-1]) else current_close
        bb_lower = float(round((sma_20 - (std_20 * 2)).iloc[-1], 2)) if not pd.isna(sma_20.iloc[-1]) else current_close
        bb_mid = float(round(sma_20.iloc[-1], 2)) if not pd.isna(sma_20.iloc[-1]) else current_close
        
        # 5. Support/Resistance & Fibonacci
        swing_low = float(round(hist['Low'].tail(30).min(), 2))
        swing_high = float(round(hist['High'].tail(30).max(), 2))
        support_1 = float(round(hist['Low'].tail(10).min(), 2))
        resistance_1 = float(round(hist['High'].tail(10).max(), 2))
        
        diff = swing_high - swing_low
        fib_0786 = float(round(swing_high - (diff * 0.786), 2))
        fib_0618 = float(round(swing_high - (diff * 0.618), 2))
        fib_0382 = float(round(swing_high - (diff * 0.382), 2))
        
        # 6. Technical Indicators
        ema_200 = float(round(hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1], 2))
        atr_14 = float(calculate_atr(hist))
        rsi_val = float(calculate_rsi(hist['Close'].values))
        
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(round(macd.iloc[-1], 2))
        signal_val = float(round(signal.iloc[-1], 2))
        
        if macd_val > signal_val and macd_val > 0: macd_status = "Strong Bullish"
        elif macd_val > signal_val and macd_val <= 0: macd_status = "Bullish Crossover (Recovery)"
        elif macd_val < signal_val and macd_val > 0: macd_status = "Bearish Crossover (Pullback)"
        else: macd_status = "Strong Bearish"
        
        if current_close > ema_200 and rsi_val > 50: trend = "Uptrend"
        elif current_close < ema_200 and rsi_val < 50: trend = "Downtrend"
        else: trend = "Sideways / Consolidation"
            
        # 8. Assemble matching training format
        technical_data = {
            "timeframe": "Daily (1D)",
            "current_price": current_close,
            "last_5_days_ohlcv": last_5_days_ohlcv,
            "graph_trend": trend,
            "ema_200": ema_200,
            "vwap_20d": vwap_val,
            "bollinger_bands": {
                "upper": bb_upper,
                "mid_sma20": bb_mid,
                "lower": bb_lower
            },
            "rsi_14": round(rsi_val, 2),
            "macd_status": macd_status,
            "volume_vs_avg_20d": f"{vol_pct}%",
            "volume_profile_trend": vol_trend,
            "atr_14": atr_14,
            "key_levels": {
                "support_1": support_1,
                "support_2_swing_low": swing_low,
                "resistance_1": resistance_1,
                "resistance_2_swing_high": swing_high
            },
            "fibonacci": {
                "fib_0.382": fib_0382,
                "fib_0.618": fib_0618,
                "fib_0.786": fib_0786
            }
        }

        return {
            "technical": technical_data
        }
    except Exception as e:
        print(f"Error fetching profile for {ticker}: {e}")
        return {"technical": {}}

# ==========================================
# Data Preparation
# ==========================================
def get_live_technical_snapshot(ticker, target_date_str=None):
    try:
        profile = fetch_stock_profile(ticker, target_date_str)
        if profile and 'technical' in profile:
            return profile['technical']
        return None
    except Exception as e:
        print(f"Error fetching technical snapshot: {e}")
        return None

# ==========================================
# AI Inference
# ==========================================
def run_pillar2_technical_quant(ticker, target_date_str=None):
    live_tech_data = get_live_technical_snapshot(ticker, target_date_str)
    
    if not live_tech_data:
        return None
        
    profile = {
        "model": MODEL_NAME,
        "context": CONTEXT
    }
    
    prompt = f"""You are an elite Chartered Market Technician (CMT) and Quant Strategist for a top-tier hedge fund.
Below is an instruction that describes a task, paired with an input that provides technical data. Write a highly analytical response that appropriately completes the request.
CRITICAL RULE: You MUST output ONLY a valid JSON object. Do not include markdown blocks (like ```json), greetings, or comments.

    ### Instruction:
    Analyze the following technical snapshot for {ticker}. Provide a {profile['context']} trading strategy. 

Your default stance is to PROTECT CAPITAL. You are highly skeptical of the market. If the volume trend is distribution, if the price is under the 200 EMA, or if the MACD is bearish, you MUST output 'Avoid / No Trade'. You only approve a 'Bullish' setup if all indicators align perfectly.

Provide your response in EXACTLY this JSON structure:
{{
  "rationale": "<Write your technical explanation here first. Evaluate trend, volume, and momentum. Conclude if it is safe to trade or if we must avoid.>",
  "technical_sentiment": "<Bullish or Bearish>",
  "trading_setup_type": "<e.g., Momentum Breakout, Support Bounce, Avoid / No Trade>",
  "risk_reward_ratio": "<e.g., 1:3.0 or 0:0>",
  "actionable_levels": {{
    "entry_price": <number>,
    "target_price": <number>,
    "stop_loss": <number>
  }}
}}

    ### Input:
    {json.dumps(live_tech_data, indent=2)}

    ### Response:
    """
    
    try:
        payload = {
            "model": profile["model"], 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        }
    
        res = requests.post(OLLAMA_URL, json=payload, timeout=60).json()
        return json.loads(res["response"])
        
    except requests.exceptions.Timeout:
        print(f"  ⚠️ Timeout Error: Ollama took longer than 60 seconds. Skipping...")
        return None
    except Exception as e:
        print(f"  ⚠️ Other Error: {e}")
        return None

def dump_ai_output_to_file(ai_output, winloss):
    final_result = {winloss: ai_output}
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
        f.write(",\n")

def run_backtest():
    config = BACKTEST_CONFIGS[ACTIVE_HORIZON]
    eval_days = config["eval_days"]
    interval = config["interval"]
    max_lookback = config["max_lookback"]
    
    print(f"► Starting Backtest for {TARGET_AMOUNT} iterations...")
    print(f"   ► Testing Horizon: {ACTIVE_HORIZON} | Window: {eval_days} days | Interval: {interval}\n")
    
    all_tickers = SHAY_TICKERS 
    
    total_signals = 0
    wins = 0
    losses = 0
    misses = 0
    skips = 0
    open_trades = 0

    for i in range(TARGET_AMOUNT):
        ticker = random.choice(all_tickers)
        days_ago = random.randint(eval_days, max_lookback)
        target_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        print(f"[{i+1}/{TARGET_AMOUNT}] Evaluating {ticker} on {target_date_str}...")
        
        try:
            ai_output = run_pillar2_technical_quant(ticker, target_date_str)
        except Exception as e:
            print(f"  ⚠️ Error running AI for {ticker}: {e}")
            continue
        
        if not ai_output:
            print(f"  ⏭️ Skipped: No valid technical data or AI output for {ticker}.")
            skips += 1
            continue

        setup_type = ai_output.get("trading_setup_type", "")
        sentiment = ai_output.get("technical_sentiment", "")
        
        skip_keywords = ["Avoid", "Wait", "No Trade", "Neutral"]
        if sentiment == "Bearish" or any(kw in setup_type for kw in skip_keywords):
            print(f"  ⏭️ Skipped: AI suggested {setup_type} ({sentiment}).")
            dump_ai_output_to_file(ai_output, "SKIPPED")
            skips += 1
            continue

        try:
            levels = ai_output.get("actionable_levels", {})
            
            entry_p = levels.get("entry_price")
            target_p = levels.get("target_price")
            stop_p = levels.get("stop_loss")
            
            entry_p = float(entry_p) if entry_p not in ["", None] else 0.0
            target_p = float(target_p) if target_p not in ["", None] else 0.0
            stop_p = float(stop_p) if stop_p not in ["", None] else 0.0
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"  ⏭️ Skipped: Actionable levels format error for {ticker}: {e}")
            dump_ai_output_to_file(ai_output, "SKIPPED")
            skips += 1
            continue

        if entry_p <= 0 or target_p <= 0 or stop_p <= 0:
            print(f"  ⏭️ Skipped: Invalid price level data (Prices must be greater than 0)")
            dump_ai_output_to_file(ai_output, "SKIPPED")
            skips += 1
            continue

        # === Normal loop ===
        total_signals += 1
        
        end_date = target_date + datetime.timedelta(days=eval_days)
        future_df = yf.Ticker(ticker).history(start=target_date, end=end_date, interval=interval)

        if future_df.empty:
            print("  ⚠️ Skipped: No future market data available.")
            dump_ai_output_to_file(ai_output, "SKIPPED")
            skips += 1
            total_signals -= 1 
            continue

        entered = False
        trade_result = "OPEN" 
        
        for date, row in future_df.iterrows():
            low = row['Low']
            high = row['High']

            if not entered:
                if low <= entry_p:
                    entered = True
                    if high >= target_p:
                        trade_result = "WIN"
                        break
                    elif low <= stop_p:
                        trade_result = "LOSS"
                        break
            else:
                if high >= target_p:
                    trade_result = "WIN"
                    break
                elif low <= stop_p:
                    trade_result = "LOSS"
                    break
        
        if not entered:
            misses += 1
            print(f"  📉 Result: MISS (Price never dropped to entry of ${entry_p})")
            dump_ai_output_to_file(ai_output, "MISS")
        elif trade_result == "WIN":
            wins += 1
            print(f"  ✅ Result: WIN (Hit Target ${target_p})")
            dump_ai_output_to_file(ai_output, "WIN")
        elif trade_result == "LOSS":
            losses += 1
            print(f"  ❌ Result: LOSS (Hit Stop ${stop_p})")
            dump_ai_output_to_file(ai_output, "LOSS")
        else:
            open_trades += 1
            print(f"  ⏳ Result: EXPIRED ({eval_days} days passed, neither Target nor Stop was hit)")
            dump_ai_output_to_file(ai_output, "EXPIRED")
        

    print("\n" + "="*40)
    print(f"► BACKTEST RESULTS ({ACTIVE_HORIZON})")
    print("="*40)
    print(f"Total Iterations:   {TARGET_AMOUNT}")
    print(f"Valid Trade Setups: {total_signals}")
    print(f"Trades Triggered:   {wins + losses + open_trades}")
    print("-" * 40)
    print(f"Wins:               {wins}")
    print(f"Losses:             {losses}")
    print(f"Expired (No hit):   {open_trades}")
    print(f"Missed (No entry):  {misses}")
    print(f"Skipped (No Trade): {skips}")
    
    closed_trades = wins + losses
    if closed_trades > 0:
        win_rate = (wins / closed_trades) * 100
        print(f"\n►  WIN RATE: {win_rate:.2f}% (Out of {closed_trades} closed trades)")
    else:
        print("\n►  WIN RATE: N/A (No trades closed)")
    print("="*40)
    
# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    run_backtest()