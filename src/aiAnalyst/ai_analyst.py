import requests
import json
import datetime
from aiAnalyst.config import OLLAMA_GENERATE_URL, OLLAMA_MODEL

def query_ollama_text(prompt, model=OLLAMA_MODEL):
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "{}")
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return "{}"

def analyze_overall_sentiment(ticker, stock_info, all_news_list, horizon="Short-term"):
    horizon_map = {
        "Short-term": "Short-term (< 1 Week)",
        "Mid-term": "Mid-term (1 Week - 3 Months)",
        "Long-term": "Long-term (3 Months - 1 Year)"
    }
    mapped_horizon = horizon_map.get(horizon, f"{horizon} (< 1 Week)")

    if not all_news_list:
        news_text = "NONE (No significant catalysts in the last 3 days)"
    else:
        news_text = "\n".join([f"{i+1}. {n['headline']} - {n['summary']}" for i, n in enumerate(all_news_list[:4])])

    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    input_text = f"""Stock: {ticker} | Sector: {stock_info.get('sector')} | Industry: {stock_info.get('industry')}
Date: {today_str} | Horizon: {mapped_horizon}

[ Fundamentals & Macro ]
P/E Ratio: {stock_info.get('pe')} | Forward P/E: {stock_info.get('fpe')} | Analyst Target: {stock_info.get('analyst_target')}
Macro Market Trend: {stock_info.get('macro_trend')}

[ Price Action & Volatility ]
Current Price: {stock_info.get('current_price')} USD
Last 5 Days Close: {stock_info.get('last_5_days')}
ATR (14): {stock_info.get('atr')} USD
Volume: {stock_info.get('volume_pct')}% of 20-day avg

[ Key Levels (Price Action) ]
Resistance 2: {stock_info.get('resistance_2')}
Resistance 1: {stock_info.get('resistance_1')}
--- Current Price: {stock_info.get('current_price')} ---
Support 1: {stock_info.get('support_1')}
Support 2: {stock_info.get('support_2')}

[ Fibonacci Levels ]
Fib 1.618 (Extension): {stock_info.get('fib_1618')}
Fib 0.786 (Retracement): {stock_info.get('fib_0786')}
Fib 0.618 (Retracement): {stock_info.get('fib_0618')}
Fib 0.382 (Retracement): {stock_info.get('fib_0382')}

[ Technicals ]
RSI (14): {stock_info.get('rsi')} | EMA200: {stock_info.get('ema_200')}
MACD: {stock_info.get('macd')} | Graph Trend: {stock_info.get('graph_trend')}

[ Market Context ]
News (Last 3 Days):
{news_text}"""

    instruction = f"Analyze {ticker} for {mapped_horizon} investment."
    
    prompt = f"""You are an elite Quantitative Trading AI.
Below is an instruction that describes a task, paired with an input that provides market context, technical indicators, and news.
Write a response that appropriately completes the request.
CRITICAL RULE: You MUST output ONLY a valid JSON object. Do not include markdown blocks (like ```json), greetings, or comments.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    raw_response = query_ollama_text(prompt, model=OLLAMA_MODEL)
    
    try:
        data = json.loads(raw_response)
        current_price = stock_info.get('current_price', 0)
        
        if current_price > 0 and data.get("action") == "BUY":
            max_multipliers = {
                "Short-term (< 1 Week)": 1.15,
                "Mid-term (1 Week - 3 Months)": 1.30,
                "Long-term (3 Months - 1 Year)": 1.50
            }
            max_allowed = current_price * max_multipliers.get(mapped_horizon, 1.20)
            
            target_val = data.get("target_price")
            if isinstance(target_val, (int, float)) and target_val > max_allowed:
                data["target_price"] = round(max_allowed, 2)

        return data
        
    except json.JSONDecodeError:
        print("Failed to decode JSON from AI.")
        return {"action": "ERROR", "raw_output": raw_response}