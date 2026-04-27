import requests
import json
import datetime
from aiAnalyst.config import OLLAMA_GENERATE_URL

def query_ollama_text(prompt, model="stock-quant"):
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
        news_text = "NONE (No significant catalysts today)"
    else:
        news_text = "\n".join([f"{i+1}. {n['headline']} - {n['summary']}" for i, n in enumerate(all_news_list[:3])])

    today_str = datetime.date.today().strftime('%Y-%m-%d')
    input_text = f"""Stock: {ticker} | Date: {today_str} | Horizon: {mapped_horizon}
[ Price Action ]
Current Price: {stock_info.get('current_price', 0)} USD
Last 5 Days Close: {stock_info.get('last_5_days', [])}
Support: {stock_info.get('support', 0)} | Resistance: {stock_info.get('resistance', 0)}
Volume: {stock_info.get('volume_pct', 100)}% of 20-day avg

[ Technicals ]
RSI (14): {stock_info.get('rsi', 50)} | EMA200: {stock_info.get('ema_200', 0)}
MACD: {stock_info.get('macd', 'Unknown')} | Graph Trend: {stock_info.get('graph_trend', 'Unknown')}

[ Market Context ]
News:
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
    
    raw_response = query_ollama_text(prompt, model="stock-quant")
    
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        print("Failed to decode JSON from AI.")
        return {"action": "ERROR", "raw_output": raw_response}