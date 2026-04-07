import requests
import json
import re
from config import OLLAMA_GENERATE_URL

def query_ollama_text(prompt, model="stock-analyst"):
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return ""

def calculate_fib_string(current_price, high, low):
    price_swing = high - low
    if price_swing <= 0:
        return "N/A"

    retracements = {
        "38.2%": round(high - (price_swing * 0.382), 2),
        "61.8%": round(high - (price_swing * 0.618), 2),
        "78.6%": round(high - (price_swing * 0.786), 2)
    }

    closest_level = min(retracements.keys(), key=lambda k: abs(current_price - retracements[k]))

    parts = []
    for level in ["38.2%", "61.8%", "78.6%"]:
        price = retracements[level]
        if level == closest_level:
            parts.append(f">>{level}: {price} USD<<")
        else:
            parts.append(f"{level}: {price} USD")
            
    return f"Buy (Fib level: {', '.join(parts)})"

def parse_lora_output(raw_text, current_price, high, low):
    data = {"trend": "neutral", "scale": 0, "reason": ""}
    
    trend_match = re.search(r"Overall Tr[ea]nd:\s*([A-Za-z]+)\s*scale\s*(\d+)", raw_text, re.IGNORECASE)
    if trend_match:
        data["trend"] = trend_match.group(1).lower()
        data["scale"] = int(trend_match.group(2))

    is_buy = re.search(r"Worth buying:\s*Buy", raw_text, re.IGNORECASE)
    
    if is_buy:
        fib_recommendation = calculate_fib_string(current_price, high, low)
        raw_text = re.sub(r"Worth buying:.*", f"Worth buying: {fib_recommendation}", raw_text)
    
    data["raw_output"] = raw_text
    return data

def build_lora_prompt(ticker, stock_info, news_items):
    news_text = ""
    for i, n in enumerate(news_items, 1):
        news_text += f"{i}. {n['headline']} - {n['summary']}\n"
        
    price = stock_info.get('current_price', 0.0)
    high = stock_info.get('high_52w', 0.0)
    low = stock_info.get('low_52w', 0.0)
    trend = stock_info.get('graph_trend', 'Neutral')

    prompt = f"""Stock: {ticker}
Current Price: {price} USD
52W High: {high} USD
52W Low: {low} USD
Graph Trend: {trend}
News Today:
{news_text.strip()}"""
    return prompt

def analyze_overall_sentiment(ticker, stock_info, all_news_list):
    if not all_news_list: return None
    
    prompt = build_lora_prompt(ticker, stock_info, all_news_list)
    raw_response = query_ollama_text(prompt, model="stock-analyst")
    
    return parse_lora_output(
        raw_response, 
        stock_info['current_price'], 
        stock_info['high_52w'], 
        stock_info['low_52w']
    )

def analyze_individual_news(ticker, stock_info, headline, summary):
    news_item = [{"headline": headline, "summary": summary}]
    prompt = build_lora_prompt(ticker, stock_info, news_item)
    raw_response = query_ollama_text(prompt, model="stock-analyst")
    return parse_lora_output(raw_response)

def re_rank_results(query, retrieved_docs):
    if not retrieved_docs: return []
    docs_text = "\n".join([f"- {d}" for d in retrieved_docs])
    prompt = f"""You are a strict filtering assistant. User Query: "{query}"\nSearch Results:\n{docs_text}\nCRITICAL RULES: Output ONLY the relevant sentences. If none are relevant, output "NONE". Do not explain anything."""
    
    # ใช้ Llama3 เดิมสำหรับงานทั่วไป
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload)
        return response.json().get("response", "")
    except Exception:
        return ""