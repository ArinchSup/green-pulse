import yfinance as yf
import requests
import json
import sqlite3
from aiAnalyst.news_fetcher import fetch_news, fetch_stock_profile
from aiAnalyst.database_manager import get_latest_prediction
from aiAnalyst.config import OLLAMA_GENERATE_URL, DB_NAME

def extract_ticker(user_question):
    extraction_prompt = f"""Extract the US stock ticker symbol from the user's question.
Convert company names (e.g., 'Apple', 'Nvidia', 'Tesla') to correct tickers (e.g., 'AAPL', 'NVDA', 'TSLA').
CRITICAL RULE: You must output ONLY a valid JSON object. Do not add any other text.

Example 1: {{"ticker": "AAPL"}}
Example 2: {{"ticker": "NONE"}}

User Question: {user_question}"""

    payload = {
        "model": "llama3.1", 
        "prompt": extraction_prompt,
        "stream": False,
        "format": "json" 
    }
    
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload)
        raw_text = response.json().get("response", "{}")
        
        import json
        data = json.loads(raw_text)
        
        ticker = data.get("ticker", "NONE").strip().upper()
        
        import re
        match = re.search(r'^[A-Z]+$', ticker)
        return ticker if match else "NONE"
        
    except Exception as e:
        return "NONE"


def get_latest_prediction(ticker):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        horizons = ["Short-term", "Mid-term", "Long-term"]
        quant_summaries = []
        
        for h in horizons:
            cursor.execute('''
                SELECT action, confidence_score, entry_price, target_price, stop_loss, rationale, analysis_date
                FROM ai_predictions 
                WHERE ticker = ? AND horizon LIKE ?
                ORDER BY id DESC LIMIT 1
            ''', (ticker, f"{h}%"))
            
            row = cursor.fetchone()
            if row:
                action, conf, entry, target, stop, rationale, date = row
                
                summary_block = (
                    f"[{h} Analysis - {date}]\n"
                    f"Signal: {action} with {conf}% Confidence\n"
                    f"Price Targets -> Entry: {entry} | Target: {target} | Stop: {stop}\n"
                    f"Quant Reasoning: {rationale}\n"
                )
                quant_summaries.append(summary_block)
                
        conn.close()
        
        if quant_summaries:
            return "\n".join(quant_summaries)
        else:
            return "No recent quant analysis available in database."
            
    except Exception as e:
        return f"Error fetching quant data: {e}"
    
def ask_stock_bot(user_question):
    ticker = extract_ticker(user_question)
    
    if ticker == "NONE" or not ticker:
        general_prompt = f"""You are an expert Financial Advisor Chatbot. 
Answer politely. If they ask about a stock, tell them to mention the stock name clearly.
Question: {user_question}"""
        payload = {"model": "llama3.1", "prompt": general_prompt, "stream": False}
        res = requests.post(OLLAMA_GENERATE_URL, json=payload)
        return res.json().get("response", "Error")
    
    stock_info = fetch_stock_profile(ticker)
    recent_news = fetch_news(ticker, days_back=3)
    
    quant_analysis = get_latest_prediction(ticker)
    
    news_text = "No recent news."
    if recent_news:
        news_text = "\n".join([f"- {n['headline']}: {n['summary']}" for n in recent_news[:5]])

    system_prompt = f"""You are an expert Financial Advisor Chatbot representing an elite AI trading firm.
Answer the user's question clearly in the language they asked (e.g., Thai or English) using ONLY the provided context. 
If the user asks for advice or direction, you MUST incorporate the 'Our Proprietary Quant System' analysis into your answer to explain our firm's official stance on this stock.

=== STOCK FACT SHEET: {ticker} ===
[ Fundamentals & Technicals ]
Current Price: {stock_info.get('current_price')}
Graph Trend: {stock_info.get('graph_trend')} | Macro Trend: {stock_info.get('macro_trend')}
P/E Ratio: {stock_info.get('pe')} | Volatility (ATR): {stock_info.get('atr')}
Support 1: {stock_info.get('support_1')} | Resistance 1: {stock_info.get('resistance_1')}

[ Our Proprietary Quant System Analysis ]
{quant_analysis}

[ Recent News ]
{news_text}
==================================

User Question: {user_question}
"""

    payload = {
        "model": "llama3.1", 
        "prompt": system_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload)
        return response.json().get("response", "Error generating response.")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    while True:
        question = input("\n👤 You: ")
        if question.lower() in ['exit', 'quit']:
            break
            
        answer = ask_stock_bot(question)
        print(f"\n📈 AI Bot:\n{answer}")