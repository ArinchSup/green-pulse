import requests
import json
from aiAnalyst.news_fetcher import fetch_news, fetch_stock_profile
from aiAnalyst.database_manager import get_latest_prediction 
from aiAnalyst.config import OLLAMA_GENERATE_URL

def extract_tickers(user_question):
    extraction_prompt = f"""Extract US stock ticker symbols from the user's question.
- If multiple companies are mentioned, list all of them.
- If an industry/sector is mentioned (e.g., 'semiconductors', 'EV', 'AI'), return the top 2-3 representative ticker symbols for that sector (e.g., NVDA, AMD, TSM for semiconductors).
CRITICAL RULE: You must output ONLY a valid JSON object with an array key named "tickers".

Example 1: {{"tickers": ["MSFT", "META"]}}
Example 2: {{"tickers": ["NVDA", "AMD", "TSM"]}}
Example 3: {{"tickers": []}}

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
        data = json.loads(raw_text)
        
        tickers = data.get("tickers", [])
        clean_tickers = [str(t).strip().upper() for t in tickers if isinstance(t, str)]
        return clean_tickers[:3] 
        
    except Exception as e:
        return []

def ask_stock_bot(user_question, chat_history):
    
    MAX_HISTORY = 4
    history_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
    extraction_text = f"Context:\n{history_context}\n\nLatest Question: {user_question}"
    
    tickers = extract_tickers(extraction_text)

    OLLAMA_CHAT_URL = OLLAMA_GENERATE_URL.replace("/api/generate", "/api/chat")

    if not tickers:
        general_system = "You are an expert Financial Advisor Chatbot. Answer politely."
        messages = []
        messages.append({"role": "system", "content": general_system}) 
        messages.extend(chat_history) 
        messages.append({"role": "user", "content": user_question})
        
        payload = {"model": "llama3.1", "messages": messages, "stream": False}
        try:
            response = requests.post(OLLAMA_CHAT_URL, json=payload)
            bot_reply = response.json().get("message", {}).get("content", "Error")
            
            chat_history.append({"role": "user", "content": user_question})
            chat_history.append({"role": "assistant", "content": bot_reply})
            
            if len(chat_history) > MAX_HISTORY * 2:
                chat_history = chat_history[-(MAX_HISTORY * 2):]
                
            return bot_reply, chat_history 
            
        except Exception as e:
            return str(e), chat_history

    print(f"🔎 AI find ticker: {', '.join(tickers)}")
    
    all_contexts = ""
    for ticker in tickers:
        stock_info = fetch_stock_profile(ticker)
        recent_news = fetch_news(ticker, days_back=3)
        quant_analysis = get_latest_prediction(ticker)
        
        news_text = "No recent news."
        if recent_news:
            news_text = "\n".join([f"- {n['headline']}: {n['summary']}" for n in recent_news[:3]])
            
        all_contexts += f"""
=== FACT SHEET: {ticker} ===
Current Price: {stock_info.get('current_price')} | P/E: {stock_info.get('pe')}
Trend: {stock_info.get('graph_trend')} | Macro: {stock_info.get('macro_trend')}
[ Quant System Analysis ]
{quant_analysis}
[ News ]
{news_text}
===========================
"""

    system_prompt = f"""You are an expert Financial Advisor Chatbot representing an elite AI trading firm.
Answer the user's question clearly in the language they asked using ONLY the provided context.
Compare the stocks if asked, using the Proprietary Quant System analysis as your firm's official stance.

[ LIVE STOCK DATA ]
{all_contexts}"""

    messages = []
    messages.append({"role": "system", "content": system_prompt}) 
    messages.extend(chat_history)                                 
    messages.append({"role": "user", "content": user_question})   

    payload = {
        "model": "llama3.1", 
        "messages": messages, 
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload)
        bot_reply = response.json().get("message", {}).get("content", "Error generating response.")
        
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": bot_reply})
        
        if len(chat_history) > MAX_HISTORY * 2:
            chat_history = chat_history[-(MAX_HISTORY * 2):]
            
        return bot_reply, chat_history
        
    except Exception as e:
        return str(e), chat_history

if __name__ == "__main__":
    local_history = [] 
    
    while True:
        question = input("\n👤 You: ")
        if question.lower() in ['exit', 'quit']:
            break
            
        answer, local_history = ask_stock_bot(question, local_history)
        print(f"\n📈 AI Bot:\n{answer}")