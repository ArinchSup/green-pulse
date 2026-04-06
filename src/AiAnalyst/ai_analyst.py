import requests
import json
from config import OLLAMA_GENERATE_URL

def query_ollama(prompt):
    payload = {"model": "llama3", "prompt": prompt, "format": "json", "stream": False}
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload)
        response.raise_for_status()
        return json.loads(response.json().get("response", "{}"))
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return None

def analyze_individual_news(ticker, headline, summary):
    prompt = f"""You are an expert quantitative AI analyst for US Stocks.
Analyze this specific news and determine its impact on the stock.

CRITICAL RULES:
1. Output EXACTLY as a valid JSON object. No other text.
2. 'trend' MUST be exactly "bullish", "bearish", or "neutral".
3. 'scale' is an integer 1-5 (1=affect stock price 1 to 2 percent, 2=affect stock price 2 to 5 percent, 3=affect stock price 5 to 10 percent, 4=affect stock price 10 to 20 percent, 5=affect stock price more than 20 percent).
4. 'reason' is 2-3 sentences.

Stock: {ticker}
Headline: {headline}
Summary: {summary}"""
    return query_ollama(prompt)

def analyze_overall_sentiment(ticker, all_news_list):
    if not all_news_list: return None
    combined = "\n".join([f"{n['headline']} - {n['summary']}" for n in all_news_list])
    prompt = f"""You are an expert quantitative AI analyst for US Stocks.
Analyze the following batch of recent news items and determine the OVERALL aggregate market impact for the stock.

CRITICAL RULES:
1. Output EXACTLY as a valid JSON object. No other text.
2. 'trend' MUST be exactly "bullish", "bearish", or "neutral".
3. 'scale' is an integer 1-5 representing the combined impact weight where 1=affect stock price 1 to 2 percent, 2=affect stock price 2 to 5 percent, 3=affect stock price 5 to 10 percent, 4=affect stock price 10 to 20 percent, 5=affect stock price more than 20 percent.
4. 'reason' is exactly 2-3 sentences summarizing the overall fundamental sentiment derived from all news.

Stock: {ticker}
Recent News Data:
{combined}"""
    return query_ollama(prompt)

def re_rank_results(query, retrieved_docs):
    """
    ส่งข่าวที่ Vector DB หามาได้ ไปให้ AI ตรวจสอบซ้ำว่าตรงประเด็นไหม
    """
    if not retrieved_docs: return []
    
    # รวมข่าวเป็นรายการเดียว
    docs_text = "\n".join([f"- {d}" for d in retrieved_docs])
    
    prompt = f"""You are a strict filtering assistant. 
User Query: "{query}"

Below are search results from a database. 
Some might be irrelevant. Your task is to return ONLY the results that are DIRECTLY related to the query.

Search Results:
{docs_text}

CRITICAL RULES:
1. Output ONLY the relevant sentences.
2. If none are relevant, output "NONE".
3. Do not explain anything.
"""
    response = query_ollama(prompt) 
    return response