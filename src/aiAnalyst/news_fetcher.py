import json
import finnhub
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
#from aiAnalyst.config import FINNHUB_API_KEY, EXCLUDE_KEYWORDS
from config import FINNHUB_API_KEY, EXCLUDE_KEYWORDS

# 🌟 ตั้งค่า Ollama AI
OLLAMA_URL = "http://localhost:11434/api/generate"
AGENT1_MODEL = "llama3.1"  
AGENT2_MODEL = "news-v3"   

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

ALLOWED_TAGS = [
    "#Earnings", "#Guidance", "#Business_Momentum",   
    "#Product_Launch", "#Partnership", "#Merger_Acquisition", "#Leadership",          
    "#Lawsuit_Regulation", "#Macro_Economy", "#Scandal", "#Other"                
]

def clean_text(text):
    if not text: return ""
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u00a0': ' ', '\n': ' '
    }
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.strip()

def is_relevant_news(headline, summary):
    safe_summary = summary if summary else ""
    full_text = (headline + " " + safe_summary).lower()
    return not any(keyword in full_text for keyword in EXCLUDE_KEYWORDS)

def get_similarity(a, b): 
    return SequenceMatcher(None, a, b).ratio()

def deduplicate_news(news_list, threshold=0.7):
    if not news_list: return []
    unique_news = []
    for news in news_list:
        is_duplicate = False
        for existing in unique_news:
            similarity = get_similarity(news['headline'].lower(), existing['headline'].lower())
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_news.append(news)
    return unique_news

def fetch_news(ticker, days_back=3): 
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back)
    try:
        news_data = finnhub_client.company_news(ticker, _from=str(start_date), to=str(end_date))
        clean_news = []
        for item in (news_data or []):
            headline = item['headline']
            summary = item.get('summary', '') or headline
            if ticker in item.get('related', '') and is_relevant_news(headline, summary):
                dt_object = datetime.datetime.fromtimestamp(item['datetime'])
                clean_news.append({
                    "id": item['id'], "time": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
                    "headline": headline, "summary": summary,
                    "related_tags": item.get('related', ''), "url": item['url']
                })
        return deduplicate_news(clean_news, threshold=0.7)
    except Exception as e:
        return []

def calculate_atr(df, period=14):
    if len(df) < period: return 0.0
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return float(round(atr.iloc[-1], 2))

def get_macro_trend(target_date, ticker):
    crypto_stocks = ["MSTR", "IREN", "WULF", "CIFR", "CORZ", "MARA", "CLSK"]
    space_small_caps = ["ASTS", "RKLB", "BKSY", "LUNR", "RDW", "KTOS", "UMAC"]
    nuclear_energy = ["CEG", "VST", "CCJ", "OKLO", "LEU", "UUUU"]
    
    if ticker in crypto_stocks: index_ticker, index_name = "BTC-USD", "Bitcoin"
    elif ticker in space_small_caps: index_ticker, index_name = "IWM", "Russell 2000"
    elif ticker in nuclear_energy: index_ticker, index_name = "XLU", "Utilities Sector"
    elif ticker in ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD"]: index_ticker, index_name = "QQQ", "NASDAQ"
    else: index_ticker, index_name = "SPY", "S&P 500"

    end_fetch = target_date + datetime.timedelta(days=5)
    start_fetch = target_date - datetime.timedelta(days=365)
    try:
        idx_data = yf.Ticker(index_ticker).history(start=start_fetch, end=end_fetch)
        if idx_data.empty: return f"Neutral (No data for {index_name})"
        if idx_data.index.tz is not None: idx_data.index = idx_data.index.tz_convert(None)
        
        target_dt = pd.to_datetime(target_date)
        if target_dt.tzinfo is not None: target_dt = target_dt.tz_convert(None)
            
        valid_dates = idx_data.index[idx_data.index <= target_dt]
        if len(valid_dates) == 0: return f"Neutral (No data for {index_name})"
        
        past_idx = idx_data.loc[:valid_dates[-1]]
        if len(past_idx) < 20: return f"Neutral (Insufficient data for {index_name})"
        
        curr_p = past_idx['Close'].iloc[-1]
        ema200_p = past_idx['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        trend_status = "Bullish" if curr_p > ema200_p else "Bearish"
        position = "Above" if curr_p > ema200_p else "Below"
        return f"{trend_status} ({position} EMA200 of {index_name})"
    except Exception as e:
        return "Neutral (Error fetching macro)"

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    if down == 0: return 100
    rs = up / down
    return float(round(100. - 100. / (1. + rs), 2))

def fetch_stock_profile(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty: raise ValueError("No historical data")
        
        info = yf.Ticker(ticker).info
        
        # ==========================================
        # Set 1: Fundamental Data (for Pillar 1)
        # ==========================================
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        raw_summary = info.get('longBusinessSummary', 'No description available.')
        business_summary = raw_summary[:400] + "..." if len(raw_summary) > 400 else raw_summary
        
        pe_raw = info.get('trailingPE')
        pe = round(pe_raw, 2) if isinstance(pe_raw, (int, float)) else "N/A"
        fpe_raw = info.get('forwardPE')
        fpe = round(fpe_raw, 2) if isinstance(fpe_raw, (int, float)) else "N/A"
        
        target_raw = info.get('targetMeanPrice')
        analyst_target = round(target_raw, 2) if isinstance(target_raw, (int, float)) else "N/A"
        
        market_cap = info.get('marketCap', 'N/A')
        revenue_growth = info.get('revenueGrowth', 'N/A')
        profit_margin = info.get('profitMargin', 'N/A')
        roe = info.get('returnOnEquity', 'N/A')
        debt_to_equity = info.get('debtToEquity', 'N/A')
        free_cashflow = info.get('freeCashflow', 'N/A')
        
        def format_pct(val): return f"{round(val * 100, 2)}%" if isinstance(val, (int, float)) else "N/A"
        def format_num(val): return f"${round(val / 1e9, 2)}B" if isinstance(val, (int, float)) else "N/A"

        fundamental_data = {
            "sector": sector, "industry": industry,
            "business_summary": business_summary,
            "current_price": float(round(hist['Close'].iloc[-1], 2)),
            "pe": pe, "fpe": fpe, 
            "analyst_target": analyst_target,
            "market_cap": format_num(market_cap),
            "revenue_growth": format_pct(revenue_growth),
            "profit_margin": format_pct(profit_margin),
            "roe": format_pct(roe),
            "debt_to_equity": debt_to_equity,
            "free_cashflow": format_num(free_cashflow)
        }

        # ==========================================
        # Set 2: Technical Data (for Pillar 2)
        # ==========================================
        current_close = float(round(hist['Close'].iloc[-1], 2))
        last_5_days = [float(x) for x in hist['Close'].tail(5).round(2).tolist()]
        
        avg_vol_20d = hist['Volume'].tail(20).mean()
        current_vol = hist['Volume'].iloc[-1]
        vol_pct = round((current_vol / avg_vol_20d) * 100) if avg_vol_20d > 0 else 100
        
        swing_low = float(round(hist['Low'].tail(30).min(), 2))
        swing_high = float(round(hist['High'].tail(30).max(), 2))
        support_1 = float(round(hist['Low'].tail(10).min(), 2))
        resistance_1 = float(round(hist['High'].tail(10).max(), 2))
        
        diff = swing_high - swing_low
        fib_1618 = float(round(swing_high + (diff * 0.618), 2))
        fib_0786 = float(round(swing_high - (diff * 0.786), 2))
        fib_0618 = float(round(swing_high - (diff * 0.382), 2))
        fib_0382 = float(round(swing_high - (diff * 0.618), 2))
        
        ema_200 = float(round(hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1], 2))
        atr_14 = float(calculate_atr(hist))
        rsi_val = float(calculate_rsi(hist['Close'].values))
        
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_status = "Bullish Crossover" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
        
        if current_close > ema_200 and macd_status == "Bullish Crossover": trend = "Strong Uptrend"
        elif current_close < ema_200 and macd_status == "Bearish": trend = "Downtrend"
        else: trend = "Consolidation / Sideways"
            
        macro_trend = get_macro_trend(datetime.date.today(), ticker)

        technical_data = {
            "current_price": current_close,
            "macro_trend": macro_trend,
            "last_5_days": last_5_days, "atr": atr_14, "volume_pct": vol_pct,
            "support_1": support_1, "support_2": swing_low,
            "resistance_1": resistance_1, "resistance_2": swing_high,
            "fib_1618": fib_1618, "fib_0786": fib_0786, "fib_0618": fib_0618, "fib_0382": fib_0382,
            "rsi": round(rsi_val, 2), "ema_200": ema_200, "macd": macd_status, "graph_trend": trend
        }

        return {
            "fundamental": fundamental_data,
            "technical": technical_data
        }
    except Exception as e:
        print(f"Error fetching profile for {ticker}: {e}")
        return {"fundamental": {}, "technical": {}}
    
def scrape_full_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        full_text = " ".join([p.get_text() for p in paragraphs])
        
        return clean_text(full_text[:3000]) if full_text else ""
    except Exception as e:
        print(f"Scrape Error for {url}: {e}")
        return ""
    
def get_daily_price_change(ticker, target_date_str):
    try:
        start_date = pd.to_datetime(target_date_str) - datetime.timedelta(days=5)
        end_date = pd.to_datetime(target_date_str) + datetime.timedelta(days=1)
        
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if df.empty: return 0.0
        
        target_dt = pd.to_datetime(target_date_str).tz_localize(df.index.tz)
        past_data = df.loc[df.index <= target_dt]
        
        if len(past_data) >= 2:
            current_close = past_data['Close'].iloc[-1]
            prev_close = past_data['Close'].iloc[-2]
            pct_change = ((current_close - prev_close) / prev_close) * 100
            return float(round(pct_change, 2))
        return 0.0
    except Exception as e:
        print(f"Price Change Error for {ticker}: {e}")
        return 0.0


# ==========================================
# AI Agents Integration
# ==========================================

def run_agent_1_reporter(full_text, ticker):
    if not full_text.strip(): return None
    prompt = f"""
    You are a ruthless Financial Analyst focusing on {ticker}.
    Read the article below.
    
    STEP 1: Classify the news. Is this article DIRECTLY about the fundamentals of {ticker}? 
    If the article is primarily about another company, an analyst rating, or an opinion piece, the answer is FALSE.

    STEP 2: Output EXACTLY this JSON structure:
    {{
        "is_fundamental_news": <true or false>,
        "rejection_reason": "<Leave empty if true. If false, briefly state why>",
        "summary": "<If true, write a punchy 2-line summary of the business impact. If false, leave empty>",
        "tags": ["<If true, pick 1 or 2 tags STRICTLY from {ALLOWED_TAGS} ONLY. Do not invent new tags.>"]
    }}
    
    Respond strictly in English. Output ONLY valid JSON.
    Article: {full_text}
    """
    try:
        res = requests.post(OLLAMA_URL, json={"model": AGENT1_MODEL, "prompt": prompt, "stream": False, "format": "json"}).json()
        return json.loads(res["response"])
    except Exception as e:
        print(f"Agent 1 Error: {e}")
        return None

def run_news_v3_inference(ticker, summaries):
    if not summaries: return None
    
    instruction = f"Analyze the impact of today's fundamental news on {ticker}. Identify the primary driver and PREDICT the 3D impact assessment."
    input_str = json.dumps(summaries, indent=2, ensure_ascii=False)
    
    prompt = f"""You are an expert Chief Investment Editor and Quant Analyst.
Below is an instruction that describes a task, paired with an input containing impact-focused summaries of fundamental news.
Write a response that appropriately completes the request.
CRITICAL RULE: You MUST output ONLY a valid JSON object. Do not include markdown blocks (like ```json), greetings, or comments.

### Instruction:
{instruction}

### Input:
{input_str}

### Response:
"""
    try:
        res = requests.post(OLLAMA_URL, json={"model": AGENT2_MODEL, "prompt": prompt, "stream": False, "format": "json"}).json()
        return json.loads(res["response"])
    except Exception as e:
        print(f"news-v3 Error: {e}")
        return None

def get_live_news_impact(ticker, days_back=1):
    print(f"📡 Fetching LIVE news for {ticker}...")
    news_list = fetch_news(ticker, days_back=days_back)
    
    if not news_list:
        return {"status": "No news today"}
        
    daily_summaries = []
    
    for news in news_list[:5]:
        full_text = scrape_full_text(news.get('url', ''))
        if not full_text: full_text = clean_text(news.get('headline', '') + " " + news.get('summary', ''))
        
        agent1_response = run_agent_1_reporter(full_text, ticker)
        
        if agent1_response and agent1_response.get('is_fundamental_news') is True:
            valid_tags = [t for t in agent1_response.get('tags', []) if t in ALLOWED_TAGS]
            if not valid_tags: valid_tags = ["#Other"]
            
            summary_to_keep = {
                "summary": clean_text(agent1_response.get('summary', '')),
                "tags": valid_tags
            }
            daily_summaries.append(summary_to_keep)
            
    if not daily_summaries:
        return {"status": "No fundamental news found"}
        
    print(f"🧠 Running 'news-v3' analysis on {len(daily_summaries)} valid news items...")
    impact_result = run_news_v3_inference(ticker, daily_summaries)
    
    return impact_result

if __name__ == "__main__":
    ticker = "AAPL"
    result = get_live_news_impact(ticker)
    print(result)