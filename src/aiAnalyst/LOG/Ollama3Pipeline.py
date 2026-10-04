import finnhub
import datetime
import sqlite3
import requests
import json

# ==========================================
# Configuration
# ==========================================
FINNHUB_API_KEY = "d77st29r01qsamsi2qe0d77st29r01qsamsi2qeg"
TARGET_TICKER = "IREN"
DB_NAME = "rag_stock_news.db"
OLLAMA_URL = "http://localhost:11434/api/generate"

EXCLUDE_KEYWORDS = ["etf", "etfs", "mutual fund", "index fund", "model portfolio", "magnificent seven"]

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ==========================================
# Database Management
# ==========================================
def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_news (
            news_id TEXT PRIMARY KEY,
            ticker TEXT,
            published_at DATETIME,
            headline TEXT,
            summary TEXT,
            tags TEXT,
            url TEXT,
            ai_trend TEXT,
            ai_scale INTEGER,
            ai_reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_and_get_new_news(ticker, clean_news_list):
    if not clean_news_list:
        return []

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    newly_added_news = []
    
    for news in clean_news_list:
        cursor.execute('''
            INSERT OR IGNORE INTO stock_news 
            (news_id, ticker, published_at, headline, summary, tags, url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(news['id']), ticker, news['time'], news['headline'], news['summary'], news['related_tags'], news['url']))
        
        if cursor.rowcount == 1:
            newly_added_news.append(news)
            
    conn.commit()
    conn.close()
    return newly_added_news

def update_individual_analysis(news_id, analysis_data):
    if not analysis_data: 
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    trend = analysis_data.get('trend') or analysis_data.get('Trend') or 'unknown'
    scale = analysis_data.get('scale') or analysis_data.get('Scale') or 0
    reason = analysis_data.get('reason') or analysis_data.get('Reason') or ''
    
    cursor.execute('''
        UPDATE stock_news 
        SET ai_trend = ?, ai_scale = ?, ai_reason = ?
        WHERE news_id = ?
    ''', (str(trend).lower(), scale, reason, str(news_id)))
    
    conn.commit()
    conn.close()

# ==========================================
# Data Fetching & Filtering
# ==========================================
def is_relevant_news(headline, summary):
    safe_summary = summary if summary else ""
    full_text = (headline + " " + safe_summary).lower()
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in full_text: 
            return False
    return True

def fetch_news(ticker, days_back=1):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    try:
        news_data = finnhub_client.company_news(ticker, _from=str(start_date), to=str(end_date))
        clean_news = []
        for item in (news_data or []):
            headline = item['headline']
            summary = item.get('summary', '')
            related_tags = item.get('related', '')
            
            if not summary or len(summary.strip()) < 5:
                summary = headline 
                
            if ticker in related_tags and is_relevant_news(headline, summary):
                dt_object = datetime.datetime.fromtimestamp(item['datetime'])
                clean_news.append({
                    "id": item['id'],
                    "time": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
                    "headline": headline,
                    "summary": summary,
                    "related_tags": related_tags,
                    "url": item['url']
                })
        return clean_news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# ==========================================
# AI Analysis Modules
# ==========================================
def query_ollama(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
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
    if not all_news_list:
        return None
        
    combined_news = ""
    for idx, news in enumerate(all_news_list, 1):
        combined_news += f"{idx}. {news['headline']} - {news['summary']}\n"

    prompt = f"""You are an expert quantitative AI analyst for US Stocks.
Analyze the following batch of recent news items and determine the OVERALL aggregate market impact for the stock.

CRITICAL RULES:
1. Output EXACTLY as a valid JSON object. No other text.
2. 'trend' MUST be exactly "bullish", "bearish", or "neutral".
3. 'scale' is an integer 1-5 representing the combined impact weight where 1=affect stock price 1 to 2 percent, 2=affect stock price 2 to 5 percent, 3=affect stock price 5 to 10 percent, 4=affect stock price 10 to 20 percent, 5=affect stock price more than 20 percent.
4. 'reason' is exactly 2-3 sentences summarizing the overall fundamental sentiment derived from all news.

Stock: {ticker}
Recent News Data:
{combined_news}"""
    
    return query_ollama(prompt)

# ==========================================
# Main Pipeline Execution
# ==========================================
def run_pipeline():
    print(f"Starting pipeline for ticker: {TARGET_TICKER}")
    
    setup_database()
    
    print("Fetching data from Finnhub...")
    all_fetched_news = fetch_news(TARGET_TICKER, days_back=2)
    print(f"Total relevant news retrieved: {len(all_fetched_news)}")
    
    if not all_fetched_news:
        print("No news to process. Exiting.")
        return

    # 1. Overall Analysis (Process all fetched news together)
    print("\n--- Performing Overall Sentiment Analysis ---")
    overall_analysis = analyze_overall_sentiment(TARGET_TICKER, all_fetched_news)
    
    if overall_analysis:
        o_trend = overall_analysis.get('trend', 'unknown').upper()
        o_scale = overall_analysis.get('scale', 0)
        o_reason = overall_analysis.get('reason', 'N/A')
        print(f"OVERALL TREND : {o_trend} (Scale: {o_scale})")
        print(f"OVERALL REASON: {o_reason}")
    else:
        print("Failed to generate overall analysis.")

    # 2. Individual Analysis (Process only new news)
    print("\n--- Processing Individual New Entries ---")
    new_news_to_process = save_and_get_new_news(TARGET_TICKER, all_fetched_news)
    print(f"New entries added to database: {len(new_news_to_process)}")
    
    for idx, news in enumerate(new_news_to_process, 1):
        print(f"\nProcessing [{idx}/{len(new_news_to_process)}]: {news['headline']}")
        
        analysis = analyze_individual_news(TARGET_TICKER, news['headline'], news['summary'])
        
        if analysis:
            i_trend = analysis.get('trend', 'unknown').upper()
            i_scale = analysis.get('scale', 0)
            i_reason = analysis.get('reason', 'N/A')
            
            print(f"Result: {i_trend} | Scale: {i_scale}")
            print(f"Reason: {i_reason}")
            
            update_individual_analysis(news['id'], analysis)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()