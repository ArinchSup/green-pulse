import finnhub
import datetime
import numpy as np
import yfinance as yf
from difflib import SequenceMatcher
from aiAnalyst.config import FINNHUB_API_KEY, EXCLUDE_KEYWORDS

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def is_relevant_news(headline, summary):
    safe_summary = summary if summary else ""
    full_text = (headline + " " + safe_summary).lower()
    return not any(keyword in full_text for keyword in EXCLUDE_KEYWORDS)

def get_similarity(a, b): # calculate similarity between two strings result 0-1 (0 = completely different, 1 = identical)
    return SequenceMatcher(None, a, b).ratio()

def deduplicate_news(news_list, threshold=0.7):
    if not news_list:
        return []

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

def fetch_news(ticker, days_back=1):
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
                
        # print(f"Total fetched: {len(clean_news)}")
        clean_news = deduplicate_news(clean_news, threshold=0.7)
        # print(f"After deduplication: {len(clean_news)}")
        
        return clean_news
    except Exception as e:
        # print(f"Error fetching news: {e}")
        return []

def determine_graph_trend(price, high, low, dp, rsi, ema200):
    if high == low or high <= 0:
        return "Neutral"

    position = (price - low) / (high - low)
    is_above_ema = price > ema200

    # 1. Extreme Cases
    if rsi > 80 and dp > 3.0: return "Parabolic"
    if rsi < 25 and dp < -4.0: return "Extreme Crash"

    # 2. Ceiling/Breakout Zone
    if position >= 0.98:
        if dp > 2.0: return "Blue Sky Breakout"
        elif dp < -2.0: return "Bullish Pullback"
        else: return "Testing Ceiling"

    # 3. Upper Zone: 80% - 98%
    elif position >= 0.80:
        if dp > 2.5: return "Strong Breakout"
        elif dp < -2.5: return "Bullish Pullback"
        elif rsi > 70: return "Parabolic"
        else: return "Strong Uptrend" if is_above_ema else "Mature Uptrend"

    # 4. Upper Zone: 60% - 80%
    elif position >= 0.60:
        if dp > 3.0: return "Sharp Momentum"
        elif dp < -3.0: return "Bearish Pullback"
        elif abs(dp) > 4.0: return "High Volatility"
        else: return "Steady Uptrend"

    # 5. Upper Zone: 40% - 60%
    elif position >= 0.40:
        if dp > 2.5: return "Bullish Recovery"
        elif dp < -2.5: return "Bearish Pullback"
        elif abs(dp) > 3.5: return "High Volatility"
        else: return "Sideways Consolidation"

    # 6. Upper Zone: 15% - 40%
    elif position >= 0.15:
        if dp > 3.0: return "Recovery"
        elif dp < -3.5: return "Sharp Crash"
        elif not is_above_ema: return "Bearish Channel"
        else: return "Strong Downtrend"

    # 7. Upper Zone: 5% - 15%
    elif position >= 0.05:
        if dp > 4.0: return "Floor Reversal"
        elif dp < -4.0: return "Extreme Crash"
        else: return "Strong Downtrend"

    # 8. Upper Zone: < 5%
    else:
        if dp > 3.0: return "Floor Reversal"
        elif dp < -3.0: return "Extreme Crash"
        else: return "Near Floor"

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    if down == 0: return 100
    rs = up / down
    return 100. - 100. / (1. + rs)

def fetch_stock_profile(ticker):
    try:
        quote = finnhub_client.quote(ticker)
        curr_price = quote.get('c', 0.0)
        daily_pct = quote.get('dp', 0.0)

        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty: raise ValueError("No historical data")

        close_prices = hist['Close'].values
        rsi_val = calculate_rsi(close_prices)
        ema_200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
        
        high_52w = float(hist['High'].max())
        low_52w = float(hist['Low'].min())

        trend = determine_graph_trend(curr_price, high_52w, low_52w, daily_pct, rsi_val, ema_200)

        return {
            "current_price": curr_price,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "graph_trend": trend,
            "rsi": round(rsi_val, 2),
            "ema_200": round(ema_200, 2)
        }
    except Exception as e:
        # print(f"Error fetching profile: {e}")
        return {
            "current_price": 0,
            "high_52w": 0,
            "low_52w": 0,
            "graph_trend": "Unknown",
            "rsi": 50,
            "ema_200": 0
        }