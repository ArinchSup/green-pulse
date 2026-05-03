import finnhub
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from difflib import SequenceMatcher
from aiAnalyst.config import FINNHUB_API_KEY, EXCLUDE_KEYWORDS

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

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
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        pe_raw = info.get('trailingPE')
        pe = round(pe_raw, 2) if isinstance(pe_raw, (int, float)) else "N/A"
        fpe_raw = info.get('forwardPE')
        fpe = round(fpe_raw, 2) if isinstance(fpe_raw, (int, float)) else "N/A"
        target_raw = info.get('targetMeanPrice')
        analyst_target = round(target_raw, 2) if isinstance(target_raw, (int, float)) else "N/A"

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

        return {
            "sector": sector, "industry": industry,
            "pe": pe, "fpe": fpe, "analyst_target": analyst_target,
            "macro_trend": macro_trend, "current_price": current_close,
            "last_5_days": last_5_days, "atr": atr_14, "volume_pct": vol_pct,
            "support_1": support_1, "support_2": swing_low,
            "resistance_1": resistance_1, "resistance_2": swing_high,
            "fib_1618": fib_1618, "fib_0786": fib_0786, "fib_0618": fib_0618, "fib_0382": fib_0382,
            "rsi": round(rsi_val, 2), "ema_200": ema_200, "macd": macd_status, "graph_trend": trend
        }
    except Exception as e:
        print(f"Error fetching profile for {ticker}: {e}")
        return {"current_price": 0.0}