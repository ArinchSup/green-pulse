import random
import datetime
import yfinance as yf
import pandas as pd
import json
import time

# Import your upgraded Pillar 1 LLM engine
from class_ai_pillar1_news import get_news_catalyst_score

TARGET_AMOUNT = 20  # Start small because LLM inference is slow
OUTPUT_FILE = "pillar1_news_backtest_results.json"

SHAY_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "AEHR", "COHR", "LITE", "CRWD", "PLTR", "ASTS", "RKLB", 
    "CEG", "WULF", "IREN", "AVGO", "MU", "ORCL", "LLY"
]

def run_pillar1_backtest():
    print(f"🚀 Starting Pillar 1 (LLM Catalyst) Backtest for {TARGET_AMOUNT} iterations...")
    print("   ► Note: Finnhub free tier limits news history to 365 days.")
    
    total_bullish, total_bearish, total_neutral = 0, 0, 0
    bullish_wins = 0  # Count of Bullish setups that were profitable 30 days later

    for i in range(TARGET_AMOUNT):
        ticker = random.choice(SHAY_TICKERS)
        
        # Pick a random date between 100 and 360 days ago
        days_ago = random.randint(100, 360)
        target_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        print(f"\n[{i+1}/{TARGET_AMOUNT}] Testing {ticker} on {target_date_str}...")
        
        # 1. Have the LLM grade the historical news
        try:
            catalyst_analysis = get_news_catalyst_score(ticker, target_date_str=target_date_str, days_back=3)
        except Exception as e:
            print(f"  ⚠️ Error running AI: {e}")
            continue
            
        sentiment = catalyst_analysis.get("catalyst_sentiment", "Neutral")
        score = catalyst_analysis.get("catalyst_score", 0)
        
        if sentiment == "Neutral" or score == 0:
            print("  ⏭️ Skipped: No major catalyst found (Neutral).")
            total_neutral += 1
            continue
            
        # 2. Fetch future price data to verify the LLM's prediction
        # Get ~100 calendar days of future data to easily calculate 30D and 90D returns
        end_date = target_date + datetime.timedelta(days=100)
        df_future = yf.Ticker(ticker).history(start=target_date, end=end_date)
        
        if len(df_future) < 20:
            print("  ⚠️ Skipped: Not enough future price data found.")
            continue
            
        # Base price right after the news dropped
        entry_price = df_future['Close'].iloc[0]
        
        # Forward Returns (Approx 21 trading days in a month, 63 in 3 months)
        forward_30d_price = df_future['Close'].iloc[min(21, len(df_future)-1)]
        forward_90d_price = df_future['Close'].iloc[min(63, len(df_future)-1)]
        
        return_30d = ((forward_30d_price - entry_price) / entry_price) * 100
        return_90d = ((forward_90d_price - entry_price) / entry_price) * 100

        print(f"  🧠 LLM Verdict: {sentiment} (Score: {score})")
        print(f"  📈 Forward Returns -> 30-Day: {return_30d:.2f}% | 90-Day: {return_90d:.2f}%")

        if sentiment == "Bullish":
            total_bullish += 1
            if return_30d > 0: bullish_wins += 1
        elif sentiment == "Bearish":
            total_bearish += 1

        # Save to JSON log
        log_entry = {
            "ticker": ticker,
            "date": target_date_str,
            "llm_analysis": catalyst_analysis,
            "forward_returns": {
                "30_day_pct": round(return_30d, 2),
                "90_day_pct": round(return_90d, 2)
            }
        }
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=4, ensure_ascii=False)
            f.write("\n")
            
        # Brief pause to respect API limits
        time.sleep(1)

    print("\n" + "="*40)
    print(f"📊 PILLAR 1 BACKTEST RESULTS")
    print("="*40)
    print(f"Total Bullish Catalysts Found: {total_bullish}")
    print(f"Total Bearish Catalysts Found: {total_bearish}")
    if total_bullish > 0:
        print(f"Bullish 30-Day Win Rate:       {(bullish_wins/total_bullish)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_pillar1_backtest()