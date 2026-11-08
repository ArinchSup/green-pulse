import random
import datetime
import time
import json
import pandas as pd
import yfinance as yf

# 🌟 1. Fix the Seed for Determinism
random.seed(42)

# 🌟 2. Import All Three Pillars
from class_ai_pillar2 import run_pillar2_technical_quant
from class_ai_pillar1_news import get_news_catalyst_score
from class_ai_pillar1_funda_helper import give_me_foundation

SEARCH_ATTEMPTS = 200
MAX_FORWARD_DAYS = 90
OUTPUT_FILE = "three_pillar_backtest_log.json"

SHAY_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI", "CRDO"
]

def check_macro_regime(target_date_str, index_ticker="SPY", sma_period=200):
    """
    Checks if the broader market is in a bullish regime.
    Returns True if SPY's Close is strictly greater than its 200-day SMA.
    """
    target_dt = pd.to_datetime(target_date_str)
    # Pull ~320 calendar days to guarantee we have 200 trading days of data
    start_dt = target_dt - datetime.timedelta(days=320)
    end_dt = target_dt + datetime.timedelta(days=1)
    
    try:
        spy_df = yf.Ticker(index_ticker).history(start=start_dt, end=end_dt)
        if len(spy_df) < sma_period:
            return True  # Fallback to approve if Yahoo Finance data drops
            
        spy_df['SMA_200'] = spy_df['Close'].rolling(window=sma_period).mean()
        
        latest_close = spy_df['Close'].iloc[-1]
        latest_sma = spy_df['SMA_200'].iloc[-1]
        
        return latest_close > latest_sma
    except Exception:
        return True # Fallback to approve on network error
    
def calculate_composite_fundamental(ticker):
    """Calculates an aggregate 0-10 score from your friend's engine."""
    try:
        data = give_me_foundation(ticker)
        p = data.get("profitability")
        g = data.get("growth")
        f = data.get("financial_health")
        
        valid = [x for x in (p, g, f) if x is not None]
        if not valid:
            return 5.0  # Neutral default if data unavailable
        return round(sum(valid) / len(valid), 1)
    except:
        return 5.0

def simulate_forward_price(ticker, start_date_str, target_price, stop_loss):
    start_dt = pd.to_datetime(start_date_str)
    end_dt = start_dt + datetime.timedelta(days=MAX_FORWARD_DAYS)
    try:
        df_future = yf.Ticker(ticker).history(start=start_dt + datetime.timedelta(days=1), end=end_dt)
        if df_future.empty: return "Expired"
        for _, row in df_future.iterrows():
            high, low = row['High'], row['Low']
            if high >= target_price and low > stop_loss: return "Win"
            if low <= stop_loss and high < target_price: return "Loss"
            if low <= stop_loss and high >= target_price: return "Loss"
        return "Expired"
    except:
        return "Expired"

def run_triad_backtest():
    print("="*60)
    print("🚀 RUNNING 4-PILLAR INTEGRATED BACKTEST (SEED: 42 + MACRO)")
    print("="*60)

    stats = {
        "Standalone_XGB": {"Win": 0, "Loss": 0, "Expired": 0},
        "Filtered_Triad": {"Win": 0, "Loss": 0, "Expired": 0}
    }
    
    macro_vetoes = 0
    funda_vetoes = 0
    news_vetoes = 0

    for i in range(SEARCH_ATTEMPTS):
        ticker = random.choice(SHAY_TICKERS)
        days_ago = random.randint(100, 360)
        target_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        target_date_str = target_date.strftime("%Y-%m-%d")

        # ---------------------------------------------------------
        # 1. PILLAR 2: Technical Breakout (XGBoost)
        # ---------------------------------------------------------
        try:
            tech = run_pillar2_technical_quant(ticker, target_date_str=target_date_str)
        except:
            continue

        if not tech or tech.get("technical_sentiment") != "Bullish":
            continue

        target = tech["actionable_levels"]["target_price"]
        stop = tech["actionable_levels"]["stop_loss"]
        outcome = simulate_forward_price(ticker, target_date_str, target, stop)
        stats["Standalone_XGB"][outcome] += 1

        print(f"\n[{i+1}/{SEARCH_ATTEMPTS}] {ticker} Breakout on {target_date_str} (Outcome: {outcome})")

        # ---------------------------------------------------------
        # 2. THE MACRO REGIME FILTER (SPY > 200 SMA)
        # ---------------------------------------------------------
        is_bull_market = check_macro_regime(target_date_str)
        if not is_bull_market:
            print(f"  🛑 MACRO VETO: SPY is below its 200-day SMA (Bear Market).")
            macro_vetoes += 1
            continue

        # ---------------------------------------------------------
        # 3. PILLAR 1.5: Fundamentals Check (Friend's Engine)
        # ---------------------------------------------------------
        fund_score = calculate_composite_fundamental(ticker)
        print(f"  📊 Fundamentals Score: {fund_score}/10")

        if fund_score < 4.5:
            print(f"  🛑 FUNDAMENTAL VETO: Poor financial health / cash burn (Score {fund_score}).")
            funda_vetoes += 1
            continue

        # ---------------------------------------------------------
        # 4. PILLAR 1: News Catalyst Check (Llama 3.1)
        # ---------------------------------------------------------
        try:
            news = get_news_catalyst_score(ticker, target_date_str=target_date_str, days_back=3)
            news_sent = news.get("catalyst_sentiment", "Neutral")
            news_score = news.get("catalyst_score", 5)
        except:
            news_sent, news_score = "Neutral", 5

        if news_sent == "Bearish" and news_score >= 5:
            print(f"  🛑 NEWS VETO: Bearish catalyst detected ({news_sent} - Score {news_score}).")
            news_vetoes += 1
            continue

        # Trade passes all 4 defenses
        print(f"  ✅ PIPELINE APPROVED: Trade executed.")
        stats["Filtered_Triad"][outcome] += 1

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------
    total_xgb = sum(stats["Standalone_XGB"].values())
    total_triad = sum(stats["Filtered_Triad"].values())

    wr_xgb = (stats["Standalone_XGB"]["Win"] / total_xgb * 100) if total_xgb else 0
    wr_triad = (stats["Filtered_Triad"]["Win"] / total_triad * 100) if total_triad else 0

    print("\n" + "="*60)
    print("📊 FULL PIPELINE INTEGRATION COMPARISON")
    print("="*60)
    print(f"Standalone XGBoost:      {total_xgb} trades | Win Rate: {wr_xgb:.2f}% (W:{stats['Standalone_XGB']['Win']} L:{stats['Standalone_XGB']['Loss']})")
    print(f"Filtered Pipeline (All): {total_triad} trades | Win Rate: {wr_triad:.2f}% (W:{stats['Filtered_Triad']['Win']} L:{stats['Filtered_Triad']['Loss']})")
    print("-" * 60)
    print(f"Trades Dropped by MACRO Regime: {macro_vetoes}")
    print(f"Trades Dropped by Fundamentals: {funda_vetoes}")
    print(f"Trades Dropped by News:         {news_vetoes}")
    print("="*60)

if __name__ == "__main__":
    run_triad_backtest()