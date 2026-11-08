import random
import datetime
import yfinance as yf
import pandas as pd
import json
import requests

# 🌟 Import the AI function AND the active horizon setting!
from ai_pillar2 import run_pillar2_technical_quant, ACTIVE_HORIZON

TARGET_AMOUNT = 1000
OUTPUT_FILE = f"pillar2_backtest_results_1000_{ACTIVE_HORIZON.lower()}.json"

# ==========================================
# ⚙️ DYNAMIC BACKTEST CONFIGURATIONS
# ==========================================
BACKTEST_CONFIGS = {
    "SHORT": {
        "eval_days": 14,           # 2 weeks future testing window
        "interval": "1h",          # Intraday precision for tight targets
        "max_lookback": 400        # yfinance 1h data limit is 730 days
    },
    "MID": {
        "eval_days": 90,           # 3 months future testing window
        "interval": "1d",          # Daily precision
        "max_lookback": 400      # Plenty of daily history
    },
    "LONG": {
        "eval_days": 365,          # 1 year future testing window
        "interval": "1d",          # Daily precision
        "max_lookback": 2000       # Plenty of daily history
    }
}

SHAY_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CRWV", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI"
]

DEFENSIVE_TICKERS = [
    "JPM", "BAC", "V", "MA", "BRK-B", "GS",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE",
    "WMT", "PG", "KO", "PEP", "COST",
    "XOM", "CVX", "CAT", "GE", "LMT", "BA"
]

def run_backtest():
    config = BACKTEST_CONFIGS[ACTIVE_HORIZON]
    eval_days = config["eval_days"]
    interval = config["interval"]
    max_lookback = config["max_lookback"]
    
    print(f"🚀 Starting Backtest for {TARGET_AMOUNT} iterations...")
    print(f"   ► Testing Horizon: {ACTIVE_HORIZON} | Window: {eval_days} days | Interval: {interval}\n")
    
    all_tickers = SHAY_TICKERS 
    
    total_signals = 0
    wins = 0
    losses = 0
    misses = 0
    skips = 0
    open_trades = 0 

    for i in range(TARGET_AMOUNT):
        ticker = random.choice(all_tickers)
        
        # 🌟 DYNAMIC DATE: Ensures we don't ask for a date that exceeds our eval window
        days_ago = random.randint(eval_days, max_lookback)
        target_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        print(f"[{i+1}/{TARGET_AMOUNT}] Evaluating {ticker} on {target_date_str}...")
        
        # Fetch the AI Prediction
        try:
            ai_output = run_pillar2_technical_quant(ticker, target_date_str)
        except Exception as e:
            print(f"  ⚠️ Error running AI for {ticker}: {e}")
            continue
            
        # 🌟 NEW FIX: Catch empty data/None immediately!
        if not ai_output:
            print(f"  ⏭️ Skipped: No valid technical data or AI output for {ticker}.")
            skips += 1
            continue

        # === ก่อนหน้านี้ที่มีการเช็ค skip_keywords ===
        setup_type = ai_output.get("trading_setup_type", "")
        sentiment = ai_output.get("technical_sentiment", "")
        
        skip_keywords = ["Avoid", "Wait", "No Trade", "Neutral"]
        if sentiment == "Bearish" or any(kw in setup_type for kw in skip_keywords):
            print(f"  ⏭️ Skipped: AI suggested {setup_type} ({sentiment}).")
            skips += 1
            continue

        # 🌟 จุดที่ต้องแก้เพิ่ม: ตัวป้องกันการพังจาก Data Type ผิดพลาด (String vs Float)
        try:
            levels = ai_output.get("actionable_levels", {})
            
            # ดึงค่าออกมาพร้อมตรวจสอบและแปลงประเภทข้อมูลให้เป็น float ทันที
            entry_p = levels.get("entry_price")
            target_p = levels.get("target_price")
            stop_p = levels.get("stop_loss")
            
            # บังคับแปลงกรณีหลุดมาเป็น String หรือค่าว่าง
            entry_p = float(entry_p) if entry_p not in ["", None] else 0.0
            target_p = float(target_p) if target_p not in ["", None] else 0.0
            stop_p = float(stop_p) if stop_p not in ["", None] else 0.0
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"  ⚠️ Skipped: Actionable levels format error for {ticker}: {e}")
            skips += 1
            continue

        # ตรวจสอบความสมเหตุสมผลของราคาเบื้องต้น ป้องกันการหารด้วยศูนย์หรือราคาติดลบ
        if entry_p <= 0 or target_p <= 0 or stop_p <= 0:
            print(f"  ⏭️ Skipped: Invalid price level data (Prices must be greater than 0)")
            skips += 1
            continue

        # === หลังจากนี้เป็นส่วนเริ่มคำนวณราคาของลูป Backtest ปกติ ===
        total_signals += 1
        print(f"  🎯 Setup Found -> Entry: ${entry_p:.2f} | Target: ${target_p:.2f} | Stop: ${stop_p:.2f}")
        
        # 🌟 DYNAMIC FETCH: Pulls exact window and exact timeframe interval (1H or 1D)
        end_date = target_date + datetime.timedelta(days=eval_days)
        future_df = yf.Ticker(ticker).history(start=target_date, end=end_date, interval=interval)

        if future_df.empty:
            print("  ⚠️ Skipped: No future market data available.")
            skips += 1
            total_signals -= 1 # Rollback counter if data is missing
            continue

        entered = False
        trade_result = "OPEN" 
        
        for date, row in future_df.iterrows():
            low = row['Low']
            high = row['High']

            if not entered:
                if low <= entry_p:
                    entered = True
                    if high >= target_p:
                        trade_result = "WIN"
                        break
                    elif low <= stop_p:
                        trade_result = "LOSS"
                        break
            else:
                if high >= target_p:
                    trade_result = "WIN"
                    break
                elif low <= stop_p:
                    trade_result = "LOSS"
                    break
        
        winloss = " "
        if not entered:
            misses += 1
            print(f"  📉 Result: MISS (Price never dropped to entry of ${entry_p})")
            winloss = "MISS"
        elif trade_result == "WIN":
            wins += 1
            print(f"  ✅ Result: WIN (Hit Target ${target_p})")
            winloss = "WIN"
        elif trade_result == "LOSS":
            losses += 1
            print(f"  ❌ Result: LOSS (Hit Stop ${stop_p})")
            winloss = "LOSS"
        else:
            open_trades += 1
            # 🌟 Dynamic Expiration String
            print(f"  ⏳ Result: EXPIRED ({eval_days} days passed, neither Target nor Stop was hit)")
            winloss = "EXPIRED"
        
        final_result = {winloss: ai_output}
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                json.dump(final_result, f, indent=4, ensure_ascii=False)
                f.write("\n") # Adds a newline so the file is easily readable

    # --- Print Final Statistics ---
    print("\n" + "="*40)
    print(f"📊 BACKTEST RESULTS ({ACTIVE_HORIZON})")
    print("="*40)
    print(f"Total Iterations:   {TARGET_AMOUNT}")
    print(f"Valid Trade Setups: {total_signals}")
    print(f"Trades Triggered:   {wins + losses + open_trades}")
    print("-" * 40)
    print(f"Wins:               {wins}")
    print(f"Losses:             {losses}")
    print(f"Expired (No hit):   {open_trades}")
    print(f"Missed (No entry):  {misses}")
    print(f"Skipped (No Trade): {skips}")
    
    closed_trades = wins + losses
    if closed_trades > 0:
        win_rate = (wins / closed_trades) * 100
        print(f"\n🎯 WIN RATE: {win_rate:.2f}% (Out of {closed_trades} closed trades)")
    else:
        print("\n🎯 WIN RATE: N/A (No trades closed)")
    print("="*40)

if __name__ == "__main__":
    run_backtest()