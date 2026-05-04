import sqlite3
import datetime
import yfinance as yf
import pandas as pd
from aiAnalyst.config import DB_NAME

def check_pending_predictions():
    print("Starting Prediction Checker...")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, ticker, analysis_date, target_date, action, target_price, stop_loss 
        FROM ai_predictions 
        WHERE status = 'pending' AND action IN ('BUY', 'SELL')
    ''')
    pending_records = cursor.fetchall()
    
    if not pending_records:
        print("✅ No pending predictions to check.")
        conn.close()
        return

    updated_count = {'hit_target': 0, 'hit_stoploss': 0, 'expired': 0}

    for row in pending_records:
        record_id, ticker, analysis_date_str, target_date_str, action, target_price, stop_loss = row
        
        analysis_date = datetime.datetime.strptime(analysis_date_str, "%Y-%m-%d %H:%M:%S")
        target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.datetime.now()
        
        try:
            start_fetch = analysis_date.strftime('%Y-%m-%d')
            hist = yf.Ticker(ticker).history(start=start_fetch)
            
            if hist.empty:
                continue

            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

            valid_hist = hist[hist.index >= analysis_date]
            
            new_status = None
            
            for date_idx, day_data in valid_hist.iterrows():
                high = float(day_data['High'])
                low = float(day_data['Low'])
                
                if action == 'BUY':
                    if target_price and high >= target_price:
                        new_status = 'hit_target'
                        break
                    elif stop_loss and low <= stop_loss:
                        new_status = 'hit_stoploss'
                        break
                        
                elif action == 'SELL':
                    if target_price and low <= target_price:
                        new_status = 'hit_target'
                        break
                    elif stop_loss and high >= stop_loss:
                        new_status = 'hit_stoploss'
                        break

            if new_status is None:
                if now > target_date:
                    new_status = 'expired'
            
            if new_status:
                cursor.execute('''
                    UPDATE ai_predictions 
                    SET status = ? 
                    WHERE id = ?
                ''', (new_status, record_id))
                
                updated_count[new_status] += 1
                
                if new_status == 'hit_target':
                    print(f"  🟢 [WIN] {ticker} ({action}) hit Target Price {target_price}!")
                elif new_status == 'hit_stoploss':
                    print(f"  🔴 [LOSS] {ticker} ({action}) hit Stop Loss {stop_loss}!")
                elif new_status == 'expired':
                    print(f"  🟡 [EXPIRED] {ticker} ({action}) timeframe ended without hitting limits.")

        except Exception as e:
            print(f"  ⚠️ Error checking {ticker}: {e}")
            
    conn.commit()
    conn.close()
    
    print("\n📊 Check Summary:")
    print(f"   Targets Hit : {updated_count['hit_target']}")
    print(f"   Stop Losses : {updated_count['hit_stoploss']}")
    print(f"   Expired     : {updated_count['expired']}")
    print("✅ Checker finished.")

if __name__ == "__main__":
    check_pending_predictions()