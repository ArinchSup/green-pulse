import random
import datetime
import pandas as pd
import yfinance as yf
import requests
import io

random.seed(56)

from class_ai_pillar2 import run_pillar2_technical_quant
from class_ai_pillar1_funda_helper import give_me_foundation

SEARCH_ATTEMPTS = 1000
MAX_FORWARD_DAYS = 90
MIN_CONFIDENCE = 0.65  

USE_SP500_TICKERS = False  # Set to False to go back to SHAY_TICKERS

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

def get_sp500_tickers():
    """Fetches live S&P 500 symbols from Wikipedia using a disguised browser header."""
    try:
        print("Fetching latest S&P 500 tickers from Wikipedia...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        
        tables = pd.read_html(io.StringIO(response.text))
        
        tickers = tables[0]['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers] 
    except Exception as e:
        print(f"===> Failed to fetch S&P 500: {e}")
        return SHAY_TICKERS

ACTIVE_TICKERS = get_sp500_tickers() if USE_SP500_TICKERS else SHAY_TICKERS

def calculate_composite_fundamental(ticker):
    """Calculates an aggregate 0-10 score from your friend's engine."""
    try:
        data = give_me_foundation(ticker)
        p = data.get("profitability")
        g = data.get("growth")
        f = data.get("financial_health")
        
        valid = [x for x in (p, g, f) if x is not None]
        if not valid:
            return 5.0
        return round(sum(valid) / len(valid), 1)
    except:
        return 5.0

def simulate_forward_price(ticker, start_date_str, target_price, stop_loss):
    start_dt = pd.to_datetime(start_date_str)
    end_dt = start_dt + datetime.timedelta(days=MAX_FORWARD_DAYS)
    try:
        df_future = yf.Ticker(ticker).history(start=start_dt + datetime.timedelta(days=1), end=end_dt)
        if df_future.empty: return "Expired", 0.0
        
        entry_price = float(df_future['Open'].iloc[0])
        if entry_price <= 0: return "Expired", 0.0
        
        halfway_target = entry_price + ((target_price - entry_price) * 0.5)
        
        # Hardcoded Maximum Loss Cap (e.g., -12.5%)
        MAX_LOSS_PCT = -0.125 
        hard_stop_price = entry_price * (1.0 + MAX_LOSS_PCT)
        
        # Take the tighter (higher) stop between your model's stop and the hard cap
        current_stop = max(stop_loss, hard_stop_price)
        
        stop_moved_to_breakeven = False
        
        for _, row in df_future.iterrows():
            high, low = row['High'], row['Low']
            
            # 1. Trailing Stop: Lock it at entry permanently once triggered
            # (Note: Only move to breakeven if breakeven is higher than your hard stop)
            if high >= halfway_target and not stop_moved_to_breakeven:
                stop_moved_to_breakeven = True
                current_stop = entry_price

            # 2. Hit Target?
            if high >= target_price and low > current_stop: 
                ret_pct = ((target_price - entry_price) / entry_price) * 100
                return "Win", ret_pct
                
            # 3. Hit Stop / Hard Cap / Breakeven?
            if low <= current_stop and high < target_price:
                ret_pct = ((current_stop - entry_price) / entry_price) * 100
                if stop_moved_to_breakeven:
                    return "Breakeven", 0.0 # Strict 0.0% for breakeven
                else:
                    return "Loss", ret_pct
                
            # 4. Same-day extreme volatility
            if low <= current_stop and high >= target_price: 
                ret_pct = ((current_stop - entry_price) / entry_price) * 100
                if stop_moved_to_breakeven:
                    return "Breakeven", 0.0
                else:
                    return "Loss", ret_pct
                
        # Expired
        final_close = float(df_future['Close'].iloc[-1])
        ret_pct = ((final_close - entry_price) / entry_price) * 100
        return "Expired", ret_pct
    except:
        return "Expired", 0.0

def run_fast_backtest():
    print("="*60)
    print(f">> RUNNING FAST BACKTEST (Trailing Stop + Confidence Buckets) <<")
    print("="*60)

    stats = {
        "Standalone_XGB": {"Win": 0, "Loss": 0, "Expired": 0, "Breakeven": 0, "Net_Return": 0.0},
        "Filtered_Pipeline": {"Win": 0, "Loss": 0, "Expired": 0, "Breakeven": 0, "Net_Return": 0.0}
    }
    
    conf_buckets = {
        "65-70": {"Win": 0, "Loss": 0, "Breakeven": 0, "Expired": 0, "Net_Return": 0.0},
        "71-75": {"Win": 0, "Loss": 0, "Breakeven": 0, "Expired": 0, "Net_Return": 0.0},
        "76-80": {"Win": 0, "Loss": 0, "Breakeven": 0, "Expired": 0, "Net_Return": 0.0},
        "81-85": {"Win": 0, "Loss": 0, "Breakeven": 0, "Expired": 0, "Net_Return": 0.0},
        "86-90": {"Win": 0, "Loss": 0, "Breakeven": 0, "Expired": 0, "Net_Return": 0.0},
        "91-100": {"Win": 0, "Loss": 0, "Breakeven": 0, "Expired": 0, "Net_Return": 0.0}
    }
    
    funda_vetoes = 0
    conf_vetoes = 0

    for i in range(SEARCH_ATTEMPTS):
        ticker = random.choice(ACTIVE_TICKERS)
        days_ago = random.randint(100, 360)
        target_date_str = (datetime.date.today() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")

        try:
            tech = run_pillar2_technical_quant(ticker, target_date_str=target_date_str)
        except:
            continue

        if not tech or tech.get("technical_sentiment") != "Bullish":
            continue

        conf = tech.get("confidence", 0.0)
        if conf < MIN_CONFIDENCE:
            conf_vetoes += 1
            continue

        target = tech["actionable_levels"]["target_price"]
        stop = tech["actionable_levels"]["stop_loss"]
        
        outcome, ret_pct = simulate_forward_price(ticker, target_date_str, target, stop)
        
        stats["Standalone_XGB"][outcome] += 1
        stats["Standalone_XGB"]["Net_Return"] += ret_pct

        conf_pct = int(round(conf * 100))
        if conf_pct <= 70: bucket = "65-70"
        elif conf_pct <= 75: bucket = "71-75"
        elif conf_pct <= 80: bucket = "76-80"
        elif conf_pct <= 85: bucket = "81-85"
        elif conf_pct <= 90: bucket = "86-90"
        else: bucket = "91-100"
        
        conf_buckets[bucket][outcome] += 1
        conf_buckets[bucket]["Net_Return"] += ret_pct

        print(f"\n[{i+1}/{SEARCH_ATTEMPTS}] {ticker} on {target_date_str} (Conf: {conf:.2f} | {outcome} | P/L: {ret_pct:+.2f}%)")

        fund_score = calculate_composite_fundamental(ticker)
        if fund_score < 4.5:
            funda_vetoes += 1
            continue

        stats["Filtered_Pipeline"][outcome] += 1
        stats["Filtered_Pipeline"]["Net_Return"] += ret_pct

    # ---------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("FAST PIPELINE RESULTS (MULTI-METRIC & RETURN EVALUATION)")
    print("="*70)

    for name, key in [("Standalone XGBoost", "Standalone_XGB"), ("Filtered Pipeline", "Filtered_Pipeline")]:
        w = stats[key]["Win"]
        l = stats[key]["Loss"]
        be = stats[key]["Breakeven"]
        exp = stats[key]["Expired"]
        net_ret = stats[key]["Net_Return"]
        
        total_active = w + l + be
        total_resolved = w + l
        
        strict_wr = (w / total_active * 100) if total_active > 0 else 0
        resolved_wr = (w / total_resolved * 100) if total_resolved > 0 else 0
        cap_preservation = ((w + be) / total_active * 100) if total_active > 0 else 0
        avg_ret = (net_ret / (total_active + exp)) if (total_active + exp) > 0 else 0
        
        print(f"{name}: {total_active + exp} trades triggered (W:{w} | L:{l} | BE:{be} | Exp:{exp})")
        print(f"  1. Strict Win Rate        [ W / (W+L+BE) ] : {strict_wr:.2f}%")
        print(f"  2. Resolved Win Rate      [ W / (W+L) ]    : {resolved_wr:.2f}%")
        print(f"  3. Capital Preservation   [ (W+BE) / Active] : {cap_preservation:.2f}%")
        print(f"  > Net Return (Sum)       : {net_ret:+.2f}%")
        print(f"  > Average Return / Trade : {avg_ret:+.2f}%")
        print("-" * 70)
        
    print(f"Trades dropped by low confidence (<{MIN_CONFIDENCE}): {conf_vetoes}")
    print(f"Trades dropped by Fundamentals: {funda_vetoes}")
    
    print("\n" + "="*70)
    print(">> PERFORMANCE BY CONFIDENCE INTERVAL (STANDALONE XGBOOST)")
    print("="*70)
    
    for bucket in ["65-70", "71-75", "76-80", "81-85", "86-90", "91-100"]:
        b_stats = conf_buckets[bucket]
        bw, bl, bbe, bexp = b_stats["Win"], b_stats["Loss"], b_stats["Breakeven"], b_stats["Expired"]
        bnet = b_stats["Net_Return"]
        
        btotal_active = bw + bl + bbe
        btotal_resolved = bw + bl
        btotal_trades = btotal_active + bexp
        
        if btotal_trades == 0:
            continue
            
        b_resolved_wr = (bw / btotal_resolved * 100) if btotal_resolved > 0 else 0
        b_avg_ret = (bnet / btotal_trades)
        
        print(f"[{bucket}%] Trades: {btotal_trades:3d} | W:{bw:2d} L:{bl:2d} BE:{bbe:2d} Exp:{bexp:2d} | Resolved WR: {b_resolved_wr:5.2f}% | Avg Ret: {b_avg_ret:+.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_fast_backtest()