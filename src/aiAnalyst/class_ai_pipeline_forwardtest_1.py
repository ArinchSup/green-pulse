import sqlite3
import datetime
import yfinance as yf
from class_ai_pillar2 import run_pillar2_technical_quant

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DB_FILE = "forward_test.db"
MAX_LOSS_PCT = -0.125  # -12.5% hard stop cap

SHAY_TICKERS = [
    "MSTR", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL",
    "MRVL", "ALAB", "OSS", "AEHR", "COHR", "LITE", "AAOI",
    "DOCN", "ZS", "NET", "PANW", "CRWD", "SYM", "ISRG", "PATH", "MDB", "SNOW",
    "PLTR", "UMAC", "ONDS", "INTC", "ASML", "TSM", "RDW",
    "BKSY", "ASTS", "RKLB", "CEG", "BE", "UAMY", "FCX", "IDR",
    "CRML", "MP", "WULF", "CIFR", "NBIS", "IREN", "LEU", "GEV", "UUUU",
    "OKLO", "APLD", "AVGO", "RDDT", "MU", "ORCL", "LLY", "OSCR",
    "DUOL", "PAYX", "SOFI", "CRDO", "CLFD", "INFQ", "IONQ", "ZETA", 
]

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            target_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            halfway_target REAL NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,       -- 'OPEN', 'WIN', 'LOSS', 'BREAKEVEN', 'EXPIRED'
            stop_at_breakeven INTEGER DEFAULT 0,
            exit_date TEXT,
            exit_price REAL,
            pnl_pct REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def is_already_open(ticker):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM trades WHERE ticker = ? AND status = 'OPEN'", (ticker,))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def log_trade(ticker, signal_date, entry, target, stop, confidence):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Calculate -12.5% hard stop and halfway target
    hard_stop = round(entry * (1.0 + MAX_LOSS_PCT), 2)
    effective_stop = max(stop, hard_stop)
    halfway_target = round(entry + ((target - entry) * 0.5), 2)
    
    cursor.execute("""
        INSERT INTO trades (ticker, signal_date, entry_price, target_price, stop_loss, halfway_target, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (ticker, signal_date, entry, target, effective_stop, halfway_target, confidence))
    
    conn.commit()
    conn.close()

def run_scanner():
    init_db()
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    print("=" * 65)
    print(f"🚀 RUNNING FORWARD SCANNER ({today_str})")
    print("=" * 65)

    triggered_count = 0

    for ticker in SHAY_TICKERS:
        if is_already_open(ticker):
            print(f"⏩ {ticker:<5} | Existing active position in DB. Skipping duplicate.")
            continue

        try:
            tech = run_pillar2_technical_quant(ticker, target_date_str=today_str)
        except Exception as e:
            print(f"⚠️ {ticker:<5} | Error running technical quant: {e}")
            continue

        if not tech or tech.get("technical_sentiment") != "Bullish":
            continue

        conf = tech.get("confidence", 0.0)
        target = tech["actionable_levels"]["target_price"]
        stop = tech["actionable_levels"]["stop_loss"]
        
        # Pull the last finalized daily candle
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist.empty:
            continue
        
        # 🌟 Pull the exact date and close price from the last settled candle
        last_candle_date = hist.index[-1].strftime("%Y-%m-%d")
        entry_price = float(hist["Close"].iloc[-1])

        log_trade(ticker, last_candle_date, entry_price, target, stop, conf)
        triggered_count += 1
        print(f"✅ EXECUTED: {ticker:<5} | Date: {last_candle_date} | Entry: ${entry_price:.2f} | TP: ${target:.2f} | SL: ${stop:.2f} | Conf: {conf:.2f}")

    print("-" * 65)
    print(f"Scan complete. {triggered_count} new trade(s) logged to '{DB_FILE}'.\n")

if __name__ == "__main__":
    run_scanner()