import yfinance as yf
import sys
import psycopg2
from psycopg2.extras import execute_values
from urllib.parse import quote

def fetch_and_push(symbol, db_url, db_password, region):
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max")
    if hist.empty:
        print(f"No data found for {symbol}")
        sys.exit(2)

    hist = hist.reset_index()
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist['symbol'] = symbol
    data_to_insert = hist[[
        'symbol', 'Date', 'Open', 'High', 'Low', 'Close', 
        'Volume', 'Dividends', 'Stock Splits'
    ]].values.tolist()
    encoded_password = quote(db_password, safe='')
    conn_str = f"postgres://postgres.apbkobhfnmcqqzqeeqss:{encoded_password}@aws-0-{region}.pooler.supabase.com:5432/postgres"
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()

        query = """
            INSERT INTO stocks (symbol, date, open, high, low, close, volume, dividends, stock_splits)
            VALUES %s
            ON CONFLICT (symbol, date) DO NOTHING;
        """

        execute_values(cur, query, data_to_insert)
        conn.commit()
        print(f"SUCCESS:{symbol}:{len(data_to_insert)}_rows")
        
    except Exception as e:
        print(f"DATABASE_ERROR:{e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: fetch.py <SYMBOL> <DB_URL> <DB_PASSWORD> <REGION>", file=sys.stderr)
        sys.exit(1)
        
    fetch_and_push(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])