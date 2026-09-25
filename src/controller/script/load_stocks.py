"""Load five years of daily prices from Yahoo Finance into app.stocks.

Demo tool: run from a laptop, never from the server. Safe to rerun.
Usage:   python script/load_stocks.py AAPL MSFT ...
Needs:   DATABASE_URL in the environment (the same value the Go code uses).
"""
import os
import sys

import psycopg2
import yfinance as yf
from psycopg2.extras import execute_values

INSERT_SQL = """
    INSERT INTO app.stocks
        (symbol, date, open, high, low, close, volume, dividends, stock_splits)
    VALUES %s
    ON CONFLICT (symbol, date) DO NOTHING
"""


def rows_for(symbol):
    hist = yf.Ticker(symbol).history(period="5y")
    if hist.empty:
        raise ValueError(f"no data for {symbol}")
    hist = hist.reset_index()
    return [
        (
            symbol,
            r["Date"].date(),
            float(r["Open"]),
            float(r["High"]),
            float(r["Low"]),
            float(r["Close"]),
            int(r["Volume"]),
            float(r["Dividends"]),
            float(r["Stock Splits"]),
        )
        for _, r in hist.iterrows()
    ]


def main():
    symbols = [s.upper() for s in sys.argv[1:]]
    if not symbols:
        sys.exit("usage: python script/load_stocks.py AAPL MSFT ...")

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        sys.exit("DATABASE_URL is not set")

    conn = psycopg2.connect(db_url)
    try:
        for symbol in symbols:
            rows = rows_for(symbol)
            with conn, conn.cursor() as cur:
                execute_values(cur, INSERT_SQL, rows, page_size=len(rows))
                print(f"{symbol}: {len(rows)} rows from Yahoo, {cur.rowcount} new")
    finally:
        conn.close()


if __name__ == "__main__":
    main()