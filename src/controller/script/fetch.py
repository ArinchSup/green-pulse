import yfinance as yf
import sys
import os

def fetch(symbol: str):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max")
    hist.index = hist.index.strftime("%Y-%m-%d")
    hist.index.name = "date"
    hist.columns = ["open", "high", "low", "close", "volume", "dividends", "stock_splits"]

    os.makedirs("./data/raw", exist_ok=True)
    path = f"./data/raw/{symbol.upper()}.csv"
    hist.to_csv(path)
    print(f"save:{path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage; fetch.py <SYMBOL>", file=sys.stderr)
        sys.exist(1)
    fetch(sys.argv[1])