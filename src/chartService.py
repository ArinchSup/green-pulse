from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
import yfinance as yf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChartRequest(BaseModel):
    ticker: str
    period: str

@app.post("/chart")
def get_chart_data(req: ChartRequest):
    real_ticker = "^GSPC" if req.ticker in ["SPX", "S&P 500", "SPY"] else req.ticker

    yf_params = {
        "1D": {"period": "1d",  "interval": "5m"},
        "1W": {"period": "5d",  "interval": "15m"},
        "1M": {"period": "1mo", "interval": "1d"},
        "3M": {"period": "3mo", "interval": "1d"},
        "1Y": {"period": "1y",  "interval": "1d"},
        "5Y": {"period": "5y",  "interval": "1wk"},
    }

    params = yf_params.get(req.period, {"period": "1mo", "interval": "1d"})

    try:
        hist = yf.Ticker(real_ticker).history(**params)
        if hist.empty:
            raise HTTPException(status_code=404, detail="Ticker not found or no data")
        return {
            "data": [
                {"i": i, "t": idx.strftime("%Y-%m-%d %H:%M"), "value": round(row["Close"], 2)}
                for i, (idx, row) in enumerate(hist.iterrows())
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_UP_WORDS   = {"rises", "rise", "gains", "gain", "surges", "surge", "beats", "jumps", "jump", "rally", "rallies", "higher", "up"}
_DOWN_WORDS = {"falls", "fall", "drops", "drop", "misses", "miss", "cuts", "cut", "slumps", "slump", "lower", "down", "declines", "decline"}

def _sentiment(headline: str) -> str:
    words = set(headline.lower().split())
    if words & _UP_WORDS:
        return "up"
    if words & _DOWN_WORDS:
        return "down"
    return "neutral"

class NewsRequest(BaseModel):
    ticker: str

@app.post("/news")
def get_news(req: NewsRequest):
    real_ticker = "^GSPC" if req.ticker in ["SPX", "S&P 500", "SPY"] else req.ticker
    try:
        articles = yf.Ticker(real_ticker).news or []
        result = []
        for a in articles[:10]:
            ts = a.get("providerPublishTime") or a.get("content", {}).get("pubDate")
            if isinstance(ts, int):
                t = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")
            else:
                t = "--:--"
            headline = a.get("title") or a.get("content", {}).get("title", "")
            source   = a.get("publisher") or a.get("content", {}).get("provider", {}).get("displayName", "")
            result.append({
                "time":     t,
                "source":   source,
                "tag":      req.ticker.upper(),
                "headline": headline,
                "impact":   _sentiment(headline),
            })
        return {"news": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
