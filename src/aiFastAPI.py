from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import aimain  
import ai_chatbot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class StockRequest(BaseModel):
    ticker: str = "AAPL"  # Default value
    horizon: str = "Mid-term"  # Default value
    
class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []

@app.post("/analyze")
async def analyze_stock(req: StockRequest):
    report = aimain.run_pipeline(req.ticker, req.horizon)
    return report

@app.post("/chat")
async def chat_with_bot(req: ChatRequest):
    reply, new_history = ai_chatbot.ask_stock_bot(req.question, req.history)
    return {
        "response": reply,
        "history": new_history
    }
    
# For graph ui parts 
class ChartRequest(BaseModel):
    ticker: str
    period: str

@app.post("/chart")
async def get_chart_data(req: ChartRequest):
    real_ticker = "^GSPC" if req.ticker in ["SPX", "S&P 500"] else req.ticker
    
    yf_params = {
        "1D": {"period": "1d", "interval": "5m"},
        "1W": {"period": "5d", "interval": "15m"},
        "1M": {"period": "1mo", "interval": "1d"},
        "3M": {"period": "3mo", "interval": "1d"},
        "1Y": {"period": "1y", "interval": "1d"},
        "5Y": {"period": "5y", "interval": "1wk"},
    }
    
    params = yf_params.get(req.period, {"period": "1mo", "interval": "1d"})
    
    try:
        import yfinance as yf
        stock = yf.Ticker(real_ticker)
        hist = stock.history(**params)
        
        if hist.empty:
            return {"data": []}
            
        chart_data = []
        for i, (index, row) in enumerate(hist.iterrows()):
            chart_data.append({
                "i": i,
                "t": index.strftime("%Y-%m-%d %H:%M"),
                "value": round(row["Close"], 2)
            })
            
        return {"data": chart_data}
    except Exception as e:
        return {"error": str(e), "data": []}

# For running: 
# cd src 
# uvicorn aiFastAPI:app