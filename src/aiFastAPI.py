from fastapi import FastAPI
from pydantic import BaseModel
import aimain  
import ai_chatbot

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str = "AAPL"  # Default value
    horizon: str = "Mid-term"  # Default value

@app.post("/analyze")
async def analyze_stock(req: StockRequest):
    report = aimain.run_pipeline(req.ticker, req.horizon)
    return report

@app.post("/chat")
async def chat_with_bot(req: ai_chatbot.StockRequest):
    response = ai_chatbot.ask_stock_bot(req.question)
    return {"response": response}

# For running: 
# cd src 
# uvicorn aiFastAPI:app