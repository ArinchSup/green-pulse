from fastapi import FastAPI
from pydantic import BaseModel
from rpds import List
from sympy import Dict
import aimain  
import ai_chatbot

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str = "AAPL"  # Default value
    horizon: str = "Mid-term"  # Default value
    
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []

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

# For running: 
# cd src 
# uvicorn aiFastAPI:app