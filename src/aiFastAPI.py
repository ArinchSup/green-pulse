from fastapi import FastAPI
from pydantic import BaseModel
import aimain  

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str = "AAPL"  # Default value
    horizon: str = "Mid-term"  # Default value

@app.post("/analyze")
async def analyze_stock(req: StockRequest):
    report = aimain.run_pipeline(req.ticker, req.horizon)
    return report

# For running: 
# cd src 
# uvicorn aiFastAPI:app