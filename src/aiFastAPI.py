from fastapi import FastAPI
from pydantic import BaseModel
import aimain  

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str = "AAPL"  # Default value
    horizon: str = "Mid-term"  # Default value

@app.post("/analyze")
async def analyze_stock(req: StockRequest):
    # เรียกใช้ฟังก์ชันหลักตัวเดียวกับที่ใช้ใน Terminal
    report = aimain.run_pipeline(req.ticker, req.horizon)
    
    # FastAPI จะแปลง dict เป็น JSON ให้ frontend อัตโนมัติ
    return report

# For running: 
# cd src 
# uvicorn aiFastAPI:app