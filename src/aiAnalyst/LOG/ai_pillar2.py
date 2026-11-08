import json
import requests

# 🌟 นำเข้าฟังก์ชันดึงข้อมูลจาก news_fetcher ที่อัปเดตแล้ว
from news_fetcher import fetch_stock_profile

OLLAMA_URL = "http://localhost:11434/api/generate"

# ==========================================
# ⚙️ SELECT YOUR TRADING HORIZON
# ==========================================
ACTIVE_HORIZON = "MID"  # Options: "SHORT", "MID", "LONG"
TURN_ON_PYTHON_FILTER = True

HORIZON_PROFILES = {
    "SHORT": {
        "model": "technical-short-v46", # Change this when your Short model is trained
        "context": "Short-Term (1-2 weeks, using an Hourly 1H Chart)"
    },
    "MID": {
        "model": "technical-mid-v5535",               # Your current Mid-Term model
        "context": "Mid-Term (1-3 months, using a Daily 1D Chart)"
    },
    "LONG": {
        "model": "technical-long-v43",  # Change this when your Long model is trained
        "context": "Long-Term (3-12 months, using a Daily 1D Chart)"
    }
}

# ==========================================
# 🛠️ MODULE 1: Data Preparation
# ==========================================
def get_live_technical_snapshot(ticker, target_date_str=None):
    try:
        profile = fetch_stock_profile(ticker, target_date_str)
        if profile and 'technical' in profile:
            return profile['technical']
        return None
    except Exception as e:
        print(f"Error fetching technical snapshot: {e}")
        return None

# ==========================================
# 🧠 MODULE 2: AI Inference (Calling Ollama)
# ==========================================
def run_pillar2_technical_quant(ticker, target_date_str=None):
    live_tech_data = get_live_technical_snapshot(ticker, target_date_str)
    
    # print(live_tech_data)
    
    if not live_tech_data:
        return None
        
    profile = HORIZON_PROFILES[ACTIVE_HORIZON]
        
    # 🌟 1. SIMPLIFIED PROMPT: No Math, No Numbers. Just Text.
    # This completely eliminates JSON formatting errors.
    prompt = f"""You are an elite Chartered Market Technician (CMT) and Quant Strategist for a top-tier hedge fund.
Below is an instruction that describes a task, paired with an input that provides technical data. Write a highly analytical response that appropriately completes the request.
CRITICAL RULE: You MUST output ONLY a valid JSON object. Do not include markdown blocks (like ```json), greetings, or comments.

### Instruction:
Analyze the following technical snapshot for {ticker}. Provide a {profile['context']} trading strategy. 

Your default stance is to PROTECT CAPITAL. You are highly skeptical of the market. If the volume trend is distribution, if the price is under the 200 EMA, or if the MACD is bearish, you MUST output 'Avoid / No Trade'. You only approve a 'Bullish' setup if the indicators align perfectly.

Provide your response in EXACTLY this JSON structure:
{{
  "rationale": "<Write your technical explanation here first. Evaluate trend, volume, and momentum. Conclude if it is safe to trade or if we must avoid.>",
  "technical_sentiment": "<Bullish or Bearish>",
  "trading_setup_type": "<e.g., Momentum Breakout, Support Bounce, Avoid / No Trade>"
}}

### Input:
{json.dumps(live_tech_data, indent=2)}

### Response:
"""
    try:
        payload = {
            "model": profile["model"], 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=60).json()
        ai_output = json.loads(res["response"])
        
        # 🌟 2. THE QUANT OVERRIDE: Python handles all the math
        sentiment = ai_output.get("technical_sentiment", "")
        
        if sentiment == "Bullish":
            curr_p = float(live_tech_data.get('current_price', 0))
            atr = float(live_tech_data.get('atr_14', 0))
            res_1 = float(live_tech_data['key_levels']['resistance_1'])
            res_2 = float(live_tech_data['key_levels']['resistance_2_swing_high'])
            
            if ACTIVE_HORIZON == "SHORT":
                atr_stop_mult = 3.0
                atr_target_mult = 3.5
            else: # MID
                atr_stop_mult = 2.5
                atr_target_mult = 4.0
                
            stop_p = round(curr_p - (atr * atr_stop_mult), 2)
            default_target = round(curr_p + (atr * atr_target_mult), 2)
            
            risk = abs(curr_p - stop_p)
            reward = abs(default_target - curr_p)
            stop_drop_pct = risk / curr_p if curr_p > 0 else 1.0
            
            # Change this line in your Quant Override:
            max_stop = 0.06 if ACTIVE_HORIZON == "SHORT" else 0.15  # 🌟 Eased from 0.10 to 0.15
            
            if TURN_ON_PYTHON_FILTER:
            # 🌟 STRICT SKIP GATEKEEPER
                if risk <= 0 or (reward / risk) < 1.5 or stop_drop_pct > max_stop:
                    ai_output["technical_sentiment"] = "Bearish"
                    ai_output["trading_setup_type"] = "Avoid (Python Override: Bad Math or High Volatility)"
                    ai_output["actionable_levels"] = {"entry_price": 0.0, "target_price": 0.0, "stop_loss": 0.0}
                else:
                    ai_output["actionable_levels"] = {
                        "entry_price": curr_p,
                        "target_price": default_target,
                        "stop_loss": stop_p
                    }
                    ai_output["risk_reward_ratio"] = f"1:{round(reward/risk, 1)}"
            
            else:
                ai_output["actionable_levels"] = {
                        "entry_price": curr_p,
                        "target_price": default_target,
                        "stop_loss": stop_p
                    }
                ai_output["risk_reward_ratio"] = f"1:{round(reward/risk, 1)}"
                
        else:
            ai_output["actionable_levels"] = {"entry_price": 0.0, "target_price": 0.0, "stop_loss": 0.0}
            
        return ai_output

    except requests.exceptions.Timeout:
        print(f"  ⚠️ Timeout Error: Ollama took longer than 60 seconds.")
        return None
    except Exception as e:
        print(f"  ⚠️ Pillar 2 Error: {e}")
        return None
    
# ==========================================
# 🚀 ทดสอบระบบ (Master Execution)
# ==========================================
if __name__ == "__main__":
    test_ticker = "META"
    print(f"🕵️‍♂️ [Pillar 2] กำลังเริ่มวิเคราะห์กราฟเทคนิคอลสำหรับ: {test_ticker}")
    print(f"   ► Horizon Setting: {ACTIVE_HORIZON} ({HORIZON_PROFILES[ACTIVE_HORIZON]['model']})")
    
    print("\n📊 1. Fetching Live Technical Snapshot from news_fetcher...")
    quant_analysis = run_pillar2_technical_quant(test_ticker)
    
    if quant_analysis:
        print(f"\n🧠 2. Running {HORIZON_PROFILES[ACTIVE_HORIZON]['model']} Quant Model...")
        print("\n✅ === FINAL QUANT SETUP ===")
        print(json.dumps(quant_analysis, indent=4, ensure_ascii=False))
    else:
        print("\n❌ Failed to generate quant analysis.")