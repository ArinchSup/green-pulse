import json
import requests

from news_fetcher import fetch_stock_profile, get_live_news_impact

OLLAMA_URL = "http://localhost:11434/api/generate"

def run_pillar1_fundamental_analyst(ticker, stock_profile, macro_trend, news_impact):
    """
    Pillar 1: The Fundamental Analyst (อัปเกรดระบบล็อคตรรกะ Deterministic)
    """
    if not news_impact or "status" in news_impact:
        news_text = "NO SIGNIFICANT FUNDAMENTAL NEWS RECENTLY. Focus your analysis ENTIRELY on the valuation metrics, financial health, and macro trends."
    else:
        news_text = json.dumps(news_impact, indent=2)

    prompt = f"""
    You are a STRICT and CONSERVATIVE Senior Fundamental Equity Analyst at a top-tier hedge fund.
    Your task is to analyze the fundamental health and near-term outlook for {ticker}.
    
    [DATA ROOM]
    1. COMPANY PROFILE & FINANCIALS:
    - Sector: {stock_profile.get('sector', 'N/A')} / Industry: {stock_profile.get('industry', 'N/A')}
    - Current Price: ${stock_profile.get('current_price', 'N/A')}
    - Market Cap: {stock_profile.get('market_cap', 'N/A')}
    - Trailing P/E: {stock_profile.get('pe', 'N/A')} | Forward P/E: {stock_profile.get('fpe', 'N/A')}
    - Revenue Growth: {stock_profile.get('revenue_growth', 'N/A')}
    - Profit Margin: {stock_profile.get('profit_margin', 'N/A')}
    - ROE: {stock_profile.get('roe', 'N/A')}
    - Debt to Equity: {stock_profile.get('debt_to_equity', 'N/A')}
    - Free Cash Flow: {stock_profile.get('free_cashflow', 'N/A')}
    - Analyst Target Price: ${stock_profile.get('analyst_target', 'N/A')}
    - Macro Environment: {macro_trend}

    2. RECENT NEWS IMPACT:
    {news_text}

    [FUNDAMENTAL SCORING RUBRIC (STRICTLY ENFORCED)]
    To prevent bipolar analysis, you MUST follow this logic hierarchy:
    1. PROFITABILITY IS KING: Negative ROE, negative Profit Margin, or missing Free Cash Flow must be heavily penalized.
    2. VALUATION IN CONTEXT (DO NOT OVER-INDEX ON P/E): A high Forward P/E is ACCEPTABLE if Revenue Growth and ROE are exceptionally strong (Growth Stock). A low P/E is a "Value Trap" if growth is negative. Weigh valuation holistically against growth and macro conditions.
    3. IGNORE ANALYST HYPE: Do not justify a "Bullish" rating solely based on the 'Analyst Target Price' if the core financials are weak.

    [INSTRUCTION]
    Evaluate the fundamental conviction for {ticker} based strictly on the Data Room and the Scoring Rubric.
    
    Output EXACTLY in this JSON format:
    {{
        "fundamental_sentiment": "Bullish, Bearish, or Neutral",
        "conviction_score_1_to_10": <integer>,
        "key_rationale": "A punchy 3-sentence explanation justifying your sentiment and score. Ensure you balance growth, profitability, and valuation, rather than relying solely on the P/E ratio."
    }}
    """
    try:
        payload = {
            "model": "llama3.1", 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        }
        res = requests.post(OLLAMA_URL, json=payload).json()
        return json.loads(res["response"])
    except Exception as e:
        print(f"Pillar 1 Error: {e}")
        return None

# ==========================================
# Test Run
# ==========================================
if __name__ == "__main__":
    test_ticker = "META"
    print(f"🕵️‍♂️ [Pillar 1] กำลังเริ่มวิเคราะห์ปัจจัยพื้นฐานสำหรับ: {test_ticker}")
    
    # 1. ดึงข้อมูลพื้นฐาน (Tool)
    print("📊 1. Fetching Stock Profile...")
    profile_data = fetch_stock_profile(test_ticker)
    print(profile_data)
    
    # 2. ดึงข่าวและวิเคราะห์ Impact (Tool -> Agent 2)
    print("📰 2. Fetching and Analyzing News...")
    news_data = get_live_news_impact(test_ticker)
    
    # 3. ให้ Pillar 1 สรุปผลลัพธ์สุดท้าย (Agent 1)
    print("🧠 3. Running Fundamental Synthesis...")
    
    # ส่งท่อข้อมูลเฉพาะ Fundamental ไปให้ AI
    if 'fundamental' in profile_data and 'technical' in profile_data:
        final_analysis = run_pillar1_fundamental_analyst(test_ticker, profile_data['fundamental'], profile_data['technical']['macro_trend'], news_data)
        print("\n✅ === FINAL FUNDAMENTAL ANALYSIS ===")
        print(json.dumps(final_analysis, indent=4, ensure_ascii=False))
    else:
        print("❌ Error: ไม่พบ Key 'fundamental' หรือ 'technical' กรุณาอัปเดตฟังก์ชัน fetch_stock_profile ใน news_fetcher.py")