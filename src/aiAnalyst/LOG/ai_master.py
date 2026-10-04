import json

# 🌟 Imports จากไฟล์เสาหลัก (ใช้ไฟล์ที่คุณทำไว้แล้ว)
from news_fetcher import fetch_stock_profile, get_live_news_impact
from ai_pillar1 import run_pillar1_fundamental_analyst
from ai_pillar2 import get_live_technical_snapshot, run_pillar2_technical_quant
from ai_bullbear import generate_bull_case, generate_bear_case, run_horizon_judge

# ==========================================
# 🚀 MAIN MASTER EXECUTION PIPELINE
# ==========================================
def master_pipeline(ticker):
    print("="*80)
    print(f"🔥 INSTITUTIONAL INVESTMENT ENGINE INITIATED: {ticker}")
    print("="*80)
    
    # ---------------------------------------------------------
    # 🏛️ Pillar 1: Fundamental & News
    # ---------------------------------------------------------
    print("\n[1/4] Fetching Pillar 1 (Fundamentals & Macro/News)...")
    profile_data = fetch_stock_profile(ticker)
    news_data = get_live_news_impact(ticker, days_back=3)
    
    fund_analysis = run_pillar1_fundamental_analyst(
        ticker, 
        profile_data.get('fundamental', {}), 
        profile_data.get('technical', {}), 
        news_data
    )

    # ---------------------------------------------------------
    # 📈 Pillar 2: Technical Quant
    # ---------------------------------------------------------
    print("\n[2/4] Fetching Pillar 2 (Live Technicals & Quant AI)...")
    tech_snapshot = get_live_technical_snapshot(ticker)
    if not tech_snapshot:
        print("❌ Error: Cannot fetch live market data.")
        return

    print(f"  -> Live Price: ${tech_snapshot['current_price']} | Trend: {tech_snapshot['graph_trend']}")
    quant_setup = run_pillar2_technical_quant(ticker, tech_snapshot)

    # ---------------------------------------------------------
    # ⚖️ Pillar 3: Multi-Horizon Debate & Judgment
    # ---------------------------------------------------------
    print("\n[3/4] Executing Horizon-Specific Analyst Debates & PM Judgments...")
    horizons = ["SHORT-TERM", "MID-TERM", "LONG-TERM", "CONSERVATIVE"]
    final_dashboard = {}

    for h in horizons:
        print(f"  ⚙️ Generating cases for [{h}] Mandate...")
        
        # 🌟 สั่งทนายและผู้พิพากษาให้วิเคราะห์ "เจาะจงเฉพาะ Horizon" 
        bull = generate_bull_case(h, ticker, fund_analysis, tech_snapshot, quant_setup)
        bear = generate_bear_case(h, ticker, fund_analysis, tech_snapshot, quant_setup)
        
        verdict = run_horizon_judge(h, ticker, bull, bear, tech_snapshot, quant_setup)
        if verdict:
            final_dashboard[h] = verdict

    # ==========================================
    # 📊 PRINT FINAL TERMINAL DASHBOARD
    # ==========================================
    print("\n\n" + "="*80)
    print(f"📈 INSTITUTIONAL INVESTMENT DECISION DASHBOARD: {ticker}")
    print("="*80)
    print(f"Asset Current Price: ${tech_snapshot['current_price']}")
    print(f"Base Quant Recommendation  : {quant_setup.get('actionable_levels', {}).get('entry_price')} | Target: {quant_setup.get('actionable_levels', {}).get('target_price')}")
    print("-" * 80)
    
    for horizon, data in final_dashboard.items():
        levels = data.get('actionable_levels', {})
        print(f"▶️ HORIZON MANDATE: [{horizon}]")
        print(f"  [DECISION]   : {data.get('decision')} ({data.get('allocation_sizing')} Allocation)")
        if data.get('decision') != "REJECT" and data.get('decision') != "WAIT":
            print(f"  [TRADE PLAN] : Entry: {levels.get('entry_price')} | Target: {levels.get('target_price')} | Stop: {levels.get('stop_loss')}")
        print(f"  [PM VERDICT] : {data.get('pm_verdict_rationale')}")
        print("-" * 80)

if __name__ == "__main__":
    target_stock = "TSLA"  # ทดสอบเปลี่ยนชื่อหุ้นตรงนี้
    master_pipeline(target_stock)