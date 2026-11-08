import json
import requests

# 🌟 Imported the run_agent_1_reporter to act as our Bouncer
from class_news_fetcher_2 import fetch_news, scrape_full_text, clean_text, run_agent_1_reporter

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

def get_news_catalyst_score(ticker, target_date_str=None, days_back=3):
    print(f"📡 [Pillar 1] Fetching recent news for {ticker} leading up to {target_date_str or 'Today'}...")
    
    # 🌟 NEW: Pass the historical date into the fetcher
    news_list = fetch_news(ticker, target_date_str=target_date_str, days_back=days_back)
    
    if not news_list:
        print("   -> No news found passing the Python Strict Name Filter. Returning Neutral.")
        return {
            "catalyst_sentiment": "Neutral",
            "catalyst_score": 0,
            "rationale": "No recent company-specific news or press releases detected."
        }

    print(f"📰 [Pillar 1] Found {len(news_list)} initial articles. Passing through LLM Gatekeeper...")
    compiled_news = ""
    valid_articles = 0
    
    for news in news_list:
        if valid_articles >= 3: 
            break # We only need the top 10 verified business catalysts
            
        print(f"   -> Checking: {news['headline']}")
        full_text = scrape_full_text(news.get('url', ''))
        
        if not full_text: 
            full_text = clean_text(news.get('summary', ''))
            
        # 🌟 USE AGENT 1 TO FILTER GOSSIP AND NOISE
        gatekeeper_verdict = run_agent_1_reporter(full_text, ticker)
        
        if gatekeeper_verdict and gatekeeper_verdict.get('is_fundamental_news') is True:
            print("      ✅ Passed: Identified as fundamental business news.")
            compiled_news += f"\n📰 Headline: {news['headline']}\nContent: {full_text[:1500]}...\n---"
            valid_articles += 1
        else:
            reason = gatekeeper_verdict.get('rejection_reason', 'Irrelevant/Gossip') if gatekeeper_verdict else "API Error"
            print(f"      ❌ Rejected: {reason}")

    if valid_articles == 0:
        print("🧠 [Pillar 1] All news was rejected as noise or gossip. Returning Neutral.")
        return {
            "catalyst_sentiment": "Neutral",
            "catalyst_score": 0,
            "rationale": "News volume existed, but the AI identified it all as market noise or non-business gossip."
        }

    print("🧠 [Pillar 1] Analyzing confirmed catalysts with Llama 3.1...")

    # 3. The Calibrated Quant Catalyst Prompt
    prompt = f"""You are a ruthless, skeptical Event-Driven Quant Analyst at a multi-billion dollar hedge fund.
Your job is to read verified business news for {ticker} and determine if there is an UNPRICED fundamental catalyst that will move the stock significantly over the next 1-3 months.

[NEWS DATA ROOM]
{compiled_news}

[STRICT SCORING & SENTIMENT RULES]
1. SENTIMENT CLASSIFICATION:
   - If catalyst_score is 1 to 3: catalyst_sentiment MUST be "Bearish" (or "Neutral" if purely irrelevant noise).
   - If catalyst_score is 4 to 6: catalyst_sentiment MUST be "Neutral" (routine operations do not justify a directional Bullish bet).
   - If catalyst_score is 7 to 10: catalyst_sentiment MUST be "Bullish" (genuine, unpriced market catalyst).

2. SCORING RUBRIC:
   - 1 to 3: Immaterial noise, executive interviews, or news that is 100% priced in.
   - 4 to 6: Normal business operations, expected product updates, minor partnerships. (Default tier).
   - 7 to 8: Substantial surprise (>10% EPS beat/miss, unexpected leadership change, tier-1 multi-billion-dollar deal).
   - 9 to 10: Transformational catalyst (major M&A, existential litigation, federal antitrust intervention).

[INSTRUCTION]
1. GROUNDING: Evaluate EXCLUSIVELY what is written in the [NEWS DATA ROOM]. Do not extrapolate.
2. MATERIALITY: Scale your score to the company's size. A product update for a mega-cap requires massive volume to move the stock price.
3. SKEPTICISM: You must state why this news might ALREADY be priced in or exaggerated by corporate PR before scoring.

Output EXACTLY in this JSON format:
{{
    "skepticism_check": "<1 sentence stating why this news might be PR hype, immaterial to total revenue, or already priced in by the market>",
    "catalyst_sentiment": "<Bullish, Bearish, or Neutral>",
    "catalyst_score": <int 1-10>,
    "rationale": "<2 sentences clearly explaining the real financial impact on the company over the next 1-3 months.>"
}}
"""
    try:
        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": {
                "temperature": 0.1,  
                "seed": 42
            }
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=60).json()
        ai_output = json.loads(res["response"])
        return ai_output
        
    except Exception as e:
        print(f"⚠️ Pillar 1 Error: {e}")
        return {
            "catalyst_sentiment": "Neutral",
            "catalyst_score": 0,
            "rationale": f"System error reading news: {e}"
        }

if __name__ == "__main__":
    test_ticker = "META"
    print("="*50)
    print(f"PILLAR 1: MICRO NEWS GATEKEEPER ({test_ticker})")
    print("="*50)
    catalyst_analysis = get_news_catalyst_score(test_ticker, days_back=3)
    print("\n✅ === FINAL NEWS CATALYST SCORE ===")
    print(json.dumps(catalyst_analysis, indent=4, ensure_ascii=False))