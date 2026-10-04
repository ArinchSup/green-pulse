import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1" 

# ==========================================
# ⚖️ HORIZON MANDATES (กฎเหล็กสำหรับแต่ละกลยุทธ์)
# ==========================================
HORIZON_RULES = {
    "SHORT-TERM": "Focus 90% on Technical Momentum (Price action, Volume, RSI, MACD). Ignore long-term valuation metrics. Target fast institutional flow and immediate momentum. Holding period: 1-5 days.",
    "MID-TERM": "Focus 50% Fundamental / 50% Technical. Seek confluence between strong business momentum and optimal swing trading setups (pullbacks/breakouts). Target strictly 8% to 20%+ profit margins. Holding period: Weeks to Months.",
    "LONG-TERM": "Focus 80% on Fundamentals / 20% on Technicals. Prioritize structural growth, revenue momentum, and cheap valuation. Use technicals ONLY for optimal entry optimization. Holding period: 6-12+ Months.",
    "CONSERVATIVE": "Focus strictly on Capital Preservation, Downside Risk Mitigation, and Low Volatility (ATR). Highly sensitive to debt levels and technical breakdowns. Reject speculative setups."
}

# ==========================================
# The Institutional Perma-Bull
# ==========================================
def generate_bull_case(horizon, ticker, fund_data, tech_data, quant_setup):
    mandate = HORIZON_RULES[horizon]
    prompt = f"""
    You are an Elite Buy-Side Equity Analyst at a Tier-1 Hedge Fund specializing in {horizon} strategies.
    Your mandate: {mandate}
    
    Construct a rigorous, institutional-grade Bull Case for {ticker}.
    
    [DATA ROOM]
    - Fundamental Data: {json.dumps(fund_data)}
    - Raw Technical Snapshot: {json.dumps(tech_data)}
    - Quant Desk Setup: {json.dumps(quant_setup)}
    
    [INSTRUCTION]
    1. Synthesize the provided data to build a compelling bullish thesis explicitly tailored to the {horizon} mandate.
    2. Justify why the Quant Desk's entry/target levels make strategic sense.
    3. Use professional financial terminology (e.g., alpha generation, confluence, momentum divergence, fundamental catalysts).
    
    CRITICAL RULE: Output ONLY a valid JSON object. No markdown.
    {{
        "bull_arguments": ["Institutional-grade reason 1", "Reason 2", "Reason 3"],
        "bull_thesis": "A punchy 2-sentence executive summary of the upside potential."
    }}
    """
    return call_ollama(prompt, temperature=0.3)

# ==========================================
# 🔴 The Institutional Perma-Bear
# ==========================================
def generate_bear_case(horizon, ticker, fund_data, tech_data, quant_setup):
    mandate = HORIZON_RULES[horizon]
    prompt = f"""
    You are a Ruthless Short-Seller and Chief Risk Officer at a Tier-1 Hedge Fund specializing in {horizon} strategies.
    Your mandate: {mandate}
    
    Construct a rigorous, institutional-grade Bear Case (Risk Assessment) for {ticker}.
    
    [DATA ROOM]
    - Fundamental Data: {json.dumps(fund_data)}
    - Raw Technical Snapshot: {json.dumps(tech_data)}
    - Quant Desk Setup: {json.dumps(quant_setup)}
    
    [INSTRUCTION]
    1. Ruthlessly attack the asset's weaknesses explicitly based on the {horizon} mandate.
    2. Expose the flaws in the Quant Desk's setup (e.g., entry is a bull trap, overbought RSI, heavy debt).
    3. Use professional financial terminology (e.g., macro headwinds, downside exposure, valuation premium, momentum exhaustion).
    
    CRITICAL RULE: Output ONLY a valid JSON object. No markdown.
    {{
        "bear_arguments": ["Severe risk 1", "Risk 2", "Risk 3"],
        "bear_thesis": "A punchy 2-sentence executive summary of why capital deployment should be avoided."
    }}
    """
    return call_ollama(prompt, temperature=0.3)

# ==========================================
# 🧑‍⚖️ The Multi-Horizon Lead PM (Judge)
# ==========================================
def run_horizon_judge(horizon, ticker, bull_case, bear_case, tech_data, quant_setup):
    mandate = HORIZON_RULES[horizon]
    prompt = f"""
    You are the Lead Portfolio Manager (PM) managing a multi-billion dollar {horizon} portfolio.
    Your exact mandate: {mandate}
    
    Evaluate the institutional debate for {ticker}:
    [THE BULL CASE]: {json.dumps(bull_case)}
    [THE BEAR CASE]: {json.dumps(bear_case)}
    [RAW TECHNICALS]: {json.dumps(tech_data)}
    [QUANT DESK RECOMMENDATION]: {json.dumps(quant_setup.get('actionable_levels'))}
    
    [INSTRUCTION]
    1. Weigh the arguments based strictly on your {horizon} mandate.
    2. Determine the final decision (BUY, WAIT, or REJECT).
    3. If BUY, approve or slightly adjust the Quant Desk's Entry, Target, and Stop Loss to fit your horizon's risk appetite. If WAIT/REJECT, set prices to 0.0.
    
    CRITICAL RULE: Output ONLY a valid JSON object. No markdown.
    {{
        "horizon": "{horizon}",
        "decision": "BUY, WAIT, or REJECT",
        "allocation_sizing": "0%, 5%, 10%, or 15%",
        "actionable_levels": {{
            "entry_price": <Float number>,
            "target_price": <Float number>,
            "stop_loss": <Float number>
        }},
        "pm_verdict_rationale": "A 3-sentence professional verdict explaining your risk-adjusted assessment and conflict resolution."
    }}
    """
    return call_ollama(prompt, temperature=0.0)

# ==========================================
# 🔌 Ollama Connector
# ==========================================
def call_ollama(prompt, temperature=0.2):
    try:
        payload = {
            "model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": temperature, "seed": 42}
        }
        res = requests.post(OLLAMA_URL, json=payload).json()
        return json.loads(res["response"])
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return None