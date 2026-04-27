import aiAnalyst.database_manager as db
import aiAnalyst.news_fetcher as fetcher
import aiAnalyst.ai_analyst as ai
import aiAnalyst.vectordb_manager as vector

def cleanup_pipeline(days=2):
    
    old_ids = db.get_old_news_ids(days=days)
    
    if old_ids:
        vector.delete_from_vector_db(old_ids)
        db.delete_old_news_from_db(old_ids)
        
def run_pipeline(TARGET_TICKER = "IREN", TARGET_HORIZON = "Short-term"):
    db.setup_database()
    cleanup_pipeline(days=2)
    
    all_news = fetcher.fetch_news(TARGET_TICKER, days_back=1)
    stock_info = fetcher.fetch_stock_profile(TARGET_TICKER) 
    
    if stock_info["current_price"] == 0:
        return {"status": "error", "message": f"Ticker {TARGET_TICKER} not found or no data available."}
    
    if not all_news: 
        all_news = []

    if len(all_news) > 30:
        all_news = all_news[:30]
        
    overall = ai.analyze_overall_sentiment(TARGET_TICKER, stock_info, all_news, horizon=TARGET_HORIZON)
    
    result = {
        "ticker": TARGET_TICKER,
        "horizon": TARGET_HORIZON,
        "stock_info": stock_info,
        "ai_analysis": overall if overall else {"action": "FAILED"},
        "news_count": len(all_news)
    }
    
    return result

if __name__ == "__main__":
    data = run_pipeline("HOOD", "Short-term")
    print(f"\n[CLI Mode] Analysis for {data['ticker']}:")
    print(data['ai_analysis'])