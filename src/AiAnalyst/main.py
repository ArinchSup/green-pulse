import database_manager as db
import news_fetcher as fetcher
import ai_analyst as ai
import vectordb_manager as vector

TARGET_TICKER = "AAPL"

def run_pipeline():
    print(f"Starting Hybrid Pipeline for: {TARGET_TICKER}")
    db.setup_database()
    
    # 1. Fetch Recent News
    all_news = fetcher.fetch_news(TARGET_TICKER, days_back=1)
    if not all_news: 
        print("No news found.")
        return

    # 2. Overall Sentiment Analysis (Added back for you)
    print("\n--- Performing Overall Sentiment Analysis ---")
    overall = ai.analyze_overall_sentiment(TARGET_TICKER, all_news)
    if overall:
        print(f"OVERALL TREND : {overall.get('trend', 'N/A').upper()}")
        print(f"OVERALL SCALE : {overall.get('scale', 0)}")
        print(f"OVERALL REASON: {overall.get('reason', 'N/A')}")

    # 3. Process Individual News
    print("\n--- Processing Individual News Entries ---")
    new_entries = db.save_and_get_new_news(TARGET_TICKER, all_news)
    print(f"New news detected: {len(new_entries)}")
    
    for news in new_entries:
        vector.add_to_vector_db(
            news_id=news['id'],
            text=f"{news['headline']} - {news['summary']}",
            metadata={"ticker": TARGET_TICKER, "url": news['url']}
        )
        
        analysis = ai.analyze_individual_news(TARGET_TICKER, news['headline'], news['summary'])
        
        if analysis:
            db.update_individual_analysis(news['id'], analysis)
            
            trend = analysis.get('trend', 'UNKNOWN').upper()
            scale = analysis.get('scale', 0)
            reason = analysis.get('reason', 'N/A')
            
            print(f"\n[NEWS]: {news['headline']}")
            print(f" -> RESULT: {trend} (Scale: {scale})")
            print(f" -> REASON: {reason}")

    print("\nPipeline Execution Completed.")

if __name__ == "__main__":
    run_pipeline()