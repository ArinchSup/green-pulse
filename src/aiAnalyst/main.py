import database_manager as db
import news_fetcher as fetcher
import ai_analyst as ai
import vectordb_manager as vector

TARGET_TICKER = "NVDA"  

def cleanup_pipeline(days=2):
    print(f"\n--- Starting Data Cleanup (Older than {days} days) ---")
    
    old_ids = db.get_old_news_ids(days=days)
    
    if old_ids:
        print(f"Found {len(old_ids)} old news items to remove.")
        
        vector.delete_from_vector_db(old_ids)
        
        db.delete_old_news_from_db(old_ids)
    else:
        print("No old news found. Database is clean.")
        
def run_pipeline():
    print(f"Starting Hybrid Pipeline for: {TARGET_TICKER}")
    db.setup_database()
    cleanup_pipeline(days=2)
    
    # 1. Fetch Recent News & Stock Profile
    all_news = fetcher.fetch_news(TARGET_TICKER, days_back=1)
    stock_info = fetcher.fetch_stock_profile(TARGET_TICKER) 
    
    if not all_news: 
        print("No news found.")
        return

    # Limit news to top 30 for analysis to prevent overload
    if len(all_news) > 30:
        print(f"Total news found: {len(all_news)}. Limiting to top 30 for analysis.")
        all_news = all_news[:30]
        
    # 2. Overall Sentiment Analysis 
    print("\n--- Performing Overall Sentiment Analysis ---")
    overall = ai.analyze_overall_sentiment(TARGET_TICKER, stock_info, all_news)
    if overall:
        print("\n[FULL AI ANALYSIS]")
        print(overall.get('raw_output', ''))
        print("-" * 50)


    '''
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
        
        # ส่ง stock_info เข้าไปด้วย
        analysis = ai.analyze_individual_news(TARGET_TICKER, stock_info, news['headline'], news['summary'])
        
        if analysis:
            db.update_individual_analysis(news['id'], analysis)
            
            trend = analysis.get('trend', 'UNKNOWN').upper()
            scale = analysis.get('scale', 0)
            
            print(f"\n[NEWS]: {news['headline']}")
            print(f" -> RESULT: {trend} (Scale: {scale})")
    '''

    print("\nPipeline Execution Completed.")

if __name__ == "__main__":
    run_pipeline()