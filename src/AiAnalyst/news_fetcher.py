import finnhub
import datetime
from config import FINNHUB_API_KEY, EXCLUDE_KEYWORDS

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def is_relevant_news(headline, summary):
    safe_summary = summary if summary else ""
    full_text = (headline + " " + safe_summary).lower()
    return not any(keyword in full_text for keyword in EXCLUDE_KEYWORDS)

def fetch_news(ticker, days_back=1):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back)
    try:
        news_data = finnhub_client.company_news(ticker, _from=str(start_date), to=str(end_date))
        clean_news = []
        for item in (news_data or []):
            headline = item['headline']
            summary = item.get('summary', '') or headline
            if ticker in item.get('related', '') and is_relevant_news(headline, summary):
                dt_object = datetime.datetime.fromtimestamp(item['datetime'])
                clean_news.append({
                    "id": item['id'], "time": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
                    "headline": headline, "summary": summary,
                    "related_tags": item.get('related', ''), "url": item['url']
                })
        return clean_news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []