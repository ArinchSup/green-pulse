# database_manager.py
import sqlite3
from config import DB_NAME

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_news (
            news_id TEXT PRIMARY KEY,
            ticker TEXT,
            published_at DATETIME,
            headline TEXT,
            summary TEXT,
            tags TEXT,
            url TEXT,
            ai_trend TEXT,
            ai_scale INTEGER,
            ai_reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_and_get_new_news(ticker, clean_news_list):
    if not clean_news_list:
        return []
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    newly_added_news = []
    for news in clean_news_list:
        cursor.execute('''
            INSERT OR IGNORE INTO stock_news 
            (news_id, ticker, published_at, headline, summary, tags, url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(news['id']), ticker, news['time'], news['headline'], news['summary'], news['related_tags'], news['url']))
        if cursor.rowcount == 1:
            newly_added_news.append(news)
    conn.commit()
    conn.close()
    return newly_added_news

def update_individual_analysis(news_id, analysis_data):
    if not analysis_data: return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    trend = analysis_data.get('trend') or analysis_data.get('Trend') or 'unknown'
    scale = analysis_data.get('scale') or analysis_data.get('Scale') or 0
    reason = analysis_data.get('reason') or analysis_data.get('Reason') or ''
    cursor.execute('''
        UPDATE stock_news SET ai_trend = ?, ai_scale = ?, ai_reason = ? WHERE news_id = ?
    ''', (str(trend).lower(), scale, reason, str(news_id)))
    conn.commit()
    conn.close()