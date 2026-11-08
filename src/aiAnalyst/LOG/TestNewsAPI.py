# API_KEY = "d77st29r01qsamsi2qe0d77st29r01qsamsi2qeg"
import finnhub
import datetime

# 1. ใส่ API Key ของ Finnhub
API_KEY = "d77st29r01qsamsi2qe0d77st29r01qsamsi2qeg"
finnhub_client = finnhub.Client(api_key=API_KEY)

# 2. รายชื่อคำต้องห้าม (กรองข่าวกองทุน หรือข่าวภาพรวมที่ไม่เกี่ยวกับปัจจัยพื้นฐานบริษัท)
# ปรับให้เบาลง เน้นเตะข่าวกองทุนล้วนๆ
EXCLUDE_KEYWORDS = [
    "etf", "etfs", "mutual fund", "index fund", 
    "model portfolio", "magnificent seven" 
]

def is_relevant_news(headline, summary):
    """
    ด่านที่ 1: ตรวจสอบว่ามีคำต้องห้ามในหัวข้อหรือเนื้อหาหรือไม่
    """
    # ป้องกันกรณี summary เป็น None (บางข่าวไม่มี summary)
    safe_summary = summary if summary else ""
    full_text = (headline + " " + safe_summary).lower()
    
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in full_text:
            return False # เจอคำต้องห้าม เตะทิ้ง
    return True # ผ่านด่าน

def fetch_and_filter_news(target_ticker, days_back=1):
    """
    ฟังก์ชันหลัก: ดึงข่าวและกรองข้อมูล
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    print(f"กำลังดึงข่าวและวิเคราะห์ Tag สำหรับ {target_ticker} ตั้งแต่ {start_date} ถึง {end_date}...")
    
    try:
        # เรียก API ดึงข่าว (ใช้ company_news เพื่อดึงข่าวที่ระบบ Tag ไว้ให้แล้ว)
        news_data = finnhub_client.company_news(target_ticker, _from=str(start_date), to=str(end_date))
        
        if not news_data:
            print("ไม่มีข่าวในช่วงเวลานี้")
            return []
            
        clean_news = []
        for item in news_data:
            headline = item['headline']
            summary = item.get('summary', '')
            
            # ดึง Tag หุ้นที่เกี่ยวข้องออกมา (ใช้ .get ป้องกันกรณี API ไม่ส่งคีย์นี้มา)
            related_tags = item.get('related', '')
            
            # ด่านที่ 2: เช็กว่า Target Ticker ของเราอยู่ใน Tag จริงๆ ใช่ไหม
            # (เป็นการรีเช็กอีกรอบเผื่อ API ส่งข่าวรวมๆ มา)
            if target_ticker in related_tags:
                
                # นำไปผ่านด่านที่ 1 (กรองคำต้องห้าม)
                if is_relevant_news(headline, summary):
                    dt_object = datetime.datetime.fromtimestamp(item['datetime'])
                    news_dict = {
                        "id": item['id'],
                        "time": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
                        "headline": headline,
                        "summary": summary,
                        "related_tags": related_tags, # เก็บ Tag ไว้ดูด้วย
                        "url": item['url']
                    }
                    # ในฟังก์ชัน fetch_and_filter_news
                    if not summary or len(summary.strip()) < 20: 
                        continue # ข้ามข่าวนี้ไปเลย ไม่เก็บลง Database
                    clean_news.append(news_dict)
                else:
                    # ข่าวโดนกรองทิ้งเพราะติด Keyword (คอมเมนต์ออกได้ตอนใช้งานจริง)
                    print(f"[กรองทิ้ง - ติด Keyword] {headline}")
            else:
                # ข่าวโดนกรองทิ้งเพราะไม่มี Tag หุ้นของเรา (คอมเมนต์ออกได้ตอนใช้งานจริง)
                print(f"[กรองทิ้ง - ไม่พบ Tag {target_ticker}] {headline}")
                
        return clean_news

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        return []

# ทดลองรันใช้งานจริง
if __name__ == "__main__":
    # ลองใช้ IREN แบบที่คุณยกตัวอย่างมา
    ticker = "IREN" 
    final_news = fetch_and_filter_news(ticker, days_back=2)
    
    print(f"\n✅ สำเร็จ! ได้ข่าวที่คลีนแล้วจำนวน {len(final_news)} ข่าว")
    for news in final_news[:3]: # แสดงแค่ 3 ข่าวแรก
        print(f"\n[{news['time']}] {news['headline']}")
        print(f"Tags: {news['related_tags']}")
        print(f"Summary: {news['summary']}")
        print("-" * 50)