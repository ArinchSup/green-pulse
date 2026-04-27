import os
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
DB_NAME = "rag_stock_news.db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
EXCLUDE_KEYWORDS = ["etf", "etfs", "mutual fund", "index fund", "model portfolio", "magnificent seven"]
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "stock_news_vectors"
# Lower is more strict
DISTANCE_THRESHOLD = 0.42