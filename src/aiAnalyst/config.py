import os
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
DB_NAME = "rag_stock_news.db"
isLocal = os.getenv("IS_LOCAL")
if isLocal == "true":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
else:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = "stock-quant-v2-1"
EXCLUDE_KEYWORDS = ["etf", "etfs", "mutual fund", "index fund", "model portfolio", "magnificent seven"]
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "stock_news_vectors"
# Lower is more strict
DISTANCE_THRESHOLD = 0.42