import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# 1. สร้างตัวเชื่อมต่อไปหา Ollama Embedding
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text", # ใช้โมเดลเฉพาะทางที่เราเพิ่งโหลดมา
)

chroma_client = chromadb.PersistentClient(path="./my_vector_db")

# 2. ลบ Collection เก่าทิ้งก่อน (เพราะ Vector เก่ากับใหม่ความยาวไม่เท่ากัน จะ Error)
try:
    chroma_client.delete_collection(name="stock_news")
except Exception:
    pass

# 3. สร้าง Collection ใหม่ โดยยัด ollama_ef เข้าไปเป็นสมองหลัก!
collection = chroma_client.create_collection(
    name="stock_news",
    embedding_function=ollama_ef
)

# ข้อมูลเดิมเลยครับ
documents = [
    "IREN has acquired 10,000 new mining rigs.",
    "Apple releases the new iPhone 16.",
    "Bitcoin price drops to $60,000 due to inflation.",
    "Tesla introduces a new electric vehicle model."
]
ids = ["news_1", "news_2", "news_3", "news_4"]

print("⏳ กำลังให้ Ollama แปลงข้อความเป็น Vector (อาจจะใช้เวลาแป๊บนึง)...")
collection.add(documents=documents, ids=ids)
print("✅ บันทึกลง Database สำเร็จ!\n")

# ทดสอบค้นหาคำเดิมเป๊ะๆ
query = "Are there any updates on cryptocurrency hardware?"

# แก้ตรงส่วน query
results = collection.query(
    query_texts=[query],
    n_results=2 
)

print("\nผลลัพธ์ที่ได้ (พร้อมคะแนนระยะห่าง):")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    dist = results['distances'][0][i] # ดึงค่าระยะห่างออกมาดู
    print(f"- {doc} (Distance: {dist:.4f})")
    
# ตัวอย่าง Logic การกรอง
threshold = 0.42
filtered_results = []

for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    dist = results['distances'][0][i]
    
    if dist < threshold:
        filtered_results.append(doc)
    else:
        print(f"Skipping: '{doc}' because distance {dist:.4f} is too high.")