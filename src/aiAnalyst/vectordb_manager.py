import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from aiAnalyst.config import CHROMA_PATH, COLLECTION_NAME, OLLAMA_BASE_URL, EMBEDDING_MODEL, DISTANCE_THRESHOLD

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=f"{OLLAMA_BASE_URL}/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=ollama_ef
)

def add_to_vector_db(news_id, text, metadata):
    try:
        collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(news_id)]
        )
    except Exception as e:
        print(f"Error adding to Vector DB: {e}")

def semantic_search(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    filtered_docs = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            
            if dist < DISTANCE_THRESHOLD:
                filtered_docs.append({"text": doc, "distance": dist, "metadata": meta})
            
    return filtered_docs

def delete_from_vector_db(news_ids):
    if not news_ids:
        return
    
    try:
        string_ids = [str(nid) for nid in news_ids]
        collection.delete(ids=string_ids)
        print(f"Successfully removed {len(string_ids)} vectors from ChromaDB.")
    except Exception as e:
        print(f"Error deleting from Vector DB: {e}")