import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory store
store = {
    "chunks": [],
    "embeddings": []
}

def create_collection(collection_name, reset=False):
    if reset:
        store["chunks"] = []
        store["embeddings"] = []
    return collection_name

def add_chunks(collection, chunks, doc_name):
    embeddings = model.encode(chunks)
    store["chunks"].extend(chunks)
    store["embeddings"].extend(embeddings)

def search_chunks(collection, query, n_results=3):
    if not store["chunks"]:
        return []
    
    query_embedding = model.encode([query])[0]
    embeddings = np.array(store["embeddings"])
    
    # Cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-n_results:][::-1]
    return [store["chunks"][i] for i in top_indices]