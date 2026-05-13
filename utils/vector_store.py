import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_collection(collection_name, reset=False):
    return collection_name

def add_chunks(collection, chunks, doc_name, session_state):
    embeddings = model.encode(chunks).tolist()
    if "store_chunks" not in session_state or doc_name == "reset":
        session_state.store_chunks = []
        session_state.store_embeddings = []
    session_state.store_chunks.extend(chunks)
    session_state.store_embeddings.extend(embeddings)

def search_chunks(collection, query, n_results, session_state):
    if not session_state.store_chunks:
        return []
    query_embedding = model.encode([query])[0]
    embeddings = np.array(session_state.store_embeddings)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[-n_results:][::-1]
    return [session_state.store_chunks[i] for i in top_indices]