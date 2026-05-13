import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_client():
    client = chromadb.PersistentClient(path="data/chroma_db")
    return client

def create_collection(collection_name):
    client = get_client()
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def add_chunks(collection, chunks, doc_name):
    embeddings = model.encode(chunks).tolist()
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

def search_chunks(collection, query, n_results=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results["documents"][0]