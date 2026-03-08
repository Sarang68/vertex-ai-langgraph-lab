# vector_db/pinecone_index.py
from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def create_index(name: str = "rag-lab", dim: int = 768):
    if name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=name, dimension=dim, metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-central1")
        )
    return pc.Index(name)

def upsert_chunks(index, chunks, embeddings, namespace: str = "default"):
    """Namespace isolates tenants — critical for multi-tenant RAG."""
    vectors = [
        {"id": c.chunk_id, "values": emb, "metadata": {
            "text": c.text[:1000],  # Pinecone metadata limit
            "doc_id": c.doc_id,
            "access_groups": c.metadata.get("access_groups", []),
            "source_type": c.metadata.get("source_type", "unknown")
        }}
        for c, emb in zip(chunks, embeddings)
    ]
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100], namespace=namespace)
    print(f"Upserted {len(vectors)} vectors to namespace '{namespace}'")
