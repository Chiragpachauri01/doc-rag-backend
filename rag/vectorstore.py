import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

QDRANT_COLLECTION = "docs"
EMBEDDING_SIZE = 768   # Gemini embedding size

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

def create_collection_if_not_exists():
    if not client.collection_exists(QDRANT_COLLECTION):
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=Distance.COSINE,
            ),
        )

def add_chunks(chunks):
    points = []
    for chunk in chunks:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                },
            )
        )

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

def search_chunks(query_embedding, top_k=5):
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k
    )
    return result.points