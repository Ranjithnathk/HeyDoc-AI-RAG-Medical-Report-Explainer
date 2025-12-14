import os
import hashlib
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from rag.chunking import TextChunk
from rag.embeddings import embed_texts

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "heydocai-medkb")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

# For text-embedding-3-small
EMBED_DIMS = 1536

def _make_id(meta: dict) -> str:
    """
    Stable ID per chunk so re-runs don't duplicate vectors.
    """
    base = f"{meta.get('source')}|p{meta.get('page')}|c{meta.get('chunk_id')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def get_index():
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )

    return pc.Index(INDEX_NAME)

def upsert_chunks(chunks: List[TextChunk], batch_size: int = 64):
    index = get_index()

    total = len(chunks)
    print(f"Upserting {total} chunks into Pinecone index '{INDEX_NAME}'...")

    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.text for c in batch]
        vectors = embed_texts(texts)

        upserts = []
        for c, vec in zip(batch, vectors):
            meta = dict(c.metadata)
            meta["text"] = c.text  # store snippet for citations
            vec_id = _make_id(meta)

            upserts.append({
                "id": vec_id,
                "values": vec,
                "metadata": meta
            })

        index.upsert(vectors=upserts)
        print(f"{min(start + batch_size, total)}/{total} upserted")

    print("Upsert complete.")
