import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone

from rag.embeddings import embed_texts
from rag.ranking import rank_and_filter

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "heydocai-medkb")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def retrieve_top_k(
    query: str,
    top_k: int = 12,              # fetch more initially
    min_score: float = 0.50,      # filter after retrieval
    final_top_k: int = 6,         # return fewer, higher-signal
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    query_embedding = embed_texts([query])[0]

    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )

    raw_results = []
    for match in res.get("matches", []):
        meta = match.get("metadata", {}) or {}
        raw_results.append({
            "text": meta.get("text", ""),
            "score": float(match.get("score", 0.0)),
            "metadata": meta,
        })

    return rank_and_filter(
        raw_results,
        min_score=min_score,
        final_top_k=final_top_k,
        max_context_chars=4500,
        per_chunk_char_cap=900,
    )
