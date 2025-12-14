from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from rag.loaders import DocumentChunk


@dataclass
class TextChunk:
    text: str
    metadata: Dict[str, Any]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Works well for PDFs and is predictable.
    """
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_documents(
    docs: List[DocumentChunk],
    chunk_size: int = 1000,
    overlap: int = 150,
    min_chunk_chars: int = 200,
) -> List[TextChunk]:
    """
    Convert page-level DocumentChunks into smaller TextChunks.
    Keeps metadata for citations (source + page).
    Adds chunk_id within a page.
    """
    all_chunks: List[TextChunk] = []

    for doc in docs:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)

        for idx, piece in enumerate(pieces):
            if len(piece) < min_chunk_chars:
                continue

            meta = dict(doc.metadata)
            meta.update({
                "chunk_id": idx,
                "chunk_size": chunk_size,
                "overlap": overlap,
            })

            all_chunks.append(TextChunk(text=piece, metadata=meta))

    return all_chunks
